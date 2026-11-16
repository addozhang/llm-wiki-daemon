//! Write gateway. Single writer per file, optimistic concurrency, and NO
//! delete or generic whole-file overwrite tool. Every write is guarded by the
//! content hash returned from `vault_read`; the derived index is updated in
//! this same process immediately afterwards.
//!
//! Scope-authorised writes: tools are ON by default but may ONLY touch paths
//! inside the configured wiki root whose frontmatter says `layer: wiki`.
//! `vault_create` is the bootstrap primitive for the very first wiki page.
//!
//! Two kinds of change, with different guardrails:
//!   * incremental (patch / append): add content or append provenance.
//!   * revision (rewrite): re-compile the body, or REPLACE sources / related
//!     so wrong provenance can actually be removed. Requires a `reason`;
//!     writes `updated` / `revision`, forces `verified: false`, and records
//!     the change in the daemon log. The page holds only current truth;
//!     history lives in the daemon log and in the vault's git history.
//!
//! `layer` stays immutable and `verified` can only be set true by a human,
//! so a note cannot be promoted into a writable page. `vault_lint` is
//! read-only and always available; `[writes] enabled` is an emergency kill
//! switch.
use crate::audit::AuditLog;
use crate::index::{VaultIndex, content_hash};
use anyhow::{Context, Result, bail};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as AsyncMutex;
use walkdir::WalkDir;

pub const WRITE_TOOLS: [&str; 4] = [
    "vault_patch",
    "vault_append",
    "vault_create",
    "vault_rewrite",
];

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PatchRequest {
    /// Existing wiki page, addressed by an Obsidian wikilink ([[Note]]) or a
    /// bare name. `path` is accepted as a legacy alias.
    #[serde(alias = "path")]
    pub target: String,
    /// Which part to patch: "frontmatter", "section" or "end".
    pub part: String,
    /// frontmatter: field name; section: heading text without the leading hashes.
    #[serde(default)]
    pub key: Option<String>,
    /// "set" (frontmatter scalar), "append" for frontmatter lists (sources/related), section or end.
    pub op: String,
    /// New value (plain text). YAML quoting is handled by the daemon.
    pub value: String,
    /// Content hash from vault_read. A mismatch returns a conflict instead of writing.
    pub expected_version: String,
    /// Optional note explaining this incremental change (recorded in the audit trail).
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AppendRequest {
    /// Existing wiki page, addressed by wikilink ([[Note]]) or bare name.
    #[serde(alias = "path")]
    pub target: String,
    pub text: String,
    pub expected_version: String,
    /// Optional heading; when set the text is appended to that section.
    #[serde(default)]
    pub heading: Option<String>,
    /// Optional note explaining the append (recorded in the audit trail).
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateRequest {
    /// Path of a NEW wiki page; must be under wiki/. Existing files are never overwritten.
    pub path: String,
    /// Page type: concept | topic | reference | entity | synthesis.
    pub kind: String,
    /// Provenance: paths (or [[wikilinks]]) of existing raw/note notes. Must be non-empty.
    pub sources: Vec<String>,
    /// One-line summary; required so a wiki page is never an empty shell.
    pub summary: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub title: Option<String>,
    /// high | medium | low (default medium).
    #[serde(default)]
    pub confidence: Option<String>,
    /// Optional note explaining why this page exists (recorded in the audit trail).
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RewriteRequest {
    /// Existing wiki page, addressed by wikilink ([[Note]]) or bare name.
    #[serde(alias = "path")]
    pub target: String,
    /// Content hash from vault_read. A mismatch returns a conflict.
    pub expected_version: String,
    /// Why this revision is happening. Recorded in the daemon audit log, never
    /// stored in the page, so the page only carries current truth.
    pub reason: String,
    /// New Markdown body. With `heading`, replaces just that section instead.
    #[serde(default)]
    pub text: Option<String>,
    /// Section heading to replace (without leading hashes); requires `text`.
    #[serde(default)]
    pub heading: Option<String>,
    /// Full replacement for sources. Entries may be removed; must be non-empty.
    #[serde(default)]
    pub sources: Option<Vec<String>>,
    /// Full replacement for related; may be empty to clear it.
    #[serde(default)]
    pub related: Option<Vec<String>>,
    /// Full replacement for supersedes (conclusions retired by newer sources);
    /// may be empty to clear it. Entries are wikilinks, verified via the index.
    #[serde(default)]
    pub supersedes: Option<Vec<String>>,
    /// New one-line summary. Should be supplied whenever the body changes:
    /// a summary written at create time goes stale as sections are rewritten.
    #[serde(default)]
    pub summary: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LintRequest {
    /// Maximum number of issues to return (1-200).
    #[serde(default = "default_lint_limit")]
    pub limit: i64,
}

fn default_lint_limit() -> i64 {
    50
}

pub struct WriteStore {
    index: Arc<VaultIndex>,
    enabled: bool,
    wiki_root: String,
    audit: Arc<AuditLog>,
    locks: Mutex<HashMap<String, Arc<AsyncMutex<()>>>>,
}

impl WriteStore {
    pub fn new(
        index: Arc<VaultIndex>,
        enabled: bool,
        wiki_root: String,
        audit: Arc<AuditLog>,
    ) -> Self {
        Self {
            index,
            enabled,
            wiki_root: wiki_root.trim().trim_end_matches('/').to_string(),
            audit,
            locks: Mutex::new(HashMap::new()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Audit failures never roll back a write that already reached disk; they
    /// are surfaced as warnings so the operator can inspect the log directory.
    fn audit_write(
        &self,
        op: &str,
        target: &str,
        path: &str,
        reason: Option<&str>,
        version_before: Option<&str>,
        version_after: &str,
        changes: Vec<String>,
    ) {
        if let Err(err) = self.audit.write_event(
            op,
            target,
            path,
            reason,
            version_before,
            version_after,
            changes,
        ) {
            tracing::warn!("审计日志写入失败: {err:#}");
        }
    }

    fn in_wiki_root(&self, rel: &str) -> bool {
        rel.starts_with(&format!("{}/", self.wiki_root))
    }

    /// Every write must satisfy BOTH checks: the path lives in the configured
    /// wiki root AND the file says `layer: wiki`. Directory alone is not enough
    /// (a stray note under wiki/ must stay read-only); layer alone is not
    /// enough (a `layer: wiki` file in 笔记/ must not be writable either).
    fn ensure_wiki_scope(&self, rel: &str, text: &str) -> Result<()> {
        if !self.in_wiki_root(rel) {
            bail!("写操作只允许作用于 {} 下的页面: {rel}", self.wiki_root);
        }
        let layer = frontmatter_layer(text);
        if layer.as_deref() != Some("wiki") {
            bail!(
                "只允许修改 layer: wiki 的页面（{rel} 的 layer 是 {layer:?}）；note / raw 只读"
            );
        }
        Ok(())
    }

    fn ensure_enabled(&self) -> Result<()> {
        if !self.enabled {
            bail!("写功能已被紧急总闸关闭：config.toml [writes] enabled = false");
        }
        Ok(())
    }

    fn lock_for(&self, rel: &str) -> Arc<AsyncMutex<()>> {
        let mut locks = self.locks.lock().unwrap();
        locks
            .entry(rel.to_string())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }

    /// Reject absolute paths, traversal, non-Markdown targets and denied areas.
    /// Canonicalizing the parent also blocks symlink escapes.
    fn resolve(&self, rel: &str, must_exist: bool) -> Result<(String, PathBuf)> {
        if rel.is_empty()
            || rel.starts_with('/')
            || rel.contains('\\')
            || rel.split('/').any(|p| p == ".." || p == "." || p.is_empty())
        {
            bail!("非法路径: {rel}");
        }
        if !rel.ends_with(".md") {
            bail!("只允许操作 .md 文件: {rel}");
        }
        if self.index.is_denied(rel) {
            bail!("路径位于排除区，禁止写入: {rel}");
        }
        let root = self.index.root();
        let abs = root.join(rel);

        if must_exist {
            let canonical =
                std::fs::canonicalize(&abs).with_context(|| format!("文件不存在: {rel}"))?;
            if !canonical.starts_with(root) || !canonical.is_file() {
                bail!("路径越出 vault 或不是文件: {rel}");
            }
            Ok((rel.to_string(), canonical))
        } else {
            let parent = abs.parent().context("缺少父目录")?;
            if !parent.exists() {
                // Bootstrap: a fresh vault may have no wiki subtree at all.
                // Creating directories under the configured wiki root is safe.
                if !self.in_wiki_root(rel) {
                    bail!(
                        "父目录不存在，且路径不在 wiki root {} 下: {rel}",
                        self.wiki_root
                    );
                }
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("创建 wiki 子目录失败: {}", parent.display()))?;
            }
            let canonical_parent =
                std::fs::canonicalize(parent).with_context(|| format!("父目录不存在: {rel}"))?;
            if !canonical_parent.starts_with(root) {
                bail!("父目录越出 vault: {rel}");
            }
            if abs.exists() {
                bail!("文件已存在，拒绝覆盖: {rel}");
            }
            let name = abs.file_name().context("缺少文件名")?;
            Ok((rel.to_string(), canonical_parent.join(name)))
        }
    }

    /// Writes use the same wikilink addressing contract as reads. Fall back to
    /// the literal path when the link index has not yet seen a freshly created
    /// file (the watcher has a debounce window).
    fn resolve_write_target(&self, target: &str) -> Result<(String, PathBuf)> {
        if let Ok(res) = self.index.resolve(target) {
            if let Some(rel) = res.path {
                return self.resolve(&rel, true);
            }
        }
        self.resolve(target, true)
    }

    /// `audit_op` distinguishes a direct patch from an append (which funnels
    /// through the same patch path) in the audit trail.
    pub async fn patch(&self, mut req: PatchRequest, audit_op: &str) -> Result<Value> {
        self.ensure_enabled()?;
        let (rel, abs) = self.resolve_write_target(&req.target)?;
        let gate = self.lock_for(&rel);
        let _guard = gate.lock().await;

        let text = std::fs::read_to_string(&abs).with_context(|| format!("读取失败: {rel}"))?;

        self.ensure_wiki_scope(&rel, &text)?;
        if req.part == "frontmatter" && req.op == "set" {
            match req.key.as_deref() {
                Some("layer") => {
                    bail!("layer 不可变：禁止通过 patch 修改，否则可把 note 提升为可写页面")
                }
                Some("sources") | Some("related") | Some("supersedes") => bail!(
                    "{} 不可通过 patch 整段替换；删除或修正溯源条目请用 vault_rewrite 并给出 reason",
                    req.key.as_deref().unwrap_or("")
                ),
                Some("revision") | Some("updated") => {
                    bail!("{} 由 vault_rewrite 自动维护，不要手工 patch", req.key.as_deref().unwrap_or(""))
                }
                Some("verified") => {
                    // YAML accepts many truthy spellings; compare semantically,
                    // not just against the literal string "true".
                    let v = req.value.trim().trim_matches('"').to_ascii_lowercase();
                    // PyYAML 1.1 also treats y/Y as truthy; cover the full
                    // boolean vocabulary so `verified` cannot be flipped.
                    if matches!(v.as_str(), "true" | "yes" | "y" | "on" | "1") {
                        bail!("verified 只能由人工确认后置 true")
                    }
                }
                _ => {}
            }
        }
        if req.part == "frontmatter"
            && req.op == "append"
            && matches!(req.key.as_deref(), Some("sources") | Some("related"))
        {
            // Provenance must stay verifiable when it grows, not only at create.
            let normalized = normalize_source(&req.value, &self.index, false)?;
            req.value = format!("[[{normalized}]]");
        }

        let before = content_hash(&text);
        if before != req.expected_version {
            return Ok(json!({
                "ok": false,
                "conflict": true,
                "path": rel,
                "current_version": before,
                "error": "expected_version 不匹配；请重新 vault_read 后重试",
            }));
        }

        let edited = apply_edit(
            &text,
            &req.part,
            req.key.as_deref(),
            &req.op,
            &req.value,
        )?;
        validate_frontmatter(&edited)?;
        if edited == text {
            return Ok(json!({"ok": false, "no_change": true, "path": rel, "version": before}));
        }

        atomic_write(&abs, &edited)?;
        self.index.upsert(&rel)?;
        let after = content_hash(&edited);
        tracing::info!("write patch {rel} ({}/{})", req.part, req.op);
        self.audit_write(
            audit_op,
            &req.target,
            &rel,
            req.reason.as_deref(),
            Some(&before),
            &after,
            vec![format!("{}/{}", req.part, req.op)],
        );
        Ok(json!({
            "ok": true,
            "path": rel,
            "operation": format!("{}/{}", req.part, req.op),
            "version_before": before,
            "version_after": after,
        }))
    }

    pub async fn append(&self, req: AppendRequest) -> Result<Value> {
        self.ensure_enabled()?;
        let (part, key) = match &req.heading {
            Some(h) if !h.trim().is_empty() => ("section".to_string(), Some(h.clone())),
            _ => ("end".to_string(), None),
        };
        let reason = req.reason;
        self.patch(
            PatchRequest {
                target: req.target,
                part,
                key,
                op: "append".to_string(),
                value: req.text,
                expected_version: req.expected_version,
                reason,
            },
            "append",
        )
        .await
    }

    /// Bootstrap primitive: creates the FIRST wiki page without requiring any
    /// pre-existing wiki page. It is safe because it can only emit a
    /// schema-compliant page under wiki/ with verifiable provenance.
    pub async fn create(&self, req: CreateRequest) -> Result<Value> {
        self.ensure_enabled()?;
        let (rel, abs) = self.resolve(&req.path, false)?;
        if !self.in_wiki_root(&rel) {
            bail!(
                "vault_create 只能建 {} 下的页面: {rel}",
                self.wiki_root
            );
        }
        if !WIKI_KINDS.contains(&req.kind.as_str()) {
            bail!("非法 kind: {}（可选 {WIKI_KINDS:?}）", req.kind);
        }
        let confidence = req
            .confidence
            .clone()
            .unwrap_or_else(|| "medium".to_string());
        if !CONFIDENCE_LEVELS.contains(&confidence.as_str()) {
            bail!("非法 confidence: {confidence}");
        }
        if req.summary.trim().is_empty() {
            bail!("summary 不能为空：wiki 页必须有一行摘要");
        }
        if req.sources.is_empty() {
            bail!("sources 不能为空：wiki 页必须溯源到真实笔记");
        }
        // Two aliases may resolve to the same note; keep each source once.
        let mut normalized: Vec<String> = Vec::with_capacity(req.sources.len());
        for source in &req.sources {
            let resolved = normalize_source(source, &self.index, false)?;
            if !normalized.contains(&resolved) {
                normalized.push(resolved);
            }
        }
        let sources_json = serde_json::to_string(
            &normalized
                .iter()
                .map(|p| format!("[[{p}]]"))
                .collect::<Vec<_>>(),
        )?;

        let gate = self.lock_for(&rel);
        let _guard = gate.lock().await;
        if abs.exists() {
            bail!("文件已存在，拒绝覆盖: {rel}");
        }
        let title = req.title.as_deref().map(str::to_string).unwrap_or_else(|| {
            Path::new(&rel)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&rel)
                .to_string()
        });
        let text = format!(
            "---\ntitle: {}\nlayer: wiki\nkind: {}\nsummary: {}\ntags: []\nsources: {}\nconfidence: {}\nverified: false\nstatus: draft\n---\n\n{}\n",
            yaml_scalar(&title),
            yaml_scalar(&req.kind),
            yaml_scalar(req.summary.trim()),
            sources_json,
            yaml_scalar(&confidence),
            req.text.trim_end(),
        );
        validate_frontmatter(&text)?;
        atomic_write(&abs, &text)?;
        self.index.upsert(&rel)?;
        let version = content_hash(&text);
        tracing::info!(
            "write create {rel} (wiki/{}, {} sources)",
            req.kind,
            normalized.len()
        );
        self.audit_write(
            "create",
            &req.path,
            &rel,
            req.reason.as_deref(),
            None,
            &version,
            vec![format!("kind={}", req.kind), format!("sources={}", normalized.len())],
        );
        Ok(json!({
            "ok": true,
            "path": rel,
            "version": version,
            "created": true,
            "layer": "wiki",
            "kind": req.kind,
            "source_count": normalized.len(),
        }))
    }

    /// Revision primitive: re-compile a wiki page's body and/or correct its
    /// provenance. Unlike patch/append this may REMOVE entries and paragraphs,
    /// because keeping a wrong source or a wrong claim is exactly what lets
    /// errors compound through later retrievals. Audit goes to the daemon log
    /// and the vault's git history, never into the page itself.
    pub async fn rewrite(&self, req: RewriteRequest) -> Result<Value> {
        self.ensure_enabled()?;
        let reason = req.reason.trim();
        if reason.is_empty() {
            bail!("reason 必填：任何修订都要说明原因，用于 daemon 审计日志");
        }
        if req.text.is_none()
            && req.sources.is_none()
            && req.related.is_none()
            && req.supersedes.is_none()
            && req.summary.is_none()
        {
            bail!("没有提供任何要修订的内容（text / sources / related / supersedes / summary 至少给一项）");
        }
          if req.heading.is_some() && req.text.is_none() {
              bail!("heading 只能与 text 一起用于 section 修订");
          }
          // An empty text would splice away a whole section (or body); an
          // empty heading would silently turn a section edit into a full
          // body replacement. Both must fail before any write happens.
          if let Some(new_text) = &req.text {
              if new_text.trim().is_empty() {
                  bail!("text 不能为空：整节或整篇替换会清空正文");
              }
          }
          if let Some(h) = &req.heading {
              if h.trim().is_empty() {
                  bail!("heading 不能为空：要替换整篇正文请省略 heading");
              }
          }
        let (rel, abs) = self.resolve_write_target(&req.target)?;
        let gate = self.lock_for(&rel);
        let _guard = gate.lock().await;
        let text = std::fs::read_to_string(&abs).with_context(|| format!("读取失败: {rel}"))?;
        self.ensure_wiki_scope(&rel, &text)?;

        let before = content_hash(&text);
        if before != req.expected_version {
            return Ok(json!({
                "ok": false,
                "conflict": true,
                "path": rel,
                "current_version": before,
                "error": "expected_version 不匹配；请重新 vault_read 后重试",
            }));
        }

        let mut edited = text.clone();
        let mut changes: Vec<String> = Vec::new();

        if let Some(srcs) = &req.sources {
            if srcs.is_empty() {
                bail!("sources 结果不能为空：wiki 页至少要保留一条有效来源");
            }
            let mut items: Vec<String> = Vec::with_capacity(srcs.len());
            for source in srcs {
                let item = format!("[[{}]]", normalize_source(source, &self.index, false)?);
                if !items.contains(&item) {
                    items.push(item);
                }
            }
            edited = set_frontmatter(&edited, "sources", &serde_json::to_string(&items)?)?;
            changes.push(format!("sources={}", items.len()));
        }
        if let Some(links) = &req.related {
            let mut items: Vec<String> = Vec::with_capacity(links.len());
            for link in links {
                let item = format!("[[{}]]", normalize_source(link, &self.index, true)?);
                if !items.contains(&item) {
                    items.push(item);
                }
            }
            edited = set_frontmatter(&edited, "related", &serde_json::to_string(&items)?)?;
            changes.push(format!("related={}", items.len()));
        }
        if let Some(retired) = &req.supersedes {
            let mut items: Vec<String> = Vec::with_capacity(retired.len());
            for link in retired {
                let item = format!("[[{}]]", normalize_source(link, &self.index, true)?);
                if !items.contains(&item) {
                    items.push(item);
                }
            }
            edited = set_frontmatter(&edited, "supersedes", &serde_json::to_string(&items)?)?;
            changes.push(format!("supersedes={}", items.len()));
        }
        if let Some(new_text) = &req.text {
            match req.heading.as_deref() {
                Some(h) if !h.trim().is_empty() => {
                    edited = replace_section(&edited, h, new_text)?;
                    changes.push(format!("section:{}", h.trim()));
                }
                _ => {
                    edited = replace_body(&edited, new_text)?;
                    changes.push("body".to_string());
                }
            }
        }
        // A summary is a create-time snapshot. Body rewrites must refresh it,
        // otherwise `vault_resolve` previews keep advertising an older page.
        if let Some(sum) = &req.summary {
            let sum = sum.trim();
            if sum.is_empty() {
                bail!("summary 不能为空：wiki 页必须有一行摘要");
            }
            edited = set_frontmatter(&edited, "summary", sum)?;
            changes.push("summary".to_string());
        }
        let summary_not_updated = req.text.is_some() && req.summary.is_none();

        let revision = frontmatter_value(&text, "revision")
            .and_then(|v| v.trim().trim_matches('"').parse::<u64>().ok())
            .unwrap_or(0)
            + 1;
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        edited = set_frontmatter(&edited, "updated", &today)?;
        edited = set_frontmatter(&edited, "revision", &revision.to_string())?;
        edited = set_frontmatter(&edited, "verified", "false")?;
        validate_frontmatter(&edited)?;

        atomic_write(&abs, &edited)?;
        self.index.upsert(&rel)?;
        let after = content_hash(&edited);
        tracing::info!(
            "write revise {rel} | reason={reason} | changes={} | {before} -> {after}",
            changes.join(",")
        );
        let mut audit_changes = changes.clone();
        if summary_not_updated {
            audit_changes.push("summary_not_updated".to_string());
        }
        self.audit_write(
            "rewrite",
            &req.target,
            &rel,
            Some(reason),
            Some(&before),
            &after,
            audit_changes,
        );
        Ok(json!({
            "ok": true,
            "path": rel,
            "revised": changes,
            "summary_not_updated": summary_not_updated,
            "reason": reason,
            "revision": revision,
            "updated": today,
            "verified": false,
            "version_before": before,
            "version_after": after,
        }))
    }

    /// Read-only lint. Always available, regardless of the writes switch.
    pub fn lint(&self, limit: i64) -> Result<Value> {
        let limit = limit.clamp(1, 200) as usize;
        let root = self.index.root().to_path_buf();
        let mut scanned = 0usize;
        let mut total = 0usize;
        let mut issues: Vec<Value> = Vec::new();

        let walker = WalkDir::new(&root).into_iter().filter_entry(|e| {
            if e.depth() == 0 {
                return true;
            }
            let rel = e
                .path()
                .strip_prefix(&root)
                .unwrap_or(e.path())
                .to_string_lossy()
                .replace('\\', "/");
            !self.index.is_denied(&rel)
        });

        for entry in walker {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            let rel = path
                .strip_prefix(&root)?
                .to_string_lossy()
                .replace('\\', "/");
            if self.index.is_denied(&rel) {
                continue;
            }
            let text = std::fs::read_to_string(path).unwrap_or_default();
            scanned += 1;

            let mut problems = Vec::new();
            if let Err(err) = validate_frontmatter(&text) {
                problems.push(err.to_string());
            }
            let fm_block: Vec<&str> = text
                .lines()
                .skip(1)
                .take_while(|l| l.trim_end() != "---")
                .collect();
            for field in ["title", "layer"] {
                if !fm_block
                    .iter()
                    .any(|l| l.starts_with(&format!("{field}:")))
                {
                    problems.push(format!("缺字段 {field}"));
                }
            }

            for problem in problems {
                total += 1;
                if issues.len() < limit {
                    issues.push(json!({"path": rel, "issue": problem}));
                }
            }
        }
        Ok(json!({
            "scanned": scanned,
            "issue_count": total,
            "returned": issues.len(),
            "truncated": total > issues.len(),
            "issues": issues,
        }))
    }
}

fn yaml_scalar(value: &str) -> String {
    let t = value.trim();
    // Only a real JSON array (used by sources/related/supersedes) may stay
    // unquoted. A scalar like "[已完成]" must not become a YAML sequence.
    if t.starts_with('[')
        && t.ends_with(']')
        && serde_json::from_str::<Vec<serde_json::Value>>(t).is_ok()
    {
        return t.to_string();
    }
    if matches!(t, "true" | "false" | "null" | "~") || t.parse::<f64>().is_ok() {
        return t.to_string();
    }
    serde_json::to_string(value)
        .unwrap_or_else(|_| format!("\"{}\"", value.replace('"', "\\\"")))
}

fn validate_frontmatter(text: &str) -> Result<()> {
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.first().map(|l| l.trim_end()) != Some("---") {
        bail!("缺少 frontmatter 起始 ---");
    }
    let end = lines
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, l)| l.trim_end() == "---")
        .map(|(i, _)| i)
        .context("frontmatter 未闭合")?;
    let mut seen = HashSet::new();
    for line in &lines[1..end] {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || line.starts_with(' ')
            || line.starts_with('\t')
            || trimmed.starts_with('-')
        {
            continue;
        }
        if let Some((key, _)) = line.split_once(':') {
            let key = key.trim();
            if !seen.insert(key.to_string()) {
                bail!("重复 frontmatter 键: {key}");
            }
        }
    }
    Ok(())
}

fn set_frontmatter(text: &str, field: &str, value: &str) -> Result<String> {
    if field.is_empty()
        || !field
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        bail!("非法 frontmatter 字段名: {field}");
    }
    let lines: Vec<&str> = text.split('\n').collect();
    let end = lines
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, l)| l.trim_end() == "---")
        .map(|(i, _)| i)
        .context("frontmatter 未闭合")?;
    let scalar = yaml_scalar(value);
    let mut out: Vec<String> = Vec::with_capacity(lines.len() + 1);
    let mut found = false;
    for (i, line) in lines.iter().enumerate() {
        let top_level = i > 0
            && i < end
            && !line.starts_with(' ')
            && !line.starts_with('\t')
            && !line.trim_start().starts_with('-');
        if top_level {
            if let Some((key, _)) = line.split_once(':') {
                if key.trim() == field {
                    out.push(format!("{field}: {scalar}"));
                    found = true;
                    continue;
                }
            }
        }
        out.push((*line).to_string());
    }
    if !found {
        out.insert(end, format!("{field}: {scalar}"));
    }
    Ok(out.join("\n"))
}

fn append_at_end(text: &str, value: &str) -> Result<String> {
    let mut out = text.to_string();
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(value);
    if !value.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

/// Mark lines inside fenced code blocks. A `#` in a shell/JS snippet must
/// never be treated as a Markdown heading when locating section boundaries.
fn code_line_mask(lines: &[&str]) -> Vec<bool> {
    let mut mask = vec![false; lines.len()];
    let mut in_fence = false;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            mask[i] = true;
            in_fence = !in_fence;
            continue;
        }
        mask[i] = in_fence;
    }
    mask
}

fn frontmatter_end(lines: &[&str]) -> Result<usize> {
    lines
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, l)| l.trim_end() == "---")
        .map(|(i, _)| i)
        .context("frontmatter 未闭合")
}

fn append_to_section(text: &str, heading: &str, value: &str) -> Result<String> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mask = code_line_mask(&lines);
    let fm_end = frontmatter_end(&lines)?;
    let wanted = heading.trim().trim_start_matches('#').trim().to_lowercase();
    let start = lines
        .iter()
        .enumerate()
        .skip(fm_end + 1)
        .find(|(i, l)| {
            if mask[*i] {
                return false;
            }
            let t = l.trim_start();
            t.starts_with('#')
                && t.trim_start_matches('#').trim().to_lowercase() == wanted
        })
        .map(|(i, _)| i)
        .with_context(|| format!("找不到标题: {heading}"))?;
    let level = lines[start].chars().take_while(|c| *c == '#').count();
    let mut insert_at = lines.len();
    for i in (start + 1)..lines.len() {
        if mask[i] {
            continue;
        }
        let t = lines[i].trim_start();
        if t.starts_with('#') {
            let lvl = t.chars().take_while(|c| *c == '#').count();
            if lvl <= level {
                insert_at = i;
                break;
            }
        }
    }
    let mut out: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    let mut block = value.to_string();
    if !block.ends_with('\n') {
        block.push('\n');
    }
    out.insert(insert_at, block);
    Ok(out.join("\n"))
}

fn apply_edit(
    text: &str,
    part: &str,
    key: Option<&str>,
    op: &str,
    value: &str,
) -> Result<String> {
    match (part, op) {
        ("frontmatter", "set") => set_frontmatter(
            text,
            key.filter(|k| !k.is_empty())
                .context("frontmatter set 需要 key")?,
            value,
        ),
        ("frontmatter", "append") => {
            let field = key
                .filter(|k| !k.is_empty())
                .context("frontmatter append 需要 key")?;
            if !matches!(field, "sources" | "related") {
                bail!("只有 sources / related 支持追加: {field}");
            }
            append_frontmatter_list(text, field, value)
        }
        ("end", "append") => append_at_end(text, value),
        ("section", "append") => append_to_section(
            text,
            key.filter(|k| !k.is_empty())
                .context("section append 需要 key（标题文本）")?,
            value,
        ),
        _ => bail!("不支持的 part/op 组合: {part}/{op}"),
    }
}

/// Replace everything after the frontmatter with a freshly compiled body.
/// Frontmatter (including sources) is preserved byte-for-byte.
fn replace_body(text: &str, new_body: &str) -> Result<String> {
    let lines: Vec<&str> = text.split('\n').collect();
    let end = frontmatter_end(&lines)?;
    let mut out = lines[..=end].join("\n");
    out.push_str("\n\n");
    out.push_str(new_body.trim_end());
    out.push('\n');
    Ok(out)
}

/// Replace the content of one heading section, leaving every other section and
/// the whole frontmatter untouched. Headings inside the frontmatter are ignored.
fn replace_section(text: &str, heading: &str, value: &str) -> Result<String> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mask = code_line_mask(&lines);
    let fm_end = frontmatter_end(&lines)?;
    let wanted = heading.trim().trim_start_matches('#').trim().to_lowercase();
    let start = lines
        .iter()
        .enumerate()
        .skip(fm_end + 1)
        .find(|(i, l)| {
            if mask[*i] {
                return false;
            }
            let t = l.trim_start();
            t.starts_with('#') && t.trim_start_matches('#').trim().to_lowercase() == wanted
        })
        .map(|(i, _)| i)
        .with_context(|| format!("找不到正文标题: {heading}"))?;
    let level = lines[start].chars().take_while(|c| *c == '#').count();
    let mut stop = lines.len();
    for i in (start + 1)..lines.len() {
        if mask[i] {
            continue;
        }
        let t = lines[i].trim_start();
        if t.starts_with('#') {
            let lvl = t.chars().take_while(|c| *c == '#').count();
            if lvl <= level {
                stop = i;
                break;
            }
        }
    }
    let replacement: Vec<String> = value
        .trim_end()
        .split('\n')
        .map(|s| s.to_string())
        .collect();
    let mut out: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    out.splice((start + 1)..stop, replacement);
    if out.last().is_none_or(|s| !s.is_empty()) {
        out.push(String::new());
    }
    Ok(out.join("\n"))
}

fn atomic_write(abs: &Path, text: &str) -> Result<()> {
    let name = abs
        .file_name()
        .and_then(|s| s.to_str())
        .context("缺少文件名")?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let tmp = abs.with_file_name(format!(".{name}.lwd-{}-{nanos}.tmp", std::process::id()));
    std::fs::write(&tmp, text.as_bytes()).with_context(|| format!("写临时文件失败: {}", tmp.display()))?;
    if let Err(err) = std::fs::rename(&tmp, abs) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err).with_context(|| format!("原子替换失败: {}", abs.display()));
    }
    Ok(())
}

const WIKI_KINDS: [&str; 5] = ["concept", "topic", "reference", "entity", "synthesis"];
const CONFIDENCE_LEVELS: [&str; 3] = ["high", "medium", "low"];

/// Read one top-level frontmatter scalar without a full YAML parser.
fn frontmatter_value(text: &str, field: &str) -> Option<String> {
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.first().map(|l| l.trim_end()) != Some("---") {
        return None;
    }
    let end = lines
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, l)| l.trim_end() == "---")
        .map(|(i, _)| i)?;
    for line in &lines[1..end] {
        if line.starts_with(' ') || line.starts_with('\t') {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim() == field {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

fn frontmatter_layer(text: &str) -> Option<String> {
    frontmatter_value(text, "layer")
        .map(|v| v.trim_matches('"').trim_matches('\'').trim().to_string())
}

/// Provenance uses the SAME wikilink addressing contract as reads: the raw
/// `[[...]]` is resolved through the link index, so `[[Note]]`, aliases and
/// bare names all work. `sources` must point to raw/note; `related` may also
/// point to another wiki page. This stops fabricated or ambiguous provenance.
fn normalize_source(raw: &str, index: &VaultIndex, allow_wiki: bool) -> Result<String> {
    if raw.trim().is_empty() {
        bail!("sources 条目为空");
    }
    let res = index.resolve(raw)?;
    let rel = res.path.with_context(|| {
        let hint = res
            .preview
            .get("candidates")
            .map(|c| format!("；请用候选 target 重试：{c}"))
            .unwrap_or_default();
        format!("引用无法解析: {raw}{hint}")
    })?;
    if index.is_denied(&rel) {
        bail!("引用指向排除区: {raw}");
    }
    let layer = res
        .preview
        .get("layer")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let ok = matches!(layer, "raw" | "note") || (allow_wiki && layer == "wiki");
    if !ok {
        bail!(
            "引用必须指向层 {} 的笔记（{raw} 为 {layer:?}）",
            if allow_wiki { "raw/note/wiki" } else { "raw/note" }
        );
    }
    Ok(rel)
}

/// Append one item to a flow-style YAML list such as sources: ["[[a]]"].
fn append_frontmatter_list(text: &str, field: &str, value: &str) -> Result<String> {
    let lines: Vec<&str> = text.split('\n').collect();
    let end = lines
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, l)| l.trim_end() == "---")
        .map(|(i, _)| i)
        .context("frontmatter 未闭合")?;
    let prefix = format!("{field}:");
    let pos = lines[1..end]
        .iter()
        .position(|l| l.starts_with(&prefix))
        .with_context(|| format!("找不到 {field} 字段"))?
        + 1;
    let raw = lines[pos].split_once(':').unwrap().1.trim();
    let mut items: Vec<String> = serde_json::from_str(raw)
        .with_context(|| format!("{field} 不是可追加的流式数组"))?;
    let item = if value.starts_with("[[") {
        value.trim().to_string()
    } else {
        format!("[[{}]]", value.trim())
    };
    if items.iter().any(|x| x == &item) {
        return Ok(text.to_string());
    }
    items.push(item);
    let mut out: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    out[pos] = format!("{field}: {}", serde_json::to_string(&items)?);
    Ok(out.join("\n"))
}

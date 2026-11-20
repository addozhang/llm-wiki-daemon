use crate::index::VaultIndex;
use crate::store::{
    AppendRequest, CreateRequest, LintRequest, PatchRequest, RewriteRequest, WRITE_TOOLS,
    WriteStore,
};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, ListToolsResult, PaginatedRequestParams, ResultType},
    service::RequestContext,
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchRequest {
    /// Keywords to look for in title, summary, tags or note body.
    pub query: String,
    /// Optional layer filter: raw, note or wiki.
    #[serde(default)]
    pub layer: Option<String>,
    /// Optional status filter: inbox, draft, published or archived.
    #[serde(default)]
    pub status: Option<String>,
    /// Maximum number of hits, between 1 and 100.
    #[serde(default = "default_limit")]
    pub limit: i64,
}

fn default_limit() -> i64 {
    10
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadRequest {
    /// An Obsidian wikilink as written ([[Note|alias]]), a bare note name, or a
    /// vault-relative path. Pass the wikilink verbatim; do not convert it to a
    /// path. `path` is accepted as a legacy alias for old clients.
    #[serde(alias = "path")]
    pub target: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResolveRequest {
    /// Wikilinks to preview in one call. Returns indexed metadata (title,
    /// summary, layer, kind, status, version) and resolved/ambiguous flags —
    /// never note bodies. Use this to decide which links deserve a full read.
    pub targets: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StatusRequest {}

#[derive(Clone)]
pub struct VaultServer {
    #[allow(dead_code, reason = "tool_handler macro reads this router field")]
    tool_router: ToolRouter<Self>,
    index: Arc<VaultIndex>,
    store: Arc<WriteStore>,
    wiki_root: String,
}

fn text_result(value: serde_json::Value) -> CallToolResult {
    let body = serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string());
    CallToolResult::success(vec![ContentBlock::text(body)])
}

fn store_result(result: anyhow::Result<serde_json::Value>) -> Result<CallToolResult, McpError> {
    match result {
        Ok(value) => Ok(text_result(value)),
        Err(err) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
            "{err:#}"
        ))])),
    }
}

fn writes_disabled() -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![ContentBlock::text(
        "写功能已被紧急总闸关闭：config.toml [writes] enabled = false",
    )]))
}

#[tool_router]
impl VaultServer {
    pub fn new(index: Arc<VaultIndex>, store: Arc<WriteStore>, wiki_root: String) -> Self {
        Self {
            tool_router: Self::tool_router(),
            index,
            store,
            wiki_root,
        }
    }

    #[tool(description = "Search vault notes through the derived index (FTS5 + CJK trigram + substring fallback), fused by RRF rank. Returns path, title, summary, layer, status, tags, date, score (RRF, higher = more relevant), bm25 (raw per-index value) and a short snippet with **highlighted** matches for relevance triage. Prefer several precise short queries over one long sentence; use layer/status to narrow scope.")]
    async fn vault_search(
        &self,
        Parameters(req): Parameters<SearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        match self.index.search(
            &req.query,
            req.layer.as_deref(),
            req.status.as_deref(),
            req.limit,
        ) {
            Ok(hits) => Ok(text_result(serde_json::json!({
                "count": hits.len(),
                "hits": hits,
            }))),
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(
                e.to_string(),
            )])),
        }
    }

    #[tool(description = "Read one note by an Obsidian wikilink ([[Note|alias]]), a bare note name, or a vault-relative path. Pass the wikilink verbatim - the daemon resolves it through its link index. Returns frontmatter, body, a version hash, `links` (outgoing wikilinks + indexed metadata) and `backlinks` (notes referencing this page). Ambiguous links return candidate targets to retry with.")]
    async fn vault_read(
        &self,
        Parameters(req): Parameters<ReadRequest>,
    ) -> Result<CallToolResult, McpError> {
        match self.index.read(&req.target) {
            Ok(value) => Ok(text_result(value)),
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(
                e.to_string(),
            )])),
        }
    }

    #[tool(description = "PREVIEW wikilinks in bulk without reading note bodies. Give the [[wikilinks]] as written; returns title, summary, layer, kind, status, version and resolved/ambiguous flags for each. Use this to decide which links are worth a full vault_read. Ambiguous links come back with candidate `target` values to retry with.")]
    async fn vault_resolve(
        &self,
        Parameters(req): Parameters<ResolveRequest>,
    ) -> Result<CallToolResult, McpError> {
        match self.index.preview(&req.targets) {
            Ok(results) => Ok(text_result(serde_json::json!({
                "count": results.len(),
                "resolved": results.iter().filter(|v| v.get("resolved") == Some(&serde_json::json!(true))).count(),
                "results": results,
            }))),
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(
                e.to_string(),
            )])),
        }
    }

    #[tool(description = "Report daemon health: indexed note count, vault root, schema presence, index database path, watcher counters and whether writes are enabled.")]
    async fn vault_status(
        &self,
        Parameters(_req): Parameters<StatusRequest>,
    ) -> Result<CallToolResult, McpError> {
        match self.index.status() {
            Ok(mut value) => {
                if let Some(obj) = value.as_object_mut() {
                    obj.insert(
                        "daemon".to_string(),
                        serde_json::json!(env!("CARGO_PKG_VERSION")),
                    );
                    obj.insert("protocol".to_string(), serde_json::json!("MCP Streamable HTTP"));
                    obj.insert(
                        "writes_enabled".to_string(),
                        serde_json::json!(self.store.enabled()),
                    );
                    obj.insert(
                        "mode".to_string(),
                        serde_json::json!(if self.store.enabled() {
                            "read-write"
                        } else {
                            "read-only"
                        }),
                    );
                    obj.insert(
                        "wiki_root".to_string(),
                        serde_json::json!(self.wiki_root),
                    );
                }
                Ok(text_result(value))
            }
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(
                e.to_string(),
            )])),
        }
    }

    #[tool(description = "PATCH a wiki page addressed by target (a wikilink such as [[Note]]; `path` is a legacy alias). `part` selects what to change: frontmatter (op=set for scalars, op=append for sources/related), section (op=append, key=heading) or end (op=append). Requires expected_version from vault_read; conflicts are returned, never silently overwritten. layer / revision / updated are immutable here; use vault_rewrite for provenance replacements.")]
    async fn vault_patch(
        &self,
        Parameters(req): Parameters<PatchRequest>,
    ) -> Result<CallToolResult, McpError> {
        if !self.store.enabled() {
            return writes_disabled();
        }
        store_result(self.store.patch(req, "patch").await)
    }

    #[tool(description = "Append text to a wiki page addressed by target (a wikilink such as [[Note]]; `path` is a legacy alias), at EOF or under a given heading. Requires expected_version from vault_read.")]
    async fn vault_append(
        &self,
        Parameters(req): Parameters<AppendRequest>,
    ) -> Result<CallToolResult, McpError> {
        if !self.store.enabled() {
            return writes_disabled();
        }
        store_result(self.store.append(req).await)
    }

    #[tool(description = "Bootstrap primitive: create a NEW wiki page under the configured wiki root (see vault_status.wiki_root) with layer: wiki. Requires kind, summary and non-empty sources pointing to existing raw/note notes. Existing files are never overwritten.")]
    async fn vault_create(
        &self,
        Parameters(req): Parameters<CreateRequest>,
    ) -> Result<CallToolResult, McpError> {
        if !self.store.enabled() {
            return writes_disabled();
        }
        store_result(self.store.create(req).await)
    }

    #[tool(description = "REVISE a wiki page addressed by target (a wikilink such as [[Note]]; `path` is a legacy alias) when content or provenance is wrong. text replaces the whole body, or one section when heading is given. sources / related / supersedes are full replacements, so wrong or stale entries can be REMOVED (sources must stay non-empty). summary is an optional one-line replacement: supply it whenever the body changes, covering every H2 section, or the response will report summary_not_updated=true. Requires a reason for the audit log and expected_version from vault_read. Automatically writes updated / revision and resets verified to false; layer is untouched. Never use append to add a correction note - fix the wrong content itself.")]
    async fn vault_rewrite(
        &self,
        Parameters(req): Parameters<RewriteRequest>,
    ) -> Result<CallToolResult, McpError> {
        if !self.store.enabled() {
            return writes_disabled();
        }
        store_result(self.store.rewrite(req).await)
    }

    #[tool(description = "Read-only lint: missing frontmatter, unclosed frontmatter, duplicate keys, missing title/layer fields, and orphan wiki pages with no incoming links.")]
    async fn vault_lint(
        &self,
        Parameters(req): Parameters<LintRequest>,
    ) -> Result<CallToolResult, McpError> {
        store_result(self.store.lint(req.limit))
    }
}

/// Delivered on `initialize`, so every session that connects to this MCP
/// server gets the wiki contract without any global stub or cwd dependency.
/// Keep it short: dsh-mcp-client injects it into the model context (limit 32 KB).
#[tool_handler(
    name = "llm-wiki-daemon",
    instructions = "LLM Wiki Daemon — 人机共享的 Markdown 知识库。vault 是唯一真相源，SQLite 索引是可随时重建的派生缓存；机器检索只走下列工具，不存在需要整份阅读的静态目录文件。

【默认：Query，只读】
1. 先看已编译结论：vault_search(query, layer=wiki)；不够再回退全库 raw/note。查询策略：中文或长句先拆成 2-4 个精确短词分别搜、再合并去重，别把整句丢进一次 query；已知归属时用 layer / status 收窄范围，能显著降低无关召回。
2. 用 vault_resolve 预览命中项，以及 vault_read 返回的 links / frontmatter_all.sources（均为 wikilink 数组），按命中片段 snippet（**高亮**了命中词）和一行摘要判断相关性，只对选中的原始 wikilink 调 vault_read；默认只展开一层、不递归。snippet 只是定位线索、上下文有限，下结论前仍须 vault_read 全文，不要仅凭 snippet 回答。
3. 回答带 [[引用]]（vault 相对路径）。
4. 收尾自检：本次是否产生了综合、对比、纠错、决策或新的因果联系？若有，向人提议「是否回写为 wiki 页」，经确认才写。默认不写 —— 不要把每次问答都写进 wiki，也不要让有价值的结论只留在对话里。

【人显式触发：Ingest】
仅当人给出新来源或明确要求 ingest 时执行：
1. vault_read 读该来源，先向人汇报要点、确认侧重，不要直接动笔。
2. vault_search(layer=wiki) 找出所有相关页 —— 一个来源可能触及多页；逐页判断新增还是修订。
3. 增量优先用 vault_create / vault_patch / vault_append；结论被新来源推翻才用 vault_rewrite 重编译整节或整篇，并在 supersedes 记录旧结论。每页都要把该来源登记进 sources。
4. **改正文必须同步 summary**：summary 是创建时的快照，重编译后要重读全文、按所有 `##` 章节重写一行摘要，覆盖全篇而非只概括开头；用 vault_rewrite 的 summary 参数或 vault_patch 更新。只改正文不给 summary 时，daemon 会返回 `summary_not_updated: true` 并在审计里留痕，看到后必须补上。摘要的验收标准是「读者能否据此判断这页讲什么、值不值得读全文」。
5. 逐页报告改了什么；发现错误来源或错误正文直接改掉，不要保留、不要追加更正说明。

【周期性、人触发：Lint】
只报告、不自动改。可用 vault_lint 做 frontmatter 体检；语义清单：页面间矛盾、被新来源取代的旧结论、无入链孤儿页、重要概念缺页、缺失交叉引用、**summary 覆盖度与 stale**（是否覆盖所有 `##` 章节、正文改过但摘要逐字未动、有无完全没被提到的 H2）、可由新来源补齐的数据缺口；顺带提出值得继续追问的问题。经人确认后用 vault_rewrite 修正并写 supersedes。

【寻址与写入铁律】
wikilink 是全链路唯一寻址：把 [[...]] 原文交给各工具的 target（vault_create.path 是唯一例外，表示新页位置）；vault_patch 的部位参数叫 part。vault_search 也返回可直接回喂的 target；resolved=false 是断链，ambiguous=true 时用 candidates[].target 重试，绝不自造路径、不拼 .md、不猜目录。
写工具常开但只作用于 wiki_root 下且 layer: wiki 的页面（wiki_root 见 vault_status）；raw / note 只读，归档/ 不动。每次写必须带 vault_read 的最新 expected_version，收到 conflict 就重读重试；vault_rewrite 必填 reason，其余写操作建议填。layer / revision / updated 不可手工改，verified 只能人工置 true。系统不提供删除或通用整文件覆盖工具。操作时间线在 daemon 的 audit.jsonl 与 wiki 页的 updated / revision / sources 中；不存在需要维护的静态 log 文件。"
)]
impl ServerHandler for VaultServer {
    /// Write tools are absent from tools/list unless explicitly enabled. The
    /// tool implementations also re-check the switch, so a direct tools/call
    /// cannot bypass this.
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let mut tools = self.tool_router.list_all();
        if !self.store.enabled() {
            tools.retain(|t| !WRITE_TOOLS.contains(&t.name.as_ref()));
        }
        Ok(ListToolsResult {
            result_type: Some(ResultType::COMPLETE),
            tools,
            meta: None,
            next_cursor: None,
            ttl_ms: None,
            cache_scope: None,
        })
    }
}

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params, params_from_iter, types::Value as SqlValue};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub path: String,
    /// The wikilink that addresses this note, ready to pass to vault_read.
    pub target: String,
    pub title: String,
    pub summary: String,
    pub layer: String,
    pub status: String,
    pub tags: String,
    pub date: String,
    /// RRF fusion score for cross-index ranking (higher = more relevant).
    /// Ranks, not raw bm25, are fused because the unicode61 and trigram
    /// tables have non-comparable IDF statistics.
    pub score: f64,
    /// Original bm25 from whichever index first surfaced this note (negative
    /// = more relevant). Kept for debugging; not comparable across indexes.
    pub bm25: f64,
    /// Bounded context window around the first match, with **highlight**.
    /// Lets the agent decide relevance without a full vault_read.
    pub snippet: String,
}

#[derive(Debug, Clone, Default)]
struct Frontmatter {
    title: String,
    date: String,
    tags: Vec<String>,
    layer: String,
    summary: String,
    status: String,
    kind: String,
    aliases: Vec<String>,
}

impl Frontmatter {
    fn set(&mut self, key: &str, value: &str) {
        let v = value.trim().trim_matches('"').trim_matches('\'').trim();
        match key {
            "title" => self.title = v.to_string(),
            "date" => self.date = v.to_string(),
            "layer" => self.layer = v.to_string(),
            "summary" => self.summary = v.to_string(),
            "status" => self.status = v.to_string(),
            "kind" => self.kind = v.to_string(),
            "tags" | "aliases" => {
                let inner = v.trim_start_matches('[').trim_end_matches(']');
                let dest = if key == "tags" {
                    &mut self.tags
                } else {
                    &mut self.aliases
                };
                for part in inner.split(',') {
                    let t = part.trim().trim_matches('"').trim_matches('\'').trim();
                    if !t.is_empty() && !dest.contains(&t.to_string()) {
                        dest.push(t.to_string());
                    }
                }
            }
            _ => {}
        }
    }
}

/// All names an Obsidian wikilink may use to address one note. Keys are
/// lower-cased and stripped of the `.md` suffix, so `[[Note]]`, `[[Note.md]]`,
/// `[[path/Note]]` and frontmatter `aliases` all land on the same row.
fn ref_keys(rel: &str, fm: &Frontmatter) -> Vec<(String, &'static str)> {
    let no_ext = rel.strip_suffix(".md").unwrap_or(rel);
    let stem = no_ext.rsplit('/').next().unwrap_or(no_ext);
    let mut out: Vec<(String, &'static str)> = Vec::new();
    for (key, kind) in [
        (no_ext.to_lowercase(), "path"),
        (stem.to_lowercase(), "stem"),
    ] {
        if !key.is_empty() && !out.iter().any(|(k, _)| k == &key) {
            out.push((key, kind));
        }
    }
    for alias in &fm.aliases {
        let key = alias.to_lowercase();
        if !key.is_empty() && !out.iter().any(|(k, _)| k == &key) {
            out.push((key, "alias"));
        }
    }
    out
}

/// Minimal deterministic frontmatter reader for the schema fields plus
/// `aliases` / `kind`. A full YAML crate is intentionally deferred.
fn parse_note(text: &str) -> (Frontmatter, String) {
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.first().map(|l| l.trim_end()) != Some("---") {
        return (Frontmatter::default(), text.to_string());
    }
    let Some(end) = lines
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, l)| l.trim_end() == "---")
        .map(|(i, _)| i)
    else {
        return (Frontmatter::default(), text.to_string());
    };

    let mut fm = Frontmatter::default();
    let mut current: Option<String> = None;
    for line in &lines[1..end] {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let is_continuation = line.starts_with(' ') || line.starts_with('\t') || trimmed.starts_with("- ");
        if is_continuation {
            if let Some(k) = &current {
                let v = trimmed.strip_prefix("- ").unwrap_or(trimmed);
                fm.set(k, v);
            }
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim().to_string();
            current = Some(key.clone());
            if !v.trim().is_empty() {
                fm.set(&key, v.trim());
            }
        }
    }
    (fm, lines[end + 1..].join("\n"))
}

/// Stable content hash used as the CAS `version` and for index dedup.
/// FNV-1a 64-bit over UTF-8 bytes: unlike `DefaultHasher`, its output is
/// specified and will not change across Rust compiler versions, so an
/// upgrade cannot invalidate every in-flight `expected_version`.
pub fn content_hash(s: &str) -> String {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(PRIME);
    }
    format!("{:016x}", h)
}

/// Detect CJK code points in a query. Trigram tokenisation only helps for
/// these; Latin queries stay on the original unicode61 + substring path.
fn has_cjk(s: &str) -> bool {
    s.chars().any(|c| {
        ('\u{3400}'..='\u{9fff}').contains(&c)   // CJK Unified (incl. Ext A)
            || ('\u{f900}'..='\u{faff}').contains(&c) // Compatibility Ideographs
            || ('\u{3040}'..='\u{30ff}').contains(&c) // Hiragana / Katakana
            || ('\u{ac00}'..='\u{d7af}').contains(&c) // Hangul
    })
}

/// Bounded context window around the first matching term in a note body.
/// Generated in Rust rather than via FTS5 `snippet()` so every retrieval path
/// (FTS5, trigram, substring fallback) produces the same shape of evidence,
/// and so CJK matches — which unicode61 cannot tokenise — still get a window.
/// ASCII case-insensitive; CJK compared literally. Output is whitespace-flat
/// and capped at `max_chars` characters (characters, not bytes).
fn make_snippet(body: &str, query: &str, max_chars: usize) -> String {
    let flat: String = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.is_empty() {
        return String::new();
    }
    let lower = flat.to_lowercase();
    // Prefer the longest quoted phrase, then individual words: find the
    // earliest occurrence among all query terms.
    let mut best: Option<(usize, usize)> = None; // (byte offset, term length)
    let terms: Vec<&str> = query
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .collect();
    for term in &terms {
        let needle = term.to_lowercase();
        if needle.is_empty() {
            continue;
        }
        if let Some(pos) = lower.find(&needle) {
            if best.is_none_or(|(b, _)| pos < b) {
                best = Some((pos, needle.len()));
            }
        }
    }
    let chars: Vec<char> = flat.chars().collect();
    let (start, end) = match best {
        Some((byte_pos, term_len)) => {
            let center = flat[..byte_pos].chars().count();
            let half = max_chars / 2;
            let s = center.saturating_sub(half);
            let e = (center + term_len.max(max_chars / 2)).min(chars.len());
            (s, e)
        }
        None => (0, chars.len().min(max_chars)),
    };

    // Collect every query-term occurrence inside the window, in char offsets,
    // and highlight all of them. A multi-word query should show the agent
    // where EACH term landed, not just the earliest one.
    let mut marks: Vec<(usize, usize)> = Vec::new(); // (char start, char end)
    for term in &terms {
        let needle = term.to_lowercase();
        if needle.is_empty() {
            continue;
        }
        let mut search_from = 0usize;
        while let Some(rel) = lower[search_from..].find(&needle) {
            let byte_pos = search_from + rel;
            let cs = flat[..byte_pos].chars().count();
            let ce = cs + needle.chars().count();
            if cs >= end {
                break;
            }
            if ce > start {
                marks.push((cs.max(start), ce.min(end)));
            }
            search_from = byte_pos + needle.len();
        }
    }
    marks.sort_unstable();
    marks.dedup();
    // Merge overlapping ranges (e.g. "network" and "network policy").
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (cs, ce) in marks {
        if let Some(last) = merged.last_mut() {
            if cs <= last.1 {
                last.1 = last.1.max(ce);
                continue;
            }
        }
        merged.push((cs, ce));
    }

    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    let mut mi = 0usize;
    for i in start..end {
        while mi < merged.len() && merged[mi].1 <= i {
            mi += 1;
        }
        if merged.get(mi).is_some_and(|(s, _)| *s == i) {
            out.push_str("**");
        }
        out.push(chars[i]);
        if merged.get(mi).is_some_and(|(_, e)| *e == i + 1) {
            out.push_str("**");
        }
    }
    if end < chars.len() {
        out.push('…');
    }
    out
}

/// Reciprocal Rank Fusion constant. Borrowed from the standard RRF formula.
const RRF_K: f64 = 60.0;

/// Fuse several ranked result lists into one ordering by RANK, not by score.
///
/// The unicode61 and trigram FTS tables have different IDF statistics, so a
/// bm25 of -7 in one is not comparable to -7 in the other. RRF sidesteps this:
/// a hit's contribution is `weight / (K + rank)`, which only depends on where
/// it placed in each list. The substring fallback is down-weighted so it can
/// top up recall without outranking a real index hit.
fn rrf_fuse(lists: &[(Vec<SearchHit>, f64)], limit: usize) -> Vec<SearchHit> {
    let mut fused: HashMap<String, (SearchHit, f64)> = HashMap::new();
    for (hits, weight) in lists {
        for (rank, hit) in hits.iter().enumerate() {
            let contribution = weight / (RRF_K + rank as f64 + 1.0);
            match fused.get_mut(&hit.path) {
                Some((_, total)) => *total += contribution,
                None => {
                    fused.insert(hit.path.clone(), (hit.clone(), contribution));
                }
            }
        }
    }
    let mut out: Vec<(SearchHit, f64)> = fused.into_values().collect();
    // Higher fused score first; stable tie-break on path for determinism.
    out.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| a.0.path.cmp(&b.0.path))
    });
    out.into_iter()
        .take(limit)
        .map(|(mut hit, fused_score)| {
            hit.score = fused_score;
            hit
        })
        .collect()
}

/// Collect whole Obsidian wikilinks from prose, verbatim (`[[Note|alias]]`).
/// These strings are the addressing contract: they can be handed straight to
/// `vault_read` / `vault_resolve`. Fenced code, inline code and media embeds
/// (![[img.png]]) are skipped.
fn extract_wikilinks(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        // Blank out inline code spans so `[[x]]` in examples is not a link.
        let mut clean = String::with_capacity(line.len());
        let mut in_inline = false;
        for ch in line.chars() {
            if ch == '`' {
                in_inline = !in_inline;
                clean.push(' ');
            } else if in_inline {
                clean.push(' ');
            } else {
                clean.push(ch);
            }
        }
        let line = clean.as_str();
        let bytes = line.as_bytes();
        let mut i = 0;
        while i + 3 < bytes.len() {
            if bytes[i] == b'[' && bytes[i + 1] == b'[' {
                if i > 0 && bytes[i - 1] == b'!' {
                    i += 2;
                    continue;
                }
                let rest = &line[i + 2..];
                if let Some(j) = rest.find("]]") {
                    out.push(format!("[[{}]]", &rest[..j]));
                    i += 2 + j + 2;
                    continue;
                }
            }
            i += 1;
        }
    }
    out
}

#[derive(Debug, Clone)]
struct TargetParts {
    /// Normalized lookup key (lower-cased, no `.md`).
    key: String,
    alias: Option<String>,
    anchor: Option<String>,
    block: Option<String>,
    media: bool,
}

/// Split one address into a lookup key plus its presentation parts. Accepts
/// `[[Note|alias#heading]]`, `[[path/Note]]`, `Note`, or `path/Note.md`.
fn parse_target(target: &str) -> TargetParts {
    let mut s = target.trim();
    if s.starts_with("[[") && s.ends_with("]]") {
        s = s[2..s.len() - 2].trim();
    }
    let mut alias = None;
    if let Some((a, b)) = s.split_once('|') {
        s = a;
        if !b.trim().is_empty() {
            alias = Some(b.trim().to_string());
        }
    }
    let mut anchor = None;
    if let Some((a, b)) = s.split_once('#') {
        s = a;
        if !b.trim().is_empty() {
            anchor = Some(b.trim().to_string());
        }
    }
    // An anchor may also carry a block ref: [[Note#Heading^block]].
    let mut block = None;
    if let Some(a) = anchor.clone() {
        if let Some((h, b)) = a.split_once('^') {
            anchor = Some(h.trim().to_string());
            if !b.trim().is_empty() {
                block = Some(b.trim().to_string());
            }
        }
    }
    if let Some((a, b)) = s.split_once('^') {
        s = a;
        if !b.trim().is_empty() {
            block = Some(b.trim().to_string());
        }
    }
    // Aliases may carry the anchor too: [[Note|alias#heading]].
    if let Some(a) = alias.clone() {
        if let Some((text, h)) = a.split_once('#') {
            alias = Some(text.trim().to_string());
            if anchor.is_none() && !h.trim().is_empty() {
                anchor = Some(h.trim().to_string());
            }
        }
    }
    if alias.as_deref().is_some_and(|a| a.is_empty()) {
        alias = None;
    }

    let name = s.trim();
    // Only known attachment extensions are media. A dot in a note title
    // (e.g. `[[Apple Container 0.8.0 ...]]`, `[[3.2-研究素材]]`) is NOT one.
    const MEDIA_EXT: [&str; 31] = [
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".tiff", ".ico", ".heic",
        ".heif", ".avif", ".pdf", ".mp3", ".wav", ".m4a", ".mp4", ".mov", ".webm", ".avi",
        ".mkv", ".canvas", ".excalidraw", ".docx", ".xlsx", ".pptx", ".epub", ".zip", ".gz",
        ".dmg", ".7z",
    ];
    let lower = name.to_ascii_lowercase();
    let media = MEDIA_EXT.iter().any(|e| lower.ends_with(e));
    let key = name
        .strip_suffix(".md")
        .unwrap_or(name)
        .trim()
        .to_lowercase();
    TargetParts {
        key,
        alias,
        anchor,
        block,
        media,
    }
}

/// Extract normalized outgoing link targets `(to_ref, kind)` from both the
/// note body and frontmatter provenance fields (`sources`, `related`,
/// `supersedes`). Used to build the `edges` table for backlinks and orphan
/// wiki detection without needing two-pass resolution during indexing.
fn extract_edges(text: &str, body: &str) -> Vec<(String, &'static str)> {
    let mut out: Vec<(String, &'static str)> = Vec::new();
    let mut seen: HashSet<(String, &'static str)> = HashSet::new();
    let mut push = |raw: &str, kind: &'static str| {
        let parts = parse_target(raw);
        if parts.key.is_empty() || parts.media {
            return;
        }
        if seen.insert((parts.key.clone(), kind)) {
            out.push((parts.key, kind));
        }
    };

    for link in extract_wikilinks(body) {
        push(&link, "body");
    }

    let lines: Vec<&str> = text.split('\n').collect();
    if lines.first().map(|l| l.trim_end()) == Some("---") {
        if let Some(end) = lines
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, l)| l.trim_end() == "---")
            .map(|(i, _)| i)
        {
            let mut i = 1usize;
            while i < end {
                let line = lines[i];
                let trimmed = line.trim();
                i += 1;
                if trimmed.is_empty()
                    || line.starts_with(' ')
                    || line.starts_with('\t')
                    || trimmed.starts_with('-')
                {
                    continue;
                }
                let Some((key, value)) = line.split_once(':') else {
                    continue;
                };
                let kind = match key.trim() {
                    "sources" => "source",
                    "related" => "related",
                    "supersedes" => "supersedes",
                    _ => continue,
                };
                let raw = value.trim();
                if !raw.is_empty() {
                    if let Ok(arr) = serde_json::from_str::<Vec<String>>(raw) {
                        for item in arr {
                            push(&item, kind);
                        }
                    } else {
                        let inner = raw.trim_start_matches('[').trim_end_matches(']');
                        for part in inner.split(',') {
                            let item = part.trim().trim_matches('"').trim_matches('\'').trim();
                            if !item.is_empty() {
                                push(item, kind);
                            }
                        }
                    }
                } else {
                    let mut j = i;
                    while j < end {
                        let t = lines[j].trim();
                        if let Some(item) = t.strip_prefix("- ") {
                            let cleaned = item.trim().trim_matches('"').trim_matches('\'').trim();
                            if !cleaned.is_empty() {
                                push(cleaned, kind);
                            }
                            j += 1;
                        } else if t.is_empty() {
                            j += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }

    out
}

#[derive(Debug, Clone)]
pub struct LinkResolution {
    /// Internal filesystem location. Never exposed as an addressing hint.
    pub path: Option<String>,
    /// Lightweight metadata keyed by the original wikilink.
    pub preview: serde_json::Value,
}

fn preview_entry(target: &str, parts: &TargetParts) -> serde_json::Map<String, serde_json::Value> {
    let mut e = serde_json::Map::new();
    e.insert("target".to_string(), serde_json::json!(target));
    e.insert("key".to_string(), serde_json::json!(parts.key));
    if let Some(a) = &parts.alias {
        e.insert("alias".to_string(), serde_json::json!(a));
    }
    if let Some(a) = &parts.anchor {
        e.insert("anchor".to_string(), serde_json::json!(a));
    }
    if let Some(b) = &parts.block {
        e.insert("block".to_string(), serde_json::json!(b));
    }
    e
}

pub struct VaultIndex {
    conn: Mutex<Connection>,
    root: PathBuf,
    db_path: PathBuf,
    deny_paths: Vec<String>,
    schema_present: bool,
    /// Number of successful incremental index changes since startup.
    updates: AtomicU64,
    /// Number of filesystem paths observed by the watcher (including no-ops).
    events: AtomicU64,
}

impl VaultIndex {
    pub fn open(
        root: PathBuf,
        db_path: &Path,
        deny_paths: Vec<String>,
        schema: &str,
    ) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("创建索引目录失败: {}", parent.display()))?;
        }
        let conn = Connection::open(db_path)
            .with_context(|| format!("打开索引库失败: {}", db_path.display()))?;
        conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;")?;
        Ok(Self {
            conn: Mutex::new(conn),
            root,
            db_path: db_path.to_path_buf(),
            deny_paths,
            schema_present: true,
            updates: AtomicU64::new(0),
            events: AtomicU64::new(0),
        }
        .with_schema_flag(schema))
    }

    fn with_schema_flag(mut self, schema: &str) -> Self {
        self.schema_present = self.root.join(schema).exists();
        self
    }

    pub fn is_denied(&self, rel: &str) -> bool {
        self.deny_paths
            .iter()
            .any(|d| rel == d.trim_end_matches('/') || rel.starts_with(d))
            || matches!(rel, "AGENTS.md" | "index.md" | "log.md")
    }

    /// Full rebuild. The SQLite DB is a disposable cache: it is always derived
    /// from the Markdown files and is never written back into them.
    pub fn rebuild(&self) -> Result<usize> {
        let mut conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "DROP TABLE IF EXISTS notes;
             DROP TABLE IF EXISTS notes_tri;
             DROP TABLE IF EXISTS note_meta;
             DROP TABLE IF EXISTS refs;
             DROP TABLE IF EXISTS edges;
             CREATE VIRTUAL TABLE notes USING fts5(
                 path UNINDEXED,
                 title, summary, tags, body,
                 layer UNINDEXED, status UNINDEXED, date UNINDEXED,
                 hash UNINDEXED,
                 tokenize = 'unicode61 remove_diacritics 2'
             );
             CREATE VIRTUAL TABLE notes_tri USING fts5(
                 path UNINDEXED,
                 title, summary, tags, body,
                 layer UNINDEXED, status UNINDEXED, date UNINDEXED,
                 hash UNINDEXED,
                 tokenize = 'trigram'
             );
             CREATE TABLE note_meta (
                 path TEXT PRIMARY KEY,
                 title TEXT, summary TEXT, layer TEXT,
                 status TEXT, kind TEXT, hash TEXT
             );
             CREATE TABLE refs (
                 ref TEXT NOT NULL,
                 path TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 PRIMARY KEY (ref, path)
             );
             CREATE INDEX refs_ref ON refs(ref);
             CREATE TABLE edges (
                 from_path TEXT NOT NULL,
                 to_ref TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 PRIMARY KEY (from_path, to_ref, kind)
             );
             CREATE INDEX edges_to_ref ON edges(to_ref);",
        )?;
        let tx = conn.transaction()?;
        let root = self.root.clone();
        let deny = self.deny_paths.clone();
        let mut count = 0usize;

        let walker = WalkDir::new(&root).into_iter().filter_entry(move |e| {
            if e.depth() == 0 {
                return true;
            }
            let rel = e
                .path()
                .strip_prefix(&root)
                .unwrap_or(e.path())
                .to_string_lossy()
                .replace('\\', "/");
            !deny
                .iter()
                .any(|d| rel == d.trim_end_matches('/') || rel.starts_with(d))
        });

        {
            let mut ins_notes = tx.prepare(
                "INSERT INTO notes
                 (path, title, summary, tags, body, layer, status, date, hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;
            let mut ins_tri = tx.prepare(
                "INSERT INTO notes_tri
                 (path, title, summary, tags, body, layer, status, date, hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;
            let mut ins_meta = tx.prepare(
                "INSERT OR REPLACE INTO note_meta
                 (path, title, summary, layer, status, kind, hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            let mut ins_refs = tx.prepare(
                "INSERT OR IGNORE INTO refs (ref, path, kind) VALUES (?1, ?2, ?3)",
            )?;
            let mut ins_edges = tx.prepare(
                "INSERT OR IGNORE INTO edges (from_path, to_ref, kind) VALUES (?1, ?2, ?3)",
            )?;
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
                    .strip_prefix(&self.root)?
                    .to_string_lossy()
                    .replace('\\', "/");
                if self.is_denied(&rel) {
                    continue;
                }
                let text = std::fs::read_to_string(path).unwrap_or_default();
                let (fm, body) = parse_note(&text);
                let hash = content_hash(&text);
                ins_notes.execute(params![
                    rel,
                    fm.title,
                    fm.summary,
                    fm.tags.join(", "),
                    body,
                    fm.layer,
                    fm.status,
                    fm.date,
                    hash,
                ])?;
                ins_tri.execute(params![
                    rel,
                    fm.title,
                    fm.summary,
                    fm.tags.join(", "),
                    body,
                    fm.layer,
                    fm.status,
                    fm.date,
                    hash,
                ])?;
                ins_meta.execute(params![
                    rel,
                    fm.title,
                    fm.summary,
                    fm.layer,
                    fm.status,
                    fm.kind,
                    hash,
                ])?;
                for (key, kind) in ref_keys(&rel, &fm) {
                    ins_refs.execute(params![key, rel, kind])?;
                }
                for (to_ref, kind) in extract_edges(&text, &body) {
                    ins_edges.execute(params![rel, to_ref, kind])?;
                }
                count += 1;
            }
        }
        tx.commit()?;
        Ok(count)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Observability: count a filesystem path seen by the watcher even when its
    /// content hash is unchanged, so a plain `touch` can prove FSEvents works
    /// on the real vault without modifying any note or index row.
    pub fn note_watcher_event(&self) {
        self.events.fetch_add(1, Ordering::Relaxed);
    }

    /// Apply one path from a filesystem event. Non-Markdown, out-of-vault and
    /// denied paths are ignored; a path that no longer exists removes its row.
    /// Returns true only when the derived index actually changed.
    pub fn apply_path(&self, abs: &Path) -> Result<bool> {
        let Ok(rel_path) = abs.strip_prefix(&self.root) else {
            return Ok(false);
        };
        let rel = rel_path.to_string_lossy().replace('\\', "/");
        if !rel.ends_with(".md") || self.is_denied(&rel) {
            return Ok(false);
        }
        if abs.is_file() {
            self.upsert(&rel)
        } else {
            self.remove(&rel)
        }
    }

    /// Insert or replace one note. The content hash makes duplicate FSEvents
    /// (and no-op touches) cheap no-ops, and lets the daemon's own writes be
    /// ignored when the watcher later sees their filesystem events.
    pub fn upsert(&self, rel: &str) -> Result<bool> {
        if self.is_denied(rel) {
            return Ok(false);
        }
        // Read the file INSIDE the index lock. A watcher event that raced with
        // a daemon write can otherwise read the old bytes before the lock and
        // then overwrite the freshly-indexed row with stale content.
        let mut conn = self.conn.lock().unwrap();
        let text = match std::fs::read_to_string(self.root.join(rel)) {
            Ok(t) => t,
            // Only a genuinely missing file should drop the index rows. A
            // transient permission / IO error must not wipe a valid entry.
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                let tx = conn.transaction()?;
                let n = tx.execute("DELETE FROM notes WHERE path = ?1", params![rel])?;
                tx.execute("DELETE FROM notes_tri WHERE path = ?1", params![rel])?;
                tx.execute("DELETE FROM note_meta WHERE path = ?1", params![rel])?;
                tx.execute("DELETE FROM refs WHERE path = ?1", params![rel])?;
                tx.execute("DELETE FROM edges WHERE from_path = ?1", params![rel])?;
                tx.commit()?;
                drop(conn);
                if n > 0 {
                    self.updates.fetch_add(1, Ordering::Relaxed);
                    return Ok(true);
                }
                return Ok(false);
            }
            Err(err) => {
                return Err(err).with_context(|| format!("读取笔记失败: {rel}"));
            }
        };
        let hash = content_hash(&text);
        let existing: Option<String> = conn
            .query_row(
                "SELECT hash FROM note_meta WHERE path = ?1",
                params![rel],
                |r| r.get(0),
            )
            .optional()?;
        if existing.as_deref() == Some(hash.as_str()) {
            return Ok(false);
        }
        let (fm, body) = parse_note(&text);
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM notes WHERE path = ?1", params![rel])?;
        tx.execute("DELETE FROM notes_tri WHERE path = ?1", params![rel])?;
        tx.execute("DELETE FROM note_meta WHERE path = ?1", params![rel])?;
        tx.execute("DELETE FROM refs WHERE path = ?1", params![rel])?;
        tx.execute("DELETE FROM edges WHERE from_path = ?1", params![rel])?;
        tx.execute(
            "INSERT INTO notes (path, title, summary, tags, body, layer, status, date, hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                rel,
                fm.title,
                fm.summary,
                fm.tags.join(", "),
                body,
                fm.layer,
                fm.status,
                fm.date,
                hash,
            ],
        )?;
        tx.execute(
            "INSERT INTO notes_tri (path, title, summary, tags, body, layer, status, date, hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                rel,
                fm.title,
                fm.summary,
                fm.tags.join(", "),
                body,
                fm.layer,
                fm.status,
                fm.date,
                hash,
            ],
        )?;
        tx.execute(
            "INSERT INTO note_meta (path, title, summary, layer, status, kind, hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                rel,
                fm.title,
                fm.summary,
                fm.layer,
                fm.status,
                fm.kind,
                hash,
            ],
        )?;
        for (key, kind) in ref_keys(rel, &fm) {
            tx.execute(
                "INSERT OR IGNORE INTO refs (ref, path, kind) VALUES (?1, ?2, ?3)",
                params![key, rel, kind],
            )?;
        }
        for (to_ref, kind) in extract_edges(&text, &body) {
            tx.execute(
                "INSERT OR IGNORE INTO edges (from_path, to_ref, kind) VALUES (?1, ?2, ?3)",
                params![rel, to_ref, kind],
            )?;
        }
        tx.commit()?;
        drop(conn);
        self.updates.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    pub fn remove(&self, rel: &str) -> Result<bool> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        let n = tx.execute("DELETE FROM notes WHERE path = ?1", params![rel])?;
        tx.execute("DELETE FROM notes_tri WHERE path = ?1", params![rel])?;
        tx.execute("DELETE FROM note_meta WHERE path = ?1", params![rel])?;
        tx.execute("DELETE FROM refs WHERE path = ?1", params![rel])?;
        tx.execute("DELETE FROM edges WHERE from_path = ?1", params![rel])?;
        tx.commit()?;
        drop(conn);
        if n > 0 {
            self.updates.fetch_add(1, Ordering::Relaxed);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Each SQL passed here selects nine columns in order:
    /// path, title, summary, layer, status, tags, date, score, body.
    fn rows_from_match(
        conn: &Connection,
        sql: &str,
        pattern: &str,
        query: &str,
        layer: Option<&str>,
        status: Option<&str>,
        limit: i64,
    ) -> Result<Vec<SearchHit>> {
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(params![pattern, layer, status, limit], |r| {
            let path: String = r.get(0)?;
            let target = format!("[[{}]]", path.strip_suffix(".md").unwrap_or(&path));
            let body: String = r.get(8)?;
            let raw_score: f64 = r.get(7)?;
            Ok(SearchHit {
                path,
                target,
                title: r.get(1)?,
                summary: r.get(2)?,
                layer: r.get(3)?,
                status: r.get(4)?,
                tags: r.get(5)?,
                date: r.get(6)?,
                score: 0.0, // filled by rrf_fuse
                bm25: raw_score,
                snippet: make_snippet(&body, query, 120),
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn search(
        &self,
        query: &str,
        layer: Option<&str>,
        status: Option<&str>,
        limit: i64,
    ) -> Result<Vec<SearchHit>> {
        let conn = self.conn.lock().unwrap();
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let words: Vec<&str> = query.split_whitespace().collect();
        let fts = if words.len() > 1 {
            words
                .iter()
                .map(|w| format!("\"{}\"", w.replace('"', "\"\"")))
                .collect::<Vec<_>>()
                .join(" OR ")
        } else {
            format!("\"{}\"", query.replace('"', "\"\""))
        };
        let limit = limit.clamp(1, 100);
        // Each path must contribute a deeper pool than the final result count:
        // RRF can only fuse docs it actually sees, so a note ranked #12 in one
        // index and #2 in another must be in both lists to be promoted.
        let pool = (limit * 5).clamp(limit, 100);

        // P1: weighted bm25 — title/summary/tags outrank a passing body mention.
        // Column order matches the FTS5 table definition; UNINDEXED columns get 0.
        let sql = "SELECT path, title, summary, layer, status, tags, date,
                          bm25(notes, 0.0, 10.0, 6.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0) AS score,
                          body
                   FROM notes
                   WHERE notes MATCH ?1
                     AND (?2 IS NULL OR layer = ?2)
                     AND (?3 IS NULL OR status = ?3)
                   ORDER BY score
                   LIMIT ?4";
        // Each retrieval path produces its own ranked list. RRF then fuses
        // them by rank, so cross-index score comparability is irrelevant.
        let primary = Self::rows_from_match(&conn, sql, &fts, query, layer, status, pool)?;
        let mut lists: Vec<(Vec<SearchHit>, f64)> = vec![(primary, 1.0)];

        // P2: trigram tokeniser segments CJK, so a Chinese query that gets no
        // (or weak) unicode61 ranking is re-run on the parallel trigram index.
        if has_cjk(query) {
            let tri_sql = "SELECT path, title, summary, layer, status, tags, date,
                                  bm25(notes_tri, 0.0, 10.0, 6.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0) AS score,
                                  body
                           FROM notes_tri
                           WHERE notes_tri MATCH ?1
                             AND (?2 IS NULL OR layer = ?2)
                             AND (?3 IS NULL OR status = ?3)
                           ORDER BY score
                           LIMIT ?4";
            let tri = Self::rows_from_match(&conn, tri_sql, &fts, query, layer, status, pool)?;
            lists.push((tri, 1.0));
        }

        // P3: field-weighted, multi-term substring scan.
        // SQLite's trigram tokenizer ignores terms shorter than 3 chars, so
        // 2-character CJK concepts (e.g. "缓存", "架构") and multi-word CJK
        // queries rely on this path. Scoring per term by field weight
        // (title 10, summary 6, tags 3, body 1) ensures title matches and
        // multi-term hits rank first instead of arbitrary table order.
        // Down-weighted in RRF (0.1) so it tops up recall without outranking
        // real FTS5 hits when those exist.
        let fallback = Self::substring_fallback(&conn, &words, query, layer, status, pool)?;
        lists.push((fallback, 0.1));

        Ok(rrf_fuse(&lists, limit as usize))
    }

    fn substring_fallback(
        conn: &Connection,
        words: &[&str],
        query: &str,
        layer: Option<&str>,
        status: Option<&str>,
        limit: i64,
    ) -> Result<Vec<SearchHit>> {
        let mut terms: Vec<String> = Vec::new();
        for w in words.iter().take(16) {
            let lower = w.to_lowercase();
            if !lower.is_empty() && !terms.contains(&lower) {
                terms.push(lower);
            }
        }
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        // Bindings: ?1 = layer, ?2 = status, ?3 = limit, ?4.. = lowercased terms.
        let mut score_parts: Vec<String> = Vec::with_capacity(terms.len());
        let mut match_parts: Vec<String> = Vec::with_capacity(terms.len());
        for i in 0..terms.len() {
            let p = i + 4;
            score_parts.push(format!(
                "(CASE WHEN instr(lower(title), ?{p}) > 0 THEN 10.0 ELSE 0.0 END \
                 + CASE WHEN instr(lower(summary), ?{p}) > 0 THEN 6.0 ELSE 0.0 END \
                 + CASE WHEN instr(lower(tags), ?{p}) > 0 THEN 3.0 ELSE 0.0 END \
                 + CASE WHEN instr(lower(body), ?{p}) > 0 THEN 1.0 ELSE 0.0 END)"
            ));
            match_parts.push(format!(
                "(instr(lower(title), ?{p}) > 0 OR instr(lower(summary), ?{p}) > 0 \
                 OR instr(lower(tags), ?{p}) > 0 OR instr(lower(body), ?{p}) > 0)"
            ));
        }

        let sql = format!(
            "SELECT path, title, summary, layer, status, tags, date,
                    ({score_expr}) AS fscore, body
             FROM notes
             WHERE ({where_expr})
               AND (?1 IS NULL OR layer = ?1)
               AND (?2 IS NULL OR status = ?2)
             ORDER BY fscore DESC, path ASC
             LIMIT ?3",
            score_expr = score_parts.join(" + "),
            where_expr = match_parts.join(" OR "),
        );

        let mut bind: Vec<SqlValue> = Vec::with_capacity(3 + terms.len());
        bind.push(match layer {
            Some(s) => SqlValue::Text(s.to_string()),
            None => SqlValue::Null,
        });
        bind.push(match status {
            Some(s) => SqlValue::Text(s.to_string()),
            None => SqlValue::Null,
        });
        bind.push(SqlValue::Integer(limit));
        for t in terms {
            bind.push(SqlValue::Text(t));
        }

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(bind), |r| {
            let path: String = r.get(0)?;
            let target = format!("[[{}]]", path.strip_suffix(".md").unwrap_or(&path));
            let body: String = r.get(8)?;
            Ok(SearchHit {
                path,
                target,
                title: r.get(1)?,
                summary: r.get(2)?,
                layer: r.get(3)?,
                status: r.get(4)?,
                tags: r.get(5)?,
                date: r.get(6)?,
                score: 0.0, // filled by rrf_fuse
                bm25: 0.0,  // fallback is not an FTS5 bm25 hit
                snippet: make_snippet(&body, query, 120),
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Read one note addressed by an Obsidian wikilink (or a bare name / path).
    /// The address is resolved through the `refs` index, never by guessing a
    /// filesystem path in the client.
    pub fn read(&self, target: &str) -> Result<serde_json::Value> {
        let resolution = self.resolve(target)?;
        let rel = resolution.path.with_context(|| {
            let hint = resolution
                .preview
                .get("candidates")
                .map(|c| format!("；请用候选 target 重试：{c}"))
                .unwrap_or_default();
            format!("无法解析 wikilink {target}{hint}")
        })?;
        let candidate = self.root.join(&rel);
        let canonical = std::fs::canonicalize(&candidate)
            .with_context(|| format!("文件不存在: {}", target))?;
        if !canonical.starts_with(&self.root) {
            bail!("路径越出 vault 范围: {}", target);
        }
        let text = std::fs::read_to_string(&canonical)?;
        let (fm, body) = parse_note(&text);

        // The standard six fields are typed for indexing; wiki pages carry
        // extra schema keys (kind, sources, confidence, verified, ...). Expose
        // every top-level frontmatter key so Query can trace provenance.
        let mut all = serde_json::Map::new();
        let lines: Vec<&str> = text.split('\n').collect();
        if lines.first().map(|l| l.trim_end()) == Some("---") {
            if let Some(end) = lines
                .iter()
                .enumerate()
                .skip(1)
                .find(|(_, l)| l.trim_end() == "---")
                .map(|(i, _)| i)
            {
                let mut i = 1usize;
                while i < end {
                    let line = lines[i];
                    let trimmed = line.trim();
                    i += 1;
                    if trimmed.is_empty()
                        || line.starts_with(' ')
                        || line.starts_with('\t')
                        || trimmed.starts_with('-')
                    {
                        continue;
                    }
                    let Some((key, value)) = line.split_once(':') else {
                        continue;
                    };
                    let key = key.trim();
                    if key.is_empty() {
                        continue;
                    }
                    let raw = value.trim().trim_matches('"');
                    let is_provenance = matches!(key, "sources" | "related" | "supersedes");
                    // Provenance fields are usually a one-line JSON array (the
                    // daemon writes them that way), but a human may use a YAML
                    // block list. Return a real array in both cases so the value
                    // can be fed straight into vault_resolve.targets[].
                    let parsed = if is_provenance && !raw.is_empty() {
                        serde_json::from_str::<serde_json::Value>(raw)
                            .ok()
                            .filter(|v| v.is_array())
                    } else if is_provenance {
                        let mut items = Vec::new();
                        let mut j = i;
                        while j < end {
                            let t = lines[j].trim();
                            if let Some(item) = t.strip_prefix("- ") {
                                items.push(serde_json::json!(item.trim().trim_matches('"')));
                                j += 1;
                            } else if t.is_empty() {
                                j += 1;
                            } else {
                                break;
                            }
                        }
                        if items.is_empty() {
                            None
                        } else {
                            Some(serde_json::Value::Array(items))
                        }
                    } else {
                        None
                    };
                    all.insert(
                        key.to_string(),
                        parsed.unwrap_or_else(|| serde_json::Value::String(raw.to_string())),
                    );
                }
            }
        }

        // `links` carries the wikilinks AS WRITTEN plus their indexed metadata.
        // The agent passes `target` straight back to vault_read / vault_resolve;
        // no path conversion happens on the client side.
        let targets: Vec<String> = extract_wikilinks(&body);
        let links = self.preview(&targets)?;
        let backlinks = self.backlinks(&rel)?;
        Ok(serde_json::json!({
            "path": rel,
            "target": target,
            "version": content_hash(&text),
            "frontmatter": {
                "title": fm.title,
                "date": fm.date,
                "tags": fm.tags,
                "layer": fm.layer,
                "summary": fm.summary,
                "status": fm.status,
            },
            "frontmatter_all": all,
            "body": body,
            "links": links,
            "backlinks": backlinks,
        }))
    }

    /// Resolve ONE wikilink against the derived link index. Returns only
    /// lightweight metadata; the resolved filesystem path is kept internal
    /// for `read` and is never advertised as an addressing hint.
    pub fn resolve(&self, target: &str) -> Result<LinkResolution> {
        let parts = parse_target(target);
        let mut entry = preview_entry(target, &parts);
        if parts.key.is_empty() {
            entry.insert("resolved".to_string(), serde_json::json!(false));
            return Ok(LinkResolution {
                path: None,
                preview: serde_json::Value::Object(entry),
            });
        }

        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT m.path, m.title, m.summary, m.layer, m.status, m.kind, m.hash, r.kind
               FROM refs r JOIN note_meta m ON m.path = r.path
              WHERE r.ref = ?1
              LIMIT 20",
        )?;
        let rows: Vec<(String, String, String, String, String, String, String, String)> = stmt
            .query_map(params![parts.key], |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                    r.get(7)?,
                ))
            })?
            .filter_map(Result::ok)
            .collect();

        // An attachment is only "media" when the link index has no note under
        // that name. A note literally named `Book.pdf.md` is still a note.
        if rows.is_empty() && parts.media {
            entry.insert("resolved".to_string(), serde_json::json!(false));
            entry.insert("reason".to_string(), serde_json::json!("media"));
            return Ok(LinkResolution {
                path: None,
                preview: serde_json::Value::Object(entry),
            });
        }

        // An explicit path wins over a bare name when both forms collide.
        let exact: Vec<_> = rows.iter().filter(|r| r.7 == "path").collect();
        let chosen: Vec<_> = if exact.is_empty() {
            rows.iter().collect()
        } else {
            exact
        };

        let path = match chosen.len() {
            1 => {
                let r = chosen[0];
                entry.insert("resolved".to_string(), serde_json::json!(true));
                entry.insert("title".to_string(), serde_json::json!(r.1));
                entry.insert("summary".to_string(), serde_json::json!(r.2));
                entry.insert("layer".to_string(), serde_json::json!(r.3));
                entry.insert("status".to_string(), serde_json::json!(r.4));
                entry.insert("kind".to_string(), serde_json::json!(r.5));
                entry.insert("version".to_string(), serde_json::json!(r.6));
                Some(r.0.clone())
            }
            0 => {
                entry.insert("resolved".to_string(), serde_json::json!(false));
                entry.insert("reason".to_string(), serde_json::json!("unresolved"));
                None
            }
            _ => {
                entry.insert("resolved".to_string(), serde_json::json!(false));
                entry.insert("ambiguous".to_string(), serde_json::json!(true));
                let candidates: Vec<serde_json::Value> = chosen
                    .iter()
                    .map(|r| {
                        let no_ext = r.0.strip_suffix(".md").unwrap_or(&r.0);
                        serde_json::json!({
                            "target": format!("[[{no_ext}]]"),
                            "title": r.1,
                            "summary": r.2,
                            "layer": r.3,
                            "status": r.4,
                            "kind": r.5,
                            "version": r.6,
                        })
                    })
                    .collect();
                entry.insert("candidates".to_string(), serde_json::json!(candidates));
                None
            }
        };
        Ok(LinkResolution {
            path,
            preview: serde_json::Value::Object(entry),
        })
    }

    /// Batch preview for many wikilinks: metadata only, never note bodies.
    /// This is the "decide what is worth reading" step of agentic retrieval.
    pub fn preview(&self, targets: &[String]) -> Result<Vec<serde_json::Value>> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for target in targets {
            if out.len() >= 200 {
                break;
            }
            let parts = parse_target(target);
            if parts.key.is_empty() || !seen.insert(parts.key.clone()) {
                continue;
            }
            let resolved = self.resolve(target)?;
            let is_media = resolved
                .preview
                .get("reason")
                .and_then(|v| v.as_str())
                == Some("media");
            if !is_media {
                out.push(resolved.preview);
            }
        }
        Ok(out)
    }

    /// Notes that reference `rel` via body wikilinks or frontmatter provenance
    /// (`sources`, `related`, `supersedes`), resolved dynamically through `refs`.
    pub fn backlinks(&self, rel: &str) -> Result<Vec<serde_json::Value>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT DISTINCT e.from_path, m.title, m.summary, m.layer, m.status, m.kind, m.hash
               FROM refs r
               JOIN edges e ON e.to_ref = r.ref
               JOIN note_meta m ON m.path = e.from_path
              WHERE r.path = ?1 AND e.from_path != ?1
              ORDER BY e.from_path
              LIMIT 200",
        )?;
        let rows = stmt.query_map(params![rel], |r| {
            let path: String = r.get(0)?;
            let no_ext = path.strip_suffix(".md").unwrap_or(&path);
            Ok(serde_json::json!({
                "target": format!("[[{no_ext}]]"),
                "title": r.get::<_, String>(1)?,
                "summary": r.get::<_, String>(2)?,
                "layer": r.get::<_, String>(3)?,
                "status": r.get::<_, String>(4)?,
                "kind": r.get::<_, String>(5)?,
                "version": r.get::<_, String>(6)?,
            }))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Wiki pages (`layer = 'wiki'`) with no incoming link from any other note.
    pub fn orphan_wiki_pages(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT m.path
               FROM note_meta m
              WHERE m.layer = 'wiki'
                AND NOT EXISTS (
                    SELECT 1
                      FROM refs r
                      JOIN edges e ON e.to_ref = r.ref
                     WHERE r.path = m.path AND e.from_path != m.path
                )
              ORDER BY m.path",
        )?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn status(&self) -> Result<serde_json::Value> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT count(*) FROM notes", [], |r| r.get(0))?;
        let refs: i64 = conn.query_row("SELECT count(*) FROM refs", [], |r| r.get(0))?;
        let edges: i64 = conn.query_row("SELECT count(*) FROM edges", [], |r| r.get(0))?;
        Ok(serde_json::json!({
            "indexed_notes": count,
            "link_refs": refs,
            "link_edges": edges,
            "vault_root": self.root.to_string_lossy(),
            "schema_present": self.schema_present,
            "index_db": self.db_path.to_string_lossy(),
            "incremental_updates": self.updates.load(Ordering::Relaxed),
            "watcher_events": self.events.load(Ordering::Relaxed),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQ: AtomicU64 = AtomicU64::new(0);

    fn note(_rel: &str, title: &str, summary: &str, body: &str) -> String {
        format!(
            "---\ntitle: \"{title}\"\nsummary: \"{summary}\"\nlayer: wiki\n---\n\n{body}\n"
        )
    }

    struct TestVault {
        root: PathBuf,
    }
    impl Drop for TestVault {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn fixture() -> (TestVault, VaultIndex) {
        let root = std::env::temp_dir().join(format!(
            "llm-wiki-test-{}-{}",
            std::process::id(),
            TEST_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let guard = TestVault { root: root.clone() };
        std::fs::write(
            root.join("网络策略.md"),
            note(
                "网络策略.md",
                "Kubernetes 网络策略",
                "How to isolate pods",
                "Kubernetes 的 NetworkPolicy 用来控制 Pod 之间的流量隔离，是集群安全的核心。",
            ),
        )
        .unwrap();
        std::fs::write(
            root.join("存储卷.md"),
            note(
                "存储卷.md",
                "持久化存储",
                "Persistent volumes",
                "本文讨论网络附加存储与 PersistentVolume 的绑定，和网络策略无关的内容在末尾提及 network。",
            ),
        )
        .unwrap();
        std::fs::write(
            root.join("english.md"),
            note(
                "english.md",
                "Service Mesh",
                "Traffic management between services",
                "A service mesh provides traffic management, and mentions the network policy concept briefly.",
            ),
        )
        .unwrap();
        // Both concepts in title/summary: strong on BOTH the unicode61 path
        // (OpenTelemetry) and the trigram path (插桩).
        std::fs::write(
            root.join("双概念.md"),
            note(
                "双概念.md",
                "OpenTelemetry 自动插桩",
                "Auto-instrumentation using OpenTelemetry",
                "OpenTelemetry 的自动插桩可以在不修改代码的情况下采集遥测数据。",
            ),
        )
        .unwrap();
        // Only the Latin word: strong on unicode61, absent from trigram list.
        std::fs::write(
            root.join("仅英文.md"),
            note(
                "仅英文.md",
                "OpenTelemetry Collector",
                "Collecting telemetry data",
                "The OpenTelemetry Collector receives and exports telemetry data in pipelines.",
            ),
        )
        .unwrap();
        let idx = VaultIndex::open(root.clone(), &root.join("index.sqlite"), vec![], "AGENTS.md")
            .unwrap();
        idx.rebuild().unwrap();
        (guard, idx)
    }

    #[test]
    fn cjk_query_ranks_title_match_first() {
        let (_dir, idx) = fixture();
        let hits = idx.search("网络策略", None, None, 10).unwrap();
        assert!(!hits.is_empty(), "CJK query must not lose all hits");
        assert_eq!(hits[0].path, "网络策略.md", "title match should rank first");
    }

    #[test]
    fn every_hit_has_snippet_with_highlight() {
        let (_dir, idx) = fixture();
        for q in ["网络策略", "NetworkPolicy", "traffic management"] {
            let hits = idx.search(q, None, None, 10).unwrap();
            assert!(!hits.is_empty(), "query {q} should match something");
            assert!(
                hits.iter().any(|h| h.snippet.contains("**")),
                "query {q} should produce at least one highlighted snippet"
            );
            assert!(
                hits.iter().all(|h| h.snippet.chars().count() <= 200),
                "snippets must stay bounded"
            );
        }
    }

    #[test]
    fn latin_query_still_works() {
        let (_dir, idx) = fixture();
        let hits = idx.search("service mesh", None, None, 10).unwrap();
        assert_eq!(hits.first().map(|h| h.path.as_str()), Some("english.md"));
    }

    #[test]
    fn substring_fallback_keeps_cjk_hits() {
        let (_dir, idx) = fixture();
        let hits = idx.search("流量隔离", None, None, 10).unwrap();
        assert!(
            hits.iter().any(|h| h.path == "网络策略.md"),
            "substring-only CJK term must still surface the note"
        );
    }

    #[test]
    fn weighted_bm25_prefers_summary_over_body() {
        let (_dir, idx) = fixture();
        // "Persistent volumes" is in 存储卷 summary; "network" appears only in
        // its body. The exact phrase in summary should outrank a body mention.
        let hits = idx.search("Persistent volumes", None, None, 10).unwrap();
        assert_eq!(hits.first().map(|h| h.path.as_str()), Some("存储卷.md"));
    }

    #[test]
    fn rrf_fuses_mixed_cjk_latin_query() {
        let (_dir, idx) = fixture();
        // "OpenTelemetry" ranks on the unicode61 path, "插桩" on the trigram
        // path. 双概念.md wins on both, so RRF must put it first even though
        // neither single path sees the full query.
        let hits = idx.search("OpenTelemetry 插桩", None, None, 10).unwrap();
        assert_eq!(
            hits.first().map(|h| h.path.as_str()),
            Some("双概念.md"),
            "RRF should surface the page matching both concepts first"
        );
        // A Latin-only page must still be present (from the primary list).
        assert!(
            hits.iter().any(|h| h.path == "仅英文.md"),
            "Latin-only page must survive fusion"
        );
    }

    #[test]
    fn score_is_rrf_and_bm25_is_preserved() {
        let (_dir, idx) = fixture();
        let hits = idx.search("网络策略", None, None, 10).unwrap();
        let top = &hits[0];
        assert!(
            top.score > 0.0,
            "RRF score should be a positive fused value, got {}",
            top.score
        );
        assert!(
            top.bm25 < 0.0,
            "original bm25 should be preserved (negative), got {}",
            top.bm25
        );
        // RRF scores must be monotonically non-increasing across the list.
        for w in hits.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "hits must be sorted by descending RRF score"
            );
        }
    }

    #[test]
    fn content_hash_is_stable_across_runs() {
        // Pin a known value so a future accidental hash-algorithm change is
        // caught immediately (FNV-1a 64-bit over UTF-8 bytes).
        assert_eq!(content_hash("hello"), "a430d84680aabd0b");
        assert_eq!(content_hash(""), "cbf29ce484222325");
        assert_ne!(content_hash("a"), content_hash("b"));
    }

    #[test]
    fn snippet_highlights_every_query_term() {
        let body = "First we deploy a cluster, then we observe the traffic with probes.";
        let snip = make_snippet(body, "deploy observe", 120);
        assert_eq!(
            snip.matches("**").count(),
            4,
            "both terms should be wrapped: expected 4 markers, got {snip}"
        );
        assert!(snip.contains("**deploy**"), "{snip}");
        assert!(snip.contains("**observe**"), "{snip}");
    }

    #[test]
    fn deep_pool_lets_later_rank_survive_small_limit() {
        let (_dir, idx) = fixture();
        // Even asking for a single result, fusion draws from a deeper pool
        // (limit * 5) on each path. A CJK query where the top primary hit and
        // the top trigram hit differ must still return the best-fused page.
        let hits = idx.search("OpenTelemetry 插桩", None, None, 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].path, "双概念.md",
            "deep pool must surface the doubly-matched page even with limit=1"
        );
    }

    #[test]
    fn two_char_cjk_fallback_ranks_title_over_body() {
        let (dir, idx) = fixture();
        std::fs::write(
            dir.root.join("正文提缓存.md"),
            note(
                "正文提缓存.md",
                "性能调优指南",
                "General performance tips",
                "在高并发场景下合理使用缓存可以降低数据库压力。",
            ),
        )
        .unwrap();
        std::fs::write(
            dir.root.join("标题有缓存.md"),
            note(
                "标题有缓存.md",
                "分布式缓存架构",
                "Distributed cache design",
                "介绍常见的淘汰策略和一致性方案。",
            ),
        )
        .unwrap();
        idx.rebuild().unwrap();

        // "缓存" is 2 chars (below FTS5 trigram minimum of 3), so it relies on
        // the weighted substring fallback. Title match must rank first.
        let hits = idx.search("缓存", None, None, 10).unwrap();
        assert!(hits.len() >= 2, "expected at least 2 hits, got {hits:?}");
        assert_eq!(
            hits[0].path, "标题有缓存.md",
            "2-char CJK title match must outrank body-only match"
        );
    }

    #[test]
    fn multi_term_cjk_fallback_matches_separated_terms() {
        let (dir, idx) = fixture();
        std::fs::write(
            dir.root.join("双词命中.md"),
            note(
                "双词命中.md",
                "分布式缓存设计",
                "Cache isolation",
                "多租户场景下需要对不同业务的流量进行严格隔离。",
            ),
        )
        .unwrap();
        std::fs::write(
            dir.root.join("单词命中.md"),
            note(
                "单词命中.md",
                "普通笔记",
                "Only mentions cache",
                "这里只提到了一次缓存。",
            ),
        )
        .unwrap();
        idx.rebuild().unwrap();

        // Neither "缓存" nor "隔离" is >= 3 chars, and they are not contiguous
        // in the document. Multi-term fallback must find and rank 双词命中 first.
        let hits = idx.search("缓存 隔离", None, None, 10).unwrap();
        assert_eq!(
            hits.first().map(|h| h.path.as_str()),
            Some("双词命中.md"),
            "doc matching both separated 2-char CJK terms must rank first"
        );
    }

    #[test]
    fn backlinks_and_orphan_wiki_pages_track_body_and_frontmatter_edges() {
        let root = std::env::temp_dir().join(format!(
            "llm-wiki-edges-{}-{}",
            std::process::id(),
            TEST_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("raw")).unwrap();
        std::fs::create_dir_all(root.join("wiki")).unwrap();
        let root = std::fs::canonicalize(&root).unwrap();
        let _guard = TestVault { root: root.clone() };

        std::fs::write(
            root.join("raw/source.md"),
            "---\ntitle: \"Raw Source\"\nlayer: raw\n---\n\n原始资料内容。\n",
        )
        .unwrap();
        std::fs::write(
            root.join("wiki/target.md"),
            "---\ntitle: \"Target Page\"\nlayer: wiki\nsources: [\"[[raw/source]]\"]\n---\n\n被引用页。\n",
        )
        .unwrap();
        std::fs::write(
            root.join("wiki/referrer.md"),
            "---\ntitle: \"Referrer Page\"\nlayer: wiki\nsources:\n  - \"[[raw/source]]\"\n---\n\n正文引用了 [[target]]。\n",
        )
        .unwrap();
        std::fs::write(
            root.join("wiki/orphan.md"),
            "---\ntitle: \"Orphan Page\"\nlayer: wiki\nsources: [\"[[raw/source]]\"]\n---\n\n没有任何页面引用我。\n",
        )
        .unwrap();

        let idx = VaultIndex::open(root.clone(), &root.join("index.sqlite"), vec![], "AGENTS.md")
            .unwrap();
        idx.rebuild().unwrap();

        // `raw/source.md` is referenced by all 3 wiki pages via frontmatter `sources`
        // (including the YAML block list in `wiki/referrer.md`).
        let source_doc = idx.read("[[raw/source]]").unwrap();
        let source_bl = source_doc["backlinks"].as_array().unwrap();
        assert_eq!(source_bl.len(), 3, "expected 3 backlinks to raw/source, got {source_bl:?}");

        // `wiki/target.md` is referenced by `wiki/referrer.md` in its body.
        let target_doc = idx.read("[[target]]").unwrap();
        let target_bl = target_doc["backlinks"].as_array().unwrap();
        assert_eq!(target_bl.len(), 1);
        assert_eq!(target_bl[0]["target"], "[[wiki/referrer]]");

        // `wiki/referrer.md` and `wiki/orphan.md` have no incoming links; `wiki/target.md` does.
        let orphans = idx.orphan_wiki_pages().unwrap();
        assert_eq!(orphans, vec!["wiki/orphan.md", "wiki/referrer.md"]);

        // Adding a body link from target -> orphan removes orphan incrementally via upsert.
        std::fs::write(
            root.join("wiki/target.md"),
            "---\ntitle: \"Target Page\"\nlayer: wiki\nsources: [\"[[raw/source]]\"]\n---\n\n现在引用了 [[orphan]]。\n",
        )
        .unwrap();
        assert!(idx.upsert("wiki/target.md").unwrap());
        let orphans_after = idx.orphan_wiki_pages().unwrap();
        assert_eq!(orphans_after, vec!["wiki/referrer.md"]);
    }
}

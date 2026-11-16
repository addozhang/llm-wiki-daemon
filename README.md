# LLM Wiki Daemon

An MCP server that turns an Obsidian vault into a **persistent, LLM-maintained wiki**,
following the pattern described by Andrej Karpathy in
[LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

Most document Q&A systems are RAG: every question re-retrieves raw chunks and
re-derives the answer from scratch. Nothing accumulates. This project instead
**compiles knowledge once and keeps it current** — a wiki layer of interlinked
Markdown pages sits between you and your immutable sources, and the LLM owns it.

> Status: `v0.1.0`, a working personal project. It was built around one
> specific vault's conventions, but the vault root, wiki directory and
> frontmatter fields are configurable. Expect to adapt the schema to your own
> vault. macOS is the primary platform (launchd + FSEvents); the daemon itself
> is plain Rust + SQLite and does not depend on Obsidian running.

## What it does

- **Three layers.** `raw` sources are read-only truth; `wiki/` pages are owned
  by the LLM; a schema document (`AGENTS.md` in your vault) defines conventions.
- **File-first, one writer.** The vault directory is the durable truth; one
  long-running Rust daemon owns the SQLite index; Obsidian is only a viewer.
  No dependency on the client's working directory.
- **A session contract, not a global prompt.** The MCP `instructions` field
  delivered on `initialize` tells any connected agent how to Query, Ingest and
  Lint, without polluting unrelated sessions.
- **Wikilinks are the addressing scheme.** Agents pass `[[Note]]` verbatim;
  the daemon resolves it through a link-name index. They never assemble paths.
- **Safe by scope, not by a global switch.** Writes are always on, but only
  pages inside the configured `wiki_root` with `layer: wiki` can be changed.
- **Auditable.** Every write is CAS-guarded and recorded in a structured JSONL
  audit trail outside the vault.

## Requirements

- Rust 1.88+ (MSRV in `Cargo.toml`)
- A Markdown vault whose notes carry YAML frontmatter. The default parser reads
  `title`, `date`, `tags`, `layer`, `summary`, `status`, `kind`, `aliases`;
  `layer` must be `raw`, `note` or `wiki`. See your vault's `AGENTS.md` for the
  full convention set.
- An MCP client that supports Streamable HTTP (e.g. OpenCode, `dsh`).
  Example configs: [`mcp-config.example.json`](mcp-config.example.json),
  [`dsh/llm-wiki.patch.yml`](dsh/llm-wiki.patch.yml).

## Quick start

```bash
git clone https://github.com/addozhang/llm-wiki-daemon.git
cd llm-wiki-daemon
cp config.example.toml config.toml
# edit the three absolute paths in config.toml
cargo build --release
./target/release/llm-wiki-daemon config.toml
```

The server listens on `http://127.0.0.1:8788/mcp` by default. **Every request
must include an allowed `Origin` header** — see [`mcp-config.example.json`](mcp-config.example.json)
for a client config. A missing or mismatched Origin is rejected with 403.

```bash
python3 scripts/check_contract.py --vault /path/to/your/vault
```

`check_contract.py` uses `tools/list` as the authority and verifies that the
session contract, the example config and the documentation still describe the
same tool surface. Run it after changing any tool parameter.

## Configuration

Copy [`config.example.toml`](config.example.toml) to `config.toml` (gitignored)
and replace the three absolute paths. All paths must be absolute; `vault.root`
must exist, while the index DB and log directories are created for you.

| Section | Key settings |
| --- | --- |
| `[server]` | `bind` (default `127.0.0.1`), `port` 8788; non-loopback requires `allow_non_loopback = true` + a token |
| `[security]` | `allowed_hosts`, `allowed_origins`, `deny_paths` |
| `[vault]` | `root`, `schema`, `index_db` |
| `[wiki]` | `root` — a vault-relative directory that owns every wiki page |
| `[watcher]` | `enabled`, `debounce_ms` (default 200, macOS FSEvents) |
| `[writes]` | `enabled` (default `true`, emergency kill switch only), `require_token` (default `false`) |
| `[logging]` | `dir`, `audit_max_bytes` (default 8 MiB) |
| `[auth]` | `token_env` — name of the env var holding the bearer token |

**Security model.** Host and Origin are both validated. With a non-empty
`allowed_origins`, a request without Origin, with a wrong value, or with
`Sec-Fetch-Site: cross-site` is rejected — this protects against browser
cross-site requests. It does **not** isolate local processes: a local program
can set the Origin itself. For local isolation set `require_token = true` and
provide `LLM_WIKI_DAEMON_TOKEN`. Paths are `canonicalize`d and must stay inside
the vault root; absolute paths, `..`, non-`.md` files and symlink escapes are
rejected. No credentials belong in any config file.

## Architecture

```
vault/  (durable truth, human + agent share it)
   ├── raw/   note/   ...     immutable sources, read-only to the daemon
   ├── wiki/                  LLM-owned compiled pages
   └── AGENTS.md              schema: conventions + workflows
daemon (Rust, single process)
   ├── SQLite  notes(FTS5) | note_meta | refs(link-name -> path)
   ├── watcher (FSEvents)    incremental upsert/remove, hash-deduplicated
   ├── 4 write tools         CAS + per-file lock + atomic rename + audit
   └── MCP initialize        delivers the session contract
Obsidian                      just a viewer/editor of the same directory
```

`index.md` and `log.md` from the original idea are **not files here**: the
wiki-directory catalog is served by `vault_search(layer=wiki)` and the
append-only timeline by `audit.jsonl` plus each page's `updated` / `revision`
fields. The pattern defines responsibilities, not filenames.

## Tool surface

Nine tools are exposed by default. `target` is the only addressing parameter
(`path` is a legacy alias); `vault_create.path` is the one exception because a
new page has no link name yet.

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `vault_search` | `query, layer?, status?, limit?` | Weighted FTS5 + CJK trigram + substring fallback; returns ready-to-feed `target` and a short `snippet` with `**highlighted**` matches |
| `vault_read` | `target` | Standard fields, `frontmatter_all`, body, `version`, and resolved `links` |
| `vault_resolve` | `targets[]` | Batch preview: metadata only, no body. Ambiguity gives `candidates[]` |
| `vault_status` | — | Index counts, `link_refs`, `wiki_root`, mode, watcher counters |
| `vault_lint` | `limit?` | Frontmatter health: missing / unclosed / duplicate keys, missing `title` / `layer` |
| `vault_patch` | `target, part(frontmatter\|section\|end), key?, op(set\|append), value, expected_version` | Incremental edit |
| `vault_append` | `target, text, expected_version, heading?` | Append at end or under a heading |
| `vault_create` | `path, kind, sources[], summary, text?, title?, confidence?` | Bootstrap a new wiki page with verifiable provenance |
| `vault_rewrite` | `target, expected_version, reason, text?, heading?, summary?, sources[]?, related[]?, supersedes[]?` | Revise or re-compile; may delete wrong entries |

### Write gateway

- **Double scope check**: path inside `wiki_root` **and** `layer: wiki`. Neither
  check alone is enough. `vault_create` bootstraps the first page even when the
  wiki directory does not exist yet.
- **Current state, not a ledger**: wrong body / `sources` / `related` must be
  corrected or removed with `vault_rewrite`, not followed by a correction note.
  History lives in the audit log and the vault's git history.
- **CAS**: every write carries `expected_version` from `vault_read`; a mismatch
  returns `conflict` with the current version. Per-file async locks serialise
  writes; atomic temp-file + rename; the index is updated in the same process.
- **No privilege escalation**: `layer` is immutable, `verified` can only be set
  true by a human, `revision` / `updated` are maintained by `vault_rewrite`, and
  every `sources` entry is verified to point at a real `raw`/`note` file.
- **No delete tool and no generic whole-file overwrite.**

`sources` are the compilation basis; body `[[links]]` are cross-references.
Default expansion depth is one level; unresolved links are reported, never
guessed. When `vault_rewrite` changes the body without a new `summary`, the
response reports `summary_not_updated: true` — a create-time one-line summary
goes stale as sections grow.

### Session contract

The MCP `instructions` text (in [`src/mcp.rs`](src/mcp.rs)) has four parts:

1. **Query** (default, read-only): look at `layer=wiki` first, fall back to all
   notes if needed, preview before reading, answer with `[[citations]]`. A
   closing self-check *proposes* filing a valuable answer back as a wiki page —
   it does not write without human confirmation.
2. **Ingest** (human-triggered): read the new source, discuss key takeaways,
   find every affected wiki page (one source can touch several), revise each
   and register the source. Body rewrites must refresh `summary`.
3. **Lint** (periodic, human-triggered): contradictions, stale claims, orphan
   pages, missing concept pages, missing cross-references, summary coverage —
   **report only, never auto-fix**.
4. **Addressing and write rules**: wikilinks, CAS, scope, `reason`, audit.

`vault_lint` only does mechanical frontmatter checks; semantic Lint is meant to
be performed by the agent using the existing tools.

## Audit and logs (outside the vault)

`daemon.log.YYYY-MM-DD` is a daily-rotated tracing log. `audit.jsonl` records
one JSON object per write with `ts / op / target / path / reason /
version_before / version_after / changes`; archives are timestamped and never
overwritten. Audit writes are best-effort (no fsync, warn on failure), so this
is not an absolute durability guarantee. Your vault's git history remains the
other audit trail.

## Deployment

[`deploy/com.example.llm-wiki-daemon.plist`](deploy/com.example.llm-wiki-daemon.plist)
is a launchd template. Replace `__PROJECT_DIR__` and `__LOG_DIR__`, rename the
label, copy it to `~/Library/LaunchAgents/`, then:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/<your-label>.plist
launchctl kickstart -k gui/$(id -u)/<your-label>   # reload after a rebuild
launchctl bootout   gui/$(id -u)/<your-label>       # stop
```

Only run one instance against a vault — multiple processes would contend for
port 8788 and the index. Don't start `target/debug/llm-wiki-daemon` while
launchd owns the port.

## Known boundaries and deliberate choices

- `vault_lint` has no semantic checks; those belong to the agent.
- Query and lint passes are not recorded in `audit.jsonl` (only writes are).
- The Origin check protects browsers, not local processes; use token auth for
  local isolation.
- Write tools are on by default. Safety comes from scope checks, not a switch.
- A watcher debounce has a bounded maximum wait so a steady event stream cannot
  starve indexing.
- Not implemented on purpose: semantic/vector search, a full YAML crate,
  automatic git commits, page deletion, and extra tools. Nine small primitives
  plus an agent turning out to be enough is the point of the design.

## Contributing

Issues and PRs are welcome. If you change the tool surface, update the session
contract, `mcp-config.example.json` and this README together, then run
`check_contract.py`. Please do not add credentials or machine-specific paths to
tracked files.

## License

[MIT](LICENSE) © 2026 Addo Zhang

#!/usr/bin/env python3
"""Check the daemon contract against `tools/list`, which is the authority.

Documents are split by reader (2026-11-15):
  * AGENTS.md  -> vault schema only; MUST NOT contain a tool table again.
  * instructions (MCP initialize) -> runtime session contract, unique source.
  * README.md / mcp-config.example.json -> implementation docs for humans.

The tool surface used to be described in several places and drifted repeatedly.
This script enforces the split so it cannot silently come back.

Usage:
    python3 scripts/check_contract.py --vault /path/to/vault    # live daemon
    LLM_WIKI_VAULT=/path/to/vault python3 scripts/check_contract.py
    python3 scripts/check_contract.py --tools-json f.json       # offline snapshot
    python3 scripts/check_contract.py --url http://127.0.0.1:8788/mcp

Exit code is non-zero when any hard check fails.
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def resolve_vault(explicit=None):
    """The vault root is machine-specific: CLI arg > env > local config.toml."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("LLM_WIKI_VAULT")
    if env:
        return Path(env)
    cfg = PROJECT / "config.toml"
    if cfg.exists() and tomllib is not None:
        try:
            with cfg.open("rb") as f:
                return Path(tomllib.load(f)["vault"]["root"])
        except Exception:
            pass
    return None

WRITE_TOOLS = {"vault_patch", "vault_append", "vault_create", "vault_rewrite"}
READ_TOOLS = {"vault_status", "vault_search", "vault_read", "vault_resolve", "vault_lint"}
EXPECTED = WRITE_TOOLS | READ_TOOLS

# Parameters that MUST exist in each tool's input schema. A change here is a
# breaking contract change and should be deliberate.
REQUIRED_PARAMS = {
    "vault_patch": {"target", "part", "op", "value", "expected_version"},
    "vault_append": {"target", "text", "expected_version"},
    "vault_create": {"path", "kind", "sources", "summary"},
    "vault_rewrite": {"target", "expected_version", "reason"},
    "vault_read": {"target"},
    "vault_resolve": {"targets"},
    "vault_search": {"query"},
    "vault_status": set(),
    "vault_lint": set(),
}


def rpc(url, method, params, rid, origin):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Origin": origin,
    }
    body = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read().decode()
        if raw.strip().startswith("data:"):
            raw = "\n".join(l[5:].strip() for l in raw.splitlines() if l.startswith("data:"))
        return json.loads(raw)


def load_tools(args):
    if args.tools_json:
        return json.loads(Path(args.tools_json).read_text(encoding="utf-8"))
    origin = args.origin or args.url.replace("/mcp", "")
    init = rpc(args.url, "initialize", {
        "protocolVersion": "2026-07-28",
        "capabilities": {},
        "clientInfo": {"name": "contract-check", "version": "1"},
    }, 1, origin)
    listing = rpc(args.url, "tools/list", {}, 2, origin)
    return {"tools": listing["result"]["tools"],
            "instructions": init["result"].get("instructions", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8788/mcp")
    ap.add_argument("--origin", default=None)
    ap.add_argument("--tools-json", default=None,
                    help="offline file: {'tools': [...], 'instructions': '...'}")
    ap.add_argument("--vault", default=None,
                    help="vault root; defaults to $LLM_WIKI_VAULT or vault.root in config.toml")
    ap.add_argument("--agents", default=None, help="path to the vault AGENTS.md")
    ap.add_argument("--readme", default=str(PROJECT / "README.md"))
    ap.add_argument("--example", default=str(PROJECT / "mcp-config.example.json"))
    args = ap.parse_args()
    vault = resolve_vault(args.vault)
    if vault is None and not args.agents:
        ap.error("未找到 vault 根目录：请用 --vault / LLM_WIKI_VAULT，或创建 config.toml（见 config.example.toml）")
    agents_path = Path(args.agents) if args.agents else vault / "AGENTS.md"

    data = load_tools(args)
    tools = {t["name"]: t for t in data["tools"]}
    names = set(tools)
    results = []

    def check(label, ok, detail=""):
        results.append((label, bool(ok), detail))

    check("工具数量 == 9", names == EXPECTED,
          f"实际 {len(names)} 个: {sorted(names)}")
    check("工具名集合一致", names == EXPECTED,
          f"缺 {sorted(EXPECTED - names)} / 多 {sorted(names - EXPECTED)}")

    for name, required in REQUIRED_PARAMS.items():
        if name not in tools:
            continue
        props = set((tools[name].get("inputSchema") or {}).get("properties", {}))
        missing = required - props
        check(f"{name} 必填参数齐全", not missing,
              f"缺 {sorted(missing)}" if missing else f"有 {sorted(props)}")

    # Only vault_create uses `path` as the address; all other write tools must
    # accept a wikilink `target`.
    for name in WRITE_TOOLS - {"vault_create"}:
        if name in tools:
            props = set((tools[name].get("inputSchema") or {}).get("properties", {}))
            check(f"{name} 用 target 寻址", "target" in props)

    # A summary written at create time goes stale after a body rewrite.
    # The tool must expose a way to refresh it and the contract must demand it.
    rewrite_props = set((tools.get("vault_rewrite") or {}).get("inputSchema", {}).get("properties", {}))
    check("vault_rewrite 支持 summary 同步", "summary" in rewrite_props,
          "有 summary 字段" if "summary" in rewrite_props else "缺少 summary，正文改写后摘要必然失真")

    instructions = data.get("instructions", "")
    check("instructions 要求同步 summary",
          "改正文必须同步 summary" in instructions and "summary_not_updated" in instructions,
          "已写明" if "改正文必须同步 summary" in instructions else "缺少同步规则")
    mentioned = set(re.findall(r"vault_[a-z_]+", instructions))
    ghosts = mentioned - names
    check("instructions 未提到不存在的工具", not ghosts,
          f"幽灵工具 {sorted(ghosts)}" if ghosts else f"提及 {sorted(mentioned)}")
    missing_writes = WRITE_TOOLS - mentioned
    check("instructions 覆盖全部写工具", not missing_writes,
          f"未提及 {sorted(missing_writes)}" if missing_writes else "4 个写工具均在契约中")

    # AGENTS.md is the vault schema; it must NOT re-introduce a tool table.
    agents = agents_path.read_text(encoding="utf-8")
    table = set(re.findall(r"\|\s*`(vault_[a-z_]+)`\s*\|", agents))
    check("AGENTS.md 不含工具表", not table,
          f"重新引入了 {sorted(table)}" if table else "工具面只在 README / 契约中")

    readme = Path(args.readme).read_text(encoding="utf-8")
    documented = set(re.findall(r"vault_[a-z_]+", readme))
    missing_docs = names - documented
    check("README.md 承接全部工具文档", not missing_docs,
          f"未记录 {sorted(missing_docs)}" if missing_docs else f"覆盖 {len(names)} 个工具")

    example = json.loads(Path(args.example).read_text(encoding="utf-8"))
    surface = set(example.get("dsh_headless", {}).get("tool_surface", []))
    check("mcp-config.example.json tool_surface 一致", surface == names,
          f"缺 {sorted(names - surface)} / 多 {sorted(surface - names)}")

    width = max(len(r[0]) for r in results)
    failed = 0
    for label, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<{width}}  {detail}")
        if not ok:
            failed += 1
    print(f"\n  {len(results) - failed}/{len(results)} 项通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

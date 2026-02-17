#!/usr/bin/env python3
"""Stable entrypoint for launching Ableton MCP from any working directory."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _bootstrap_repo_path(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _set_launch_cwd() -> None:
    # Preserve caller cwd so relative source paths can be resolved consistently.
    os.environ.setdefault("ABLETON_MCP_LAUNCH_CWD", os.getcwd())


def _smoke_check() -> int:
    from MCP_Server import server

    async def _list_tool_count() -> int:
        tools = await server.mcp.list_tools()
        return len(tools)

    try:
        tool_count = asyncio.run(_list_tool_count())
    except Exception as exc:
        print(f"SMOKE_CHECK_FAILED: {exc}", file=sys.stderr)
        return 1

    if tool_count <= 0:
        print("SMOKE_CHECK_FAILED: no MCP tools are registered", file=sys.stderr)
        return 1

    print(f"SMOKE_CHECK_OK: {tool_count} tools registered")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Ableton MCP server with stable repo-root path handling."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Validate imports and tool registration without starting the server loop.",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    _set_launch_cwd()
    _bootstrap_repo_path(repo_root)

    # Internal paths should resolve predictably regardless of caller cwd.
    os.chdir(repo_root)

    if args.smoke:
        return _smoke_check()

    from MCP_Server.server import main as server_main

    server_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Ableton MCP Agent Notes

## Running from any project folder

Use the launcher script in this repo so startup does not depend on your current working directory.

1. Start MCP with:
   `/Users/mikeonator/Documents/Code/ableton-mcp-analysis/scripts/run_mcp_server.sh`
2. Run a quick import/tool-registration check from any folder with:
   `/Users/mikeonator/Documents/Code/ableton-mcp-analysis/scripts/mcp_healthcheck.sh`
3. The launcher prefers `/Users/mikeonator/Documents/Code/ableton-mcp-analysis/.venv/bin/python` and falls back to `python3` with a clear stderr warning if the venv is missing.

## Recommended Codex MCP config (absolute paths)

```json
{
  "mcpServers": {
    "AbletonMCP": {
      "command": "/Users/mikeonator/Documents/Code/ableton-mcp-analysis/scripts/run_mcp_server.sh",
      "args": [],
      "cwd": "/Users/mikeonator/Documents/Code/ableton-mcp-analysis"
    }
  }
}
```

Use absolute `command` and `cwd` values so the MCP starts reliably when Codex is launched from external Ableton project directories.

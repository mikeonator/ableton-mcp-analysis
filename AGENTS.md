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

## Export-Based Ears (No Loopback)

The preferred "ears" workflow in this repo is file-based (export + analysis), not live loopback capture.

If audio analysis is requested and no `wav_path` is provided, the agent MUST:
1. Call `plan_exports` to create a manifest and exact output paths.
2. Tell the user what to export and where.
3. Call `check_exports_ready` and STOP if `status="WAIT_FOR_USER_EXPORT"`.
4. Continue with `analyze_export_job` only when `ready=true`.

Default directories are project-relative:
- Exports: `<project>/AbletonMCP/exports`
- Analysis JSONs + manifests: `<project>/AbletonMCP/analysis`

Fallback mode:
- If project root is unavailable, server falls back to cache paths under `~/.ableton_mcp_analysis/cache/` and returns warnings.

Recommended Ableton export settings:
- WAV, PCM, 24-bit
- Normalize OFF
- Sample rate matches project

`analyze_audio_file` also accepts `.mp3`, `.aif/.aiff`, `.flac` (and `.m4a` where ffmpeg can decode).
Non-WAV inputs are decoded to `AbletonMCP/analysis/tmp_decoded/` via `ffmpeg` before analysis.

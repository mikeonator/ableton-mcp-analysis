# Ableton MCP Agent Notes

## Scope (Important)

This `AGENTS.md` is intended to be copied into an **Ableton project folder** when using `AbletonMCP` there.

- It is **not** a generic Codex instruction file for unrelated coding projects.
- It should be used for music-production workflows in an Ableton session/project context.
- Bootstrap behavior in `AbletonMCP` should only copy this file into folders that look like Ableton projects (for example, a folder containing a `.als` file).

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

## Teacher Workflow Support (Mixing + Mastering PDFs)

This project is expected to help the user perform the workflows taught in:

- `Inserts & Sends / Aux / Bus` session-prep lesson
- `Mixing` (seven stages) lesson
- `Mastering` (goals + chain) lesson

The agent should treat these as the **preferred workflow sequence** when helping in an Ableton project.

### Session Prep (Inserts, Sends, Aux, Busses)

The agent should help the user:

- Understand and inspect **sends vs inserts** decisions
- Inspect **aux/return tracks** and what effects live on them
- Inspect **bus/group routing** (drum bus, vocal bus, instrument bus, etc.)
- Inspect **master bus** chain and routing

Use these tools first:

- `get_mix_topology`
- `get_send_matrix`
- `get_return_tracks_info`
- `get_master_track_device_chain`
- `build_mix_context_profile`

### Mixing (Seven Stages)

The agent should guide the user through the teacher’s mixing sequence:

1. Importing & organizing
2. Gain staging (VU/gain-structure mindset, loudest section first)
3. Fader mix (core band focus: lead vox, bass, kick, snare)
4. Automation pass (micro volume moves / section favoring)
5. Sub mix (group/family buses + EQ/compression glue)
6. Stereo and stem printing
7. Hand-off to mastering

The agent should use project data to support the user’s decisions, not force automation. Prefer:

- `build_mix_master_context` for stage-readiness + missing-data actions
- `get_mix_topology` and `get_send_matrix` for routing/submix/send structure
- `infer_mix_context_tags` / `save_mix_context_tags` to map tracks to musical roles (lead vocal, kick, snare, etc.)
- `get_automation_overview` and `get_track_automation_targets` for automation-state coverage
- `get_automation_envelope_points` when available for envelope reads

Automation note:

- If `get_automation_envelope_points` returns `supported=false`, the agent should still help the user perform the automation pass manually and use export-based loudness timelines / listening notes as a proxy for automation hotspots.

### Mastering (Goals + Chain)

The agent should guide the user with the teacher’s mastering priorities:

- Know the goal/target for the song
- Don’t fix mix problems in mastering (send back to mix if needed)
- Improve loudness, quality, and translation across systems
- Follow a mastering-chain mindset (meter continuity, cleaning EQ, saturation, glue compression, tonal EQ, stereo widening, limiting)

Use these tools:

- `analyze_mastering_file`
- `analyze_export_job(..., analysis_profile=\"mastering\")`
- `build_mix_master_context`

For mastering/stem analysis requests with no exported file paths, the agent MUST use the export workflow:

1. `plan_exports`
2. Tell the user what to export and where
3. `check_exports_ready`
4. Stop if `WAIT_FOR_USER_EXPORT`
5. `analyze_export_job`

### Working Style for This Ableton Project AGENTS.md

When helping with these lessons, the agent should:

- Be explicit about what is **observed data** vs **mix/master judgment advice**
- Prefer reading session/routing/analysis data before suggesting changes
- Preserve the teacher’s stage order unless the user asks to skip around
- Offer practical next steps in the DAW (what to adjust manually) when a capability is unavailable in the Remote Script API

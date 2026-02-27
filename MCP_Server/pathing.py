"""Project/cache path resolution and bootstrap helpers for export workflows."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


_DEFAULT_EXPORT_ROOT_MODE = "project"
_DEFAULT_EXPORT_REL_DIR = "AbletonMCP/exports"
_DEFAULT_ANALYSIS_REL_DIR = "AbletonMCP/analysis"
_CACHE_EXPORT_REL_DIR = "audio_exports"
_CACHE_ANALYSIS_REL_DIR = "audio_analysis"
_CACHE_BASE_DIR = os.path.expanduser("~/.ableton_mcp_analysis/cache")
_CONFIG_PATH = os.path.expanduser("~/.ableton_mcp_analysis/config.json")


def _load_optional_config() -> Dict[str, Any]:
    """Load optional JSON config payload from disk."""
    if not os.path.exists(_CONFIG_PATH):
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _config_or_env(config_payload: Dict[str, Any], key: str, env_key: str, default: Any) -> Any:
    """Return environment override, config value, or default."""
    env_value = os.environ.get(env_key)
    if env_value is not None and str(env_value).strip():
        return env_value
    if isinstance(config_payload, dict) and key in config_payload:
        value = config_payload.get(key)
        if value is not None and (not isinstance(value, str) or value.strip()):
            return value
    return default


def get_repo_root() -> str:
    """Return repository root inferred from module location."""
    return str(Path(__file__).resolve().parent.parent)


def _looks_like_ableton_project_root(project_root: str) -> Dict[str, Any]:
    """
    Heuristic detection for an Ableton project folder.

    We only want to auto-bootstrap a project-local AGENTS.md when AbletonMCP is being
    used inside an actual music project, not arbitrary coding folders.
    """
    if not isinstance(project_root, str) or not project_root:
        return {
            "is_ableton_project": False,
            "signals": [],
            "reason": "project_root_missing"
        }

    root = os.path.abspath(project_root)
    if not os.path.isdir(root):
        return {
            "is_ableton_project": False,
            "signals": [],
            "reason": "project_root_not_dir"
        }

    # Allow explicit override for unusual project layouts.
    force_flag = os.environ.get("ABLETON_MCP_FORCE_PROJECT_BOOTSTRAP")
    if isinstance(force_flag, str) and force_flag.strip().lower() in {"1", "true", "yes", "on"}:
        return {
            "is_ableton_project": True,
            "signals": ["env_override:ABLETON_MCP_FORCE_PROJECT_BOOTSTRAP"],
            "reason": "env_override"
        }

    signals = []
    try:
        entries = list(Path(root).iterdir())
    except Exception:
        return {
            "is_ableton_project": False,
            "signals": [],
            "reason": "listdir_failed"
        }

    # Strong signal: a Live set file at project root.
    if any(entry.is_file() and entry.suffix.lower() == ".als" for entry in entries):
        signals.append("top_level_als")

    # Common Ableton project directories/files. We treat these as supporting signals.
    common_names = {
        "Ableton Project Info",
        "Project Info",
        "Samples",
        "Backup",
        "Recordings",
        "AbletonMCP",
    }
    present_names = {entry.name for entry in entries}
    for name in sorted(common_names):
        if name in present_names:
            signals.append("entry:" + name)

    # A valid "help me mix/master in this session" folder should usually have a Live set.
    # We still accept strong project-folder patterns (project metadata + media folders) if no .als
    # is present yet, but require at least 2 supporting signals to avoid polluting random code repos.
    supporting_count = len([s for s in signals if s != "top_level_als"])
    is_ableton_project = ("top_level_als" in signals) or (supporting_count >= 2)

    return {
        "is_ableton_project": bool(is_ableton_project),
        "signals": signals,
        "reason": "detected" if is_ableton_project else "insufficient_ableton_signals"
    }


def get_project_root() -> Optional[str]:
    """
    Resolve project root path.

    Order:
    1) PROJECT_ROOT env override
    2) current working directory (preferred default)
    3) ABLETON_MCP_LAUNCH_CWD fallback

    Special-case:
    - When the launcher forces cwd to the repository root but preserved the
      original invocation cwd in ABLETON_MCP_LAUNCH_CWD, prefer the preserved
      cwd so project-relative exports/analysis map to the project folder.
    """
    explicit_root = os.environ.get("PROJECT_ROOT")
    if isinstance(explicit_root, str) and explicit_root.strip():
        candidate = os.path.abspath(os.path.expanduser(explicit_root.strip()))
        if os.path.isdir(candidate):
            return candidate
        return None

    launch_cwd = os.environ.get("ABLETON_MCP_LAUNCH_CWD")
    launch_candidate = None
    if isinstance(launch_cwd, str) and launch_cwd.strip():
        candidate = os.path.abspath(os.path.expanduser(launch_cwd.strip()))
        if os.path.isdir(candidate):
            launch_candidate = candidate

    cwd_candidate = os.path.abspath(os.getcwd())
    if os.path.isdir(cwd_candidate):
        try:
            repo_root = os.path.abspath(get_repo_root())
        except Exception:
            repo_root = None

        if (
            isinstance(repo_root, str)
            and cwd_candidate == repo_root
            and isinstance(launch_candidate, str)
            and launch_candidate != cwd_candidate
        ):
            return launch_candidate
        return cwd_candidate

    if isinstance(launch_candidate, str):
        return launch_candidate

    return None


def resolve_pathing() -> Dict[str, Any]:
    """Resolve export/analysis directories with fallback mode handling."""
    config_payload = _load_optional_config()
    requested_mode = str(
        _config_or_env(
            config_payload=config_payload,
            key="export_root_mode",
            env_key="EXPORT_ROOT_MODE",
            default=_DEFAULT_EXPORT_ROOT_MODE
        )
    ).strip().lower()
    if requested_mode not in {"project", "cache"}:
        requested_mode = _DEFAULT_EXPORT_ROOT_MODE

    export_rel_dir = str(
        _config_or_env(
            config_payload=config_payload,
            key="export_rel_dir",
            env_key="EXPORT_REL_DIR",
            default=_DEFAULT_EXPORT_REL_DIR
        )
    ).strip() or _DEFAULT_EXPORT_REL_DIR
    analysis_rel_dir = str(
        _config_or_env(
            config_payload=config_payload,
            key="analysis_rel_dir",
            env_key="ANALYSIS_REL_DIR",
            default=_DEFAULT_ANALYSIS_REL_DIR
        )
    ).strip() or _DEFAULT_ANALYSIS_REL_DIR

    warnings = []
    project_root = get_project_root()
    resolved_mode = requested_mode

    if resolved_mode == "project" and project_root is None:
        warnings.append("project_root_unavailable_fallback_to_cache")
        resolved_mode = "cache"

    if resolved_mode == "project":
        export_dir = os.path.abspath(os.path.join(project_root, export_rel_dir))
        analysis_dir = os.path.abspath(os.path.join(project_root, analysis_rel_dir))
    else:
        export_dir = os.path.abspath(os.path.join(_CACHE_BASE_DIR, _CACHE_EXPORT_REL_DIR))
        analysis_dir = os.path.abspath(os.path.join(_CACHE_BASE_DIR, _CACHE_ANALYSIS_REL_DIR))

    return {
        "requested_mode": requested_mode,
        "resolved_mode": resolved_mode,
        "project_root": project_root,
        "export_dir": export_dir,
        "analysis_dir": analysis_dir,
        "export_rel_dir": export_rel_dir,
        "analysis_rel_dir": analysis_rel_dir,
        "warnings": warnings
    }


def get_export_dir() -> str:
    """Return resolved export directory path."""
    return str(resolve_pathing()["export_dir"])


def get_analysis_dir() -> str:
    """Return resolved analysis directory path."""
    return str(resolve_pathing()["analysis_dir"])


def ensure_dirs_exist() -> Dict[str, Any]:
    """Ensure export and analysis directories exist."""
    resolved = resolve_pathing()
    os.makedirs(resolved["export_dir"], exist_ok=True)
    os.makedirs(resolved["analysis_dir"], exist_ok=True)
    return resolved


def ensure_project_agents_md() -> Dict[str, Any]:
    """Ensure AGENTS.md exists in project root when operating in project mode."""
    resolved = resolve_pathing()
    if resolved["resolved_mode"] != "project":
        return {
            "ok": True,
            "copied": False,
            "skipped": True,
            "reason": "not_project_mode",
            "warnings": resolved.get("warnings", [])
        }

    project_root = resolved.get("project_root")
    if not isinstance(project_root, str) or not project_root:
        return {
            "ok": False,
            "copied": False,
            "skipped": True,
            "reason": "project_root_unavailable",
            "warnings": ["project_root_unavailable"]
        }

    detection = _looks_like_ableton_project_root(project_root)
    if not detection.get("is_ableton_project"):
        return {
            "ok": True,
            "copied": False,
            "skipped": True,
            "reason": "not_ableton_project_folder",
            "agents_path": os.path.join(project_root, "AGENTS.md"),
            "warnings": resolved.get("warnings", []),
            "ableton_project_detection": detection,
        }

    project_agents = os.path.join(project_root, "AGENTS.md")
    if os.path.exists(project_agents):
        return {
            "ok": True,
            "copied": False,
            "skipped": True,
            "reason": "already_exists",
            "agents_path": project_agents,
            "warnings": resolved.get("warnings", []),
            "ableton_project_detection": detection,
        }

    source_agents = os.path.join(get_repo_root(), "AGENTS.md")
    if not os.path.exists(source_agents):
        return {
            "ok": False,
            "copied": False,
            "skipped": True,
            "reason": "source_agents_missing",
            "agents_path": project_agents,
            "warnings": ["source_agents_missing"],
            "ableton_project_detection": detection,
        }

    try:
        shutil.copy2(source_agents, project_agents)
        return {
            "ok": True,
            "copied": True,
            "skipped": False,
            "reason": "copied_from_repo",
            "agents_path": project_agents,
            "warnings": resolved.get("warnings", []),
            "ableton_project_detection": detection,
        }
    except Exception as exc:
        return {
            "ok": False,
            "copied": False,
            "skipped": True,
            "reason": "copy_failed",
            "agents_path": project_agents,
            "warnings": [f"copy_failed:{str(exc)}"],
            "ableton_project_detection": detection,
        }


def bootstrap_project_environment() -> Dict[str, Any]:
    """Ensure pathing directories and project AGENTS.md baseline are in place."""
    resolved = ensure_dirs_exist()
    agents_result = ensure_project_agents_md()
    warnings = []
    for value in [resolved.get("warnings", []), agents_result.get("warnings", [])]:
        if isinstance(value, list):
            warnings.extend(value)
    return {
        "ok": True,
        "pathing": resolved,
        "agents": agents_result,
        "warnings": warnings
    }

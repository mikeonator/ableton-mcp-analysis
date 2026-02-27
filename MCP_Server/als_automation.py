"""Read arrangement automation envelopes from a saved Ableton .als file."""

from __future__ import annotations

import gzip
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


_PRE_ROLL_SENTINEL_TIME = -63072000.0


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        return int(str(value).strip())
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(int(value))
        return float(str(value).strip())
    except Exception:
        return None


def _safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def _safe_file_mtime(path_value: str) -> float:
    try:
        return float(os.path.getmtime(path_value))
    except Exception:
        return 0.0


def _iso_utc_from_ts(ts: float) -> Optional[str]:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _looks_like_device_node(elem: ET.Element) -> bool:
    return elem.find("./ParametersListWrapper") is not None and elem.find("./Pointee") is not None


def _iter_normal_track_nodes(live_set_root: ET.Element) -> List[ET.Element]:
    tracks_parent = live_set_root.find("./Tracks")
    if tracks_parent is None:
        return []
    rows: List[ET.Element] = []
    for track_node in list(tracks_parent):
        if not isinstance(track_node.tag, str):
            continue
        if track_node.tag == "ReturnTrack":
            continue
        rows.append(track_node)
    return rows


def _iter_return_track_nodes(live_set_root: ET.Element) -> List[ET.Element]:
    tracks_parent = live_set_root.find("./Tracks")
    if tracks_parent is None:
        return []
    rows: List[ET.Element] = []
    for track_node in list(tracks_parent):
        if not isinstance(track_node.tag, str):
            continue
        if track_node.tag != "ReturnTrack":
            continue
        rows.append(track_node)
    return rows


def _get_main_track_node(live_set_root: ET.Element) -> Optional[ET.Element]:
    # Live 12 uses MainTrack in the .als structure; older exports may use MasterTrack.
    main_track = live_set_root.find("./MainTrack")
    if main_track is not None:
        return main_track
    return live_set_root.find("./MasterTrack")


def _track_name_from_als(track_node: ET.Element) -> Optional[str]:
    for path in (
        "./Name/EffectiveName",
        "./Name/UserName",
        "./Name/Name",
    ):
        node = track_node.find(path)
        if node is None:
            continue
        text = _safe_text(node.attrib.get("Value"))
        if text:
            return text
    return None


def _read_xml_from_als_file(als_path: str) -> Tuple[Optional[ET.Element], Optional[str]]:
    try:
        with gzip.open(als_path, "rb") as handle:
            data = handle.read()
    except Exception as exc:
        return None, "gzip_open_failed:{0}".format(str(exc))
    try:
        return ET.fromstring(data), None
    except Exception as exc:
        return None, "xml_parse_failed:{0}".format(str(exc))


def _resolve_als_file(project_root: Optional[str], als_file_path: Optional[str] = None) -> Dict[str, Any]:
    override_path = _safe_text(als_file_path)
    if override_path:
        expanded = os.path.abspath(os.path.expanduser(override_path))
        if not os.path.exists(expanded):
            return {
                "ok": True,
                "supported": False,
                "reason": "als_file_path_not_found",
                "als_file_path": expanded,
                "warnings": ["als_file_path_not_found"],
            }
        if not os.path.isfile(expanded):
            return {
                "ok": True,
                "supported": False,
                "reason": "als_file_path_not_file",
                "als_file_path": expanded,
                "warnings": ["als_file_path_not_file"],
            }
        if expanded.lower().endswith(".als") is False:
            return {
                "ok": True,
                "supported": False,
                "reason": "als_file_path_invalid_extension",
                "als_file_path": expanded,
                "warnings": ["als_file_path_invalid_extension"],
            }
        mtime = _safe_file_mtime(expanded)
        resolved_project_root = os.path.abspath(project_root) if isinstance(project_root, str) and project_root else os.path.dirname(expanded)
        return {
            "ok": True,
            "supported": True,
            "project_root": resolved_project_root,
            "als_file_path": expanded,
            "als_file_mtime_unix": mtime,
            "als_file_mtime_utc": _iso_utc_from_ts(mtime),
            "candidate_count": 1,
            "candidate_paths": [expanded],
            "warnings": ["als_file_path_override"],
        }

    if not isinstance(project_root, str) or not project_root:
        return {
            "ok": True,
            "supported": False,
            "reason": "project_root_unavailable",
            "warnings": ["project_root_unavailable"],
        }
    root = os.path.abspath(project_root)
    if not os.path.isdir(root):
        return {
            "ok": True,
            "supported": False,
            "reason": "project_root_not_dir",
            "project_root": root,
            "warnings": ["project_root_not_dir"],
        }

    candidates: List[Tuple[float, str]] = []
    try:
        for entry in Path(root).iterdir():
            if entry.is_file() and entry.suffix.lower() == ".als":
                try:
                    mtime = float(entry.stat().st_mtime)
                except Exception:
                    mtime = 0.0
                candidates.append((mtime, str(entry)))
    except Exception:
        return {
            "ok": True,
            "supported": False,
            "reason": "als_scan_failed",
            "project_root": root,
            "warnings": ["als_scan_failed"],
        }

    if not candidates:
        return {
            "ok": True,
            "supported": False,
            "reason": "top_level_als_not_found",
            "project_root": root,
            "warnings": ["top_level_als_not_found"],
        }

    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected_mtime, selected_path = candidates[0]
    warnings: List[str] = []
    if len(candidates) > 1:
        warnings.append("multiple_top_level_als_files_selected_newest_mtime")

    return {
        "ok": True,
        "supported": True,
        "project_root": root,
        "als_file_path": selected_path,
        "als_file_mtime_unix": selected_mtime,
        "als_file_mtime_utc": _iso_utc_from_ts(selected_mtime),
        "candidate_count": len(candidates),
        "candidate_paths": [row[1] for row in candidates[:10]],
        "warnings": warnings,
    }


def _device_name_hint_from_node(device_node: ET.Element) -> Optional[str]:
    for path in ("./Name/EffectiveName", "./Name/UserName", "./UserName", "./OriginalName", "./PluginDesc/VstPluginInfo/PlugName"):
        node = device_node.find(path)
        if node is None:
            continue
        if "Value" in node.attrib:
            text = _safe_text(node.attrib.get("Value"))
        else:
            text = _safe_text(node.text)
        if text:
            return text
    return None


def _macro_display_name_for_device(device_node: ET.Element, macro_index: int) -> Optional[str]:
    if macro_index < 0:
        return None
    node = device_node.find("./MacroDisplayNames.{0}".format(macro_index))
    if node is None:
        return None
    return _safe_text(node.attrib.get("Value"))


def _resolve_track_mixer_target_id(track_node: ET.Element, mixer_target: str, send_index: Optional[int]) -> Tuple[Optional[int], Dict[str, Any]]:
    mixer = track_node.find("./DeviceChain/Mixer")
    debug: Dict[str, Any] = {"scope": "track_mixer", "mixer_target": mixer_target}
    if mixer is None:
        return None, {"error": "mixer_node_missing", "message": "ALS track mixer node missing", "debug": debug}

    if mixer_target == "volume":
        target_node = mixer.find("./Volume/AutomationTarget")
        target_id = _safe_int(target_node.attrib.get("Id") if target_node is not None else None)
        debug["parameter_tag"] = "Volume"
        return target_id, {"debug": debug} if target_id is not None else {
            "error": "automation_target_missing",
            "message": "Track volume automation target not found in ALS",
            "debug": debug,
        }

    if mixer_target == "panning":
        # ALS stores pan under <Pan>; Live API field is "panning".
        target_node = mixer.find("./Pan/AutomationTarget")
        target_id = _safe_int(target_node.attrib.get("Id") if target_node is not None else None)
        debug["parameter_tag"] = "Pan"
        return target_id, {"debug": debug} if target_id is not None else {
            "error": "automation_target_missing",
            "message": "Track pan automation target not found in ALS",
            "debug": debug,
        }

    if mixer_target == "send":
        if send_index is None or send_index < 0:
            return None, {
                "error": "invalid_send_index",
                "message": "send_index is required for mixer_target='send'",
                "debug": debug,
            }
        holders = mixer.findall("./Sends/TrackSendHolder")
        debug["send_count"] = len(holders)
        if send_index >= len(holders):
            return None, {
                "error": "invalid_send_index",
                "message": "send_index out of range in ALS mixer sends",
                "debug": debug,
            }
        holder = holders[send_index]
        target_node = holder.find("./Send/AutomationTarget")
        target_id = _safe_int(target_node.attrib.get("Id") if target_node is not None else None)
        debug["send_index"] = int(send_index)
        debug["parameter_tag"] = "Send"
        return target_id, {"debug": debug} if target_id is not None else {
            "error": "automation_target_missing",
            "message": "Track send automation target not found in ALS",
            "debug": debug,
        }

    return None, {
        "error": "invalid_mixer_target",
        "message": "mixer_target must be one of: volume, panning, send",
        "debug": debug,
    }


def _resolve_main_track_target_id(main_track_node: ET.Element, target_kind: str) -> Tuple[Optional[int], Dict[str, Any]]:
    mixer = main_track_node.find("./DeviceChain/Mixer")
    debug: Dict[str, Any] = {"scope": "main_track", "target_kind": target_kind}
    if mixer is None:
        return None, {"error": "mixer_node_missing", "message": "ALS main track mixer node missing", "debug": debug}

    target_map = {
        "volume": ("Volume", "master_mixer"),
        "panning": ("Pan", "master_mixer"),
        "crossfade": ("CrossFade", "master_mixer"),
        "tempo": ("Tempo", "global"),
        "time_signature": ("TimeSignature", "global"),
    }
    if target_kind not in target_map:
        return None, {
            "error": "invalid_target_kind",
            "message": "target_kind must be one of: volume, panning, crossfade, tempo, time_signature",
            "debug": debug,
        }

    xml_tag, target_scope = target_map[target_kind]
    target_node = mixer.find("./{0}/AutomationTarget".format(xml_tag))
    target_id = _safe_int(target_node.attrib.get("Id") if target_node is not None else None)
    debug["parameter_tag"] = xml_tag
    debug["target_scope"] = target_scope
    if target_id is None:
        return None, {
            "error": "automation_target_missing",
            "message": "Main track automation target not found in ALS",
            "debug": debug,
        }
    return target_id, {"debug": debug, "target_scope": target_scope, "parameter_tag": xml_tag}


def _iter_top_level_track_devices(track_node: ET.Element) -> List[ET.Element]:
    devices_parent = track_node.find("./DeviceChain/DeviceChain/Devices")
    if devices_parent is None:
        return []
    return [child for child in list(devices_parent) if isinstance(child.tag, str)]


def _collect_device_parameter_targets(device_node: ET.Element) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    device_name_hint = _device_name_hint_from_node(device_node)

    def visit(node: ET.Element, root_device: ET.Element) -> None:
        for child in list(node):
            if child is not root_device and _looks_like_device_node(child):
                # Nested devices (e.g. rack chains) are not part of the top-level device parameter list.
                continue

            target_node = child.find("./AutomationTarget")
            target_id = _safe_int(target_node.attrib.get("Id") if target_node is not None else None)
            if target_id is not None:
                parameter_name_hint = _safe_text(child.tag)
                parameter_display_name_hint = None
                macro_match = re.match(r"^MacroControls\.(\d+)$", str(child.tag))
                if macro_match:
                    macro_index = _safe_int(macro_match.group(1))
                    if macro_index is not None:
                        parameter_display_name_hint = _macro_display_name_for_device(root_device, macro_index)
                rows.append({
                    "parameter_name_hint": parameter_name_hint,
                    "parameter_display_name_hint": parameter_display_name_hint,
                    "automation_target_id": int(target_id),
                    "xml_tag": child.tag,
                    "device_name_hint": device_name_hint,
                })

            visit(child, root_device)

    visit(device_node, device_node)
    return rows


def _resolve_device_parameter_target_id(
    track_node: ET.Element,
    device_index: Optional[int],
    parameter_index: Optional[int]
) -> Tuple[Optional[int], Dict[str, Any]]:
    if device_index is None or device_index < 0:
        return None, {
            "error": "invalid_device_index",
            "message": "device_index must be a non-negative integer",
        }
    if parameter_index is None or parameter_index < 0:
        return None, {
            "error": "invalid_parameter_index",
            "message": "parameter_index must be a non-negative integer",
        }

    devices = _iter_top_level_track_devices(track_node)
    if device_index >= len(devices):
        return None, {
            "error": "invalid_device_index",
            "message": "device_index out of range in ALS top-level device chain",
            "device_count": len(devices),
        }

    device_node = devices[device_index]
    target_rows = _collect_device_parameter_targets(device_node)
    if parameter_index >= len(target_rows):
        return None, {
            "error": "invalid_parameter_index",
            "message": "parameter_index out of range in ALS device parameter list",
            "device_count": len(devices),
            "parameter_count": len(target_rows),
            "device_tag": device_node.tag,
        }

    row = target_rows[parameter_index]
    return row.get("automation_target_id"), {
        "device_tag": device_node.tag,
        "device_count": len(devices),
        "parameter_count": len(target_rows),
        "device_name_hint": row.get("device_name_hint"),
        "parameter_name_hint": row.get("parameter_name_hint"),
        "parameter_display_name_hint": row.get("parameter_display_name_hint"),
        "parameter_xml_tag": row.get("xml_tag"),
    }


def _find_track_automation_envelope(track_node: ET.Element, pointee_id: int) -> Optional[ET.Element]:
    for env in track_node.findall("./AutomationEnvelopes/Envelopes/AutomationEnvelope"):
        pointee = env.find("./EnvelopeTarget/PointeeId")
        candidate = _safe_int(pointee.attrib.get("Value") if pointee is not None else None)
        if candidate == pointee_id:
            return env
    return None


def _parse_bool_text(text: Optional[str]) -> Optional[bool]:
    normalized = (_safe_text(text) or "").lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def _event_value_from_xml(event_node: ET.Element) -> Tuple[Any, str]:
    event_tag = _safe_text(event_node.tag) or "Event"
    raw_text = event_node.attrib.get("Value")

    if event_tag == "BoolEvent":
        bool_value = _parse_bool_text(raw_text)
        if bool_value is not None:
            return bool_value, "bool"
        return _safe_text(raw_text), "bool_text"

    if event_tag == "EnumEvent":
        int_value = _safe_int(raw_text)
        if int_value is not None:
            return int_value, "int"
        float_value = _safe_float(raw_text)
        if float_value is not None:
            return float_value, "float"
        return _safe_text(raw_text), "text"

    # Default: float envelopes (most mixer/device automation)
    float_value = _safe_float(raw_text)
    if float_value is not None:
        return float_value, "float"
    int_value = _safe_int(raw_text)
    if int_value is not None:
        return int_value, "int"
    return _safe_text(raw_text), "text"


def _parse_envelope_points(
    envelope_node: ET.Element,
    start_time_beats: Optional[float] = None,
    end_time_beats: Optional[float] = None
) -> List[Dict[str, Any]]:
    events = envelope_node.find("./Automation/Events")
    if events is None:
        return []

    points: List[Dict[str, Any]] = []
    for idx, event_node in enumerate(list(events)):
        if not isinstance(event_node.tag, str):
            continue
        time_beats = _safe_float(event_node.attrib.get("Time"))
        if time_beats is None:
            continue

        if start_time_beats is not None and time_beats < start_time_beats:
            continue
        if end_time_beats is not None and time_beats > end_time_beats:
            continue

        value, value_kind = _event_value_from_xml(event_node)
        row: Dict[str, Any] = {
            "point_index": idx,
            "time_beats": float(time_beats),
            "value": value,
            "event_type": _safe_text(event_node.tag),
            "value_kind": value_kind,
        }
        event_id = _safe_int(event_node.attrib.get("Id"))
        if event_id is not None:
            row["event_id"] = event_id
        if time_beats <= _PRE_ROLL_SENTINEL_TIME:
            row["is_pre_roll_default"] = True

        if event_node.tag == "FloatEvent":
            curve: Dict[str, float] = {}
            for attr_name in ("CurveControl1X", "CurveControl1Y", "CurveControl2X", "CurveControl2Y"):
                attr_value = _safe_float(event_node.attrib.get(attr_name))
                if attr_value is not None:
                    curve[attr_name] = attr_value
            if curve:
                row["shape"] = "bezier"
                row["curve"] = curve

        points.append(row)

    return points


def _make_tracklike_probe_result(
    *,
    scope_label: str,
    index: Optional[int],
    track_name: Optional[str],
    target_payload: Dict[str, Any],
    envelope: Optional[ET.Element],
    points: List[Dict[str, Any]],
    warnings: List[str],
    als_info: Dict[str, Any],
    als_path: Optional[str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": True,
        "supported": True,
        "source": "als_file",
        "source_kind": "arrangement_automation",
        "track_scope": scope_label,
        "target": target_payload,
        "envelope_exists": envelope is not None,
        "point_access_supported": True,
        "points": points,
        "warnings": list(warnings),
        "als_file_path": als_path,
        "als_file_mtime_utc": als_info.get("als_file_mtime_utc"),
        "als_candidate_count": als_info.get("candidate_count"),
    }
    if track_name:
        result["track_name"] = track_name
    if index is not None:
        result["index"] = int(index)
    return result


def _make_target_missing_result(
    *,
    scope_label: str,
    index: Optional[int],
    track_name: Optional[str],
    target_payload: Dict[str, Any],
    warnings: List[str],
    als_info: Dict[str, Any],
    als_path: Optional[str],
    target_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": True,
        "supported": True,
        "source": "als_file",
        "source_kind": "arrangement_automation",
        "track_scope": scope_label,
        "target": target_payload,
        "envelope_exists": False,
        "point_access_supported": True,
        "points": [],
        "warnings": list(warnings),
        "als_file_path": als_path,
        "als_file_mtime_utc": als_info.get("als_file_mtime_utc"),
        "als_candidate_count": als_info.get("candidate_count"),
    }
    if track_name:
        result["track_name"] = track_name
    if index is not None:
        result["index"] = int(index)
    if isinstance(target_debug, dict):
        if target_debug.get("error"):
            result["resolution_error"] = target_debug.get("error")
        if target_debug.get("message"):
            result["resolution_message"] = target_debug.get("message")
        if isinstance(target_debug.get("debug"), dict):
            result["target_debug"] = target_debug.get("debug")
    return result


def read_arrangement_automation_from_project_als(
    project_root: Optional[str],
    track_index: int,
    scope: str,
    mixer_target: str,
    als_file_path: Optional[str] = None,
    send_index: Optional[int] = None,
    device_index: Optional[int] = None,
    parameter_index: Optional[int] = None,
    start_time_beats: Optional[float] = None,
    end_time_beats: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return exact arrangement automation points from the saved .als file (best effort).

    This is a fallback for cases where the Live Python API cannot expose arrangement automation
    breakpoints directly in the Remote Script context.
    """
    scope_value = (_safe_text(scope) or "track_mixer").lower()
    mixer_target_value = (_safe_text(mixer_target) or "volume").lower()

    als_info = _resolve_als_file(project_root, als_file_path=als_file_path)
    if als_info.get("supported") is not True:
        return {
            "ok": True,
            "supported": False,
            "reason": als_info.get("reason") or "als_unavailable",
            "track_index": int(track_index),
            "target": {
                "scope": scope_value,
                "mixer_target": mixer_target_value if scope_value == "track_mixer" else None,
                "send_index": send_index,
                "device_index": device_index,
                "parameter_index": parameter_index,
            },
            "envelope_exists": None,
            "point_access_supported": False,
            "points": [],
            "warnings": list(als_info.get("warnings") or []),
            "als_file_path": als_info.get("als_file_path"),
        }

    als_path = _safe_text(als_info.get("als_file_path"))
    root, read_error = _read_xml_from_als_file(als_path)
    if root is None:
        return {
            "ok": True,
            "supported": False,
            "reason": "als_parse_failed",
            "track_index": int(track_index),
            "target": {
                "scope": scope_value,
                "mixer_target": mixer_target_value if scope_value == "track_mixer" else None,
                "send_index": send_index,
                "device_index": device_index,
                "parameter_index": parameter_index,
            },
            "envelope_exists": None,
            "point_access_supported": False,
            "points": [],
            "warnings": list(als_info.get("warnings") or []) + ([read_error] if read_error else []),
            "als_file_path": als_path,
        }

    live_set = root.find("./LiveSet")
    if live_set is None:
        return {
            "ok": True,
            "supported": False,
            "reason": "live_set_node_missing",
            "track_index": int(track_index),
            "target": {"scope": scope_value},
            "envelope_exists": None,
            "point_access_supported": False,
            "points": [],
            "warnings": list(als_info.get("warnings") or []) + ["live_set_node_missing"],
            "als_file_path": als_path,
        }

    normal_tracks = _iter_normal_track_nodes(live_set)
    if track_index < 0 or track_index >= len(normal_tracks):
        return {
            "ok": False,
            "error": "invalid_track_index",
            "message": "track_index out of range for ALS normal tracks",
            "track_index": int(track_index),
            "track_count": len(normal_tracks),
            "als_file_path": als_path,
        }

    track_node = normal_tracks[track_index]
    track_name = _track_name_from_als(track_node)
    warnings: List[str] = list(als_info.get("warnings") or [])
    warnings.append("saved_als_snapshot_only")

    target_id: Optional[int] = None
    target_debug: Dict[str, Any] = {}
    if scope_value == "track_mixer":
        target_id, target_info = _resolve_track_mixer_target_id(track_node, mixer_target_value, send_index)
        target_debug = target_info if isinstance(target_info, dict) else {}
    elif scope_value == "device_parameter":
        target_id, target_info = _resolve_device_parameter_target_id(
            track_node, _safe_int(device_index), _safe_int(parameter_index)
        )
        target_debug = target_info if isinstance(target_info, dict) else {}
    else:
        return {
            "ok": False,
            "error": "invalid_scope",
            "message": "scope must be 'track_mixer' or 'device_parameter'",
            "track_index": int(track_index),
            "scope": scope_value,
            "als_file_path": als_path,
        }

    if target_id is None:
        result: Dict[str, Any] = {
            "ok": True,
            "supported": True,
            "track_index": int(track_index),
            "track_name": track_name,
            "target": {
                "scope": scope_value,
                "mixer_target": mixer_target_value if scope_value == "track_mixer" else None,
                "send_index": _safe_int(send_index),
                "device_index": _safe_int(device_index),
                "parameter_index": _safe_int(parameter_index),
            },
            "envelope_exists": False,
            "point_access_supported": True,
            "points": [],
            "warnings": warnings,
            "als_file_path": als_path,
            "als_file_mtime_utc": als_info.get("als_file_mtime_utc"),
            "source": "als_file",
        }
        if isinstance(target_debug, dict):
            if target_debug.get("error"):
                result["resolution_error"] = target_debug.get("error")
            if target_debug.get("message"):
                result["resolution_message"] = target_debug.get("message")
            debug_payload = target_debug.get("debug")
            if isinstance(debug_payload, dict):
                result["target_debug"] = debug_payload
            if target_debug.get("device_name_hint"):
                result["target"]["device_name_hint"] = target_debug.get("device_name_hint")
            if target_debug.get("parameter_name_hint"):
                result["target"]["parameter_name_hint"] = target_debug.get("parameter_name_hint")
            if target_debug.get("parameter_display_name_hint"):
                result["target"]["parameter_display_name_hint"] = target_debug.get("parameter_display_name_hint")
            if target_debug.get("parameter_xml_tag"):
                result["target"]["parameter_xml_tag"] = target_debug.get("parameter_xml_tag")
            if target_debug.get("device_tag"):
                result["target"]["device_xml_tag"] = target_debug.get("device_tag")
        return result

    envelope = _find_track_automation_envelope(track_node, target_id)
    points = _parse_envelope_points(envelope, start_time_beats=start_time_beats, end_time_beats=end_time_beats) if envelope is not None else []

    target_payload: Dict[str, Any] = {
        "scope": scope_value,
        "automation_target_id": int(target_id),
    }
    if scope_value == "track_mixer":
        target_payload["mixer_target"] = mixer_target_value
        if mixer_target_value == "send":
            target_payload["send_index"] = _safe_int(send_index)
    else:
        target_payload["device_index"] = _safe_int(device_index)
        target_payload["parameter_index"] = _safe_int(parameter_index)
        if isinstance(target_debug, dict):
            if target_debug.get("device_name_hint"):
                target_payload["device_name_hint"] = target_debug.get("device_name_hint")
            if target_debug.get("parameter_name_hint"):
                target_payload["parameter_name_hint"] = target_debug.get("parameter_name_hint")
            if target_debug.get("parameter_display_name_hint"):
                target_payload["parameter_display_name_hint"] = target_debug.get("parameter_display_name_hint")
            if target_debug.get("parameter_xml_tag"):
                target_payload["parameter_xml_tag"] = target_debug.get("parameter_xml_tag")
            if target_debug.get("device_tag"):
                target_payload["device_xml_tag"] = target_debug.get("device_tag")

    return {
        "ok": True,
        "supported": True,
        "source": "als_file",
        "source_kind": "arrangement_automation",
        "track_index": int(track_index),
        "track_name": track_name,
        "target": target_payload,
        "envelope_exists": envelope is not None,
        "point_access_supported": True,
        "points": points,
        "warnings": warnings,
        "als_file_path": als_path,
        "als_file_mtime_utc": als_info.get("als_file_mtime_utc"),
        "als_candidate_count": als_info.get("candidate_count"),
    }


def enumerate_non_track_arrangement_automation_from_project_als(
    project_root: Optional[str],
    als_file_path: Optional[str] = None,
    start_time_beats: Optional[float] = None,
    end_time_beats: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Enumerate return/master/global arrangement automation points from a saved .als file.

    This complements `read_arrangement_automation_from_project_als`, which targets normal tracks
    only. Returned payloads mirror the same point schema (`points`, `envelope_exists`, etc.)
    so server-side callers can summarize them uniformly.
    """
    als_info = _resolve_als_file(project_root, als_file_path=als_file_path)
    if als_info.get("supported") is not True:
        return {
            "ok": True,
            "supported": False,
            "reason": als_info.get("reason") or "als_unavailable",
            "returns": [],
            "master": None,
            "global": None,
            "warnings": list(als_info.get("warnings") or []),
            "als_file_path": als_info.get("als_file_path"),
        }

    als_path = _safe_text(als_info.get("als_file_path"))
    root, read_error = _read_xml_from_als_file(als_path)
    if root is None:
        return {
            "ok": True,
            "supported": False,
            "reason": "als_parse_failed",
            "returns": [],
            "master": None,
            "global": None,
            "warnings": list(als_info.get("warnings") or []) + ([read_error] if read_error else []),
            "als_file_path": als_path,
        }

    live_set = root.find("./LiveSet")
    if live_set is None:
        return {
            "ok": True,
            "supported": False,
            "reason": "live_set_node_missing",
            "returns": [],
            "master": None,
            "global": None,
            "warnings": list(als_info.get("warnings") or []) + ["live_set_node_missing"],
            "als_file_path": als_path,
        }

    warnings: List[str] = list(als_info.get("warnings") or [])
    if "saved_als_snapshot_only" not in warnings:
        warnings.append("saved_als_snapshot_only")

    def _probe_track_mixer_target_from_node(
        *,
        track_node: ET.Element,
        track_scope: str,
        index: Optional[int],
        track_name: Optional[str],
        mixer_target: str,
        send_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        target_id, target_info = _resolve_track_mixer_target_id(track_node, mixer_target, send_index)
        target_payload: Dict[str, Any] = {
            "scope": "track_mixer",
            "mixer_target": mixer_target,
        }
        if mixer_target == "send":
            target_payload["send_index"] = _safe_int(send_index)

        if target_id is None:
            return _make_target_missing_result(
                scope_label=track_scope,
                index=index,
                track_name=track_name,
                target_payload=target_payload,
                warnings=warnings,
                als_info=als_info,
                als_path=als_path,
                target_debug=target_info if isinstance(target_info, dict) else None,
            )

        target_payload["automation_target_id"] = int(target_id)
        envelope = _find_track_automation_envelope(track_node, target_id)
        points = _parse_envelope_points(
            envelope,
            start_time_beats=start_time_beats,
            end_time_beats=end_time_beats
        ) if envelope is not None else []

        return _make_tracklike_probe_result(
            scope_label=track_scope,
            index=index,
            track_name=track_name,
            target_payload=target_payload,
            envelope=envelope,
            points=points,
            warnings=warnings,
            als_info=als_info,
            als_path=als_path,
        )

    def _probe_main_target(
        *,
        main_track_node: ET.Element,
        target_kind: str,
        track_name: Optional[str],
    ) -> Dict[str, Any]:
        target_id, target_info = _resolve_main_track_target_id(main_track_node, target_kind)
        target_payload: Dict[str, Any] = {
            "scope": "main_track",
            "target_kind": target_kind,
        }

        if target_id is None:
            return _make_target_missing_result(
                scope_label="global" if target_kind in {"tempo", "time_signature"} else "master",
                index=None,
                track_name=track_name,
                target_payload=target_payload,
                warnings=warnings,
                als_info=als_info,
                als_path=als_path,
                target_debug=target_info if isinstance(target_info, dict) else None,
            )

        target_payload["automation_target_id"] = int(target_id)
        if isinstance(target_info, dict):
            debug_payload = target_info.get("debug")
            if isinstance(debug_payload, dict):
                if _safe_text(debug_payload.get("parameter_tag")):
                    target_payload["parameter_tag"] = _safe_text(debug_payload.get("parameter_tag"))
        envelope = _find_track_automation_envelope(main_track_node, target_id)
        points = _parse_envelope_points(
            envelope,
            start_time_beats=start_time_beats,
            end_time_beats=end_time_beats
        ) if envelope is not None else []

        scope_label = "global" if target_kind in {"tempo", "time_signature"} else "master"
        return _make_tracklike_probe_result(
            scope_label=scope_label,
            index=None,
            track_name=track_name,
            target_payload=target_payload,
            envelope=envelope,
            points=points,
            warnings=warnings,
            als_info=als_info,
            als_path=als_path,
        )

    returns_out: List[Dict[str, Any]] = []
    return_tracks = _iter_return_track_nodes(live_set)
    for return_index, return_track in enumerate(return_tracks):
        return_name = _track_name_from_als(return_track) or "Return {0}".format(return_index)
        mixer_targets = {
            "volume": _probe_track_mixer_target_from_node(
                track_node=return_track,
                track_scope="return",
                index=return_index,
                track_name=return_name,
                mixer_target="volume",
            ),
            "panning": _probe_track_mixer_target_from_node(
                track_node=return_track,
                track_scope="return",
                index=return_index,
                track_name=return_name,
                mixer_target="panning",
            ),
        }

        send_holders = return_track.findall("./DeviceChain/Mixer/Sends/TrackSendHolder")
        send_rows: List[Dict[str, Any]] = []
        for send_index in range(len(send_holders)):
            send_rows.append({
                "send_index": int(send_index),
                "probe": _probe_track_mixer_target_from_node(
                    track_node=return_track,
                    track_scope="return",
                    index=return_index,
                    track_name=return_name,
                    mixer_target="send",
                    send_index=send_index,
                )
            })

        returns_out.append({
            "index": int(return_index),
            "track_name": return_name,
            "targets": mixer_targets,
            "sends": send_rows,
        })

    main_track = _get_main_track_node(live_set)
    master_out: Optional[Dict[str, Any]] = None
    global_out: Optional[Dict[str, Any]] = None
    if main_track is not None:
        main_name = _track_name_from_als(main_track) or "Main"
        master_out = {
            "track_name": main_name,
            "targets": {
                "volume": _probe_main_target(main_track_node=main_track, target_kind="volume", track_name=main_name),
                "panning": _probe_main_target(main_track_node=main_track, target_kind="panning", track_name=main_name),
                "crossfade": _probe_main_target(main_track_node=main_track, target_kind="crossfade", track_name=main_name),
            }
        }
        global_out = {
            "track_name": main_name,
            "targets": {
                "tempo": _probe_main_target(main_track_node=main_track, target_kind="tempo", track_name=main_name),
                "time_signature": _probe_main_target(main_track_node=main_track, target_kind="time_signature", track_name=main_name),
            }
        }
    else:
        warnings.append("main_track_node_missing")

    return {
        "ok": True,
        "supported": True,
        "source": "als_file",
        "source_kind": "arrangement_automation_non_track_inventory",
        "returns": returns_out,
        "master": master_out,
        "global": global_out,
        "warnings": warnings,
        "als_file_path": als_path,
        "als_file_mtime_utc": als_info.get("als_file_mtime_utc"),
        "als_candidate_count": als_info.get("candidate_count"),
    }


# Exhaustive automation inventory (ALS-first, saved-state exact)
_ALS_AUTOMATION_INVENTORY_SCHEMA_VERSION = 1
_KNOWN_EVENT_TYPES = {"FloatEvent", "BoolEvent", "EnumEvent"}


def _build_parent_map(root: ET.Element) -> Dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in list(parent)}


def _same_tag_sibling_index(node: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> Optional[int]:
    parent = parent_map.get(node)
    if parent is None:
        return None
    same_tag = [child for child in list(parent) if isinstance(child.tag, str) and child.tag == node.tag]
    if len(same_tag) <= 1:
        return None
    for idx, child in enumerate(same_tag):
        if child is node:
            return idx
    return None


def _node_path_from_live_set(node: ET.Element, live_set: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> str:
    segments: List[str] = []
    cur: Optional[ET.Element] = node
    while cur is not None and cur is not live_set:
        if isinstance(cur.tag, str):
            seg = cur.tag
            idx = _same_tag_sibling_index(cur, parent_map)
            if idx is not None:
                seg = "{0}[{1}]".format(seg, int(idx))
            segments.append(seg)
        cur = parent_map.get(cur)
    segments.append("LiveSet")
    return "/".join(reversed(segments))


def _relative_path_from_container(full_path: str, container_path: str) -> str:
    if not isinstance(full_path, str) or not isinstance(container_path, str):
        return full_path if isinstance(full_path, str) else ""
    prefix = container_path.rstrip("/") + "/"
    if full_path.startswith(prefix):
        return full_path[len(prefix):]
    return full_path


def _iter_ancestors(node: ET.Element, parent_map: Dict[ET.Element, ET.Element], stop_at: Optional[ET.Element] = None) -> List[ET.Element]:
    rows: List[ET.Element] = []
    cur = node
    while cur is not None:
        rows.append(cur)
        if stop_at is not None and cur is stop_at:
            break
        cur = parent_map.get(cur)
    return rows


def _find_nearest_ancestor_with_tag(
    node: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
    tag_names: Tuple[str, ...],
    stop_at: Optional[ET.Element] = None
) -> Optional[ET.Element]:
    cur = node
    while cur is not None:
        if isinstance(cur.tag, str) and cur.tag in tag_names:
            return cur
        if stop_at is not None and cur is stop_at:
            return None
        cur = parent_map.get(cur)
    return None


def _has_ancestor_tag(
    node: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
    tag_names: Tuple[str, ...],
    stop_at: Optional[ET.Element] = None
) -> bool:
    cur = parent_map.get(node)
    while cur is not None:
        if isinstance(cur.tag, str) and cur.tag in tag_names:
            return True
        if stop_at is not None and cur is stop_at:
            return False
        cur = parent_map.get(cur)
    return False


def _track_kind_from_xml_tag(track_tag: Optional[str]) -> str:
    tag = _safe_text(track_tag) or ""
    tag_lower = tag.lower()
    if tag_lower == "audiotrack":
        return "audio"
    if tag_lower == "miditrack":
        return "midi"
    if tag_lower == "grouptrack":
        return "group"
    if tag_lower == "returntrack":
        return "return"
    if tag_lower in {"maintrack", "mastertrack"}:
        return "main"
    if tag_lower == "preheartrack":
        return "prehear"
    return tag_lower or "unknown"


def _target_event_summary(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    event_types: List[str] = []
    unsupported_count = 0
    first_time: Optional[float] = None
    last_time: Optional[float] = None
    for row in points:
        if not isinstance(row, dict):
            continue
        event_type = _safe_text(row.get("event_type"))
        if event_type and event_type not in event_types:
            event_types.append(event_type)
        if event_type and event_type not in _KNOWN_EVENT_TYPES:
            unsupported_count += 1
        time_beats = _safe_float(row.get("time_beats"))
        if time_beats is None:
            continue
        if first_time is None or time_beats < first_time:
            first_time = time_beats
        if last_time is None or time_beats > last_time:
            last_time = time_beats
    return {
        "point_count": len(points),
        "event_types": event_types,
        "first_point_time_beats": first_time,
        "last_point_time_beats": last_time,
        "unsupported_event_types_count": unsupported_count,
    }


def _container_locator(scope: str, index: Optional[int]) -> str:
    scope_value = (_safe_text(scope) or "unknown").lower()
    if index is None:
        return scope_value
    return "{0}:{1}".format(scope_value, int(index))


def _device_index_within_devices_parent(device_node: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> Optional[int]:
    parent = parent_map.get(device_node)
    if parent is None or _safe_text(parent.tag) != "Devices":
        return None
    device_siblings = [child for child in list(parent) if isinstance(child.tag, str)]
    for idx, child in enumerate(device_siblings):
        if child is device_node:
            return idx
    return None


def _chain_or_branch_index(node: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> Optional[int]:
    tag = _safe_text(getattr(node, "tag", None))
    if tag not in {"Chain", "Branch"}:
        return None
    parent = parent_map.get(node)
    if parent is None:
        return None
    same_tag = [child for child in list(parent) if isinstance(child.tag, str) and child.tag == tag]
    for idx, child in enumerate(same_tag):
        if child is node:
            return idx
    return None


def _device_path_metadata_for_parameter_node(
    parameter_node: ET.Element,
    container_node: ET.Element,
    live_set: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
) -> Dict[str, Any]:
    ancestors = _iter_ancestors(parameter_node, parent_map, stop_at=container_node)
    device_nodes = [node for node in ancestors if _looks_like_device_node(node)]
    device_nodes = list(reversed(device_nodes))  # outermost -> innermost
    if not device_nodes:
        return {}

    nearest_device = device_nodes[-1]
    device_index_path: List[int] = []
    device_xml_tag_path: List[str] = []
    rack_chain_index_path: List[int] = []
    chain_tag_path: List[str] = []

    for node in device_nodes:
        device_xml_tag_path.append(_safe_text(node.tag) or "unknown")
        idx = _device_index_within_devices_parent(node, parent_map)
        if idx is not None:
            device_index_path.append(int(idx))

        cur = parent_map.get(node)
        chain_nodes_for_device: List[ET.Element] = []
        while cur is not None and cur is not container_node:
            if _safe_text(cur.tag) in {"Chain", "Branch"}:
                chain_nodes_for_device.append(cur)
            cur = parent_map.get(cur)
        for chain_node in reversed(chain_nodes_for_device):
            chain_idx = _chain_or_branch_index(chain_node, parent_map)
            if chain_idx is not None:
                rack_chain_index_path.append(int(chain_idx))
                chain_tag_path.append(_safe_text(chain_node.tag) or "Chain")

    # Deduplicate chain path entries while preserving order (same ancestors repeated per nested device path walk)
    dedup_chain_pairs: List[Tuple[str, int]] = []
    for tag, idx in zip(chain_tag_path, rack_chain_index_path):
        pair = (tag, idx)
        if pair not in dedup_chain_pairs:
            dedup_chain_pairs.append(pair)
    if dedup_chain_pairs:
        chain_tag_path = [p[0] for p in dedup_chain_pairs]
        rack_chain_index_path = [p[1] for p in dedup_chain_pairs]
    else:
        chain_tag_path = []
        rack_chain_index_path = []

    return {
        "nearest_device_node": nearest_device,
        "device_path": {
            "device_index_path": device_index_path,
            "rack_chain_index_path": rack_chain_index_path,
            "rack_chain_tag_path": chain_tag_path,
            "device_xml_tag_path": device_xml_tag_path,
            "als_device_path": _node_path_from_live_set(nearest_device, live_set, parent_map),
        }
    }


def _parameter_index_map_for_device(device_node: ET.Element) -> Dict[int, int]:
    """
    Map automation_target_id -> local parameter index for one device node.

    Uses the existing local collector, which intentionally excludes nested child devices when run
    on a given root device, making it suitable for per-device local parameter ordering.
    """
    rows = _collect_device_parameter_targets(device_node)
    mapping: Dict[int, int] = {}
    for idx, row in enumerate(rows):
        target_id = _safe_int(row.get("automation_target_id")) if isinstance(row, dict) else None
        if target_id is None:
            continue
        # Preserve first occurrence if duplicates happen.
        if target_id not in mapping:
            mapping[target_id] = int(idx)
    return mapping


def _mixer_target_from_parameter_node(
    parameter_node: ET.Element,
    container_scope: str,
    container_node: ET.Element,
    parent_map: Dict[ET.Element, ET.Element]
) -> Dict[str, Any]:
    parameter_tag = _safe_text(parameter_node.tag) or "unknown"
    domain = None
    mixer_target = None
    extra: Dict[str, Any] = {"parameter_tag": parameter_tag}

    if parameter_tag == "Volume":
        mixer_target = "volume"
    elif parameter_tag == "Pan":
        mixer_target = "panning"
    elif parameter_tag == "Send":
        mixer_target = "send"
        holder = _find_nearest_ancestor_with_tag(parameter_node, parent_map, ("TrackSendHolder",), stop_at=container_node)
        if holder is not None:
            send_idx = _same_tag_sibling_index(holder, parent_map)
            if send_idx is None:
                # Some ALS layouts can still have multiple holders but no explicit duplicate-tag indexing
                send_idx = 0
            extra["send_index"] = int(send_idx)

    if container_scope == "track":
        domain = "track_mixer" if mixer_target else "unclassified_track_mixer"
    elif container_scope == "return":
        domain = "return_mixer" if mixer_target else "unclassified_non_track"
    elif container_scope == "prehear":
        domain = "prehear_mixer" if mixer_target else "unclassified_non_track"
    elif container_scope == "main":
        global_tags = {"Tempo", "TimeSignature", "GlobalGrooveAmount"}
        if parameter_tag in global_tags:
            domain = "global_song"
            extra["global_target_tag"] = parameter_tag
            extra["target_kind"] = parameter_tag.lower()
        else:
            domain = "main_mixer" if (mixer_target or parameter_tag) else "unclassified_non_track"
    else:
        domain = "unclassified_non_track"

    if mixer_target:
        extra["mixer_target"] = mixer_target
    return {
        "domain": domain,
        "mixer_target": mixer_target,
        "extra": extra,
    }


def _discover_tracklike_target_rows(
    container_node: ET.Element,
    container_scope: str,
    container_index: Optional[int],
    container_name: Optional[str],
    live_set: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
    normal_track_index_by_node: Dict[ET.Element, int],
    return_track_index_by_node: Dict[ET.Element, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Discover all ALS AutomationTarget nodes under one track-like container (excluding clip descendants).

    Returns (rows, diagnostics).
    """
    container_path = _node_path_from_live_set(container_node, live_set, parent_map)
    locator = _container_locator(container_scope, container_index)
    rows: List[Dict[str, Any]] = []
    duplicate_counter: Dict[int, int] = {}
    device_param_index_cache: Dict[int, Dict[int, int]] = {}

    for parameter_node in container_node.iter():
        if not isinstance(parameter_node.tag, str):
            continue
        if _has_ancestor_tag(parameter_node, parent_map, ("AudioClip", "MidiClip"), stop_at=container_node):
            continue

        target_node = parameter_node.find("./AutomationTarget")
        if target_node is None:
            continue
        target_id = _safe_int(target_node.attrib.get("Id"))
        if target_id is None:
            continue

        duplicate_counter[target_id] = duplicate_counter.get(target_id, 0) + 1

        parameter_node_path = _node_path_from_live_set(parameter_node, live_set, parent_map)
        target_node_path = _node_path_from_live_set(target_node, live_set, parent_map)
        relative_parameter_path = _relative_path_from_container(parameter_node_path, container_path)
        relative_target_path = _relative_path_from_container(target_node_path, container_path)

        row: Dict[str, Any] = {
            "inventory_kind": "automation_target",
            "container_scope": container_scope,
            "container_index": container_index,
            "container_locator": locator,
            "container_name": container_name,
            "container_track_kind": _track_kind_from_xml_tag(_safe_text(container_node.tag)),
            "als_container_path": container_path,
            "als_parameter_node_path": parameter_node_path,
            "als_target_path": target_node_path,
            "als_parameter_node_path_relative": relative_parameter_path,
            "als_target_path_relative": relative_target_path,
            "automation_target_id": int(target_id),
            "classification": {},
            "location": {
                "track_index": int(container_index) if container_scope == "track" and container_index is not None else None,
                "return_index": int(container_index) if container_scope == "return" and container_index is not None else None,
                "track_name": container_name,
                "track_scope": container_scope,
                "clip_scope": None,
                "clip_name": None,
                "clip_type": None,
            },
            "warnings": [],
        }

        # Classify mixer targets first.
        if "/DeviceChain/Mixer/" in parameter_node_path:
            mixer_info = _mixer_target_from_parameter_node(
                parameter_node=parameter_node,
                container_scope=container_scope,
                container_node=container_node,
                parent_map=parent_map,
            )
            classification = {
                "domain": mixer_info.get("domain"),
                "scope": "track_mixer" if mixer_info.get("mixer_target") else "non_track_mixer",
                "mixer_target": mixer_info.get("mixer_target"),
            }
            extra = mixer_info.get("extra") if isinstance(mixer_info.get("extra"), dict) else {}
            for key, value in extra.items():
                classification[key] = value
            row["classification"] = classification
        else:
            device_meta = _device_path_metadata_for_parameter_node(
                parameter_node=parameter_node,
                container_node=container_node,
                live_set=live_set,
                parent_map=parent_map,
            )
            nearest_device = device_meta.get("nearest_device_node") if isinstance(device_meta, dict) else None
            if isinstance(nearest_device, ET.Element):
                device_node = nearest_device
                device_name_hint = _device_name_hint_from_node(device_node)
                param_xml_tag = _safe_text(parameter_node.tag)
                param_display_name_hint = None
                macro_match = re.match(r"^MacroControls\.(\d+)$", str(param_xml_tag or ""))
                if macro_match:
                    macro_index = _safe_int(macro_match.group(1))
                    if macro_index is not None:
                        param_display_name_hint = _macro_display_name_for_device(device_node, macro_index)

                cache_key = id(device_node)
                target_to_param_index = device_param_index_cache.get(cache_key)
                if target_to_param_index is None:
                    target_to_param_index = _parameter_index_map_for_device(device_node)
                    device_param_index_cache[cache_key] = target_to_param_index

                device_path_payload = device_meta.get("device_path") if isinstance(device_meta.get("device_path"), dict) else {}
                device_index_path = device_path_payload.get("device_index_path", []) if isinstance(device_path_payload, dict) else []
                legacy_top_level_device_index = None
                if isinstance(device_index_path, list) and len(device_index_path) == 1:
                    legacy_top_level_device_index = _safe_int(device_index_path[0])
                legacy_top_level_parameter_index = None
                local_parameter_index = target_to_param_index.get(int(target_id)) if isinstance(target_to_param_index, dict) else None
                if legacy_top_level_device_index is not None and local_parameter_index is not None:
                    legacy_top_level_parameter_index = int(local_parameter_index)

                domain_map = {
                    "track": "track_device_parameter",
                    "return": "return_device_parameter",
                    "main": "main_device_parameter",
                    "prehear": "prehear_device_parameter",
                }
                classification = {
                    "domain": domain_map.get(container_scope, "unclassified_tracklike"),
                    "scope": "device_parameter",
                    "device_name_hint": device_name_hint,
                    "parameter_name_hint": param_xml_tag,
                    "parameter_display_name_hint": param_display_name_hint,
                    "parameter_xml_tag": param_xml_tag,
                    "device_path": device_path_payload,
                    "device_parameter_index_local": int(local_parameter_index) if local_parameter_index is not None else None,
                    "legacy_top_level_device_index": legacy_top_level_device_index,
                    "legacy_top_level_parameter_index": legacy_top_level_parameter_index,
                }
                row["classification"] = classification
            else:
                row["classification"] = {
                    "domain": "unclassified_tracklike",
                    "scope": "unknown",
                    "parameter_xml_tag": _safe_text(parameter_node.tag),
                }
                row["warnings"].append("unclassified_target_no_device_or_mixer_context")

        rows.append(row)

    duplicate_rows: List[Dict[str, Any]] = []
    duplicate_ids_total = 0
    if rows:
        for row in rows:
            target_id = _safe_int(row.get("automation_target_id"))
            if target_id is None:
                continue
            if int(duplicate_counter.get(target_id, 0)) > 1:
                row_warnings = row.get("warnings")
                if not isinstance(row_warnings, list):
                    row_warnings = []
                    row["warnings"] = row_warnings
                if "duplicate_target_id_in_container" not in row_warnings:
                    row_warnings.append("duplicate_target_id_in_container")
                duplicate_rows.append({
                    "container_locator": locator,
                    "automation_target_id": int(target_id),
                    "als_target_path": row.get("als_target_path"),
                })
        duplicate_ids_total = sum(max(0, count - 1) for count in duplicate_counter.values())

    diagnostics = {
        "duplicate_rows": duplicate_rows,
        "duplicate_ids_total": int(duplicate_ids_total),
    }
    return rows, diagnostics


def _discover_tracklike_envelope_rows(
    container_node: ET.Element,
    container_scope: str,
    container_index: Optional[int],
    container_name: Optional[str],
    live_set: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
    start_time_beats: Optional[float],
    end_time_beats: Optional[float],
) -> List[Dict[str, Any]]:
    container_path = _node_path_from_live_set(container_node, live_set, parent_map)
    locator = _container_locator(container_scope, container_index)
    rows: List[Dict[str, Any]] = []
    for env_index, env in enumerate(container_node.findall("./AutomationEnvelopes/Envelopes/AutomationEnvelope")):
        pointee = env.find("./EnvelopeTarget/PointeeId")
        target_id = _safe_int(pointee.attrib.get("Value") if pointee is not None else None)
        if target_id is None:
            continue
        points = _parse_envelope_points(env, start_time_beats=start_time_beats, end_time_beats=end_time_beats)
        env_path = _node_path_from_live_set(env, live_set, parent_map)
        rows.append({
            "inventory_kind": "automation_envelope",
            "envelope_kind": "AutomationEnvelope",
            "container_scope": container_scope,
            "container_index": container_index,
            "container_locator": locator,
            "container_name": container_name,
            "als_container_path": container_path,
            "als_envelope_path": env_path,
            "als_envelope_path_relative": _relative_path_from_container(env_path, container_path),
            "automation_target_id": int(target_id),
            "points": points,
            "summary": _target_event_summary(points),
            "envelope_index": int(env_index),
        })
    return rows


def _discover_clip_envelope_rows(
    live_set: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
    normal_track_index_by_node: Dict[ET.Element, int],
    return_track_index_by_node: Dict[ET.Element, int],
    include_arrangement_clip_envelopes: bool,
    include_session_clip_envelopes: bool,
    start_time_beats: Optional[float],
    end_time_beats: Optional[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    session_clip_candidates = 0
    arrangement_clip_candidates = 0
    session_clip_envelope_rows = 0
    arrangement_clip_envelope_rows = 0
    clip_nodes_seen = 0

    for clip_node in live_set.iter():
        if not isinstance(clip_node.tag, str) or clip_node.tag not in {"AudioClip", "MidiClip"}:
            continue
        clip_nodes_seen += 1
        ancestors = _iter_ancestors(clip_node, parent_map, stop_at=live_set)
        ancestor_tags = {_safe_text(node.tag) for node in ancestors if isinstance(node.tag, str)}
        is_arrangement = "ArrangerAutomation" in ancestor_tags
        is_session = "ClipSlot" in ancestor_tags or "ClipSlots" in ancestor_tags

        if is_session:
            session_clip_candidates += 1
        if is_arrangement:
            arrangement_clip_candidates += 1

        clip_scope: Optional[str] = None
        container_scope: Optional[str] = None
        if is_arrangement and include_arrangement_clip_envelopes:
            clip_scope = "arrangement"
            container_scope = "clip_arrangement"
        elif is_session and include_session_clip_envelopes:
            clip_scope = "session"
            container_scope = "clip_session"

        if clip_scope is None or container_scope is None:
            continue

        clip_name = None
        name_node = clip_node.find("./Name")
        if name_node is not None:
            clip_name = _safe_text(name_node.attrib.get("Value")) or _safe_text(name_node.text)
        if clip_name is None:
            clip_name = _safe_text(clip_node.findtext("./Name/EffectiveName"))

        clip_path = _node_path_from_live_set(clip_node, live_set, parent_map)

        # Locate owning track container (normal or return)
        owning_track = _find_nearest_ancestor_with_tag(
            clip_node,
            parent_map,
            ("AudioTrack", "MidiTrack", "GroupTrack", "ReturnTrack"),
            stop_at=live_set
        )
        track_scope = None
        track_index = None
        return_index = None
        track_name = None
        if isinstance(owning_track, ET.Element):
            if owning_track.tag == "ReturnTrack":
                track_scope = "return"
                idx = return_track_index_by_node.get(owning_track)
                return_index = int(idx) if idx is not None else None
            else:
                track_scope = "track"
                idx = normal_track_index_by_node.get(owning_track)
                track_index = int(idx) if idx is not None else None
            track_name = _track_name_from_als(owning_track)

        clip_envelope_nodes: List[ET.Element] = []
        for tag_name in ("ClipEnvelope", "AutomationEnvelope"):
            for env in clip_node.findall(".//{0}".format(tag_name)):
                if isinstance(env.tag, str):
                    clip_envelope_nodes.append(env)

        seen_env_nodes: List[ET.Element] = []
        dedup_envs: List[ET.Element] = []
        for env in clip_envelope_nodes:
            if env in seen_env_nodes:
                continue
            seen_env_nodes.append(env)
            dedup_envs.append(env)

        for env in dedup_envs:
            pointee = env.find("./EnvelopeTarget/PointeeId")
            target_id = _safe_int(pointee.attrib.get("Value") if pointee is not None else None)
            if target_id is None:
                continue
            points = _parse_envelope_points(env, start_time_beats=start_time_beats, end_time_beats=end_time_beats)
            env_path = _node_path_from_live_set(env, live_set, parent_map)
            rows.append({
                "inventory_kind": "clip_envelope",
                "envelope_kind": _safe_text(env.tag) or "ClipEnvelope",
                "container_scope": container_scope,
                "container_index": None,
                "container_locator": "{0}:{1}".format(container_scope, clip_path),
                "container_name": clip_name,
                "als_container_path": clip_path,
                "als_envelope_path": env_path,
                "als_envelope_path_relative": _relative_path_from_container(env_path, clip_path),
                "automation_target_id": int(target_id),
                "points": points,
                "summary": _target_event_summary(points),
                "classification": {
                    "domain": "clip_envelope",
                    "scope": "clip_envelope",
                    "clip_scope": clip_scope,
                    "target_metadata_resolved": False,
                },
                "location": {
                    "track_scope": track_scope,
                    "track_index": track_index,
                    "return_index": return_index,
                    "track_name": track_name,
                    "clip_scope": clip_scope,
                    "clip_name": clip_name,
                    "clip_type": "audio" if clip_node.tag == "AudioClip" else "midi",
                    "clip_als_path": clip_path,
                },
                "warnings": [],
            })
            if clip_scope == "session":
                session_clip_envelope_rows += 1
            else:
                arrangement_clip_envelope_rows += 1

    diagnostics = {
        "session_clip_candidates_detected": int(session_clip_candidates),
        "arrangement_clip_candidates_detected": int(arrangement_clip_candidates),
        "session_clip_envelope_rows": int(session_clip_envelope_rows),
        "arrangement_clip_envelope_rows": int(arrangement_clip_envelope_rows),
        "clip_nodes_seen": int(clip_nodes_seen),
    }
    return rows, diagnostics


def _stable_inventory_key(row: Dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return "invalid"
    kind = _safe_text(row.get("inventory_kind")) or "target"
    container_locator = _safe_text(row.get("container_locator")) or "unknown"
    target_id = _safe_int(row.get("automation_target_id"))
    target_id_text = "none" if target_id is None else str(int(target_id))
    target_path = _safe_text(row.get("als_target_path_relative")) or _safe_text(row.get("als_target_path")) or ""
    env_path = _safe_text(row.get("als_envelope_path_relative")) or _safe_text(row.get("als_envelope_path")) or ""
    return "{0}|{1}|{2}|{3}|{4}".format(kind, container_locator, target_id_text, target_path, env_path)


def _build_target_row_from_tracklike_target(
    target_row: Dict[str, Any],
    envelope_row: Optional[Dict[str, Any]],
    duplicate_target_id_in_container: bool,
) -> Dict[str, Any]:
    classification = target_row.get("classification") if isinstance(target_row.get("classification"), dict) else {}
    points = list(envelope_row.get("points") or []) if isinstance(envelope_row, dict) and isinstance(envelope_row.get("points"), list) else []
    summary = envelope_row.get("summary") if isinstance(envelope_row, dict) and isinstance(envelope_row.get("summary"), dict) else _target_event_summary(points)

    out: Dict[str, Any] = {
        "inventory_kind": "automation_target",
        "container_scope": target_row.get("container_scope"),
        "container_index": target_row.get("container_index"),
        "container_locator": target_row.get("container_locator"),
        "container_name": target_row.get("container_name"),
        "container_track_kind": target_row.get("container_track_kind"),
        "automation_target_id": target_row.get("automation_target_id"),
        "classification": classification,
        "location": target_row.get("location") if isinstance(target_row.get("location"), dict) else {},
        "als_container_path": target_row.get("als_container_path"),
        "als_target_path": target_row.get("als_target_path"),
        "als_target_path_relative": target_row.get("als_target_path_relative"),
        "als_parameter_node_path": target_row.get("als_parameter_node_path"),
        "als_parameter_node_path_relative": target_row.get("als_parameter_node_path_relative"),
        "envelope": {
            "exists": envelope_row is not None,
            "envelope_kind": envelope_row.get("envelope_kind") if isinstance(envelope_row, dict) else None,
            "als_envelope_path": envelope_row.get("als_envelope_path") if isinstance(envelope_row, dict) else None,
            "point_count": summary.get("point_count"),
            "event_types": summary.get("event_types"),
            "first_point_time_beats": summary.get("first_point_time_beats"),
            "last_point_time_beats": summary.get("last_point_time_beats"),
        },
        "points": points,
        "warnings": list(target_row.get("warnings") or []) if isinstance(target_row.get("warnings"), list) else [],
    }
    if duplicate_target_id_in_container and "duplicate_target_id_in_container" not in out["warnings"]:
        out["warnings"].append("duplicate_target_id_in_container")
    if isinstance(envelope_row, dict):
        for warning in list(envelope_row.get("warnings") or []):
            if isinstance(warning, str) and warning not in out["warnings"]:
                out["warnings"].append(warning)
        if envelope_row.get("summary", {}).get("unsupported_event_types_count"):
            out["warnings"].append("unsupported_event_types_present")
    out["inventory_key"] = _stable_inventory_key(out)
    out["target_ref"] = {
        "schema_version": _ALS_AUTOMATION_INVENTORY_SCHEMA_VERSION,
        "inventory_key": out["inventory_key"],
        "source": "als_inventory",
        "container_scope": out.get("container_scope"),
        "container_locator": out.get("container_locator"),
        "automation_target_id": out.get("automation_target_id"),
    }
    return out


def _build_target_row_from_clip_envelope_row(clip_row: Dict[str, Any]) -> Dict[str, Any]:
    points = list(clip_row.get("points") or []) if isinstance(clip_row.get("points"), list) else []
    summary = clip_row.get("summary") if isinstance(clip_row.get("summary"), dict) else _target_event_summary(points)
    out: Dict[str, Any] = {
        "inventory_kind": "clip_envelope",
        "container_scope": clip_row.get("container_scope"),
        "container_index": None,
        "container_locator": clip_row.get("container_locator"),
        "container_name": clip_row.get("container_name"),
        "automation_target_id": clip_row.get("automation_target_id"),
        "classification": clip_row.get("classification") if isinstance(clip_row.get("classification"), dict) else {
            "domain": "clip_envelope",
            "scope": "clip_envelope",
            "clip_scope": "arrangement",
            "target_metadata_resolved": False,
        },
        "location": clip_row.get("location") if isinstance(clip_row.get("location"), dict) else {},
        "als_container_path": clip_row.get("als_container_path"),
        "als_target_path": None,
        "als_target_path_relative": None,
        "als_parameter_node_path": None,
        "als_parameter_node_path_relative": None,
        "envelope": {
            "exists": True,
            "envelope_kind": clip_row.get("envelope_kind"),
            "als_envelope_path": clip_row.get("als_envelope_path"),
            "point_count": summary.get("point_count"),
            "event_types": summary.get("event_types"),
            "first_point_time_beats": summary.get("first_point_time_beats"),
            "last_point_time_beats": summary.get("last_point_time_beats"),
        },
        "points": points,
        "warnings": list(clip_row.get("warnings") or []) if isinstance(clip_row.get("warnings"), list) else [],
    }
    if summary.get("unsupported_event_types_count"):
        out["warnings"].append("unsupported_event_types_present")
    out["inventory_key"] = _stable_inventory_key(out)
    out["target_ref"] = {
        "schema_version": _ALS_AUTOMATION_INVENTORY_SCHEMA_VERSION,
        "inventory_key": out["inventory_key"],
        "source": "als_inventory",
        "container_scope": out.get("container_scope"),
        "container_locator": out.get("container_locator"),
        "automation_target_id": out.get("automation_target_id"),
    }
    return out


def _build_als_inventory_sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    scope_order_map = {
        "track": 0,
        "return": 1,
        "main": 2,
        "prehear": 3,
        "clip_arrangement": 4,
        "clip_session": 5,
    }
    scope = _safe_text(row.get("container_scope")) or "unknown"
    classification = row.get("classification") if isinstance(row.get("classification"), dict) else {}
    location = row.get("location") if isinstance(row.get("location"), dict) else {}
    return (
        scope_order_map.get(scope, 99),
        _safe_int(row.get("container_index")) if _safe_int(row.get("container_index")) is not None else 10**9,
        _safe_int(location.get("track_index")) if _safe_int(location.get("track_index")) is not None else 10**9,
        _safe_int(location.get("return_index")) if _safe_int(location.get("return_index")) is not None else 10**9,
        _safe_text(location.get("clip_als_path")) or "",
        _safe_text(classification.get("domain")) or "",
        _safe_int(row.get("automation_target_id")) if _safe_int(row.get("automation_target_id")) is not None else 10**9,
        _safe_text(row.get("als_target_path")) or _safe_text(row.get("als_container_path")) or "",
        _safe_text(row.get("inventory_key")) or "",
    )


def build_als_automation_inventory(
    project_root: Optional[str],
    als_file_path: Optional[str] = None,
    include_arrangement_clip_envelopes: bool = True,
    include_session_clip_envelopes: bool = True,
    start_time_beats: Optional[float] = None,
    end_time_beats: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build an exhaustive saved-state automation inventory from a Live 12.3.x .als file.

    This is ALS-first and read-only. It is intended as the canonical source for exact automation
    visibility in supported domains.
    """
    als_info = _resolve_als_file(project_root, als_file_path=als_file_path)
    if als_info.get("supported") is not True:
        return {
            "ok": True,
            "supported": False,
            "schema_version": _ALS_AUTOMATION_INVENTORY_SCHEMA_VERSION,
            "reason": als_info.get("reason") or "als_unavailable",
            "source": "als_file",
            "targets": [],
            "orphan_envelopes": [],
            "unclassified_targets": [],
            "warnings": list(als_info.get("warnings") or []),
            "als_file_path": als_info.get("als_file_path"),
            "completeness": {
                "status": "partial",
                "targets_discovered_total": 0,
                "envelopes_discovered_total": 0,
                "targets_with_envelopes": 0,
                "targets_without_envelopes": 0,
                "orphan_envelopes_total": 0,
                "unclassified_targets_total": 0,
                "duplicate_target_ids_total": 0,
                "unsupported_event_types_total": 0,
                "warnings": list(als_info.get("warnings") or []),
            },
        }

    als_path = _safe_text(als_info.get("als_file_path"))
    root, read_error = _read_xml_from_als_file(als_path)
    if root is None:
        warnings = list(als_info.get("warnings") or [])
        if read_error:
            warnings.append(read_error)
        return {
            "ok": True,
            "supported": False,
            "schema_version": _ALS_AUTOMATION_INVENTORY_SCHEMA_VERSION,
            "reason": "als_parse_failed",
            "source": "als_file",
            "targets": [],
            "orphan_envelopes": [],
            "unclassified_targets": [],
            "warnings": warnings,
            "als_file_path": als_path,
            "completeness": {
                "status": "partial",
                "targets_discovered_total": 0,
                "envelopes_discovered_total": 0,
                "targets_with_envelopes": 0,
                "targets_without_envelopes": 0,
                "orphan_envelopes_total": 0,
                "unclassified_targets_total": 0,
                "duplicate_target_ids_total": 0,
                "unsupported_event_types_total": 0,
                "warnings": warnings,
            },
        }

    live_set = root.find("./LiveSet")
    if live_set is None:
        warnings = list(als_info.get("warnings") or []) + ["live_set_node_missing"]
        return {
            "ok": True,
            "supported": False,
            "schema_version": _ALS_AUTOMATION_INVENTORY_SCHEMA_VERSION,
            "reason": "live_set_node_missing",
            "source": "als_file",
            "targets": [],
            "orphan_envelopes": [],
            "unclassified_targets": [],
            "warnings": warnings,
            "als_file_path": als_path,
            "completeness": {
                "status": "partial",
                "targets_discovered_total": 0,
                "envelopes_discovered_total": 0,
                "targets_with_envelopes": 0,
                "targets_without_envelopes": 0,
                "orphan_envelopes_total": 0,
                "unclassified_targets_total": 0,
                "duplicate_target_ids_total": 0,
                "unsupported_event_types_total": 0,
                "warnings": warnings,
            },
        }

    parent_map = _build_parent_map(live_set)
    normal_tracks = _iter_normal_track_nodes(live_set)
    return_tracks = _iter_return_track_nodes(live_set)
    main_track = _get_main_track_node(live_set)
    prehear_track = live_set.find("./PreHearTrack")

    normal_track_index_by_node: Dict[ET.Element, int] = {node: idx for idx, node in enumerate(normal_tracks)}
    return_track_index_by_node: Dict[ET.Element, int] = {node: idx for idx, node in enumerate(return_tracks)}

    warnings: List[str] = list(als_info.get("warnings") or [])
    if "saved_als_snapshot_only" not in warnings:
        warnings.append("saved_als_snapshot_only")

    container_rows: List[Tuple[ET.Element, str, Optional[int], Optional[str]]] = []
    for idx, node in enumerate(normal_tracks):
        container_rows.append((node, "track", int(idx), _track_name_from_als(node)))
    for idx, node in enumerate(return_tracks):
        container_rows.append((node, "return", int(idx), _track_name_from_als(node)))
    if isinstance(main_track, ET.Element):
        container_rows.append((main_track, "main", None, _track_name_from_als(main_track) or "Main"))
    else:
        warnings.append("main_track_node_missing")
    if isinstance(prehear_track, ET.Element):
        container_rows.append((prehear_track, "prehear", None, _track_name_from_als(prehear_track) or "PreHear"))
    else:
        warnings.append("prehear_track_node_missing")

    all_target_rows: List[Dict[str, Any]] = []
    all_envelope_rows: List[Dict[str, Any]] = []
    duplicate_target_id_rows: List[Dict[str, Any]] = []
    duplicate_target_ids_total = 0

    for container_node, scope, idx, name in container_rows:
        target_rows, diagnostics = _discover_tracklike_target_rows(
            container_node=container_node,
            container_scope=scope,
            container_index=idx,
            container_name=name,
            live_set=live_set,
            parent_map=parent_map,
            normal_track_index_by_node=normal_track_index_by_node,
            return_track_index_by_node=return_track_index_by_node,
        )
        env_rows = _discover_tracklike_envelope_rows(
            container_node=container_node,
            container_scope=scope,
            container_index=idx,
            container_name=name,
            live_set=live_set,
            parent_map=parent_map,
            start_time_beats=start_time_beats,
            end_time_beats=end_time_beats,
        )
        all_target_rows.extend(target_rows)
        all_envelope_rows.extend(env_rows)
        if isinstance(diagnostics, dict):
            duplicate_target_id_rows.extend(list(diagnostics.get("duplicate_rows") or []))
            duplicate_target_ids_total += int(diagnostics.get("duplicate_ids_total", 0) or 0)

    clip_rows: List[Dict[str, Any]] = []
    clip_diag = {
        "session_clip_candidates_detected": 0,
        "arrangement_clip_candidates_detected": 0,
        "session_clip_envelope_rows": 0,
        "arrangement_clip_envelope_rows": 0,
        "clip_nodes_seen": 0,
    }
    if include_arrangement_clip_envelopes or include_session_clip_envelopes:
        clip_rows, clip_diag = _discover_clip_envelope_rows(
            live_set=live_set,
            parent_map=parent_map,
            normal_track_index_by_node=normal_track_index_by_node,
            return_track_index_by_node=return_track_index_by_node,
            include_arrangement_clip_envelopes=bool(include_arrangement_clip_envelopes),
            include_session_clip_envelopes=bool(include_session_clip_envelopes),
            start_time_beats=start_time_beats,
            end_time_beats=end_time_beats,
        )

    targets_by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in all_target_rows:
        locator = _safe_text(row.get("container_locator")) or "unknown"
        target_id = _safe_int(row.get("automation_target_id"))
        if target_id is None:
            continue
        key = (locator, int(target_id))
        targets_by_key.setdefault(key, []).append(row)

    envelopes_by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in all_envelope_rows:
        locator = _safe_text(row.get("container_locator")) or "unknown"
        target_id = _safe_int(row.get("automation_target_id"))
        if target_id is None:
            continue
        key = (locator, int(target_id))
        envelopes_by_key.setdefault(key, []).append(row)

    matched_env_ids: Dict[int, bool] = {}
    final_targets: List[Dict[str, Any]] = []
    orphan_envelopes: List[Dict[str, Any]] = []
    unsupported_event_types_total = 0

    for key, target_rows in targets_by_key.items():
        envelope_candidates = list(envelopes_by_key.get(key) or [])
        duplicate_target_key = len(target_rows) > 1
        duplicate_envelope_key = len(envelope_candidates) > 1

        if duplicate_target_key and envelope_candidates:
            for env in envelope_candidates:
                orphan_row = {
                    "container_locator": env.get("container_locator"),
                    "container_scope": env.get("container_scope"),
                    "container_index": env.get("container_index"),
                    "container_name": env.get("container_name"),
                    "automation_target_id": env.get("automation_target_id"),
                    "envelope_kind": env.get("envelope_kind"),
                    "als_envelope_path": env.get("als_envelope_path"),
                    "point_count": env.get("summary", {}).get("point_count") if isinstance(env.get("summary"), dict) else None,
                    "event_types": env.get("summary", {}).get("event_types") if isinstance(env.get("summary"), dict) else None,
                    "warnings": ["ambiguous_target_match_duplicate_target_id"],
                }
                orphan_envelopes.append(orphan_row)
                matched_env_ids[id(env)] = True
                unsupported_event_types_total += int((env.get("summary") or {}).get("unsupported_event_types_count", 0) or 0)

            for target_row in target_rows:
                final = _build_target_row_from_tracklike_target(
                    target_row=target_row,
                    envelope_row=None,
                    duplicate_target_id_in_container=True,
                )
                final["warnings"].append("envelope_match_skipped_due_to_duplicate_target_id")
                final_targets.append(final)
            continue

        if duplicate_envelope_key and target_rows:
            # Ambiguous: attach none, surface all envelopes as orphans.
            for env in envelope_candidates:
                orphan_row = {
                    "container_locator": env.get("container_locator"),
                    "container_scope": env.get("container_scope"),
                    "container_index": env.get("container_index"),
                    "container_name": env.get("container_name"),
                    "automation_target_id": env.get("automation_target_id"),
                    "envelope_kind": env.get("envelope_kind"),
                    "als_envelope_path": env.get("als_envelope_path"),
                    "point_count": env.get("summary", {}).get("point_count") if isinstance(env.get("summary"), dict) else None,
                    "event_types": env.get("summary", {}).get("event_types") if isinstance(env.get("summary"), dict) else None,
                    "warnings": ["ambiguous_multiple_envelopes_same_target_id"],
                }
                orphan_envelopes.append(orphan_row)
                matched_env_ids[id(env)] = True
                unsupported_event_types_total += int((env.get("summary") or {}).get("unsupported_event_types_count", 0) or 0)
            for target_row in target_rows:
                final = _build_target_row_from_tracklike_target(
                    target_row=target_row,
                    envelope_row=None,
                    duplicate_target_id_in_container=False,
                )
                final["warnings"].append("envelope_match_skipped_due_to_multiple_envelopes")
                final_targets.append(final)
            continue

        env_row = envelope_candidates[0] if envelope_candidates else None
        if env_row is not None:
            matched_env_ids[id(env_row)] = True
            unsupported_event_types_total += int((env_row.get("summary") or {}).get("unsupported_event_types_count", 0) or 0)
        for target_row in target_rows:
            final_targets.append(_build_target_row_from_tracklike_target(
                target_row=target_row,
                envelope_row=env_row,
                duplicate_target_id_in_container=False,
            ))

    for env in all_envelope_rows:
        if matched_env_ids.get(id(env)):
            continue
        orphan_envelopes.append({
            "container_locator": env.get("container_locator"),
            "container_scope": env.get("container_scope"),
            "container_index": env.get("container_index"),
            "container_name": env.get("container_name"),
            "automation_target_id": env.get("automation_target_id"),
            "envelope_kind": env.get("envelope_kind"),
            "als_envelope_path": env.get("als_envelope_path"),
            "point_count": env.get("summary", {}).get("point_count") if isinstance(env.get("summary"), dict) else None,
            "event_types": env.get("summary", {}).get("event_types") if isinstance(env.get("summary"), dict) else None,
            "warnings": ["orphan_envelope_no_matching_target"],
        })
        unsupported_event_types_total += int((env.get("summary") or {}).get("unsupported_event_types_count", 0) or 0)

    for clip_row in clip_rows:
        unsupported_event_types_total += int((clip_row.get("summary") or {}).get("unsupported_event_types_count", 0) or 0)
        final_targets.append(_build_target_row_from_clip_envelope_row(clip_row))

    # Build unclassified target summaries and mark row warnings.
    unclassified_targets: List[Dict[str, Any]] = []
    for row in final_targets:
        classification = row.get("classification") if isinstance(row.get("classification"), dict) else {}
        domain = _safe_text(classification.get("domain")) or ""
        if domain.startswith("unclassified"):
            if "unclassified_target" not in row.get("warnings", []):
                row.setdefault("warnings", []).append("unclassified_target")
            unclassified_targets.append({
                "inventory_key": row.get("inventory_key"),
                "container_locator": row.get("container_locator"),
                "container_scope": row.get("container_scope"),
                "automation_target_id": row.get("automation_target_id"),
                "domain": domain,
                "als_target_path": row.get("als_target_path"),
                "warnings": list(row.get("warnings") or []),
            })

    final_targets.sort(key=_build_als_inventory_sort_key)
    orphan_envelopes.sort(key=lambda row: (
        _safe_text(row.get("container_scope")) or "",
        _safe_int(row.get("container_index")) if _safe_int(row.get("container_index")) is not None else 10**9,
        _safe_int(row.get("automation_target_id")) if _safe_int(row.get("automation_target_id")) is not None else 10**9,
        _safe_text(row.get("als_envelope_path")) or "",
    ))
    unclassified_targets.sort(key=lambda row: (
        _safe_text(row.get("container_scope")) or "",
        _safe_int(row.get("automation_target_id")) if _safe_int(row.get("automation_target_id")) is not None else 10**9,
        _safe_text(row.get("als_target_path")) or "",
    ))

    targets_with_envelopes = sum(1 for row in final_targets if isinstance(row.get("envelope"), dict) and row["envelope"].get("exists") is True)
    targets_without_envelopes = len(final_targets) - targets_with_envelopes
    envelopes_discovered_total = len(all_envelope_rows) + len(clip_rows)
    session_clip_candidates_detected = int(clip_diag.get("session_clip_candidates_detected", 0) or 0)

    completeness_warnings: List[str] = []
    if duplicate_target_ids_total > 0:
        completeness_warnings.append("duplicate_target_ids_detected")
    if orphan_envelopes:
        completeness_warnings.append("orphan_envelopes_present")
    if unclassified_targets:
        completeness_warnings.append("unclassified_targets_present")
    if unsupported_event_types_total > 0:
        completeness_warnings.append("unsupported_event_types_present")

    completeness_status = "complete"
    if orphan_envelopes or unsupported_event_types_total > 0:
        completeness_status = "partial"

    session = {
        "track_count_normal": len(normal_tracks),
        "track_count_returns": len(return_tracks),
        "has_main_track": isinstance(main_track, ET.Element),
        "has_prehear_track": isinstance(prehear_track, ET.Element),
        "clip_nodes_seen_total": int(clip_diag.get("clip_nodes_seen", 0) or 0),
        "session_clip_candidates_detected": session_clip_candidates_detected,
        "arrangement_clip_candidates_detected": int(clip_diag.get("arrangement_clip_candidates_detected", 0) or 0),
        "session_clip_envelope_rows": int(clip_diag.get("session_clip_envelope_rows", 0) or 0),
        "arrangement_clip_envelope_rows": int(clip_diag.get("arrangement_clip_envelope_rows", 0) or 0),
    }

    scope_statement = {
        "milestone_scope": "saved_state_exact_automation_inventory_live_12_3_x",
        "session_clip_envelopes_excluded": not bool(include_session_clip_envelopes),
        "reason": "session_clip_envelopes_included" if bool(include_session_clip_envelopes) else "user_scoped_out_for_current_milestone",
        "session_clip_envelopes_included": bool(include_session_clip_envelopes),
        "arrangement_clip_envelopes_included": bool(include_arrangement_clip_envelopes),
    }

    return {
        "ok": True,
        "supported": True,
        "schema_version": _ALS_AUTOMATION_INVENTORY_SCHEMA_VERSION,
        "source": "als_file",
        "als_file_path": als_path,
        "als_file_mtime_utc": als_info.get("als_file_mtime_utc"),
        "als_candidate_count": als_info.get("candidate_count"),
        "scope_statement": scope_statement,
        "session": session,
        "targets": final_targets,
        "orphan_envelopes": orphan_envelopes,
        "unclassified_targets": unclassified_targets,
        "duplicate_target_id_rows": duplicate_target_id_rows,
        "completeness": {
            "status": completeness_status,
            "targets_discovered_total": len(final_targets),
            "envelopes_discovered_total": envelopes_discovered_total,
            "targets_with_envelopes": targets_with_envelopes,
            "targets_without_envelopes": targets_without_envelopes,
            "orphan_envelopes_total": len(orphan_envelopes),
            "unclassified_targets_total": len(unclassified_targets),
            "duplicate_target_ids_total": int(duplicate_target_ids_total),
            "unsupported_event_types_total": int(unsupported_event_types_total),
            "warnings": completeness_warnings,
        },
        "warnings": warnings,
    }


def get_als_automation_target_points_from_inventory(
    inventory_payload: Dict[str, Any],
    target_ref: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve exact points for one target from a canonical ALS inventory payload.
    """
    if not isinstance(inventory_payload, dict):
        return {
            "ok": False,
            "error": "invalid_inventory_payload",
            "message": "inventory_payload must be a dict"
        }
    if not isinstance(target_ref, dict):
        return {
            "ok": False,
            "error": "invalid_target_ref",
            "message": "target_ref must be an object"
        }

    inventory_key = _safe_text(target_ref.get("inventory_key"))
    if not inventory_key:
        return {
            "ok": False,
            "error": "invalid_target_ref",
            "message": "target_ref.inventory_key is required"
        }

    for row in list(inventory_payload.get("targets") or []):
        if not isinstance(row, dict):
            continue
        if _safe_text(row.get("inventory_key")) != inventory_key:
            continue
        envelope = row.get("envelope") if isinstance(row.get("envelope"), dict) else {}
        points = list(row.get("points") or []) if isinstance(row.get("points"), list) else []
        return {
            "ok": True,
            "source": "als_inventory",
            "inventory_key": inventory_key,
            "target_ref": row.get("target_ref"),
            "classification": row.get("classification"),
            "location": row.get("location"),
            "container_scope": row.get("container_scope"),
            "container_locator": row.get("container_locator"),
            "automation_target_id": row.get("automation_target_id"),
            "envelope_exists": bool(envelope.get("exists", False)),
            "envelope_kind": envelope.get("envelope_kind"),
            "als_envelope_path": envelope.get("als_envelope_path"),
            "points": points,
            "warnings": list(row.get("warnings") or []) if isinstance(row.get("warnings"), list) else [],
        }

    return {
        "ok": False,
        "error": "target_ref_not_found",
        "message": "No target with the requested inventory_key exists in the inventory payload",
        "inventory_key": inventory_key,
    }

# ableton_mcp_server.py
from mcp.server.fastmcp import FastMCP, Context
import socket
import json
import logging
import os
import hashlib
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Union, Optional, Tuple
from MCP_Server.audio_analysis import analyze_wav_file, AudioAnalysisError
from MCP_Server.pathing import (
    get_export_dir,
    get_analysis_dir,
    ensure_dirs_exist,
    bootstrap_project_environment,
    resolve_pathing,
)

try:
    import numpy as np
except Exception:
    np = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None

try:
    from scipy import signal as sp_signal
except Exception:
    sp_signal = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AbletonMCPServer")

@dataclass
class AbletonConnection:
    host: str
    port: int
    sock: socket.socket = None
    
    def connect(self) -> bool:
        """Connect to the Ableton Remote Script socket server"""
        if self.sock:
            return True
            
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Ableton at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ableton: {str(e)}")
            self.sock = None
            return False
    
    def disconnect(self):
        """Disconnect from the Ableton Remote Script"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Ableton: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self, sock, buffer_size=8192):
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        sock.settimeout(15.0)  # Increased timeout for operations that might take longer
        
        try:
            while True:
                try:
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        if not chunks:
                            raise Exception("Connection closed before receiving any data")
                        break
                    
                    chunks.append(chunk)
                    
                    # Check if we've received a complete JSON object
                    try:
                        data = b''.join(chunks)
                        json.loads(data.decode('utf-8'))
                        logger.info(f"Received complete response ({len(data)} bytes)")
                        return data
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue receiving
                        continue
                except socket.timeout:
                    logger.warning("Socket timeout during chunked receive")
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f"Socket connection error during receive: {str(e)}")
                    raise
        except Exception as e:
            logger.error(f"Error during receive: {str(e)}")
            raise
            
        # If we get here, we either timed out or broke out of the loop
        if chunks:
            data = b''.join(chunks)
            logger.info(f"Returning data after receive completion ({len(data)} bytes)")
            try:
                json.loads(data.decode('utf-8'))
                return data
            except json.JSONDecodeError:
                raise Exception("Incomplete JSON response received")
        else:
            raise Exception("No data received")

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to Ableton and return the response"""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Ableton")
        
        command = {
            "type": command_type,
            "params": params or {}
        }
        
        # Check if this is a state-modifying command
        is_modifying_command = command_type in [
            "create_midi_track", "create_audio_track", "set_track_name",
            "create_clip", "add_notes_to_clip", "set_clip_name",
            "set_tempo", "fire_clip", "stop_clip", "set_device_parameter",
            "start_playback", "stop_playback", "load_instrument_or_effect",
            "set_transport_state", "set_tracks_mixer_state"
        ]
        
        try:
            logger.info(f"Sending command: {command_type} with params: {params}")
            
            # Send the command
            self.sock.sendall(json.dumps(command).encode('utf-8'))
            logger.info(f"Command sent, waiting for response...")
            
            # For state-modifying commands, add a small delay to give Ableton time to process
            if is_modifying_command:
                import time
                time.sleep(0.1)  # 100ms delay
            
            # Set timeout based on command type
            timeout = 15.0 if is_modifying_command else 10.0
            self.sock.settimeout(timeout)
            
            # Receive the response
            response_data = self.receive_full_response(self.sock)
            logger.info(f"Received {len(response_data)} bytes of data")
            
            # Parse the response
            response = json.loads(response_data.decode('utf-8'))
            logger.info(f"Response parsed, status: {response.get('status', 'unknown')}")
            
            if response.get("status") == "error":
                logger.error(f"Ableton error: {response.get('message')}")
                raise Exception(response.get("message", "Unknown error from Ableton"))
            
            # For state-modifying commands, add another small delay after receiving response
            if is_modifying_command:
                import time
                time.sleep(0.1)  # 100ms delay
            
            return response.get("result", {})
        except socket.timeout:
            logger.error("Socket timeout while waiting for response from Ableton")
            self.sock = None
            raise Exception("Timeout waiting for Ableton response")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Socket connection error: {str(e)}")
            self.sock = None
            raise Exception(f"Connection to Ableton lost: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Ableton: {str(e)}")
            if 'response_data' in locals() and response_data:
                logger.error(f"Raw response (first 200 bytes): {response_data[:200]}")
            self.sock = None
            raise Exception(f"Invalid response from Ableton: {str(e)}")
        except Exception as e:
            logger.error(f"Error communicating with Ableton: {str(e)}")
            self.sock = None
            raise Exception(f"Communication error with Ableton: {str(e)}")

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    try:
        logger.info("AbletonMCP server starting up")
        try:
            bootstrap_result = bootstrap_project_environment()
            warnings = bootstrap_result.get("warnings", [])
            if isinstance(warnings, list) and warnings:
                for warning in warnings:
                    logger.warning(f"Pathing bootstrap warning: {warning}")
        except Exception as bootstrap_exc:
            logger.warning(f"Pathing bootstrap failed: {str(bootstrap_exc)}")
        
        try:
            ableton = get_ableton_connection()
            logger.info("Successfully connected to Ableton on startup")
        except Exception as e:
            logger.warning(f"Could not connect to Ableton on startup: {str(e)}")
            logger.warning("Make sure the Ableton Remote Script is running")
        
        yield {}
    finally:
        global _ableton_connection
        if _ableton_connection:
            logger.info("Disconnecting from Ableton on shutdown")
            _ableton_connection.disconnect()
            _ableton_connection = None
        logger.info("AbletonMCP server shut down")

# Create the MCP server with lifespan support
mcp = FastMCP(
    "AbletonMCP",
    lifespan=server_lifespan
)

# Global connection for resources
_ableton_connection = None

_SOURCE_CACHE_VERSION = 2
_SOURCE_CACHE_DIR = os.path.expanduser("~/.ableton_mcp_analysis/cache")
_LAUNCH_CWD = os.path.abspath(os.environ.get("ABLETON_MCP_LAUNCH_CWD", os.getcwd()))
_SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".aiff", ".aif", ".mp3", ".m4a", ".flac"}
_ANALYZE_AUDIO_INPUT_FORMATS = {"wav", "mp3", "aif", "aiff", "flac", "m4a"}
_DEVICE_CAPABILITIES_SCHEMA_VERSION = 1
_DEVICE_CAPABILITIES_CACHE_PATH = os.path.join(_SOURCE_CACHE_DIR, "device_capabilities.json")
_DEVICE_INVENTORY_CACHE_SCHEMA_VERSION = 1
_DEVICE_INVENTORY_CACHE_PATH = os.path.join(_SOURCE_CACHE_DIR, "device_inventory_cache.json")
_PROJECT_SNAPSHOT_SCHEMA_VERSION = 1
_PROJECT_SNAPSHOT_DIR = os.path.join(_SOURCE_CACHE_DIR, "project_snapshots")
_AUDIO_ANALYSIS_CACHE_VERSION = 2
_DEFAULT_AUDIO_WINDOW_SEC = 1.0
_DEFAULT_RMS_THRESHOLD_DBFS = -60.0
_DEVICE_CAPABILITY_BUCKETS = [
    "eq",
    "compression",
    "reverb",
    "delay",
    "modulation",
    "distortion",
    "imaging",
    "metering",
    "utility",
    "filter",
    "pitch",
    "dynamics",
    "unknown"
]
_KNOWN_BROWSER_ROOTS = [
    "Audio Effects",
    "Plugins",
    "MIDI Effects",
    "Instruments",
    "Max for Live",
    "Sounds",
    "Drums",
    "Clips",
    "Current Project",
    "Packs",
    "Samples",
    "User Library",
    "User Folders"
]

def get_ableton_connection():
    """Get or create a persistent Ableton connection"""
    global _ableton_connection
    
    if _ableton_connection is not None:
        try:
            # Test the connection with a simple ping
            # We'll try to send an empty message, which should fail if the connection is dead
            # but won't affect Ableton if it's alive
            _ableton_connection.sock.settimeout(1.0)
            _ableton_connection.sock.sendall(b'')
            return _ableton_connection
        except Exception as e:
            logger.warning(f"Existing connection is no longer valid: {str(e)}")
            try:
                _ableton_connection.disconnect()
            except:
                pass
            _ableton_connection = None
    
    # Connection doesn't exist or is invalid, create a new one
    if _ableton_connection is None:
        # Try to connect up to 3 times with a short delay between attempts
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Connecting to Ableton (attempt {attempt}/{max_attempts})...")
                _ableton_connection = AbletonConnection(host="localhost", port=9877)
                if _ableton_connection.connect():
                    logger.info("Created new persistent connection to Ableton")
                    
                    # Validate connection with a simple command
                    try:
                        # Get session info as a test
                        _ableton_connection.send_command("get_session_info")
                        logger.info("Connection validated successfully")
                        return _ableton_connection
                    except Exception as e:
                        logger.error(f"Connection validation failed: {str(e)}")
                        _ableton_connection.disconnect()
                        _ableton_connection = None
                        # Continue to next attempt
                else:
                    _ableton_connection = None
            except Exception as e:
                logger.error(f"Connection attempt {attempt} failed: {str(e)}")
                if _ableton_connection:
                    _ableton_connection.disconnect()
                    _ableton_connection = None
            
            # Wait before trying again, but only if we have more attempts left
            if attempt < max_attempts:
                import time
                time.sleep(1.0)
        
        # If we get here, all connection attempts failed
        if _ableton_connection is None:
            logger.error("Failed to connect to Ableton after multiple attempts")
            raise Exception("Could not connect to Ableton. Make sure the Remote Script is running.")
    
    return _ableton_connection


def _safe_bool(value: Any) -> bool:
    """Convert any truthy/falsy value to a bool."""
    return bool(value)


def _probe_chain(root: Any, chain: List[str]) -> Any:
    """Probe nested dict/attribute chains defensively."""
    current = root
    for key in chain:
        if current is None:
            return None

        if isinstance(current, dict):
            if key not in current:
                return None
            current = current.get(key)
            continue

        if not hasattr(current, key):
            return None
        current = getattr(current, key)

    return current


def _clip_type_from_track(track_info: Dict[str, Any], clip_info: Dict[str, Any]) -> Dict[str, bool]:
    """Infer clip type from explicit clip fields, then track type as fallback."""
    if isinstance(clip_info, dict):
        is_audio_clip = clip_info.get("is_audio_clip")
        is_midi_clip = clip_info.get("is_midi_clip")

        if isinstance(is_audio_clip, bool) or isinstance(is_midi_clip, bool):
            return {
                "is_audio_clip": _safe_bool(is_audio_clip),
                "is_midi_clip": _safe_bool(is_midi_clip)
            }

    return {
        "is_audio_clip": _safe_bool(track_info.get("is_audio_track")),
        "is_midi_clip": _safe_bool(track_info.get("is_midi_track"))
    }


class SourceAnalysisError(Exception):
    """Structured error used by source analysis helpers."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class LiveCaptureError(Exception):
    """Structured error used by range/analysis helper utilities."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _utc_now_iso() -> str:
    """Return UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _normalize_source_path(file_path: str) -> str:
    """Expand and normalize a source file path."""
    expanded = os.path.expanduser(file_path)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(_LAUNCH_CWD, expanded))


def _ensure_cache_dir() -> None:
    """Ensure source analysis cache directory exists."""
    os.makedirs(_SOURCE_CACHE_DIR, exist_ok=True)


def _ensure_audio_analysis_cache_dir() -> None:
    """Ensure audio analysis cache directory exists."""
    os.makedirs(get_analysis_dir(), exist_ok=True)


def _sanitize_filename_token(value: str, fallback: str = "section") -> str:
    """Sanitize text for filesystem-safe filename tokens."""
    if not isinstance(value, str):
        return fallback
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    token = token.strip("._-")
    if not token:
        return fallback
    return token[:120]


def _audio_analysis_cache_key(payload: Dict[str, Any]) -> str:
    """Build stable cache key for audio analysis inputs."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _audio_analysis_cache_path(cache_key: str) -> str:
    """Return cache file path for audio analysis key."""
    return os.path.join(get_analysis_dir(), f"{cache_key}.json")


def _load_audio_analysis_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    """Load audio analysis cache payload when available."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _write_audio_analysis_cache(cache_path: str, payload: Dict[str, Any]) -> None:
    """Persist audio analysis cache payload."""
    _ensure_audio_analysis_cache_dir()
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _audio_input_format(file_path: str) -> str:
    """Return normalized input format token (without leading dot)."""
    extension = os.path.splitext(file_path)[1].strip().lower()
    if extension.startswith("."):
        extension = extension[1:]
    return extension


def _decoded_audio_tmp_dir() -> str:
    """Return directory used for decoded non-wav analysis artifacts."""
    return os.path.join(get_analysis_dir(), "tmp_decoded")


def _ensure_decoded_audio_tmp_dir() -> str:
    """Ensure decoded non-wav cache directory exists and return path."""
    decoded_dir = _decoded_audio_tmp_dir()
    os.makedirs(decoded_dir, exist_ok=True)
    return decoded_dir


def _decoded_wav_cache_path(
    source_path: str,
    stat_size: int,
    stat_mtime: float
) -> str:
    """Build deterministic decoded WAV cache path from source metadata."""
    cache_input = f"{source_path}|{int(stat_size)}|{float(stat_mtime):.6f}"
    cache_key = hashlib.sha256(cache_input.encode("utf-8")).hexdigest()
    return os.path.join(_ensure_decoded_audio_tmp_dir(), f"{cache_key}.wav")


def _decode_source_to_wav(
    source_path: str,
    input_format: str,
    stat_size: int,
    stat_mtime: float
) -> Tuple[str, Optional[str]]:
    """Return analysis WAV path, decoding source with ffmpeg when needed."""
    if input_format == "wav":
        return source_path, None

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise LiveCaptureError(
            "FFMPEG_NOT_FOUND",
            "ffmpeg required to analyze mp3/aif/flac."
        )

    decoded_wav_path = _decoded_wav_cache_path(
        source_path=source_path,
        stat_size=int(stat_size),
        stat_mtime=float(stat_mtime)
    )

    decoded_exists, decoded_size, _ = _safe_file_stats(decoded_wav_path)
    if (
        decoded_exists
        and isinstance(decoded_size, int)
        and decoded_size > 44
        and _check_wav_header(decoded_wav_path)
    ):
        return decoded_wav_path, decoded_wav_path

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        decoded_wav_path,
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120.0
        )
    except subprocess.TimeoutExpired:
        raise LiveCaptureError("decode_timeout", "ffmpeg decode timed out")
    except Exception as exc:
        raise LiveCaptureError("decode_failed", f"ffmpeg decode failed: {str(exc)}")

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        if not stderr:
            stderr = "Unknown ffmpeg decode failure"
        raise LiveCaptureError("decode_failed", stderr)

    decoded_exists, decoded_size, _ = _safe_file_stats(decoded_wav_path)
    if not decoded_exists or not isinstance(decoded_size, int) or decoded_size <= 44:
        raise LiveCaptureError("decode_failed", "Decoded WAV is empty or invalid")
    if not _check_wav_header(decoded_wav_path):
        raise LiveCaptureError("decode_failed", "Decoded WAV header validation failed")

    return decoded_wav_path, decoded_wav_path


def _source_cache_path(file_path: str) -> str:
    """Build cache file path from source path SHA256 hash."""
    key = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
    return os.path.join(_SOURCE_CACHE_DIR, f"{key}.json")


def _load_source_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    """Load cache payload from disk."""
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None

    return None


def _write_source_cache(cache_path: str, payload: Dict[str, Any]) -> None:
    """Write cache payload to disk."""
    _ensure_cache_dir()
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _safe_file_stats(file_path: str) -> Tuple[bool, Optional[int], Optional[float]]:
    """Return existence, size, and mtime for file path."""
    if not os.path.exists(file_path):
        return False, None, None

    try:
        stat_result = os.stat(file_path)
        return True, int(stat_result.st_size), float(stat_result.st_mtime)
    except Exception:
        return False, None, None


def _is_source_cache_valid(
    cache_payload: Optional[Dict[str, Any]],
    file_path: str,
    file_exists: bool,
    stat_size: Optional[int],
    stat_mtime: Optional[float]
) -> bool:
    """Validate cache payload against current file metadata."""
    if not isinstance(cache_payload, dict):
        return False
    if cache_payload.get("version") != _SOURCE_CACHE_VERSION:
        return False
    if cache_payload.get("file_path") != file_path:
        return False
    if not file_exists:
        return False
    if cache_payload.get("file_exists") is not True:
        return False
    if cache_payload.get("stat_size") != stat_size:
        return False
    if cache_payload.get("stat_mtime") != stat_mtime:
        return False
    return True


def _device_id_from_inventory_entry(name: str, path_parts: List[str]) -> str:
    """Build stable device id from inventory name/path."""
    key = f"{name}|{'/'.join(path_parts)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _normalize_inventory_devices_for_capabilities(inventory_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert device inventory payload into stable device descriptors."""
    devices_payload = inventory_payload.get("devices", [])
    if not isinstance(devices_payload, list):
        return []

    normalized: List[Dict[str, Any]] = []
    seen_device_ids = set()
    for item in devices_payload:
        if not isinstance(item, dict):
            continue

        name_value = item.get("name")
        if not isinstance(name_value, str) or not name_value.strip():
            continue
        name = name_value.strip()

        raw_path = item.get("path")
        path_parts: List[str] = []
        if isinstance(raw_path, list):
            path_parts = [part.strip() for part in raw_path if isinstance(part, str) and part.strip()]
        elif isinstance(raw_path, str) and raw_path.strip():
            path_parts = [part.strip() for part in raw_path.split("/") if part.strip()]

        if not path_parts:
            path_parts = [name]

        root = path_parts[0] if path_parts else None
        item_type = item.get("item_type")
        if not isinstance(item_type, str):
            item_type = None

        device_id = _device_id_from_inventory_entry(name, path_parts)
        if device_id in seen_device_ids:
            continue
        seen_device_ids.add(device_id)

        normalized.append({
            "device_id": device_id,
            "name": name,
            "path": path_parts,
            "root": root,
            "item_type": item_type
        })

    normalized.sort(key=lambda entry: (entry["name"].lower(), "/".join(entry["path"]).lower()))
    return normalized


def _compute_inventory_hash(devices: List[Dict[str, Any]]) -> str:
    """Compute stable hash for normalized inventory devices."""
    hashable_records = []
    for item in devices:
        if not isinstance(item, dict):
            continue
        hashable_records.append({
            "device_id": item.get("device_id"),
            "name": item.get("name"),
            "path": item.get("path"),
            "root": item.get("root"),
            "item_type": item.get("item_type")
        })
    hashable_records.sort(key=lambda entry: str(entry.get("device_id")))
    payload = json.dumps(hashable_records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_inventory_scan_params() -> Dict[str, Any]:
    """Default scan parameters for device inventory snapshots."""
    return {
        "roots": ["Audio Effects", "Plugins"],
        "max_depth": 5,
        "max_items_per_folder": 500,
        "include_presets": False
    }


def _normalize_inventory_scan_params(scan_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize and sanitize inventory scan params."""
    defaults = _default_inventory_scan_params()
    if not isinstance(scan_params, dict):
        return defaults

    roots_input = scan_params.get("roots")
    roots: List[str] = []
    if isinstance(roots_input, list):
        seen_roots = set()
        for value in roots_input:
            if not isinstance(value, str):
                continue
            root_name = value.strip()
            if not root_name or root_name in seen_roots:
                continue
            seen_roots.add(root_name)
            roots.append(root_name)
    if not roots:
        roots = list(defaults["roots"])

    try:
        max_depth = max(0, int(scan_params.get("max_depth", defaults["max_depth"])))
    except Exception:
        max_depth = defaults["max_depth"]

    try:
        max_items_per_folder = max(1, int(scan_params.get("max_items_per_folder", defaults["max_items_per_folder"])))
    except Exception:
        max_items_per_folder = defaults["max_items_per_folder"]

    include_presets = bool(scan_params.get("include_presets", defaults["include_presets"]))

    return {
        "roots": roots,
        "max_depth": max_depth,
        "max_items_per_folder": max_items_per_folder,
        "include_presets": include_presets
    }


def _scan_params_key(scan_params: Dict[str, Any]) -> str:
    """Build stable hash key for normalized scan params."""
    normalized = _normalize_inventory_scan_params(scan_params)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inventory_cache_path() -> str:
    """Return inventory snapshot cache path."""
    return _DEVICE_INVENTORY_CACHE_PATH


def _load_inventory_cache() -> Optional[Dict[str, Any]]:
    """Load normalized inventory snapshot cache payload."""
    cache_path = _inventory_cache_path()
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != _DEVICE_INVENTORY_CACHE_SCHEMA_VERSION:
        return None

    inventory_hash = payload.get("inventory_hash")
    devices_payload = payload.get("devices")
    if not isinstance(inventory_hash, str) or not isinstance(devices_payload, list):
        return None

    normalized_scan_params = _normalize_inventory_scan_params(payload.get("scan_params"))
    devices = _normalize_inventory_devices_for_capabilities({"devices": devices_payload})

    created_at = payload.get("created_at")
    if not isinstance(created_at, str):
        created_at = None

    return {
        "schema_version": _DEVICE_INVENTORY_CACHE_SCHEMA_VERSION,
        "created_at": created_at,
        "scan_params": normalized_scan_params,
        "scan_params_key": _scan_params_key(normalized_scan_params),
        "inventory_hash": inventory_hash,
        "devices": devices
    }


def _save_inventory_cache(payload: Dict[str, Any]) -> None:
    """Write inventory snapshot cache payload to disk."""
    _ensure_cache_dir()
    with open(_inventory_cache_path(), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _inventory_cache_age_sec(scan_params: Dict[str, Any]) -> Optional[float]:
    """Return age of matching inventory snapshot cache in seconds."""
    cache_payload = _load_inventory_cache()
    if not isinstance(cache_payload, dict):
        return None

    if cache_payload.get("scan_params_key") != _scan_params_key(scan_params):
        return None

    created_at = cache_payload.get("created_at")
    if not isinstance(created_at, str):
        return None

    try:
        created_dt = datetime.fromisoformat(created_at)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age = (now - created_dt.astimezone(timezone.utc)).total_seconds()
        return max(0.0, float(age))
    except Exception:
        return None


def _get_or_build_inventory(
    scan_params: Dict[str, Any],
    force_refresh: bool
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], bool, List[str]]:
    """
    Return normalized inventory devices and hash from cache or fresh scan.

    Returns (devices, inventory_hash, cache_hit, warnings).
    """
    normalized_scan_params = _normalize_inventory_scan_params(scan_params)
    params_key = _scan_params_key(normalized_scan_params)
    warnings: List[str] = []

    cached_payload = _load_inventory_cache()
    if not force_refresh and isinstance(cached_payload, dict):
        if cached_payload.get("scan_params_key") == params_key:
            cached_devices = cached_payload.get("devices")
            cached_inventory_hash = cached_payload.get("inventory_hash")
            if isinstance(cached_devices, list) and isinstance(cached_inventory_hash, str):
                return cached_devices, cached_inventory_hash, True, warnings
            warnings.append("inventory_cache_invalid_payload")
        else:
            warnings.append("inventory_cache_scan_params_mismatch")

    try:
        inventory_payload = get_device_inventory(
            None,
            roots=normalized_scan_params["roots"],
            max_depth=normalized_scan_params["max_depth"],
            max_items_per_folder=normalized_scan_params["max_items_per_folder"],
            include_presets=normalized_scan_params["include_presets"]
        )
    except Exception as exc:
        warnings.append(f"inventory_scan_failed:{str(exc)}")
        return None, None, False, warnings

    if not isinstance(inventory_payload, dict):
        warnings.append("inventory_scan_failed:invalid_inventory_response")
        return None, None, False, warnings
    if inventory_payload.get("ok") is not True:
        inventory_error = inventory_payload.get("error")
        if isinstance(inventory_error, str):
            warnings.append(f"inventory_scan_failed:{inventory_error}")
        else:
            warnings.append("inventory_scan_failed:device_inventory_failed")
        return None, None, False, warnings

    devices = _normalize_inventory_devices_for_capabilities(inventory_payload)
    inventory_hash = _compute_inventory_hash(devices)
    cache_payload = {
        "schema_version": _DEVICE_INVENTORY_CACHE_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "scan_params": normalized_scan_params,
        "scan_params_key": params_key,
        "inventory_hash": inventory_hash,
        "devices": devices
    }
    try:
        _save_inventory_cache(cache_payload)
    except Exception as exc:
        warnings.append(f"inventory_cache_write_failed:{str(exc)}")

    return devices, inventory_hash, False, warnings


def _ensure_project_snapshot_dir() -> None:
    """Ensure project snapshot cache directory exists."""
    os.makedirs(_PROJECT_SNAPSHOT_DIR, exist_ok=True)


def _project_device_key(track_index: int, device_index: int) -> str:
    """Build stable per-track device key."""
    return f"{int(track_index)}:{int(device_index)}"


def _stable_snapshot_device_key(
    track_index: int,
    class_name: Any,
    device_name: Any,
    occurrence_index: int
) -> str:
    """Build stable per-track snapshot device key resilient to index shifts."""
    class_name_value = class_name if isinstance(class_name, str) and class_name else "unknown_class"
    device_name_value = device_name if isinstance(device_name, str) and device_name else "unknown_device"
    return f"{int(track_index)}|{class_name_value}|{device_name_value}|{int(occurrence_index)}"


def _resolve_snapshot_device_key(
    device_payload: Dict[str, Any],
    track_index: int,
    occurrence_index: int
) -> str:
    """Resolve stable device key from snapshot payload, with legacy fallback."""
    existing = device_payload.get("device_key")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    return _stable_snapshot_device_key(
        track_index=track_index,
        class_name=device_payload.get("class_name"),
        device_name=device_payload.get("name"),
        occurrence_index=occurrence_index
    )


def _safe_json_file_load(file_path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON object file safely."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _safe_json_file_write(file_path: str, payload: Dict[str, Any]) -> None:
    """Write JSON object file safely."""
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _project_snapshot_timestamp_token() -> str:
    """Return UTC timestamp token for snapshot ids."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_project_snapshot_file(snapshot_id: str) -> Optional[str]:
    """Resolve snapshot id or path to snapshot json file."""
    if not isinstance(snapshot_id, str):
        return None

    candidate = snapshot_id.strip()
    if not candidate:
        return None

    if os.path.isabs(candidate):
        if os.path.exists(candidate) and candidate.endswith(".json"):
            return candidate
        return None

    if candidate.endswith(".json"):
        direct = os.path.join(_PROJECT_SNAPSHOT_DIR, candidate)
    else:
        direct = os.path.join(_PROJECT_SNAPSHOT_DIR, f"{candidate}.json")

    if os.path.exists(direct):
        return direct

    prefix = candidate if not candidate.endswith(".json") else candidate[:-5]
    try:
        matches = []
        for name in os.listdir(_PROJECT_SNAPSHOT_DIR):
            if not name.endswith(".json"):
                continue
            if name.endswith(".params.json"):
                continue
            if name.startswith(prefix):
                matches.append(os.path.join(_PROJECT_SNAPSHOT_DIR, name))
        if len(matches) == 1:
            return matches[0]
    except Exception:
        return None

    return None


def _build_snapshot_project_hash(session_payload: Dict[str, Any], tracks_payload: List[Dict[str, Any]]) -> str:
    """Compute stable project hash from compact snapshot payload."""
    hash_payload = {
        "session": {
            "tempo": session_payload.get("tempo"),
            "signature_numerator": session_payload.get("signature_numerator"),
            "signature_denominator": session_payload.get("signature_denominator"),
            "track_count": session_payload.get("track_count")
        },
        "tracks": []
    }

    for track in tracks_payload:
        if not isinstance(track, dict):
            continue
        hash_payload["tracks"].append({
            "track_index": track.get("track_index"),
            "track_name": track.get("track_name"),
            "mixer": track.get("mixer"),
            "devices": track.get("devices")
        })

    hash_payload["tracks"].sort(key=lambda item: int(item.get("track_index", -1)))
    serialized = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _resolve_snapshot_sidecar_file(snapshot_payload: Dict[str, Any], snapshot_file_path: str) -> Optional[str]:
    """Resolve a parameter sidecar path for a saved project snapshot."""
    sidecar_hint = snapshot_payload.get("parameter_sidecar_file")
    if isinstance(sidecar_hint, str) and sidecar_hint.strip():
        sidecar_hint = sidecar_hint.strip()
        if os.path.isabs(sidecar_hint):
            if os.path.exists(sidecar_hint):
                return sidecar_hint
        else:
            candidate = os.path.join(os.path.dirname(snapshot_file_path), sidecar_hint)
            if os.path.exists(candidate):
                return candidate

    if snapshot_file_path.endswith(".json"):
        fallback = snapshot_file_path[:-5] + ".params.json"
    else:
        fallback = snapshot_file_path + ".params.json"
    if os.path.exists(fallback):
        return fallback

    return None


def _snapshot_parameter_index_map(parameter_rows: Any) -> Dict[int, Dict[str, Any]]:
    """Build parameter index map from compact parameter rows."""
    mapped: Dict[int, Dict[str, Any]] = {}
    if not isinstance(parameter_rows, list):
        return mapped
    for row in parameter_rows:
        if not isinstance(row, dict):
            continue
        parameter_index = row.get("parameter_index")
        try:
            parameter_index = int(parameter_index)
        except Exception:
            continue
        mapped[parameter_index] = {
            "parameter_index": parameter_index,
            "name": row.get("name"),
            "value": row.get("value")
        }
    return mapped


def _load_device_capabilities_cache() -> Dict[str, Any]:
    """Load device capability cache from disk."""
    default_payload = {
        "schema_version": _DEVICE_CAPABILITIES_SCHEMA_VERSION,
        "inventory_hash": None,
        "updated_at": None,
        "classifications": {}
    }

    if not os.path.exists(_DEVICE_CAPABILITIES_CACHE_PATH):
        return default_payload

    try:
        with open(_DEVICE_CAPABILITIES_CACHE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return default_payload

    if not isinstance(payload, dict):
        return default_payload

    classifications_raw = payload.get("classifications")
    classifications_map: Dict[str, Dict[str, Any]] = {}

    if isinstance(classifications_raw, dict):
        for device_id, entry in classifications_raw.items():
            if not isinstance(device_id, str) or not device_id:
                continue
            if not isinstance(entry, dict):
                continue
            classifications_map[device_id] = dict(entry)
    elif isinstance(classifications_raw, list):
        for entry in classifications_raw:
            if not isinstance(entry, dict):
                continue
            device_id = entry.get("device_id")
            if not isinstance(device_id, str) or not device_id:
                continue
            classifications_map[device_id] = dict(entry)

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int):
        schema_version = _DEVICE_CAPABILITIES_SCHEMA_VERSION

    inventory_hash = payload.get("inventory_hash")
    if not isinstance(inventory_hash, str):
        inventory_hash = None

    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str):
        updated_at = None

    return {
        "schema_version": schema_version,
        "inventory_hash": inventory_hash,
        "updated_at": updated_at,
        "classifications": classifications_map
    }


def _write_device_capabilities_cache(payload: Dict[str, Any]) -> None:
    """Persist device capability cache to disk."""
    _ensure_cache_dir()
    with open(_DEVICE_CAPABILITIES_CACHE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _get_current_inventory_devices_and_hash(ctx: Context) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Optional[str]]:
    """
    Return current inventory devices with stable hash.

    Returns (devices, inventory_hash, error_message).
    """
    _ = ctx
    devices, inventory_hash, _, warnings = _get_or_build_inventory(
        scan_params=_default_inventory_scan_params(),
        force_refresh=False
    )
    if devices is None or inventory_hash is None:
        if warnings:
            last_warning = warnings[-1]
            if isinstance(last_warning, str) and ":" in last_warning:
                return None, None, last_warning.split(":", 1)[1]
            return None, None, last_warning
        return None, None, "inventory_unavailable"
    return devices, inventory_hash, None


def _decode_audio_samples(file_path: str) -> Tuple[Any, int, int, str]:
    """Decode audio and return NxC samples, sample rate, channels, and backend."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension not in _SUPPORTED_AUDIO_EXTENSIONS:
        raise SourceAnalysisError(
            "decode_failed",
            f"Unsupported audio format '{extension}'. Supported: {sorted(_SUPPORTED_AUDIO_EXTENSIONS)}"
        )
    if np is None:
        raise SourceAnalysisError("unsupported_decode_backend", "numpy is not available")

    decode_errors: List[str] = []

    if sf is not None:
        try:
            decoded, sample_rate = sf.read(file_path, always_2d=True, dtype="float32")
            if decoded.size == 0:
                raise SourceAnalysisError("decode_failed", "Decoded audio is empty")
            channels = int(decoded.shape[1])
            return decoded.astype(np.float32), int(sample_rate), channels, "soundfile"
        except SourceAnalysisError:
            raise
        except Exception as exc:
            decode_errors.append(f"soundfile: {str(exc)}")

    if AudioSegment is not None:
        try:
            segment = AudioSegment.from_file(file_path)
            if len(segment) == 0:
                raise SourceAnalysisError("decode_failed", "Decoded audio is empty")

            channels = int(segment.channels)
            sample_rate = int(segment.frame_rate)
            sample_width = int(segment.sample_width)
            pcm = np.array(segment.get_array_of_samples(), dtype=np.float32)

            if channels > 1:
                pcm = pcm.reshape((-1, channels))
            else:
                pcm = pcm.reshape((-1, 1))

            full_scale = float(2 ** max(1, (8 * sample_width - 1)))
            normalized = (pcm / full_scale).astype(np.float32)
            return normalized, sample_rate, channels, "pydub"
        except SourceAnalysisError:
            raise
        except Exception as exc:
            decode_errors.append(f"pydub: {str(exc)}")

    if extension in {".mp3", ".m4a"} and sf is None and AudioSegment is None:
        raise SourceAnalysisError(
            "unsupported_decode_backend",
            "No decode backend for mp3/m4a (soundfile/pydub unavailable)"
        )

    message = "Failed to decode audio"
    if decode_errors:
        message += f" ({'; '.join(decode_errors)})"
    raise SourceAnalysisError("decode_failed", message)


def _band_energy_db(spectrum: Any, frequencies: Any, low_hz: float, high_hz: float) -> float:
    """Compute mean band magnitude in dB."""
    if np is None:
        return 0.0

    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not np.any(mask):
        return -120.0

    band_mag = float(np.mean(spectrum[mask]))
    return float(20.0 * math.log10(max(band_mag, 1e-12)))


def _safe_db(value: float, floor_db: float = -120.0) -> float:
    """Convert linear amplitude to dB with flooring."""
    if value <= 0.0:
        return floor_db
    db_value = 20.0 * math.log10(value)
    return max(floor_db, float(db_value))


def _average_spectrum_welch(signal_mono: Any, sample_rate: int) -> Tuple[Any, Any]:
    """Build a Welch-style averaged magnitude spectrum."""
    if np is None:
        raise SourceAnalysisError("unsupported_decode_backend", "numpy is not available")

    frame_size = 4096
    hop = frame_size // 2
    if signal_mono.shape[0] < frame_size:
        padded = np.pad(signal_mono, (0, frame_size - signal_mono.shape[0]))
        signal_mono = padded.astype(np.float32)

    window = np.hanning(frame_size).astype(np.float32)
    spectra = []
    for start in range(0, signal_mono.shape[0] - frame_size + 1, hop):
        frame = signal_mono[start:start + frame_size]
        if frame.shape[0] < frame_size:
            continue
        frame_spectrum = np.abs(np.fft.rfft(frame * window))
        spectra.append(frame_spectrum)

    if not spectra:
        frame_spectrum = np.abs(np.fft.rfft(signal_mono[:frame_size] * window))
        spectra = [frame_spectrum]

    stacked = np.stack(spectra, axis=0)
    avg_spectrum = np.mean(stacked, axis=0)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / float(sample_rate))
    return frequencies, avg_spectrum


def _find_resonant_peaks(frequencies: Any, spectrum_db: Any) -> List[Dict[str, float]]:
    """Find robust spectral peaks from residual over smoothed baseline."""
    if np is None or len(spectrum_db) < 8:
        return []

    kernel_size = max(31, int(len(spectrum_db) / 96))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    baseline = np.convolve(spectrum_db, kernel, mode="same")
    prominence = spectrum_db - baseline

    valid_mask = (frequencies >= 80.0) & (frequencies <= 12000.0)
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size < 3:
        return []

    candidate_indices: List[int] = []
    if sp_signal is not None:
        valid_prom = prominence[valid_indices]
        peak_local_indices, props = sp_signal.find_peaks(valid_prom, prominence=6.0, distance=3)
        candidate_indices = [int(valid_indices[idx]) for idx in peak_local_indices]
    else:
        for idx in valid_indices[1:-1]:
            if prominence[idx] < 6.0:
                continue
            if spectrum_db[idx] <= spectrum_db[idx - 1] or spectrum_db[idx] <= spectrum_db[idx + 1]:
                continue
            candidate_indices.append(int(idx))

    def _passes_spacing(existing_hz: List[float], hz: float) -> bool:
        min_ratio = 2.0 ** (1.0 / 12.0)
        for val in existing_hz:
            ratio = max(hz, val) / max(1e-9, min(hz, val))
            if ratio < min_ratio:
                return False
        return True

    candidates = sorted(
        [(idx, float(prominence[idx])) for idx in candidate_indices],
        key=lambda item: item[1],
        reverse=True
    )

    peaks: List[Dict[str, float]] = []
    selected_hz: List[float] = []
    for idx, prom in candidates:
        if prom < 6.0:
            continue
        hz = float(frequencies[idx])
        if not _passes_spacing(selected_hz, hz):
            continue
        peaks.append({
            "hz": round(hz, 2),
            "prominence_db": round(prom, 2)
        })
        selected_hz.append(hz)
        if len(peaks) >= 5:
            break

    return peaks


def _format_frequency(hz: float) -> str:
    """Format frequency in Hz or kHz for summaries."""
    if hz >= 1000.0:
        return f"{hz / 1000.0:.1f} kHz"
    return f"{hz:.0f} Hz"


def _build_source_summary(
    band_energies_db: Dict[str, float],
    peaks: List[Dict[str, float]],
    lufs_integrated: Optional[float],
    true_peak_dbtp: Optional[float]
) -> str:
    """Build short deterministic source summary."""
    descriptors: List[str] = []
    sub = band_energies_db.get("sub_20_60", -120.0)
    low = band_energies_db.get("low_60_200", -120.0)
    lowmid = band_energies_db.get("lowmid_200_500", -120.0)
    mid = band_energies_db.get("mid_500_2000", -120.0)
    highmid = band_energies_db.get("highmid_2000_6000", -120.0)
    air = band_energies_db.get("air_6000_12000", -120.0)

    if highmid > mid + 2.5 or air > mid + 2.5:
        descriptors.append("Bright high-mids")
    elif highmid < mid - 2.5 and air < mid - 2.5:
        descriptors.append("Soft top-end")

    if sub > lowmid + 2.5:
        descriptors.append("strong sub")
    elif sub < lowmid - 2.5:
        descriptors.append("modest sub")

    if lowmid > mid + 2.0:
        descriptors.append("forward low-mids")
    elif mid > lowmid + 2.0:
        descriptors.append("forward mids")

    if not descriptors:
        descriptors.append("Balanced spectrum")

    if lufs_integrated is not None:
        descriptors.append(f"LUFS {lufs_integrated:.1f}")

    if true_peak_dbtp is not None:
        descriptors.append(f"true peak {true_peak_dbtp:+.1f} dBTP")

    if peaks:
        top_peak = peaks[0]
        descriptors.append(f"resonance ~{_format_frequency(float(top_peak['hz']))}")

    summary = ", ".join(descriptors)
    if not summary.endswith("."):
        summary += "."
    return summary


def _analyze_audio_source(file_path: str, stat_size: int, stat_mtime: float) -> Dict[str, Any]:
    """Run a deterministic spectral analysis over decoded audio."""
    if np is None:
        raise SourceAnalysisError("unsupported_decode_backend", "numpy is not available")

    samples, sample_rate, channels, decode_backend = _decode_audio_samples(file_path)
    if samples.size == 0:
        raise SourceAnalysisError("decode_failed", "Decoded audio is empty")

    analysis_notes: List[str] = [f"decode_backend={decode_backend}"]
    mono = samples.mean(axis=1).astype(np.float32)
    sample_peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    crest_factor_db = float(_safe_db(max(sample_peak, 1e-12) / max(rms, 1e-12)))
    sample_peak_dbfs = _safe_db(sample_peak)

    true_peak = sample_peak
    true_peak_dbtp = sample_peak_dbfs
    if sp_signal is not None:
        try:
            channel_true_peaks = []
            for channel_index in range(samples.shape[1]):
                oversampled = sp_signal.resample_poly(samples[:, channel_index], up=4, down=1)
                channel_true_peaks.append(float(np.max(np.abs(oversampled))))
            if channel_true_peaks:
                true_peak = max(channel_true_peaks)
                true_peak_dbtp = _safe_db(true_peak)
        except Exception as exc:
            analysis_notes.append(f"true_peak_fallback_sample_peak ({str(exc)})")
    else:
        analysis_notes.append("true_peak_fallback_sample_peak (scipy unavailable)")

    lufs_integrated: Optional[float] = None
    if pyln is not None:
        try:
            meter = pyln.Meter(sample_rate)
            if channels == 1:
                lufs_integrated = float(meter.integrated_loudness(mono.astype(np.float64)))
            else:
                lufs_integrated = float(meter.integrated_loudness(samples.astype(np.float64)))
        except Exception as exc:
            analysis_notes.append(f"lufs_unavailable ({str(exc)})")
    else:
        analysis_notes.append("lufs_unavailable (pyloudnorm unavailable)")

    # Analyze at most 120 seconds for predictable runtime on large files.
    max_samples = int(sample_rate * 120)
    analysis_signal = mono[:max_samples] if mono.shape[0] > max_samples else mono
    if analysis_signal.shape[0] < 32:
        raise SourceAnalysisError("decode_failed", "Audio too short for spectral analysis")

    frequencies, spectrum = _average_spectrum_welch(analysis_signal, sample_rate)
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))

    band_energies_db = {
        "sub_20_60": round(_band_energy_db(spectrum, frequencies, 20.0, 60.0), 3),
        "low_60_200": round(_band_energy_db(spectrum, frequencies, 60.0, 200.0), 3),
        "lowmid_200_500": round(_band_energy_db(spectrum, frequencies, 200.0, 500.0), 3),
        "mid_500_2000": round(_band_energy_db(spectrum, frequencies, 500.0, 2000.0), 3),
        "highmid_2000_6000": round(_band_energy_db(spectrum, frequencies, 2000.0, 6000.0), 3),
        "air_6000_12000": round(_band_energy_db(spectrum, frequencies, 6000.0, 12000.0), 3),
    }
    resonant_peaks_hz = _find_resonant_peaks(frequencies, spectrum_db)
    summary = _build_source_summary(
        band_energies_db=band_energies_db,
        peaks=resonant_peaks_hz,
        lufs_integrated=lufs_integrated,
        true_peak_dbtp=true_peak_dbtp
    )

    return {
        "ok": True,
        "file_path": file_path,
        "file_exists": True,
        "stat_size": stat_size,
        "stat_mtime": stat_mtime,
        "analyzed_at": _utc_now_iso(),
        "version": _SOURCE_CACHE_VERSION,
        "duration_sec": round(float(mono.shape[0]) / float(sample_rate), 4),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "peak": round(sample_peak, 6),
        "sample_peak": round(sample_peak, 6),
        "sample_peak_dbfs": round(sample_peak_dbfs, 3),
        "true_peak": round(float(true_peak), 6),
        "true_peak_dbtp": round(float(true_peak_dbtp), 3),
        "lufs_integrated": None if lufs_integrated is None else round(float(lufs_integrated), 3),
        "rms": round(rms, 6),
        "crest_factor_db": round(crest_factor_db, 3),
        "band_energies_db": band_energies_db,
        "resonant_peaks_hz": resonant_peaks_hz,
        "summary": summary,
        "analysis_notes": analysis_notes
    }


def _beats_per_bar(signature_numerator: int, signature_denominator: int) -> float:
    """Convert time signature into beats per bar."""
    if signature_numerator <= 0 or signature_denominator <= 0:
        raise LiveCaptureError("invalid_time_signature", "Session time signature is invalid")
    return float(signature_numerator) * (4.0 / float(signature_denominator))


def _bars_to_seconds(
    bar_number: int,
    tempo: float,
    signature_numerator: int,
    signature_denominator: int
) -> float:
    """Convert 1-based bar position to absolute seconds from project start."""
    if bar_number < 1:
        raise LiveCaptureError("invalid_bar", "Bar numbers must be >= 1")
    if tempo <= 0:
        raise LiveCaptureError("invalid_tempo", "Session tempo must be > 0")

    beats_per_bar = _beats_per_bar(signature_numerator, signature_denominator)
    beats_from_start = (float(bar_number) - 1.0) * beats_per_bar
    return max(0.0, beats_from_start * (60.0 / tempo))


def _resolve_capture_range_seconds(
    session_payload: Dict[str, Any],
    start_bar: Optional[int],
    end_bar: Optional[int],
    start_time_sec: Optional[float],
    duration_sec: Optional[float],
    max_duration_sec: Optional[float] = None
) -> Dict[str, Any]:
    """Resolve mixed bar/time inputs into a validated [start, end, duration] section."""
    try:
        tempo = float(session_payload.get("tempo"))
    except Exception:
        raise LiveCaptureError("invalid_session_info", "tempo missing or invalid in session info")
    if tempo <= 0.0:
        raise LiveCaptureError("invalid_session_info", "tempo must be > 0")

    try:
        signature_numerator = int(session_payload.get("signature_numerator"))
        signature_denominator = int(session_payload.get("signature_denominator"))
    except Exception:
        raise LiveCaptureError("invalid_session_info", "time signature missing or invalid in session info")

    beats_per_bar = _beats_per_bar(signature_numerator, signature_denominator)

    if start_time_sec is not None:
        try:
            start_sec = float(start_time_sec)
        except Exception:
            raise LiveCaptureError("invalid_start_time_sec", "start_time_sec must be numeric")
        if start_sec < 0.0:
            raise LiveCaptureError("invalid_start_time_sec", "start_time_sec must be >= 0")
        resolved_start_bar = None
    else:
        if start_bar is None:
            start_bar = 1
        try:
            start_bar_value = int(start_bar)
        except Exception:
            raise LiveCaptureError("invalid_start_bar", "start_bar must be an integer")
        start_sec = _bars_to_seconds(
            bar_number=start_bar_value,
            tempo=tempo,
            signature_numerator=signature_numerator,
            signature_denominator=signature_denominator
        )
        resolved_start_bar = start_bar_value

    end_sec_from_bar: Optional[float] = None
    resolved_end_bar: Optional[int] = None
    if end_bar is not None:
        try:
            end_bar_value = int(end_bar)
        except Exception:
            raise LiveCaptureError("invalid_end_bar", "end_bar must be an integer")
        end_sec_from_bar = _bars_to_seconds(
            bar_number=end_bar_value,
            tempo=tempo,
            signature_numerator=signature_numerator,
            signature_denominator=signature_denominator
        )
        resolved_end_bar = end_bar_value

    if duration_sec is not None:
        try:
            resolved_duration_sec = float(duration_sec)
        except Exception:
            raise LiveCaptureError("invalid_duration_sec", "duration_sec must be numeric")
        if resolved_duration_sec <= 0.0:
            raise LiveCaptureError("invalid_duration_sec", "duration_sec must be > 0")
        end_sec = start_sec + resolved_duration_sec
    elif end_sec_from_bar is not None:
        resolved_duration_sec = end_sec_from_bar - start_sec
        if resolved_duration_sec <= 0.0:
            raise LiveCaptureError(
                "invalid_range",
                "Resolved end time must be greater than start time"
            )
        end_sec = end_sec_from_bar
    else:
        raise LiveCaptureError(
            "missing_range",
            "Provide duration_sec, or provide end_bar, to define capture length"
        )

    if max_duration_sec is not None and resolved_duration_sec > float(max_duration_sec):
        raise LiveCaptureError(
            "duration_too_long",
            f"Requested duration {resolved_duration_sec:.3f}s exceeds max {float(max_duration_sec):.1f}s"
        )

    return {
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "duration_sec": float(resolved_duration_sec),
        "tempo": float(tempo),
        "signature_numerator": int(signature_numerator),
        "signature_denominator": int(signature_denominator),
        "beats_per_bar": float(beats_per_bar),
        "start_bar": resolved_start_bar,
        "end_bar": resolved_end_bar
    }


def _extract_track_kind(track_payload: Dict[str, Any]) -> str:
    """Resolve track kind field with backwards-compatible fallbacks."""
    if not isinstance(track_payload, dict):
        return "unknown"
    kind_value = track_payload.get("track_kind")
    if isinstance(kind_value, str) and kind_value.strip():
        return kind_value.strip().lower()

    if bool(track_payload.get("is_group_track")):
        return "group"
    if bool(track_payload.get("is_audio_track")) and bool(track_payload.get("is_midi_track")):
        return "hybrid"
    if bool(track_payload.get("is_audio_track")):
        return "audio"
    if bool(track_payload.get("is_midi_track")):
        return "midi"
    return "unknown"


def _normalize_arrangement_target(target: str, track_index: Optional[int]) -> Dict[str, Any]:
    """Validate arrangement target selector."""
    target_value = target.strip().lower() if isinstance(target, str) else ""
    if target_value not in {"track", "group", "mix"}:
        raise LiveCaptureError("invalid_target", "target must be 'track', 'group', or 'mix'")

    if target_value in {"track", "group"}:
        if track_index is None:
            raise LiveCaptureError("missing_track_index", "track_index is required for target='track'/'group'")
        try:
            normalized_index = int(track_index)
        except Exception:
            raise LiveCaptureError("invalid_track_index", "track_index must be an integer")
        if normalized_index < 0:
            raise LiveCaptureError("invalid_track_index", "track_index must be >= 0")
        return {"target": target_value, "track_index": normalized_index}

    return {"target": "mix", "track_index": None}


def _get_tracks_mixer_state_rows(ableton: AbletonConnection) -> List[Dict[str, Any]]:
    """Return normalized track mixer/state rows including grouping metadata."""
    try:
        payload = ableton.send_command("get_tracks_mixer_state")
        raw_states = payload.get("states", []) if isinstance(payload, dict) else []
    except Exception:
        raw_states = []

    rows: List[Dict[str, Any]] = []
    if isinstance(raw_states, list):
        for row in raw_states:
            if not isinstance(row, dict):
                continue
            try:
                track_index = int(row.get("track_index"))
            except Exception:
                continue
            group_track_index_raw = row.get("group_track_index")
            group_track_index = None
            if group_track_index_raw is not None:
                try:
                    group_track_index = int(group_track_index_raw)
                except Exception:
                    group_track_index = None

            rows.append({
                "track_index": track_index,
                "track_name": row.get("track_name"),
                "track_kind": _extract_track_kind(row),
                "is_group_track": bool(row.get("is_group_track")),
                "group_track_index": group_track_index
            })
    if rows:
        rows.sort(key=lambda item: item["track_index"])
        return rows

    # Fallback path for older backends.
    session = ableton.send_command("get_session_info")
    track_count = int(session.get("track_count", 0)) if isinstance(session, dict) else 0
    for track_index in range(max(0, track_count)):
        try:
            track_info = ableton.send_command("get_track_info", {"track_index": track_index})
        except Exception:
            continue
        if not isinstance(track_info, dict):
            continue

        group_track_index_raw = track_info.get("group_track_index")
        group_track_index = None
        if group_track_index_raw is not None:
            try:
                group_track_index = int(group_track_index_raw)
            except Exception:
                group_track_index = None

        rows.append({
            "track_index": track_index,
            "track_name": track_info.get("name"),
            "track_kind": _extract_track_kind(track_info),
            "is_group_track": bool(track_info.get("is_group_track")),
            "group_track_index": group_track_index
        })
    rows.sort(key=lambda item: item["track_index"])
    return rows


def _collect_group_tree_indices(
    track_rows: List[Dict[str, Any]],
    group_track_index: int,
    include_children: bool
) -> List[int]:
    """Return group track index + optional recursive child indices."""
    known_indices = {int(row["track_index"]) for row in track_rows if isinstance(row, dict) and "track_index" in row}
    if group_track_index not in known_indices:
        raise LiveCaptureError("invalid_track_index", f"track_index {group_track_index} out of range")

    if not include_children:
        return [group_track_index]

    remaining = [group_track_index]
    seen = {group_track_index}
    result = [group_track_index]
    while remaining:
        current = remaining.pop(0)
        for row in track_rows:
            if not isinstance(row, dict):
                continue
            parent = row.get("group_track_index")
            if parent != current:
                continue
            child_index = row.get("track_index")
            if not isinstance(child_index, int):
                continue
            if child_index in seen:
                continue
            seen.add(child_index)
            result.append(child_index)
            if row.get("track_kind") == "group":
                remaining.append(child_index)
    result.sort()
    return result


def _clip_interval_from_metadata(clip_payload: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Extract [start_beat, end_beat] interval from arrangement clip metadata."""
    if not isinstance(clip_payload, dict):
        return None

    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except Exception:
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    start_beat = None
    for key in ("start_time_beats", "start_time", "clip_start_beats"):
        start_beat = _as_float(clip_payload.get(key))
        if start_beat is not None:
            break

    end_beat = None
    for key in ("end_time_beats", "end_time", "clip_end_beats"):
        end_beat = _as_float(clip_payload.get(key))
        if end_beat is not None:
            break

    length_beat = None
    for key in ("length_beats", "length", "duration_beats"):
        length_beat = _as_float(clip_payload.get(key))
        if length_beat is not None:
            break

    if start_beat is None:
        return None
    if end_beat is None and length_beat is not None:
        end_beat = float(start_beat) + float(length_beat)
    if end_beat is None:
        return None
    if end_beat <= start_beat:
        return None
    return float(start_beat), float(end_beat)


def _collect_arrangement_intervals_for_tracks(
    ctx: Context,
    track_indices: List[int]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Collect arrangement clip intervals for the requested tracks."""
    intervals: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for track_index in track_indices:
        payload = list_arrangement_clips(ctx, track_index)
        if not isinstance(payload, dict):
            warnings.append(f"track_{track_index}:invalid_arrangement_response")
            continue
        if not payload.get("supported"):
            warnings.append(f"track_{track_index}:arrangement_not_supported")
            continue
        clips = payload.get("clips", [])
        if not isinstance(clips, list):
            warnings.append(f"track_{track_index}:invalid_clip_list")
            continue
        for clip in clips:
            interval = _clip_interval_from_metadata(clip if isinstance(clip, dict) else {})
            if interval is None:
                continue
            start_beat, end_beat = interval
            intervals.append({
                "track_index": int(track_index),
                "clip_index": clip.get("clip_index") if isinstance(clip, dict) else None,
                "start_beat": start_beat,
                "end_beat": end_beat
            })
    return intervals, warnings


def _build_activity_timeline(
    intervals: List[Dict[str, Any]],
    beats_per_bar: float,
    start_bar: int,
    end_bar: int,
    bar_resolution: int
) -> Dict[str, Any]:
    """Build coarse per-bar activity flags from beat intervals."""
    if start_bar < 1:
        raise LiveCaptureError("invalid_start_bar", "start_bar must be >= 1")
    if end_bar < start_bar:
        raise LiveCaptureError("invalid_range", "end_bar must be >= start_bar")
    if bar_resolution <= 0:
        raise LiveCaptureError("invalid_bar_resolution", "bar_resolution must be > 0")

    bars: List[Dict[str, Any]] = []
    first_active_bar: Optional[int] = None
    active_bar_count = 0

    normalized_intervals: List[Tuple[float, float]] = []
    for row in intervals:
        if not isinstance(row, dict):
            continue
        try:
            start_beat = float(row.get("start_beat"))
            end_beat = float(row.get("end_beat"))
        except Exception:
            continue
        if end_beat <= start_beat:
            continue
        normalized_intervals.append((start_beat, end_beat))

    for bar in range(start_bar, end_bar + 1, bar_resolution):
        segment_start_beat = (float(bar) - 1.0) * beats_per_bar
        segment_end_bar = min(end_bar + 1, bar + bar_resolution)
        segment_end_beat = (float(segment_end_bar) - 1.0) * beats_per_bar

        has_clip = False
        for clip_start_beat, clip_end_beat in normalized_intervals:
            if clip_start_beat < segment_end_beat and clip_end_beat > segment_start_beat:
                has_clip = True
                break

        if has_clip and first_active_bar is None:
            first_active_bar = int(bar)
        if has_clip:
            active_bar_count += 1

        bars.append({
            "bar": int(bar),
            "has_clip": bool(has_clip)
        })

    return {
        "bars": bars,
        "first_active_bar": first_active_bar,
        "active_bar_count": active_bar_count,
        "detected_activity": first_active_bar is not None
    }


def _suggest_export_filename(
    target: str,
    track_index: Optional[int],
    suggest_filename: Optional[str],
    start_sec: float,
    end_sec: float
) -> str:
    """Build stable recommended filename for manual Ableton exports."""
    if isinstance(suggest_filename, str) and suggest_filename.strip():
        base = _sanitize_filename_token(suggest_filename, fallback="export_section")
    else:
        target_token = target if target != "track" else f"track_{track_index}"
        start_token = f"{int(round(start_sec * 1000.0))}ms"
        end_token = f"{int(round(end_sec * 1000.0))}ms"
        base = _sanitize_filename_token(f"{target_token}_{start_token}_{end_token}", fallback="export_section")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{base}.wav"


def _format_clock(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm string."""
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    hours, rem = divmod(total_ms, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _seconds_to_bar_number(seconds: float, tempo: float, beats_per_bar: float) -> int:
    """Convert absolute seconds to 1-based bar number."""
    if tempo <= 0.0 or beats_per_bar <= 0.0:
        return 1
    beats = max(0.0, float(seconds)) * (float(tempo) / 60.0)
    return int(math.floor(beats / beats_per_bar)) + 1


def _default_wav_settings() -> Dict[str, Any]:
    """Return default export WAV settings."""
    return {
        "bit_depth": 24,
        "sample_rate": "project",
        "normalize": False,
        "dither": False
    }


def _normalize_wav_settings(wav_settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize and validate plan-export WAV settings."""
    defaults = _default_wav_settings()
    if not isinstance(wav_settings, dict):
        return defaults

    normalized = dict(defaults)
    bit_depth_value = wav_settings.get("bit_depth", defaults["bit_depth"])
    if bit_depth_value in {24, 32}:
        normalized["bit_depth"] = bit_depth_value

    sample_rate_value = wav_settings.get("sample_rate", defaults["sample_rate"])
    if sample_rate_value in {44100, 48000, 96000, "project"}:
        normalized["sample_rate"] = sample_rate_value

    normalized["normalize"] = bool(wav_settings.get("normalize", defaults["normalize"]))
    normalized["dither"] = bool(wav_settings.get("dither", defaults["dither"]))
    return normalized


def _export_manifest_filename(job_name: str) -> str:
    """Return stable manifest filename for export jobs."""
    job_token = _sanitize_filename_token(job_name, fallback="export_job")
    return f"{job_token}__export_manifest.json"


def _resolve_export_manifest_path(
    manifest_path: Optional[str],
    job_name: Optional[str]
) -> str:
    """Resolve export manifest path from explicit path or job name."""
    if isinstance(manifest_path, str) and manifest_path.strip():
        return _normalize_source_path(manifest_path.strip())
    if isinstance(job_name, str) and job_name.strip():
        return os.path.join(get_analysis_dir(), _export_manifest_filename(job_name.strip()))
    raise LiveCaptureError("missing_manifest_reference", "Provide manifest_path or job_name")


def _load_export_manifest(path: str) -> Dict[str, Any]:
    """Load and validate export manifest JSON."""
    normalized = _normalize_source_path(path)
    if not os.path.exists(normalized):
        raise LiveCaptureError("manifest_not_found", f"Manifest not found: {normalized}")
    payload = _safe_json_file_load(normalized)
    if not isinstance(payload, dict):
        raise LiveCaptureError("manifest_invalid", f"Manifest is not valid JSON: {normalized}")
    return payload


def _check_wav_header(path: str) -> bool:
    """Validate WAV header readability."""
    try:
        if sf is not None:
            with sf.SoundFile(path, mode="r") as handle:
                return int(handle.samplerate) > 0 and int(handle.channels) > 0
    except Exception:
        pass

    try:
        import wave
        with wave.open(path, "rb") as handle:
            return (
                int(handle.getframerate()) > 0
                and int(handle.getnchannels()) > 0
                and int(handle.getnframes()) >= 0
            )
    except Exception:
        return False


def _wait_for_export_payload(
    manifest_path: str,
    export_dir: str,
    missing: List[str]
) -> Dict[str, Any]:
    """Return standardized WAIT_FOR_USER_EXPORT payload."""
    return {
        "ok": True,
        "ready": False,
        "status": "WAIT_FOR_USER_EXPORT",
        "message_to_user": (
            "Export the listed WAVs to the export folder, then rerun check_exports_ready."
        ),
        "missing": missing,
        "export_dir": export_dir,
        "manifest_path": manifest_path
    }


def _range_payload_for_item(
    session_payload: Dict[str, Any],
    item: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve item bar/time range to seconds with no max-duration cap."""
    return _resolve_capture_range_seconds(
        session_payload=session_payload,
        start_bar=item.get("start_bar"),
        end_bar=item.get("end_bar"),
        start_time_sec=item.get("start_time_sec"),
        duration_sec=item.get("duration_sec"),
        max_duration_sec=None
    )


def _coerce_json_dict(payload: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse payload into a dict when possible."""
    if isinstance(payload, dict):
        return payload, None

    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return None, "empty_payload"
        if stripped.startswith("Error"):
            return None, stripped
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed, None
            return None, "parsed_json_not_dict"
        except Exception:
            return None, "json_parse_failed"

    return None, "unsupported_payload_type"


def _normalize_browser_token(token: str) -> str:
    """Normalize browser token for comparison and deduping."""
    lowered = token.strip().lower()
    normalized = []
    prev_is_sep = False
    for ch in lowered:
        if ch.isalnum():
            normalized.append(ch)
            prev_is_sep = False
        else:
            if not prev_is_sep:
                normalized.append("_")
                prev_is_sep = True
    normalized_token = "".join(normalized).strip("_")
    return normalized_token


def _browser_path_key(parts: List[str]) -> str:
    """Build a normalized key for browser paths."""
    return "/".join(_normalize_browser_token(part) for part in parts if isinstance(part, str) and part.strip())


def _infer_inventory_item_type(item: Dict[str, Any], item_name: str) -> Optional[str]:
    """Infer browser item type from known fields when possible."""
    if _safe_bool(item.get("is_device")):
        return "device"

    textual_fields = []
    for key in ("type", "item_type", "kind", "class_name", "class_display_name", "uri"):
        value = item.get(key)
        if isinstance(value, str):
            textual_fields.append(value.lower())

    joined = " ".join(textual_fields + [item_name.lower()])
    if "instrument" in joined:
        return "instrument"
    if "audio effect" in joined:
        return "audio_effect"
    if "midi effect" in joined:
        return "midi_effect"
    if "max for live" in joined or "m4l" in joined:
        return "max_for_live"
    if "preset" in joined:
        return "preset"
    if "rack" in joined:
        return "rack"
    if "device" in joined or "plugin" in joined or "plug-in" in joined:
        return "device"
    return None


def _is_clearly_preset_item(item: Dict[str, Any], item_name: str) -> bool:
    """Return True when an item is clearly a preset/rack artifact."""
    inferred = _infer_inventory_item_type(item, item_name)
    if inferred in {"preset", "rack"}:
        return True

    lower_name = item_name.lower()
    preset_extensions = {".adg", ".adv", ".aupreset", ".vstpreset", ".fxp", ".fxb"}
    extension = os.path.splitext(lower_name)[1]
    if extension in preset_extensions:
        return True

    uri = item.get("uri")
    if isinstance(uri, str):
        uri_lower = uri.lower()
        if "preset" in uri_lower or "rack" in uri_lower:
            return True

    return False


def _humanize_browser_root_name(raw_name: str) -> str:
    """Convert root labels/tokens into a stable display name."""
    collapsed = " ".join(raw_name.strip().replace("_", " ").split())
    if not collapsed:
        return ""

    lowercase_words = {"a", "an", "and", "for", "in", "of", "on", "or", "the", "to"}
    uppercase_words = {"api", "au", "cv", "eq", "fx", "lfo", "m4l", "midi", "osc", "vst"}

    words = []
    for index, token in enumerate(collapsed.split(" ")):
        lower = token.lower()
        if lower in uppercase_words:
            words.append(lower.upper())
        elif index > 0 and lower in lowercase_words:
            words.append(lower)
        else:
            words.append(lower.capitalize())

    return " ".join(words)


def _looks_like_browser_root_token(token: str) -> bool:
    """Return True for likely browser root tokens and False for API methods/properties."""
    normalized = _normalize_browser_token(token)
    if not normalized:
        return False
    if normalized.startswith("add_") or normalized.startswith("remove_"):
        return False
    if normalized.endswith("_listener") or normalized.endswith("_has_listener"):
        return False

    excluded = {
        "colors",
        "filter_type",
        "full_refresh",
        "hotswap_target",
        "legacy_libraries",
        "load_item",
        "preview_item",
        "relation_to_hotswap_target",
        "stop_preview"
    }
    return normalized not in excluded


def _extract_browser_root_entries(browser_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract top-level browser roots from varying tree schemas."""
    entries: List[Dict[str, Any]] = []
    entries_by_name: Dict[str, Dict[str, Any]] = {}

    def ensure_entry(display_name: str) -> Optional[Dict[str, Any]]:
        name = display_name.strip()
        if not name:
            return None
        if name in entries_by_name:
            return entries_by_name[name]
        entry = {
            "display_name": name,
            "path_candidates": []
        }
        entries.append(entry)
        entries_by_name[name] = entry
        return entry

    def add_path_candidate(entry: Dict[str, Any], value: Any) -> None:
        if isinstance(value, (int, float)):
            candidate = str(value).strip()
        elif isinstance(value, str):
            candidate = value.strip()
        else:
            return

        if not candidate:
            return

        if candidate not in entry["path_candidates"]:
            entry["path_candidates"].append(candidate)

    def add_entry(display_name: str, path_candidates: Optional[List[Any]] = None) -> None:
        entry = ensure_entry(display_name)
        if entry is None:
            return

        candidate_values = list(path_candidates or [])
        candidate_values.extend([
            display_name,
            _normalize_browser_token(display_name)
        ])
        for value in candidate_values:
            add_path_candidate(entry, value)

    def add_entry_from_dict(item: Dict[str, Any]) -> None:
        name_value: Optional[str] = None
        for key in ("name", "display_name", "title", "label"):
            raw_name = item.get(key)
            if isinstance(raw_name, str) and raw_name.strip():
                name_value = raw_name
                break
        if name_value is None:
            return

        display_name = _humanize_browser_root_name(name_value)
        path_candidates: List[Any] = [
            item.get("path"),
            item.get("path_key"),
            item.get("key"),
            item.get("id"),
            item.get("uri"),
            name_value
        ]
        add_entry(display_name, path_candidates)

    def parse_collection(items: List[Any]) -> None:
        dict_items = [item for item in items if isinstance(item, dict)]
        if dict_items:
            for item in dict_items:
                add_entry_from_dict(item)
            return

        string_items = [item for item in items if isinstance(item, str)]
        for token in string_items:
            if not _looks_like_browser_root_token(token):
                continue
            add_entry(_humanize_browser_root_name(token), [token])

    if isinstance(browser_tree, dict):
        for value in browser_tree.values():
            if isinstance(value, list):
                parse_collection(value)

    # If top-level extraction failed, fallback to recursive list scanning.
    if not entries:
        def find_lists(node: Any, depth: int = 0) -> None:
            if depth > 5:
                return
            if isinstance(node, dict):
                for value in node.values():
                    find_lists(value, depth + 1)
                return
            if isinstance(node, list):
                parse_collection(node)
                if entries:
                    return
                for value in node:
                    find_lists(value, depth + 1)

        find_lists(browser_tree, 0)

    return entries


# Core Tool endpoints

@mcp.tool()
def get_session_info(ctx: Context) -> str:
    """Get detailed information about the current Ableton session"""
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("get_session_info")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting session info from Ableton: {str(e)}")
        return f"Error getting session info: {str(e)}"

@mcp.tool()
def get_track_info(ctx: Context, track_index: int) -> str:
    """
    Get detailed information about a specific track in Ableton.
    
    Parameters:
    - track_index: The index of the track to get information about
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("get_track_info", {"track_index": track_index})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting track info from Ableton: {str(e)}")
        return f"Error getting track info: {str(e)}"

@mcp.tool()
def create_midi_track(ctx: Context, index: int = -1) -> str:
    """
    Create a new MIDI track in the Ableton session.
    
    Parameters:
    - index: The index to insert the track at (-1 = end of list)
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("create_midi_track", {"index": index})
        return f"Created new MIDI track: {result.get('name', 'unknown')}"
    except Exception as e:
        logger.error(f"Error creating MIDI track: {str(e)}")
        return f"Error creating MIDI track: {str(e)}"


@mcp.tool()
def set_track_name(ctx: Context, track_index: int, name: str) -> str:
    """
    Set the name of a track.
    
    Parameters:
    - track_index: The index of the track to rename
    - name: The new name for the track
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("set_track_name", {"track_index": track_index, "name": name})
        return f"Renamed track to: {result.get('name', name)}"
    except Exception as e:
        logger.error(f"Error setting track name: {str(e)}")
        return f"Error setting track name: {str(e)}"

@mcp.tool()
def create_clip(ctx: Context, track_index: int, clip_index: int, length: float = 4.0) -> str:
    """
    Create a new MIDI clip in the specified track and clip slot.
    
    Parameters:
    - track_index: The index of the track to create the clip in
    - clip_index: The index of the clip slot to create the clip in
    - length: The length of the clip in beats (default: 4.0)
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("create_clip", {
            "track_index": track_index, 
            "clip_index": clip_index, 
            "length": length
        })
        return f"Created new clip at track {track_index}, slot {clip_index} with length {length} beats"
    except Exception as e:
        logger.error(f"Error creating clip: {str(e)}")
        return f"Error creating clip: {str(e)}"

@mcp.tool()
def add_notes_to_clip(
    ctx: Context, 
    track_index: int, 
    clip_index: int, 
    notes: List[Dict[str, Union[int, float, bool]]]
) -> str:
    """
    Add MIDI notes to a clip.
    
    Parameters:
    - track_index: The index of the track containing the clip
    - clip_index: The index of the clip slot containing the clip
    - notes: List of note dictionaries, each with pitch, start_time, duration, velocity, and mute
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("add_notes_to_clip", {
            "track_index": track_index,
            "clip_index": clip_index,
            "notes": notes
        })
        return f"Added {len(notes)} notes to clip at track {track_index}, slot {clip_index}"
    except Exception as e:
        logger.error(f"Error adding notes to clip: {str(e)}")
        return f"Error adding notes to clip: {str(e)}"

@mcp.tool()
def set_clip_name(ctx: Context, track_index: int, clip_index: int, name: str) -> str:
    """
    Set the name of a clip.
    
    Parameters:
    - track_index: The index of the track containing the clip
    - clip_index: The index of the clip slot containing the clip
    - name: The new name for the clip
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("set_clip_name", {
            "track_index": track_index,
            "clip_index": clip_index,
            "name": name
        })
        return f"Renamed clip at track {track_index}, slot {clip_index} to '{name}'"
    except Exception as e:
        logger.error(f"Error setting clip name: {str(e)}")
        return f"Error setting clip name: {str(e)}"

@mcp.tool()
def set_tempo(ctx: Context, tempo: float) -> str:
    """
    Set the tempo of the Ableton session.
    
    Parameters:
    - tempo: The new tempo in BPM
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("set_tempo", {"tempo": tempo})
        return f"Set tempo to {tempo} BPM"
    except Exception as e:
        logger.error(f"Error setting tempo: {str(e)}")
        return f"Error setting tempo: {str(e)}"


@mcp.tool()
def load_instrument_or_effect(ctx: Context, track_index: int, uri: str) -> str:
    """
    Load an instrument or effect onto a track using its URI.
    
    Parameters:
    - track_index: The index of the track to load the instrument on
    - uri: The URI of the instrument or effect to load (e.g., 'query:Synths#Instrument%20Rack:Bass:FileId_5116')
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("load_browser_item", {
            "track_index": track_index,
            "item_uri": uri
        })
        
        # Check if the instrument was loaded successfully
        if result.get("loaded", False):
            new_devices = result.get("new_devices", [])
            if new_devices:
                return f"Loaded instrument with URI '{uri}' on track {track_index}. New devices: {', '.join(new_devices)}"
            else:
                devices = result.get("devices_after", [])
                return f"Loaded instrument with URI '{uri}' on track {track_index}. Devices on track: {', '.join(devices)}"
        else:
            return f"Failed to load instrument with URI '{uri}'"
    except Exception as e:
        logger.error(f"Error loading instrument by URI: {str(e)}")
        return f"Error loading instrument by URI: {str(e)}"

@mcp.tool()
def fire_clip(ctx: Context, track_index: int, clip_index: int) -> str:
    """
    Start playing a clip.
    
    Parameters:
    - track_index: The index of the track containing the clip
    - clip_index: The index of the clip slot containing the clip
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("fire_clip", {
            "track_index": track_index,
            "clip_index": clip_index
        })
        return f"Started playing clip at track {track_index}, slot {clip_index}"
    except Exception as e:
        logger.error(f"Error firing clip: {str(e)}")
        return f"Error firing clip: {str(e)}"

@mcp.tool()
def stop_clip(ctx: Context, track_index: int, clip_index: int) -> str:
    """
    Stop playing a clip.
    
    Parameters:
    - track_index: The index of the track containing the clip
    - clip_index: The index of the clip slot containing the clip
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("stop_clip", {
            "track_index": track_index,
            "clip_index": clip_index
        })
        return f"Stopped clip at track {track_index}, slot {clip_index}"
    except Exception as e:
        logger.error(f"Error stopping clip: {str(e)}")
        return f"Error stopping clip: {str(e)}"

@mcp.tool()
def start_playback(ctx: Context) -> str:
    """Start playing the Ableton session."""
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("start_playback")
        return "Started playback"
    except Exception as e:
        logger.error(f"Error starting playback: {str(e)}")
        return f"Error starting playback: {str(e)}"

@mcp.tool()
def stop_playback(ctx: Context) -> str:
    """Stop playing the Ableton session."""
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("stop_playback")
        return "Stopped playback"
    except Exception as e:
        logger.error(f"Error stopping playback: {str(e)}")
        return f"Error stopping playback: {str(e)}"

@mcp.tool()
def get_browser_tree(ctx: Context, category_type: str = "all") -> str:
    """
    Get a hierarchical tree of browser categories from Ableton.
    
    Parameters:
    - category_type: Type of categories to get ('all', 'instruments', 'sounds', 'drums', 'audio_effects', 'midi_effects')
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("get_browser_tree", {
            "category_type": category_type
        })
        
        # Check if we got any categories
        if "available_categories" in result and len(result.get("categories", [])) == 0:
            available_cats = result.get("available_categories", [])
            return (f"No categories found for '{category_type}'. "
                   f"Available browser categories: {', '.join(available_cats)}")
        
        # Format the tree in a more readable way
        total_folders = result.get("total_folders", 0)
        formatted_output = f"Browser tree for '{category_type}' (showing {total_folders} folders):\n\n"
        
        def format_tree(item, indent=0):
            output = ""
            if item:
                prefix = "  " * indent
                name = item.get("name", "Unknown")
                path = item.get("path", "")
                has_more = item.get("has_more", False)
                
                # Add this item
                output += f"{prefix}• {name}"
                if path:
                    output += f" (path: {path})"
                if has_more:
                    output += " [...]"
                output += "\n"
                
                # Add children
                for child in item.get("children", []):
                    output += format_tree(child, indent + 1)
            return output
        
        # Format each category
        for category in result.get("categories", []):
            formatted_output += format_tree(category)
            formatted_output += "\n"
        
        return formatted_output
    except Exception as e:
        error_msg = str(e)
        if "Browser is not available" in error_msg:
            logger.error(f"Browser is not available in Ableton: {error_msg}")
            return f"Error: The Ableton browser is not available. Make sure Ableton Live is fully loaded and try again."
        elif "Could not access Live application" in error_msg:
            logger.error(f"Could not access Live application: {error_msg}")
            return f"Error: Could not access the Ableton Live application. Make sure Ableton Live is running and the Remote Script is loaded."
        else:
            logger.error(f"Error getting browser tree: {error_msg}")
            return f"Error getting browser tree: {error_msg}"

@mcp.tool()
def get_browser_items_at_path(ctx: Context, path: str) -> str:
    """
    Get browser items at a specific path in Ableton's browser.
    
    Parameters:
    - path: Path in the format "category/folder/subfolder"
            where category is one of the available browser categories in Ableton
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("get_browser_items_at_path", {
            "path": path
        })
        
        # Check if there was an error with available categories
        if "error" in result and "available_categories" in result:
            error = result.get("error", "")
            available_cats = result.get("available_categories", [])
            return (f"Error: {error}\n"
                   f"Available browser categories: {', '.join(available_cats)}")
        
        return json.dumps(result, indent=2)
    except Exception as e:
        error_msg = str(e)
        if "Browser is not available" in error_msg:
            logger.error(f"Browser is not available in Ableton: {error_msg}")
            return f"Error: The Ableton browser is not available. Make sure Ableton Live is fully loaded and try again."
        elif "Could not access Live application" in error_msg:
            logger.error(f"Could not access Live application: {error_msg}")
            return f"Error: Could not access the Ableton Live application. Make sure Ableton Live is running and the Remote Script is loaded."
        elif "Unknown or unavailable category" in error_msg:
            logger.error(f"Invalid browser category: {error_msg}")
            return f"Error: {error_msg}. Please check the available categories using get_browser_tree."
        elif "Path part" in error_msg and "not found" in error_msg:
            logger.error(f"Path not found: {error_msg}")
            return f"Error: {error_msg}. Please check the path and try again."
        else:
            logger.error(f"Error getting browser items at path: {error_msg}")
            return f"Error getting browser items at path: {error_msg}"

@mcp.tool()
def load_drum_kit(ctx: Context, track_index: int, rack_uri: str, kit_path: str) -> str:
    """
    Load a drum rack and then load a specific drum kit into it.
    
    Parameters:
    - track_index: The index of the track to load on
    - rack_uri: The URI of the drum rack to load (e.g., 'Drums/Drum Rack')
    - kit_path: Path to the drum kit inside the browser (e.g., 'drums/acoustic/kit1')
    """
    try:
        ableton = get_ableton_connection()
        
        # Step 1: Load the drum rack
        result = ableton.send_command("load_browser_item", {
            "track_index": track_index,
            "item_uri": rack_uri
        })
        
        if not result.get("loaded", False):
            return f"Failed to load drum rack with URI '{rack_uri}'"
        
        # Step 2: Get the drum kit items at the specified path
        kit_result = ableton.send_command("get_browser_items_at_path", {
            "path": kit_path
        })
        
        if "error" in kit_result:
            return f"Loaded drum rack but failed to find drum kit: {kit_result.get('error')}"
        
        # Step 3: Find a loadable drum kit
        kit_items = kit_result.get("items", [])
        loadable_kits = [item for item in kit_items if item.get("is_loadable", False)]
        
        if not loadable_kits:
            return f"Loaded drum rack but no loadable drum kits found at '{kit_path}'"
        
        # Step 4: Load the first loadable kit
        kit_uri = loadable_kits[0].get("uri")
        load_result = ableton.send_command("load_browser_item", {
            "track_index": track_index,
            "item_uri": kit_uri
        })
        
        return f"Loaded drum rack and kit '{loadable_kits[0].get('name')}' on track {track_index}"
    except Exception as e:
        logger.error(f"Error loading drum kit: {str(e)}")
        return f"Error loading drum kit: {str(e)}"


@mcp.tool()
def list_session_clips(ctx: Context, track_index: int) -> Dict[str, Any]:
    """
    List non-empty Session View clip slots for a track.

    Parameters:
    - track_index: 0-based track index
    """
    try:
        ableton = get_ableton_connection()
        session_info = ableton.send_command("get_session_info")
        track_count = session_info.get("track_count")

        if isinstance(track_count, int) and (track_index < 0 or track_index >= track_count):
            return {
                "error": "invalid_track_index",
                "message": f"track_index {track_index} out of range (0-{max(track_count - 1, 0)})",
                "track_index": track_index,
                "track_count": track_count
            }

        track_info = ableton.send_command("get_track_info", {"track_index": track_index})
        clips = []

        for slot in track_info.get("clip_slots", []):
            if not slot.get("has_clip"):
                continue

            clip_info = slot.get("clip") or {}
            clip_types = _clip_type_from_track(track_info, clip_info)

            clips.append({
                "clip_slot_index": slot.get("index"),
                "clip_name": clip_info.get("name"),
                "is_audio_clip": clip_types["is_audio_clip"],
                "is_midi_clip": clip_types["is_midi_clip"]
            })

        return {
            "track_index": track_index,
            "track_name": track_info.get("name"),
            "clips": clips
        }
    except Exception as e:
        logger.error(f"Error listing session clips: {str(e)}")
        return {
            "error": "list_session_clips_failed",
            "message": str(e),
            "track_index": track_index
        }


@mcp.tool()
def get_session_clip_source_path(ctx: Context, track_index: int, clip_slot_index: int) -> Dict[str, Any]:
    """
    Get source file path metadata for a Session View clip.

    Parameters:
    - track_index: 0-based track index
    - clip_slot_index: 0-based clip slot index in Session View
    """
    path_chains = [
        ("clip.sample.file_path", ["clip", "sample", "file_path"]),
        ("clip.sample_path", ["clip", "sample_path"]),
        ("clip.file_path", ["clip", "file_path"]),
        ("clip.path", ["clip", "path"]),
        ("clip.sample.filepath", ["clip", "sample", "filepath"])
    ]

    try:
        ableton = get_ableton_connection()
        session_info = ableton.send_command("get_session_info")
        track_count = session_info.get("track_count")

        if isinstance(track_count, int) and (track_index < 0 or track_index >= track_count):
            return {
                "error": "invalid_track_index",
                "message": f"track_index {track_index} out of range (0-{max(track_count - 1, 0)})",
                "track_index": track_index,
                "track_count": track_count,
                "clip_slot_index": clip_slot_index
            }

        track_info = ableton.send_command("get_track_info", {"track_index": track_index})
        clip_slots = track_info.get("clip_slots", [])

        if clip_slot_index < 0 or clip_slot_index >= len(clip_slots):
            return {
                "error": "invalid_clip_slot_index",
                "message": f"clip_slot_index {clip_slot_index} out of range (0-{max(len(clip_slots) - 1, 0)})",
                "track_index": track_index,
                "clip_slot_index": clip_slot_index
            }

        slot = clip_slots[clip_slot_index]
        if not slot.get("has_clip"):
            return {
                "track_index": track_index,
                "track_name": track_info.get("name"),
                "clip_slot_index": clip_slot_index,
                "clip_name": None,
                "is_audio_clip": False,
                "is_midi_clip": False,
                "file_path": None,
                "exists_on_disk": None,
                "error": "empty_clip_slot",
                "debug_tried": [name for name, _ in path_chains]
            }

        clip_info = slot.get("clip") or {}
        clip_root = {"clip": clip_info}
        clip_types = _clip_type_from_track(track_info, clip_info)

        found_file_path = None
        found_chain = None
        debug_tried = []

        for chain_name, chain_keys in path_chains:
            debug_tried.append(chain_name)
            probed = _probe_chain(clip_root, chain_keys)
            if isinstance(probed, str) and probed.strip():
                found_file_path = probed.strip()
                found_chain = chain_name
                break

        if found_file_path is None:
            exists_on_disk = None
        elif not os.path.isabs(found_file_path):
            exists_on_disk = None
        else:
            try:
                exists_on_disk = os.path.exists(found_file_path)
            except Exception:
                exists_on_disk = None

        result = {
            "track_index": track_index,
            "track_name": track_info.get("name"),
            "clip_slot_index": clip_slot_index,
            "clip_name": clip_info.get("name"),
            "is_audio_clip": clip_types["is_audio_clip"],
            "is_midi_clip": clip_types["is_midi_clip"],
            "file_path": found_file_path,
            "exists_on_disk": exists_on_disk
        }

        if found_chain is not None:
            result["debug_found_at"] = found_chain
        else:
            result["debug_tried"] = debug_tried

        return result
    except Exception as e:
        logger.error(f"Error getting session clip source path: {str(e)}")
        return {
            "error": "get_session_clip_source_path_failed",
            "message": str(e),
            "track_index": track_index,
            "clip_slot_index": clip_slot_index
        }


@mcp.tool()
def list_arrangement_clips(ctx: Context, track_index: int) -> Dict[str, Any]:
    """
    List arrangement clips for a track.
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("list_arrangement_clips", {"track_index": track_index})
        if isinstance(result, dict):
            return result

        return {
            "supported": False,
            "track_index": track_index,
            "reason": "invalid_response",
            "debug": {
                "backend_command": "list_arrangement_clips",
                "backend_result_type": str(type(result))
            }
        }
    except Exception as e:
        logger.error(f"Error listing arrangement clips: {str(e)}")
        return {
            "supported": False,
            "track_index": track_index,
            "reason": "arrangement_clip_access_failed",
            "debug": {
                "exception_type": type(e).__name__,
                "exception_message": str(e)
            }
        }


@mcp.tool()
def get_arrangement_clip_source_path(ctx: Context, track_index: int, clip_index: int) -> Dict[str, Any]:
    """
    Get source path metadata for an arrangement clip.

    Parameters:
    - track_index: 0-based track index
    - clip_index: 0-based arrangement clip index
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("get_arrangement_clip_source_path", {
            "track_index": track_index,
            "clip_index": clip_index
        })
        if isinstance(result, dict):
            return result

        return {
            "error": "invalid_response",
            "track_index": track_index,
            "clip_index": clip_index,
            "message": "Backend returned a non-dict response",
            "debug": {
                "backend_command": "get_arrangement_clip_source_path",
                "backend_result_type": str(type(result))
            }
        }
    except Exception as e:
        logger.error(f"Error getting arrangement clip source path: {str(e)}")
        return {
            "error": "get_arrangement_clip_source_path_failed",
            "track_index": track_index,
            "clip_index": clip_index,
            "message": str(e)
        }


@mcp.tool()
def get_detail_clip_source_path(ctx: Context) -> Dict[str, Any]:
    """
    Get source path metadata for the currently selected Detail View clip.
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("get_detail_clip_source_path")
        if isinstance(result, dict):
            return result

        return {
            "error": "invalid_response",
            "message": "Backend returned a non-dict response",
            "debug": {
                "backend_command": "get_detail_clip_source_path",
                "backend_result_type": str(type(result))
            }
        }
    except Exception as e:
        logger.error(f"Error getting detail clip source path: {str(e)}")
        return {
            "error": "get_detail_clip_source_path_failed",
            "message": str(e)
        }


@mcp.tool()
def get_track_device_chain(ctx: Context, track_index: int) -> Dict[str, Any]:
    """
    Get ordered device chain metadata for a track.

    Parameters:
    - track_index: 0-based track index
    """
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("get_track_devices", {"track_index": track_index})
        if isinstance(result, dict):
            if "ok" not in result:
                result = dict(result)
                result["ok"] = True
            return result

        return {
            "ok": False,
            "error": "invalid_response",
            "message": "Backend returned a non-dict response",
            "track_index": track_index,
            "debug": {
                "backend_command": "get_track_devices",
                "backend_result_type": str(type(result))
            }
        }
    except Exception as e:
        logger.error(f"Error getting track device chain: {str(e)}")
        return {
            "ok": False,
            "error": "get_track_device_chain_failed",
            "message": str(e),
            "track_index": track_index
        }


@mcp.tool()
def get_device_parameters(
    ctx: Context,
    track_index: int,
    device_index: int,
    offset: int = 0,
    limit: int = 64
) -> Dict[str, Any]:
    """
    Get paged device parameters for a track device.

    Parameters:
    - track_index: 0-based track index
    - device_index: 0-based device index on the track
    - offset: parameter offset
    - limit: page size
    """
    try:
        offset_value = int(offset)
        limit_value = int(limit)
    except Exception:
        return {
            "ok": False,
            "error": "invalid_paging",
            "message": "offset and limit must be integers",
            "track_index": track_index,
            "device_index": device_index,
            "offset": offset,
            "limit": limit
        }

    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("get_device_parameters", {
            "track_index": track_index,
            "device_index": device_index,
            "offset": offset_value,
            "limit": limit_value
        })
        if isinstance(result, dict):
            if "ok" not in result:
                result = dict(result)
                result["ok"] = True
            return result

        return {
            "ok": False,
            "error": "invalid_response",
            "message": "Backend returned a non-dict response",
            "track_index": track_index,
            "device_index": device_index,
            "offset": offset_value,
            "limit": limit_value,
            "debug": {
                "backend_command": "get_device_parameters",
                "backend_result_type": str(type(result))
            }
        }
    except Exception as e:
        logger.error(f"Error getting device parameters: {str(e)}")
        return {
            "ok": False,
            "error": "get_device_parameters_failed",
            "message": str(e),
            "track_index": track_index,
            "device_index": device_index,
            "offset": offset_value,
            "limit": limit_value
        }


@mcp.tool()
def snapshot_device_parameters(ctx: Context, track_index: int, device_index: int) -> Dict[str, Any]:
    """
    Build a deterministic all-parameter snapshot hash for a track device.

    Parameters:
    - track_index: 0-based track index
    - device_index: 0-based device index on the track
    """
    page_size = 256

    def _snapshot_rows(parameter_payload: Any) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not isinstance(parameter_payload, list):
            return rows

        for parameter in parameter_payload:
            if not isinstance(parameter, dict):
                continue
            parameter_index = parameter.get("parameter_index")
            try:
                parameter_index = int(parameter_index)
            except Exception:
                continue

            rows.append({
                "parameter_index": parameter_index,
                "name": parameter.get("name"),
                "value": parameter.get("value")
            })
        return rows

    try:
        ableton = get_ableton_connection()
        first_page = ableton.send_command("get_device_parameters", {
            "track_index": track_index,
            "device_index": device_index,
            "offset": 0,
            "limit": page_size
        })
        if not isinstance(first_page, dict):
            return {
                "ok": False,
                "error": "invalid_response",
                "message": "Backend returned a non-dict response",
                "track_index": track_index,
                "device_index": device_index,
                "debug": {
                    "backend_command": "get_device_parameters",
                    "backend_result_type": str(type(first_page))
                }
            }
        if first_page.get("ok") is False:
            return first_page

        total_parameters_raw = first_page.get("total_parameters")
        try:
            total_parameters = int(total_parameters_raw)
            if total_parameters < 0:
                total_parameters = 0
        except Exception:
            total_parameters = len(_snapshot_rows(first_page.get("parameters", [])))

        device_name = first_page.get("device_name")
        snapshot_by_index: Dict[int, Dict[str, Any]] = {}
        first_rows = _snapshot_rows(first_page.get("parameters", []))
        for row in first_rows:
            snapshot_by_index[row["parameter_index"]] = row

        next_offset = page_size
        while next_offset < total_parameters:
            page_result = ableton.send_command("get_device_parameters", {
                "track_index": track_index,
                "device_index": device_index,
                "offset": next_offset,
                "limit": page_size
            })
            if not isinstance(page_result, dict):
                return {
                    "ok": False,
                    "error": "invalid_response",
                    "message": "Backend returned a non-dict response during snapshot pagination",
                    "track_index": track_index,
                    "device_index": device_index,
                    "offset": next_offset
                }
            if page_result.get("ok") is False:
                return page_result

            rows = _snapshot_rows(page_result.get("parameters", []))
            if not rows:
                break

            for row in rows:
                snapshot_by_index[row["parameter_index"]] = row

            next_offset += page_size

        snapshot = sorted(snapshot_by_index.values(), key=lambda item: item["parameter_index"])
        if len(snapshot) < total_parameters:
            return {
                "ok": False,
                "error": "incomplete_snapshot",
                "message": (
                    f"Expected {total_parameters} parameters but only captured {len(snapshot)}"
                ),
                "track_index": track_index,
                "device_index": device_index,
                "device_name": device_name,
                "parameter_count": len(snapshot)
            }

        snapshot_payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        snapshot_hash = "sha256:" + hashlib.sha256(snapshot_payload.encode("utf-8")).hexdigest()

        return {
            "ok": True,
            "track_index": track_index,
            "device_index": device_index,
            "device_name": device_name,
            "parameter_count": len(snapshot),
            "snapshot_hash": snapshot_hash,
            "snapshot": snapshot
        }
    except Exception as e:
        logger.error(f"Error snapshotting device parameters: {str(e)}")
        return {
            "ok": False,
            "error": "snapshot_device_parameters_failed",
            "message": str(e),
            "track_index": track_index,
            "device_index": device_index
        }


@mcp.tool()
def snapshot_project_state(ctx: Context, include_device_hashes: bool = True) -> Dict[str, Any]:
    """
    Capture a compact project state snapshot and persist it to cache.

    Parameters:
    - include_device_hashes: include per-device parameter snapshot hashes
    """
    warnings: List[str] = []

    try:
        ableton = get_ableton_connection()
        session_payload = ableton.send_command("get_session_info")
        if not isinstance(session_payload, dict):
            return {
                "ok": False,
                "error": "invalid_session_info",
                "message": "Backend returned a non-dict session response"
            }

        track_count = session_payload.get("track_count")
        if not isinstance(track_count, int) or track_count < 0:
            return {
                "ok": False,
                "error": "invalid_session_info",
                "message": "track_count missing or invalid"
            }

        session_compact = {
            "tempo": session_payload.get("tempo"),
            "signature_numerator": session_payload.get("signature_numerator"),
            "signature_denominator": session_payload.get("signature_denominator"),
            "track_count": track_count,
            "return_track_count": session_payload.get("return_track_count")
        }

        tracks_payload: List[Dict[str, Any]] = []
        sidecar_devices: Dict[str, Dict[str, Any]] = {}
        track_with_devices_count = 0
        device_count = 0
        plugin_device_count = 0
        devices_with_hash_count = 0

        include_hashes = bool(include_device_hashes)
        for track_index in range(track_count):
            try:
                track_info = ableton.send_command("get_track_info", {"track_index": track_index})
            except Exception as exc:
                track_info = {}
                warnings.append(f"track_info_failed:{track_index}:{str(exc)}")
            if not isinstance(track_info, dict):
                track_info = {}
                warnings.append(f"track_info_invalid:{track_index}")

            try:
                chain_payload = ableton.send_command("get_track_devices", {"track_index": track_index})
            except Exception as exc:
                chain_payload = {"ok": False}
                warnings.append(f"track_device_chain_failed:{track_index}:{str(exc)}")
            if not isinstance(chain_payload, dict):
                chain_payload = {"ok": False}
                warnings.append(f"track_device_chain_invalid:{track_index}")

            track_name = None
            if isinstance(chain_payload.get("track_name"), str):
                track_name = chain_payload.get("track_name")
            elif isinstance(track_info.get("name"), str):
                track_name = track_info.get("name")

            track_row = {
                "track_index": track_index,
                "track_name": track_name,
                "mixer": {
                    "volume": track_info.get("volume"),
                    "panning": track_info.get("panning"),
                    "mute": track_info.get("mute"),
                    "solo": track_info.get("solo"),
                    "arm": track_info.get("arm")
                },
                "device_count": 0,
                "devices": []
            }

            devices_payload = chain_payload.get("devices", [])
            if not isinstance(devices_payload, list):
                devices_payload = []

            if devices_payload:
                track_with_devices_count += 1

            device_occurrence_counts: Dict[Tuple[str, str], int] = {}
            for device in devices_payload:
                if not isinstance(device, dict):
                    continue

                device_index = device.get("device_index")
                if not isinstance(device_index, int):
                    continue

                is_plugin = device.get("is_plugin")
                class_name = device.get("class_name")
                device_name = device.get("name") if isinstance(device.get("name"), str) else None
                class_name_key = class_name if isinstance(class_name, str) and class_name else "unknown_class"
                device_name_key = device_name if isinstance(device_name, str) and device_name else "unknown_device"
                occurrence_tuple = (class_name_key, device_name_key)
                occurrence_index = device_occurrence_counts.get(occurrence_tuple, 0)
                device_occurrence_counts[occurrence_tuple] = occurrence_index + 1
                device_key = _stable_snapshot_device_key(
                    track_index=track_index,
                    class_name=class_name_key,
                    device_name=device_name_key,
                    occurrence_index=occurrence_index
                )

                if is_plugin is True or class_name == "PluginDevice":
                    plugin_device_count += 1
                device_count += 1

                device_row = {
                    "device_key": device_key,
                    "device_index": device_index,
                    "name": device_name,
                    "device_name": device_name,
                    "class_name": class_name,
                    "is_plugin": is_plugin,
                    "plugin_format": device.get("plugin_format"),
                    "vendor": device.get("vendor"),
                    "parameter_count": device.get("parameter_count"),
                    "snapshot_hash": None
                }

                if include_hashes:
                    snapshot_payload = snapshot_device_parameters(ctx, track_index, device_index)
                    if isinstance(snapshot_payload, dict) and snapshot_payload.get("ok"):
                        snapshot_hash = snapshot_payload.get("snapshot_hash")
                        if isinstance(snapshot_hash, str):
                            device_row["snapshot_hash"] = snapshot_hash
                            devices_with_hash_count += 1

                        snapshot_rows = snapshot_payload.get("snapshot", [])
                        sidecar_devices[device_key] = {
                            "device_key": device_key,
                            "legacy_device_key": _project_device_key(track_index, device_index),
                            "track_index": track_index,
                            "track_name": track_name,
                            "device_index": device_index,
                            "device_name": device_name,
                            "class_name": class_name,
                            "parameters": sorted(
                                _snapshot_parameter_index_map(snapshot_rows).values(),
                                key=lambda item: item["parameter_index"]
                            )
                        }
                    else:
                        snapshot_error = None
                        if isinstance(snapshot_payload, dict):
                            snapshot_error = snapshot_payload.get("error")
                        warnings.append(
                            f"device_snapshot_failed:{track_index}:{device_index}:{snapshot_error or 'unknown'}"
                        )

                track_row["devices"].append(device_row)

            track_row["device_count"] = len(track_row["devices"])
            tracks_payload.append(track_row)

        project_hash = _build_snapshot_project_hash(session_compact, tracks_payload)
        created_at = _utc_now_iso()
        timestamp_token = _project_snapshot_timestamp_token()
        snapshot_base = f"{timestamp_token}_{project_hash}"
        snapshot_id = snapshot_base

        _ensure_project_snapshot_dir()
        snapshot_file_path = os.path.join(_PROJECT_SNAPSHOT_DIR, f"{snapshot_id}.json")
        suffix = 2
        while os.path.exists(snapshot_file_path):
            snapshot_id = f"{snapshot_base}_{suffix}"
            snapshot_file_path = os.path.join(_PROJECT_SNAPSHOT_DIR, f"{snapshot_id}.json")
            suffix += 1

        snapshot_payload = {
            "schema_version": _PROJECT_SNAPSHOT_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "created_at": created_at,
            "project_hash": project_hash,
            "include_device_hashes": include_hashes,
            "session": session_compact,
            "track_count": track_count,
            "tracks": tracks_payload
        }

        if include_hashes:
            sidecar_file_name = f"{snapshot_id}.params.json"
            sidecar_file_path = os.path.join(_PROJECT_SNAPSHOT_DIR, sidecar_file_name)
            sidecar_payload = {
                "schema_version": _PROJECT_SNAPSHOT_SCHEMA_VERSION,
                "snapshot_id": snapshot_id,
                "created_at": created_at,
                "devices": sidecar_devices
            }
            _safe_json_file_write(sidecar_file_path, sidecar_payload)
            snapshot_payload["parameter_sidecar_file"] = sidecar_file_name

        _safe_json_file_write(snapshot_file_path, snapshot_payload)

        result = {
            "ok": True,
            "snapshot_id": snapshot_id,
            "file_path": snapshot_file_path,
            "project_hash": project_hash,
            "include_device_hashes": include_hashes,
            "summary": {
                "track_count": track_count,
                "tracks_with_devices": track_with_devices_count,
                "device_count": device_count,
                "plugin_device_count": plugin_device_count,
                "devices_with_hashes": devices_with_hash_count
            }
        }
        if warnings:
            result["warnings"] = warnings
        return result
    except Exception as e:
        logger.error(f"Error snapshotting project state: {str(e)}")
        return {
            "ok": False,
            "error": "snapshot_project_state_failed",
            "message": str(e)
        }


@mcp.tool()
def diff_project_state(
    ctx: Context,
    before_snapshot_id: str,
    after_snapshot_id: str,
    include_parameter_diffs: bool = False
) -> Dict[str, Any]:
    """
    Diff two saved project state snapshots.

    Parameters:
    - before_snapshot_id: snapshot id (or file name/path) for the baseline
    - after_snapshot_id: snapshot id (or file name/path) for the comparison snapshot
    - include_parameter_diffs: include compact parameter deltas for devices with changed hashes
    """
    _ = ctx

    def _track_map(payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        mapped: Dict[int, Dict[str, Any]] = {}
        tracks = payload.get("tracks", [])
        if not isinstance(tracks, list):
            return mapped
        for track in tracks:
            if not isinstance(track, dict):
                continue
            track_index = track.get("track_index")
            try:
                track_index = int(track_index)
            except Exception:
                continue
            mapped[track_index] = track
        return mapped

    def _device_signature(device: Dict[str, Any]) -> str:
        return f"{device.get('name')}|{device.get('class_name')}"

    def _multiset_counts(values: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return counts

    def _keyed_devices(track_index: int, devices: List[Any]) -> List[Dict[str, Any]]:
        keyed: List[Dict[str, Any]] = []
        occurrence_counts: Dict[Tuple[str, str], int] = {}
        for device in devices:
            if not isinstance(device, dict):
                continue

            class_name = device.get("class_name")
            device_name = device.get("device_name")
            if not isinstance(device_name, str):
                device_name = device.get("name")

            class_name_key = class_name if isinstance(class_name, str) and class_name else "unknown_class"
            device_name_key = device_name if isinstance(device_name, str) and device_name else "unknown_device"
            occurrence_tuple = (class_name_key, device_name_key)
            occurrence_index = occurrence_counts.get(occurrence_tuple, 0)
            occurrence_counts[occurrence_tuple] = occurrence_index + 1

            resolved_key = _resolve_snapshot_device_key(device, track_index, occurrence_index)
            row = dict(device)
            row["_resolved_device_key"] = resolved_key
            row["_resolved_device_name"] = device_name
            keyed.append(row)
        return keyed

    def _sidecar_device_payload(
        sidecar_devices: Dict[str, Any],
        device_key: Any,
        track_index: Any,
        device_index: Any
    ) -> Dict[str, Any]:
        if not isinstance(sidecar_devices, dict):
            return {}

        if isinstance(device_key, str) and device_key:
            payload = sidecar_devices.get(device_key)
            if isinstance(payload, dict):
                return payload

        if isinstance(track_index, int) and isinstance(device_index, int):
            legacy_key = _project_device_key(track_index, device_index)
            payload = sidecar_devices.get(legacy_key)
            if isinstance(payload, dict):
                return payload
        return {}

    warnings: List[str] = []

    before_path = _resolve_project_snapshot_file(before_snapshot_id)
    if before_path is None:
        return {
            "ok": False,
            "error": "before_snapshot_not_found",
            "message": f"Could not resolve snapshot '{before_snapshot_id}'"
        }

    after_path = _resolve_project_snapshot_file(after_snapshot_id)
    if after_path is None:
        return {
            "ok": False,
            "error": "after_snapshot_not_found",
            "message": f"Could not resolve snapshot '{after_snapshot_id}'"
        }

    before_payload = _safe_json_file_load(before_path)
    if not isinstance(before_payload, dict):
        return {
            "ok": False,
            "error": "before_snapshot_invalid",
            "message": f"Could not read snapshot file '{before_path}'"
        }

    after_payload = _safe_json_file_load(after_path)
    if not isinstance(after_payload, dict):
        return {
            "ok": False,
            "error": "after_snapshot_invalid",
            "message": f"Could not read snapshot file '{after_path}'"
        }

    before_tracks = _track_map(before_payload)
    after_tracks = _track_map(after_payload)

    before_track_indices = set(before_tracks.keys())
    after_track_indices = set(after_tracks.keys())

    added_tracks = []
    for track_index in sorted(after_track_indices - before_track_indices):
        track = after_tracks.get(track_index, {})
        added_tracks.append({
            "track_index": track_index,
            "track_name": track.get("track_name")
        })

    removed_tracks = []
    for track_index in sorted(before_track_indices - after_track_indices):
        track = before_tracks.get(track_index, {})
        removed_tracks.append({
            "track_index": track_index,
            "track_name": track.get("track_name")
        })

    session_changes = []
    before_session = before_payload.get("session", {})
    after_session = after_payload.get("session", {})
    for key in ["tempo", "signature_numerator", "signature_denominator"]:
        before_value = before_session.get(key) if isinstance(before_session, dict) else None
        after_value = after_session.get(key) if isinstance(after_session, dict) else None
        if before_value != after_value:
            session_changes.append({
                "field": key,
                "before": before_value,
                "after": after_value
            })

    mixer_changes = []
    device_chain_changes = []
    changed_device_hashes = []

    shared_tracks = sorted(before_track_indices & after_track_indices)
    for track_index in shared_tracks:
        before_track = before_tracks.get(track_index, {})
        after_track = after_tracks.get(track_index, {})

        before_mixer = before_track.get("mixer", {}) if isinstance(before_track.get("mixer"), dict) else {}
        after_mixer = after_track.get("mixer", {}) if isinstance(after_track.get("mixer"), dict) else {}

        mixer_delta = {}
        for field in ["volume", "panning", "mute", "solo", "arm"]:
            before_value = before_mixer.get(field)
            after_value = after_mixer.get(field)
            if before_value != after_value:
                mixer_delta[field] = {"before": before_value, "after": after_value}
        if mixer_delta:
            mixer_changes.append({
                "track_index": track_index,
                "track_name": after_track.get("track_name") or before_track.get("track_name"),
                "changes": mixer_delta
            })

        before_devices = before_track.get("devices", []) if isinstance(before_track.get("devices"), list) else []
        after_devices = after_track.get("devices", []) if isinstance(after_track.get("devices"), list) else []

        before_keyed_devices = _keyed_devices(track_index, before_devices)
        after_keyed_devices = _keyed_devices(track_index, after_devices)

        before_chain = [_device_signature(device) for device in before_keyed_devices]
        after_chain = [_device_signature(device) for device in after_keyed_devices]
        before_chain_keys = [
            device.get("_resolved_device_key")
            for device in before_keyed_devices
            if isinstance(device.get("_resolved_device_key"), str)
        ]
        after_chain_keys = [
            device.get("_resolved_device_key")
            for device in after_keyed_devices
            if isinstance(device.get("_resolved_device_key"), str)
        ]

        before_signatures_by_key: Dict[str, str] = {}
        for device in before_keyed_devices:
            device_key = device.get("_resolved_device_key")
            if not isinstance(device_key, str):
                continue
            before_signatures_by_key[device_key] = _device_signature(device)

        after_signatures_by_key: Dict[str, str] = {}
        for device in after_keyed_devices:
            device_key = device.get("_resolved_device_key")
            if not isinstance(device_key, str):
                continue
            after_signatures_by_key[device_key] = _device_signature(device)

        if before_chain_keys != after_chain_keys:
            before_counts = _multiset_counts(before_chain_keys)
            after_counts = _multiset_counts(after_chain_keys)

            added_device_keys = []
            for device_key, count in after_counts.items():
                delta = count - before_counts.get(device_key, 0)
                for _ in range(max(0, delta)):
                    added_device_keys.append(device_key)

            removed_device_keys = []
            for device_key, count in before_counts.items():
                delta = count - after_counts.get(device_key, 0)
                for _ in range(max(0, delta)):
                    removed_device_keys.append(device_key)

            added_devices = [after_signatures_by_key.get(key, key) for key in added_device_keys]
            removed_devices = [before_signatures_by_key.get(key, key) for key in removed_device_keys]

            device_chain_changes.append({
                "track_index": track_index,
                "track_name": after_track.get("track_name") or before_track.get("track_name"),
                "before_chain": before_chain,
                "after_chain": after_chain,
                "before_chain_keys": before_chain_keys,
                "after_chain_keys": after_chain_keys,
                "added_devices": added_devices,
                "removed_devices": removed_devices,
                "added_device_keys": added_device_keys,
                "removed_device_keys": removed_device_keys,
                "reordered": (
                    sorted(before_chain_keys) == sorted(after_chain_keys)
                    and before_chain_keys != after_chain_keys
                )
            })

        before_by_device_key = {}
        for device in before_keyed_devices:
            device_key = device.get("_resolved_device_key")
            if isinstance(device_key, str):
                before_by_device_key[device_key] = device

        after_by_device_key = {}
        for device in after_keyed_devices:
            device_key = device.get("_resolved_device_key")
            if isinstance(device_key, str):
                after_by_device_key[device_key] = device

        for device_key in sorted(set(before_by_device_key.keys()) & set(after_by_device_key.keys())):
            before_device = before_by_device_key[device_key]
            after_device = after_by_device_key[device_key]

            before_hash = before_device.get("snapshot_hash")
            after_hash = after_device.get("snapshot_hash")
            if isinstance(before_hash, str) and isinstance(after_hash, str) and before_hash != after_hash:
                changed_device_hashes.append({
                    "track_index": track_index,
                    "track_name": after_track.get("track_name") or before_track.get("track_name"),
                    "device_key": device_key,
                    "device_index": after_device.get("device_index"),
                    "before_device_index": before_device.get("device_index"),
                    "after_device_index": after_device.get("device_index"),
                    "device_name": after_device.get("device_name") or after_device.get("name"),
                    "class_name": after_device.get("class_name"),
                    "before_hash": before_hash,
                    "after_hash": after_hash
                })

    parameter_diffs: List[Dict[str, Any]] = []
    if include_parameter_diffs and changed_device_hashes:
        before_sidecar_path = _resolve_snapshot_sidecar_file(before_payload, before_path)
        after_sidecar_path = _resolve_snapshot_sidecar_file(after_payload, after_path)

        before_sidecar = _safe_json_file_load(before_sidecar_path) if isinstance(before_sidecar_path, str) else None
        after_sidecar = _safe_json_file_load(after_sidecar_path) if isinstance(after_sidecar_path, str) else None

        before_sidecar_devices = before_sidecar.get("devices", {}) if isinstance(before_sidecar, dict) else {}
        after_sidecar_devices = after_sidecar.get("devices", {}) if isinstance(after_sidecar, dict) else {}

        if not isinstance(before_sidecar_devices, dict):
            before_sidecar_devices = {}
        if not isinstance(after_sidecar_devices, dict):
            after_sidecar_devices = {}

        if not before_sidecar_devices:
            warnings.append("before_parameter_sidecar_unavailable")
        if not after_sidecar_devices:
            warnings.append("after_parameter_sidecar_unavailable")

        max_parameter_diffs = 50
        for device_change in changed_device_hashes:
            track_index = device_change.get("track_index")
            if not isinstance(track_index, int):
                continue

            device_key = device_change.get("device_key")
            before_device_index = device_change.get("before_device_index")
            after_device_index = device_change.get("after_device_index")

            before_device_payload = _sidecar_device_payload(
                before_sidecar_devices,
                device_key=device_key,
                track_index=track_index,
                device_index=before_device_index
            )
            after_device_payload = _sidecar_device_payload(
                after_sidecar_devices,
                device_key=device_key,
                track_index=track_index,
                device_index=after_device_index
            )

            before_param_map = _snapshot_parameter_index_map(before_device_payload.get("parameters", []))
            after_param_map = _snapshot_parameter_index_map(after_device_payload.get("parameters", []))

            if not before_param_map and not after_param_map:
                continue

            changed_parameters_all = []

            for parameter_index in sorted(set(before_param_map.keys()) & set(after_param_map.keys())):
                before_parameter = before_param_map[parameter_index]
                after_parameter = after_param_map[parameter_index]
                before_value = before_parameter.get("value")
                after_value = after_parameter.get("value")
                before_name = before_parameter.get("name")
                after_name = after_parameter.get("name")
                if before_value == after_value:
                    continue

                row = {
                    "parameter_index": parameter_index,
                    "name": after_name if isinstance(after_name, str) else before_name,
                    "before": before_value,
                    "after": after_value
                }
                if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
                    row["delta"] = float(after_value) - float(before_value)
                changed_parameters_all.append(row)

            truncated = False
            changed_parameter_count = len(changed_parameters_all)
            changed_parameters = changed_parameters_all
            if changed_parameter_count > max_parameter_diffs:
                changed_parameters = changed_parameters_all[:max_parameter_diffs]
                truncated = True

            if changed_parameters_all:
                parameter_diffs.append({
                    "device_key": device_key,
                    "track_index": track_index,
                    "track_name": device_change.get("track_name"),
                    "device_index": device_change.get("device_index"),
                    "device_name": device_change.get("device_name"),
                    "changed_parameter_count": changed_parameter_count,
                    "changes": changed_parameters,
                    "changed_parameters": changed_parameters,
                    "added_parameters": [],
                    "removed_parameters": [],
                    "truncated": truncated
                })

    before_id = before_payload.get("snapshot_id")
    if not isinstance(before_id, str):
        before_id = os.path.basename(before_path).replace(".json", "")
    after_id = after_payload.get("snapshot_id")
    if not isinstance(after_id, str):
        after_id = os.path.basename(after_path).replace(".json", "")

    result = {
        "ok": True,
        "before_snapshot_id": before_id,
        "after_snapshot_id": after_id,
        "before_file_path": before_path,
        "after_file_path": after_path,
        "summary": {
            "session_changes": len(session_changes),
            "added_tracks": len(added_tracks),
            "removed_tracks": len(removed_tracks),
            "mixer_changes": len(mixer_changes),
            "device_chain_changes": len(device_chain_changes),
            "changed_device_hashes": len(changed_device_hashes),
            "parameter_diff_devices": len(parameter_diffs)
        },
        "session_changes": session_changes,
        "added_tracks": added_tracks,
        "removed_tracks": removed_tracks,
        "mixer_changes": mixer_changes,
        "device_chain_changes": device_chain_changes,
        "changed_device_hashes": changed_device_hashes
    }
    if include_parameter_diffs:
        result["parameter_diffs"] = parameter_diffs
    if warnings:
        result["warnings"] = warnings
    return result


@mcp.tool()
def summarize_diff_for_llm(ctx: Context, before_snapshot_id: str, after_snapshot_id: str) -> Dict[str, Any]:
    """
    Build deterministic, compact change summary strings from a project snapshot diff.
    """
    diff_payload = diff_project_state(
        ctx=ctx,
        before_snapshot_id=before_snapshot_id,
        after_snapshot_id=after_snapshot_id,
        include_parameter_diffs=True
    )
    if not isinstance(diff_payload, dict):
        return {
            "ok": False,
            "error": "diff_failed",
            "message": "diff_project_state returned invalid payload"
        }
    if diff_payload.get("ok") is not True:
        return diff_payload

    def _device_label(raw: Any) -> str:
        if isinstance(raw, str) and raw:
            if "|" in raw:
                return raw.split("|", 1)[0]
            return raw
        return "Unknown Device"

    def _format_delta(value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):+.4f}"
        return "changed"

    details: List[str] = []

    mixer_changes = diff_payload.get("mixer_changes", [])
    if isinstance(mixer_changes, list):
        for mixer_change in sorted(
            [row for row in mixer_changes if isinstance(row, dict)],
            key=lambda item: int(item.get("track_index", -1))
        ):
            track_index = mixer_change.get("track_index")
            changes = mixer_change.get("changes", {})
            if not isinstance(changes, dict):
                continue
            for field in ["volume", "panning", "mute", "solo", "arm"]:
                if field not in changes or not isinstance(changes[field], dict):
                    continue
                before_value = changes[field].get("before")
                after_value = changes[field].get("after")
                details.append(
                    f"Track {track_index}: {field} {before_value} -> {after_value}"
                )

    device_chain_changes = diff_payload.get("device_chain_changes", [])
    if isinstance(device_chain_changes, list):
        for chain_change in sorted(
            [row for row in device_chain_changes if isinstance(row, dict)],
            key=lambda item: int(item.get("track_index", -1))
        ):
            track_index = chain_change.get("track_index")
            removed_devices = chain_change.get("removed_devices", [])
            if isinstance(removed_devices, list):
                for removed in removed_devices:
                    details.append(f"Track {track_index}: Removed {_device_label(removed)}")
            added_devices = chain_change.get("added_devices", [])
            if isinstance(added_devices, list):
                for added in added_devices:
                    details.append(f"Track {track_index}: Added {_device_label(added)}")
            if chain_change.get("reordered") is True:
                details.append(f"Track {track_index}: Reordered device chain")

    parameter_diffs = diff_payload.get("parameter_diffs", [])
    if isinstance(parameter_diffs, list):
        sorted_parameter_diffs = sorted(
            [row for row in parameter_diffs if isinstance(row, dict)],
            key=lambda item: (
                int(item.get("track_index", -1)),
                str(item.get("device_name") or "")
            )
        )
        for parameter_diff in sorted_parameter_diffs:
            track_index = parameter_diff.get("track_index")
            device_name = parameter_diff.get("device_name")
            if not isinstance(device_name, str) or not device_name:
                device_name = "Unknown Device"
            changes = parameter_diff.get("changes", [])
            if not isinstance(changes, list) or not changes:
                continue
            compact_changes = []
            for change in changes[:5]:
                if not isinstance(change, dict):
                    continue
                parameter_name = change.get("name")
                if not isinstance(parameter_name, str) or not parameter_name:
                    parameter_name = f"Param {change.get('parameter_index')}"
                if ":" in parameter_name:
                    parameter_name = parameter_name.split(":")[-1].strip() or parameter_name
                if "delta" in change:
                    compact_changes.append(f"{parameter_name} {_format_delta(change.get('delta'))}")
                else:
                    compact_changes.append(f"{parameter_name} changed")
            if compact_changes:
                details.append(
                    f"Track {track_index}: {device_name} - {', '.join(compact_changes)}"
                )

    devices_added = 0
    devices_removed = 0
    if isinstance(device_chain_changes, list):
        for chain_change in device_chain_changes:
            if not isinstance(chain_change, dict):
                continue
            added_devices = chain_change.get("added_devices", [])
            removed_devices = chain_change.get("removed_devices", [])
            if isinstance(added_devices, list):
                devices_added += len(added_devices)
            if isinstance(removed_devices, list):
                devices_removed += len(removed_devices)

    details = sorted(dict.fromkeys(details))

    return {
        "ok": True,
        "before_snapshot_id": diff_payload.get("before_snapshot_id"),
        "after_snapshot_id": diff_payload.get("after_snapshot_id"),
        "summary": {
            "mixer_changes": len(mixer_changes) if isinstance(mixer_changes, list) else 0,
            "devices_added": devices_added,
            "devices_removed": devices_removed,
            "devices_modified": len(parameter_diffs) if isinstance(parameter_diffs, list) else 0
        },
        "details": details
    }


@mcp.tool()
def get_device_inventory(
    ctx: Context,
    roots: Optional[List[str]] = None,
    max_depth: int = 5,
    max_items_per_folder: int = 500,
    include_presets: bool = False
) -> Dict[str, Any]:
    """
    Enumerate loadable devices/effects/instruments from Ableton's browser.

    Parameters:
    - roots: Optional browser roots to scan. Defaults to ["Audio Effects", "Plugins"].
    - max_depth: Maximum recursive folder depth
    - max_items_per_folder: Hard cap per folder scan
    - include_presets: Include clearly preset/rack items when True
    """
    try:
        ableton = get_ableton_connection()
        browser_tree = ableton.send_command("get_browser_tree", {"category_type": "all"})
        if not isinstance(browser_tree, dict) or not browser_tree:
            return {
                "ok": False,
                "error": "browser_tree_unavailable"
            }

        root_entries = _extract_browser_root_entries(browser_tree)
        if not root_entries:
            return {
                "ok": False,
                "error": "browser_tree_unavailable"
            }

        try:
            depth_limit = max(0, int(max_depth))
        except Exception:
            depth_limit = 5
        try:
            items_per_folder_limit = max(1, int(max_items_per_folder))
        except Exception:
            items_per_folder_limit = 500

        overall_item_cap = 5000
        max_folder_calls = 300
        folder_calls = 0
        scanned_roots: List[str] = []
        discovered_roots = [entry.get("display_name") for entry in root_entries if isinstance(entry.get("display_name"), str)]
        available_roots: List[str] = []
        seen_available_roots = set()
        for root_name in discovered_roots + _KNOWN_BROWSER_ROOTS:
            if not isinstance(root_name, str):
                continue
            normalized_name = root_name.strip()
            if not normalized_name or normalized_name in seen_available_roots:
                continue
            seen_available_roots.add(normalized_name)
            available_roots.append(normalized_name)
        devices: List[Dict[str, Any]] = []
        folders_truncated: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        truncated = False

        device_dedupe_keys = set()
        visited_folders = set()

        root_lookup = {
            entry["display_name"]: entry
            for entry in root_entries
            if isinstance(entry.get("display_name"), str)
        }
        for root_name in _KNOWN_BROWSER_ROOTS:
            if root_name in root_lookup:
                continue
            root_lookup[root_name] = {
                "display_name": root_name,
                "path_candidates": [f"query:{root_name}", root_name, _normalize_browser_token(root_name)]
            }

        if roots is None:
            requested_roots = ["Audio Effects", "Plugins"]
        elif isinstance(roots, list):
            requested_roots = [
                value.strip() for value in roots
                if isinstance(value, str) and value.strip()
            ]
        else:
            requested_roots = []

        seen_requested = set()
        deduped_requested_roots: List[str] = []
        for root_name in requested_roots:
            if root_name in seen_requested:
                continue
            seen_requested.add(root_name)
            deduped_requested_roots.append(root_name)
        requested_roots = deduped_requested_roots

        valid_roots = [root_name for root_name in requested_roots if root_name in root_lookup]
        roots_not_found = [root_name for root_name in requested_roots if root_name not in root_lookup]

        def fetch_folder(path_key_parts: List[str]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
            nonlocal folder_calls
            if folder_calls >= max_folder_calls:
                return None, "max_folder_calls"
            folder_calls += 1
            path_string = "/".join(path_key_parts)
            try:
                response = ableton.send_command("get_browser_items_at_path", {"path": path_string})
            except Exception as exc:
                return None, str(exc)

            if not isinstance(response, dict):
                return None, f"invalid_response_type:{type(response)}"
            if response.get("error"):
                return None, str(response.get("error"))

            items = response.get("items", [])
            if not isinstance(items, list):
                return [], None
            normalized_items = [item for item in items if isinstance(item, dict)]
            return normalized_items, None

        def fetch_root_folder(root_entry: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Optional[str]]:
            path_candidates = root_entry.get("path_candidates", [])
            if not isinstance(path_candidates, list):
                return None, None, "no_path_candidates"

            last_error: Optional[str] = None
            seen_candidates = set()
            for value in path_candidates:
                if not isinstance(value, str):
                    continue
                candidate = value.strip()
                if not candidate or candidate in seen_candidates:
                    continue
                seen_candidates.add(candidate)

                items, error = fetch_folder([candidate])
                if error is None:
                    return items, candidate, None
                if error == "max_folder_calls":
                    return None, None, error
                last_error = error

            return None, None, last_error or "root_not_resolvable"

        def traverse_folder(
            path_key_parts: List[str],
            path_display_parts: List[str],
            depth: int,
            prefetched_items: Optional[List[Dict[str, Any]]] = None
        ) -> None:
            nonlocal truncated
            if truncated:
                return

            folder_key = _browser_path_key(path_display_parts)
            if folder_key in visited_folders:
                return
            visited_folders.add(folder_key)

            items = prefetched_items
            if items is None:
                fetched_items, error = fetch_folder(path_key_parts)
                if error is not None:
                    if error == "max_folder_calls":
                        truncated = True
                        folders_truncated.append({
                            "path": path_display_parts,
                            "reason": "max_folder_calls"
                        })
                        return
                    errors.append({
                        "path": path_display_parts,
                        "error": error
                    })
                    return
                items = fetched_items or []

            if len(items) > items_per_folder_limit:
                folders_truncated.append({
                    "path": path_display_parts,
                    "reason": "max_items_per_folder"
                })
                items = items[:items_per_folder_limit]

            for item in items:
                if truncated:
                    break

                item_name = item.get("name")
                if not isinstance(item_name, str) or not item_name.strip():
                    continue
                item_name = item_name.strip()

                item_display_path = path_display_parts + [item_name]
                is_loadable = _safe_bool(item.get("is_loadable"))
                is_folder = _safe_bool(item.get("is_folder"))

                if is_loadable:
                    if include_presets or not _is_clearly_preset_item(item, item_name):
                        item_id = None
                        for id_key in ("id", "item_id", "uri"):
                            id_value = item.get(id_key)
                            if isinstance(id_value, (str, int)):
                                item_id = str(id_value)
                                break

                        dedupe_key = item_id or _browser_path_key(item_display_path)
                        if dedupe_key not in device_dedupe_keys:
                            device_dedupe_keys.add(dedupe_key)
                            devices.append({
                                "name": item_name,
                                "path": item_display_path,
                                "item_id": item_id,
                                "item_type": _infer_inventory_item_type(item, item_name)
                            })

                            if len(devices) >= overall_item_cap:
                                truncated = True
                                break

                if is_folder and depth < depth_limit and not truncated:
                    traverse_folder(
                        path_key_parts=path_key_parts + [item_name],
                        path_display_parts=item_display_path,
                        depth=depth + 1
                    )

        for root_name in valid_roots:
            if truncated:
                break

            root_entry = root_lookup.get(root_name)
            if not isinstance(root_entry, dict):
                continue

            root_items, resolved_root_path, root_error = fetch_root_folder(root_entry)
            if root_error is not None:
                if root_error == "max_folder_calls":
                    truncated = True
                    folders_truncated.append({
                        "path": [root_name],
                        "reason": "max_folder_calls"
                    })
                    break
                errors.append({
                    "path": [root_name],
                    "error": root_error
                })
                continue

            if not isinstance(resolved_root_path, str) or not resolved_root_path.strip():
                errors.append({
                    "path": [root_name],
                    "error": "root_not_resolvable"
                })
                continue

            scanned_roots.append(root_name)
            traverse_folder(
                path_key_parts=[resolved_root_path],
                path_display_parts=[root_name],
                depth=0,
                prefetched_items=root_items
            )

        return {
            "ok": True,
            "requested_roots": requested_roots,
            "available_roots": available_roots,
            "scanned_roots": scanned_roots,
            "roots_not_found": roots_not_found,
            "max_depth": depth_limit,
            "include_presets": bool(include_presets),
            "truncated": truncated,
            "folders_truncated": folders_truncated,
            "devices": devices,
            "errors": errors
        }
    except Exception as e:
        logger.error(f"Error getting device inventory: {str(e)}")
        return {
            "ok": False,
            "error": "device_inventory_failed",
            "message": str(e)
        }


@mcp.tool()
def get_capability_map_status(ctx: Context) -> Dict[str, Any]:
    """Return classification cache status for current inventory."""
    cache_payload = _load_device_capabilities_cache()
    classifications = cache_payload.get("classifications", {})
    if not isinstance(classifications, dict):
        classifications = {}

    scan_params = _default_inventory_scan_params()
    devices, inventory_hash, inventory_cache_hit, inventory_warnings = _get_or_build_inventory(
        scan_params=scan_params,
        force_refresh=False
    )
    cache_age_sec = _inventory_cache_age_sec(scan_params)
    if devices is None or inventory_hash is None:
        inventory_message = "Failed to resolve current inventory"
        if inventory_warnings:
            inventory_message = inventory_warnings[-1]
            if ":" in inventory_message:
                inventory_message = inventory_message.split(":", 1)[1]

        return {
            "ok": False,
            "error": "inventory_unavailable",
            "message": inventory_message,
            "inventory_hash": None,
            "classified_count": len(classifications),
            "unclassified_count": 0,
            "cache_hit": bool(inventory_cache_hit),
            "cache_age_sec": cache_age_sec,
            "scan_params": scan_params,
            "cache_path": _DEVICE_CAPABILITIES_CACHE_PATH,
            "schema_version": _DEVICE_CAPABILITIES_SCHEMA_VERSION
        }

    classified_count = 0
    for device in devices:
        device_id = device.get("device_id")
        if isinstance(device_id, str) and device_id in classifications:
            classified_count += 1

    return {
        "ok": True,
        "inventory_hash": inventory_hash,
        "classified_count": classified_count,
        "unclassified_count": max(0, len(devices) - classified_count),
        "cache_hit": bool(inventory_cache_hit),
        "cache_age_sec": cache_age_sec,
        "scan_params": scan_params,
        "cache_path": _DEVICE_CAPABILITIES_CACHE_PATH,
        "schema_version": _DEVICE_CAPABILITIES_SCHEMA_VERSION
    }
    if inventory_warnings:
        result["warnings"] = inventory_warnings
    return result


@mcp.tool()
def get_devices_for_classification(
    ctx: Context,
    max_items: int = 200,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """Return inventory devices that are missing classifications."""
    try:
        limit = max(1, int(max_items))
    except Exception:
        limit = 200

    scan_params = _default_inventory_scan_params()
    devices, inventory_hash, inventory_cache_hit, inventory_warnings = _get_or_build_inventory(
        scan_params=scan_params,
        force_refresh=bool(force_refresh)
    )
    if devices is None or inventory_hash is None:
        inventory_message = "Failed to resolve current inventory"
        if inventory_warnings:
            inventory_message = inventory_warnings[-1]
            if ":" in inventory_message:
                inventory_message = inventory_message.split(":", 1)[1]

        return {
            "ok": False,
            "error": "inventory_unavailable",
            "message": inventory_message,
            "inventory_hash": None,
            "devices": [],
            "truncated": False,
            "cache_hit": bool(inventory_cache_hit)
        }

    cache_payload = _load_device_capabilities_cache()
    classifications = cache_payload.get("classifications", {})
    if not isinstance(classifications, dict):
        classifications = {}

    unclassified_devices = []
    for device in devices:
        device_id = device.get("device_id")
        if isinstance(device_id, str) and device_id in classifications:
            continue
        unclassified_devices.append(device)

    truncated = len(unclassified_devices) > limit
    result = {
        "ok": True,
        "inventory_hash": inventory_hash,
        "devices": unclassified_devices[:limit],
        "truncated": truncated,
        "cache_hit": bool(inventory_cache_hit)
    }
    if inventory_warnings:
        result["warnings"] = inventory_warnings
    return result


@mcp.tool()
def save_device_classifications(
    ctx: Context,
    inventory_hash: str,
    classifications: List[Dict[str, Any]],
    overwrite: bool = False
) -> Dict[str, Any]:
    """Persist device capability classifications from an external LLM."""
    if not isinstance(classifications, list):
        return {
            "ok": False,
            "error": "invalid_classifications",
            "message": "classifications must be a list",
            "saved_count": 0,
            "skipped_count": 0,
            "errors": [{"index": None, "error": "classifications_not_list"}],
            "cache_path": _DEVICE_CAPABILITIES_CACHE_PATH
        }

    allowed_buckets = set(_DEVICE_CAPABILITY_BUCKETS)
    cache_payload = _load_device_capabilities_cache()
    existing = cache_payload.get("classifications", {})
    if not isinstance(existing, dict):
        existing = {}

    saved_count = 0
    skipped_count = 0
    errors: List[Dict[str, Any]] = []

    for idx, entry in enumerate(classifications):
        if not isinstance(entry, dict):
            errors.append({"index": idx, "error": "invalid_entry_type"})
            continue

        device_id = entry.get("device_id")
        if not isinstance(device_id, str) or not device_id.strip():
            errors.append({"index": idx, "error": "invalid_device_id"})
            continue
        device_id = device_id.strip()

        bucket = entry.get("bucket")
        if not isinstance(bucket, str) or bucket not in allowed_buckets:
            errors.append({
                "index": idx,
                "device_id": device_id,
                "error": "invalid_bucket",
                "allowed_buckets": _DEVICE_CAPABILITY_BUCKETS
            })
            continue

        confidence_raw = entry.get("confidence")
        try:
            confidence = float(confidence_raw)
        except Exception:
            errors.append({"index": idx, "device_id": device_id, "error": "invalid_confidence"})
            continue

        if confidence < 0.0 or confidence > 1.0:
            errors.append({"index": idx, "device_id": device_id, "error": "confidence_out_of_range"})
            continue

        if (not overwrite) and (device_id in existing):
            skipped_count += 1
            continue

        notes_value = entry.get("notes")
        notes = None
        if isinstance(notes_value, str) and notes_value.strip():
            notes = notes_value.strip()

        stored = {
            "device_id": device_id,
            "bucket": bucket,
            "confidence": round(confidence, 6),
            "updated_at": _utc_now_iso()
        }
        if notes is not None:
            stored["notes"] = notes

        existing[device_id] = stored
        saved_count += 1

    _, current_inventory_hash, inventory_error = _get_current_inventory_devices_and_hash(ctx)
    warnings: List[str] = []
    if inventory_error is not None:
        warnings.append(f"current_inventory_hash_unavailable:{inventory_error}")
    elif isinstance(inventory_hash, str) and inventory_hash and current_inventory_hash != inventory_hash:
        warnings.append("inventory_hash_mismatch")

    next_inventory_hash: Optional[str] = current_inventory_hash
    if next_inventory_hash is None and isinstance(inventory_hash, str) and inventory_hash:
        next_inventory_hash = inventory_hash

    output_payload = {
        "schema_version": _DEVICE_CAPABILITIES_SCHEMA_VERSION,
        "inventory_hash": next_inventory_hash,
        "updated_at": _utc_now_iso(),
        "classifications": existing
    }
    _write_device_capabilities_cache(output_payload)

    return {
        "ok": True,
        "saved_count": saved_count,
        "skipped_count": skipped_count,
        "errors": errors,
        "cache_path": _DEVICE_CAPABILITIES_CACHE_PATH,
        "warnings": warnings,
        "classified_total": len(existing),
        "current_inventory_hash": current_inventory_hash,
        "submitted_inventory_hash": inventory_hash
    }


@mcp.tool()
def scan_arrangement_activity(
    ctx: Context,
    target: str,
    track_index: Optional[int] = None,
    include_children: bool = False,
    bar_resolution: int = 1,
    start_bar: Optional[int] = None,
    end_bar: Optional[int] = None
) -> Dict[str, Any]:
    """
    Scan arrangement clip activity by bar range (structural, no audio decoding).

    Parameters:
    - target: "track" | "group" | "mix"
    - track_index: required for target "track"/"group"
    - include_children: include child tracks for group target
    - bar_resolution: bars per timeline bucket
    - start_bar/end_bar: optional scan range (inclusive)
    """
    _ = ctx
    try:
        target_request = _normalize_arrangement_target(target=target, track_index=track_index)
        ableton = get_ableton_connection()
        session_payload = ableton.send_command("get_session_info")
        if not isinstance(session_payload, dict):
            raise LiveCaptureError("invalid_session_info", "Failed to retrieve session info")

        track_rows = _get_tracks_mixer_state_rows(ableton)
        if not track_rows:
            raise LiveCaptureError("track_scan_failed", "No tracks available in session")

        try:
            resolution = max(1, int(bar_resolution))
        except Exception:
            raise LiveCaptureError("invalid_bar_resolution", "bar_resolution must be an integer >= 1")

        requested_track_index = target_request.get("track_index")
        warnings: List[str] = []

        if target_request["target"] == "track":
            indices = [int(requested_track_index)]
        elif target_request["target"] == "group":
            indices = _collect_group_tree_indices(
                track_rows=track_rows,
                group_track_index=int(requested_track_index),
                include_children=bool(include_children)
            )
        else:
            if include_children:
                indices = [int(row["track_index"]) for row in track_rows if isinstance(row.get("track_index"), int)]
            else:
                indices = [
                    int(row["track_index"])
                    for row in track_rows
                    if isinstance(row.get("track_index"), int) and row.get("track_kind") != "group"
                ]
            if not indices:
                warnings.append("mix_filter_result_empty_falling_back_to_all_tracks")
                indices = [int(row["track_index"]) for row in track_rows if isinstance(row.get("track_index"), int)]

        intervals, interval_warnings = _collect_arrangement_intervals_for_tracks(ctx, indices)
        warnings.extend(interval_warnings)

        tempo = float(session_payload.get("tempo"))
        signature_numerator = int(session_payload.get("signature_numerator"))
        signature_denominator = int(session_payload.get("signature_denominator"))
        beats_per_bar = _beats_per_bar(signature_numerator, signature_denominator)

        if start_bar is None:
            resolved_start_bar = 1
        else:
            try:
                resolved_start_bar = int(start_bar)
            except Exception:
                raise LiveCaptureError("invalid_start_bar", "start_bar must be an integer")
            if resolved_start_bar < 1:
                raise LiveCaptureError("invalid_start_bar", "start_bar must be >= 1")

        if end_bar is None:
            max_end_beat = 0.0
            for row in intervals:
                try:
                    max_end_beat = max(max_end_beat, float(row.get("end_beat", 0.0)))
                except Exception:
                    continue
            if max_end_beat <= 0.0:
                resolved_end_bar = resolved_start_bar
            else:
                resolved_end_bar = max(resolved_start_bar, int(math.ceil(max_end_beat / beats_per_bar)))
        else:
            try:
                resolved_end_bar = int(end_bar)
            except Exception:
                raise LiveCaptureError("invalid_end_bar", "end_bar must be an integer")
            if resolved_end_bar < resolved_start_bar:
                raise LiveCaptureError("invalid_range", "end_bar must be >= start_bar")

        timeline = _build_activity_timeline(
            intervals=intervals,
            beats_per_bar=beats_per_bar,
            start_bar=resolved_start_bar,
            end_bar=resolved_end_bar,
            bar_resolution=resolution
        )

        result = {
            "ok": True,
            "target": target_request["target"],
            "track_index": requested_track_index,
            "include_children": bool(include_children),
            "bar_resolution": int(resolution),
            "start_bar": int(resolved_start_bar),
            "end_bar": int(resolved_end_bar),
            "bars": timeline["bars"],
            "first_active_bar": timeline["first_active_bar"],
            "active_bar_count": int(timeline["active_bar_count"]),
            "detected_activity": bool(timeline["detected_activity"]),
            "track_indices_scanned": indices,
            "track_count_scanned": len(indices),
            "clip_interval_count": len(intervals),
            "tempo": tempo,
            "signature_numerator": signature_numerator,
            "signature_denominator": signature_denominator
        }
        if warnings:
            result["warnings"] = warnings
        return result
    except LiveCaptureError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "message": exc.message,
            "bars": [],
            "first_active_bar": None,
            "active_bar_count": 0,
            "detected_activity": False
        }
    except Exception as exc:
        logger.error(f"Error scanning arrangement activity: {str(exc)}")
        return {
            "ok": False,
            "error": "scan_arrangement_activity_failed",
            "message": str(exc),
            "bars": [],
            "first_active_bar": None,
            "active_bar_count": 0,
            "detected_activity": False
        }


@mcp.tool()
def analyze_audio_file(
    ctx: Context,
    wav_path: str,
    window_sec: float = _DEFAULT_AUDIO_WINDOW_SEC,
    rms_threshold_db: float = _DEFAULT_RMS_THRESHOLD_DBFS,
    start_time_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    auto_trim_silence: bool = True
) -> Dict[str, Any]:
    """
    Analyze an audio file for signal presence, dynamics, and spectral balance.

    Parameters:
    - wav_path: absolute or launch-cwd-relative path to WAV/MP3/AIF/FLAC/M4A file
    - window_sec: analysis window size in seconds
    - rms_threshold_db: silence threshold in dBFS
    - start_time_sec/duration_sec: optional segment selection
    - auto_trim_silence: trim leading silence for metrics
    """
    _ = ctx
    try:
        normalized_input_path = _normalize_source_path(wav_path)
        input_format = _audio_input_format(normalized_input_path)
        file_exists, stat_size, stat_mtime = _safe_file_stats(normalized_input_path)
        if not file_exists:
            return {
                "ok": False,
                "error": "MISSING_WAV",
                "status": "WAIT_FOR_USER_EXPORT",
                "message": "Audio file does not exist yet. Export it first, then rerun analyze_audio_file.",
                "wav_path": normalized_input_path,
                "export_dir": get_export_dir()
            }
        if stat_size is None or stat_mtime is None:
            return {
                "ok": False,
                "error": "stat_failed",
                "message": "Could not read audio file metadata",
                "wav_path": normalized_input_path
            }
        if input_format not in _ANALYZE_AUDIO_INPUT_FORMATS:
            return {
                "ok": False,
                "error": "unsupported_audio_format",
                "message": (
                    "Supported formats: "
                    + ", ".join(sorted(_ANALYZE_AUDIO_INPUT_FORMATS))
                ),
                "wav_path": normalized_input_path
            }

        cache_input = {
            "version": _AUDIO_ANALYSIS_CACHE_VERSION,
            "source_path": normalized_input_path,
            "input_format": input_format,
            "stat_size": int(stat_size),
            "stat_mtime": float(stat_mtime),
            "window_sec": float(window_sec),
            "rms_threshold_db": float(rms_threshold_db),
            "start_time_sec": None if start_time_sec is None else float(start_time_sec),
            "duration_sec": None if duration_sec is None else float(duration_sec),
            "auto_trim_silence": bool(auto_trim_silence)
        }
        cache_key = _audio_analysis_cache_key(cache_input)
        cache_path = _audio_analysis_cache_path(cache_key)
        cached = _load_audio_analysis_cache(cache_path)
        if isinstance(cached, dict) and cached.get("ok") is True:
            result = dict(cached)
            if "original_path" not in result:
                result["original_path"] = normalized_input_path
            if "input_format" not in result:
                result["input_format"] = input_format
            if input_format != "wav" and "decoded_wav_path" not in result:
                decoded_path = _decoded_wav_cache_path(
                    source_path=normalized_input_path,
                    stat_size=int(stat_size),
                    stat_mtime=float(stat_mtime)
                )
                decoded_exists, decoded_size, _ = _safe_file_stats(decoded_path)
                if decoded_exists and isinstance(decoded_size, int) and decoded_size > 44:
                    result["decoded_wav_path"] = decoded_path
            result["cache_hit"] = True
            return result

        analysis_wav_path, decoded_wav_path = _decode_source_to_wav(
            source_path=normalized_input_path,
            input_format=input_format,
            stat_size=int(stat_size),
            stat_mtime=float(stat_mtime)
        )

        analysis = analyze_wav_file(
            wav_path=analysis_wav_path,
            window_sec=float(window_sec),
            rms_threshold_db=float(rms_threshold_db),
            start_time_sec=start_time_sec,
            duration_sec=duration_sec,
            auto_trim_silence=bool(auto_trim_silence)
        )
        result = dict(analysis)
        result["analyzed_at"] = _utc_now_iso()
        result["original_path"] = normalized_input_path
        result["input_format"] = input_format
        if isinstance(decoded_wav_path, str):
            result["decoded_wav_path"] = decoded_wav_path
        result["cache_hit"] = False
        _write_audio_analysis_cache(cache_path, result)
        return result
    except (AudioAnalysisError, LiveCaptureError) as exc:
        return {
            "ok": False,
            "error": exc.code,
            "message": exc.message,
            "wav_path": _normalize_source_path(wav_path)
        }
    except Exception as exc:
        logger.error(f"Error analyzing audio file '{wav_path}': {str(exc)}")
        return {
            "ok": False,
            "error": "analyze_audio_file_failed",
            "message": str(exc),
            "wav_path": _normalize_source_path(wav_path)
        }


@mcp.tool()
def scan_audio_presence_from_file(
    ctx: Context,
    wav_path: str,
    start_time_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    window_sec: float = _DEFAULT_AUDIO_WINDOW_SEC,
    rms_threshold_db: float = _DEFAULT_RMS_THRESHOLD_DBFS
) -> Dict[str, Any]:
    """
    Presence-focused wrapper over analyze_audio_file for exported audio files.
    """
    analysis = analyze_audio_file(
        ctx=ctx,
        wav_path=wav_path,
        window_sec=window_sec,
        rms_threshold_db=rms_threshold_db,
        start_time_sec=start_time_sec,
        duration_sec=duration_sec,
        auto_trim_silence=False
    )
    if not isinstance(analysis, dict) or not analysis.get("ok"):
        if isinstance(analysis, dict):
            return analysis
        return {
            "ok": False,
            "error": "scan_audio_presence_from_file_failed",
            "message": "Unexpected analysis response"
        }

    return {
        "ok": True,
        "wav_path": analysis.get("wav_path"),
        "windows": analysis.get("windows", []),
        "first_signal_time_sec": analysis.get("first_signal_time_sec"),
        "overall_peak_dbfs": analysis.get("overall_peak_dbfs"),
        "detected_audio": bool(analysis.get("detected_audio")),
        "cache_hit": bool(analysis.get("cache_hit", False))
    }


@mcp.tool()
def export_instructions_for_section(
    ctx: Context,
    target: str,
    track_index: Optional[int] = None,
    start_bar: Optional[int] = None,
    end_bar: Optional[int] = None,
    start_time_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    suggest_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build manual Ableton export instructions for a section (no loopback required).
    """
    _ = ctx
    try:
        target_request = _normalize_arrangement_target(target=target, track_index=track_index)
        ableton = get_ableton_connection()
        session_payload = ableton.send_command("get_session_info")
        if not isinstance(session_payload, dict):
            raise LiveCaptureError("invalid_session_info", "Failed to retrieve session info")

        range_payload = _resolve_capture_range_seconds(
            session_payload=session_payload,
            start_bar=start_bar,
            end_bar=end_bar,
            start_time_sec=start_time_sec,
            duration_sec=duration_sec,
            max_duration_sec=None
        )

        track_rows = _get_tracks_mixer_state_rows(ableton)
        track_lookup = {
            int(row["track_index"]): row
            for row in track_rows
            if isinstance(row, dict) and isinstance(row.get("track_index"), int)
        }

        selected_track = None
        if target_request["target"] in {"track", "group"}:
            selected_track = track_lookup.get(int(target_request["track_index"]))
            if selected_track is None:
                raise LiveCaptureError("invalid_track_index", f"track_index {target_request['track_index']} out of range")

        suggested_name = _suggest_export_filename(
            target=target_request["target"],
            track_index=target_request.get("track_index"),
            suggest_filename=suggest_filename,
            start_sec=range_payload["start_sec"],
            end_sec=range_payload["end_sec"]
        )
        suggested_output_path = os.path.join(get_export_dir(), suggested_name)

        instructions: List[str] = []
        instructions.append("Open the Live set in Arrangement View.")
        instructions.append(
            "Set the time selection from "
            + _format_clock(range_payload["start_sec"])
            + " to "
            + _format_clock(range_payload["end_sec"])
            + f" (duration {range_payload['duration_sec']:.3f}s)."
        )

        if target_request["target"] == "mix":
            instructions.append("Leave all intended mix tracks active; do not solo a single track.")
        elif target_request["target"] == "track":
            track_name = selected_track.get("track_name") if isinstance(selected_track, dict) else None
            instructions.append(
                "Solo track "
                + str(target_request["track_index"])
                + (f" ({track_name})" if isinstance(track_name, str) and track_name else "")
                + " for export."
            )
        else:
            track_name = selected_track.get("track_name") if isinstance(selected_track, dict) else None
            instructions.append(
                "Solo group track "
                + str(target_request["track_index"])
                + (f" ({track_name})" if isinstance(track_name, str) and track_name else "")
                + " so child tracks are rendered through the group."
            )

        instructions.append("Go to File -> Export Audio/Video.")
        instructions.append("Set file format to WAV (PCM), bit depth 24-bit, normalize OFF.")
        instructions.append("Set sample rate to match the project sample rate.")
        instructions.append("Save the export to: " + suggested_output_path)
        instructions.append("Then run analyze_audio_file with that WAV path.")

        return {
            "ok": True,
            "export_type": target_request["target"],
            "track_index": target_request.get("track_index"),
            "start_time_sec": round(range_payload["start_sec"], 6),
            "end_time_sec": round(range_payload["end_sec"], 6),
            "duration_sec": round(range_payload["duration_sec"], 6),
            "instructions": instructions,
            "suggested_output_path": suggested_output_path
        }
    except LiveCaptureError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "message": exc.message,
            "instructions": []
        }
    except Exception as exc:
        logger.error(f"Error building export instructions: {str(exc)}")
        return {
            "ok": False,
            "error": "export_instructions_for_section_failed",
            "message": str(exc),
            "instructions": []
        }


@mcp.tool()
def plan_exports(
    ctx: Context,
    job_name: str,
    items: List[Dict[str, Any]],
    wav_settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a deterministic manual-export plan and persist an export manifest.

    If audio analysis is requested and no WAV path exists yet, this tool should be
    called first to plan exports and produce expected output paths.
    """
    _ = ctx
    try:
        if not isinstance(job_name, str) or not job_name.strip():
            raise LiveCaptureError("invalid_job_name", "job_name must be a non-empty string")
        if not isinstance(items, list) or not items:
            raise LiveCaptureError("invalid_items", "items must be a non-empty list")

        pathing_info = ensure_dirs_exist()
        export_dir = str(pathing_info.get("export_dir"))
        analysis_dir = str(pathing_info.get("analysis_dir"))
        warnings: List[str] = []
        pathing_warnings = pathing_info.get("warnings", [])
        if isinstance(pathing_warnings, list):
            warnings.extend(pathing_warnings)

        ableton = get_ableton_connection()
        session_payload = ableton.send_command("get_session_info")
        if not isinstance(session_payload, dict):
            raise LiveCaptureError("invalid_session_info", "Failed to retrieve session info")

        tempo = float(session_payload.get("tempo"))
        signature_numerator = int(session_payload.get("signature_numerator"))
        signature_denominator = int(session_payload.get("signature_denominator"))
        beats_per_bar = _beats_per_bar(signature_numerator, signature_denominator)

        track_rows = _get_tracks_mixer_state_rows(ableton)
        track_lookup = {
            int(row["track_index"]): row
            for row in track_rows
            if isinstance(row, dict) and isinstance(row.get("track_index"), int)
        }

        normalized_wav_settings = _normalize_wav_settings(wav_settings)
        timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_token = _sanitize_filename_token(job_name, fallback="export_job")

        planned_items: List[Dict[str, Any]] = []
        manifest_items: List[Dict[str, Any]] = []
        expected_paths: List[str] = []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise LiveCaptureError("invalid_items", f"items[{idx}] must be an object")

            target_request = _normalize_arrangement_target(
                target=item.get("target"),
                track_index=item.get("track_index")
            )
            range_payload = _range_payload_for_item(session_payload=session_payload, item=item)

            start_bar_value = range_payload.get("start_bar")
            if not isinstance(start_bar_value, int):
                start_bar_value = _seconds_to_bar_number(
                    seconds=range_payload["start_sec"],
                    tempo=tempo,
                    beats_per_bar=beats_per_bar
                )
            end_bar_value = range_payload.get("end_bar")
            if not isinstance(end_bar_value, int):
                end_bar_value = _seconds_to_bar_number(
                    seconds=max(range_payload["start_sec"], range_payload["end_sec"] - 1e-9),
                    tempo=tempo,
                    beats_per_bar=beats_per_bar
                )

            item_id = f"{job_token}__item_{idx + 1:02d}"
            track_token = ""
            if target_request["target"] in {"track", "group"}:
                track_token = f"__track{int(target_request['track_index'])}"

            hint_token = ""
            filename_hint = item.get("filename_hint")
            if isinstance(filename_hint, str) and filename_hint.strip():
                hint_token = "__" + _sanitize_filename_token(filename_hint, fallback="section")

            suggested_filename = (
                f"{job_token}__{target_request['target']}"
                f"{track_token}__bars{int(start_bar_value)}-{int(end_bar_value)}"
                f"{hint_token}__{timestamp_token}.wav"
            )
            suggested_output_path = os.path.join(export_dir, suggested_filename)

            instructions = [
                "Open the Live set in Arrangement View.",
                (
                    "Set the time selection from "
                    + _format_clock(range_payload["start_sec"])
                    + " to "
                    + _format_clock(range_payload["end_sec"])
                    + f" (bars {int(start_bar_value)}-{int(end_bar_value)})."
                )
            ]

            if target_request["target"] == "mix":
                instructions.append("Rendered Track: Master.")
            else:
                track_row = track_lookup.get(int(target_request["track_index"]))
                track_name = track_row.get("track_name") if isinstance(track_row, dict) else None
                if target_request["target"] == "track":
                    instructions.append(
                        "Solo track "
                        + str(target_request["track_index"])
                        + (f" ({track_name})" if isinstance(track_name, str) and track_name else "")
                        + " before export."
                    )
                else:
                    instructions.append(
                        "Solo group track "
                        + str(target_request["track_index"])
                        + (f" ({track_name})" if isinstance(track_name, str) and track_name else "")
                        + " to render group output."
                    )
                instructions.append("Rendered Track: Selected Tracks Only.")

            instructions.extend([
                "Go to File -> Export Audio/Video.",
                (
                    "Set WAV export: bit depth "
                    + str(normalized_wav_settings["bit_depth"])
                    + "-bit, sample rate "
                    + str(normalized_wav_settings["sample_rate"])
                    + ", normalize "
                    + ("ON" if normalized_wav_settings["normalize"] else "OFF")
                    + ", dither "
                    + ("ON" if normalized_wav_settings["dither"] else "OFF")
                    + "."
                ),
                "Save exported WAV to: " + suggested_output_path
            ])

            planned_items.append({
                "item_id": item_id,
                "target": target_request["target"],
                "track_index": target_request.get("track_index"),
                "start_time_sec": round(range_payload["start_sec"], 6),
                "end_time_sec": round(range_payload["end_sec"], 6),
                "suggested_output_path": suggested_output_path,
                "instructions": instructions
            })

            manifest_items.append({
                "item_id": item_id,
                "target": target_request["target"],
                "track_index": target_request.get("track_index"),
                "start_bar": int(start_bar_value),
                "end_bar": int(end_bar_value),
                "start_time_sec": round(range_payload["start_sec"], 6),
                "end_time_sec": round(range_payload["end_sec"], 6),
                "duration_sec": round(range_payload["duration_sec"], 6),
                "suggested_filename": suggested_filename,
                "suggested_output_path": suggested_output_path,
                "instructions": instructions
            })
            expected_paths.append(suggested_output_path)

        manifest_path = os.path.join(analysis_dir, _export_manifest_filename(job_name))
        manifest_payload = {
            "schema_version": 1,
            "created_at": _utc_now_iso(),
            "job_name": job_name,
            "export_dir": export_dir,
            "analysis_dir": analysis_dir,
            "session": {
                "tempo": tempo,
                "signature_numerator": signature_numerator,
                "signature_denominator": signature_denominator,
                "track_count": session_payload.get("track_count")
            },
            "wav_settings": normalized_wav_settings,
            "items": manifest_items,
            "expected_paths": expected_paths
        }
        _safe_json_file_write(manifest_path, manifest_payload)

        result = {
            "ok": True,
            "job_name": job_name,
            "export_dir": export_dir,
            "analysis_dir": analysis_dir,
            "manifest_path": manifest_path,
            "wav_settings": normalized_wav_settings,
            "items": planned_items
        }
        if warnings:
            result["warnings"] = warnings
        return result
    except LiveCaptureError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "message": exc.message,
            "items": []
        }
    except Exception as exc:
        logger.error(f"Error planning exports: {str(exc)}")
        return {
            "ok": False,
            "error": "plan_exports_failed",
            "message": str(exc),
            "items": []
        }


@mcp.tool()
def check_exports_ready(
    ctx: Context,
    manifest_path: Optional[str] = None,
    job_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check whether all expected WAV exports from a manifest are present and readable.
    """
    _ = ctx
    try:
        resolved_manifest_path = _resolve_export_manifest_path(
            manifest_path=manifest_path,
            job_name=job_name
        )
        manifest = _load_export_manifest(resolved_manifest_path)

        export_dir_value = manifest.get("export_dir")
        if not isinstance(export_dir_value, str) or not export_dir_value.strip():
            export_dir_value = get_export_dir()

        expected_paths_raw = manifest.get("expected_paths", [])
        expected_paths: List[str] = []
        if isinstance(expected_paths_raw, list):
            for value in expected_paths_raw:
                if isinstance(value, str) and value.strip():
                    expected_paths.append(_normalize_source_path(value))
        if not expected_paths:
            raise LiveCaptureError("manifest_invalid", "Manifest has no expected_paths")

        present = []
        missing = []
        for path in expected_paths:
            file_exists, stat_size, stat_mtime = _safe_file_stats(path)
            header_ok = _check_wav_header(path) if file_exists else False
            if file_exists and isinstance(stat_size, int) and stat_size > 0 and header_ok:
                modified_at = None
                if isinstance(stat_mtime, float):
                    modified_at = datetime.fromtimestamp(stat_mtime, tz=timezone.utc).isoformat()
                present.append({
                    "path": path,
                    "bytes": stat_size,
                    "modified_at": modified_at
                })
            else:
                missing.append(path)

        if missing:
            payload = _wait_for_export_payload(
                manifest_path=resolved_manifest_path,
                export_dir=export_dir_value,
                missing=missing
            )
            payload["present"] = present
            return payload

        return {
            "ok": True,
            "ready": True,
            "missing": [],
            "present": present,
            "manifest_path": resolved_manifest_path,
            "export_dir": export_dir_value
        }
    except LiveCaptureError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "message": exc.message
        }
    except Exception as exc:
        logger.error(f"Error checking exports ready: {str(exc)}")
        return {
            "ok": False,
            "error": "check_exports_ready_failed",
            "message": str(exc)
        }


@mcp.tool()
def analyze_export_job(
    ctx: Context,
    manifest_path: Optional[str] = None,
    job_name: Optional[str] = None,
    window_sec: float = _DEFAULT_AUDIO_WINDOW_SEC,
    rms_threshold_db: float = _DEFAULT_RMS_THRESHOLD_DBFS
) -> Dict[str, Any]:
    """
    Analyze all exported WAVs in a manifest. Returns WAIT_FOR_USER_EXPORT if missing.
    """
    try:
        readiness = check_exports_ready(ctx=ctx, manifest_path=manifest_path, job_name=job_name)
        if not isinstance(readiness, dict):
            return {
                "ok": False,
                "error": "invalid_readiness_response",
                "message": "check_exports_ready returned non-dict response"
            }
        if readiness.get("ok") is not True:
            return readiness
        if readiness.get("ready") is not True:
            return readiness

        resolved_manifest_path = _resolve_export_manifest_path(
            manifest_path=manifest_path or readiness.get("manifest_path"),
            job_name=job_name
        )
        manifest = _load_export_manifest(resolved_manifest_path)

        job_name_value = manifest.get("job_name")
        if not isinstance(job_name_value, str) or not job_name_value.strip():
            job_name_value = "export_job"
        job_token = _sanitize_filename_token(job_name_value, fallback="export_job")

        analysis_dir = manifest.get("analysis_dir")
        if not isinstance(analysis_dir, str) or not analysis_dir.strip():
            analysis_dir = get_analysis_dir()
        os.makedirs(analysis_dir, exist_ok=True)

        items = manifest.get("items", [])
        if not isinstance(items, list):
            raise LiveCaptureError("manifest_invalid", "Manifest items must be a list")

        results = []
        peak_values = []
        rms_values = []
        notes = []

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                notes.append(f"item_{index}:invalid_item_entry")
                continue

            item_id = item.get("item_id")
            if not isinstance(item_id, str) or not item_id.strip():
                item_id = f"{job_token}__item_{index + 1:02d}"

            wav_path = item.get("suggested_output_path")
            if not isinstance(wav_path, str) or not wav_path.strip():
                notes.append(f"{item_id}:missing_wav_path")
                continue

            analysis = analyze_audio_file(
                ctx=ctx,
                wav_path=wav_path,
                window_sec=window_sec,
                rms_threshold_db=rms_threshold_db,
                auto_trim_silence=True
            )

            analysis_file_name = (
                f"{job_token}__{_sanitize_filename_token(item_id, fallback='item')}"
                "__analysis.json"
            )
            analysis_json_path = os.path.join(analysis_dir, analysis_file_name)
            _safe_json_file_write(analysis_json_path, analysis if isinstance(analysis, dict) else {"ok": False})

            entry = {
                "item_id": item_id,
                "wav_path": _normalize_source_path(wav_path),
                "analysis_json_path": analysis_json_path
            }

            if isinstance(analysis, dict) and analysis.get("ok"):
                entry["detected_audio"] = bool(analysis.get("detected_audio"))
                entry["first_signal_time_sec"] = analysis.get("first_signal_time_sec")
                entry["metrics"] = analysis.get("metrics")
                metrics = analysis.get("metrics", {})
                if isinstance(metrics, dict):
                    peak_db = metrics.get("overall_peak_dbfs")
                    rms_db = metrics.get("overall_rms_dbfs")
                    if isinstance(peak_db, (int, float)):
                        peak_values.append(float(peak_db))
                    if isinstance(rms_db, (int, float)):
                        rms_values.append(float(rms_db))
            else:
                entry["detected_audio"] = False
                entry["first_signal_time_sec"] = None
                if isinstance(analysis, dict):
                    entry["error"] = analysis.get("error")
                    entry["message"] = analysis.get("message")
                notes.append(f"{item_id}:analysis_failed")

            results.append(entry)

        summary = {
            "count": len(results),
            "any_missing": False,
            "loudest_peak_dbfs": max(peak_values) if peak_values else None,
            "average_rms_dbfs": (sum(rms_values) / len(rms_values)) if rms_values else None,
            "notes": notes
        }

        return {
            "ok": True,
            "job_name": job_name_value,
            "results": results,
            "summary": summary
        }
    except LiveCaptureError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "message": exc.message
        }
    except Exception as exc:
        logger.error(f"Error analyzing export job: {str(exc)}")
        return {
            "ok": False,
            "error": "analyze_export_job_failed",
            "message": str(exc)
        }


@mcp.tool()
def suggest_export_ranges(
    ctx: Context,
    target: str,
    track_index: Optional[int] = None,
    include_children: bool = False,
    bar_resolution: int = 1,
    min_active_bars: int = 1,
    max_ranges: int = 5
) -> Dict[str, Any]:
    """
    Suggest contiguous active bar ranges for export based on arrangement activity.
    """
    activity = scan_arrangement_activity(
        ctx=ctx,
        target=target,
        track_index=track_index,
        include_children=include_children,
        bar_resolution=bar_resolution
    )
    if not isinstance(activity, dict) or not activity.get("ok"):
        if isinstance(activity, dict):
            return activity
        return {
            "ok": False,
            "error": "suggest_export_ranges_failed",
            "message": "scan_arrangement_activity returned invalid response"
        }

    bars = activity.get("bars", [])
    if not isinstance(bars, list):
        bars = []

    try:
        resolution = max(1, int(activity.get("bar_resolution", bar_resolution)))
        min_bars = max(1, int(min_active_bars))
        max_ranges_value = max(1, int(max_ranges))
    except Exception:
        return {
            "ok": False,
            "error": "invalid_range_params",
            "message": "bar_resolution, min_active_bars, and max_ranges must be integers"
        }

    ranges = []
    current_start = None
    current_end_bucket = None
    previous_bar = None
    for row in bars:
        if not isinstance(row, dict):
            continue
        bar_value = row.get("bar")
        has_clip = bool(row.get("has_clip"))
        if not isinstance(bar_value, int):
            try:
                bar_value = int(bar_value)
            except Exception:
                continue

        if has_clip:
            if current_start is None:
                current_start = bar_value
                current_end_bucket = bar_value
            elif previous_bar is not None and bar_value == previous_bar + resolution:
                current_end_bucket = bar_value
            else:
                end_bar_value = int(current_end_bucket + resolution - 1)
                length_bars = max(1, end_bar_value - int(current_start) + 1)
                if length_bars >= min_bars:
                    ranges.append({
                        "start_bar": int(current_start),
                        "end_bar": end_bar_value,
                        "length_bars": length_bars
                    })
                current_start = bar_value
                current_end_bucket = bar_value
        else:
            if current_start is not None:
                end_bar_value = int(current_end_bucket + resolution - 1)
                length_bars = max(1, end_bar_value - int(current_start) + 1)
                if length_bars >= min_bars:
                    ranges.append({
                        "start_bar": int(current_start),
                        "end_bar": end_bar_value,
                        "length_bars": length_bars
                    })
                current_start = None
                current_end_bucket = None
        previous_bar = bar_value

    if current_start is not None and current_end_bucket is not None:
        end_bar_value = int(current_end_bucket + resolution - 1)
        length_bars = max(1, end_bar_value - int(current_start) + 1)
        if length_bars >= min_bars:
            ranges.append({
                "start_bar": int(current_start),
                "end_bar": end_bar_value,
                "length_bars": length_bars
            })

    ranges.sort(key=lambda entry: (-int(entry["length_bars"]), int(entry["start_bar"])))
    ranges = ranges[:max_ranges_value]

    return {
        "ok": True,
        "target": activity.get("target"),
        "track_index": activity.get("track_index"),
        "bar_resolution": resolution,
        "min_active_bars": min_bars,
        "max_ranges": max_ranges_value,
        "detected_activity": bool(activity.get("detected_activity")),
        "first_active_bar": activity.get("first_active_bar"),
        "ranges": ranges
    }


@mcp.tool()
def get_export_workflow_help(ctx: Context) -> Dict[str, Any]:
    """Return canonical human-in-the-loop export workflow guidance."""
    _ = ctx
    pathing = resolve_pathing()
    export_dir = pathing.get("export_dir")
    analysis_dir = pathing.get("analysis_dir")
    return {
        "ok": True,
        "message": (
            "AbletonMCP uses manual exports as a reliable bridge for audio analysis: "
            "plan exports first, export WAVs in Ableton, then run analysis tools once files exist."
        ),
        "steps": [
            "Call plan_exports with job_name and items.",
            "Export each WAV in Ableton to the suggested output paths.",
            "Call check_exports_ready and wait until ready=true.",
            "Call analyze_export_job to analyze all exported files."
        ],
        "recommended_settings": [
            "WAV (PCM), 24-bit by default",
            "Normalize OFF",
            "Dither OFF",
            "Sample rate = project"
        ],
        "export_dir": export_dir,
        "analysis_dir": analysis_dir
    }


@mcp.tool()
def get_session_snapshot(
    ctx: Context,
    track_indices: Optional[List[int]] = None,
    include_arrangement_clip_sources: bool = True
) -> Dict[str, Any]:
    """
    Return a compact snapshot of session + track state and arrangement clip source paths.

    Parameters:
    - track_indices: Optional list of track indices. If omitted, includes all normal tracks.
    - include_arrangement_clip_sources: Include arrangement clip file path details
    """
    errors: List[Dict[str, Any]] = []

    try:
        session_payload, session_error = _coerce_json_dict(get_session_info(ctx))
        if session_payload is None:
            try:
                ableton = get_ableton_connection()
                direct_session = ableton.send_command("get_session_info")
                if isinstance(direct_session, dict):
                    session_payload = direct_session
            except Exception:
                pass

        if session_payload is None:
            return {
                "ok": False,
                "error": "session_info_unavailable",
                "message": session_error or "Failed to retrieve session info"
            }

        track_count = session_payload.get("track_count")

        if track_indices is None:
            if not isinstance(track_count, int):
                return {
                    "ok": False,
                    "error": "track_count_missing",
                    "message": "track_count not available in session info",
                    "session": session_payload
                }
            indices_to_scan = list(range(track_count))
        else:
            indices_to_scan = []
            seen_indices = set()
            for value in track_indices:
                try:
                    track_index = int(value)
                except Exception:
                    errors.append({
                        "track_index": value,
                        "error": "invalid_track_index_type"
                    })
                    continue

                if track_index in seen_indices:
                    continue
                seen_indices.add(track_index)
                indices_to_scan.append(track_index)

        tracks: List[Dict[str, Any]] = []
        for track_index in indices_to_scan:
            if isinstance(track_count, int) and (track_index < 0 or track_index >= track_count):
                errors.append({
                    "track_index": track_index,
                    "error": "invalid_track_index"
                })
                continue

            track_payload, track_error = _coerce_json_dict(get_track_info(ctx, track_index))
            if track_payload is None:
                errors.append({
                    "track_index": track_index,
                    "error": "track_info_unavailable",
                    "message": track_error
                })
                continue

            track_entry: Dict[str, Any] = {
                "track_index": track_index,
                "track_info": track_payload
            }

            if include_arrangement_clip_sources:
                arrangement_payload = list_arrangement_clips(ctx, track_index)
                if not isinstance(arrangement_payload, dict):
                    track_entry["arrangement"] = {"supported": False}
                elif not arrangement_payload.get("supported"):
                    track_entry["arrangement"] = {
                        "supported": False,
                        "reason": arrangement_payload.get("reason")
                    }
                else:
                    clips_out: List[Dict[str, Any]] = []
                    clips = arrangement_payload.get("clips", [])
                    if isinstance(clips, list):
                        for clip in clips:
                            if not isinstance(clip, dict):
                                continue

                            clip_index = clip.get("clip_index")
                            clip_entry: Dict[str, Any] = {
                                "clip_index": clip_index,
                                "clip_name": clip.get("clip_name"),
                                "is_audio_clip": clip.get("is_audio_clip"),
                                "is_midi_clip": clip.get("is_midi_clip")
                            }

                            if isinstance(clip_index, int):
                                source_payload = get_arrangement_clip_source_path(ctx, track_index, clip_index)
                                if isinstance(source_payload, dict):
                                    if "file_path" in source_payload:
                                        clip_entry["file_path"] = source_payload.get("file_path")
                                    if "exists_on_disk" in source_payload:
                                        clip_entry["exists_on_disk"] = source_payload.get("exists_on_disk")
                                    if source_payload.get("error"):
                                        clip_entry["source_error"] = source_payload.get("error")
                                else:
                                    clip_entry["source_error"] = "invalid_source_response"
                            else:
                                clip_entry["source_error"] = "invalid_clip_index"

                            clips_out.append(clip_entry)

                    track_entry["arrangement"] = {
                        "supported": True,
                        "clip_count": len(clips_out),
                        "clips": clips_out
                    }

            tracks.append(track_entry)

        return {
            "ok": True,
            "session": session_payload,
            "tracks": tracks,
            "errors": errors
        }
    except Exception as e:
        logger.error(f"Error getting session snapshot: {str(e)}")
        return {
            "ok": False,
            "error": "session_snapshot_failed",
            "message": str(e),
            "errors": errors
        }


@mcp.tool()
def analyze_source(ctx: Context, file_path: str, force: bool = False) -> Dict[str, Any]:
    """
    Analyze a source audio file and cache results by absolute path hash.

    Parameters:
    - file_path: Source file path (absolute or with ~ expansion)
    - force: Recompute analysis even when cache is valid
    """
    try:
        normalized_path = _normalize_source_path(file_path)
        cache_path = _source_cache_path(normalized_path)
        file_exists, stat_size, stat_mtime = _safe_file_stats(normalized_path)

        cache_payload = _load_source_cache(cache_path)
        cache_valid = _is_source_cache_valid(
            cache_payload=cache_payload,
            file_path=normalized_path,
            file_exists=file_exists,
            stat_size=stat_size,
            stat_mtime=stat_mtime
        )

        if not force and cache_valid:
            cached_result = dict(cache_payload)
            cached_result["cache_hit"] = True
            return cached_result

        if not file_exists:
            return {
                "ok": False,
                "error": "file_not_found",
                "message": "Source file does not exist on disk",
                "file_path": normalized_path,
                "file_exists": False,
                "cache_hit": False
            }

        if stat_size is None or stat_mtime is None:
            return {
                "ok": False,
                "error": "stat_failed",
                "message": "Could not read file metadata",
                "file_path": normalized_path,
                "file_exists": True,
                "cache_hit": False
            }

        analysis_payload = _analyze_audio_source(
            file_path=normalized_path,
            stat_size=stat_size,
            stat_mtime=stat_mtime
        )
        _write_source_cache(cache_path, analysis_payload)

        result = dict(analysis_payload)
        result["cache_hit"] = False
        return result
    except SourceAnalysisError as e:
        return {
            "ok": False,
            "error": e.code,
            "message": e.message,
            "file_path": _normalize_source_path(file_path),
            "cache_hit": False
        }
    except Exception as e:
        logger.error(f"Error analyzing source file '{file_path}': {str(e)}")
        return {
            "ok": False,
            "error": "analysis_failed",
            "message": str(e),
            "file_path": _normalize_source_path(file_path),
            "cache_hit": False
        }


@mcp.tool()
def index_sources_from_live_set(ctx: Context, track_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Index and analyze unique arrangement clip sources in the current Live set.

    Parameters:
    - track_indices: Optional list of 0-based track indices to scan
    """
    try:
        ableton = get_ableton_connection()
        if track_indices is None:
            session_info = ableton.send_command("get_session_info")
            track_count = session_info.get("track_count")
            if not isinstance(track_count, int):
                return {
                    "ok": False,
                    "error": "invalid_session_info",
                    "message": "track_count missing from get_session_info response"
                }
            indices_to_scan = list(range(track_count))
        else:
            seen_indices = set()
            indices_to_scan = []
            for index in track_indices:
                try:
                    index_int = int(index)
                except Exception:
                    continue
                if index_int in seen_indices:
                    continue
                seen_indices.add(index_int)
                indices_to_scan.append(index_int)

        unique_paths: List[str] = []
        seen_paths = set()
        tracks_with_arrangement_clips = 0
        clips_seen = 0

        for track_index in indices_to_scan:
            arrangement_result = list_arrangement_clips(ctx, track_index)
            if not isinstance(arrangement_result, dict):
                continue
            if not arrangement_result.get("supported"):
                continue

            clips = arrangement_result.get("clips", [])
            if not isinstance(clips, list):
                continue
            if clips:
                tracks_with_arrangement_clips += 1

            for clip in clips:
                clip_index = clip.get("clip_index")
                if not isinstance(clip_index, int):
                    continue
                clips_seen += 1
                source_result = get_arrangement_clip_source_path(ctx, track_index, clip_index)
                if not isinstance(source_result, dict):
                    continue
                source_path = source_result.get("file_path")
                if not isinstance(source_path, str) or not source_path.strip():
                    continue
                normalized_path = _normalize_source_path(source_path)
                if normalized_path in seen_paths:
                    continue
                seen_paths.add(normalized_path)
                unique_paths.append(normalized_path)

        sources = []
        analyzed_ok = 0
        analyzed_failed = 0

        for source_path in unique_paths:
            analysis_result = analyze_source(ctx, source_path, force=False)
            source_entry = {
                "file_path": source_path,
                "ok": bool(analysis_result.get("ok", False)),
                "cache_hit": bool(analysis_result.get("cache_hit", False))
            }
            if analysis_result.get("ok"):
                source_entry["summary"] = analysis_result.get("summary")
                analyzed_ok += 1
            else:
                source_entry["error"] = analysis_result.get("error")
                source_entry["message"] = analysis_result.get("message")
                analyzed_failed += 1
            sources.append(source_entry)

        return {
            "ok": True,
            "tracks_scanned": len(indices_to_scan),
            "tracks_with_arrangement_clips": tracks_with_arrangement_clips,
            "clips_seen": clips_seen,
            "unique_sources_found": len(unique_paths),
            "analyzed_ok": analyzed_ok,
            "analyzed_failed": analyzed_failed,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error indexing sources from live set: {str(e)}")
        return {
            "ok": False,
            "error": "index_failed",
            "message": str(e)
        }

# Main execution
def main():
    """Run the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()

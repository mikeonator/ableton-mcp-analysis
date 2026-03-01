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
import copy
import sys
import wave
from datetime import datetime, timezone
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Union, Optional, Tuple
from MCP_Server.audio_analysis import analyze_wav_file, AudioAnalysisError
from MCP_Server.als_automation import (
    read_arrangement_automation_from_project_als,
    enumerate_non_track_arrangement_automation_from_project_als,
    build_als_automation_inventory,
    get_als_automation_target_points_from_inventory,
    read_time_locators_from_project_als,
)
from MCP_Server.pathing import (
    get_export_dir,
    get_analysis_dir,
    ensure_dirs_exist,
    bootstrap_project_environment,
    resolve_pathing,
    get_project_root,
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
_MIX_CONTEXT_TAGS_SCHEMA_VERSION = 1
_MIX_CONTEXT_TAGS_FILE_NAME = "mix_context_tags.json"
_DEFAULT_AUDIO_WINDOW_SEC = 1.0
_DEFAULT_RMS_THRESHOLD_DBFS = -60.0
_DEFAULT_MASTERING_TARGET_LUFS = -14.0
_MASTERING_CLIP_THRESHOLD = 0.9999
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
_BROWSER_ROOT_ALIAS_MAP = {
    "audio_effect": "Audio Effects",
    "audio_effects": "Audio Effects",
    "audiofx": "Audio Effects",
    "audio_fx": "Audio Effects",
    "plugin": "Plugins",
    "plugins": "Plugins",
    "midi_effect": "MIDI Effects",
    "midi_effects": "MIDI Effects",
    "midifx": "MIDI Effects",
    "midi_fx": "MIDI Effects",
    "instrument": "Instruments",
    "instruments": "Instruments",
    "sound": "Sounds",
    "sounds": "Sounds",
    "drum": "Drums",
    "drums": "Drums",
    "clip": "Clips",
    "clips": "Clips",
    "current_project": "Current Project",
    "currentproject": "Current Project",
    "max_for_live": "Max for Live",
    "maxforlive": "Max for Live",
    "m4l": "Max for Live",
    "pack": "Packs",
    "packs": "Packs",
    "sample": "Samples",
    "samples": "Samples",
    "user_library": "User Library",
    "userlibrary": "User Library",
    "user_folders": "User Folders",
    "userfolders": "User Folders",
}
_BROWSER_ROOT_TO_TOKEN = {
    "Audio Effects": "audio_effects",
    "Plugins": "plugins",
    "MIDI Effects": "midi_effects",
    "Instruments": "instruments",
    "Max for Live": "max_for_live",
    "Sounds": "sounds",
    "Drums": "drums",
    "Clips": "clips",
    "Current Project": "current_project",
    "Packs": "packs",
    "Samples": "samples",
    "User Library": "user_library",
    "User Folders": "user_folders",
}
_DEVICE_INVENTORY_RUNTIME_CACHE: Dict[str, Dict[str, Any]] = {}

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


def _resolve_ffmpeg_path() -> Dict[str, Any]:
    """Resolve ffmpeg path using env override first, then PATH lookup."""
    override_raw = os.environ.get("ABLETON_MCP_FFMPEG_PATH")
    override = override_raw.strip() if isinstance(override_raw, str) else None
    attempted_paths: List[str] = []

    if override:
        override_path = os.path.abspath(os.path.expanduser(override))
        attempted_paths.append(override_path)
        if os.path.isfile(override_path) and os.access(override_path, os.X_OK):
            return {
                "available": True,
                "path": override_path,
                "source": "env_override",
                "attempted_paths": attempted_paths,
            }
        return {
            "available": False,
            "path": None,
            "source": "env_override",
            "attempted_paths": attempted_paths,
            "message": "ABLETON_MCP_FFMPEG_PATH is set but not executable"
        }

    discovered = shutil.which("ffmpeg")
    if discovered:
        discovered_path = os.path.abspath(discovered)
        attempted_paths.append(discovered_path)
        return {
            "available": True,
            "path": discovered_path,
            "source": "path_lookup",
            "attempted_paths": attempted_paths,
        }

    attempted_paths.append("ffmpeg")
    return {
        "available": False,
        "path": None,
        "source": "path_lookup",
        "attempted_paths": attempted_paths,
        "message": "ffmpeg not found in PATH"
    }


def _audio_dependency_diagnostics() -> Dict[str, Any]:
    """Return deterministic decode dependency diagnostics for current runtime."""
    ffmpeg_probe = _resolve_ffmpeg_path()
    return {
        "python_executable": _safe_text_value(getattr(sys, "executable", None)),
        "numpy_available": np is not None,
        "soundfile_available": sf is not None,
        "pydub_available": AudioSegment is not None,
        "scipy_available": sp_signal is not None,
        "pyloudnorm_available": pyln is not None,
        "ffmpeg_available": bool(ffmpeg_probe.get("available")),
        "ffmpeg_path": ffmpeg_probe.get("path"),
        "ffmpeg_resolution_source": ffmpeg_probe.get("source"),
        "ffmpeg_attempted_paths": ffmpeg_probe.get("attempted_paths", []),
    }


def _decode_source_to_wav(
    source_path: str,
    input_format: str,
    stat_size: int,
    stat_mtime: float
) -> Tuple[str, Optional[str]]:
    """Return analysis WAV path, decoding source with ffmpeg when needed."""
    if input_format == "wav":
        return source_path, None

    ffmpeg_probe = _resolve_ffmpeg_path()
    ffmpeg_path = ffmpeg_probe.get("path")
    if not ffmpeg_probe.get("available") or not isinstance(ffmpeg_path, str):
        attempted_paths = ffmpeg_probe.get("attempted_paths", [])
        attempted_text = ", ".join([str(path) for path in attempted_paths]) if isinstance(attempted_paths, list) else "unknown"
        probe_message = ffmpeg_probe.get("message") or "ffmpeg executable unavailable"
        raise LiveCaptureError(
            "FFMPEG_NOT_FOUND",
            "ffmpeg required to analyze mp3/aif/flac. "
            "Set ABLETON_MCP_FFMPEG_PATH or install ffmpeg on PATH. "
            "Attempted: {0}. Detail: {1}".format(attempted_text, probe_message)
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
            canonical_root = _canonicalize_browser_root_name(root_name) or root_name
            if canonical_root in seen_roots:
                continue
            seen_roots.add(canonical_root)
            roots.append(canonical_root)
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
            include_presets=normalized_scan_params["include_presets"],
            audio_only=False,
            include_max_for_live_audio=True,
            response_mode="full",
            offset=0,
            limit=50000,
            use_cache=True,
            force_refresh=force_refresh,
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


def _decode_audio_samples(file_path: str) -> Tuple[Any, int, int, str, Dict[str, Any]]:
    """Decode audio and return NxC samples, sample rate, channels, backend, and decode diagnostics."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension not in _SUPPORTED_AUDIO_EXTENSIONS:
        raise SourceAnalysisError(
            "decode_failed",
            f"Unsupported audio format '{extension}'. Supported: {sorted(_SUPPORTED_AUDIO_EXTENSIONS)}"
        )
    if np is None:
        diagnostics = _audio_dependency_diagnostics()
        raise SourceAnalysisError(
            "unsupported_decode_backend",
            "numpy is not available in this Python runtime "
            f"({diagnostics.get('python_executable')})"
        )

    decode_errors: List[str] = []
    attempted_backends: List[str] = []

    base_pipeline = {
        "python_executable": _safe_text_value(getattr(sys, "executable", None)),
        "extension": extension,
        "dependencies": {
            "numpy_available": np is not None,
            "soundfile_available": sf is not None,
            "pydub_available": AudioSegment is not None,
            "scipy_available": sp_signal is not None,
            "pyloudnorm_available": pyln is not None,
        },
        "ffmpeg_probe": _resolve_ffmpeg_path(),
    }

    if sf is not None:
        attempted_backends.append("soundfile")
        try:
            decoded, sample_rate = sf.read(file_path, always_2d=True, dtype="float32")
            if decoded.size == 0:
                raise SourceAnalysisError("decode_failed", "Decoded audio is empty")
            channels = int(decoded.shape[1])
            pipeline = dict(base_pipeline)
            pipeline["attempted_backends"] = list(attempted_backends)
            pipeline["decode"] = {
                "backend": "soundfile",
                "fallback_used": False
            }
            return decoded.astype(np.float32), int(sample_rate), channels, "soundfile", pipeline
        except SourceAnalysisError:
            raise
        except Exception as exc:
            decode_errors.append(f"soundfile: {str(exc)}")

    # WAV fallback path without external decode dependencies.
    if extension == ".wav":
        attempted_backends.append("stdlib_wav")
        try:
            with wave.open(file_path, "rb") as wf:
                channels = int(wf.getnchannels())
                sample_rate = int(wf.getframerate())
                sample_width = int(wf.getsampwidth())
                frame_count = int(wf.getnframes())
                if frame_count <= 0 or channels <= 0:
                    raise SourceAnalysisError("decode_failed", "Decoded audio is empty")
                raw = wf.readframes(frame_count)

            if sample_width == 1:
                pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                pcm = (pcm - 128.0) / 128.0
            elif sample_width == 2:
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 3:
                byte_array = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                signed = (
                    byte_array[:, 0].astype(np.int32)
                    | (byte_array[:, 1].astype(np.int32) << 8)
                    | (byte_array[:, 2].astype(np.int32) << 16)
                )
                sign_mask = signed & 0x800000
                signed = signed - (sign_mask << 1)
                pcm = signed.astype(np.float32) / 8388608.0
            elif sample_width == 4:
                pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise SourceAnalysisError(
                    "decode_failed",
                    f"Unsupported WAV sample width: {sample_width} bytes"
                )

            if channels > 1:
                pcm = pcm.reshape((-1, channels))
            else:
                pcm = pcm.reshape((-1, 1))

            if pcm.size == 0:
                raise SourceAnalysisError("decode_failed", "Decoded audio is empty")

            pipeline = dict(base_pipeline)
            pipeline["attempted_backends"] = list(attempted_backends)
            pipeline["decode"] = {
                "backend": "stdlib_wav",
                "fallback_used": sf is None
            }
            return pcm.astype(np.float32), int(sample_rate), channels, "stdlib_wav", pipeline
        except SourceAnalysisError:
            raise
        except Exception as exc:
            decode_errors.append(f"stdlib_wav: {str(exc)}")

    if AudioSegment is not None:
        attempted_backends.append("pydub")
        try:
            ffmpeg_probe = _resolve_ffmpeg_path()
            if ffmpeg_probe.get("available") and isinstance(ffmpeg_probe.get("path"), str):
                try:
                    AudioSegment.converter = str(ffmpeg_probe.get("path"))
                except Exception:
                    pass
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
            pipeline = dict(base_pipeline)
            pipeline["attempted_backends"] = list(attempted_backends)
            pipeline["decode"] = {
                "backend": "pydub",
                "fallback_used": sf is None
            }
            return normalized, sample_rate, channels, "pydub", pipeline
        except SourceAnalysisError:
            raise
        except Exception as exc:
            decode_errors.append(f"pydub: {str(exc)}")

    # Last-resort fallback: force-decode through ffmpeg to WAV and parse via stdlib.
    ffmpeg_probe = _resolve_ffmpeg_path()
    if ffmpeg_probe.get("available") and isinstance(ffmpeg_probe.get("path"), str):
        attempted_backends.append("ffmpeg_wav_decode")
        ffmpeg_tmp_path = os.path.join(
            _ensure_decoded_audio_tmp_dir(),
            "{0}.wav".format(hashlib.sha256(
                "{0}|{1}|{2}".format(file_path, os.path.getmtime(file_path), os.path.getsize(file_path)).encode("utf-8")
            ).hexdigest())
        )
        cmd = [
            str(ffmpeg_probe.get("path")),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            file_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            ffmpeg_tmp_path,
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120.0
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip() or "Unknown ffmpeg decode failure"
                decode_errors.append("ffmpeg_wav_decode: {0}".format(stderr))
            else:
                with wave.open(ffmpeg_tmp_path, "rb") as wf:
                    channels = int(wf.getnchannels())
                    sample_rate = int(wf.getframerate())
                    sample_width = int(wf.getsampwidth())
                    frame_count = int(wf.getnframes())
                    raw = wf.readframes(frame_count)

                if frame_count <= 0 or channels <= 0:
                    raise SourceAnalysisError("decode_failed", "Decoded audio is empty")

                if sample_width == 2:
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 1:
                    pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                    pcm = (pcm - 128.0) / 128.0
                elif sample_width == 3:
                    byte_array = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                    signed = (
                        byte_array[:, 0].astype(np.int32)
                        | (byte_array[:, 1].astype(np.int32) << 8)
                        | (byte_array[:, 2].astype(np.int32) << 16)
                    )
                    sign_mask = signed & 0x800000
                    signed = signed - (sign_mask << 1)
                    pcm = signed.astype(np.float32) / 8388608.0
                elif sample_width == 4:
                    pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    raise SourceAnalysisError(
                        "decode_failed",
                        "Unsupported ffmpeg-decoded sample width: {0} bytes".format(sample_width)
                    )

                if channels > 1:
                    pcm = pcm.reshape((-1, channels))
                else:
                    pcm = pcm.reshape((-1, 1))

                pipeline = dict(base_pipeline)
                pipeline["attempted_backends"] = list(attempted_backends)
                pipeline["decode"] = {
                    "backend": "ffmpeg_wav_decode",
                    "fallback_used": True
                }
                pipeline["ffmpeg_decode_path"] = ffmpeg_tmp_path
                return pcm.astype(np.float32), int(sample_rate), int(channels), "ffmpeg_wav_decode", pipeline
        except SourceAnalysisError:
            raise
        except Exception as exc:
            decode_errors.append("ffmpeg_wav_decode: {0}".format(str(exc)))

    message = "Failed to decode audio"
    if decode_errors:
        message += f" ({'; '.join(decode_errors)})"
    diagnostics = _audio_dependency_diagnostics()
    message += (
        " | python={0} numpy={1} soundfile={2} pydub={3} ffmpeg={4}".format(
            diagnostics.get("python_executable"),
            diagnostics.get("numpy_available"),
            diagnostics.get("soundfile_available"),
            diagnostics.get("pydub_available"),
            diagnostics.get("ffmpeg_path"),
        )
    )
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

    samples, sample_rate, channels, decode_backend, decode_pipeline = _decode_audio_samples(file_path)
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

    result = {
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
    if isinstance(decode_pipeline, dict):
        result["analysis_pipeline"] = decode_pipeline
    return result


def _series_percentile(values: List[float], pct: float) -> Optional[float]:
    """Compute percentile from Python list using numpy when available."""
    if np is None or not values:
        return None
    try:
        return float(np.percentile(np.asarray(values, dtype=np.float64), pct))
    except Exception:
        return None


def _window_loudness_series(
    samples: Any,
    sample_rate: int,
    meter: Any,
    window_sec: float,
    hop_sec: float
) -> List[Dict[str, Any]]:
    """Compute sliding-window loudness series using pyloudnorm meter."""
    if meter is None or pyln is None or np is None:
        return []

    window_frames = max(1, int(round(float(window_sec) * float(sample_rate))))
    hop_frames = max(1, int(round(float(hop_sec) * float(sample_rate))))
    total_frames = int(samples.shape[0])

    rows: List[Dict[str, Any]] = []
    if total_frames <= 0:
        return rows

    for start in range(0, max(total_frames - window_frames + 1, 1), hop_frames):
        end = min(total_frames, start + window_frames)
        segment = samples[start:end]
        if segment.shape[0] <= 0:
            continue
        # pyloudnorm is unstable on tiny windows; skip clearly too-short tails.
        if segment.shape[0] < max(4, int(0.25 * sample_rate)):
            continue
        try:
            if segment.shape[1] == 1:
                loudness = float(meter.integrated_loudness(segment[:, 0].astype(np.float64)))
            else:
                loudness = float(meter.integrated_loudness(segment.astype(np.float64)))
        except Exception:
            continue
        if math.isnan(loudness) or math.isinf(loudness):
            continue
        rows.append({
            "start_sec": round(float(start) / float(sample_rate), 6),
            "duration_sec": round(float(end - start) / float(sample_rate), 6),
            "lufs": round(loudness, 3)
        })
        if end >= total_frames:
            break

    return rows


def _stereo_correlation_series(
    samples: Any,
    sample_rate: int,
    window_sec: float
) -> List[Dict[str, Any]]:
    """Compute stereo correlation over fixed windows."""
    if np is None or samples is None or int(samples.shape[0]) <= 0:
        return []
    if int(samples.shape[1]) < 2:
        return []

    lr = samples[:, :2].astype(np.float32)
    window_frames = max(1, int(round(float(window_sec) * float(sample_rate))))
    rows: List[Dict[str, Any]] = []

    for start in range(0, int(lr.shape[0]), window_frames):
        end = min(int(lr.shape[0]), start + window_frames)
        segment = lr[start:end]
        if int(segment.shape[0]) < 2:
            continue
        left = segment[:, 0].astype(np.float64)
        right = segment[:, 1].astype(np.float64)
        left_std = float(np.std(left))
        right_std = float(np.std(right))
        corr = 0.0
        if left_std > 1e-12 and right_std > 1e-12:
            try:
                corr = float(np.corrcoef(left, right)[0, 1])
            except Exception:
                corr = 0.0
        corr = max(-1.0, min(1.0, corr))
        rows.append({
            "start_sec": round(float(start) / float(sample_rate), 6),
            "duration_sec": round(float(end - start) / float(sample_rate), 6),
            "correlation": round(corr, 4)
        })
    return rows


def _build_mastering_summary(
    lufs_integrated: Optional[float],
    true_peak_dbtp: Optional[float],
    correlation_min: Optional[float],
    stereo_width_score: Optional[float],
    clipped_sample_count: int,
    inter_sample_peak_risk: bool
) -> str:
    """Build deterministic mastering summary text."""
    parts: List[str] = []
    if lufs_integrated is not None:
        parts.append(f"Integrated loudness {lufs_integrated:.1f} LUFS")
    if true_peak_dbtp is not None:
        parts.append(f"true peak {true_peak_dbtp:+.1f} dBTP")
    if correlation_min is not None:
        parts.append(f"min stereo corr {correlation_min:.2f}")
    if stereo_width_score is not None:
        parts.append(f"width score {stereo_width_score:.0f}/100")
    if clipped_sample_count > 0:
        parts.append(f"{clipped_sample_count} clipped samples")
    if inter_sample_peak_risk:
        parts.append("inter-sample peak risk")
    if not parts:
        parts.append("Mastering metrics unavailable")
    summary = ", ".join(parts)
    if not summary.endswith("."):
        summary += "."
    return summary


def _extract_top_peak_events(
    samples: Any,
    sample_rate: int,
    sample_peak: float,
    max_events: int = 20
) -> Tuple[List[Dict[str, Any]], float]:
    """Extract compact near-peak/clipped contiguous ranges with timestamps."""
    if np is None or samples is None or int(samples.size) == 0:
        return [], 0.0

    frame_peak = np.max(np.abs(samples), axis=1)
    if frame_peak.size == 0:
        return [], 0.0

    base_threshold = float(sample_peak) * 0.98
    clip_or_peak_threshold = min(float(_MASTERING_CLIP_THRESHOLD), float(sample_peak))
    near_peak_threshold = float(max(base_threshold, clip_or_peak_threshold))
    if near_peak_threshold <= 0.0:
        near_peak_threshold = float(sample_peak)

    active_indices = np.where(frame_peak >= near_peak_threshold)[0]
    if active_indices.size == 0:
        return [], near_peak_threshold

    ranges: List[Tuple[int, int]] = []
    start = int(active_indices[0])
    end = int(active_indices[0])
    for idx in active_indices[1:]:
        idx = int(idx)
        if idx == end + 1:
            end = idx
            continue
        ranges.append((start, end))
        start = idx
        end = idx
    ranges.append((start, end))

    events: List[Dict[str, Any]] = []
    for start_frame, end_frame in ranges:
        segment = samples[start_frame:end_frame + 1, :]
        if int(segment.size) == 0:
            continue
        segment_abs = np.abs(segment)
        peak_offset = np.unravel_index(int(np.argmax(segment_abs)), segment_abs.shape)
        peak_value = float(segment_abs[peak_offset])
        peak_channel = int(peak_offset[1])
        peak_frame = int(start_frame + int(peak_offset[0]))
        events.append({
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_sec": round(float(start_frame) / float(sample_rate), 6),
            "end_sec": round(float(end_frame) / float(sample_rate), 6),
            "duration_sec": round(float(max(1, end_frame - start_frame + 1)) / float(sample_rate), 6),
            "peak_frame": peak_frame,
            "peak_sec": round(float(peak_frame) / float(sample_rate), 6),
            "peak": round(peak_value, 6),
            "peak_dbfs": round(_safe_db(peak_value), 3),
            "peak_channel": peak_channel,
            "is_clipped": bool(peak_value >= float(_MASTERING_CLIP_THRESHOLD)),
            "sample_count": int(max(1, end_frame - start_frame + 1)),
        })

    events.sort(key=lambda row: (-float(row.get("peak", 0.0)), int(row.get("start_frame", 0))))
    if max_events > 0 and len(events) > max_events:
        events = events[:max_events]
    return events, near_peak_threshold


def _analyze_mastering_source(
    file_path: str,
    stat_size: int,
    stat_mtime: float,
    window_sec: float = 1.0,
    short_term_sec: float = 3.0,
    momentary_sec: float = 0.4,
    true_peak_threshold_dbtp: float = -1.0
) -> Dict[str, Any]:
    """Compute mastering-oriented analysis metrics for a decoded audio file."""
    if np is None:
        raise SourceAnalysisError("unsupported_decode_backend", "numpy is not available")
    if window_sec <= 0.0:
        raise SourceAnalysisError("invalid_window_sec", "window_sec must be > 0")
    if short_term_sec <= 0.0 or momentary_sec <= 0.0:
        raise SourceAnalysisError("invalid_window_sec", "short_term_sec and momentary_sec must be > 0")

    samples, sample_rate, channels, decode_backend, decode_pipeline = _decode_audio_samples(file_path)
    if samples.size == 0:
        raise SourceAnalysisError("decode_failed", "Decoded audio is empty")

    notes: List[str] = [f"decode_backend={decode_backend}"]
    total_frames = int(samples.shape[0])
    duration_sec = float(total_frames) / float(sample_rate) if sample_rate > 0 else 0.0

    # Use first two channels for stereo metrics when multichannel content is provided.
    stereo_view = samples[:, :2] if channels >= 2 else None
    if channels > 2:
        notes.append("stereo_metrics_use_first_two_channels")

    mono = samples.mean(axis=1).astype(np.float32)
    sample_peak = float(np.max(np.abs(samples)))
    sample_peak_dbfs = _safe_db(sample_peak)
    abs_samples = np.abs(samples)
    peak_flat_index = int(np.argmax(abs_samples))
    peak_sample_frame, peak_sample_channel = np.unravel_index(peak_flat_index, abs_samples.shape)
    peak_sample_frame = int(peak_sample_frame)
    peak_sample_channel = int(peak_sample_channel)
    peak_sample_time_sec = float(peak_sample_frame) / float(sample_rate) if sample_rate > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    rms_dbfs = _safe_db(rms)
    crest_factor_db = float(sample_peak_dbfs - rms_dbfs)

    true_peak = sample_peak
    true_peak_dbtp = sample_peak_dbfs
    if sp_signal is not None:
        try:
            channel_true_peaks: List[float] = []
            for channel_index in range(samples.shape[1]):
                oversampled = sp_signal.resample_poly(samples[:, channel_index], up=4, down=1)
                channel_true_peaks.append(float(np.max(np.abs(oversampled))))
            if channel_true_peaks:
                true_peak = max(channel_true_peaks)
                true_peak_dbtp = _safe_db(true_peak)
        except Exception as exc:
            notes.append(f"true_peak_fallback_sample_peak ({str(exc)})")
    else:
        notes.append("true_peak_fallback_sample_peak (scipy unavailable)")

    meter = None
    lufs_integrated: Optional[float] = None
    lufs_short_term_series: List[Dict[str, Any]] = []
    lufs_momentary_series: List[Dict[str, Any]] = []
    loudness_range_lra: Optional[float] = None

    if pyln is not None:
        try:
            meter = pyln.Meter(sample_rate)
            if channels == 1:
                lufs_integrated = float(meter.integrated_loudness(mono.astype(np.float64)))
            else:
                lufs_integrated = float(meter.integrated_loudness(samples.astype(np.float64)))
            if math.isnan(lufs_integrated) or math.isinf(lufs_integrated):
                lufs_integrated = None
        except Exception as exc:
            notes.append(f"lufs_unavailable ({str(exc)})")
            meter = None
            lufs_integrated = None
    else:
        notes.append("lufs_unavailable (pyloudnorm unavailable)")

    if meter is not None:
        lufs_short_term_series = _window_loudness_series(
            samples=samples if channels > 1 else mono.reshape((-1, 1)),
            sample_rate=sample_rate,
            meter=meter,
            window_sec=float(short_term_sec),
            hop_sec=float(window_sec)
        )
        lufs_momentary_series = _window_loudness_series(
            samples=samples if channels > 1 else mono.reshape((-1, 1)),
            sample_rate=sample_rate,
            meter=meter,
            window_sec=float(momentary_sec),
            hop_sec=float(window_sec)
        )
        try:
            if hasattr(meter, "loudness_range"):
                if channels == 1:
                    loudness_range_lra = float(meter.loudness_range(mono.astype(np.float64)))
                else:
                    loudness_range_lra = float(meter.loudness_range(samples.astype(np.float64)))
        except Exception as exc:
            notes.append(f"lra_unavailable ({str(exc)})")

    if loudness_range_lra is None and lufs_short_term_series:
        short_vals = [float(row["lufs"]) for row in lufs_short_term_series if isinstance(row.get("lufs"), (int, float))]
        p10 = _series_percentile(short_vals, 10.0)
        p95 = _series_percentile(short_vals, 95.0)
        if p10 is not None and p95 is not None:
            loudness_range_lra = float(max(0.0, p95 - p10))
            notes.append("lra_approximated_from_short_term_percentiles")

    stereo_correlation = _stereo_correlation_series(samples, sample_rate, window_sec=float(window_sec))
    correlation_values = [float(row["correlation"]) for row in stereo_correlation if isinstance(row.get("correlation"), (int, float))]
    correlation_min = min(correlation_values) if correlation_values else None
    correlation_avg = (sum(correlation_values) / len(correlation_values)) if correlation_values else None

    mid_side_energy_ratio = None
    mid_side_energy_ratio_db = None
    stereo_width_score = None
    mono_foldown_peak_delta_db = None
    mono_foldown_rms_delta_db = None
    dc_offset_l = None
    dc_offset_r = None
    if stereo_view is not None and int(stereo_view.shape[1]) >= 2:
        left = stereo_view[:, 0].astype(np.float64)
        right = stereo_view[:, 1].astype(np.float64)
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        mid_rms = float(np.sqrt(np.mean(np.square(mid))))
        side_rms = float(np.sqrt(np.mean(np.square(side))))
        mid_side_energy_ratio = float(side_rms / max(mid_rms, 1e-12))
        mid_side_energy_ratio_db = float(_safe_db(mid_side_energy_ratio))

        channel_peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))))
        mono_peak = float(np.max(np.abs(mid)))
        mono_foldown_peak_delta_db = float(_safe_db(mono_peak) - _safe_db(channel_peak))

        channel_rms_avg = (
            float(np.sqrt(np.mean(np.square(left)))) + float(np.sqrt(np.mean(np.square(right))))
        ) / 2.0
        mono_foldown_rms_delta_db = float(_safe_db(float(np.sqrt(np.mean(np.square(mid))))) - _safe_db(channel_rms_avg))

        dc_offset_l = float(np.mean(left))
        dc_offset_r = float(np.mean(right))

        corr_basis = 0.0 if correlation_avg is None else float(correlation_avg)
        width_linear = max(0.0, min(1.0, mid_side_energy_ratio / 1.0))
        score = (0.55 * ((1.0 - corr_basis) / 2.0) + 0.45 * width_linear) * 100.0
        stereo_width_score = float(max(0.0, min(100.0, score)))
    elif channels == 1:
        notes.append("stereo_metrics_unavailable_mono_source")
    else:
        notes.append("stereo_metrics_unavailable")

    clipped_sample_count = int(np.sum(np.abs(samples) >= float(_MASTERING_CLIP_THRESHOLD)))
    inter_sample_peak_risk = bool(
        isinstance(true_peak_dbtp, (int, float)) and true_peak_dbtp > float(true_peak_threshold_dbtp)
    )
    top_peak_events, near_peak_threshold = _extract_top_peak_events(
        samples=samples,
        sample_rate=sample_rate,
        sample_peak=sample_peak,
        max_events=20
    )

    max_samples = int(sample_rate * 120)
    analysis_signal = mono[:max_samples] if mono.shape[0] > max_samples else mono
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

    plr_db = None
    if isinstance(lufs_integrated, (int, float)):
        plr_db = float(true_peak_dbtp - lufs_integrated)

    summary = _build_mastering_summary(
        lufs_integrated=lufs_integrated,
        true_peak_dbtp=true_peak_dbtp,
        correlation_min=correlation_min,
        stereo_width_score=stereo_width_score,
        clipped_sample_count=clipped_sample_count,
        inter_sample_peak_risk=inter_sample_peak_risk
    )

    analysis_pipeline: Dict[str, Any] = {}
    if isinstance(decode_pipeline, dict):
        analysis_pipeline = dict(decode_pipeline)
    analysis_pipeline["mastering"] = {
        "clip_threshold": float(_MASTERING_CLIP_THRESHOLD),
        "near_peak_threshold": round(float(near_peak_threshold), 6),
        "true_peak_threshold_dbtp": float(true_peak_threshold_dbtp),
    }

    return {
        "ok": True,
        "file_path": file_path,
        "file_exists": True,
        "stat_size": int(stat_size),
        "stat_mtime": float(stat_mtime),
        "analyzed_at": _utc_now_iso(),
        "duration_sec": round(duration_sec, 4),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "window_sec": round(float(window_sec), 6),
        "short_term_window_sec": round(float(short_term_sec), 6),
        "momentary_window_sec": round(float(momentary_sec), 6),
        "sample_peak": round(sample_peak, 6),
        "sample_peak_dbfs": round(sample_peak_dbfs, 3),
        "peak_sample_frame": int(peak_sample_frame),
        "peak_sample_time_sec": round(float(peak_sample_time_sec), 6),
        "peak_sample_channel": int(peak_sample_channel),
        "true_peak": round(float(true_peak), 6),
        "true_peak_dbtp": round(float(true_peak_dbtp), 3),
        "rms": round(rms, 6),
        "rms_dbfs": round(rms_dbfs, 3),
        "crest_factor_db": round(float(crest_factor_db), 3),
        "lufs_integrated": None if lufs_integrated is None else round(float(lufs_integrated), 3),
        "loudness_range_lra": None if loudness_range_lra is None else round(float(loudness_range_lra), 3),
        "lufs_short_term_series": lufs_short_term_series,
        "lufs_momentary_series": lufs_momentary_series,
        "stereo_correlation_series": stereo_correlation,
        "correlation_min": None if correlation_min is None else round(float(correlation_min), 4),
        "correlation_avg": None if correlation_avg is None else round(float(correlation_avg), 4),
        "mid_side_energy_ratio": None if mid_side_energy_ratio is None else round(float(mid_side_energy_ratio), 4),
        "mid_side_energy_ratio_db": None if mid_side_energy_ratio_db is None else round(float(mid_side_energy_ratio_db), 3),
        "stereo_width_score": None if stereo_width_score is None else round(float(stereo_width_score), 2),
        "mono_foldown_peak_delta_db": None if mono_foldown_peak_delta_db is None else round(float(mono_foldown_peak_delta_db), 3),
        "mono_foldown_rms_delta_db": None if mono_foldown_rms_delta_db is None else round(float(mono_foldown_rms_delta_db), 3),
        "dc_offset_l": None if dc_offset_l is None else round(float(dc_offset_l), 8),
        "dc_offset_r": None if dc_offset_r is None else round(float(dc_offset_r), 8),
        "clipped_sample_count": clipped_sample_count,
        "near_peak_threshold": round(float(near_peak_threshold), 6),
        "top_peak_events": top_peak_events,
        "inter_sample_peak_risk": inter_sample_peak_risk,
        "true_peak_threshold_dbtp": round(float(true_peak_threshold_dbtp), 3),
        "plr_db": None if plr_db is None else round(float(plr_db), 3),
        "band_energies_db": band_energies_db,
        "resonant_peaks_hz": resonant_peaks_hz,
        "summary": summary,
        "analysis_notes": notes,
        "analysis_pipeline": analysis_pipeline
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


def _canonicalize_browser_root_name(raw_name: Any, available_roots: Optional[List[str]] = None) -> Optional[str]:
    """Resolve human/token root labels to canonical display names."""
    raw_text = _safe_text_value(raw_name)
    if not raw_text:
        return None

    normalized = _normalize_browser_token(raw_text)
    if not normalized:
        return None

    alias_hit = _BROWSER_ROOT_ALIAS_MAP.get(normalized)
    if alias_hit:
        return alias_hit

    candidate_pool = list(_KNOWN_BROWSER_ROOTS)
    if isinstance(available_roots, list):
        for value in available_roots:
            text = _safe_text_value(value)
            if text:
                candidate_pool.append(text)

    for candidate in candidate_pool:
        candidate_text = _safe_text_value(candidate)
        if not candidate_text:
            continue
        if _normalize_browser_token(candidate_text) == normalized:
            return candidate_text

    return None


def _root_name_to_browser_token(root_name: Any) -> Optional[str]:
    """Convert canonical root display name to backend path token."""
    canonical = _canonicalize_browser_root_name(root_name)
    if not canonical:
        return None
    if canonical in _BROWSER_ROOT_TO_TOKEN:
        return _BROWSER_ROOT_TO_TOKEN[canonical]
    return _normalize_browser_token(canonical)


def _canonicalize_browser_path(path: str) -> str:
    """Canonicalize browser root segment while preserving child path segments."""
    if not isinstance(path, str):
        return path
    parts = [part for part in path.split("/") if part]
    if not parts:
        return path
    root_token = _root_name_to_browser_token(parts[0])
    if root_token:
        parts[0] = root_token
    return "/".join(parts)


def _inventory_item_is_audio_relevant(item: Dict[str, Any], include_max_for_live_audio: bool) -> bool:
    """Filter inventory entries to audio-effects-focused rows."""
    path_parts = item.get("path")
    if not isinstance(path_parts, list) or not path_parts:
        return False

    root = _safe_text_value(path_parts[0])
    if root in {"Audio Effects", "Plugins"}:
        return True

    if root != "Max for Live":
        return False

    if not include_max_for_live_audio:
        return False

    if len(path_parts) < 2:
        return False
    category = _normalize_browser_token(path_parts[1])
    return category == "max_audio_effect"


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


def _safe_text_value(value: Any) -> Optional[str]:
    """Return a trimmed string or None."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if value is None:
        return None
    try:
        stripped = str(value).strip()
        return stripped or None
    except Exception:
        return None


def _safe_int_value(value: Any) -> Optional[int]:
    """Return int when coercion is safe."""
    try:
        return int(value)
    except Exception:
        return None


def _safe_float_value(value: Any) -> Optional[float]:
    """Return float when coercion is safe and finite."""
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _normalize_topology_devices(devices: Any) -> List[Dict[str, Any]]:
    """Normalize device-chain rows for topology payloads."""
    rows: List[Dict[str, Any]] = []
    if not isinstance(devices, list):
        return rows

    for device in devices:
        if not isinstance(device, dict):
            continue
        device_index = _safe_int_value(device.get("device_index"))
        row = {
            "device_index": device_index,
            "name": device.get("name"),
            "class_name": device.get("class_name"),
            "is_plugin": device.get("is_plugin"),
            "plugin_format": device.get("plugin_format"),
            "vendor": device.get("vendor"),
            "parameter_count": device.get("parameter_count")
        }
        if isinstance(device.get("parameters"), list):
            row["parameters"] = device.get("parameters")
        rows.append(row)
    return rows


def _normalize_topology_track_row(row: Any, scope: str) -> Optional[Dict[str, Any]]:
    """Normalize one topology row for tracks/returns."""
    if not isinstance(row, dict):
        return None
    index = _safe_int_value(row.get("index"))
    if index is None:
        return None

    mixer_raw = row.get("mixer", {})
    mixer = mixer_raw if isinstance(mixer_raw, dict) else {}

    normalized = {
        "scope": scope,
        "index": index,
        "name": row.get("name"),
        "mixer": {
            "volume": mixer.get("volume"),
            "panning": mixer.get("panning"),
            "mute": mixer.get("mute"),
            "solo": mixer.get("solo")
        },
        "routing": row.get("routing") if isinstance(row.get("routing"), dict) else {
            "input_type": None,
            "input_channel": None,
            "output_type": None,
            "output_channel": None,
        },
        "devices": _normalize_topology_devices(row.get("devices", []))
    }

    if scope == "track":
        normalized["track_kind"] = row.get("track_kind")
        normalized["is_group_track"] = bool(row.get("is_group_track"))
        normalized["group_track_index"] = _safe_int_value(row.get("group_track_index"))
        normalized["mixer"]["arm"] = mixer.get("arm")
        sends_out = []
        sends = row.get("sends", [])
        if isinstance(sends, list):
            for send in sends:
                if not isinstance(send, dict):
                    continue
                sends_out.append({
                    "send_index": _safe_int_value(send.get("send_index")),
                    "target_return_index": _safe_int_value(send.get("target_return_index")),
                    "name": send.get("name"),
                    "value": send.get("value"),
                    "min": send.get("min"),
                    "max": send.get("max"),
                    "is_enabled": send.get("is_enabled"),
                    "automation_state": send.get("automation_state")
                })
        normalized["sends"] = sends_out
    return normalized


def _normalize_mix_topology_payload(payload: Any) -> Dict[str, Any]:
    """Normalize backend mix topology response into a stable schema."""
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "error": "invalid_response",
            "message": "Backend returned non-dict topology response",
            "session": {},
            "tracks": [],
            "returns": [],
            "master": None,
            "edges": [],
            "warnings": ["invalid_response"]
        }

    session_raw = payload.get("session", {})
    session = session_raw if isinstance(session_raw, dict) else {}
    tracks = []
    returns = []

    for row in payload.get("tracks", []):
        normalized = _normalize_topology_track_row(row, scope="track")
        if normalized is not None:
            tracks.append(normalized)

    for row in payload.get("returns", []):
        normalized = _normalize_topology_track_row(row, scope="return")
        if normalized is not None:
            returns.append(normalized)

    master_payload = payload.get("master")
    master = None
    if isinstance(master_payload, dict):
        master_mixer = master_payload.get("mixer", {})
        master = {
            "scope": "master",
            "name": master_payload.get("name") or "Master",
            "mixer": master_mixer if isinstance(master_mixer, dict) else {"volume": None, "panning": None},
            "devices": _normalize_topology_devices(master_payload.get("devices", []))
        }

    edges = [row for row in payload.get("edges", []) if isinstance(row, dict)]
    warnings = [str(row) for row in payload.get("warnings", [])] if isinstance(payload.get("warnings"), list) else []

    normalized = {
        "ok": bool(payload.get("ok", True)),
        "session": {
            "tempo": session.get("tempo"),
            "signature_numerator": session.get("signature_numerator"),
            "signature_denominator": session.get("signature_denominator"),
            "track_count": session.get("track_count", len(tracks)),
            "return_track_count": session.get("return_track_count", len(returns))
        },
        "tracks": sorted(tracks, key=lambda item: int(item["index"])),
        "returns": sorted(returns, key=lambda item: int(item["index"])),
        "master": master,
        "edges": edges,
        "warnings": warnings
    }

    if payload.get("error"):
        normalized["error"] = payload.get("error")
    if payload.get("message"):
        normalized["message"] = payload.get("message")
    return normalized


def _derive_send_matrix_from_topology(topology: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact send matrix view from a normalized topology payload."""
    rows: List[Dict[str, Any]] = []
    active_routes: List[Dict[str, Any]] = []
    return_lookup: Dict[int, Optional[str]] = {}

    for return_row in topology.get("returns", []):
        if isinstance(return_row, dict):
            idx = _safe_int_value(return_row.get("index"))
            if idx is not None:
                return_lookup[idx] = _safe_text_value(return_row.get("name"))

    for track in topology.get("tracks", []):
        if not isinstance(track, dict):
            continue
        track_index = _safe_int_value(track.get("index"))
        sends = track.get("sends", [])
        if track_index is None or not isinstance(sends, list):
            continue

        send_rows = []
        for send in sends:
            if not isinstance(send, dict):
                continue
            send_index = _safe_int_value(send.get("send_index"))
            target_return_index = _safe_int_value(send.get("target_return_index"))
            amount = _safe_float_value(send.get("value"))
            send_row = {
                "send_index": send_index,
                "target_return_index": target_return_index,
                "target_return_name": (
                    return_lookup.get(target_return_index)
                    if isinstance(target_return_index, int)
                    else None
                ),
                "name": send.get("name"),
                "value": amount,
                "is_enabled": send.get("is_enabled")
            }
            send_rows.append(send_row)

            if (
                isinstance(send_index, int)
                and isinstance(target_return_index, int)
                and isinstance(amount, float)
                and amount > 0.0
            ):
                active_routes.append({
                    "track_index": track_index,
                    "track_name": track.get("name"),
                    "send_index": send_index,
                    "target_return_index": target_return_index,
                    "target_return_name": return_lookup.get(target_return_index),
                    "amount": amount
                })

        rows.append({
            "track_index": track_index,
            "track_name": track.get("name"),
            "sends": send_rows
        })

    return {
        "ok": bool(topology.get("ok")),
        "track_count": len(rows),
        "return_count": len(return_lookup),
        "rows": rows,
        "active_routes": active_routes,
        "warnings": topology.get("warnings", []),
    }


def _mix_context_tags_file_path() -> str:
    """Return project-relative mix context tags file path."""
    ensure_dirs_exist()
    return os.path.join(get_analysis_dir(), _MIX_CONTEXT_TAGS_FILE_NAME)


def _empty_mix_context_tags_payload() -> Dict[str, Any]:
    """Return stable empty mix-context tag payload."""
    return {
        "ok": True,
        "schema_version": _MIX_CONTEXT_TAGS_SCHEMA_VERSION,
        "updated_at": None,
        "track_roles": {},
        "return_roles": {},
        "master_roles": [],
        "metadata": {
            "source": "manual",
            "inference_version": 1
        }
    }


def _normalize_role_token(value: Any) -> Optional[str]:
    """Normalize role names into snake_case-ish tokens."""
    token = _safe_text_value(value)
    if token is None:
        return None
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", token.strip().lower()).strip("_")
    return normalized or None


def _normalize_role_list(values: Any) -> List[str]:
    """Normalize and dedupe a list of roles preserving order."""
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for value in values:
        token = _normalize_role_token(value)
        if token is None or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _normalize_role_map(payload: Any, prefix: str) -> Dict[str, List[str]]:
    """Normalize role maps keyed by IDs like track:0 / return:0."""
    out: Dict[str, List[str]] = {}
    if not isinstance(payload, dict):
        return out
    for raw_key, raw_roles in payload.items():
        key = _safe_text_value(raw_key)
        if key is None:
            continue
        if ":" not in key:
            key = f"{prefix}:{key}"
        roles = _normalize_role_list(raw_roles)
        if roles:
            out[key] = roles
    return out


def _normalize_mix_context_tags_input(payload: Any) -> Dict[str, Any]:
    """Normalize persisted or user-provided tag payload into stable schema."""
    base = _empty_mix_context_tags_payload()
    if not isinstance(payload, dict):
        return base

    metadata = payload.get("metadata", {})
    metadata_out = dict(base["metadata"])
    if isinstance(metadata, dict):
        source_value = _safe_text_value(metadata.get("source"))
        if source_value:
            metadata_out["source"] = source_value
        inference_version = _safe_int_value(metadata.get("inference_version"))
        if inference_version is not None:
            metadata_out["inference_version"] = inference_version

    master_roles_raw = payload.get("master_roles")
    if not isinstance(master_roles_raw, list) and isinstance(payload.get("master"), list):
        master_roles_raw = payload.get("master")

    normalized = {
        "ok": True,
        "schema_version": _MIX_CONTEXT_TAGS_SCHEMA_VERSION,
        "updated_at": payload.get("updated_at"),
        "track_roles": _normalize_role_map(payload.get("track_roles"), "track"),
        "return_roles": _normalize_role_map(payload.get("return_roles"), "return"),
        "master_roles": _normalize_role_list(master_roles_raw),
        "metadata": metadata_out
    }
    return normalized


def _load_mix_context_tags_payload() -> Dict[str, Any]:
    """Load mix-context tags from analysis dir if present."""
    path = _mix_context_tags_file_path()
    if not os.path.exists(path):
        payload = _empty_mix_context_tags_payload()
        payload["tags_file_path"] = path
        return payload
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        payload = _normalize_mix_context_tags_input(raw)
        payload["tags_file_path"] = path
        return payload
    except Exception as exc:
        payload = _empty_mix_context_tags_payload()
        payload["tags_file_path"] = path
        payload["warnings"] = [f"tags_load_failed:{str(exc)}"]
        return payload


def _write_mix_context_tags_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Persist normalized mix-context tags to disk."""
    normalized = _normalize_mix_context_tags_input(payload)
    normalized["updated_at"] = _utc_now_iso()
    path = _mix_context_tags_file_path()
    ensure_dirs_exist()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)
    normalized["tags_file_path"] = path
    return normalized


def _merge_role_maps_prefer_explicit(
    explicit_map: Dict[str, List[str]],
    inferred_map: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Merge inferred roles under explicit roles without overwriting explicit assignments."""
    merged: Dict[str, List[str]] = {}
    keys = sorted(set(explicit_map.keys()) | set(inferred_map.keys()))
    for key in keys:
        if key in explicit_map and explicit_map.get(key):
            merged[key] = list(explicit_map[key])
        elif key in inferred_map and inferred_map.get(key):
            merged[key] = list(inferred_map[key])
    return merged


def _normalized_name_key(value: Any) -> str:
    """Normalize names for heuristic matching."""
    text = _safe_text_value(value) or ""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _infer_roles_from_name(name: str, scope: str, is_group_track: bool = False) -> Tuple[List[str], Dict[str, float]]:
    """Infer semantic roles from track/return/master names."""
    roles: List[str] = []
    confidence: Dict[str, float] = {}
    key = _normalized_name_key(name)
    if not key:
        return roles, confidence

    def add(role: str, score: float) -> None:
        if role not in roles:
            roles.append(role)
        current = confidence.get(role, 0.0)
        confidence[role] = max(current, float(score))

    # Track-oriented roles
    if scope == "track":
        if ("lead vocal" in key) or ("lead vox" in key) or (key.startswith("vox lead")) or ("leadvox" in key.replace(" ", "")):
            add("lead_vocal", 0.98)
        elif "vox" in key or "vocal" in key:
            if "back" in key or "bgv" in key or "harmony" in key or "double" in key:
                add("backing_vocal", 0.92)
            else:
                add("vocal", 0.75)
        if "kick" in key:
            add("kick", 0.96)
        if "snare" in key or re.search(r"\bsn\b", key):
            add("snare", 0.9)
        if "bass" in key and "brass" not in key:
            add("bass", 0.9)

        if is_group_track:
            if "drum" in key or "perc" in key:
                add("drums_bus", 0.95)
            if "gtr" in key or "guitar" in key:
                add("guitars_bus", 0.95)
            if "vox" in key or "vocal" in key:
                add("vocals_bus", 0.95)
            if "key" in key or "synth" in key or "piano" in key:
                add("keys_bus", 0.9)
            if not any(role.endswith("_bus") for role in roles):
                add("submix_bus", 0.55)

    if scope == "return":
        if "reverb" in key or "verb" in key:
            add("main_reverb", 0.95)
        if "delay" in key or "echo" in key:
            add("main_delay", 0.95)
        if ("parallel" in key or "ny" in key) and ("comp" in key or "compress" in key):
            add("parallel_compression", 0.97)
        elif "comp" in key or "compress" in key:
            add("compression_return", 0.75)
        if "dist" in key or "satur" in key:
            add("distortion_return", 0.7)

    if scope == "master":
        add("mix_bus", 0.99)
    return roles, confidence


def _topology_row_id(scope: str, row: Dict[str, Any]) -> str:
    """Build stable ID for topology rows."""
    if scope == "master":
        return "master"
    index_value = _safe_int_value(row.get("index"))
    if index_value is None:
        return f"{scope}:unknown"
    return f"{scope}:{index_value}"


def _stage_status(has_all: bool, has_partial: bool) -> str:
    """Return stage status enum."""
    if has_all:
        return "sufficient"
    if has_partial:
        return "partial"
    return "missing"


def _collect_device_parameters_for_track(
    ctx: Context,
    track_index: int,
    device_index: int,
    page_size: int = 256
) -> Dict[str, Any]:
    """Fetch all paged device parameters including automation state."""
    first_page = get_device_parameters(ctx, track_index=track_index, device_index=device_index, offset=0, limit=page_size)
    if not isinstance(first_page, dict):
        return {"ok": False, "error": "invalid_response", "parameters": []}
    if first_page.get("ok") is False:
        return first_page

    total_parameters = _safe_int_value(first_page.get("total_parameters"))
    if total_parameters is None:
        total_parameters = len(first_page.get("parameters", [])) if isinstance(first_page.get("parameters"), list) else 0

    params_out: List[Dict[str, Any]] = []
    if isinstance(first_page.get("parameters"), list):
        params_out.extend([row for row in first_page["parameters"] if isinstance(row, dict)])

    next_offset = len(params_out)
    while next_offset < total_parameters:
        page = get_device_parameters(ctx, track_index=track_index, device_index=device_index, offset=next_offset, limit=page_size)
        if not isinstance(page, dict) or page.get("ok") is False:
            break
        rows = page.get("parameters", [])
        if not isinstance(rows, list) or not rows:
            break
        params_out.extend([row for row in rows if isinstance(row, dict)])
        next_offset += len(rows)

    # Deduplicate by parameter_index
    by_index: Dict[int, Dict[str, Any]] = {}
    for row in params_out:
        parameter_index = _safe_int_value(row.get("parameter_index"))
        if parameter_index is None:
            continue
        by_index[parameter_index] = row

    return {
        "ok": True,
        "track_index": track_index,
        "device_index": device_index,
        "parameters": [by_index[idx] for idx in sorted(by_index.keys())],
        "parameter_count": len(by_index)
    }


def _sanitize_locator_rows(locators: Any) -> List[Dict[str, Any]]:
    """Normalize locator rows for stable API payloads."""
    rows: List[Dict[str, Any]] = []
    if not isinstance(locators, list):
        return rows

    for idx, row in enumerate(locators):
        if not isinstance(row, dict):
            continue
        time_beats = _safe_float_value(row.get("time_beats"))
        if time_beats is None:
            continue
        row_index = _safe_int_value(row.get("index"))
        if row_index is None:
            row_index = idx
        name = _safe_text_value(row.get("name")) or f"Locator {int(row_index) + 1}"
        rows.append(
            {
                "index": int(row_index),
                "name": name,
                "time_beats": float(time_beats),
            }
        )

    rows.sort(key=lambda item: (float(item.get("time_beats", 0.0)), int(item.get("index", 0))))
    for idx, row in enumerate(rows):
        row["index"] = int(idx)
    return rows


def _locator_neighbors(
    locators: List[Dict[str, Any]],
    current_song_time_beats: Optional[float],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return previous/next locator for current song time."""
    if not isinstance(locators, list) or current_song_time_beats is None:
        return None, None

    previous_locator = None
    next_locator = None
    for row in locators:
        time_value = _safe_float_value(row.get("time_beats"))
        if time_value is None:
            continue
        if time_value <= current_song_time_beats:
            previous_locator = row
        if time_value > current_song_time_beats:
            next_locator = row
            break
    return previous_locator, next_locator


def _inject_session_locator_summary(session_payload: Dict[str, Any], locator_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Inject additive locator summary fields into session payload."""
    if not isinstance(session_payload, dict):
        return session_payload
    out = dict(session_payload)

    locators = _sanitize_locator_rows(locator_payload.get("locators"))
    out["has_time_locators"] = bool(len(locators) > 0)
    out["time_locator_count"] = int(len(locators))
    out["time_locators_preview"] = list(locators[:5])
    return out


# Core Tool endpoints

@mcp.tool()
def get_session_info(ctx: Context) -> str:
    """Get detailed information about the current Ableton session"""
    try:
        ableton = get_ableton_connection()
        result = ableton.send_command("get_session_info")
        if isinstance(result, dict):
            has_locator_fields = (
                "has_time_locators" in result
                and "time_locator_count" in result
                and "time_locators_preview" in result
            )
            if not has_locator_fields:
                locator_payload = get_time_locators(ctx, include_als_fallback=True)
                if isinstance(locator_payload, dict):
                    result = _inject_session_locator_summary(result, locator_payload)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting session info from Ableton: {str(e)}")
        return f"Error getting session info: {str(e)}"


@mcp.tool()
def get_time_locators(
    ctx: Context,
    als_file_path: Optional[str] = None,
    include_als_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Return timeline locator/cue-point visibility from Live runtime with optional .als fallback.
    """
    _ = ctx

    warnings: List[str] = []
    runtime_payload: Optional[Dict[str, Any]] = None
    runtime_error: Optional[str] = None
    current_song_time_beats: Optional[float] = None

    try:
        ableton = get_ableton_connection()
        runtime_candidate = ableton.send_command("get_time_locators")
        if isinstance(runtime_candidate, dict):
            runtime_payload = dict(runtime_candidate)
            current_song_time_beats = _safe_float_value(
                runtime_payload.get("current_song_time_beats")
                if runtime_payload.get("current_song_time_beats") is not None
                else runtime_payload.get("current_song_time_sec")
            )
    except Exception as exc:
        runtime_error = str(exc)

    if runtime_payload is not None:
        runtime_locators = _sanitize_locator_rows(runtime_payload.get("locators"))
        runtime_supported = bool(runtime_payload.get("supported")) and bool(runtime_payload.get("ok"))
        if runtime_supported:
            previous_locator = runtime_payload.get("previous_locator")
            next_locator = runtime_payload.get("next_locator")
            if current_song_time_beats is not None:
                inferred_previous, inferred_next = _locator_neighbors(runtime_locators, current_song_time_beats)
                if inferred_previous is not None:
                    previous_locator = inferred_previous
                if inferred_next is not None:
                    next_locator = inferred_next

            payload: Dict[str, Any] = {
                "ok": True,
                "supported": True,
                "source": "runtime",
                "locators": runtime_locators,
                "locator_count": int(len(runtime_locators)),
                "warnings": list(runtime_payload.get("warnings") or []),
            }
            if current_song_time_beats is not None:
                payload["current_song_time_beats"] = float(current_song_time_beats)
            if isinstance(previous_locator, dict):
                payload["previous_locator"] = previous_locator
            if isinstance(next_locator, dict):
                payload["next_locator"] = next_locator
            return payload

        runtime_reason = _safe_text_value(runtime_payload.get("reason")) or "runtime_locator_unavailable"
        warnings.append(runtime_reason)

    if runtime_error:
        warnings.append("runtime_locator_error:{0}".format(runtime_error))

    if not include_als_fallback:
        return {
            "ok": True,
            "supported": False,
            "source": "runtime",
            "reason": warnings[-1] if warnings else "runtime_locator_unavailable",
            "locators": [],
            "locator_count": 0,
            "warnings": warnings,
        }

    project_root = get_project_root()
    fallback = read_time_locators_from_project_als(project_root=project_root, als_file_path=als_file_path)
    if not isinstance(fallback, dict):
        return {
            "ok": False,
            "error": "locator_fallback_failed",
            "message": "ALS locator fallback returned invalid payload",
            "warnings": warnings,
        }

    fallback_payload = dict(fallback)
    fallback_payload["warnings"] = list(warnings) + list(fallback_payload.get("warnings") or [])
    if fallback_payload.get("ok") is not True:
        return fallback_payload

    fallback_locators = _sanitize_locator_rows(fallback_payload.get("locators"))
    fallback_payload["locators"] = fallback_locators
    fallback_payload["locator_count"] = int(len(fallback_locators))
    if fallback_payload.get("source") is None:
        fallback_payload["source"] = "als_file"
    if current_song_time_beats is not None:
        fallback_payload["current_song_time_beats"] = float(current_song_time_beats)
        previous_locator, next_locator = _locator_neighbors(fallback_locators, current_song_time_beats)
        fallback_payload["previous_locator"] = previous_locator
        fallback_payload["next_locator"] = next_locator
    return fallback_payload

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
        canonical_path = _canonicalize_browser_path(path)
        result = ableton.send_command("get_browser_items_at_path", {
            "path": canonical_path
        })
        if isinstance(result, dict):
            result = dict(result)
            result["requested_path"] = path
            result["resolved_path"] = canonical_path
        
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
def get_mix_topology(
    ctx: Context,
    include_device_chains: bool = True,
    include_device_parameters: bool = False
) -> Dict[str, Any]:
    """
    Return normalized routing/bus/send topology for tracks, returns, and master.

    Parameters:
    - include_device_chains: include device-chain metadata for tracks/returns/master
    - include_device_parameters: request full parameter payloads (best-effort; may be partial on older backends)
    """
    _ = ctx
    try:
        ableton = get_ableton_connection()
        raw = ableton.send_command("get_mix_topology", {
            "include_device_chains": bool(include_device_chains),
            "include_device_parameters": bool(include_device_parameters)
        })
        normalized = _normalize_mix_topology_payload(raw)
        if include_device_parameters:
            warnings = list(normalized.get("warnings", []))
            for scope_key in ("tracks", "returns"):
                rows = normalized.get(scope_key, [])
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    for device in row.get("devices", []):
                        if not isinstance(device, dict):
                            continue
                        if "parameters" not in device:
                            device["parameters"] = []
                    # If the backend provided no inline parameter payloads, note that once.
                if rows and all(
                    isinstance(row, dict)
                    and isinstance(row.get("devices"), list)
                    and all(isinstance(dev, dict) and isinstance(dev.get("parameters"), list) and len(dev["parameters"]) == 0 for dev in row["devices"])
                    for row in rows
                ):
                    warnings.append(f"{scope_key}_device_parameters_unavailable_or_empty")
            master_row = normalized.get("master")
            if isinstance(master_row, dict):
                for device in master_row.get("devices", []):
                    if isinstance(device, dict) and "parameters" not in device:
                        device["parameters"] = []
            normalized["warnings"] = warnings
        return normalized
    except Exception as e:
        logger.error(f"Error getting mix topology: {str(e)}")
        return {
            "ok": False,
            "error": "get_mix_topology_failed",
            "message": str(e),
            "session": {},
            "tracks": [],
            "returns": [],
            "master": None,
            "edges": [],
            "warnings": []
        }


@mcp.tool()
def get_send_matrix(ctx: Context) -> Dict[str, Any]:
    """Return a compact send matrix derived from get_mix_topology."""
    topology = get_mix_topology(ctx, include_device_chains=False, include_device_parameters=False)
    if not isinstance(topology, dict):
        return {
            "ok": False,
            "error": "invalid_topology_response",
            "message": "get_mix_topology returned invalid payload",
            "rows": [],
            "active_routes": []
        }
    if topology.get("ok") is not True:
        return topology
    return _derive_send_matrix_from_topology(topology)


@mcp.tool()
def get_return_tracks_info(ctx: Context, include_device_chains: bool = True) -> Dict[str, Any]:
    """
    Return return-track mixer/routing/device-chain metadata.

    Parameters:
    - include_device_chains: include return-track device chains
    """
    topology = get_mix_topology(
        ctx,
        include_device_chains=bool(include_device_chains),
        include_device_parameters=False
    )
    if not isinstance(topology, dict):
        return {
            "ok": False,
            "error": "invalid_topology_response",
            "message": "get_mix_topology returned invalid payload",
            "returns": []
        }
    if topology.get("ok") is not True:
        return topology
    returns = topology.get("returns", [])
    return {
        "ok": True,
        "return_count": len(returns) if isinstance(returns, list) else 0,
        "returns": returns if isinstance(returns, list) else [],
        "warnings": topology.get("warnings", [])
    }


@mcp.tool()
def get_master_track_device_chain(ctx: Context) -> Dict[str, Any]:
    """Return master-track device-chain metadata and mixer state."""
    topology = get_mix_topology(ctx, include_device_chains=True, include_device_parameters=False)
    if not isinstance(topology, dict):
        return {
            "ok": False,
            "error": "invalid_topology_response",
            "message": "get_mix_topology returned invalid payload"
        }
    if topology.get("ok") is not True:
        return topology

    master = topology.get("master")
    if not isinstance(master, dict):
        return {
            "ok": False,
            "error": "master_unavailable",
            "message": "Master track topology was not available",
            "warnings": topology.get("warnings", [])
        }

    devices = master.get("devices", [])
    return {
        "ok": True,
        "scope": "master",
        "track_name": master.get("name") or "Master",
        "mixer": master.get("mixer", {}),
        "device_count": len(devices) if isinstance(devices, list) else 0,
        "devices": devices if isinstance(devices, list) else [],
        "warnings": topology.get("warnings", [])
    }


@mcp.tool()
def get_track_device_chain(
    ctx: Context,
    track_index: int,
    include_nested: bool = False,
    max_depth: int = 4
) -> Dict[str, Any]:
    """
    Get ordered device chain metadata for a track.

    Parameters:
    - track_index: 0-based track index
    - include_nested: include rack-contained nested devices
    - max_depth: nested rack traversal depth (when include_nested=True)
    """
    try:
        max_depth_value = int(max_depth)
    except Exception:
        max_depth_value = 4

    try:
        ableton = get_ableton_connection()
        result = ableton.send_command(
            "get_track_devices",
            {
                "track_index": track_index,
                "include_nested": bool(include_nested),
                "max_depth": max_depth_value,
            },
        )
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
                "include_nested": bool(include_nested),
                "max_depth": max_depth_value,
                "backend_result_type": str(type(result))
            }
        }
    except Exception as e:
        logger.error(f"Error getting track device chain: {str(e)}")
        return {
            "ok": False,
            "error": "get_track_device_chain_failed",
            "message": str(e),
            "track_index": track_index,
            "include_nested": bool(include_nested),
            "max_depth": max_depth_value,
        }


@mcp.tool()
def get_device_parameters(
    ctx: Context,
    track_index: int,
    device_index: int,
    offset: int = 0,
    limit: int = 64,
    device_path: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Get paged device parameters for a track device.

    Parameters:
    - track_index: 0-based track index
    - device_index: 0-based device index on the track
    - offset: parameter offset
    - limit: page size
    - device_path: optional nested path [top_device, chain, device, ...]
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

    device_path_value = None
    if device_path is not None:
        if not isinstance(device_path, list) or not device_path:
            return {
                "ok": False,
                "error": "invalid_device_path",
                "message": "device_path must be a non-empty list of integers when provided",
                "track_index": track_index,
                "device_index": device_index,
                "device_path": device_path,
            }
        parsed_path: List[int] = []
        for value in device_path:
            parsed_value = _safe_int_value(value)
            if parsed_value is None:
                return {
                    "ok": False,
                    "error": "invalid_device_path",
                    "message": "device_path entries must be integers",
                    "track_index": track_index,
                    "device_index": device_index,
                    "device_path": device_path,
                }
            parsed_path.append(int(parsed_value))
        if parsed_path[0] != int(device_index):
            return {
                "ok": False,
                "error": "device_path_mismatch",
                "message": "device_path[0] must match device_index",
                "track_index": track_index,
                "device_index": int(device_index),
                "device_path": parsed_path,
            }
        if len(parsed_path) % 2 == 0:
            return {
                "ok": False,
                "error": "invalid_device_path",
                "message": "device_path must have odd length [top_device, chain, device, ...]",
                "track_index": track_index,
                "device_index": int(device_index),
                "device_path": parsed_path,
            }
        device_path_value = parsed_path

    try:
        ableton = get_ableton_connection()
        command_payload = {
            "track_index": track_index,
            "device_index": device_index,
            "offset": offset_value,
            "limit": limit_value
        }
        if isinstance(device_path_value, list):
            command_payload["device_path"] = device_path_value
        result = ableton.send_command("get_device_parameters", command_payload)
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
            "device_path": device_path_value,
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
            "limit": limit_value,
            "device_path": device_path_value
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
    include_presets: bool = False,
    audio_only: bool = False,
    include_max_for_live_audio: bool = True,
    response_mode: str = "compact",
    offset: int = 0,
    limit: int = 200,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Enumerate loadable devices/effects/instruments from Ableton's browser.

    Parameters:
    - roots: Optional browser roots to scan. Defaults to ["Audio Effects", "Plugins"].
    - max_depth: Maximum recursive folder depth
    - max_items_per_folder: Hard cap per folder scan
    - include_presets: Include clearly preset/rack items when True
    - audio_only: Include only audio-relevant items (Audio Effects, Plugins, optional Max for Live audio effects)
    - include_max_for_live_audio: Include Max for Live > Max Audio Effect rows when audio_only=True
    - response_mode: "compact" (paged, default) or "full"
    - offset: Pagination offset used when response_mode="compact"
    - limit: Pagination page size used when response_mode="compact"
    - use_cache: Reuse runtime cache for identical scan params
    - force_refresh: Ignore cache and rescan browser
    """
    _ = ctx
    try:
        if roots is not None and not isinstance(roots, list):
            return {
                "ok": False,
                "error": "invalid_roots_type",
                "message": "roots must be a list of strings",
                "roots_type": str(type(roots)),
            }

        response_mode_value = (_safe_text_value(response_mode) or "compact").lower()
        if response_mode_value not in {"compact", "full"}:
            response_mode_value = "compact"

        try:
            offset_value = max(0, int(offset))
        except Exception:
            offset_value = 0
        try:
            limit_value = max(1, int(limit))
        except Exception:
            limit_value = 200

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
        if roots is None:
            requested_roots_raw = ["Audio Effects", "Plugins"]
        else:
            requested_roots_raw = [
                value.strip()
                for value in roots
                if isinstance(value, str) and value.strip()
            ]

        # Canonicalize requested roots while preserving not-found diagnostics by original input token.
        requested_roots: List[str] = []
        roots_not_found: List[str] = []
        seen_requested = set()
        for raw_value in requested_roots_raw:
            canonical = _canonicalize_browser_root_name(raw_value, available_roots=available_roots)
            if not canonical:
                roots_not_found.append(raw_value)
                continue
            if canonical in seen_requested:
                continue
            seen_requested.add(canonical)
            requested_roots.append(canonical)

        # Build root lookup map with canonical display names and tokenized path candidates.
        root_lookup: Dict[str, Dict[str, Any]] = {}
        for entry in root_entries:
            display = _safe_text_value(entry.get("display_name"))
            canonical_display = _canonicalize_browser_root_name(display, available_roots=available_roots) or display
            if not canonical_display:
                continue
            target = root_lookup.get(canonical_display)
            if not isinstance(target, dict):
                target = {"display_name": canonical_display, "path_candidates": []}
                root_lookup[canonical_display] = target
            path_candidates = entry.get("path_candidates", [])
            if isinstance(path_candidates, list):
                for candidate in path_candidates:
                    candidate_text = _safe_text_value(candidate)
                    if candidate_text and candidate_text not in target["path_candidates"]:
                        target["path_candidates"].append(candidate_text)
            root_token = _root_name_to_browser_token(canonical_display)
            for candidate in [canonical_display, root_token, _normalize_browser_token(canonical_display)]:
                candidate_text = _safe_text_value(candidate)
                if candidate_text and candidate_text not in target["path_candidates"]:
                    target["path_candidates"].append(candidate_text)

        for known_root in _KNOWN_BROWSER_ROOTS:
            if known_root in root_lookup:
                continue
            root_lookup[known_root] = {
                "display_name": known_root,
                "path_candidates": [
                    value
                    for value in [
                        known_root,
                        _root_name_to_browser_token(known_root),
                        _normalize_browser_token(known_root),
                    ]
                    if _safe_text_value(value)
                ],
            }

        unresolved_requested = [name for name in requested_roots if name not in root_lookup]
        for unresolved in unresolved_requested:
            if unresolved not in roots_not_found:
                roots_not_found.append(unresolved)
        valid_roots = [name for name in requested_roots if name in root_lookup]

        scan_params = _normalize_inventory_scan_params(
            {
                "roots": valid_roots,
                "max_depth": depth_limit,
                "max_items_per_folder": items_per_folder_limit,
                "include_presets": bool(include_presets),
            }
        )
        runtime_cache_key = _scan_params_key(scan_params)
        cached_scan_payload = _DEVICE_INVENTORY_RUNTIME_CACHE.get(runtime_cache_key)
        cache_hit = bool(
            use_cache
            and not force_refresh
            and isinstance(cached_scan_payload, dict)
            and isinstance(cached_scan_payload.get("devices"), list)
        )

        scan_payload: Optional[Dict[str, Any]] = copy.deepcopy(cached_scan_payload) if cache_hit else None

        if scan_payload is None:
            folder_calls = 0
            scanned_roots: List[str] = []
            devices: List[Dict[str, Any]] = []
            folders_truncated: List[Dict[str, Any]] = []
            errors: List[Dict[str, Any]] = []
            truncated = False
            device_dedupe_keys = set()
            visited_folders = set()

            def fetch_folder(path_key_parts: List[str]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
                nonlocal folder_calls
                if folder_calls >= max_folder_calls:
                    return None, "max_folder_calls"
                folder_calls += 1

                path_string = "/".join(path_key_parts)
                path_string = _canonicalize_browser_path(path_string)
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
                    candidate = _safe_text_value(value)
                    if not candidate or candidate in seen_candidates:
                        continue
                    seen_candidates.add(candidate)
                    candidate_path = _canonicalize_browser_path(candidate)
                    items, error = fetch_folder([candidate_path])
                    if error is None:
                        return items, candidate_path, None
                    if error == "max_folder_calls":
                        return None, None, error
                    last_error = error

                return None, None, last_error or "root_not_resolvable"

            def traverse_folder(
                path_key_parts: List[str],
                path_display_parts: List[str],
                depth: int,
                prefetched_items: Optional[List[Dict[str, Any]]] = None,
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
                            folders_truncated.append({"path": path_display_parts, "reason": "max_folder_calls"})
                            return
                        errors.append({"path": path_display_parts, "error": error})
                        return
                    items = fetched_items or []

                if len(items) > items_per_folder_limit:
                    folders_truncated.append({"path": path_display_parts, "reason": "max_items_per_folder"})
                    items = items[:items_per_folder_limit]

                for item in items:
                    if truncated:
                        break

                    item_name = _safe_text_value(item.get("name"))
                    if not item_name:
                        continue
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
                                devices.append(
                                    {
                                        "name": item_name,
                                        "path": item_display_path,
                                        "item_id": item_id,
                                        "item_type": _infer_inventory_item_type(item, item_name),
                                    }
                                )
                                if len(devices) >= overall_item_cap:
                                    truncated = True
                                    break

                    if is_folder and depth < depth_limit and not truncated:
                        traverse_folder(
                            path_key_parts=path_key_parts + [item_name],
                            path_display_parts=item_display_path,
                            depth=depth + 1,
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
                        folders_truncated.append({"path": [root_name], "reason": "max_folder_calls"})
                        break
                    errors.append({"path": [root_name], "error": root_error})
                    continue

                if not isinstance(resolved_root_path, str) or not resolved_root_path.strip():
                    errors.append({"path": [root_name], "error": "root_not_resolvable"})
                    continue

                scanned_roots.append(root_name)
                traverse_folder(
                    path_key_parts=[resolved_root_path],
                    path_display_parts=[root_name],
                    depth=0,
                    prefetched_items=root_items,
                )

            scan_payload = {
                "requested_roots": list(requested_roots),
                "available_roots": list(available_roots),
                "scanned_roots": scanned_roots,
                "roots_not_found": list(roots_not_found),
                "max_depth": depth_limit,
                "max_items_per_folder": items_per_folder_limit,
                "include_presets": bool(include_presets),
                "truncated": bool(truncated),
                "folders_truncated": folders_truncated,
                "devices": devices,
                "errors": errors,
            }
            if use_cache:
                _DEVICE_INVENTORY_RUNTIME_CACHE[runtime_cache_key] = copy.deepcopy(scan_payload)

        if not isinstance(scan_payload, dict):
            return {
                "ok": False,
                "error": "device_inventory_failed",
                "message": "inventory scan payload missing",
            }

        devices_all = scan_payload.get("devices", [])
        if not isinstance(devices_all, list):
            devices_all = []
        devices_filtered = [item for item in devices_all if isinstance(item, dict)]

        if bool(audio_only):
            devices_filtered = [
                item for item in devices_filtered
                if _inventory_item_is_audio_relevant(item, include_max_for_live_audio=bool(include_max_for_live_audio))
            ]

        devices_filtered.sort(
            key=lambda row: (
                (_safe_text_value(row.get("name")) or "").lower(),
                _browser_path_key(row.get("path", [])) if isinstance(row.get("path"), list) else "",
            )
        )

        total_devices = len(devices_filtered)
        if response_mode_value == "full":
            paged_devices = devices_filtered
            returned_count = total_devices
            has_more = False
        else:
            start = min(offset_value, total_devices)
            end = min(total_devices, start + limit_value)
            paged_devices = devices_filtered[start:end]
            returned_count = len(paged_devices)
            has_more = bool(end < total_devices)

        return {
            "ok": True,
            "requested_roots": requested_roots_raw,
            "normalized_requested_roots": scan_payload.get("requested_roots", requested_roots),
            "available_roots": scan_payload.get("available_roots", available_roots),
            "scanned_roots": scan_payload.get("scanned_roots", []),
            "roots_not_found": scan_payload.get("roots_not_found", roots_not_found),
            "max_depth": scan_payload.get("max_depth", depth_limit),
            "max_items_per_folder": scan_payload.get("max_items_per_folder", items_per_folder_limit),
            "include_presets": bool(include_presets),
            "audio_only": bool(audio_only),
            "include_max_for_live_audio": bool(include_max_for_live_audio),
            "response_mode": response_mode_value,
            "offset": int(offset_value),
            "limit": int(limit_value),
            "total_devices": int(total_devices),
            "returned_count": int(returned_count),
            "has_more": bool(has_more),
            "cache_hit": bool(cache_hit),
            "cache_key": runtime_cache_key,
            "truncated": bool(scan_payload.get("truncated", False)),
            "folders_truncated": scan_payload.get("folders_truncated", []),
            "devices": paged_devices,
            "errors": scan_payload.get("errors", []),
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
def analyze_mastering_file(
    ctx: Context,
    file_path: str,
    window_sec: float = 1.0
) -> Dict[str, Any]:
    """
    Analyze a file with mastering-oriented loudness, peak, stereo, and spectral metrics.

    Parameters:
    - file_path: absolute or launch-cwd-relative path to WAV/MP3/AIF/FLAC/M4A file
    - window_sec: timeline hop/window size used for correlation and loudness series sampling
    """
    _ = ctx
    try:
        normalized_path = _normalize_source_path(file_path)
        file_exists, stat_size, stat_mtime = _safe_file_stats(normalized_path)
        if not file_exists:
            return {
                "ok": False,
                "error": "file_not_found",
                "message": "Audio file does not exist",
                "file_path": normalized_path
            }
        if stat_size is None or stat_mtime is None:
            return {
                "ok": False,
                "error": "stat_failed",
                "message": "Could not read file metadata",
                "file_path": normalized_path
            }

        return _analyze_mastering_source(
            file_path=normalized_path,
            stat_size=int(stat_size),
            stat_mtime=float(stat_mtime),
            window_sec=float(window_sec)
        )
    except SourceAnalysisError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "message": exc.message,
            "file_path": _normalize_source_path(file_path)
        }
    except Exception as exc:
        logger.error(f"Error analyzing mastering file '{file_path}': {str(exc)}")
        return {
            "ok": False,
            "error": "analyze_mastering_file_failed",
            "message": str(exc),
            "file_path": _normalize_source_path(file_path)
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
    rms_threshold_db: float = _DEFAULT_RMS_THRESHOLD_DBFS,
    analysis_profile: str = "mix"
) -> Dict[str, Any]:
    """
    Analyze all exported WAVs in a manifest. Returns WAIT_FOR_USER_EXPORT if missing.

    Parameters:
    - analysis_profile: "mix" (default) or "mastering"
    """
    try:
        profile = str(analysis_profile or "mix").strip().lower()
        if profile not in {"mix", "mastering"}:
            return {
                "ok": False,
                "error": "invalid_analysis_profile",
                "message": "analysis_profile must be 'mix' or 'mastering'"
            }

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
        lufs_values = []
        true_peak_values = []
        correlation_min_values = []
        items_exceeding_true_peak_threshold: List[str] = []
        items_below_target_loudness: List[str] = []
        items_above_target_loudness: List[str] = []
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

            if profile == "mastering":
                analysis = analyze_mastering_file(
                    ctx=ctx,
                    file_path=wav_path,
                    window_sec=window_sec
                )
            else:
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
                "analysis_json_path": analysis_json_path,
                "analysis_profile": profile
            }

            if isinstance(analysis, dict) and analysis.get("ok"):
                if profile == "mastering":
                    entry["detected_audio"] = True
                    entry["mastering"] = {
                        "lufs_integrated": analysis.get("lufs_integrated"),
                        "true_peak_dbtp": analysis.get("true_peak_dbtp"),
                        "sample_peak_dbfs": analysis.get("sample_peak_dbfs"),
                        "correlation_min": analysis.get("correlation_min"),
                        "stereo_width_score": analysis.get("stereo_width_score"),
                        "inter_sample_peak_risk": analysis.get("inter_sample_peak_risk"),
                        "summary": analysis.get("summary")
                    }
                    peak_db = analysis.get("sample_peak_dbfs")
                    if isinstance(peak_db, (int, float)):
                        peak_values.append(float(peak_db))
                    lufs_value = analysis.get("lufs_integrated")
                    if isinstance(lufs_value, (int, float)):
                        lufs_values.append(float(lufs_value))
                        if float(lufs_value) < float(_DEFAULT_MASTERING_TARGET_LUFS):
                            items_below_target_loudness.append(item_id)
                        elif float(lufs_value) > float(_DEFAULT_MASTERING_TARGET_LUFS):
                            items_above_target_loudness.append(item_id)
                    true_peak = analysis.get("true_peak_dbtp")
                    if isinstance(true_peak, (int, float)):
                        true_peak_values.append(float(true_peak))
                    corr_min = analysis.get("correlation_min")
                    if isinstance(corr_min, (int, float)):
                        correlation_min_values.append(float(corr_min))
                    if analysis.get("inter_sample_peak_risk") is True:
                        items_exceeding_true_peak_threshold.append(item_id)
                else:
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
            "analysis_profile": profile,
            "notes": notes
        }
        if profile == "mastering":
            summary["target_lufs"] = _DEFAULT_MASTERING_TARGET_LUFS
            summary["lufs_integrated_range"] = (
                {
                    "min": min(lufs_values),
                    "max": max(lufs_values)
                }
                if lufs_values else None
            )
            summary["true_peak_max_dbtp"] = max(true_peak_values) if true_peak_values else None
            summary["correlation_min"] = min(correlation_min_values) if correlation_min_values else None
            summary["items_exceeding_true_peak_threshold"] = items_exceeding_true_peak_threshold
            summary["items_below_target_loudness"] = items_below_target_loudness
            summary["items_above_target_loudness"] = items_above_target_loudness

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


def _automation_state_is_active(value: Any) -> Optional[bool]:
    """Best-effort normalization of parameter automation_state values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        try:
            return int(value) != 0
        except Exception:
            return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return None
        if lowered in {"0", "none", "off", "disabled", "not_automated", "no_automation"}:
            return False
        return True
    return None


def _build_mix_stage_readiness(
    topology: Optional[Dict[str, Any]],
    tags_profile: Optional[Dict[str, Any]],
    automation_overview: Optional[Dict[str, Any]],
    source_inventory: Optional[Dict[str, Any]],
    export_analysis: Optional[Dict[str, Any]],
    include_mastering_metrics: bool
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build deterministic stage-readiness rows and missing-data actions."""
    topology_ok = isinstance(topology, dict) and topology.get("ok") is True
    tags_ok = isinstance(tags_profile, dict) and tags_profile.get("ok") is True
    automation_ok = isinstance(automation_overview, dict) and automation_overview.get("ok") is True
    source_ok = isinstance(source_inventory, dict) and source_inventory.get("ok") is True
    export_ok = isinstance(export_analysis, dict) and export_analysis.get("ok") is True
    export_ready = export_ok and export_analysis.get("ready", True) is not False
    export_profile = export_analysis.get("summary", {}).get("analysis_profile") if isinstance(export_analysis, dict) and isinstance(export_analysis.get("summary"), dict) else None

    group_bus_visible = False
    sends_visible = False
    return_visible = False
    master_chain_visible = False
    if topology_ok:
        track_rows = topology.get("tracks", [])
        return_rows = topology.get("returns", [])
        if isinstance(track_rows, list):
            group_bus_visible = any(bool(row.get("is_group_track")) for row in track_rows if isinstance(row, dict))
            sends_visible = any(isinstance(row.get("sends"), list) and len(row.get("sends")) > 0 for row in track_rows if isinstance(row, dict))
        if isinstance(return_rows, list):
            return_visible = len(return_rows) >= 0
        master = topology.get("master")
        master_chain_visible = isinstance(master, dict) and isinstance(master.get("devices"), list)

    automation_supported = False
    if automation_ok:
        automation_supported = bool(automation_overview.get("supported")) or bool(automation_overview.get("tracks_with_device_automation", 0))

    stages: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []

    def add_stage(stage_id: str, has_all: bool, has_partial: bool, data_present: List[str], data_missing: List[str], recommended_next_steps: List[str]) -> None:
        stages.append({
            "stage": stage_id,
            "status": _stage_status(has_all=has_all, has_partial=has_partial),
            "data_present": data_present,
            "data_missing": data_missing,
            "recommended_next_steps": recommended_next_steps
        })

    add_stage(
        "mix_session_prep_inserts_sends",
        has_all=bool(topology_ok and sends_visible and return_visible and master_chain_visible),
        has_partial=bool(topology_ok),
        data_present=[x for x in [
            "mix_topology" if topology_ok else None,
            "send_matrix" if topology_ok and sends_visible else None,
            "return_track_chains" if topology_ok and return_visible else None,
            "master_chain" if topology_ok and master_chain_visible else None
        ] if x],
        data_missing=[x for x in [
            None if topology_ok else "mix_topology",
            None if sends_visible else "send_levels_or_send_slots",
            None if return_visible else "return_track_visibility",
            None if master_chain_visible else "master_track_device_chain"
        ] if x],
        recommended_next_steps=[x for x in [
            "Call get_mix_topology to inspect sends, returns, and master routing." if not topology_ok else None,
            "Call get_send_matrix for a compact send routing view." if topology_ok else None
        ] if x]
    )

    add_stage(
        "mix_stage_1_import_organize",
        has_all=bool(tags_ok and topology_ok and source_ok),
        has_partial=bool(topology_ok or source_ok),
        data_present=[x for x in [
            "track_topology" if topology_ok else None,
            "mix_context_tags" if tags_ok else None,
            "source_inventory" if source_ok else None
        ] if x],
        data_missing=[x for x in [
            None if topology_ok else "track_topology",
            None if tags_ok else "semantic_track_roles",
            None if source_ok else "source_inventory"
        ] if x],
        recommended_next_steps=[x for x in [
            "Call build_mix_context_profile to merge inferred and explicit track roles." if not tags_ok else None,
            "Call index_sources_from_live_set to analyze source clips." if not source_ok else None
        ] if x]
    )

    add_stage(
        "mix_stage_2_gain_staging",
        has_all=bool(source_ok and export_ok and export_ready),
        has_partial=bool(source_ok or export_ok),
        data_present=[x for x in [
            "source_spectral_loudness_summaries" if source_ok else None,
            "export_analysis_mix_metrics" if export_ok and export_profile == "mix" else None
        ] if x],
        data_missing=[x for x in [
            "real_time_metering" if True else None,
            None if (export_ok and export_profile == "mix") else "exported_mix_loudness_windows"
        ] if x],
        recommended_next_steps=[x for x in [
            "Use plan_exports -> check_exports_ready -> analyze_export_job for exported stems/mix windows." if not export_ok else None
        ] if x]
    )

    add_stage(
        "mix_stage_3_fader_mix",
        has_all=bool(topology_ok and tags_ok),
        has_partial=bool(topology_ok),
        data_present=[x for x in [
            "topology" if topology_ok else None,
            "semantic_roles" if tags_ok else None,
            "send_matrix" if topology_ok and sends_visible else None
        ] if x],
        data_missing=[x for x in [
            None if topology_ok else "topology",
            None if tags_ok else "semantic_roles"
        ] if x],
        recommended_next_steps=[x for x in [
            "Save explicit mix context tags for lead vocal/core band tracks." if not tags_ok else None
        ] if x]
    )

    add_stage(
        "mix_stage_4_automation_pass",
        has_all=False,
        has_partial=bool(automation_ok),
        data_present=[x for x in [
            "device_parameter_automation_states" if automation_ok else None
        ] if x],
        data_missing=[x for x in [
            "automation_envelope_points",
            "track_volume_pan_automation_envelopes"
        ] if x],
        recommended_next_steps=[x for x in [
            "Call get_automation_overview and get_track_automation_targets to inspect automation-state coverage." if not automation_ok else "Call get_automation_envelope_points for specific targets; use export-based loudness timelines as a proxy when envelope point access is unavailable."
        ] if x]
    )

    add_stage(
        "mix_stage_5_sub_mix",
        has_all=bool(topology_ok and group_bus_visible and master_chain_visible),
        has_partial=bool(topology_ok),
        data_present=[x for x in [
            "group_bus_membership" if topology_ok and group_bus_visible else None,
            "return_tracks" if topology_ok else None,
            "master_chain" if topology_ok and master_chain_visible else None
        ] if x],
        data_missing=[x for x in [
            None if topology_ok else "mix_topology",
            None if group_bus_visible else "group_track_submix_structure",
            None if master_chain_visible else "master_chain"
        ] if x],
        recommended_next_steps=[x for x in [
            "Inspect get_mix_topology edges to confirm submix bus membership and routing." if topology_ok else None
        ] if x]
    )

    add_stage(
        "mix_stage_6_printing_stereo_and_stems",
        has_all=bool(export_ok and export_ready),
        has_partial=True,
        data_present=[x for x in [
            "export_workflow_tools_available",
            "export_analysis_results" if export_ok else None
        ] if x],
        data_missing=[x for x in [
            None if export_ok else "analyzed_exports"
        ] if x],
        recommended_next_steps=[x for x in [
            "Call plan_exports, export WAVs manually, then check_exports_ready and analyze_export_job." if not export_ok else None
        ] if x]
    )

    mastering_export_ok = export_ok and export_profile == "mastering"
    add_stage(
        "mastering_goals",
        has_all=bool(mastering_export_ok or (include_mastering_metrics and export_ok)),
        has_partial=bool(topology_ok or source_ok or export_ok),
        data_present=[x for x in [
            "master_track_chain" if topology_ok and master_chain_visible else None,
            "mastering_metrics" if mastering_export_ok else None
        ] if x],
        data_missing=[x for x in [
            None if mastering_export_ok else "mastering_metrics_batch_or_single"
        ] if x],
        recommended_next_steps=[x for x in [
            "Run analyze_mastering_file on a printed mix, or analyze_export_job(analysis_profile='mastering')." if not mastering_export_ok else None
        ] if x]
    )

    add_stage(
        "mastering_chain_eq_sat_comp_eq_stereo_limit",
        has_all=bool(topology_ok and master_chain_visible and mastering_export_ok),
        has_partial=bool(topology_ok or mastering_export_ok),
        data_present=[x for x in [
            "master_chain" if topology_ok and master_chain_visible else None,
            "mastering_metrics" if mastering_export_ok else None
        ] if x],
        data_missing=[x for x in [
            None if topology_ok and master_chain_visible else "master_chain_visibility",
            None if mastering_export_ok else "true_peak_lufs_stereo_metrics"
        ] if x],
        recommended_next_steps=[x for x in [
            "Use get_master_track_device_chain to inspect chain order and tools." if not (topology_ok and master_chain_visible) else None,
            "Use analyze_export_job with analysis_profile='mastering' for printed mixes/stems." if not mastering_export_ok else None
        ] if x]
    )

    if not topology_ok:
        actions.append({
            "kind": "tool_call",
            "tool": "get_mix_topology",
            "params": {"include_device_chains": True, "include_device_parameters": False},
            "reason": "Need routing, sends, returns, and master chain visibility."
        })
    if not tags_ok:
        actions.append({
            "kind": "tool_call",
            "tool": "build_mix_context_profile",
            "params": {},
            "reason": "Generate inferred semantic roles and merge with saved tags."
        })
    if not automation_ok:
        actions.append({
            "kind": "tool_call",
            "tool": "get_automation_overview",
            "params": {},
            "reason": "Inspect automation-state coverage for track devices."
        })
    if not source_ok:
        actions.append({
            "kind": "tool_call",
            "tool": "index_sources_from_live_set",
            "params": {},
            "reason": "Analyze source clip files for spectral/loudness summaries."
        })
    if not export_ok:
        actions.append({
            "kind": "workflow",
            "tool": "plan_exports",
            "params": {"job_name": "mix_or_master_prints", "items": [{"target": "mix", "start_bar": 1, "end_bar": 9}]},
            "reason": "Export-based ears workflow is required for print/stem analysis."
        })

    return stages, actions


@mcp.tool()
def infer_mix_context_tags(ctx: Context) -> Dict[str, Any]:
    """
    Infer semantic mix-role tags from topology names (tracks, returns, master).
    Returns confidence scores but does not persist anything.
    """
    topology = get_mix_topology(ctx, include_device_chains=False, include_device_parameters=False)
    if not isinstance(topology, dict):
        return {
            "ok": False,
            "error": "invalid_topology_response",
            "message": "get_mix_topology returned invalid payload"
        }
    if topology.get("ok") is not True:
        return topology

    track_roles: Dict[str, List[str]] = {}
    return_roles: Dict[str, List[str]] = {}
    master_roles: List[str] = []
    confidence: Dict[str, Dict[str, float]] = {}
    reasons: Dict[str, List[Dict[str, Any]]] = {}

    for row in topology.get("tracks", []):
        if not isinstance(row, dict):
            continue
        item_id = _topology_row_id("track", row)
        roles, scores = _infer_roles_from_name(
            name=str(row.get("name") or ""),
            scope="track",
            is_group_track=bool(row.get("is_group_track"))
        )
        if roles:
            track_roles[item_id] = roles
            confidence[item_id] = {role: round(float(scores.get(role, 0.0)), 4) for role in roles}
            reasons[item_id] = [{
                "role": role,
                "confidence": round(float(scores.get(role, 0.0)), 4),
                "basis": "name_heuristic"
            } for role in roles]

    for row in topology.get("returns", []):
        if not isinstance(row, dict):
            continue
        item_id = _topology_row_id("return", row)
        roles, scores = _infer_roles_from_name(
            name=str(row.get("name") or ""),
            scope="return",
            is_group_track=False
        )
        if roles:
            return_roles[item_id] = roles
            confidence[item_id] = {role: round(float(scores.get(role, 0.0)), 4) for role in roles}
            reasons[item_id] = [{
                "role": role,
                "confidence": round(float(scores.get(role, 0.0)), 4),
                "basis": "name_heuristic"
            } for role in roles]

    master_row = topology.get("master")
    if isinstance(master_row, dict):
        roles, scores = _infer_roles_from_name(
            name=str(master_row.get("name") or "Master"),
            scope="master",
            is_group_track=False
        )
        if roles:
            master_roles = roles
            confidence["master"] = {role: round(float(scores.get(role, 0.0)), 4) for role in roles}
            reasons["master"] = [{
                "role": role,
                "confidence": round(float(scores.get(role, 0.0)), 4),
                "basis": "default_master_role"
            } for role in roles]

    return {
        "ok": True,
        "schema_version": _MIX_CONTEXT_TAGS_SCHEMA_VERSION,
        "inferred_at": _utc_now_iso(),
        "track_roles": track_roles,
        "return_roles": return_roles,
        "master_roles": master_roles,
        "confidence": confidence,
        "suggestions": reasons,
        "metadata": {
            "source": "inference",
            "inference_version": 1
        },
        "warnings": topology.get("warnings", [])
    }


@mcp.tool()
def get_mix_context_tags(ctx: Context) -> Dict[str, Any]:
    """Load persisted mix-context semantic tags from the project analysis folder."""
    _ = ctx
    return _load_mix_context_tags_payload()


@mcp.tool()
def save_mix_context_tags(ctx: Context, tags: Dict[str, Any], merge: bool = True) -> Dict[str, Any]:
    """
    Persist mix-context semantic tags to the project analysis folder.

    Parameters:
    - tags: payload with track_roles / return_roles / master_roles
    - merge: when true, preserve existing explicit tags and merge in new keys
    """
    _ = ctx
    incoming = _normalize_mix_context_tags_input(tags)
    if not merge:
        incoming["metadata"]["source"] = "manual"
        return _write_mix_context_tags_payload(incoming)

    existing = _load_mix_context_tags_payload()
    merged = _empty_mix_context_tags_payload()
    merged["track_roles"] = _merge_role_maps_prefer_explicit(
        explicit_map=existing.get("track_roles", {}),
        inferred_map=incoming.get("track_roles", {})
    )
    merged["return_roles"] = _merge_role_maps_prefer_explicit(
        explicit_map=existing.get("return_roles", {}),
        inferred_map=incoming.get("return_roles", {})
    )
    merged["master_roles"] = (
        existing.get("master_roles")
        if isinstance(existing.get("master_roles"), list) and existing.get("master_roles")
        else incoming.get("master_roles", [])
    )
    merged["metadata"] = {
        "source": "manual",
        "inference_version": 1
    }
    return _write_mix_context_tags_payload(merged)


@mcp.tool()
def build_mix_context_profile(ctx: Context) -> Dict[str, Any]:
    """Build merged topology + explicit tags + inferred suggestions profile."""
    topology = get_mix_topology(ctx, include_device_chains=False, include_device_parameters=False)
    if not isinstance(topology, dict):
        return {
            "ok": False,
            "error": "invalid_topology_response",
            "message": "get_mix_topology returned invalid payload"
        }
    if topology.get("ok") is not True:
        return topology

    explicit = get_mix_context_tags(ctx)
    inferred = infer_mix_context_tags(ctx)

    explicit_track_roles = explicit.get("track_roles", {}) if isinstance(explicit, dict) else {}
    explicit_return_roles = explicit.get("return_roles", {}) if isinstance(explicit, dict) else {}
    explicit_master_roles = explicit.get("master_roles", []) if isinstance(explicit, dict) else []
    inferred_track_roles = inferred.get("track_roles", {}) if isinstance(inferred, dict) else {}
    inferred_return_roles = inferred.get("return_roles", {}) if isinstance(inferred, dict) else {}
    inferred_master_roles = inferred.get("master_roles", []) if isinstance(inferred, dict) else []

    merged_track_roles = _merge_role_maps_prefer_explicit(
        explicit_map=explicit_track_roles if isinstance(explicit_track_roles, dict) else {},
        inferred_map=inferred_track_roles if isinstance(inferred_track_roles, dict) else {}
    )
    merged_return_roles = _merge_role_maps_prefer_explicit(
        explicit_map=explicit_return_roles if isinstance(explicit_return_roles, dict) else {},
        inferred_map=inferred_return_roles if isinstance(inferred_return_roles, dict) else {}
    )
    merged_master_roles = (
        explicit_master_roles if isinstance(explicit_master_roles, list) and explicit_master_roles else (
            inferred_master_roles if isinstance(inferred_master_roles, list) else []
        )
    )

    return {
        "ok": True,
        "schema_version": _MIX_CONTEXT_TAGS_SCHEMA_VERSION,
        "topology": topology,
        "explicit_tags": explicit,
        "inference_suggestions": inferred,
        "merged_roles": {
            "track_roles": merged_track_roles,
            "return_roles": merged_return_roles,
            "master_roles": merged_master_roles
        },
        "metadata": {
            "merge_strategy": "explicit_overrides_inference",
            "inference_version": 1
        }
    }


@mcp.tool()
def get_track_automation_targets(ctx: Context, track_index: int) -> Dict[str, Any]:
    """
    Return track-level automation targets (track volume/pan support + device parameters with automation state).

    Parameters:
    - track_index: 0-based track index
    """
    track_payload, track_error = _coerce_json_dict(get_track_info(ctx, track_index))
    if track_payload is None:
        return {
            "ok": False,
            "error": "track_info_unavailable",
            "message": track_error or "Failed to load track info",
            "track_index": track_index
        }

    chain_payload = get_track_device_chain(ctx, track_index)
    if not isinstance(chain_payload, dict) or chain_payload.get("ok") is False:
        return {
            "ok": False,
            "error": "track_device_chain_unavailable",
            "message": (
                chain_payload.get("message") if isinstance(chain_payload, dict) else "Invalid device chain response"
            ),
            "track_index": track_index
        }

    devices = chain_payload.get("devices", [])
    if not isinstance(devices, list):
        devices = []

    device_rows = []
    automated_device_count = 0
    automated_parameter_count = 0
    warnings: List[str] = []

    for device in devices:
        if not isinstance(device, dict):
            continue
        device_index = _safe_int_value(device.get("device_index"))
        if device_index is None:
            continue
        params_payload = _collect_device_parameters_for_track(ctx, track_index=track_index, device_index=device_index)
        parameters = params_payload.get("parameters", []) if isinstance(params_payload, dict) else []
        if not isinstance(parameters, list):
            parameters = []
        if isinstance(params_payload, dict) and params_payload.get("ok") is False:
            warnings.append(f"device_param_fetch_failed:{track_index}:{device_index}")

        parameter_rows = []
        device_has_automation = False
        for row in parameters:
            if not isinstance(row, dict):
                continue
            automation_state = row.get("automation_state")
            automated = _automation_state_is_active(automation_state)
            if automated is True:
                device_has_automation = True
                automated_parameter_count += 1
            parameter_rows.append({
                "parameter_index": row.get("parameter_index"),
                "name": row.get("name"),
                "automation_state": automation_state,
                "automated": automated
            })

        if device_has_automation:
            automated_device_count += 1

        device_rows.append({
            "device_index": device_index,
            "device_name": device.get("name"),
            "class_name": device.get("class_name"),
            "parameter_count": len(parameter_rows),
            "automated_parameter_count": sum(1 for p in parameter_rows if p.get("automated") is True),
            "has_automation": device_has_automation,
            "parameters": parameter_rows
        })

    # Track volume/pan automation envelopes are not exposed yet; keep explicit and stable.
    track_mixer_targets = {
        "volume": {
            "supported": False,
            "automation_state": None,
            "automated": None,
            "reason": "track_mixer_automation_state_not_exposed"
        },
        "panning": {
            "supported": False,
            "automation_state": None,
            "automated": None,
            "reason": "track_mixer_automation_state_not_exposed"
        }
    }

    return {
        "ok": True,
        "track_index": track_index,
        "track_name": track_payload.get("name"),
        "supported": True,
        "track_mixer_targets": track_mixer_targets,
        "devices": device_rows,
        "summary": {
            "device_count": len(device_rows),
            "devices_with_automation": automated_device_count,
            "automated_parameter_count": automated_parameter_count
        },
        "warnings": warnings
    }


def _apply_als_arrangement_automation_fallback(
    base_result: Dict[str, Any],
    track_index: int,
    scope: str,
    mixer_target: str,
    als_file_path: Optional[str],
    send_index: Optional[int],
    device_index: Optional[int],
    parameter_index: Optional[int],
    start_time_beats: Optional[float],
    end_time_beats: Optional[float],
) -> Dict[str, Any]:
    """
    Merge exact arrangement automation points from the saved .als file when Live API access is missing.

    This keeps the public tool stable while adding a stronger fallback for arrangement automation
    breakpoints (track mixer and device parameters).
    """
    if not isinstance(base_result, dict):
        return base_result

    existing_points = base_result.get("points")
    has_point_rows = isinstance(existing_points, list) and len(existing_points) > 0
    point_access_supported = bool(base_result.get("point_access_supported", False))

    # If Live API already returned direct point access rows, do not replace them.
    if point_access_supported and has_point_rows:
        return base_result

    try:
        project_root = get_project_root()
        als_result = read_arrangement_automation_from_project_als(
            project_root=project_root,
            track_index=int(track_index),
            scope=scope,
            mixer_target=mixer_target,
            als_file_path=als_file_path,
            send_index=send_index,
            device_index=device_index,
            parameter_index=parameter_index,
            start_time_beats=start_time_beats,
            end_time_beats=end_time_beats,
        )
    except Exception as exc:
        merged = dict(base_result)
        warnings = list(merged.get("warnings") or [])
        warnings.append("als_fallback_exception")
        merged["warnings"] = warnings
        merged["als_fallback_error"] = str(exc)
        return merged

    if not isinstance(als_result, dict):
        return base_result

    # Preserve unsupported ALS fallback info as diagnostics, but only replace payload if ALS can answer.
    if als_result.get("ok") is not True or als_result.get("supported") is not True:
        merged = dict(base_result)
        warnings = list(merged.get("warnings") or [])
        for warning in list(als_result.get("warnings") or []):
            if isinstance(warning, str) and warning not in warnings:
                warnings.append(warning)
        merged["warnings"] = warnings
        if isinstance(als_result.get("reason"), str):
            merged["als_fallback_reason"] = als_result.get("reason")
        if isinstance(als_result.get("als_file_path"), str):
            merged["als_file_path"] = als_result.get("als_file_path")
        return merged

    def _normalize_name_for_match(value: Any) -> Optional[str]:
        text = _safe_text_value(value)
        if not text:
            return None
        # split camel case and normalize punctuation/spacing
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        text = text.replace("_", " ").replace(".", " ").replace("-", " ")
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text or None

    def _device_name_candidates_from_als_target(als_target_payload: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []
        xml_tag = _safe_text_value(als_target_payload.get("device_xml_tag"))
        name_hint = _safe_text_value(als_target_payload.get("device_name_hint"))
        for value in (name_hint, xml_tag):
            normalized = _normalize_name_for_match(value)
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        # Common internal ALS tags -> Live UI names
        xml_alias_map = {
            "stereogain": ["utility"],
            "eqeight": ["eq eight"],
            "compressor2": ["compressor"],
            "autofilter": ["auto filter"],
            "hybridreverb": ["hybrid reverb"],
            "drumbus": ["drum buss"],
            "gluecompressor": ["glue compressor"],
            "saturator": ["saturator"],
        }
        xml_key = _normalize_name_for_match(xml_tag)
        if xml_key and xml_key in xml_alias_map:
            for alias in xml_alias_map.get(xml_key, []):
                normalized = _normalize_name_for_match(alias)
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        return candidates

    def _parameter_name_candidates_from_als_target(als_target_payload: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []
        for value in (
            als_target_payload.get("parameter_display_name_hint"),
            als_target_payload.get("parameter_name_hint"),
            als_target_payload.get("parameter_xml_tag"),
        ):
            normalized = _normalize_name_for_match(value)
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        xml_tag = _safe_text_value(als_target_payload.get("parameter_xml_tag"))
        xml_norm = _normalize_name_for_match(xml_tag)
        macro_match = re.match(r"^macro controls (\d+)$", xml_norm or "")
        if macro_match:
            idx = _safe_int_value(macro_match.group(1))
            if idx is not None:
                macro_alias = _normalize_name_for_match("Macro {0}".format(int(idx) + 1))
                if macro_alias and macro_alias not in candidates:
                    candidates.append(macro_alias)
        return candidates

    def _verify_device_parameter_mapping(
        live_target_payload: Dict[str, Any],
        als_target_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        live_device = _normalize_name_for_match(live_target_payload.get("device_name"))
        live_param = _normalize_name_for_match(live_target_payload.get("parameter_name"))
        als_device_candidates = _device_name_candidates_from_als_target(als_target_payload)
        als_param_candidates = _parameter_name_candidates_from_als_target(als_target_payload)

        verification = {
            "kind": "device_parameter",
            "live_device_name": _safe_text_value(live_target_payload.get("device_name")),
            "live_parameter_name": _safe_text_value(live_target_payload.get("parameter_name")),
            "als_device_candidates": als_device_candidates,
            "als_parameter_candidates": als_param_candidates,
        }

        # We harden by refusing an ALS merge when parameter mapping is unverifiable or mismatched.
        if not live_param or not als_param_candidates:
            verification["ok"] = False
            verification["reason"] = "device_parameter_mapping_unverifiable"
            return verification
        if live_param not in als_param_candidates:
            verification["ok"] = False
            verification["reason"] = "device_parameter_mapping_mismatch"
            return verification

        # Device names are advisory, but if present and contradictory we also block.
        if live_device and als_device_candidates and live_device not in als_device_candidates:
            verification["ok"] = False
            verification["reason"] = "device_name_mapping_mismatch"
            return verification

        verification["ok"] = True
        verification["reason"] = "verified"
        return verification

    als_target_for_verify = als_result.get("target") if isinstance(als_result.get("target"), dict) else {}
    live_target_for_verify = base_result.get("target") if isinstance(base_result.get("target"), dict) else {}
    if (_safe_text_value(scope) or "").lower() == "device_parameter":
        verification = _verify_device_parameter_mapping(live_target_for_verify, als_target_for_verify)
        if verification.get("ok") is not True:
            merged = dict(base_result)
            warnings = list(merged.get("warnings") or [])
            reason = _safe_text_value(verification.get("reason")) or "device_parameter_mapping_ambiguous"
            if reason not in warnings:
                warnings.append(reason)
            merged["warnings"] = warnings
            merged["als_mapping_verification"] = verification
            merged["als_fallback_reason"] = reason
            if isinstance(als_result.get("als_file_path"), str):
                merged["als_file_path"] = als_result.get("als_file_path")
            return merged

    merged = dict(base_result)
    merged["live_api_supported"] = bool(base_result.get("supported", False)) if "supported" in base_result else None
    if base_result.get("reason") is not None:
        merged["live_api_reason"] = base_result.get("reason")

    # Replace point-related fields with exact arrangement data from .als
    merged["supported"] = True
    merged["envelope_exists"] = als_result.get("envelope_exists")
    merged["point_access_supported"] = bool(als_result.get("point_access_supported", False))
    merged["points"] = list(als_result.get("points") or [])
    merged["source"] = "hybrid"
    merged["point_source"] = "als_arrangement_file"
    merged["arrangement_automation_source"] = "als_file"
    merged["als_fallback_used"] = True
    if isinstance(als_result.get("source_kind"), str):
        merged["arrangement_automation_source_kind"] = als_result.get("source_kind")
    if isinstance(als_result.get("als_file_path"), str):
        merged["als_file_path"] = als_result.get("als_file_path")
    if isinstance(als_result.get("als_file_mtime_utc"), str):
        merged["als_file_mtime_utc"] = als_result.get("als_file_mtime_utc")
    if _safe_int_value(als_result.get("als_candidate_count")) is not None:
        merged["als_candidate_count"] = _safe_int_value(als_result.get("als_candidate_count"))

    if not isinstance(merged.get("target"), dict):
        merged["target"] = {}
    als_target = als_result.get("target")
    if isinstance(als_target, dict):
        for key, value in als_target.items():
            if key not in merged["target"] or merged["target"].get(key) in (None, ""):
                merged["target"][key] = value
        # Keep the resolved automation target id even if target already existed.
        if "automation_target_id" in als_target:
            merged["target"]["automation_target_id"] = als_target.get("automation_target_id")

    if not _safe_text_value(merged.get("track_name")) and _safe_text_value(als_result.get("track_name")):
        merged["track_name"] = als_result.get("track_name")

    warnings = list(merged.get("warnings") or [])
    for warning in list(als_result.get("warnings") or []):
        if isinstance(warning, str) and warning not in warnings:
            warnings.append(warning)
    merged["warnings"] = warnings
    if (_safe_text_value(scope) or "").lower() == "device_parameter":
        merged["als_mapping_verification"] = {"ok": True, "reason": "verified"}

    return merged


@mcp.tool()
def get_clip_automation_envelope_points(
    ctx: Context,
    track_index: int,
    clip_scope: str = "session",
    clip_slot_index: Optional[int] = None,
    clip_index: Optional[int] = None,
    scope: str = "device_parameter",
    mixer_target: str = "volume",
    send_index: Optional[int] = None,
    device_index: Optional[int] = None,
    parameter_index: Optional[int] = None,
    start_time_beats: Optional[float] = None,
    end_time_beats: Optional[float] = None,
    sample_points: int = 0
) -> Dict[str, Any]:
    """
    Return clip automation envelope points (best effort) for a session/arrangement clip and target parameter.

    Parameters:
    - track_index: 0-based normal track index
    - clip_scope: "session" or "arrangement"
    - clip_slot_index: required for clip_scope="session"
    - clip_index: required for clip_scope="arrangement"
    - scope: "track_mixer" or "device_parameter"
    - mixer_target: for track_mixer scope: "volume", "panning", or "send"
    - send_index: required when scope="track_mixer" and mixer_target="send"
    - device_index: required when scope="device_parameter"
    - parameter_index: required when scope="device_parameter"
    - start_time_beats/end_time_beats: optional sampling range when point access is unavailable
    - sample_points: optional sampled fallback count via envelope.value_at_time
    """
    _ = ctx

    track_index_value = _safe_int_value(track_index)
    if track_index_value is None or track_index_value < 0:
        return {
            "ok": False,
            "error": "invalid_track_index",
            "message": "track_index must be a non-negative integer",
            "track_index": track_index
        }

    clip_scope_value = (_safe_text_value(clip_scope) or "session").lower()
    if clip_scope_value not in {"session", "arrangement"}:
        return {
            "ok": False,
            "error": "invalid_clip_scope",
            "message": "clip_scope must be 'session' or 'arrangement'",
            "track_index": track_index_value,
            "clip_scope": clip_scope
        }

    if clip_scope_value == "session":
        clip_slot_index_value = _safe_int_value(clip_slot_index)
        if clip_slot_index_value is None or clip_slot_index_value < 0:
            return {
                "ok": False,
                "error": "invalid_clip_slot_index",
                "message": "clip_slot_index must be a non-negative integer for clip_scope='session'",
                "track_index": track_index_value,
                "clip_scope": clip_scope_value,
                "clip_slot_index": clip_slot_index
            }
    else:
        clip_index_value = _safe_int_value(clip_index)
        if clip_index_value is None or clip_index_value < 0:
            return {
                "ok": False,
                "error": "invalid_clip_index",
                "message": "clip_index must be a non-negative integer for clip_scope='arrangement'",
                "track_index": track_index_value,
                "clip_scope": clip_scope_value,
                "clip_index": clip_index
            }

    scope_value = (_safe_text_value(scope) or "device_parameter").lower()
    mixer_target_value = (_safe_text_value(mixer_target) or "volume").lower()

    if scope_value not in {"track_mixer", "device_parameter"}:
        return {
            "ok": False,
            "error": "invalid_scope",
            "message": "scope must be 'track_mixer' or 'device_parameter'",
            "track_index": track_index_value,
            "scope": scope
        }

    if scope_value == "track_mixer" and mixer_target_value not in {"volume", "panning", "send"}:
        return {
            "ok": False,
            "error": "invalid_mixer_target",
            "message": "mixer_target must be one of: volume, panning, send",
            "track_index": track_index_value,
            "scope": scope_value,
            "mixer_target": mixer_target
        }

    if scope_value == "track_mixer" and mixer_target_value == "send":
        send_index_value = _safe_int_value(send_index)
        if send_index_value is None or send_index_value < 0:
            return {
                "ok": False,
                "error": "invalid_send_index",
                "message": "send_index must be a non-negative integer for mixer_target='send'",
                "track_index": track_index_value,
                "scope": scope_value,
                "mixer_target": mixer_target_value,
                "send_index": send_index
            }

    if scope_value == "device_parameter":
        device_index_value = _safe_int_value(device_index)
        parameter_index_value = _safe_int_value(parameter_index)
        if device_index_value is None or device_index_value < 0:
            return {
                "ok": False,
                "error": "invalid_device_index",
                "message": "device_index must be a non-negative integer for scope='device_parameter'",
                "track_index": track_index_value,
                "scope": scope_value,
                "device_index": device_index
            }
        if parameter_index_value is None or parameter_index_value < 0:
            return {
                "ok": False,
                "error": "invalid_parameter_index",
                "message": "parameter_index must be a non-negative integer for scope='device_parameter'",
                "track_index": track_index_value,
                "scope": scope_value,
                "device_index": device_index_value,
                "parameter_index": parameter_index
            }

    params: Dict[str, Any] = {
        "track_index": track_index_value,
        "clip_scope": clip_scope_value,
        "scope": scope_value,
        "mixer_target": mixer_target_value,
        "sample_points": max(0, _safe_int_value(sample_points) or 0),
    }
    if clip_slot_index is not None:
        params["clip_slot_index"] = _safe_int_value(clip_slot_index)
    if clip_index is not None:
        params["clip_index"] = _safe_int_value(clip_index)
    if send_index is not None:
        params["send_index"] = _safe_int_value(send_index)
    if device_index is not None:
        params["device_index"] = _safe_int_value(device_index)
    if parameter_index is not None:
        params["parameter_index"] = _safe_int_value(parameter_index)
    if start_time_beats is not None:
        params["start_time_beats"] = _safe_float_value(start_time_beats)
    if end_time_beats is not None:
        params["end_time_beats"] = _safe_float_value(end_time_beats)

    try:
        ableton = get_ableton_connection()
        raw = ableton.send_command("get_clip_automation_envelope_points", params)
    except Exception as e:
        message = str(e)
        if "Unknown command: get_clip_automation_envelope_points" in message:
            return {
                "ok": True,
                "supported": False,
                "reason": "backend_command_unavailable",
                "message": message,
                "track_index": track_index_value,
                "clip_scope": clip_scope_value,
                "clip_slot_index": _safe_int_value(clip_slot_index),
                "clip_index": _safe_int_value(clip_index),
                "target": {
                    "scope": scope_value,
                    "mixer_target": mixer_target_value,
                    "send_index": _safe_int_value(send_index),
                    "device_index": _safe_int_value(device_index),
                    "parameter_index": _safe_int_value(parameter_index),
                },
                "point_access_supported": False,
                "envelope_exists": None,
                "points": [],
                "sampled_series": [],
                "warnings": ["backend_command_unavailable"]
            }
        logger.error(f"Error getting clip automation envelope points: {message}")
        return {
            "ok": False,
            "error": "get_clip_automation_envelope_points_failed",
            "message": message,
            "track_index": track_index_value,
            "clip_scope": clip_scope_value,
            "clip_slot_index": _safe_int_value(clip_slot_index),
            "clip_index": _safe_int_value(clip_index),
        }

    payload, payload_error = _coerce_json_dict(raw)
    if payload is None:
        return {
            "ok": False,
            "error": "invalid_response",
            "message": payload_error or "Backend returned invalid payload",
            "track_index": track_index_value,
            "clip_scope": clip_scope_value
        }

    result = dict(payload)
    if "ok" not in result:
        result["ok"] = True

    result["supported"] = bool(result.get("supported", False)) if "supported" in result else False
    result["envelope_exists"] = result.get("envelope_exists")
    result["point_access_supported"] = bool(result.get("point_access_supported", False)) if "point_access_supported" in result else False
    if not isinstance(result.get("warnings"), list):
        result["warnings"] = []

    result.setdefault("track_index", track_index_value)
    result.setdefault("clip_scope", clip_scope_value)
    if clip_scope_value == "session" and "clip_slot_index" not in result:
        result["clip_slot_index"] = _safe_int_value(clip_slot_index)
    if clip_scope_value == "arrangement" and "clip_index" not in result:
        result["clip_index"] = _safe_int_value(clip_index)

    clip_payload = result.get("clip")
    if not isinstance(clip_payload, dict):
        clip_payload = {}
    clip_payload.setdefault("clip_scope", clip_scope_value)
    result["clip"] = clip_payload

    target_payload = result.get("target")
    if not isinstance(target_payload, dict):
        target_payload = {}
    target_payload.setdefault("scope", scope_value)
    if scope_value == "track_mixer":
        target_payload.setdefault("mixer_target", mixer_target_value)
    result["target"] = target_payload

    def _normalize_point_rows(rows: Any, point_index_key: str = "point_index") -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not isinstance(rows, list):
            return normalized
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            time_beats = _safe_float_value(row.get("time_beats"))
            if time_beats is None:
                time_beats = _safe_float_value(row.get("time"))
            if time_beats is None:
                continue

            raw_value = row.get("value")
            if isinstance(raw_value, bool):
                value = raw_value
            elif isinstance(raw_value, (int, float)):
                value = raw_value
            else:
                parsed_float = _safe_float_value(raw_value)
                if parsed_float is not None:
                    value = parsed_float
                elif isinstance(raw_value, str):
                    value = raw_value
                else:
                    continue

            point_index_value = _safe_int_value(row.get(point_index_key))
            normalized_row: Dict[str, Any] = {
                point_index_key: point_index_value if point_index_value is not None else idx,
                "time_beats": time_beats,
                "value": value,
            }
            for optional_key in ("shape", "event_type", "value_kind"):
                optional_value = _safe_text_value(row.get(optional_key))
                if optional_value:
                    normalized_row[optional_key] = optional_value
            event_id_value = _safe_int_value(row.get("event_id"))
            if event_id_value is not None:
                normalized_row["event_id"] = event_id_value
            if row.get("is_pre_roll_default") is True:
                normalized_row["is_pre_roll_default"] = True
            curve_payload = row.get("curve")
            if isinstance(curve_payload, dict):
                curve_clean: Dict[str, float] = {}
                for curve_key, curve_value in curve_payload.items():
                    curve_float = _safe_float_value(curve_value)
                    if curve_float is None:
                        continue
                    curve_clean[str(curve_key)] = curve_float
                if curve_clean:
                    normalized_row["curve"] = curve_clean
            normalized.append(normalized_row)
        normalized.sort(key=lambda row: (_safe_float_value(row.get("time_beats")) or 0.0, _safe_int_value(row.get(point_index_key)) or 0))
        return normalized

    result["points"] = _normalize_point_rows(result.get("points"), point_index_key="point_index")
    result["sampled_series"] = _normalize_point_rows(result.get("sampled_series"), point_index_key="sample_index")
    return result


@mcp.tool()
def get_automation_envelope_points(
    ctx: Context,
    track_index: int,
    scope: str = "track_mixer",
    mixer_target: str = "volume",
    als_file_path: Optional[str] = None,
    send_index: Optional[int] = None,
    device_index: Optional[int] = None,
    parameter_index: Optional[int] = None,
    start_time_beats: Optional[float] = None,
    end_time_beats: Optional[float] = None,
    sample_points: int = 0
) -> Dict[str, Any]:
    """
    Return automation envelope points (best effort) for a track mixer target or device parameter.

    Parameters:
    - track_index: 0-based normal track index
    - scope: "track_mixer" or "device_parameter"
    - mixer_target: for track_mixer scope: "volume", "panning", or "send"
    - als_file_path: optional explicit .als file path for arrangement automation fallback
    - send_index: required when scope="track_mixer" and mixer_target="send"
    - device_index: required when scope="device_parameter"
    - parameter_index: required when scope="device_parameter"
    - start_time_beats/end_time_beats: optional sampling range when point access is unavailable
    - sample_points: optional sampled fallback count via envelope.value_at_time
    """
    _ = ctx

    track_index_value = _safe_int_value(track_index)
    if track_index_value is None or track_index_value < 0:
        return {
            "ok": False,
            "error": "invalid_track_index",
            "message": "track_index must be a non-negative integer",
            "track_index": track_index
        }

    scope_value = (_safe_text_value(scope) or "track_mixer").lower()
    mixer_target_value = (_safe_text_value(mixer_target) or "volume").lower()

    if scope_value not in {"track_mixer", "device_parameter"}:
        return {
            "ok": False,
            "error": "invalid_scope",
            "message": "scope must be 'track_mixer' or 'device_parameter'",
            "track_index": track_index_value,
            "scope": scope
        }

    if scope_value == "track_mixer" and mixer_target_value not in {"volume", "panning", "send"}:
        return {
            "ok": False,
            "error": "invalid_mixer_target",
            "message": "mixer_target must be one of: volume, panning, send",
            "track_index": track_index_value,
            "scope": scope_value,
            "mixer_target": mixer_target
        }

    if scope_value == "track_mixer" and mixer_target_value == "send":
        send_index_value = _safe_int_value(send_index)
        if send_index_value is None or send_index_value < 0:
            return {
                "ok": False,
                "error": "invalid_send_index",
                "message": "send_index must be a non-negative integer for mixer_target='send'",
                "track_index": track_index_value,
                "scope": scope_value,
                "mixer_target": mixer_target_value,
                "send_index": send_index
            }

    if scope_value == "device_parameter":
        device_index_value = _safe_int_value(device_index)
        parameter_index_value = _safe_int_value(parameter_index)
        if device_index_value is None or device_index_value < 0:
            return {
                "ok": False,
                "error": "invalid_device_index",
                "message": "device_index must be a non-negative integer for scope='device_parameter'",
                "track_index": track_index_value,
                "scope": scope_value,
                "device_index": device_index
            }
        if parameter_index_value is None or parameter_index_value < 0:
            return {
                "ok": False,
                "error": "invalid_parameter_index",
                "message": "parameter_index must be a non-negative integer for scope='device_parameter'",
                "track_index": track_index_value,
                "scope": scope_value,
                "device_index": device_index_value,
                "parameter_index": parameter_index
            }

    params: Dict[str, Any] = {
        "track_index": track_index_value,
        "scope": scope_value,
        "mixer_target": mixer_target_value,
        "sample_points": max(0, _safe_int_value(sample_points) or 0),
    }
    if send_index is not None:
        params["send_index"] = _safe_int_value(send_index)
    if device_index is not None:
        params["device_index"] = _safe_int_value(device_index)
    if parameter_index is not None:
        params["parameter_index"] = _safe_int_value(parameter_index)
    if start_time_beats is not None:
        params["start_time_beats"] = _safe_float_value(start_time_beats)
    if end_time_beats is not None:
        params["end_time_beats"] = _safe_float_value(end_time_beats)
    als_file_path_value = _safe_text_value(als_file_path)

    try:
        ableton = get_ableton_connection()
        raw = ableton.send_command("get_automation_envelope_points", params)
    except Exception as e:
        message = str(e)
        if "Unknown command: get_automation_envelope_points" in message:
            base_result = {
                "ok": True,
                "supported": False,
                "reason": "backend_command_unavailable",
                "message": message,
                "track_index": track_index_value,
                "target": {
                    "scope": scope_value,
                    "mixer_target": mixer_target_value,
                    "send_index": _safe_int_value(send_index),
                    "device_index": _safe_int_value(device_index),
                    "parameter_index": _safe_int_value(parameter_index),
                },
                "point_access_supported": False,
                "envelope_exists": None,
                "points": [],
                "sampled_series": [],
                "warnings": ["backend_command_unavailable"]
            }
            return _apply_als_arrangement_automation_fallback(
                base_result=base_result,
                track_index=track_index_value,
                scope=scope_value,
                mixer_target=mixer_target_value,
                als_file_path=als_file_path_value,
                send_index=_safe_int_value(send_index),
                device_index=_safe_int_value(device_index),
                parameter_index=_safe_int_value(parameter_index),
                start_time_beats=_safe_float_value(start_time_beats),
                end_time_beats=_safe_float_value(end_time_beats),
            )
        logger.error(f"Error getting automation envelope points: {message}")
        return {
            "ok": False,
            "error": "get_automation_envelope_points_failed",
            "message": message,
            "track_index": track_index_value,
            "target": {
                "scope": scope_value,
                "mixer_target": mixer_target_value,
                "send_index": _safe_int_value(send_index),
                "device_index": _safe_int_value(device_index),
                "parameter_index": _safe_int_value(parameter_index),
            }
        }

    payload, payload_error = _coerce_json_dict(raw)
    if payload is None:
        return {
            "ok": False,
            "error": "invalid_response",
            "message": payload_error or "Backend returned invalid payload",
            "track_index": track_index_value
        }

    result = dict(payload)
    if "ok" not in result:
        result["ok"] = True

    # Normalize top-level booleans / lists for stability.
    result["supported"] = bool(result.get("supported", False)) if "supported" in result else False
    result["envelope_exists"] = result.get("envelope_exists")
    result["point_access_supported"] = bool(result.get("point_access_supported", False)) if "point_access_supported" in result else False

    if not isinstance(result.get("warnings"), list):
        result["warnings"] = []

    target_payload = result.get("target")
    if not isinstance(target_payload, dict):
        target_payload = {}
    target_payload.setdefault("scope", scope_value)
    if scope_value == "track_mixer":
        target_payload.setdefault("mixer_target", mixer_target_value)
    result["target"] = target_payload

    result = _apply_als_arrangement_automation_fallback(
        base_result=result,
        track_index=track_index_value,
        scope=scope_value,
        mixer_target=mixer_target_value,
        als_file_path=als_file_path_value,
        send_index=_safe_int_value(send_index),
        device_index=_safe_int_value(device_index),
        parameter_index=_safe_int_value(parameter_index),
        start_time_beats=_safe_float_value(start_time_beats),
        end_time_beats=_safe_float_value(end_time_beats),
    )

    def _normalize_point_rows(rows: Any, point_index_key: str = "point_index") -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not isinstance(rows, list):
            return normalized
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            time_beats = _safe_float_value(row.get("time_beats"))
            if time_beats is None:
                time_beats = _safe_float_value(row.get("time"))
            if time_beats is None:
                continue
            raw_value = row.get("value")
            if isinstance(raw_value, bool):
                value = raw_value
            elif isinstance(raw_value, (int, float)):
                value = raw_value
            else:
                parsed_float = _safe_float_value(raw_value)
                if parsed_float is not None:
                    value = parsed_float
                elif isinstance(raw_value, str):
                    value = raw_value
                else:
                    continue
            point_index_value = _safe_int_value(row.get(point_index_key))
            normalized_row = {
                point_index_key: point_index_value if point_index_value is not None else idx,
                "time_beats": time_beats,
                "value": value,
            }
            shape_value = _safe_text_value(row.get("shape"))
            if shape_value:
                normalized_row["shape"] = shape_value
            event_type = _safe_text_value(row.get("event_type"))
            if event_type:
                normalized_row["event_type"] = event_type
            value_kind = _safe_text_value(row.get("value_kind"))
            if value_kind:
                normalized_row["value_kind"] = value_kind
            event_id = _safe_int_value(row.get("event_id"))
            if event_id is not None:
                normalized_row["event_id"] = event_id
            if row.get("is_pre_roll_default") is True:
                normalized_row["is_pre_roll_default"] = True
            curve_payload = row.get("curve")
            if isinstance(curve_payload, dict):
                curve_clean: Dict[str, float] = {}
                for curve_key, curve_value in curve_payload.items():
                    curve_float = _safe_float_value(curve_value)
                    if curve_float is None:
                        continue
                    curve_clean[str(curve_key)] = curve_float
                if curve_clean:
                    normalized_row["curve"] = curve_clean
            normalized.append(normalized_row)
        normalized.sort(key=lambda row: (_safe_float_value(row.get("time_beats")) or 0.0, _safe_int_value(row.get(point_index_key)) or 0))
        return normalized

    result["points"] = _normalize_point_rows(result.get("points"), point_index_key="point_index")
    result["sampled_series"] = _normalize_point_rows(result.get("sampled_series"), point_index_key="sample_index")

    return result


@mcp.tool()
def get_automation_overview(ctx: Context, track_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Return automation-state coverage overview for track devices (read-only; envelope points not included).

    Parameters:
    - track_indices: optional subset of normal tracks
    """
    session_payload, session_error = _coerce_json_dict(get_session_info(ctx))
    if session_payload is None:
        try:
            ableton = get_ableton_connection()
            session_direct = ableton.send_command("get_session_info")
            if isinstance(session_direct, dict):
                session_payload = session_direct
        except Exception:
            session_payload = None
    if session_payload is None:
        return {
            "ok": False,
            "error": "session_info_unavailable",
            "message": session_error or "Failed to retrieve session info"
        }

    track_count = _safe_int_value(session_payload.get("track_count"))
    indices_to_scan: List[int] = []
    if track_indices is None:
        if track_count is None:
            return {
                "ok": False,
                "error": "track_count_missing",
                "message": "track_count missing from session info"
            }
        indices_to_scan = list(range(max(0, track_count)))
    else:
        seen = set()
        for value in track_indices:
            idx = _safe_int_value(value)
            if idx is None or idx < 0:
                continue
            if idx in seen:
                continue
            seen.add(idx)
            indices_to_scan.append(idx)

    tracks = []
    tracks_with_device_automation = 0
    total_automated_parameters = 0
    warnings: List[str] = []

    for track_index in indices_to_scan:
        result = get_track_automation_targets(ctx, track_index)
        if not isinstance(result, dict):
            warnings.append(f"invalid_track_automation_result:{track_index}")
            continue
        if result.get("ok") is not True:
            tracks.append({
                "track_index": track_index,
                "ok": False,
                "error": result.get("error"),
                "message": result.get("message")
            })
            continue

        summary = result.get("summary", {})
        if isinstance(summary, dict):
            if int(summary.get("devices_with_automation", 0) or 0) > 0:
                tracks_with_device_automation += 1
            total_automated_parameters += int(summary.get("automated_parameter_count", 0) or 0)

        tracks.append({
            "track_index": track_index,
            "track_name": result.get("track_name"),
            "ok": True,
            "devices_with_automation": summary.get("devices_with_automation"),
            "automated_parameter_count": summary.get("automated_parameter_count"),
            "track_mixer_targets": result.get("track_mixer_targets"),
            "warnings": result.get("warnings", [])
        })

    return {
        "ok": True,
        "supported": True,
        "envelope_points_supported": False,
        "tracks_scanned": len(indices_to_scan),
        "tracks_with_device_automation": tracks_with_device_automation,
        "total_automated_parameters": total_automated_parameters,
        "tracks": tracks,
        "warnings": warnings,
        "fallback_guidance": [
            "Device parameter automation states are available in overview responses.",
            "Call get_automation_envelope_points for specific mixer/device targets when you need point data, or use export-based loudness timelines as a proxy."
        ]
    }


@mcp.tool()
def enumerate_project_automation(
    ctx: Context,
    track_indices: Optional[List[int]] = None,
    als_file_path: Optional[str] = None,
    include_arrangement_mixer_points: bool = True,
    include_clip_envelopes: bool = True,
    include_device_parameter_points: bool = False,
    include_return_master_context: bool = True,
    max_clips_per_track: int = 4,
    sample_points: int = 0,
    start_time_beats: Optional[float] = None,
    end_time_beats: Optional[float] = None,
    include_point_payloads: bool = False
) -> Dict[str, Any]:
    """
    Enumerate automation coverage across the current project (broad scan + targeted point probes).

    This is a read-only inventory tool intended to provide a broader project-wide picture than
    `get_automation_overview`, while still reusing the safe single-target probe tools for exact points.

    Parameters:
    - track_indices: optional subset of normal tracks to scan
    - als_file_path: optional explicit .als file for arrangement automation fallback
    - include_arrangement_mixer_points: probe track mixer arrangement automation (volume/pan/send)
    - include_clip_envelopes: probe clip envelopes for discovered session/arrangement clips (limited per track)
    - include_device_parameter_points: probe exact device-parameter arrangement envelopes for automated params
    - include_return_master_context: include return/master chain inventory + explicit automation coverage gaps
    - max_clips_per_track: cap clip-envelope probes per track per clip scope
    - sample_points/start_time_beats/end_time_beats: optional sampling hints for point probe fallbacks
    - include_point_payloads: include full `points`/`sampled_series` in probe rows (can be large)
    """
    _ = ctx

    def _point_probe_summary(payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {
                "ok": False,
                "error": "invalid_probe_payload",
                "probe_payload_type": str(type(payload))
            }

        summary: Dict[str, Any] = {
            "ok": bool(payload.get("ok", False)),
            "supported": bool(payload.get("supported", False)) if "supported" in payload else None,
            "reason": payload.get("reason"),
            "error": payload.get("error"),
            "envelope_exists": payload.get("envelope_exists"),
            "point_access_supported": payload.get("point_access_supported"),
            "point_source": payload.get("point_source"),
            "source": payload.get("source"),
            "warnings": list(payload.get("warnings") or []) if isinstance(payload.get("warnings"), list) else [],
        }

        target_payload = payload.get("target")
        if isinstance(target_payload, dict):
            target_summary: Dict[str, Any] = {}
            for key in (
                "scope", "mixer_target", "send_index",
                "device_index", "device_name",
                "parameter_index", "parameter_name",
                "automation_target_id"
            ):
                if key in target_payload:
                    target_summary[key] = target_payload.get(key)
            if target_summary:
                summary["target"] = target_summary

        clip_payload = payload.get("clip")
        if isinstance(clip_payload, dict):
            clip_summary: Dict[str, Any] = {}
            for key in (
                "clip_scope", "clip_slot_index", "clip_index",
                "clip_name", "is_audio_clip", "is_midi_clip", "length_beats"
            ):
                if key in clip_payload:
                    clip_summary[key] = clip_payload.get(key)
            if clip_summary:
                summary["clip"] = clip_summary

        points = payload.get("points")
        if isinstance(points, list):
            summary["point_count"] = len(points)
            if points:
                first_time = _safe_float_value(points[0].get("time_beats")) if isinstance(points[0], dict) else None
                last_time = _safe_float_value(points[-1].get("time_beats")) if isinstance(points[-1], dict) else None
                if first_time is not None:
                    summary["first_point_time_beats"] = first_time
                if last_time is not None:
                    summary["last_point_time_beats"] = last_time
        else:
            summary["point_count"] = 0

        sampled = payload.get("sampled_series")
        if isinstance(sampled, list):
            summary["sample_count"] = len(sampled)

        if "als_fallback_used" in payload:
            summary["als_fallback_used"] = bool(payload.get("als_fallback_used"))
        if isinstance(payload.get("als_fallback_reason"), str):
            summary["als_fallback_reason"] = payload.get("als_fallback_reason")
        if isinstance(payload.get("als_file_path"), str):
            summary["als_file_path"] = payload.get("als_file_path")
        if isinstance(payload.get("als_mapping_verification"), dict):
            summary["als_mapping_verification"] = payload.get("als_mapping_verification")

        if include_point_payloads:
            summary["points"] = list(points or []) if isinstance(points, list) else []
            summary["sampled_series"] = list(sampled or []) if isinstance(sampled, list) else []

        return summary

    def _compact_track_target_summary(payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {"ok": False, "error": "invalid_track_target_payload"}
        if payload.get("ok") is not True:
            return {
                "ok": False,
                "error": payload.get("error"),
                "message": payload.get("message")
            }

        summary = payload.get("summary", {})
        out: Dict[str, Any] = {
            "ok": True,
            "track_name": payload.get("track_name"),
            "summary": summary if isinstance(summary, dict) else {},
            "track_mixer_targets": payload.get("track_mixer_targets"),
            "warnings": list(payload.get("warnings") or []) if isinstance(payload.get("warnings"), list) else [],
        }

        devices_out: List[Dict[str, Any]] = []
        for device in list(payload.get("devices") or []):
            if not isinstance(device, dict):
                continue
            if device.get("has_automation") is not True and int(device.get("automated_parameter_count", 0) or 0) <= 0:
                continue
            device_row = {
                "device_index": device.get("device_index"),
                "device_name": device.get("device_name"),
                "class_name": device.get("class_name"),
                "automated_parameter_count": device.get("automated_parameter_count"),
            }
            automated_params = []
            for param in list(device.get("parameters") or []):
                if not isinstance(param, dict):
                    continue
                if param.get("automated") is not True:
                    continue
                automated_params.append({
                    "parameter_index": param.get("parameter_index"),
                    "name": param.get("name"),
                    "automation_state": param.get("automation_state"),
                    "automated": True
                })
            device_row["automated_parameters"] = automated_params
            devices_out.append(device_row)
        out["devices_with_automation"] = devices_out
        return out

    session_payload, session_error = _coerce_json_dict(get_session_info(ctx))
    if session_payload is None:
        return {
            "ok": False,
            "error": "session_info_unavailable",
            "message": session_error or "Failed to retrieve session info"
        }

    overview = get_automation_overview(ctx, track_indices=track_indices)
    if not isinstance(overview, dict) or overview.get("ok") is not True:
        return {
            "ok": False,
            "error": "automation_overview_unavailable",
            "message": (
                overview.get("message") if isinstance(overview, dict) else "Invalid automation overview response"
            ),
            "details": overview if isinstance(overview, dict) else None
        }

    max_clips_value = _safe_int_value(max_clips_per_track)
    if max_clips_value is None:
        max_clips_value = 4
    max_clips_value = max(0, min(max_clips_value, 128))
    sample_points_value = max(0, _safe_int_value(sample_points) or 0)
    start_time_beats_value = _safe_float_value(start_time_beats)
    end_time_beats_value = _safe_float_value(end_time_beats)
    als_file_path_value = _safe_text_value(als_file_path)

    send_counts_by_track: Dict[int, int] = {}
    topology_warnings: List[str] = []
    if include_arrangement_mixer_points:
        try:
            topology = get_mix_topology(ctx, include_device_chains=False, include_device_parameters=False)
            if isinstance(topology, dict) and topology.get("ok") is True:
                for row in list(topology.get("tracks") or []):
                    if not isinstance(row, dict):
                        continue
                    idx = _safe_int_value(row.get("index"))
                    if idx is None:
                        continue
                    sends = row.get("sends")
                    send_counts_by_track[idx] = len(sends) if isinstance(sends, list) else 0
            elif isinstance(topology, dict):
                topology_warnings.append("mix_topology_unavailable")
        except Exception:
            topology_warnings.append("mix_topology_exception")

    coverage = {
        "tracks_scanned": 0,
        "tracks_with_device_automation": int(overview.get("tracks_with_device_automation", 0) or 0),
        "total_automated_parameters": int(overview.get("total_automated_parameters", 0) or 0),
        "arrangement_mixer_point_queries": 0,
        "arrangement_mixer_envelopes_found": 0,
        "device_parameter_point_queries": 0,
        "device_parameter_envelopes_found": 0,
        "device_parameter_als_mismatch_blocks": 0,
        "clip_envelope_probes": 0,
        "clip_envelopes_found": 0,
        "session_clips_seen": 0,
        "arrangement_clips_seen": 0,
        "return_point_queries": 0,
        "return_envelopes_found": 0,
        "master_point_queries": 0,
        "master_envelopes_found": 0,
        "global_point_queries": 0,
        "global_envelopes_found": 0,
    }

    tracks_out: List[Dict[str, Any]] = []
    warnings: List[str] = list(overview.get("warnings") or []) if isinstance(overview.get("warnings"), list) else []
    warnings.extend([w for w in topology_warnings if w not in warnings])
    gaps: List[Dict[str, Any]] = []

    for overview_row in list(overview.get("tracks") or []):
        if not isinstance(overview_row, dict):
            continue
        track_index_value = _safe_int_value(overview_row.get("track_index"))
        if track_index_value is None:
            continue
        coverage["tracks_scanned"] += 1

        track_row: Dict[str, Any] = {
            "track_index": track_index_value,
            "track_name": overview_row.get("track_name"),
            "overview": {
                "ok": bool(overview_row.get("ok", False)),
                "devices_with_automation": overview_row.get("devices_with_automation"),
                "automated_parameter_count": overview_row.get("automated_parameter_count"),
                "track_mixer_targets": overview_row.get("track_mixer_targets"),
                "warnings": list(overview_row.get("warnings") or []) if isinstance(overview_row.get("warnings"), list) else []
            }
        }

        if overview_row.get("ok") is not True:
            tracks_out.append(track_row)
            continue

        track_targets_payload: Optional[Dict[str, Any]] = None
        needs_track_targets = bool(include_device_parameter_points) or bool(include_clip_envelopes)
        if needs_track_targets and int(overview_row.get("automated_parameter_count", 0) or 0) > 0:
            try:
                track_targets_payload = get_track_automation_targets(ctx, track_index_value)
            except Exception as exc:
                track_targets_payload = {
                    "ok": False,
                    "error": "get_track_automation_targets_failed",
                    "message": str(exc)
                }
            track_row["track_targets"] = _compact_track_target_summary(track_targets_payload)

        automated_device_params: List[Dict[str, Any]] = []
        if isinstance(track_targets_payload, dict) and track_targets_payload.get("ok") is True:
            for device in list(track_targets_payload.get("devices") or []):
                if not isinstance(device, dict):
                    continue
                device_index_value = _safe_int_value(device.get("device_index"))
                if device_index_value is None:
                    continue
                device_name_value = _safe_text_value(device.get("device_name"))
                for param in list(device.get("parameters") or []):
                    if not isinstance(param, dict):
                        continue
                    if param.get("automated") is not True:
                        continue
                    param_index_value = _safe_int_value(param.get("parameter_index"))
                    if param_index_value is None:
                        continue
                    automated_device_params.append({
                        "device_index": device_index_value,
                        "device_name": device_name_value,
                        "parameter_index": param_index_value,
                        "parameter_name": _safe_text_value(param.get("name")),
                    })

        if include_arrangement_mixer_points:
            arrangement_mixer: Dict[str, Any] = {"targets": {}, "send_count_hint": send_counts_by_track.get(track_index_value, 0)}
            for mixer_target_value in ("volume", "panning"):
                probe_payload = get_automation_envelope_points(
                    ctx,
                    track_index=track_index_value,
                    scope="track_mixer",
                    mixer_target=mixer_target_value,
                    als_file_path=als_file_path_value,
                    start_time_beats=start_time_beats_value,
                    end_time_beats=end_time_beats_value,
                    sample_points=sample_points_value
                )
                probe_summary = _point_probe_summary(probe_payload)
                arrangement_mixer["targets"][mixer_target_value] = probe_summary
                coverage["arrangement_mixer_point_queries"] += 1
                if probe_summary.get("envelope_exists") is True:
                    coverage["arrangement_mixer_envelopes_found"] += 1

            send_probe_rows: List[Dict[str, Any]] = []
            send_count = max(0, _safe_int_value(send_counts_by_track.get(track_index_value)) or 0)
            for send_idx in range(send_count):
                probe_payload = get_automation_envelope_points(
                    ctx,
                    track_index=track_index_value,
                    scope="track_mixer",
                    mixer_target="send",
                    send_index=send_idx,
                    als_file_path=als_file_path_value,
                    start_time_beats=start_time_beats_value,
                    end_time_beats=end_time_beats_value,
                    sample_points=sample_points_value
                )
                probe_summary = _point_probe_summary(probe_payload)
                send_probe_rows.append({
                    "send_index": send_idx,
                    "probe": probe_summary
                })
                coverage["arrangement_mixer_point_queries"] += 1
                if probe_summary.get("envelope_exists") is True:
                    coverage["arrangement_mixer_envelopes_found"] += 1
            arrangement_mixer["sends"] = send_probe_rows
            track_row["arrangement_mixer"] = arrangement_mixer

        if include_device_parameter_points:
            device_point_rows: List[Dict[str, Any]] = []
            for target in automated_device_params:
                probe_payload = get_automation_envelope_points(
                    ctx,
                    track_index=track_index_value,
                    scope="device_parameter",
                    device_index=target["device_index"],
                    parameter_index=target["parameter_index"],
                    als_file_path=als_file_path_value,
                    start_time_beats=start_time_beats_value,
                    end_time_beats=end_time_beats_value,
                    sample_points=sample_points_value
                )
                probe_summary = _point_probe_summary(probe_payload)
                device_point_rows.append({
                    "device_index": target["device_index"],
                    "device_name": target.get("device_name"),
                    "parameter_index": target["parameter_index"],
                    "parameter_name": target.get("parameter_name"),
                    "probe": probe_summary
                })
                coverage["device_parameter_point_queries"] += 1
                if probe_summary.get("envelope_exists") is True:
                    coverage["device_parameter_envelopes_found"] += 1
                if _safe_text_value(probe_summary.get("als_fallback_reason")) in {
                    "device_parameter_mapping_mismatch",
                    "device_name_mapping_mismatch",
                    "device_parameter_mapping_unverifiable",
                }:
                    coverage["device_parameter_als_mismatch_blocks"] += 1
            track_row["device_parameter_points"] = {
                "query_count": len(device_point_rows),
                "targets": device_point_rows
            }

        if include_clip_envelopes:
            clip_inventory: Dict[str, Any] = {
                "session": {"supported": True, "clips": [], "warnings": []},
                "arrangement": {"supported": True, "clips": [], "warnings": []},
            }

            session_clips_payload = list_session_clips(ctx, track_index_value)
            if isinstance(session_clips_payload, dict) and "error" not in session_clips_payload:
                session_clips = session_clips_payload.get("clips", [])
                if not isinstance(session_clips, list):
                    session_clips = []
                coverage["session_clips_seen"] += len(session_clips)
                for clip_row in session_clips[:max_clips_value]:
                    if not isinstance(clip_row, dict):
                        continue
                    clip_slot_index_value = _safe_int_value(clip_row.get("clip_slot_index"))
                    if clip_slot_index_value is None:
                        continue
                    clip_probe_rows: List[Dict[str, Any]] = []
                    for mixer_target_value in ("volume", "panning"):
                        clip_probe = get_clip_automation_envelope_points(
                            ctx,
                            track_index=track_index_value,
                            clip_scope="session",
                            clip_slot_index=clip_slot_index_value,
                            scope="track_mixer",
                            mixer_target=mixer_target_value,
                            start_time_beats=start_time_beats_value,
                            end_time_beats=end_time_beats_value,
                            sample_points=sample_points_value
                        )
                        clip_probe_summary = _point_probe_summary(clip_probe)
                        clip_probe_rows.append({
                            "target_kind": "track_mixer",
                            "mixer_target": mixer_target_value,
                            "probe": clip_probe_summary
                        })
                        coverage["clip_envelope_probes"] += 1
                        if clip_probe_summary.get("envelope_exists") is True:
                            coverage["clip_envelopes_found"] += 1
                    for target in automated_device_params:
                        clip_probe = get_clip_automation_envelope_points(
                            ctx,
                            track_index=track_index_value,
                            clip_scope="session",
                            clip_slot_index=clip_slot_index_value,
                            scope="device_parameter",
                            device_index=target["device_index"],
                            parameter_index=target["parameter_index"],
                            start_time_beats=start_time_beats_value,
                            end_time_beats=end_time_beats_value,
                            sample_points=sample_points_value
                        )
                        clip_probe_summary = _point_probe_summary(clip_probe)
                        clip_probe_rows.append({
                            "target_kind": "device_parameter",
                            "device_index": target["device_index"],
                            "device_name": target.get("device_name"),
                            "parameter_index": target["parameter_index"],
                            "parameter_name": target.get("parameter_name"),
                            "probe": clip_probe_summary
                        })
                        coverage["clip_envelope_probes"] += 1
                        if clip_probe_summary.get("envelope_exists") is True:
                            coverage["clip_envelopes_found"] += 1
                    clip_inventory["session"]["clips"].append({
                        "clip_slot_index": clip_slot_index_value,
                        "clip_name": clip_row.get("clip_name"),
                        "is_audio_clip": clip_row.get("is_audio_clip"),
                        "is_midi_clip": clip_row.get("is_midi_clip"),
                        "probes": clip_probe_rows
                    })
            else:
                clip_inventory["session"]["supported"] = False
                if isinstance(session_clips_payload, dict):
                    clip_inventory["session"]["reason"] = session_clips_payload.get("error") or session_clips_payload.get("reason")
                    clip_inventory["session"]["message"] = session_clips_payload.get("message")

            arrangement_clips_payload = list_arrangement_clips(ctx, track_index_value)
            if isinstance(arrangement_clips_payload, dict) and arrangement_clips_payload.get("supported") is not False and "error" not in arrangement_clips_payload:
                arrangement_clips = arrangement_clips_payload.get("clips", [])
                if not isinstance(arrangement_clips, list):
                    arrangement_clips = []
                coverage["arrangement_clips_seen"] += len(arrangement_clips)
                for clip_row in arrangement_clips[:max_clips_value]:
                    if not isinstance(clip_row, dict):
                        continue
                    clip_index_value = _safe_int_value(clip_row.get("clip_index"))
                    if clip_index_value is None:
                        continue
                    clip_probe_rows: List[Dict[str, Any]] = []
                    for mixer_target_value in ("volume", "panning"):
                        clip_probe = get_clip_automation_envelope_points(
                            ctx,
                            track_index=track_index_value,
                            clip_scope="arrangement",
                            clip_index=clip_index_value,
                            scope="track_mixer",
                            mixer_target=mixer_target_value,
                            start_time_beats=start_time_beats_value,
                            end_time_beats=end_time_beats_value,
                            sample_points=sample_points_value
                        )
                        clip_probe_summary = _point_probe_summary(clip_probe)
                        clip_probe_rows.append({
                            "target_kind": "track_mixer",
                            "mixer_target": mixer_target_value,
                            "probe": clip_probe_summary
                        })
                        coverage["clip_envelope_probes"] += 1
                        if clip_probe_summary.get("envelope_exists") is True:
                            coverage["clip_envelopes_found"] += 1
                    for target in automated_device_params:
                        clip_probe = get_clip_automation_envelope_points(
                            ctx,
                            track_index=track_index_value,
                            clip_scope="arrangement",
                            clip_index=clip_index_value,
                            scope="device_parameter",
                            device_index=target["device_index"],
                            parameter_index=target["parameter_index"],
                            start_time_beats=start_time_beats_value,
                            end_time_beats=end_time_beats_value,
                            sample_points=sample_points_value
                        )
                        clip_probe_summary = _point_probe_summary(clip_probe)
                        clip_probe_rows.append({
                            "target_kind": "device_parameter",
                            "device_index": target["device_index"],
                            "device_name": target.get("device_name"),
                            "parameter_index": target["parameter_index"],
                            "parameter_name": target.get("parameter_name"),
                            "probe": clip_probe_summary
                        })
                        coverage["clip_envelope_probes"] += 1
                        if clip_probe_summary.get("envelope_exists") is True:
                            coverage["clip_envelopes_found"] += 1
                    clip_inventory["arrangement"]["clips"].append({
                        "clip_index": clip_index_value,
                        "clip_name": clip_row.get("clip_name"),
                        "is_audio_clip": clip_row.get("is_audio_clip"),
                        "is_midi_clip": clip_row.get("is_midi_clip"),
                        "start_time": clip_row.get("start_time"),
                        "end_time": clip_row.get("end_time"),
                        "probes": clip_probe_rows
                    })
            else:
                clip_inventory["arrangement"]["supported"] = False
                if isinstance(arrangement_clips_payload, dict):
                    clip_inventory["arrangement"]["reason"] = arrangement_clips_payload.get("error") or arrangement_clips_payload.get("reason")
                    clip_inventory["arrangement"]["message"] = arrangement_clips_payload.get("message")
                    debug_payload = arrangement_clips_payload.get("debug")
                    if isinstance(debug_payload, dict):
                        clip_inventory["arrangement"]["debug"] = debug_payload

            clip_inventory["session"]["clip_count"] = len(clip_inventory["session"]["clips"])
            clip_inventory["arrangement"]["clip_count"] = len(clip_inventory["arrangement"]["clips"])
            track_row["clip_inventory"] = clip_inventory

        tracks_out.append(track_row)

    returns_master_context = None
    if include_return_master_context:
        returns_payload = get_return_tracks_info(ctx, include_device_chains=True)
        master_payload = get_master_track_device_chain(ctx)
        non_track_als_inventory = None
        non_track_als_summary = None
        non_track_als_reason = None
        try:
            non_track_als_inventory = enumerate_non_track_arrangement_automation_from_project_als(
                project_root=get_project_root(),
                als_file_path=als_file_path_value,
                start_time_beats=start_time_beats_value,
                end_time_beats=end_time_beats_value,
            )
        except Exception as exc:
            non_track_als_reason = "non_track_als_inventory_exception"
            non_track_als_inventory = {
                "ok": True,
                "supported": False,
                "reason": "non_track_als_inventory_exception",
                "message": str(exc),
                "returns": [],
                "master": None,
                "global": None,
                "warnings": ["non_track_als_inventory_exception"],
            }

        if isinstance(non_track_als_inventory, dict) and non_track_als_inventory.get("supported") is True:
            returns_summary_rows: List[Dict[str, Any]] = []
            for return_row in list(non_track_als_inventory.get("returns") or []):
                if not isinstance(return_row, dict):
                    continue
                target_summaries: Dict[str, Any] = {}
                for key, value in (return_row.get("targets") or {}).items():
                    if not isinstance(key, str):
                        continue
                    probe_summary = _point_probe_summary(value)
                    target_summaries[key] = probe_summary
                    coverage["return_point_queries"] += 1
                    if probe_summary.get("envelope_exists") is True:
                        coverage["return_envelopes_found"] += 1
                send_rows_summary: List[Dict[str, Any]] = []
                for send_row in list(return_row.get("sends") or []):
                    if not isinstance(send_row, dict):
                        continue
                    probe_summary = _point_probe_summary(send_row.get("probe"))
                    send_rows_summary.append({
                        "send_index": _safe_int_value(send_row.get("send_index")),
                        "probe": probe_summary
                    })
                    coverage["return_point_queries"] += 1
                    if probe_summary.get("envelope_exists") is True:
                        coverage["return_envelopes_found"] += 1
                returns_summary_rows.append({
                    "index": _safe_int_value(return_row.get("index")),
                    "track_name": return_row.get("track_name"),
                    "targets": target_summaries,
                    "sends": send_rows_summary
                })

            master_summary = None
            master_targets_in = non_track_als_inventory.get("master")
            if isinstance(master_targets_in, dict):
                master_target_rows: Dict[str, Any] = {}
                for key, value in (master_targets_in.get("targets") or {}).items():
                    if not isinstance(key, str):
                        continue
                    probe_summary = _point_probe_summary(value)
                    master_target_rows[key] = probe_summary
                    coverage["master_point_queries"] += 1
                    if probe_summary.get("envelope_exists") is True:
                        coverage["master_envelopes_found"] += 1
                master_summary = {
                    "track_name": master_targets_in.get("track_name"),
                    "targets": master_target_rows
                }

            global_summary = None
            global_targets_in = non_track_als_inventory.get("global")
            if isinstance(global_targets_in, dict):
                global_target_rows: Dict[str, Any] = {}
                for key, value in (global_targets_in.get("targets") or {}).items():
                    if not isinstance(key, str):
                        continue
                    probe_summary = _point_probe_summary(value)
                    global_target_rows[key] = probe_summary
                    coverage["global_point_queries"] += 1
                    if probe_summary.get("envelope_exists") is True:
                        coverage["global_envelopes_found"] += 1
                global_summary = {
                    "track_name": global_targets_in.get("track_name"),
                    "targets": global_target_rows
                }

            non_track_als_summary = {
                "ok": True,
                "supported": True,
                "als_file_path": non_track_als_inventory.get("als_file_path"),
                "als_file_mtime_utc": non_track_als_inventory.get("als_file_mtime_utc"),
                "warnings": list(non_track_als_inventory.get("warnings") or []) if isinstance(non_track_als_inventory.get("warnings"), list) else [],
                "returns": returns_summary_rows,
                "master": master_summary,
                "global": global_summary,
            }
        else:
            if isinstance(non_track_als_inventory, dict):
                non_track_als_reason = _safe_text_value(non_track_als_inventory.get("reason")) or non_track_als_reason
                non_track_als_summary = {
                    "ok": True,
                    "supported": False,
                    "reason": non_track_als_reason or "als_unavailable",
                    "als_file_path": non_track_als_inventory.get("als_file_path"),
                    "warnings": list(non_track_als_inventory.get("warnings") or []) if isinstance(non_track_als_inventory.get("warnings"), list) else [],
                }
            else:
                non_track_als_summary = {
                    "ok": True,
                    "supported": False,
                    "reason": non_track_als_reason or "als_unavailable",
                    "warnings": ["non_track_als_inventory_unavailable"],
                }
        returns_master_context = {
            "returns": returns_payload,
            "master": master_payload,
            "automation_point_enumeration": {
                "returns_supported": bool(isinstance(non_track_als_summary, dict) and non_track_als_summary.get("supported") is True),
                "master_supported": bool(isinstance(non_track_als_summary, dict) and non_track_als_summary.get("supported") is True),
                "global_supported": bool(isinstance(non_track_als_summary, dict) and non_track_als_summary.get("supported") is True),
                "reason": non_track_als_reason,
            },
            "als_non_track_automation": non_track_als_summary,
        }
        if not (isinstance(non_track_als_summary, dict) and non_track_als_summary.get("supported") is True):
            gaps.append({
                "area": "return_master_automation",
                "status": "partial",
                "reason": non_track_als_reason or "return_master_automation_als_unavailable"
            })
            gaps.append({
                "area": "global_song_automation",
                "status": "partial",
                "reason": non_track_als_reason or "global_song_automation_als_unavailable"
            })
        else:
            gaps.append({
                "area": "global_song_automation",
                "status": "partial",
                "reason": "only_tempo_and_time_signature_are_enumerated"
            })
    else:
        gaps.append({
            "area": "return_master_automation",
            "status": "missing",
            "reason": "include_return_master_context_false"
        })
        gaps.append({
            "area": "global_song_automation",
            "status": "missing",
            "reason": "include_return_master_context_false"
        })

    gaps.append({
        "area": "automation_unsaved_edits_visibility",
        "status": "partial",
        "reason": "als_fallback_reads_saved_set_only"
    })

    if coverage["device_parameter_als_mismatch_blocks"] > 0:
        gaps.append({
            "area": "device_parameter_arrangement_points",
            "status": "partial",
            "reason": "some_als_device_parameter_mappings_blocked_for_safety"
        })

    return {
        "ok": True,
        "supported": True,
        "scope": "project_automation_inventory",
        "session": {
            "tempo": session_payload.get("tempo"),
            "signature_numerator": session_payload.get("signature_numerator"),
            "signature_denominator": session_payload.get("signature_denominator"),
            "track_count": session_payload.get("track_count"),
            "return_track_count": session_payload.get("return_track_count"),
        },
        "options": {
            "track_indices": list(track_indices) if isinstance(track_indices, list) else None,
            "als_file_path": als_file_path_value,
            "include_arrangement_mixer_points": bool(include_arrangement_mixer_points),
            "include_clip_envelopes": bool(include_clip_envelopes),
            "include_device_parameter_points": bool(include_device_parameter_points),
            "include_return_master_context": bool(include_return_master_context),
            "max_clips_per_track": max_clips_value,
            "sample_points": sample_points_value,
            "start_time_beats": start_time_beats_value,
            "end_time_beats": end_time_beats_value,
            "include_point_payloads": bool(include_point_payloads),
        },
        "coverage": coverage,
        "overview": {
            "tracks_scanned": overview.get("tracks_scanned"),
            "tracks_with_device_automation": overview.get("tracks_with_device_automation"),
            "total_automated_parameters": overview.get("total_automated_parameters"),
            "envelope_points_supported": overview.get("envelope_points_supported"),
        },
        "tracks": tracks_out,
        "returns_master": returns_master_context,
        "gaps": gaps,
        "warnings": warnings,
        "notes": [
            "This tool broadens automation visibility by combining overview scans with targeted point probes.",
            "Clip envelopes are only probed for discovered clips and selected candidate targets (mixer volume/pan + automated device params).",
            "Arrangement device-parameter ALS fallback is safety-gated and may refuse ambiguous mappings."
        ]
    }


@mcp.tool()
def build_mix_master_context(
    ctx: Context,
    profile: str = "full",
    include_source_inventory: bool = True,
    include_export_analysis: bool = False,
    include_mastering_metrics: bool = False,
    manifest_path: Optional[str] = None,
    job_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a deterministic, LLM-ready mix/mastering context payload with stage-readiness statuses.

    Parameters:
    - profile: reserved profile selector (currently informational)
    - include_source_inventory: include source clip inventory summaries
    - include_export_analysis: include export-based analysis if manifest/job available
    - include_mastering_metrics: when export analysis is requested, use mastering profile
    - manifest_path/job_name: optional export-manifest selectors for export analysis
    """
    profile_value = _safe_text_value(profile) or "full"

    mix_context_profile = build_mix_context_profile(ctx)
    topology = mix_context_profile.get("topology") if isinstance(mix_context_profile, dict) else None

    session_snapshot = get_session_snapshot(
        ctx=ctx,
        track_indices=None,
        include_arrangement_clip_sources=False
    )
    automation_overview = get_automation_overview(ctx)

    source_inventory = None
    if include_source_inventory:
        source_inventory = index_sources_from_live_set(ctx)

    export_analysis = None
    export_analysis_request = {
        "enabled": bool(include_export_analysis),
        "analysis_profile": "mastering" if include_mastering_metrics else "mix",
        "manifest_path": manifest_path,
        "job_name": job_name
    }
    if include_export_analysis:
        if (isinstance(manifest_path, str) and manifest_path.strip()) or (isinstance(job_name, str) and job_name.strip()):
            export_analysis = analyze_export_job(
                ctx=ctx,
                manifest_path=manifest_path,
                job_name=job_name,
                analysis_profile="mastering" if include_mastering_metrics else "mix"
            )
        else:
            export_analysis = {
                "ok": False,
                "error": "missing_export_manifest_selector",
                "message": "Provide manifest_path or job_name to include export analysis"
            }

    project_snapshot_summary = None
    try:
        snapshot = snapshot_project_state(ctx, include_device_hashes=False)
        if isinstance(snapshot, dict) and snapshot.get("ok") is True:
            project_snapshot_summary = {
                "snapshot_id": snapshot.get("snapshot_id"),
                "project_hash": snapshot.get("project_hash"),
                "summary": snapshot.get("summary")
            }
        elif isinstance(snapshot, dict):
            project_snapshot_summary = {
                "ok": False,
                "error": snapshot.get("error"),
                "message": snapshot.get("message")
            }
    except Exception as exc:
        project_snapshot_summary = {
            "ok": False,
            "error": "snapshot_project_state_failed",
            "message": str(exc)
        }

    stages, missing_actions = _build_mix_stage_readiness(
        topology=topology if isinstance(topology, dict) else None,
        tags_profile=mix_context_profile if isinstance(mix_context_profile, dict) else None,
        automation_overview=automation_overview if isinstance(automation_overview, dict) else None,
        source_inventory=source_inventory if isinstance(source_inventory, dict) else None,
        export_analysis=export_analysis if isinstance(export_analysis, dict) else None,
        include_mastering_metrics=bool(include_mastering_metrics)
    )

    future_mutation_api_spec = {
        "status": "planned_not_implemented",
        "endpoints": [
            "set_mixer_values(scope, index, volume?, panning?, mute?, solo?, arm?)",
            "set_send_level(track_index, send_index, value)",
            "set_device_parameter(scope, index, device_index, parameter_index, value)",
            "load_effect_on_target(scope, index, uri)",
            "set_track_output_routing(track_index, output_type, output_channel?)",
            "apply_mix_template_sends(template_name|template_spec)"
        ]
    }

    return {
        "ok": True,
        "profile": profile_value,
        "topology": topology,
        "tags": {
            "explicit": mix_context_profile.get("explicit_tags") if isinstance(mix_context_profile, dict) else None,
            "inference_suggestions": mix_context_profile.get("inference_suggestions") if isinstance(mix_context_profile, dict) else None,
            "merged_roles": mix_context_profile.get("merged_roles") if isinstance(mix_context_profile, dict) else None
        },
        "session_snapshot": session_snapshot,
        "project_snapshot_summary": project_snapshot_summary,
        "automation_overview": automation_overview,
        "source_inventory": source_inventory if include_source_inventory else None,
        "export_analysis_request": export_analysis_request,
        "export_analysis": export_analysis,
        "stage_readiness": stages,
        "missing_data_actions": missing_actions,
        "future_mutation_api_spec": future_mutation_api_spec
    }


_ALS_EXHAUSTIVE_INVENTORY_CACHE_MAX_ENTRIES = 8
_ALS_EXHAUSTIVE_INVENTORY_CACHE: Dict[str, Dict[str, Any]] = {}
_ALS_EXHAUSTIVE_INVENTORY_CACHE_ORDER: List[str] = []


def _normalized_cache_float(value: Any) -> Optional[float]:
    parsed = _safe_float_value(value)
    if parsed is None:
        return None
    return round(float(parsed), 6)


def _als_exhaustive_inventory_cache_key(
    als_file_path: Optional[str],
    include_arrangement_clip_envelopes: bool,
    include_session_clip_envelopes: bool,
    start_time_beats: Optional[float],
    end_time_beats: Optional[float],
) -> Optional[str]:
    path_value = _safe_text_value(als_file_path)
    if not path_value:
        return None
    abs_path = os.path.abspath(os.path.expanduser(path_value))
    try:
        stat = os.stat(abs_path)
    except Exception:
        return None
    payload = {
        "kind": "als_exhaustive_inventory",
        "als_file_path": abs_path,
        "als_file_mtime_unix": round(float(stat.st_mtime), 6),
        "als_file_size_bytes": int(stat.st_size),
        "include_arrangement_clip_envelopes": bool(include_arrangement_clip_envelopes),
        "include_session_clip_envelopes": bool(include_session_clip_envelopes),
        "start_time_beats": _normalized_cache_float(start_time_beats),
        "end_time_beats": _normalized_cache_float(end_time_beats),
        "schema_version": 1,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def _cache_put_als_exhaustive_inventory(cache_key: str, payload: Dict[str, Any]) -> None:
    if not isinstance(cache_key, str) or not cache_key:
        return
    if not isinstance(payload, dict):
        return
    _ALS_EXHAUSTIVE_INVENTORY_CACHE[cache_key] = payload
    if cache_key in _ALS_EXHAUSTIVE_INVENTORY_CACHE_ORDER:
        _ALS_EXHAUSTIVE_INVENTORY_CACHE_ORDER.remove(cache_key)
    _ALS_EXHAUSTIVE_INVENTORY_CACHE_ORDER.append(cache_key)
    while len(_ALS_EXHAUSTIVE_INVENTORY_CACHE_ORDER) > _ALS_EXHAUSTIVE_INVENTORY_CACHE_MAX_ENTRIES:
        evicted = _ALS_EXHAUSTIVE_INVENTORY_CACHE_ORDER.pop(0)
        _ALS_EXHAUSTIVE_INVENTORY_CACHE.pop(evicted, None)


def _als_exhaustive_inventory_disk_cache_path(cache_key: str) -> Optional[str]:
    if not isinstance(cache_key, str) or not cache_key:
        return None
    try:
        analysis_dir = get_analysis_dir()
    except Exception:
        return None
    if not isinstance(analysis_dir, str) or not analysis_dir:
        return None
    cache_dir = os.path.join(analysis_dir, "automation_inventory_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        return None
    return os.path.join(cache_dir, f"{cache_key}.json")


def _cache_get_als_exhaustive_inventory_disk(cache_key: str) -> Optional[Dict[str, Any]]:
    cache_path = _als_exhaustive_inventory_disk_cache_path(cache_key)
    if not cache_path or not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _cache_put_als_exhaustive_inventory_disk(cache_key: str, payload: Dict[str, Any]) -> Optional[str]:
    cache_path = _als_exhaustive_inventory_disk_cache_path(cache_key)
    if not cache_path or not isinstance(payload, dict):
        return None
    try:
        temp_path = cache_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)
        os.replace(temp_path, cache_path)
        return cache_path
    except Exception:
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass
        return None


def _build_or_get_als_exhaustive_inventory(
    *,
    project_root: Optional[str],
    als_file_path: Optional[str],
    include_arrangement_clip_envelopes: bool,
    include_session_clip_envelopes: bool,
    start_time_beats: Optional[float],
    end_time_beats: Optional[float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    requested_cache_key = _als_exhaustive_inventory_cache_key(
        als_file_path=als_file_path,
        include_arrangement_clip_envelopes=include_arrangement_clip_envelopes,
        include_session_clip_envelopes=include_session_clip_envelopes,
        start_time_beats=start_time_beats,
        end_time_beats=end_time_beats,
    )
    if requested_cache_key and requested_cache_key in _ALS_EXHAUSTIVE_INVENTORY_CACHE:
        return _ALS_EXHAUSTIVE_INVENTORY_CACHE[requested_cache_key], {
            "cache_hit": True,
            "cache_key": requested_cache_key,
            "cache_layer": "process_memory",
        }
    if requested_cache_key:
        disk_payload = _cache_get_als_exhaustive_inventory_disk(requested_cache_key)
        if isinstance(disk_payload, dict):
            _cache_put_als_exhaustive_inventory(requested_cache_key, disk_payload)
            return disk_payload, {
                "cache_hit": True,
                "cache_key": requested_cache_key,
                "cache_layer": "analysis_disk",
            }

    inventory = build_als_automation_inventory(
        project_root=project_root,
        als_file_path=als_file_path,
        include_arrangement_clip_envelopes=include_arrangement_clip_envelopes,
        include_session_clip_envelopes=include_session_clip_envelopes,
        start_time_beats=start_time_beats,
        end_time_beats=end_time_beats,
    )
    if not isinstance(inventory, dict):
        return {
            "ok": False,
            "supported": False,
            "error": "invalid_inventory_payload",
            "message": "build_als_automation_inventory returned invalid payload",
        }, {"cache_hit": False}

    resolved_key = _als_exhaustive_inventory_cache_key(
        als_file_path=_safe_text_value(inventory.get("als_file_path")),
        include_arrangement_clip_envelopes=include_arrangement_clip_envelopes,
        include_session_clip_envelopes=include_session_clip_envelopes,
        start_time_beats=start_time_beats,
        end_time_beats=end_time_beats,
    )
    if resolved_key and inventory.get("supported") is True:
        _cache_put_als_exhaustive_inventory(resolved_key, inventory)
        _cache_put_als_exhaustive_inventory_disk(resolved_key, inventory)
    return inventory, {
        "cache_hit": False,
        "cache_key": resolved_key,
        "cache_layer": "process_memory" if resolved_key else None,
    }


def _parse_page_cursor(cursor: Optional[str]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    if cursor is None:
        return 0, None
    if isinstance(cursor, int):
        value = int(cursor)
    elif isinstance(cursor, str):
        text = cursor.strip()
        if not text:
            return 0, None
        if not re.fullmatch(r"\d+", text):
            return None, {
                "ok": False,
                "error": "invalid_cursor",
                "message": "cursor must be a non-negative integer string",
                "cursor": cursor,
            }
        value = int(text)
    else:
        return None, {
            "ok": False,
            "error": "invalid_cursor",
            "message": "cursor must be a string offset",
            "cursor": cursor,
        }
    if value < 0:
        return None, {
            "ok": False,
            "error": "invalid_cursor",
            "message": "cursor must be non-negative",
            "cursor": cursor,
        }
    return value, None


def _sanitize_exhaustive_inventory_row(row: Dict[str, Any], include_point_payloads: bool) -> Dict[str, Any]:
    out = copy.deepcopy(row) if isinstance(row, dict) else {}
    if include_point_payloads:
        return out
    out.pop("points", None)
    return out


def _build_exhaustive_runtime_hint_index(
    ctx: Context,
    page_rows: List[Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    track_indices_needed: List[int] = []
    seen_track_indices = set()
    for row in page_rows:
        if not isinstance(row, dict):
            continue
        if (_safe_text_value(row.get("container_scope")) or "") != "track":
            continue
        classification = row.get("classification") if isinstance(row.get("classification"), dict) else {}
        scope_value = _safe_text_value(classification.get("scope"))
        if scope_value not in {"device_parameter", "track_mixer"}:
            continue
        location = row.get("location") if isinstance(row.get("location"), dict) else {}
        track_index_value = _safe_int_value(location.get("track_index"))
        if track_index_value is None or track_index_value < 0 or track_index_value in seen_track_indices:
            continue
        seen_track_indices.add(track_index_value)
        track_indices_needed.append(track_index_value)

    runtime_index: Dict[int, Dict[str, Any]] = {}
    warnings: List[str] = []
    for track_index_value in track_indices_needed:
        try:
            payload = get_track_automation_targets(ctx, track_index_value)
        except Exception as exc:
            warnings.append(f"live_hint_track_query_exception:{track_index_value}")
            runtime_index[track_index_value] = {
                "ok": False,
                "error": "exception",
                "message": str(exc),
            }
            continue

        if not isinstance(payload, dict) or payload.get("ok") is not True:
            warnings.append(f"live_hint_track_query_failed:{track_index_value}")
            runtime_index[track_index_value] = payload if isinstance(payload, dict) else {
                "ok": False,
                "error": "invalid_track_automation_targets_payload",
            }
            continue

        devices_index: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for device in list(payload.get("devices") or []):
            if not isinstance(device, dict):
                continue
            device_index_value = _safe_int_value(device.get("device_index"))
            if device_index_value is None:
                continue
            for param in list(device.get("parameters") or []):
                if not isinstance(param, dict):
                    continue
                parameter_index_value = _safe_int_value(param.get("parameter_index"))
                if parameter_index_value is None:
                    continue
                devices_index[(int(device_index_value), int(parameter_index_value))] = param

        runtime_index[track_index_value] = {
            "ok": True,
            "track_name": payload.get("track_name"),
            "track_mixer_targets": payload.get("track_mixer_targets") if isinstance(payload.get("track_mixer_targets"), dict) else {},
            "device_parameters": devices_index,
        }
    return runtime_index, warnings


def _attach_exhaustive_live_hints_to_row(
    row: Dict[str, Any],
    runtime_index: Dict[int, Dict[str, Any]],
) -> None:
    if not isinstance(row, dict):
        return
    classification = row.get("classification") if isinstance(row.get("classification"), dict) else {}
    location = row.get("location") if isinstance(row.get("location"), dict) else {}
    container_scope = (_safe_text_value(row.get("container_scope")) or "").lower()
    classification_scope = (_safe_text_value(classification.get("scope")) or "").lower()

    def _set_hint(payload: Dict[str, Any]) -> None:
        row["live_hints"] = payload

    if container_scope != "track":
        _set_hint({
            "available": False,
            "mapping_confidence": "none",
            "unsaved_risk": "unknown",
            "reason": "non_track_runtime_hint_overlay_not_implemented",
        })
        return

    track_index_value = _safe_int_value(location.get("track_index"))
    if track_index_value is None:
        _set_hint({
            "available": False,
            "mapping_confidence": "none",
            "unsaved_risk": "unknown",
            "reason": "track_index_unavailable",
        })
        return

    runtime_track = runtime_index.get(track_index_value)
    if not isinstance(runtime_track, dict) or runtime_track.get("ok") is not True:
        _set_hint({
            "available": False,
            "mapping_confidence": "none",
            "unsaved_risk": "unknown",
            "reason": "track_runtime_automation_targets_unavailable",
        })
        return

    if classification_scope == "track_mixer":
        mixer_target = _safe_text_value(classification.get("mixer_target"))
        track_mixer_targets = runtime_track.get("track_mixer_targets") if isinstance(runtime_track.get("track_mixer_targets"), dict) else {}
        mixer_payload = track_mixer_targets.get(mixer_target) if isinstance(track_mixer_targets, dict) and isinstance(mixer_target, str) else None
        if not isinstance(mixer_payload, dict):
            _set_hint({
                "available": False,
                "mapping_confidence": "none",
                "unsaved_risk": "unknown",
                "reason": "track_mixer_runtime_hint_missing",
            })
            return
        _set_hint({
            "available": bool(mixer_payload.get("supported", False)) if "supported" in mixer_payload else False,
            "mapping_confidence": "high",
            "unsaved_risk": "unknown",
            "automation_state": mixer_payload.get("automation_state"),
            "automated": mixer_payload.get("automated"),
            "reason": mixer_payload.get("reason"),
        })
        return

    if classification_scope == "device_parameter":
        legacy_device_index = _safe_int_value(classification.get("legacy_top_level_device_index"))
        legacy_parameter_index = _safe_int_value(classification.get("legacy_top_level_parameter_index"))
        if legacy_device_index is None or legacy_parameter_index is None:
            _set_hint({
                "available": False,
                "mapping_confidence": "none",
                "unsaved_risk": "unknown",
                "reason": "legacy_top_level_mapping_unavailable",
            })
            return
        param_payload = (runtime_track.get("device_parameters") or {}).get((int(legacy_device_index), int(legacy_parameter_index)))
        if not isinstance(param_payload, dict):
            _set_hint({
                "available": False,
                "mapping_confidence": "low",
                "unsaved_risk": "unknown",
                "reason": "runtime_parameter_not_found",
                "legacy_top_level_device_index": legacy_device_index,
                "legacy_top_level_parameter_index": legacy_parameter_index,
            })
            return
        _set_hint({
            "available": True,
            "mapping_confidence": "high",
            "unsaved_risk": "unknown",
            "automation_state": param_payload.get("automation_state"),
            "automated": param_payload.get("automated"),
            "parameter_index": param_payload.get("parameter_index"),
            "parameter_name": param_payload.get("name"),
            "legacy_top_level_device_index": legacy_device_index,
            "legacy_top_level_parameter_index": legacy_parameter_index,
        })
        return

    _set_hint({
        "available": False,
        "mapping_confidence": "none",
        "unsaved_risk": "unknown",
        "reason": "runtime_hint_not_supported_for_scope",
        "classification_scope": classification_scope or None,
    })


def _attach_exhaustive_live_hints(
    ctx: Context,
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    runtime_index, warnings = _build_exhaustive_runtime_hint_index(ctx, rows)
    for row in rows:
        _attach_exhaustive_live_hints_to_row(row, runtime_index)
    return rows, warnings


@mcp.tool()
def enumerate_project_automation_exhaustive(
    ctx: Context,
    als_file_path: Optional[str] = None,
    include_point_payloads: bool = False,
    include_live_hints: bool = True,
    include_arrangement_clip_envelopes: bool = True,
    include_session_clip_envelopes: bool = True,
    include_unclassified: bool = True,
    include_orphans: bool = True,
    start_time_beats: Optional[float] = None,
    end_time_beats: Optional[float] = None,
    page_size: int = 500,
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Exhaustive, ALS-first saved-state automation inventory (paged).

    This tool is the canonical exact inventory for automation visibility in the current milestone.

    Parameters:
    - include_arrangement_clip_envelopes: include arrangement clip envelope rows
    - include_session_clip_envelopes: include session clip envelope rows
    """
    _ = ctx
    page_size_value = _safe_int_value(page_size)
    if page_size_value is None:
        page_size_value = 500
    page_size_value = max(1, min(int(page_size_value), 5000))

    cursor_offset, cursor_error = _parse_page_cursor(cursor)
    if cursor_error is not None:
        return cursor_error
    cursor_offset = int(cursor_offset or 0)

    inventory, cache_meta = _build_or_get_als_exhaustive_inventory(
        project_root=get_project_root(),
        als_file_path=_safe_text_value(als_file_path),
        include_arrangement_clip_envelopes=bool(include_arrangement_clip_envelopes),
        include_session_clip_envelopes=bool(include_session_clip_envelopes),
        start_time_beats=_safe_float_value(start_time_beats),
        end_time_beats=_safe_float_value(end_time_beats),
    )
    if not isinstance(inventory, dict):
        return {
            "ok": False,
            "error": "invalid_inventory_payload",
            "message": "ALS automation inventory builder returned invalid payload",
        }

    all_targets = list(inventory.get("targets") or []) if isinstance(inventory.get("targets"), list) else []
    total_targets = len(all_targets)
    start_index = min(cursor_offset, total_targets)
    end_index = min(start_index + page_size_value, total_targets)
    page_rows = [
        _sanitize_exhaustive_inventory_row(row, include_point_payloads=bool(include_point_payloads))
        for row in all_targets[start_index:end_index]
        if isinstance(row, dict)
    ]

    warnings = list(inventory.get("warnings") or []) if isinstance(inventory.get("warnings"), list) else []
    if bool(include_live_hints):
        try:
            page_rows, live_warnings = _attach_exhaustive_live_hints(ctx, page_rows)
            for warning in live_warnings:
                if isinstance(warning, str) and warning not in warnings:
                    warnings.append(warning)
        except Exception as exc:
            if "exhaustive_live_hint_overlay_failed" not in warnings:
                warnings.append("exhaustive_live_hint_overlay_failed")
            warnings.append(f"exhaustive_live_hint_overlay_error:{str(exc)}")

    next_cursor = str(end_index) if end_index < total_targets else None
    page_payload = {
        "page_size": page_size_value,
        "cursor": str(start_index),
        "next_cursor": next_cursor,
        "returned_targets": len(page_rows),
        "total_targets": total_targets,
    }

    out: Dict[str, Any] = {
        "ok": bool(inventory.get("ok", False)),
        "supported": bool(inventory.get("supported", False)) if "supported" in inventory else False,
        "schema_version": inventory.get("schema_version"),
        "source": inventory.get("source", "als_file"),
        "als_file_path": inventory.get("als_file_path"),
        "als_file_mtime_utc": inventory.get("als_file_mtime_utc"),
        "scope_statement": inventory.get("scope_statement"),
        "session": inventory.get("session"),
        "completeness": inventory.get("completeness"),
        "page": page_payload,
        "targets": page_rows,
        "warnings": warnings,
    }
    if "reason" in inventory:
        out["reason"] = inventory.get("reason")
    if bool(include_orphans):
        out["orphan_envelopes"] = list(inventory.get("orphan_envelopes") or []) if isinstance(inventory.get("orphan_envelopes"), list) else []
    if bool(include_unclassified):
        out["unclassified_targets"] = list(inventory.get("unclassified_targets") or []) if isinstance(inventory.get("unclassified_targets"), list) else []
    if isinstance(inventory.get("duplicate_target_id_rows"), list):
        out["duplicate_target_id_rows"] = list(inventory.get("duplicate_target_id_rows") or [])
    if isinstance(cache_meta, dict):
        out["cache"] = {
            "cache_hit": bool(cache_meta.get("cache_hit", False)),
            "cache_layer": cache_meta.get("cache_layer"),
        }
    if bool(include_live_hints):
        out["live_hints_included"] = True
    return out


@mcp.tool()
def get_automation_target_points(
    ctx: Context,
    target_ref: Dict[str, Any],
    als_file_path: Optional[str] = None,
    include_live_hints: bool = False,
    start_time_beats: Optional[float] = None,
    end_time_beats: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return exact saved automation points for one target discovered by enumerate_project_automation_exhaustive.

    Parameters:
    - target_ref: canonical target reference object from enumerate_project_automation_exhaustive
    - als_file_path: optional explicit .als file path
    - include_live_hints: include best-effort runtime hints (top-level normal-track mappings only)
    - start_time_beats/end_time_beats: optional beat-range filter applied during ALS inventory build
    """
    if not isinstance(target_ref, dict):
        return {
            "ok": False,
            "error": "invalid_target_ref",
            "message": "target_ref must be an object"
        }

    inventory, cache_meta = _build_or_get_als_exhaustive_inventory(
        project_root=get_project_root(),
        als_file_path=_safe_text_value(als_file_path),
        include_arrangement_clip_envelopes=True,
        include_session_clip_envelopes=True,
        start_time_beats=_safe_float_value(start_time_beats),
        end_time_beats=_safe_float_value(end_time_beats),
    )
    if not isinstance(inventory, dict):
        return {
            "ok": False,
            "error": "invalid_inventory_payload",
            "message": "ALS automation inventory builder returned invalid payload",
        }
    if inventory.get("supported") is not True:
        return {
            "ok": True,
            "supported": False,
            "reason": inventory.get("reason") or "als_inventory_unavailable",
            "target_ref": target_ref,
            "als_file_path": inventory.get("als_file_path"),
            "warnings": list(inventory.get("warnings") or []) if isinstance(inventory.get("warnings"), list) else [],
        }

    resolved = get_als_automation_target_points_from_inventory(inventory, target_ref)
    if not isinstance(resolved, dict):
        return {
            "ok": False,
            "error": "invalid_target_resolution_payload",
            "message": "ALS target resolution helper returned invalid payload",
        }

    out = dict(resolved)
    out.setdefault("source", "als_inventory")
    out["als_file_path"] = inventory.get("als_file_path")
    out["als_file_mtime_utc"] = inventory.get("als_file_mtime_utc")
    out["scope_statement"] = inventory.get("scope_statement")
    if isinstance(cache_meta, dict):
        out["cache"] = {
            "cache_hit": bool(cache_meta.get("cache_hit", False)),
            "cache_layer": cache_meta.get("cache_layer"),
        }

    if bool(include_live_hints) and out.get("ok") is True:
        row_like = {
            "container_scope": out.get("container_scope"),
            "classification": out.get("classification"),
            "location": out.get("location"),
        }
        runtime_index, live_warnings = _build_exhaustive_runtime_hint_index(ctx, [row_like])
        _attach_exhaustive_live_hints_to_row(row_like, runtime_index)
        out["live_hints"] = row_like.get("live_hints")
        if live_warnings:
            warnings = list(out.get("warnings") or []) if isinstance(out.get("warnings"), list) else []
            for warning in live_warnings:
                if isinstance(warning, str) and warning not in warnings:
                    warnings.append(warning)
            out["warnings"] = warnings
        out["live_hints_included"] = True

    return out

# Main execution
def main():
    """Run the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()

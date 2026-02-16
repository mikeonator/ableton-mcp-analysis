# ableton_mcp_server.py
from mcp.server.fastmcp import FastMCP, Context
import socket
import json
import logging
import os
import hashlib
import math
from datetime import datetime, timezone
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Union, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    import soundfile as sf
except Exception:
    sf = None

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
            "start_playback", "stop_playback", "load_instrument_or_effect"
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

_SOURCE_CACHE_VERSION = 1
_SOURCE_CACHE_DIR = os.path.expanduser("~/.ableton_mcp_analysis/cache")
_SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".aiff", ".aif", ".mp3", ".m4a", ".flac"}

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


def _utc_now_iso() -> str:
    """Return UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _normalize_source_path(file_path: str) -> str:
    """Expand and normalize a source file path."""
    return os.path.abspath(os.path.expanduser(file_path))


def _ensure_cache_dir() -> None:
    """Ensure source analysis cache directory exists."""
    os.makedirs(_SOURCE_CACHE_DIR, exist_ok=True)


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


def _decode_audio_samples(file_path: str) -> Tuple[Any, int, int]:
    """Decode audio and return mono samples, sample rate, and original channels."""
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
            mono = decoded.mean(axis=1).astype(np.float32)
            return mono, int(sample_rate), channels
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
                mono = pcm.mean(axis=1)
            else:
                mono = pcm

            full_scale = float(2 ** max(1, (8 * sample_width - 1)))
            mono = (mono / full_scale).astype(np.float32)
            return mono, sample_rate, channels
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


def _find_resonant_peaks(frequencies: Any, spectrum_db: Any) -> List[Dict[str, float]]:
    """Find top local spectral spikes above a smoothed baseline."""
    if np is None or len(spectrum_db) < 5:
        return []

    kernel_size = max(9, int(len(spectrum_db) / 256))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    baseline = np.convolve(spectrum_db, kernel, mode="same")
    prominence = spectrum_db - baseline

    candidates = []
    for idx in range(1, len(spectrum_db) - 1):
        hz = float(frequencies[idx])
        if hz < 20.0 or hz > 12000.0:
            continue
        if spectrum_db[idx] <= spectrum_db[idx - 1] or spectrum_db[idx] <= spectrum_db[idx + 1]:
            continue
        if prominence[idx] < 2.5:
            continue
        candidates.append((idx, float(prominence[idx])))

    candidates.sort(key=lambda item: item[1], reverse=True)

    peaks: List[Dict[str, float]] = []
    for idx, prom in candidates:
        hz = float(frequencies[idx])
        if any(abs(existing["hz"] - hz) < 60.0 for existing in peaks):
            continue
        peaks.append({
            "hz": round(hz, 2),
            "prominence_db": round(prom, 2)
        })
        if len(peaks) >= 5:
            break

    return peaks


def _format_frequency(hz: float) -> str:
    """Format frequency in Hz or kHz for summaries."""
    if hz >= 1000.0:
        return f"{hz / 1000.0:.1f} kHz"
    return f"{hz:.0f} Hz"


def _build_source_summary(band_energies_db: Dict[str, float], peaks: List[Dict[str, float]]) -> str:
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

    if peaks:
        top_peak = peaks[0]
        descriptors.append(f"resonance around {_format_frequency(float(top_peak['hz']))}")

    summary = ", ".join(descriptors)
    if not summary.endswith("."):
        summary += "."
    return summary


def _analyze_audio_source(file_path: str, stat_size: int, stat_mtime: float) -> Dict[str, Any]:
    """Run a deterministic spectral analysis over decoded audio."""
    if np is None:
        raise SourceAnalysisError("unsupported_decode_backend", "numpy is not available")

    mono, sample_rate, channels = _decode_audio_samples(file_path)
    if mono.size == 0:
        raise SourceAnalysisError("decode_failed", "Decoded audio is empty")

    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    crest_factor_db = float(20.0 * math.log10(max(peak, 1e-12) / max(rms, 1e-12)))

    # Analyze at most 120 seconds for predictable runtime on large files.
    max_samples = int(sample_rate * 120)
    analysis_signal = mono[:max_samples] if mono.shape[0] > max_samples else mono
    fft_size = int(analysis_signal.shape[0])
    if fft_size < 32:
        raise SourceAnalysisError("decode_failed", "Audio too short for spectral analysis")

    window = np.hanning(fft_size).astype(np.float32)
    windowed = analysis_signal * window
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / float(sample_rate))
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
    summary = _build_source_summary(band_energies_db, resonant_peaks_hz)

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
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "crest_factor_db": round(crest_factor_db, 3),
        "band_energies_db": band_energies_db,
        "resonant_peaks_hz": resonant_peaks_hz,
        "summary": summary
    }


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

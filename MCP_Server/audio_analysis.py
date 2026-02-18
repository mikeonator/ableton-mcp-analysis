"""Chunked WAV analysis helpers for export-based signal awareness."""

from __future__ import annotations

import math
import os
import wave
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency guard
    np = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency guard
    sf = None


_ANALYSIS_SCHEMA_VERSION = 1
_MIN_DBFS = -120.0
_MAX_SPECTRAL_SECONDS = 120.0
_DEFAULT_CHUNK_FRAMES = 65536


class AudioAnalysisError(Exception):
    """Structured error raised by audio analysis helpers."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _safe_db(value: float, floor_db: float = _MIN_DBFS) -> float:
    """Convert linear amplitude to dBFS with floor protection."""
    if value <= 0.0:
        return float(floor_db)
    db_value = 20.0 * math.log10(value)
    return max(float(floor_db), float(db_value))


def _band_energy_db(spectrum: Any, frequencies: Any, low_hz: float, high_hz: float) -> float:
    """Compute mean magnitude in a frequency band."""
    if np is None:
        return _MIN_DBFS

    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not np.any(mask):
        return _MIN_DBFS
    band_mag = float(np.mean(spectrum[mask]))
    return _safe_db(max(band_mag, 1e-12))


def _average_spectrum_welch(signal_mono: Any, sample_rate: int) -> Tuple[Any, Any]:
    """Build a simple Welch-style averaged magnitude spectrum."""
    if np is None:
        raise AudioAnalysisError("numpy_unavailable", "numpy is required for spectral analysis")

    frame_size = 4096
    hop = frame_size // 2
    if signal_mono.shape[0] < frame_size:
        signal_mono = np.pad(signal_mono, (0, frame_size - signal_mono.shape[0]))

    window = np.hanning(frame_size).astype(np.float32)
    spectra = []
    for start in range(0, signal_mono.shape[0] - frame_size + 1, hop):
        frame = signal_mono[start:start + frame_size]
        if frame.shape[0] < frame_size:
            continue
        spectra.append(np.abs(np.fft.rfft(frame * window)))

    if not spectra:
        spectra = [np.abs(np.fft.rfft(signal_mono[:frame_size] * window))]

    stacked = np.stack(spectra, axis=0)
    avg_spectrum = np.mean(stacked, axis=0)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / float(sample_rate))
    return frequencies, avg_spectrum


def _decode_int24_le(raw_bytes: bytes, channels: int) -> Any:
    """Decode little-endian packed 24-bit PCM into float32 NxC samples."""
    if np is None:
        raise AudioAnalysisError("numpy_unavailable", "numpy is required for waveform decoding")
    byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    if byte_array.size % (channels * 3) != 0:
        byte_array = byte_array[: byte_array.size - (byte_array.size % (channels * 3))]
    if byte_array.size == 0:
        return np.zeros((0, channels), dtype=np.float32)

    triplets = byte_array.reshape((-1, 3))
    values = (
        triplets[:, 0].astype(np.int32)
        | (triplets[:, 1].astype(np.int32) << 8)
        | (triplets[:, 2].astype(np.int32) << 16)
    )
    sign_bit = 1 << 23
    values = (values ^ sign_bit) - sign_bit
    normalized = (values.astype(np.float32) / float(1 << 23)).reshape((-1, channels))
    return normalized


def _pcm_bytes_to_float32(raw_bytes: bytes, channels: int, sample_width: int) -> Any:
    """Convert PCM bytes to float32 NxC samples."""
    if np is None:
        raise AudioAnalysisError("numpy_unavailable", "numpy is required for waveform decoding")

    if sample_width == 1:
        data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        return _decode_int24_le(raw_bytes=raw_bytes, channels=channels)
    elif sample_width == 4:
        data = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioAnalysisError(
            "unsupported_wav_encoding",
            f"Unsupported sample width: {sample_width} bytes"
        )

    if channels <= 0:
        raise AudioAnalysisError("invalid_wav_channels", "WAV channel count must be > 0")
    frame_count = data.size // channels
    if frame_count <= 0:
        return np.zeros((0, channels), dtype=np.float32)
    usable = data[: frame_count * channels]
    return usable.reshape((frame_count, channels)).astype(np.float32)


def _iter_wav_frames_soundfile(
    wav_path: str,
    start_time_sec: float,
    duration_sec: Optional[float],
    chunk_frames: int
) -> Tuple[int, int, int, Iterator[Tuple[int, Any]]]:
    """Return WAV metadata + iterator using soundfile backend."""
    if sf is None:
        raise AudioAnalysisError("soundfile_unavailable", "soundfile backend not available")

    audio = sf.SoundFile(wav_path, mode="r")
    sample_rate = int(audio.samplerate)
    channels = int(audio.channels)
    total_frames = int(len(audio))

    start_frame = max(0, int(round(start_time_sec * float(sample_rate))))
    if start_frame > total_frames:
        start_frame = total_frames

    if duration_sec is None:
        segment_frames = total_frames - start_frame
    else:
        requested_frames = max(0, int(round(duration_sec * float(sample_rate))))
        segment_frames = min(requested_frames, max(0, total_frames - start_frame))

    audio.seek(start_frame)

    def _generator() -> Iterator[Tuple[int, Any]]:
        read_frames = 0
        try:
            while read_frames < segment_frames:
                frames_to_read = min(chunk_frames, segment_frames - read_frames)
                block = audio.read(frames=frames_to_read, dtype="float32", always_2d=True)
                if block is None or block.size == 0:
                    break
                block = block.astype(np.float32) if np is not None else block
                yield read_frames, block
                read_frames += int(block.shape[0])
        finally:
            audio.close()

    return sample_rate, channels, segment_frames, _generator()


def _iter_wav_frames_wave(
    wav_path: str,
    start_time_sec: float,
    duration_sec: Optional[float],
    chunk_frames: int
) -> Tuple[int, int, int, Iterator[Tuple[int, Any]]]:
    """Return WAV metadata + iterator using stdlib wave backend."""
    if np is None:
        raise AudioAnalysisError("numpy_unavailable", "numpy is required for waveform analysis")

    wf = wave.open(wav_path, "rb")
    sample_rate = int(wf.getframerate())
    channels = int(wf.getnchannels())
    total_frames = int(wf.getnframes())
    sample_width = int(wf.getsampwidth())

    start_frame = max(0, int(round(start_time_sec * float(sample_rate))))
    if start_frame > total_frames:
        start_frame = total_frames

    if duration_sec is None:
        segment_frames = total_frames - start_frame
    else:
        requested_frames = max(0, int(round(duration_sec * float(sample_rate))))
        segment_frames = min(requested_frames, max(0, total_frames - start_frame))

    wf.setpos(start_frame)

    def _generator() -> Iterator[Tuple[int, Any]]:
        read_frames = 0
        try:
            while read_frames < segment_frames:
                frames_to_read = min(chunk_frames, segment_frames - read_frames)
                raw = wf.readframes(frames_to_read)
                if not raw:
                    break
                block = _pcm_bytes_to_float32(
                    raw_bytes=raw,
                    channels=channels,
                    sample_width=sample_width
                )
                if block.size == 0:
                    break
                yield read_frames, block
                read_frames += int(block.shape[0])
        finally:
            wf.close()

    return sample_rate, channels, segment_frames, _generator()


def _iter_wav_frames(
    wav_path: str,
    start_time_sec: float,
    duration_sec: Optional[float],
    chunk_frames: int
) -> Tuple[int, int, int, Iterator[Tuple[int, Any]], str]:
    """Select best WAV decode backend and return stream iterator."""
    if sf is not None:
        try:
            sr, ch, seg_frames, generator = _iter_wav_frames_soundfile(
                wav_path=wav_path,
                start_time_sec=start_time_sec,
                duration_sec=duration_sec,
                chunk_frames=chunk_frames
            )
            return sr, ch, seg_frames, generator, "soundfile"
        except Exception:
            pass

    sr, ch, seg_frames, generator = _iter_wav_frames_wave(
        wav_path=wav_path,
        start_time_sec=start_time_sec,
        duration_sec=duration_sec,
        chunk_frames=chunk_frames
    )
    return sr, ch, seg_frames, generator, "wave"


def _window_metrics(window_samples: Any, rms_threshold_db: float) -> Dict[str, Any]:
    """Compute RMS/peak metrics for one analysis window."""
    mono = window_samples.mean(axis=1).astype(np.float32)
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    peak = float(np.max(np.abs(window_samples)))
    rms_dbfs = _safe_db(rms)
    peak_dbfs = _safe_db(peak)
    silent = bool(rms_dbfs < float(rms_threshold_db))
    return {
        "rms": rms,
        "peak": peak,
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "silent": silent,
        "mono": mono
    }


def _empty_metrics() -> Dict[str, Any]:
    """Return default metrics for silent/empty analysis."""
    return {
        "overall_peak_dbfs": _MIN_DBFS,
        "overall_rms_dbfs": _MIN_DBFS,
        "crest_factor": 0.0,
        "spectral_bands": {
            "low": _MIN_DBFS,
            "low_mid": _MIN_DBFS,
            "high_mid": _MIN_DBFS,
            "high": _MIN_DBFS
        }
    }


def analyze_wav_file(
    wav_path: str,
    window_sec: float = 1.0,
    rms_threshold_db: float = -60.0,
    start_time_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    auto_trim_silence: bool = True,
    chunk_frames: int = _DEFAULT_CHUNK_FRAMES,
    spectral_cap_sec: float = _MAX_SPECTRAL_SECONDS
) -> Dict[str, Any]:
    """Analyze a WAV file using chunked window metrics + coarse spectral summary."""
    if np is None:
        raise AudioAnalysisError("numpy_unavailable", "numpy is required for waveform analysis")
    if not isinstance(wav_path, str) or not wav_path.strip():
        raise AudioAnalysisError("invalid_wav_path", "wav_path must be a non-empty string")
    if window_sec <= 0.0:
        raise AudioAnalysisError("invalid_window_sec", "window_sec must be > 0")
    if duration_sec is not None and duration_sec <= 0.0:
        raise AudioAnalysisError("invalid_duration_sec", "duration_sec must be > 0 when provided")

    path = wav_path.strip()
    if not path.lower().endswith(".wav"):
        raise AudioAnalysisError("invalid_wav_path", "Only WAV files are supported in analyze_wav_file")
    if not os.path.exists(path):
        raise AudioAnalysisError("file_not_found", "WAV file does not exist")

    resolved_start_time_sec = max(0.0, float(start_time_sec or 0.0))
    sr, channels, segment_frames, frame_iter, backend = _iter_wav_frames(
        wav_path=path,
        start_time_sec=resolved_start_time_sec,
        duration_sec=duration_sec,
        chunk_frames=max(1024, int(chunk_frames))
    )
    if segment_frames <= 0:
        raise AudioAnalysisError("empty_segment", "Selected segment contains no audio frames")

    window_frames = max(1, int(round(float(window_sec) * float(sr))))
    spectral_cap_frames = max(32, int(round(float(spectral_cap_sec) * float(sr))))

    carry = np.zeros((0, channels), dtype=np.float32)
    windows: List[Dict[str, Any]] = []
    first_signal_time_sec: Optional[float] = None
    detected_audio = False
    full_overall_peak = 0.0

    metrics_sum_sq = 0.0
    metrics_sample_count = 0
    metrics_peak = 0.0
    spectral_chunks: List[Any] = []
    spectral_frames_collected = 0

    def _accumulate_metrics(window_info: Dict[str, Any]) -> None:
        nonlocal metrics_sum_sq, metrics_sample_count, metrics_peak
        nonlocal spectral_frames_collected
        mono = window_info["mono"]
        metrics_sum_sq += float(np.sum(np.square(mono, dtype=np.float64)))
        metrics_sample_count += int(mono.shape[0])
        metrics_peak = max(metrics_peak, float(window_info["peak"]))

        if spectral_frames_collected >= spectral_cap_frames:
            return
        remaining = spectral_cap_frames - spectral_frames_collected
        take = min(remaining, int(mono.shape[0]))
        if take > 0:
            spectral_chunks.append(mono[:take].astype(np.float32))
            spectral_frames_collected += take

    def _process_window(window_samples: Any, start_sample: int) -> None:
        nonlocal first_signal_time_sec, detected_audio, full_overall_peak
        info = _window_metrics(window_samples=window_samples, rms_threshold_db=float(rms_threshold_db))
        full_overall_peak = max(full_overall_peak, float(info["peak"]))
        window_start_sec = float(start_sample) / float(sr)

        windows.append({
            "start_sec": round(window_start_sec, 6),
            "rms_dbfs": round(float(info["rms_dbfs"]), 3),
            "peak_dbfs": round(float(info["peak_dbfs"]), 3),
            "silent": bool(info["silent"])
        })

        if (not info["silent"]) and first_signal_time_sec is None:
            first_signal_time_sec = window_start_sec
            detected_audio = True

        collect_for_metrics = True
        if auto_trim_silence:
            collect_for_metrics = detected_audio
        if collect_for_metrics:
            _accumulate_metrics(info)

    processed_frames = 0
    for chunk_offset, chunk in frame_iter:
        if chunk is None or chunk.size == 0:
            continue
        if chunk.shape[1] != channels:
            raise AudioAnalysisError("decode_failed", "Decoder returned inconsistent channel count")

        buffer = np.concatenate([carry, chunk], axis=0)
        carry_start_sample = processed_frames - int(carry.shape[0])
        cursor = 0
        while (buffer.shape[0] - cursor) >= window_frames:
            window = buffer[cursor:cursor + window_frames]
            start_sample = carry_start_sample + cursor
            _process_window(window_samples=window, start_sample=start_sample)
            cursor += window_frames
        carry = buffer[cursor:]
        processed_frames = chunk_offset + int(chunk.shape[0])

    if carry.shape[0] > 0:
        start_sample = processed_frames - int(carry.shape[0])
        _process_window(window_samples=carry, start_sample=start_sample)

    if metrics_sample_count <= 0:
        metrics = _empty_metrics()
    else:
        metrics_rms = math.sqrt(metrics_sum_sq / float(metrics_sample_count))
        crest_factor = float(metrics_peak) / max(metrics_rms, 1e-12)

        if spectral_chunks:
            spectral_signal = np.concatenate(spectral_chunks, axis=0).astype(np.float32)
            frequencies, spectrum = _average_spectrum_welch(spectral_signal, sr)
            spectral_bands = {
                "low": round(_band_energy_db(spectrum, frequencies, 20.0, 150.0), 3),
                "low_mid": round(_band_energy_db(spectrum, frequencies, 150.0, 500.0), 3),
                "high_mid": round(_band_energy_db(spectrum, frequencies, 500.0, 2000.0), 3),
                "high": round(_band_energy_db(spectrum, frequencies, 2000.0, 16000.0), 3)
            }
        else:
            spectral_bands = _empty_metrics()["spectral_bands"]

        metrics = {
            "overall_peak_dbfs": round(_safe_db(float(metrics_peak)), 3),
            "overall_rms_dbfs": round(_safe_db(float(metrics_rms)), 3),
            "crest_factor": round(float(crest_factor), 4),
            "spectral_bands": spectral_bands
        }

    return {
        "ok": True,
        "schema_version": _ANALYSIS_SCHEMA_VERSION,
        "wav_path": path,
        "sample_rate": int(sr),
        "channels": int(channels),
        "segment_start_time_sec": round(float(resolved_start_time_sec), 6),
        "segment_duration_sec": round(float(segment_frames) / float(sr), 6),
        "window_sec": round(float(window_sec), 6),
        "rms_threshold_db": round(float(rms_threshold_db), 3),
        "windows": windows,
        "first_signal_time_sec": None if first_signal_time_sec is None else round(first_signal_time_sec, 6),
        "detected_audio": bool(detected_audio),
        "overall_peak_dbfs": round(_safe_db(float(full_overall_peak)), 3),
        "metrics": metrics,
        "analysis_notes": [
            f"decode_backend={backend}",
            f"auto_trim_silence={bool(auto_trim_silence)}"
        ],
    }

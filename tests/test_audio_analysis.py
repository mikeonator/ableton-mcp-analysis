import math
import os
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from MCP_Server.audio_analysis import analyze_wav_file
from MCP_Server import server


def _write_wav_mono_16bit(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _write_wav_stereo_16bit(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    if clipped.ndim != 2 or clipped.shape[1] != 2:
        raise ValueError("samples must be Nx2")
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.reshape((-1,)).tobytes())


class AnalyzeWavFileTests(unittest.TestCase):
    def test_detects_first_signal_with_leading_silence(self) -> None:
        sr = 48000
        duration_sec = 1.0
        t = np.arange(int(sr * duration_sec), dtype=np.float32) / float(sr)
        tone = 0.2 * np.sin(2.0 * math.pi * 440.0 * t)
        leading = np.zeros(int(sr * 0.5), dtype=np.float32)
        trailing = tone[: int(sr * 0.5)]
        samples = np.concatenate([leading, trailing], axis=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "late_start.wav"
            _write_wav_mono_16bit(wav_path, samples, sr)

            result = analyze_wav_file(
                wav_path=str(wav_path),
                window_sec=0.25,
                rms_threshold_db=-50.0,
                auto_trim_silence=True
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["detected_audio"])
        self.assertIsNotNone(result["first_signal_time_sec"])
        self.assertAlmostEqual(result["first_signal_time_sec"], 0.5, places=2)
        self.assertGreater(result["metrics"]["overall_peak_dbfs"], -20.0)
        self.assertGreater(result["metrics"]["overall_rms_dbfs"], -35.0)

    def test_reports_silence_for_silent_wav(self) -> None:
        sr = 48000
        samples = np.zeros(sr, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "silent.wav"
            _write_wav_mono_16bit(wav_path, samples, sr)

            result = analyze_wav_file(
                wav_path=str(wav_path),
                window_sec=0.25,
                rms_threshold_db=-60.0,
                auto_trim_silence=True
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["detected_audio"])
        self.assertIsNone(result["first_signal_time_sec"])
        self.assertEqual(result["metrics"]["overall_peak_dbfs"], -120.0)
        self.assertEqual(result["metrics"]["overall_rms_dbfs"], -120.0)
        self.assertTrue(all(window["silent"] for window in result["windows"]))

    def test_analyze_audio_file_decodes_mp3_and_preserves_spectral_balance(self) -> None:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            self.skipTest("ffmpeg not installed")

        sr = 48000
        duration_sec = 2.0
        t = np.arange(int(sr * duration_sec), dtype=np.float32) / float(sr)
        samples = (
            0.25 * np.sin(2.0 * math.pi * 80.0 * t)
            + 0.20 * np.sin(2.0 * math.pi * 300.0 * t)
            + 0.15 * np.sin(2.0 * math.pi * 1200.0 * t)
            + 0.10 * np.sin(2.0 * math.pi * 6000.0 * t)
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "reference.wav"
            mp3_path = Path(tmpdir) / "reference.mp3"
            _write_wav_mono_16bit(wav_path, samples, sr)

            completed = subprocess.run(
                [
                    ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(wav_path),
                    str(mp3_path),
                ],
                capture_output=True,
                text=True,
                timeout=60.0,
            )
            if completed.returncode != 0:
                self.fail(f"ffmpeg encode failed: {(completed.stderr or '').strip()}")

            with mock.patch.dict(
                os.environ,
                {
                    "PROJECT_ROOT": tmpdir,
                    "EXPORT_ROOT_MODE": "project",
                    "EXPORT_REL_DIR": "AbletonMCP/exports",
                    "ANALYSIS_REL_DIR": "AbletonMCP/analysis",
                },
                clear=False,
            ):
                wav_result = server.analyze_audio_file(None, str(wav_path), window_sec=0.25)
                mp3_result = server.analyze_audio_file(None, str(mp3_path), window_sec=0.25)

            self.assertTrue(wav_result["ok"])
            self.assertTrue(mp3_result["ok"])
            self.assertEqual(mp3_result.get("input_format"), "mp3")
            self.assertEqual(mp3_result.get("original_path"), str(mp3_path))
            self.assertTrue(isinstance(mp3_result.get("decoded_wav_path"), str))
            self.assertTrue(os.path.exists(mp3_result["decoded_wav_path"]))

            wav_bands = wav_result["metrics"]["spectral_bands"]
            mp3_bands = mp3_result["metrics"]["spectral_bands"]
            keys = ["low", "low_mid", "high_mid", "high"]

            def _norm_bands_db(bands):
                linear = [10.0 ** (float(bands[key]) / 20.0) for key in keys]
                total = max(sum(linear), 1e-12)
                return [value / total for value in linear]

            wav_norm = _norm_bands_db(wav_bands)
            mp3_norm = _norm_bands_db(mp3_bands)
            for wav_share, mp3_share in zip(wav_norm, mp3_norm):
                self.assertLess(abs(wav_share - mp3_share), 0.20)

    def test_analyze_audio_file_returns_ffmpeg_not_found_for_non_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mp3_path = Path(tmpdir) / "stub.mp3"
            mp3_path.write_bytes(b"ID3stub")

            with mock.patch.dict(
                os.environ,
                {
                    "PROJECT_ROOT": tmpdir,
                    "EXPORT_ROOT_MODE": "project",
                    "EXPORT_REL_DIR": "AbletonMCP/exports",
                    "ANALYSIS_REL_DIR": "AbletonMCP/analysis",
                },
                clear=False,
            ):
                with mock.patch.object(server.shutil, "which", return_value=None):
                    result = server.analyze_audio_file(None, str(mp3_path))

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "FFMPEG_NOT_FOUND")

    def test_analyze_mastering_file_reports_stereo_metrics_and_clipping(self) -> None:
        sr = 48000
        duration_sec = 2.0
        t = np.arange(int(sr * duration_sec), dtype=np.float32) / float(sr)
        # Opposite-polarity channels yield strong negative correlation and mono cancellation.
        left = 1.2 * np.sin(2.0 * math.pi * 997.0 * t).astype(np.float32)
        right = -left
        stereo = np.stack([left, right], axis=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "master_test.wav"
            _write_wav_stereo_16bit(wav_path, stereo, sr)

            result = server.analyze_mastering_file(None, str(wav_path), window_sec=0.25)

        self.assertTrue(result["ok"])
        self.assertEqual(result["channels"], 2)
        self.assertIn("lufs_integrated", result)
        self.assertIn("true_peak_dbtp", result)
        self.assertIn("stereo_correlation_series", result)
        self.assertGreaterEqual(len(result["stereo_correlation_series"]), 1)
        self.assertIsInstance(result["clipped_sample_count"], int)
        self.assertGreater(result["clipped_sample_count"], 0)
        self.assertIsInstance(result.get("peak_sample_frame"), int)
        self.assertIsInstance(result.get("peak_sample_time_sec"), float)
        self.assertIsInstance(result.get("peak_sample_channel"), int)
        self.assertIsInstance(result.get("top_peak_events"), list)
        self.assertTrue(result["top_peak_events"])
        self.assertIn("analysis_pipeline", result)
        self.assertIn("decode", result["analysis_pipeline"])

        corr_values = [
            row["correlation"]
            for row in result["stereo_correlation_series"]
            if isinstance(row, dict) and isinstance(row.get("correlation"), (int, float))
        ]
        self.assertTrue(corr_values)
        self.assertLess(max(corr_values), -0.7)

        self.assertLessEqual(
            float(result["sample_peak_dbfs"]),
            0.0
        )
        self.assertGreaterEqual(
            float(result["true_peak_dbtp"]),
            float(result["sample_peak_dbfs"]) - 0.25
        )
        self.assertIsInstance(result.get("summary"), str)
        self.assertTrue(result["summary"])

    def test_analyze_mastering_file_handles_mono_sources(self) -> None:
        sr = 44100
        t = np.arange(sr, dtype=np.float32) / float(sr)
        mono = 0.25 * np.sin(2.0 * math.pi * 220.0 * t).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "mono_master.wav"
            _write_wav_mono_16bit(wav_path, mono, sr)
            result = server.analyze_mastering_file(None, str(wav_path), window_sec=0.5)

        self.assertTrue(result["ok"])
        self.assertEqual(result["channels"], 1)
        self.assertEqual(result["stereo_correlation_series"], [])
        self.assertIsNone(result["stereo_width_score"])
        self.assertIsInstance(result.get("analysis_notes"), list)

    def test_ffmpeg_path_resolution_honors_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_ffmpeg = Path(tmpdir) / "ffmpeg"
            fake_ffmpeg.write_text("#!/bin/sh\\nexit 0\\n", encoding="utf-8")
            fake_ffmpeg.chmod(0o755)

            with mock.patch.dict(
                os.environ,
                {"ABLETON_MCP_FFMPEG_PATH": str(fake_ffmpeg)},
                clear=False,
            ):
                with mock.patch.object(server.shutil, "which", return_value=None):
                    probe = server._resolve_ffmpeg_path()

        self.assertTrue(probe["available"])
        self.assertEqual(probe["path"], str(fake_ffmpeg))
        self.assertEqual(probe["source"], "env_override")

    def test_analyze_mastering_file_reports_pipeline_when_soundfile_missing(self) -> None:
        sr = 44100
        t = np.arange(sr, dtype=np.float32) / float(sr)
        stereo = np.stack(
            [
                0.2 * np.sin(2.0 * math.pi * 440.0 * t),
                0.2 * np.sin(2.0 * math.pi * 660.0 * t),
            ],
            axis=1,
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "fallback_decode.wav"
            _write_wav_stereo_16bit(wav_path, stereo, sr)

            with mock.patch.object(server, "sf", None):
                with mock.patch.object(server, "AudioSegment", None):
                    result = server.analyze_mastering_file(None, str(wav_path), window_sec=0.5)

        self.assertTrue(result["ok"])
        pipeline = result.get("analysis_pipeline", {})
        self.assertEqual(pipeline.get("decode", {}).get("backend"), "stdlib_wav")


if __name__ == "__main__":
    unittest.main()

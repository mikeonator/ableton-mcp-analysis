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


if __name__ == "__main__":
    unittest.main()

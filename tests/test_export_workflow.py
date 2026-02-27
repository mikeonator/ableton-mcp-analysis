import json
import os
import tempfile
import unittest
from unittest import mock

from MCP_Server import server
from MCP_Server import pathing


class _FakeAbletonConnection:
    def send_command(self, command_type, params=None):
        if command_type == "get_session_info":
            return {
                "tempo": 120.0,
                "signature_numerator": 4,
                "signature_denominator": 4,
                "track_count": 2
            }
        if command_type == "get_tracks_mixer_state":
            return {
                "ok": True,
                "states": [
                    {
                        "track_index": 0,
                        "track_name": "Track 1",
                        "track_kind": "audio",
                        "is_group_track": False,
                        "group_track_index": None,
                        "solo": False,
                        "mute": False
                    },
                    {
                        "track_index": 1,
                        "track_name": "Track 2",
                        "track_kind": "audio",
                        "is_group_track": False,
                        "group_track_index": None,
                        "solo": False,
                        "mute": False
                    }
                ]
            }
        raise RuntimeError(f"Unexpected command in fake connection: {command_type}")


class ExportWorkflowTests(unittest.TestCase):
    def test_bootstrap_creates_project_dirs_and_agents_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mark temp folder as an Ableton project so AGENTS.md bootstrap is allowed.
            with open(os.path.join(tmpdir, "DemoSong.als"), "w", encoding="utf-8") as handle:
                handle.write("placeholder")
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
                project_agents_path = os.path.join(tmpdir, "AGENTS.md")
                self.assertFalse(os.path.exists(project_agents_path))

                result = pathing.bootstrap_project_environment()

                self.assertTrue(result["ok"])
                self.assertTrue(os.path.isdir(os.path.join(tmpdir, "AbletonMCP", "exports")))
                self.assertTrue(os.path.isdir(os.path.join(tmpdir, "AbletonMCP", "analysis")))
                self.assertTrue(os.path.exists(project_agents_path))

    def test_bootstrap_skips_agents_md_in_non_ableton_project_folders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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
                result = pathing.bootstrap_project_environment()

            self.assertTrue(result["ok"])
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "AbletonMCP", "exports")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "AbletonMCP", "analysis")))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "AGENTS.md")))
            self.assertEqual(result["agents"]["reason"], "not_ableton_project_folder")
            self.assertFalse(result["agents"]["copied"])

    def test_plan_exports_writes_manifest_under_project_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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
                with mock.patch.object(server, "get_ableton_connection", return_value=_FakeAbletonConnection()):
                    result = server.plan_exports(
                        None,
                        job_name="demo_job",
                        items=[
                            {
                                "target": "mix",
                                "start_bar": 1,
                                "end_bar": 9,
                                "filename_hint": "passA"
                            }
                        ],
                    )

                self.assertTrue(result["ok"])
                self.assertTrue(result["manifest_path"].startswith(tmpdir))
                self.assertTrue(result["export_dir"].startswith(tmpdir))
                self.assertTrue(result["analysis_dir"].startswith(tmpdir))
                self.assertEqual(len(result["items"]), 1)
                self.assertTrue(result["items"][0]["suggested_output_path"].startswith(result["export_dir"]))
                self.assertTrue(os.path.exists(result["manifest_path"]))

                with open(result["manifest_path"], "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                self.assertEqual(manifest["job_name"], "demo_job")
                self.assertEqual(len(manifest["expected_paths"]), 1)
                self.assertTrue(manifest["expected_paths"][0].startswith(result["export_dir"]))

    def test_check_and_analyze_return_wait_when_exports_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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
                with mock.patch.object(server, "get_ableton_connection", return_value=_FakeAbletonConnection()):
                    plan = server.plan_exports(
                        None,
                        job_name="wait_job",
                        items=[{"target": "mix", "start_bar": 1, "end_bar": 5}],
                    )

                self.assertTrue(plan["ok"])
                manifest_path = plan["manifest_path"]

                readiness = server.check_exports_ready(None, manifest_path=manifest_path)
                self.assertTrue(readiness["ok"])
                self.assertFalse(readiness["ready"])
                self.assertEqual(readiness["status"], "WAIT_FOR_USER_EXPORT")
                self.assertGreaterEqual(len(readiness["missing"]), 1)

                analysis = server.analyze_export_job(None, manifest_path=manifest_path)
                self.assertTrue(analysis["ok"])
                self.assertFalse(analysis["ready"])
                self.assertEqual(analysis["status"], "WAIT_FOR_USER_EXPORT")
                self.assertGreaterEqual(len(analysis["missing"]), 1)

    def test_analyze_export_job_mastering_profile_summarizes_batch_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")
            analysis_dir = os.path.join(tmpdir, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            wav_a = os.path.join(tmpdir, "mix_a.wav")
            wav_b = os.path.join(tmpdir, "mix_b.wav")
            manifest = {
                "job_name": "master_job",
                "analysis_dir": analysis_dir,
                "items": [
                    {"item_id": "mix_a", "suggested_output_path": wav_a},
                    {"item_id": "mix_b", "suggested_output_path": wav_b},
                ],
            }

            def _fake_mastering(ctx=None, file_path=None, window_sec=1.0):
                if file_path == wav_a:
                    return {
                        "ok": True,
                        "sample_peak_dbfs": -0.5,
                        "true_peak_dbtp": -0.2,
                        "lufs_integrated": -12.5,
                        "correlation_min": 0.15,
                        "stereo_width_score": 62.0,
                        "inter_sample_peak_risk": True,
                        "summary": "A"
                    }
                return {
                    "ok": True,
                    "sample_peak_dbfs": -2.0,
                    "true_peak_dbtp": -1.1,
                    "lufs_integrated": -15.4,
                    "correlation_min": 0.45,
                    "stereo_width_score": 44.0,
                    "inter_sample_peak_risk": False,
                    "summary": "B"
                }

            with mock.patch.object(server, "check_exports_ready", return_value={
                "ok": True,
                "ready": True,
                "manifest_path": manifest_path
            }):
                with mock.patch.object(server, "_resolve_export_manifest_path", return_value=manifest_path):
                    with mock.patch.object(server, "_load_export_manifest", return_value=manifest):
                        with mock.patch.object(server, "analyze_mastering_file", side_effect=_fake_mastering):
                            result = server.analyze_export_job(
                                None,
                                manifest_path=manifest_path,
                                analysis_profile="mastering"
                            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["job_name"], "master_job")
            self.assertEqual(result["summary"]["analysis_profile"], "mastering")
            self.assertEqual(result["summary"]["true_peak_max_dbtp"], -0.2)
            self.assertEqual(result["summary"]["correlation_min"], 0.15)
            self.assertEqual(result["summary"]["items_exceeding_true_peak_threshold"], ["mix_a"])
            self.assertIn("mix_b", result["summary"]["items_below_target_loudness"])
            self.assertIn("mix_a", result["summary"]["items_above_target_loudness"])
            self.assertEqual(len(result["results"]), 2)
            self.assertEqual(result["results"][0]["analysis_profile"], "mastering")


if __name__ == "__main__":
    unittest.main()

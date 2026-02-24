import unittest
from unittest import mock

from MCP_Server import server


class BuildMixMasterContextTests(unittest.TestCase):
    def test_build_mix_master_context_returns_stage_readiness_and_actions(self):
        mix_context_profile = {
            "ok": True,
            "topology": {
                "ok": True,
                "tracks": [
                    {"index": 0, "name": "Lead Vox", "is_group_track": False, "sends": [{"send_index": 0}]},
                    {"index": 1, "name": "Drum Bus", "is_group_track": True, "sends": []},
                ],
                "returns": [{"index": 0, "name": "A Reverb"}],
                "master": {"name": "Master", "devices": [{"device_index": 0, "name": "Limiter"}]},
                "warnings": []
            },
            "explicit_tags": {"track_roles": {"track:0": ["lead_vocal"]}, "return_roles": {}, "master_roles": ["mix_bus"]},
            "inference_suggestions": {"track_roles": {"track:1": ["drums_bus"]}},
            "merged_roles": {"track_roles": {"track:0": ["lead_vocal"], "track:1": ["drums_bus"]}, "return_roles": {}, "master_roles": ["mix_bus"]},
        }

        with mock.patch.object(server, "build_mix_context_profile", return_value=mix_context_profile):
            with mock.patch.object(server, "get_session_snapshot", return_value={"ok": True, "session": {"track_count": 2}, "tracks": []}):
                with mock.patch.object(server, "get_automation_overview", return_value={
                    "ok": True,
                    "supported": True,
                    "tracks_with_device_automation": 1
                }):
                    with mock.patch.object(server, "index_sources_from_live_set", return_value={"ok": True, "unique_sources_found": 2, "sources": []}):
                        with mock.patch.object(server, "analyze_export_job", return_value={
                            "ok": True,
                            "summary": {
                                "analysis_profile": "mastering",
                                "lufs_integrated_range": {"min": -15.0, "max": -12.0}
                            }
                        }):
                            with mock.patch.object(server, "snapshot_project_state", return_value={
                                "ok": True,
                                "snapshot_id": "snap1",
                                "project_hash": "abc",
                                "summary": {"track_count": 2}
                            }):
                                result = server.build_mix_master_context(
                                    None,
                                    include_source_inventory=True,
                                    include_export_analysis=True,
                                    include_mastering_metrics=True,
                                    manifest_path="/tmp/fake_manifest.json"
                                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["export_analysis_request"]["analysis_profile"], "mastering")
        stage_ids = [row["stage"] for row in result["stage_readiness"]]
        self.assertEqual(stage_ids, [
            "mix_session_prep_inserts_sends",
            "mix_stage_1_import_organize",
            "mix_stage_2_gain_staging",
            "mix_stage_3_fader_mix",
            "mix_stage_4_automation_pass",
            "mix_stage_5_sub_mix",
            "mix_stage_6_printing_stereo_and_stems",
            "mastering_goals",
            "mastering_chain_eq_sat_comp_eq_stereo_limit",
        ])
        self.assertIsInstance(result["missing_data_actions"], list)
        self.assertIn("future_mutation_api_spec", result)
        self.assertEqual(result["tags"]["merged_roles"]["master_roles"], ["mix_bus"])

    def test_build_mix_master_context_requires_manifest_or_job_for_export_analysis(self):
        with mock.patch.object(server, "build_mix_context_profile", return_value={"ok": False}):
            with mock.patch.object(server, "get_session_snapshot", return_value={"ok": False}):
                with mock.patch.object(server, "get_automation_overview", return_value={"ok": False}):
                    with mock.patch.object(server, "snapshot_project_state", return_value={"ok": False, "error": "x"}):
                        result = server.build_mix_master_context(
                            None,
                            include_source_inventory=False,
                            include_export_analysis=True,
                            include_mastering_metrics=False
                        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["export_analysis"]["error"], "missing_export_manifest_selector")


if __name__ == "__main__":
    unittest.main()

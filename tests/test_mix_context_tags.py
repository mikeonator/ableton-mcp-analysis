import os
import tempfile
import unittest
from unittest import mock

from MCP_Server import server


def _fake_topology():
    return {
        "ok": True,
        "session": {"track_count": 4, "return_track_count": 2},
        "tracks": [
            {"index": 0, "name": "Lead Vox", "is_group_track": False, "sends": []},
            {"index": 1, "name": "Kick", "is_group_track": False, "sends": []},
            {"index": 2, "name": "Snare", "is_group_track": False, "sends": []},
            {"index": 3, "name": "Drum Bus", "is_group_track": True, "sends": []},
        ],
        "returns": [
            {"index": 0, "name": "Verb A"},
            {"index": 1, "name": "NY Comp"},
        ],
        "master": {"name": "Master", "devices": []},
        "edges": [],
        "warnings": []
    }


class MixContextTagsTests(unittest.TestCase):
    def test_infer_mix_context_tags_detects_common_roles(self):
        with mock.patch.object(server, "get_mix_topology", return_value=_fake_topology()):
            result = server.infer_mix_context_tags(None)

        self.assertTrue(result["ok"])
        self.assertIn("track:0", result["track_roles"])
        self.assertIn("lead_vocal", result["track_roles"]["track:0"])
        self.assertIn("kick", result["track_roles"]["track:1"])
        self.assertIn("snare", result["track_roles"]["track:2"])
        self.assertIn("drums_bus", result["track_roles"]["track:3"])
        self.assertIn("main_reverb", result["return_roles"]["return:0"])
        self.assertIn("parallel_compression", result["return_roles"]["return:1"])
        self.assertIn("mix_bus", result["master_roles"])
        self.assertIn("track:0", result["confidence"])

    def test_save_merge_preserves_existing_explicit_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "PROJECT_ROOT": tmpdir,
                "EXPORT_ROOT_MODE": "project",
                "EXPORT_REL_DIR": "AbletonMCP/exports",
                "ANALYSIS_REL_DIR": "AbletonMCP/analysis",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                first = server.save_mix_context_tags(None, {
                    "track_roles": {"track:0": ["lead_vocal"]},
                    "return_roles": {"return:0": ["main_reverb"]},
                    "master_roles": ["mix_bus"]
                }, merge=False)
                self.assertTrue(first["ok"])

                merged = server.save_mix_context_tags(None, {
                    "track_roles": {
                        "track:0": ["backing_vocal"],
                        "track:1": ["snare"]
                    }
                }, merge=True)
                self.assertTrue(merged["ok"])

                loaded = server.get_mix_context_tags(None)

        self.assertEqual(loaded["track_roles"]["track:0"], ["lead_vocal"])
        self.assertEqual(loaded["track_roles"]["track:1"], ["snare"])
        self.assertEqual(loaded["return_roles"]["return:0"], ["main_reverb"])
        self.assertEqual(loaded["master_roles"], ["mix_bus"])

    def test_build_mix_context_profile_prefers_explicit_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "PROJECT_ROOT": tmpdir,
                "EXPORT_ROOT_MODE": "project",
                "EXPORT_REL_DIR": "AbletonMCP/exports",
                "ANALYSIS_REL_DIR": "AbletonMCP/analysis",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                server.save_mix_context_tags(None, {
                    "track_roles": {"track:0": ["backing_vocal"]},
                    "master_roles": ["mix_bus"]
                }, merge=False)

                with mock.patch.object(server, "get_mix_topology", return_value=_fake_topology()):
                    profile = server.build_mix_context_profile(None)

        self.assertTrue(profile["ok"])
        merged = profile["merged_roles"]
        self.assertEqual(merged["track_roles"]["track:0"], ["backing_vocal"])
        self.assertIn("kick", merged["track_roles"]["track:1"])
        self.assertEqual(merged["master_roles"], ["mix_bus"])


if __name__ == "__main__":
    unittest.main()

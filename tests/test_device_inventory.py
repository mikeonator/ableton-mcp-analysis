import unittest
from unittest import mock

from MCP_Server import server


class _FakeInventoryConnection:
    def __init__(self):
        self.path_map = {
            "audio_effects": [
                {
                    "name": "EQ Eight",
                    "is_folder": False,
                    "is_device": True,
                    "is_loadable": True,
                    "uri": "query:AudioFx#EQ%20Eight",
                },
                {
                    "name": "Dynamics",
                    "is_folder": True,
                    "is_device": False,
                    "is_loadable": False,
                    "uri": "query:AudioFx#Dynamics",
                },
            ],
            "audio_effects/dynamics": [
                {
                    "name": "Compressor",
                    "is_folder": False,
                    "is_device": True,
                    "is_loadable": True,
                    "uri": "query:AudioFx#Compressor",
                }
            ],
            "plugins": [
                {
                    "name": "Ozone 11",
                    "is_folder": False,
                    "is_device": True,
                    "is_loadable": True,
                    "uri": "query:Plugins#VST3:iZotope:Ozone%2011",
                }
            ],
            "midi_effects": [
                {
                    "name": "Arpeggiator",
                    "is_folder": False,
                    "is_device": True,
                    "is_loadable": True,
                    "uri": "query:MidiFx#Arpeggiator",
                }
            ],
            "max_for_live": [
                {
                    "name": "Max Audio Effect",
                    "is_folder": True,
                    "is_device": False,
                    "is_loadable": False,
                    "uri": "query:M4L#Max%20Audio%20Effect",
                },
                {
                    "name": "Max MIDI Effect",
                    "is_folder": True,
                    "is_device": False,
                    "is_loadable": False,
                    "uri": "query:M4L#Max%20MIDI%20Effect",
                },
            ],
            "max_for_live/max_audio_effect": [
                {
                    "name": "Stereo Scope",
                    "is_folder": False,
                    "is_device": True,
                    "is_loadable": True,
                    "uri": "query:M4L#Max%20Audio%20Effect:StereoScope",
                }
            ],
            "max_for_live/max_midi_effect": [
                {
                    "name": "MIDI LFO",
                    "is_folder": False,
                    "is_device": True,
                    "is_loadable": True,
                    "uri": "query:M4L#Max%20MIDI%20Effect:MIDI%20LFO",
                }
            ],
        }

    @staticmethod
    def _normalize_path(path: str) -> str:
        parts = [part for part in str(path).split("/") if part]
        out = []
        for part in parts:
            out.append(server._normalize_browser_token(part))
        return "/".join(out)

    def send_command(self, command_type, params=None):
        params = params or {}
        if command_type == "get_browser_tree":
            return {
                "categories": [
                    {"name": "Audio Effects"},
                    {"name": "Plugins"},
                    {"name": "MIDI Effects"},
                    {"name": "Max for Live"},
                ]
            }
        if command_type == "get_browser_items_at_path":
            path = params.get("path", "")
            key = self._normalize_path(path)
            if key not in self.path_map:
                return {"path": path, "error": f"Path part not found: {path}", "items": []}
            return {"path": path, "items": list(self.path_map[key])}
        raise RuntimeError(f"Unexpected command: {command_type}")


class DeviceInventoryTests(unittest.TestCase):
    def setUp(self):
        server._DEVICE_INVENTORY_RUNTIME_CACHE.clear()

    def test_rejects_non_list_roots(self):
        with mock.patch.object(server, "get_ableton_connection", return_value=_FakeInventoryConnection()):
            result = server.get_device_inventory(None, roots="Plugins")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "invalid_roots_type")

    def test_root_alias_normalization_accepts_token_forms(self):
        with mock.patch.object(server, "get_ableton_connection", return_value=_FakeInventoryConnection()):
            result = server.get_device_inventory(
                None,
                roots=["Max_for_live", "audio_effects"],
                max_depth=2,
                response_mode="full",
                use_cache=False,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["roots_not_found"], [])
        self.assertIn("Max for Live", result["normalized_requested_roots"])
        self.assertIn("Audio Effects", result["normalized_requested_roots"])
        self.assertIn("Max for Live", result["scanned_roots"])
        self.assertIn("Audio Effects", result["scanned_roots"])

    def test_compact_mode_returns_paged_devices(self):
        with mock.patch.object(server, "get_ableton_connection", return_value=_FakeInventoryConnection()):
            result = server.get_device_inventory(
                None,
                roots=["Audio Effects"],
                max_depth=2,
                response_mode="compact",
                offset=0,
                limit=1,
                use_cache=False,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["response_mode"], "compact")
        self.assertEqual(result["returned_count"], 1)
        self.assertGreaterEqual(result["total_devices"], 2)
        self.assertTrue(result["has_more"])

    def test_audio_only_filter_includes_max_for_live_audio_not_midi(self):
        with mock.patch.object(server, "get_ableton_connection", return_value=_FakeInventoryConnection()):
            result = server.get_device_inventory(
                None,
                roots=["Audio Effects", "MIDI Effects", "Max for Live"],
                max_depth=2,
                response_mode="full",
                audio_only=True,
                include_max_for_live_audio=True,
                use_cache=False,
            )

        self.assertTrue(result["ok"])
        names = {row["name"] for row in result["devices"]}
        self.assertIn("EQ Eight", names)
        self.assertIn("Compressor", names)
        self.assertIn("Stereo Scope", names)
        self.assertNotIn("Arpeggiator", names)
        self.assertNotIn("MIDI LFO", names)


if __name__ == "__main__":
    unittest.main()

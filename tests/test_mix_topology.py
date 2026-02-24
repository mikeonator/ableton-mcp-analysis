import unittest
from unittest import mock

from MCP_Server import server


class _FakeAbletonConnection:
    def send_command(self, command_type, params=None):
        if command_type == "get_mix_topology":
            return {
                "ok": True,
                "session": {
                    "tempo": 120.0,
                    "signature_numerator": 4,
                    "signature_denominator": 4,
                    "track_count": 2,
                    "return_track_count": 1
                },
                "tracks": [
                    {
                        "scope": "track",
                        "index": 0,
                        "name": "Lead Vox",
                        "track_kind": "audio",
                        "is_group_track": False,
                        "group_track_index": 1,
                        "mixer": {
                            "volume": 0.0,
                            "panning": 0.0,
                            "mute": False,
                            "solo": False,
                            "arm": False
                        },
                        "sends": [
                            {
                                "send_index": 0,
                                "target_return_index": 0,
                                "name": "A Reverb",
                                "value": 0.25,
                                "min": 0.0,
                                "max": 1.0,
                                "is_enabled": True
                            }
                        ],
                        "routing": {
                            "input_type": None,
                            "input_channel": None,
                            "output_type": "Master",
                            "output_channel": "Stereo"
                        },
                        "devices": [
                            {
                                "device_index": 0,
                                "name": "EQ Eight",
                                "class_name": "AudioEffectGroupDevice",
                                "is_plugin": False,
                                "plugin_format": None,
                                "vendor": None,
                                "parameter_count": 10
                            }
                        ]
                    },
                    {
                        "scope": "track",
                        "index": 1,
                        "name": "Vox Bus",
                        "track_kind": "group",
                        "is_group_track": True,
                        "group_track_index": None,
                        "mixer": {
                            "volume": -3.0,
                            "panning": 0.0,
                            "mute": False,
                            "solo": False,
                            "arm": None
                        },
                        "sends": [],
                        "routing": {"output_type": "Master", "output_channel": "Stereo"},
                        "devices": []
                    }
                ],
                "returns": [
                    {
                        "scope": "return",
                        "index": 0,
                        "name": "A Reverb",
                        "mixer": {"volume": -6.0, "panning": 0.0, "mute": False, "solo": False},
                        "routing": {"output_type": "Master", "output_channel": "Stereo"},
                        "devices": [
                            {
                                "device_index": 0,
                                "name": "Hybrid Reverb",
                                "class_name": "AudioEffectGroupDevice",
                                "is_plugin": False,
                                "plugin_format": None,
                                "vendor": None,
                                "parameter_count": 32
                            }
                        ]
                    }
                ],
                "master": {
                    "scope": "master",
                    "name": "Master",
                    "mixer": {"volume": 0.0, "panning": 0.0},
                    "devices": [
                        {
                            "device_index": 0,
                            "name": "Limiter",
                            "class_name": "AudioEffectGroupDevice",
                            "is_plugin": False,
                            "plugin_format": None,
                            "vendor": None,
                            "parameter_count": 8
                        }
                    ]
                },
                "edges": [
                    {"from": "track:0", "to": "track:1", "kind": "group_membership"},
                    {"from": "track:0", "to": "return:0", "kind": "send", "send_index": 0, "amount": 0.25},
                    {"from": "track:1", "to": "master", "kind": "output_routing"}
                ],
                "warnings": []
            }
        raise RuntimeError(f"Unexpected command: {command_type}")


class MixTopologyToolsTests(unittest.TestCase):
    def test_get_mix_topology_normalizes_and_returns_tracks_returns_master(self):
        with mock.patch.object(server, "get_ableton_connection", return_value=_FakeAbletonConnection()):
            result = server.get_mix_topology(None)

        self.assertTrue(result["ok"])
        self.assertEqual(result["session"]["track_count"], 2)
        self.assertEqual(len(result["tracks"]), 2)
        self.assertEqual(len(result["returns"]), 1)
        self.assertEqual(result["master"]["name"], "Master")
        self.assertEqual(result["tracks"][0]["sends"][0]["target_return_index"], 0)

    def test_get_send_matrix_derives_active_routes(self):
        with mock.patch.object(server, "get_ableton_connection", return_value=_FakeAbletonConnection()):
            result = server.get_send_matrix(None)

        self.assertTrue(result["ok"])
        self.assertEqual(result["track_count"], 2)
        self.assertEqual(len(result["active_routes"]), 1)
        self.assertEqual(result["active_routes"][0]["track_name"], "Lead Vox")
        self.assertEqual(result["active_routes"][0]["target_return_name"], "A Reverb")

    def test_return_and_master_chain_convenience_wrappers(self):
        with mock.patch.object(server, "get_ableton_connection", return_value=_FakeAbletonConnection()):
            returns_result = server.get_return_tracks_info(None, include_device_chains=True)
            master_result = server.get_master_track_device_chain(None)

        self.assertTrue(returns_result["ok"])
        self.assertEqual(returns_result["return_count"], 1)
        self.assertEqual(returns_result["returns"][0]["name"], "A Reverb")
        self.assertTrue(master_result["ok"])
        self.assertEqual(master_result["device_count"], 1)
        self.assertEqual(master_result["devices"][0]["name"], "Limiter")

if __name__ == "__main__":
    unittest.main()

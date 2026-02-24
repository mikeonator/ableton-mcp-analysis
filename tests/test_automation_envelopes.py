import unittest
from unittest import mock

from MCP_Server import server


class _FakeAutomationEnvelopeConn:
    def send_command(self, command_type, params=None):
        if command_type != "get_automation_envelope_points":
            raise RuntimeError("Unexpected command: {0}".format(command_type))
        return {
            "ok": True,
            "supported": True,
            "track_index": 3,
            "track_name": "Lead Vox",
            "target": {
                "scope": "track_mixer",
                "mixer_target": "volume",
                "parameter_name": "Track Volume"
            },
            "automation_state": 1,
            "envelope_exists": True,
            "point_access_supported": True,
            "points": [
                {"point_index": 1, "time_beats": 8.0, "value": 0.42},
                {"point_index": 0, "time_beats": 4.0, "value": 0.35}
            ],
            "sampled_series": [],
            "warnings": []
        }


class _UnknownCommandConn:
    def send_command(self, command_type, params=None):
        raise Exception("Unknown command: get_automation_envelope_points")


class AutomationEnvelopeToolTests(unittest.TestCase):
    def test_get_automation_envelope_points_returns_normalized_points(self):
        with mock.patch.object(server, "get_ableton_connection", return_value=_FakeAutomationEnvelopeConn()):
            result = server.get_automation_envelope_points(
                None,
                track_index=3,
                scope="track_mixer",
                mixer_target="volume"
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertTrue(result["point_access_supported"])
        self.assertEqual(result["track_name"], "Lead Vox")
        self.assertEqual(result["target"]["scope"], "track_mixer")
        self.assertEqual(result["target"]["mixer_target"], "volume")
        self.assertEqual([row["time_beats"] for row in result["points"]], [4.0, 8.0])

    def test_get_automation_envelope_points_unknown_backend_command_is_graceful(self):
        with mock.patch.object(server, "get_ableton_connection", return_value=_UnknownCommandConn()):
            result = server.get_automation_envelope_points(
                None,
                track_index=0,
                scope="track_mixer",
                mixer_target="volume"
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["supported"])
        self.assertEqual(result["reason"], "backend_command_unavailable")
        self.assertEqual(result["points"], [])
        self.assertEqual(result["sampled_series"], [])

    def test_get_automation_envelope_points_validates_device_parameter_indexes(self):
        result = server.get_automation_envelope_points(
            None,
            track_index=0,
            scope="device_parameter",
            device_index=None,
            parameter_index=0
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "invalid_device_index")


if __name__ == "__main__":
    unittest.main()

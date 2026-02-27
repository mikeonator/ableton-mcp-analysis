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


class _UnsupportedSongEnvelopeConn:
    def send_command(self, command_type, params=None):
        if command_type != "get_automation_envelope_points":
            raise RuntimeError("Unexpected command: {0}".format(command_type))
        return {
            "ok": True,
            "supported": False,
            "reason": "song_automation_envelope_unavailable",
            "track_index": 0,
            "track_name": "Lead Vox",
            "target": {
                "scope": "track_mixer",
                "mixer_target": "volume",
                "parameter_name": "Track Volume",
            },
            "automation_state": 1,
            "envelope_exists": False,
            "point_access_supported": False,
            "points": [],
            "sampled_series": [],
            "warnings": ["song_automation_envelope_unavailable"],
        }


class _FakeClipAutomationEnvelopeConn:
    def send_command(self, command_type, params=None):
        if command_type != "get_clip_automation_envelope_points":
            raise RuntimeError("Unexpected command: {0}".format(command_type))
        return {
            "ok": True,
            "supported": True,
            "track_index": 12,
            "track_name": "Lead Vox",
            "clip_scope": "session",
            "clip_slot_index": 3,
            "clip_name": "Lead Vox Clip",
            "clip": {
                "clip_scope": "session",
                "clip_slot_index": 3,
                "clip_name": "Lead Vox Clip",
                "is_audio_clip": True,
                "length_beats": 16.0,
            },
            "target": {
                "scope": "device_parameter",
                "device_index": 0,
                "device_name": "Auto Filter",
                "parameter_index": 2,
                "parameter_name": "Frequency",
            },
            "automation_state": 1,
            "envelope_exists": True,
            "point_access_supported": True,
            "points": [
                {"point_index": 2, "time_beats": 4.0, "value": 2000.0},
                {"point_index": 1, "time_beats": 2.0, "value": 1200.0},
            ],
            "sampled_series": [],
            "warnings": [],
        }


class _UnknownClipCommandConn:
    def send_command(self, command_type, params=None):
        raise Exception("Unknown command: get_clip_automation_envelope_points")


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

    def test_get_automation_envelope_points_uses_als_fallback_when_live_api_unavailable(self):
        als_payload = {
            "ok": True,
            "supported": True,
            "source": "als_file",
            "source_kind": "arrangement_automation",
            "track_index": 0,
            "track_name": "Lead Vox",
            "target": {
                "scope": "track_mixer",
                "mixer_target": "volume",
                "automation_target_id": 1234,
            },
            "envelope_exists": True,
            "point_access_supported": True,
            "points": [
                {"point_index": 0, "time_beats": 1.0, "value": 0.25},
                {"point_index": 1, "time_beats": 2.0, "value": 0.5},
            ],
            "warnings": ["saved_als_snapshot_only"],
            "als_file_path": "/tmp/Test.als",
            "als_file_mtime_utc": "2026-02-25T00:00:00Z",
        }
        with mock.patch.object(server, "get_ableton_connection", return_value=_UnsupportedSongEnvelopeConn()):
            with mock.patch.object(server, "get_project_root", return_value="/tmp/project"):
                with mock.patch.object(server, "read_arrangement_automation_from_project_als", return_value=als_payload):
                    result = server.get_automation_envelope_points(
                        None,
                        track_index=0,
                        scope="track_mixer",
                        mixer_target="volume",
                    )

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertTrue(result["als_fallback_used"])
        self.assertEqual(result["point_source"], "als_arrangement_file")
        self.assertEqual(result["target"]["automation_target_id"], 1234)
        self.assertEqual([row["time_beats"] for row in result["points"]], [1.0, 2.0])
        self.assertIn("saved_als_snapshot_only", result["warnings"])

    def test_get_automation_envelope_points_refuses_mismatched_device_parameter_als_mapping(self):
        als_payload = {
            "ok": True,
            "supported": True,
            "source": "als_file",
            "source_kind": "arrangement_automation",
            "track_index": 0,
            "track_name": "Lead Vox",
            "target": {
                "scope": "device_parameter",
                "device_index": 0,
                "parameter_index": 10,
                "automation_target_id": 84045,
                "device_xml_tag": "StereoGain",
                "parameter_xml_tag": "Gain",
                "device_name_hint": None,
                "parameter_name_hint": "Gain",
                "parameter_display_name_hint": None,
            },
            "envelope_exists": True,
            "point_access_supported": True,
            "points": [{"point_index": 0, "time_beats": 1.0, "value": 1.0}],
            "warnings": ["saved_als_snapshot_only"],
            "als_file_path": "/tmp/Test.als",
        }
        live_payload_conn = _UnsupportedSongEnvelopeConn()
        # mutate target to emulate runtime metadata from Live for a mismatched param
        def _send_command(command_type, params=None):
            payload = live_payload_conn.send_command(command_type, params)
            payload["target"] = {
                "scope": "device_parameter",
                "device_index": 0,
                "parameter_index": 10,
                "device_name": "Utility",
                "parameter_name": "Mute",
            }
            return payload
        with mock.patch.object(server, "get_ableton_connection") as mock_conn_factory:
            mock_conn = mock.Mock()
            mock_conn.send_command.side_effect = _send_command
            mock_conn_factory.return_value = mock_conn
            with mock.patch.object(server, "get_project_root", return_value="/tmp/project"):
                with mock.patch.object(server, "read_arrangement_automation_from_project_als", return_value=als_payload):
                    result = server.get_automation_envelope_points(
                        None,
                        track_index=0,
                        scope="device_parameter",
                        device_index=0,
                        parameter_index=10,
                    )

        self.assertTrue(result["ok"])
        self.assertFalse(result.get("als_fallback_used", False))
        self.assertEqual(result["reason"], "song_automation_envelope_unavailable")
        self.assertEqual(result["points"], [])
        self.assertEqual(result["als_fallback_reason"], "device_parameter_mapping_mismatch")
        self.assertIn("device_parameter_mapping_mismatch", result["warnings"])

    def test_get_automation_envelope_points_passes_explicit_als_file_path_to_fallback(self):
        captured = {}
        def _fake_read_arrangement_automation_from_project_als(**kwargs):
            captured.update(kwargs)
            return {
                "ok": True,
                "supported": False,
                "reason": "top_level_als_not_found",
                "warnings": [],
            }

        with mock.patch.object(server, "get_ableton_connection", return_value=_UnsupportedSongEnvelopeConn()):
            with mock.patch.object(server, "get_project_root", return_value="/tmp/project"):
                with mock.patch.object(server, "read_arrangement_automation_from_project_als", side_effect=_fake_read_arrangement_automation_from_project_als):
                    _ = server.get_automation_envelope_points(
                        None,
                        track_index=0,
                        scope="track_mixer",
                        mixer_target="volume",
                        als_file_path="/tmp/DeadEnemiesMidterm.als",
                    )

        self.assertEqual(captured.get("als_file_path"), "/tmp/DeadEnemiesMidterm.als")

    def test_get_clip_automation_envelope_points_returns_normalized_points(self):
        with mock.patch.object(server, "get_ableton_connection", return_value=_FakeClipAutomationEnvelopeConn()):
            result = server.get_clip_automation_envelope_points(
                None,
                track_index=12,
                clip_scope="session",
                clip_slot_index=3,
                scope="device_parameter",
                device_index=0,
                parameter_index=2,
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertTrue(result["point_access_supported"])
        self.assertEqual(result["clip_scope"], "session")
        self.assertEqual(result["clip_slot_index"], 3)
        self.assertEqual(result["clip"]["clip_name"], "Lead Vox Clip")
        self.assertEqual([row["time_beats"] for row in result["points"]], [2.0, 4.0])

    def test_get_clip_automation_envelope_points_unknown_backend_command_is_graceful(self):
        with mock.patch.object(server, "get_ableton_connection", return_value=_UnknownClipCommandConn()):
            result = server.get_clip_automation_envelope_points(
                None,
                track_index=0,
                clip_scope="session",
                clip_slot_index=0,
                scope="track_mixer",
                mixer_target="volume",
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["supported"])
        self.assertEqual(result["reason"], "backend_command_unavailable")
        self.assertEqual(result["points"], [])
        self.assertEqual(result["sampled_series"], [])

    def test_get_clip_automation_envelope_points_validates_clip_selector(self):
        result = server.get_clip_automation_envelope_points(
            None,
            track_index=0,
            clip_scope="arrangement",
            clip_index=None,
            scope="device_parameter",
            device_index=0,
            parameter_index=0,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "invalid_clip_index")


if __name__ == "__main__":
    unittest.main()

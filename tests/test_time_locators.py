import json
import unittest
from unittest import mock

from MCP_Server import server


class _RuntimeLocatorConnection:
    def __init__(self, locator_payload=None, session_payload=None):
        self._locator_payload = locator_payload if isinstance(locator_payload, dict) else {}
        self._session_payload = session_payload if isinstance(session_payload, dict) else {}

    def send_command(self, command_type, params=None):
        if command_type == "get_time_locators":
            return dict(self._locator_payload)
        if command_type == "get_session_info":
            return dict(self._session_payload)
        raise RuntimeError(f"Unexpected command: {command_type}")


class TimeLocatorToolsTests(unittest.TestCase):
    def test_get_time_locators_prefers_runtime_when_supported(self):
        runtime_payload = {
            "ok": True,
            "supported": True,
            "locators": [
                {"index": 1, "name": "Verse", "time_beats": 32.0},
                {"index": 0, "name": "Intro", "time_beats": 8.0},
            ],
            "current_song_time_beats": 10.0,
        }
        with mock.patch.object(server, "get_ableton_connection", return_value=_RuntimeLocatorConnection(locator_payload=runtime_payload)):
            result = server.get_time_locators(None, include_als_fallback=False)

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertEqual(result["source"], "runtime")
        self.assertEqual(result["locator_count"], 2)
        self.assertEqual([row["name"] for row in result["locators"]], ["Intro", "Verse"])
        self.assertEqual(result["previous_locator"]["name"], "Intro")
        self.assertEqual(result["next_locator"]["name"], "Verse")

    def test_get_time_locators_uses_als_fallback_when_runtime_unavailable(self):
        fallback_payload = {
            "ok": True,
            "supported": True,
            "source": "als_file",
            "locators": [{"index": 0, "name": "Drop", "time_beats": 64.0}],
            "locator_count": 1,
            "warnings": ["saved_als_snapshot_only"],
        }

        with mock.patch.object(server, "get_ableton_connection", side_effect=RuntimeError("socket down")):
            with mock.patch.object(server, "get_project_root", return_value="/tmp/project"):
                with mock.patch.object(server, "read_time_locators_from_project_als", return_value=fallback_payload):
                    result = server.get_time_locators(None, include_als_fallback=True)

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertEqual(result["source"], "als_file")
        self.assertEqual(result["locator_count"], 1)
        self.assertIn("runtime_locator_error:socket down", result["warnings"])

    def test_get_time_locators_handles_empty_runtime_locator_list(self):
        runtime_payload = {
            "ok": True,
            "supported": True,
            "locators": [],
            "current_song_time_beats": 0.0,
        }
        with mock.patch.object(server, "get_ableton_connection", return_value=_RuntimeLocatorConnection(locator_payload=runtime_payload)):
            result = server.get_time_locators(None, include_als_fallback=False)

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertEqual(result["locator_count"], 0)
        self.assertEqual(result["locators"], [])

    def test_get_session_info_backfills_locator_summary_fields(self):
        with mock.patch.object(
            server,
            "get_ableton_connection",
            return_value=_RuntimeLocatorConnection(session_payload={"tempo": 120.0, "track_count": 2}),
        ):
            with mock.patch.object(
                server,
                "get_time_locators",
                return_value={"ok": True, "locators": [{"index": 0, "name": "Intro", "time_beats": 1.0}]},
            ):
                payload = server.get_session_info(None)

        parsed = json.loads(payload)
        self.assertEqual(parsed["tempo"], 120.0)
        self.assertTrue(parsed["has_time_locators"])
        self.assertEqual(parsed["time_locator_count"], 1)
        self.assertEqual(parsed["time_locators_preview"][0]["name"], "Intro")


if __name__ == "__main__":
    unittest.main()

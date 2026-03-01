import unittest
from unittest import mock

from MCP_Server import server


class _CaptureConnection:
    def __init__(self):
        self.calls = []

    def send_command(self, command_type, params=None):
        self.calls.append((command_type, dict(params or {})))
        return {"ok": True, "command": command_type, "params": dict(params or {})}


class DeviceChainNestedToolTests(unittest.TestCase):
    def test_get_track_device_chain_forwards_nested_params(self):
        fake_conn = _CaptureConnection()
        with mock.patch.object(server, "get_ableton_connection", return_value=fake_conn):
            result = server.get_track_device_chain(None, track_index=3, include_nested=True, max_depth=6)

        self.assertTrue(result["ok"])
        self.assertEqual(len(fake_conn.calls), 1)
        command, params = fake_conn.calls[0]
        self.assertEqual(command, "get_track_devices")
        self.assertEqual(params["track_index"], 3)
        self.assertTrue(params["include_nested"])
        self.assertEqual(params["max_depth"], 6)

    def test_get_device_parameters_forwards_device_path(self):
        fake_conn = _CaptureConnection()
        with mock.patch.object(server, "get_ableton_connection", return_value=fake_conn):
            result = server.get_device_parameters(
                None,
                track_index=1,
                device_index=2,
                device_path=[2, 0, 1],
                offset=4,
                limit=16,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(len(fake_conn.calls), 1)
        command, params = fake_conn.calls[0]
        self.assertEqual(command, "get_device_parameters")
        self.assertEqual(params["track_index"], 1)
        self.assertEqual(params["device_index"], 2)
        self.assertEqual(params["device_path"], [2, 0, 1])
        self.assertEqual(params["offset"], 4)
        self.assertEqual(params["limit"], 16)

    def test_get_device_parameters_validates_device_path_mismatch(self):
        fake_conn = _CaptureConnection()
        with mock.patch.object(server, "get_ableton_connection", return_value=fake_conn):
            result = server.get_device_parameters(
                None,
                track_index=1,
                device_index=2,
                device_path=[1, 0, 0],
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "device_path_mismatch")
        self.assertEqual(fake_conn.calls, [])


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock

from MCP_Server import server


class AutomationEnumeratorTests(unittest.TestCase):
    def _fake_exhaustive_inventory(self):
        base_row = {
            "inventory_kind": "automation_target",
            "container_scope": "track",
            "container_index": 0,
            "container_locator": "track:0",
            "container_name": "Lead Vox",
            "automation_target_id": 200,
            "classification": {
                "domain": "track_device_parameter",
                "scope": "device_parameter",
                "legacy_top_level_device_index": 0,
                "legacy_top_level_parameter_index": 1,
            },
            "location": {
                "track_index": 0,
                "track_name": "Lead Vox",
                "track_scope": "track",
                "clip_scope": None,
            },
            "envelope": {
                "exists": True,
                "envelope_kind": "AutomationEnvelope",
                "als_envelope_path": "LiveSet/Tracks/AudioTrack[0]/AutomationEnvelopes/Envelopes/AutomationEnvelope[0]",
                "point_count": 2,
                "event_types": ["FloatEvent"],
                "first_point_time_beats": 0.0,
                "last_point_time_beats": 8.0,
            },
            "points": [
                {"point_index": 0, "time_beats": 0.0, "value": 0.1, "event_type": "FloatEvent", "value_kind": "float"},
                {"point_index": 1, "time_beats": 8.0, "value": 0.8, "event_type": "FloatEvent", "value_kind": "float"},
            ],
            "warnings": [],
        }
        row1 = dict(base_row)
        row1["inventory_key"] = "k1"
        row1["target_ref"] = {
            "schema_version": 1,
            "inventory_key": "k1",
            "source": "als_inventory",
            "container_scope": "track",
            "container_locator": "track:0",
            "automation_target_id": 200,
        }

        row2 = dict(base_row)
        row2["automation_target_id"] = 201
        row2["inventory_key"] = "k2"
        row2["target_ref"] = dict(row1["target_ref"], inventory_key="k2", automation_target_id=201)
        row2["classification"] = {
            "domain": "track_mixer",
            "scope": "track_mixer",
            "mixer_target": "volume",
        }

        row3 = dict(base_row)
        row3["container_scope"] = "clip_arrangement"
        row3["container_locator"] = "clip_arrangement:foo"
        row3["container_index"] = None
        row3["automation_target_id"] = 301
        row3["classification"] = {
            "domain": "clip_envelope",
            "scope": "clip_envelope",
            "clip_scope": "arrangement",
            "target_metadata_resolved": False,
        }
        row3["location"] = {
            "track_scope": "track",
            "track_index": 0,
            "track_name": "Lead Vox",
            "clip_scope": "arrangement",
            "clip_name": "11_Tom3",
            "clip_type": "audio",
            "clip_als_path": "LiveSet/Tracks/AudioTrack[0]/.../AudioClip[0]",
        }
        row3["envelope"] = dict(base_row["envelope"], envelope_kind="ClipEnvelope")
        row3["inventory_key"] = "k3"
        row3["target_ref"] = {
            "schema_version": 1,
            "inventory_key": "k3",
            "source": "als_inventory",
            "container_scope": "clip_arrangement",
            "container_locator": "clip_arrangement:foo",
            "automation_target_id": 301,
        }

        return {
            "ok": True,
            "supported": True,
            "schema_version": 1,
            "source": "als_file",
            "als_file_path": "/tmp/DeadEnemiesMidterm.als",
            "als_file_mtime_utc": "2026-02-26T12:00:00Z",
            "scope_statement": {
                "session_clip_envelopes_excluded": False,
                "session_clip_envelopes_included": True,
                "reason": "session_clip_envelopes_included",
                "arrangement_clip_envelopes_included": True,
            },
            "session": {
                "track_count_normal": 2,
                "track_count_returns": 1,
                "has_main_track": True,
                "has_prehear_track": True,
            },
            "targets": [row1, row2, row3],
            "orphan_envelopes": [{"automation_target_id": 999}],
            "unclassified_targets": [{"inventory_key": "ux"}],
            "duplicate_target_id_rows": [],
            "completeness": {
                "status": "complete",
                "targets_discovered_total": 3,
                "envelopes_discovered_total": 3,
                "targets_with_envelopes": 3,
                "targets_without_envelopes": 0,
                "orphan_envelopes_total": 1,
                "unclassified_targets_total": 1,
                "duplicate_target_ids_total": 0,
                "unsupported_event_types_total": 0,
                "warnings": [],
            },
            "warnings": ["saved_als_snapshot_only"],
        }

    def test_enumerate_project_automation_aggregates_inventory(self):
        session_info = {
            "tempo": 120.0,
            "signature_numerator": 4,
            "signature_denominator": 4,
            "track_count": 2,
            "return_track_count": 1,
        }
        automation_overview = {
            "ok": True,
            "supported": True,
            "envelope_points_supported": False,
            "tracks_scanned": 2,
            "tracks_with_device_automation": 1,
            "total_automated_parameters": 1,
            "tracks": [
                {
                    "track_index": 0,
                    "track_name": "Lead Vox",
                    "ok": True,
                    "devices_with_automation": 1,
                    "automated_parameter_count": 1,
                    "track_mixer_targets": {},
                    "warnings": [],
                },
                {
                    "track_index": 1,
                    "track_name": "Guitars",
                    "ok": True,
                    "devices_with_automation": 0,
                    "automated_parameter_count": 0,
                    "track_mixer_targets": {},
                    "warnings": [],
                },
            ],
            "warnings": [],
        }
        track_targets = {
            "ok": True,
            "track_name": "Lead Vox",
            "summary": {
                "device_count": 1,
                "devices_with_automation": 1,
                "automated_parameter_count": 1,
            },
            "track_mixer_targets": {},
            "devices": [
                {
                    "device_index": 0,
                    "device_name": "Utility",
                    "class_name": "StereoGain",
                    "has_automation": True,
                    "automated_parameter_count": 1,
                    "parameters": [
                        {
                            "parameter_index": 10,
                            "name": "Mute",
                            "automation_state": 1,
                            "automated": True,
                        }
                    ],
                }
            ],
            "warnings": [],
        }
        mix_topology = {
            "ok": True,
            "tracks": [
                {"index": 0, "sends": [{"send_index": 0}]},
                {"index": 1, "sends": []},
            ],
        }
        non_track_inventory = {
            "ok": True,
            "supported": True,
            "als_file_path": "/tmp/DeadEnemiesMidterm.als",
            "warnings": ["saved_als_snapshot_only"],
            "returns": [
                {
                    "index": 0,
                    "track_name": "A Verb",
                    "targets": {
                        "volume": {
                            "ok": True,
                            "supported": True,
                            "envelope_exists": True,
                            "point_access_supported": True,
                            "points": [{"point_index": 0, "time_beats": 1.0, "value": 0.2}],
                            "sampled_series": [],
                            "warnings": [],
                            "target": {"scope": "track_mixer", "mixer_target": "volume"},
                        },
                        "panning": {
                            "ok": True,
                            "supported": True,
                            "envelope_exists": False,
                            "point_access_supported": True,
                            "points": [],
                            "sampled_series": [],
                            "warnings": [],
                            "target": {"scope": "track_mixer", "mixer_target": "panning"},
                        },
                    },
                    "sends": [],
                }
            ],
            "master": {
                "track_name": "Main",
                "targets": {
                    "volume": {
                        "ok": True,
                        "supported": True,
                        "envelope_exists": True,
                        "point_access_supported": True,
                        "points": [{"point_index": 0, "time_beats": 4.0, "value": 0.9}],
                        "sampled_series": [],
                        "warnings": [],
                        "target": {"scope": "main_track", "target_kind": "volume"},
                    },
                    "panning": {
                        "ok": True,
                        "supported": True,
                        "envelope_exists": False,
                        "point_access_supported": True,
                        "points": [],
                        "sampled_series": [],
                        "warnings": [],
                        "target": {"scope": "main_track", "target_kind": "panning"},
                    },
                    "crossfade": {
                        "ok": True,
                        "supported": True,
                        "envelope_exists": False,
                        "point_access_supported": True,
                        "points": [],
                        "sampled_series": [],
                        "warnings": [],
                        "target": {"scope": "main_track", "target_kind": "crossfade"},
                    },
                },
            },
            "global": {
                "track_name": "Main",
                "targets": {
                    "tempo": {
                        "ok": True,
                        "supported": True,
                        "envelope_exists": True,
                        "point_access_supported": True,
                        "points": [{"point_index": 0, "time_beats": 0.0, "value": 120.0}],
                        "sampled_series": [],
                        "warnings": [],
                        "target": {"scope": "main_track", "target_kind": "tempo"},
                    },
                    "time_signature": {
                        "ok": True,
                        "supported": True,
                        "envelope_exists": False,
                        "point_access_supported": True,
                        "points": [],
                        "sampled_series": [],
                        "warnings": [],
                        "target": {"scope": "main_track", "target_kind": "time_signature"},
                    },
                },
            },
        }

        def _fake_arrangement_probe(ctx, **kwargs):
            target_scope = kwargs.get("scope")
            mixer_target = kwargs.get("mixer_target")
            if target_scope == "device_parameter":
                return {
                    "ok": True,
                    "supported": False,
                    "reason": "song_automation_envelope_unavailable",
                    "envelope_exists": False,
                    "point_access_supported": False,
                    "points": [],
                    "sampled_series": [],
                    "warnings": ["device_parameter_mapping_mismatch"],
                    "als_fallback_reason": "device_parameter_mapping_mismatch",
                    "target": {
                        "scope": "device_parameter",
                        "device_index": kwargs.get("device_index"),
                        "parameter_index": kwargs.get("parameter_index"),
                        "device_name": "Utility",
                        "parameter_name": "Mute",
                    },
                }
            envelope_exists = kwargs.get("track_index") == 0 and mixer_target == "volume"
            return {
                "ok": True,
                "supported": True,
                "envelope_exists": envelope_exists,
                "point_access_supported": True,
                "points": (
                    [{"point_index": 0, "time_beats": 1.0, "value": 0.5}] if envelope_exists else []
                ),
                "sampled_series": [],
                "warnings": [],
                "point_source": "als_arrangement_file",
                "target": {
                    "scope": "track_mixer",
                    "mixer_target": mixer_target,
                    "send_index": kwargs.get("send_index"),
                },
            }

        def _fake_clip_probe(ctx, **kwargs):
            is_hit = (
                kwargs.get("track_index") == 0
                and kwargs.get("clip_scope") == "session"
                and kwargs.get("clip_slot_index") == 0
                and kwargs.get("scope") == "track_mixer"
                and kwargs.get("mixer_target") == "volume"
            )
            return {
                "ok": True,
                "supported": True,
                "envelope_exists": is_hit,
                "point_access_supported": True,
                "points": ([{"point_index": 0, "time_beats": 2.0, "value": 0.8}] if is_hit else []),
                "sampled_series": [],
                "warnings": [],
                "clip": {
                    "clip_scope": kwargs.get("clip_scope"),
                    "clip_slot_index": kwargs.get("clip_slot_index"),
                    "clip_index": kwargs.get("clip_index"),
                    "clip_name": "Lead Clip",
                },
                "target": {
                    "scope": kwargs.get("scope"),
                    "mixer_target": kwargs.get("mixer_target"),
                    "device_index": kwargs.get("device_index"),
                    "parameter_index": kwargs.get("parameter_index"),
                },
            }

        def _fake_list_session_clips(ctx, track_index):
            if track_index == 0:
                return {
                    "track_index": 0,
                    "track_name": "Lead Vox",
                    "clips": [
                        {
                            "clip_slot_index": 0,
                            "clip_name": "Lead Clip",
                            "is_audio_clip": True,
                            "is_midi_clip": False,
                        }
                    ],
                }
            return {"track_index": track_index, "track_name": "Guitars", "clips": []}

        def _fake_list_arrangement_clips(ctx, track_index):
            return {
                "supported": False,
                "track_index": track_index,
                "reason": "arrangement_clip_access_failed",
                "message": "not exposed",
            }

        with mock.patch.object(server, "get_session_info", return_value=session_info), \
            mock.patch.object(server, "get_automation_overview", return_value=automation_overview), \
            mock.patch.object(server, "get_mix_topology", return_value=mix_topology), \
            mock.patch.object(server, "get_track_automation_targets", return_value=track_targets), \
            mock.patch.object(server, "get_automation_envelope_points", side_effect=_fake_arrangement_probe), \
            mock.patch.object(server, "get_clip_automation_envelope_points", side_effect=_fake_clip_probe), \
            mock.patch.object(server, "list_session_clips", side_effect=_fake_list_session_clips), \
            mock.patch.object(server, "list_arrangement_clips", side_effect=_fake_list_arrangement_clips), \
            mock.patch.object(server, "get_return_tracks_info", return_value={"ok": True, "returns": []}), \
            mock.patch.object(server, "get_master_track_device_chain", return_value={"ok": True, "devices": []}), \
            mock.patch.object(server, "enumerate_non_track_arrangement_automation_from_project_als", return_value=non_track_inventory):
            result = server.enumerate_project_automation(
                None,
                include_device_parameter_points=True,
                include_clip_envelopes=True,
                include_arrangement_mixer_points=True,
                include_return_master_context=True,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["coverage"]["tracks_scanned"], 2)
        self.assertEqual(result["coverage"]["arrangement_mixer_point_queries"], 5)
        self.assertEqual(result["coverage"]["arrangement_mixer_envelopes_found"], 1)
        self.assertEqual(result["coverage"]["device_parameter_point_queries"], 1)
        self.assertEqual(result["coverage"]["device_parameter_als_mismatch_blocks"], 1)
        self.assertEqual(result["coverage"]["clip_envelope_probes"], 3)
        self.assertEqual(result["coverage"]["clip_envelopes_found"], 1)
        self.assertEqual(result["coverage"]["session_clips_seen"], 1)
        self.assertEqual(result["coverage"]["arrangement_clips_seen"], 0)
        self.assertEqual(result["coverage"]["return_point_queries"], 2)
        self.assertEqual(result["coverage"]["return_envelopes_found"], 1)
        self.assertEqual(result["coverage"]["master_point_queries"], 3)
        self.assertEqual(result["coverage"]["master_envelopes_found"], 1)
        self.assertEqual(result["coverage"]["global_point_queries"], 2)
        self.assertEqual(result["coverage"]["global_envelopes_found"], 1)
        self.assertEqual(result["tracks"][0]["clip_inventory"]["session"]["clip_count"], 1)
        self.assertFalse(result["tracks"][0]["clip_inventory"]["arrangement"]["supported"])
        self.assertEqual(
            result["tracks"][0]["clip_inventory"]["arrangement"]["reason"],
            "arrangement_clip_access_failed"
        )
        self.assertTrue(result["returns_master"]["automation_point_enumeration"]["returns_supported"])
        self.assertTrue(result["returns_master"]["automation_point_enumeration"]["master_supported"])
        self.assertTrue(result["returns_master"]["automation_point_enumeration"]["global_supported"])
        self.assertEqual(result["returns_master"]["als_non_track_automation"]["returns"][0]["track_name"], "A Verb")
        self.assertTrue(result["returns_master"]["als_non_track_automation"]["master"]["targets"]["volume"]["envelope_exists"])
        self.assertTrue(result["returns_master"]["als_non_track_automation"]["global"]["targets"]["tempo"]["envelope_exists"])
        gap_reasons = {g.get("reason") for g in result["gaps"] if isinstance(g, dict)}
        self.assertIn("only_tempo_and_time_signature_are_enumerated", gap_reasons)
        self.assertIn("some_als_device_parameter_mappings_blocked_for_safety", gap_reasons)

    def test_enumerate_project_automation_passes_explicit_als_file_path(self):
        captured = []

        def _capture_arrangement_probe(ctx, **kwargs):
            captured.append(kwargs)
            return {
                "ok": True,
                "supported": False,
                "reason": "song_automation_envelope_unavailable",
                "envelope_exists": False,
                "point_access_supported": False,
                "points": [],
                "sampled_series": [],
                "warnings": [],
                "target": {
                    "scope": kwargs.get("scope"),
                    "mixer_target": kwargs.get("mixer_target"),
                    "send_index": kwargs.get("send_index"),
                },
            }

        with mock.patch.object(server, "get_session_info", return_value={
            "tempo": 120.0,
            "signature_numerator": 4,
            "signature_denominator": 4,
            "track_count": 1,
            "return_track_count": 0,
        }), \
            mock.patch.object(server, "get_automation_overview", return_value={
                "ok": True,
                "tracks_scanned": 1,
                "tracks_with_device_automation": 0,
                "total_automated_parameters": 0,
                "envelope_points_supported": False,
                "tracks": [{
                    "track_index": 0,
                    "track_name": "Lead Vox",
                    "ok": True,
                    "devices_with_automation": 0,
                    "automated_parameter_count": 0,
                    "track_mixer_targets": {},
                    "warnings": [],
                }],
                "warnings": [],
            }), \
            mock.patch.object(server, "get_mix_topology", return_value={"ok": True, "tracks": [{"index": 0, "sends": []}]}), \
            mock.patch.object(server, "get_automation_envelope_points", side_effect=_capture_arrangement_probe), \
            mock.patch.object(server, "enumerate_non_track_arrangement_automation_from_project_als", return_value={
                "ok": True,
                "supported": False,
                "reason": "top_level_als_not_found",
                "warnings": [],
                "returns": [],
                "master": None,
                "global": None,
            }):
            result = server.enumerate_project_automation(
                None,
                als_file_path="/tmp/DeadEnemiesMidterm.als",
                include_clip_envelopes=False,
                include_device_parameter_points=False,
                include_return_master_context=False,
            )

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(len(captured), 2)  # volume + panning
        self.assertTrue(all(call.get("als_file_path") == "/tmp/DeadEnemiesMidterm.als" for call in captured))

    def test_enumerate_project_automation_exhaustive_pages_and_strips_points(self):
        fake_inventory = self._fake_exhaustive_inventory()
        with mock.patch.object(
            server,
            "_build_or_get_als_exhaustive_inventory",
            return_value=(fake_inventory, {"cache_hit": True, "cache_layer": "process_memory"})
        ):
            result = server.enumerate_project_automation_exhaustive(
                None,
                include_point_payloads=False,
                include_live_hints=False,
                include_unclassified=False,
                include_orphans=False,
                page_size=2,
                cursor=None,
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertEqual(result["page"]["returned_targets"], 2)
        self.assertEqual(result["page"]["next_cursor"], "2")
        self.assertEqual(result["page"]["total_targets"], 3)
        self.assertFalse(result["scope_statement"]["session_clip_envelopes_excluded"])
        self.assertTrue(result["scope_statement"]["session_clip_envelopes_included"])
        self.assertNotIn("orphan_envelopes", result)
        self.assertNotIn("unclassified_targets", result)
        self.assertNotIn("points", result["targets"][0])
        self.assertTrue(result["cache"]["cache_hit"])

    def test_get_automation_target_points_resolves_from_exhaustive_inventory(self):
        fake_inventory = self._fake_exhaustive_inventory()
        target_ref = fake_inventory["targets"][0]["target_ref"]

        with mock.patch.object(
            server,
            "_build_or_get_als_exhaustive_inventory",
            return_value=(fake_inventory, {"cache_hit": False, "cache_layer": "process_memory"})
        ) as mock_build, mock.patch.object(
            server,
            "_build_exhaustive_runtime_hint_index",
            return_value=(
                {
                    0: {
                        "ok": True,
                        "track_name": "Lead Vox",
                        "track_mixer_targets": {},
                        "device_parameters": {
                            (0, 1): {"parameter_index": 1, "name": "Gain", "automation_state": 1, "automated": True}
                        },
                    }
                },
                []
            )
        ):
            result = server.get_automation_target_points(
                None,
                target_ref=target_ref,
                include_live_hints=True,
                start_time_beats=0.0,
                end_time_beats=8.0,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["inventory_key"], "k1")
        self.assertTrue(result["envelope_exists"])
        self.assertEqual(len(result["points"]), 2)
        self.assertEqual(result["als_file_path"], "/tmp/DeadEnemiesMidterm.als")
        self.assertFalse(result["scope_statement"]["session_clip_envelopes_excluded"])
        self.assertTrue(result["scope_statement"]["session_clip_envelopes_included"])
        self.assertTrue(result["live_hints"]["available"])
        self.assertEqual(result["live_hints"]["automation_state"], 1)
        self.assertTrue(result["live_hints_included"])
        kwargs = mock_build.call_args.kwargs
        self.assertTrue(kwargs["include_session_clip_envelopes"])
        self.assertTrue(kwargs["include_arrangement_clip_envelopes"])
        self.assertEqual(kwargs["start_time_beats"], 0.0)
        self.assertEqual(kwargs["end_time_beats"], 8.0)


if __name__ == "__main__":
    unittest.main()

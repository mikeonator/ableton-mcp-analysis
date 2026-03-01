import gzip
import os
import tempfile
import unittest

from MCP_Server.als_automation import (
    read_arrangement_automation_from_project_als,
    enumerate_non_track_arrangement_automation_from_project_als,
    build_als_automation_inventory,
    get_als_automation_target_points_from_inventory,
    read_time_locators_from_project_als,
)


_SYNTH_ALS_XML = """<?xml version="1.0" encoding="utf-8"?>
<Ableton>
  <LiveSet>
    <Tracks>
      <AudioTrack Id="10">
        <Name><EffectiveName Value="Lead Vox" /></Name>
        <AutomationEnvelopes>
          <Envelopes>
            <AutomationEnvelope>
              <EnvelopeTarget><PointeeId Value="200" /></EnvelopeTarget>
              <Automation>
                <Events>
                  <FloatEvent Id="0" Time="-63072000" Value="0.50" />
                  <FloatEvent Id="1" Time="8" Value="0.75" CurveControl1X="0.1" CurveControl1Y="0.2" />
                </Events>
              </Automation>
            </AutomationEnvelope>
            <AutomationEnvelope>
              <EnvelopeTarget><PointeeId Value="301" /></EnvelopeTarget>
              <Automation>
                <Events>
                  <FloatEvent Id="0" Time="4" Value="0.15" />
                  <FloatEvent Id="1" Time="12" Value="0.65" />
                </Events>
              </Automation>
            </AutomationEnvelope>
          </Envelopes>
        </AutomationEnvelopes>
        <DeviceChain>
          <Mixer>
            <Volume><AutomationTarget Id="200" /></Volume>
            <Pan><AutomationTarget Id="201" /></Pan>
            <Sends>
              <TrackSendHolder Id="0"><Send><AutomationTarget Id="202" /></Send></TrackSendHolder>
            </Sends>
          </Mixer>
          <DeviceChain>
            <Devices>
              <InstrumentGroupDevice Id="0">
                <ParametersListWrapper LomId="0" />
                <Pointee Id="900" />
                <On><AutomationTarget Id="300" /></On>
                <MacroControls.0><AutomationTarget Id="301" /></MacroControls.0>
                <Branches>
                  <Branch Id="0">
                    <DeviceChain>
                      <Devices>
                        <OriginalSimpler Id="8">
                          <ParametersListWrapper LomId="0" />
                          <Pointee Id="901" />
                          <On><AutomationTarget Id="9999" /></On>
                        </OriginalSimpler>
                      </Devices>
                    </DeviceChain>
                  </Branch>
                </Branches>
              </InstrumentGroupDevice>
            </Devices>
          </DeviceChain>
        </DeviceChain>
        <MainSequencer>
          <ClipSlotList>
            <ClipSlot Id="0">
              <ClipSlot>
                <Value>
                  <AudioClip Id="501">
                    <Name Value="SessionLead" />
                    <Envelopes>
                      <Envelopes>
                        <ClipEnvelope>
                          <EnvelopeTarget><PointeeId Value="97199" /></EnvelopeTarget>
                          <Automation>
                            <Events>
                              <FloatEvent Id="0" Time="0" Value="0.20" />
                              <FloatEvent Id="1" Time="4" Value="0.65" />
                            </Events>
                          </Automation>
                        </ClipEnvelope>
                      </Envelopes>
                    </Envelopes>
                  </AudioClip>
                </Value>
              </ClipSlot>
            </ClipSlot>
          </ClipSlotList>
          <Sample>
            <ArrangerAutomation>
              <Events>
                <AudioClip Id="500">
                  <Name Value="11_Tom3" />
                  <Envelopes>
                    <Envelopes>
                      <ClipEnvelope>
                        <EnvelopeTarget><PointeeId Value="301" /></EnvelopeTarget>
                        <Automation>
                          <Events>
                            <FloatEvent Id="0" Time="1" Value="0.10" />
                            <FloatEvent Id="1" Time="2" Value="0.80" CurveControl1X="0.2" CurveControl1Y="0.3" />
                          </Events>
                        </Automation>
                      </ClipEnvelope>
                    </Envelopes>
                  </Envelopes>
                </AudioClip>
              </Events>
            </ArrangerAutomation>
          </Sample>
        </MainSequencer>
      </AudioTrack>
      <ReturnTrack Id="20">
        <Name><EffectiveName Value="A Verb" /></Name>
        <AutomationEnvelopes>
          <Envelopes>
            <AutomationEnvelope>
              <EnvelopeTarget><PointeeId Value="210" /></EnvelopeTarget>
              <Automation>
                <Events>
                  <FloatEvent Id="0" Time="2" Value="0.33" />
                </Events>
              </Automation>
            </AutomationEnvelope>
          </Envelopes>
        </AutomationEnvelopes>
        <DeviceChain>
          <Mixer>
            <Volume><AutomationTarget Id="210" /></Volume>
            <Pan><AutomationTarget Id="211" /></Pan>
            <Sends />
          </Mixer>
          <DeviceChain><Devices /></DeviceChain>
        </DeviceChain>
      </ReturnTrack>
      <MidiTrack Id="30">
        <Name><EffectiveName Value="Bass" /></Name>
        <AutomationEnvelopes>
          <Envelopes>
            <AutomationEnvelope>
              <EnvelopeTarget><PointeeId Value="400" /></EnvelopeTarget>
              <Automation>
                <Events>
                  <BoolEvent Id="0" Time="16" Value="true" />
                  <BoolEvent Id="1" Time="20" Value="false" />
                </Events>
              </Automation>
            </AutomationEnvelope>
          </Envelopes>
        </AutomationEnvelopes>
        <DeviceChain>
          <Mixer>
            <Volume><AutomationTarget Id="400" /></Volume>
            <Pan><AutomationTarget Id="401" /></Pan>
            <Sends />
          </Mixer>
          <DeviceChain><Devices /></DeviceChain>
        </DeviceChain>
      </MidiTrack>
    </Tracks>
    <MainTrack Id="99">
      <Name><EffectiveName Value="Main" /></Name>
      <AutomationEnvelopes>
        <Envelopes>
          <AutomationEnvelope>
            <EnvelopeTarget><PointeeId Value="5" /></EnvelopeTarget>
            <Automation>
              <Events>
                <FloatEvent Id="0" Time="1" Value="0.90" />
              </Events>
            </Automation>
          </AutomationEnvelope>
          <AutomationEnvelope>
            <EnvelopeTarget><PointeeId Value="8" /></EnvelopeTarget>
            <Automation>
              <Events>
                <FloatEvent Id="0" Time="0" Value="120.0" />
                <FloatEvent Id="1" Time="16" Value="128.0" />
              </Events>
            </Automation>
          </AutomationEnvelope>
          <AutomationEnvelope>
            <EnvelopeTarget><PointeeId Value="10" /></EnvelopeTarget>
            <Automation>
              <Events>
                <EnumEvent Id="0" Time="0" Value="0" />
                <EnumEvent Id="1" Time="32" Value="1" />
              </Events>
            </Automation>
          </AutomationEnvelope>
        </Envelopes>
      </AutomationEnvelopes>
      <DeviceChain>
        <Mixer>
          <Volume><AutomationTarget Id="5" /></Volume>
          <Pan><AutomationTarget Id="3" /></Pan>
          <Tempo><AutomationTarget Id="8" /></Tempo>
          <TimeSignature><AutomationTarget Id="10" /></TimeSignature>
          <CrossFade><AutomationTarget Id="13" /></CrossFade>
        </Mixer>
        <DeviceChain><Devices /></DeviceChain>
      </DeviceChain>
    </MainTrack>
    <PreHearTrack Id="100">
      <Name><EffectiveName Value="Cue" /></Name>
      <AutomationEnvelopes>
        <Envelopes />
      </AutomationEnvelopes>
      <DeviceChain>
        <Mixer>
          <Volume><AutomationTarget Id="501" /></Volume>
          <Pan><AutomationTarget Id="502" /></Pan>
          <Speaker><AutomationTarget Id="503" /></Speaker>
          <CrossFadeState><AutomationTarget Id="504" /></CrossFadeState>
        </Mixer>
        <DeviceChain><Devices /></DeviceChain>
      </DeviceChain>
    </PreHearTrack>
  </LiveSet>
</Ableton>
"""


_LOCATOR_ALS_XML = """<?xml version="1.0" encoding="utf-8"?>
<Ableton>
  <LiveSet>
    <Locators>
      <Locators>
        <Locator Id="2">
          <Time Value="32" />
          <Name>
            <EffectiveName Value="Verse" />
          </Name>
        </Locator>
        <Locator Id="1">
          <Time Value="8" />
          <Name>
            <EffectiveName Value="Intro" />
          </Name>
        </Locator>
      </Locators>
    </Locators>
  </LiveSet>
</Ableton>
"""


class AlsAutomationParserTests(unittest.TestCase):
    def _write_als(self, project_root: str, file_name: str = "Test Set.als") -> str:
        os.makedirs(project_root, exist_ok=True)
        als_path = os.path.join(project_root, file_name)
        with gzip.open(als_path, "wb") as handle:
            handle.write(_SYNTH_ALS_XML.encode("utf-8"))
        return als_path

    def test_reads_track_mixer_volume_points_from_als(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_als(tmpdir)

            result = read_arrangement_automation_from_project_als(
                project_root=tmpdir,
                track_index=0,
                scope="track_mixer",
                mixer_target="volume",
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertTrue(result["envelope_exists"])
        self.assertEqual(result["track_name"], "Lead Vox")
        self.assertEqual(result["target"]["automation_target_id"], 200)
        self.assertEqual([row["time_beats"] for row in result["points"]], [-63072000.0, 8.0])
        self.assertEqual(result["points"][1]["shape"], "bezier")
        self.assertIn("saved_als_snapshot_only", result["warnings"])

    def test_skips_return_tracks_when_mapping_normal_track_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_als(tmpdir)

            result = read_arrangement_automation_from_project_als(
                project_root=tmpdir,
                track_index=1,  # second normal track, after a return track in XML order
                scope="track_mixer",
                mixer_target="volume",
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertEqual(result["track_name"], "Bass")
        self.assertEqual(result["target"]["automation_target_id"], 400)
        self.assertEqual([row["value"] for row in result["points"]], [True, False])
        self.assertEqual(result["points"][0]["value_kind"], "bool")

    def test_device_parameter_mapping_uses_top_level_device_parameter_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_als(tmpdir)

            result = read_arrangement_automation_from_project_als(
                project_root=tmpdir,
                track_index=0,
                scope="device_parameter",
                mixer_target="volume",
                device_index=0,
                parameter_index=1,  # MacroControls.0 (nested Simpler On should be skipped)
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertTrue(result["envelope_exists"])
        self.assertEqual(result["target"]["automation_target_id"], 301)
        self.assertEqual(result["target"]["parameter_xml_tag"], "MacroControls.0")
        self.assertEqual([row["time_beats"] for row in result["points"]], [4.0, 12.0])

    def test_reads_with_explicit_als_file_path_override_when_project_folder_scan_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            als_path = self._write_als(tmpdir, file_name="DeadEnemiesMidterm.als")

            # project_root intentionally invalid to prove explicit path override is used.
            result = read_arrangement_automation_from_project_als(
                project_root=os.path.join(tmpdir, "missing"),
                track_index=0,
                scope="track_mixer",
                mixer_target="volume",
                als_file_path=als_path,
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertEqual(result["als_file_path"], als_path)
        self.assertIn("als_file_path_override", result["warnings"])

    def test_enumerates_return_master_and_global_automation_from_als(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_als(tmpdir)
            result = enumerate_non_track_arrangement_automation_from_project_als(
                project_root=tmpdir
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertEqual(result["returns"][0]["track_name"], "A Verb")
        self.assertTrue(result["returns"][0]["targets"]["volume"]["envelope_exists"])
        self.assertEqual(result["returns"][0]["targets"]["volume"]["points"][0]["time_beats"], 2.0)
        self.assertEqual(result["master"]["track_name"], "Main")
        self.assertTrue(result["master"]["targets"]["volume"]["envelope_exists"])
        self.assertEqual(result["master"]["targets"]["volume"]["target"]["automation_target_id"], 5)
        self.assertTrue(result["global"]["targets"]["tempo"]["envelope_exists"])
        self.assertEqual([p["value"] for p in result["global"]["targets"]["tempo"]["points"]], [120.0, 128.0])
        self.assertEqual([p["value"] for p in result["global"]["targets"]["time_signature"]["points"]], [0, 1])

    def test_builds_exhaustive_inventory_including_session_and_arrangement_clip_envelopes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_als(tmpdir)
            result = build_als_automation_inventory(project_root=tmpdir)

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertEqual(result["source"], "als_file")
        self.assertFalse(result["scope_statement"]["session_clip_envelopes_excluded"])
        self.assertTrue(result["scope_statement"]["session_clip_envelopes_included"])
        self.assertTrue(result["scope_statement"]["arrangement_clip_envelopes_included"])
        self.assertEqual(result["session"]["track_count_normal"], 2)
        self.assertEqual(result["session"]["track_count_returns"], 1)
        self.assertTrue(result["session"]["has_main_track"])
        self.assertTrue(result["session"]["has_prehear_track"])
        self.assertGreater(result["completeness"]["targets_discovered_total"], 0)
        self.assertEqual(result["completeness"]["orphan_envelopes_total"], 0)

        clip_rows = [
            row for row in result["targets"]
            if isinstance(row, dict) and row.get("container_scope") == "clip_arrangement"
        ]
        self.assertEqual(len(clip_rows), 1)
        self.assertEqual(clip_rows[0]["classification"]["domain"], "clip_envelope")
        self.assertEqual(clip_rows[0]["envelope"]["envelope_kind"], "ClipEnvelope")
        self.assertEqual(clip_rows[0]["envelope"]["point_count"], 2)
        self.assertEqual(clip_rows[0]["location"]["clip_name"], "11_Tom3")

        session_clip_rows = [
            row for row in result["targets"]
            if isinstance(row, dict) and row.get("container_scope") == "clip_session"
        ]
        self.assertEqual(len(session_clip_rows), 1)
        self.assertEqual(session_clip_rows[0]["classification"]["clip_scope"], "session")
        self.assertEqual(session_clip_rows[0]["envelope"]["envelope_kind"], "ClipEnvelope")
        self.assertEqual(session_clip_rows[0]["envelope"]["point_count"], 2)
        self.assertEqual(session_clip_rows[0]["location"]["clip_name"], "SessionLead")

        nested_param_rows = [
            row for row in result["targets"]
            if isinstance(row, dict)
            and row.get("classification", {}).get("scope") == "device_parameter"
            and row.get("automation_target_id") == 9999
        ]
        self.assertEqual(len(nested_param_rows), 1)
        nested_device_path = nested_param_rows[0]["classification"]["device_path"]
        self.assertGreaterEqual(len(nested_device_path["device_index_path"]), 1)
        self.assertEqual(nested_param_rows[0]["envelope"]["exists"], False)

    def test_resolves_target_points_from_exhaustive_inventory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_als(tmpdir)
            inventory = build_als_automation_inventory(project_root=tmpdir)

        target_row = next(
            row for row in inventory["targets"]
            if row.get("automation_target_id") == 200 and row.get("container_scope") == "track"
        )
        resolved = get_als_automation_target_points_from_inventory(inventory, target_row["target_ref"])

        self.assertTrue(resolved["ok"])
        self.assertTrue(resolved["envelope_exists"])
        self.assertEqual(resolved["envelope_kind"], "AutomationEnvelope")
        self.assertEqual([p["time_beats"] for p in resolved["points"]], [-63072000.0, 8.0])
        self.assertEqual(resolved["target_ref"]["inventory_key"], target_row["target_ref"]["inventory_key"])

    def test_can_exclude_session_clip_envelopes_from_exhaustive_inventory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_als(tmpdir)
            inventory = build_als_automation_inventory(
                project_root=tmpdir,
                include_arrangement_clip_envelopes=True,
                include_session_clip_envelopes=False,
            )

        self.assertTrue(inventory["ok"])
        self.assertTrue(inventory["scope_statement"]["session_clip_envelopes_excluded"])
        session_rows = [
            row for row in inventory["targets"]
            if isinstance(row, dict) and row.get("container_scope") == "clip_session"
        ]
        self.assertEqual(session_rows, [])

    def test_reads_time_locators_from_als_and_sorts_by_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            als_path = os.path.join(tmpdir, "Locators.als")
            with gzip.open(als_path, "wb") as handle:
                handle.write(_LOCATOR_ALS_XML.encode("utf-8"))

            result = read_time_locators_from_project_als(
                project_root=tmpdir,
                als_file_path=als_path,
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["supported"])
        self.assertEqual(result["locator_count"], 2)
        self.assertEqual([row["name"] for row in result["locators"]], ["Intro", "Verse"])
        self.assertEqual([row["time_beats"] for row in result["locators"]], [8.0, 32.0])
        self.assertIn("saved_als_snapshot_only", result["warnings"])


if __name__ == "__main__":
    unittest.main()

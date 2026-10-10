import unittest

from scenarios.scenario_matrix_registry import ScenarioMatrixRegistry
from scenarios.scenario_generator import ScenarioGenerator
from scenarios.scenario_registry import ScenarioRegistry


class TestScenarioGenerator(unittest.TestCase):
    def test_registry_resolves_template_with_overrides(self) -> None:
        template = ScenarioRegistry().resolve(
            {
                "scenario_name": "high_mobility_sparse",
                "template_overrides": {
                    "radio": {"carrier_frequency_hz": 28.0e9},
                    "channel": {"los_mode": "nlos"},
                },
            }
        )

        self.assertEqual(template.name, "high_mobility_sparse")
        self.assertEqual(template.radio.carrier_frequency_hz, 28.0e9)
        self.assertEqual(template.channel.los_mode, "nlos")

    def test_generator_populates_geometry_and_link_defaults(self) -> None:
        registry = ScenarioRegistry()
        template = registry.resolve({"scenario_name": "high_mobility_sparse"})
        bundle = ScenarioGenerator().generate(
            template=template,
            scenario_config={"scenario_name": "high_mobility_sparse"},
            sim_config={
                "total_slots": 30,
                "slot_duration_ms": 1.0,
                "rng_seed": 7,
            },
            cell_config={
                "cell_id": "cell_0",
                "tx_power_dbm": 30.0,
                "current_codebook_id": 0,
                "current_beam_profile_id": 0,
            },
            ue_config={
                "ues": [
                    {
                        "ue_id": "ue_0",
                        "queue_bytes": 1000,
                        "throughput_mbps": 0.0,
                        "sinr_db": 10.0,
                        "radio_quality_db": 9.0,
                        "bler": 0.1,
                        "slice_id": "embb",
                        "traffic_type": "video",
                        "latency_budget_ms": 100.0,
                        "priority": 1,
                    }
                ]
            },
            beamforming_config={
                "antenna": {},
                "codebooks": {
                    0: {
                        "beams": {
                            0: 0.0,
                            1: 0.2,
                            2: 0.4,
                        }
                    }
                },
                "beam_profiles": {0: {"profile_gain_db": 0.0}},
            },
        )

        self.assertEqual(bundle.scenario_name, "high_mobility_sparse")
        self.assertEqual(bundle.beamforming_config["antenna"]["carrier_frequency_hz"], 3.5e9)
        ue_item = bundle.ue_config["ues"][0]
        geometry = bundle.ue_geometries["ue_0"]

        self.assertGreater(geometry.distance_2d_m, 0.0)
        self.assertGreater(geometry.distance_3d_m, geometry.distance_2d_m)
        self.assertEqual(ue_item["scenario_type"], "uma")
        self.assertIn(ue_item["los_state"], {"los", "nlos"})
        self.assertGreater(ue_item["pathloss_db"], 0.0)
        self.assertGreater(ue_item["doppler_hz"], 0.0)
        self.assertGreaterEqual(ue_item["serving_beam_id"], 0)
        self.assertIn(ue_item["channel_model_type"], {"tdl_c", "tdl_d"})
        self.assertIn("angle_spread_deg", ue_item)
        self.assertIn("spatial_correlation_level", ue_item)
        self.assertIn("large_scale", ue_item)
        self.assertIn("small_scale", ue_item)
        self.assertEqual(ue_item["large_scale"]["pathloss_db"], ue_item["pathloss_db"])
        self.assertEqual(ue_item["small_scale"]["doppler_hz"], ue_item["doppler_hz"])

    def test_generator_populates_candidate_cells_for_multi_cell_scene(self) -> None:
        registry = ScenarioRegistry()
        template = registry.resolve({"scenario_name": "mixed_traffic_congested"})

        bundle = ScenarioGenerator().generate(
            template=template,
            scenario_config={"scenario_name": "mixed_traffic_congested"},
            sim_config={"total_slots": 10, "slot_duration_ms": 1.0, "rng_seed": 5},
            cell_config={
                "cell_id": "cell_0",
                "tx_power_dbm": 30.0,
                "current_codebook_id": 0,
                "current_beam_profile_id": 0,
            },
            ue_config={
                "ues": [
                    {
                        "ue_id": "ue_0",
                        "queue_bytes": 1000,
                        "throughput_mbps": 0.0,
                        "sinr_db": 10.0,
                        "radio_quality_db": 9.0,
                        "bler": 0.1,
                        "slice_id": "embb",
                        "traffic_type": "video",
                        "latency_budget_ms": 100.0,
                        "priority": 1,
                    }
                ]
            },
            beamforming_config={
                "antenna": {},
                "codebooks": {0: {"beams": {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6}}},
                "beam_profiles": {0: {"profile_gain_db": 0.0}},
            },
        )

        ue_item = bundle.ue_config["ues"][0]

        self.assertGreaterEqual(len(bundle.cell_config["managed_cell_ids"]), 3)
        self.assertGreaterEqual(len(ue_item["candidate_cells"]), 3)
        self.assertEqual(ue_item["serving_cell_id"], ue_item["candidate_cells"][0]["cell_id"])
        self.assertIn("cell_layout", bundle.scenario_metadata)

    def test_generator_builds_heterogeneous_macro_mmwave_cell_profiles(self) -> None:
        registry = ScenarioRegistry()
        template = registry.resolve({"scenario_name": "heterogeneous_macro_mmwave_benchmark"})

        bundle = ScenarioGenerator().generate(
            template=template,
            scenario_config={"scenario_name": "heterogeneous_macro_mmwave_benchmark"},
            sim_config={"total_slots": 10, "slot_duration_ms": 1.0, "rng_seed": 9},
            cell_config={
                "cell_id": "cell_0",
                "tx_power_dbm": 30.0,
                "current_codebook_id": 0,
                "current_beam_profile_id": 0,
            },
            ue_config={
                "ues": [
                    {
                        "ue_id": "ue_hotspot_probe",
                        "queue_bytes": 1200,
                        "throughput_mbps": 0.0,
                        "sinr_db": 12.0,
                        "radio_quality_db": 11.0,
                        "bler": 0.08,
                        "slice_id": "embb",
                        "traffic_type": "video",
                        "latency_budget_ms": 100.0,
                        "priority": 1,
                        "x_m": 126.0,
                        "y_m": 92.0,
                    }
                ]
            },
            beamforming_config={
                "antenna": {},
                "codebooks": {0: {"beams": {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6}}},
                "beam_profiles": {0: {"profile_gain_db": 0.0}},
            },
        )

        cell_profiles = bundle.cell_config["cell_radio_profiles"]
        ue_item = bundle.ue_config["ues"][0]

        self.assertEqual(cell_profiles["cell_0"]["carrier_frequency_hz"], 3.5e9)
        self.assertEqual(cell_profiles["cell_1"]["carrier_frequency_hz"], 28.0e9)
        self.assertEqual(cell_profiles["cell_2"]["carrier_frequency_hz"], 26.0e9)
        self.assertEqual(cell_profiles["cell_0"]["rf_band"], "sub6")
        self.assertEqual(cell_profiles["cell_1"]["rf_band"], "mmwave")
        self.assertGreater(cell_profiles["cell_0"]["adjacent_channel_leakage_ratio_db"], 0.0)
        self.assertGreater(cell_profiles["cell_1"]["adjacent_channel_selectivity_db"], 0.0)
        self.assertGreater(cell_profiles["cell_2"]["out_of_band_isolation_db"], 0.0)
        self.assertEqual(ue_item["serving_cell_id"], "cell_1")
        self.assertEqual(ue_item["serving_cell_type"], "mmwave_hotspot")
        self.assertEqual(ue_item["candidate_cells"][0]["cell_type"], "mmwave_hotspot")
        self.assertEqual(ue_item["candidate_cells"][0]["carrier_frequency_hz"], 28.0e9)
        self.assertEqual(ue_item["candidate_cells"][0]["rf_band"], "mmwave")
        self.assertIn("adjacent_channel_leakage_ratio_db", ue_item["candidate_cells"][0])
        self.assertIn("adjacent_channel_selectivity_db", ue_item["candidate_cells"][0])
        self.assertIn("out_of_band_isolation_db", ue_item["candidate_cells"][0])
        self.assertIn("cell_radio_profiles", bundle.scenario_metadata)

    def test_generator_expands_oran_paper_medium_scene(self) -> None:
        case_map = {
            case.case_id: case
            for case in ScenarioMatrixRegistry().expand_matrix("oran_paper_eval_suite")
        }
        case = case_map["oran_medium_nominal_mixed"]
        template = ScenarioRegistry().resolve(
            {
                "scenario_name": case.scenario_name,
                "template_overrides": case.template_overrides,
            }
        )

        bundle = ScenarioGenerator().generate(
            template=template,
            scenario_config={"scenario_name": case.scenario_name},
            sim_config={"total_slots": 10, "slot_duration_ms": 1.0, "rng_seed": 11},
            cell_config={
                "cell_id": "cell_0",
                "tx_power_dbm": 30.0,
                "current_codebook_id": 0,
                "current_beam_profile_id": 0,
            },
            ue_config=case.ue_overrides,
            beamforming_config={
                "antenna": {},
                "codebooks": {0: {"beams": {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6}}},
                "beam_profiles": {0: {"profile_gain_db": 0.0}},
            },
        )

        self.assertEqual(len(bundle.cell_config["managed_cell_ids"]), 7)
        self.assertEqual(len(bundle.ue_config["ues"]), 42)
        self.assertGreaterEqual(len(bundle.ue_config["ues"][0]["candidate_cells"]), 1)
        self.assertIn("cell_6", bundle.cell_config["cell_radio_profiles"])
        rf_bands = {
            str(profile.get("rf_band", ""))
            for profile in bundle.cell_config["cell_radio_profiles"].values()
        }
        self.assertTrue({"sub6", "mmwave"}.issubset(rf_bands))


if __name__ == "__main__":
    unittest.main()

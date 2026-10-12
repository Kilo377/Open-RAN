from __future__ import annotations

import unittest

from scenarios.scenario_matrix_registry import ScenarioMatrixRegistry


class ScenarioMatrixRegistryTest(unittest.TestCase):
    def test_registry_lists_builtin_matrices(self) -> None:
        registry = ScenarioMatrixRegistry()

        matrix_names = registry.list_matrix_names()

        self.assertIn("scenario_catalog", matrix_names)
        self.assertIn("control_delay_scan", matrix_names)
        self.assertIn("radio_bandwidth_scs_scan", matrix_names)
        self.assertIn("orchestration_benchmark_suite", matrix_names)
        self.assertIn("oran_paper_eval_suite", matrix_names)

    def test_control_delay_scan_expands_cross_product(self) -> None:
        registry = ScenarioMatrixRegistry()

        cases = registry.expand_matrix("control_delay_scan")

        self.assertEqual(len(cases), 9)
        case_ids = {case.case_id for case in cases}
        self.assertIn("mixed_control_delay_scan__d1__w4", case_ids)
        self.assertIn("mixed_control_delay_scan__d5__w12", case_ids)

    def test_catalog_contains_large_population_cases(self) -> None:
        registry = ScenarioMatrixRegistry()

        cases = registry.expand_matrix("scenario_catalog")
        case_map = {case.case_id: case for case in cases}

        self.assertEqual(len(cases), 8)
        self.assertEqual(len(case_map["scene_c_mixed_traffic_congested"].ue_overrides["ues"]), 12)
        self.assertEqual(len(case_map["scene_g_mmtc_sparse_reporting"].ue_overrides["ues"]), 20)

    def test_orchestration_benchmark_suite_contains_fixed_cases(self) -> None:
        registry = ScenarioMatrixRegistry()

        cases = registry.expand_matrix("orchestration_benchmark_suite")
        case_map = {case.case_id: case for case in cases}

        self.assertEqual(len(cases), 5)
        self.assertIn("benchmark_hotspot_overload", case_map)
        self.assertIn("benchmark_edge_coverage", case_map)
        ue_items = case_map["benchmark_nominal_cluster"].ue_overrides["ues"]
        traffic_types = {str(ue.get("traffic_type", "")) for ue in ue_items}
        self.assertTrue({"video", "control", "phone", "iot"}.issubset(traffic_types))
        self.assertEqual(
            case_map["benchmark_stale_observation"].template_overrides["control"]["obs_delay_slots"],
            5,
        )

    def test_oran_paper_eval_suite_has_scale_and_stress_cases(self) -> None:
        registry = ScenarioMatrixRegistry()

        cases = registry.expand_matrix("oran_paper_eval_suite")
        case_map = {case.case_id: case for case in cases}

        self.assertEqual(set(case_map), {
            "oran_small_debug_mixed", "oran_medium_nominal_mixed",
            "oran_medium_dense_hotspot_high_mobility", "oran_medium_urllc_burst",
            "oran_medium_interference_edge", "oran_medium_hotspot_shift_blockage",
            "oran_medium_high_mobility_interference", "oran_medium_low_load_energy_saving",
            "oran_large_19cell_scalability",
        })
        self.assertEqual(len(case_map["oran_small_debug_mixed"].ue_overrides["ues"]), 24)
        self.assertEqual(len(case_map["oran_medium_nominal_mixed"].ue_overrides["ues"]), 42)
        self.assertEqual(
            len(case_map["oran_medium_dense_hotspot_high_mobility"].ue_overrides["ues"]),
            56,
        )
        self.assertEqual(len(case_map["oran_large_19cell_scalability"].ue_overrides["ues"]), 120)
        self.assertEqual(
            case_map["oran_medium_nominal_mixed"].template_overrides["topology"]["num_cells"],
            7,
        )
        self.assertEqual(
            case_map["oran_large_19cell_scalability"].template_overrides["topology"]["num_cells"],
            19,
        )
        traffic_types = {
            str(ue.get("traffic_type", ""))
            for ue in case_map["oran_medium_nominal_mixed"].ue_overrides["ues"]
        }
        self.assertTrue({"video", "control", "iot", "phone"}.issubset(traffic_types))


if __name__ == "__main__":
    unittest.main()

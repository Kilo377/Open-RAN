import unittest

from phy.sionna_engine import SionnaEngine


class TestSionnaResourceGrid(unittest.TestCase):
    def test_resource_grid_view_counts_re_and_positions(self) -> None:
        engine = SionnaEngine({"subcarriers_per_prb": 12, "max_ofdm_symbols_per_slot": 14})

        resource_view = engine._build_resource_grid_view(
            {
                "allocated_prbs": 6,
                "allocated_symbols": 4,
                "start_prb": 2,
                "start_symbol": 1,
                "slot_total_prbs": 12,
                "slot_total_symbols": 14,
            }
        )

        self.assertEqual(resource_view["start_prb"], 2)
        self.assertEqual(resource_view["end_prb"], 8)
        self.assertEqual(resource_view["start_symbol"], 1)
        self.assertEqual(resource_view["end_symbol"], 5)
        self.assertEqual(resource_view["resource_elements"], 6 * 4 * 12)

    def test_overlap_penalty_is_zero_for_disjoint_resources(self) -> None:
        engine = SionnaEngine({"enable_overlap_interference": True})
        disjoint_a = {
            "ue_id": "ue_a",
            "is_scheduled": True,
            "allocated_prbs": 4,
            "allocated_symbols": 8,
            "start_prb": 0,
            "start_symbol": 0,
            "slot_total_prbs": 12,
            "slot_total_symbols": 14,
            "beam_direction_deg": 0.0,
            "precoder_id": 1,
            "layer_count": 1,
            "num_rx_ant": 2,
        }
        disjoint_b = {
            "ue_id": "ue_b",
            "is_scheduled": True,
            "allocated_prbs": 4,
            "allocated_symbols": 8,
            "start_prb": 5,
            "start_symbol": 0,
            "slot_total_prbs": 12,
            "slot_total_symbols": 14,
            "beam_direction_deg": 90.0,
            "precoder_id": 2,
            "layer_count": 1,
            "num_rx_ant": 2,
        }

        penalty_db, overlap_fraction, _, _, _ = engine._compute_overlap_interference_penalty_db(
            disjoint_a,
            [disjoint_a, disjoint_b],
        )

        self.assertEqual(overlap_fraction, 0.0)
        self.assertEqual(penalty_db, 0.0)

    def test_overlap_penalty_grows_for_shared_resources(self) -> None:
        engine = SionnaEngine(
            {
                "enable_overlap_interference": True,
                "same_precoder_coupling_multiplier": 1.5,
                "min_spatial_coupling": 0.1,
            }
        )
        overlapping_a = {
            "ue_id": "ue_a",
            "is_scheduled": True,
            "allocated_prbs": 8,
            "allocated_symbols": 10,
            "start_prb": 0,
            "start_symbol": 0,
            "slot_total_prbs": 12,
            "slot_total_symbols": 14,
            "beam_direction_deg": 0.0,
            "precoder_id": 1,
            "layer_count": 2,
            "num_rx_ant": 2,
        }
        overlapping_b = {
            "ue_id": "ue_b",
            "is_scheduled": True,
            "allocated_prbs": 8,
            "allocated_symbols": 10,
            "start_prb": 2,
            "start_symbol": 0,
            "slot_total_prbs": 12,
            "slot_total_symbols": 14,
            "beam_direction_deg": 5.0,
            "precoder_id": 1,
            "layer_count": 2,
            "num_rx_ant": 2,
        }

        penalty_db, overlap_fraction, _, _, _ = engine._compute_overlap_interference_penalty_db(
            overlapping_a,
            [overlapping_a, overlapping_b],
        )

        self.assertGreater(overlap_fraction, 0.0)
        self.assertGreater(penalty_db, 0.0)

    def test_adjacent_prb_leakage_penalty_is_positive_without_overlap(self) -> None:
        engine = SionnaEngine(
            {
                "enable_overlap_interference": True,
                "adjacent_prb_leakage_factor": 0.2,
                "min_spatial_coupling": 0.1,
            }
        )
        left = {
            "ue_id": "ue_left",
            "is_scheduled": True,
            "allocated_prbs": 4,
            "allocated_symbols": 8,
            "start_prb": 0,
            "start_symbol": 0,
            "slot_total_prbs": 12,
            "slot_total_symbols": 14,
            "beam_direction_deg": 0.0,
            "precoder_id": 1,
            "layer_count": 2,
            "num_rx_ant": 2,
        }
        right = {
            "ue_id": "ue_right",
            "is_scheduled": True,
            "allocated_prbs": 4,
            "allocated_symbols": 8,
            "start_prb": 4,
            "start_symbol": 0,
            "slot_total_prbs": 12,
            "slot_total_symbols": 14,
            "beam_direction_deg": 10.0,
            "precoder_id": 1,
            "layer_count": 2,
            "num_rx_ant": 2,
        }

        penalty_db, overlap_fraction, _, _, _ = engine._compute_overlap_interference_penalty_db(
            left,
            [left, right],
        )

        self.assertEqual(overlap_fraction, 0.0)
        self.assertGreater(penalty_db, 0.0)

    def test_mu_mimo_group_reduces_penalty_and_adds_gain(self) -> None:
        engine = SionnaEngine(
            {
                "enable_overlap_interference": True,
                "same_group_interference_reduction": 0.8,
                "mu_mimo_bonus_db_max": 2.0,
            }
        )
        base_input = {
            "is_scheduled": True,
            "allocated_prbs": 6,
            "allocated_symbols": 8,
            "start_prb": 0,
            "start_symbol": 0,
            "slot_total_prbs": 12,
            "slot_total_symbols": 14,
            "layer_count": 1,
            "num_rx_ant": 2,
        }
        grouped_a = {
            "ue_id": "ue_a",
            "beam_direction_deg": 0.0,
            "precoder_id": 1,
            "spatial_group_id": 0,
            "spatial_group_size": 2,
            "spatial_compatibility_score": 0.9,
            **base_input,
        }
        grouped_b = {
            "ue_id": "ue_b",
            "beam_direction_deg": 120.0,
            "precoder_id": 2,
            "spatial_group_id": 0,
            "spatial_group_size": 2,
            "spatial_compatibility_score": 0.9,
            **base_input,
        }
        ungrouped_b = {
            "ue_id": "ue_b",
            "beam_direction_deg": 120.0,
            "precoder_id": 2,
            **base_input,
        }

        grouped_penalty_db, _, grouped_gain_db, grouped_score, grouped_size = engine._compute_overlap_interference_penalty_db(
            grouped_a,
            [grouped_a, grouped_b],
        )
        ungrouped_penalty_db, _, ungrouped_gain_db, _, _ = engine._compute_overlap_interference_penalty_db(
            grouped_a,
            [grouped_a, ungrouped_b],
        )

        self.assertLess(grouped_penalty_db, ungrouped_penalty_db)
        self.assertGreater(grouped_gain_db, 0.0)
        self.assertGreater(grouped_score, 0.0)
        self.assertEqual(grouped_size, 2)
        self.assertEqual(ungrouped_gain_db, 0.0)


if __name__ == "__main__":
    unittest.main()

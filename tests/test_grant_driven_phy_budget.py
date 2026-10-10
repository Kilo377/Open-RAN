import unittest

from models.phy_input import PhyInput, PhyUeInput
from models.phy_runtime_context import PhyRuntimeContext
from phy.custom_phy_adapter import CustomPhyAdapter


class TestGrantDrivenPhyBudget(unittest.TestCase):
    def test_custom_phy_respects_grant_target_mcs_and_budget(self) -> None:
        adapter = CustomPhyAdapter(
            codebook_map={0: {"base_gain_db": 0.0, "beams": {0: 0.0}}},
            beam_profile_map={0: {"profile_gain_db": 0.0}},
        )

        runtime_context = PhyRuntimeContext(
            slot_id=1,
            sim_time_ms=1.0,
            slot_duration_ms=1.0,
            cell_id="cell_0",
            scheduler_mode="latency_aware",
            tx_power_dbm=30.0,
            current_codebook_id=0,
            current_beam_profile_id=0,
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=20e6,
            subcarrier_spacing_hz=30e3,
            cyclic_prefix_type="normal",
            num_ofdm_symbols_per_slot=14,
            n_prbs_total=24,
            pathloss_db=100.0,
            shadowing_db=0.0,
            channel_profile="tdl_c",
            codebook_id=0,
            beam_profile_id=0,
        )

        low_mcs_input = PhyUeInput(
            ue_id="ue_low",
            queue_bytes=1000,
            sinr_db=12.0,
            radio_quality_db=12.0,
            bler=0.1,
            serving_beam_id=0,
            is_scheduled=True,
            slice_id="embb",
            traffic_type="video",
            latency_budget_ms=100.0,
            priority=1,
            allocated_prbs=12,
            allocated_symbols=6,
            is_retx=False,
            requested_bytes=1000,
            grant_bytes_budget=60,
            target_mcs=3,
            mcs_cap=3,
            channel_quality_confidence=0.42,
            link_adaptation_reason="low_confidence_guarded",
            num_tx_ant=4,
            num_rx_ant=2,
            rank_indicator=1,
            layer_count=1,
            precoder_id=0,
            combiner_id=0,
            beamforming_gain_db=0.0,
        )
        high_mcs_input = PhyUeInput(
            ue_id="ue_high",
            queue_bytes=1000,
            sinr_db=12.0,
            radio_quality_db=12.0,
            bler=0.1,
            serving_beam_id=0,
            is_scheduled=True,
            slice_id="embb",
            traffic_type="video",
            latency_budget_ms=100.0,
            priority=1,
            allocated_prbs=12,
            allocated_symbols=6,
            is_retx=False,
            requested_bytes=1000,
            grant_bytes_budget=180,
            target_mcs=9,
            mcs_cap=9,
            channel_quality_confidence=0.93,
            link_adaptation_reason="high_confidence_high_se",
            num_tx_ant=4,
            num_rx_ant=2,
            rank_indicator=1,
            layer_count=1,
            precoder_id=0,
            combiner_id=0,
            beamforming_gain_db=0.0,
        )

        phy_input = PhyInput(
            slot_id=1,
            sim_time_ms=1.0,
            cell_id="cell_0",
            tx_power_dbm=30.0,
            current_codebook_id=0,
            current_beam_profile_id=0,
            scheduled_ue_ids=["ue_low", "ue_high"],
            ue_inputs=[low_mcs_input, high_mcs_input],
            runtime_context=runtime_context,
        )

        result = adapter.run_slot(phy_input)
        self.assertEqual(len(result.ue_results), 2)

        low_result = next(item for item in result.ue_results if item.ue_id == "ue_low")
        high_result = next(item for item in result.ue_results if item.ue_id == "ue_high")

        self.assertEqual(low_result.mcs_index, 3)
        self.assertEqual(high_result.mcs_index, 9)
        self.assertAlmostEqual(low_result.channel_quality_confidence or 0.0, 0.42, places=2)
        self.assertEqual(low_result.link_adaptation_reason, "low_confidence_guarded")
        self.assertAlmostEqual(high_result.channel_quality_confidence or 0.0, 0.93, places=2)
        self.assertEqual(high_result.link_adaptation_reason, "high_confidence_high_se")
        self.assertLessEqual(low_result.scheduled_bytes, 60)
        self.assertLessEqual(high_result.scheduled_bytes, 180)
        self.assertGreater(high_result.scheduled_bytes, low_result.scheduled_bytes)

    def test_custom_phy_applies_tbs_method_and_dmrs_overhead(self) -> None:
        adapter = CustomPhyAdapter(
            codebook_map={0: {"base_gain_db": 0.0, "beams": {0: 0.0}}},
            beam_profile_map={0: {"profile_gain_db": 0.0}},
        )

        runtime_context = PhyRuntimeContext(
            slot_id=2,
            sim_time_ms=2.0,
            slot_duration_ms=1.0,
            cell_id="cell_0",
            scheduler_mode="round_robin",
            tx_power_dbm=30.0,
            current_codebook_id=0,
            current_beam_profile_id=0,
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=20e6,
            subcarrier_spacing_hz=30e3,
            cyclic_prefix_type="normal",
            num_ofdm_symbols_per_slot=14,
            n_prbs_total=24,
            pathloss_db=100.0,
            shadowing_db=0.0,
            channel_profile="tdl_c",
            codebook_id=0,
            beam_profile_id=0,
        )

        baseline_input = PhyUeInput(
            ue_id="ue_baseline",
            queue_bytes=2000,
            sinr_db=14.0,
            radio_quality_db=14.0,
            bler=0.1,
            serving_beam_id=0,
            is_scheduled=True,
            slice_id="embb",
            traffic_type="video",
            latency_budget_ms=100.0,
            priority=1,
            allocated_prbs=10,
            allocated_symbols=8,
            requested_bytes=2000,
            grant_bytes_budget=2000,
            target_mcs=9,
            mcs_cap=9,
            mcs_table_id="qam256",
            tbs_selection_method="resource_elements",
            dmrs_pattern_id="dmrs_type1_default",
            num_tx_ant=4,
            num_rx_ant=2,
            rank_indicator=2,
            layer_count=2,
            precoder_id=0,
            combiner_id=0,
        )
        guarded_input = PhyUeInput(
            ue_id="ue_guarded",
            queue_bytes=2000,
            sinr_db=14.0,
            radio_quality_db=14.0,
            bler=0.1,
            serving_beam_id=0,
            is_scheduled=True,
            slice_id="embb",
            traffic_type="video",
            latency_budget_ms=20.0,
            priority=1,
            allocated_prbs=10,
            allocated_symbols=8,
            requested_bytes=2000,
            grant_bytes_budget=2000,
            target_mcs=9,
            mcs_cap=9,
            mcs_table_id="qam256",
            tbs_selection_method="latency_guarded",
            dmrs_pattern_id="dmrs_type1_dense",
            num_tx_ant=4,
            num_rx_ant=2,
            rank_indicator=2,
            layer_count=2,
            precoder_id=0,
            combiner_id=0,
        )

        result = adapter.run_slot(
            PhyInput(
                slot_id=2,
                sim_time_ms=2.0,
                cell_id="cell_0",
                tx_power_dbm=30.0,
                current_codebook_id=0,
                current_beam_profile_id=0,
                scheduled_ue_ids=["ue_baseline", "ue_guarded"],
                ue_inputs=[baseline_input, guarded_input],
                runtime_context=runtime_context,
            )
        )

        baseline_result = next(item for item in result.ue_results if item.ue_id == "ue_baseline")
        guarded_result = next(item for item in result.ue_results if item.ue_id == "ue_guarded")

        self.assertEqual(baseline_result.mcs_table_id, "qam256")
        self.assertEqual(guarded_result.tbs_selection_method, "latency_guarded")
        self.assertEqual(guarded_result.dmrs_pattern_id, "dmrs_type1_dense")
        self.assertLess(guarded_result.scheduled_bytes, baseline_result.scheduled_bytes)


if __name__ == "__main__":
    unittest.main()

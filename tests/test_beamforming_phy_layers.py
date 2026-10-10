import unittest

from models.phy_input import PhyInput, PhyUeInput
from models.phy_runtime_context import PhyRuntimeContext
from phy.custom_phy_adapter import CustomPhyAdapter


class TestBeamformingPhyLayers(unittest.TestCase):
    def test_custom_phy_scales_tb_budget_with_layer_count(self) -> None:
        adapter = CustomPhyAdapter(
            codebook_map={0: {"base_gain_db": 0.0, "beams": {0: 0.0}}},
            beam_profile_map={0: {"profile_gain_db": 0.0}},
        )

        runtime_context = PhyRuntimeContext(
            slot_id=1,
            sim_time_ms=1.0,
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
            doppler_hz=10.0,
        )

        one_layer = PhyUeInput(
            ue_id="ue_l1",
            queue_bytes=2000,
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
            allocated_symbols=10,
            requested_bytes=2000,
            grant_bytes_budget=2000,
            target_mcs=6,
            mcs_cap=6,
            reported_ri=1,
            reported_pmi=0,
            num_tx_ant=8,
            num_rx_ant=2,
            rank_indicator=1,
            layer_count=1,
            precoder_id=0,
            combiner_id=0,
            beamforming_gain_db=3.0,
        )
        two_layers = PhyUeInput(
            ue_id="ue_l2",
            queue_bytes=2000,
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
            allocated_symbols=10,
            requested_bytes=2000,
            grant_bytes_budget=2000,
            target_mcs=6,
            mcs_cap=6,
            reported_ri=2,
            reported_pmi=0,
            num_tx_ant=8,
            num_rx_ant=2,
            rank_indicator=2,
            layer_count=2,
            precoder_id=0,
            combiner_id=1,
            beamforming_gain_db=3.0,
        )

        phy_input = PhyInput(
            slot_id=1,
            sim_time_ms=1.0,
            cell_id="cell_0",
            tx_power_dbm=30.0,
            current_codebook_id=0,
            current_beam_profile_id=0,
            scheduled_ue_ids=["ue_l1", "ue_l2"],
            ue_inputs=[one_layer, two_layers],
            runtime_context=runtime_context,
        )

        result = adapter.run_slot(phy_input)
        one_layer_result = next(item for item in result.ue_results if item.ue_id == "ue_l1")
        two_layer_result = next(item for item in result.ue_results if item.ue_id == "ue_l2")

        self.assertEqual(one_layer_result.num_layers, 1)
        self.assertEqual(two_layer_result.num_layers, 2)
        self.assertEqual(two_layer_result.channel_rank_limit, 2)
        self.assertGreater(two_layer_result.post_eq_sinr_db, one_layer_result.post_eq_sinr_db)
        self.assertGreater(
            two_layer_result.transport_block_size_bytes,
            one_layer_result.transport_block_size_bytes,
        )


if __name__ == "__main__":
    unittest.main()

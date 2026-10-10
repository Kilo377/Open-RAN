import unittest

from models.phy_input import PhyInput, PhyUeInput
from models.phy_runtime_context import PhyRuntimeContext
from phy.custom_phy_adapter import CustomPhyAdapter
from phy.mimo_link_model import SingleUserMimoModel
from phy.sionna_engine import SionnaEngine


class TestSingleUserMimoEffects(unittest.TestCase):
    def test_mimo_model_penalizes_precoder_mismatch(self) -> None:
        model = SingleUserMimoModel()

        matched = model.evaluate(
            pre_eq_sinr_db=14.0,
            num_tx_ant=4,
            num_rx_ant=2,
            layer_count=2,
            rank_indicator=2,
            precoder_id=1,
            combiner_id=0,
            reported_ri=2,
            reported_pmi=1,
            delay_spread_ns=200.0,
            doppler_hz=10.0,
            channel_model_type="tdl_d",
        )
        mismatched = model.evaluate(
            pre_eq_sinr_db=14.0,
            num_tx_ant=4,
            num_rx_ant=2,
            layer_count=2,
            rank_indicator=2,
            precoder_id=3,
            combiner_id=0,
            reported_ri=2,
            reported_pmi=0,
            delay_spread_ns=200.0,
            doppler_hz=10.0,
            channel_model_type="tdl_d",
        )

        self.assertGreater(matched.precoder_match_score, mismatched.precoder_match_score)
        self.assertGreater(matched.post_eq_sinr_db, mismatched.post_eq_sinr_db)

    def test_custom_phy_reflects_precoder_mismatch_in_bler_and_post_eq_sinr(self) -> None:
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
            channel_profile="tdl_d",
            codebook_id=0,
            beam_profile_id=0,
            doppler_hz=10.0,
        )

        matched_input = PhyUeInput(
            ue_id="ue_match",
            queue_bytes=1500,
            sinr_db=12.0,
            radio_quality_db=9.0,
            bler=0.1,
            serving_beam_id=0,
            is_scheduled=True,
            slice_id="embb",
            traffic_type="video",
            latency_budget_ms=100.0,
            priority=1,
            allocated_prbs=12,
            allocated_symbols=10,
            requested_bytes=1500,
            grant_bytes_budget=1500,
            target_mcs=8,
            mcs_cap=8,
            reported_ri=2,
            reported_pmi=1,
            num_tx_ant=4,
            num_rx_ant=2,
            rank_indicator=2,
            layer_count=2,
            precoder_id=1,
            combiner_id=0,
            beamforming_gain_db=2.0,
            link_channel_model_type="tdl_d",
            link_delay_spread_ns=200.0,
            link_doppler_hz=10.0,
        )
        mismatched_input = PhyUeInput(
            ue_id="ue_mismatch",
            queue_bytes=1500,
            sinr_db=12.0,
            radio_quality_db=9.0,
            bler=0.1,
            serving_beam_id=0,
            is_scheduled=True,
            slice_id="embb",
            traffic_type="video",
            latency_budget_ms=100.0,
            priority=1,
            allocated_prbs=12,
            allocated_symbols=10,
            requested_bytes=1500,
            grant_bytes_budget=1500,
            target_mcs=8,
            mcs_cap=8,
            reported_ri=2,
            reported_pmi=0,
            num_tx_ant=4,
            num_rx_ant=2,
            rank_indicator=2,
            layer_count=2,
            precoder_id=3,
            combiner_id=0,
            beamforming_gain_db=2.0,
            link_channel_model_type="tdl_d",
            link_delay_spread_ns=200.0,
            link_doppler_hz=10.0,
        )

        result = adapter.run_slot(
            PhyInput(
                slot_id=1,
                sim_time_ms=1.0,
                cell_id="cell_0",
                tx_power_dbm=30.0,
                current_codebook_id=0,
                current_beam_profile_id=0,
                scheduled_ue_ids=["ue_match", "ue_mismatch"],
                ue_inputs=[matched_input, mismatched_input],
                runtime_context=runtime_context,
            )
        )

        matched_result = next(item for item in result.ue_results if item.ue_id == "ue_match")
        mismatched_result = next(item for item in result.ue_results if item.ue_id == "ue_mismatch")

        self.assertGreater(matched_result.precoder_match_score, mismatched_result.precoder_match_score)
        self.assertGreater(matched_result.post_eq_sinr_db, mismatched_result.post_eq_sinr_db)
        self.assertLess(matched_result.bler, mismatched_result.bler)

    def test_sionna_engine_mimo_evaluation_prefers_supported_rank(self) -> None:
        engine = SionnaEngine({"bandwidth_hz": 20e6})

        rank_one = engine._evaluate_single_user_mimo(
            {
                "num_tx_ant": 2,
                "num_rx_ant": 2,
                "layer_count": 1,
                "rank_indicator": 1,
                "precoder_id": 0,
                "combiner_id": 0,
                "reported_ri": 1,
                "reported_pmi": 0,
                "delay_spread_ns": 100.0,
                "doppler_hz": 5.0,
                "channel_profile": "tdl_d",
            },
            pre_eq_sinr_db=14.0,
        )
        rank_two = engine._evaluate_single_user_mimo(
            {
                "num_tx_ant": 2,
                "num_rx_ant": 2,
                "layer_count": 2,
                "rank_indicator": 2,
                "precoder_id": 0,
                "combiner_id": 0,
                "reported_ri": 2,
                "reported_pmi": 0,
                "delay_spread_ns": 100.0,
                "doppler_hz": 5.0,
                "channel_profile": "tdl_d",
            },
            pre_eq_sinr_db=14.0,
        )

        self.assertEqual(rank_two["channel_rank_limit"], 2)
        self.assertGreater(rank_two["post_eq_sinr_db"], rank_one["post_eq_sinr_db"])


if __name__ == "__main__":
    unittest.main()

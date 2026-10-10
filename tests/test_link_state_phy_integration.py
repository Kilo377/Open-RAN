import unittest
from types import SimpleNamespace

from models.phy_input import PhyInput, PhyUeInput
from models.phy_runtime_context import PhyRuntimeContext
from phy.custom_phy_adapter import CustomPhyAdapter


class TestLinkStatePhyIntegration(unittest.TestCase):
    def test_custom_phy_uses_per_ue_link_pathloss(self) -> None:
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
            pathloss_db=120.0,
            shadowing_db=6.0,
            channel_profile="tdl_c",
            codebook_id=0,
            beam_profile_id=0,
            doppler_hz=30.0,
        )

        strong_link = PhyUeInput(
            ue_id="ue_strong",
            queue_bytes=500,
            sinr_db=10.0,
            radio_quality_db=12.0,
            bler=0.1,
            serving_beam_id=0,
            is_scheduled=True,
            slice_id="embb",
            traffic_type="video",
            latency_budget_ms=100.0,
            priority=1,
            allocated_prbs=8,
            allocated_symbols=6,
            requested_bytes=500,
            grant_bytes_budget=200,
            target_mcs=5,
            mcs_cap=5,
            num_tx_ant=4,
            num_rx_ant=2,
            rank_indicator=1,
            layer_count=1,
            precoder_id=0,
            combiner_id=0,
            beamforming_gain_db=0.0,
            link_pathloss_db=100.0,
            link_shadowing_db=0.0,
            link_doppler_hz=10.0,
            link_fast_fading_seed=11,
        )
        weak_link = PhyUeInput(
            ue_id="ue_weak",
            queue_bytes=500,
            sinr_db=10.0,
            radio_quality_db=12.0,
            bler=0.1,
            serving_beam_id=0,
            is_scheduled=True,
            slice_id="embb",
            traffic_type="video",
            latency_budget_ms=100.0,
            priority=1,
            allocated_prbs=8,
            allocated_symbols=6,
            requested_bytes=500,
            grant_bytes_budget=200,
            target_mcs=5,
            mcs_cap=5,
            num_tx_ant=4,
            num_rx_ant=2,
            rank_indicator=1,
            layer_count=1,
            precoder_id=0,
            combiner_id=0,
            beamforming_gain_db=0.0,
            link_pathloss_db=130.0,
            link_shadowing_db=8.0,
            link_doppler_hz=10.0,
            link_fast_fading_seed=11,
        )

        phy_input = PhyInput(
            slot_id=1,
            sim_time_ms=1.0,
            cell_id="cell_0",
            tx_power_dbm=30.0,
            current_codebook_id=0,
            current_beam_profile_id=0,
            scheduled_ue_ids=["ue_strong", "ue_weak"],
            ue_inputs=[strong_link, weak_link],
            runtime_context=runtime_context,
        )

        result = adapter.run_slot(phy_input)
        strong_result = next(item for item in result.ue_results if item.ue_id == "ue_strong")
        weak_result = next(item for item in result.ue_results if item.ue_id == "ue_weak")

        self.assertGreater(strong_result.sinr_db, weak_result.sinr_db)


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from typing import Any, cast

from core.simulator import Simulator
from models.phy_input import PhyInput, PhyUeInput
from models.phy_runtime_context import PhyRuntimeContext
from models.scheduling import UeGrant
from phy.custom_phy_adapter import CustomPhyAdapter
from state.resource_grid_state import ResourceGridState


class TestPhyResourceGridSliceMapping(unittest.TestCase):
    def test_simulator_expands_grant_to_explicit_resource_slice(self) -> None:
        simulator = cast(Any, Simulator.__new__(Simulator))
        simulator.state = SimpleNamespace(
            resource_grid=ResourceGridState(
                n_prbs_total=16,
                n_subcarriers_per_prb=12,
                n_ofdm_symbols_per_slot=14,
            )
        )

        grant = UeGrant(
            ue_id="ue_0",
            allocated_prbs=3,
            allocated_symbols=4,
            start_prb=2,
            start_symbol=5,
            resource_block_indices=[2, 3, 4],
            ofdm_symbol_indices=[5, 6, 7, 8],
        )

        resource_grid_slice = simulator._build_phy_resource_grid_slice(grant=grant)

        self.assertEqual(resource_grid_slice.resource_block_indices, [2, 3, 4])
        self.assertEqual(resource_grid_slice.allocated_symbol_indices, [5, 6, 7, 8])
        self.assertEqual(resource_grid_slice.allocated_subcarrier_indices[0], 24)
        self.assertEqual(resource_grid_slice.allocated_subcarrier_indices[-1], 59)
        self.assertEqual(resource_grid_slice.resource_elements, 144)

    def test_custom_phy_prefers_explicit_resource_slice_over_raw_counts(self) -> None:
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

        compact_slice_input = PhyUeInput(
            ue_id="ue_compact",
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
            allocated_prbs=10,
            allocated_symbols=10,
            start_prb=0,
            start_symbol=0,
            resource_block_indices=[0, 1],
            allocated_symbol_indices=[0, 1, 2],
            allocated_subcarrier_indices=list(range(24)),
            requested_bytes=1000,
            grant_bytes_budget=1000,
            target_mcs=6,
            mcs_cap=6,
            num_tx_ant=4,
            num_rx_ant=2,
            rank_indicator=1,
            layer_count=1,
            precoder_id=0,
            combiner_id=0,
            beamforming_gain_db=0.0,
        )

        result = adapter.run_slot(
            phy_input=PhyInput(
                slot_id=1,
                sim_time_ms=1.0,
                cell_id="cell_0",
                tx_power_dbm=30.0,
                current_codebook_id=0,
                current_beam_profile_id=0,
                scheduled_ue_ids=["ue_compact"],
                ue_inputs=[compact_slice_input],
                runtime_context=runtime_context,
            )
        )

        ue_result = result.ue_results[0]
        self.assertEqual(ue_result.resource_grid_re_count, 72)
        self.assertLessEqual(ue_result.scheduled_bytes, 72)


if __name__ == "__main__":
    unittest.main()

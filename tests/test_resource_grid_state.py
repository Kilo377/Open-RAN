import unittest
from types import SimpleNamespace
from typing import Any, cast

from core.metrics import build_observation
from core.state_updater import StateUpdater
from models.phy_result import PhyResult, PhyUeResult
from models.scheduling import SchedulingDecision, UeGrant


class TestResourceGridState(unittest.TestCase):
    def test_apply_phy_result_populates_resource_grid_state(self) -> None:
        state = SimpleNamespace(
            slot_id=5,
            sim_time_ms=5.0,
            slot_duration_ms=1.0,
            ues=[
                SimpleNamespace(
                    ue_id="ue_0",
                    queue_bytes=400,
                    throughput_mbps=0.0,
                    sinr_db=8.0,
                    radio_quality_db=7.5,
                    bler=0.1,
                    serving_beam_id=0,
                    slice_id="embb",
                    traffic_type="video",
                    latency_budget_ms=50.0,
                    priority=1,
                    harq=SimpleNamespace(
                        process_id=0,
                        pending_retx_bytes=0,
                        retransmission_count=0,
                        last_tx_bytes=0,
                        last_tx_was_retx=False,
                        last_delivered_bytes=0,
                        is_waiting_ack=False,
                    ),
                    rank=2,
                    pmi=1,
                    ri=2,
                )
            ],
            cell=SimpleNamespace(
                cell_id="cell_0",
                tx_power_dbm=30.0,
                current_codebook_id=1,
                current_beam_profile_id=0,
            ),
            scheduler=SimpleNamespace(
                scheduler_mode="round_robin",
                packing_strategy="frequency_first",
                scheduled_ue_ids=["ue_0"],
                grants=[
                    UeGrant(
                        ue_id="ue_0",
                        allocated_prbs=6,
                        allocated_symbols=4,
                        start_prb=2,
                        start_symbol=1,
                        resource_block_indices=[2, 3, 4, 5, 6, 7],
                        ofdm_symbol_indices=[1, 2, 3, 4],
                        grant_bytes_budget=180,
                    )
                ],
                last_decision=SchedulingDecision(
                    slot_id=5,
                    scheduler_mode="round_robin",
                    total_prbs=12,
                    total_symbols=14,
                    grants=[],
                    unallocated_prbs=6,
                    unallocated_symbols=10,
                ),
            ),
            beamforming={"ue_0": SimpleNamespace(layer_count=2)},
        )
        phy_result = PhyResult(
            slot_id=5,
            ue_results=[
                PhyUeResult(
                    ue_id="ue_0",
                    is_scheduled=True,
                    allocated_prbs=6,
                    allocated_symbols=4,
                    start_prb=2,
                    start_symbol=1,
                    scheduled_bytes=180,
                    delivered_bytes=180,
                    ack=True,
                    num_layers=2,
                    resource_grid_re_count=288,
                    resource_grid_overlap_fraction=0.0,
                    interference_penalty_db=0.15,
                )
            ],
        )

        updater = StateUpdater()
        updater.apply_phy_result(state, phy_result)

        self.assertEqual(state.resource_grid.total_prbs, 12)
        self.assertEqual(state.resource_grid.total_symbols, 14)
        self.assertEqual(state.resource_grid.allocated_resource_elements, 288)
        self.assertAlmostEqual(state.resource_grid.utilization, 288.0 / 2016.0)
        self.assertEqual(len(state.resource_grid.allocations), 1)
        self.assertEqual(state.resource_grid.allocations[0].start_prb, 2)
        self.assertEqual(state.resource_grid.allocations[0].start_symbol, 1)
        self.assertEqual(state.resource_grid.allocations[0].resource_block_indices, [2, 3, 4, 5, 6, 7])
        self.assertEqual(state.resource_grid.allocations[0].allocated_symbol_indices, [1, 2, 3, 4])
        self.assertEqual(state.resource_grid.ue_resource_slices["ue_0"].start_prb, 2)
        self.assertEqual(state.resource_grid.prb_symbol_occupancy_map[2][1], 1)
        self.assertEqual(state.resource_grid.prb_symbol_occupancy_map[7][4], 1)

        observation = build_observation(cast(Any, state))
        self.assertEqual(observation.resource_grid_allocated_re, 288)
        self.assertAlmostEqual(observation.resource_grid_utilization or 0.0, 288.0 / 2016.0)
        self.assertEqual(observation.ue_observations[0].grant_start_prb, 2)
        self.assertEqual(observation.ue_observations[0].grant_resource_block_indices, [2, 3, 4, 5, 6, 7])
        self.assertEqual(observation.ue_observations[0].grant_allocated_symbol_indices, [1, 2, 3, 4])
        self.assertEqual(observation.ue_observations[0].resource_grid_re_count, 288)
        self.assertEqual(observation.scheduler_packing_strategy, "frequency_first")

    def test_resource_grid_tracks_adjacent_interference_groups(self) -> None:
        state = SimpleNamespace(
            slot_id=8,
            sim_time_ms=8.0,
            slot_duration_ms=1.0,
            ues=[
                SimpleNamespace(
                    ue_id="ue_0",
                    queue_bytes=300,
                    throughput_mbps=0.0,
                    sinr_db=8.0,
                    radio_quality_db=7.5,
                    bler=0.1,
                    serving_beam_id=0,
                    slice_id="embb",
                    traffic_type="video",
                    latency_budget_ms=50.0,
                    priority=1,
                    harq=SimpleNamespace(
                        process_id=0,
                        pending_retx_bytes=0,
                        retransmission_count=0,
                        last_tx_bytes=0,
                        last_tx_was_retx=False,
                        last_delivered_bytes=0,
                        is_waiting_ack=False,
                    ),
                    rank=1,
                    pmi=0,
                    ri=1,
                ),
                SimpleNamespace(
                    ue_id="ue_1",
                    queue_bytes=300,
                    throughput_mbps=0.0,
                    sinr_db=8.0,
                    radio_quality_db=7.5,
                    bler=0.1,
                    serving_beam_id=1,
                    slice_id="embb",
                    traffic_type="video",
                    latency_budget_ms=50.0,
                    priority=1,
                    harq=SimpleNamespace(
                        process_id=1,
                        pending_retx_bytes=0,
                        retransmission_count=0,
                        last_tx_bytes=0,
                        last_tx_was_retx=False,
                        last_delivered_bytes=0,
                        is_waiting_ack=False,
                    ),
                    rank=1,
                    pmi=0,
                    ri=1,
                ),
            ],
            cell=SimpleNamespace(
                cell_id="cell_0",
                tx_power_dbm=30.0,
                current_codebook_id=1,
                current_beam_profile_id=0,
            ),
            scheduler=SimpleNamespace(
                scheduler_mode="round_robin",
                packing_strategy="frequency_first",
                scheduled_ue_ids=["ue_0", "ue_1"],
                grants=[
                    UeGrant(
                        ue_id="ue_0",
                        allocated_prbs=4,
                        allocated_symbols=6,
                        start_prb=0,
                        start_symbol=0,
                        resource_block_indices=[0, 1, 2, 3],
                        ofdm_symbol_indices=[0, 1, 2, 3, 4, 5],
                        grant_bytes_budget=100,
                        spatial_group_id=0,
                        spatial_group_size=2,
                        spatial_compatibility_score=0.8,
                    ),
                    UeGrant(
                        ue_id="ue_1",
                        allocated_prbs=4,
                        allocated_symbols=6,
                        start_prb=4,
                        start_symbol=0,
                        resource_block_indices=[4, 5, 6, 7],
                        ofdm_symbol_indices=[0, 1, 2, 3, 4, 5],
                        grant_bytes_budget=100,
                        spatial_group_id=0,
                        spatial_group_size=2,
                        spatial_compatibility_score=0.8,
                    ),
                ],
                last_decision=SchedulingDecision(
                    slot_id=8,
                    scheduler_mode="round_robin",
                    total_prbs=12,
                    total_symbols=14,
                    grants=[],
                    unallocated_prbs=4,
                    unallocated_symbols=2,
                ),
            ),
            beamforming={
                "ue_0": SimpleNamespace(layer_count=1),
                "ue_1": SimpleNamespace(layer_count=1),
            },
        )
        phy_result = PhyResult(
            slot_id=8,
            ue_results=[
                PhyUeResult(
                    ue_id="ue_0",
                    is_scheduled=True,
                    allocated_prbs=4,
                    allocated_symbols=6,
                    start_prb=0,
                    start_symbol=0,
                    scheduled_bytes=100,
                    delivered_bytes=80,
                    ack=True,
                    num_layers=1,
                    resource_grid_re_count=288,
                    resource_grid_overlap_fraction=0.0,
                    interference_penalty_db=0.4,
                    spatial_group_id=0,
                    spatial_group_size=2,
                    spatial_compatibility_score=0.8,
                    spatial_multiplexing_gain_db=1.2,
                    spatial_mode="mu_mimo",
                ),
                PhyUeResult(
                    ue_id="ue_1",
                    is_scheduled=True,
                    allocated_prbs=4,
                    allocated_symbols=6,
                    start_prb=4,
                    start_symbol=0,
                    scheduled_bytes=100,
                    delivered_bytes=75,
                    ack=True,
                    num_layers=1,
                    resource_grid_re_count=288,
                    resource_grid_overlap_fraction=0.0,
                    interference_penalty_db=0.35,
                    spatial_group_id=0,
                    spatial_group_size=2,
                    spatial_compatibility_score=0.8,
                    spatial_multiplexing_gain_db=1.2,
                    spatial_mode="mu_mimo",
                ),
            ],
        )

        updater = StateUpdater()
        updater.apply_phy_result(state, phy_result)

        self.assertEqual(state.resource_grid.adjacent_interference_pair_count, 1)
        self.assertEqual(len(state.resource_grid.interference_groups), 1)
        self.assertEqual(len(state.resource_grid.spatial_groups), 1)
        self.assertEqual(state.resource_grid.mu_mimo_group_count, 1)
        self.assertEqual(state.resource_grid.mu_mimo_ue_count, 2)
        self.assertEqual(sorted(state.resource_grid.interference_groups[0]), ["ue_0", "ue_1"])
        self.assertEqual(state.resource_grid.allocations[0].interference_group_id, 0)
        self.assertEqual(state.resource_grid.allocations[1].interference_group_id, 0)
        self.assertEqual(state.resource_grid.allocations[0].spatial_group_id, 0)
        self.assertEqual(state.resource_grid.allocations[1].spatial_group_id, 0)

        observation = build_observation(cast(Any, state))
        self.assertEqual(observation.resource_grid_adjacent_interference_pair_count, 1)
        self.assertEqual(observation.resource_grid_interference_group_count, 1)
        self.assertEqual(observation.resource_grid_mu_mimo_group_count, 1)
        self.assertEqual(observation.resource_grid_mu_mimo_ue_count, 2)
        self.assertEqual(observation.ue_observations[0].interference_group_id, 0)
        self.assertEqual(observation.ue_observations[1].interference_group_id, 0)
        self.assertEqual(observation.ue_observations[0].spatial_group_id, 0)
        self.assertEqual(observation.ue_observations[0].spatial_group_size, 2)
        self.assertEqual(observation.ue_observations[0].spatial_mode, "mu_mimo")


if __name__ == "__main__":
    unittest.main()

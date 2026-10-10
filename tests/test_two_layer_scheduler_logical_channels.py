import unittest
from types import SimpleNamespace

from core.state_updater import StateUpdater
from models.phy_result import PhyResult, PhyUeResult
from models.scheduling import LogicalChannelGrant, UeGrant
from ran.scheduler import SimpleScheduler
from state.harq_state import HarqState
from state.logical_channel_group_state import LogicalChannelGroupState
from state.logical_channel_state import LogicalChannelState
from state.resource_grid_state import ResourceGridState
from traffic.traffic_engine import TrafficEngine


class TestTwoLayerSchedulerLogicalChannels(unittest.TestCase):
    def test_scheduler_allocates_grant_bytes_across_logical_channels(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=1,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )

        ue = SimpleNamespace(
            ue_id="ue_0",
            queue_bytes=650,
            latency_budget_ms=40.0,
            priority=2,
            sinr_db=10.0,
            slice_id="embb",
            traffic_type="mixed",
            logical_channels=[
                LogicalChannelState(
                    logical_channel_id="lc_ctrl",
                    drb_id=2,
                    lcg_id=0,
                    queue_bytes=150,
                    priority=4,
                    packet_delay_budget_ms=10.0,
                    slice_id="urllc",
                    traffic_type="control",
                ),
                LogicalChannelState(
                    logical_channel_id="lc_video",
                    drb_id=1,
                    lcg_id=1,
                    queue_bytes=500,
                    priority=1,
                    packet_delay_budget_ms=80.0,
                    slice_id="embb",
                    traffic_type="video",
                ),
            ],
            harq=SimpleNamespace(
                pending_retx_bytes=0,
                pending_retx_logical_channels=[],
                process_id=0,
            ),
        )

        state = SimpleNamespace(
            slot_id=4,
            ues=[ue],
            control=SimpleNamespace(active_policy_constraints=[]),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                scheduled_ue_ids=[],
                grants=[],
                last_decision=None,
                rr_pointer=0,
                priority_boost_by_ue={},
                last_served_slot_by_ue={},
            ),
        )

        scheduler.run(state)

        self.assertEqual(len(state.scheduler.grants), 1)
        grant = state.scheduler.grants[0]
        self.assertGreater(grant.grant_bytes_budget, 0)
        self.assertGreaterEqual(len(grant.logical_channel_allocations), 1)
        self.assertEqual(grant.logical_channel_allocations[0].logical_channel_id, "lc_ctrl")
        self.assertEqual(
            sum(allocation.scheduled_bytes for allocation in grant.logical_channel_allocations),
            grant.grant_bytes_budget,
        )
        self.assertEqual(ue.scheduled_logical_channel_ids[0], "lc_ctrl")
        self.assertIn(2, ue.scheduled_drb_ids)
        self.assertIn(0, ue.scheduled_lcg_ids)

    def test_scheduler_serves_pbr_credit_before_best_effort(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=1,
            min_prbs_per_grant=4,
            min_symbols_per_grant=4,
        )

        lc_pbr = LogicalChannelState(
            logical_channel_id="lc_pbr",
            drb_id=1,
            queue_bytes=400,
            priority=1,
            packet_delay_budget_ms=80.0,
            slice_id="embb",
            traffic_type="video",
            prioritized_bit_rate_bps=800000,
            bucket_size_duration_ms=10.0,
        )
        lc_best_effort = LogicalChannelState(
            logical_channel_id="lc_be",
            drb_id=2,
            queue_bytes=400,
            priority=5,
            packet_delay_budget_ms=10.0,
            slice_id="urllc",
            traffic_type="control",
        )
        ue = SimpleNamespace(
            ue_id="ue_0",
            queue_bytes=800,
            latency_budget_ms=20.0,
            priority=2,
            sinr_db=9.0,
            slice_id="mixed",
            traffic_type="mixed",
            logical_channels=[lc_pbr, lc_best_effort],
            harq=SimpleNamespace(
                pending_retx_bytes=0,
                pending_retx_logical_channels=[],
                process_id=0,
            ),
        )
        state = SimpleNamespace(
            slot_id=1,
            slot_duration_ms=1.0,
            ues=[ue],
            control=SimpleNamespace(active_policy_constraints=[]),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                scheduled_ue_ids=[],
                grants=[],
                last_decision=None,
                rr_pointer=0,
                priority_boost_by_ue={},
                last_served_slot_by_ue={},
            ),
        )

        scheduler.run(state)

        grant = state.scheduler.grants[0]
        allocations_by_lc = {
            allocation.logical_channel_id: allocation
            for allocation in grant.logical_channel_allocations
        }
        expected_prioritized_bytes = min(100, int(getattr(grant, "grant_bytes_budget", 0)))
        remaining_best_effort_budget = max(
            0,
            int(getattr(grant, "grant_bytes_budget", 0)) - expected_prioritized_bytes,
        )
        self.assertEqual(allocations_by_lc["lc_pbr"].prioritized_bytes, expected_prioritized_bytes)
        self.assertEqual(allocations_by_lc["lc_pbr"].used_pbr_credit_bytes, expected_prioritized_bytes)
        if remaining_best_effort_budget > 0:
            self.assertEqual(allocations_by_lc["lc_be"].best_effort_bytes, remaining_best_effort_budget)
        else:
            self.assertNotIn("lc_be", allocations_by_lc)
        self.assertEqual(lc_pbr.pbr_credit_bytes, float(max(0, 100 - expected_prioritized_bytes)))

    def test_scheduler_uses_lcg_bsr_state_to_break_logical_channel_ties(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=1,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )

        ue = SimpleNamespace(
            ue_id="ue_0",
            queue_bytes=400,
            latency_budget_ms=40.0,
            priority=2,
            sinr_db=10.0,
            slice_id="embb",
            traffic_type="mixed",
            logical_channels=[
                LogicalChannelState(
                    logical_channel_id="lc_stable",
                    drb_id=1,
                    lcg_id=0,
                    queue_bytes=200,
                    priority=2,
                    packet_delay_budget_ms=50.0,
                    slice_id="embb",
                    traffic_type="video",
                ),
                LogicalChannelState(
                    logical_channel_id="lc_growth",
                    drb_id=2,
                    lcg_id=1,
                    queue_bytes=200,
                    priority=2,
                    packet_delay_budget_ms=50.0,
                    slice_id="embb",
                    traffic_type="video",
                ),
            ],
            logical_channel_groups=[
                LogicalChannelGroupState(
                    lcg_id=0,
                    logical_channel_ids=["lc_stable"],
                    drb_ids=[1],
                    queue_bytes=200,
                    bsr_reported_bytes=200,
                    bsr_pending_bytes=0,
                    bsr_age_slots=1,
                    last_bsr_report_slot=4,
                    bsr_triggered=False,
                ),
                LogicalChannelGroupState(
                    lcg_id=1,
                    logical_channel_ids=["lc_growth"],
                    drb_ids=[2],
                    queue_bytes=200,
                    bsr_reported_bytes=120,
                    bsr_pending_bytes=80,
                    bsr_age_slots=0,
                    last_bsr_report_slot=5,
                    last_bsr_trigger_slot=5,
                    bsr_triggered=True,
                    bsr_trigger_reason="buffer_growth",
                ),
            ],
            harq=SimpleNamespace(
                pending_retx_bytes=0,
                pending_retx_logical_channels=[],
                process_id=0,
            ),
        )

        state = SimpleNamespace(
            slot_id=5,
            ues=[ue],
            control=SimpleNamespace(active_policy_constraints=[]),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                scheduled_ue_ids=[],
                grants=[],
                last_decision=None,
                rr_pointer=0,
                priority_boost_by_ue={},
                last_served_slot_by_ue={},
            ),
        )

        scheduler.run(state)

        grant = state.scheduler.grants[0]
        self.assertGreaterEqual(len(grant.logical_channel_allocations), 1)
        self.assertEqual(grant.logical_channel_allocations[0].lcg_id, 1)
        self.assertEqual(grant.logical_channel_allocations[0].logical_channel_id, "lc_growth")
        self.assertIn(1, ue.scheduled_lcg_ids)

    def test_scheduler_builds_demand_objects_and_stage_trace(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=1,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )

        ue = SimpleNamespace(
            ue_id="ue_0",
            queue_bytes=240,
            latency_budget_ms=20.0,
            priority=3,
            sinr_db=10.0,
            slice_id="urllc",
            traffic_type="control",
            cqi=8,
            ri=1,
            pmi=0,
            logical_channels=[
                LogicalChannelState(
                    logical_channel_id="lc_ctrl",
                    drb_id=2,
                    lcg_id=0,
                    queue_bytes=240,
                    priority=3,
                    packet_delay_budget_ms=20.0,
                    slice_id="urllc",
                    traffic_type="control",
                )
            ],
            harq=SimpleNamespace(
                pending_retx_bytes=0,
                pending_retx_logical_channels=[],
                process_id=0,
                retransmission_count=0,
            ),
        )
        state = SimpleNamespace(
            slot_id=9,
            slot_duration_ms=1.0,
            ues=[ue],
            control=SimpleNamespace(active_policy_constraints=[]),
            measurement=SimpleNamespace(
                ue_measurements={
                    "ue_0": SimpleNamespace(
                        queue_bytes=220,
                        age_slots=1,
                        queue_age_slots=3,
                        last_measurement_update_slot=7,
                    )
                }
            ),
            links={
                "ue_0": SimpleNamespace(
                    cqi=9,
                    ri=1,
                    pmi=0,
                )
            },
            connections={},
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                scheduled_ue_ids=[],
                grants=[],
                last_decision=None,
                rr_pointer=0,
                priority_boost_by_ue={},
                last_served_slot_by_ue={},
            ),
        )

        scheduler.run(state)

        self.assertEqual(len(state.scheduler.last_demands), 1)
        demand = state.scheduler.last_demands[0]
        self.assertEqual(demand.ue_id, "ue_0")
        self.assertEqual(demand.true_backlog_bytes, 240)
        self.assertEqual(demand.backlog_bytes, 240)
        self.assertEqual(demand.observed_backlog_bytes, 220)
        self.assertEqual(demand.demand_bytes, 240)
        self.assertEqual(demand.hol_delay_ms, 3.0)
        self.assertEqual(demand.cqi_validity_age_slots, 2)
        self.assertEqual(demand.qos_class, "urllc")
        self.assertEqual(demand.five_qi, 80)
        self.assertGreater(demand.channel_quality_confidence, 0.5)
        self.assertEqual(demand.demand_reason, "delayed_measurement")
        self.assertEqual(len(demand.logical_channel_demands), 1)
        self.assertEqual(demand.logical_channel_demands[0].demand_bytes, 240)
        self.assertEqual(len(demand.logical_channel_group_demands), 1)
        self.assertEqual(demand.logical_channel_group_demands[0].demand_bytes, 240)
        self.assertEqual(state.scheduler.last_stage_trace.demand_visible_ue_ids, ["ue_0"])
        self.assertEqual(state.scheduler.last_stage_trace.feasible_candidate_ue_ids, ["ue_0"])
        self.assertEqual(state.scheduler.last_stage_trace.selected_new_tx_ue_ids, ["ue_0"])
        self.assertEqual(state.scheduler.last_stage_trace.granted_ue_ids, ["ue_0"])
        self.assertEqual(
            state.scheduler.last_stage_trace.selection_reasons_by_ue["ue_0"],
            "selected_for_latency_priority",
        )
        self.assertEqual(state.scheduler.grants[0].selection_reason, "selected_for_latency_priority")
        self.assertEqual(
            state.scheduler.grants[0].lc_multiplexing_summary["logical_channel_count"],
            1,
        )

    def test_scheduler_marks_new_tx_as_harq_preempted_when_retx_uses_slot_budget(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=1,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )

        ue_retx = SimpleNamespace(
            ue_id="ue_retx",
            queue_bytes=180,
            latency_budget_ms=20.0,
            priority=2,
            sinr_db=9.0,
            slice_id="embb",
            traffic_type="video",
            logical_channels=[
                LogicalChannelState(
                    logical_channel_id="lc_retx",
                    drb_id=1,
                    lcg_id=0,
                    queue_bytes=180,
                    priority=2,
                    packet_delay_budget_ms=50.0,
                    slice_id="embb",
                    traffic_type="video",
                )
            ],
            harq=SimpleNamespace(
                pending_retx_bytes=180,
                pending_retx_logical_channels=[],
                process_id=0,
                retransmission_count=1,
                ndi=1,
                redundancy_version=2,
                last_tx_bytes=180,
                last_delivered_bytes=0,
                is_waiting_ack=False,
                process_pool_size=4,
                next_new_tx_process_id=1,
                ndi_by_process_id={0: 1},
                processes=[],
            ),
        )
        ue_new = SimpleNamespace(
            ue_id="ue_new",
            queue_bytes=160,
            latency_budget_ms=30.0,
            priority=1,
            sinr_db=11.0,
            slice_id="embb",
            traffic_type="video",
            logical_channels=[
                LogicalChannelState(
                    logical_channel_id="lc_new",
                    drb_id=2,
                    lcg_id=1,
                    queue_bytes=160,
                    priority=1,
                    packet_delay_budget_ms=60.0,
                    slice_id="embb",
                    traffic_type="video",
                )
            ],
            harq=SimpleNamespace(
                pending_retx_bytes=0,
                pending_retx_logical_channels=[],
                process_id=1,
                retransmission_count=0,
                ndi=0,
                redundancy_version=0,
                last_tx_bytes=0,
                last_delivered_bytes=0,
                is_waiting_ack=False,
                process_pool_size=4,
                next_new_tx_process_id=1,
                ndi_by_process_id={},
                processes=[],
            ),
        )
        state = SimpleNamespace(
            slot_id=6,
            slot_duration_ms=1.0,
            ues=[ue_retx, ue_new],
            control=SimpleNamespace(active_policy_constraints=[]),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                scheduled_ue_ids=[],
                grants=[],
                last_decision=None,
                rr_pointer=0,
                priority_boost_by_ue={},
                last_served_slot_by_ue={},
            ),
        )

        scheduler.run(state)

        self.assertEqual(state.scheduler.scheduled_ue_ids, ["ue_retx"])
        self.assertEqual(
            state.scheduler.last_stage_trace.selection_reasons_by_ue["ue_retx"],
            "selected_for_harq_retransmission",
        )
        self.assertEqual(
            state.scheduler.last_stage_trace.rejection_reasons_by_ue["ue_new"],
            "new_tx_preempted_by_retransmissions",
        )
        self.assertEqual(
            state.scheduler.last_stage_trace.harq_preemption_causes_by_ue["ue_new"],
            "retx_reserved_slot_budget",
        )
        self.assertIn("harq_retx", state.scheduler.grants[0].constraint_hits)
        self.assertEqual(state.scheduler.grants[0].selection_reason, "selected_for_harq_retransmission")

    def test_traffic_engine_updates_logical_channel_queues_and_aggregate_queue(self) -> None:
        traffic_engine = TrafficEngine(
            ue_config={
                "ues": [
                    {
                        "ue_id": "ue_0",
                        "logical_channels": [
                            {
                                "logical_channel_id": "lc_0",
                                "arrival_model": "cbr",
                                "arrival_rate_bps": 8000,
                            },
                            {
                                "logical_channel_id": "lc_1",
                                "arrival_model": "cbr",
                                "arrival_rate_bps": 16000,
                            },
                        ],
                    }
                ]
            },
            slot_duration_ms=1.0,
            rng_seed=1,
        )

        ue = SimpleNamespace(
            ue_id="ue_0",
            queue_bytes=30,
            logical_channels=[
                LogicalChannelState(logical_channel_id="lc_0", queue_bytes=10),
                LogicalChannelState(logical_channel_id="lc_1", queue_bytes=20),
            ],
        )
        state = SimpleNamespace(slot_id=1, ues=[ue])

        arrivals_by_ue, arrivals_by_logical_channel = traffic_engine.apply_arrivals(state)

        self.assertEqual(arrivals_by_ue["ue_0"], 3)
        self.assertEqual(arrivals_by_logical_channel["ue_0:lc_0"], 1)
        self.assertEqual(arrivals_by_logical_channel["ue_0:lc_1"], 2)
        self.assertEqual(ue.logical_channels[0].queue_bytes, 11)
        self.assertEqual(ue.logical_channels[1].queue_bytes, 22)
        self.assertEqual(ue.queue_bytes, 33)

    def test_state_updater_writes_harq_and_queue_by_logical_channel(self) -> None:
        updater = StateUpdater()
        ue = SimpleNamespace(
            ue_id="ue_0",
            queue_bytes=300,
            throughput_mbps=0.0,
            sinr_db=8.0,
            radio_quality_db=8.0,
            bler=0.1,
            serving_beam_id=0,
            slice_id="embb",
            traffic_type="mixed",
            latency_budget_ms=40.0,
            priority=2,
            rank=1,
            pmi=0,
            ri=1,
            harq=HarqState(),
            logical_channels=[
                LogicalChannelState(
                    logical_channel_id="lc_ctrl",
                    drb_id=2,
                    lcg_id=0,
                    queue_bytes=100,
                    priority=4,
                    packet_delay_budget_ms=10.0,
                    slice_id="urllc",
                    traffic_type="control",
                ),
                LogicalChannelState(
                    logical_channel_id="lc_video",
                    drb_id=1,
                    lcg_id=1,
                    queue_bytes=200,
                    priority=1,
                    packet_delay_budget_ms=80.0,
                    slice_id="embb",
                    traffic_type="video",
                ),
            ],
        )
        grant_allocations = [
            LogicalChannelGrant(
                logical_channel_id="lc_ctrl",
                drb_id=2,
                scheduled_bytes=100,
                queue_bytes_before=100,
                priority=4,
                packet_delay_budget_ms=10.0,
                slice_id="urllc",
                traffic_type="control",
            ),
            LogicalChannelGrant(
                logical_channel_id="lc_video",
                drb_id=1,
                scheduled_bytes=80,
                queue_bytes_before=200,
                priority=1,
                packet_delay_budget_ms=80.0,
                slice_id="embb",
                traffic_type="video",
            ),
        ]

        state = SimpleNamespace(
            slot_id=5,
            slot_duration_ms=1.0,
            scheduler=SimpleNamespace(
                scheduled_ue_ids=["ue_0"],
                grants=[
                    UeGrant(
                        ue_id="ue_0",
                        allocated_prbs=4,
                        allocated_symbols=4,
                        is_retx=False,
                        requested_bytes=180,
                        grant_bytes_budget=180,
                        logical_channel_allocations=grant_allocations,
                    )
                ],
                last_decision=SimpleNamespace(total_prbs=4, total_symbols=4),
            ),
            ues=[ue],
            connections={},
            links={},
            beamforming={},
            resource_grid=ResourceGridState(),
        )

        nack_result = PhyResult(
            slot_id=5,
            ue_results=[
                PhyUeResult(
                    ue_id="ue_0",
                    is_scheduled=True,
                    allocated_prbs=4,
                    allocated_symbols=4,
                    is_retx=False,
                    scheduled_bytes=180,
                    delivered_bytes=0,
                    transport_block_size_bytes=180,
                    ack=False,
                    sinr_db=8.0,
                    post_eq_sinr_db=8.0,
                )
            ],
        )
        updater.apply_phy_result(state, nack_result)

        self.assertEqual(ue.queue_bytes, 300)
        self.assertEqual(ue.logical_channels[0].queue_bytes, 100)
        self.assertEqual(ue.logical_channels[1].queue_bytes, 200)
        self.assertEqual(ue.harq.pending_retx_bytes, 180)
        self.assertEqual(
            [entry.pending_bytes for entry in ue.harq.pending_retx_logical_channels],
            [100, 80],
        )
        self.assertEqual(
            [logical_channel_group.queue_bytes for logical_channel_group in ue.logical_channel_groups],
            [100, 200],
        )

        retx_allocations = [
            LogicalChannelGrant(
                logical_channel_id="lc_ctrl",
                drb_id=2,
                lcg_id=0,
                scheduled_bytes=100,
                queue_bytes_before=100,
                priority=4,
                packet_delay_budget_ms=10.0,
                slice_id="urllc",
                traffic_type="control",
                is_retx=True,
            ),
            LogicalChannelGrant(
                logical_channel_id="lc_video",
                drb_id=1,
                lcg_id=1,
                scheduled_bytes=80,
                queue_bytes_before=200,
                priority=1,
                packet_delay_budget_ms=80.0,
                slice_id="embb",
                traffic_type="video",
                is_retx=True,
            ),
        ]
        state.scheduler.grants = [
            UeGrant(
                ue_id="ue_0",
                allocated_prbs=4,
                allocated_symbols=4,
                is_retx=True,
                requested_bytes=180,
                grant_bytes_budget=180,
                logical_channel_allocations=retx_allocations,
            )
        ]

        ack_result = PhyResult(
            slot_id=6,
            ue_results=[
                PhyUeResult(
                    ue_id="ue_0",
                    is_scheduled=True,
                    allocated_prbs=4,
                    allocated_symbols=4,
                    is_retx=True,
                    scheduled_bytes=180,
                    delivered_bytes=180,
                    transport_block_size_bytes=180,
                    ack=True,
                    sinr_db=9.0,
                    post_eq_sinr_db=9.0,
                )
            ],
        )
        updater.apply_phy_result(state, ack_result)

        self.assertEqual(ue.logical_channels[0].queue_bytes, 0)
        self.assertEqual(ue.logical_channels[1].queue_bytes, 120)
        self.assertEqual(ue.queue_bytes, 120)
        self.assertEqual(ue.harq.pending_retx_bytes, 0)
        self.assertEqual(len(ue.harq.pending_retx_logical_channels), 0)
        self.assertEqual(
            [logical_channel_group.queue_bytes for logical_channel_group in ue.logical_channel_groups],
            [0, 120],
        )

    def test_state_updater_refreshes_lcg_bsr_reporting(self) -> None:
        updater = StateUpdater()
        ue = SimpleNamespace(
            ue_id="ue_0",
            logical_channels=[
                LogicalChannelState(
                    logical_channel_id="lc_0",
                    drb_id=1,
                    lcg_id=0,
                    queue_bytes=100,
                    priority=1,
                    packet_delay_budget_ms=50.0,
                    slice_id="embb",
                    traffic_type="video",
                )
            ],
        )
        state = SimpleNamespace(slot_id=1, ues=[ue])

        updater.refresh_logical_channel_groups(state)

        self.assertEqual(len(ue.logical_channel_groups), 1)
        logical_channel_group = ue.logical_channel_groups[0]
        self.assertEqual(logical_channel_group.bsr_reported_bytes, 100)
        self.assertEqual(logical_channel_group.bsr_pending_bytes, 0)
        self.assertEqual(logical_channel_group.last_bsr_report_slot, 1)
        self.assertEqual(logical_channel_group.bsr_trigger_reason, "initial")
        self.assertTrue(logical_channel_group.bsr_triggered)

        ue.logical_channels[0].queue_bytes = 160
        state.slot_id = 2
        updater.refresh_logical_channel_groups(state)

        logical_channel_group = ue.logical_channel_groups[0]
        self.assertEqual(logical_channel_group.bsr_reported_bytes, 160)
        self.assertEqual(logical_channel_group.last_bsr_report_slot, 2)
        self.assertEqual(logical_channel_group.bsr_trigger_reason, "buffer_growth")
        self.assertEqual(logical_channel_group.bsr_age_slots, 0)

        state.slot_id = 7
        updater.refresh_logical_channel_groups(state)
        logical_channel_group = ue.logical_channel_groups[0]
        self.assertEqual(logical_channel_group.last_bsr_report_slot, 7)
        self.assertEqual(logical_channel_group.bsr_trigger_reason, "periodic")
        self.assertTrue(logical_channel_group.bsr_triggered)


if __name__ == "__main__":
    unittest.main()

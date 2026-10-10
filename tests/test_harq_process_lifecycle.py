import unittest
from types import SimpleNamespace

from core.state_updater import StateUpdater
from models.phy_result import PhyResult, PhyUeResult
from models.scheduling import UeGrant, UeSchedulingDemand
from ran.scheduler import SimpleScheduler
from state.harq_state import HarqProcessState


class TestHarqProcessLifecycle(unittest.TestCase):
    def test_scheduler_reserves_new_tx_harq_process_with_lifecycle(self) -> None:
        scheduler = SimpleScheduler()
        harq = SimpleNamespace(
            process_pool_size=4,
            next_new_tx_process_id=1,
            ndi_by_process_id={},
            processes=[],
            process_id=0,
            ndi=0,
            redundancy_version=0,
            last_retransmission_round=0,
            last_harq_process_id=None,
        )
        ue = SimpleNamespace(
            ue_id="ue_0",
            traffic_type="control",
            latency_budget_ms=8.0,
            harq=harq,
        )
        demand = UeSchedulingDemand(
            ue_id="ue_0",
            true_backlog_bytes=256,
            demand_bytes=256,
            backlog_bytes=256,
            observed_backlog_bytes=256,
            hol_delay_ms=1.0,
            priority=3,
            qos_class="urllc",
            five_qi=80,
            slice_id="urllc",
            traffic_type="control",
            bsr_like_bytes=256,
            bsr_reported_bytes=256,
            bsr_pending_bytes=0,
            harq_pending_bytes=0,
            harq_urgency_score=0.0,
            cqi=10,
            ri=1,
            pmi=0,
            cqi_validity_age_slots=0,
            ri_validity_age_slots=0,
            pmi_validity_age_slots=0,
            measurement_age_slots=0,
            channel_quality_confidence=0.95,
            connection_state="connected",
            demand_source="unit_test",
            demand_reason="test_new_tx",
        )
        state = SimpleNamespace(slot_id=12, slot_duration_ms=1.0)

        process_id, redundancy_version, ndi, retransmission_round = scheduler._reserve_new_tx_harq_context(
            state,
            ue,
            demand,
        )

        self.assertEqual(process_id, 1)
        self.assertEqual(redundancy_version, 0)
        self.assertEqual(retransmission_round, 0)
        self.assertEqual(ndi, 1)
        self.assertEqual(harq.next_new_tx_process_id, 2)
        self.assertEqual(len(harq.processes), 1)
        reserved_process = harq.processes[0]
        self.assertTrue(reserved_process.is_busy)
        self.assertTrue(reserved_process.is_waiting_ack)
        self.assertEqual(reserved_process.process_id, 1)
        self.assertEqual(reserved_process.tb_id, "ue_0:slot12:proc1:ndi1")
        self.assertEqual(reserved_process.max_retransmission_rounds, 2)
        self.assertEqual(reserved_process.service_type, "control")
        self.assertEqual(reserved_process.retx_deadline_slot, 15)
        self.assertGreaterEqual(int(reserved_process.timeout_slot), int(reserved_process.retx_deadline_slot))

    def test_state_updater_flushes_harq_process_on_max_rounds(self) -> None:
        harq = SimpleNamespace(
            process_id=1,
            is_waiting_ack=False,
            last_decision="NACK",
            retransmission_count=1,
            pending_retx_bytes=120,
            last_tx_bytes=120,
            last_tx_was_retx=True,
            last_delivered_bytes=0,
            redundancy_version=2,
            ndi=1,
            processes=[
                HarqProcessState(
                    process_id=1,
                    is_busy=True,
                    ndi=1,
                    tb_id="ue_0:slot3:proc1:ndi1",
                    redundancy_version=2,
                    retransmission_round=1,
                    pending_bytes=120,
                    max_retransmission_rounds=1,
                    retx_deadline_slot=10,
                    timeout_slot=10,
                )
            ],
            harq_flush_count=0,
            last_flush_reason=None,
            last_timeout_slot=None,
        )
        ue = SimpleNamespace(
            ue_id="ue_0",
            queue_bytes=300,
            throughput_mbps=0.0,
            sinr_db=8.0,
            radio_quality_db=8.0,
            bler=0.2,
            serving_beam_id=0,
            slice_id="embb",
            traffic_type="video",
            latency_budget_ms=50.0,
            priority=1,
            harq=harq,
            is_scheduled=True,
            allocated_prbs=4,
            allocated_symbols=4,
            is_retx=True,
            priority_score=10.0,
        )
        state = SimpleNamespace(
            slot_id=5,
            sim_time_ms=5.0,
            slot_duration_ms=1.0,
            ues=[ue],
            avg_sinr_db=None,
            avg_bler=None,
            total_tb_size_bytes=0,
            last_phy_result=None,
            cell=SimpleNamespace(tx_power_dbm=30.0, current_codebook_id=0, current_beam_profile_id=0),
            traffic=SimpleNamespace(arrival_bytes_total=0, arrival_bytes_by_ue={}),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                scheduled_ue_ids=["ue_0"],
                grants=[
                    UeGrant(
                        ue_id="ue_0",
                        allocated_prbs=4,
                        allocated_symbols=4,
                        is_retx=True,
                        requested_bytes=120,
                        harq_process_id=1,
                        harq_redundancy_version=2,
                        harq_ndi=1,
                        harq_retransmission_round=1,
                    )
                ],
            ),
        )
        phy_result = PhyResult(
            slot_id=5,
            ue_results=[
                PhyUeResult(
                    ue_id="ue_0",
                    is_scheduled=True,
                    allocated_prbs=4,
                    allocated_symbols=4,
                    is_retx=True,
                    requested_bytes=120,
                    harq_process_id=1,
                    harq_redundancy_version=2,
                    harq_ndi=1,
                    harq_retransmission_round=1,
                    scheduled_bytes=120,
                    delivered_bytes=0,
                    ack=False,
                    sinr_db=5.0,
                    bler=0.6,
                    tb_size_bytes=120,
                    transport_block_size_bytes=120,
                    throughput_mbps=0.0,
                )
            ],
        )

        StateUpdater().apply_phy_result(state, phy_result)

        self.assertEqual(ue.queue_bytes, 300)
        self.assertEqual(ue.harq.pending_retx_bytes, 0)
        self.assertEqual(len(ue.harq.processes), 0)
        self.assertEqual(ue.harq.harq_flush_count, 1)
        self.assertEqual(ue.harq.last_flush_reason, "max_rounds")
        self.assertEqual(ue.harq.last_decision, "NACK_MAX_ROUNDS")

    def test_state_updater_flushes_harq_process_on_timeout(self) -> None:
        harq = SimpleNamespace(
            process_id=2,
            is_waiting_ack=False,
            last_decision="NACK",
            retransmission_count=1,
            pending_retx_bytes=64,
            last_tx_bytes=64,
            last_tx_was_retx=True,
            last_delivered_bytes=0,
            redundancy_version=2,
            ndi=1,
            processes=[
                HarqProcessState(
                    process_id=2,
                    is_busy=True,
                    ndi=1,
                    tb_id="ue_1:slot2:proc2:ndi1",
                    redundancy_version=2,
                    retransmission_round=1,
                    pending_bytes=64,
                    max_retransmission_rounds=4,
                    retx_deadline_slot=8,
                    timeout_slot=4,
                )
            ],
            harq_flush_count=0,
            last_flush_reason=None,
            last_timeout_slot=None,
        )
        ue = SimpleNamespace(
            ue_id="ue_1",
            queue_bytes=128,
            throughput_mbps=0.0,
            sinr_db=6.0,
            radio_quality_db=6.0,
            bler=0.3,
            serving_beam_id=0,
            slice_id="urllc",
            traffic_type="control",
            latency_budget_ms=10.0,
            priority=3,
            harq=harq,
            is_scheduled=True,
            allocated_prbs=2,
            allocated_symbols=4,
            is_retx=True,
            priority_score=20.0,
        )
        state = SimpleNamespace(
            slot_id=5,
            sim_time_ms=5.0,
            slot_duration_ms=1.0,
            ues=[ue],
            avg_sinr_db=None,
            avg_bler=None,
            total_tb_size_bytes=0,
            last_phy_result=None,
            cell=SimpleNamespace(tx_power_dbm=30.0, current_codebook_id=0, current_beam_profile_id=0),
            traffic=SimpleNamespace(arrival_bytes_total=0, arrival_bytes_by_ue={}),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                scheduled_ue_ids=["ue_1"],
                grants=[
                    UeGrant(
                        ue_id="ue_1",
                        allocated_prbs=2,
                        allocated_symbols=4,
                        is_retx=True,
                        requested_bytes=64,
                        harq_process_id=2,
                        harq_redundancy_version=2,
                        harq_ndi=1,
                        harq_retransmission_round=1,
                    )
                ],
            ),
        )
        phy_result = PhyResult(
            slot_id=5,
            ue_results=[
                PhyUeResult(
                    ue_id="ue_1",
                    is_scheduled=True,
                    allocated_prbs=2,
                    allocated_symbols=4,
                    is_retx=True,
                    requested_bytes=64,
                    harq_process_id=2,
                    harq_redundancy_version=2,
                    harq_ndi=1,
                    harq_retransmission_round=1,
                    scheduled_bytes=64,
                    delivered_bytes=0,
                    ack=False,
                    sinr_db=4.0,
                    bler=0.7,
                    tb_size_bytes=64,
                    transport_block_size_bytes=64,
                    throughput_mbps=0.0,
                )
            ],
        )

        StateUpdater().apply_phy_result(state, phy_result)

        self.assertEqual(ue.harq.pending_retx_bytes, 0)
        self.assertEqual(len(ue.harq.processes), 0)
        self.assertEqual(ue.harq.harq_flush_count, 1)
        self.assertEqual(ue.harq.last_flush_reason, "timeout")
        self.assertEqual(ue.harq.last_timeout_slot, 5)
        self.assertEqual(ue.harq.last_decision, "NACK_TIMEOUT")


if __name__ == "__main__":
    unittest.main()

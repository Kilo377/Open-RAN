import unittest
from types import SimpleNamespace

from core.state_updater import StateUpdater
from models.phy_result import PhyResult, PhyUeResult
from models.scheduling import UeGrant
from state.harq_state import HarqProcessState


class TestHarqResourceAccounting(unittest.TestCase):
    def test_state_updater_splits_new_and_retx_bytes(self) -> None:
        retx_harq = SimpleNamespace(
            process_id=2,
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
                    process_id=2,
                    ndi=1,
                    redundancy_version=2,
                    retransmission_round=1,
                    pending_bytes=120,
                )
            ],
        )
        new_harq = SimpleNamespace(
            process_id=1,
            is_waiting_ack=False,
            last_decision="IDLE",
            retransmission_count=0,
            pending_retx_bytes=0,
            last_tx_bytes=0,
            last_tx_was_retx=False,
            last_delivered_bytes=0,
            redundancy_version=0,
            ndi=1,
            next_new_tx_process_id=2,
            process_pool_size=4,
            ndi_by_process_id={1: 1},
            processes=[],
        )

        ue_retx = SimpleNamespace(
            ue_id="ue_retx",
            queue_bytes=500,
            throughput_mbps=0.0,
            sinr_db=8.0,
            radio_quality_db=8.0,
            bler=0.2,
            serving_beam_id=0,
            slice_id="embb",
            traffic_type="video",
            latency_budget_ms=50.0,
            priority=1,
            harq=retx_harq,
            is_scheduled=True,
            allocated_prbs=4,
            allocated_symbols=4,
            is_retx=True,
            priority_score=10.0,
        )
        ue_new = SimpleNamespace(
            ue_id="ue_new",
            queue_bytes=700,
            throughput_mbps=0.0,
            sinr_db=10.0,
            radio_quality_db=10.0,
            bler=0.1,
            serving_beam_id=0,
            slice_id="urllc",
            traffic_type="cbr",
            latency_budget_ms=10.0,
            priority=2,
            harq=new_harq,
            is_scheduled=True,
            allocated_prbs=4,
            allocated_symbols=4,
            is_retx=False,
            priority_score=20.0,
        )

        state = SimpleNamespace(
            slot_id=4,
            sim_time_ms=4.0,
            slot_duration_ms=1.0,
            avg_sinr_db=None,
            avg_bler=None,
            total_tb_size_bytes=0,
            last_phy_result=None,
            ues=[ue_retx, ue_new],
            cell=SimpleNamespace(
                tx_power_dbm=30.0,
                current_codebook_id=0,
                current_beam_profile_id=0,
            ),
            traffic=SimpleNamespace(arrival_bytes_total=0, arrival_bytes_by_ue={}),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                scheduled_ue_ids=["ue_retx", "ue_new"],
                grants=[
                    UeGrant(
                        ue_id="ue_retx",
                        allocated_prbs=4,
                        allocated_symbols=4,
                        is_retx=True,
                        requested_bytes=120,
                        harq_process_id=2,
                        harq_redundancy_version=2,
                        harq_ndi=1,
                        harq_retransmission_round=1,
                    ),
                    UeGrant(
                        ue_id="ue_new",
                        allocated_prbs=4,
                        allocated_symbols=4,
                        is_retx=False,
                        requested_bytes=200,
                        harq_process_id=1,
                        harq_redundancy_version=0,
                        harq_ndi=1,
                        harq_retransmission_round=0,
                    ),
                ],
            ),
        )

        phy_result = PhyResult(
            slot_id=4,
            ue_results=[
                PhyUeResult(
                    ue_id="ue_retx",
                    is_scheduled=True,
                    allocated_prbs=4,
                    allocated_symbols=4,
                    is_retx=True,
                    requested_bytes=120,
                    harq_process_id=2,
                    harq_redundancy_version=2,
                    harq_ndi=1,
                    harq_retransmission_round=1,
                    scheduled_bytes=120,
                    delivered_bytes=120,
                    ack=True,
                    sinr_db=12.0,
                    bler=0.05,
                    tb_size_bytes=120,
                    transport_block_size_bytes=120,
                    throughput_mbps=0.96,
                ),
                PhyUeResult(
                    ue_id="ue_new",
                    is_scheduled=True,
                    allocated_prbs=4,
                    allocated_symbols=4,
                    is_retx=False,
                    requested_bytes=200,
                    harq_process_id=1,
                    harq_redundancy_version=0,
                    harq_ndi=1,
                    harq_retransmission_round=0,
                    scheduled_bytes=200,
                    delivered_bytes=0,
                    ack=False,
                    sinr_db=7.0,
                    bler=0.4,
                    tb_size_bytes=200,
                    transport_block_size_bytes=200,
                    throughput_mbps=0.0,
                ),
            ],
            avg_sinr_db=9.5,
            avg_bler=0.225,
            total_tb_size_bytes=320,
        )

        updater = StateUpdater()
        updater.apply_phy_result(state, phy_result)

        self.assertEqual(ue_retx.queue_bytes, 380)
        self.assertEqual(ue_retx.retx_tx_bytes, 120)
        self.assertEqual(ue_retx.delivered_retx_bytes, 120)
        self.assertEqual(ue_retx.harq.pending_retx_bytes, 0)
        self.assertEqual(ue_retx.harq.retransmission_count, 0)
        self.assertEqual(len(ue_retx.harq.processes), 0)

        self.assertEqual(ue_new.queue_bytes, 700)
        self.assertEqual(ue_new.new_tx_bytes, 200)
        self.assertEqual(ue_new.delivered_new_bytes, 0)
        self.assertEqual(ue_new.harq.pending_retx_bytes, 200)
        self.assertEqual(ue_new.harq.retransmission_count, 1)
        self.assertEqual(len(ue_new.harq.processes), 1)
        self.assertEqual(ue_new.harq.processes[0].process_id, 1)
        self.assertEqual(ue_new.harq.processes[0].redundancy_version, 2)
        self.assertEqual(ue_new.harq.processes[0].ndi, 1)
        self.assertEqual(ue_new.harq.processes[0].retransmission_round, 1)

        slot_records = []
        updater.record_slot_metrics(
            state=state,
            phy_result=phy_result,
            slot_records=slot_records,
        )
        self.assertEqual(len(slot_records), 1)
        true_metrics = slot_records[0]["true_metrics"]
        self.assertEqual(true_metrics["new_tx_bytes_this_slot"], 200)
        self.assertEqual(true_metrics["retx_tx_bytes_this_slot"], 120)
        self.assertEqual(true_metrics["delivered_new_bytes_this_slot"], 0)
        self.assertEqual(true_metrics["delivered_retx_bytes_this_slot"], 120)


if __name__ == "__main__":
    unittest.main()

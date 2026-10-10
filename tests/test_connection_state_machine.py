import unittest
from types import SimpleNamespace
from typing import Any, cast

from core.metrics import build_observation
from core.state_updater import StateUpdater
from models.phy_result import PhyResult, PhyUeResult
from ran.scheduler import SimpleScheduler
from state.connection_state import ConnectionState
from state.link_state import LinkState


class TestConnectionStateMachine(unittest.TestCase):
    def test_scheduler_only_services_connected_ues(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=2,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )
        ue_connected = SimpleNamespace(
            ue_id="ue_connected",
            queue_bytes=600,
            latency_budget_ms=20.0,
            priority=2,
            sinr_db=10.0,
            radio_quality_db=10.0,
            slice_id="urllc",
            serving_beam_id=0,
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=0),
        )
        ue_out_of_sync = SimpleNamespace(
            ue_id="ue_oos",
            queue_bytes=1200,
            latency_budget_ms=10.0,
            priority=3,
            sinr_db=11.0,
            radio_quality_db=11.0,
            slice_id="urllc",
            serving_beam_id=0,
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=1),
        )
        state = SimpleNamespace(
            slot_id=5,
            cell=SimpleNamespace(cell_id="cell_0"),
            ues=[ue_connected, ue_out_of_sync],
            connections={
                "ue_connected": ConnectionState(
                    ue_id="ue_connected",
                    serving_cell_id="cell_0",
                    connection_state="connected",
                    c_rnti=4096,
                ),
                "ue_oos": ConnectionState(
                    ue_id="ue_oos",
                    serving_cell_id="cell_0",
                    connection_state="out_of_sync",
                    c_rnti=4097,
                ),
            },
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

        scheduled = scheduler.run(state)

        self.assertEqual(scheduled, ["ue_connected"])
        self.assertEqual(len(state.scheduler.grants), 1)
        self.assertEqual(state.scheduler.grants[0].ue_id, "ue_connected")

    def test_observation_exposes_connection_fields(self) -> None:
        state = SimpleNamespace(
            slot_id=3,
            cell=SimpleNamespace(
                cell_id="cell_0",
                current_codebook_id=0,
                current_beam_profile_id=0,
                tx_power_dbm=30.0,
            ),
            scheduler=SimpleNamespace(
                scheduler_mode="round_robin",
                scheduled_ue_ids=[],
            ),
            ues=[
                SimpleNamespace(
                    ue_id="ue_0",
                    queue_bytes=100,
                    throughput_mbps=0.0,
                    sinr_db=6.0,
                    bler=0.2,
                    serving_beam_id=1,
                    slice_id="embb",
                    traffic_type="video",
                    latency_budget_ms=100.0,
                    priority=1,
                    radio_quality_db=5.0,
                    cqi=4,
                    rank=1,
                    pmi=2,
                )
            ],
            connections={
                "ue_0": ConnectionState(
                    ue_id="ue_0",
                    serving_cell_id="cell_0",
                    connection_state="connected",
                    c_rnti=4100,
                    serving_beam_id=1,
                    rank_indicator=1,
                    pmi=2,
                    cqi=4,
                    active_harq_processes=[0],
                )
            },
            measurement=None,
        )

        observation = build_observation(cast(Any, state))
        ue_observation = observation.ue_observations[0]

        self.assertEqual(observation.connected_ue_count, 1)
        self.assertEqual(observation.out_of_sync_ue_count, 0)
        self.assertEqual(ue_observation.serving_cell_id, "cell_0")
        self.assertEqual(ue_observation.connection_state, "connected")
        self.assertEqual(ue_observation.c_rnti, 4100)
        self.assertEqual(ue_observation.active_harq_process_count, 1)

    def test_state_updater_marks_connection_out_of_sync_after_consecutive_nacks(self) -> None:
        state = SimpleNamespace(
            slot_id=9,
            slot_duration_ms=1.0,
            scheduler=SimpleNamespace(
                scheduled_ue_ids=["ue_0"],
                grants=[
                    SimpleNamespace(
                        ue_id="ue_0",
                        allocated_prbs=6,
                        allocated_symbols=4,
                        is_retx=True,
                        priority_score=10.0,
                        grant_bytes_budget=180,
                        target_mcs=4,
                    )
                ],
            ),
            ues=[
                SimpleNamespace(
                    ue_id="ue_0",
                    queue_bytes=400,
                    throughput_mbps=0.0,
                    sinr_db=4.0,
                    bler=0.5,
                    serving_beam_id=2,
                    is_scheduled=True,
                    harq=SimpleNamespace(
                        pending_retx_bytes=180,
                        process_id=3,
                        retransmission_count=7,
                        last_decision="NACK",
                    ),
                )
            ],
            connections={
                "ue_0": ConnectionState(
                    ue_id="ue_0",
                    serving_cell_id="cell_0",
                    connection_state="connected",
                    c_rnti=4101,
                    serving_beam_id=0,
                    consecutive_nack_count=7,
                )
            },
            links={
                "ue_0": LinkState(
                    ue_id="ue_0",
                    serving_cell_id="cell_0",
                    pathloss_db=110.0,
                    shadowing_db=2.0,
                    doppler_hz=20.0,
                    delay_spread_ns=200.0,
                    channel_model_type="tdl_c",
                    fast_fading_seed=33,
                )
            },
        )
        phy_result = PhyResult(
            slot_id=9,
            cell_id="cell_0",
            ue_results=[
                PhyUeResult(
                    ue_id="ue_0",
                    is_scheduled=True,
                    allocated_prbs=6,
                    allocated_symbols=4,
                    is_retx=True,
                    scheduled_bytes=180,
                    delivered_bytes=0,
                    grant_bytes_budget=180,
                    ack=False,
                    sinr_db=2.0,
                    bler=0.8,
                    cqi=1,
                    rank=1,
                    pmi=1,
                    throughput_mbps=0.0,
                )
            ],
        )

        updater = StateUpdater()
        updater.apply_phy_result(state, phy_result)

        connection = state.connections["ue_0"]
        link = state.links["ue_0"]
        self.assertEqual(connection.connection_state, "out_of_sync")
        self.assertEqual(connection.out_of_sync_reason, "consecutive_nack_threshold")
        self.assertEqual(connection.cqi, 1)
        self.assertEqual(connection.pmi, 1)
        self.assertEqual(link.last_post_eq_sinr_db, 2.0)
        self.assertEqual(link.cqi, 1)
        self.assertEqual(link.pmi, 1)


if __name__ == "__main__":
    unittest.main()

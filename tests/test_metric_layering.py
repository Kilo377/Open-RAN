import unittest
from types import SimpleNamespace

from core.state_updater import StateUpdater
from models.action import ControlMetadata
from models.observation import Observation
from models.oran_lifecycle import (
    A1EIJob,
    A1EIType,
    A1PolicyInstance,
    A1PolicyStatus,
    A1PolicyType,
    E2ControlOutcome,
    E2ControlRequest,
    E2Indication,
    E2Subscription,
    OranLifecycleResources,
)
from models.phy_result import PhyResult, PhyUeResult
from models.scheduling import SchedulingDecision, SchedulingStageTrace, UeGrant, UeSchedulingDemand


class TestMetricLayering(unittest.TestCase):
    def test_record_slot_metrics_separates_true_observable_and_traces(self) -> None:
        state = SimpleNamespace(
            slot_id=7,
            sim_time_ms=7.0,
            slot_duration_ms=1.0,
            ues=[
                SimpleNamespace(
                    ue_id="ue_0",
                    queue_bytes=320,
                    throughput_mbps=1.2,
                    sinr_db=9.0,
                    radio_quality_db=8.5,
                    harq=SimpleNamespace(pending_retx_bytes=40),
                )
            ],
            cell=SimpleNamespace(
                tx_power_dbm=30.0,
                current_codebook_id=1,
                current_beam_profile_id=0,
            ),
            traffic=SimpleNamespace(arrival_bytes_total=120, arrival_bytes_by_ue={"ue_0": 120}),
            measurement=SimpleNamespace(
                produced_at_slot=7,
                delivered_at_slot=None,
                age_slots=0,
                queue_delay_slots=1,
                sinr_window_slots=8,
                sinr_noise_std_db_applied=2.0,
                bler_window_tbs=20,
                throughput_granularity_slots=10,
            ),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                scheduled_ue_ids=["ue_0"],
                grants=[
                    UeGrant(
                        ue_id="ue_0",
                        allocated_prbs=6,
                        allocated_symbols=4,
                        is_retx=False,
                        requested_bytes=180,
                        grant_bytes_budget=160,
                    )
                ],
                last_decision=SchedulingDecision(
                    slot_id=7,
                    scheduler_mode="latency_aware",
                    total_prbs=12,
                    total_symbols=14,
                    grants=[],
                    unallocated_prbs=6,
                    unallocated_symbols=10,
                ),
                last_demands=[
                    UeSchedulingDemand(
                        ue_id="ue_0",
                        true_backlog_bytes=320,
                        demand_bytes=320,
                        backlog_bytes=320,
                        observed_backlog_bytes=360,
                        hol_delay_ms=1.0,
                        priority=2,
                        qos_class="embb",
                        five_qi=9,
                        slice_id="embb",
                        traffic_type="video",
                        bsr_like_bytes=320,
                        bsr_reported_bytes=300,
                        bsr_pending_bytes=20,
                        harq_pending_bytes=40,
                        harq_urgency_score=140.0,
                        cqi=7,
                        ri=1,
                        pmi=0,
                        cqi_validity_age_slots=1,
                        ri_validity_age_slots=1,
                        pmi_validity_age_slots=1,
                        measurement_age_slots=0,
                        channel_quality_confidence=0.8,
                    )
                ],
                last_stage_trace=SchedulingStageTrace(
                    demand_source="mac_state+measurement_metadata",
                    demand_visible_ue_ids=["ue_0"],
                    harq_urgent_ue_ids=["ue_0"],
                    feasible_candidate_ue_ids=["ue_0"],
                    retx_candidate_ue_ids=["ue_0"],
                    new_tx_candidate_ue_ids=[],
                    selected_retx_ue_ids=["ue_0"],
                    selected_new_tx_ue_ids=[],
                    grant_candidate_ue_ids=["ue_0"],
                    granted_ue_ids=["ue_0"],
                    selection_reasons_by_ue={"ue_0": "selected_for_harq_retransmission"},
                    resource_pressure_source_by_ue={"ue_0": "shared_slot_budget_under_harq_pressure"},
                    constraint_hits_by_ue={"ue_0": ["harq_retx"]},
                    lc_multiplexing_summaries_by_ue={
                        "ue_0": {
                            "logical_channel_count": 1,
                            "scheduled_bytes_total": 160,
                        }
                    },
                ),
            ),
            control=SimpleNamespace(
                decision_metadata=ControlMetadata(
                    source="controller_manager",
                    issued_at_slot=7,
                    reason="merged_control_decision",
                    decision_id="decision:7",
                ),
                actuation_outcomes=[{"status": "applied"}],
                input_observation_trace={
                    "observation_slot_id": 6,
                    "observation_produced_at_slot": 6,
                    "observation_delivered_at_slot": 7,
                    "measurement_produced_at_slot": 6,
                    "measurement_delivered_at_slot": 7,
                    "measurement_age_slots": 1,
                    "measurement_queue_delay_slots": 1,
                    "measurement_sinr_window_slots": 8,
                    "measurement_sinr_noise_std_db_applied": 2.0,
                    "measurement_bler_window_tbs": 20,
                    "measurement_throughput_granularity_slots": 10,
                },
                oran_lifecycle_resources=OranLifecycleResources(
                    a1_policy_types=[
                        A1PolicyType(policy_type_id="pt:1", name="codebook_policy")
                    ],
                    a1_policy_instances=[
                        A1PolicyInstance(
                            instance_id="pi:1",
                            policy_type_id="pt:1",
                            source_name="codebook_rapp",
                            status="active",
                        )
                    ],
                    a1_policy_statuses=[
                        A1PolicyStatus(
                            instance_id="pi:1",
                            status="active",
                            reason="policy_update_applied_to_epoch",
                        )
                    ],
                    a1_ei_types=[
                        A1EIType(ei_type_id="eit:1", name="observation_context")
                    ],
                    a1_ei_jobs=[
                        A1EIJob(
                            job_id="ej:1",
                            ei_type_id="eit:1",
                            consumer_name="codebook_rapp",
                            status="active",
                        )
                    ],
                    e2_subscriptions=[
                        E2Subscription(
                            subscription_id="sub:1",
                            subscriber_name="beam_xapp",
                            status="active",
                        )
                    ],
                    e2_indications=[
                        E2Indication(
                            indication_id="ind:1",
                            subscription_id="sub:1",
                            producer_name="controller_manager",
                        )
                    ],
                    e2_control_requests=[
                        E2ControlRequest(
                            request_id="req:1",
                            source_name="controller_manager",
                            status="applied",
                        ),
                        E2ControlRequest(
                            request_id="req:2",
                            source_name="controller_manager",
                            status="queued",
                        ),
                    ],
                    e2_control_outcomes=[
                        E2ControlOutcome(
                            outcome_id="out:1",
                            request_id="req:1",
                            status="applied",
                        )
                    ],
                ),
            ),
        )
        phy_result = PhyResult(
            slot_id=7,
            ue_results=[
                PhyUeResult(
                    ue_id="ue_0",
                    is_scheduled=True,
                    allocated_prbs=6,
                    allocated_symbols=4,
                    is_retx=False,
                    scheduled_bytes=160,
                    delivered_bytes=160,
                    ack=True,
                    sinr_db=10.5,
                    post_eq_sinr_db=10.5,
                    effective_sinr_db=10.5,
                    bler=0.08,
                    throughput_mbps=1.2,
                    transport_block_size_bytes=160,
                    beam_gain_db=1.1,
                    mcs_index=5,
                    spatial_mode="mu_mimo",
                    spatial_group_id=2,
                    spatial_multiplexing_gain_db=1.4,
                )
            ],
        )
        shadow_phy_result = PhyResult(
            slot_id=7,
            ue_results=[
                PhyUeResult(
                    ue_id="ue_0",
                    is_scheduled=True,
                    allocated_prbs=6,
                    allocated_symbols=4,
                    is_retx=False,
                    scheduled_bytes=150,
                    delivered_bytes=150,
                    ack=True,
                    sinr_db=9.5,
                    post_eq_sinr_db=9.4,
                    bler=0.12,
                    throughput_mbps=1.0,
                    transport_block_size_bytes=150,
                    spatial_mode="mu_mimo",
                    spatial_group_id=2,
                    spatial_multiplexing_gain_db=1.0,
                )
            ],
        )
        observation = Observation(
            slot_id=7,
            avg_sinr_db=8.7,
            avg_bler=0.08,
            total_queue_bytes=360,
            total_throughput_mbps=0.9,
            cell_id="cell_0",
            current_codebook_id=1,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="latency_aware",
            scheduled_ue_ids=["ue_0"],
            num_scheduled_ues=1,
            ue_observations=[],
            produced_at_slot=7,
            delivered_at_slot=None,
            measurement_produced_at_slot=7,
            measurement_delivered_at_slot=None,
            measurement_age_slots=0,
            measurement_queue_delay_slots=1,
            measurement_sinr_window_slots=8,
            measurement_sinr_noise_std_db_applied=2.0,
            measurement_bler_window_tbs=20,
            measurement_throughput_granularity_slots=10,
            measurement_queue_age_slots_mean=1.0,
            measurement_sinr_effective_sample_count_mean=4.0,
            measurement_bler_sample_count_mean=3.0,
            measurement_throughput_sample_count_mean=5.0,
        )

        slot_records = []
        updater = StateUpdater()
        updater.record_slot_metrics(
            state=state,
            observation=observation,
            slot_records=slot_records,
            phy_result=phy_result,
            shadow_phy_result=shadow_phy_result,
            shadow_phy_label="custom_phy",
        )

        self.assertEqual(len(slot_records), 1)
        record = slot_records[0]
        self.assertIn("true_metrics", record)
        self.assertIn("observable_metrics", record)
        self.assertIn("measurement_trace", record)
        self.assertIn("control_trace", record)
        self.assertIn("scheduler_trace", record)
        self.assertIn("phy_comparison_trace", record)
        self.assertEqual(record["true_metrics"]["queue_bytes"], 320)
        self.assertEqual(record["true_metrics"]["new_tx_bytes_this_slot"], 160)
        self.assertEqual(record["observable_metrics"]["queue_bytes"], 360)
        self.assertEqual(record["observable_metrics"]["sinr_db"], 8.7)
        self.assertEqual(record["measurement_trace"]["age_slots"], 0)
        self.assertEqual(record["measurement_trace"]["sinr_window_slots"], 8)
        self.assertEqual(record["control_trace"]["input_measurement_age_slots"], 1)
        self.assertEqual(record["control_trace"]["a1_policy_instance_count"], 1)
        self.assertEqual(record["control_trace"]["e2_subscription_count"], 1)
        self.assertEqual(record["control_trace"]["e2_control_request_count"], 2)
        self.assertEqual(
            record["control_trace"]["e2_control_request_status_counts"],
            {"applied": 1, "queued": 1},
        )
        self.assertEqual(record["scheduler_trace"]["demand_count"], 1)
        self.assertEqual(record["scheduler_trace"]["demand_true_backlog_bytes_total"], 320)
        self.assertEqual(record["scheduler_trace"]["demand_visible_bytes_total"], 320)
        self.assertEqual(record["scheduler_trace"]["demand_bsr_reported_bytes_total"], 300)
        self.assertEqual(record["scheduler_trace"]["demand_bsr_pending_bytes_total"], 20)
        self.assertEqual(record["scheduler_trace"]["selected_retx_count"], 1)
        self.assertEqual(record["scheduler_trace"]["harq_urgent_count"], 1)
        self.assertEqual(record["scheduler_trace"]["demand_visible_ue_ids"], ["ue_0"])
        self.assertEqual(record["scheduler_trace"]["granted_ue_ids"], ["ue_0"])
        self.assertEqual(
            record["scheduler_trace"]["selection_reasons_by_ue"]["ue_0"],
            "selected_for_harq_retransmission",
        )
        self.assertEqual(
            record["scheduler_trace"]["resource_pressure_source_by_ue"]["ue_0"],
            "shared_slot_budget_under_harq_pressure",
        )
        self.assertEqual(record["scheduler_trace"]["constraint_hits_by_ue"]["ue_0"], ["harq_retx"])
        self.assertEqual(record["control_trace"]["decision_id"], "decision:7")
        self.assertEqual(record["phy_comparison_trace"]["shadow_phy_label"], "custom_phy")
        self.assertEqual(record["phy_comparison_trace"]["compared_ue_count"], 1)
        self.assertAlmostEqual(record["phy_comparison_trace"]["mean_abs_sinr_gap_db"], 1.0)
        self.assertAlmostEqual(record["phy_comparison_trace"]["mean_abs_post_eq_sinr_gap_db"], 1.1)
        self.assertAlmostEqual(record["phy_comparison_trace"]["max_abs_post_eq_sinr_gap_db"], 1.1)
        self.assertEqual(record["phy_comparison_trace"]["worst_ue_id_by_post_eq_sinr_gap"], "ue_0")
        self.assertEqual(len(record["phy_comparison_trace"]["per_ue_diffs"]), 1)
        self.assertEqual(record["phy_comparison_trace"]["per_ue_diffs"][0]["ue_id"], "ue_0")
        self.assertEqual(record["phy_comparison_trace"]["per_ue_diffs"][0]["primary_spatial_group_id"], 2)
        self.assertEqual(len(record["phy_comparison_trace"]["spatial_group_summaries"]), 1)
        self.assertEqual(record["phy_comparison_trace"]["spatial_group_summaries"][0]["spatial_group_id"], 2)
        self.assertAlmostEqual(
            record["phy_comparison_trace"]["spatial_group_summaries"][0]["mean_post_eq_sinr_gap_db"],
            1.1,
        )


if __name__ == "__main__":
    unittest.main()

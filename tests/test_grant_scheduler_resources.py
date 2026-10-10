import unittest
from types import SimpleNamespace

from ran.scheduler import SimpleScheduler


class TestGrantSchedulerResources(unittest.TestCase):
    def test_scheduler_produces_multi_ue_grants_with_resource_budget(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=2,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )

        ue_a = SimpleNamespace(
            ue_id="ue_a",
            queue_bytes=500,
            latency_budget_ms=80.0,
            priority=1,
            sinr_db=9.0,
            slice_id="embb",
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=0),
        )
        ue_b = SimpleNamespace(
            ue_id="ue_b",
            queue_bytes=800,
            latency_budget_ms=10.0,
            priority=3,
            sinr_db=11.0,
            slice_id="urllc",
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=0),
        )
        ue_c = SimpleNamespace(
            ue_id="ue_c",
            queue_bytes=600,
            latency_budget_ms=200.0,
            priority=1,
            sinr_db=8.0,
            slice_id="mmtc",
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=0),
        )

        state = SimpleNamespace(
            slot_id=3,
            ues=[ue_a, ue_b, ue_c],
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

        self.assertEqual(len(scheduled), 2)
        self.assertIn("ue_b", scheduled)
        self.assertEqual(len(state.scheduler.grants), 2)
        self.assertEqual(sum(grant.allocated_prbs for grant in state.scheduler.grants), 12)
        self.assertEqual(sum(grant.allocated_symbols for grant in state.scheduler.grants), 6)
        self.assertEqual(state.scheduler.grants[0].start_prb, 0)
        self.assertEqual(
            state.scheduler.grants[1].start_prb,
            state.scheduler.grants[0].allocated_prbs,
        )
        self.assertTrue(all(grant.start_symbol == 0 for grant in state.scheduler.grants))
        self.assertEqual(
            state.scheduler.grants[0].resource_block_indices,
            list(range(state.scheduler.grants[0].allocated_prbs)),
        )
        self.assertEqual(
            state.scheduler.grants[0].ofdm_symbol_indices,
            list(range(state.scheduler.grants[0].allocated_symbols)),
        )
        self.assertEqual(state.scheduler.grants[0].dmrs_pattern_id, "dmrs_type1_default")
        self.assertTrue(all(grant.grant_bytes_budget > 0 for grant in state.scheduler.grants))
        self.assertTrue(all(grant.target_mcs is not None for grant in state.scheduler.grants))
        self.assertIsNotNone(state.scheduler.last_decision)
        self.assertEqual(state.scheduler.last_decision.scheduled_ue_ids, scheduled)

    def test_scheduler_emits_retx_grant_before_new_tx(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=2,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )

        ue_retx = SimpleNamespace(
            ue_id="ue_retx",
            queue_bytes=900,
            latency_budget_ms=100.0,
            priority=1,
            sinr_db=8.0,
            slice_id="embb",
            harq=SimpleNamespace(pending_retx_bytes=180, process_id=3),
        )
        ue_new = SimpleNamespace(
            ue_id="ue_new",
            queue_bytes=700,
            latency_budget_ms=20.0,
            priority=2,
            sinr_db=10.0,
            slice_id="urllc",
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=1),
        )
        ue_other = SimpleNamespace(
            ue_id="ue_other",
            queue_bytes=500,
            latency_budget_ms=50.0,
            priority=1,
            sinr_db=9.0,
            slice_id="embb",
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=2),
        )

        state = SimpleNamespace(
            slot_id=9,
            ues=[ue_retx, ue_new, ue_other],
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

        self.assertEqual(scheduled[0], "ue_retx")
        retx_grant = state.scheduler.grants[0]
        self.assertTrue(retx_grant.is_retx)
        self.assertEqual(retx_grant.requested_bytes, 180)
        self.assertGreaterEqual(retx_grant.allocated_prbs, 2)
        self.assertGreaterEqual(retx_grant.allocated_symbols, 2)
        self.assertEqual(retx_grant.start_prb, 0)
        self.assertEqual(retx_grant.start_symbol, 0)
        self.assertEqual(retx_grant.resource_block_indices, list(range(retx_grant.allocated_prbs)))
        self.assertEqual(retx_grant.ofdm_symbol_indices, list(range(retx_grant.allocated_symbols)))
        self.assertIsNotNone(retx_grant.target_mcs)
        self.assertLessEqual(retx_grant.grant_bytes_budget, retx_grant.requested_bytes)
        self.assertEqual(retx_grant.harq_process_id, 3)
        self.assertEqual(retx_grant.harq_redundancy_version, 0)
        self.assertEqual(retx_grant.harq_ndi, 0)
        self.assertEqual(retx_grant.harq_retransmission_round, 0)

    def test_scheduler_assigns_new_tx_harq_context(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=1,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )

        ue = SimpleNamespace(
            ue_id="ue_new",
            queue_bytes=600,
            latency_budget_ms=20.0,
            priority=2,
            sinr_db=10.0,
            slice_id="urllc",
            harq=SimpleNamespace(
                pending_retx_bytes=0,
                process_id=0,
                next_new_tx_process_id=1,
                process_pool_size=4,
                ndi_by_process_id={1: 0},
                processes=[],
            ),
        )

        state = SimpleNamespace(
            slot_id=10,
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
        self.assertFalse(grant.is_retx)
        self.assertEqual(grant.harq_process_id, 1)
        self.assertEqual(grant.harq_redundancy_version, 0)
        self.assertEqual(grant.harq_ndi, 1)
        self.assertEqual(grant.harq_retransmission_round, 0)

    def test_scheduler_assigns_mcs_table_tbs_method_and_validity_age(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=1,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )

        ue = SimpleNamespace(
            ue_id="ue_embb",
            queue_bytes=1200,
            latency_budget_ms=15.0,
            priority=1,
            sinr_db=14.0,
            slice_id="embb",
            traffic_type="video",
            radio_quality_db=14.0,
            ri=2,
            pmi=1,
            cqi=10,
            logical_channels=[],
            harq=SimpleNamespace(
                pending_retx_bytes=0,
                process_id=0,
                next_new_tx_process_id=0,
                process_pool_size=8,
                ndi_by_process_id={0: 0},
                processes=[],
            ),
        )
        state = SimpleNamespace(
            slot_id=12,
            ues=[ue],
            links={"ue_embb": SimpleNamespace(cqi=10, ri=2, pmi=1)},
            measurement=SimpleNamespace(
                ue_measurements={
                    "ue_embb": SimpleNamespace(last_measurement_update_slot=6)
                }
            ),
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
        self.assertEqual(grant.mcs_table_id, "qam64_low_se")
        self.assertEqual(grant.tbs_selection_method, "guarded_resource_elements")
        self.assertEqual(grant.dmrs_pattern_id, "dmrs_type1_dense")
        self.assertEqual(grant.cqi_validity_age_slots, 6)
        self.assertEqual(grant.ri_validity_age_slots, 6)
        self.assertEqual(grant.pmi_validity_age_slots, 6)
        self.assertLess(grant.channel_quality_confidence or 1.0, 0.5)
        self.assertEqual(grant.link_adaptation_reason, "stale_low_confidence_guarded")
        self.assertEqual(ue.harq.next_new_tx_process_id, 1)
        self.assertEqual(ue.harq.ndi_by_process_id[0], 1)

    def test_scheduler_caps_poor_link_new_tx_as_probe(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=1,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )

        ue = SimpleNamespace(
            ue_id="ue_probe",
            queue_bytes=6000,
            latency_budget_ms=80.0,
            priority=1,
            sinr_db=-9.0,
            post_eq_sinr_db=-11.0,
            bler=0.90,
            slice_id="embb",
            traffic_type="video",
            radio_quality_db=-6.0,
            ri=1,
            pmi=0,
            cqi=2,
            logical_channels=[],
            harq=SimpleNamespace(
                pending_retx_bytes=0,
                process_id=0,
                next_new_tx_process_id=0,
                process_pool_size=4,
                ndi_by_process_id={0: 0},
                processes=[],
                harq_flush_count=1,
            ),
        )
        state = SimpleNamespace(
            slot_id=18,
            ues=[ue],
            links={"ue_probe": SimpleNamespace(cqi=2, ri=1, pmi=0)},
            measurement=SimpleNamespace(
                ue_measurements={"ue_probe": SimpleNamespace(last_measurement_update_slot=10)}
            ),
            control=SimpleNamespace(active_policy_constraints=[]),
            scheduler=SimpleNamespace(
                scheduler_mode="slice_aware",
                scheduled_ue_ids=[],
                grants=[],
                last_decision=None,
                rr_pointer=0,
                priority_boost_by_ue={},
                last_served_slot_by_ue={},
                slice_weight_adjustment={"embb": 1.0},
            ),
        )

        scheduler.run(state)

        grant = state.scheduler.grants[0]
        self.assertLessEqual(grant.requested_bytes, 512)
        self.assertEqual(grant.mcs_table_id, "qam64_low_se")
        self.assertEqual(grant.tbs_selection_method, "guarded_resource_elements")
        self.assertEqual(grant.link_adaptation_reason, "poor_link_probe_guarded")
        self.assertIn("poor_link_probe", grant.constraint_hits)
        self.assertLessEqual(grant.grant_bytes_budget, grant.requested_bytes)

    def test_scheduler_prefers_healthier_candidate_over_poor_link_probe(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=1,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )

        ue_poor = SimpleNamespace(
            ue_id="ue_poor",
            queue_bytes=5000,
            latency_budget_ms=80.0,
            priority=2,
            sinr_db=-8.0,
            post_eq_sinr_db=-9.0,
            bler=0.82,
            slice_id="embb",
            traffic_type="video",
            radio_quality_db=-6.0,
            ri=1,
            pmi=0,
            cqi=3,
            logical_channels=[],
            harq=SimpleNamespace(
                pending_retx_bytes=0,
                process_id=0,
                next_new_tx_process_id=0,
                process_pool_size=4,
                ndi_by_process_id={0: 0},
                processes=[],
                harq_flush_count=1,
            ),
        )
        ue_good = SimpleNamespace(
            ue_id="ue_good",
            queue_bytes=5000,
            latency_budget_ms=80.0,
            priority=2,
            sinr_db=9.0,
            post_eq_sinr_db=10.0,
            bler=0.08,
            slice_id="embb",
            traffic_type="video",
            radio_quality_db=9.0,
            ri=1,
            pmi=0,
            cqi=10,
            logical_channels=[],
            harq=SimpleNamespace(
                pending_retx_bytes=0,
                process_id=1,
                next_new_tx_process_id=1,
                process_pool_size=4,
                ndi_by_process_id={1: 0},
                processes=[],
                harq_flush_count=0,
            ),
        )
        state = SimpleNamespace(
            slot_id=19,
            ues=[ue_poor, ue_good],
            links={
                "ue_poor": SimpleNamespace(cqi=3, ri=1, pmi=0),
                "ue_good": SimpleNamespace(cqi=10, ri=1, pmi=0),
            },
            measurement=SimpleNamespace(
                ue_measurements={
                    "ue_poor": SimpleNamespace(last_measurement_update_slot=11),
                    "ue_good": SimpleNamespace(last_measurement_update_slot=18),
                }
            ),
            control=SimpleNamespace(active_policy_constraints=[]),
            scheduler=SimpleNamespace(
                scheduler_mode="slice_aware",
                scheduled_ue_ids=[],
                grants=[],
                last_decision=None,
                rr_pointer=0,
                priority_boost_by_ue={},
                last_served_slot_by_ue={},
                slice_weight_adjustment={"embb": 1.0},
            ),
        )

        scheduled = scheduler.run(state)

        self.assertEqual(scheduled, ["ue_good"])
        self.assertEqual(state.scheduler.grants[0].ue_id, "ue_good")
        self.assertEqual(
            state.scheduler.last_decision.stage_trace.rejection_reasons_by_ue.get("ue_poor"),
            "new_tx_not_selected_under_slice_priority",
        )

    def test_scheduler_supports_time_first_packing(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=2,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
            packing_strategy="time_first",
        )

        ue_a = SimpleNamespace(
            ue_id="ue_a",
            queue_bytes=500,
            latency_budget_ms=40.0,
            priority=2,
            sinr_db=9.0,
            slice_id="embb",
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=0),
        )
        ue_b = SimpleNamespace(
            ue_id="ue_b",
            queue_bytes=700,
            latency_budget_ms=20.0,
            priority=3,
            sinr_db=10.0,
            slice_id="urllc",
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=1),
        )

        state = SimpleNamespace(
            slot_id=11,
            ues=[ue_a, ue_b],
            control=SimpleNamespace(active_policy_constraints=[]),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                packing_strategy="time_first",
                scheduled_ue_ids=[],
                grants=[],
                last_decision=None,
                rr_pointer=0,
                priority_boost_by_ue={},
                last_served_slot_by_ue={},
            ),
        )

        scheduler.run(state)

        self.assertEqual(state.scheduler.packing_strategy, "time_first")
        self.assertEqual(state.scheduler.grants[0].start_prb, 0)
        self.assertEqual(state.scheduler.grants[1].start_prb, 0)
        self.assertEqual(state.scheduler.grants[0].start_symbol, 0)
        self.assertEqual(
            state.scheduler.grants[1].start_symbol,
            state.scheduler.grants[0].allocated_symbols,
        )
        self.assertEqual(
            state.scheduler.grants[0].resource_block_indices,
            list(range(state.scheduler.grants[0].allocated_prbs)),
        )
        self.assertEqual(
            state.scheduler.grants[1].ofdm_symbol_indices,
            list(
                range(
                    state.scheduler.grants[1].start_symbol,
                    state.scheduler.grants[1].start_symbol + state.scheduler.grants[1].allocated_symbols,
                )
            ),
        )

    def test_scheduler_can_form_spatial_reuse_group(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=12,
            total_symbols_per_slot=6,
            max_ues_per_slot=2,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
            enable_spatial_multiplexing=True,
            max_mu_users_per_group=2,
            min_mu_beam_separation_deg=45.0,
            max_total_spatial_layers=4,
        )

        ue_a = SimpleNamespace(
            ue_id="ue_a",
            queue_bytes=700,
            latency_budget_ms=40.0,
            priority=2,
            sinr_db=11.0,
            slice_id="embb",
            serving_beam_id=0,
            ri=1,
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=0),
        )
        ue_b = SimpleNamespace(
            ue_id="ue_b",
            queue_bytes=650,
            latency_budget_ms=35.0,
            priority=2,
            sinr_db=10.0,
            slice_id="embb",
            serving_beam_id=1,
            ri=1,
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=1),
        )

        state = SimpleNamespace(
            slot_id=13,
            ues=[ue_a, ue_b],
            antenna=SimpleNamespace(gnb_num_tx_ant=4),
            beamforming={
                "ue_a": SimpleNamespace(layer_count=1, precoder_id=0, beam_direction_deg=0.0),
                "ue_b": SimpleNamespace(layer_count=1, precoder_id=2, beam_direction_deg=120.0),
            },
            control=SimpleNamespace(active_policy_constraints=[]),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                packing_strategy="frequency_first",
                enable_spatial_multiplexing=True,
                max_mu_users_per_group=2,
                min_mu_beam_separation_deg=45.0,
                max_total_spatial_layers=4,
                scheduled_ue_ids=[],
                grants=[],
                last_decision=None,
                rr_pointer=0,
                priority_boost_by_ue={},
                last_served_slot_by_ue={},
            ),
        )

        scheduler.run(state)

        self.assertEqual(len(state.scheduler.grants), 2)
        first_grant = state.scheduler.grants[0]
        second_grant = state.scheduler.grants[1]
        self.assertIsNotNone(first_grant.spatial_group_id)
        self.assertEqual(first_grant.spatial_group_id, second_grant.spatial_group_id)
        self.assertEqual(first_grant.start_prb, second_grant.start_prb)
        self.assertEqual(first_grant.start_symbol, second_grant.start_symbol)
        self.assertEqual(first_grant.spatial_group_size, 2)
        self.assertEqual(second_grant.spatial_group_size, 2)
        self.assertGreater(second_grant.spatial_compatibility_score, 0.0)


if __name__ == "__main__":
    unittest.main()

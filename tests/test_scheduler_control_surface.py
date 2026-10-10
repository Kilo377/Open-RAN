import unittest
from types import SimpleNamespace

from ran.scheduler import SimpleScheduler


class TestSchedulerControlSurface(unittest.TestCase):
    def test_slice_weight_adjustment_changes_slice_aware_winner(self) -> None:
        scheduler = SimpleScheduler(
            total_prbs_per_slot=8,
            total_symbols_per_slot=4,
            max_ues_per_slot=1,
            min_prbs_per_grant=2,
            min_symbols_per_grant=2,
        )

        ue_embb = SimpleNamespace(
            ue_id="ue_embb",
            queue_bytes=500,
            latency_budget_ms=80.0,
            priority=1,
            sinr_db=10.0,
            slice_id="embb",
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=0),
        )
        ue_urllc = SimpleNamespace(
            ue_id="ue_urllc",
            queue_bytes=500,
            latency_budget_ms=80.0,
            priority=1,
            sinr_db=10.0,
            slice_id="urllc",
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=1),
        )

        state = SimpleNamespace(
            slot_id=11,
            ues=[ue_embb, ue_urllc],
            control=SimpleNamespace(active_policy_constraints=[]),
            scheduler=SimpleNamespace(
                scheduler_mode="slice_aware",
                scheduler_policy_hint="slice_bias",
                latency_bias=None,
                slice_weight_adjustment={"embb": 2.0, "urllc": 0.5},
                scheduled_ue_ids=[],
                grants=[],
                last_decision=None,
                rr_pointer=0,
                priority_boost_by_ue={},
                last_served_slot_by_ue={},
            ),
        )

        scheduled = scheduler.run(state)

        self.assertEqual(scheduled, ["ue_embb"])
        self.assertEqual(state.scheduler.grants[0].ue_id, "ue_embb")


if __name__ == "__main__":
    unittest.main()

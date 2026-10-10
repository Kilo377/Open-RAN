import unittest
from types import SimpleNamespace


from models.action import PolicyConstraint
from ran.scheduler import SimpleScheduler


class TestMinUeServiceIntervalConstraint(unittest.TestCase):
    def test_constraint_boosts_target_ue(self) -> None:
        scheduler = SimpleScheduler()

        ue_a = SimpleNamespace(
            ue_id="ue_a",
            queue_bytes=100,
            latency_budget_ms=1.0,
            priority=1,
            sinr_db=10.0,
            slice_id="embb",
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=0),
        )
        ue_b = SimpleNamespace(
            ue_id="ue_b",
            queue_bytes=100,
            latency_budget_ms=100.0,
            priority=1,
            sinr_db=10.0,
            slice_id="embb",
            harq=SimpleNamespace(pending_retx_bytes=0, process_id=0),
        )

        constraint = PolicyConstraint(
            constraint_type="min_ue_service_interval",
            scope="ue",
            target_id="ue_b",
            params={"max_gap_slots": 5},
            valid_from_slot=0,
            valid_until_slot=None,
            source_name="test",
        )

        state = SimpleNamespace(
            slot_id=10,
            ues=[ue_a, ue_b],
            control=SimpleNamespace(active_policy_constraints=[constraint]),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                scheduled_ue_ids=[],
                grants=[],
                last_decision=None,
                rr_pointer=0,
                priority_boost_by_ue={},
                last_served_slot_by_ue={"ue_a": 9, "ue_b": 0},
            ),
        )

        scheduled = scheduler.run(state)
        self.assertEqual(scheduled, ["ue_b", "ue_a"])
        self.assertEqual(state.scheduler.last_served_slot_by_ue.get("ue_b"), 10)
        self.assertTrue(any(grant.ue_id == "ue_b" for grant in state.scheduler.grants))


if __name__ == "__main__":
    unittest.main()

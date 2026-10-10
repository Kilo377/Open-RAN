import unittest

from models.action import ConfigAction, ControlDecision
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.controller_manager import ControllerManager
from ric.interfaces.e2_like import E2LikeInterface
from ric.near_rt.xapp_base import XAppBase


class DummyXApp(XAppBase):
    def __init__(
        self,
        *,
        name: str,
        priority: int,
        controlled_fields: list[str],
        scheduler_mode: str,
    ) -> None:
        super().__init__(name=name, priority=priority, controlled_fields=controlled_fields)
        self._scheduler_mode = scheduler_mode

    def run(self, observation: Observation, policy_snapshot: PolicySnapshot) -> ControlDecision:
        _ = observation
        _ = policy_snapshot
        return ControlDecision(
            config_action=ConfigAction(
                scheduler_mode=self._scheduler_mode,
                source_name=self.name,
            )
        )


class TestControlMetadataAndTrace(unittest.TestCase):
    def test_controller_manager_records_trace_and_stamps_metadata(self) -> None:
        xapp_winner = DummyXApp(
            name="winner_xapp",
            priority=10,
            controlled_fields=["scheduler_mode"],
            scheduler_mode="latency_aware",
        )
        xapp_dropped = DummyXApp(
            name="dropped_xapp",
            priority=5,
            controlled_fields=[],
            scheduler_mode="slice_aware",
        )

        manager = ControllerManager(
            a1_interfaces=[],
            e2_interfaces=[E2LikeInterface(xapp_winner), E2LikeInterface(xapp_dropped)],
            xapp_interval_slots=1,
            rapp_interval_slots=100000,
        )

        observation = Observation(
            slot_id=1,
            avg_sinr_db=None,
            avg_bler=None,
            total_queue_bytes=0,
            total_throughput_mbps=0.0,
            cell_id="cell_0",
            current_codebook_id=0,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="round_robin",
            scheduled_ue_ids=[],
            num_scheduled_ues=0,
            ue_observations=[],
        )

        decision = manager.update(slot_id=1, observation=observation)

        self.assertEqual(decision.metadata.source, "controller_manager")
        self.assertEqual(decision.metadata.issued_at_slot, 1)

        self.assertEqual(len(decision.trace.received_config_actions), 2)
        sources = {a.metadata.source for a in decision.trace.received_config_actions}
        self.assertEqual(sources, {"winner_xapp", "dropped_xapp"})
        for a in decision.trace.received_config_actions:
            self.assertEqual(a.metadata.issued_at_slot, 1)

        self.assertIn("scheduler_mode", decision.trace.selected_config_action_by_field)
        winner = decision.trace.selected_config_action_by_field["scheduler_mode"]
        self.assertEqual(winner.source, "winner_xapp")
        self.assertEqual(winner.value, "latency_aware")

        dropped = [
            d
            for d in decision.trace.dropped_items_with_reason
            if d.source == "dropped_xapp" and d.field_name == "scheduler_mode"
        ]
        self.assertTrue(dropped)


if __name__ == "__main__":
    unittest.main()

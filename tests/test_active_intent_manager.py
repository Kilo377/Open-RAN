import unittest

from ric.non_rt.active_intent_manager import ActiveIntentEvent, ActiveIntentManager


class TestActiveIntentManager(unittest.TestCase):
    def test_active_intent_is_piecewise_constant_without_new_events(self) -> None:
        manager = ActiveIntentManager(
            initial_intent_text="提升10%吞吐",
            initial_epoch=1,
            initial_change_reason="initial_test_intent",
            source_name="test_operator",
        )

        first_epoch_intent, first_changed = manager.get_epoch_state(1)
        second_epoch_intent, second_changed = manager.get_epoch_state(2)
        third_epoch_intent, third_changed = manager.get_epoch_state(3)

        self.assertIsNotNone(first_epoch_intent)
        self.assertTrue(first_changed)
        self.assertFalse(second_changed)
        self.assertFalse(third_changed)
        self.assertEqual(first_epoch_intent.intent_id, second_epoch_intent.intent_id)
        self.assertEqual(second_epoch_intent.intent_id, third_epoch_intent.intent_id)
        self.assertEqual(first_epoch_intent.last_update_epoch, 1)
        self.assertEqual(second_epoch_intent.last_update_epoch, 1)
        self.assertEqual(third_epoch_intent.last_change_reason, "initial_test_intent")

    def test_active_intent_changes_only_on_registered_event(self) -> None:
        manager = ActiveIntentManager(
            initial_intent_text="提升10%吞吐",
            initial_epoch=1,
            initial_change_reason="initial_test_intent",
            source_name="test_operator",
        )
        manager.update_intent(
            ActiveIntentEvent(
                epoch_id=4,
                intent_text="降低URLLC时延",
                change_reason="operator_policy_change",
                source_name="test_operator",
            )
        )

        third_epoch_intent, third_changed = manager.get_epoch_state(3)
        fourth_epoch_intent, fourth_changed = manager.get_epoch_state(4)

        self.assertFalse(third_changed)
        self.assertNotEqual(third_epoch_intent.intent_id, fourth_epoch_intent.intent_id)
        self.assertTrue(fourth_changed)
        self.assertEqual(fourth_epoch_intent.last_update_epoch, 4)
        self.assertEqual(fourth_epoch_intent.last_change_reason, "operator_policy_change")


if __name__ == "__main__":
    unittest.main()

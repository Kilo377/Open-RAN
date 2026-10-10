import unittest

from models.observation import Observation
from ric.controller_manager import ControllerManager
from ric.interfaces.a1_like import A1LikeInterface
from ric.interfaces.e2_like import E2LikeInterface
from ric.near_rt.beam_xapp import BeamXApp
from ric.non_rt.codebook_rapp import CodebookRApp


class TestOranLifecycleResources(unittest.TestCase):
    def test_controller_manager_builds_a1_and_e2_lifecycle_resources(self) -> None:
        manager = ControllerManager(
            a1_interfaces=[A1LikeInterface(CodebookRApp())],
            e2_interfaces=[
                E2LikeInterface(
                    BeamXApp(
                        name="beam_xapp",
                        priority=20,
                        sinr_threshold_db=12.0,
                        queue_threshold_bytes=2000,
                    )
                )
            ],
            xapp_interval_slots=1,
            rapp_interval_slots=1,
        )

        observation = Observation(
            slot_id=1,
            avg_sinr_db=8.0,
            avg_bler=0.15,
            total_queue_bytes=6000,
            total_throughput_mbps=4.0,
            cell_id="cell_0",
            current_codebook_id=0,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="round_robin",
            scheduled_ue_ids=[],
            num_scheduled_ues=1,
            ue_observations=[],
            runtime_mode="multi_cell_ready",
        )

        decision = manager.update(slot_id=1, observation=observation)
        resources = decision.oran_lifecycle_resources

        self.assertGreaterEqual(len(resources.a1_policy_types), 1)
        self.assertGreaterEqual(len(resources.a1_policy_instances), 1)
        self.assertGreaterEqual(len(resources.a1_policy_statuses), 1)
        self.assertGreaterEqual(len(resources.a1_ei_types), 1)
        self.assertGreaterEqual(len(resources.a1_ei_jobs), 1)
        self.assertEqual(len(resources.e2_subscriptions), 1)
        self.assertEqual(resources.e2_subscriptions[0].subscriber_name, "beam_xapp")
        self.assertEqual(len(resources.e2_indications), 1)
        self.assertEqual(len(resources.e2_control_requests), 1)
        self.assertIsNotNone(decision.config_action.control_request_id)
        self.assertEqual(
            resources.e2_control_requests[0].request_id,
            decision.config_action.control_request_id,
        )
        self.assertEqual(resources.e2_control_requests[0].status, "issued")


if __name__ == "__main__":
    unittest.main()

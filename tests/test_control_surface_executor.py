import unittest
from types import SimpleNamespace
from typing import Any, cast

from models.action import ConfigAction
from ran.local_executor import LocalExecutor


class TestControlSurfaceExecutor(unittest.TestCase):
    def test_executor_maps_control_surface_to_runtime_state(self) -> None:
        executor = LocalExecutor()
        state = SimpleNamespace(
            slot_id=5,
            cell=SimpleNamespace(
                tx_power_dbm=30.0,
                current_codebook_id=0,
                current_beam_profile_id=0,
                beam_candidate_preference="balanced",
                power_cap_dbm=None,
            ),
            scheduler=SimpleNamespace(
                scheduler_mode="round_robin",
                scheduler_policy_hint="balanced",
                latency_bias=None,
                slice_weight_adjustment={},
            ),
        )

        target_cell_ids = executor.apply_action(
            cast(Any, state),
            ConfigAction(
                scheduler_policy_hint="slice_bias",
                slice_weight_adjustment={"URLLC": 1.4, "eMBB": 0.9},
                beam_candidate_preference="coverage",
                power_cap_dbm=31.0,
            ),
        )

        self.assertEqual(target_cell_ids, ["cell_0"])
        self.assertEqual(state.scheduler.scheduler_mode, "slice_aware")
        self.assertEqual(state.scheduler.scheduler_policy_hint, "slice_bias")
        self.assertEqual(
            state.scheduler.slice_weight_adjustment,
            {"urllc": 1.4, "embb": 0.9},
        )
        self.assertIsNone(state.scheduler.latency_bias)
        self.assertEqual(state.cell.beam_candidate_preference, "coverage")
        self.assertEqual(state.cell.current_codebook_id, 2)
        self.assertEqual(state.cell.current_beam_profile_id, 1)
        self.assertEqual(state.cell.power_cap_dbm, 31.0)
        self.assertEqual(state.cell.tx_power_dbm, 31.0)

    def test_executor_clears_incompatible_surface_state_when_hint_changes(self) -> None:
        executor = LocalExecutor()
        state = SimpleNamespace(
            slot_id=6,
            cell=SimpleNamespace(
                tx_power_dbm=31.0,
                current_codebook_id=2,
                current_beam_profile_id=1,
                beam_candidate_preference="coverage",
                power_cap_dbm=31.0,
            ),
            scheduler=SimpleNamespace(
                scheduler_mode="slice_aware",
                scheduler_policy_hint="slice_bias",
                latency_bias=None,
                slice_weight_adjustment={"urllc": 1.3},
            ),
        )

        target_cell_ids = executor.apply_action(
            cast(Any, state),
            ConfigAction(
                scheduler_policy_hint="latency_bias",
                latency_bias=1.6,
            ),
        )

        self.assertEqual(target_cell_ids, ["cell_0"])
        self.assertEqual(state.scheduler.scheduler_mode, "latency_aware")
        self.assertEqual(state.scheduler.scheduler_policy_hint, "latency_bias")
        self.assertEqual(state.scheduler.latency_bias, 1.6)
        self.assertEqual(state.scheduler.slice_weight_adjustment, {})

    def test_executor_supports_target_cell_scoped_action(self) -> None:
        executor = LocalExecutor()
        primary_runtime = SimpleNamespace(
            cell=SimpleNamespace(
                cell_id="cell_0",
                tx_power_dbm=30.0,
                current_codebook_id=0,
                current_beam_profile_id=0,
                beam_candidate_preference="balanced",
                power_cap_dbm=None,
                traffic_steering_preference="balanced",
                mobility_bias_db=0.0,
            ),
            scheduler=SimpleNamespace(
                scheduler_mode="round_robin",
                scheduler_policy_hint="balanced",
                latency_bias=None,
                slice_weight_adjustment={},
                scheduled_ue_ids=[],
            ),
        )
        target_runtime = SimpleNamespace(
            cell=SimpleNamespace(
                cell_id="cell_1",
                tx_power_dbm=28.0,
                current_codebook_id=0,
                current_beam_profile_id=0,
                beam_candidate_preference="balanced",
                power_cap_dbm=None,
                traffic_steering_preference="balanced",
                mobility_bias_db=0.0,
            ),
            scheduler=SimpleNamespace(
                scheduler_mode="round_robin",
                scheduler_policy_hint="balanced",
                latency_bias=None,
                slice_weight_adjustment={},
                scheduled_ue_ids=[],
            ),
        )
        state = SimpleNamespace(
            slot_id=7,
            cell=primary_runtime.cell,
            scheduler=primary_runtime.scheduler,
            managed_cells={"cell_0": primary_runtime, "cell_1": target_runtime},
        )

        target_cell_ids = executor.apply_action(
            cast(Any, state),
            ConfigAction(
                target_cell_id="cell_1",
                scheduler_policy_hint="latency_bias",
                latency_bias=1.4,
                beam_candidate_preference="coverage",
                power_cap_dbm=27.0,
            ),
        )

        self.assertEqual(target_cell_ids, ["cell_1"])
        self.assertEqual(state.scheduler.scheduler_mode, "round_robin")
        self.assertEqual(target_runtime.scheduler.scheduler_mode, "latency_aware")
        self.assertEqual(target_runtime.scheduler.latency_bias, 1.4)
        self.assertEqual(target_runtime.cell.current_codebook_id, 2)
        self.assertEqual(target_runtime.cell.current_beam_profile_id, 1)
        self.assertEqual(target_runtime.cell.tx_power_dbm, 27.0)
        self.assertEqual(state.cell.current_codebook_id, 0)


if __name__ == "__main__":
    unittest.main()

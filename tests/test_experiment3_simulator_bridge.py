from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import unittest

from experiments.experiment3_simulator_bridge import run_experiment3_simulator_bridge


class Experiment3SimulatorBridgeTest(unittest.TestCase):
    def test_bridge_runs_actual_simulator_and_preserves_full_method_safety(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "bridge"
            result = run_experiment3_simulator_bridge(
                output_dir,
                seeds=[1],
                training_seeds=[1001],
                episode_length=40,
                training_episodes=3,
                method_names=[
                    "no_near_rt_tuning",
                    "ddpg_projection_only",
                    "full_contract_constrained_ddpg",
                ],
                event_names=["traffic_burst"],
                severity_names=["low"],
            )

            self.assertEqual(len(result["episode_metrics"]), 3)
            for artifact_name in (
                "time_series.csv",
                "episode_metrics.csv",
                "summary.csv",
                "event_summary.csv",
                "recovery_curve.csv",
                "training_diagnostics.csv",
                "run_plan.json",
            ):
                self.assertTrue((output_dir / artifact_name).exists())

            with (output_dir / "run_plan.json").open("r", encoding="utf-8") as json_file:
                run_plan = json.load(json_file)
            self.assertEqual(run_plan["simulator"], "core.simulator.Simulator")
            self.assertEqual(run_plan["workload_profile"]["ue_count"], 15)

            with (output_dir / "time_series.csv").open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertIn("action_scheduler_priority", rows[0])
            self.assertIn("mean_estimated_queue_delay_ms", rows[0])

            full_rows = [
                row for row in rows if row["method"] == "full_contract_constrained_ddpg"
            ]
            projection_rows = [
                row for row in rows if row["method"] == "ddpg_projection_only"
            ]
            self.assertTrue(
                all(float(row["raw_action_violation"]) == 0.0 for row in full_rows)
            )
            self.assertTrue(
                all(float(row["executed_action_violation"]) == 0.0 for row in full_rows)
            )
            self.assertGreater(
                sum(float(row["raw_action_violation"]) for row in projection_rows),
                0.0,
            )
            self.assertTrue(
                all(float(row["executed_action_violation"]) == 0.0 for row in projection_rows)
            )


if __name__ == "__main__":
    unittest.main()

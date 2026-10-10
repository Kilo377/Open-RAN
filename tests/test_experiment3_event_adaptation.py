from __future__ import annotations

import csv
import importlib.util
import tempfile
from pathlib import Path
import unittest

from experiments.event_driven_contract_adaptation import (
    _heldout_evaluation_cases,
    run_event_driven_contract_adaptation_experiment,
)


class Experiment3EventAdaptationTest(unittest.TestCase):
    def test_heldout_protocol_separates_iid_and_ood_cases(self) -> None:
        cases = _heldout_evaluation_cases(test_seeds=[1, 2], validation_seeds=[2001])
        split_counts: dict[str, int] = {}
        for split_name, seed, event, severity_name, _ in cases:
            split_counts[split_name] = split_counts.get(split_name, 0) + 1
            if split_name == "validation":
                self.assertEqual(seed, 2001)
                self.assertIn(event.event_id, {"E1", "E2", "E3", "E4"})
                self.assertEqual(severity_name, "medium")
            elif split_name == "test_iid":
                self.assertIn(seed, {1, 2})
                self.assertIn(event.event_id, {"E1", "E2", "E3", "E4"})
                self.assertIn(severity_name, {"low", "medium"})
            elif split_name == "test_ood_severity":
                self.assertIn(event.event_id, {"E1", "E2", "E3", "E4"})
                self.assertEqual(severity_name, "high")
            elif split_name == "test_ood_event":
                self.assertIn(event.event_id, {"E5", "E6"})

        self.assertEqual(split_counts["validation"], 4)
        self.assertEqual(split_counts["test_iid"], 16)
        self.assertEqual(split_counts["test_ood_severity"], 8)
        self.assertEqual(split_counts["test_ood_event"], 12)

    def test_runner_writes_contract_adaptation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "exp3"
            has_matplotlib = importlib.util.find_spec("matplotlib") is not None
            result = run_event_driven_contract_adaptation_experiment(
                output_dir,
                seeds=[1],
                episode_length=60,
                event_time_fraction=0.40,
                shock_fraction=0.10,
                recovery_window=4,
                method_suite="main",
                training_episodes=6,
                write_figures=has_matplotlib,
            )

            self.assertTrue((output_dir / "run_plan.json").exists())
            self.assertTrue((output_dir / "time_series.csv").exists())
            self.assertTrue((output_dir / "episode_metrics.csv").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "training_diagnostics.csv").exists())
            if has_matplotlib:
                self.assertTrue((output_dir / "figures" / "figure1_event_kpi_timeseries.png").exists())
            self.assertEqual(len(result["episode_metrics"]), 6 * 3 * 6)

            with (output_dir / "episode_metrics.csv").open("r", encoding="utf-8", newline="") as csv_file:
                episode_rows = list(csv.DictReader(csv_file))
            full_rows = [row for row in episode_rows if row["method"] == "full_contract_constrained_ddpg"]
            unconstrained_rows = [row for row in episode_rows if row["method"] == "unconstrained_ddpg"]
            self.assertGreater(len(full_rows), 0)
            self.assertGreater(len(unconstrained_rows), 0)
            self.assertTrue(all(float(row["executed_action_violation_rate"]) == 0.0 for row in full_rows))
            self.assertGreater(
                sum(float(row["raw_action_violation_rate"]) for row in unconstrained_rows),
                0.0,
            )
            with (output_dir / "training_diagnostics.csv").open("r", encoding="utf-8", newline="") as csv_file:
                training_rows = list(csv.DictReader(csv_file))
            self.assertIn("neural_algorithm", training_rows[0])
            self.assertGreater(
                len([row for row in training_rows if row["method"] == "full_contract_constrained_ddpg"]),
                0,
            )


if __name__ == "__main__":
    unittest.main()

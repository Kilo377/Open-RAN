import tempfile
import unittest
from pathlib import Path

from experiments.conflict_taxonomy_eval import (
    evaluate_conflict_taxonomy_modes,
    write_conflict_taxonomy_outputs,
)


class TestConflictTaxonomyEval(unittest.TestCase):
    def test_typed_taxonomy_balances_safety_and_over_rejection(self) -> None:
        _, summary = evaluate_conflict_taxonomy_modes(
            modes=[
                "no_conflict_check",
                "binary_conflict_flag",
                "pacifista_like_profiler",
                "typed_taxonomy",
            ]
        )

        typed = summary["typed_taxonomy"]
        no_check = summary["no_conflict_check"]
        binary = summary["binary_conflict_flag"]
        pacifista = summary["pacifista_like_profiler"]

        self.assertEqual(typed["case_count"], 10)
        self.assertEqual(typed["hard_conflict_escape_rate"], 0.0)
        self.assertEqual(typed["policy_conflict_escape_rate"], 0.0)
        self.assertEqual(typed["risk_conflict_realization_rate"], 0.0)
        self.assertGreater(typed["soft_conflict_arbitration_success_rate"], 0.0)
        self.assertGreater(no_check["unsafe_plan_rate"], typed["unsafe_plan_rate"])
        self.assertGreater(binary["over_rejection_rate"], typed["over_rejection_rate"])
        self.assertGreater(pacifista["policy_conflict_escape_rate"], typed["policy_conflict_escape_rate"])

    def test_conflict_taxonomy_outputs_are_written(self) -> None:
        results, summary = evaluate_conflict_taxonomy_modes(modes=["typed_taxonomy"])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_conflict_taxonomy_outputs(
                output_dir=output_dir,
                results=results,
                summary=summary,
                config={"test": True},
            )

            self.assertTrue((output_dir / "case_results.csv").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "config_snapshot.json").exists())


if __name__ == "__main__":
    unittest.main()

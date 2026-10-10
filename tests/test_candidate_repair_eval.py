import tempfile
import unittest
from pathlib import Path

from experiments.candidate_repair_eval import (
    evaluate_candidate_repair_modes,
    write_candidate_repair_outputs,
)


class TestCandidateRepairEval(unittest.TestCase):
    def test_deterministic_repair_improves_executability_without_hard_escape(self) -> None:
        _, summary = evaluate_candidate_repair_modes(
            modes=["deterministic", "reject_only", "schema_only", "no_projection"]
        )

        deterministic = summary["deterministic"]
        reject_only = summary["reject_only"]
        schema_only = summary["schema_only"]
        no_projection = summary["no_projection"]

        self.assertEqual(deterministic["candidate_count"], 10)
        self.assertEqual(deterministic["hard_conflict_escape_rate"], 0.0)
        self.assertGreater(
            deterministic["feasible_candidate_rate"],
            reject_only["feasible_candidate_rate"],
        )
        self.assertGreater(deterministic["repair_success_rate"], 0.0)
        self.assertGreater(schema_only["hard_conflict_escape_rate"], 0.0)
        self.assertGreater(no_projection["unknown_xapp_escape_rate"], 0.0)

    def test_candidate_repair_outputs_are_written(self) -> None:
        results, summary = evaluate_candidate_repair_modes(modes=["deterministic"])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_candidate_repair_outputs(
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

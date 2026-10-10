from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from experiments.paper_sandbox_eval import run_selected_experiments


class PaperSandboxEvalTest(unittest.TestCase):
    def test_selected_experiments_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "paper_eval"
            outputs = run_selected_experiments(
                root_output_dir=output_dir,
                seeds=[1],
                experiments=["2", "5", "6", "7", "8", "9"],
                llm_provider="deepseek",
                llm_model="deepseek-v4-pro",
            )
            self.assertIn("Experiment 2: failed-case memory", outputs)
            self.assertTrue((output_dir / "paper_evaluation_artifact_index.csv").exists())
            self.assertTrue((output_dir / "exp2_failed_case_memory" / "summary.csv").exists())
            self.assertTrue((output_dir / "exp5_near_rt_tuning" / "figures" / "near_rt_latency_timeseries.png").exists())
            self.assertTrue((output_dir / "exp6_llm_safe_pipeline" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()

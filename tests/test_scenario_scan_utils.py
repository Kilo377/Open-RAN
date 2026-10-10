from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.scenario_scan_utils import aggregate_case_records, build_case_summary_record, write_matrix_summary
from models.scenario_matrix import ScenarioMatrixCase


class ScenarioScanUtilsTest(unittest.TestCase):
    def test_build_case_summary_record_flattens_nested_summary(self) -> None:
        case = ScenarioMatrixCase(
            case_id="case_a",
            description="demo",
            scenario_name="baseline_light_load_static",
            tags=["sanity"],
            coordinates={"obs_delay_slots": 1},
        )
        final_summary = {
            "final_true_metrics": {
                "queue_bytes": 100,
                "throughput_mbps": 12.5,
            },
            "final_observable_metrics": {
                "bler": 0.1,
            },
        }

        record = build_case_summary_record(case=case, final_summary=final_summary, output_dir="outputs/case_a")

        self.assertEqual(record["case_id"], "case_a")
        self.assertIn("summary_final_true_metrics_queue_bytes", record)
        self.assertEqual(record["summary_final_true_metrics_throughput_mbps"], 12.5)
        self.assertEqual(record["summary_final_observable_metrics_bler"], 0.1)

    def test_aggregate_case_records_computes_means(self) -> None:
        records = [
            {
                "case_id": "a",
                "scenario_name": "benchmark_nominal_cluster",
                "summary_final_true_metrics_throughput_mbps": 10.0,
                "summary_experimental_metrics_intent_loss_mean_normalized": 0.10,
            },
            {
                "case_id": "b",
                "scenario_name": "benchmark_edge_coverage",
                "summary_final_true_metrics_throughput_mbps": 14.0,
                "summary_experimental_metrics_intent_loss_mean_normalized": 0.25,
            },
        ]

        aggregates = aggregate_case_records(records)

        self.assertEqual(aggregates["case_count"], 2)
        self.assertEqual(aggregates["summary_final_true_metrics_throughput_mbps_mean"], 12.0)
        self.assertEqual(aggregates["summary_final_true_metrics_throughput_mbps_min"], 10.0)
        self.assertEqual(aggregates["summary_final_true_metrics_throughput_mbps_max"], 14.0)
        self.assertEqual(aggregates["reference_scenario_name"], "benchmark_nominal_cluster")
        self.assertAlmostEqual(
            aggregates["cross_scenario_generalization_gap"]["mean_gap"],
            0.15,
        )

    def test_write_matrix_summary_creates_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_records = [
                {
                    "case_id": "case_a",
                    "summary_final_true_metrics_queue_bytes": 100,
                }
            ]

            write_matrix_summary(
                output_dir=tmp_dir,
                matrix_name="scenario_catalog",
                matrix_description="demo",
                case_records=case_records,
            )

            self.assertTrue((Path(tmp_dir) / "matrix_summary.json").exists())
            self.assertTrue((Path(tmp_dir) / "matrix_cases.csv").exists())


if __name__ == "__main__":
    unittest.main()

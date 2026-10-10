import csv
import tempfile
import unittest
from pathlib import Path

from core.result_writer import ResultWriter


class TestResultWriterLayeredCsv(unittest.TestCase):
    def test_write_slot_metrics_flattens_layered_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ResultWriter(output_dir=temp_dir)
            writer.write_slot_metrics(
                [
                    {
                        "slot_id": 1,
                        "sim_time_ms": 1.0,
                        "true_metrics": {"queue_bytes": 100, "new_tx_bytes_this_slot": 80},
                        "observable_metrics": {"queue_bytes": 120, "sinr_db": 5.5},
                        "measurement_trace": {"age_slots": 1, "sinr_window_slots": 8},
                        "control_trace": {"decision_id": "decision:1"},
                        "scheduler_trace": {
                            "demand_count": 2,
                            "granted_ue_ids": ["ue_0"],
                        },
                        "phy_comparison_trace": {
                            "mean_abs_sinr_gap_db": 1.25,
                            "ack_match_rate": 1.0,
                            "per_ue_diffs": [{"ue_id": "ue_0", "post_eq_sinr_gap_db": 0.7}],
                        },
                    }
                ]
            )

            output_path = Path(temp_dir) / "slot_metrics.csv"
            with output_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])

            self.assertEqual(len(rows), 1)
            self.assertIn("true_queue_bytes", fieldnames)
            self.assertIn("observable_queue_bytes", fieldnames)
            self.assertIn("measurement_age_slots", fieldnames)
            self.assertIn("control_decision_id", fieldnames)
            self.assertIn("scheduler_demand_count", fieldnames)
            self.assertIn("phy_compare_mean_abs_sinr_gap_db", fieldnames)
            self.assertEqual(rows[0]["true_queue_bytes"], "100")
            self.assertEqual(rows[0]["observable_queue_bytes"], "120")
            self.assertEqual(rows[0]["measurement_age_slots"], "1")
            self.assertEqual(rows[0]["control_decision_id"], "decision:1")
            self.assertEqual(rows[0]["scheduler_demand_count"], "2")
            self.assertEqual(rows[0]["phy_compare_mean_abs_sinr_gap_db"], "1.25")
            self.assertIn("phy_compare_per_ue_diffs", fieldnames)
            self.assertEqual(
                rows[0]["phy_compare_per_ue_diffs"],
                '[{"ue_id": "ue_0", "post_eq_sinr_gap_db": 0.7}]',
            )

    def test_write_snapshots_outputs_layered_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ResultWriter(output_dir=temp_dir)
            writer.write_snapshots(
                [
                    {
                        "slot_id": 1,
                        "true_metrics": {"queue_bytes": 100},
                        "observable_metrics": {"queue_bytes": 120},
                        "ues": [
                            {
                                "ue_id": "ue_0",
                                "true_state": {"queue_bytes": 100},
                                "observable_state": {"queue_bytes": 120},
                            }
                        ],
                    }
                ]
            )

            output_path = Path(temp_dir) / "slot_snapshots.json"
            self.assertTrue(output_path.exists())
            self.assertIn('"observable_state"', output_path.read_text(encoding="utf-8"))

    def test_write_orchestration_epoch_history_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ResultWriter(output_dir=temp_dir)
            writer.write_orchestration_epoch_history(
                [
                    {
                        "epoch_id": 1,
                        "active_intent_id": "intent:test",
                        "selected_xapps": ["xapp_a"],
                    }
                ]
            )

            output_path = Path(temp_dir) / "orchestration_epoch_history.json"
            self.assertTrue(output_path.exists())
            self.assertIn('"active_intent_id"', output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

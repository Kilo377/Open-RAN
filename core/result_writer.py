"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: result_writer.py
Description: This file defines helper functions for standardized result
recording. It supports writing slot-level metrics to CSV files and final
simulation summaries to JSON files.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Any


class ResultWriter:
    """
    Writes simulator outputs into structured files.
    INPUT
    output_dir: The output directory path.
    OUTPUT
    A result writer object that stores CSV and JSON results.
    EXAMPLE OUTPUT
    writer = ResultWriter(output_dir="outputs")
    writer.write_slot_metrics(slot_records)
    writer.write_final_summary(summary)
    """

    def __init__(self, output_dir: str) -> None:
        """
        Initializes the result writer and ensures the output directory exists.
        INPUT
        output_dir: The output directory path.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        self.output_dir = Path("outputs")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _flatten_record(
        self,
        record: Dict[str, Any],
        *,
        prefix: str = "",
    ) -> Dict[str, Any]:
        flattened: Dict[str, Any] = {}
        special_prefix_map = {
            "true_metrics": "true_",
            "observable_metrics": "observable_",
            "measurement_trace": "measurement_",
            "control_trace": "control_",
            "scheduler_trace": "scheduler_",
            "phy_comparison_trace": "phy_compare_",
        }

        for key, value in record.items():
            child_prefix = special_prefix_map.get(key, f"{prefix}{key}_")
            if prefix == "" and key in special_prefix_map:
                child_prefix = special_prefix_map[key]

            if isinstance(value, dict):
                flattened.update(self._flatten_record(value, prefix=child_prefix))
            elif isinstance(value, (list, tuple)):
                flattened[f"{prefix}{key}"] = json.dumps(value, ensure_ascii=False)
            else:
                flattened[f"{prefix}{key}"] = value

        return flattened

    def write_slot_metrics(self, slot_records: List[Dict[str, Any]]) -> None:
        """
        Writes slot-level metrics to a CSV file.
        INPUT
        slot_records: A list of dictionaries containing slot-level metrics.
        OUTPUT
        None. A CSV file is written to disk.
        EXAMPLE OUTPUT
        outputs/slot_metrics.csv is created.
        """
        output_path = self.output_dir / "slot_metrics.csv"

        if len(slot_records) == 0:
            return

        flattened_records = [self._flatten_record(record) for record in slot_records]
        field_names: list[str] = []
        for record in flattened_records:
            for field_name in record.keys():
                if field_name not in field_names:
                    field_names.append(field_name)

        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(flattened_records)

    def write_final_summary(self, summary: Dict[str, Any]) -> None:
        """
        Writes final simulation summary data to a JSON file.
        INPUT
        summary: A dictionary containing final summary metrics.
        OUTPUT
        None. A JSON file is written to disk.
        EXAMPLE OUTPUT
        outputs/final_summary.json is created.
        """
        output_path = self.output_dir / "final_summary.json"

        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(summary, json_file, indent=4, ensure_ascii=False)

    def write_snapshots(self, snapshots: List[Dict[str, Any]]) -> None:
        """
        Writes layered slot snapshots to a JSON file.
        INPUT
        snapshots: A list of layered per-slot snapshot dictionaries.
        OUTPUT
        None. A JSON file is written to disk.
        EXAMPLE OUTPUT
        outputs/slot_snapshots.json is created.
        """
        if str(os.environ.get("ORAN_WRITE_SLOT_SNAPSHOTS", "1")).strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return
        output_path = self.output_dir / "slot_snapshots.json"

        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(snapshots, json_file, indent=4, ensure_ascii=False)

    def write_orchestration_epoch_history(
        self,
        orchestration_epoch_history: List[Dict[str, Any]],
    ) -> None:
        """
        Writes Non-RT orchestration epoch history to a JSON file.
        INPUT
        orchestration_epoch_history: A list of epoch-level orchestration dictionaries.
        OUTPUT
        None. A JSON file is written to disk.
        EXAMPLE OUTPUT
        outputs/orchestration_epoch_history.json is created.
        """
        output_path = self.output_dir / "orchestration_epoch_history.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(orchestration_epoch_history, json_file, indent=4, ensure_ascii=False)

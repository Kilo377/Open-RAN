"""Merge JSON-backed orchestration memory stores into one cumulative store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge orchestration memory JSON stores.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--max-success", type=int, default=256)
    parser.add_argument("--max-failure", type=int, default=256)
    parser.add_argument("--max-reflections", type=int, default=256)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _merge_items(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    anonymous_index = 0
    for item in list(existing or []) + list(incoming or []):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("case_id", "") or item.get("outcome_summary", "") or "").strip()
        if item_id == "":
            item_id = f"anonymous:{anonymous_index}"
            anonymous_index += 1
        by_id[item_id] = dict(item)
    return list(by_id.values())


def _trim(items: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    limit = max(1, int(max_items))
    return list(items or [])[-limit:]


def main() -> None:
    args = _parse_args()
    output_path = Path(str(args.output)).resolve()
    success_cases: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    reflection_notes: list[dict[str, Any]] = []
    used_inputs: list[str] = []
    for raw_input in list(args.inputs or []):
        path = Path(str(raw_input)).resolve()
        if not path.exists():
            continue
        payload = _load(path)
        success_cases = _merge_items(success_cases, list(payload.get("success_cases", []) or []))
        failure_cases = _merge_items(failure_cases, list(payload.get("failure_cases", []) or []))
        reflection_notes = _merge_items(reflection_notes, list(payload.get("reflection_notes", []) or []))
        used_inputs.append(str(path))

    output = {
        "schema_version": 1,
        "success_cases": _trim(success_cases, args.max_success),
        "failure_cases": _trim(failure_cases, args.max_failure),
        "reflection_notes": _trim(reflection_notes, args.max_reflections),
        "merge_metadata": {
            "input_count": len(used_inputs),
            "inputs": used_inputs,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Wrote {output_path} "
        f"success={len(output['success_cases'])} "
        f"failure={len(output['failure_cases'])} "
        f"reflections={len(output['reflection_notes'])}"
    )


if __name__ == "__main__":
    main()

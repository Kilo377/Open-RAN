"""Run Experiment 1 chunk configs sequentially or with limited parallelism."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "configs" / "exp1_chunks" / "chunk_manifest.csv"
DEFAULT_LOG_DIR = Path("E:/oran_slot_sim_outputs/paper_eval/run_logs/exp1_chunks")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1 chunk queue.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--chunk-id", action="append", default=[], help="Specific chunk_id to run.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N selected chunks.")
    parser.add_argument("--limit", type=int, default=0, help="Run at most N chunks.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel chunk processes.")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOG_DIR))
    return parser.parse_args()


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _row_id(row: dict[str, str]) -> str:
    chunk_id = str(row.get("chunk_id", "") or "").strip()
    if chunk_id:
        return chunk_id
    task_id = str(row.get("task_id", "") or "task").strip()
    seed = str(row.get("seed", "") or "").strip()
    return f"{task_id}_seed{seed}" if seed else task_id


def _case_json_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as input_file:
            loaded = json.load(input_file)
        return isinstance(loaded, dict) and len(loaded) > 0
    except Exception:
        return False


def _completion_count(output_dir: Path) -> int:
    cases_dir = output_dir / "cases"
    if not cases_dir.exists():
        return 0
    count = 0
    for path in cases_dir.glob("*/final_summary.json"):
        if _case_json_is_valid(path):
            count += 1
    return count


def _worker_error_count(output_dir: Path) -> int:
    cases_dir = output_dir / "cases"
    if not cases_dir.exists():
        return 0
    return len(list(cases_dir.glob("*/worker_error.json")))


def _select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = list(rows)
    requested_ids = {str(item) for item in list(args.chunk_id or []) if str(item).strip() != ""}
    if requested_ids:
        selected = [row for row in selected if _row_id(row) in requested_ids]
    offset = max(0, int(args.offset or 0))
    if offset > 0:
        selected = selected[offset:]
    limit = max(0, int(args.limit or 0))
    if limit > 0:
        selected = selected[:limit]
    return selected


def _run_one(row: dict[str, str], *, logs_dir: Path, skip_completed: bool) -> dict[str, Any]:
    chunk_id = _row_id(row)
    config_path = Path(str(row.get("config_path", "") or "")).resolve()
    output_dir = Path(str(row.get("output_dir", "") or "")).resolve()
    planned_runs = int(float(row.get("planned_runs", 0) or 0))
    before_count = _completion_count(output_dir)
    if skip_completed and planned_runs > 0 and before_count >= planned_runs:
        return {
            "chunk_id": chunk_id,
            "status": "skipped_completed",
            "returncode": 0,
            "planned_runs": planned_runs,
            "completed_runs": before_count,
            "worker_error_count": _worker_error_count(output_dir),
            "output_dir": str(output_dir),
        }

    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{chunk_id}.out.log"
    stderr_path = logs_dir / f"{chunk_id}.err.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_experiment1_transfer.py"),
        "--config",
        str(config_path),
    ]
    if skip_completed:
        command.append("--skip-completed")

    with stdout_path.open("a", encoding="utf-8") as stdout_file, stderr_path.open("a", encoding="utf-8") as stderr_file:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )

    after_count = _completion_count(output_dir)
    return {
        "chunk_id": chunk_id,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "planned_runs": planned_runs,
        "completed_runs": after_count,
        "worker_error_count": _worker_error_count(output_dir),
        "output_dir": str(output_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def main() -> None:
    args = _parse_args()
    manifest_path = Path(str(args.manifest)).resolve()
    rows = _select_rows(_load_manifest(manifest_path), args)
    if len(rows) == 0:
        raise RuntimeError("No chunk rows selected.")

    if bool(args.dry_run):
        for row in rows:
            output_dir = Path(str(row.get("output_dir", "") or "")).resolve()
            completed = _completion_count(output_dir)
            print(
                f"{_row_id(row)} intent={row.get('intent_slug')} seed={row.get('seed')} "
                f"planned={row.get('planned_runs')} completed={completed} output={output_dir}"
            )
        print(f"Selected {len(rows)} chunks.")
        return

    logs_dir = Path(str(args.logs_dir)).resolve()
    jobs = max(1, int(args.jobs or 1))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                _run_one,
                row,
                logs_dir=logs_dir,
                skip_completed=bool(args.skip_completed),
            )
            for row in rows
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[{result['status']}] {result['chunk_id']} "
                f"completed={result['completed_runs']}/{result['planned_runs']} "
                f"errors={result['worker_error_count']} rc={result['returncode']}"
            )

    summary_path = logs_dir / "last_chunk_queue_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "jobs": jobs,
                "chunk_count": len(rows),
                "results": sorted(results, key=lambda item: str(item.get("chunk_id", ""))),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    failures = [result for result in results if int(result.get("returncode", 1)) != 0]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

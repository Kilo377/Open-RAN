"""Read-only progress snapshot for a finite RT/SYS matrix."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def last_record(path):
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 1000000))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeError):
            pass
    return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/rt_sys_targeted_matrix_nr')
    args = parser.parse_args()
    plan = json.loads((args.root / 'run_plan.json').read_text())
    rows = []
    for case in plan:
        name = f'{case["scene"]}_u{case["ues"]}_s{case["seed"]}_dt{case["path_update_ms"]}'
        path = args.root / name
        status = json.loads((path / 'run_status.json').read_text()) if (path / 'run_status.json').exists() else {}
        rows.append({'case': name, 'stage': status.get('stage', 'queued'),
            'slots': last_record(path / 'rt_sys_trace.jsonl').get('slot_id', 0),
            'completed': status.get('completed', False)})
    matrix = json.loads((args.root / 'matrix_status.json').read_text()) if (args.root / 'matrix_status.json').exists() else {}
    print(json.dumps({'completed_cases': sum(row['completed'] for row in rows),
        'total_cases': len(rows), 'finished_slots': sum(row['slots'] for row in rows),
        'planned_slots': sum(case['slots'] for case in plan), 'cases': rows,
        'matrix': matrix, 'memory': last_record(args.root / 'device_memory_samples.jsonl')}, ensure_ascii=False))

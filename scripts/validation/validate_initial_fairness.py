"""Compare actual baseline/orchestrated starts and exogenous packet arrivals."""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phy.rt_sys.contracts import source_fingerprint


def main():
    out = ROOT / 'outputs/rt_research_readiness_20261013/initial_fairness'
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for scene in ('simple_street_canyon', 'munich'):
        starts, arrivals = [], []
        for mode in ('baseline', 'orchestrated'):
            folder = out / f'{scene}_{mode}'
            if folder.exists():
                raise ValueError(f'Refusing overwrite {folder}')
            folder.mkdir()
            cmd = [sys.executable, str(ROOT / 'experiments/run_rt_sys.py'), '--scene', scene,
                   '--cells', '3', '--ues', '4', '--slots', '12', '--seed', '513',
                   '--initial-queue-bytes', '0', '--ue-speed-mps', '0', '--selection-mode', 'fixed_plan',
                   '--feedback-mode', 'nr_bler_bank', '--control-mode', mode, '--output', str(folder)]
            with (folder / 'process.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            status = json.loads((folder / 'run_status.json').read_text())
            assert status['completed'] and status['spark_verified']
            starts.append(json.loads((folder / 'initial_conditions.json').read_text()))
            arrivals.append([e for line in (folder / 'packet_events.jsonl').read_text().splitlines()
                             if (e := json.loads(line))['kind'] == 'arrival'])
        assert starts[0] == starts[1], 'Initial physical/business/association conditions differ'
        assert arrivals[0] == arrivals[1], 'Exogenous packet arrival streams differ'
        assert all(u['queue_bytes'] == 0 for u in starts[0]['ues'])
        cases.append(dict(scene=scene, passed=True, environment_sha256=starts[0]['environment_sha256'],
                          arrival_events=len(arrivals[0]), slots=12))
    report = dict(passed=True, source_sha256=source_fingerprint(ROOT), cases=cases,
                  scope='Identical initial environment and exogenous arrivals; controller actions may diverge afterward')
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

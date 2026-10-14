"""Save effective control fingerprints and RET analog weights from executed traces.

This is read-only postprocessing. Digital per-resource RZF weights remain in the
original precoding trace; they are channel results, not configuration settings.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phy.rt_sys.contracts import digest
from phy.rt_sys.paper_radio import ret_basis


def export(root, output):
    validation = json.loads((root / 'validation.json').read_text())
    assert validation['passed']
    result = dict(source_sha256=validation['source_sha256'], cases={},
                  definition='Effective runtime controls + actual cell budget + RET analog basis catalogue; digital RZF results are stored separately in rt_sys_trace.jsonl')
    for case in validation['results']:
        folder = root / case['case']
        snapshot = json.loads((folder / 'config_snapshot.json').read_text())
        manifest = snapshot['manifest']
        assert snapshot['source_sha256'] == result['source_sha256']
        rows, cols = manifest['tx_rows'], manifest['tx_cols']
        known, changes, previous = {}, [], {}
        for line in (folder / 'rt_sys_trace.jsonl').open():
            trace = json.loads(line)
            for cid, controls in trace['paper_interfaces']['runtime_config'].items():
                weights = None
                if controls['ret_enabled']:
                    weights = []
                    for beam in range(cols if controls['ret_codebook_id'] == 1 else 1):
                        basis = ret_basis(rows, cols, controls['ret_tilt_deg'], controls['ret_codebook_id'], beam)
                        weights.append(dict(beam=beam, real=basis.real.tolist(), imag=basis.imag.tolist()))
                config = dict(cell_id=cid, controls=controls, tx_power_dbm=trace['cells'][cid]['tx_power_dbm'],
                              budget_w=trace['cells'][cid]['budget_w'], tx_rows=rows, tx_cols=cols,
                              ret_analog_weights=weights, source_sha256=result['source_sha256'])
                sha = digest(config)
                known.setdefault(sha, config)
                if previous.get(cid) != sha:
                    changes.append(dict(slot=trace['slot_id'], cell_id=cid, sha256=sha))
                    previous[cid] = sha
        result['cases'][case['case']] = dict(configurations=known, effective_changes=changes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(cases=len(result['cases']), source_sha256=result['source_sha256'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    export(args.root, args.output)

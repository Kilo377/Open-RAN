"""Read-only LLM entry-point audit, separate from quantitative test labels.

No user preset/intent ledger is opened or written. Structured preregistered
ground truth is not changed in response to language-model output.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path

from smo.intent_language import IntentLanguage, resolve_ranges
import smo.intent_language as language_module


FIELDS = ('kpi', 'lower', 'upper', 'lower_inclusive', 'upper_inclusive')


def canonical(ranges):
    return sorted(({k:r.get(k) for k in FIELDS} for r in ranges), key=lambda r:r['kpi'])


def audit(inputs_path, catalog_path, output, key):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new language-audit output file')
    inputs = json.loads(Path(inputs_path).read_text())
    catalog = json.loads(Path(catalog_path).read_text())
    first = inputs['cases'][0]
    graph, frame = inputs['graphs'][first['graph_id']], inputs['contexts'][first['context_id']]['frame']
    language = IntentLanguage()
    language.configure(key)
    result = dict(schema='oran.study-language-audit.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        inputs_sha256=hashlib.sha256(Path(inputs_path).read_bytes()).hexdigest(),
        catalog_sha256=hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest(),
        parser_source_sha256=hashlib.sha256(Path(language_module.__file__).read_bytes()).hexdigest(),
        source_snapshot=frame.get('slot'), ledger_modified=False, preset_modified=False,
        quantitative_ground_truth_modified=False, deployment_results_read=False, cases=[])
    cases = [dict(spec, expect_question=False) for spec in catalog['intents']]
    cases += [dict(id='ambiguous-latency', text='请降低时延。', expect_question=True),
              dict(id='unsupported-individual', text='将UE3单独的平均功率降低10%。', expect_question=True)]
    try:
        for spec in cases:
            row = dict(id=spec['id'], text=spec['text'], expect_question=spec['expect_question'])
            try:
                parsed, provenance = language.parse(spec['text'], graph,
                    validator=lambda raw:resolve_ranges(raw, graph, frame))
                row.update(parsed=parsed, provenance=provenance)
                if spec['expect_question']:
                    row['passed'] = bool(parsed['questions'])
                else:
                    expected = resolve_ranges(dict(ranges=deepcopy(spec['ranges']), questions=[]), graph, frame)
                    row['expected'] = expected
                    row['passed'] = not parsed['questions'] and canonical(parsed['ranges']) == canonical(expected['ranges'])
            except Exception as error:
                # IntentLanguage sanitizes provider errors; never record raw
                # HTTP bodies, credentials or provider request headers.
                row.update(passed=False, error_type=type(error).__name__, error=str(error))
            result['cases'].append(row)
            result['passed'] = sum(c['passed'] for c in result['cases'])
            output.parent.mkdir(parents=True, exist_ok=True)
            temp = output.with_suffix('.tmp')
            temp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
            temp.replace(output)
            print(json.dumps(dict(id=row['id'], passed=row['passed'], done=len(result['cases']),
                total=len(cases)), ensure_ascii=False), flush=True)
    finally:
        language.clear_key()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.inputs, args.catalog, args.output, getpass.getpass('Session-only DeepSeek key: '))

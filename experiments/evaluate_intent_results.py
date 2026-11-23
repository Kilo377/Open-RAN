"""Re-score a retained batch's measured intent outcomes; never run simulators."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from smo.result_evaluation import enrich_batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--risk-report',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    raw=a.batch.read_bytes();batch=enrich_batch(json.loads(raw))
    risks={}
    if a.risk_report:
        risk=json.loads(a.risk_report.read_text())
        if risk['batch_id']!=batch['id']:raise ValueError('Risk report belongs to a different batch')
        risks={r['group_id']:r['score'] for r in risk['rows']}
    rows=[]
    for member in batch['members']:
        e=member.get('comparison',{}).get('result_evaluation')
        rows.append(dict(group_id=member['group_id'],xapps=member['selected_names'],
            phase=member['phase'],risk_score=risks.get(member['group_id']),
            before_score=e['before']['score']if e else None,
            after_score=e['after']['score']if e else None,
            score_change=e['change']if e else None,
            intent_met=e['after']['all_satisfied']if e else None,
            no_regression=e['no_regression']if e else None,
            regression_kpis=e['regression_kpis']if e else [],
            remaining_violation_mean=e['violation']['after']['mean']if e else None,
            remaining_violation_max=e['violation']['after']['max']if e else None,
            bottleneck_kpis=e['after']['bottleneck_kpis']if e else [],
            job_id=member.get('job_id')))
    result=dict(schema='oran.observed-intent-result-report.v2',batch_id=batch['id'],
        method=batch['result_evaluation_method'],reference=batch['result_evaluation_reference'],
        observe_slots=batch['observe_slots'],rows=rows,new_simulations=0,
        source_batch_sha256=hashlib.sha256(raw).hexdigest(),raw_batch_modified=False,
        engine_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'smo/result_evaluation.py').read_bytes()).hexdigest(),
        risk_used_in_result_score=False,automatic_selection=False,
        limits=['One retained seed; overlapping combinations are not independent replications.',
            'Frozen intent thresholds are not rebased to the evaluation baseline.',
            'Before/after comparison is a short-window response, not a causal estimate.',
            'Any supplied risk report is joined only for inspection and does not enter the evaluation.'])
    (a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
    (a.output/'enriched-batch.json').write_text(json.dumps(batch,ensure_ascii=False,allow_nan=False,indent=2))
    stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(['xApps','Before score','After score','Score change','Intent met','No regression','Regressed KPI',
        'Mean remaining normalized violation','Maximum remaining normalized violation','Bottleneck KPI','Conflict risk score','Job ID'])
    def fmt(v):return '—'if v is None else f'{v:+.4f}'
    lines=['# Observed intent result trial','',
        'E = mean normalized improvement when no requested KPI regresses; otherwise the worst regression. BEFORE = 0.',
        'Only requested KPIs participate. Improvement, interval attainment and remaining violation are distinct.',
        'The input is the retained equal-window data. No new simulation or modified intent thresholds.','',
        '| xApps | Improvement | No regression | Intent met | Remaining violation (mean) | Risk |',
        '|---|---:|---|---|---:|---:|']
    for r in rows:
        writer.writerow([' + '.join(r['xapps']),r['before_score'],r['after_score'],r['score_change'],r['intent_met'],r['no_regression'],
            ' + '.join(r['regression_kpis']),r['remaining_violation_mean'],r['remaining_violation_max'],
            ' + '.join(r['bottleneck_kpis']),r['risk_score'],r['job_id']])
        lines.append('| '+' + '.join(r['xapps'])+f" | {fmt(r['after_score'])} | {r['no_regression']} | {r['intent_met']} | "+
            fmt(r['remaining_violation_mean'])+' | '+fmt(r['risk_score'])+' |')
    lines+=['','Common scales, fixed from the retained BEFORE baseline: '+json.dumps(result['reference']),
        '', 'Zero improvement can coexist with remaining violation. Attainment uses the original open/closed rules, not a new epsilon.',
        '', '## Limits','']+['- '+s for s in result['limits']]
    (a.output/'results.csv').write_text('\ufeff'+stream.getvalue())
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    assert a.batch.read_bytes()==raw
    print(json.dumps(dict(batch_id=batch['id'],groups=len(rows),
        valid=sum(r['after_score']is not None for r in rows),
        met=sum(r['intent_met']is True for r in rows),new_simulations=0)))


if __name__=='__main__':main()

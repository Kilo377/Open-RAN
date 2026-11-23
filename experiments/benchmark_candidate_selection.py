"""Read-only preselection replay; archived outcomes are scoring labels only."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import time

from smo.candidate_selection import SelectionConfig, filter_catalog, METHODS
from smo.contracts import build_graph, validate_influences
from smo.graph_combinations import enumerate_combinations
from smo.population import graph_view


VALID = {'satisfied','below_range','above_range','outside_range'}
CONFIGS = {f'threshold_{int(t*100):02d}':SelectionConfig(coverage_threshold=t)
    for t in (.05,.10,.20)}
CASES = {
    'ue_power_and_mec_demand': ['ue_power_w','mec_expected_gflops'],
    'ue_power_and_mec_delay': ['ue_power_w','mec_offloaded_latency_ms'],
    'bs_power_and_mec_delay': ['ap_power_w','mec_offloaded_latency_ms'],
    'throughput_and_bs_power': ['network_goodput_mbps','ap_power_w'],
    'throughput_and_urllc': ['network_goodput_mbps','urllc_failure_events_per_s'],
    'mec_delay': ['mec_offloaded_latency_ms'],
    'four_targets': ['ue_power_w','mec_expected_gflops','ap_power_w','network_goodput_mbps'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_member(member, required):
    comparison = member.get('comparison', {})
    rows = {r['kpi']:r for r in comparison.get('rows', [])}
    complete = all(k in rows and rows[k].get('before',{}).get('status') in VALID
        and rows[k].get('after',{}).get('status') in VALID for k in required)
    passed = complete and all(rows[k]['after']['status']=='satisfied' for k in required)
    gap = comparison.get('after', {}).get('worst_normalized_gap') if complete else None
    return dict(complete=complete, passed=passed,
        worst_normalized_gap=gap, observed_slots=comparison.get('observe_slots'),
        missing_kpis=[k for k in required if k not in rows or
            rows[k].get('before',{}).get('status') not in VALID or rows[k].get('after',{}).get('status') not in VALID])


def summarize_scores(members, required, primary, structural):
    matched = [m for m in members if frozenset(m['app_ids']) in structural and m.get('comparison')]
    scored = [(m,score_member(m,required)) for m in matched]
    valid = [(m,s) for m,s in scored if s['complete']]
    passed = [(m,s) for m,s in valid if s['passed']]
    kept = [(m,s) for m,s in valid if frozenset(m['app_ids']) in primary]
    kept_passed = [(m,s) for m,s in passed if frozenset(m['app_ids']) in primary]
    gaps = [s['worst_normalized_gap'] for _,s in valid if s['worst_normalized_gap'] is not None]
    kept_gaps = [s['worst_normalized_gap'] for _,s in kept if s['worst_normalized_gap'] is not None]
    return dict(measured_groups=len(matched), valid_groups=len(valid), insufficient_groups=len(scored)-len(valid),
        passing_groups=len(passed),primary_measured_groups=sum(frozenset(m['app_ids'])in primary for m,_ in scored),
        primary_valid_groups=len(kept),primary_passing_groups=len(kept_passed),
        passing_group_recall=len(kept_passed)/len(passed) if passed else None,
        any_known_passing_retained=bool(kept_passed) if passed else None,
        best_observed_gap=min(gaps) if gaps else None,
        best_primary_observed_gap=min(kept_gaps) if kept_gaps else None,
        observed_slots=sorted({s['observed_slots'] for _,s in scored if s['observed_slots'] is not None}),
        missing_kpis=dict(Counter(k for _,s in scored for k in s['missing_kpis'])),
        scored_members=[dict(group_id=m['group_id'],app_ids=m['app_ids'],job_id=m.get('job_id'),
            in_primary=frozenset(m['app_ids']) in primary,**s) for m,s in scored])


def trial(graph, ranges, snapshot, config, method='influence_coverage_v1'):
    # Deliberately pass no live states, proposals, action spaces or outcomes.
    frame = dict(slot=snapshot.get('slot'),time_ms=snapshot.get('time_ms'),states={})
    plan = dict(snapshot=dict(slot=frame['slot']),ranges=deepcopy(ranges),questions=[])
    raw = enumerate_combinations(graph,frame,plan)
    start = time.perf_counter()
    result = filter_catalog(graph,raw,method=method,config=config)
    elapsed = time.perf_counter()-start
    return raw,result,elapsed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--batch-root',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=METHODS[1:],default='influence_coverage_v1')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    hashes={};graphs={};structural=[];empirical=[];catalogs=[]
    for context in ('native_1x','steady_2x','burst_2x'):
        source=a.analysis/context/'all-effects.json';hashes[str(source)]=sha(source)
        matrix=json.loads(source.read_text())
        graphs[context]=validate_influences(graph_view(build_graph()),source)
        (a.output/('graph-'+context+'.json')).write_text(json.dumps(graphs[context],indent=2))
        for case,kpis in CASES.items():
            # Probe capability directions using calibration reference bounds.
            # These are structural test cases, not deployed numeric targets.
            references={k:next((r.get('reference_kpi_mean') for r in matrix['estimates']
                if r['kpi']==k and r.get('reference_kpi_mean') is not None),0.) for k in kpis}
            ranges=[dict(kpi=k,lower=references[k] if k=='network_goodput_mbps' else 0.,
                upper=None if k=='network_goodput_mbps' else references[k],
                lower_inclusive=k!='network_goodput_mbps',upper_inclusive=True)for k in kpis]
            for name,config in CONFIGS.items():
                raw,result,elapsed=trial(graphs[context],ranges,dict(slot=0,time_ms=0),config,a.method)
                summary=result['initial_selection']
                structural.append(dict(scenario=context,case=case,configuration=name,
                    original_groups=len(raw['groups']),primary_groups=result['group_count'],
                    reserve_groups=result['reserve_group_count'],reduction_percent=summary['pruned_fraction']*100,
                    filter_ms=elapsed*1000,exhaustive_fallback=summary['exhaustive_fallback']))
                catalogs.append(dict(scenario=context,case=case,configuration=name,
                    **{k:v for k,v in summary.items() if k!='app_capabilities'},
                    primary=[dict(id=g['id'],app_ids=g['app_ids'],**g['initial_selection'])for g in result['groups']],
                    reserve=[dict(id=g['id'],app_ids=g['app_ids'],**g['initial_selection'])for g in result['reserve_groups']]))
                if name=='threshold_10':
                    (a.output/(context+'-'+case+'-capabilities.json')).write_text(json.dumps(summary['app_capabilities'],indent=2))
    for root in a.batch_root:
        for path in sorted(root.glob('*/batch.json')):
            batch=json.loads(path.read_text())
            if not any(m.get('comparison') for m in batch['members']):continue
            hashes[str(path)]=sha(path)
            for name,config in CONFIGS.items():
                raw,result,elapsed=trial(graphs['native_1x'],batch['ranges'],batch['snapshot'],config,a.method)
                stats=summarize_scores(batch['members'],raw['required_kpis'],
                    {frozenset(g['app_ids']) for g in result['groups']},
                    {frozenset(g['app_ids']) for g in raw['groups']})
                empirical.append(dict(batch_id=batch['id'],source=str(path),configuration=name,
                    scenario='native_1x',required_kpis=raw['required_kpis'],
                    structural_groups=len(raw['groups']),primary_groups=result['group_count'],
                    reserve_groups=result['reserve_group_count'],**stats))
    aggregate=[]
    for name in CONFIGS:
        ss=[r for r in structural if r['configuration']==name]
        es=[r for r in empirical if r['configuration']==name]
        with_pass=[r for r in es if r['passing_groups']]
        passing=sum(r['passing_groups']for r in es);kept=sum(r['primary_passing_groups']for r in es)
        aggregate.append(dict(configuration=name,
            structural_original=sum(r['original_groups']for r in ss),
            structural_primary=sum(r['primary_groups']for r in ss),
            structural_reduction_percent=100*(1-sum(r['primary_groups']for r in ss)/sum(r['original_groups']for r in ss)),
            evaluated_batch_count=len(es),valid_historical_groups=sum(r['valid_groups']for r in es),
            insufficient_historical_groups=sum(r['insufficient_groups']for r in es),
            passing_historical_groups=passing,primary_passing_historical_groups=kept,
            passing_group_recall=kept/passing if passing else None,
            batches_with_known_passing=len(with_pass),
            batches_retaining_known_passing=sum(r['any_known_passing_retained']for r in with_pass),
            minimum_primary_count=min(r['primary_groups']for r in ss),
            maximum_primary_count=max(r['primary_groups']for r in ss)))
    output=dict(schema='oran.initial-candidate-selection-benchmark.v1',new_simulations=0,
        selection_method=a.method,
        default_runtime_changed=False,graph_modified=False,conflict_method_modified=False,
        default_experimental_configuration='threshold_05',aggregate=aggregate,
        structural_trials=structural,historical_trials=empirical,
        limitations=['Historical evaluations are native-scene, short-window records, not new validation across all three traffic scenes.',
            'Graphs/calibration and historical deployment snapshots or controller versions may differ; this is a retrospective screen.',
            'Historical catalog coverage is partial; unmeasured combinations are not successes or failures.',
            'Missing mature MEC/task samples are excluded from outcome scoring.',
            'Repeated batches and shared seeds are not independent statistical replications.',
            'Primary/reserve depends on influence capability only; no numerical intent attainment, conflict score or winning policy is inferred.'])
    (a.output/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    (a.output/'catalogs.json').write_text(json.dumps(catalogs,indent=2,allow_nan=False))
    with(a.output/'structural-results.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(structural[0]));writer.writeheader();writer.writerows(structural)
    fields=[k for k in empirical[0]if k not in ('scored_members','missing_kpis')]
    with(a.output/'historical-results.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(empirical)
    lines=['# Initial xApp combination selection trial','',
        'Method: '+a.method+'; current actions and gain signs are not used for capability direction.','',
        'Graph influence screening only. Existing graph, conflict quantification and final strategy selection are unchanged. No new simulation.','',
        '| Influence threshold | Structural candidates retained | Reduction | Known passing groups retained | Batches retaining a known pass |',
        '|---|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['configuration']} | {r['structural_primary']} / {r['structural_original']} | {r['structural_reduction_percent']:.1f}% | {r['primary_passing_historical_groups']} / {r['passing_historical_groups']} | {r['batches_retaining_known_passing']} / {r['batches_with_known_passing']} |")
    lines+=['','## Conservative experimental setting: threshold 0.05','',
        '| Scene | Intent KPI set | Original | Primary | Reserve | Reduction |','|---|---|---:|---:|---:|---:|']
    for r in structural:
        if r['configuration']=='threshold_05':lines.append(f"| {r['scenario']} | {r['case']} | {r['original_groups']} | {r['primary_groups']} | {r['reserve_groups']} | {r['reduction_percent']:.1f}% |")
    lines+=['','## Retrospective measured results (threshold 0.05)','',
        '| Batch | KPI set | Measured | Valid | Missing | Known passes: all → primary | Best observed gap: all → primary |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in empirical:
        if r['configuration']=='threshold_05':lines.append(f"| {r['batch_id'][:8]} | {', '.join(r['required_kpis'])} | {r['measured_groups']} | {r['valid_groups']} | {r['insufficient_groups']} | {r['passing_groups']} → {r['primary_passing_groups']} | {r['best_observed_gap']} → {r['best_primary_observed_gap']} |")
    lines+=['','## Limits','']+['- '+s for s in output['limitations']]
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    (a.output/'provenance.json').write_text(json.dumps(dict(input_sha256=hashes,
        module_sha256=sha(Path(__file__).resolve().parents[1]/'smo/candidate_selection.py'),
        capability_module_sha256=sha(Path(__file__).resolve().parents[1]/'smo/xapp_capabilities.py'),
        benchmark_sha256=sha(__file__),new_simulations=0,
        outcomes_used_for_selection=False,
        outputs_sha256={f.name:sha(f)for f in a.output.iterdir()if f.is_file()and (
            f.name in ('results.json','catalogs.json','structural-results.csv','historical-results.csv','report.md')
            or f.name.startswith('graph-')and f.suffix=='.json' or f.name.endswith('-capabilities.json'))}),indent=2))
    print(json.dumps(dict(output=str(a.output),aggregate=aggregate)))


if __name__=='__main__':main()

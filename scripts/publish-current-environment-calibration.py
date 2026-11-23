"""Create-only calibration certificate and same-origin UI manifest."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.current_environment_calibration import atomic,read,sha
from smo.current_calibration import verify_current_calibration
from smo.runtime_recipe import load_recipe

TARGETS=['ue_power_w','ap_power_w','network_goodput_mbps','mec_offloaded_latency_ms',
         'urllc_failure_fraction','urllc_packet_latency_ms','mec_expected_gflops']


def publish(study,base_manifest,analysis_dir='analysis',publication_dir='ui'):
    study=Path(study).resolve();manifest=read(base_manifest)
    analysis=(study/analysis_dir).resolve();publication=(study/publication_dir).resolve()
    if any(p==study or not p.is_relative_to(study)for p in(analysis,publication)):
        raise ValueError('Use distinct owned analysis/publication subdirectories')
    entry=next(e for e in manifest['scenarios'] if e['id']==manifest['default'])
    recipe,_=load_recipe(entry['runtime_recipe'])
    dataset=analysis/'measurement-dataset.json';matrix=analysis/'influences.json';design=study/'design.json'
    origin=analysis/'origin-equivalence.json'
    measurements=read(dataset)['measurements'];gains=read(matrix)
    expected=sum(len(read(study/(w+'-plan.json'))['cases']) for w in ('agx1','agx2'))
    if len(measurements)!=expected:raise ValueError('Do not publish partial calibration')
    if publication.exists():raise ValueError('Preserve prior publication; create a new version')
    certificate=dict(schema='oran.smo.current-environment-calibration.v1',environment_id=recipe['id'],
        dynamics_sha256=recipe['dynamics_sha256'],source_archive_sha256=recipe['validated_source_archive_sha256'],
        profile_sha256s={k:v['sha256'] for k,v in recipe['profiles'].items()},frozen_models=deepcopy(recipe['frozen_models']),
        combination_evaluation_data_used=False,learned_xapps_invoked=False,audited_run_count=len(measurements),
        seeds=read(design)['seeds'],confidence_level=.95,
        files={k:dict(path=str(p),sha256=sha(p)) for k,p in [('influences',matrix),('measurements',dataset),('design',design),('origin_equivalence',origin)]})
    path=publication/'calibration.json';atomic(path,certificate)
    entry.update(influences=str(matrix),current_calibration=str(path),influence_selection='significant_relations',influence_confidence_level=.90)
    verified=verify_current_calibration(entry,recipe)
    from smo.contracts import validate_influences,build_graph
    from smo.population import graph_view
    from smo.statistical_filter import filter_graph
    from smo.influence_units import align_graph
    graph=align_graph(filter_graph(validate_influences(graph_view(build_graph()),matrix),confidence_level=.90,retain_measured_contexts=True))
    active=[e for e in graph['edges'] if e['kind']=='influences']
    estimated={(r['parameter'],r['kpi']) for r in gains['estimates']}
    params=['ue_offload_fraction','ue_drx_configs','scheduler_mode','ue_scheduler_weights','cell_sleep_mode']
    target_coverage=[dict(parameter=p,kpi=k,estimated=(p,k) in estimated,
        active_in_graph=any(e['source']=='rcp:'+p and e['target']=='kpi:'+k for e in active)) for p in params for k in TARGETS]
    if not all(c['estimated'] for c in target_coverage):
        raise ValueError('Some target P–K pairs have no numeric current-environment estimate')
    if any(not r.get('projection_units',{}).get('valid') for e in active for r in e.get('sensitivities',[])):
        raise ValueError('Active gain has an invalid unit projection')
    summary=dict(**verified,total_estimated_pairs=len(estimated),target_pair_coverage=target_coverage,
        active_p_k_relations=len(active),statistical_filter=graph['statistical_filter'],
        raw_data_retained=True,prior_intents_or_tags_modified=False,prior_runtime_recipe_modified=False,
        base_manifest=dict(path=str(Path(base_manifest).resolve()),sha256=sha(base_manifest)))
    atomic(analysis/'gain-summary.json',summary)
    with (analysis/'gains.csv').open('w',newline='') as stream:
        fields=['parameter','kpi','contrast','kind','estimate','unit','ci95_low','ci95_high','independent_seeds','importance_abs','assessment']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in gains['estimates']:
            ci=r.get('confidence_interval')or [None,None]
            writer.writerow(dict(parameter=r['parameter'],kpi=r['kpi'],contrast=r.get('contrast_label',r['contrast_id']),
                kind=r['kind'],estimate=r['estimate'],unit=r['unit'],ci95_low=ci[0],ci95_high=ci[1],
                independent_seeds=r['sample_count'],importance_abs=abs(r['importance']['signed_reference_importance']),
                assessment=r.get('fit_diagnostics',{}).get('assessment',r['statistical_context']['retained_as'])))
    target=publication/'scenario-manifest.json';atomic(target,manifest)
    print(json.dumps(dict(manifest=str(target),audited_runs=len(measurements),estimates=len(gains['estimates']),
        active_p_k_relations=len(active),target_pairs_estimated=sum(c['estimated']for c in target_coverage)),ensure_ascii=False))
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--base-manifest',type=Path,required=True)
    parser.add_argument('--analysis-dir',default='analysis');parser.add_argument('--publication-dir',default='ui')
    args=parser.parse_args();publish(args.study,args.base_manifest,args.analysis_dir,args.publication_dir)

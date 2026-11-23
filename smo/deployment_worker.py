"""Owned, isolated evaluation run. Trained policies use the normal hot/RIC path."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import re
import subprocess
import time


def atomic(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False));tmp.replace(path)


def plan(request,root):
    from smo.runtime import APP_LOGS
    if not re.fullmatch(r'[0-9a-f]{32}',request['id']):raise ValueError('Invalid owned job ID')
    if type(request['seed'])is not int or not 0<=request['seed']<2**32:raise ValueError('Invalid seed')
    apps=request['app_ids'];length=request['observe_slots']
    if not isinstance(apps,list)or not apps or len(apps)!=len(set(apps))or any(a not in {'xapp:'+i for i in APP_LOGS}for a in apps):
        raise ValueError('Choose registered trained xApps')
    if type(length)is not int or not 1<=length<=20000:raise ValueError('Invalid observation slots')
    # A genuine counter row is needed at the left endpoint. Activate only after
    # W+1 baseline slots, at a normal 20ms RIC boundary. No post-action settling.
    baseline=((length+1+39)//40)*40
    recipe=request.get('runtime_recipe')
    origin=request.get('expected_origin')
    if recipe is not None:
        from smo.runtime_recipe import validate_worker_recipe
        validate_worker_recipe(recipe,root,origin,request.get('expected_source_archive_sha256'))
        if request['seed']!=recipe['seed']:raise ValueError('Runtime recipe and request seeds differ')
        baseline=recipe['baseline_slot']
    elif origin is not None:
        from smo.evaluation_origin_guard import validate_proof
        validate_proof(origin)
        if origin['baseline_slot']!=baseline or origin['observe_slots']!=length or origin['seed']!=request['seed']:
            raise ValueError('Prepared evaluation origin differs from the worker plan')
    # Bound cold-policy/20ms RIC startup by two 100ms controller periods.
    total=baseline+(recipe['tail_slots']if recipe is not None else 400)+length
    job=root/'outputs'/'smo_deployments'/request['id'];run=job/'run'
    command=['bash','scripts/orin/run-six-ue-cuda-nr.sh','--model','fast','--fast-profile','system',
        '--slots',str(total),'--seed',str(request['seed']),'--traffic-load','1',
        '--ue-services','drx-mec','--xapp-pool','chongyu',
        '--ap-services-profile',str(root/'configs/rt_sys/ap_services_reference.json'),
        '--record-compression','gzip','--live-xapps','--xapps-start','',
        '--xapp-toggle-timeline',str(job/'timeline.json'),'--xapp-decision-preview',
        '--xapp-preview-before-slot',str(baseline),
        '--channel-cache-root',str(root/'channel_cache/smo_deployments'/str(request['seed'])),
        '--live-state',str(run/'live.json'),'--output',str(run)]
    if origin is not None:command+=['--evaluation-origin-file',str(job/'expected_origin.json')]
    if recipe is not None:
        command[command.index('--traffic-load')+1]=str(recipe['traffic_load'])
        command[command.index('--ap-services-profile')+1]=recipe['profiles']['ap_services_profile']['path']
        command+=['--ue-services-config',recipe['profiles']['ue_services_config']['path'],
            '--traffic-profile',recipe['profiles']['traffic_profile']['path'],
            '--config',recipe['profiles']['base_config']['path']]
        from smo.model_release import runtime_arguments
        command+=runtime_arguments(recipe,root)
    return job,run,baseline,total,command


def evaluate(request,run,baseline):
    from smo.window_metrics import RecordedWindows
    from smo.runtime import replay
    from smo.contracts import build_graph
    from smo.population import graph_view
    from smo.intent_comparison import IntentComparison
    config=json.loads((run/'config_snapshot.json').read_text())
    index=RecordedWindows(run,config);index.sync()
    graph=graph_view(build_graph());keys=[r['kpi']for rec in request['records']for r in rec['ranges']]
    recipe=request.get('runtime_recipe')
    before_length=recipe['origin_observe_slots']if recipe is not None else request['observe_slots']
    before=index.window(keys,baseline-before_length,baseline,through=index.slots[-1])
    store=IntentComparison({'deployment_job':request['id']},run.parent/'comparisons.json')
    source=dict(run_id=request['id'],frames=[])
    # Legacy comparison retains equal windows. A declared runtime recipe uses
    # its certified BEFORE prefix here; fixed_origin_comparison below computes
    # the independently requested AFTER window with the same KPI formulas.
    ident=store.start(request['records'],graph,before,before_length,request['id'])
    result=store.review(ident,graph,source,dict(slot=index.slots[-1]),windows=index)
    if request.get('expected_origin')is not None:
        from smo.evaluation_reference import comparable
        origin=request['expected_origin'];path=run/'evaluation_origin_verified.json'
        verification=json.loads(path.read_text())if path.is_file()else{}
        if verification.get('origin_id')!=origin['origin_id']or verification.get('verified_before_activation')is not True:
            raise RuntimeError('Prepared evaluation origin was not verified before activation')
        expected={k:origin['before_frame']['states'].get('kpi:'+k)for k in keys}
        if not comparable(before['states'],{'kpi:'+k:v for k,v in expected.items()}):
            raise RuntimeError('Recorded deployment baseline differs from the prepared intent origin')
        result=fixed_origin_comparison(result,request,graph,index,before,baseline)
        result['evaluation_origin']=verification
    selected={a[5:]for a in request['app_ids']}
    unexpected={r['app_id'][5:]for r in result['application_evidence']}-selected
    if unexpected:raise RuntimeError('Unexpected xApp execution in an isolated run')
    if result['after']is None:raise RuntimeError('No complete post-actuation window; inspect retained receipts')
    result.update(deployment_job_id=request['id'],selected_app_ids=request['app_ids'],
        baseline_origin='fresh_run_all_xapps_off',intent_bounds_rebased=False,
        direct_conflict_mitigation='uniform_random',automatic_selection=False,
        model_release='20261117',causal_inference=False)
    if recipe is not None:
        result['runtime_recipe']={k:deepcopy(recipe[k])for k in ('id','recipe_sha256','validated_source_sha256','validated_source_archive_sha256','dynamics_sha256','dynamics_members','profiles')}
        result['worker_source_archive_sha256']=request['expected_source_archive_sha256']
        result['frozen_models']=deepcopy(recipe['frozen_models'])
        if recipe.get('model_release'):
            result['model_release']=recipe['model_release']['id']
            result['runtime_recipe']['model_release']=deepcopy(recipe['model_release'])
    atomic(run.parent/'comparison.json',result)
    return result


def fixed_origin_comparison(result,request,graph,index,before,baseline):
    """Common absolute window for every candidate; actual receipts stay true.

    Includes selection/actuation latency. No member may pick a more favorable
    later window from its first successful command. Legacy archives keep their
    historical receipt-anchored interpretation unchanged.
    """
    from smo.intent_ledger import intersect_intents
    from smo.intent_quality import assess,score,normalization,quality,quantify,enrich_comparison
    length=request['observe_slots'];end=baseline+length
    ranges=result['rows'];keys=[r['kpi']for r in ranges]
    after=index.window(keys,baseline,end,through=index.slots[-1])
    scales={r['kpi']:normalization(r,before['states'].get('kpi:'+r['kpi'],{}))for r in ranges}
    oldscore=score(ranges,before,scales);newscore=score(ranges,after,scales)
    rows=[]
    for target in ranges:
        old=assess(target,before,scales[target['kpi']]);new=assess(target,after,scales[target['kpi']])
        row={k:deepcopy(v)for k,v in target.items()if k not in('before','after','delta','normalization','quantification')}
        row.update(before=old,after=new,delta=new['value']-old['value']if new['value']is not None and old['value']is not None else None,
            normalization=scales[target['kpi']],quantification=quantify(target,old,new))
        rows.append(row)
    intents=[]
    for record in request['records']:
        local=intersect_intents([record],graph,before)['ranges'];old=score(local,before,scales);new=score(local,after,scales)
        intents.append(dict(id=record['id'],text=record['text'],before=old,after=new,quality=quality(old,new)))
    result=deepcopy(result)
    result.update(status='complete'if oldscore['valid']==oldscore['required']and newscore['valid']==newscore['required']else'complete_with_missing_kpis',
        before=oldscore,after=newscore,rows=rows,intents=intents,quality=quality(oldscore,newscore),
        after_window=dict(window_start_slot=baseline,slot=end,window_slots=length),target_end_slot=end,observed_after_slot=end,
        actual_elapsed_slots=length,remaining_slots=0,application_evidence=[r for r in result['application_evidence']if r['slot']<=end],
        observation_origin_slot=baseline,fixed_combination_observation_window=True,
        deployment_latency_included=True,interpretation='common_origin_short_term_response_including_selection_and_actuation_latency')
    if request.get('runtime_recipe')is not None:
        result.update(observe_slots=length,
            baseline_window=dict(window_start_slot=baseline-request['runtime_recipe']['origin_observe_slots'],slot=baseline,window_slots=request['runtime_recipe']['origin_observe_slots']),
            comparison_basis='declared_independent_before_after_windows_from_scalar_counters_and_mature_task_events',
            application_evidence=[deepcopy(r)for r in sorted(index.receipts.values(),key=lambda r:(r['slot'],r['app_id']))if baseline<r['slot']<=end])
    return enrich_comparison(result)


def run_worker(request,root):
    job,run,baseline,total,command=plan(request,root)
    if (job/'status.json').exists():raise ValueError('A deployment job cannot overwrite an earlier job')
    job.mkdir(parents=True,exist_ok=True)
    atomic(job/'request.json',request)
    if request.get('expected_origin')is not None:atomic(job/'expected_origin.json',request['expected_origin'])
    atomic(job/'timeline.json',[dict(at_ms=baseline*.5,operation='set',apps=[a[5:]for a in request['app_ids']],cell='*')])
    status=dict(id=request['id'],phase='initializing',baseline_slot=baseline,total_slots=total,
        slot=0,observe_slots=request['observe_slots'],application_slot=None,started_unix_s=time.time(),
        selected_app_ids=request['app_ids'],selected_names=request['selected_names'],
        prepared_origin_required=request.get('expected_origin')is not None)
    def publish(**values):status.update(values,updated_unix_s=time.time());atomic(job/'status.json',status)
    publish()
    with (job/'simulation.log').open('wb')as log:
        process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        publish(simulation_pid=process.pid)
        cancelled=False;application=None
        while process.poll()is None:
            if (job/'STOP').exists():
                cancelled=True;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                break
            path=run/'live.json'
            if path.exists():
                try:
                    live=json.loads(path.read_text());slot=live.get('slot',0)
                    selected={a[5:]for a in request['app_ids']}
                    for receipt in live.get('control',{}).get('history',[]):
                        at=receipt.get('apply_slot',receipt.get('slot_id'))
                        owners={p.get('implementation_id')for p in receipt.get('provenance',[])}
                        if receipt.get('status')=='applied'and type(at)is int and at>baseline and owners&selected:
                            application=min(application,at)if application else at
                    fixed=request.get('expected_origin')is not None
                    phase='collecting_baseline'if slot<=baseline else'observing'if fixed else'deploying'if application is None else'observing'
                    publish(phase=phase,slot=slot,application_slot=application,
                        observed_slots=min(request['observe_slots'],max(0,slot-baseline))if fixed else min(request['observe_slots'],max(0,slot-application+1))if application else 0)
                except (ValueError,OSError):pass
            time.sleep(.75)
        code=process.wait()
    if cancelled:publish(phase='cancelled',exit_code=code);return
    if code:
        tail=(job/'simulation.log').read_text(errors='replace')[-2000:]
        publish(phase='failed',exit_code=code,error='Simulation failed; inspect simulation.log',log_tail=tail);return
    publish(phase='evaluating',slot=total)
    try:
        result=evaluate(request,run,baseline)
        publish(phase='finished',comparison=result,application_slot=result['application_slot'],observed_slots=request['observe_slots'])
    except Exception as error:publish(phase='failed',error=str(error))


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    run_worker(json.loads(a.request.read_text()),root)


if __name__=='__main__':main()

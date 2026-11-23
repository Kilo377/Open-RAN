"""Budgeted, checkpointed population RCP identification in separate traffic contexts.

This module changes experimental stimuli, never a trained xApp. Remote copies
are immutable and all writes are confined to an owned project/RAM directory.
"""
from copy import deepcopy
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.rcp_perturbation import validate,vector
from experiments.run_rcp_perturbation import atomic,load,sha,source_hash

SCENARIOS={
    'native_1x':dict(traffic_load=1.,urllc_scenario='native',description='Native offered load, steady URLLC arrivals'),
    'steady_2x':dict(traffic_load=2.,urllc_scenario='native',description='Doubled offered load, steady URLLC arrivals'),
    'burst_2x':dict(traffic_load=2.,urllc_scenario='burst',description='Same mean load as steady_2x; fourfold URLLC peak, 20 ms on / 60 ms off'),
    'alternating_2x':dict(traffic_load=2.,urllc_scenario='alternating',description='Same mean load; 0.8 s steady / 0.8 s burst repeating'),
}


def design(root=ROOT):
    from experiments.population_rcp_design import design as population_design
    ref,_,_=population_design(root)
    ref={**ref,'drx_cycle_ms':40.}
    specs={};vectors={}
    def add(name,**changes):
        spec=validate({**ref,**changes,'arm':name})
        key=json.dumps(vector(spec,perturbed=True),sort_keys=True)
        if key in vectors:return vectors[key]
        specs[name]=spec;vectors[key]=name;return name
    add('native');add('latency_native',scheduler_mode='latency_aware');add('qos_native',scheduler_mode='qos_deadline')
    contrasts=[]
    for mode in ('micro_sleep','light_sleep','deep_sleep'):
        name=add('ap_'+mode,cell_sleep_mode=mode)
        contrasts.append(dict(id='ap_'+mode,parameter='cell_sleep_mode',low='native',high=name,reference='native',step=1.,kind='finite_contrast',
            parameter_component='all_cells',level_field='cell_sleep_mode',scope='All three cells',context='Fixed round_robin; native DRX and MEC offloading',
            population_control_eligible=True,**{'from':'active','to':mode}))
    for name,mode in [('latency','latency_aware'),('qos','qos_deadline')]:
        contrasts.append(dict(id='scheduler_'+name,parameter='scheduler_mode',low='native',high=name+'_native',reference='native',step=1.,kind='finite_contrast',
            parameter_component='all_cells',level_field='scheduler_mode',scope='All three cells; all weights 1',context='Active BSs; native DRX and MEC offloading',
            population_control_eligible=True,**{'from':'round_robin','to':mode}))
    grids=[
        dict(id='offload_population_range',parameter='ue_offload_fraction',field='offload_probability',reference=.5,levels=[0.,.25,.5,.75,1.],mode='round_robin',component='mec_workload',x_unit='fraction',scope='All MEC sources'),
        dict(id='drx_population_range',parameter='ue_drx_configs',field='drx_inactivity_ms',reference=8.,levels=[0.,1.,4.,8.,16.,40.],mode='round_robin',component='drx_enabled_population',scalar_component='inactivity_ms',x_unit='ms',scope='All DRX-enabled UEs; cycle fixed at 40 ms'),
        dict(id='weight_mec_population_range',parameter='ue_scheduler_weights',field='mec_weight',reference=1.,levels=[.5,.75,1.,1.5,2.],mode='latency_aware',component='mec_priority',x_unit='weight',scope='All MEC sources; other UE weights 1'),
        dict(id='weight_urllc_population_range',parameter='ue_scheduler_weights',field='urllc_weight',reference=1.,levels=[.5,1.,1.5,2.,3.,4.],mode='qos_deadline',component='urllc_priority',x_unit='weight',scope='All URLLC UEs; other UE weights 1'),
        dict(id='drx_cycle_population_range',parameter='ue_drx_configs',field='drx_cycle_ms',reference=40.,levels=[20.,40.,80.,160.],mode='round_robin',component='drx_enabled_population',scalar_component='cycle_ms',x_unit='ms',scope='All DRX-enabled UEs; inactivity fixed at 8 ms'),
    ]
    for g in grids:
        g['arms']=[add(g['field']+'_'+str(x).replace('.','p'),scheduler_mode=g['mode'],**{g['field']:x})for x in g['levels']]
        center=vector(specs[g['arms'][g['levels'].index(g['reference'])]],perturbed=True)
        for name in g['arms']:
            v=vector(specs[name],perturbed=True)
            if {k for k in center if center[k]!=v[k]}not in(set(),{g['parameter']}):raise ValueError('Unrelated RCP changed')
    joint=add('drx_profile_80ms_1ms',drx_cycle_ms=80.,drx_inactivity_ms=1.)
    contrasts.append(dict(id='drx_profile_80ms_1ms',parameter='ue_drx_configs',low='native',high=joint,reference='native',step=1.,kind='finite_contrast',
        parameter_component='drx_enabled_population',level_field='drx_profile',scope='All DRX-enabled UEs; exact joint profile, not separable derivative',
        context='round_robin; active BSs; native MEC offloading',population_control_eligible=True,
        **{'from':{'cycle_ms':40.,'inactivity_ms':8.},'to':{'cycle_ms':80.,'inactivity_ms':1.}}))
    return specs,grids,contrasts


def make_case(context,seed,arm,*,stage='grid',duration_s=1.8):
    if context not in SCENARIOS:raise ValueError('Unknown experimental context')
    score_start=.4;score_end=round(duration_s-.2,8)
    if duration_s<1.0 or not math.isclose(score_end/.1,round(score_end/.1),abs_tol=1e-8):raise ValueError('Use cohort-aligned windows')
    return dict(name=f'{stage}_{context}_s{seed}_{arm}',context=context,seed=seed,arm=arm,stage=stage,
        duration_s=duration_s,slots=round(duration_s*2000),switch_ms=100,
        scoring_start_ms=score_start*1000,scoring_end_ms=score_end*1000,cohort_end_ms=score_end*1000,
        expected_task_cohort_count=round((score_end-score_start)*10)*2,
        expected_task_count=round((score_end-score_start)*20)*2)


def is_completed(out,case):
    folder=Path(out)/'cases'/case['name'];seal=Path(out)/'checkpoints'/(case['name']+'.json')
    if not seal.exists():return False
    row=load(seal)
    return row.get('completed')is True and all((folder/p).is_file()and sha(folder/p)==d for p,d in row['files_sha256'].items())


def external_simulations():
    result=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().replace(b'\x00',b' ').decode(errors='replace')
            if ('experiments/run_rt_sys.py' in args or '-m smo.deployment_worker --request' in args)and int(p.parent.name)!=os.getpid():
                result.append(dict(pid=int(p.parent.name),command=args[:500]))
        except (OSError,ValueError):pass
    return result


def resources(root):
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1)
        if k in ('MemAvailable','SwapFree'):mem[k]=int(v.strip().split()[0])*1024
    return dict(available_memory_bytes=mem.get('MemAvailable',0),free_work_bytes=shutil.disk_usage(root).free,
        free_system_bytes=shutil.disk_usage('/').free)


def case_command(root,out,case,spec):
    folder=Path(out)/'cases'/case['name'];ctx=SCENARIOS[case['context']]
    return ['bash',str(root/'scripts/orin/run-six-ue-cuda-nr.sh'),'--model','fast','--fast-profile','system',
        '--duration-s',str(case['duration_s']),'--seed',str(case['seed']),'--traffic-load',str(ctx['traffic_load']),
        '--urllc-scenario',ctx['urllc_scenario'],'--ue-services','drx-mec','--xapp-pool','chongyu',
        '--ap-services-profile',str(root/'configs/rt_sys/ap_services_reference.json'),'--record-compression','gzip',
        '--perturbation-spec',str(spec),'--channel-cache-root',str(root/'channel_cache/context_identification'/str(case['seed'])/str(case['duration_s'])),
        '--replay-max-frames','100','--live-state',str(folder/'live.json'),'--output',str(folder)]


def worker(plan_path):
    plan_path=Path(plan_path).resolve();out=plan_path.parent;root=ROOT
    if not out.is_relative_to(root/'outputs')or not(root/'.smo-owned-worker.json').exists():raise ValueError('Owned worker output required')
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=load(plan_path);fingerprint=source_hash(root)
    if plan['source_sha256']!=fingerprint:raise ValueError('Frozen source differs')
    for name,d in plan['frozen_models'].items():
        if sha(root/name)!=d:raise ValueError('Frozen model differs')
    status_path=out/'study_status.json';status=dict(worker_id=plan['worker_id'],done=[],failed=[],running=[],started_unix_s=time.time(),phase='waiting_for_existing_work',completed=False)
    (out/'checkpoints').mkdir(exist_ok=True);(out/'cases').mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    env=os.environ.copy();env.update(XDG_CACHE_HOME=str(root/'.cache'),TMPDIR=str(root/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(root/'.cache/torchinductor'),CUDA_CACHE_MAXSIZE=str(128*1024**2))
    (root/'tmp').mkdir(exist_ok=True)
    wall_per_sim=[];seen=set();failures=set()
    while True:
        plan=load(plan_path);status.update(updated_unix_s=time.time(),free=resources(root),source_sha256=fingerprint)
        deadline=plan['deadline_unix_s'];remaining=deadline-time.time()
        if plan.get('stop')or remaining<plan.get('analysis_reserve_s',1200):
            status.update(phase='budget_complete',completed=True);atomic(status_path,status);break
        entries=[e for e in plan['cases']if e['name']not in seen and e['name']not in failures]
        if not (out/'ADMISSION_ALLOWED').exists()or external_simulations():
            status.update(phase='waiting_for_existing_work',external_process_count=len(external_simulations()),queued=len(entries));atomic(status_path,status);time.sleep(5);continue
        if not entries:
            status.update(phase='waiting_for_plan'if not plan.get('sealed')else'complete',completed=bool(plan.get('sealed')),queued=0);atomic(status_path,status)
            if plan.get('sealed'):break
            time.sleep(5);continue
        free=status['free']
        if free['available_memory_bytes']<8*1024**3 or free['free_work_bytes']<5*1024**3 or free['free_system_bytes']<256*1024**2:
            status.update(phase='waiting_for_storage',queued=len(entries));atomic(status_path,status);time.sleep(10);continue
        case=entries[0]
        estimate=max(120.,statistics.median(wall_per_sim[-8:])*case['duration_s']*1.35)if wall_per_sim else 600.
        if remaining<plan.get('analysis_reserve_s',1200)+estimate:
            status.update(phase='budget_complete',completed=True,unstarted=len(entries));atomic(status_path,status);break
        name=case['name'];folder=out/'cases'/name
        if is_completed(out,case):seen.add(name);status['done'].append(name);continue
        if folder.exists():
            partial=out/'partial';partial.mkdir(exist_ok=True);folder.rename(partial/(name+'_'+str(time.time_ns())))
        spec=out/'specs'/(case['arm']+'.json')
        if load(spec)!=plan['specs'][case['arm']]:raise ValueError('Stimulus changed')
        logpath=out/'logs'/(name+'.log');started=time.time()
        with logpath.open('wb')as log:
            proc=subprocess.Popen(case_command(root,out,case,spec),cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            state={**case,'pid':proc.pid,'started_unix_s':started,'slot':0};status.update(phase='running',running=[state],queued=len(entries)-1);atomic(status_path,status)
            while proc.poll()is None:
                try:
                    live=load(folder/'live.json');state['slot']=live.get('slot',0);state['stage']=live.get('stage')
                except (OSError,ValueError):pass
                status.update(updated_unix_s=time.time(),free=resources(root));atomic(status_path,status)
                if time.time()>deadline-180:
                    os.killpg(proc.pid,signal.SIGINT)
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM)
                time.sleep(2)
            code=proc.wait()
        elapsed=time.time()-started;state.update(exit_code=code,wall_s=elapsed)
        success=False
        if code==0:
            run=load(folder/'run_status.json')
            success=bool(run.get('completed')and run.get('replay',{}).get('audits_passed')and run.get('source_sha256')==fingerprint)
        if success:
            seal=dict(completed=True,case=case,worker_id=plan['worker_id'],source_sha256=fingerprint,finished_unix_s=time.time(),wall_s=elapsed,
                files_sha256={str(p.relative_to(folder)):sha(p)for p in folder.rglob('*')if p.is_file()})
            atomic(out/'checkpoints'/(name+'.json'),seal);seen.add(name);status['done'].append(name);wall_per_sim.append(elapsed/case['duration_s'])
        else:
            failures.add(name);status['failed'].append(state)
        status.update(running=[],completed=False,wall_per_sim_s=statistics.median(wall_per_sim[-8:])if wall_per_sim else None)
        atomic(status_path,status)
    status['models_unchanged']=all(sha(root/name)==d for name,d in plan['frozen_models'].items());atomic(status_path,status)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-plan',type=Path);a=p.parse_args()
    if not a.worker_plan:p.error('--worker-plan required')
    worker(a.worker_plan)


if __name__=='__main__':main()

"""Small matched six-UE intervention sweep; no training or model optimization.

Two seeds × eleven fixed arms, 0.8s each. Every arm has a common native warmup,
100ms wash-in, a 500ms scoring window and 100ms terminal follow-up. The main
simulator, native PHY/traffic/deadlines and normal RIC execution stay intact.
"""
from copy import deepcopy
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.rcp_perturbation import SCHEMA,validate,vector

REFERENCE=dict(schema=SCHEMA,arm='native',switch_ms=100,scheduler_mode='round_robin',
               offload_ue4=.5,drx_ue4_inactivity_ms=8.,weight_ue4=1.,weight_ue6=1.,cell_sleep_mode='active')
ARMS={
    'native':{}, 'offload_low':dict(offload_ue4=.4), 'offload_high':dict(offload_ue4=.6),
    'drx_short':dict(drx_ue4_inactivity_ms=4.), 'ap_micro':dict(cell_sleep_mode='micro_sleep'),
    'latency_native':dict(scheduler_mode='latency_aware'), 'qos_native':dict(scheduler_mode='qos_deadline'),
    'weight_mec_low':dict(scheduler_mode='latency_aware',weight_ue4=.75),
    'weight_mec_high':dict(scheduler_mode='latency_aware',weight_ue4=1.25),
    'weight_urllc_low':dict(scheduler_mode='qos_deadline',weight_ue6=.75),
    'weight_urllc_high':dict(scheduler_mode='qos_deadline',weight_ue6=1.25),
}
CONTRASTS=[
    dict(id='offload_ue4',parameter='ue_offload_fraction',low='offload_low',high='offload_high',
         reference='native',step=.2,kind='local_derivative',scope='UE4 only; all other RCPs fixed',
         context='native round_robin; p4=.5±.1; native DRX; active APs'),
    dict(id='drx_ue4',parameter='ue_drx_configs',low='native',high='drx_short',reference='native',
         step=1.,kind='finite_contrast',scope='UE4 only; DRX inactivity_ms 8→4, other DRX fields fixed',
         context='native round_robin; p4=.5; active APs; experimental DRX unlock, deployed X01 remains locked',
         **{'from':{'ue_4':{'inactivity_ms':8.}},'to':{'ue_4':{'inactivity_ms':4.}}}),
    dict(id='ap_sleep',parameter='cell_sleep_mode',low='native',high='ap_micro',reference='native',
         step=1.,kind='finite_contrast',scope='All three cells: active→micro_sleep',
         context='native round_robin; native DRX; p4=.5; micro profile 1ms sleep per 20ms',
         **{'from':'active','to':'micro_sleep'}),
    dict(id='scheduler_latency',parameter='scheduler_mode',low='native',high='latency_native',reference='native',
         step=1.,kind='finite_contrast',scope='All three cells; all UE weights held at 1',
         context='round_robin→latency_aware; native DRX; p4=.5; active APs',
         **{'from':'round_robin','to':'latency_aware'}),
    dict(id='scheduler_qos',parameter='scheduler_mode',low='native',high='qos_native',reference='native',
         step=1.,kind='finite_contrast',scope='All three cells; all UE weights held at 1',
         context='round_robin→qos_deadline; native DRX; p4=.5; active APs',
         **{'from':'round_robin','to':'qos_deadline'}),
    dict(id='weight_mec_ue4',parameter='ue_scheduler_weights',low='weight_mec_low',high='weight_mec_high',
         reference='latency_native',step=.5,kind='local_derivative',scope='UE4 weight only; other UE weights 1',
         context='fixed latency_aware; w4=1±.25; native DRX; p4=.5; active APs'),
    dict(id='weight_urllc_ue6',parameter='ue_scheduler_weights',low='weight_urllc_low',high='weight_urllc_high',
         reference='qos_native',step=.5,kind='local_derivative',scope='UE6 weight only; other UE weights 1',
         context='fixed qos_deadline; w6=1±.25; native DRX; p4=.5; active APs'),
]


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def load(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(root=ROOT):
    result=hashlib.sha256()
    for name in ('core','phy','ran','ric','state','models','traffic','experiments','configs'):
        for path in sorted((Path(root)/name).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.json','.yaml','.yml'):
                result.update(str(path.relative_to(root)).encode());result.update(path.read_bytes())
    return result.hexdigest()


def build_plan(output,seeds=(20261116,20261117),duration=.8,root=ROOT,*,population=False):
    root=Path(root).resolve();output=Path(output).resolve()
    if not output.is_relative_to(root/'outputs') or output==root/'outputs':
        raise ValueError('Use a dedicated project outputs/ directory')
    if len(seeds)<2 or len(seeds)!=len(set(seeds)) or any(type(s)is not int or not 0<=s<2**32 for s in seeds):
        raise ValueError('At least two distinct valid seeds are required')
    if not .8<=duration<=1.6 or not math.isclose(duration/.1,round(duration/.1),abs_tol=1e-9):
        raise ValueError('Short pilot uses .8–1.6 seconds in 100ms cohorts')
    reference,arms,contrasts=REFERENCE,ARMS,CONTRASTS
    if population:
        from experiments.population_rcp_design import design
        reference,arms,contrasts=design(root)
    specs={name:validate({**reference,**changes,'arm':name})for name,changes in arms.items()}
    # Each contrast changes exactly the intended scalar/structured RCP. The
    # scheduler family is held fixed within each numerical-weight derivative.
    for c in contrasts:
        low=vector(specs[c['low']],perturbed=True);high=vector(specs[c['high']],perturbed=True)
        if {k for k in low if low[k]!=high[k]}!={c['parameter']}:
            raise ValueError('A contrast changed an unrelated control: '+c['id'])
    cases=[]
    for arm in arms:
        for seed in seeds:
            name=f's{seed}_{arm}';spec_path=output/'specs'/(arm+'.json')
            command=['bash','scripts/orin/run-six-ue-cuda-nr.sh','--model','fast','--duration-s',str(duration),
                '--seed',str(seed),'--traffic-load','1','--ue-services','drx-mec','--xapp-pool','chongyu',
                '--ap-services-profile',str(root/'configs/rt_sys/ap_services_reference.json'),
                '--record-compression','gzip','--perturbation-spec',str(spec_path),
                '--channel-cache-root',str(root/'channel_cache'/('population_pilot'if population else'rcp_pilot')/str(seed)),
                '--output',str(output/'cases'/name)]
            cases.append(dict(name=name,seed=seed,arm=arm,output=str(output/'cases'/name),command=command))
    release=load(root/'artifacts/xapps/chongyu/20261109/release.json')
    return dict(schema='oran.rcp-perturbation-plan.v1',source_sha256=source_hash(root),
        project_root=str(root),output=str(output),seeds=list(seeds),duration_s=duration,slots=round(duration*2000),
        switch_ms=100,scoring_start_ms=200,scoring_end_ms=(duration-.1)*1000,
        followup_ms=100,specs=specs,contrasts=contrasts,cases=cases,population=population,
        frozen_models={str(root/'artifacts/xapps/chongyu/20261109'/e['path']):e['sha256']for e in release['models'].values()},
        physical_model='unchanged accepted fast system model; moving native six UEs; depth-2 RT; 1x traffic',
        identification='paired common exogenous seed/geometry/traffic; one RCP change per contrast; no algorithm tuning',
        inference='two-seed local pilot; no global causality, mature-SLA or asymptotic-significance claim')


def verify(plan):
    if source_hash()!=plan['source_sha256']:
        raise ValueError('Simulator source changed during the preregistered sweep')
    for path,digest in plan['frozen_models'].items():
        if sha(path)!=digest:raise ValueError('A trained model changed: '+path)
    for arm,spec in plan['specs'].items():
        if load(Path(plan['output'])/'specs'/(arm+'.json'))!=spec:
            raise ValueError('A fixed intervention spec changed')


def completed(case):
    p=Path(case['output'])
    if not(p/'run_status.json').exists():return False
    s=load(p/'run_status.json')
    return bool(s.get('completed')and s.get('replay',{}).get('audits_passed')and(p/'final_summary.json').is_file())


def run(plan,parallel=2,min_free_gb=.5):
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'sweep.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue=list(plan['cases']);active={};done=[];failed=[];started=time.time()
    while queue or active:
        verify(plan)
        if plan.get('requires_bridge') and (out/'bridge_failed.json').exists():
            failed.append(dict(reason='cross_device_bridge_failed'));queue=[]
        for name,item in list(active.items()):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();case=item['case']
            if code==0 and completed(case):
                done.append(name)
                files={str(p.relative_to(case['output'])):sha(p)for p in Path(case['output']).rglob('*')if p.is_file()}
                atomic(out/'checkpoints'/(name+'.json'),dict(completed=True,case=case,files_sha256=files,finished_unix_s=time.time()))
            else:failed.append(dict(name=name,exit_code=code))
            del active[name]
        if failed:
            # Stop admitting new work; let already-running cases finish and keep
            # every partial output. A retry needs an explicit --resume.
            queue=[]
        while queue and len(active)<parallel:
            if plan.get('requires_bridge') and queue[0].get('role')!='bridge':
                gate=out/'bridge_verified.json'
                if not gate.exists():break
                if load(gate).get('source_sha256')!=plan['source_sha256']:
                    raise ValueError('Bridge verification belongs to another source version')
            case=queue.pop(0);name=case['name'];folder=Path(case['output'])
            if completed(case):done.append(name);continue
            if shutil.disk_usage(out).free < (min_free_gb+.08*max(1,len(active)+1))*1024**3:
                failed.append(dict(name=name,reason='disk_reserve'));queue=[];break
            if folder.exists():
                partial=out/'partial'/(name+'_'+str(time.time_ns()));partial.parent.mkdir(exist_ok=True)
                folder.rename(partial)
            logs=out/'logs';logs.mkdir(exist_ok=True)
            log=(logs/(name+'.log')).open('a')
            process=subprocess.Popen(case['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=dict(process=process,log=log,case=case)
        running=[]
        for name,item in active.items():
            p=Path(item['case']['output'])/'run_status.json'
            status=load(p)if p.exists()else{}
            running.append(dict(name=name,pid=item['process'].pid,stage=status.get('stage','starting'),slot=status.get('last_completed_slot',0)))
        atomic(out/'study_status.json',dict(completed=len(done)==len(plan['cases'])and not failed,
            done=done,failed=failed,running=running,queued=len(queue),total=len(plan['cases']),
            waiting_for_bridge=bool(plan.get('requires_bridge')and queue and queue[0].get('role')!='bridge'and not(out/'bridge_verified.json').exists()),
            updated_unix_s=time.time(),wall_s=time.time()-started,free_bytes=shutil.disk_usage(out).free))
        if queue or active:time.sleep(3)
    verify(plan)
    if failed:raise RuntimeError('Sweep stopped; all records/checkpoints retained: '+str(failed))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[20261116,20261117])
    p.add_argument('--duration-s',type=float,default=.8)
    p.add_argument('--parallel',type=int,choices=(1,2,3),default=2)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--population',action='store_true',help='Perturb complete applicable UE groups, preserving native service-class membership')
    a=p.parse_args();output=a.output.resolve();plan_path=output/'plan.json'
    if plan_path.exists():
        if not a.resume:raise ValueError('Existing study: use --resume to retain/continue its records')
        plan=load(plan_path);verify(plan)
    else:
        plan=build_plan(output,tuple(a.seeds),a.duration_s,population=a.population)
        output.mkdir(parents=True,exist_ok=True)
        for arm,spec in plan['specs'].items():atomic(output/'specs'/(arm+'.json'),spec)
        atomic(plan_path,plan)
    print(json.dumps(dict(cases=len(plan['cases']),duration_s=plan['duration_s'],
        simulated_seconds=len(plan['cases'])*plan['duration_s'],output=str(output),source_sha256=plan['source_sha256'])),flush=True)
    if not a.prepare_only:run(plan,a.parallel)


if __name__=='__main__':main()

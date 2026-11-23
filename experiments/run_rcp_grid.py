"""Range-covering population probes with explicit reuse of compatible data."""
from copy import deepcopy
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.population_rcp_design import design
from experiments.rcp_perturbation import validate,vector
from experiments.run_rcp_perturbation import atomic,load,sha,source_hash,verify,run

DESIGN_ONLY={'rcp_perturbation.py','run_rcp_perturbation.py','population_rcp_design.py','run_rcp_grid.py'}


def dynamics_signature(root):
    """Include runtime/PHY/config code; isolate only accepted-range/design edits.

    Exact ASTs of stimulus execution/membership and constants remain pinned.
    Validation is also enforced against every executed vector by the auditor.
    """
    root=Path(root);h=hashlib.sha256()
    for name in ('core','phy','ran','ric','state','models','traffic','configs','experiments','scripts/orin'):
        for p in sorted((root/name).rglob('*')):
            if p.is_file()and p.suffix in('.py','.json','.yaml','.yml','.sh'):
                if name=='experiments'and p.name in DESIGN_ONLY:continue
                h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    tree=ast.parse((root/'experiments/rcp_perturbation.py').read_text())
    keep={'population_memberships','vector','configure','build_controller'}
    constants={'SCHEMA','POPULATION_SCHEMA','USERS','FIELDS','DRX'}
    for node in tree.body:
        selected=isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name in keep
        selected|=isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in constants for t in node.targets)
        if selected:h.update(ast.dump(node,include_attributes=False).encode())
    return h.hexdigest()


def grid_design(root):
    reference,base,contrasts=design(root)
    grids=[
        dict(id='offload_population_range',parameter='ue_offload_fraction',field='offload_probability',reference=.5,
             levels=[0.,.2,.4,.5,.6,.8,1.],mode='round_robin',component='mec_workload',x_unit='fraction',
             scope='All MEC task sources; one common offloading probability'),
        dict(id='drx_population_range',parameter='ue_drx_configs',field='drx_inactivity_ms',reference=8.,
             levels=[0.,2.,4.,8.,16.,24.,32.,40.],mode='round_robin',component='drx_enabled_population',x_unit='ms',scalar_component='inactivity_ms',
             scope='All DRX-enabled UEs; one common inactivity timer; other timers/native disabled profiles fixed',
             domain_note='Operational calibration domain: 0–40 ms, one current DRX cycle; not a model hard maximum.'),
        dict(id='weight_mec_population_range',parameter='ue_scheduler_weights',field='mec_weight',reference=1.,
             levels=[.5,.75,1.,1.25,1.5,1.75,2.],mode='latency_aware',component='mec_priority',x_unit='weight',
             scope='Entire MEC-workload group; common relative priority; other traffic weights fixed at 1'),
        dict(id='weight_urllc_population_range',parameter='ue_scheduler_weights',field='urllc_weight',reference=1.,
             levels=[.5,.75,1.,1.25,1.5,2.,3.,4.],mode='qos_deadline',component='urllc_priority',x_unit='weight',
             scope='Entire URLLC traffic group; common relative priority; other traffic weights fixed at 1')]
    specs={name:validate({**reference,**base[name],'arm':name})for name in('native','latency_native','qos_native','ap_micro')}
    vectors={json.dumps(vector(s,perturbed=True),sort_keys=True):name for name,s in specs.items()}
    for g in grids:
        g['arms']=[]
        for x in g['levels']:
            name=g['field']+'_'+str(x).replace('.','p')
            s=validate({**reference,'arm':name,'scheduler_mode':g['mode'],g['field']:x})
            key=json.dumps(vector(s,perturbed=True),sort_keys=True)
            if key in vectors:name=vectors[key]
            else:specs[name]=s;vectors[key]=name
            g['arms'].append(name)
        center=vector(specs[g['arms'][g['levels'].index(g['reference'])]],perturbed=True)
        for name in g['arms']:
            v=vector(specs[name],perturbed=True)
            if {k for k in center if center[k]!=v[k]}not in(set(),{g['parameter']}):
                raise ValueError('Grid changed an unrelated RCP')
    return specs,grids,[c for c in contrasts if c['id']in('ap_sleep','scheduler_latency','scheduler_qos')]


def build_plan(output,seeds,*,reuse=None,bridge_seed=None):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'outputs')or output==ROOT/'outputs':raise ValueError('Use project outputs/')
    if not seeds or len(seeds)!=len(set(seeds))or any(type(s)is not int or not 0<=s<2**32 for s in seeds):raise ValueError('Invalid worker seeds')
    specs,grids,contrasts=grid_design(ROOT);cases=[];reused=[]
    signature=dynamics_signature(ROOT);old=None
    if reuse:
        reuse=Path(reuse).resolve();old=load(reuse/'plan.json');old_root=Path(old['project_root'])
        if source_hash(old_root)!=old['source_sha256']:raise ValueError('Original simulator source changed')
        if dynamics_signature(old_root)!=signature:raise ValueError('Runtime dynamics differ; do not reuse old cases')
        if not load(reuse/'study_status.json').get('completed'):raise ValueError('Reuse requires a completed checkpointed study')
        for path,digest in old['frozen_models'].items():
            if sha(path)!=digest:raise ValueError('A reused trained model changed')
    old_vectors={}
    if old:
        for c in old['cases']:
            key=(c['seed'],json.dumps(vector(old['specs'][c['arm']],perturbed=True),sort_keys=True))
            old_vectors[key]=c
    def append(seed,arm,role='grid'):
        name=('bridge_'if role=='bridge'else'')+f's{seed}_{arm}'
        key=(seed,json.dumps(vector(specs[arm],perturbed=True),sort_keys=True))
        if role=='grid'and key in old_vectors:
            c=old_vectors[key]
            reused.append(dict(name=name,seed=seed,arm=arm,origin_case=c,origin_study=str(reuse),origin_plan_sha256=sha(reuse/'plan.json')))
            return
        command=['bash','scripts/orin/run-six-ue-cuda-nr.sh','--model','fast','--duration-s','.8','--seed',str(seed),
            '--traffic-load','1','--ue-services','drx-mec','--xapp-pool','chongyu','--ap-services-profile',str(ROOT/'configs/rt_sys/ap_services_reference.json'),
            '--record-compression','gzip','--perturbation-spec',str(output/'specs'/(arm+'.json')),
            '--channel-cache-root',str(ROOT/'channel_cache/grid_pilot'/str(seed)),'--output',str(output/'cases'/name)]
        cases.append(dict(name=name,seed=seed,arm=arm,role=role,output=str(output/'cases'/name),command=command))
    if bridge_seed is not None:
        for arm in('native','latency_native','qos_native'):append(bridge_seed,arm,'bridge')
    for arm in specs:
        for seed in seeds:append(seed,arm)
    release=load(ROOT/'artifacts/xapps/chongyu/20261109/release.json')
    return dict(schema='oran.rcp-grid-plan.v1',source_sha256=source_hash(),dynamics_sha256=signature,population=True,
        project_root=str(ROOT),output=str(output),seeds=list(seeds),duration_s=.8,slots=1600,switch_ms=100,
        scoring_start_ms=200,scoring_end_ms=700,followup_ms=100,specs=specs,grids=grids,contrasts=contrasts,cases=cases,reused=reused,
        reuse_source_sha256=old['source_sha256']if old else None,
        requires_bridge=bridge_seed is not None,
        frozen_models={str(ROOT/'artifacts/xapps/chongyu/20261109'/e['path']):e['sha256']for e in release['models'].values()},
        inference='Seed-blocked first-order range fit; preserve missing outcomes and nonlinear residuals; no slots-as-independent-samples')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--reuse',type=Path);p.add_argument('--bridge-seed',type=int)
    p.add_argument('--parallel',type=int,choices=(1,2,3),default=2);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();path=a.output.resolve()/'plan.json'
    if path.exists():
        if not a.resume:raise ValueError('Use --resume; existing data will be retained')
        plan=load(path);verify(plan)
    else:
        plan=build_plan(a.output,a.seeds,reuse=a.reuse,bridge_seed=a.bridge_seed)
        for arm,s in plan['specs'].items():atomic(path.parent/'specs'/(arm+'.json'),s)
        atomic(path,plan)
    print(json.dumps(dict(new_cases=len(plan['cases']),reused_cases=len(plan['reused']),levels=[len(g['levels'])for g in plan['grids']],source_sha256=plan['source_sha256'],dynamics_sha256=plan['dynamics_sha256'])),flush=True)
    if not a.prepare_only:run(plan,parallel=a.parallel)


if __name__=='__main__':main()

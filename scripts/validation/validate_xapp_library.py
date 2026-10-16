"""Training-free library acceptance. All evidence and caches stay on Spark.

Short response fixtures are intentionally controlled, not deployment capacity
results. Sleep, handover and MRO require longer than 64 ms to exercise their
unchanged timing rules; only those short fixtures are extended.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Event,Lock
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_name,'1')
STOP=Event();CHILDREN=set();LOCK=Lock()

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from ric.near_rt.xapp_library import IMPLEMENTATIONS
from ric.near_rt.rule_based_v1.catalog import PROFILES
from scripts.validation.validate_rule_based import inventory,write
from scripts.validation.validate_efficiency import audit,read,equivalent
from phy.rt_sys.contracts import source_fingerprint

SCENES=('simple_street_canyon','munich')


def short_slots(category):
    return {'cell_sleep':640,'traffic_steering':320,'load_balancing':768,'mobility_robustness':3800}.get(category,128)


def worker(a):
    from experiments import run_rt_sys as runner
    category=IMPLEMENTATIONS[a.implementation]['category_id'] if a.implementation else None
    original=runner.build_case
    def fixture(*values,**kwargs):
        manifest,c,intent=original(*values,**kwargs)
        if not a.fixture:return manifest,c,intent
        c['sim']['measurement_sinr_noise_std_db']=0
        if category!='beam_management':
            for u in manifest['ues']:u['trajectory']=deepcopy(manifest['ues'][0]['trajectory'])
            for u in c['ue']['ues']:
                u.update(dict(zip(('x_m','y_m','z_m'),manifest['ues'][0]['trajectory']['positions_m'][0])))
                u.update(priority=1,queue_bytes=20000,arrival_model='cbr',arrival_rate_bps=100000,logical_channels=[])
        if category in ('traffic_steering','load_balancing','mobility_robustness','interference_coordination'):
            p=deepcopy(manifest['cells'][1])
            manifest['cells'][0]={**p,'cell_id':'cell_0'}
            c['cell']['neighbor_cell_positions']={x['cell_id']:dict(zip(('x_m','y_m','z_m'),x['position_m'])) for x in manifest['cells']}
            c['cell']['handover_margin_db']=20 if category in ('traffic_steering','interference_coordination') else .05
            if category=='mobility_robustness':
                c['phy']['rt_sys']['paper_interfaces']['connection_model'].update(good_ms=400,handover_timeout_ms=1000)
                c['cell']['cell_radio_profiles']['cell_1']['tx_power_dbm']=29.
        if category=='rf_energy':
            for profile in c['cell']['cell_radio_profiles'].values():profile['tx_power_dbm']=29.
        if category in ('slice_sla','latency_guard','ue_fairness'):
            c['sim'].update(scheduler_max_ues_per_slot=1,scheduler_enable_spatial_multiplexing=False)
            for j,u in enumerate(c['ue']['ues']):
                u.update(slice_id='urllc' if j==0 else 'embb',latency_budget_ms=1 if j==0 and category=='latency_guard' else 100,
                    queue_bytes=40000 if j and category=='latency_guard' else 20000)
        return manifest,c,intent
    runner.build_case=fixture
    config=ROOT/'configs/rt_sys/xapp_library'/((a.implementation.replace(':','__') if a.implementation else a.profile)+'.json')
    data=read(config)
    if a.arm=='reference':data['xapp_library']['bindings']={}
    if a.arm=='legacy':
        data.pop('xapp_library');data['rule_based_v1']=dict(enabled=True,monitor=True,applications=[category],parameters={})
    snapshot=a.output.parent/(a.output.name+'_input.json');write(snapshot,data)
    timeline=[];policy=[]
    if a.fixture:
        if category in ('slice_sla','latency_guard'):
            timeline=[dict(slot=5,config=dict(target_cell_id='cell_1',ue_prb_limits={f'ue_{i}':2 for i in range(a.ues)}))]
        if category=='traffic_steering':timeline=[dict(slot=20,config=dict(target_cell_id='cell_1',tx_power_dbm=33.))]
        if category=='mobility_robustness':
            timeline=[dict(slot=5,config=dict(apply_scope='cluster',ue_prb_limits={f'ue_{i}':0 for i in range(a.ues)})),
                dict(slot=10,commands=[dict(command_type='request_handover',target_ue_id=f'ue_{i}',target_cell_id='cell_1') for i in range(a.ues)]),
                dict(slot=20,config=dict(target_cell_id='cell_0',tx_power_dbm=-100.)),
                dict(slot=2410,config=dict(target_cell_id='cell_0',tx_power_dbm=30.)),
                dict(slot=2420,commands=[dict(command_type='request_handover',target_ue_id=f'ue_{i}',target_cell_id='cell_0') for i in range(a.ues)]),
                dict(slot=3450,config=dict(target_cell_id='cell_1',tx_power_dbm=33.))]
    if a.profile:
        kinds=PROFILES[a.profile]
        known=[f'cell_{i}' for i in range(3)]
        schedule=((20,1),(80,2)) if a.slots<1000 else ((1600,1),(3200,2))
        for slot,index in schedule:
            bindings={cid:{} for cid in known}
            for category in kinds:
                choices=[i for i,s in IMPLEMENTATIONS.items() if s['category_id']==category]
                for j,cid in enumerate(known):bindings[cid][category]=choices[(index+j)%len(choices)].replace(':','__')
            policy.append(dict(slot=slot,xapp_bindings=bindings))
    argv=['--config',str(snapshot),'--scene',a.scene,'--cells','3','--ues',str(a.ues),'--slots',str(a.slots),'--seed','512',
        '--initial-queue-bytes','0','--ue-speed-mps','0' if a.fixture else '1.5',
        '--warmup-ms','0' if a.slots<1000 or a.fixture else '200','--control-mode','orchestrated',
        '--nr-execution','batched','--feedback-mode','nr_direct','--output',str(a.output)]
    if timeline:
        path=a.output.parent/(a.output.name+'_timeline.json');write(path,timeline);argv+=['--control-timeline',str(path)]
    if policy:
        path=a.output.parent/(a.output.name+'_policy.json');write(path,policy);argv+=['--policy-timeline',str(path)]
    if a.cold:argv+=['--channel-cache-root',str(a.output.parent/'cold_channel_cache'/a.scene/str(a.ues))]
    sys.argv=[sys.argv[0],*argv];runner.run_case(runner.parse_args())


def rows(path):
    with path.open() as stream:
        for line in stream:yield json.loads(line)


def execute(root,scene,ident,arm,slots,ues=4,fixture=False,profile=None,cold=False):
    if STOP.is_set():raise RuntimeError('Stopped after severe validation failure')
    label=f'{scene}_{(ident or profile).replace(":","__")}_{arm}_{ues}_{slots}'
    folder=root/label
    if (folder/'acceptance.json').exists():
        result=read(folder/'acceptance.json');assert result['source_sha256']==source_fingerprint(ROOT);return result
    if folder.exists():raise FileExistsError('Partial run retained: '+str(folder))
    cmd=[sys.executable,__file__,'--worker','--scene',scene,'--arm',arm,'--slots',str(slots),'--ues',str(ues),'--output',str(folder)]
    if ident:cmd+=['--implementation',ident]
    if fixture:cmd+=['--fixture']
    if profile:cmd+=['--profile',profile]
    if cold:cmd+=['--cold']
    start=time.perf_counter()
    with (root/(label+'.log')).open('w') as log:
        with subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT) as child:
            with LOCK:CHILDREN.add(child)
            try:
                if STOP.is_set():child.terminate()
                code=child.wait()
                if code:raise RuntimeError(f'{label} exited {code}; see process log')
            finally:
                with LOCK:CHILDREN.discard(child)
    result=audit(folder);result.update(path=str(folder),scene=scene,implementation_id=ident,profile=profile,ues=ues,cold_timing=cold,
        process_wall_s=time.perf_counter()-start,performance=read(folder/'performance.json'))
    if profile and slots<1000:
        applied=[r for r in rows(folder/'xapp_selection.jsonl') if r.get('status')=='applied' and r.get('slot_id',0)>=20]
        assert applied,'No real non-RT binding transitions'
        result['selection_transitions']=len(applied)
    write(folder/'acceptance.json',result);print('STRUCTURAL PASS',label,round(result['process_wall_s'],2),flush=True)
    return result


def response(reference,candidate,ident):
    a,b=Path(reference['path']),Path(candidate['path'])
    assert read(a/'initial_conditions.json')['environment_sha256']==read(b/'initial_conditions.json')['environment_sha256']
    requests=[r for r in rows(b/'rule_decisions.jsonl') if r.get('implementation_id')==ident and r.get('reason') in ('action_requested','command_requested')]
    changes=[]
    for r in rows(b/'control_receipts.jsonl'):
        if r.get('status')!='applied':continue
        for cid,after in r.get('after',{}).items():
            before=r['before'][cid]
            for k,value in after.items():
                action_field={'current_codebook_id':'codebook_id','current_beam_profile_id':'beam_profile_id'}.get(k,k)
                if value!=before.get(k) and any(p.get('implementation_id')==ident and action_field in p.get('fields',[]) for p in r.get('provenance',[])):
                    changes.append(dict(slot_id=r['slot_id'],cell_id=cid,field=k,before=before.get(k),after=value))
        if r.get('kind')=='command' and r.get('source')==ident.replace(':','__'):changes.append(r)
    affected=[]
    for x,y in zip(rows(a/'rt_sys_trace.jsonl'),rows(b/'rt_sys_trace.jsonl')):
        for u,v in zip(x['ues'],y['ues']):
            for k in ('delivered_bytes','queue_bytes','resource_allocation','reference_rsrp_dbm','reference_sinr_db'):
                left,right=u.get(k),v.get(k)
                different=abs(left-right)>1e-4 if isinstance(left,(float,int)) and isinstance(right,(float,int)) else left!=right
                if different:affected.append(dict(slot_id=y['slot_id'],ue_id=v['ue_id'],kpi=k));break
        if x.get('mobility_events')!=y.get('mobility_events'):affected.append(dict(slot_id=y['slot_id'],kpi='mobility_events'))
        for cid,cell in x['cells'].items():
            for key,tolerance in (('average_used_w',1e-10),('occupied_re',0)):
                if abs(cell[key]-y['cells'][cid][key])>tolerance:affected.append(dict(slot_id=y['slot_id'],cell_id=cid,kpi=key))
        if len(affected)>10:break
    # Candidate reference reports matter for sleeping cells even when they had
    # no data transmission (no fictitious whole-device power is introduced).
    final_a=read(a/'final_summary.json');final_b=read(b/'final_summary.json')
    qa=final_a['paper_interfaces']['final_reports']['ue'];qb=final_b['paper_interfaces']['final_reports']['ue']
    if any(qa[u].get('candidate_rsrp_dbm')!=qb[u].get('candidate_rsrp_dbm') for u in qa):affected.append(dict(kpi='candidate_rsrp_dbm'))
    if final_a['paper_interfaces']['events']!=final_b['paper_interfaces']['events']:
        affected.append(dict(kpi='reference_connection_and_handover_events'))
    if any(qa[u].get('mobility_counts')!=qb[u].get('mobility_counts') for u in qa):affected.append(dict(kpi='public_mobility_counts'))
    return dict(implementation_id=ident,scene=candidate['scene'],passed=bool(requests and changes and affected),
        requests=len(requests),executed_changes=changes[:10],kpi_response=affected[:10],
        interpretation='Execution and response only; improvement is not required. Missing triggers are not passes.')


def pair(out,scene,ident):
    spec=IMPLEMENTATIONS[ident];category=spec['category_id'];slots=short_slots(category);ues=5 if category=='mobility_robustness' else 4
    ref=execute(out,scene,ident,'reference',slots,ues,True)
    result=execute(out,scene,ident,'active',slots,ues,True)
    migration=None
    if spec['algorithm']=='legacy_v1':
        legacy=execute(out,scene,ident,'legacy',slots,ues,True)
        migration=dict(implementation_id=ident,scene=scene,**equivalent(legacy['path'],result['path']))
    return [ref,result],response(ref,result,ident),migration


def long_specs():
    singles=[(s,i,'active',8000 if d['category_id']=='ret_coverage' else 4000,4,None) for s in SCENES for i,d in IMPLEMENTATIONS.items()]
    combined=[(s,None,'active',slots,n,p) for s in SCENES for p in PROFILES for n,slots in ((4,4000),(24,1000))]
    return singles+combined


def estimate(cases):
    estimates=[];cold=0.;disk=0;cold_keys=set()
    for scene,ident,arm,slots,ues,profile in long_specs():
        matches=[r for r in cases if r['scene']==scene and (r['implementation_id']==ident if ident else r['profile']==profile and r['ues']==ues)]
        assert matches,'Missing timing gate'
        row=matches[-1];perf=row['performance']
        # Static response fixtures cannot predict mobile RT generation. Use
        # the slower observed dynamic-combination rate as an additional bound.
        dynamic=[r for r in cases if r['scene']==scene and r['profile'] and r['ues']==ues]
        rate=max([max(0,perf['simulation_s']-perf['stages'].get('rt_generation',{}).get('wall_s',0))/row['slots']]+
            [max(0,r['performance']['simulation_s']-r['performance']['stages'].get('rt_generation',{}).get('wall_s',0))/r['slots'] for r in dynamic])
        seconds=perf['initialization_s']+rate*slots
        estimates.append(seconds);disk+=row['output_bytes']*slots/row['slots']
        key=(scene,ues,slots)
        if key not in cold_keys:
            cold_keys.add(key)
            samples=[r for r in cases if r['scene']==scene and r['ues']==ues and r.get('cold_timing')]
            assert samples,'Fresh-cache timing required'
            cold+=max(r['performance']['stages'].get('rt_generation',{}).get('wall_s',0)/r['slots'] for r in samples)*slots
    return dict(workers=1,cases=len(estimates),slots=sum(s[3] for s in long_specs()),
        lower_s=(sum(estimates)+cold)*.8,upper_s=(sum(estimates)+cold)*1.6,estimated_disk_bytes=int(disk*1.5),
        estimated_compute_s=sum(estimates),estimated_unique_cold_rt_s=cold,unique_channel_keys=len(cold_keys),
        cold_cache_note='Fresh 8-slot RT generation extrapolated per unique map/UE/duration key; report separately, with uncertainty. Cached channels are reused across algorithms.')


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if not out.is_relative_to(ROOT/'outputs'):raise ValueError('Raw evidence must remain in Spark outputs')
    sha=source_fingerprint(ROOT);files=inventory(ROOT);start=time.time()
    status=dict(completed=False,passed=False,stage=a.stage,source_sha256=sha,cases=[],responses=[],workers=a.workers,
        timing_note='Concurrent functional gates are not isolated performance measurements; long estimate uses conservative observed rates.')
    write(out/'source_sha256.json',files)
    if not (out/'executed_sources.tar.gz').exists():
        with tarfile.open(out/'executed_sources.tar.gz','w:gz') as archive:
            for name in files:archive.add(ROOT/name,arcname=name)
    try:
        if a.stage=='short':
            if not a.filter:
                for scene in SCENES:
                    for n in (4,24):status['cases'].append(execute(out,scene,'throughput_power:legacy_v1','reference',8,n,False,cold=True))
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                futures=[pool.submit(pair,out,scene,ident) for scene in SCENES for ident in IMPLEMENTATIONS if not a.filter or ident in a.filter]
                try:
                    for future in as_completed(futures):
                        cases,result,migration=future.result()
                        status['cases']+=cases;status['responses'].append(result)
                        if migration:status.setdefault('migration',[]).append(migration)
                        status['elapsed_s']=time.time()-start;write(out/'validation.json',status)
                except BaseException:
                    STOP.set()
                    with LOCK:
                        for child in list(CHILDREN):child.terminate()
                    for future in futures:future.cancel()
                    raise
            if not a.filter:
                with ThreadPoolExecutor(max_workers=a.workers) as pool:
                    futures=[pool.submit(execute,out,scene,None,'active',128,n,False,profile)
                        for scene in SCENES for profile in PROFILES for n in (4,24)]
                    try:
                        for future in as_completed(futures):
                            status['cases'].append(future.result());write(out/'validation.json',status)
                    except BaseException:
                        STOP.set()
                        with LOCK:
                            for child in list(CHILDREN):child.terminate()
                        for future in futures:future.cancel()
                        raise
            status['passed']=all(r['passed'] for r in status['responses']) and bool(status['responses'])
            if not a.filter:write(out/'long_estimate.json',estimate(status['cases']))
        else:
            gate=read(a.gate/'validation.json')
            assert gate['completed'] and gate['passed'] and gate['source_sha256']==sha and len(gate['responses'])==56,'Short response gate not passed'
            for scene,ident,arm,slots,ues,profile in long_specs():
                status['cases'].append(execute(out,scene,ident,arm,slots,ues,False,profile));write(out/'validation.json',status)
                assert source_fingerprint(ROOT)==sha and inventory(ROOT)==files,'Source changed during long run'
            status['passed']=True
        assert inventory(ROOT)==files,'Source changed during acceptance'
        status.update(completed=True,elapsed_s=time.time()-start)
    except BaseException as error:
        status.update(error=f'{type(error).__name__}: {error}',elapsed_s=time.time()-start);raise
    finally:
        write(out/'validation.json',status)
        write(out/'SHA256.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='SHA256.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('short','long'),default='short');p.add_argument('--gate',type=Path)
    p.add_argument('--workers',type=int,choices=(1,2,3),default=1)
    p.add_argument('--filter',nargs='*');p.add_argument('--worker',action='store_true')
    p.add_argument('--scene',choices=SCENES);p.add_argument('--implementation');p.add_argument('--profile',choices=tuple(PROFILES))
    p.add_argument('--arm',choices=('reference','active','legacy'),default='active');p.add_argument('--slots',type=int,default=128)
    p.add_argument('--ues',type=int,default=4);p.add_argument('--fixture',action='store_true')
    p.add_argument('--cold',action='store_true')
    args=p.parse_args();(worker if args.worker else main)(args)

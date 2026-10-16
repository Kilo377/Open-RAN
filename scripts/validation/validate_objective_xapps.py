"""Short-only objective/parallel-control acceptance. Raw artifacts stay on Spark."""
import argparse
from copy import deepcopy
from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Event,Lock
STOP=Event();LOCK=Lock();CHILDREN=set()
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from ric.near_rt.xapp_library import IMPLEMENTATIONS
from ric.near_rt.xapp_library.objectives import OBJECTIVE_ALGORITHMS,SEARCH_ALGORITHMS
from scripts.validation.validate_rule_based import inventory,write
from scripts.validation.validate_efficiency import audit,read
from scripts.validation.validate_xapp_library import response,rows,SCENES
from phy.rt_sys.contracts import source_fingerprint
PAIRS={
 'power':('throughput_power:sum_service_power','throughput_power:weak_user_power'),
 'rf':('rf_energy:rf_efficiency_search','rf_energy:sla_admission_power'),
 'resources':('ue_fairness:sum_service_caps','ue_fairness:starvation_guard_caps'),
 'slices':('slice_sla:sla_count_weights','slice_sla:worst_slice_weights'),
 'steering':('traffic_steering:coverage_margin_steering','traffic_steering:continuity_first_steering'),
 'ret':('ret_coverage:coverage_fraction_grid','ret_coverage:sum_goodput_grid'),
 'beam':('beam_management:sum_goodput_beam','beam_management:sla_count_beam'),
}


def duration(ident):
 k=IMPLEMENTATIONS[ident]['algorithm'];category=IMPLEMENTATIONS[ident]['category_id']
 if k in SEARCH_ALGORITHMS:return 3600 if category=='ret_coverage' else 2100
 return (1600 if k=='pingpong_guard' else 3800) if category=='mobility_robustness' else 640 if category=='traffic_steering' else 128


def worker(a):
 a.output.parent.mkdir(parents=True,exist_ok=True)
 from experiments import run_rt_sys as runner
 ident=a.implementation;category=IMPLEMENTATIONS[ident]['category_id'];algorithm=IMPLEMENTATIONS[ident]['algorithm']
 source=read(ROOT/'configs/rt_sys/xapp_library'/(ident.replace(':','__')+'.json'))
 lib=source['xapp_library'];names=[ident.replace(':','__')]
 known=[f'cell_{i}' for i in range(3)]
 for item in lib['instances']:item['cell_ids']=known
 lib['bindings']={cid:{category:names[0]} for cid in known}
 if a.pair:
  left,right=PAIRS[a.pair];names=[x.replace(':','__') for x in (left,right)]
  lib=dict(enabled=True,monitor=True,selection_mode='conflict_experiment',instances=[dict(instance_id=name,implementation_id=i,cell_ids=known,priority=(10 if j==0 else 20) if a.arm=='swap' else (20 if j==0 else 10)) for j,(i,name) in enumerate(zip((left,right),names))],active_sets={cid:{category:names} for cid in known})
  source['xapp_library']=lib
  if a.arm=='left':lib['active_sets']={cid:{category:names[:1]} for cid in known}
  if a.arm=='right':lib['active_sets']={cid:{category:names[1:]} for cid in known}
 elif a.arm=='reference':lib['bindings']={}
 # Fixed, explicitly controlled response fixture. Production deployment unchanged.
 original=runner.build_case
 def fixture(*args,**kwargs):
  manifest,c,intent=original(*args,**kwargs)
  c['sim']['measurement_sinr_noise_std_db']=0
  if category!='beam_management':
   for u in manifest['ues']:u['trajectory']=deepcopy(manifest['ues'][0]['trajectory'])
   for u in c['ue']['ues']:u.update(dict(zip(('x_m','y_m','z_m'),manifest['ues'][0]['trajectory']['positions_m'][0])))
  for j,u in enumerate(c['ue']['ues']):
   u.update(queue_bytes=12000,arrival_model='cbr',arrival_rate_bps=100000 if category not in ('slice_sla','ue_fairness') else 1000000,logical_channels=[],slice_id='urllc' if j==0 else 'embb',latency_budget_ms=5 if j==0 and algorithm=='deadline_loss_weights' else 100)
  if category in ('slice_sla','ue_fairness'):c['sim'].update(scheduler_max_ues_per_slot=1,scheduler_enable_spatial_multiplexing=False)
  if category in ('traffic_steering','mobility_robustness'):
   p=deepcopy(manifest['cells'][1]);manifest['cells'][0]={**p,'cell_id':'cell_0'}
   c['cell']['neighbor_cell_positions']={x['cell_id']:dict(zip(('x_m','y_m','z_m'),x['position_m'])) for x in manifest['cells']}
   c['cell']['handover_margin_db']=40 if category=='traffic_steering' else .05
   if category=='mobility_robustness':c['cell']['cell_radio_profiles']['cell_1']['tx_power_dbm']=29.
   if category=='mobility_robustness' and algorithm=='failure_guard':c['phy']['rt_sys']['paper_interfaces']['connection_model'].update(good_ms=400,handover_timeout_ms=1000)
  if category=='ret_coverage':
   # Isolate analog response from automatic reassociation, without lowering link thresholds.
   pose=deepcopy(manifest['cells'][1])
   manifest['cells']=[{**deepcopy(pose),'cell_id':cid} for cid in known]
   c['cell']['neighbor_cell_positions']={x['cell_id']:dict(zip(('x_m','y_m','z_m'),x['position_m'])) for x in manifest['cells']}
   for cid,profile in c['cell']['cell_radio_profiles'].items():profile['tx_power_dbm']=30. if cid=='cell_0' else 0.
  if category=='rf_energy':
   for profile in c['cell']['cell_radio_profiles'].values():profile['tx_power_dbm']=29.
  return manifest,c,intent
 runner.build_case=fixture
 # Keep acceptance timing engineering parameters explicit in config snapshots.
 if category=='mobility_robustness':
  for item in lib['instances']:item['parameters']={'pingpong_ms':1000.}
 snapshot=a.output.parent/(a.output.name+'_input.json');write(snapshot,source)
 timeline=[];policy=[]
 if category=='traffic_steering':
  timeline=[dict(slot=20,config=dict(target_cell_id='cell_0',tx_power_dbm=0.))]
 if category=='mobility_robustness':
  timeline=[dict(slot=5,config=dict(apply_scope='cluster',ue_prb_limits={f'ue_{i}':0 for i in range(a.ues)})),dict(slot=10,commands=[dict(command_type='request_handover',target_ue_id=f'ue_{i}',target_cell_id='cell_1') for i in range(a.ues)]),dict(slot=20,config=dict(target_cell_id='cell_0',tx_power_dbm=-100.)),dict(slot=2410,config=dict(target_cell_id='cell_0',tx_power_dbm=30.)),dict(slot=2420,commands=[dict(command_type='request_handover',target_ue_id=f'ue_{i}',target_cell_id='cell_0') for i in range(a.ues)]),dict(slot=3450,config=dict(target_cell_id='cell_1',tx_power_dbm=33.))]
 if category=='mobility_robustness' and algorithm=='pingpong_guard':
  timeline=[dict(slot=5,config=dict(apply_scope='cluster',ue_prb_limits={f'ue_{i}':0 for i in range(a.ues)})),dict(slot=10,commands=[dict(command_type='request_handover',target_ue_id=f'ue_{i}',target_cell_id='cell_1') for i in range(a.ues)]),dict(slot=320,commands=[dict(command_type='request_handover',target_ue_id=f'ue_{i}',target_cell_id='cell_0') for i in range(a.ues)]),dict(slot=700,config=dict(target_cell_id='cell_1',tx_power_dbm=33.))]
 if a.pair and a.arm=='withdraw':policy=[dict(slot=max(20,(a.slots//2//10)*10),xapp_active_sets={cid:{category:names[1:]} for cid in known})]
 argv=['--config',str(snapshot),'--scene',a.scene,'--cells','3','--ues',str(a.ues),'--slots',str(a.slots),'--seed','512','--initial-queue-bytes','0','--ue-speed-mps','0','--warmup-ms','0','--control-mode','orchestrated','--nr-execution','batched','--feedback-mode','nr_direct','--output',str(a.output),'--channel-cache-root',str(a.output.parent/'channel_cache'/a.scene/str(a.ues)/str(a.slots))]
 if timeline:
  path=a.output.parent/(a.output.name+'_timeline.json');write(path,timeline);argv+=['--control-timeline',str(path)]
 if policy:
  path=a.output.parent/(a.output.name+'_policy.json');write(path,policy);argv+=['--policy-timeline',str(path)]
 sys.argv=[sys.argv[0],*argv];runner.run_case(runner.parse_args())


def execute(out,scene,ident,arm,slots,pair=None):
 if STOP.is_set():raise RuntimeError('Short gate stopped after a severe failure')
 label=scene+'_'+(pair or ident.replace(':','__'))+'_'+arm;folder=out/label
 argv=[sys.executable,__file__,'--worker','--scene',scene,'--implementation',ident,'--arm',arm,'--slots',str(slots),'--ues',str(5 if IMPLEMENTATIONS[ident]['category_id']=='mobility_robustness' else 4),'--output',str(folder)]
 if pair:argv+=['--pair',pair]
 start=time.time()
 if not folder.exists():
  with (out/(label+'.log')).open('w') as stream:
   with LOCK:
    if STOP.is_set():raise RuntimeError('Short gate stopped')
    child=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT);CHILDREN.add(child)
   try:code=child.wait()
   finally:
    with LOCK:CHILDREN.discard(child)
  if code:raise RuntimeError(label+' failed; see log')
 result=audit(folder);result.update(path=str(folder),scene=scene,implementation_id=ident,arm=arm,process_wall_s=time.time()-start)
 write(folder/'acceptance.json',result);return result


def single(out,scene,ident):
 n=duration(ident);ref=execute(out,scene,ident,'reference',n);active=execute(out,scene,ident,'active',n)
 result=response(ref,active,ident)
 evaluated=[r for r in rows(Path(active['path'])/'rule_decisions.jsonl') if r.get('reason')=='objective_window_evaluated' and r.get('implementation_id')==ident]
 result['objective_windows']=len(evaluated)
 if IMPLEMENTATIONS[ident]['algorithm'] in SEARCH_ALGORITHMS:result['passed'] &= len(evaluated)>=2
 return dict(scene=scene,implementation_id=ident,cases=[ref,active],response=result,passed=result['passed'])


def combination(out,scene,pair):
 left,right=PAIRS[pair];n=max(duration(left),duration(right));results=[]
 for arm in ('left','right','both','swap','withdraw'):results.append(execute(out,scene,left,arm,n,pair))
 evidence={};ids=[left.replace(':','__'),right.replace(':','__')]
 for result in results:
  folder=Path(result['path']);decisions=list(rows(folder/'rule_decisions.jsonl'))
  arb=[r for r in decisions if r.get('reason')=='request_arbitrated'];by_field={};disagreements=[]
  for row in arb:by_field.setdefault((row['slot_id'],row['cell_id'],row['field']),[]).append(row)
  for key,items in by_field.items():
   if len({json.dumps(x['requested_value'],sort_keys=True) for x in items})>1:disagreements.append(dict(slot_id=key[0],cell_id=key[1],field=key[2]))
  executed={ident:0 for ident in ids}
  for row in rows(folder/'control_receipts.jsonl'):
   if row.get('status')!='applied':continue
   for owner in row.get('provenance',[]):
    ident=owner.get('instance_id')
    if ident not in executed:continue
    changed=row.get('kind')=='command'
    for cid,after in row.get('after',{}).items():
     for field,value in after.items():
      action_field={'current_codebook_id':'codebook_id','current_beam_profile_id':'beam_profile_id'}.get(field,field)
      if value!=row.get('before',{}).get(cid,{}).get(field) and action_field in owner.get('fields',[]):changed=True
    executed[ident]+=bool(changed)
  evidence[result['arm']]=dict(arbitrated=len(arb),selected=sum(r['status']=='selected' for r in arb),overridden=sum(r['status']=='overridden' for r in arb),
   requests={ident:sum(r.get('instance_id')==ident and r['reason'] in ('action_requested','command_requested') for r in decisions) for ident in ids},
   executed_changes=executed,live_request_disagreements=len(disagreements),disagreement_examples=disagreements[:8])
  if result['arm']=='withdraw':
   cutoff=max(20,(n//2//10)*10)
   evidence['withdraw']['late_removed_requests']=sum(r.get('instance_id')==ids[0] and r.get('slot_id',0)>cutoff and r['reason'] in ('action_requested','command_requested','request_arbitrated') for r in decisions)
   selections=list(rows(folder/'xapp_selection.jsonl'))
   evidence['withdraw']['removal_applied']=any(r.get('operation')=='removed' and r.get('instance_id')==ids[0] for r in selections)
 changed=response(results[2],results[3],left)
 separate=bool(evidence['left']['executed_changes'][ids[0]] and evidence['right']['executed_changes'][ids[1]])
 switched=evidence['withdraw']['removal_applied'] and not evidence['withdraw']['late_removed_requests']
 # Different goals need not disagree on every scenario; absence is explicit,
 # never a fabricated conflict/benefit label or a positive conflict-response result.
 disagreement=bool(evidence['both']['live_request_disagreements'])
 return dict(scene=scene,pair=pair,cases=results,arbitration=evidence,priority_kpi_evidence=changed,
  individual_execution_passed=separate,withdrawal_passed=switched,
  priority_response_observed=bool(changed['kpi_response']),request_disagreement_observed=disagreement,
  passed=all(r['passed'] for r in results) and separate and switched,
  interpretation='Selection/execution gate is distinct from observing concurrent disagreement or KPI response. No conflict labels or improvement claims.')


def main(a):
 out=a.output.resolve()
 if not out.is_relative_to(ROOT/'outputs'):raise ValueError('Raw data stays in Spark outputs')
 if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new evidence directory; do not mix source versions')
 out.mkdir(parents=True,exist_ok=True);files=inventory(ROOT);sha=source_fingerprint(ROOT)
 status=dict(completed=False,passed=False,source_sha256=sha,singles=[],combinations=[],long_test_enabled=False)
 write(out/'source_sha256.json',files)
 with tarfile.open(out/'executed_sources.tar.gz','w:gz') as archive:
  for path in files:archive.add(ROOT/path,arcname=path)
 start=time.time()
 try:
  singles=[i for i,s in IMPLEMENTATIONS.items() if s['algorithm'] in OBJECTIVE_ALGORITHMS and (not a.filter or i in a.filter)]
  jobs=[] if a.only_combinations else [('single',scene,i) for scene in a.scenes for i in singles]
  if a.combinations or a.only_combinations:jobs += [('combination',scene,pair) for scene in a.scenes for pair in (a.pairs or PAIRS)]
  with ThreadPoolExecutor(max_workers=a.workers) as pool:
   futures={pool.submit(single if kind=='single' else combination,out,scene,key):kind for kind,scene,key in jobs}
   try:
    for future in as_completed(futures):
     result=future.result();status['singles' if futures[future]=='single' else 'combinations'].append(result)
     status['elapsed_s']=time.time()-start;write(out/'validation.json',status)
     print('RESULT',result.get('implementation_id',result.get('pair')),result['scene'],result['passed'],flush=True)
   except BaseException:
    STOP.set()
    with LOCK:
     for child in CHILDREN:child.terminate()
    for future in futures:future.cancel()
    raise
  assert inventory(ROOT)==files,'Source changed during short test'
  status.update(completed=True,passed=bool(jobs) and all(r['passed'] for k in ('singles','combinations') for r in status[k]),elapsed_s=time.time()-start)
 except BaseException as error:status.update(error=str(error),elapsed_s=time.time()-start);raise
 finally:
  write(out/'validation.json',status)
  write(out/'SHA256.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='SHA256.json'})


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
 p.add_argument('--workers',type=int,choices=(1,2,3),default=1);p.add_argument('--scenes',nargs='+',choices=SCENES,default=list(SCENES));p.add_argument('--filter',nargs='*');p.add_argument('--combinations',action='store_true');p.add_argument('--only-combinations',action='store_true');p.add_argument('--pairs',nargs='+',choices=tuple(PAIRS))
 p.add_argument('--worker',action='store_true');p.add_argument('--scene',choices=SCENES);p.add_argument('--implementation');p.add_argument('--arm',default='active');p.add_argument('--slots',type=int,default=128);p.add_argument('--ues',type=int,default=4);p.add_argument('--pair',choices=tuple(PAIRS))
 a=p.parse_args();(worker if a.worker else main)(a)

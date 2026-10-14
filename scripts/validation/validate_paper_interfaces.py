"""Finite Spark acceptance of paper control/observation interfaces, no algorithms."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Lock,Event
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.contracts import source_fingerprint
STOP=Event();CHILDREN=set();LOCK=Lock()


def stop(signum,frame):
    STOP.set()
    with LOCK:
        for p in list(CHILDREN):p.terminate()
    raise SystemExit(128+signum)

signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def invoke(argv,log):
    if STOP.is_set():raise InterruptedError('Stopped')
    with log.open('w') as f:
        with subprocess.Popen([sys.executable,*argv],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT) as p:
            with LOCK:CHILDREN.add(p)
            try:
                if STOP.is_set():p.terminate()
                if p.wait():raise RuntimeError(f'Failed: {log}')
            finally:
                with LOCK:CHILDREN.discard(p)


def timeline(slots,nprb,ues,ret=None):
    if ret is not None:return [dict(slot=5,config=dict(apply_scope='cluster',ret_enabled=ret,ret_tilt_deg=10 if ret else 0))]
    events=[dict(slot=5,config=dict(apply_scope='cluster',ue_prb_limits={f'ue_{i}':min(8,nprb) for i in range(ues)},time_to_trigger_ms=40)),
        dict(slot=20,config=dict(target_cell_id='cell_0',prb_power_w=[.4/nprb]*nprb)),
        dict(slot=50,config=dict(target_cell_id='cell_0',ret_enabled=True,ret_tilt_deg=10)),
        dict(slot=80,config=dict(target_cell_id='cell_0',prb_power_w=[1.]*nprb)),
        dict(slot=100,config=dict(target_cell_id='cell_0',clear_prb_power=True),commands=[dict(command_type='disconnect',target_ue_id='ue_0')]),
        dict(slot=200,commands=[dict(command_type='reconnect',target_ue_id='ue_0')]),
        dict(slot=300,config=dict(target_cell_id='cell_0',ret_tilt_deg=0,cio_offsets_db={'cell_1':3},neighbor_cell_ids=['cell_1'],steering_hysteresis_db=1)),
        dict(slot=400,config=dict(target_cell_id='cell_0',ret_enabled=False,ue_prb_limits={})),
        dict(slot=1500,config=dict(apply_scope='cluster',clear_prb_power=True,tx_power_dbm=-100)),
        dict(slot=2300,config=dict(apply_scope='cluster',tx_power_dbm=30)),
        dict(slot=3000,commands=[dict(command_type='request_handover',target_ue_id='ue_0',target_cell_id='cell_1')])]
    return [e for e in events if e['slot']<=slots]


def audit(folder,expected_slots):
    status=json.loads((folder/'run_status.json').read_text());snapshot=json.loads((folder/'config_snapshot.json').read_text());final=json.loads((folder/'final_summary.json').read_text())
    assert status['completed'] and status['spark_verified'] and status['source_sha256']==source_fingerprint(ROOT)
    stats=final['link_reference_statistics']
    assert all(stats.get(k,0)==0 for k in ('amc_probe_trials','bler_probe_trials','calibration_decode_trials','incompatible_no_tb','unsupported_no_tb'))
    counters=Counter();power_error=0.;contracts={};acked=set();last_energy={};decoded=0;refs=[];outcomes=[];last_slot=0
    for line in (folder/'rt_sys_trace.jsonl').open():
        r=json.loads(line);last_slot=r['slot_id'];paper=r['paper_interfaces'];used_prbs={c:set() for c in r['cells']}
        outcomes.extend(paper['config_outcomes']);counters.update(e['kind'] for e in paper['events'])
        for u in r['ues']:
            assert not u.get('mac_capacity_guard')
            alloc=u['resource_allocation'];cid=u['cell_id'];cfg=paper['runtime_config'][cid]
            if alloc['scheduled']:
                assert len(alloc['prbs'])<=cfg['ue_prb_limits'].get(u['ue_id'],10**9)
                used_prbs[cid].update(alloc['prbs'])
                if cfg['prb_power_w'] is not None:assert all(cfg['prb_power_w'][i]>0 for i in alloc['prbs'])
            if u['actual_decode']:
                decoded+=1;c=u['tx_contract'];assert c and c['nr_tbs_bits']//8==u['physical_tb_capacity_bytes']
                if u['is_retx']:assert contracts[u['tb_id']]==c
                else:assert u['tb_id'] not in contracts;contracts[u['tb_id']]=c
                if u['ack']:assert u['tb_id'] not in acked;acked.add(u['tb_id'])
        for cid,t in r['cells'].items():
            assert 0<=t['average_used_w']<=t['budget_w']+1e-7
            p=paper['runtime_config'][cid]['prb_power_w']
            expected=(sum(p[i] for i in used_prbs[cid]) if p is not None else t['budget_w'] if t['emitted_re_streams'] else 0.)
            assert abs(expected-t['average_used_w'])<1e-6,(cid,expected,t)
            energy=paper['rf_energy_j'][cid];delta=energy-last_energy.get(cid,0)
            power_error=max(power_error,abs(delta-t['average_used_w']*snapshot['manifest']['slot_duration_ms']/1000));last_energy[cid]=energy
        if last_slot in (10,50,70,128):refs.append(dict(slot=last_slot,ues=paper['reports']['ue']))
    assert last_slot==expected_slots and power_error<1e-8
    assert decoded==stats['actual_tb_decodes']
    delivered_ranges={}
    for line in (folder/'packet_events.jsonl').open():
        e=json.loads(line)
        if e['kind']=='delivery':
            assert e['tb_id'] in acked
            delivered_ranges.setdefault((e['ue_id'],e['packet_id']),[]).append((e['byte_start'],e['byte_end']))
    for ranges in delivered_ranges.values():
        ordered=sorted(ranges);assert all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:]))
    for m in final['packet_metrics'].values():assert m['initial_bytes']+m['offered_bytes']==m['pending_bytes']+m['overflow_bytes']+m['delivered_bytes']
    public=json.loads((folder/'public_kpi_inputs.json').read_text())
    assert public
    for frame in public:
        for k in frame['kpis']:
            if k['valid']:assert k['produced_at_slot']<=k['delivered_at_slot']<=frame['slot']
    if expected_slots>=4000:
        assert counters['connection_drop']>0
        all_events=final['paper_interfaces']['events']
        assert sum(e['kind']=='connection_started' for e in all_events)>sum(e['kind']=='connection_drop' for e in all_events)
        assert any(o.get('reason')=='prb_power_exceeds_cell_budget' for o in outcomes)
        assert any(e['kind']=='connection_started' and e['slot_id']>2300 for e in final['paper_interfaces']['events'])
    result=dict(case=folder.name,passed=True,source_sha256=status['source_sha256'],elapsed_s=status['elapsed_s'],
        actual_decodes=decoded,events=dict(counters),max_energy_error_j=power_error,peak_rss_bytes=status.get('peak_rss_bytes'),
        peak_cuda_bytes=status.get('peak_cuda_bytes'),references=refs,config_rejections=Counter(o.get('reason') for o in outcomes if o.get('status')=='rejected'),
        steady_state=final['steady_state'],packet_metrics=final['packet_metrics'],cache_key=final['rt_sys']['cache_key'])
    (folder/'paper_validation.json').write_text(json.dumps(result,indent=2));return result


def execute(out,scene,ues,seed,slots,load=1.,speed=0.,ret=None):
    from phy.rt_sys.scenario import build_case
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    _,c,_=build_case(scene,num_cells=3,num_ues=ues,total_slots=slots)
    actions=out/'timeline.json';actions.write_text(json.dumps(timeline(slots,c['sim']['scheduler_total_prbs_per_slot'],ues,ret),indent=2))
    argv=['experiments/run_rt_sys.py','--config','configs/rt_sys/paper_interfaces.json','--control-timeline',str(actions),
        '--scene',scene,'--cells','3','--ues',str(ues),'--seed',str(seed),'--slots',str(slots),'--initial-queue-bytes','0',
        '--ue-speed-mps',str(speed),'--traffic-load',str(load),'--warmup-ms',str(200 if slots>=1000 else 0),
        '--feedback-mode','nr_direct','--control-mode','baseline','--output',str(out)]
    invoke(argv,out/'process.log');return audit(out,slots)


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);sha=source_fingerprint(ROOT);start=time.perf_counter();results=[]
    status=dict(source_sha256=sha,completed=False,passed=False,stage='starting')
    def save():
        status['elapsed_s']=time.perf_counter()-start
        p=root/'pipeline_status.tmp';p.write_text(json.dumps(status,indent=2));p.replace(root/'pipeline_status.json')
    save()
    try:
        status['stage']='pytest';save();invoke(['-m','pytest','-q','tests','--disable-warnings','--maxfail=1'],root/'pytest.log')
        status['stage']='gate';save()
        from phy.rt_sys.scenario import build_case
        from phy.rt_sys.channel import RtChannelProvider
        import numpy as np
        m,_,_=build_case(num_cells=1,num_ues=1,total_slots=1)
        provider=RtChannelProvider(m,ROOT/'channel_cache',device='cuda')
        pos=provider.backend.scene.tx_array.normalized_positions
        actual_z=np.asarray(pos.z);actual_y=np.asarray(pos.y)
        expected_z=np.tile(((m['tx_rows']-1)/2-np.arange(m['tx_rows']))*.5,m['tx_cols'])
        expected_y=np.repeat((np.arange(m['tx_cols'])-(m['tx_cols']-1)/2)*.5,m['tx_rows'])
        np.testing.assert_allclose(actual_z,expected_z,atol=1e-6);np.testing.assert_allclose(actual_y,expected_y,atol=1e-6)
        (root/'ret_array_layout.json').write_text(json.dumps(dict(verified=True,y=actual_y.tolist(),z=actual_z.tolist())))
        for scene in ('simple_street_canyon','munich'):
            folder=root/(scene+'_timing8')
            invoke(['scripts/validation/benchmark_rt_cost.py','--worker','--mode','nr_direct','--scene',scene,'--slots','8',
                '--config','configs/rt_sys/paper_interfaces.json','--output',str(folder)],root/(scene+'_timing.log'))
            profile=json.loads((folder/'cost_profile.json').read_text());assert profile['total_s']<=14.64,(scene,profile['total_s'])
            results.append(execute(root/(scene+'_128'),scene,24,512,128))
            (root/'progress.json').write_text(json.dumps(dict(stage='gate',results=results),indent=2));print('PASS gate',scene,profile['total_s'],flush=True)
        status['stage']='ret_pairs';save()
        ret_pairs=[]
        for scene in ('simple_street_canyon','munich'):
            pair=[execute(root/f'{scene}_ret_{v}',scene,4,512,128,ret=v) for v in (False,True)]
            assert pair[0]['cache_key']==pair[1]['cache_key']
            a,b=[r['references'][-1]['ues'] for r in pair]
            difference=max(abs(a[u]['candidate_rsrp_dbm'][c]-b[u]['candidate_rsrp_dbm'][c]) for u in a for c in a[u]['candidate_rsrp_dbm'])
            assert difference>1e-3;ret_pairs.append(dict(scene=scene,max_reference_difference_db=difference,cache_key=pair[0]['cache_key']))
            results.extend(pair);print('PASS ret',scene,difference,flush=True)
        status['stage']='moving';save()
        for scene in ('simple_street_canyon','munich'):
            results.append(execute(root/(scene+'_moving'),scene,4,512,256,speed=1.5))
            print('PASS moving',scene,flush=True)
        status['stage']='matrix';save()
        specs=[(s,4,seed,4000,1.) for s in ('simple_street_canyon','munich') for seed in (512,513,514)]
        specs +=[(s,24,512,1000,load) for s in ('simple_street_canyon','munich') for load in (.1,1.)]
        def job(spec):
            scene,n,seed,slots,load=spec
            return execute(root/f'{scene}_u{n}_s{seed}_load{load}_{slots}',scene,n,seed,slots,load)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(job,s) for s in specs]
            for i,future in enumerate(as_completed(futures),1):
                try:result=future.result()
                except BaseException:
                    STOP.set()
                    with LOCK:
                        for child in list(CHILDREN):child.terminate()
                    for pending in futures:pending.cancel()
                    raise
                results.append(result)
                (root/'progress.json').write_text(json.dumps(dict(stage='matrix',completed=i,total=len(specs),results=results),indent=2))
                print('PASS matrix',i,len(specs),result['case'],result['elapsed_s'],flush=True)
        assert source_fingerprint(ROOT)==sha
        report=dict(passed=True,source_sha256=sha,results=results,ret_pairs=ret_pairs,scope='Control/observation integration; no trained xApp, generalization or universal QoS claim')
        (root/'validation.json').write_text(json.dumps(report,indent=2));status.update(completed=True,passed=True,stage='complete');save()
        files=[]
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            h=hashlib.sha256()
            with p.open('rb') as f:
                for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
            files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h.hexdigest()))
        (root/'data_manifest.json').write_text(json.dumps(dict(source_sha256=sha,files=files),indent=2))
    except BaseException as error:
        status.update(error=repr(error));save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

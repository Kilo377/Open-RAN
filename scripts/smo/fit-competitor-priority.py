"""Fit a supplemental population gain using retained, fixed-control probes only."""
import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from smo.action_gain_supplement import fit,SCHEMA,MEMBERS,UNITS,sha
from smo.calibration_reuse import source_files,ALLOWED_CONTROLLER_CHANGES,ADDED_CALIBRATION_TOOLS
from smo.runtime import trace_users
from smo.window_metrics import WindowIndex
from phy.rt_sys.contracts import source_fingerprint


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def settings(cfg):
    result=deepcopy(cfg['configs']);result.pop('ric',None)
    result['sim'].pop('rng_seed',None);result['sim'].pop('total_slots',None)
    result['phy']['rt_sys'].pop('rng_seed',None)
    return result


def read_lines(path):
    if not path.is_file():path=path.with_suffix(path.suffix+'.gz')
    with (gzip.open(path,'rt')if path.suffix=='.gz'else path.open())as f:
        return [json.loads(line)for line in f if line.strip()]


def audit_case(record,study,recipe,reference):
    c=record['case'];folder=Path(record['folder']).resolve();assert folder.is_relative_to(study/'workers')
    assert c['arm']=='FIXED'and c['split']=='training'and c['policy']in('fixed-01.json','fixed-02.json')
    seal=json.loads((folder/'archive-verified.json').read_text());assert json.loads((folder/'complete.json').read_text())['state']=='complete'
    run=folder/'run';pins={}
    for name in ('config_snapshot.json','rt_sys_trace.jsonl.gz','control_receipts.jsonl.gz','xapp_hot_start.json'):
        pins[name]=sha(run/name);assert seal['files']['run/'+name]==pins[name]
    hot=json.loads((run/'xapp_hot_start.json').read_text())['models']['latency_scheduler_rf']
    policy=hot['adaptive_policy'];assert hot['mode']=='frozen'and policy['kind']=='fixed'and policy['training']is False
    checkpoint=run/policy['checkpoint'];assert sha(checkpoint)==policy['sha256']==seal['files']['run/'+policy['checkpoint']]
    pins[policy['checkpoint']]=policy['sha256']
    cfg=json.loads((run/'config_snapshot.json').read_text());assert settings(cfg)==settings(reference)
    profile_pins=dict(traffic_profile=cfg['manifest']['traffic_profile']['sha256'],
        ue_services_config=cfg['ue_services_profile']['sha256'],ap_services_profile=cfg['ap_services_profile']['sha256'])
    assert all(value==recipe['profiles'][key]['sha256']for key,value in profile_pins.items())
    source=(study/'sources'/c['source']).resolve();assert source.is_relative_to(study/'sources')
    assert source_fingerprint(source)==cfg['source_sha256']
    old,new=source_files(source),source_files(ROOT)
    allowed=ALLOWED_CONTROLLER_CHANGES|ADDED_CALIBRATION_TOOLS
    assert {k:v for k,v in old.items()if k not in allowed}=={k:v for k,v in new.items()if k not in allowed}
    idx=WindowIndex(cfg)
    with gzip.open(run/'rt_sys_trace.jsonl.gz','rt')as f:
        for line in f:
            r=json.loads(line);idx.observe(r['slot_id'],trace_users(r),r.get('ue_services',{}),
                r.get('ap_services',{}).get('cells',{}),r.get('ue_service_events',[]))
    assert not idx.task_errors and len(idx.slots)==c['slots']
    assert all(slot==i+1 for i,slot in enumerate(idx.slots))
    origin=c['origin_slot'];end=origin+500
    states=idx.window(list(UNITS),origin,end,through=c['slots'])['states']
    observed={k:states['kpi:'+k]['value']for k in UNITS}
    assert all(value is not None and math.isfinite(value)for value in observed.values())
    assert states['kpi:mec_offloaded_latency_ms'].get('pending')==0
    assert states['kpi:mec_offloaded_latency_ms'].get('failed')==0
    for k,value in observed.items():
        assert math.isclose(value,record['windows']['500']['aggregate']['states']['kpi:'+k]['value'],abs_tol=1e-9)
    events=read_lines(run/'latency_rf_events.jsonl')
    decisions=[d for d in events if d.get('kind')=='decision'and origin<d['slot_id']<end]
    assert decisions and all(d['scheduler_mode']=='latency_aware'for d in decisions)
    receipts=read_lines(run/'control_receipts.jsonl.gz')
    assert all(p.get('instance_id')=='latency_scheduler_rf'for r in receipts
        if origin<r.get('apply_slot',0)<=end and r.get('status')=='applied'for p in r.get('provenance',[]))
    for d in decisions:
        matched=[r for r in receipts if r.get('apply_slot')==d['slot_id']+1 and d['cell_id']in r.get('after',{})
            and any(p.get('instance_id')=='latency_scheduler_rf'for p in r.get('provenance',[]))]
        assert matched and all(r['status']=='applied'for r in matched)
        assert any(r['after'][d['cell_id']]['scheduler_mode']==d['scheduler_mode']
            and r['after'][d['cell_id']]['ue_scheduler_weights']==d['ue_scheduler_weights']for r in matched)
    tasks=sorted((t['ue_id'],str(key),t['created_s'],t['offloaded'])for key,t in idx.tasks.items()
        if origin<t['born_slot']<=end and origin*idx.dt-1e-10<=t['created_s']<end*idx.dt-.1-1e-10)
    return observed,decisions,tasks,dict(case=c['name'],source=c['source'],source_sha256=cfg['source_sha256'],
        archive_sha256=seal['archive_sha256'],raw_sha256=pins,profile_sha256=profile_pins,
        contiguous_slots=c['slots'],sample_count=states['kpi:mec_offloaded_latency_ms']['sample_count'])


def main():
    if not __debug__:
        raise RuntimeError('The raw-data audit must run without Python -O')
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    study=a.study.resolve();manifest=json.loads(a.manifest.read_text());entry=next(e for e in manifest['scenarios']if e['id']==manifest['default'])
    recipe=json.loads(Path(entry['runtime_recipe']).read_text());reference=json.loads((Path(recipe['validated_origin']['run'])/'config_snapshot.json').read_text())
    groups=defaultdict(dict)
    for r in json.loads((study/'analysis/case-index.json').read_text()):
        c=r['case']
        if c.get('arm')=='FIXED'and c.get('split')=='training'and c.get('source')=='source-v3'and c.get('policy')in('fixed-01.json','fixed-02.json'):
            groups[c['worker'],c['seed'],c['origin_slot']][c['policy']]=r
    observations=[];excluded=[]
    for key,cases in sorted(groups.items()):
        if set(cases)!= {'fixed-01.json','fixed-02.json'}:continue
        before,after=cases['fixed-02.json'],cases['fixed-01.json']
        assert before['network_prefix_sha256']==after['network_prefix_sha256']
        left,ld,lt,lp=audit_case(before,study,recipe,reference);right,rd,rt,rp=audit_case(after,study,recipe,reference)
        assert lt==rt and len(ld)==len(rd)
        changed=set()
        for x,y in zip(ld,rd):
            assert (x['slot_id'],x['cell_id'],x['scheduler_mode'])==(y['slot_id'],y['cell_id'],y['scheduler_mode'])
            assert x['ue_ids']==y['ue_ids']
            assert set(x['ue_scheduler_weights'])==set(y['ue_scheduler_weights'])
            for uid,value in x['ue_scheduler_weights'].items():
                other=y['ue_scheduler_weights'][uid]
                if uid in ('ue_0','ue_4'):assert value==other==2.
                else:
                    assert value==1. and other==.5
                    changed.add(uid)
        if sorted(changed)!=MEMBERS:
            excluded.append(dict(context=list(key),reason='different_competitor_population',actual_members=sorted(changed)))
            print(json.dumps(excluded[-1]),flush=True)
            continue
        observations.append(dict(worker=key[0],seed=key[1],origin_slot=key[2],input_role='fixed_parameter_probe',arm='FIXED',split='training',
            learned_policy_controls_used=False,combination_evaluation_data_used=False,members=MEMBERS,
            from_weight=1.,to_weight=.5,mec_weight=2.,scheduler_mode='latency_aware',
            prefix_equal=True,task_identities_equal=True,normal_controls_verified=True,
            only_competitor_weights_changed=True,network_configuration_verified=True,
            before=left,after=right,raw_evidence={'before':lp,'after':rp}))
        print(json.dumps(dict(audited_pair=key,means=[left,right])),flush=True)
    rows=fit(observations);a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'measurements.json',dict(schema='oran.fixed-rcp-paired-measurements.v1',observations=observations,
        excluded_contexts=excluded,
        combination_evaluation_data_used=False,window_slots=500,birth_guard_ms=100,
        terminal_followup_for_KPI=False,aggregation='unchanged_online_matched_mature_birth_cohort'))
    certificate=dict(schema=SCHEMA,environment_id=recipe['id'],combination_evaluation_data_used=False,
        base_matrix_sha256=sha(entry['influences']),runtime_recipe_sha256=sha(entry['runtime_recipe']),
        measurements=dict(path='measurements.json',sha256=sha(a.output/'measurements.json')),estimates=rows,
        only_existing_parameter_kpi_edges=True,source='archived_fixed_parameter_probes; no deployed-combination outcomes',
        coordinate='joint competitor-population weight; per-UE residual projection is a first-order reference approximation')
    write(a.output/'certificate.json',certificate)
    print(json.dumps(dict(certificate=str(a.output/'certificate.json'),seeds=rows[0]['independent_seeds'],pairs=len(observations),estimates=[{k:r[k]for k in('kpi','estimate','confidence_interval')}for r in rows])))
if __name__=='__main__':main()

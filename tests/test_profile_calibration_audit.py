"""Analytic journal fixtures; no simulator and no validation data access."""
from copy import deepcopy
import json

import pytest

from experiments import profile_calibration_audit as audit


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def lines(path,rows):
    path.write_text(''.join(json.dumps(r)+'\n'for r in rows))


def fixture(tmp_path):
    profile={'cell_0':'light_sleep','cell_1':'deep_sleep','cell_2':'light_sleep'}
    plan=dict(training_seeds=[25],validation_seeds_excluded=[1,2],profiles={'native':{'mixed':profile}},
        reference_profile={c:'active'for c in profile},slot_duration_ms=.5,switch_slot=40,total_slots=85,
        scoring_start_slot=82,scoring_end_slot=85)
    plan['plan_sha256']=audit._digest(plan)
    paths={}
    for ident,policy in(('REF',plan['reference_profile']),('mixed',profile)):
        run=tmp_path/ident;run.mkdir();paths[ident]=run
        spec=dict(schema='oran.sleep-profile-probe.v1',profile_id=ident,switch_slot=40,profile=policy,calibration_plan_sha256=plan['plan_sha256'])
        config=dict(source_sha256='source',manifest=dict(rng_seed=25,total_slots=85,slot_duration_ms=.5),
            configs={'ric':dict(xapps=[],rapps=[],freeze_learning=True)},sleep_profile_probe=spec,control_mode='sleep_profile_probe')
        put(run/'config_snapshot.json',config);put(run/'run_status.json',dict(completed=True,source_sha256='source'))
        lines(run/'sleep_profile_requests.jsonl',[dict(slot=40,profile_id=ident,profile=policy,controlled_fields=['cell_sleep_mode'],calibration_plan_sha256=plan['plan_sha256'])])
        receipts=[]
        for cid,mode in policy.items():
            before=dict(cell_sleep_mode='active',ap_policy='active',ap_policy_owner=None,scheduler_mode='round_robin',ue_scheduler_weights={})
            after=dict(before,cell_sleep_mode=mode,ap_policy=mode,ap_policy_owner=audit.SOURCE)
            receipts.append(dict(kind='config',status='applied',source=audit.SOURCE,target_cell_id=cid,target_cell_ids=[cid],apply_slot=41,
                fields=[dict(field='cell_sleep_mode',value=mode)],before={cid:before},after={cid:after}))
        lines(run/'control_receipts.jsonl',receipts)
        trace=[dict(slot_id=s,xapp_bindings={},traffic_arrivals={'u':dict(exogenous_bytes=20,mec_result_bytes=0)},ue_service_events=[],
            ap_services={'cells':{c:dict(ap_policy=mode if s>=41 else'active',ap_policy_owner=audit.SOURCE if s>=41 else None)for c,mode in policy.items()}})for s in range(1,86)]
        lines(run/'rt_sys_trace.jsonl',trace)
    return plan,paths


def test_real_sleep_only_pair_has_matching_recipe_prefix_and_arrivals(tmp_path):
    plan,paths=fixture(tmp_path);proof=audit.audit_pair(paths['REF'],paths['mixed'],plan,'native',25)
    assert proof['valid'] and all(proof['checks'].values())
    assert proof['profile']['controls_written']==['cell_sleep_mode']
    assert proof['profile']['post_settle_observation_rows']==3
    assert proof['heldout_validation_data_read']is False


@pytest.mark.parametrize('mutation,reason',[
    ('extra_field','sleep_only_action'),('non_sleep_change','non_sleep_control_changed'),
    ('rejected','sleep_only_action'),('early_effect','effect_before_receipt'),
    ('missing_cell','missing_cell_telemetry'),('short_trace','missing_post_settle'),
])
def test_invalid_identification_runs_are_rejected(tmp_path,mutation,reason):
    plan,paths=fixture(tmp_path);run=paths['mixed']
    receipts=[json.loads(x)for x in(run/'control_receipts.jsonl').read_text().splitlines()]
    trace=[json.loads(x)for x in(run/'rt_sys_trace.jsonl').read_text().splitlines()]
    if mutation=='extra_field':receipts[0]['fields'].append(dict(field='tx_power_dbm',value=12))
    if mutation=='non_sleep_change':receipts[0]['after']['cell_0']['scheduler_mode']='latency_aware'
    if mutation=='rejected':receipts[0]['status']='rejected'
    if mutation=='early_effect':trace[39]['ap_services']['cells']['cell_0']['ap_policy']='light_sleep'
    if mutation=='missing_cell':trace[-1]['ap_services']['cells'].pop('cell_2')
    if mutation=='short_trace':trace.pop()
    lines(run/'control_receipts.jsonl',receipts);lines(run/'rt_sys_trace.jsonl',trace)
    with pytest.raises(ValueError,match=reason):audit.audit_run(run,plan,'native',25,'mixed')


def test_pair_exogenous_drift_is_not_accepted_as_sleep_gain(tmp_path):
    plan,paths=fixture(tmp_path);p=paths['mixed']/'rt_sys_trace.jsonl'
    trace=[json.loads(x)for x in p.read_text().splitlines()]
    trace[-1]['traffic_arrivals']['u']['exogenous_bytes']=30;lines(p,trace)
    with pytest.raises(ValueError,match='full_exogenous_sha256'):audit.audit_pair(paths['REF'],paths['mixed'],plan,'native',25)


def test_heldout_seed_is_refused_before_data_access(tmp_path):
    plan,paths=fixture(tmp_path)
    with pytest.raises(ValueError,match='not_training'):audit.audit_run(paths['mixed'],plan,'native',1,'mixed')


def test_native_dispatcher_repeats_expiry_and_unrecorded_owner_are_preserved(tmp_path):
    plan,paths=fixture(tmp_path);run=paths['mixed']
    receipts=[json.loads(x)for x in(run/'control_receipts.jsonl').read_text().splitlines()]
    for r in receipts:
        r['source']='controller_manager';r['after'][r['target_cell_id']]['ap_policy_owner']=None
    repeat=deepcopy(receipts[0]);repeat['apply_slot']=42;repeat['before']=deepcopy(repeat['after'])
    expired=deepcopy(repeat);expired.update(apply_slot=80,status='rejected',reason='expired_before_actuation')
    noop=dict(kind='config',source='controller_manager',status='applied',apply_slot=3,fields=[],
        before={'cell_0':{'scheduler_mode':'round_robin'}},after={'cell_0':{'scheduler_mode':'round_robin'}})
    lines(run/'control_receipts.jsonl',[noop,*receipts,repeat,expired])
    trace=[json.loads(x)for x in(run/'rt_sys_trace.jsonl').read_text().splitlines()]
    for row in trace:
        for cell in row['ap_services']['cells'].values():cell['ap_policy_owner']=None
    lines(run/'rt_sys_trace.jsonl',trace)
    proof=audit.audit_run(run,plan,'native',25,'mixed')
    assert proof['receipt_counts']==dict(empty_noop=1,applied_sleep=4,expired_noop=1)
    assert proof['actual_policy_owners']=={c:None for c in plan['reference_profile']}

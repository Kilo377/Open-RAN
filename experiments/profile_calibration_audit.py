"""Fail-closed evidence audit for independent fixed sleep-profile probes.

This module reads calibration runs only. It neither estimates KPI gains nor
opens a held-out combination run, and never supplies information to a policy.
"""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from experiments.sleep_profile_probe import validate
from smo.evaluation_reference import origin_configuration

SOURCE='sleep_profile_probe'


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _records(run,name):
    for suffix,opener in(('.jsonl.gz',gzip.open),('.jsonl',open)):
        path=Path(run)/(name+suffix)
        if path.exists():
            with opener(path,'rt')as stream:
                for line in stream:
                    if line.strip():yield json.loads(line)
            return


def _file_pins(run):
    paths=[Path(run)/'config_snapshot.json',Path(run)/'run_status.json']
    for name in('rt_sys_trace','control_receipts','sleep_profile_requests','diagnostic_trace'):
        paths+=list(Path(run).glob(name+'.jsonl*'))
    return {str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths if p.is_file()}


def _plan(plan):
    plan=_read(plan)if isinstance(plan,(str,Path))else deepcopy(plan)
    if _digest({k:v for k,v in plan.items()if k!='plan_sha256'})!=plan.get('plan_sha256'):
        raise ValueError('profile_probe_plan_digest_mismatch')
    return plan


def _exogenous(row):
    return dict(slot=row['slot_id'],traffic_arrivals={u:v.get('exogenous_bytes')for u,v in row.get('traffic_arrivals',{}).items()},
        task_births=[{k:v for k,v in event.items()if k!='latency_ms'}for event in row.get('ue_service_events',[])if event.get('kind')=='task_generated'])


def _non_sleep_receipt(snapshot):
    # AP policy counters/ownership are consequences of the one permitted field.
    return {cid:{k:v for k,v in values.items()if k!='cell_sleep_mode'and not k.startswith('ap_')}
        for cid,values in snapshot.items()}


def audit_run(run,plan,scene,seed,profile_id):
    """Return explicit proof or raise on a missing/invalid calibration contract."""
    run=Path(run);plan=_plan(plan)
    if seed not in plan['training_seeds']or seed in plan['validation_seeds_excluded']:
        raise ValueError('profile_probe_seed_not_training')
    if scene not in plan['profiles']:raise ValueError('profile_probe_scene_unregistered')
    profile=(plan['reference_profile']if profile_id=='REF'else plan['profiles'][scene].get(profile_id))
    if profile is None:raise ValueError('profile_probe_profile_unregistered')
    config=_read(run/'config_snapshot.json')
    spec=validate(config['sleep_profile_probe']);switch=plan['switch_slot'];end=plan['total_slots']
    expected=dict(schema=spec['schema'],profile_id=profile_id,switch_slot=switch,profile=profile,calibration_plan_sha256=plan['plan_sha256'])
    if spec!=expected:raise ValueError('profile_probe_spec_mismatch')
    manifest=config['manifest'];ric=config['configs']['ric']
    if (manifest['rng_seed']!=seed or manifest['total_slots']!=end or manifest['slot_duration_ms']!=plan['slot_duration_ms']
        or ric.get('xapps')or ric.get('rapps')or ric.get('freeze_learning')is not True
        or config.get('control_mode')!='sleep_profile_probe'):
        raise ValueError('profile_probe_not_isolated_native_recipe')
    status=_read(run/'run_status.json')
    if status.get('completed')is not True or status.get('source_sha256')!=config.get('source_sha256'):
        raise ValueError('profile_probe_run_incomplete_or_source_mismatch')
    requests=list(_records(run,'sleep_profile_requests'))
    if len(requests)!=1:raise ValueError('profile_probe_requires_one_fixed_request')
    request=requests[0]
    if any(request.get(k)!=v for k,v in dict(slot=switch,profile_id=profile_id,profile=profile,
        controlled_fields=['cell_sleep_mode'],calibration_plan_sha256=plan['plan_sha256']).items()):
        raise ValueError('profile_probe_request_mismatch')
    receipts=list(_records(run,'control_receipts'));applied={};owners={};receipt_counts={'empty_noop':0,'applied_sleep':0,'expired_noop':0}
    for receipt in receipts:
        cid=receipt.get('target_cell_id');fields=receipt.get('fields',[]);slot=receipt.get('apply_slot')
        before=receipt.get('before',{});after=receipt.get('after',{})
        if (receipt.get('kind')!='config'or receipt.get('source')not in(SOURCE,'controller_manager')
            or type(slot)is not int or not 1<=slot<=end):
            raise ValueError('profile_probe_receipt_not_exact_applied_sleep_only_action')
        if not fields:
            if receipt.get('status')!='applied'or not before or before!=after:
                raise ValueError('profile_probe_empty_action_changed_state')
            receipt_counts['empty_noop']+=1;continue
        if (cid not in profile or receipt.get('target_cell_ids')!=[cid]or type(slot)is not int
            or len(fields)!=1 or fields[0].get('field')!='cell_sleep_mode'or fields[0].get('value')!=profile[cid]):
            raise ValueError('profile_probe_receipt_not_exact_applied_sleep_only_action')
        if receipt.get('status')=='rejected'and receipt.get('reason')=='expired_before_actuation'and slot>switch+39:
            if not before or before!=after:raise ValueError('profile_probe_expired_action_changed_state')
            receipt_counts['expired_noop']+=1;continue
        if receipt.get('status')!='applied'or not switch<=slot<=switch+39:
            raise ValueError('profile_probe_receipt_not_exact_applied_sleep_only_action')
        if set(before)!={cid}or set(after)!={cid}or _non_sleep_receipt(before)!=_non_sleep_receipt(after):
            raise ValueError('profile_probe_non_sleep_control_changed')
        if after[cid].get('cell_sleep_mode')!=profile[cid]or after[cid].get('ap_policy')!=profile[cid]:
            raise ValueError('profile_probe_actual_policy_not_confirmed')
        owner=after[cid].get('ap_policy_owner')
        if cid in owners and owner!=owners[cid]:raise ValueError('profile_probe_policy_owner_changed')
        owners[cid]=owner;applied[cid]=min(slot,applied.get(cid,slot));receipt_counts['applied_sleep']+=1
    if set(applied)!=set(profile):raise ValueError('profile_probe_missing_applied_cell_receipt')
    prefix=hashlib.sha256();exogenous=hashlib.sha256();last=0;observations=0;prefix_rows=0
    for row in _records(run,'rt_sys_trace'):
        slot=row.get('slot_id')
        if type(slot)is not int or slot!=last+1:raise ValueError('profile_probe_trace_not_contiguous')
        last=slot
        if row.get('xapp_bindings'):raise ValueError('profile_probe_trained_xapp_active')
        exogenous.update(_digest(_exogenous(row)).encode())
        if slot<=switch:
            # The admission boundary may expose different requests, but the
            # physical/native prefix must be identical, before any actuation.
            physical={k:v for k,v in row.items()if k not in('control_audit','paper_interfaces','precoders')}
            prefix.update(_digest(physical).encode());prefix_rows+=1
        cells=row.get('ap_services',{}).get('cells',{})
        if set(cells)!=set(profile):raise ValueError('profile_probe_missing_cell_telemetry')
        for cid,actual in cells.items():
            if slot<applied[cid]and actual.get('ap_policy')!='active':raise ValueError('profile_probe_effect_before_receipt')
            if slot>=applied[cid]and(actual.get('ap_policy')!=profile[cid]or actual.get('ap_policy_owner')!=owners[cid]):
                raise ValueError('profile_probe_held_profile_not_executed')
        if plan['scoring_start_slot']<slot<=plan['scoring_end_slot']:observations+=1
    if last!=end or observations!=plan['scoring_end_slot']-plan['scoring_start_slot']:
        raise ValueError('profile_probe_missing_post_settle_window')
    recipe=origin_configuration(config);recipe.pop('sleep_profile_probe',None)
    return dict(schema='oran.sleep-profile-run-audit.v1',valid=True,run=str(run),scene=scene,seed=seed,profile_id=profile_id,
        calibration_plan_sha256=plan['plan_sha256'],source_sha256=config['source_sha256'],configuration_sha256=_digest(recipe),
        actual_applied_slots=applied,actual_policy_owners=owners,receipt_counts=receipt_counts,
        controller_origin_evidence='isolated fixed-probe recipe, one recorded request and exact scoped sleep-only dispatcher receipts; original owner can be unrecorded',
        prefix_end_slot=switch,prefix_rows=prefix_rows,prefix_physical_sha256=prefix.hexdigest(),
        full_exogenous_sha256=exogenous.hexdigest(),post_settle_observation_rows=observations,files_sha256=_file_pins(run),
        controls_written=['cell_sleep_mode'],trained_xapps_enabled=False,heldout_validation_data_read=False)


def audit_pair(reference_run,profile_run,plan,scene,seed):
    """Certify matched recipe, real native prefix and exogenous arrivals."""
    plan=_plan(plan);spec=_read(Path(profile_run)/'config_snapshot.json')['sleep_profile_probe']
    reference=audit_run(reference_run,plan,scene,seed,'REF')
    profile=audit_run(profile_run,plan,scene,seed,spec['profile_id'])
    checks={k:reference[k]==profile[k]for k in('source_sha256','configuration_sha256','prefix_physical_sha256','full_exogenous_sha256')}
    if not all(checks.values()):raise ValueError('profile_probe_pair_mismatch:'+','.join(k for k,v in checks.items()if not v))
    return dict(schema='oran.sleep-profile-pair-audit.v1',valid=True,scene=scene,seed=seed,
        profile_id=spec['profile_id'],calibration_plan_sha256=plan['plan_sha256'],checks=checks,reference=reference,profile=profile,
        matched_window=[plan['scoring_start_slot'],plan['scoring_end_slot']],heldout_validation_data_read=False)

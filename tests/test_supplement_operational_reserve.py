"""A wall-budget amendment cannot alter physical calls or interrupt a case."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from experiments import quantification_supplement_reserve as operational
from experiments import quantification_supplement as original


def fixture(tmp_path):
    physical=dict(parent_physical_plan_sha256='physical',source_sha256='source',source_archive_sha256='archive',
        deadline_utc='2026-11-22T11:47:24+00:00',analysis_reserve_s=3600)
    plan=dict(worker_id='agx1',supplement_plan_sha256='supplement',precontrol_admission_sha256='old-gate',
        runner_sha256='new-actual-sha',original_protocol_runner_sha256=operational.ORIGINAL_RUNNER_SHA256,
        operational_analysis_reserve_s=1800,cases=[dict(id=str(i),arm='BCE')for i in range(6)])
    cert=dict(schema='oran.supplement-operational-reserve-admission.v1',root_approved=True,
        supplement_plan_sha256='supplement',physical_plan_sha256='physical',physical_source_sha256='source',physical_archive_sha256='archive',
        original_protocol_runner_sha256=operational.ORIGINAL_RUNNER_SHA256,execution_wrapper_sha256='new-actual-sha',
        parent_precontrol_admission_sha256='old-gate',analysis_reserve_s=1800,deadline_utc=physical['deadline_utc'],
        changes_physical_recipe=False,changes_case_identity=False,extends_user_deadline=False,
        case_ids_by_worker={'agx1':[str(i)for i in range(6)]},case_specs_sha256_by_worker={'agx1':operational.digest(plan['cases'])})
    path=tmp_path/'admission.json';path.write_text(json.dumps(cert));plan['operational_admission']=dict(path=str(path),sha256=operational.sha(path))
    return plan,physical,cert,path


def test_truthful_new_wrapper_overrides_only_wall_reserve_and_keeps_deadline(tmp_path):
    plan,physical,_,_=fixture(tmp_path);before=deepcopy((plan,physical))
    reserve,meta=operational.operational_reserve(plan,physical,'new-actual-sha')
    assert reserve==1800 and (plan,physical)==before
    assert meta['original_protocol_analysis_reserve_s']==3600 and meta['deadline_utc']==physical['deadline_utc']
    assert meta['execution_wrapper_sha256']!=meta['original_protocol_runner_sha256']
    assert meta['actual_execution_host']==meta['logical_owner_worker_id']=='agx1'


@pytest.mark.parametrize('field,value',[
    ('root_approved',False),('root_approved',1),('execution_wrapper_sha256',operational.ORIGINAL_RUNNER_SHA256),
    ('analysis_reserve_s',3600),('deadline_utc','2026-11-22T12:47:24+00:00'),('physical_source_sha256','changed'),
    ('changes_physical_recipe',True),('changes_case_identity',True),('extends_user_deadline',True)])
def test_bad_or_falsely_identified_operational_admission_is_rejected(tmp_path,field,value):
    plan,physical,cert,path=fixture(tmp_path);cert[field]=value;path.write_text(json.dumps(cert));plan['operational_admission']['sha256']=operational.sha(path)
    with pytest.raises(ValueError):operational.operational_reserve(plan,physical,'new-actual-sha')


def test_case_reorder_or_edited_action_set_is_not_a_wall_budget_amendment(tmp_path):
    plan,physical,_,_=fixture(tmp_path);plan['cases'].reverse()
    with pytest.raises(ValueError):operational.operational_reserve(plan,physical,'new-actual-sha')


def test_half_hour_admission_extension_does_not_extend_the_fixed_deadline_or_last3min_stop():
    deadline=10000;new_boundary=deadline-1800-650;old_boundary=deadline-3600-650
    assert operational.admission_allowed(old_boundary,deadline,3600)
    assert not operational.admission_allowed(old_boundary+.001,deadline,3600)
    assert operational.admission_allowed(new_boundary,deadline,1800)
    assert not operational.admission_allowed(new_boundary+.001,deadline,1800)
    assert new_boundary-old_boundary==1800
    assert not operational.hard_stop_required(deadline-180,deadline)
    assert operational.hard_stop_required(deadline-180+.001,deadline)
    assert operational.hard_stop_required(0,deadline,stop=True)


def test_swap_boundary_requires_no_active_or_failed_case_and_a_stable_idle_waiter():
    waiting=dict(phase='waiting_for_original_matrix',running=[],failed=[])
    assert operational.idle_activation_boundary(waiting,original_incomplete=True,existing_waiter_alive=True)
    assert not operational.idle_activation_boundary(waiting,original_incomplete=False,existing_waiter_alive=True)
    assert not operational.idle_activation_boundary(dict(waiting,running=['physical']),original_incomplete=True,existing_waiter_alive=True)
    assert not operational.idle_activation_boundary(dict(waiting,failed=['bad']),original_incomplete=True,existing_waiter_alive=True)
    assert operational.idle_activation_boundary(dict(waiting,phase='budget_complete'),original_incomplete=False,existing_waiter_alive=False)


def test_sealed_resume_uses_identical_existing_raw_sha_gate(tmp_path):
    from experiments.quantification_study import completed
    case={'id':'case'};out=tmp_path/'outputs';job=out/'jobs/case';job.mkdir(parents=True);raw=job/'raw';raw.write_bytes(b'sealed physical data')
    (out/'checkpoints').mkdir();(out/'checkpoints/case.json').write_text(json.dumps(dict(completed=True,files_sha256={'raw':operational.sha(raw)})))
    assert completed(out,case)
    raw.write_bytes(b'corrupt')
    assert not completed(out,case)


def test_parity_of_all_12_planned_case_identities_with_the_original_protocol():
    # Use a small pinned test protocol, not private machine/run data.
    apps={k:'xapp:'+k for k in'ABCDEF'}
    p=dict(schema='oran.quantification-supplement-plan.v1',source_version='v2',physical_run_count=12,baseline_slot=2040,observe_slots=2000,total_slots=4440,
        heldout_after_kpis_used=False,original_plan_or_seals_changed=False,apps=apps,arms=[dict(id=a,app_ids=[apps[k]for k in a])for a in('BCE','BCDE')],
        group_id_prefix='study-supplement-arm:',origins=[dict(origin_key=f'{scene}_{seed}',scene=scene,seed=seed,worker_id=('agx1','agx2')[(i+j)%2],origin_id=f'origin{i}{j}')for i,scene in enumerate(('native_1x','steady_2x','burst_2x'))for j,seed in enumerate((1,2))])
    p['plan_sha256']=original.digest(p)
    assert operational.assignment(p,p['plan_sha256'])==original.assignment(p,p['plan_sha256'])

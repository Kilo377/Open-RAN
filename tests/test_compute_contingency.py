"""Compute relocation cannot alter case identity, duplicate work or hide host."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments import compute_contingency as compute


def protocol():
    p=dict(schema='oran.compute-contingency-protocol.v1',physical_source_sha256=compute.SOURCE_PIN,physical_archive_sha256=compute.ARCHIVE_PIN,
        original_protocol_runner_sha256=compute.PLANNED_WRAPPER_PIN,execution_wrapper_sha256='new-wrapper-sha',
        reference_case=dict(id=compute.REFERENCE_CASE_ID,arm='ABC'),logical_owner='agx1',actual_execution_host='agx2',
        migration_cases=[dict(id=str(i))for i in range(6)])
    p['protocol_sha256']=compute.digest(p);return p


def certificate(p,mode='parity'):
    return dict(schema='oran.compute-contingency-admission.v1',root_approved=True,mode=mode,protocol_sha256=p['protocol_sha256'],
        actual_execution_host='agx2',logical_owner='agx1',physical_source_sha256=compute.SOURCE_PIN,physical_archive_sha256=compute.ARCHIVE_PIN,
        execution_wrapper_sha256=p['execution_wrapper_sha256'],original_protocol_runner_sha256=compute.PLANNED_WRAPPER_PIN,
        actual_ssh_endpoint='lboroldn@100.72.45.61')


def test_old_wrapper_sha_cannot_be_falsely_reused_for_new_execution():
    p=protocol();c=certificate(p)
    assert compute.validate_certificate(c,p,p['protocol_sha256'],'parity','new-wrapper-sha')==c
    with pytest.raises(ValueError):compute.validate_certificate(c,p,p['protocol_sha256'],'parity',compute.PLANNED_WRAPPER_PIN)
    c['execution_wrapper_sha256']=compute.PLANNED_WRAPPER_PIN
    with pytest.raises(ValueError):compute.validate_certificate(c,p,p['protocol_sha256'],'parity','new-wrapper-sha')


def test_migration_needs_separate_root_approval_exact_parity_budget_and_claim():
    p=protocol();c=certificate(p,'migration')
    with pytest.raises(ValueError):compute.validate_certificate(c,p,p['protocol_sha256'],'migration','new-wrapper-sha')
    c.update(exact_parity_passed=True,parity_proof_sha256='proof',runtime_trigger_passed=True,owner_handoff_verified=True)
    assert compute.validate_certificate(c,p,p['protocol_sha256'],'migration','new-wrapper-sha')==c
    meta=compute.execution_metadata(p,c,'certificate-pin',mode='migration',actual_hostname='real-device-hostname')
    assert meta['logical_owner_worker_id']=='agx1'and meta['actual_execution_host']=='agx2'and meta['actual_system_hostname']=='real-device-hostname'
    assert meta['execution_wrapper_sha256']!=meta['original_protocol_runner_sha256']
    assert not meta['additional_independent_replicate']and meta['counts_toward_original132']


def test_execution_host_must_finish_all_its_original_and_supplemental_work():
    done=lambda n:dict(phase='complete',completed=True,running=[],failed=[],done=list(range(n)))
    assert compute.idle_execution_host(done(60),done(6))
    for status in(dict(done(60),running=['active']),dict(done(60),failed=['error']),done(59),dict(done(60),phase='budget_complete')):
        assert not compute.idle_execution_host(status,done(6))


def test_waiting_owner_recheck_cancels_handoff_if_any_task_started_or_main_finished():
    waiting=dict(phase='waiting_for_original_matrix',completed=False,running=[],done=[],failed=[]);main=dict(phase='running',completed=False);ids=[str(i)for i in range(6)]
    assert compute.waiting_owner_eligible(waiting,main,ids)
    assert not compute.waiting_owner_eligible(waiting,dict(main,completed=True),ids)
    for extra in(dict(requests=['1']),dict(partial_case_ids=['2']),dict(checkpoint_ids=['3']),dict(physical_children=[1])):
        assert not compute.waiting_owner_eligible(waiting,main,ids,**extra)
    assert not compute.waiting_owner_eligible(dict(waiting,running=['case']),main,ids)


def test_budget_trigger_uses_wall_times_only_and_requires_meaningful_benefit():
    args=dict(now=0,stop_new_at=14000,agx1_remaining_main=20,agx1_active_walls=[600]*8,agx2_supplement_walls=[300]*6)
    assert compute.runtime_trigger(**args)['triggered']
    assert not compute.runtime_trigger(**{**args,'stop_new_at':20000})['triggered']
    assert not compute.runtime_trigger(**{**args,'agx2_supplement_walls':[3000]*6})['triggered']
    assert not compute.runtime_trigger(**{**args,'agx1_remaining_main':0})['triggered']


def write_execution(tmp_path):
    actual=tmp_path/'actual'/'job';actual.mkdir(parents=True);raw=actual/'data/run/trace';raw.parent.mkdir(parents=True);raw.write_bytes(b'raw physical records')
    guard=raw.parent/'evaluation_origin_verified.json';guard.write_text(json.dumps(dict(origin_id='original-origin',
        verified_before_activation=True,observed_prefix_equal=True,configuration_equal=True,models_equal=True,network_dynamics_code_equal=True)))
    seal=dict(completed=True,case=dict(id='case'),source_sha256=compute.SOURCE_PIN,runner_sha256='new',
        origin_id='original-origin',
        execution_provenance=dict(kind='compute_migrated_supplement',logical_owner_worker_id='agx1',actual_execution_host='agx2',execution_wrapper_sha256='new',parity_proof_sha256='proof'),
        files_sha256={'data/run/trace':compute.sha(raw),'data/run/evaluation_origin_verified.json':compute.sha(guard)})
    path=tmp_path/'actual/checkpoint.json';path.write_text(json.dumps(seal));return actual,path,raw


def test_all_raw_bytes_verified_and_preserved_before_logical_checkpoint_publication(tmp_path):
    actual,seal,raw=write_execution(tmp_path);jobs=tmp_path/'logical/jobs';checkpoints=tmp_path/'logical/checkpoints'
    result=compute.publish_verified_job(actual,seal,jobs,checkpoints)
    assert result['published']and raw.read_bytes()==(jobs/'case/data/run/trace').read_bytes()
    assert (checkpoints/'case.json').read_bytes()==seal.read_bytes()
    assert compute.publish_verified_job(actual,seal,jobs,checkpoints)['existing']


@pytest.mark.parametrize('kind',['wrong_actual_bytes','existing_owner_bytes','wrong_host','fake_old_wrapper'])
def test_invalid_compute_evidence_never_overwrites_existing_owner_or_publishes_seal(tmp_path,kind):
    actual,seal,raw=write_execution(tmp_path);jobs=tmp_path/'logical/jobs';checkpoints=tmp_path/'logical/checkpoints'
    if kind=='wrong_actual_bytes':raw.write_bytes(b'changed')
    elif kind=='existing_owner_bytes':
        path=jobs/'case/data/run/trace';path.parent.mkdir(parents=True);path.write_bytes(b'owner-original')
    else:
        body=json.loads(seal.read_text())
        if kind=='wrong_host':body['execution_provenance']['actual_execution_host']='agx1'
        else:body['runner_sha256']=compute.PLANNED_WRAPPER_PIN
        seal.write_text(json.dumps(body))
    with pytest.raises(ValueError):compute.publish_verified_job(actual,seal,jobs,checkpoints)
    assert not(checkpoints/'case.json').exists()
    if kind=='existing_owner_bytes':assert (jobs/'case/data/run/trace').read_bytes()==b'owner-original'

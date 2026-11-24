"""Relocating one pinned released model must not admit a different override."""
from copy import deepcopy
import hashlib
import json
import pytest
from smo.evaluation_reference import origin_configuration

def config(root):
    path=root+'/outputs/trained_xapps/20261124/models/x12_mec.json'
    return {'configs':{'ric':{'xapps':[{'parameters':{
        'mode':'frozen','model_path':path,'adaptive_policy_path':path}}]}}}

def test_same_released_file_is_portable_between_owned_worker_roots():
    assert origin_configuration(config('/dev/shm/study/source'))==origin_configuration(config('/dev/shm/runtime/source'))

def test_a_distinct_unpinned_override_does_not_become_a_released_file():
    original=config('/dev/shm/runtime/source');different=deepcopy(original)
    different['configs']['ric']['xapps'][0]['parameters']['adaptive_policy_path']='/dev/shm/runtime/source/outputs/research/x12_mec.json'
    assert origin_configuration(original)!=origin_configuration(different)

@pytest.mark.parametrize('runtime_pin,study_archive_pin',[
    ('e'*64,'a'*64),(None,'a'*64),('d'*64,'f'*64),
])
def test_holdout_admission_rejects_another_executed_source_before_reading_any_kpis(tmp_path,runtime_pin,study_archive_pin):
    # Deliberately no KPI observations: this exercises identity admission only.
    from smo.x12_release_validation import validate,SCHEMA
    folder=tmp_path/'artifacts/xapps/chongyu/20261124/evidence';folder.mkdir(parents=True)
    selection=dict(model_sha256='c'*64,heldout_used_for_selection=False,
        evaluation_source_archive_sha256=study_archive_pin,evaluation_source_sha256='d'*64)
    file=folder/'selection.json';file.write_text(json.dumps(selection))
    pin=hashlib.sha256(file.read_bytes()).hexdigest()
    report=dict(schema=SCHEMA,role='holdout',synthetic_fixture=False,
        selected_model_sha256='c'*64,selection_sha256=pin,source_archive_sha256='a'*64)
    file=folder/'final-holdout-report.json';file.write_text(json.dumps(report))
    release=dict(final_report_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),selection_sha256=pin,
        source_archive_sha256='a'*64,models={'x12_mec':{'sha256':'c'*64}})
    with pytest.raises(ValueError,match='executed study source differs'):
        validate(release,{'model_release':{'id':'20261124'},'validated_source_sha256':runtime_pin},{},tmp_path)

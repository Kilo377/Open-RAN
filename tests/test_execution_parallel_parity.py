import hashlib
import json
from pathlib import Path

from experiments import execution_parallel_parity as parity


def fixture(root, ident, *, number=1.0, unknown=False, model_change=False):
    job = root/'jobs'/ident
    run = job/'data/run'
    run.mkdir(parents=True)
    def put(name, value):
        p = run/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
    put('config_snapshot.json', dict(source_sha256=parity.policy()['physical_source_sha256'],
        manifest=dict(total_slots=2, rng_seed=1), configs=dict(ric=dict(xapps=[]))))
    put('run_status.json', dict(completed=True, source_sha256=parity.policy()['physical_source_sha256'],
        replay=dict(audits_passed=True)))
    models = {}
    for i in range(6):
        name = 'model'+str(i)
        path = 'models/'+name+'.json'
        put(path, dict(weight=2 if model_change and i == 0 else 1))
        models[name] = dict(checkpoint=path, sha256=parity.sha(run/path))
    put('xapp_hot_start.json', dict(models=models))
    for name in (*parity.exact.JSONL, *parity.exact.OPTIONAL_JSONL):
        (run/(name+'.jsonl')).write_text('')
    (run/'rt_sys_trace.jsonl').write_text(json.dumps(dict(slot_id=1, counter=number))+'\n'+json.dumps(dict(slot_id=2, counter=2))+'\n')
    for name in parity.exact.CSV:
        (run/(name+'.csv')).write_text('slot,value\n1,1\n2,2\n')
    if unknown:
        (run/'unregistered.jsonl').write_text('{}\n')
    seal = dict(completed=True, source_sha256=parity.policy()['physical_source_sha256'],
        worker_id='agx2', case=dict(id=ident, seed=1, app_ids=[]),
        frozen_models={str(i):str(i) for i in range(6)},
        files_sha256={str(p.relative_to(job)):parity.sha(p) for p in job.rglob('*') if p.is_file()})
    cp = root/'checkpoints'/(ident+'.json')
    cp.parent.mkdir(exist_ok=True)
    cp.write_text(json.dumps(seal))
    return cp


def test_exact_ordered_network_records_and_numeric_spelling(tmp_path):
    a = fixture(tmp_path/'a', 'a'*32, number=1)
    b = fixture(tmp_path/'b', 'b'*32, number=1.0)
    result = parity.compare(a, b)
    assert result['passed'], result
    assert result['streams']['rt_sys_trace.jsonl']['reference']['records'] == 2


def test_small_numeric_difference_cannot_be_rounded_away(tmp_path):
    a = fixture(tmp_path/'a', 'a'*32)
    b = fixture(tmp_path/'b', 'b'*32, number=1.0000000000000002)
    result = parity.compare(a, b)
    assert not result['passed']
    assert result['exact_difference_count'] == 1


def test_unknown_network_stream_fails(tmp_path):
    a = fixture(tmp_path/'a', 'a'*32)
    b = fixture(tmp_path/'b', 'b'*32, unknown=True)
    assert not parity.compare(a, b)['passed']


def test_changed_copied_model_fails_even_with_updated_local_pin(tmp_path):
    a = fixture(tmp_path/'a', 'a'*32)
    b = fixture(tmp_path/'b', 'b'*32, model_change=True)
    assert not parity.compare(a, b)['passed']


def test_modified_raw_file_fails_checkpoint_verification(tmp_path):
    a = fixture(tmp_path/'a', 'a'*32)
    b = fixture(tmp_path/'b', 'b'*32)
    (b.parent.parent/'jobs'/('b'*32)/'data/run/rt_sys_trace.jsonl').write_text('')
    assert not parity.compare(a, b)['passed']


def test_off_inventory_allows_only_absence_of_inactive_app_logs(tmp_path):
    cp = fixture(tmp_path/'a', 'a'*32)
    run = cp.parent.parent/'jobs'/('a'*32)/'data/run'
    for name in parity.APP_STREAMS.values():
        (run/name).unlink()
    assert parity.inventory(run)  # No algorithm ran; core and shadow streams remain.


def test_active_app_log_and_core_streams_remain_required(tmp_path):
    import pytest
    cp = fixture(tmp_path/'a', 'a'*32)
    run = cp.parent.parent/'jobs'/('a'*32)/'data/run'
    (run/'energy_rf_events.jsonl').unlink()
    with pytest.raises(ValueError, match='Missing physical streams'):
        parity.inventory(run, ['xapp:ue_energy_rf'])
    (run/'xapp_decision_previews.jsonl').unlink()
    with pytest.raises(ValueError, match='Missing physical streams'):
        parity.inventory(run)


def test_present_inactive_stream_is_still_compared_in_full(tmp_path):
    a = fixture(tmp_path/'a', 'a'*32)
    b = fixture(tmp_path/'b', 'b'*32)
    run = b.parent.parent/'jobs'/('b'*32)/'data/run'
    (run/'ap_rf_events.jsonl').write_text('{"decision":1}\n')
    seal = json.loads(b.read_text())
    seal['files_sha256']['data/run/ap_rf_events.jsonl'] = parity.sha(run/'ap_rf_events.jsonl')
    b.write_text(json.dumps(seal))
    result = parity.compare(a, b)
    assert not result['passed']
    assert result['streams']['ap_rf_events.jsonl']['exact_equal'] is False

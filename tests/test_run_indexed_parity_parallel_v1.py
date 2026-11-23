import json
from pathlib import Path

import pytest

from experiments import run_indexed_parity_parallel_v1 as w


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def plans(tmp_path):
    roots=[tmp_path/'old',tmp_path/'new']
    for r in roots:
        for worker,seed in [('agx1',14),('agx2',15)]:
            cases=[dict(id=worker+arm,arm=arm,worker_id=worker,seed=seed,app_ids=apps)
                   for arm,apps in w.core.ARMS.items()]
            put(r/'protocol/pilot'/(worker+'.json'),dict(worker_id=worker,cases=cases))
            for c in cases:put(r/'runs/workers'/worker/'checkpoints'/(c['id']+'.json'),{})
    return roots


def test_stable_complete_grid(tmp_path):
    roots=plans(tmp_path);jobs=w.build_jobs(*roots)
    assert len(jobs)==14
    assert [(x['worker_id'],x['arm'])for x in jobs]==[(worker,arm)for worker in['agx1','agx2']for arm in w.core.ARMS]
    assert all(len(j['input_pins'])==4 for j in jobs)


def test_cannot_duplicate_host_or_drop_arm(tmp_path):
    roots=plans(tmp_path);p=roots[1]/'protocol/pilot/agx2.json';body=json.loads(p.read_text());body['worker_id']='agx1';put(p,body)
    with pytest.raises(ValueError,match='same-host'):w.build_jobs(*roots)


def test_snapshot_pins_fail_closed(tmp_path):
    p=tmp_path/'input';p.write_text('one');pins={str(p):w.core.sha(p)};w.require_pins(pins);p.write_text('two')
    with pytest.raises(ValueError,match='changed'):w.require_pins(pins)


def test_worker_calls_unmodified_comparison_and_preserves_failure(tmp_path,monkeypatch):
    jobs=w.build_jobs(*plans(tmp_path));j=jobs[0];calls=[]
    expected=dict(passed=False,non_metadata_differences=[dict(file='trace',pointer='/request_id')])
    monkeypatch.setattr(w.core,'compare_pair',lambda *args:(calls.append(args),expected)[1])
    pins=w.code_pins(w.core.sha(w.core.__file__))
    result=w.compute_job(j,pins)
    assert calls==[tuple(j['args'])] and result['core_result']==expected
    assert result['core_result_sha256']==w.value_sha(expected)
    assert result['execution']['pid']>0


def test_cache_lookup_keeps_order_and_restores_function(tmp_path,monkeypatch):
    roots=plans(tmp_path);jobs=w.build_jobs(*roots);computed={tuple(j['args']):dict(core_result={'pair':j['index'],'passed':False})for j in jobs}
    original=w.core.compare_pair
    def aggregate(ref,can,out,**kwargs):
        result=[w.core.compare_pair(*j['args'])for j in jobs]
        result[0]['pair']='mutated copy'
        return dict(trial_pairs=result,passed=False)
    monkeypatch.setattr(w.core,'compare_study',aggregate)
    result=w.cached_assembly(*roots,tmp_path/'out',tmp_path/'oracle',tmp_path/'diff',jobs,computed)
    assert w.core.compare_pair is original
    assert computed[tuple(jobs[0]['args'])]['core_result']['pair']==0
    assert result['passed']is False


def test_cache_cannot_skip_or_repeat_a_pair(tmp_path,monkeypatch):
    roots=plans(tmp_path);jobs=w.build_jobs(*roots);computed={tuple(j['args']):dict(core_result={'passed':True})for j in jobs}
    original=w.core.compare_pair
    def aggregate(*args,**kwargs):
        w.core.compare_pair(*jobs[0]['args']);w.core.compare_pair(*jobs[0]['args'])
    monkeypatch.setattr(w.core,'compare_study',aggregate)
    with pytest.raises(ValueError,match='duplicate'):
        w.cached_assembly(*roots,tmp_path/'out',tmp_path/'oracle',tmp_path/'diff',jobs,computed)
    assert w.core.compare_pair is original


def test_save_never_overwrites(tmp_path):
    p=tmp_path/'proof.json';w.save(p,{'passed':False})
    with pytest.raises(ValueError,match='Preserve'):w.save(p,{'passed':True})
    assert json.loads(p.read_text())=={'passed':False}


def test_result_regression_is_typed_and_exact():
    assert w.encoded({'x':1})!=w.encoded({'x':True})
    assert w.encoded({'x':.10000000000000002})!=w.encoded({'x':.10000000000000003})
    assert w.value_sha({'b':[1,2],'a':3})==w.value_sha({'a':3,'b':[1,2]})
    assert w.value_sha({'a':[1,2]})!=w.value_sha({'a':[2,1]})


def test_fresh_verify_checks_entire_raw_inputs_again(tmp_path,monkeypatch):
    jobs=w.build_jobs(*plans(tmp_path));computed={tuple(j['args']):dict(core_result={})for j in jobs};calls=[]
    monkeypatch.setattr(w.core,'verify',lambda *args:(calls.append(args),({}, {},None,{},None))[1])
    w.fresh_verify(jobs,computed)
    assert len(calls)==28
    Path(jobs[0]['args'][0]).write_text('changed')
    with pytest.raises(ValueError,match='changed'):w.fresh_verify(jobs,computed)


def test_spawned_analysis_worker_imports_pinned_core_without_simulating():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    pin=w.core.sha(w.core.__file__)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'))as pool:
        results=list(pool.map(w.code_pins,[pin,pin]))
    assert results[0]==results[1]==w.code_pins(pin)

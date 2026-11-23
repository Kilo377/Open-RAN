"""No physical/network actions: indexed wrapper identity, admission and resource contracts."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

import pytest
from experiments import environment_validation_worker_v4 as v4
from experiments import environment_validation_worker_v5 as w

ROOT=Path(__file__).resolve().parents[1]
E=Path('/Users/kilobao/Desktop/ORAN/SMO/environment-validation-20261122')
GiB=1024**3

def pinned(plan):
    plan=deepcopy(plan);plan.pop('plan_sha256',None);plan['plan_sha256']=w.digest(plan);return plan

def plan(mode='parity_validation',parallel=6):
    p=json.loads((E/'common-validation-loaded/protocol/validation/agx1.json').read_text())
    def rewrite(x):
        if isinstance(x,str):return x.replace('/source/','/source-indexed-v1/').replace('/source-v2.tgz','/source-indexed-v1.tgz').replace('/source','/source-indexed-v1')if x=='/dev/shm/oran_environment_20261122/source'else x.replace('/source/','/source-indexed-v1/').replace('/source-v2.tgz','/source-indexed-v1.tgz')
        if isinstance(x,dict):return{k:rewrite(v)for k,v in x.items()}
        if isinstance(x,list):return[rewrite(v)for v in x]
        return x
    p=rewrite(p);p.update(schema=w.SCHEMA,source_sha256=w.SOURCE_SHA256,source_archive_sha256=w.ARCHIVE_SHA256,
        runner_path=str(w.SCOPE/'runner/worker-v5.py'),runner_sha256=w.sha(w.__file__),max_parallel=parallel,
        baseline_source_sha256=w.BASELINE_SOURCE_SHA256,baseline_source_archive_sha256=w.BASELINE_ARCHIVE_SHA256,
        source_equivalence_path=str(w.source_equivalence_path()),source_equivalence_sha256=w.SOURCE_EQUIVALENCE_SHA256,
        source_trial_approval_path=str(w.source_trial_approval_path()),source_trial_approval_sha256=w.SOURCE_TRIAL_APPROVAL_SHA256,
        indexed_source_admission_path=str(w.indexed_admission_path()),indexed_source_mode=mode)
    if mode=='parity_validation':
        for c in p['cases']:c.update(slots=4440,observe_slots=2000)
    return pinned(p)


def test_true_indexed_plan_and_six_cap():
    p=plan();assert w.validate_plan(p,p['plan_sha256'])==p

@pytest.mark.parametrize('key,value',[
    ('source_sha256',w.BASELINE_SOURCE_SHA256),('source_archive_sha256',w.BASELINE_ARCHIVE_SHA256),
    ('source_root',str(w.SCOPE/'source')),('source_archive_path',str(w.SCOPE/'source-v2.tgz')),
    ('runner_path',str(w.SCOPE/'runner/worker-v4.py')),('source_equivalence_sha256','f'*64),
    ('baseline_source_sha256','a'*64),('source_trial_approval_sha256','b'*64),
    ('indexed_source_mode','adopted'),('max_parallel',7),('require_on_admission',False),
    ('host_memory_reserve_bytes',GiB),('case_peak_rss_bytes',0)])
def test_bad_plan_fields_fail_closed(key,value):
    p=plan();p[key]=value;p=pinned(p)
    with pytest.raises(ValueError):w.validate_plan(p,p['plan_sha256'])

def test_parity_trial_not_full_primary():
    p=plan();p['cases'][0]['slots']=22440;p['cases'][0]['observe_slots']=20000;p=pinned(p)
    with pytest.raises(ValueError,match='4440'):w.validate_plan(p,p['plan_sha256'])

def test_full_validation_is_explicit_separate_mode():
    p=plan(mode='primary_validation');assert p['cases'][0]['slots']==22440;w.validate_plan(p,p['plan_sha256'])

def test_original_children_scheduler_cache_launcher_ast_are_unchanged():
    trees=[ast.parse(Path(m.__file__).read_text())for m in[v4,w]]
    definitions=[{n.name:ast.dump(n,include_attributes=False)for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}for t in trees]
    for name in['OwnedChildren','schedule_cases','case_cache_root','activated_command','validate_installed_modelset','resource_admission','ui_admission','_runtime','atomic']:
        assert definitions[0][name]==definitions[1][name],name
    for name in['CASE_LAUNCHER','ENVIRONMENT_SNAPSHOT_CODE','SIX_UE_FIXED_ARGV','SIX_UE_LAUNCHER_SHA256','CACHE_ENV_KEYS']:
        assert getattr(v4,name)==getattr(w,name)
    assert w.sha(v4.__file__)=='0aa03b21946b0baa600ab25e24502ef4439fe08e503304ce70aee8209b26edb2'


def memory(available,active=0,rss=()):return w.memory_admission(dict(available_memory_bytes=available*GiB),active_count=active,rss_bytes=[v*GiB for v in rss],case_peak_rss_bytes=7*GiB,host_memory_reserve_bytes=8*GiB)

def test_six_is_a_cap_not_memory_guarantee():
    assert memory(50,4,[0,0,0,0])['allowed']
    assert not memory(46,5,[0]*5)['allowed']
    assert memory(50,5,[1]*5)['allowed']

def test_unknown_starting_futures_reserve_full_peak():
    result=memory(50,5,[2]);assert result['already_admitted_future_growth_reservation_bytes']==33*GiB
    assert result['active_measured_rss_bytes']==[2*GiB,0,0,0,0]

def test_already_used_rss_is_not_double_reserved():
    result=memory(15,1,[7]);assert result['allowed'];assert result['required_available_memory_bytes']==15*GiB

@pytest.mark.parametrize('rss',[[-1],[True],['bad']])
def test_invalid_rss_is_not_silently_admitted(rss):
    with pytest.raises(ValueError):w.memory_admission(dict(available_memory_bytes=100*GiB),active_count=1,rss_bytes=rss,case_peak_rss_bytes=7*GiB,host_memory_reserve_bytes=8*GiB)


def test_off_gate_is_separate_from_on_marker(tmp_path):
    p=plan();assert w.on_admission(p,p['cases'][0],tmp_path)['allowed'];assert not w.on_admission(p,p['cases'][1],tmp_path)['allowed']


def marker(p):return dict(plan_sha256=p['plan_sha256'],source_sha256=p['source_sha256'],runner_sha256=p['runner_sha256'],purpose='indexed_source_parity_validation',scientific_validation_claimed=False)

def test_bounded_parity_marker_does_not_claim_adoption(tmp_path):
    p=plan();(tmp_path/'ON_ADMISSION_ALLOWED').write_text(json.dumps(marker(p)));assert w.on_admission(p,p['cases'][1],tmp_path)['allowed']

@pytest.mark.parametrize('key,value',[('plan_sha256','x'),('source_sha256',w.BASELINE_SOURCE_SHA256),('scientific_validation_claimed',True)])
def test_wrong_or_overclaiming_marker_blocks_on(tmp_path,key,value):
    p=plan();m=marker(p);m[key]=value;(tmp_path/'ON_ADMISSION_ALLOWED').write_text(json.dumps(m));assert not w.on_admission(p,p['cases'][1],tmp_path)['allowed']


def physical(p):return dict(schema='oran.indexed-source-physical-prefix-admission.v1',approval_kind='primary_validation',approved=True,source_sha256=p['source_sha256'],source_archive_sha256=p['source_archive_sha256'],runner_sha256=p['runner_sha256'],OFF_prefix_equal=True,ON_control_KPI_prefix_equal=True,does_not_claim_full_run_parity=True,maximum_parallelism_authorized=6)

def test_validation_needs_actual_prefix_certificate_and_marker_pin(tmp_path):
    with patch.object(w,'SCOPE',tmp_path):
        p=plan(mode='primary_validation');cert=w.indexed_admission_path();cert.parent.mkdir(parents=True);out=tmp_path/'outputs';out.mkdir();m=marker(p);(out/'ON_ADMISSION_ALLOWED').write_text(json.dumps(m));assert not w.on_admission(p,p['cases'][1],out)['allowed']
        cert.write_text(json.dumps(physical(p)));m['indexed_source_admission_sha256']=w.sha(cert);(out/'ON_ADMISSION_ALLOWED').write_text(json.dumps(m));assert w.on_admission(p,p['cases'][1],out)['allowed']

@pytest.mark.parametrize('key,value',[('OFF_prefix_equal',False),('ON_control_KPI_prefix_equal',False),('maximum_parallelism_authorized',5),('does_not_claim_full_run_parity',False)])
def test_insufficient_actual_proof_cannot_authorize_six(tmp_path,key,value):
    with patch.object(w,'SCOPE',tmp_path):
        p=plan(mode='primary_validation');cert=w.indexed_admission_path();cert.parent.mkdir(parents=True);body=physical(p);body[key]=value;cert.write_text(json.dumps(body));out=tmp_path/'outputs';out.mkdir();m=marker(p);m['indexed_source_admission_sha256']=w.sha(cert);(out/'ON_ADMISSION_ALLOWED').write_text(json.dumps(m));assert not w.on_admission(p,p['cases'][1],out)['allowed']


def test_bounded_six_way_scheduler_off_barrier_without_physics():
    lock=threading.Lock();all_on=threading.Event();active=maximum=0;off_done=False
    cases=[dict(id='OFF',app_ids=[])]+[dict(id=str(i),app_ids=['x'])for i in range(6)]
    def execute(c):
        nonlocal active,maximum,off_done
        with lock:
            if not c['app_ids']:
                assert active==0;off_done=True;return dict(id=c['id'],phase='complete')
            assert off_done;active+=1;maximum=max(maximum,active)
            if active==6:all_on.set()
        assert all_on.wait(3)
        with lock:active-=1
        return dict(id=c['id'],phase='complete')
    result=w.schedule_cases(cases,max_parallel=6,execute=execute,admission=lambda *_:dict(allowed=True),stop_reason=lambda:None,stop_event=threading.Event(),stop_children=lambda:None,publish=lambda *_:None,poll=.001)
    assert len(result['done'])==7 and maximum==6 and not result['failed']


def coordinator():
    spec=importlib.util.spec_from_file_location('indexed_coordinator',ROOT/'scripts/run-environment-indexed.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_actual_source_diff_preflight_without_ssh_or_simulator():
    m=coordinator()
    with patch.object(m,'ssh',side_effect=AssertionError('No SSH in local preflight')):
        proof=m.validate_source_inputs()
    assert proof['changed_members']==sorted(w.INDEXED_MEMBERS)and proof['candidate_regular_members']==1388 and proof['physical_parity_claimed']is False


def test_all_remote_python_source_blocks_compile_and_plan_bytes_are_json():
    module=ast.parse((ROOT/'scripts/run-environment-indexed.py').read_text());blocks=[n.value for n in ast.walk(module)if isinstance(n,ast.Constant)and isinstance(n.value,str)and 'c=json.loads(sys.argv[1])'in n.value]
    assert len(blocks)>=4
    found=False
    for block in blocks:
        tree=ast.parse(block);compile(tree,'remote-code','exec')
        for call in ast.walk(tree):
            if isinstance(call,ast.Call)and isinstance(call.func,ast.Name)and call.func.id=='install'and len(call.args)==2 and isinstance(call.args[0],ast.BinOp)and isinstance(call.args[0].right,ast.Constant)and call.args[0].right.value=='plan.json':
                body=eval(compile(ast.Expression(call.args[1]),'plan-encoding','eval'),dict(json=json,p={'source':'indexed'}));assert json.loads(body)=={'source':'indexed'};found=True
    assert found

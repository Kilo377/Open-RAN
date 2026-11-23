"""Preregistration, paired-prefix ordering and isolated physical run controls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.quantification_study import assignment,command,completed,digest,isolated_environment,verify_plan


def plan():
    letters='ABCDEF';apps={letter:'xapp:'+letter.lower()for letter in letters}
    arms=['OFF',*letters,'AC','AD','AF','BC','BD','BE','CD','CE','DE','DF','ABC','ADF','CDE']
    value=dict(schema='oran.quantification-study-plan.v1',seeds=[20261201,20261202],scenarios=['native_1x','steady_2x','burst_2x'],
        observe_slots=2000,baseline_slot=2040,total_slots=4440,apps=apps,randomization_seed=20261122,
        arms=[dict(id=a,app_ids=[]if a=='OFF'else[apps[x]for x in a])for a in arms],physical_run_count=120)
    value['plan_sha256']=digest(value)
    return value


def test_full_grid_is_unique_balanced_and_off_precedes_its_candidate_arms():
    p=plan();schedule=assignment(p,p['plan_sha256'])
    assert {k:len(v)for k,v in schedule.items()}=={'agx2':60,'agx1':60}
    rows=[c for cases in schedule.values()for c in cases]
    assert len({r['id']for r in rows})==120
    assert len({(r['scene'],r['seed'],r['arm'])for r in rows})==120
    for cases in schedule.values():
        seen=set()
        for case in cases:
            if case['arm']=='OFF':seen.add(case['origin_key'])
            else:assert case['origin_key']in seen
    for scene in p['scenarios']:
        assert {w for w,cases in schedule.items()if any(c['scene']==scene for c in cases)}=={'agx1','agx2'}
    assert assignment(p,p['plan_sha256'])==schedule


def test_changed_preregistered_protocol_cannot_be_silently_admitted():
    p=plan();pin=p['plan_sha256'];p['observe_slots']=500
    with pytest.raises(ValueError,match='digest'):verify_plan(p,pin)
    p['plan_sha256']=digest({k:v for k,v in p.items()if k!='plan_sha256'})
    with pytest.raises(ValueError,match='protocol'):verify_plan(p,p['plan_sha256'])


def test_physical_command_changes_traffic_context_but_uses_normal_trained_hot_apps():
    p=plan();case=next(c for cases in assignment(p,p['plan_sha256']).values()for c in cases if c['scene']=='burst_2x'and c['arm']!='OFF')
    args=command(Path('/owned/source'),Path('/owned/source/outputs/quantification_study'),case,p)
    assert args[args.index('--traffic-load')+1]=='2.0'
    assert args[args.index('--urllc-scenario')+1]=='burst'
    assert args[args.index('--xapps-start')+1]==''
    assert '--live-xapps'in args and '--xapp-toggle-timeline'in args
    assert '--evaluation-origin-file'in args and '--xapp-preview-before-slot'in args
    assert args[args.index('--xapp-preview-before-slot')+1]=='2040'
    assert '--perturbation-spec'not in args and '--fixed-control-timeline'not in args
    off=deepcopy(case);off.update(arm='OFF',app_ids=[])
    assert '--evaluation-origin-file'not in command(Path('/owned/source'),Path('/owned/source/outputs/quantification_study'),off,p)


def test_python_gpu_and_package_caches_point_only_into_owned_copy():
    root=Path('/dev/shm/oran_quantification_20261122/source');env=isolated_environment(root)
    for key in('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','CUDA_CACHE_PATH','CUPY_CACHE_DIR','TORCH_HOME',
        'TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR','NUMBA_CACHE_DIR','HF_HOME','MPLCONFIGDIR','DRJIT_CACHE_DIR','UV_CACHE_DIR','PIP_CACHE_DIR'):
        assert Path(env[key]).is_relative_to(root)
    assert env['PYTHONDONTWRITEBYTECODE']=='1'


def test_resume_requires_every_sealed_raw_file_checksum(tmp_path):
    case=dict(id='a'*32);folder=tmp_path/'jobs'/case['id'];folder.mkdir(parents=True);raw=folder/'data.json';raw.write_text('retained')
    (tmp_path/'checkpoints').mkdir();seal=tmp_path/'checkpoints'/(case['id']+'.json')
    import hashlib
    seal.write_text(json.dumps(dict(completed=True,files_sha256={'data.json':hashlib.sha256(raw.read_bytes()).hexdigest()})))
    assert completed(tmp_path,case)
    raw.write_text('changed')
    assert not completed(tmp_path,case)


def test_operational_queue_round_robin_preserves_the_full_registered_grid():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/run-quantification-study.py'
    spec=importlib.util.spec_from_file_location('quant_study_coordinator',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=plan();original=assignment(p,p['plan_sha256'])['agx1'];saved=deepcopy(original)
    interleaved=module.round_robin_cases(original)
    assert original==saved and sorted(x['id']for x in interleaved)==sorted(x['id']for x in original)
    assert all(c['arm']=='OFF'for c in interleaved[:3])
    assert len({c['origin_key']for c in interleaved[3:6]})==3
    for origin in {c['origin_key']for c in original}:
        assert [c['id']for c in interleaved if c['origin_key']==origin]==[c['id']for c in original if c['origin_key']==origin]

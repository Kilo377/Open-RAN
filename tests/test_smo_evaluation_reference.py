"""Controlled BEFORE alignment; no fixture claims measured network benefits."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from smo import evaluation_reference as ref


def fixture(tmp_path, monkeypatch):
    frame=dict(slot=400,time_ms=200,states={'rcp:p':{'by_entity':{'ue_0':1}}})
    preview=dict(slot=400,status='ready',apps=[dict(app_id='a',mode='cold_activation',
        input_sha256='device-a',actions=[dict(parameter='p',current=1,proposed=2)])])
    batch=dict(id='b',seed=7,members=[])
    for i in range(2):
        job=str(i);path=tmp_path/job/'data/run';path.mkdir(parents=True)
        (path/'config_snapshot.json').write_text(json.dumps(dict(manifest={'rng_seed':7},
            configs={'ric':{'xapps':[{'parameters':{'log_path':str(path/'log')}}]}})))
        batch['members'].append(dict(job_id=job,phase='complete',comparison=dict(
            baseline={'slot':520},application_slot=561,baseline_origin='fresh_run_all_xapps_off',
            rows=[dict(kpi='k',before={'value':10.0+i*1e-10},after={'value':-123})])))
    batch['observe_slots']=500
    monkeypatch.setattr(ref,'replay',lambda *a,**k:dict(frames=[deepcopy(frame)],slot_duration_ms=.5))
    monkeypatch.setattr(ref,'records',lambda *a:[dict(slot=400),dict(slot=600)])
    class Store:
        def __init__(self,path):self.path=path
        def read(self,slot):
            r=deepcopy(preview);r['apps'][0]['input_sha256']=str(self.path);return r
    monkeypatch.setattr(ref,'PreviewStore',Store)
    return batch,frame,preview


def test_before_reference_checks_config_states_proposals_without_after_labels(tmp_path,monkeypatch):
    batch,_,_=fixture(tmp_path,monkeypatch);original=deepcopy(batch)
    first=ref.load(batch,tmp_path)
    assert first['status']=='ready'
    r=first['reference'];assert r['decision_slot']==400 and r['baseline_slot']==520
    assert r['decision_to_baseline_ms']==60 and r['matched_completed_runs']==2
    assert not r['public_input_hashes_equal'] and not r['after_kpis_used']
    assert not r['intent_bounds_rebased'] and batch==original
    batch['members'][0]['comparison']['rows'][0]['after']={'value':999,'status':'satisfied'}
    assert ref.load(batch,tmp_path)==first
    # A future preview cannot supply the predeployment policy.
    monkeypatch.setattr(ref,'records',lambda *a:[dict(slot=600)])
    assert ref.load(batch,tmp_path)['reason']=='evaluation_baseline_preview_missing'


@pytest.mark.parametrize('change',['config','before','state','proposal','activated'])
def test_mismatched_control_conditions_block_scoring(tmp_path,monkeypatch,change):
    batch,frame,preview=fixture(tmp_path,monkeypatch)
    if change=='config':
        p=tmp_path/'1/data/run/config_snapshot.json';c=json.loads(p.read_text());c['manifest']['rng_seed']=8;p.write_text(json.dumps(c))
    elif change=='before':batch['members'][1]['comparison']['rows'][0]['before']['value']=11
    elif change=='state':
        def replay(path,*a,**k):
            f=deepcopy(frame)
            if path.parts[-3]=='1':f['states']['rcp:p']['by_entity']['ue_0']=2
            return dict(frames=[f],slot_duration_ms=.5)
        monkeypatch.setattr(ref,'replay',replay)
    elif change=='proposal':
        class Store:
            def __init__(self,path):self.path=path
            def read(self,slot):
                p=deepcopy(preview)
                if self.path.parts[-3]=='1':p['apps'][0]['actions'][0]['proposed']=3
                return p
        monkeypatch.setattr(ref,'PreviewStore',Store)
    else:preview['apps'][0]['mode']='state_clone'
    assert ref.load(batch,tmp_path)['status']=='unavailable'


def test_pending_reference_is_not_a_zero_risk_result(tmp_path,monkeypatch):
    batch,_,_=fixture(tmp_path,monkeypatch)
    for m in batch['members']:m['phase']='running'
    assert ref.load(batch,tmp_path)==dict(status='unavailable',reason='evaluation_baseline_pending')


def test_baseline_preview_is_captured_before_selection_once_without_mutating_apps(tmp_path,monkeypatch):
    from experiments.xapp_preview import PreviewRecorder
    import experiments.xapp_preview as recorder
    app=SimpleNamespace(name='a',active_cells=[])
    active={};manager=SimpleNamespace(instance_selection=SimpleNamespace(apps=lambda:active))
    sink=PreviewRecorder.__new__(PreviewRecorder)
    sink.period=200;sink.before_slots={200};sink.apps=[app];sink.path=tmp_path/'previews.jsonl';sink.latest=None
    calls=[]
    def capture(manager,observation,slot,apps):
        calls.append(deepcopy(apps));return dict(slot=slot,apps=[],deployment_enabled=False)
    monkeypatch.setattr(recorder,'capture',capture)
    sink.before_selection(manager,{},200)
    active['a']=SimpleNamespace(name='a',active_cells=['cell_0'])
    sink(manager,{},200)
    assert len(calls)==1 and calls[0][0].active_cells==[] and app.active_cells==[]
    rows=[json.loads(s)for s in sink.path.read_text().splitlines()]
    assert len(rows)==1 and rows[0]['capture_phase']=='before_selection'

"""Admission, display units and archived data must share honest semantics."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from smo.__main__ import GraphService
from smo.intent_language import IntentError


def test_unprepared_or_wrong_length_origin_cannot_be_deployed():
    service=GraphService();frame={'slot':520,'evaluation_origin':{'origin_id':'same'}}
    with pytest.raises(IntentError,match='对齐评估起点'):service.evaluation_context(frame,500)
    service.evaluation_origin=dict(origin_id='same',observe_slots=500)
    with pytest.raises(IntentError):service.evaluation_context(frame,800)
    with pytest.raises(IntentError):service.evaluation_context(dict(slot=520),500)
    assert service.evaluation_context(frame,500)['evaluation_origin']['origin_id']=='same'


def test_preparing_origin_preserves_old_intents_and_uses_before_only(monkeypatch):
    service=GraphService()
    service.intent_ledger.records={'old':dict(id='old',snapshot={'slot':1420},ranges=[])}
    original=deepcopy(service.intent_ledger.records)
    proof=dict(origin_id='prepared',baseline_slot=520,observe_slots=500,before_frame={'states':{}})
    frame=dict(slot=520,time_ms=260,states={'kpi:ue_power_w':{'value':5.5}})
    monkeypatch.setattr('smo.evaluation_reference.prepare_origin',lambda *a,**k:dict(
        status='ready',frame=frame,preview={'slot':520,'apps':[{'actions':[
            dict(parameter='scheduler_mode',cell_id='c',current='round_robin',proposed='qos_deadline'),
            dict(parameter='ue_scheduler_weights',ue_id='u',current=1.,proposed=8.)]}]},proof=proof,
        run_path='/owned/recorded/run',source_job_id='before-source'))
    service.deployments=SimpleNamespace(config={'seed':1},batch=SimpleNamespace(items={
        'b':{'seed':1,'created_unix_s':1}}),output=Path('/owned'),status=lambda:{})
    bundle=service.prepare_evaluation_origin()
    assert bundle['source']['frames']==[frame]
    assert not bundle['source']['after_kpis_used']
    assert bundle['evaluation_origin']['status']=='ready'
    assert service.intent_ledger.records==original
    assert service.preview_xapps(520)['slot']==520
    assert frame['states']['rcp:scheduler_mode']['value']=='round_robin'
    assert frame['states']['rcp:ue_scheduler_weights']['value']==1.
    assert proof['before_frame']=={'states':{}}


def test_ratio_display_does_not_relabel_archived_event_rates_and_regression_is_visible():
    root=Path(__file__).resolve().parents[1]
    js="""const fs=require('fs'),vm=require('vm'),c={};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);const g=c.SmoGraph;
process.stdout.write(JSON.stringify({ratio:g.metricDisplayValue('urllc_failure_fraction',.07),
ratioUnit:g.metricDisplayUnit('urllc_failure_fraction','ratio'),
events:g.metricDisplayValue('urllc_failure_events_per_s',7),
eventsUnit:g.metricDisplayUnit('urllc_failure_events_per_s','events/s'),
partial:g.evaluationPresentation({after:{complete:true,all_satisfied:false,score:1},no_regression:true}),
regressed:g.evaluationPresentation({after:{complete:true,all_satisfied:true,score:-1},no_regression:false})}));"""
    result=json.loads(subprocess.check_output(['node','-e',js],text=True,cwd=root))
    assert result['ratio']==pytest.approx(7) and result['ratioUnit']=='%'
    assert result['events']==7 and result['eventsUnit']=='events/s'
    assert result['partial']['tone']=='partial' and result['regressed']['tone']=='regressed'

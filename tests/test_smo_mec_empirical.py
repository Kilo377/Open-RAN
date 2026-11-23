"""Analytic regression fixtures, not claimed simulator/xApp improvements."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np
import pytest

from smo.mec_empirical import fit_dataset,estimate_change,load_model
from experiments.fit_mec_local_response import extract_job,sha


def regression_fixture(two_weights=False):
    context=dict(cells=['c'],users=['u','v']if two_weights else['u'],native_drx={'u':dict(enabled=True)},
        native_offload={'u':.5},source_sha256='fixture',ue_services={},cell_radio={},ue_config=[])
    names=['mode:c:latency_aware','mode:c:qos_deadline','weight:u']+(['weight:v']if two_weights else[])
    rows=[]
    for phase in range(2):
        for i,w in enumerate((.5,.75,1.,1.25,1.5,2.)):
            rows.append(dict(x=[1.,0.,w]+([w]if two_weights else[]),y=7+3*w+phase*10,
                stratum=str(phase),trajectory_sha256=str(i),source_job_ids=[f'{phase}:{i}'],features=names,
                context=context,context_sha256='same',duration_ms=250))
    return dict(rows=rows,duplicate_response_discrepancies=[],independent_seeds=[1],window_slots=500,raw_job_count=12,usable_before_dedup=12)


def state(w=1.,two_weights=False):
    return dict(scheduler_mode={'c':'latency_aware'},ue_scheduler_weights={'u':w,**({'v':w}if two_weights else{})},
        ue_drx_configs={'u':dict(enabled=True)},ue_offload_fraction={'u':.5},cell_sleep_mode={'c':'active'})


def test_fixed_effects_fit_and_whole_trajectory_holdout_are_not_slot_replicates():
    d=regression_fixture();old=deepcopy(d);r=fit_dataset(d)
    assert d==old and r['model']['rank']==1 and r['coverage']['feature_dimension']==3
    assert r['validation']['held_out_rows']==12 and r['coverage']['unique_trajectories']==6
    assert r['validation']['mae_ms']<1e-8 and r['validation']['baseline_mae_ms']>0
    assert r['validation']['direction_accuracy']==1 and not r['causal_effect_identified']
    result=estimate_change(r,state(1),state(2),horizon_ms=250)
    assert result['delta']==pytest.approx(3)and result['status']=='conditional'
    assert result['uncertainty']=='held_out_error_envelope_not_a_confidence_interval'


def test_correlated_parameters_do_not_gain_fictitious_independent_identification():
    r=fit_dataset(regression_fixture(two_weights=True));base=state(1,True);target=state(2,True)
    assert estimate_change(r,base,target,horizon_ms=250)['delta']==pytest.approx(3)
    target['ue_scheduler_weights']['v']=1
    assert estimate_change(r,base,target,horizon_ms=250)['issues']==['empirical_unidentified_direction']


@pytest.mark.parametrize('kind',['range','window','context','category'])
def test_extrapolation_wrong_window_and_non_native_controls_are_refused(kind):
    r=fit_dataset(regression_fixture());base=state();target=state(1.5);horizon=250
    if kind=='range':target=state(4)
    if kind=='window':horizon=500
    if kind=='context':target['ue_offload_fraction']['u']=1
    if kind=='category':target['scheduler_mode']['c']='qos_deadline'
    result=estimate_change(r,base,target,horizon_ms=horizon)
    assert result['delta']is None and result['bounds']is None


def test_inconsistent_duplicates_and_mixed_environments_are_not_pooled():
    d=regression_fixture();d['duplicate_response_discrepancies']=[{}]
    with pytest.raises(ValueError):fit_dataset(d)
    d=regression_fixture();d['rows'][0]['context_sha256']='different'
    with pytest.raises(ValueError):fit_dataset(d)


def tiny_run(folder,*,offload_changed=False):
    run=folder/'data/run';run.mkdir(parents=True)
    drx=dict(enabled=True,cycle_ms=40,on_duration_ms=4,inactivity_ms=8,wakeup_ms=.5,offset_ms=0)
    cfg=dict(source_sha256='fixture',manifest=dict(slot_duration_ms=1,ues=[dict(ue_id='u')],cells=[dict(cell_id='c')]),configs=dict(
        phy=dict(rt_sys=dict(model='fast',fast_profile='system',ue_services=dict(drx=drx,drx_by_ue={},mec=dict(tasks_by_ue={
            'u':dict(offload_fraction=.5,interval_ms=3,compute_gflop=.01,deadline_ms=2)})))),
        cell=dict(cell_radio_profiles={'c':{}}),ue=dict(ues=[dict(ue_id='u',serving_cell_id='placeholder')])) )
    (run/'config_snapshot.json').write_text(json.dumps(cfg));(run/'run_status.json').write_text(json.dumps(dict(completed=True,last_completed_slot=30)))
    (folder/'job.json').write_text(json.dumps(dict(id=folder.name,phase='complete',app_ids=['fixture_only'],seed=1,comparison=dict(application_slot=11))))
    old=dict(scheduler_mode='round_robin',ue_scheduler_weights={'u':1})
    new=dict(scheduler_mode='latency_aware',ue_scheduler_weights={'u':2})
    receipts=[dict(status='applied',apply_slot=1,before={'c':old},after={'c':old}),
        dict(status='rejected',apply_slot=11,before={'c':old},after={'c':dict(scheduler_mode='qos_deadline',ue_scheduler_weights={'u':4})}),
        dict(status='applied',apply_slot=11,before={'c':old},after={'c':new})]
    with gzip.open(run/'control_receipts.jsonl.gz','wt')as f:
        for r in receipts:f.write(json.dumps(r)+'\n')
    generated=completed=0;events={}
    for ident,slot in enumerate((11,14,17)):
        birth=dict(kind='task_generated',ue_id='u',task_id=ident,created_s=slot/1000,time_s=slot/1000,deadline_s=.002,offloaded=True)
        events.setdefault(slot,[]).append(birth);events.setdefault(slot+1,[]).append(dict(birth,kind='task_completed',time_s=(slot+1)/1000,latency_ms=1.))
    with gzip.open(run/'rt_sys_trace.jsonl.gz','wt')as f:
        for slot in range(1,31):
            es=events.get(slot,[]);generated+=sum(e['kind']=='task_generated'for e in es);completed+=sum(e['kind']=='task_completed'for e in es)
            counts=dict(generated=generated,offloaded_generated=generated,completed=completed,offloaded_completed=completed,failed=0,offloaded_failed=0)
            row=dict(slot_id=slot,ues=[dict(ue_id='u',cell_id='c',queue_bytes=0)],ue_services={'u':dict(ue_task_metrics=counts,
                ue_service_policy=dict(drx=drx,offload_fraction=.7 if offload_changed and slot>=20 else .5))},
                ap_services=dict(cells={'c':dict(ap_policy='active')}),ue_service_events=es)
            f.write(json.dumps(row)+'\n')
    return folder/'job.json'


def test_extraction_reads_actual_receipts_and_mature_events_without_editing_sources(tmp_path):
    job=tiny_run(tmp_path/'case');before={str(p):sha(p)for p in job.parent.rglob('*')if p.is_file()}
    r=extract_job(job,window_slots=20)
    assert r['status']=='usable'and r['y']==1 and r['metric']['sample_count']==3
    assert r['x']==[1.,0.,2.] # rejected QoS proposal does not enter features
    assert r['context']['assignments']=={'u':'c'} # observed association, not placeholder
    assert before=={str(p):sha(p)for p in job.parent.rglob('*')if p.is_file()}


def test_non_scheduler_policy_changes_are_excluded(tmp_path):
    r=extract_job(tiny_run(tmp_path/'case',offload_changed=True),window_slots=20)
    assert r['status']=='excluded'and r['reason']=='non_native_offload'


def test_fixed_stimuli_use_declared_action_fields_and_valid_memberships(tmp_path,monkeypatch):
    from types import SimpleNamespace as N
    import sys
    import experiments.mec_response_stimulus as driver
    import experiments.fixed_control_timeline as adapter
    original=adapter.FixedControlTimeline
    monkeypatch.setattr(driver.runpy,'run_path',lambda *a,**k:None)
    monkeypatch.setattr(sys,'argv',['fixture','--owned-root',str(Path(__file__).resolve().parents[1])])
    try:
        driver.main();cls=adapter.FixedControlTimeline
        weights=dict(ue_0=2,ue_1=1,ue_2=1,ue_4=1,ue_6=1,ue_8=1)
        spec=dict(schema='oran.mec-response-stimulus.v1',name='fixture',switch_slot=560,
            scheduler_mode={c:'latency_aware'for c in('cell_0','cell_1','cell_2')},ue_scheduler_weights=weights)
        p=tmp_path/'spec.json';p.write_text(json.dumps(spec));app=cls(p)
        obs=N(slot_id=560,paper_interfaces_enabled=True,cell_observations=[N(cell_id=c,executed_config={})for c in spec['scheduler_mode']],
            ue_observations=[N(ue_id=u,serving_cell_id='cell_0'if u in('ue_0','ue_6','ue_8')else'cell_1'if u=='ue_1'else'cell_2')for u in weights])
        result=app.run(obs,None)
        assert len(result.config_actions)==3 and 'latency_bias'in app.contract.actions
    finally:adapter.FixedControlTimeline=original


def test_joint_regression_fallback_is_provenanced_and_shared_writes_are_not_added():
    from test_smo_conflict_quantification import fixture
    from smo.conflict_quantification import quantify_catalog
    g,f,c,p,a=fixture();f['states']['rcp:scheduler_mode']['by_entity']['c']='latency_aware'
    f['states']['kpi:mec_offloaded_latency_ms']=f['states'].pop('kpi:k')
    for e in g['edges']:
        if e['target']=='kpi:k':e['target']='kpi:mec_offloaded_latency_ms'
    c['required_kpis']=['mec_offloaded_latency_ms'];c['ranges'][0]['kpi']='mec_offloaded_latency_ms'
    for app,w in zip(p['apps'],(2.,1.)):
        app['actions']=[dict(parameter='scheduler_mode',cell_id='c',ue_id=None,current='latency_aware',proposed='latency_aware'),a('ue_scheduler_weights',w)]
    artifact=fit_dataset(regression_fixture());artifact['context']['native_drx']=deepcopy(f['states']['rcp:ue_drx_configs']['by_entity'])
    config=dict(source_sha256='fixture',configs=dict(phy=dict(rt_sys=dict(ue_services={})),cell=dict(cell_radio_profiles={}),ue=dict(ues=[])))
    r=quantify_catalog(g,f,c,p,config=config,horizon_ms=250,empirical_model=artifact)
    q=r['groups'][0]
    assert len(q['direct'])==1 and q['scenario_count']==2
    assert q['kpis'][0]['predicted_min']==pytest.approx(10)and q['kpis'][0]['predicted_max']==pytest.approx(13)
    assert q['kpis'][0]['violation_probability']==.5
    assert q['kpis'][0]['evidence']=='observational_executed_control_regression'
    assert len(r['joint_effects'])==1 and r['joint_effects'][0]['delta']==pytest.approx(3)
    assert any(l['delta']is None for l in r['links']) # no invented individual gain

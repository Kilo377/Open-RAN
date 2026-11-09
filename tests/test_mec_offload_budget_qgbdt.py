"""Delivered telemetry, real-policy confirmation, and measured-utility tests."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
import pytest

from ric.near_rt.pools import pool_catalog, build_pool_xapp
import ric.near_rt.pools.chongyu.mec_offload_budget_qgbdt as budget
from ric.near_rt.pools.chongyu.mec_offload_budget_qgbdt import MecOffloadBudgetQGBDT, FEATURES
from ric.near_rt.pools.chongyu.online_qgbdt import OnlineQGBDT, parameters

PROFILE=dict(interval_ms=50,compute_gflop=.05,deadline_ms=100)


def app(p=None):
    result=MecOffloadBudgetQGBDT(['cell_0','cell_1'],parameters={'task_profile':PROFILE,**(p or {})})
    result.select(['cell_0','cell_1'],{'cell_0':1,'cell_1':1});return result


def cohort(p=.5,**values):
    row=dict(index=2,generated=2,completed=1,on_time_completed=1,failed=0,pending=1,
        completion_latency_ms_sum=25.,mature_after_s=.4,offload_fraction=p,mixed_policy=False)
    row.update(values);return row


def obs(slot=1,p=.5,owner=None,cohorts=(),uid='ue_4',cell='cell_0'):
    u=NS(ue_id=uid,serving_cell_id=cell,queue_bytes=1000,hol_delay_ms=2.,reference_sinr_db=10.,
        ue_task_metrics={'offloaded_pending':0},ue_task_cohorts=deepcopy(list(cohorts)),
        ue_service_policy={'offload_fraction':p,'drx':{'enabled':True}},
        kpi_metadata={'ue_energy_total_j':{'produced_at_slot':slot},
                      'ue_task_metrics':{'produced_at_slot':slot,'valid':True}})
    c=NS(cell_id=cell,executed_config={'ue_offload_fraction':{uid:p}},
         kpi_metadata={'executed_config':{'produced_at_slot':slot,'valid':True}})
    return NS(slot_id=slot,ue_observations=[u],cell_observations=[c],control_ownership={cell:{
        'ue_offload_fraction':None,'ue_offload_fraction_by_ue':{uid:owner,'ue_0':'x01'},'ue_drx_configs':'x01'}})


def confirm(controller,p=.5):
    controller.run(obs(201,p,controller.name),NS())
    assert controller.pending['confirmed'] and controller.pending['cohort']==2


def artifact(tmp_path,monkeypatch):
    """Explicit inference fixture only, not a deployment training artifact."""
    monkeypatch.setattr(budget,'ROOT',tmp_path)
    model=OnlineQGBDT(FEATURES)
    for i in range(12):model.add([20.,0.,1.,2.,10.,0.,25.,float(i%2)/2],.9)
    path=tmp_path/'outputs'/'fixture.json';model.save(path);p=json.loads(path.read_text())
    p.update(trained_samples=12,fit_count=1,fit_parameters=parameters(137),initial_prediction=.9,
        forest=[dict(children_left=[-1],children_right=[-1],feature=[-2],threshold=[-2.],value=[0.]) for _ in range(32)])
    path.write_text(json.dumps(p));return path


def test_catalog_factory_and_exact_uid_control():
    spec=pool_catalog('chongyu')['mec_offload_budget_qgbdt']
    controller=build_pool_xapp(dict(implementation_id='mec_offload_budget_qgbdt',cell_ids=['cell_0'],parameters={'task_profile':PROFILE}),'chongyu')
    controller.select(['cell_0'],{'cell_0':1})
    assert isinstance(controller,MecOffloadBudgetQGBDT)
    assert spec['category_id']==controller.category_id=='mec_offload_budget'
    result=controller.run(obs(),NS());action=result.config_actions[0]
    assert action.ue_offload_fraction=={'ue_4':.5} and action.ue_drx_configs is None
    assert action.scheduler_mode is None and action._library_provenance==controller.identity('cell_0')
    assert controller.feasible()==(0.,.5) and controller.status=='rules_only'


def test_no_profile_or_other_ues_cannot_control_or_fabricate_data():
    controller=MecOffloadBudgetQGBDT(['cell_0']);controller.select(['cell_0'],{'cell_0':1})
    assert not controller.run(obs(),NS()).config_actions
    assert controller.status=='not_evaluable'
    for uid in ('ue_0','ue_2','ue_6'):
        controller=app();assert not controller.run(obs(uid=uid),NS()).config_actions
    assert controller.learner.sample_count==0


def test_uid_owner_and_execution_required_but_drx_ownership_is_not():
    controller=app();controller.run(obs(),NS())
    controller.run(obs(201,owner='x01'),NS())
    assert not controller.pending['confirmed'] and controller.learner.sample_count==0
    report=obs(202,owner=controller.name)
    report.cell_observations[0].executed_config['ue_offload_fraction']['ue_4']=1.
    controller.run(report,NS());assert not controller.pending['confirmed']
    controller.run(obs(203,owner=controller.name),NS())
    assert controller.pending['confirmed']


def test_future_mature_all_task_utility_retains_failures_and_pending():
    controller=app();controller.run(obs(),NS());confirm(controller)
    controller.run(obs(799,owner=controller.name,cohorts=[cohort()]),NS())
    assert controller.learner.sample_count==0
    controller.run(obs(800,owner=controller.name,cohorts=[cohort(mixed_policy=True)]),NS())
    assert controller.learner.sample_count==1
    sample=next(e for e in controller.events if e['kind']=='measured_sample')
    assert sample['deadline_penalized_mean_ms']==62.5 and sample['utility']==.34375
    assert sample['features'][5]==0. and sample['features'][6]==100. # Captured before future outcome.
    assert controller.action_samples()=={'0':0,'0.5':1}
    assert controller.pending['p']==0. # Actual measured counts drive balanced exploration.


def test_zero_completion_is_not_reported_as_zero_latency_success():
    controller=app();controller.run(obs(),NS());confirm(controller)
    controller.run(obs(800,owner=controller.name,cohorts=[cohort(completed=0,on_time_completed=0,
        failed=1,pending=1,completion_latency_ms_sum=0.)]),NS())
    sample=next(e for e in controller.events if e['kind']=='measured_sample')
    assert sample['utility']==-.25 and sample['deadline_penalized_mean_ms']==100.


@pytest.mark.parametrize('change',['owner','executed','cohort','scope','handover','future','stale','spliced','profile'])
def test_invalid_context_never_adds_a_label(change):
    controller=app();controller.run(obs(),NS());confirm(controller)
    report=obs(800,owner=controller.name,cohorts=[cohort()])
    if change=='owner':report.control_ownership['cell_0']['ue_offload_fraction_by_ue']['ue_4']='x01'
    elif change=='executed':report.ue_observations[0].ue_service_policy['offload_fraction']=1.
    elif change=='cohort':report.ue_observations[0].ue_task_cohorts[0]['offload_fraction']=None
    elif change=='scope':controller.select([],{})
    elif change=='handover':report.ue_observations[0].serving_cell_id='cell_1';report.cell_observations[0].cell_id='cell_1'
    elif change=='future':report.cell_observations[0].kpi_metadata['executed_config']['produced_at_slot']=801
    elif change=='stale':report.slot_id=1100
    elif change=='spliced':report.ue_observations[0].kpi_metadata['ue_energy_total_j']['produced_at_slot']=799
    else:report.ue_observations[0].ue_task_cohorts[0]['generated']=3
    controller.run(report,NS());assert controller.learner.sample_count==0
    assert not any(e['kind']=='measured_sample' for e in controller.events)


def test_unconfirmed_timeout_retries_without_training():
    controller=app();controller.run(obs(),NS())
    controller.run(obs(402,owner='x01'),NS())
    assert controller.invalidated==1 and controller.pending['requested_slot']==402
    assert controller.learner.sample_count==0


def test_untrained_rows_save_and_resume_counts_before_minimum_fit(tmp_path,monkeypatch):
    monkeypatch.setattr(budget,'ROOT',tmp_path);path=tmp_path/'outputs'/'resume.json'
    controller=app({'model_path':str(path)})
    controller.run(obs(),NS());confirm(controller)
    controller.run(obs(800,owner=controller.name,cohorts=[cohort()]),NS())
    assert path.is_file()
    resumed=app({'model_path':str(path)})
    assert resumed.action_samples()=={'0':0,'0.5':1} and not resumed.learner.trained
    assert resumed.run(obs(),NS()).config_actions[0].ue_offload_fraction=={'ue_4':0.}


def test_fit_requires_six_measured_rows_for_each_feasible_action(monkeypatch):
    controller=app();fits=[]
    for i in range(10):controller.learner.add([20.,0.,1.,2.,10.,0.,25.,float(i%2)/2],.9)
    monkeypatch.setattr(controller.learner,'fit',lambda:fits.append(controller.learner.sample_count) or False)
    controller.run(obs(),NS());confirm(controller)
    controller.run(obs(800,owner=controller.name,cohorts=[cohort()]),NS())
    assert controller.learner.sample_count==11 and not fits
    controller.run(obs(801,p=0.,owner=controller.name),NS())
    assert controller.pending['cohort']==5
    controller.run(obs(1400,p=0.,owner=controller.name,cohorts=[cohort(0.,index=5,mature_after_s=.7)]),NS())
    assert controller.learner.sample_count==12 and fits==[12]


def test_frozen_inference_filters_budget_and_never_adds_fits_or_rewrites(tmp_path,monkeypatch):
    path=artifact(tmp_path,monkeypatch);original=path.read_bytes()
    controller=app({'mode':'frozen','model_path':str(path)})
    def forbidden(*args,**kwargs):raise AssertionError('Frozen learner mutated')
    monkeypatch.setattr(controller.learner,'add',forbidden);monkeypatch.setattr(controller.learner,'fit',forbidden)
    result=controller.run(obs(),NS())
    assert result.config_actions[0].ue_offload_fraction=={'ue_4':.5} # Constant quantile tie -> highest feasible.
    assert controller.status=='qgbdt' and controller.learner.prediction_batches==1
    confirm(controller)
    controller.run(obs(800,owner=controller.name,cohorts=[cohort()]),NS())
    assert controller.learner.sample_count==controller.learner.trained_samples==12
    assert controller.learner.fit_count==1 and path.read_bytes()==original


def test_rules_ablation_does_not_import_or_call_learner():
    controller=app({'mode':'rules'})
    controller.learner.predict=lambda *a:(_ for _ in ()).throw(AssertionError('Rules called ML'))
    assert controller.run(obs(),NS()).config_actions[0].ue_offload_fraction=={'ue_4':.5}
    confirm(controller)
    controller.run(obs(800,owner=controller.name,cohorts=[cohort()]),NS())
    assert controller.learner.sample_count==0 and controller.status=='rules_only'


@pytest.mark.parametrize('p',[{'mode':'bad'},{'window_ms':50},{'budget_gflops':True},
    {'budget_gflops':float('nan')},{'seed':True},{'unknown':1},
    {'task_profile':dict(interval_ms=100,compute_gflop=.05,deadline_ms=100)}])
def test_invalid_configuration_rejected(p):
    with pytest.raises(ValueError):app(p)


def test_real_app_refits_after_four_new_measured_rows_and_preserves_actual_params(tmp_path,monkeypatch):
    pytest.importorskip('sklearn')
    path=artifact(tmp_path,monkeypatch)
    controller=app({'mode':'online','model_path':str(path)})
    controller.run(obs(),NS())
    for count in range(4):
        confirmed=201+600*count
        controller.run(obs(confirmed,owner=controller.name),NS())
        index=2+3*count
        controller.run(obs(800+600*count,owner=controller.name,
            cohorts=[cohort(index=index,mature_after_s=(index+2)*.1)]),NS())
        assert controller.learner.sample_count==13+count
        assert controller.learner.fit_count==(1 if count<3 else 2)
    saved=json.loads(path.read_text())
    assert saved['trained_samples']==16 and saved['fit_parameters']==parameters(137)
    assert len(saved['forest'])==32 and len(saved['x'])==16


def test_frozen_rejects_missing_untrained_or_uncovered_artifact(tmp_path,monkeypatch):
    monkeypatch.setattr(budget,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='requires a trained model'):app({'mode':'frozen'})
    path=tmp_path/'outputs'/'empty.json';OnlineQGBDT(FEATURES).save(path)
    with pytest.raises(ValueError,match='requires 12'):app({'mode':'frozen','model_path':str(path)})
    path=artifact(tmp_path,monkeypatch);payload=json.loads(path.read_text())
    for row in payload['x']:row[-1]=0.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='six per feasible action'):app({'mode':'frozen','model_path':str(path)})


@pytest.mark.parametrize('field,value',[('completion_latency_ms_sum',float('nan')),
    ('completed',3),('mature_after_s',float('inf'))])
def test_invalid_delivered_cohort_reports_not_evaluable(field,value):
    controller=app();controller.run(obs(),NS());confirm(controller)
    row=cohort();row[field]=value
    assert not controller.run(obs(800,owner=controller.name,cohorts=[row]),NS()).config_actions
    assert controller.status=='not_evaluable' and controller.learner.sample_count==0


@pytest.mark.parametrize('row',[cohort(generated=3,completed=2,on_time_completed=2,pending=1,
    completion_latency_ms_sum=50.),cohort(mature_after_s=.3)])
def test_well_formed_profile_or_horizon_mismatch_fails_closed_without_new_action(row):
    controller=app();controller.run(obs(),NS());confirm(controller)
    result=controller.run(obs(800,owner=controller.name,cohorts=[row]),NS())
    assert not result.config_actions and controller.pending is None
    assert controller.status=='not_evaluable' and controller.learner.sample_count==0
    assert any(e['kind']=='not_evaluable' for e in controller.events)


def test_confirmation_consumes_actual_public_executed_config_schema():
    from ric.near_rt.rule_based_v1.telemetry import executed_configs
    policy={'offload_fraction':.5,'drx':{'enabled':True}}
    state=NS(managed_cells={'cell_0':NS(cell=NS(),resource_grid=NS(n_prbs_total=24),
        scheduler=NS(scheduler_mode='round_robin',scheduler_policy_hint='balanced',latency_bias=None,slice_weight_adjustment={}))},
        ues=[],ue_services=NS(policies=lambda cid:{'ue_4':deepcopy(policy)},profiles={'ue_4':{}}))
    public=executed_configs(state)['cell_0']
    assert public['ue_offload_fraction']=={'ue_4':.5}
    assert 'ue_service_policies' not in public # Receipt-after has a different shape.
    controller=app();controller.run(obs(),NS())
    report=obs(201,owner=controller.name);report.cell_observations[0].executed_config=public
    controller.run(report,NS())
    assert controller.pending['confirmed']

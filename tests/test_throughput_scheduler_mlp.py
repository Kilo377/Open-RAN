"""Whole-network ACK windows and actual goodput/packet-counter labels."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
import pytest

from ric.near_rt.pools import build_pool_xapp,pool_catalog
import ric.near_rt.pools.chongyu.throughput_scheduler_mlp as throughput
from ric.near_rt.pools.chongyu.throughput_scheduler_mlp import ThroughputSchedulerMLP,USERS,URLLC,FEATURES,TARGETS,PACKET_COUNTERS,action_map
from ric.near_rt.pools.chongyu.online_throughput_mlp import OnlineThroughputMLP,parameters

CELLS=['cell_0','cell_1','cell_2']


def app(p=None):
    parameters={'policy':'guarded',**(p or {})} # Explicit protected-policy fixture.
    c=ThroughputSchedulerMLP(CELLS,parameters=parameters);c.select(CELLS,{cid:1 for cid in CELLS});return c


def obs(slot=1,snapshot=None,owner=None,bits=100000,packet_delta=None):
    users=[]
    for i,uid in enumerate(USERS):
        cid=CELLS[i//2]
        counters=dict(zip(PACKET_COUNTERS,[100,0,0,90,0]))
        if packet_delta and uid in packet_delta:
            for key,value in packet_delta[uid].items():counters[key]+=value
        meta={key:{'produced_at_slot':slot,'valid':True} for key in ('delivered_bits_total','offered_bits_total')+PACKET_COUNTERS}
        users.append(NS(ue_id=uid,serving_cell_id=cid,queue_bytes=1000,hol_delay_ms=2.,reference_sinr_db=10.+i,
            throughput_mbps=1.,offered_rate_bps=1e6,delivered_bits_total=bits,offered_bits_total=bits,
            slice_id='urllc' if uid in URLLC else 'embb',latency_budget_ms=10. if uid in URLLC else 100.,
            kpi_metadata=meta,**counters))
    snapshot=snapshot or action_map(users,0)
    cells=[NS(cell_id=cid,executed_config={**deepcopy(snapshot[cid]),'latency_bias':None},
        kpi_metadata={'executed_config':{'produced_at_slot':slot,'valid':True}}) for cid in CELLS]
    return NS(slot_id=slot,ue_observations=users,cell_observations=cells,
        control_ownership={cid:dict(scheduler_mode=owner,ue_scheduler_weights=owner) for cid in CELLS})


def start(c,index=None):
    if index is not None:
        for i in range(4):
            for _ in range(1 if i!=index else 0):
                x=[0.]*len(FEATURES);x[-4+i]=1.;c.learner.add(x,[6.]+[1.]*6+[0.,0.])
    decision=c.run(obs(),NS());snapshot=deepcopy(c.pending['snapshot'])
    c.run(obs(10,snapshot,owner=c.name),NS())
    assert c.pending['confirmed']
    c.run(obs(50,snapshot,owner=c.name),NS())
    assert c.pending['start_slot']==50
    return snapshot,decision


def artifact(tmp_path,monkeypatch):
    """Actual corrected sklearn fit on synthetic unit rows, never deployment data."""
    pytest.importorskip('sklearn')
    monkeypatch.setattr(throughput,'ROOT',tmp_path)
    m=OnlineThroughputMLP(FEATURES,TARGETS)
    for i in range(24):
        x=[0.]*len(FEATURES);x[-4+i%4]=1.;m.add(x,[6.]+[1.]*6+[0.,0.])
    assert m.fit() and m.optimizer_verified
    path=tmp_path/'outputs/model.json';m.save(path)
    return path


def test_catalog_fixed_mapper_and_real_control_identity():
    spec=pool_catalog('chongyu')['throughput_scheduler_mlp']
    c=build_pool_xapp(dict(implementation_id='throughput_scheduler_mlp',cell_ids=CELLS,parameters={'mode':'fixed','fixed_action':3}),'chongyu')
    c.select(CELLS,{cid:1 for cid in CELLS});d=c.run(obs(),NS())
    assert isinstance(c,ThroughputSchedulerMLP) and c.category_id==spec['category_id']=='throughput_scheduler'
    assert len(d.config_actions)==3 and c.status=='fixed'
    weights={uid:w for action in d.config_actions for uid,w in action.ue_scheduler_weights.items()}
    assert set(weights)==set(USERS) and all(weights[uid]>=1 for uid in URLLC)
    for a in d.config_actions:
        assert a.scheduler_mode=='latency_aware' and a._library_provenance==c.identity(a.target_cell_id)
        assert a.ue_offload_fraction is a.ue_drx_configs is None
    assert weights['ue_4']==2. and weights['ue_8']==2. # Prefer eMBB over higher-priority URLLC peers.


def test_one_global_row_uses_all_six_deltas_after_ack_and_wash_in():
    c=app();snapshot,_=start(c)
    c.run(obs(249,snapshot,c.name,bits=200000),NS());assert c.learner.sample_count==0
    packet={uid:{'admitted_packets_total':10,'deadline_missed_packets_total':1,
        'overflow_packets_total':2,'completed_packets_total':9,'late_completed_packets_total':1} for uid in URLLC}
    c.run(obs(250,snapshot,c.name,bits=200000,packet_delta=packet),NS())
    assert c.learner.sample_count==1 # Not three correlated per-cell rows.
    sample=next(v for v in c.events if v['kind']=='measured_sample')
    assert sample['targets']==pytest.approx([6.]+[1.]*6+[40.,40.])
    assert sample['duration_s']==.1 and sample['produced_start_slot']==50 and sample['produced_end_slot']==250
    assert set(sample['delivered_bits_by_ue'])==set(USERS)
    assert sample['urllc_original_latency_budgets_ms']=={'ue_2':10.,'ue_6':10.}
    assert sample['features'][-4:]==[1.,0.,0.,0.]
    assert c.pending['action']==1


def test_no_label_before_every_cell_and_uid_policy_confirmed():
    c=app();c.run(obs(),NS());snapshot=deepcopy(c.pending['snapshot'])
    report=obs(10,snapshot,c.name);report.control_ownership['cell_2']['ue_scheduler_weights']='other'
    c.run(report,NS());assert not c.pending['confirmed']
    report=obs(20,snapshot,c.name);report.cell_observations[1].executed_config['ue_scheduler_weights']['ue_4']=2.
    c.run(report,NS());assert not c.pending['confirmed']
    c.run(obs(30,snapshot,c.name),NS());assert c.pending['confirmed_slot']==30


@pytest.mark.parametrize('change',['owner','mode','weight','scope','membership','reset','spliced','deadline','profile','missing_ue'])
def test_interrupted_or_invalid_context_is_not_training_data(change):
    c=app();snapshot,_=start(c)
    r=obs(250,snapshot,c.name,bits=200000)
    if change=='owner':r.control_ownership['cell_1']['scheduler_mode']='x12'
    elif change=='mode':r.cell_observations[1].executed_config['scheduler_mode']='latency_aware'
    elif change=='weight':r.cell_observations[0].executed_config['ue_scheduler_weights']['ue_0']=2.
    elif change=='scope':c.select(['cell_0'],{'cell_0':1})
    elif change=='membership':r.ue_observations[0].serving_cell_id='cell_1'
    elif change=='reset':r.ue_observations[0].delivered_bits_total=0
    elif change=='spliced':r.ue_observations[0].kpi_metadata['delivered_bits_total']['produced_at_slot']=249
    elif change=='deadline':r.ue_observations[2].kpi_metadata['admitted_packets_total']['produced_at_slot']=249
    elif change=='profile':r.ue_observations[2].latency_budget_ms=100.
    else:r.ue_observations.pop()
    c.run(r,NS());assert c.learner.sample_count==0
    assert not any(v['kind']=='measured_sample' for v in c.events)


def test_bad_actual_urlcl_outcome_is_retained_not_filtered(tmp_path,monkeypatch):
    path=artifact(tmp_path,monkeypatch);c=app({'model_path':str(path)})
    snapshot,_=start(c)
    packet={'ue_2':{'deadline_missed_packets_total':3}}
    c.run(obs(250,snapshot,c.name,bits=200000,packet_delta=packet),NS())
    assert c.learner.sample_count==25 and c.learner._y[-1][7]==30.
    assert c.pending['action']==0 and c.last_guard_bad
    assert c.last_decisions[0]['reason']=='observed_urllc_guard_abstention'


def test_snapshot_is_held_when_public_quality_backlog_changes():
    c=app({'mode':'fixed','fixed_action':3});snapshot,_=start(c)
    r=obs(250,snapshot,c.name,bits=200000)
    for u in r.ue_observations:u.reference_sinr_db=-30.;u.queue_bytes=0
    c.run(r,NS())
    sample=next(v for v in c.events if v['kind']=='measured_sample')
    assert sample['snapshot']==snapshot and sample['features'][3]==10.
    assert c.last_decisions[0]['neutralized_requested_arm'] is True


def test_untrained_rows_resume_balanced_counts(tmp_path,monkeypatch):
    monkeypatch.setattr(throughput,'ROOT',tmp_path);p=tmp_path/'outputs/untrained.json'
    c=app({'model_path':str(p)});snapshot,_=start(c)
    c.run(obs(250,snapshot,c.name,bits=200000),NS())
    resumed=app({'model_path':str(p)})
    assert resumed.action_samples()=={'0':1,'1':0,'2':0,'3':0}
    assert resumed.run(obs(),NS()).config_actions[0].scheduler_mode=='latency_aware'
    assert resumed.pending['action']==1 and resumed.learner.fit_count==0


@pytest.mark.parametrize('policy',['guarded','throughput'])
def test_frozen_native_abstention_still_calls_actual_network_and_never_mutates(tmp_path,monkeypatch,policy):
    path=artifact(tmp_path,monkeypatch);original=path.read_bytes();c=app({'mode':'frozen','model_path':str(path),'policy':policy})
    def forbidden(*a,**k):raise AssertionError('Frozen model learned')
    monkeypatch.setattr(c.learner,'add',forbidden);monkeypatch.setattr(c.learner,'fit',forbidden)
    snapshot,_=start(c);assert c.pending['action']==0 and c.learner.prediction_batches==1
    c.run(obs(250,snapshot,c.name,bits=200000),NS())
    assert c.learner.sample_count==c.learner.trained_samples==24 and c.learner.fit_count==1
    assert c.learner.prediction_batches==2 and path.read_bytes()==original


def test_guard_rejects_high_network_prediction_that_harms_one_ue(tmp_path,monkeypatch):
    path=artifact(tmp_path,monkeypatch);c=app({'mode':'frozen','model_path':str(path)})
    rows=[[6.]+[1.]*6+[0.,0.],[8.,.1]+[1.5]*5+[0.,0.],[7.2]+[1.2]*6+[0.,0.],[9.]+[1.5]*6+[1.,0.]]
    monkeypatch.setattr(c.learner,'predict',lambda x:rows)
    c.run(obs(),NS());assert c.pending['action']==2
    assert c.status=='mlp' and all(w>=1 for m in c.pending['snapshot'].values() for w in m['ue_scheduler_weights'].values())


def test_frozen_requires_real_prefix_coverage_and_encoding(tmp_path,monkeypatch):
    monkeypatch.setattr(throughput,'ROOT',tmp_path)
    with pytest.raises(ValueError):app({'mode':'frozen'})
    path=artifact(tmp_path,monkeypatch);p=json.loads(path.read_text())
    p['x'][0][-4:]=[1.,1.,0.,0.];path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='encoding'):app({'mode':'frozen','model_path':str(path)})


@pytest.mark.parametrize('p',[{'mode':'bad'},{'window_ms':1},{'fixed_action':True},{'seed':False},{'unknown':1}])
def test_invalid_configuration_rejected(p):
    with pytest.raises(ValueError):app(p)


def test_online_guard_uses_observed_native_windows_even_before_first_fit():
    c=app();x=[0.]*len(FEATURES);x[-4]=1.
    c.learner.add(x,[6.]+[1.]*6+[0.,0.])
    assert not c.learner.trained and c._reference()==[6.]+[1.]*6+[0.,0.]
    snapshot,_=start(c)
    packet={'ue_6':{'overflow_packets_total':2}}
    c.run(obs(250,snapshot,c.name,bits=200000,packet_delta=packet),NS())
    assert c.learner.sample_count==2 and c.learner._y[-1][8]==20.
    assert c.last_guard_bad and c.pending['action']==0


def test_tiny_positive_mlp_residual_is_numerically_tolerated_and_logged(tmp_path,monkeypatch):
    path=artifact(tmp_path,monkeypatch);c=app({'mode':'frozen','model_path':str(path)})
    tolerance=c._guard_rate_tolerance()
    rows=[[6.]+[1.]*6+[0.,0.],[6.1]+[1.02]*6+[0.,0.],
          [7.2]+[1.2]*6+[tolerance*.5,tolerance*.5],
          [9.]+[1.5]*6+[1./.1,0.]]
    monkeypatch.setattr(c.learner,'predict',lambda x:rows)
    c.run(obs(),NS())
    assert c.pending['action']==2 # Small residual does not turn a healthy forecast into forced RR.
    audit=c.last_decisions[0]['prediction_guard_audit']
    assert audit['raw_predictions'][2][7]>0 and '2' not in audit['rejected_actions']
    assert '3' in audit['rejected_actions'] # A full event per window is not a numerical residual.
    assert audit['numeric_tolerance_events_per_window']==.001
    assert c.summary()['guard_numeric_tolerance_event_rate_per_s']==pytest.approx(.01)
    assert app({'window_ms':200}).summary()['guard_numeric_tolerance_event_rate_per_s']==pytest.approx(.005)


def empty_cell_obs(slot=1,snapshot=None,owner=None,bits=100000):
    r=obs(slot,owner=owner,bits=bits)
    association={'ue_0':'cell_0','ue_6':'cell_0','ue_8':'cell_0',
                 'ue_1':'cell_2','ue_2':'cell_2','ue_4':'cell_2'}
    for u in r.ue_observations:u.serving_cell_id=association[u.ue_id]
    snapshot=deepcopy(snapshot) if snapshot else action_map(r.ue_observations,0)
    snapshot.setdefault('cell_1',dict(scheduler_mode='round_robin',ue_scheduler_weights={}))
    for c in r.cell_observations:c.executed_config={**deepcopy(snapshot[c.cell_id]),'latency_bias':None}
    r.control_ownership['cell_1']['ue_scheduler_weights']=None # Real RuleLeases drops empty UID maps.
    return r


def test_empty_cell_needs_mode_owner_but_no_fictitious_uid_weight_owner():
    c=app();c.run(empty_cell_obs(),NS());snapshot=deepcopy(c.pending['snapshot'])
    assert snapshot['cell_1']['ue_scheduler_weights']=={}
    c.run(empty_cell_obs(10,snapshot,c.name),NS());assert c.pending['confirmed']
    c.run(empty_cell_obs(50,snapshot,c.name),NS())
    c.run(empty_cell_obs(250,snapshot,c.name,bits=200000),NS())
    assert c.learner.sample_count==1
    sample=next(v for v in c.events if v['kind']=='measured_sample')
    assert sample['targets'][0]==pytest.approx(6.) and len(sample['delivered_bits_by_ue'])==6


def test_occupied_cell_missing_weight_owner_still_cannot_confirm_or_label():
    c=app();c.run(empty_cell_obs(),NS());snapshot=deepcopy(c.pending['snapshot'])
    r=empty_cell_obs(10,snapshot,c.name);r.control_ownership['cell_0']['ue_scheduler_weights']=None
    c.run(r,NS());assert not c.pending['confirmed']
    c.run(empty_cell_obs(20,snapshot,c.name),NS());assert c.pending['confirmed']
    c.run(empty_cell_obs(60,snapshot,c.name),NS())
    r=empty_cell_obs(260,snapshot,c.name,bits=200000);r.control_ownership['cell_2']['ue_scheduler_weights']=None
    c.run(r,NS())
    assert c.invalidated==1 and c.learner.sample_count==0
    assert not any(v['kind']=='measured_sample' for v in c.events)


def test_legacy_optimizer_artifact_is_inspectable_but_not_promotable_to_frozen(tmp_path,monkeypatch):
    monkeypatch.setattr(throughput,'ROOT',tmp_path)
    m=OnlineThroughputMLP(FEATURES,TARGETS)
    for i in range(24):
        x=[0.]*len(FEATURES);x[-4+i%4]=1.;m.add(x,[6.]+[1.]*6+[0.,0.])
    path=tmp_path/'outputs/legacy_unit_fixture.json';m.save(path);p=json.loads(path.read_text())
    p.update(version=1,coefs=[[[0.]*8 for _ in FEATURES],[[0.]*len(TARGETS) for _ in range(8)]],
        intercepts=[[0.]*8,[0.]*len(TARGETS)],x_scaler={'mean':[0.]*len(FEATURES),'scale':[1.]*len(FEATURES)},
        y_scaler={'mean':[6.]+[1.]*6+[0.,0.],'scale':[1.]*len(TARGETS)},fit_parameters=parameters(173),
        fit_diagnostics={'n_iter':10,'loss':0.,'converged':True},trained_samples=24,fit_count=1)
    p.pop('optimizer_metadata');path.write_text(json.dumps(p));original=path.read_bytes()
    loaded=OnlineThroughputMLP.load(path,FEATURES,TARGETS)
    assert loaded.predict([[0.]*len(FEATURES)])==[[6.]+[1.]*6+[0.,0.]]
    with pytest.raises(ValueError,match='verified mean-output optimizer'):
        app({'mode':'frozen','model_path':str(path)})
    online=app({'mode':'online','model_path':str(path)})
    online.run(obs(),NS())
    assert online.status=='rules_only' and online.pending['action']==0
    assert path.read_bytes()==original


def test_bad_weighted_bootstrap_then_bad_measured_rr_releases_exploration():
    c=app();x=[0.]*len(FEATURES);x[-4]=1.
    c.learner.add(x,[6.]+[1.]*6+[0.,0.])
    snapshot,_=start(c) # Least-sampled arm is now the neutral weighted family.
    assert c.pending['action']==1
    packet={'ue_6':{'overflow_packets_total':2}}
    c.run(obs(250,snapshot,c.name,bits=200000,packet_delta=packet),NS())
    assert c.learner.sample_count==2 and c.learner._y[-1][8]==20.
    assert c.pending['action']==0 and c.bootstrap_cooldown_pending
    assert c.last_decisions[0]['reason']=='bootstrap_urllc_measured_rr_cooldown'
    rr_snapshot=deepcopy(c.pending['snapshot'])
    # Rejected/unconfirmed requests cannot release a cooldown.
    c.run(obs(260,rr_snapshot,owner='other',bits=200000,packet_delta=packet),NS())
    assert c.bootstrap_cooldown_pending
    c.run(obs(270,rr_snapshot,c.name,bits=200000,packet_delta=packet),NS())
    c.run(obs(310,rr_snapshot,c.name,bits=200000,packet_delta=packet),NS())
    worse={'ue_6':{'overflow_packets_total':4}}
    c.run(obs(510,rr_snapshot,c.name,bits=300000,packet_delta=worse),NS())
    assert c.learner.sample_count==3 and c.learner._y[-1][8]==20.
    assert c.last_measured_action==0 and c.last_guard_bad # The RR window is also bad.
    assert not c.bootstrap_cooldown_pending and c.pending['action']==2
    assert any(v['kind']=='bootstrap_cooldown_required' for v in c.events)
    assert any(v['kind']=='bootstrap_cooldown_released' and v['rr_window_guard_bad'] for v in c.events)


def test_incomplete_online_resume_keeps_all40_rows_and_does_not_blame_bad_rr(tmp_path,monkeypatch):
    monkeypatch.setattr(throughput,'ROOT',tmp_path)
    m=OnlineThroughputMLP(FEATURES,TARGETS)
    for action in [0]*25+[1]*5+[2]*5+[3]*4+[0]:
        x=[0.]*len(FEATURES);x[-4+action]=1.;m.add(x,[6.]+[1.]*6+[0.,0.])
    m._y[-1][8]=20. # Synthetic unit history: the last measured action was bad RR.
    path=tmp_path/'outputs/old40.json';m.save(path);original=path.read_bytes()
    c=app({'model_path':str(path)})
    assert c.last_measured_action==0 and c.last_guard_bad and not c.bootstrap_cooldown_pending
    c.run(obs(),NS())
    assert c.pending['action']==3 and c.action_samples()=={'0':26,'1':5,'2':5,'3':4}
    assert c.learner._x==m._x and c.learner._y==m._y and path.read_bytes()==original
    assert c.summary()['maximum_measured_rows']==throughput.MAX_ROWS


def test_frozen_postcoverage_and_fixed_guard_branches_are_unchanged():
    c=app();c.last_guard_bad=True;c.bootstrap_cooldown_pending=True
    c.mode='frozen' # Direct branch test: no inference is reached or fabricated.
    assert c._choose(obs().ue_observations)==(0,None,'rules_only','observed_urllc_guard_abstention')
    c.mode='fixed'
    assert c._choose(obs().ue_observations)==(3,None,'fixed','predeclared_fixed_policy')
    c.mode='online'
    for i in range(24):
        x=[0.]*len(FEATURES);x[-4+i%4]=1.;c.learner.add(x,[6.]+[1.]*6+[0.,0.])
    assert c._coverage()
    assert c._choose(obs().ue_observations)==(0,None,'rules_only','observed_urllc_guard_abstention')


def policy_prediction_stub():
    """A policy-unit stub: shared forecasts/weights, never a deployment model."""
    xs=[]
    for i in range(24):
        x=[0.]*len(FEATURES);x[-4+i%4]=1.;xs.append(x)
    predictions=[[6.]+[1.]*6+[0.,0.],
        [7.2]+[1.2]*6+[0.,20.],[8.4]+[1.4]*6+[0.,40.],[9.6]+[1.6]*6+[0.,60.]]
    def forbidden(*args,**kwargs):raise AssertionError('Policy selection attempted training/persistence')
    return NS(_x=xs,_y=[[6.]+[1.]*6+[0.,0.] for _ in xs],trained=True,optimizer_verified=True,
        sample_count=24,trained_samples=24,coefs=['same unit stub weights'],
        predict=lambda rows:deepcopy(predictions),add=forbidden,fit=forbidden,save=forbidden)


def test_same_forecasts_guarded_abstains_but_throughput_policy_selects_best_with_qos_disclosed():
    model=policy_prediction_stub();before=deepcopy(model.coefs)
    guarded=app({'policy':'guarded'});throughput_only=app({'policy':'throughput'})
    for c in (guarded,throughput_only):
        c.mode='frozen';c.learner=model
        c.run(obs(),NS())
    assert guarded.pending['action']==0
    assert throughput_only.pending['action']==3
    assert model.coefs==before
    assert guarded.last_decisions[0]['policy']=='guarded' and guarded.last_decisions[0]['constraints_active']
    decision=throughput_only.last_decisions[0]
    assert decision['policy']=='throughput' and not decision['constraints_active']
    assert '3' in decision['prediction_guard_audit']['unapplied_guard_warnings']
    assert decision['prediction_guard_audit']['rejected_actions']=={}
    assert throughput_only.events[-1]['policy']=='throughput' and not throughput_only.events[-1]['constraints_active']


def test_throughput_policy_bypasses_observed_guard_and_online_bootstrap_cooldown():
    guarded=app();raw=app({'policy':'throughput'})
    for c in (guarded,raw):
        x=[0.]*len(FEATURES);x[-4]=1.;c.learner.add(x,[6.]+[1.]*6+[0.,0.])
        c.last_guard_bad=True;c.bootstrap_cooldown_pending=True
    assert guarded._choose(obs().ue_observations)[0]==0
    assert raw._choose(obs().ue_observations)[0]==1
    raw.learner=policy_prediction_stub();raw.mode='frozen'
    assert raw._choose(obs().ue_observations)[0]==3


def test_policy_default_validation_and_fixed_policy_unaffected():
    c=ThroughputSchedulerMLP(CELLS);c.select(CELLS,{cid:1 for cid in CELLS})
    assert c.policy=='throughput' and not c.summary()['constraints_active']
    c.mode='frozen';c.learner=policy_prediction_stub();c.run(obs(),NS())
    assert c.pending['action']==3 # Default follows learned throughput despite poor QoS forecasts.
    for policy in ('guarded','throughput'):
        fixed=app({'mode':'fixed','fixed_action':3,'policy':policy})
        fixed.last_guard_bad=True
        assert fixed._choose(obs().ue_observations)[0]==3
        assert fixed.summary()['policy']==policy and not fixed.summary()['constraints_active']
    with pytest.raises(ValueError,match='policy'):app({'policy':'safe_sounding_unknown'})

"""Public-observation and artifact invariants; fixtures are not training data."""
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

import pytest

from ric.near_rt.pools.chongyu.latency_scheduler_adaptive import (
    ACTIONS,FEATURES,CONTEXT_FEATURES,JOINT_FEATURES,SCHEMA,LatencySchedulerAdaptive,
    joint_feature_row,load_policy,predict)
from test_adaptive_mec_membership import observation

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def policy_path():
    (ROOT/'outputs').mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='test-mec-policy-',dir=ROOT/'outputs')as folder:
        yield Path(folder)/'policy.json'

def model(**extra):
    return dict(schema=SCHEMA,kind='joint_rf',actions=ACTIONS,
        feature_names=JOINT_FEATURES,action_vectors=[[0,0],[1,1]],
        forest=[dict(children_left=[-1],children_right=[-1],feature=[-2],threshold=[-2.],value=[-.1])],**extra)

def app(path,value=None):
    path.write_text(json.dumps(value or model()))
    return LatencySchedulerAdaptive(['cell_0','cell_1','cell_2'],
        parameters=dict(mode='frozen',adaptive_policy_path=str(path)))

def test_joint_policy_does_not_deploy_a_stale_or_inactive_half(policy_path):
    a=app(policy_path);a.select(['cell_0'],{'cell_0':1})
    assert not a.run(observation(),NS()).config_actions
    assert a.status_reason=='joint_policy_requires_all_cells'
    a.select(a.allowed_cells,{cid:1 for cid in a.allowed_cells})
    stale=observation()
    next(u for u in stale.ue_observations if u.ue_id=='ue_4').kpi_metadata['ue_energy_total_j']['produced_at_slot']-=2
    assert not a.run(stale,NS()).config_actions
    assert a.status_reason=='joint_public_state_not_ready'

def test_joint_policy_uses_complete_priority_maps_and_only_its_two_controls(policy_path):
    a=app(policy_path);a.select(a.allowed_cells,{cid:1 for cid in a.allowed_cells})
    actions=a.run(observation(),NS()).config_actions
    assert len(actions)==2
    assert {uid:w for item in actions for uid,w in item.ue_scheduler_weights.items()}=={
        'ue_0':2.,'ue_4':2.,'ue_6':.5,'ue_2':.5}
    assert all(item.ue_offload_fraction is item.ue_drx_configs is None for item in actions)
    assert all(d['produced_slot']<2080 for d in a.last_decisions)
    assert not a.run(observation(2480),NS()).config_actions

def test_nonbeneficial_estimate_retains_native_scheduling(policy_path):
    p=model();p['forest'][0]['value']=[.1]
    a=app(policy_path,p);a.select(a.allowed_cells,{cid:1 for cid in a.allowed_cells})
    actions=a.run(observation(),NS()).config_actions
    assert all(item.scheduler_mode=='round_robin'and set(item.ue_scheduler_weights.values())=={1.}for item in actions)
    assert all(d['prediction']['action_vector']==[0,0]for d in a.last_decisions)

def test_joint_feature_order_is_stable():
    row=joint_feature_row([[1.]*30,[2.]*30],[0,1])
    assert len(row)==len(JOINT_FEATURES)==68
    assert row[:30]==[1.]*30 and row[30:60]==[2.]*30
    assert row[60:]==[0.,0.,1.,1.,1.,0.,2.,.5]

def test_release_installer_works_as_a_standalone_command(policy_path):
    import os
    import subprocess
    import sys
    release=ROOT/'artifacts/xapps/chongyu/20261124'
    destination=policy_path.parent/'installed'
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(ROOT/'scripts/orin/install-trained-xapps.py'),
        '--bundle',str(release),'--output',str(destination)],cwd=policy_path.parent,
        env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['models']==6
    assert (destination/'models/x12_mec.json').read_bytes()==(release/'models/x12_mec.json').read_bytes()

@pytest.mark.parametrize('mutation',[
    lambda p:p['forest'][0].update(children_left=[0],children_right=[0],feature=[0]),
    lambda p:p['forest'][0].update(value=[float('nan')]),
    lambda p:p.update(action_vectors=[[1,1]]),
    lambda p:p.update(refresh_on_membership_change='yes'),
])
def test_malformed_frozen_policy_is_rejected(policy_path,mutation):
    p=model();mutation(p);policy_path.write_text(json.dumps(p))
    with pytest.raises(ValueError):load_policy(policy_path)

def test_exported_tree_inference_matches_sklearn_float32():
    np=pytest.importorskip('numpy');ensemble=pytest.importorskip('sklearn.ensemble')
    rng=np.random.default_rng(905);x=rng.normal(size=(50,len(FEATURES)));y=rng.normal(size=50)
    rf=ensemble.RandomForestRegressor(n_estimators=9,max_depth=4,random_state=18).fit(x,y)
    forest=[]
    for estimator in rf.estimators_:
        t=estimator.tree_;forest.append(dict(children_left=t.children_left.tolist(),children_right=t.children_right.tolist(),
            feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value.ravel().tolist()))
    assert [predict(forest,row)[0]for row in x.tolist()]==pytest.approx(rf.predict(x).tolist(),abs=1e-12)

def test_online_ppo_update_is_checkpointed_without_consuming_simulator_rng(tmp_path):
    torch=pytest.importorskip('torch')
    from ric.near_rt.pools.chongyu.latency_scheduler_ppo import PPOTrainer,forward
    policy=dict(schema=SCHEMA,kind='ppo',actions=ACTIONS,allowed_actions=[0,1,2,3,4],
        rollout_rows=4,ppo_epochs=2,minibatch_rows=4)
    before=torch.random.get_rng_state().clone();python_before=random.getstate()
    trainer=PPOTrainer(policy,tmp_path/'events.jsonl',41)
    assert torch.equal(before,torch.random.get_rng_state())and python_before==random.getstate()
    ctx=[.1]*len(CONTEXT_FEATURES)
    with torch.no_grad():dist,value=trainer.evaluate(trainer.tensor(ctx))
    _,probs,v=forward(trainer.policy,ctx)
    assert probs==pytest.approx(dist.probs.tolist(),abs=1e-7)and v==pytest.approx(float(value),abs=1e-7)
    users=[NS(ue_task_metrics=dict(offloaded_completed=0,offloaded_completion_latency_ms_sum=0,
        offloaded_failed=0,offloaded_pending=1))]
    for slot in range(100,700,100):
        trainer.finish_step('cell_0',ctx,users,slot,.5);trainer.maybe_update()
        trainer.choose('cell_0',ctx,users,slot)
        users[0].ue_task_metrics['offloaded_completed']+=1
        users[0].ue_task_metrics['offloaded_completion_latency_ms_sum']+=12
    assert trainer.updates==1 and(tmp_path/'ppo-checkpoints/update-001.json').is_file()
    assert torch.equal(before,torch.random.get_rng_state())and python_before==random.getstate()
    updates=[json.loads(line)for line in(tmp_path/'ppo-training.jsonl').read_text().splitlines()if json.loads(line)['kind']=='ppo_update']
    assert updates[0]['actual_behavior_log_prob_used']and updates[0]['gae_grouped_by_cell']

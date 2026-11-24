"""Synthetic lifecycle fixtures only; these are never model-training samples."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

from ric.near_rt.pools.chongyu.latency_scheduler_adaptive import (
    ACTIONS,JOINT_FEATURES,SCHEMA,LatencySchedulerAdaptive)

ROOT=Path(__file__).resolve().parents[1]

def observation(slot=2080,moved=False):
    users=[]
    for uid,cid in [('ue_0','cell_0'),('ue_6','cell_2'if moved else'cell_0'),('ue_4','cell_2'),('ue_2','cell_2')]:
        users.append(NS(ue_id=uid,serving_cell_id=cid,slice_id='embb'if uid in('ue_0','ue_4')else'urllc',
            queue_bytes=2000,hol_delay_ms=4.,throughput_mbps=1.,offered_rate_bps=2e6,reference_sinr_db=12.,
            data_sinr_db=10.,bler=.1,retx_fraction=.1,retx_sample_count=20,ue_active_fraction=.9,
            ue_task_metrics={'offloaded_pending':1}if uid in('ue_0','ue_4')else None,
            ue_service_policy=dict(offload_fraction=.5,drx=dict(cycle_ms=40,inactivity_ms=8)),
            kpi_metadata={'ue_energy_total_j':dict(produced_at_slot=slot-2,valid=True)}))
    cells=[NS(cell_id=cid,executed_config=dict(scheduler_mode='round_robin',
        ue_scheduler_weights={u.ue_id:1. for u in users if u.serving_cell_id==cid}),
        kpi_metadata={'executed_config':dict(produced_at_slot=slot-2,valid=True)})for cid in('cell_0','cell_1','cell_2')]
    return NS(slot_id=slot,ue_observations=users,cell_observations=cells)

def exercise(refresh):
    (ROOT/'outputs').mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='test-mec-membership-',dir=ROOT/'outputs')as folder:
        path=Path(folder)/'policy.json'
        path.write_text(json.dumps(dict(schema=SCHEMA,kind='joint_rf',actions=ACTIONS,
            feature_names=JOINT_FEATURES,action_vectors=[[0,0],[1,1]],refresh_on_membership_change=refresh,
            startup_hold_ms=260,decision_ms=260,
            forest=[dict(children_left=[-1],children_right=[-1],feature=[-2],threshold=[-2.],value=[-.1])])) )
        app=LatencySchedulerAdaptive(['cell_0','cell_1','cell_2'],parameters=dict(mode='frozen',adaptive_policy_path=str(path)))
        app.select(app.allowed_cells,{cid:1 for cid in app.allowed_cells})
        assert len(app.run(observation(),NS()).config_actions)==2
        assert not app.run(observation(2120),NS()).config_actions
        result=app.run(observation(2160,moved=True),NS())
        if refresh:
            assert {a.target_cell_id:set(a.ue_scheduler_weights)for a in result.config_actions}=={
                'cell_0':{'ue_0'},'cell_2':{'ue_2','ue_4','ue_6'}}
            assert all(d['membership_refresh']for d in app.last_decisions)
            assert not app.run(observation(2200,moved=True),NS()).config_actions
        else:assert not result.config_actions

def test_membership_event_refreshes_both_cells_without_waiting_for_the_hold():exercise(True)
def test_explicit_old_execution_ablation_preserves_the_original_hold():exercise(False)

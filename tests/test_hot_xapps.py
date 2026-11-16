"""Actual released learners and the real delayed lease/executor lifecycle.

Propagation/ACKs here are explicitly analytic fixtures, not Sionna evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS

import pytest

from experiments.hot_xapps import ROOT,HotXAppSession,configure,load_timeline,resolve_ids,submit_many


@pytest.fixture
def network():
    from test_rt_sys import FakeTracer,FixtureAbstraction
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from core.factory import build_controller_registry,build_scheduler
    from core.simulator import Simulator
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    (ROOT/'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test-hot-xapps-',dir=ROOT/'outputs')as folder:
        output=Path(folder)
        manifest,configs,_=build_case(num_cells=3,num_ues=6,total_slots=1000,seed=55,initial_queue_bytes=20000,
            config_path=ROOT/'configs/rt_sys/agx_munich_6ue_depth2_fast.json')
        configs['phy']['rt_sys']['ap_services']=json.loads((ROOT/'configs/rt_sys/ap_services_reference.json').read_text())['ap_services']
        args=NS(live_xapps=True,live_xapp_test=False,xapp_pool='chongyu',phase='integration',control_mode='baseline',
            slots=1000,seed=55,xapps_start='',xapp_test_start='off',xapp_test_app=None,_live_slot_ms=.5)
        configure(configs,manifest,args,output)
        configs['ric']['xapp_interval_slots']=10
        configs['sim']['config_actuation_delay_slots']=5
        configs['phy']['rt_sys']['feedback_mode']='sys_abstraction'
        configs['phy']['rt_sys'].update(nr_execution='scalar',fast_profile='reference',fast_channel_mode='dense')
        provider=RtChannelProvider(manifest,output/'cache',backend=FakeTracer(manifest))
        adapter=RtSysPhyAdapter({**configs['phy']['rt_sys'],'device':'numpy'},abstraction=FixtureAbstraction(),channel_provider=provider)
        runtime=build_runtime_scenario_config(scenario_name='hot_fixture',scenario_type='rt_scene',
            sim_config=configs['sim'],cell_config=configs['cell'],ric_config=configs['ric'])
        sim=Simulator(runtime,build_controller_registry(configs['ric']),build_scheduler(configs['sim']),
            adapter,configs['cell'],configs['ue'],configs['beamforming'],str(output))
        session=HotXAppSession(sim,args,output,output/'live.json')
        records=[]
        def step(n):
            for _ in range(n):
                session.before_step();sim.step();session.after_step(adapter)
                records.extend(deepcopy(sim.state.control.actuation_outcomes))
        yield NS(sim=sim,session=session,step=step,records=records,output=output,args=args,configs=configs,manifest=manifest)


def selected(n):return set(n.session.snapshot()['active_app_ids'])


def test_warm_multi_activation_targeted_disable_reenable_and_delayed_restoration(network):
    n=network;s=n.session;n.step(30)
    learners={k:id(a.learner)for k,a in s.apps.items()}
    s.apply_command('enable',['ue_energy_rf','mec_offload_budget_qgbdt'],'*',n.sim.state.slot_id)
    n.step(70)
    assert selected(n)=={'ue_energy_rf','mec_offload_budget_qgbdt'}
    token=dict(s.apps['mec_offload_budget_qgbdt'].tokens)
    stale=s.apps['ue_energy_rf'].identity('cell_0')
    with pytest.raises(ValueError,match='whole network'):
        s.apply_command('enable',['throughput_scheduler_mlp'],'cell_0',n.sim.state.slot_id)
    assert selected(n)=={'ue_energy_rf','mec_offload_budget_qgbdt'}
    s.apply_command('enable',['throughput_scheduler_mlp'],'*',n.sim.state.slot_id)
    n.step(40)
    assert len(selected(n))==3
    assert s.apps['mec_offload_budget_qgbdt'].tokens==token
    assert n.sim.controller_manager.instance_selection.valid(stale)
    s.apply_command('disable',['ue_energy_rf'],'*',n.sim.state.slot_id)
    assert not n.sim.controller_manager.instance_selection.valid(stale)
    n.step(40)
    assert selected(n)=={'mec_offload_budget_qgbdt','throughput_scheduler_mlp'}
    assert s.apps['mec_offload_budget_qgbdt'].tokens==token
    s.apply_command('enable',['ue_energy_rf'],'*',n.sim.state.slot_id);n.step(40)
    assert s.apps['ue_energy_rf'].identity('cell_0')['selection_version']!=stale['selection_version']
    assert learners=={k:id(a.learner)for k,a in s.apps.items()}
    old_pending=s.apps['throughput_scheduler_mlp'].pending
    assert old_pending is not None
    s.apply_command('disable',['throughput_scheduler_mlp'],'*',n.sim.state.slot_id)
    assert s.apps['throughput_scheduler_mlp'].pending is None
    s.apply_command('enable',['throughput_scheduler_mlp'],'*',n.sim.state.slot_id)
    n.step(40)
    assert s.apps['throughput_scheduler_mlp'].pending is not old_pending
    assert id(s.apps['throughput_scheduler_mlp'].learner)==learners['throughput_scheduler_mlp']
    assert any(r['status']=='applied'and r.get('provenance')for r in n.records)
    s.apply_command('disable_all',[],'*',n.sim.state.slot_id);n.step(40)
    assert not selected(n)
    assert all(c.scheduler.scheduler_mode=='round_robin'for c in n.sim.state.managed_cells.values())
    assert all(getattr(u,'scheduler_weight',1.)==1. for u in n.sim.state.ues)
    s.verify_models()
    assert all((n.output/p['checkpoint']).is_file()for p in s.snapshot()['model_artifacts'].values())


def test_all_six_including_both_latency_objectives_coexist_with_actual_model_requests(network):
    n=network;s=n.session;s.apply_command('set',sorted(s.apps),'*',0);n.step(160)
    assert selected(n)==set(s.apps)and len(selected(n))==6
    groups=n.sim.controller_manager.instance_selection.bindings
    assert all(set(g['latency_scheduler'])=={'latency_scheduler_rf','latency_urllc_rf'}for g in groups.values())
    for ident,app in s.apps.items():
        assert app.mode=='frozen'
        assert app.learner.trained
        assert any(e.get('kind')=='decision'for e in app.events),ident
    assert any(r['status']=='applied'and len(r.get('provenance',[]))for r in n.records)
    assert n.sim.state.slot_id==160
    s.verify_models()


def test_atomic_invalid_request_stale_run_and_cell_scoped_disable(network):
    n=network;s=n.session;path=n.output/'live.json'
    data=lambda:dict(stage='simulating',xapp_test=s.snapshot())
    first=submit_many(path,data(),'enable',['x01','x13']);n.step(1)
    assert s.receipts[-1]['request_id']==first and s.receipts[-1]['status']=='accepted'
    before=deepcopy(n.sim.controller_manager.instance_selection.bindings)
    submit_many(path,data(),'enable',['x08','missing']);n.step(1)
    assert s.receipts[-1]['status']=='rejected'
    assert n.sim.controller_manager.instance_selection.bindings==before
    submit_many(path,data(),'disable',['x01'],cell='cell_1');n.step(1)
    assert 'cell_1'not in s.apps['ue_energy_rf'].active_cells
    assert s.apps['mec_offload_budget_qgbdt'].active_cells=={'cell_0','cell_1','cell_2'}
    from experiments.live_xapp_test import atomic_json
    atomic_json(s.command_dir/'stale.json',dict(run_id='old',request_id='stale',operation='disable_all',apps=[],cell='*'))
    n.step(1);assert s.receipts[-1]['status']=='rejected'and len(selected(n))==2
    with pytest.raises(ValueError,match='finished'):submit_many(path,dict(stage='completed',xapp_test=s.snapshot()),'enable',['x08'])


def test_simulation_time_timeline_is_recorded_and_does_not_reset_network(network):
    n=network;s=n.session
    requests=[(10,'enable',['ue_energy_rf','mec_offload_budget_qgbdt']),
              (50,'enable',['throughput_scheduler_mlp']),(90,'disable',['ue_energy_rf']),
              (130,'disable_all',[])]
    s.timeline=[dict(run_id=s.run_id,request_id='timeline:'+str(i),operation=op,apps=ids,
        cell='*',scheduled_slot=slot,source='simulation_timeline')for i,(slot,op,ids)in enumerate(requests)]
    n.step(160)
    assert [r['received_at_slot']for r in s.receipts]==[10,50,90,130]
    assert all(r['status']=='accepted'for r in s.receipts)
    assert n.sim.state.slot_id==160 and not selected(n)
    assert {len(seg['app_ids'])for seg in s.segments}>={0,2,3}
    events=[json.loads(r)for r in s.events.read_text().splitlines()]
    assert events[-1]['active_sets']=={} and events[-1]['operation']=='disable_all'


def test_timeline_and_alias_validation(tmp_path):
    lookup={'x01':'ue_energy_rf','ueenergyrf':'ue_energy_rf'}
    assert resolve_ids(['X-01'],lookup)==['ue_energy_rf']
    with pytest.raises(ValueError,match='Duplicate'):resolve_ids(['x01','ue_energy_rf'],lookup)
    p=tmp_path/'timeline.json';p.write_text(json.dumps([dict(at_ms=10,operation='enable',apps=['x01'])]))
    assert load_timeline(p,lookup,.5,100)[0]['scheduled_slot']==20
    p.write_text(json.dumps([dict(at_ms=10.1,operation='enable',apps=['x01'])]))
    with pytest.raises(ValueError,match='align'):load_timeline(p,lookup,.5,100)


@pytest.mark.parametrize('flags,expected',[
    ([], 'priority'),(['--live-xapps'],'uniform_random'),
    (['--live-xapps','--direct-conflict-mitigation','priority'],'priority')])
def test_cli_keeps_static_default_and_selects_live_random_default(monkeypatch,flags,expected):
    from experiments.run_rt_sys import parse_args
    monkeypatch.setattr('sys.argv',['run_rt_sys.py',*flags])
    assert parse_args().direct_conflict_mitigation==expected

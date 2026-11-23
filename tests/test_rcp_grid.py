"""Range coverage and legal endpoint execution; no synthetic network results."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import json

import pytest

from experiments.run_rcp_grid import grid_design,dynamics_signature,ROOT
from experiments.rcp_perturbation import validate,vector,build_controller


def test_grids_cover_declared_ranges_and_share_complete_native_warmup():
    specs,grids,_=grid_design(ROOT);warm=vector(specs['native'],perturbed=False)
    assert len(specs)==30
    for s in specs.values():assert vector(s,perturbed=False)==warm
    assert [(min(g['levels']),max(g['levels']))for g in grids]==[(0,1),(0,40),(.5,2),(.5,4)]
    for g in grids:
        assert len(g['levels'])>=7 and len(set(g['levels']))==len(g['levels'])
        for arm,x in zip(g['arms'],g['levels']):assert specs[arm][g['field']]==x
    s=deepcopy(specs['qos_native']);s['urllc_weight']=4;validate(s)
    s['urllc_weight']=4.01
    with pytest.raises(ValueError):validate(s)
    s['urllc_weight']=3;s['scheduler_mode']='latency_aware'
    with pytest.raises(ValueError):validate(s)


def test_reuse_signature_ignores_validator_range_but_detects_runtime_changes(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b'
    source=(ROOT/'experiments/rcp_perturbation.py').read_text()
    for root in(a,b):
        (root/'experiments').mkdir(parents=True);(root/'experiments/rcp_perturbation.py').write_text(source)
        (root/'core').mkdir();(root/'core/state.py').write_text('VALUE=1\n')
    assert dynamics_signature(a)==dynamics_signature(b)
    (b/'experiments/rcp_perturbation.py').write_text(source.replace("('switch_ms',20,500)","('switch_ms',20,501)"))
    assert dynamics_signature(a)==dynamics_signature(b)
    (b/'core/state.py').write_text('VALUE=2\n')
    assert dynamics_signature(a)!=dynamics_signature(b)


@pytest.mark.parametrize('passed',[True,False])
def test_worker_waits_for_bridge_verification_before_admitting_grid_cases(tmp_path,monkeypatch,passed):
    from experiments import run_rcp_perturbation as runner
    started=[];source='a'*64
    cases=[dict(name=n,role=n,output=str(tmp_path/'cases'/n),command=[n])for n in('bridge','grid')]
    plan=dict(output=str(tmp_path),cases=cases,requires_bridge=True,source_sha256=source)
    monkeypatch.setattr(runner,'verify',lambda plan:None)
    monkeypatch.setattr(runner,'completed',lambda case:case['name']in started)
    def launch(command,**kwargs):
        if command==['grid']:assert (tmp_path/'bridge_verified.json').exists()
        started.append(command[0]);return NS(pid=len(started),poll=lambda:0)
    def advance(_):
        if (tmp_path/'checkpoints/bridge.json').exists():
            runner.atomic(tmp_path/('bridge_verified.json'if passed else'bridge_failed.json'),dict(source_sha256=source))
    monkeypatch.setattr(runner.subprocess,'Popen',launch);monkeypatch.setattr(runner.time,'sleep',advance)
    if passed:
        runner.run(plan,parallel=2);assert started==['bridge','grid']
    else:
        with pytest.raises(RuntimeError,match='Bridge|bridge'):runner.run(plan,parallel=2)
        assert started==['bridge'] and (tmp_path/'checkpoints/bridge.json').exists()


@pytest.mark.parametrize('changes',[dict(offload_probability=0),dict(drx_inactivity_ms=0),
    dict(scheduler_mode='latency_aware',mec_weight=.5),dict(scheduler_mode='qos_deadline',urllc_weight=4),dict(drx_cycle_ms=80),dict(cell_sleep_mode='light_sleep'),dict(cell_sleep_mode='deep_sleep')])
def test_real_ric_accepts_range_endpoints_with_delayed_atomic_execution(tmp_path,changes):
    from test_rt_sys import FakeTracer,FixtureAbstraction
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from core.factory import build_controller_registry,build_scheduler
    from core.simulator import Simulator
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    from experiments.run_rt_sys import configure_ue_services
    specs,_,_=grid_design(ROOT);spec=validate({**specs['native'],**changes,'switch_ms':20})
    m,c,_=build_case(num_cells=3,num_ues=6,total_slots=600,seed=77,config_path=ROOT/'configs/rt_sys/agx_munich_6ue_depth2_fast.json')
    configure_ue_services(c,m,NS(ue_services='drx-mec'))
    c['ric'].update(rapps=[],xapps=[],xapp_interval_slots=40,run_rapps_on_start=False);c['sim']['config_actuation_delay_slots']=3
    c['phy']['rt_sys'].update(feedback_mode='sys_abstraction',fast_profile='reference',fast_channel_mode='dense',nr_execution='scalar')
    provider=RtChannelProvider(m,tmp_path/'cache',backend=FakeTracer(m))
    adapter=RtSysPhyAdapter({**c['phy']['rt_sys'],'device':'numpy'},abstraction=FixtureAbstraction(),channel_provider=provider)
    registry=build_controller_registry(c['ric']);registry.register_xapp(build_controller(spec,tmp_path))
    runtime=build_runtime_scenario_config(scenario_name='grid_fixture',scenario_type='rt_scene',sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
    sim=Simulator(runtime,registry,build_scheduler(c['sim']),adapter,c['cell'],c['ue'],c['beamforming'],str(tmp_path/'run'))
    for _ in range(42):sim.step()
    assert all(u.scheduler_weight==1 for u in sim.state.ues)
    sim.step();target=vector(spec,perturbed=True)
    assert {u.ue_id:u.scheduler_weight for u in sim.state.ues}==target['ue_scheduler_weights']
    assert all(sim.state.ue_services.offload[u]==p for u,p in target['ue_offload_fraction'].items())
    assert all(sim.state.ue_services.drx[u]==p for u,p in target['ue_drx_configs'].items())
    assert not any(r['status']=='rejected'for r in sim.state.control.actuation_outcomes)

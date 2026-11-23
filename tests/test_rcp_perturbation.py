"""Actual delayed fixed-control execution with an explicit analytic PHY fixture."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from experiments.rcp_perturbation import validate,vector,build_controller,configure
from experiments.run_rcp_perturbation import REFERENCE,ARMS,CONTRASTS

ROOT=Path(__file__).resolve().parents[1]


def test_preregistered_contrasts_change_one_rcp_only_and_share_native_warmup():
    warm=vector(REFERENCE,perturbed=False)
    for name,changes in ARMS.items():
        spec=validate({**REFERENCE,**changes,'arm':name})
        assert vector(spec,perturbed=False)==warm
    for c in CONTRASTS:
        a=vector({**REFERENCE,**ARMS[c['low']]},perturbed=True)
        b=vector({**REFERENCE,**ARMS[c['high']]},perturbed=True)
        assert {k for k in a if a[k]!=b[k]}=={c['parameter']}
        if c['kind']=='local_derivative':assert a['scheduler_mode']==b['scheduler_mode']


@pytest.mark.parametrize('changes',[dict(offload_ue4=1.1),dict(weight_ue6=float('nan')),
    dict(scheduler_mode='made_up'),dict(switch_ms=105),dict(extra_action=1)])
def test_invalid_stimuli_are_rejected(changes):
    with pytest.raises(ValueError):validate({**REFERENCE,**changes})


@pytest.mark.parametrize('other_mode',['active_xapp','live','timeline','training_phase'])
def test_identification_cannot_run_with_unrelated_controllers(tmp_path,other_mode):
    path=tmp_path/'arm.json';path.write_text(json.dumps(REFERENCE))
    configs=dict(ric=dict(xapps=[],rapps=[]),phy=dict(rt_sys=dict(
        ue_services=dict(enabled=True,mec=dict(enabled=True)),ap_services=dict(enabled=True))))
    manifest=dict(ues=[dict(ue_id=u)for u in ('ue_0','ue_1','ue_2','ue_4','ue_6','ue_8')],slot_duration_ms=.5,total_slots=1600)
    args=NS(perturbation_spec=path,control_mode='baseline',phase='integration',live_xapps=False)
    if other_mode=='active_xapp':configs['ric']['xapps']=[dict(type='pool_xapp')]
    if other_mode=='live':args.live_xapps=True
    if other_mode=='timeline':args.control_timeline=path
    if other_mode=='training_phase':args.phase='train'
    before=deepcopy(configs)
    with pytest.raises(ValueError):configure(configs,manifest,args)
    assert configs==before


def test_real_ric_delays_and_applies_only_the_declared_intervention(tmp_path):
    from test_rt_sys import FakeTracer,FixtureAbstraction
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from core.factory import build_controller_registry,build_scheduler
    from core.simulator import Simulator
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    m,c,_=build_case(num_cells=3,num_ues=6,total_slots=600,seed=77,config_path=ROOT/'configs/rt_sys/agx_munich_6ue_depth2_fast.json')
    c['ric'].update(rapps=[],xapps=[],xapp_interval_slots=40,run_rapps_on_start=False)
    c['sim']['config_actuation_delay_slots']=3
    c['phy']['rt_sys']['ap_services']=json.loads((ROOT/'configs/rt_sys/ap_services_reference.json').read_text())['ap_services']
    c['phy']['rt_sys'].update(feedback_mode='sys_abstraction',fast_profile='reference',fast_channel_mode='dense',nr_execution='scalar')
    provider=RtChannelProvider(m,tmp_path/'cache',backend=FakeTracer(m))
    adapter=RtSysPhyAdapter({**c['phy']['rt_sys'],'device':'numpy'},abstraction=FixtureAbstraction(),channel_provider=provider)
    registry=build_controller_registry(c['ric'])
    registry.register_xapp(build_controller({**REFERENCE,'arm':'offload_high','offload_ue4':.6,'switch_ms':20},tmp_path))
    runtime=build_runtime_scenario_config(scenario_name='perturb_fixture',scenario_type='rt_scene',sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
    sim=Simulator(runtime,registry,build_scheduler(c['sim']),adapter,c['cell'],c['ue'],c['beamforming'],str(tmp_path/'run'))
    for _ in range(42):sim.step()
    assert sim.state.ue_services.offload['ue_4']==.5  # request at 40, apply at 43
    sim.step()
    assert sim.state.ue_services.offload['ue_4']==.6
    assert sim.state.ue_services.offload['ue_0']==.5
    assert all(cell.scheduler.scheduler_mode=='round_robin'for cell in sim.state.managed_cells.values())
    assert all(getattr(u,'scheduler_weight',1)==1 for u in sim.state.ues)
    assert sim.state.ue_services.drx['ue_4']['inactivity_ms']==8
    assert not sim.controller_manager.a1_interfaces
    assert any(r['status']=='applied'and any(f['field']=='ue_offload_fraction'for f in r.get('fields',[]))for r in sim.state.control.actuation_outcomes)

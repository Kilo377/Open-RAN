"""Complete service-class interventions through real delayed RIC execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from experiments.population_rcp_design import design
from experiments.rcp_perturbation import vector,validate,build_controller
from smo.identify import fit_contrasts
from smo.contracts import KPIS

ROOT=Path(__file__).resolve().parents[1]


def test_every_population_contrast_changes_only_one_rcp_and_full_applicable_group():
    ref,arms,contrasts=design(ROOT)
    warm=vector(ref,perturbed=False)
    for arm,changes in arms.items():
        assert vector(validate({**ref,**changes,'arm':arm}),perturbed=False)==warm
    for c in contrasts:
        a=vector({**ref,**arms[c['low']]},perturbed=True)
        b=vector({**ref,**arms[c['high']]},perturbed=True)
        assert {k for k in a if a[k]!=b[k]}=={c['parameter']}
        if c['parameter'] in ('ue_offload_fraction','ue_drx_configs','ue_scheduler_weights'):
            changed={uid for uid in a[c['parameter']]if a[c['parameter']][uid]!=b[c['parameter']][uid]}
            group={'offload_population':'mec','drx_population':'drx','weight_mec_population':'mec','weight_urllc_population':'urllc'}[c['id']]
            assert changed==set(ref['groups'][group]) and len(changed)>1
    bad=deepcopy(ref);bad['groups']['mec']=bad['groups']['mec'][:1]
    with pytest.raises(ValueError):validate(bad)


@pytest.mark.parametrize('arm',['offload_high','drx_short','weight_mec_high','weight_urllc_high'])
def test_real_executor_applies_group_values_only_after_the_normal_delay(tmp_path,arm):
    from test_rt_sys import FakeTracer,FixtureAbstraction
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from core.factory import build_controller_registry,build_scheduler
    from core.simulator import Simulator
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    from experiments.run_rt_sys import configure_ue_services
    ref,arms,_=design(ROOT);spec=validate({**ref,**arms[arm],'arm':arm,'switch_ms':20})
    m,c,_=build_case(num_cells=3,num_ues=6,total_slots=600,seed=77,config_path=ROOT/'configs/rt_sys/agx_munich_6ue_depth2_fast.json')
    configure_ue_services(c,m,NS(ue_services='drx-mec'))
    c['ric'].update(rapps=[],xapps=[],xapp_interval_slots=40,run_rapps_on_start=False)
    c['sim']['config_actuation_delay_slots']=3
    c['phy']['rt_sys'].update(feedback_mode='sys_abstraction',fast_profile='reference',fast_channel_mode='dense',nr_execution='scalar')
    provider=RtChannelProvider(m,tmp_path/'cache',backend=FakeTracer(m))
    adapter=RtSysPhyAdapter({**c['phy']['rt_sys'],'device':'numpy'},abstraction=FixtureAbstraction(),channel_provider=provider)
    registry=build_controller_registry(c['ric']);registry.register_xapp(build_controller(spec,tmp_path))
    runtime=build_runtime_scenario_config(scenario_name='population_fixture',scenario_type='rt_scene',sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
    sim=Simulator(runtime,registry,build_scheduler(c['sim']),adapter,c['cell'],c['ue'],c['beamforming'],str(tmp_path/'run'))
    for _ in range(42):sim.step()
    assert all(sim.state.ue_services.offload[u]==.5 for u in ref['groups']['mec'])
    assert all(sim.state.ue_services.drx[u]['inactivity_ms']==8 for u in ref['groups']['drx'])
    assert all(u.scheduler_weight==1 for u in sim.state.ues)
    sim.step();target=vector(spec,perturbed=True)
    assert all(sim.state.ue_services.offload[u]==p for u,p in target['ue_offload_fraction'].items())
    assert all(sim.state.ue_services.drx[u]==p for u,p in target['ue_drx_configs'].items())
    assert {u.ue_id:u.scheduler_weight for u in sim.state.ues}==target['ue_scheduler_weights']
    assert all(cell.scheduler.scheduler_mode==target['scheduler_mode']for cell in sim.state.managed_cells.values())
    assert not any(r['status']=='rejected'for r in sim.state.control.actuation_outcomes)


def test_estimator_keeps_mec_and_urllc_population_priority_contexts_separate():
    ref,arms,contrasts=design(ROOT)
    plan=dict(population=True,seeds=[1,2],contrasts=contrasts,specs={arm:{**ref,**v}for arm,v in arms.items()},
        scoring_start_ms=200,scoring_end_ms=700,switch_ms=100)
    cases=[dict(seed=s,arm=a,metrics=dict.fromkeys(KPIS,float(i+1)))for s in plan['seeds']for i,a in enumerate(arms)]
    rows=fit_contrasts(plan,cases,'a'*64,population=True)
    assert len(rows)==7*len(KPIS) and all(r['population_control_eligible']for r in rows)
    weights=[r for r in rows if r['parameter']=='ue_scheduler_weights'and r['kpi']=='network_goodput_mbps']
    assert {r['parameter_component']for r in weights}=={'mec_priority','urllc_priority'}
    assert all(r['parameter_levels']==dict(low=.75,reference=1.,high=1.25)for r in weights)


def test_uniform_qos_weight_scaling_preserves_priority_order():
    from ran.scheduler import SimpleScheduler
    scheduler=SimpleScheduler()
    state=NS(cell=NS(bandwidth_hz=10e6),scheduler=NS(priority_boost_by_ue={},qos_average_rate_mbps={}))
    ues=[NS(ue_id=str(i),latency_budget_ms=pdb,hol_delay_ms=hol,priority=p,sinr_db=sinr,throughput_mbps=rate,scheduler_weight=1.)
         for i,(pdb,hol,p,sinr,rate)in enumerate([(10.,2.,3,10.,1.),(100.,5.,1,15.,5.),(50.,3.,2,7.,.5)])]
    scores=[scheduler._qos_deadline_score(state,u)for u in ues]
    for u in ues:u.scheduler_weight=1.25
    scaled=[scheduler._qos_deadline_score(state,u)for u in ues]
    assert scaled==pytest.approx([v*1.25 for v in scores])
    assert sorted(range(3),key=scores.__getitem__)==sorted(range(3),key=scaled.__getitem__)

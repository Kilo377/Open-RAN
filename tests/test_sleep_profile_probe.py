"""Fixed experimental sleep stimuli never run trained policies or other RCPs."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace
import pytest

from experiments.sleep_profile_probe import SCHEMA,validate,configure,build_controller


def spec():
    return dict(schema=SCHEMA,profile_id='mixed',switch_slot=2040,
        profile={'cell_0':'light_sleep','cell_1':'deep_sleep','cell_2':'light_sleep'},calibration_plan_sha256='a'*64)


def test_one_shot_three_cell_sleep_only_at_the_preregistered_ric_boundary(tmp_path):
    stimulus=build_controller(spec(),tmp_path)
    def observe(slot):return SimpleNamespace(slot_id=slot,cell_observations=[SimpleNamespace(cell_id='cell_'+str(i))for i in range(3)])
    for slot in(40,2000):assert all(a.cell_sleep_mode is None for a in stimulus.run(observe(slot),None).config_actions)
    result=stimulus.run(observe(2040),None)
    assert len(result.config_actions)==3
    for action in result.config_actions:
        values=asdict(action)
        assert action.cell_sleep_mode==spec()['profile'][action.target_cell_id]
        assert action.ue_drx_configs is None and action.ue_offload_fraction is None and action.ue_scheduler_weights is None and action.scheduler_mode is None
        assert action.effective_from_slot==2040 and action.effective_until_slot==2079
    assert all(a.cell_sleep_mode is None for a in stimulus.run(observe(2080),None).config_actions)
    assert all(a.cell_sleep_mode is None for a in stimulus.run(observe(2040),None).config_actions)
    rows=[json.loads(line)for line in(tmp_path/'sleep_profile_requests.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['controlled_fields']==['cell_sleep_mode']


def test_probe_configuration_preserves_every_native_phy_parameter(tmp_path):
    cfg=dict(ric=dict(xapps=[],rapps=[]),phy=dict(rt_sys=dict(ap_services={'enabled':True},ue_services={'enabled':True},
        paper_interfaces={'enabled':True,'rule_observations':True},untouched=dict(scheduler='round_robin',drx=40,offload=.5))))
    before=deepcopy(cfg['phy']);path=tmp_path/'spec.json';path.write_text(json.dumps(spec()))
    args=SimpleNamespace(sleep_profile_spec=path);manifest=dict(cells=[{'cell_id':'cell_'+str(i)}for i in range(3)],slot_duration_ms=.5,total_slots=4440)
    assert configure(cfg,manifest,args)==spec() and cfg['phy']==before
    assert cfg['ric']['xapps']==[] and cfg['ric']['rapps']==[] and cfg['ric']['xapp_interval_slots']==40
    for name in('live_xapps','live_xapp_test','perturbation_spec','control_timeline','policy_timeline'):
        bad=SimpleNamespace(sleep_profile_spec=path,**{name:True})
        with pytest.raises(ValueError,match='independent'):configure(cfg,manifest,bad)


def test_probe_rejects_unregistered_fields_bad_profiles_and_wrong_tick():
    for change in(dict(profile={'cell_0':'light_sleep'}),dict(switch_slot=2041),dict(scheduler_mode='latency_aware')):
        with pytest.raises(ValueError):validate({**spec(),**change})


def test_calibration_grid_keeps_training_seeds_and_reference_pairs_together():
    from experiments.sleep_profile_calibration import assignment,digest
    p=dict(schema='oran.parameter-profile-calibration-plan.v1',training_seeds=[20261125,20261126,20261127],validation_seeds_excluded=[20261201,20261202],
        switch_slot=2040,total_slots=4440,scoring_start_slot=2440,scoring_end_slot=4440,reference_profile={cid:'active'for cid in spec()['profile']},
        profiles={scene:{'mixed':spec()['profile']}for scene in('native_1x','steady_2x','burst_2x')})
    p['plan_sha256']=digest(p);grid=assignment(p,p['plan_sha256']);rows=[r for cases in grid.values()for r in cases]
    assert len(rows)==18 and len({r['id']for r in rows})==18
    assert {r['seed']for r in rows}.isdisjoint(p['validation_seeds_excluded'])
    for cases in grid.values():
        for i in range(0,len(cases),2):
            assert cases[i]['arm']=='REF' and cases[i]['origin_key']==cases[i+1]['origin_key']


def test_real_ric_e2_receipts_only_apply_sleep_and_hold_policy_after_request_expiry(tmp_path):
    from pathlib import Path
    from test_rt_sys import FakeTracer,FixtureAbstraction
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from core.factory import build_controller_registry,build_scheduler
    from core.simulator import Simulator
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    root=Path(__file__).resolve().parents[1]
    manifest,configs,_=build_case(num_cells=3,num_ues=6,total_slots=100,seed=77,
        config_path=root/'configs/rt_sys/agx_munich_6ue_depth2_fast.json')
    configs['ric'].update(rapps=[],xapps=[],xapp_interval_slots=40,run_rapps_on_start=False)
    configs['sim']['config_actuation_delay_slots']=3
    configs['phy']['rt_sys']['ap_services']=json.loads((root/'configs/rt_sys/ap_services_reference.json').read_text())['ap_services']
    configs['phy']['rt_sys'].update(feedback_mode='sys_abstraction',fast_profile='reference',fast_channel_mode='dense',nr_execution='scalar')
    provider=RtChannelProvider(manifest,tmp_path/'cache',backend=FakeTracer(manifest))
    adapter=RtSysPhyAdapter({**configs['phy']['rt_sys'],'device':'numpy'},abstraction=FixtureAbstraction(),channel_provider=provider)
    registry=build_controller_registry(configs['ric']);registry.register_xapp(build_controller({**spec(),'switch_slot':40},tmp_path))
    runtime=build_runtime_scenario_config(scenario_name='sleep_probe_fixture',scenario_type='rt_scene',
        sim_config=configs['sim'],cell_config=configs['cell'],ric_config=configs['ric'])
    sim=Simulator(runtime,registry,build_scheduler(configs['sim']),adapter,configs['cell'],configs['ue'],configs['beamforming'],str(tmp_path/'run'))
    receipts=[]
    for i in range(85):
        sim.step();receipts.extend(deepcopy(sim.state.control.actuation_outcomes))
        if i==41:assert all(c.cell.cell_sleep_mode=='active'for c in sim.state.managed_cells.values())
    applied=[r for r in receipts if r.get('status')=='applied'and r.get('fields')]
    assert len(applied)>=3
    assert all({f['field']for f in r['fields']}=={'cell_sleep_mode'}for r in applied)
    (tmp_path/'real-sleep-receipts.json').write_text(json.dumps(receipts,indent=2))
    assert {cid:cell.cell.cell_sleep_mode for cid,cell in sim.state.managed_cells.items()}==spec()['profile']
    assert all(cell.scheduler.scheduler_mode=='round_robin'for cell in sim.state.managed_cells.values())
    assert all(getattr(u,'scheduler_weight',1)==1 for u in sim.state.ues)
    assert sim.state.ue_services.offload['ue_4']==.5 and sim.state.ue_services.drx['ue_4']['inactivity_ms']==8
    assert len((tmp_path/'sleep_profile_requests.jsonl').read_text().splitlines())==1

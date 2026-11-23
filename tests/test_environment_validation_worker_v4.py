"""Pinned external geometry and admission guards; no simulator subprocesses."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Event

import pytest

from experiments import environment_validation_worker_v4 as worker
from experiments import environment_validation_worker_v3 as v3


def seal(value):
    result = deepcopy(value)
    result.pop('plan_sha256', None)
    result['plan_sha256'] = worker.digest(result)
    return result


@pytest.fixture
def plan(tmp_path, monkeypatch):
    scope = tmp_path/'owned'
    scope.mkdir()
    monkeypatch.setattr(worker, 'SCOPE', scope)
    monkeypatch.setattr(v3, 'SCOPE', scope)
    root = scope/'source'
    base = root/worker.BASE_CONFIG
    base.parent.mkdir(parents=True)
    original = json.loads((Path(v3.__file__).resolve().parents[1]/worker.BASE_CONFIG).read_text())
    worker.atomic(base, original)
    path = scope/'protocol/profiles/geometry-test_ue0_road0_phase065_base.json'
    worker.atomic(path, {**original, 'ue_route_overrides': deepcopy(worker.ROUTE_OVERRIDE)})
    case = dict(id='a'*32, name='p_OFF', origin_key='p_s1', scene='p', seed=1,
        arm='OFF', app_ids=[], worker_id='agx1', slots=22440, baseline_slot=2040,
        origin_observe_slots=2000, observe_slots=20000, traffic_load=1.5,
        traffic_profile=str(scope/'protocol/p.json'), traffic_profile_sha256='2'*64,
        ue_services_config=str(scope/'protocol/service.json'), ue_services_config_sha256='3'*64,
        base_config=str(path), base_config_sha256=worker.sha(path))
    on = {**case, 'id':'b'*32, 'name':'p_A', 'arm':'A', 'app_ids':['xapp:ue_energy_rf']}
    return seal(dict(schema=worker.SCHEMA, study_id='geometry-test', stage='validation', worker_id='agx1',
        source_root=str(root), source_archive_path=str(scope/'source-v2.tgz'),
        source_sha256=worker.SOURCE_SHA256, source_archive_sha256=worker.ARCHIVE_SHA256,
        frozen_models={f'model/{i}':str(i)*64 for i in range(6)}, runner_sha256=worker.sha(worker.__file__),
        runner_path=str(scope/'runner/worker-v4.py'), active_plan_path=str(scope/'plans/geometry-test/validation.json'),
        output_root=str(root/'outputs/environment_validation/geometry-test/validation'),
        require_on_admission=True, geometry_change=deepcopy(worker.GEOMETRY_CHANGE),
        max_parallel=2, deadline_utc='2099-01-01T00:00:00+00:00', poll_seconds=.1, cases=[case,on]))


def test_v3_is_untouched_and_valid_v4_plan_is_namespaced(plan):
    assert worker.sha(v3.__file__) == '3247cfdd9e6e03f30784a57f571ce24b43d167c275e4dc16a586aea56724ad79'
    worker.validate_plan(plan, plan['plan_sha256'], plan_path=plan['active_plan_path'])
    assert worker.validate_base_config(plan, plan['cases'][0]) == Path(plan['cases'][0]['base_config'])


@pytest.mark.parametrize('mutate', [
    lambda x:x['radio'].update(slot_duration_ms=1),
    lambda x:x['rt_sys']['paper_interfaces'].update(connection_events=True),
    lambda x:x.update(ue_speed_mps=0),
    lambda x:x['ue_route_overrides'].update(ue_6=dict(route_index=0,start_fraction=.65)),
    lambda x:x['ue_route_overrides']['ue_0'].update(start_fraction=.6),
    lambda x:x['ue_route_overrides']['ue_0'].update(route_index=1),
    lambda x:x['ue_route_overrides']['ue_0'].update(extra='ignored-by-production-but-rejected-here'),
    lambda x:x['radio'].update(num_rx_ant=True),
    lambda x:x.update(unknown_setting=True),
])
def test_any_extra_physical_control_or_geometry_edit_is_rejected_even_with_new_sha(plan, mutate):
    case = plan['cases'][0]
    path = Path(case['base_config'])
    changed = worker.read(path)
    mutate(changed)
    worker.atomic(path, changed)
    case['base_config_sha256'] = worker.sha(path)
    with pytest.raises(ValueError, match='Only UE0'):
        worker.validate_base_config(plan, case)


def test_base_content_hash_and_owned_exact_filename_are_mandatory(plan, tmp_path):
    case = deepcopy(plan['cases'][0])
    case['base_config_sha256'] = 'f'*64
    with pytest.raises(ValueError):worker.validate_base_config(plan, case)
    case['base_config'] = str(tmp_path/'outside.json')
    with pytest.raises(ValueError):worker.validate_base_config(plan, case)


def test_on_cannot_pair_with_an_off_using_another_base_pin(plan):
    p = deepcopy(plan)
    p['cases'][1]['base_config_sha256'] = 'f'*64
    p = seal(p)
    with pytest.raises(ValueError, match='matching same-host OFF'):
        worker.validate_plan(p, p['plan_sha256'])


@pytest.mark.parametrize('value', [None, False, 1, 'true'])
def test_v4_cannot_disable_explicit_on_gate(plan, value):
    p = seal({**plan, 'require_on_admission':value})
    with pytest.raises(ValueError, match='explicit ON admission'):
        worker.validate_plan(p, p['plan_sha256'])


def test_command_only_appends_pinned_config_to_v3_case_argv(plan):
    out = Path(plan['output_root'])
    case = plan['cases'][1]
    oldplan = {**plan, 'runner_path':str(worker.SCOPE/'runner/worker-v3.py')}
    old = v3.command(oldplan, case, out)
    new = worker.command(plan, case, out)
    assert new[:-2] == old
    assert new[-2:] == ['--config', case['base_config']]
    assert '--config' not in old
    assert worker.pairing_recipe(case)['base_config_sha256'] == case['base_config_sha256']


def test_production_case_builder_preserves_five_routes_and_all_radio_controls(plan):
    import numpy as np
    from phy.rt_sys.scenario import build_case
    args = dict(scene='munich',num_cells=3,num_ues=6,seed=20261212,total_slots=22440,
                device='cuda',xapp_pool='chongyu')
    root=Path(plan['source_root'])
    old, old_configs, old_intent=build_case(config_path=root/worker.BASE_CONFIG,**args)
    new, new_configs, new_intent=build_case(config_path=plan['cases'][0]['base_config'],**args)
    assert old_intent==new_intent
    assert new['ue_route_overrides']==worker.ROUTE_OVERRIDE
    assert new['ues'][1:]==old['ues'][1:]
    without=lambda m:{k:v for k,v in m.items()if k not in ('ues','ue_route_overrides')}
    assert without(new)==without(old)
    route=new['ues'][0]['trajectory']
    assert route['times_s']==pytest.approx([0,5.6,11.22])
    assert np.asarray(route['positions_m'])==pytest.approx(np.array([[-11.4,-30,1.5],[-3,-30,1.5],[-11.43,-30,1.5]]))
    for u in new['ues']:
        p=np.asarray(u['trajectory']['positions_m']);t=np.asarray(u['trajectory']['times_s'])
        assert np.linalg.norm(np.diff(p,axis=0)/np.diff(t)[:,None],axis=1)==pytest.approx(1.5)
    old_configs['ue']['ues'][0].update({k:new_configs['ue']['ues'][0][k]for k in ('x_m','y_m','z_m')})
    assert new_configs==old_configs


def test_stage_admission_never_implies_on_admission_and_off_can_finish_first(plan):
    out = Path(plan['output_root'])
    out.mkdir(parents=True)
    (out/'ADMISSION_ALLOWED').touch()
    off, on = plan['cases']
    assert worker.on_admission(plan, off, out)['allowed']
    assert not worker.on_admission(plan, on, out)['allowed']
    started = []
    def execute(case):
        started.append(case['id'])
        return dict(id=case['id'], phase='complete')
    def publish(state):
        if state['done'] == [off['id']] and not state['running']:
            assert started == [off['id']]
            (out/'ON_ADMISSION_ALLOWED').touch()
    result = worker.schedule_cases(plan['cases'], max_parallel=2, execute=execute,
        admission=lambda c,n:worker.on_admission(plan,c,out), stop_reason=lambda:None,
        stop_event=Event(), stop_children=lambda:None, publish=publish, poll=.001)
    assert result['done'] == [off['id'],on['id']]
    (out/'ON_ADMISSION_ALLOWED').unlink()
    assert not worker.on_admission(plan,on,out)['allowed']

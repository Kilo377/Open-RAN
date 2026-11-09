"""Controller unit tests with explicitly delivered, delayed public reports."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

import ric.near_rt.pools.chongyu.ue_energy_rf as energy
from ric.near_rt.pools.chongyu.online_rf import OnlineEnergyRF
from ric.near_rt.pools.chongyu.ue_energy_rf import ACTIONS, FEATURES, UeEnergyRF


def controller(parameters=None, cells=None):
    cells = cells or ['cell_0']
    app = UeEnergyRF(cells, parameters=parameters)
    app.select(cells, {cell: 1 for cell in cells})
    return app


def observation(slot=1, *, uid='ue_0', action=1, owner=None, energy_j=1., bits=100000,
                cohorts=None, tasks=True, cell='cell_0'):
    user = NS(ue_id=uid, serving_cell_id=cell, queue_bytes=1000, hol_delay_ms=2.,
              throughput_mbps=1., offered_rate_bps=1e6, reference_sinr_db=10.,
              ue_energy_total_j=energy_j, delivered_bits_total=bits, offered_bits_total=bits,
              ue_task_metrics={'pending': 0} if tasks else None,
              ue_task_cohorts=cohorts if cohorts is not None else [],
              ue_service_policy=deepcopy(ACTIONS[action]),
              kpi_metadata={'ue_energy_total_j': {'produced_at_slot': slot}})
    return NS(slot_id=slot, ue_observations=[user],
              control_ownership={cell: {'ue_drx_configs': owner,
                                        'ue_offload_fraction': owner}})


def mature_cohort():
    return dict(index=2, generated=2, completed=2, on_time_completed=2,
                completion_latency_ms_sum=40., mature_after_s=.4)


def confirm_and_start(app, *, action=1):
    app.run(observation(201, action=action, owner=app.name), NS())
    assert app.pending['ue_0']['confirmed']
    assert app.pending['ue_0']['cohort'] == 2
    app.run(observation(400, action=action, owner=app.name, energy_j=1., bits=100000), NS())


def trained_fixture(tmp_path, monkeypatch):
    # Test-only one-leaf tree: exercises deployment loading, not RF training.
    monkeypatch.setattr(energy, 'ROOT', tmp_path)
    path = tmp_path / 'outputs' / 'fixture.json'
    model = OnlineEnergyRF(FEATURES, min_samples=2, trees=1)
    model.add([0.] * len(FEATURES), [.5, 1., 0., 25.])
    model.add([1.] * len(FEATURES), [.5, 1., 0., 25.])
    model.save(path)
    data = json.loads(path.read_text())
    data.update(trained_samples=2, fit_count=1, fit_parameters={'max_features': 1.0}, forest=[dict(children_left=[-1],
        children_right=[-1], feature=[-2], threshold=[-2.], value=[[.5, 1., 0., 25.]])])
    path.write_text(json.dumps(data))
    return path


def test_commands_are_cell_scoped_and_use_registered_identity():
    app = controller()
    result = app.run(observation(), NS())
    assert len(result.config_actions) == 1
    action = result.config_actions[0]
    assert action.target_cell_id == 'cell_0'
    assert action.source_name == app.name
    assert action.ue_drx_configs == {'ue_0': ACTIONS[1]['drx']}
    assert action.ue_offload_fraction == {'ue_0': .5}
    assert action._library_provenance == app.identity('cell_0')
    assert app.learner.sample_count == 0


def test_both_target_ues_in_same_cell_share_one_complete_transaction():
    app = controller()
    obs = observation()
    obs.ue_observations.append(observation(uid='ue_4').ue_observations[0])
    result = app.run(obs, NS())
    assert len(result.config_actions) == 1
    action = result.config_actions[0]
    assert set(action.ue_drx_configs) == set(action.ue_offload_fraction) == {'ue_0', 'ue_4'}
    assert app.pending['ue_0']['features'][0] == 0.
    assert app.pending['ue_4']['features'][0] == 1.


def test_task_cohorts_are_registered_public_model_telemetry():
    from ric.xapp_contract import KPI_REGISTRY
    spec = KPI_REGISTRY['ue_task_cohorts']
    assert spec.entity == 'ue'
    assert spec.unit == 'record'
    assert spec.source == 'paper:ue_task_cohorts:system_UE_power_DRX_MEC_model'


@pytest.mark.parametrize('owner', [None, 'another_xapp'])
def test_unconfirmed_or_overridden_action_times_out_and_retries_without_sample(owner):
    app = controller()
    app.run(observation(1), NS())
    # The request has not acquired ownership at the 200 ms boundary.
    before = app.run(observation(401, owner=owner), NS())
    assert not before.config_actions
    assert app.pending['ue_0']['requested_slot'] == 1
    assert not app.pending['ue_0']['confirmed']
    # Once the timeout has elapsed, retry rather than stall indefinitely.
    after = app.run(observation(402, owner=owner), NS())
    assert len(after.config_actions) == 1
    assert app.pending['ue_0']['requested_slot'] == 402
    assert not app.pending['ue_0']['confirmed']
    assert app.invalidated == 1
    assert app.learner.sample_count == 0
    assert not any(e['kind'] == 'measured_sample' for e in app.events)


def test_leaving_active_scope_discards_confirmed_inflight_sample():
    app = controller()
    app.run(observation(), NS())
    confirm_and_start(app)
    assert app.pending['ue_0']['confirmed']
    result = app.run(observation(800, cell='cell_1', owner=app.name,
                                 cohorts=[mature_cohort()]), NS())
    assert not result.config_actions
    assert 'ue_0' not in app.pending
    assert app.invalidated == 1 and app.learner.sample_count == 0
    assert any(e['kind'] == 'sample_invalidated' and e['reason'] == 'left_active_scope'
               for e in app.events)


def test_unconfirmed_or_overridden_commands_cannot_generate_training_labels():
    app = controller()
    app.run(observation(), NS())
    # Requested settings alone are insufficient: no executed ownership yet.
    app.run(observation(800, owner=None, cohorts=[mature_cohort()]), NS())
    assert not app.pending['ue_0']['confirmed']
    assert app.learner.sample_count == 0
    app.run(observation(801, action=0, owner=app.name, cohorts=[mature_cohort()]), NS())
    assert not app.pending['ue_0']['confirmed']
    assert app.learner.sample_count == 0


def test_only_mature_post_confirmation_cohort_produces_measured_sample():
    app = controller()
    app.run(observation(), NS())
    confirm_and_start(app)
    # A completed old cohort cannot label the newly executed action.
    old = dict(mature_cohort(), index=1, mature_after_s=.3)
    app.run(observation(600, owner=app.name, cohorts=[old]), NS())
    assert app.learner.sample_count == 0
    app.run(observation(799, owner=app.name, cohorts=[mature_cohort()]), NS())
    assert app.learner.sample_count == 0
    app.run(observation(800, owner=app.name, energy_j=1.2, bits=300000,
                        cohorts=[mature_cohort()]), NS())
    assert app.learner.sample_count == 1
    sample = next(e for e in app.events if e['kind'] == 'measured_sample')
    assert sample['targets'] == pytest.approx([1., 1., 0., 20.])
    assert sample['prediction_before_outcome'] is None
    assert sample['measurement_start_s'] == .2 and sample['end_s'] == .4
    assert sample['confirmation'] == 'executed_policy_and_ownership'
    assert app.windows['ue_0'] == 1


@pytest.mark.parametrize('uid', ['ue_1', 'ue_2', 'ue_6', 'ue_8'])
def test_non_target_and_low_latency_ues_are_never_controlled(uid):
    app = controller()
    assert not app.run(observation(uid=uid), NS()).config_actions
    assert not app.pending and app.learner.sample_count == 0


def test_missing_task_model_and_inactive_cells_are_not_controlled():
    app = controller()
    assert not app.run(observation(tasks=False), NS()).config_actions
    assert not app.run(observation(cell='cell_1'), NS()).config_actions
    user = observation().ue_observations[0]
    user.ue_task_cohorts = None
    assert not app.run(NS(slot_id=1, ue_observations=[user]), NS()).config_actions


@pytest.mark.parametrize('change', ['owner', 'policy', 'association'])
def test_confirmed_sample_is_invalidated_by_changed_execution(change):
    app = controller(cells=['cell_0', 'cell_1'])
    app.run(observation(), NS())
    confirm_and_start(app)
    obs = observation(700, owner=app.name)
    if change == 'owner':
        obs.control_ownership['cell_0']['ue_offload_fraction'] = 'another_xapp'
    elif change == 'policy':
        obs.ue_observations[0].ue_service_policy = deepcopy(ACTIONS[0])
    else:
        # Still an allowed active cell, so invalidation is actually observed.
        obs.ue_observations[0].serving_cell_id = 'cell_1'
        obs.control_ownership['cell_1'] = obs.control_ownership['cell_0']
    app.run(obs, NS())
    assert app.invalidated == 1
    assert app.learner.sample_count == 0
    assert any(e['kind'] == 'sample_invalidated' for e in app.events)


def test_reset_discards_inflight_labels_but_preserves_learning():
    app = controller()
    app.learner.add([0.] * len(FEATURES), [1., 1., 0., 20.])
    app.run(observation(), NS())
    assert app.pending
    app.reset()
    assert not app.pending and not app.last_decisions
    assert app.learner.sample_count == 1


def test_frozen_requires_existing_trained_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(energy, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='requires a trained model'):
        controller({'mode': 'frozen'})
    missing = tmp_path / 'outputs' / 'missing.json'
    with pytest.raises(ValueError, match='not trained'):
        controller({'mode': 'frozen', 'model_path': str(missing)})
    empty = tmp_path / 'outputs' / 'empty.json'
    OnlineEnergyRF(FEATURES).save(empty)
    with pytest.raises(ValueError, match='not trained'):
        controller({'mode': 'frozen', 'model_path': str(empty)})


def test_frozen_policy_uses_predictions_without_adding_samples_or_fitting(tmp_path, monkeypatch):
    path = trained_fixture(tmp_path, monkeypatch)
    app = controller({'mode': 'frozen', 'model_path': str(path)})
    def forbidden(*args, **kwargs):
        raise AssertionError('Frozen model attempted training')
    monkeypatch.setattr(app.learner, 'add', forbidden)
    monkeypatch.setattr(app.learner, 'fit', forbidden)
    result = app.run(observation(), NS())
    # Equal-power predictions now correctly retain the native baseline.
    assert result.config_actions[0].ue_offload_fraction == {'ue_0': .5}
    assert app.last_decisions[0]['reason'] == 'no_predicted_energy_gain_baseline'
    confirm_and_start(app, action=1)
    app.run(observation(800, action=1, owner=app.name, energy_j=1.2, bits=300000,
                        cohorts=[mature_cohort()]), NS())
    assert app.learner.sample_count == app.learner.trained_samples == 2
    assert app.learner.fit_count == 1
    assert app.windows['ue_0'] == 1
    assert app.learner.prediction_batches >= 2


def test_conservative_deployment_keeps_native_drx_but_still_uses_rf():
    app=controller()
    app.windows['ue_0']=9
    app.learner.predict=lambda rows:[[10.-i,1.,0.,25.] for i in range(len(rows))]
    chosen,_,reason=app._choose(observation().ue_observations[0])
    assert chosen==2 and reason=='rf_minimum_predicted_feasible_power'
    app.parameters['deploy_drx']=True
    assert app._choose(observation().ue_observations[0])[0]==8


def test_energy_controller_does_not_spend_more_than_predicted_baseline():
    app=controller()
    app.windows['ue_0']=9
    app.learner.predict=lambda rows:[[1.,1.,0.,25.] for _ in rows]
    chosen,_,reason=app._choose(observation().ue_observations[0])
    assert chosen==1 and reason=='no_predicted_energy_gain_baseline'

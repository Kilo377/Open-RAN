"""X-12 control/label invariants with delivered public report fixtures."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from models.action import ConfigAction
from phy.rt_sys.control import audit_action
import ric.near_rt.pools.chongyu.latency_scheduler_rf as latency
from ric.near_rt.pools.chongyu.latency_scheduler_rf import ACTIONS, WEIGHTS, FEATURES, LatencySchedulerRF
from ric.near_rt.pools.chongyu.online_rf import OnlineEnergyRF
from ric.near_rt.pools.chongyu.ue_energy_rf import ACTIONS as ENERGY_ACTIONS


def app(parameters=None, cells=None):
    cells = cells or ['cell_0']
    result = LatencySchedulerRF(cells, parameters=parameters)
    result.select(cells, {cid: 1 for cid in cells})
    return result


def obs(slot=1, *, uids=('ue_0',), cell='cell_0', mode='round_robin', bias=None,
        owner=None, cohorts=None, power_total=1., bits=100000, service=None, weight=1.):
    users = [NS(ue_id=uid, serving_cell_id=cell, queue_bytes=1000, hol_delay_ms=2.,
                throughput_mbps=1., offered_rate_bps=1e6, reference_sinr_db=10.,
                ue_energy_total_j=power_total, delivered_bits_total=bits, offered_bits_total=bits,
                ue_active_fraction=.8, ue_task_metrics={'pending': 0, 'offloaded_pending': 0},
                ue_task_cohorts=deepcopy(cohorts or []),
                ue_service_policy=deepcopy(service or ENERGY_ACTIONS[1]),
                kpi_metadata={'ue_energy_total_j': {'produced_at_slot': slot, 'valid': True}})
             for uid in uids]
    c = NS(cell_id=cell, total_queue_bytes=2000,
           executed_config=dict(scheduler_mode=mode, latency_bias=bias, ue_scheduler_weights={uid:weight for uid in uids}),
           kpi_metadata={'executed_config': {'produced_at_slot': slot, 'valid': True}})
    return NS(slot_id=slot, ue_observations=users, cell_observations=[c],
              control_ownership={cell: dict(scheduler_mode=owner, ue_scheduler_weights=owner)})


def cohort(**values):
    result = dict(index=2, generated=2, completed=1, on_time_completed=1, failed=0,
                  pending=1, completion_latency_ms_sum=40., mature_after_s=.4,
                  offloaded_generated=2, offloaded_completed=1,
                  offloaded_on_time_completed=1, offloaded_failed=0,
                  offloaded_pending=1, offloaded_completion_latency_ms_sum=40.,
                  mixed_policy=False, drx_config=deepcopy(ENERGY_ACTIONS[1]['drx']),
                  offload_fraction=.5)
    result.update(values)
    return result


def confirm(controller, action=0, uids=('ue_0',)):
    mode=ACTIONS[action]['scheduler_mode'];weight=ACTIONS[action]['weight']
    controller.run(obs(201, uids=uids, mode=mode, weight=weight, owner=controller.name), NS())
    assert controller.pending['cell_0']['confirmed']
    assert controller.pending['cell_0']['cohort'] == 2
    controller.run(obs(400, uids=uids, mode=mode, weight=weight, owner=controller.name), NS())


def artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(latency, 'ROOT', tmp_path)
    path = tmp_path / 'outputs' / 'latency.json'
    model = OnlineEnergyRF(FEATURES, min_samples=2, trees=1)
    for value in [0., 1.]:
        model.add([value] * len(FEATURES), [.5, 1., 0., 25.])
    model.save(path)
    data = json.loads(path.read_text())
    # A test-only inference fixture, not a fabricated deployment artifact.
    data.update(trained_samples=2, fit_count=1, fit_parameters={'max_features':1.0}, forest=[dict(children_left=[-1],
        children_right=[-1], feature=[-2], threshold=[-2.], value=[[.5, 1., 0., 25.]])])
    path.write_text(json.dumps(data))
    return path


def test_x12_mapping_is_supported_and_disjoint_from_x01():
    controller = app()
    decision = controller.run(obs(), NS())
    assert len(decision.config_actions) == 1
    action = decision.config_actions[0]
    assert action.scheduler_mode == 'round_robin' and action.ue_scheduler_weights == {'ue_0':1.}
    assert action.latency_bias is None
    assert action.target_cell_id == 'cell_0' and action.source_name == controller.name
    assert action._library_provenance == controller.identity('cell_0')
    assert action.ue_drx_configs is action.ue_offload_fraction is None
    assert {row['route'] for row in audit_action(action)} == {'scheduler'}
    assert not set(controller.writable_actions) & {'ue_drx_configs', 'ue_offload_fraction'}
    spec = json.loads((latency.Path(latency.__file__).with_name('x12.json')).read_text())
    assert spec['rcp']['ue_scheduler_weights']['values'] == list(WEIGHTS)
    assert spec['implementation'] == controller.implementation_id


@pytest.mark.parametrize('index',range(4))
def test_fixed_ablation_executes_declared_mode_and_weight_without_learning(index,monkeypatch):
    controller=app(dict(mode='fixed',fixed_action=index))
    def forbidden(*args,**kwargs):raise AssertionError('Fixed ablation must not fit or predict')
    monkeypatch.setattr(controller.learner,'fit',forbidden)
    monkeypatch.setattr(controller.learner,'predict',forbidden)
    first=controller.run(obs(),NS()).config_actions[0]
    assert first.scheduler_mode==ACTIONS[index]['scheduler_mode']
    assert first.ue_scheduler_weights=={'ue_0':ACTIONS[index]['weight']}
    confirm(controller,action=index)
    controller.run(obs(800,mode=ACTIONS[index]['scheduler_mode'],weight=ACTIONS[index]['weight'],
                       owner=controller.name,cohorts=[cohort()]),NS())
    assert controller.learner.sample_count==controller.learner.trained_samples==0
    assert controller.summary()['algorithm']=='FixedSchedulerAblation'
    assert controller.summary()['rf_selected_actions']==0
    assert controller.summary()['fixed_selected_actions']>=1
    assert all(e['reason']=='predeclared_fixed_scheduler_action' for e in controller.events if e['kind']=='decision')


@pytest.mark.parametrize('value',[-1,4,True,.5])
def test_fixed_ablation_rejects_invalid_action(value):
    with pytest.raises(ValueError,match='fixed scheduler'):
        app(dict(mode='fixed',fixed_action=value))


def test_two_target_ues_in_one_cell_have_one_real_scheduler_action():
    controller = app()
    result = controller.run(obs(uids=('ue_0', 'ue_4')), NS())
    assert len(result.config_actions) == 1
    assert controller.pending['cell_0']['ue_ids'] == ('ue_0', 'ue_4')
    assert controller.pending['cell_0']['features'][:2] == [1., 1.]


def test_non_target_ues_do_not_start_scheduler_control():
    controller = app()
    assert not controller.run(obs(uids=('ue_2', 'ue_6')), NS()).config_actions
    assert not controller.pending


def test_actual_scheduler_and_both_owners_required_before_sampling():
    controller = app()
    controller.run(obs(), NS())
    controller.run(obs(200, mode='latency_aware', weight=1., owner=controller.name), NS())
    assert not controller.pending['cell_0']['confirmed']
    report = obs(201, mode='round_robin', weight=1., owner=controller.name)
    report.control_ownership['cell_0']['ue_scheduler_weights'] = 'another'
    controller.run(report, NS())
    assert not controller.pending['cell_0']['confirmed']
    assert controller.learner.sample_count == 0



@pytest.mark.parametrize('weights', [None, {}, {'ue_0': .5}])
def test_execution_confirmation_requires_explicit_matching_target_weights(weights):
    controller = app()
    controller.run(obs(), NS())
    report = obs(201, owner=controller.name)
    report.cell_observations[0].executed_config['ue_scheduler_weights'] = weights
    controller.run(report, NS())
    assert not controller.pending['cell_0']['confirmed']
    assert controller.learner.sample_count == 0


def test_exploration_covers_native_rr_and_all_three_target_weights_twice():
    controller = app()
    report = obs()
    choices = []
    for count in range(8):
        controller.windows['cell_0'] = count
        index, _, reason = controller._choose('cell_0', report.ue_observations,
                                              report.cell_observations[0])
        choices.append(index)
        assert reason == 'balanced_online_exploration'
    assert choices == [0, 3, 1, 2, 0, 3, 1, 2]


def test_unconfirmed_request_retries_after_timeout_without_a_label():
    controller = app()
    controller.run(obs(), NS())
    assert not controller.run(obs(401), NS()).config_actions
    result = controller.run(obs(402), NS())
    assert result.config_actions
    assert controller.pending['cell_0']['requested_slot'] == 402
    assert controller.invalidated == 1 and controller.learner.sample_count == 0


def test_measured_label_is_task_e2e_and_pending_tasks_count_as_bad():
    controller = app()
    controller.run(obs(), NS())
    confirm(controller)
    controller.run(obs(799, mode='round_robin', weight=1., owner=controller.name,
                       cohorts=[cohort()]), NS())
    assert controller.learner.sample_count == 0
    controller.run(obs(800, mode='round_robin', weight=1., owner=controller.name,
                       cohorts=[cohort()], power_total=1.2, bits=300000), NS())
    sample = next(e for e in controller.events if e['kind'] == 'measured_sample')
    assert sample['targets'] == pytest.approx([1., 1., .5, 40.])
    assert sample['targets'][3] != 2.  # The HOL feature is not the task label.
    assert controller.learner.sample_count == 1
    assert controller.windows['cell_0'] == 1


def test_zero_completions_cannot_look_like_zero_latency_success():
    controller = app()
    controller.run(obs(), NS())
    confirm(controller)
    missing = cohort(completed=0, on_time_completed=0, pending=2,
                     completion_latency_ms_sum=0., offloaded_completed=0,
                     offloaded_on_time_completed=0, offloaded_pending=2,
                     offloaded_completion_latency_ms_sum=0.)
    controller.run(obs(800, mode='round_robin', weight=1., owner=controller.name,
                       cohorts=[missing], power_total=1.2, bits=300000), NS())
    sample = next(e for e in controller.events if e['kind'] == 'measured_sample')
    assert sample['targets'][2:] == [1., 100.]


@pytest.mark.parametrize('change', ['owner', 'scheduler', 'x01', 'mixed', 'generation_policy', 'membership', 'scope', 'counter'])
def test_interrupted_context_does_not_become_a_training_sample(change):
    controller = app()
    controller.run(obs(), NS())
    confirm(controller)
    report = obs(800, mode='round_robin', weight=1., owner=controller.name,
                 cohorts=[cohort()], power_total=1.2, bits=300000)
    if change == 'owner':
        report.control_ownership['cell_0']['scheduler_mode'] = 'another'
    elif change == 'scheduler':
        report.cell_observations[0].executed_config['ue_scheduler_weights']['ue_0'] = 2.
    elif change == 'x01':
        report.ue_observations[0].ue_service_policy['offload_fraction'] = 1.
    elif change == 'mixed':
        report.ue_observations[0].ue_task_cohorts[0]['mixed_policy'] = True
    elif change == 'generation_policy':
        report.ue_observations[0].ue_task_cohorts[0]['offload_fraction'] = 1.
    elif change == 'membership':
        report.ue_observations.append(obs(800, uids=('ue_4',)).ue_observations[0])
    elif change == 'scope':
        controller.select([], {})
    else:
        report.ue_observations[0].offered_bits_total = 0
    controller.run(report, NS())
    assert controller.invalidated == 1
    assert controller.learner.sample_count == 0
    assert not any(e['kind'] == 'measured_sample' for e in controller.events)


def test_spliced_or_future_reports_cannot_confirm_execution():
    controller = app()
    controller.run(obs(), NS())
    report = obs(201, mode='round_robin', weight=1., owner=controller.name)
    report.ue_observations[0].kpi_metadata['ue_energy_total_j']['produced_at_slot'] = 199
    controller.run(report, NS())
    assert not controller.pending['cell_0']['confirmed']
    report = obs(202, mode='round_robin', weight=1., owner=controller.name)
    report.cell_observations[0].kpi_metadata['executed_config']['produced_at_slot'] = 205
    controller.run(report, NS())
    assert not controller.pending['cell_0']['confirmed']


def test_frozen_requires_trained_latency_feature_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(latency, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='requires a trained model'):
        app({'mode': 'frozen'})
    with pytest.raises(ValueError, match='not trained'):
        app({'mode': 'frozen', 'model_path': str(tmp_path / 'outputs/missing.json')})
    path = tmp_path / 'outputs' / 'wrong.json'
    OnlineEnergyRF(['not_latency_features']).save(path)
    with pytest.raises(ValueError, match='schema mismatch'):
        app({'mode': 'frozen', 'model_path': str(path)})


def test_frozen_selection_penalizes_bad_tasks_and_never_fits(tmp_path, monkeypatch):
    path = artifact(tmp_path, monkeypatch)
    controller = app({'mode': 'frozen', 'model_path': str(path)})
    def forbidden(*args, **kwargs):
        raise AssertionError('Frozen latency policy attempted training')
    monkeypatch.setattr(controller.learner, 'add', forbidden)
    monkeypatch.setattr(controller.learner, 'fit', forbidden)
    # Weight .5 has the best completed-only latency, but worse risk score than 1.
    monkeypatch.setattr(controller.learner, 'predict', lambda rows: [[.6,1.,0.,30.],
                                                                  [.2, 1., .14, 10.],
                                                                  [.5, 1., 0., 20.],
                                                                  [.1, 1., .5, 1.]])
    result = controller.run(obs(), NS())
    assert result.config_actions[0].ue_scheduler_weights == {'ue_0':1.}
    assert result.config_actions[0].scheduler_mode == 'latency_aware'
    assert controller.last_decisions[0]['reason'] == 'rf_minimum_predicted_e2e_risk'
    confirm(controller,action=2)
    controller.run(obs(800, mode='latency_aware', weight=1., owner=controller.name,
                       cohorts=[cohort()], power_total=1.2, bits=300000), NS())
    assert controller.learner.sample_count == controller.learner.trained_samples == 2
    assert controller.learner.fit_count == 1 and controller.windows['cell_0'] == 1
    assert controller.predicted_choices >= 2


def test_no_feasible_prediction_abstains_with_native_rr(tmp_path, monkeypatch):
    path = artifact(tmp_path, monkeypatch)
    controller = app({'mode': 'frozen', 'model_path': str(path)})
    monkeypatch.setattr(controller.learner, 'predict', lambda rows: [[1., .1, 1., 100.]] * 4)
    action = controller.run(obs(), NS()).config_actions[0]
    assert action.scheduler_mode == 'round_robin' and action.ue_scheduler_weights == {'ue_0':1.}
    assert action.latency_bias is None
    assert controller.last_decisions[0]['reason'] == 'predicted_qos_guard_native_rr'


def test_small_predicted_gain_abstains_with_native_rr_and_neutral_weights(tmp_path, monkeypatch):
    path = artifact(tmp_path, monkeypatch)
    controller = app({'mode': 'frozen', 'model_path': str(path)})
    monkeypatch.setattr(controller.learner, 'predict', lambda rows: [[1., 1., 0., 20.],
        [1., 1., 0., 19.9], [1., 1., 0., 21.], [1., 1., 0., 22.]])
    action = controller.run(obs(), NS()).config_actions[0]
    assert action.scheduler_mode == 'round_robin'
    assert action.ue_scheduler_weights == {'ue_0': 1.}
    assert controller.predicted_choices == 0
    assert controller.last_decisions[0]['reason'] == 'no_predicted_latency_gain_native_rr'



def test_frozen_native_rr_abstention_still_records_real_rf_inference(tmp_path, monkeypatch):
    path = artifact(tmp_path, monkeypatch)
    controller = app({'mode': 'frozen', 'model_path': str(path)})
    before = controller.summary()
    # The artifact has identical constant predictions for all four arms, so
    # native RR is the RF's genuine abstention decision, with no training.
    action = controller.run(obs(), NS()).config_actions[0]
    after = controller.summary()
    assert action.scheduler_mode == 'round_robin'
    assert after['rf_selected_actions'] == 0
    assert after['prediction_batches'] == before['prediction_batches'] + 1
    assert after['samples'] == before['samples'] == 2
    assert after['trained_samples'] == before['trained_samples'] == 2
    assert after['fit_count'] == before['fit_count'] == 1



def local_success(index=2, **values):
    row = cohort(index=index, completed=2, on_time_completed=2, pending=0,
                 completion_latency_ms_sum=50., mature_after_s=(index + 2) * .1,
                 offloaded_generated=0, offloaded_completed=0,
                 offloaded_on_time_completed=0, offloaded_failed=0,
                 offloaded_pending=0, offloaded_completion_latency_ms_sum=0.)
    row.update(values)
    return row


def test_local_cohort_holds_same_action_then_routes_actual_offloaded_mean():
    controller = app()
    controller.run(obs(), NS())
    confirm(controller)
    local = local_success()
    remote = local_success(3, completion_latency_ms_sum=105.,
                           offloaded_generated=1, offloaded_completed=1,
                           offloaded_on_time_completed=1,
                           offloaded_completion_latency_ms_sum=80., mature_after_s=.6)
    # A telemetry row may already expose a future bucket; it is not yet a label.
    first = controller.run(obs(800, owner=controller.name, cohorts=[local, remote]), NS())
    assert not first.config_actions and controller.learner.sample_count == 0
    assert controller.pending['cell_0']['next_cohort'] == 3
    controller.run(obs(1000, owner=controller.name, cohorts=[local, remote]), NS())
    assert controller.learner.sample_count == 0
    controller.run(obs(1200, owner=controller.name, cohorts=[local, remote],
                       power_total=1.4, bits=500000), NS())
    sample = next(e for e in controller.events if e['kind'] == 'measured_sample')
    assert sample['targets'] == pytest.approx([1., 1., 0., 80.])
    assert sample['targets'][3] != (50. + 105.) / 4  # All-task mean.
    assert sample['offloaded_generated'] == sample['offloaded_completed'] == 1
    assert sample['all_task_generated'] == 4
    assert [c['index'] for c in sample['cohorts']] == [2, 3]
    assert all(c['ue_id'] == 'ue_0' for c in sample['cohorts'])
    assert controller.learner.sample_count == 1


@pytest.mark.parametrize('route', ['offloaded_failed', 'offloaded_pending', 'unassigned_failure'])
def test_failures_and_pending_are_not_discarded_without_offloaded_completion(route):
    controller = app()
    controller.run(obs(), NS())
    confirm(controller)
    row = local_success(completed=1, on_time_completed=1,
                        completion_latency_ms_sum=25., failed=1)
    if route != 'unassigned_failure':
        row['offloaded_generated'] = 1
        row[route] = 1
        if route == 'offloaded_pending':
            row.update(failed=0, pending=1)
    controller.run(obs(800, owner=controller.name, cohorts=[row],
                       power_total=1.2, bits=300000), NS())
    sample = next(e for e in controller.events if e['kind'] == 'measured_sample')
    assert sample['targets'][2:] == [.5, 100.]
    assert sample['offloaded_completed'] == 0
    assert controller.learner.sample_count == 1


def test_no_offloaded_demand_is_bounded_and_returns_native_rr_without_data():
    controller = app()
    controller.windows['cell_0'] = 1  # Deliberately test a non-native probe arm.
    controller.run(obs(), NS())
    confirm(controller, action=3)
    for index in range(2, 7):
        result = controller.run(obs(800 + (index - 2) * 200, mode='latency_aware',
            weight=2., owner=controller.name, cohorts=[local_success(index)]), NS())
        if index < 6:
            assert not result.config_actions
            assert controller.pending['cell_0']['action'] == 3
    assert result.config_actions[0].scheduler_mode == 'round_robin'
    assert result.config_actions[0].ue_scheduler_weights == {'ue_0': 1.}
    assert controller.learner.sample_count == 0
    assert controller.windows['cell_0'] == 1
    assert controller.summary()['no_offloaded_demand_windows'] == 1
    assert any(e['kind'] == 'missing_offloaded_demand' for e in controller.events)
    assert not any(e['kind'] == 'measured_sample' for e in controller.events)


def test_zero_offload_policy_abstains_and_feature_uses_actual_pending_offloaded():
    controller = app()
    policy = deepcopy(ENERGY_ACTIONS[1]); policy['offload_fraction'] = 0.
    report = obs(service=policy)
    report.ue_observations[0].ue_task_metrics.update(pending=999, offloaded_pending=0)
    result = controller.run(report, NS())
    assert result.config_actions[0].scheduler_mode == 'round_robin'
    assert controller.last_decisions[0]['reason'] == 'no_offloaded_demand_native_rr'
    report.ue_observations[0].ue_task_metrics['offloaded_pending'] = 2
    assert controller._features(report.ue_observations, report.cell_observations[0], 0)[7] == 2
    assert FEATURES[7] == 'pending_offloaded_tasks'


def test_old_all_task_feature_artifact_is_rejected_despite_same_dimension(tmp_path, monkeypatch):
    path = artifact(tmp_path, monkeypatch)
    data = json.loads(path.read_text())
    data['feature_names'][7] = 'pending_tasks'
    assert len(data['feature_names']) == len(FEATURES) == 15
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='schema mismatch'):
        app({'mode': 'frozen', 'model_path': str(path)})


def test_missing_actual_offloaded_counters_cannot_supply_an_all_task_label():
    controller = app()
    controller.run(obs(), NS())
    confirm(controller)
    row = cohort()
    del row['offloaded_completed']
    controller.run(obs(800, owner=controller.name, cohorts=[row]), NS())
    assert controller.learner.sample_count == 0
    assert any(e.get('reason') == 'missing_offloaded_task_counters' for e in controller.events)


def test_reset_discards_pending_windows_without_erasing_training():
    controller = app()
    controller.learner.add([0.] * len(FEATURES), [1., 1., 0., 20.])
    controller.run(obs(), NS())
    controller.reset()
    assert not controller.pending and not controller.last_decisions
    assert controller.learner.sample_count == 1


@pytest.mark.parametrize('parameters', [{'window_ms': 50}, {'explore_windows': 1},
    {'trees': 0}, {'min_samples': True}, {'bad_task_penalty_ms': float('nan')},
    {'max_task_bad_rate': 2}, {'unknown': 1}])
def test_invalid_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        app(parameters)


@pytest.mark.parametrize('setting', [.5, .7])
def test_x12_passes_configured_fraction_only_on_actual_measured_sample(monkeypatch, setting):
    controller = app({'max_features': setting})
    fits = []
    monkeypatch.setattr(controller.learner, 'fit', lambda **kwargs: fits.append(kwargs) or False)
    controller.run(obs(), NS()); confirm(controller)
    assert not fits
    controller.run(obs(800, owner=controller.name, cohorts=[cohort()],
                       power_total=1.2, bits=300000), NS())
    assert fits == [{'max_features': setting}]
    assert controller.learner.sample_count == 1
    assert controller.summary()['configured_fit_parameters'] == {'max_features': setting}
    assert controller.summary()['artifact_fit_parameters'] is None


def test_frozen_legacy_forest_is_not_mislabelled_by_new_x12_fit_default(tmp_path, monkeypatch):
    path = artifact(tmp_path, monkeypatch)
    data = json.loads(path.read_text()); data['version'] = 1; data.pop('fit_parameters')
    path.write_text(json.dumps(data)); original = path.read_bytes()
    controller = app({'mode': 'frozen', 'model_path': str(path)})
    controller.run(obs(), NS())
    assert controller.summary()['configured_fit_parameters'] == {'max_features': .5}
    assert controller.summary()['artifact_fit_parameters'] == {'max_features': 1.}
    assert path.read_bytes() == original


def test_real_resumed_fit_migrates_only_after_new_observed_outcome(tmp_path, monkeypatch):
    pytest.importorskip('sklearn')
    path = artifact(tmp_path, monkeypatch)
    data = json.loads(path.read_text()); data['version'] = 1; data.pop('fit_parameters')
    path.write_text(json.dumps(data)); original = path.read_bytes()
    controller = app({'mode': 'online', 'model_path': str(path)})
    assert controller.summary()['artifact_fit_parameters'] == {'max_features': 1.}
    controller.run(obs(), NS()); confirm(controller)
    assert path.read_bytes() == original and controller.learner.fit_count == 1
    controller.run(obs(800, owner=controller.name, cohorts=[cohort()],
                       power_total=1.2, bits=300000), NS())
    saved = json.loads(path.read_text())
    assert saved['version'] == 2 and saved['fit_parameters'] == {'max_features': .5}
    assert saved['fit_count'] == 2 and saved['trained_samples'] == len(saved['x']) == 3
    assert controller.summary()['artifact_fit_parameters'] == {'max_features': .5}


@pytest.mark.parametrize('setting', [True, False, 0, -.1, 1.01, float('nan'), float('inf'), '.5'])
def test_x12_rejects_invalid_max_features(setting):
    with pytest.raises(ValueError, match='max_features'):
        app({'max_features': setting})

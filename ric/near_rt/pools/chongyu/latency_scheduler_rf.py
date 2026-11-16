"""X-12 RF scheduler using measured task end-to-end outcomes.

The paper's w5G is a real per-target UE scheduler priority multiplier.
A native round-robin/weight-one arm permits abstaining from a family change. OnlineEnergyRF is reused unchanged because
its four target meanings are also useful here: power (audit), delivered/offered
ratio, all-task unsuccessful fraction, and offloaded completed-task E2E latency.
Only mature observed cohorts form labels; all-local successful cohorts are
accumulated under the same action rather than supplying a local-compute label.
"""
from collections import deque
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from models.action import ConfigAction, ControlDecision
from ric.near_rt.pools.base import PoolXApp
from .online_rf import OnlineEnergyRF


ROOT = Path(__file__).resolve().parents[4]
TARGETS = ('ue_0', 'ue_4')
WEIGHTS = (.5, 1., 2.)
ACTIONS = (dict(scheduler_mode='round_robin', weight=1.),
           dict(scheduler_mode='latency_aware', weight=.5),
           dict(scheduler_mode='latency_aware', weight=1.),
           dict(scheduler_mode='latency_aware', weight=2.))
BASELINE = 0
FEATURES = ['has_ue0', 'has_ue4', 'target_queue_kb', 'target_hol_max_ms',
            'target_goodput_mbps', 'target_offered_mbps', 'min_reference_sinr_db',
            'pending_offloaded_tasks', 'cell_queue_kb', 'mean_active_fraction',
            'mean_offload_fraction', 'mean_drx_cycle_ms',
            'mean_drx_inactivity_ms', 'scheduler_is_latency_aware', 'candidate_ue_weight']
FIELDS = ('scheduler_mode', 'ue_scheduler_weights')


class LatencySchedulerRF(PoolXApp):
    implementation_id = 'latency_scheduler_rf'
    category_id = 'latency_scheduler'
    description = 'X-12: online RF UE scheduling priority from measured task E2E latency'
    # Cohort/task requirements are target-specific: the other four UEs have no
    # task model. They are checked explicitly rather than rejecting all reports.
    required_kpis = ('queue_bytes', 'hol_delay_ms', 'throughput_mbps',
                     'offered_rate_bps', 'reference_sinr_db', 'delivered_bits_total',
                     'offered_bits_total', 'ue_energy_total_j', 'ue_service_policy',
                     'ue_active_fraction', 'executed_config')
    writable_actions = FIELDS
    min_action_ms = 100

    def validate_parameters(self, p):
        super().validate_parameters(p)
        allowed = {'mode', 'model_path', 'log_path', 'window_ms', 'explore_windows',
                   'min_samples', 'trees', 'seed', 'min_delivery_ratio',
                   'max_task_bad_rate', 'bad_task_penalty_ms', 'max_features', 'fixed_action', 'complete_scheduler_bundle'}
        if set(p) - allowed:
            raise ValueError('Unknown latency RF parameters: ' + str(sorted(set(p) - allowed)))
        if p.get('mode', 'online') not in ('online', 'frozen', 'fixed'):
            raise ValueError('Latency RF mode must be online, frozen or fixed')
        if type(p.get('complete_scheduler_bundle',False)) is not bool:
            raise ValueError('complete_scheduler_bundle must be boolean')
        if type(p.get('fixed_action',3)) is not int or not 0 <= p.get('fixed_action',3) < len(ACTIONS):
            raise ValueError('Invalid fixed scheduler action')
        if p.get('window_ms', 100) != 100:
            raise ValueError('Version 1 requires 100 ms task cohorts')
        for key, default, low, high in [('explore_windows', 8, 4, 100),
                ('min_samples', 12, 2, 512), ('trees', 32, 1, 128),
                ('seed', 1, 0, 2**31 - 1)]:
            value = p.get(key, default)
            if type(value) is not int or not low <= value <= high:
                raise ValueError('Invalid ' + key)
        for key, default, low, high in [('min_delivery_ratio', .65, 0, 1),
                ('max_task_bad_rate', .15, 0, 1), ('bad_task_penalty_ms', 100., 1., 1000.)]:
            value = p.get(key, default)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Invalid ' + key)
        max_features = p.get('max_features', .5)
        if type(max_features) not in (int, float) or not math.isfinite(max_features) or not 0 < max_features <= 1:
            raise ValueError('Invalid max_features')
        for key in ('model_path', 'log_path'):
            if p.get(key) is not None and not Path(p[key]).resolve().is_relative_to(ROOT / 'outputs'):
                raise ValueError('Latency RF artifacts must stay under project outputs')
        if p.get('mode') == 'frozen' and not p.get('model_path'):
            raise ValueError('Frozen latency RF requires a trained model')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = self.parameters
        self.mode = p.get('mode', 'online')
        self.model_path = Path(p['model_path']) if p.get('model_path') else None
        self.log_path = Path(p['log_path']) if p.get('log_path') else None
        self.learner = OnlineEnergyRF(FEATURES, min_samples=p.get('min_samples', 12),
                                     trees=p.get('trees', 32), seed=p.get('seed', 1))
        if self.model_path and self.model_path.is_file():
            self.learner = OnlineEnergyRF.load(self.model_path, FEATURES)
        if self.mode == 'frozen' and not self.learner.trained:
            raise ValueError('Frozen latency RF artifact is not trained')
        self.pending, self.windows = {}, {}
        self.no_demand_cells = set()
        self.no_demand_windows = 0
        self.events = deque(maxlen=512)
        self.prediction_errors = []
        self.predicted_choices = self.invalidated = 0
        self.fixed_choices = 0
        if self.mode == 'online' and self.learner.trained:
            self.windows = {cell: p.get('explore_windows', 8) for cell in self.allowed_cells}
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists():
                raise ValueError('Choose a fresh latency RF log')

    def _log(self, kind, **values):
        row = dict(kind=kind, mode=self.mode, **values)
        self.events.append(row)
        if self.log_path:
            with self.log_path.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')

    def _features(self, users, cell, index):
        n = len(users)
        return [float(any(u.ue_id == uid for u in users)) for uid in TARGETS] + [
            sum(u.queue_bytes for u in users) / 1000., max(u.hol_delay_ms for u in users),
            sum(u.throughput_mbps for u in users), sum(u.offered_rate_bps for u in users) / 1e6,
            min(u.reference_sinr_db for u in users),
            sum(u.ue_task_metrics['offloaded_pending'] for u in users),
            getattr(cell, 'total_queue_bytes', 0) / 1000.,
            sum(u.ue_active_fraction for u in users) / n,
            sum(u.ue_service_policy['offload_fraction'] for u in users) / n,
            sum(u.ue_service_policy['drx']['cycle_ms'] for u in users) / n,
            sum(u.ue_service_policy['drx']['inactivity_ms'] for u in users) / n,
            float(ACTIONS[index]['scheduler_mode']=='latency_aware'), ACTIONS[index]['weight']]

    def _choose(self, cid, users, cell):
        if self.mode == 'fixed':
            self.fixed_choices += 1
            return self.parameters.get('fixed_action',3), None, 'predeclared_fixed_scheduler_action'
        if cid in self.no_demand_cells:
            self.no_demand_cells.discard(cid)
            return BASELINE, None, 'no_offloaded_demand_native_rr'
        if all(u.ue_service_policy['offload_fraction'] == 0 and
               u.ue_task_metrics['offloaded_pending'] == 0 for u in users):
            return BASELINE, None, 'no_offloaded_demand_native_rr'
        count = self.windows.get(cid, 0)
        if self.mode == 'online' and count < self.parameters.get('explore_windows', 8):
            return (0, 3, 1, 2)[count % 4], None, 'balanced_online_exploration'
        predictions = self.learner.predict([self._features(users, cell, i) for i in range(len(ACTIONS))])
        if predictions is None:
            return BASELINE, None, 'untrained_native_rr'
        feasible = [i for i, v in enumerate(predictions)
                    if v[1] >= self.parameters.get('min_delivery_ratio', .65)
                    and v[2] <= self.parameters.get('max_task_bad_rate', .15)]
        if not feasible:
            self._log('predicted_qos_guard', cell_id=cid, predictions=predictions)
            return BASELINE, predictions[BASELINE], 'predicted_qos_guard_native_rr'
        penalty = self.parameters.get('bad_task_penalty_ms', 100.)
        index = min(feasible, key=lambda i: predictions[i][3] + penalty * predictions[i][2])
        def score(i):
            return predictions[i][3] + penalty * predictions[i][2]
        if index == BASELINE or score(index) > score(BASELINE) * .99:
            return BASELINE, predictions[BASELINE], 'no_predicted_latency_gain_native_rr'
        self.predicted_choices += 1
        return index, predictions[index], 'rf_minimum_predicted_e2e_risk'

    @staticmethod
    def _policies(users):
        return {u.ue_id: deepcopy(u.ue_service_policy) for u in users}

    def _invalidate(self, cid, reason, produced=None):
        self.pending.pop(cid, None)
        self.invalidated += 1
        self._log('sample_invalidated', cell_id=cid, produced_slot=produced, reason=reason)

    def _observe(self, cid, users, cell, produced, owners):
        current = self.pending.get(cid)
        if current is None:
            return
        ids = tuple(u.ue_id for u in users)
        if ids != current['ue_ids']:
            self._invalidate(cid, 'target_membership_changed', produced)
            return
        if self._policies(users) != current['service_policies']:
            self._invalidate(cid, 'x01_service_policy_changed', produced)
            return
        cfg = cell.executed_config
        expected = ACTIONS[current['action']]
        weights = cfg.get('ue_scheduler_weights')
        matches = (cfg.get('scheduler_mode') == expected['scheduler_mode']
                   and isinstance(weights, dict)
                   and all(uid in weights and weights[uid] == expected['weight'] for uid in ids))
        if expected['scheduler_mode'] == 'latency_aware' and cfg.get('latency_bias') not in (None, 1.):
            matches = False
        owned = all(owners.get(field) == self.name for field in FIELDS)
        if not matches or not owned:
            if current.get('confirmed') or produced - current['requested_slot'] > round(200 / self.dt):
                self._invalidate(cid, 'scheduler_execution_or_ownership_changed', produced)
            return
        now = produced * self.dt / 1000
        if not current.get('confirmed'):
            current.update(confirmed=True, cohort=math.floor(now / .1 + 1e-8) + 1,
                           next_cohort=math.floor(now / .1 + 1e-8) + 1, mature_cohorts=[])
            self._log('executed_policy_confirmed', cell_id=cid, action=current['action'],
                      produced_slot=produced, cohort=current['cohort'],
                      executed_config={key: cfg[key] for key in FIELDS}, owners={key: owners[key] for key in FIELDS})
        start = current['cohort'] * .1
        if now >= start and 'measurement_start_s' not in current:
            current.update(measurement_start_s=now,
                counters={u.ue_id: [u.ue_energy_total_j, u.delivered_bits_total, u.offered_bits_total] for u in users})
        # A local-only 25 ms completion is not a sample of the offloaded AR
        # objective. Keep the executed action and accumulate consecutive mature
        # buckets, without revisiting earlier buckets or looking at future ones.
        while True:
            index = current['next_cohort']
            cohorts = [next((c for c in u.ue_task_cohorts if c['index'] == index), None)
                       for u in users]
            if any(c is None or not c['generated'] for c in cohorts):
                if now > index * .1 + .6:
                    self._invalidate(cid, 'missing_task_cohort', produced)
                return
            for u, cohort in zip(users, cohorts):
                policy = current['service_policies'][u.ue_id]
                if cohort.get('mixed_policy') or cohort.get('drx_config') != policy['drx'] or cohort.get('offload_fraction') != policy['offload_fraction']:
                    self._invalidate(cid, 'mixed_or_changed_task_generation_policy', produced)
                    return
                if any(key not in cohort for key in (
                        'offloaded_generated', 'offloaded_completed',
                        'offloaded_on_time_completed', 'offloaded_failed',
                        'offloaded_pending', 'offloaded_completion_latency_ms_sum')):
                    self._invalidate(cid, 'missing_offloaded_task_counters', produced)
                    return
            if 'measurement_start_s' not in current or any(now + 1e-9 < c['mature_after_s'] for c in cohorts):
                return
            current['mature_cohorts'].extend(dict(deepcopy(c), ue_id=u.ue_id)
                                              for u, c in zip(users, cohorts))
            current['next_cohort'] += 1
            accumulated = current['mature_cohorts']
            generated = sum(c['generated'] for c in accumulated)
            on_time = sum(c['on_time_completed'] for c in accumulated)
            offloaded_generated = sum(c['offloaded_generated'] for c in accumulated)
            if offloaded_generated or generated > on_time:
                break
            self._log('held_without_offloaded_demand', cell_id=cid,
                      action=current['action'], produced_slot=produced, cohort=index)
            if now >= start + .6 - 1e-9:
                self._log('missing_offloaded_demand', cell_id=cid,
                          action=current['action'], produced_slot=produced,
                          mature_cohorts=deepcopy(accumulated))
                self.no_demand_windows += 1
                self.no_demand_cells.add(cid)
                self.pending.pop(cid)
                return
        cohorts = current['mature_cohorts']
        duration = now - current['measurement_start_s']
        if duration <= 0:
            return
        deltas = [[value - old for value, old in zip(
            [u.ue_energy_total_j, u.delivered_bits_total, u.offered_bits_total], current['counters'][u.ue_id])] for u in users]
        if any(v < -1e-9 for row in deltas for v in row):
            self._invalidate(cid, 'counter_reset', produced)
            return
        generated = sum(c['generated'] for c in cohorts)
        completed = sum(c['offloaded_completed'] for c in cohorts)
        on_time = sum(c['on_time_completed'] for c in cohorts)
        completed_mean = (sum(c['offloaded_completion_latency_ms_sum'] for c in cohorts) / completed
                          if completed else 100.)
        targets = [sum(d[0] for d in deltas) / duration,
                   min(2., sum(d[1] for d in deltas) / max(1., sum(d[2] for d in deltas))),
                   1 - on_time / generated, completed_mean]
        self._log('measured_sample', cell_id=cid, ue_ids=list(ids), action=current['action'],
                  features=current['features'], targets=targets, cohorts=deepcopy(cohorts),
                  objective='offloaded_task_e2e', offloaded_generated=offloaded_generated,
                  offloaded_completed=completed, all_task_generated=generated,
                  confirmation='executed_scheduler_policy_and_ownership',
                  measurement_start_s=current['measurement_start_s'], end_s=now,
                  prediction_before_outcome=current['prediction'])
        if current['prediction'] is not None:
            self.prediction_errors.append([abs(a - b) for a, b in zip(targets, current['prediction'])])
        if self.mode == 'online' and self.learner.sample_count < 512:
            self.learner.add(current['features'], targets)
        self.windows[cid] = self.windows.get(cid, 0) + 1
        self.pending.pop(cid)

    def run(self, observation, policy_snapshot):
        begun = time.perf_counter()
        self.last_decisions = []
        cells = {c.cell_id: c for c in observation.cell_observations if c.cell_id in self.active_cells}
        groups = {}
        for u in observation.ue_observations:
            if u.ue_id in TARGETS and u.serving_cell_id in cells and getattr(u, 'ue_task_metrics', None) is not None and getattr(u, 'ue_task_cohorts', None) is not None and 'offloaded_pending' in u.ue_task_metrics:
                groups.setdefault(u.serving_cell_id, []).append(u)
        for users in groups.values():
            users.sort(key=lambda u: u.ue_id)
        for cid in list(self.pending):
            if cid not in groups:
                self._invalidate(cid, 'left_active_scope_or_task_model_missing')
        before = self.learner.sample_count
        available = []
        for cid, users in sorted(groups.items()):
            cell = cells[cid]
            meta = cell.kpi_metadata.get('executed_config', {})
            produced = meta.get('produced_at_slot')
            if not isinstance(getattr(cell, 'executed_config', None), dict) or produced is None or produced > observation.slot_id or meta.get('valid') is False:
                continue
            # One delayed report must describe both the executed scheduler and
            # the counters used to score it; no new/old report splicing.
            if any(u.kpi_metadata.get('ue_energy_total_j', {}).get('produced_at_slot') != produced for u in users):
                continue
            owners = getattr(observation, 'control_ownership', {}).get(cid, {})
            self._observe(cid, users, cell, produced, owners)
            available.append((cid, users, cell))
        if self.mode == 'online' and self.learner.sample_count != before and self.learner.fit(max_features=self.parameters.get('max_features', .5)):
            if self.model_path:
                self.learner.save(self.model_path)
            self._log('rf_fit', samples=self.learner.trained_samples, fit_count=self.learner.fit_count)
        actions = []
        for cid, users, cell in available:
            if cid in self.pending:
                continue
            index, prediction, reason = self._choose(cid, users, cell)
            self.pending[cid] = dict(action=index, prediction=prediction,
                features=self._features(users, cell, index), ue_ids=tuple(u.ue_id for u in users),
                service_policies=self._policies(users), requested_slot=observation.slot_id, confirmed=False)
            weights={u.ue_id:ACTIONS[index]['weight'] for u in users}
            if self.parameters.get('complete_scheduler_bundle',False):
                # Non-target UEs already have neutral weight in this policy.
                # Spell it out for atomic whole-scheduler conflict arbitration.
                weights={u.ue_id:weights.get(u.ue_id,1.) for u in observation.ue_observations
                         if u.serving_cell_id==cid}
            action = ConfigAction(source_name=self.name, target_cell_id=cid,
                                  scheduler_mode=ACTIONS[index]['scheduler_mode'],
                                  ue_scheduler_weights=weights)
            action._library_provenance = self.identity(cid)
            actions.append(action)
            decision = dict(slot_id=observation.slot_id, action=index,
                            scheduler_mode=ACTIONS[index]['scheduler_mode'],
                            ue_scheduler_weight=ACTIONS[index]['weight'],ue_ids=[u.ue_id for u in users],
                            reason=reason, trained_samples=self.learner.trained_samples, **self.identity(cid))
            self.last_decisions.append(decision)
            self._log('decision', **decision)
        self.decision_seconds += time.perf_counter() - begun
        result = ControlDecision(config_actions=actions)
        if not actions:
            result.config_actions.clear()
        return result

    def reset(self):
        super().reset()
        self.pending.clear()
        self.no_demand_cells.clear()

    def summary(self):
        n = len(self.prediction_errors)
        return dict(mode=self.mode, algorithm='FixedSchedulerAblation' if self.mode == 'fixed' else 'RandomForestRegressor_batched_online',
                    fixed_action=self.parameters.get('fixed_action',3) if self.mode == 'fixed' else None,
                    fixed_selected_actions=self.fixed_choices,
                    objective='offloaded_completed_task_e2e_ms_plus_all_task_bad_rate_penalty',
                    serializer_target_mapping={'task_mean_latency_ms': 'offloaded_mean_completion_latency_ms'},
                    no_offloaded_demand_windows=self.no_demand_windows,
                    native_mapping='target_UE_priority_multiplier_with_native_RR_abstention',
                    serializer_target_schema=['energy_w', 'normalized_goodput', 'task_late_rate', 'task_mean_latency_ms'],
                    configured_fit_parameters={'max_features': self.parameters.get('max_features', .5)},
                    artifact_fit_parameters=deepcopy(self.learner.fit_parameters),
                    samples=self.learner.sample_count, trained_samples=self.learner.trained_samples,
                    fit_count=self.learner.fit_count, fit_s=self.learner.fit_s,
                    predict_s=self.learner.predict_s, decision_s=self.decision_seconds,
                    rf_selected_actions=self.predicted_choices,
                    prediction_batches=self.learner.prediction_batches, invalidated_samples=self.invalidated,
                    prequential_mae=[sum(e[i] for e in self.prediction_errors) / n for i in range(4)] if n else None,
                    windows=dict(self.windows), model_path=str(self.model_path) if self.model_path else None)

    def descriptor(self):
        return dict(super().descriptor(), learning=self.summary(),
                    target_ue_ids=list(TARGETS),
                    target_required_kpis=['ue_task_metrics', 'ue_task_cohorts'])

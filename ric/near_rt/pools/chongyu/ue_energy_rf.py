"""X-01: measured-outcome, batched-online RF control of UE DRX/offload.

Only delivered public reports are consumed. Predictions are never labels.
The controller waits for executed-policy confirmation and mature task cohorts.
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

FEATURES = ['is_ue4', 'queue_kb', 'hol_ms', 'goodput_mbps', 'offered_mbps',
            'reference_sinr_db', 'pending_tasks', 'cycle_ms', 'on_duration_ms',
            'inactivity_ms', 'offload_fraction']
PROFILES = [(40., 4., 8.), (20., 4., 1.), (80., 4., 1.)]
ACTIONS = [dict(drx=dict(enabled=True, cycle_ms=c, on_duration_ms=o,
                         inactivity_ms=i, wakeup_ms=.5, offset_ms=0.), offload_fraction=f)
           for c, o, i in PROFILES for f in (0., .5, 1.)]
BASELINE = 1
EXPLORATION = [1, 5, 6, 0, 8, 3, 2, 7, 4]
ROOT = Path(__file__).resolve().parents[4]


class UeEnergyRF(PoolXApp):
    implementation_id = 'ue_energy_rf'
    category_id = 'ue_energy_saving'
    description = 'X-01: online Random Forest UE energy saving with measured QoS feedback'
    required_kpis = ('ue_energy_total_j', 'ue_service_policy', 'queue_bytes',
                     'hol_delay_ms', 'throughput_mbps', 'reference_sinr_db',
                     'offered_rate_bps', 'delivered_bits_total', 'offered_bits_total')
    writable_actions = ('ue_drx_configs', 'ue_offload_fraction')
    min_action_ms = 100

    def validate_parameters(self, p):
        super().validate_parameters(p)
        unknown = set(p) - {'mode', 'model_path', 'log_path', 'window_ms', 'explore_windows',
                            'min_samples', 'trees', 'seed', 'min_delivery_ratio', 'max_task_bad_rate', 'deploy_drx'}
        if unknown:
            raise ValueError('Unknown energy RF parameters: ' + str(sorted(unknown)))
        if type(p.get('deploy_drx',False)) is not bool:raise ValueError('deploy_drx must be boolean')
        if p.get('mode', 'online') not in ('online', 'frozen'):
            raise ValueError('Energy RF mode must be online or frozen')
        if p.get('window_ms', 100) != 100:
            raise ValueError('Version 1 requires the declared 100 ms task cohorts')
        for key, default, low, high in [('explore_windows', 9, 9, 100), ('min_samples', 12, 2, 512), ('trees', 32, 1, 128), ('seed', 1, 0, 2**31-1)]:
            v = p.get(key, default)
            if type(v) is not int or not low <= v <= high:
                raise ValueError('Invalid ' + key)
        for key, default in [('min_delivery_ratio', .65), ('max_task_bad_rate', .15)]:
            v = p.get(key, default)
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError('Invalid ' + key)
        for key in ('model_path', 'log_path'):
            if p.get(key) is not None:
                path = Path(p[key]).resolve()
                if not path.is_relative_to(ROOT / 'outputs'):
                    raise ValueError('Energy artifacts must stay under project outputs')
        if p.get('mode') == 'frozen' and not p.get('model_path'):
            raise ValueError('Frozen RF requires a trained model')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = self.parameters
        self.mode = p.get('mode', 'online')
        self.model_path = Path(p['model_path']) if p.get('model_path') else None
        self.log_path = Path(p['log_path']) if p.get('log_path') else None
        self.learner = OnlineEnergyRF(FEATURES, min_samples=p.get('min_samples', 12),
                                     trees=p.get('trees', 32), seed=p.get('seed', 1))
        if self.model_path and self.model_path.exists():
            self.learner = OnlineEnergyRF.load(self.model_path, FEATURES)
        if self.mode == 'frozen' and not self.learner.trained:
            raise ValueError('Frozen RF artifact is not trained')
        self.pending = {}
        self.windows = {}
        self.events = deque(maxlen=512)
        self.prediction_errors = []
        self.predicted_choices = 0
        self.invalidated = 0
        if self.mode == 'online' and self.learner.trained:
            self.windows = {uid: p.get('explore_windows', 9) for uid in ('ue_0', 'ue_4')}
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists():
                raise ValueError('Choose a fresh online RF log')

    def _log(self, kind, **values):
        row = dict(kind=kind, mode=self.mode, **values)
        self.events.append(row)
        if self.log_path:
            with self.log_path.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')

    def _features(self, u, action):
        task = getattr(u, 'ue_task_metrics', None) or {}
        drx = action['drx']
        return [float(u.ue_id == 'ue_4'), u.queue_bytes / 1000., u.hol_delay_ms,
                u.throughput_mbps, u.offered_rate_bps / 1e6, u.reference_sinr_db,
                task.get('pending', 0), drx['cycle_ms'], drx['on_duration_ms'],
                drx['inactivity_ms'], action['offload_fraction']]

    def _choose(self, u):
        n = self.windows.get(u.ue_id, 0)
        if self.mode == 'online' and n < self.parameters.get('explore_windows', 9):
            return EXPLORATION[n % len(EXPLORATION)], None, 'balanced_online_exploration'
        values = self.learner.predict([self._features(u, a) for a in ACTIONS])
        if values is None:
            return BASELINE, None, 'untrained_baseline'
        delivery = self.parameters.get('min_delivery_ratio', .65)
        bad = self.parameters.get('max_task_bad_rate', .15)
        # Small pilot data is insufficient to authorize aggressive DRX in
        # deployment. Learn offload first; broader DRX remains an explicit opt-in.
        eligible=range(len(ACTIONS)) if self.parameters.get('deploy_drx',False) else range(3)
        feasible = [i for i in eligible if values[i][1] >= delivery and values[i][2] <= bad and values[i][3] <= 100.]
        if not feasible:
            self._log('predicted_qos_guard', ue_id=u.ue_id, predictions=values)
            return BASELINE, values[BASELINE], 'predicted_qos_guard_baseline'
        selected = min(feasible, key=lambda i: values[i][0])
        if values[selected][0] >= values[BASELINE][0] * .99:
            return BASELINE, values[BASELINE], 'no_predicted_energy_gain_baseline'
        self.predicted_choices += 1
        return selected, values[selected], 'rf_minimum_predicted_feasible_power'

    def _observe(self, u, produced, owner):
        uid = u.ue_id
        current = self.pending.get(uid)
        if not current:
            return
        expected = ACTIONS[current['action']]
        actual = u.ue_service_policy
        matches = actual['drx'] == expected['drx'] and actual['offload_fraction'] == expected['offload_fraction']
        if owner != self.name or u.serving_cell_id != current['cell'] or not matches:
            if current.get('confirmed') or produced - current.get('requested_slot', produced) > round(200 / self.dt):
                self.invalidated += 1
                self._log('sample_invalidated', ue_id=uid, produced_slot=produced,
                          reason='ownership_policy_or_association_changed')
                self.pending.pop(uid)
            return
        now = produced * self.dt / 1000
        if not current.get('confirmed'):
            # Align to a future cohort so no pre-action task enters its label.
            current.update(confirmed=True, cohort=math.floor(now / .1 + 1e-8) + 1)
            self._log('executed_policy_confirmed', ue_id=uid, action=current['action'],
                      produced_slot=produced, cohort=current['cohort'])
        cohort_start = current['cohort'] * .1
        if now >= cohort_start and 'energy_start' not in current:
            current.update(energy_start=u.ue_energy_total_j, bits_start=u.delivered_bits_total, offered_start=u.offered_bits_total,
                           measurement_start_s=now)
        cohorts = getattr(u, 'ue_task_cohorts', None) or []
        cohort = next((c for c in cohorts if c['index'] == current['cohort']), None)
        if not cohort or now + 1e-9 < cohort['mature_after_s'] or 'energy_start' not in current or not cohort['generated']:
            return
        if cohort.get('mixed_policy', False):
            self.invalidated += 1
            self.pending.pop(uid)
            self._log('sample_invalidated', ue_id=uid, reason='mixed_task_generation_policy')
            return
        duration = now - current['measurement_start_s']
        if duration <= 0:
            return
        power = (u.ue_energy_total_j - current['energy_start']) / duration
        offered = max(u.offered_bits_total - current['offered_start'], 1.)
        ratio = max(0., (u.delivered_bits_total - current['bits_start']) / offered)
        late = 1 - cohort['on_time_completed'] / cohort['generated']
        latency = cohort['completion_latency_ms_sum'] / cohort['completed'] if cohort['completed'] else 100.
        targets = [power, min(2., ratio), late, latency]
        self._log('measured_sample', ue_id=uid, action=current['action'], features=current['features'],
                  targets=targets, cohort=cohort, confirmation='executed_policy_and_ownership',
                  measurement_start_s=current['measurement_start_s'], end_s=now,
                  prediction_before_outcome=current['prediction'])
        if current['prediction'] is not None:
            self.prediction_errors.append([abs(a-b) for a, b in zip(targets, current['prediction'])])
        if self.mode == 'online' and self.learner.sample_count < 512:
            self.learner.add(current['features'], targets)
        self.windows[uid] = self.windows.get(uid, 0) + 1
        self.pending.pop(uid)

    def run(self, observation, policy_snapshot):
        started = time.perf_counter()
        self.last_decisions = []
        before = self.learner.sample_count
        users = []
        visible={u.ue_id for u in observation.ue_observations if u.serving_cell_id in self.active_cells}
        for uid in list(self.pending):
            if uid not in visible:
                self.pending.pop(uid);self.invalidated += 1
                self._log('sample_invalidated',ue_id=uid,reason='left_active_scope')
        for u in observation.ue_observations:
            if u.ue_id not in ('ue_0', 'ue_4') or u.serving_cell_id not in self.active_cells:
                continue
            # Non-task UEs have no task model and are never controlled here.
            if getattr(u, 'ue_task_metrics', None) is None or getattr(u, 'ue_task_cohorts', None) is None:
                continue
            meta = u.kpi_metadata['ue_energy_total_j']
            ownership = observation.control_ownership.get(u.serving_cell_id, {})
            offload_owner = ownership.get('ue_offload_fraction_by_ue', {}).get(u.ue_id, ownership.get('ue_offload_fraction'))
            owner = offload_owner if ownership.get('ue_drx_configs') == self.name else None
            self._observe(u, meta['produced_at_slot'], owner)
            users.append(u)
        if self.mode == 'online' and self.learner.sample_count != before:
            if self.learner.fit():
                if self.model_path:
                    self.model_path.parent.mkdir(parents=True, exist_ok=True)
                    self.learner.save(self.model_path)
                self._log('rf_fit', samples=self.learner.trained_samples, fit_count=self.learner.fit_count)
        actions = {}
        for u in users:
            if u.ue_id in self.pending:
                continue
            index, prediction, reason = self._choose(u)
            self.pending[u.ue_id] = dict(action=index, prediction=prediction,
                features=self._features(u, ACTIONS[index]), cell=u.serving_cell_id, confirmed=False, requested_slot=observation.slot_id)
            action = actions.setdefault(u.serving_cell_id, ConfigAction(source_name=self.name,
                target_cell_id=u.serving_cell_id, ue_drx_configs={}, ue_offload_fraction={}))
            action.ue_drx_configs[u.ue_id] = deepcopy(ACTIONS[index]['drx'])
            action.ue_offload_fraction[u.ue_id] = ACTIONS[index]['offload_fraction']
            self.last_decisions.append(dict(slot_id=observation.slot_id, ue_id=u.ue_id,
                action=index, reason=reason, trained_samples=self.learner.trained_samples,
                **self.identity(u.serving_cell_id)))
            self._log('decision', **self.last_decisions[-1])
        for action in actions.values():
            action._library_provenance = self.identity(action.target_cell_id)
        self.decision_seconds += time.perf_counter() - started
        decision=ControlDecision(config_actions=list(actions.values()))
        if not actions:decision.config_actions.clear()
        return decision

    def reset(self):
        super().reset()
        self.pending.clear()

    def summary(self):
        n = len(self.prediction_errors)
        return dict(mode=self.mode, algorithm='RandomForestRegressor_batched_online',
                    samples=self.learner.sample_count, trained_samples=self.learner.trained_samples,
                    fit_count=self.learner.fit_count, fit_s=self.learner.fit_s,
                    predict_s=self.learner.predict_s, decision_s=self.decision_seconds,
                    rf_selected_actions=self.predicted_choices, invalidated_samples=self.invalidated,
                    prequential_mae=[sum(e[i] for e in self.prediction_errors)/n for i in range(4)] if n else None,
                    windows=dict(self.windows), model_path=str(self.model_path) if self.model_path else None)

    def descriptor(self):
        return dict(super().descriptor(), learning=self.summary())

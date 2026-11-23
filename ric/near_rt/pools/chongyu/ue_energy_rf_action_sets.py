"""X-01: measured-outcome, batched-online RF control of UE DRX/offload.

Only delivered public reports are consumed. Predictions are never labels.
The controller waits for executed-policy confirmation and mature task cohorts.
"""
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from models.action import ConfigAction, ControlDecision
from ric.near_rt.pools.base import PoolXApp
from .online_rf import OnlineEnergyRF, TARGET_NAMES

FEATURES = ['is_ue4', 'queue_kb', 'hol_ms', 'goodput_mbps', 'offered_mbps',
            'reference_sinr_db', 'pending_tasks', 'cycle_ms', 'on_duration_ms',
            'inactivity_ms', 'offload_fraction']
PROFILES = [(40., 4., 8.), (20., 4., 1.), (80., 4., 1.)]
EXTENDED_PROFILES = PROFILES + [(40., 4., 0.), (80., 4., 0.), (160., 4., 0.),
                                (80., 1., 0.), (160., 1., 0.)]
BALANCED_PROFILES = EXTENDED_PROFILES + [(10., 8., 0.), (10., 6., 0.),
    (20., 10., 0.), (20., 8., 0.), (40., 20., 0.), (40., 10., 0.)]
ACTION_SET_VERSIONS = ('legacy9', 'extended24', 'balanced42')
BALANCED42_EXPLORATION_SCHEME = 'sha256-json-seed-uid-cycle-arm-v1'
ACTIONS = [dict(drx=dict(enabled=True, cycle_ms=c, on_duration_ms=o,
                         inactivity_ms=i, wakeup_ms=.5, offset_ms=0.), offload_fraction=f)
           for c, o, i in PROFILES for f in (0., .5, 1.)]
BASELINE = 1
EXPLORATION = [1, 5, 6, 0, 8, 3, 2, 7, 4]
ROOT = Path(__file__).resolve().parents[4]


def actions_for_version(version='legacy9'):
    """Return an instance-owned catalogue; legacy indices keep their meaning."""
    if not isinstance(version, str) or version not in ACTION_SET_VERSIONS:
        raise ValueError('Invalid action_set_version')
    profiles = {'legacy9': PROFILES, 'extended24': EXTENDED_PROFILES,
                'balanced42': BALANCED_PROFILES}[version]
    return [dict(drx=dict(enabled=True, cycle_ms=c, on_duration_ms=o,
                          inactivity_ms=i, wakeup_ms=.5, offset_ms=0.), offload_fraction=f)
            for c, o, i in profiles for f in (0., .5, 1.)]


def balanced42_exploration_order(seed, uid, cycle):
    """A cross-version deterministic permutation; never touches an RNG."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Invalid balanced exploration seed')
    if uid not in ('ue_0', 'ue_4') or type(cycle) is not int or cycle < 0:
        raise ValueError('Invalid balanced exploration UE or cycle')
    def key(arm):
        raw = json.dumps([BALANCED42_EXPLORATION_SCHEME, seed, uid, cycle, arm],
                         separators=(',', ':'), allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest(), arm
    return sorted(range(42), key=key)


def balanced42_schedule(seed, windows=84):
    """Full prospective per-UE schedule, indexed only by valid label count."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Invalid balanced exploration seed')
    if type(windows) is not int or not 0 <= windows <= 100:
        raise ValueError('Invalid balanced exploration windows')
    return {uid: [arm for cycle in range((windows + 41) // 42)
                  for arm in balanced42_exploration_order(seed, uid, cycle)][:windows]
            for uid in ('ue_0', 'ue_4')}


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
                            'balanced_start_windows', 'min_samples', 'trees', 'seed',
                            'min_delivery_ratio', 'max_task_bad_rate', 'deploy_drx',
                            'action_set_version'}
        if unknown:
            raise ValueError('Unknown energy RF parameters: ' + str(sorted(unknown)))
        if type(p.get('deploy_drx',False)) is not bool:raise ValueError('deploy_drx must be boolean')
        if not isinstance(p.get('action_set_version', 'legacy9'), str) or p.get('action_set_version', 'legacy9') not in ACTION_SET_VERSIONS:
            raise ValueError('Invalid action_set_version')
        if p.get('mode', 'online') not in ('online', 'frozen'):
            raise ValueError('Energy RF mode must be online or frozen')
        if p.get('window_ms', 100) != 100:
            raise ValueError('Version 1 requires the declared 100 ms task cohorts')
        for key, default, low, high in [('explore_windows', 9, 9, 100),
                ('balanced_start_windows', 0, 0, 100), ('min_samples', 12, 2, 512),
                ('trees', 32, 1, 128), ('seed', 1, 0, 2**31-1)]:
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
        self.action_set_version = p.get('action_set_version', 'legacy9')
        self.actions = actions_for_version(self.action_set_version)
        self.exploration = list(EXPLORATION) + list(range(len(ACTIONS), len(self.actions)))
        self.action_catalogue_sha256 = hashlib.sha256(json.dumps(self.actions,
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.balanced_exploration_metadata = {}
        if self.mode == 'online' and self.action_set_version == 'balanced42':
            schedule = balanced42_schedule(p.get('seed', 1),
                max(84, p.get('balanced_start_windows', 0), p.get('explore_windows', 9)))
            self.balanced_exploration_metadata = dict(
                balanced_exploration_scheme=BALANCED42_EXPLORATION_SCHEME,
                balanced_exploration_seed=p.get('seed', 1),
                balanced_exploration_cycle_size=42,
                balanced_exploration_schedule=schedule,
                balanced_exploration_schedule_sha256=hashlib.sha256(json.dumps(schedule,
                    sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest())
        self.description = type(self).description + f' [{self.action_set_version}; {len(self.actions)} actions]'
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
        # Episode-local measured progress is deliberately not loaded with a
        # resumed RF artifact. Invalidated/request-only windows do not advance
        # this schedule; each UE independently completes its declared cycle.
        self.balanced_windows = {}
        self.measured_action_counts = {}
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
        self._log('action_catalogue', action_catalogue=deepcopy(self.actions),
                  feature_names=list(FEATURES), target_names=list(TARGET_NAMES),
                  exploration_order=None if self.balanced_exploration_metadata else list(self.exploration), baseline_action=BASELINE,
                  **deepcopy(self.balanced_exploration_metadata))

    def _log(self, kind, **values):
        row = dict(kind=kind, mode=self.mode, **values)
        row.update(action_set_version=self.action_set_version, action_count=len(self.actions),
                   action_catalogue_sha256=self.action_catalogue_sha256)
        row.update({key: value for key, value in self.balanced_exploration_metadata.items()
                    if key != 'balanced_exploration_schedule'})
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
        balanced = self.balanced_windows.get(u.ue_id, 0)
        if self.mode == 'online' and balanced < self.parameters.get('balanced_start_windows', 0):
            index = (balanced42_exploration_order(self.parameters.get('seed', 1), u.ue_id, balanced // 42)[balanced % 42]
                     if self.action_set_version == 'balanced42' else self.exploration[balanced % len(self.exploration)])
            return index, None, 'episode_balanced_exploration'
        n = self.windows.get(u.ue_id, 0)
        if self.mode == 'online' and n < self.parameters.get('explore_windows', 9):
            index = (balanced42_exploration_order(self.parameters.get('seed', 1), u.ue_id, n // 42)[n % 42]
                     if self.action_set_version == 'balanced42' else self.exploration[n % len(self.exploration)])
            return index, None, 'balanced_online_exploration'
        values = self.learner.predict([self._features(u, a) for a in self.actions])
        if values is None:
            return BASELINE, None, 'untrained_baseline'
        delivery = self.parameters.get('min_delivery_ratio', .65)
        bad = self.parameters.get('max_task_bad_rate', .15)
        # Small pilot data is insufficient to authorize aggressive DRX in
        # deployment. Learn offload first; broader DRX remains an explicit opt-in.
        eligible=range(len(self.actions)) if self.parameters.get('deploy_drx',False) else range(3)
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
        expected = self.actions[current['action']]
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
        counts = self.measured_action_counts.setdefault(uid, {})
        counts[current['action']] = counts.get(current['action'], 0) + 1
        if current.get('balanced_start'):
            self.balanced_windows[uid] = self.balanced_windows.get(uid, 0) + 1
        self.windows[uid] = self.windows.get(uid, 0) + 1
        self.pending.pop(uid)

    def _save_fit_checkpoint(self):
        """Publish an immutable per-fit artifact, independently of latest.json."""
        directory = self.model_path.parent / 'checkpoints'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'fit-{self.learner.fit_count:05d}-samples-{self.learner.trained_samples:04d}.json'
        temporary = directory / ('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            self.learner.save(temporary)
            # Same-directory hard-link publication is atomic and refuses an
            # existing destination, unlike the rolling model's os.replace.
            os.link(temporary, path)
            path.chmod(0o444)
        finally:
            temporary.unlink(missing_ok=True)
        return dict(checkpoint_path=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

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
                checkpoint = {}
                if self.model_path:
                    self.model_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint = self._save_fit_checkpoint()
                    self.learner.save(self.model_path)
                self._log('rf_fit', samples=self.learner.trained_samples, fit_count=self.learner.fit_count, **checkpoint)
        actions = {}
        for u in users:
            if u.ue_id in self.pending:
                continue
            index, prediction, reason = self._choose(u)
            self.pending[u.ue_id] = dict(action=index, prediction=prediction,
                balanced_start=reason == 'episode_balanced_exploration',
                features=self._features(u, self.actions[index]), cell=u.serving_cell_id, confirmed=False, requested_slot=observation.slot_id)
            action = actions.setdefault(u.serving_cell_id, ConfigAction(source_name=self.name,
                target_cell_id=u.serving_cell_id, ue_drx_configs={}, ue_offload_fraction={}))
            action.ue_drx_configs[u.ue_id] = deepcopy(self.actions[index]['drx'])
            action.ue_offload_fraction[u.ue_id] = self.actions[index]['offload_fraction']
            self.last_decisions.append(dict(slot_id=observation.slot_id, ue_id=u.ue_id,
                action=index, reason=reason, trained_samples=self.learner.trained_samples,
                action_set_version=self.action_set_version, action_count=len(self.actions),
                action_catalogue_sha256=self.action_catalogue_sha256,
                action_parameters=deepcopy(self.actions[index]),
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
        return dict(mode=self.mode, deploy_drx=self.parameters.get('deploy_drx',False), algorithm='RandomForestRegressor_batched_online',
                    **deepcopy(self.balanced_exploration_metadata),
                    action_set_version=self.action_set_version, action_count=len(self.actions),
                    action_catalogue=deepcopy(self.actions), action_catalogue_sha256=self.action_catalogue_sha256,
                    exploration_order=None if self.balanced_exploration_metadata else list(self.exploration),
                    balanced_start_windows=self.parameters.get('balanced_start_windows',0),
                    balanced_completed_windows=dict(self.balanced_windows),
                    measured_action_counts=deepcopy(self.measured_action_counts),
                    samples=self.learner.sample_count, trained_samples=self.learner.trained_samples,
                    fit_count=self.learner.fit_count, fit_s=self.learner.fit_s,
                    predict_s=self.learner.predict_s, decision_s=self.decision_seconds,
                    rf_selected_actions=self.predicted_choices, invalidated_samples=self.invalidated,
                    prequential_mae=[sum(e[i] for e in self.prediction_errors)/n for i in range(4)] if n else None,
                    windows=dict(self.windows), model_path=str(self.model_path) if self.model_path else None)

    def descriptor(self):
        return dict(super().descriptor(), learning=self.summary())

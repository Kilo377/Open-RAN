"""Opt-in QoS/PDB action family for the measured URLLC RF xApp.

The MAC updates deadline urgency every slot; this xApp selects a configuration
after executed ownership confirmation, 20 ms wash-in and a 100/200 ms outcome
window. Labels include carryover and are not arrival-cohort SLA probabilities.
RF training never sees private channel truth or packet-ID future outcomes.
"""
from collections import deque
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import random
import uuid

from models.action import ControlDecision
from ric.near_rt.pools.base import PoolXApp
from .latency_urllc_rf import LatencyUrllcRF, ROOT, URLLC, PACKET_COUNTERS
from .online_urllc_qos_rf import OnlineUrllcQosRF, TARGETS, MIN_ROWS, MAX_ROWS

ACTIONS = ('native_rr', 'qos_urllc_1', 'qos_urllc_2', 'qos_urllc_4')
FEATURES = ['network_offered_mbps', 'network_reported_goodput_mbps',
    'network_queue_in_offered_ms', 'backlogged_ue_fraction']
for _uid in URLLC:
    FEATURES += [_uid+'_'+name for name in ('hol_over_pdb', 'queue_over_deadline_arrivals',
        'offered_mbps', 'reference_sinr_db', 'goodput_over_offered',
        'retx_fraction', 'retx_report_available')]
FEATURES += ['action_'+str(i) for i in range(4)]
EPSILON = .1
MIN_ROWS_PER_ARM = 10
REFIT_ROWS = 20


def action_map(users, index):
    if type(index) is not int or not 0 <= index < 4:
        raise ValueError('Invalid QoS URLLC RF arm')
    result = {}
    for u in users:
        values = result.setdefault(u.serving_cell_id,
            dict(scheduler_mode='round_robin' if index == 0 else 'qos_deadline', ue_scheduler_weights={}))
        values['ue_scheduler_weights'][u.ue_id] = (1., 1., 2., 4.)[index] if u.ue_id in URLLC else 1.
    return result


class LatencyUrllcQosRF(LatencyUrllcRF):
    description = 'Opt-in X-12 URLLC QoS/PDB RF; both UE2 and UE6 deadline outcomes'
    required_kpis = LatencyUrllcRF.required_kpis + ('hol_delay_ms',)
    action_names = ACTIONS
    target_ue_ids = URLLC
    monitored_peer_ue_ids = ()
    non_rr_scheduler_mode = 'qos_deadline'
    target_names = TARGETS
    max_rows = MAX_ROWS
    _action_map = staticmethod(action_map)

    def validate_parameters(self, p):
        if p.get('policy_version') != 'qos_v2':
            raise ValueError('Explicit policy_version=qos_v2 required')
        balanced = p.get('balanced_start_windows', 0)
        if type(balanced) is not int or not 0 <= balanced <= 80 or balanced % 4:
            raise ValueError('balanced_start_windows must be a multiple of four between zero and 80')
        if balanced and p.get('mode', 'online') != 'online':
            raise ValueError('balanced_start_windows requires online mode')
        base = dict(p); base.pop('policy_version'); base.pop('balanced_start_windows', None)
        super().validate_parameters(base)

    def __init__(self, *args, **kwargs):
        PoolXApp.__init__(self, *args, **kwargs)
        self.mode = self.parameters.get('mode', 'online')
        self.window_ms = self.parameters.get('window_ms', 100)
        self.model_path = Path(self.parameters['model_path']) if self.parameters.get('model_path') else None
        self.log_path = Path(self.parameters['log_path']) if self.parameters.get('log_path') else None
        self.learner = OnlineUrllcQosRF(FEATURES, self.parameters.get('seed', 179))
        if self.model_path and self.model_path.is_file():
            self.learner = OnlineUrllcQosRF.load(self.model_path, FEATURES)
        if any(sum(x[-4:]) != 1 or any(v not in (0., 1.) for v in x[-4:]) for x in self.learner._x):
            raise ValueError('Invalid persisted QoS URLLC action encoding')
        if self.mode == 'frozen' and (not self.learner.trained or not self._coverage(self.learner.trained_samples)):
            raise ValueError('Frozen QoS URLLC RF requires 40 fitted measured rows and ten per arm')
        self.balanced_start_windows = self.parameters.get('balanced_start_windows', 0)
        self.run_start_samples = self.learner.sample_count
        self.run_start_model_sha256 = self.learner.artifact_sha256
        if self.mode == 'online' and self.balanced_start_windows > MAX_ROWS-self.run_start_samples:
            raise ValueError('Insufficient measured-row capacity for requested balanced run start')
        self.pending = None
        self.events = deque(maxlen=256)
        self.invalidated = self.rf_selected_actions = self.rules_only_actions = self.fixed_actions = 0
        self.online_exploration_actions = 0
        self.status, self.status_reason = 'not_evaluable', 'no_valid_public_report'
        self.last_prediction_audit = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists():
                raise ValueError('Choose a fresh QoS URLLC RF event log')
        if self.balanced_start_windows:
            self._log('run_balanced_start', starting_measured_rows=self.run_start_samples,
                starting_model_sha256=self.run_start_model_sha256,
                requested_new_windows=self.balanced_start_windows,
                requested_new_windows_per_action=self.balanced_start_windows//4,
                counts=self._balanced_start_counts())

    def _balanced_start_counts(self):
        rows = self.learner._x[self.run_start_samples:self.run_start_samples+self.balanced_start_windows]
        return {str(i): sum(x[-4+i] == 1. for x in rows) for i in range(4)}

    def _coverage(self, limit=None):
        n = self.learner.sample_count if limit is None else limit
        return n >= MIN_ROWS and all(v >= MIN_ROWS_PER_ARM for v in self.action_samples(limit).values())

    def _features(self, users, index):
        by_id = {u.ue_id: u for u in users}
        offered = sum(u.offered_rate_bps for u in users)
        values = [offered/1e6, sum(u.throughput_mbps for u in users),
            min(10000., 8000*sum(u.queue_bytes for u in users)/max(offered, 1.)),
            sum(u.queue_bytes > 0 for u in users)/len(users)]
        for uid in URLLC:
            u = by_id[uid]
            retx = getattr(u, 'retx_fraction', None)
            values += [min(100., u.hol_delay_ms/u.latency_budget_ms),
                min(1000., u.queue_bytes*8/max(u.offered_rate_bps*u.latency_budget_ms/1000, 1.)),
                u.offered_rate_bps/1e6, u.reference_sinr_db,
                min(100., u.throughput_mbps*1e6/max(u.offered_rate_bps, 1.)),
                retx if retx is not None else 0., float(retx is not None)]
        return values + [float(i == index) for i in range(4)]

    def _choose(self, users):
        self.last_prediction_audit = None
        if self.mode == 'fixed':
            return self.parameters.get('fixed_action', 3), None, 'fixed', 'predeclared_fixed_qos_policy'
        if self.mode == 'online' and self.learner.sample_count < MAX_ROWS:
            if self.learner.sample_count-self.run_start_samples < self.balanced_start_windows:
                counts = self._balanced_start_counts()
                index = min(range(4), key=lambda i: (counts[str(i)], i))
                return index, None, 'rules_only', 'balanced_new_context_measured_qos_exploration'
            if not self._coverage():
                counts = self.action_samples()
                index = min(range(4), key=lambda i: (counts[str(i)], i))
                return index, None, 'rules_only', 'balanced_measured_qos_exploration'
            # Sampling depends only on the saved seed and measured-row count;
            # a resumed pending decision does not advance a hidden RNG state.
            rng = random.Random(self.learner.seed + 1009*self.learner.sample_count)
            if rng.random() < EPSILON:
                self.online_exploration_actions += 1
                return rng.randrange(4), None, 'exploration', 'epsilon_greedy_measured_qos_exploration'
        if not self.learner.trained:
            return 0, None, 'rules_only', 'untrained_native_rr_abstention'
        predictions = self.learner.predict([self._features(users, i) for i in range(4)])
        tolerance = .001/(self.window_ms/1000)
        # Both URLLC UEs are protected, rather than optimizing only UE6. This
        # comparison is an empirical model heuristic, never an SLA guarantee.
        feasible = [i for i in range(4) if all(predictions[i][j] <= predictions[0][j]+tolerance for j in (4, 5))]
        best_failure = min(predictions[i][0] for i in feasible)
        candidates = [i for i in feasible if predictions[i][0] <= best_failure+tolerance]
        means = [p[1]/p[2] if p[2] > 0 else None for p in predictions]
        finite = [means[i] for i in candidates if means[i] is not None]
        if finite:
            best_mean = min(finite)
            candidates = [i for i in candidates if means[i] is not None and means[i] <= best_mean*1.01+1e-9]
        index = max(candidates, key=lambda i: (predictions[i][3], -i))
        if 0 in candidates and predictions[index][3] < 1.01*predictions[0][3]:
            index = 0
        self.last_prediction_audit = dict(raw_predictions=deepcopy(predictions),
            target_order=TARGETS, predicted_completion_mean_ms=means,
            predicted_worst_ue_failure_events_per_s=[max(p[4:6]) for p in predictions],
            per_ue_guard_feasible=feasible, failure_latency_finalists=candidates, selected_action=index,
            event_rate_resolution_per_s=tolerance,
            selection='aggregate both-URLLC failure event rate; completed mean within 1%; network goodput; RR abstention on ties',
            peer_guard='each URLLC predicted failure rate <= same-context RR; empirical, not calibrated safety',
            label_scope='outcome_window_including_carryover_not_arrival_cohort_probability')
        self.rf_selected_actions += 1
        return index, predictions[index], 'rf', 'measured_qos_RF_inference'

    def _sample_targets(self, deltas, delivered, duration):
        rows = [deltas[uid] for uid in URLLC]
        failures = [(d[0]+d[1])/duration for d in rows]
        return [sum(failures), sum(d[3] for d in rows)/duration,
            sum(d[2] for d in rows)/duration, sum(delivered.values())/duration/1e6, *failures]

    def _retain_sample(self, current, targets):
        if self.mode != 'online' or self.learner.sample_count >= MAX_ROWS:
            return
        self.learner.add(current['features'], targets)
        run_new = self.learner.sample_count-self.run_start_samples
        if self.balanced_start_windows and run_new <= self.balanced_start_windows:
            self._log('run_balanced_sample', starting_measured_rows=self.run_start_samples,
                new_measured_windows=run_new, requested_new_windows=self.balanced_start_windows,
                action_counts=self._balanced_start_counts(),
                complete=run_new == self.balanced_start_windows)
        fitted = False
        if self._coverage() and (not self.learner.trained or self.learner.sample_count-self.learner.trained_samples >= REFIT_ROWS):
            fitted = self.learner.fit()
            if fitted:
                self._log('rf_fit', samples=self.learner.trained_samples, fit_count=self.learner.fit_count,
                    fit_parameters=deepcopy(self.learner.fit_parameters), target_scales=self.learner.target_scales)
        if self.model_path:
            self.learner.save(self.model_path)
            if fitted:
                # Immutable fit checkpoints accompany the atomically replaced
                # latest measured-data artifact. Frozen inference writes none.
                raw = self.model_path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                folder = self.model_path.parent/(self.model_path.stem+'_checkpoints')
                folder.mkdir(parents=True, exist_ok=True)
                checkpoint = folder/f'fit-{self.learner.fit_count:04d}-rows-{self.learner.trained_samples:04d}-{digest[:16]}.json'
                temporary = folder/('.'+uuid.uuid4().hex+'.tmp')
                try:
                    temporary.write_bytes(raw)
                    try:
                        # Publish a complete immutable file without replacing
                        # any existing checkpoint, even after an interrupted fit.
                        os.link(temporary, checkpoint)
                    except FileExistsError:
                        if checkpoint.read_bytes() != raw:
                            raise ValueError('Conflicting immutable QoS RF checkpoint')
                finally:
                    temporary.unlink(missing_ok=True)
                self._log('model_checkpoint', path=str(checkpoint), sha256=digest,
                    fit_count=self.learner.fit_count, trained_samples=self.learner.trained_samples)

    def run(self, observation, policy_snapshot):
        def empty():
            result = ControlDecision(config_actions=[])
            result.config_actions.clear()
            return result
        for u in observation.ue_observations:
            if u.ue_id not in URLLC:
                continue
            hol = getattr(u, 'hol_delay_ms', None)
            if type(hol) not in (int, float) or not math.isfinite(hol) or hol < 0:
                self._unavailable('valid_public_HOL_required_for_qos_v2')
                self.last_decisions = []
                return empty()
            metadata = u.kpi_metadata.get('hol_delay_ms', {})
            produced = metadata.get('produced_at_slot')
            if metadata.get('valid') is False or type(produced) is not int or not 0 <= observation.slot_id-produced <= math.ceil(100/self.dt):
                self._unavailable('valid_fresh_public_HOL_required_for_qos_v2')
                self.last_decisions = []
                return empty()
            retx = getattr(u, 'retx_fraction', None)
            if retx is not None and (type(retx) not in (int, float) or not math.isfinite(retx) or not 0 <= retx <= 1):
                self._unavailable('invalid_public_retx_fraction')
                self.last_decisions = []
                return empty()
        return super().run(observation, policy_snapshot)

    def summary(self):
        return dict(mode=self.mode, policy_version='qos_v2', status=self.status, status_reason=self.status_reason,
            algorithm='RandomForestRegressor', target_ue_ids=list(URLLC), monitored_peer_ue_ids=[],
            target_order=TARGETS, feature_names=FEATURES, samples=self.learner.sample_count,
            trained_samples=self.learner.trained_samples, fit_count=self.learner.fit_count,
            action_samples=self.action_samples(), fitted_action_samples=self.action_samples(self.learner.trained_samples),
            minimum_measured_rows=MIN_ROWS, maximum_measured_rows=MAX_ROWS, minimum_rows_per_action=MIN_ROWS_PER_ARM,
            epsilon_after_coverage=EPSILON, refit_every_measured_rows=REFIT_ROWS,
            balanced_start_windows=self.balanced_start_windows,
            run_start_samples=self.run_start_samples, run_start_model_sha256=self.run_start_model_sha256,
            run_new_measured_samples=self.learner.sample_count-self.run_start_samples,
            run_balanced_action_samples=self._balanced_start_counts(),
            run_balanced_start_complete=self.learner.sample_count-self.run_start_samples >= self.balanced_start_windows,
            prediction_batches=self.learner.prediction_batches, rf_selected_actions=self.rf_selected_actions,
            rules_only_actions=self.rules_only_actions, online_exploration_actions=self.online_exploration_actions,
            fixed_actions=self.fixed_actions, invalidated_samples=self.invalidated,
            fit_parameters=deepcopy(self.learner.fit_parameters), target_scales=self.learner.target_scales,
            fit_s=self.learner.fit_s, predict_s=self.learner.predict_s, decision_s=self.decision_seconds,
            model_path=str(self.model_path) if self.model_path else None, model_sha256=self.learner.artifact_sha256,
            original_deadline_ms=10., outcome_window_ms=self.window_ms, wash_in_ms=20,
            actual_decision_period='execution/reporting delays + 20 ms wash-in + configured outcome window',
            label_scope='outcome_window_including_carryover_not_arrival_cohort_probability',
            latency_label='actual completed-packet mean, survivor-only; no capped-latency label available on public counters',
            selection='both-URLLC aggregate failure rate, completed mean latency, network goodput; each-UE empirical RR guard',
            scheduler_mapping='opt-in qos_deadline per-slot PDB/PF priority; UE2+UE6 weights1/2/4; native RR unchanged')

    def descriptor(self):
        return dict(PoolXApp.descriptor(self), learning=self.summary(), target_ue_ids=list(URLLC),
            monitored_peer_ue_ids=[], actions=list(ACTIONS))

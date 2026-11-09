"""X-12-inspired RF for the original UE6 URLLC packet deadline.

This is an explicit packet-level adaptation of the paper's AR scheduler xApp,
using the existing latency-aware priority adapter. It does not replace the MEC
X-12, change NR/traffic, or claim the paper specifies these hyperparameters.
Labels are outcome-window counter deltas including carryover packets, not an
arrival-cohort deadline probability. Matched mature packet-ID audits remain the
authoritative SLA test. No HOL value is used as completed-packet latency.
"""
from collections import deque
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from models.action import ConfigAction, ControlDecision
from ric.near_rt.pools.base import PoolXApp
from .online_packet_rf import OnlinePacketRF, TARGETS, MIN_ROWS, MAX_ROWS

ROOT = Path(__file__).resolve().parents[4]
USERS = ('ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8')
URLLC = ('ue_2', 'ue_6')
ACTIONS = ('native_rr', 'weighted_neutral', 'urllc_1_5', 'urllc_2')
FEATURES = ['network_queue_kb', 'network_reported_goodput_mbps', 'network_offered_mbps',
            'ue_6_queue_kb', 'ue_6_reference_sinr_db', 'ue_6_reported_goodput_mbps',
            'ue_2_queue_kb', 'backlogged_ue_fraction'] + ['action_'+str(i) for i in range(4)]
FIELDS = ('scheduler_mode', 'ue_scheduler_weights')
PACKET_COUNTERS = ('deadline_failed_packets_total', 'overflow_packets_total',
    'completed_packets_total', 'completion_latency_ms_total', 'admitted_packets_total',
    'on_time_completed_packets_total', 'late_completed_packets_total')
COUNTER_MAPS = ('deadline_failed_packets_by_ue', 'overflow_packets_by_ue',
    'completed_packets_by_ue', 'completion_latency_ms_by_ue', 'admitted_packets_by_ue',
    'on_time_completed_packets_by_ue', 'late_completed_packets_by_ue')
# A numerical tie resolution, not an allowance on the original 10 ms SLA.
EVENT_RESOLUTION_PER_WINDOW = .001


def action_map(users, index):
    if type(index) is not int or not 0 <= index < 4:
        raise ValueError('Invalid URLLC RF arm')
    result = {}
    for u in users:
        result.setdefault(u.serving_cell_id, dict(scheduler_mode='round_robin' if index == 0 else 'latency_aware',
            ue_scheduler_weights={}))['ue_scheduler_weights'][u.ue_id] = (1.5 if index == 2 else 2.) if u.ue_id == 'ue_6' and index >= 2 else 1.
    return result


class LatencyUrllcRF(PoolXApp):
    implementation_id = 'latency_urllc_rf'
    category_id = 'latency_scheduler'
    description = 'X-12 packet adaptation: measured UE6 10 ms deadline/latency RF'
    required_kpis = ('queue_bytes', 'reference_sinr_db', 'throughput_mbps', 'offered_rate_bps',
        'delivered_bits_total', 'offered_bits_total', 'executed_config') + PACKET_COUNTERS
    writable_actions = FIELDS
    min_action_ms = 100
    action_names = ACTIONS
    target_ue_ids = ('ue_6',)
    monitored_peer_ue_ids = ('ue_2',)
    non_rr_scheduler_mode = 'latency_aware'
    _action_map = staticmethod(action_map)
    target_names = TARGETS
    max_rows = MAX_ROWS

    def __new__(cls, *args, **kwargs):
        # Explicit versioning keeps saved v1 models and all legacy runs intact.
        parameters = kwargs.get('parameters', args[3] if len(args) > 3 else None)
        if cls is LatencyUrllcRF and isinstance(parameters, dict) and parameters.get('policy_version') == 'qos_v2':
            from .latency_urllc_qos_rf import LatencyUrllcQosRF
            return object.__new__(LatencyUrllcQosRF)
        return object.__new__(cls)

    def validate_parameters(self, p):
        super().validate_parameters(p)
        if set(p) - {'mode', 'model_path', 'log_path', 'window_ms', 'seed', 'fixed_action'}:
            raise ValueError('Unknown URLLC RF parameters')
        if p.get('mode', 'online') not in ('online', 'frozen', 'fixed'):
            raise ValueError('Invalid URLLC RF mode')
        if type(p.get('window_ms', 100)) not in (int, float) or p.get('window_ms', 100) not in (100, 200):
            raise ValueError('URLLC RF window must be 100 or 200 ms')
        if type(p.get('seed', 179)) is not int or not 0 <= p.get('seed', 179) <= 2**32-1:
            raise ValueError('Invalid URLLC RF seed')
        if type(p.get('fixed_action', 3)) is not int or not 0 <= p.get('fixed_action', 3) < 4:
            raise ValueError('Invalid fixed URLLC RF arm')
        for key in ('model_path', 'log_path'):
            if p.get(key) is not None and not Path(p[key]).resolve().is_relative_to(ROOT / 'outputs'):
                raise ValueError('URLLC RF artifacts must stay under outputs')
        if p.get('mode') == 'frozen' and not p.get('model_path'):
            raise ValueError('Frozen URLLC RF requires a measured model')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = self.parameters.get('mode', 'online')
        self.window_ms = self.parameters.get('window_ms', 100)
        self.model_path = Path(self.parameters['model_path']) if self.parameters.get('model_path') else None
        self.log_path = Path(self.parameters['log_path']) if self.parameters.get('log_path') else None
        self.learner = OnlinePacketRF(FEATURES, self.parameters.get('seed', 179))
        if self.model_path and self.model_path.is_file():
            self.learner = OnlinePacketRF.load(self.model_path, FEATURES)
        if any(sum(x[-4:]) != 1 or any(v not in (0., 1.) for v in x[-4:]) for x in self.learner._x):
            raise ValueError('Invalid persisted URLLC action encoding')
        if self.mode == 'frozen' and (not self.learner.trained or not self._coverage(self.learner.trained_samples)):
            raise ValueError('Frozen URLLC RF requires 24 fitted measured rows and six per arm')
        self.pending = None
        self.events = deque(maxlen=256)
        self.invalidated = self.rf_selected_actions = self.rules_only_actions = self.fixed_actions = 0
        self.status, self.status_reason = 'not_evaluable', 'no_valid_public_report'
        self.last_prediction_audit = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists():
                raise ValueError('Choose a fresh URLLC RF event log')

    def action_samples(self, limit=None):
        rows = self.learner._x if limit is None else self.learner._x[:limit]
        return {str(i): sum(x[-4+i] == 1. for x in rows) for i in range(4)}

    def _coverage(self, limit=None):
        n = self.learner.sample_count if limit is None else limit
        return n >= MIN_ROWS and all(v >= 6 for v in self.action_samples(limit).values())

    def _log(self, kind, **values):
        row = dict(kind=kind, mode=self.mode, **values)
        self.events.append(row)
        if self.log_path:
            with self.log_path.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')

    def _invalidate(self, reason, produced=None):
        if self.pending is not None:
            self.pending = None
            self.invalidated += 1
            self._log('sample_invalidated', reason=reason, produced_slot=produced)

    def _unavailable(self, reason):
        self._invalidate(reason)
        if (self.status, self.status_reason) != ('not_evaluable', reason):
            self._log('not_evaluable', reason=reason, persistence='No new request; an empty decision does not roll back a held control')
        self.status, self.status_reason = 'not_evaluable', reason

    @staticmethod
    def _packet(u):
        values = [getattr(u, key, None) for key in PACKET_COUNTERS]
        for key, value in zip(PACKET_COUNTERS, values):
            if key == 'completion_latency_ms_total':
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid public completion latency sum')
            elif type(value) is not int or value < 0:
                raise ValueError('Invalid public packet counter')
        if values[5] + values[6] != values[2]:
            raise ValueError('Inconsistent public completed packet counts')
        return values

    def _features(self, users, index):
        by_id = {u.ue_id: u for u in users}
        target, peer = by_id['ue_6'], by_id['ue_2']
        return [sum(u.queue_bytes for u in users)/1000, sum(u.throughput_mbps for u in users),
            sum(u.offered_rate_bps for u in users)/1e6, target.queue_bytes/1000,
            target.reference_sinr_db, target.throughput_mbps, peer.queue_bytes/1000,
            sum(u.queue_bytes > 0 for u in users)/len(users)] + [float(i == index) for i in range(4)]

    def _choose(self, users):
        self.last_prediction_audit = None
        if self.mode == 'fixed':
            return self.parameters.get('fixed_action', 3), None, 'fixed', 'predeclared_fixed_policy'
        if self.mode == 'online' and self.learner.sample_count < MAX_ROWS and not self._coverage():
            counts = self.action_samples()
            index = min(range(4), key=lambda i: (counts[str(i)], i))
            return index, None, 'rules_only', 'balanced_measured_packet_exploration'
        if not self.learner.trained:
            return 0, None, 'rules_only', 'untrained_native_rr_abstention'
        predictions = self.learner.predict([self._features(users, i) for i in range(4)])
        tolerance = EVENT_RESOLUTION_PER_WINDOW/(self.window_ms/1000)
        feasible = [i for i in range(4) if predictions[i][4] <= predictions[0][4]+tolerance]
        lowest_failure = min(predictions[i][0] for i in feasible)
        candidates = [i for i in feasible if predictions[i][0] <= lowest_failure+tolerance]
        means = [p[1]/p[2] if p[2] > 0 else None for p in predictions]
        finite_means = [means[i] for i in candidates if means[i] is not None]
        if finite_means:
            best_mean = min(finite_means)
            candidates = [i for i in candidates if means[i] is not None and means[i] <= best_mean*1.01+1e-9]
        index = max(candidates, key=lambda i: (predictions[i][3], -i))
        # Equal latency/deadline predictions require a material goodput gain
        # before departing from native RR. RR remains a genuine learned arm.
        if 0 in candidates and predictions[index][3] < 1.01*predictions[0][3]:
            index = 0
        self.last_prediction_audit = dict(raw_predictions=deepcopy(predictions),
            predicted_completion_mean_ms=means, peer_guard_feasible=feasible,
            failure_latency_finalists=candidates, selected_action=index,
            event_resolution_per_window=EVENT_RESOLUTION_PER_WINDOW,
            event_rate_resolution_per_s=tolerance,
            selection='UE6 minimum unique failure event rate; completion mean within 1%; then network goodput',
            peer_guard='UE2 predicted failure event rate <= same-context RR + numerical resolution; not calibrated SLA safety',
            label_scope='outcome_window_including_carryover_not_arrival_cohort_probability')
        self.rf_selected_actions += 1
        return index, predictions[index], 'rf', 'measured_packet_RF_inference'

    @staticmethod
    def _membership(users):
        return {u.ue_id: u.serving_cell_id for u in users}

    def _sample_targets(self, deltas, delivered, duration):
        target, peer = deltas['ue_6'], deltas['ue_2']
        return [(target[0]+target[1])/duration, target[3]/duration, target[2]/duration,
            sum(delivered.values())/duration/1e6, (peer[0]+peer[1])/duration]

    def _retain_sample(self, current, targets):
        if self.mode == 'online' and self.learner.sample_count < MAX_ROWS:
            self.learner.add(current['features'], targets)
            if self._coverage() and (not self.learner.trained or self.learner.sample_count-self.learner.trained_samples >= 4):
                if self.learner.fit():
                    self._log('rf_fit', samples=self.learner.trained_samples, fit_count=self.learner.fit_count,
                        fit_parameters=deepcopy(self.learner.fit_parameters))
            if self.model_path:
                self.learner.save(self.model_path)

    def _observe(self, users, cells, produced, owners):
        current = self.pending
        if current is None:
            return
        if self._membership(users) != current['membership']:
            self._invalidate('handover_or_membership_changed', produced)
            return
        if produced < current['requested_slot']:
            return
        matches = True
        for cid, expected in current['snapshot'].items():
            cfg, owned = cells[cid].executed_config, owners.get(cid, {})
            matches = matches and owned.get('scheduler_mode') == self.name and cfg.get('scheduler_mode') == expected['scheduler_mode']
            weights = expected['ue_scheduler_weights']
            if weights:
                matches = matches and owned.get('ue_scheduler_weights') == self.name
                matches = matches and cfg.get('ue_scheduler_weights', {}) == weights
            if expected['scheduler_mode'] == 'latency_aware' and cfg.get('latency_bias') not in (None, 1.):
                matches = False
        if not matches:
            if current.get('confirmed') or (produced-current['requested_slot'])*self.dt > 200:
                self._invalidate('execution_or_ownership_changed', produced)
            return
        if not current.get('confirmed'):
            current.update(confirmed=True, confirmed_slot=produced)
            self._log('executed_policy_confirmed', action=current['action'], produced_slot=produced,
                executed_snapshot=deepcopy(current['snapshot']), owners=deepcopy(owners))
        if (produced-current['confirmed_slot'])*self.dt < 20:
            return
        if 'start_slot' not in current:
            current.update(start_slot=produced,
                counters={u.ue_id: [u.delivered_bits_total, u.offered_bits_total] for u in users},
                packets={u.ue_id: self._packet(u) for u in users if u.ue_id in URLLC})
            return
        if (produced-current['start_slot'])*self.dt+1e-9 < self.window_ms:
            return
        duration = (produced-current['start_slot'])*self.dt/1000
        delivered = {u.ue_id: u.delivered_bits_total-current['counters'][u.ue_id][0] for u in users}
        offered = {u.ue_id: u.offered_bits_total-current['counters'][u.ue_id][1] for u in users}
        deltas = {u.ue_id: [a-b for a,b in zip(self._packet(u), current['packets'][u.ue_id])] for u in users if u.ue_id in URLLC}
        if any(v < 0 for v in list(delivered.values())+list(offered.values())+[v for d in deltas.values() for v in d]):
            self._invalidate('cumulative_counter_reset', produced)
            return
        targets = self._sample_targets(deltas, delivered, duration)
        maps = {name: {uid: d[i] for uid,d in deltas.items()} for i,name in enumerate(COUNTER_MAPS)}
        self._log('measured_sample', action=current['action'], features=current['features'], targets=targets,
            produced_start_slot=current['start_slot'], produced_end_slot=produced, duration_s=duration,
            delivered_bits_by_ue=delivered, offered_bits_by_ue=offered, **maps,
            completion_mean_ms_by_ue={uid: d[3]/d[2] if d[2] else None for uid,d in deltas.items()},
            original_latency_budgets_ms={u.ue_id: u.latency_budget_ms for u in users if u.ue_id in URLLC},
            membership=deepcopy(current['membership']), snapshot=deepcopy(current['snapshot']),
            target_order=self.target_names, label_scope='outcome_window_including_carryover_not_arrival_cohort_probability',
            execution_confirmation='all_cells_modes_and_all_served_uid_weights_owned_and_executed',
            retained_for_training=self.mode == 'online' and self.learner.sample_count < self.max_rows)
        self._retain_sample(current, targets)
        self.pending = None

    def run(self, observation, policy_snapshot):
        started = time.perf_counter()
        self.last_decisions, actions = [], []
        users = sorted(observation.ue_observations, key=lambda u: u.ue_id)
        cells = {c.cell_id: c for c in observation.cell_observations}
        reason = None
        if tuple(u.ue_id for u in users) != USERS or self.active_cells != set(self.allowed_cells) or any(u.serving_cell_id not in self.active_cells for u in users):
            reason = 'whole_six_ue_network_scope_required'
        elif any(cid not in cells for cid in self.allowed_cells):
            reason = 'missing_managed_cell'
        try:
            if reason is None:
                stamps = [cells[cid].kpi_metadata.get('executed_config', {}).get('produced_at_slot') for cid in self.allowed_cells]
                produced = stamps[0]
                if type(produced) is not int or any(s != produced for s in stamps) or produced > observation.slot_id or (observation.slot_id-produced)*self.dt > 100:
                    raise ValueError('Invalid common public report time')
                if any(cells[cid].kpi_metadata.get('executed_config', {}).get('valid') is False for cid in self.allowed_cells):
                    raise ValueError('Invalid executed config report')
                for u in users:
                    for name in ('queue_bytes', 'reference_sinr_db', 'throughput_mbps', 'offered_rate_bps', 'delivered_bits_total', 'offered_bits_total'):
                        value = getattr(u, name, None)
                        if type(value) not in (int, float) or not math.isfinite(value) or (name != 'reference_sinr_db' and value < 0):
                            raise ValueError('Invalid public '+name)
                    for name in ('delivered_bits_total', 'offered_bits_total'):
                        meta = u.kpi_metadata.get(name, {})
                        if meta.get('produced_at_slot') != produced or meta.get('valid') is False:
                            raise ValueError('Spliced cumulative report')
                    if u.ue_id in URLLC:
                        if u.slice_id.lower() != 'urllc' or u.latency_budget_ms != 10.:
                            raise ValueError('Original 10 ms URLLC profile required')
                        self._packet(u)
                        for name in PACKET_COUNTERS:
                            meta = u.kpi_metadata.get(name, {})
                            if meta.get('produced_at_slot') != produced or meta.get('valid') is False:
                                raise ValueError('Invalid or spliced packet counter')
        except (ValueError, AttributeError, TypeError) as error:
            reason = str(error)
        if reason:
            self._unavailable(reason)
        else:
            self._observe(users, cells, produced, getattr(observation, 'control_ownership', {}))
            if self.pending is None:
                index, prediction, status, why = self._choose(users)
                snapshot = self._action_map(users, index)
                for cid in self.allowed_cells:
                    snapshot.setdefault(cid, dict(scheduler_mode='round_robin' if index == 0 else self.non_rr_scheduler_mode, ue_scheduler_weights={}))
                self.pending = dict(action=index, features=self._features(users, index), snapshot=deepcopy(snapshot),
                    membership=self._membership(users), requested_slot=observation.slot_id, confirmed=False)
                self.status, self.status_reason = status, why
                self.fixed_actions += status == 'fixed'
                self.rules_only_actions += status == 'rules_only'
                for cid, values in snapshot.items():
                    action = ConfigAction(source_name=self.name, target_cell_id=cid, **deepcopy(values))
                    action._library_provenance = self.identity(cid)
                    actions.append(action)
                decision = dict(slot_id=observation.slot_id, action=index, action_name=self.action_names[index], status=status,
                    features=deepcopy(self.pending['features']),
                    reason=why, snapshot=deepcopy(snapshot), membership=self._membership(users), prediction=prediction,
                    prediction_audit=deepcopy(self.last_prediction_audit), instance_id=self.name,
                    implementation_id=self.implementation_id, target_ue_ids=list(self.target_ue_ids), monitored_peer_ue_ids=list(self.monitored_peer_ue_ids))
                self.last_decisions.append(decision)
                self._log('decision', **decision)
        self.decision_seconds += time.perf_counter()-started
        result = ControlDecision(config_actions=actions)
        if not actions:
            result.config_actions.clear()
        return result

    def reset(self):
        super().reset()
        self.pending = None

    def summary(self):
        return dict(mode=self.mode, status=self.status, status_reason=self.status_reason,
            algorithm='RandomForestRegressor', target_ue_ids=['ue_6'], monitored_peer_ue_ids=['ue_2'],
            target_order=TARGETS, feature_names=FEATURES, samples=self.learner.sample_count,
            trained_samples=self.learner.trained_samples, fit_count=self.learner.fit_count,
            action_samples=self.action_samples(), fitted_action_samples=self.action_samples(self.learner.trained_samples),
            minimum_measured_rows=MIN_ROWS, maximum_measured_rows=MAX_ROWS, minimum_rows_per_action=6,
            prediction_batches=self.learner.prediction_batches, rf_selected_actions=self.rf_selected_actions,
            rules_only_actions=self.rules_only_actions, fixed_actions=self.fixed_actions, invalidated_samples=self.invalidated,
            fit_parameters=deepcopy(self.learner.fit_parameters), fit_s=self.learner.fit_s, predict_s=self.learner.predict_s,
            decision_s=self.decision_seconds, model_path=str(self.model_path) if self.model_path else None,
            model_sha256=self.learner.artifact_sha256, original_deadline_ms=10.,
            label_scope='outcome_window_including_carryover_not_arrival_cohort_probability',
            selection='UE6 unique failure event rate, completed mean latency, network goodput tie-break; UE2 heuristic peer guard',
            scheduler_mapping='existing_latency_aware_base_scores_plus_UID_weights_and_unchanged_NR_packing')

    def descriptor(self):
        return dict(super().descriptor(), learning=self.summary(), target_ue_ids=['ue_6'],
            monitored_peer_ue_ids=['ue_2'], actions=list(ACTIONS))

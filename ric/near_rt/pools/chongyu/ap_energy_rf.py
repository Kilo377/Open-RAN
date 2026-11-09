"""X-02-inspired whole-AP energy RF using the opt-in AP availability model.

The paper's RF energy-control idea is adapted to 5G cell duty policies; it is
not a Wi-Fi reproduction. Only cell_sleep_mode is writable. Policies apply to
the whole network, never a test-selected UE or cell or a forced RF arm. An
empty-cell-only arm is a measured negative control.

Whole-AP joules include the explicit modeled base/duty/wakeup components, not
just emitted RF power. QoS heads are actual outcome-window first-failure plus
disjoint overflow rates, not calibrated arrival-cohort failure probabilities.
The default objective is energy first: every QoS cost remains visible.
"""
from collections import deque
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from models.action import ConfigAction, ControlDecision
from ric.near_rt.pools.base import PoolXApp
from .online_ap_rf import OnlineApRF, TARGETS, MIN_ROWS, MAX_ROWS

ROOT = Path(__file__).resolve().parents[4]
USERS = ('ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8')
URLLC = ('ue_2', 'ue_6')
ACTIONS = ('all_active', 'empty_only', 'occupied_micro_sleep', 'occupied_light_sleep')
FEATURES = ['network_queue_kb', 'network_reported_goodput_mbps', 'network_offered_mbps',
    'maximum_cell_queue_kb', 'maximum_cell_offered_mbps', 'minimum_occupied_cell_offered_mbps',
    'backlogged_ue_fraction', 'empty_cell_fraction'] + ['action_'+str(i) for i in range(4)]
PACKET_COUNTERS = ('deadline_failed_packets_total', 'overflow_packets_total',
                   'admitted_packets_total', 'completed_packets_total')
PACKET_MAPS = ('deadline_failed_packets_by_ue', 'overflow_packets_by_ue',
               'admitted_packets_by_ue', 'completed_packets_by_ue')
AP_COUNTERS = ('ap_energy_total_j', 'ap_available_slots_total', 'ap_sleep_slots_total', 'ap_waking_slots_total')
AP_PROOF = ('ap_policy', 'ap_policy_owner', 'ap_policy_generation', 'ap_policy_applied_slot',
            'ap_cycle_period_slots', 'ap_cycle_phase_slot')
AP_NULLABLE_PROOF = ('ap_policy_owner', 'ap_policy_applied_slot')


def action_map(users, cell_ids, index):
    """Complete cell policy, derived only from the public served-UE membership."""
    if type(index) is not int or not 0 <= index < 4:
        raise ValueError('Invalid AP RF arm')
    occupied = {u.serving_cell_id for u in users}
    result = {cid: 'active' if index == 0 or cid in occupied else 'deep_sleep' for cid in cell_ids}
    if index >= 2:
        for cid in occupied:
            result[cid] = 'micro_sleep' if index == 2 else 'light_sleep'
    return result


class ApEnergyRF(PoolXApp):
    implementation_id = 'ap_energy_rf'
    category_id = 'ap_energy_saving'
    description = 'X-02 5G adaptation: measured whole-AP joules RF with explicit duty policies'
    required_kpis = ('queue_bytes', 'throughput_mbps', 'offered_rate_bps',
        'delivered_bits_total', 'offered_bits_total', 'executed_config') + PACKET_COUNTERS + AP_COUNTERS + tuple(k for k in AP_PROOF if k not in AP_NULLABLE_PROOF)
    writable_actions = ('cell_sleep_mode',)
    min_action_ms = 100

    def validate_parameters(self, p):
        super().validate_parameters(p)
        if set(p) - {'mode', 'model_path', 'log_path', 'window_ms', 'seed', 'fixed_action'}:
            raise ValueError('Unknown AP RF parameters')
        if p.get('mode', 'online') not in ('online', 'frozen', 'fixed'):
            raise ValueError('Invalid AP RF mode')
        if type(p.get('window_ms', 100)) not in (int, float) or p.get('window_ms', 100) not in (100, 200):
            raise ValueError('AP RF window must be 100 or 200 ms')
        if type(p.get('seed', 181)) is not int or not 0 <= p.get('seed', 181) <= 2**32-1:
            raise ValueError('Invalid AP RF seed')
        if type(p.get('fixed_action', 1)) is not int or not 0 <= p.get('fixed_action', 1) < 4:
            raise ValueError('Invalid fixed AP RF arm')
        for key in ('model_path', 'log_path'):
            if p.get(key) is not None and not Path(p[key]).resolve().is_relative_to(ROOT / 'outputs'):
                raise ValueError('AP RF artifacts must stay under outputs')
        if p.get('mode') == 'frozen' and not p.get('model_path'):
            raise ValueError('Frozen AP RF requires a measured model')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = self.parameters.get('mode', 'online')
        self.window_ms = self.parameters.get('window_ms', 100)
        self.model_path = Path(self.parameters['model_path']) if self.parameters.get('model_path') else None
        self.log_path = Path(self.parameters['log_path']) if self.parameters.get('log_path') else None
        self.learner = OnlineApRF(FEATURES, self.parameters.get('seed', 181))
        if self.model_path and self.model_path.is_file():
            self.learner = OnlineApRF.load(self.model_path, FEATURES)
        if any(sum(x[-4:]) != 1 or any(v not in (0., 1.) for v in x[-4:]) for x in self.learner._x):
            raise ValueError('Invalid persisted AP action encoding')
        if self.mode == 'frozen' and (not self.learner.trained or not self._coverage(self.learner.trained_samples)):
            raise ValueError('Frozen AP RF requires 24 fitted measured global rows and six per arm')
        self.pending = None
        self.events = deque(maxlen=256)
        self.invalidated = self.rf_selected_actions = self.rules_only_actions = self.fixed_actions = 0
        self.status, self.status_reason = 'not_evaluable', 'no_valid_public_report'
        self.last_prediction_audit = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists():
                raise ValueError('Choose a fresh AP RF event log')

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
            self._log('not_evaluable', reason=reason, persistence='No new request; an empty decision does not undo a held AP policy')
        self.status, self.status_reason = 'not_evaluable', reason

    @staticmethod
    def _membership(users):
        return {u.ue_id: u.serving_cell_id for u in users}

    @staticmethod
    def _packet(u):
        values = [getattr(u, key, None) for key in PACKET_COUNTERS]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError('Invalid AP public packet counter')
        return values

    def _features(self, users, index):
        occupied = {u.serving_cell_id for u in users}
        queues = [sum(u.queue_bytes for u in users if u.serving_cell_id == cid) for cid in self.allowed_cells]
        offered = [sum(u.offered_rate_bps for u in users if u.serving_cell_id == cid) for cid in occupied]
        return [sum(u.queue_bytes for u in users)/1000, sum(u.throughput_mbps for u in users),
            sum(u.offered_rate_bps for u in users)/1e6, max(queues)/1000,
            max(offered)/1e6, min(offered)/1e6, sum(u.queue_bytes > 0 for u in users)/len(users),
            1-len(occupied)/len(self.allowed_cells)] + [float(i == index) for i in range(4)]

    def _choose(self, users):
        self.last_prediction_audit = None
        if self.mode == 'fixed':
            return self.parameters.get('fixed_action', 1), None, 'fixed', 'predeclared_fixed_policy'
        if self.mode == 'online' and self.learner.sample_count < MAX_ROWS and not self._coverage():
            counts = self.action_samples()
            index = min(range(4), key=lambda i: (counts[str(i)], i))
            return index, None, 'rules_only', 'balanced_measured_AP_exploration'
        if not self.learner.trained:
            return 0, None, 'rules_only', 'untrained_all_active_abstention'
        predictions = self.learner.predict([self._features(users, i) for i in range(4)])
        lowest = min(p[0] for p in predictions)
        # Equal predicted energy (one part per million) favors lower actual
        # forecasted URLLC failures, then goodput. This is not a QoS constraint.
        tolerance = max(1e-9, abs(lowest)*1e-6)
        candidates = [i for i in range(4) if predictions[i][0] <= lowest+tolerance]
        index = min(candidates, key=lambda i: (predictions[i][2]+predictions[i][3], -predictions[i][1], i))
        if predictions[0][0]-predictions[index][0] < .01*predictions[0][0]:
            index = 0
        self.last_prediction_audit = dict(raw_predictions=deepcopy(predictions), selected_action=index,
            energy_tie_tolerance_w=tolerance, energy_finalists=candidates,
            selection='minimum predicted whole-AP power; equal-energy QoS/goodput tie-break; 1% all-active abstention margin',
            constraints_active=False, objective='energy_first_research_QoS_recorded_not_constrained')
        self.rf_selected_actions += 1
        return index, predictions[index], 'rf', 'measured_whole_AP_energy_RF_inference'

    def _observe(self, users, cells, produced):
        current = self.pending
        if current is None:
            return
        if self._membership(users) != current['membership']:
            self._invalidate('handover_or_membership_changed', produced)
            return
        if produced < current['requested_slot']:
            return
        matches = all(cells[cid].executed_config.get('cell_sleep_mode') == requested
            and cells[cid].ap_policy == requested and cells[cid].ap_policy_owner == self.name
            and type(cells[cid].ap_policy_applied_slot) is int
            and cells[cid].ap_policy_applied_slot <= produced for cid, requested in current['snapshot'].items())
        if not matches:
            if current.get('confirmed') or (produced-current['requested_slot'])*self.dt > 200:
                self._invalidate('AP_execution_or_actual_owner_changed', produced)
            return
        generations = {cid: cells[cid].ap_policy_generation for cid in self.allowed_cells}
        context = {cid: {key: deepcopy(value) for key,value in cells[cid].executed_config.items()
                         if key != 'cell_sleep_mode'} for cid in self.allowed_cells}
        if current.get('confirmed') and generations != current['generations']:
            self._invalidate('AP_policy_generation_changed', produced)
            return
        if current.get('confirmed') and context != current['fixed_control_context']:
            self._invalidate('other_executed_control_changed_during_AP_label', produced)
            return
        if not current.get('confirmed'):
            current.update(confirmed=True, confirmed_slot=produced, generations=generations,
                           fixed_control_context=context)
            self._log('executed_policy_confirmed', action=current['action'], produced_slot=produced,
                executed_snapshot=deepcopy(current['snapshot']), generations=deepcopy(generations),
                actual_owners={cid: cells[cid].ap_policy_owner for cid in self.allowed_cells},
                actual_applied_slots={cid: cells[cid].ap_policy_applied_slot for cid in self.allowed_cells})
        periods = {cid: cells[cid].ap_cycle_period_slots for cid in self.allowed_cells}
        if any(produced-cells[cid].ap_policy_applied_slot < periods[cid] for cid in self.allowed_cells):
            return
        if 'start_slot' not in current:
            current.update(start_slot=produced, period_slots=periods,
                start_phases={cid: cells[cid].ap_cycle_phase_slot for cid in self.allowed_cells},
                energy={cid: cells[cid].ap_energy_total_j for cid in self.allowed_cells},
                availability={cid: [getattr(cells[cid], key) for key in AP_COUNTERS[1:]] for cid in self.allowed_cells},
                counters={u.ue_id: [u.delivered_bits_total, u.offered_bits_total] for u in users},
                packets={u.ue_id: self._packet(u) for u in users if u.ue_id in URLLC})
            return
        slots = produced-current['start_slot']
        if slots*self.dt+1e-9 < self.window_ms:
            return
        if periods != current['period_slots']:
            self._invalidate('AP_cycle_profile_changed', produced)
            return
        if any(slots % period for period in periods.values()) or any(cells[cid].ap_cycle_phase_slot != current['start_phases'][cid] for cid in self.allowed_cells):
            # Reports need not land on an exact cycle boundary; equal phases
            # after an integer period suffice, and preserve all real outcomes.
            return
        duration = slots*self.dt/1000
        energy = {cid: cells[cid].ap_energy_total_j-current['energy'][cid] for cid in self.allowed_cells}
        availability = {cid: [getattr(cells[cid], key)-value for key,value in zip(AP_COUNTERS[1:], current['availability'][cid])] for cid in self.allowed_cells}
        delivered = {u.ue_id: u.delivered_bits_total-current['counters'][u.ue_id][0] for u in users}
        offered = {u.ue_id: u.offered_bits_total-current['counters'][u.ue_id][1] for u in users}
        packets = {u.ue_id: [a-b for a,b in zip(self._packet(u), current['packets'][u.ue_id])] for u in users if u.ue_id in URLLC}
        if any(v < 0 for v in list(energy.values())+list(delivered.values())+list(offered.values())+[v for a in availability.values() for v in a]+[v for a in packets.values() for v in a]):
            self._invalidate('cumulative_counter_reset', produced)
            return
        if any(sum(values) != slots for values in availability.values()):
            self._invalidate('inconsistent_AP_availability_accounting', produced)
            return
        targets = [sum(energy.values())/duration, sum(delivered.values())/duration/1e6] + [(packets[uid][0]+packets[uid][1])/duration for uid in URLLC]
        maps = {name: {uid: values[i] for uid,values in packets.items()} for i,name in enumerate(PACKET_MAPS)}
        self._log('measured_sample', action=current['action'], features=current['features'], targets=targets,
            produced_start_slot=current['start_slot'], produced_end_slot=produced, duration_s=duration,
            ap_energy_j_by_cell=energy, ap_available_slots_by_cell={cid:v[0] for cid,v in availability.items()},
            ap_sleep_slots_by_cell={cid:v[1] for cid,v in availability.items()}, ap_waking_slots_by_cell={cid:v[2] for cid,v in availability.items()},
            cycle_period_slots_by_cell=periods, start_cycle_phase_by_cell=current['start_phases'],
            end_cycle_phase_by_cell={cid: cells[cid].ap_cycle_phase_slot for cid in self.allowed_cells},
            delivered_bits_by_ue=delivered, offered_bits_by_ue=offered, **maps,
            membership=deepcopy(current['membership']), snapshot=deepcopy(current['snapshot']),
            actual_policy_generations=deepcopy(current['generations']), target_order=TARGETS,
            fixed_control_context=deepcopy(current['fixed_control_context']),
            original_latency_budgets_ms={u.ue_id: u.latency_budget_ms for u in users if u.ue_id in URLLC},
            execution_confirmation='all_AP_policies_actual_consumer_owner_ACK_after_full_period_settling',
            label_scope='global_AP_energy_and_outcome_window_including_carryover_not_arrival_cohort_failure_probability',
            retained_for_training=self.mode == 'online' and self.learner.sample_count < MAX_ROWS)
        if self.mode == 'online' and self.learner.sample_count < MAX_ROWS:
            self.learner.add(current['features'], targets)
            if self._coverage() and (not self.learner.trained or self.learner.sample_count-self.learner.trained_samples >= 4):
                if self.learner.fit():
                    self._log('rf_fit', samples=self.learner.trained_samples, fit_count=self.learner.fit_count,
                        fit_parameters=deepcopy(self.learner.fit_parameters))
            if self.model_path:
                self.learner.save(self.model_path)
        self.pending = None

    def run(self, observation, policy_snapshot):
        started = time.perf_counter()
        self.last_decisions, actions = [], []
        users = sorted(observation.ue_observations, key=lambda u: u.ue_id)
        cells = {c.cell_id: c for c in observation.cell_observations}
        reason = None
        if tuple(u.ue_id for u in users) != USERS or self.active_cells != set(self.allowed_cells) or any(u.serving_cell_id not in self.active_cells for u in users):
            reason = 'whole_six_UE_three_cell_public_scope_required'
        elif len(self.allowed_cells) != 3 or any(cid not in cells for cid in self.allowed_cells):
            reason = 'three_managed_AP_reports_required'
        try:
            if reason is None:
                stamps = [cells[cid].kpi_metadata.get('executed_config', {}).get('produced_at_slot') for cid in self.allowed_cells]
                produced = stamps[0]
                if type(produced) is not int or any(s != produced for s in stamps) or produced > observation.slot_id or (observation.slot_id-produced)*self.dt > 100:
                    raise ValueError('Invalid common AP public report time')
                for cid in self.allowed_cells:
                    cell = cells[cid]
                    for key in ('executed_config',)+AP_COUNTERS+AP_PROOF:
                        meta = cell.kpi_metadata.get(key, {})
                        nullable_missing = key in AP_NULLABLE_PROOF and getattr(cell, key, None) is None
                        if meta.get('produced_at_slot') != produced or (meta.get('valid') is False and not nullable_missing):
                            raise ValueError('Invalid or spliced AP public report')
                    for key in AP_COUNTERS:
                        value = getattr(cell, key, None)
                        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key != 'ap_energy_total_j' and type(value) is not int):
                            raise ValueError('Invalid actual AP energy/availability counter')
                    period, phase = cell.ap_cycle_period_slots, cell.ap_cycle_phase_slot
                    window_slots = self.window_ms/self.dt
                    if type(period) is not int or period <= 0 or type(phase) is not int or not 0 <= phase < period or not math.isclose(window_slots, round(window_slots), abs_tol=1e-9) or round(window_slots) % period:
                        raise ValueError('AP label window must contain integer actual profile periods')
                    if cell.ap_policy not in ('active', 'micro_sleep', 'light_sleep', 'deep_sleep') or type(cell.ap_policy_generation) is not int or cell.ap_policy_generation < 0:
                        raise ValueError('Invalid public AP policy generation')
                for u in users:
                    for key in ('queue_bytes', 'throughput_mbps', 'offered_rate_bps', 'delivered_bits_total', 'offered_bits_total'):
                        value = getattr(u, key, None)
                        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                            raise ValueError('Invalid AP public UE '+key)
                    for key in ('delivered_bits_total', 'offered_bits_total'):
                        meta = u.kpi_metadata.get(key, {})
                        if meta.get('produced_at_slot') != produced or meta.get('valid') is False:
                            raise ValueError('Spliced UE cumulative report')
                    if u.ue_id in URLLC:
                        if u.slice_id.lower() != 'urllc' or u.latency_budget_ms != 10.:
                            raise ValueError('Original URLLC 10 ms profile required')
                        self._packet(u)
                        for key in PACKET_COUNTERS:
                            meta = u.kpi_metadata.get(key, {})
                            if meta.get('produced_at_slot') != produced or meta.get('valid') is False:
                                raise ValueError('Spliced AP packet outcome report')
        except (ValueError, AttributeError, TypeError) as error:
            reason = str(error)
        if reason:
            self._unavailable(reason)
        else:
            self._observe(users, cells, produced)
            if self.pending is None:
                index, prediction, status, why = self._choose(users)
                snapshot = action_map(users, self.allowed_cells, index)
                features = self._features(users, index)
                self.pending = dict(action=index, features=features, snapshot=deepcopy(snapshot),
                    membership=self._membership(users), requested_slot=observation.slot_id, confirmed=False)
                self.status, self.status_reason = status, why
                self.fixed_actions += status == 'fixed'
                self.rules_only_actions += status == 'rules_only'
                for cid, mode in snapshot.items():
                    action = ConfigAction(source_name=self.name, target_cell_id=cid, cell_sleep_mode=mode)
                    action._library_provenance = self.identity(cid)
                    actions.append(action)
                decision = dict(slot_id=observation.slot_id, action=index, action_name=ACTIONS[index], features=deepcopy(features),
                    status=status, reason=why, snapshot=deepcopy(snapshot), membership=self._membership(users), prediction=prediction,
                    prediction_audit=deepcopy(self.last_prediction_audit), instance_id=self.name, implementation_id=self.implementation_id,
                    deployment_scope='whole_network_all_currently_occupied_cells',
                    publicly_empty_cell_ids=sorted(set(self.allowed_cells)-{u.serving_cell_id for u in users}),
                    effective_same_as_empty_only=index >= 2 and snapshot == action_map(users, self.allowed_cells, 1),
                    constraints_active=False)
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
        return dict(mode=self.mode, status=self.status, status_reason=self.status_reason, algorithm='RandomForestRegressor',
            objective='energy_first_research_QoS_recorded_not_constrained', constraints_active=False,
            deployment_scope='whole_network_all_currently_occupied_cells', target_order=TARGETS, feature_names=FEATURES,
            samples=self.learner.sample_count, trained_samples=self.learner.trained_samples, fit_count=self.learner.fit_count,
            action_samples=self.action_samples(), fitted_action_samples=self.action_samples(self.learner.trained_samples),
            minimum_measured_rows=MIN_ROWS, maximum_measured_rows=MAX_ROWS, minimum_rows_per_action=6,
            prediction_batches=self.learner.prediction_batches, rf_selected_actions=self.rf_selected_actions,
            rules_only_actions=self.rules_only_actions, fixed_actions=self.fixed_actions, invalidated_samples=self.invalidated,
            fit_parameters=deepcopy(self.learner.fit_parameters), fit_s=self.learner.fit_s, predict_s=self.learner.predict_s,
            decision_s=self.decision_seconds, model_path=str(self.model_path) if self.model_path else None,
            model_sha256=self.learner.artifact_sha256, energy_scope='explicit_opt_in_whole_AP_power_model_not_RF_only_or_hardware_measurement',
            label_scope='global_one_row_per_actual_window_not_three_correlated_cell_rows',
            cycle_measurement='one_full_actual_period_settling_then_integer_period_equal_phase_window')

    def descriptor(self):
        return dict(super().descriptor(), learning=self.summary(), actions=list(ACTIONS))

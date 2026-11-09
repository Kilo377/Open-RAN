"""Lightweight, slot-clock UE power, connected-DRX and task-service models.

This is an explicit system-level extension, not a Sionna/OAI implementation.
DRX conservatively stays active for pending HARQ and handover. MEC uploads use
a finite, shared abstract UL bearer; result packets use the real DL ledger.
All numerical power/compute values are configurable scenario assumptions.
Offloaded-route metrics cover tasks assigned that route after initial admission.
Pre-route rejects remain unassigned failures in the overall task QoS denominator.
"""
from collections import Counter, deque
from copy import deepcopy
import heapq
import math
import random
import time


ACTION_FIELDS = frozenset({'ue_drx_configs', 'ue_offload_fraction'})
KPI_UNITS = {'ue_energy_total_j': 'J', 'ue_power_w': 'W',
             'ue_radio_state': 'state', 'ue_active_fraction': 'ratio',
             'ue_task_metrics': 'record', 'ue_task_cohorts': 'record', 'ue_service_policy': 'record'}
DRX_DEFAULTS = dict(enabled=False, cycle_ms=40., on_duration_ms=4.,
                    inactivity_ms=8., wakeup_ms=.5, offset_ms=0.)
POWER_DEFAULTS = dict(sleep_w=.04, monitor_w=.25, receive_w=.9,
                      transmit_w=1.4, waking_w=.9, wakeup_j=.0002,
                      cpu_idle_w=.05, cpu_compute_w=.8)
OFFLOADED_COUNTER_DEFAULTS = dict(offloaded_generated=0, offloaded_completed=0,
    offloaded_on_time_completed=0, offloaded_failed=0, offloaded_pending=0,
    offloaded_completion_latency_ms_sum=0.)
TASK_STAGE_SCHEMA = 'oran.mec-stage-timing.v1'
TASK_STAGES = ('admission', 'upload', 'backhaul_to_edge', 'edge_compute',
               'backhaul_to_result', 'downlink', 'local_compute')
TASK_SERVICE_STAGES = ('upload', 'edge_compute', 'local_compute')
UPLOAD_BLOCK_REASONS = ('radio_unavailable', 'not_connected', 'no_path')
TASK_DEADLINE_SEMANTICS = (
    'soft observation deadline: overdue tasks are counted once but continue '
    'receiving service; deadlines do not cancel queued work')


def _stage_aggregate():
    return dict(count=0, elapsed_ms_sum=0., stage_ms_sum=Counter(),
                service_ms_sum=Counter(), upload_blocked_ms_sum=0.,
                upload_blocked_by_reason_ms_sum=Counter(),
                terminal_stage_counts=Counter(), outcome_counts=Counter())


def _add_stage_timing(target, timing, outcome=None):
    target['count'] += 1
    target['elapsed_ms_sum'] += timing['elapsed_ms']
    target['stage_ms_sum'].update(timing['stage_ms'])
    target['service_ms_sum'].update(timing['service_ms'])
    target['upload_blocked_ms_sum'] += timing['upload_blocked_ms']
    target['upload_blocked_by_reason_ms_sum'].update(timing['upload_blocked_by_reason_ms'])
    if outcome is not None:
        target['terminal_stage_counts'][timing['terminal_stage']] += 1
        target['outcome_counts'][outcome] += 1


def _merge_stage_aggregate(target, source):
    for key in ('count', 'elapsed_ms_sum', 'upload_blocked_ms_sum'):
        target[key] += source[key]
    for key in ('stage_ms_sum', 'service_ms_sum',
                'upload_blocked_by_reason_ms_sum', 'terminal_stage_counts', 'outcome_counts'):
        target[key].update(source[key])


def _stage_aggregate_report(value):
    count = value['count']
    stages = {s: value['stage_ms_sum'][s] for s in TASK_STAGES}
    service = {s: value['service_ms_sum'][s] for s in TASK_SERVICE_STAGES}
    blocked = value['upload_blocked_ms_sum']
    return dict(count=count, elapsed_ms_sum=value['elapsed_ms_sum'],
                elapsed_mean_ms=value['elapsed_ms_sum']/count if count else None,
                stage_ms_sum=stages,
                stage_mean_ms={s: v/count if count else None for s,v in stages.items()},
                service_ms_sum=service,
                wait_or_quantization_ms_sum={s: max(0., stages[s]-service[s]-(blocked if s=='upload' else 0.))
                                             for s in TASK_SERVICE_STAGES},
                upload_blocked_ms_sum=blocked,
                upload_blocked_by_reason_ms_sum={s: value['upload_blocked_by_reason_ms_sum'][s]
                                                for s in UPLOAD_BLOCK_REASONS},
                terminal_stage_counts=dict(value['terminal_stage_counts']),
                outcome_counts=dict(value['outcome_counts']))


def finite(value, name, minimum=0., maximum=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError('invalid_' + name)
    return float(value)


def drx_config(value, dt):
    if not isinstance(value, dict) or set(value) - set(DRX_DEFAULTS):
        raise ValueError('invalid_drx_fields')
    result = dict(DRX_DEFAULTS)
    result['wakeup_ms'] = math.ceil(result['wakeup_ms'] / dt) * dt
    result.update(value)
    if type(result['enabled']) is not bool:
        raise ValueError('invalid_drx_enabled')
    for key in set(DRX_DEFAULTS) - {'enabled'}:
        n = finite(result[key], key) / dt
        if not math.isclose(n, round(n), abs_tol=1e-8, rel_tol=1e-10):
            raise ValueError('drx_timing_must_align_to_slots')
    if not 0 < result['on_duration_ms'] <= result['cycle_ms'] or not 0 <= result['offset_ms'] < result['cycle_ms'] or result['wakeup_ms'] > result['cycle_ms'] - result['on_duration_ms']:
        raise ValueError('invalid_drx_cycle')
    return result


class UeServices:
    def __init__(self, state, config, dt_ms, seed=1):
        if type(config.get('stage_metrics', False)) is not bool:
            raise ValueError('invalid_stage_metrics_enabled')
        self.stage_metrics_enabled = config.get('stage_metrics', False)
        self.state = state
        self.config = deepcopy(config)
        if 'enabled' in config and type(config['enabled']) is not bool:
            raise ValueError('invalid_ue_services_enabled')
        self.dt_ms = finite(dt_ms, 'slot_duration', 1e-9)
        self.dt = self.dt_ms / 1000
        self.slot = 0
        self.now = 0.
        self.wall_s = 0.
        cohort_ms = finite(config.get('task_cohort_ms', 0), 'task_cohort_ms')
        cohort_slots = cohort_ms / self.dt_ms
        if cohort_ms and (cohort_slots < 1 or not math.isclose(cohort_slots, round(cohort_slots), abs_tol=1e-8, rel_tol=1e-10)):
            raise ValueError('task_cohort_must_align_to_slots')
        self.cohort_slots = round(cohort_slots)
        self.cohort_seconds = self.cohort_slots * self.dt
        self.cohorts = {}
        self.cohort_cache = {}
        self.cohort_dirty = set()
        self.users = {u.ue_id: u for u in state.ues}
        self.power = dict(POWER_DEFAULTS, **config.get('power', {}))
        if set(self.power) != set(POWER_DEFAULTS):
            raise ValueError('unknown_power_parameter')
        for key, value in self.power.items():
            finite(value, key)
        if min(self.power['monitor_w'], self.power['receive_w'], self.power['transmit_w']) < self.power['sleep_w']:
            raise ValueError('sleep_power_exceeds_active_power')
        self.drx = {uid: drx_config(config.get('drx', {}), self.dt_ms) for uid in self.users}
        overrides = config.get('drx_by_ue', {})
        if set(overrides) - self.users.keys():
            raise ValueError('unknown_drx_ue')
        for uid, value in overrides.items():
            self.drx[uid] = drx_config(dict(self.drx[uid], **value), self.dt_ms)
        self.inactivity_until = dict.fromkeys(self.users, 0)
        self.wake_until = dict.fromkeys(self.users, 0)
        self.last_active = dict.fromkeys(self.users, True)
        self.energy = {uid: Counter() for uid in self.users}
        self.last_power = dict.fromkeys(self.users, 0.)
        self.active_slots = Counter()
        self.radio_states = dict.fromkeys(self.users, 'monitor')
        self.mec = deepcopy(config.get('mec', {}))
        if set(self.mec) - {'enabled', 'local_gflops', 'server_gflops', 'uplink_bps_per_cell', 'backhaul_one_way_ms', 'max_queued_tasks', 'tasks_by_ue'}:
            raise ValueError('unknown_mec_parameter')
        self.mec_enabled = self.mec.get('enabled', False)
        if type(self.mec_enabled) is not bool:
            raise ValueError('invalid_mec_enabled')
        self.local_rate = finite(self.mec.get('local_gflops', 2.), 'local_gflops', 1e-9)
        self.edge_rate = finite(self.mec.get('server_gflops', 40.), 'server_gflops', 1e-9)
        self.ul_rate = finite(self.mec.get('uplink_bps_per_cell', 5e6), 'uplink_bps', 1.)
        self.backhaul_s = finite(self.mec.get('backhaul_one_way_ms', 2.), 'backhaul_ms') / 1000
        self.max_queue = self.mec.get('max_queued_tasks', 32)
        if type(self.max_queue) is not int or self.max_queue < 1:
            raise ValueError('invalid_task_queue_bound')
        self.tasks = {}
        # Diagnostic state is separate from physical task dictionaries. No
        # routing, queue, budget, timing or RNG decision consumes these fields.
        self.task_timings = {}
        self.stage_clock_s = 0.
        self.terminal_stage_metrics = {
            uid: {kind: dict(all=_stage_aggregate(), routes={})
                  for kind in ('completed', 'failed')} for uid in self.users
        } if self.stage_metrics_enabled else {}
        self.events = []
        self.sequence = 0
        self.future = []
        self.local = {uid: deque() for uid in self.users}
        self.upload = {cid: deque() for cid in state.managed_cells}
        self.edge = {cid: deque() for cid in state.managed_cells}
        self.stats = {uid: Counter(OFFLOADED_COUNTER_DEFAULTS) for uid in self.users}
        self.server_stats = {cid: Counter() for cid in state.managed_cells}
        self.offload = dict.fromkeys(self.users, .5)
        self.profiles = {}
        profiles = self.mec.get('tasks_by_ue', {}) if self.mec_enabled else {}
        if set(profiles) - self.users.keys():
            raise ValueError('unknown_task_ue')
        for uid, profile in profiles.items():
            p = dict(interval_ms=50., input_bytes=1000, result_bytes=200,
                     compute_gflop=.05, deadline_ms=100., offload_fraction=.5, **{})
            if set(profile) - set(p):
                raise ValueError('unknown_task_parameter')
            p.update(profile)
            for key in ('input_bytes', 'result_bytes'):
                if type(p[key]) is not int or p[key] < 1:
                    raise ValueError('invalid_' + key)
            p['compute_gflop'] = finite(p['compute_gflop'], 'compute_gflop', 1e-12)
            p['deadline_ms'] = finite(p['deadline_ms'], 'task_deadline_ms', self.dt_ms)
            slots = finite(p['interval_ms'], 'task_interval_ms', self.dt_ms) / self.dt_ms
            if not math.isclose(slots, round(slots), abs_tol=1e-8):
                raise ValueError('task_interval_must_align_to_slots')
            p['interval_slots'] = round(slots)
            self.offload[uid] = finite(p['offload_fraction'], 'offload_fraction', 0, 1)
            self.profiles[uid] = p
        self.rng = {uid: random.Random(f'{seed}:ue-services:{uid}') for uid in self.users}
        self.reports = {}
        for ue in self.users.values():
            ue.ue_radio_available = True
            ue.packet_ledger.service_delivery_listener = self.delivered
        self._reports()

    def validate_action(self, action, targets):
        for field in ACTION_FIELDS:
            values = getattr(action, field, None)
            if values is None:
                continue
            if not isinstance(values, dict):
                raise ValueError('invalid_' + field)
            for uid, value in values.items():
                if uid not in self.users or self.state.connections[uid].serving_cell_id not in targets:
                    raise ValueError('ue_service_target_outside_cell_scope')
                if field == 'ue_drx_configs':
                    drx_config(dict(self.drx[uid], **value) if isinstance(value, dict) else value, self.dt_ms)
                else:
                    if not self.mec_enabled or uid not in self.profiles:
                        raise ValueError('task_model_not_configured_for_ue')
                    finite(value, 'offload_fraction', 0, 1)

    def apply_action(self, action):
        for uid, value in (action.ue_drx_configs or {}).items():
            self.drx[uid] = drx_config(dict(self.drx[uid], **value), self.dt_ms)
        self.offload.update(action.ue_offload_fraction or {})

    def policies(self, cid):
        return {uid: dict(drx=deepcopy(self.drx[uid]), offload_fraction=self.offload[uid])
                for uid in self.users if self.state.connections[uid].serving_cell_id == cid}

    def begin_slot(self, slot):
        start = time.perf_counter()
        self.slot = slot
        self.now = (slot - 1) * self.dt
        if self.stage_metrics_enabled:
            self.stage_clock_s = self.now
        self.events = []
        self.result_arrivals = Counter()
        while self.future and self.future[0][0] <= self.now + 1e-12:
            _, _, kind, tid = heapq.heappop(self.future)
            task = self.tasks.get(tid)
            if task is None:
                continue
            if kind == 'edge':
                self._enqueue(self.edge[task['cell']], task, 'compute', task['gflop'])
            else:
                self._inject(task)
        for uid, profile in self.profiles.items():
            if (slot - 1) % profile['interval_slots'] == 0:
                self._generate(uid, profile)
        self.wall_s += time.perf_counter() - start

    def before_scheduling(self):
        start = time.perf_counter()
        uploading = {self.tasks[tid]['uid'] for q in self.upload.values() for tid in q if tid in self.tasks}
        for uid, ue in self.users.items():
            p = self.drx[uid]
            cycle = round(p['cycle_ms'] / self.dt_ms)
            phase = (self.slot - 1 - round(p['offset_ms'] / self.dt_ms)) % cycle
            on = round(p['on_duration_ms'] / self.dt_ms)
            wake = round(p['wakeup_ms'] / self.dt_ms)
            connection = self.state.connections[uid]
            # Conservative: do not hide HARQ retransmissions behind a DRX cycle.
            harq = any(p.is_busy or p.is_waiting_ack or p.pending_bytes for p in ue.harq.processes)
            forced = harq or uid in uploading or bool(getattr(connection, 'handover_pending_target_cell_id', None)) or connection.connection_state == 'handover_interruption'
            desired = not p['enabled'] or phase < on or forced or self.slot < self.inactivity_until[uid]
            prewake = p['enabled'] and wake > 0 and phase >= cycle - wake
            if (desired or prewake) and not self.last_active[uid] and self.wake_until[uid] <= self.slot:
                self.wake_until[uid] = self.slot + wake
                self.energy[uid]['transition_j'] += self.power['wakeup_j']
            waking = self.slot < self.wake_until[uid]
            active = desired and not waking
            # last_active tracks powered radio, including the pre-wake interval.
            self.last_active[uid] = desired or prewake or waking
            ue.ue_radio_available = active
            self.radio_states[uid] = 'waking' if waking else 'monitor' if active else 'sleep'
        self.wall_s += time.perf_counter() - start

    def _generate(self, uid, p):
        self.sequence += 1
        tid = self.sequence
        task = dict(id=tid, uid=uid, created=self.now, gflop=p['compute_gflop'],
                    input_bytes=p['input_bytes'], result_bytes=p['result_bytes'],
                    deadline_s=p['deadline_ms'] / 1000, overdue=False,
                    cell=self.state.connections[uid].serving_cell_id)
        self.tasks[tid] = task
        if self.stage_metrics_enabled:
            self.task_timings[tid] = dict(stage='admission', entered_s=self.now,
                stage_s=Counter(), service_s=Counter(), upload_blocked_s=Counter())
        self.stats[uid]['generated'] += 1
        self._cohort_generate(task)
        # This audit record outlives bounded public cohort telemetry. Route is
        # filled after the original admission/RNG decision, before an enqueue
        # can fail; pre-route rejects deliberately retain offloaded=None.
        generated_event = dict(kind='task_generated', task_id=tid, ue_id=uid, time_s=self.now,
            created_s=task['created'], deadline_s=task['deadline_s'],
            cohort_index=task.get('cohort_index'), offloaded=None)
        self.events.append(generated_event)
        if sum(t['uid'] == uid for t in self.tasks.values()) > self.max_queue:
            self._finish(task, 'queue_overflow')
            return
        # Preserve the original single draw, after initial per-UE admission.
        # Pre-route rejects remain unassigned failures in the all-task guard.
        if self.rng[uid].random() < self.offload[uid]:
            task['offloaded'] = True
            generated_event['offloaded'] = True
            self.stats[uid]['offloaded'] += 1
            self.stats[uid]['offloaded_generated'] += 1
            self.stats[uid]['offloaded_pending'] += 1
            self._cohort_count(task, 'offloaded')
            self._cohort_count(task, 'offloaded_generated')
            self._cohort_count(task, 'offloaded_pending')
            self._enqueue(self.upload[task['cell']], task, 'upload', p['input_bytes'] * 8.)
        else:
            task['offloaded'] = False
            generated_event['offloaded'] = False
            self.stats[uid]['local'] += 1
            self._cohort_count(task, 'local')
            self._enqueue(self.local[uid], task, 'compute', p['compute_gflop'])

    def _cohort_generate(self, task):
        if not self.cohort_slots:
            return
        uid = task['uid']
        # Creation is exactly a slot boundary. Integer arithmetic avoids
        # floating-point floor(0.3 / 0.1) putting a task in the wrong bucket.
        index = (self.slot - 1) // self.cohort_slots
        task['cohort_index'] = index
        rows = self.cohorts.setdefault(uid, {})
        if index not in rows:
            rows[index] = dict(index=index, start_s=index * self.cohort_seconds,
                               end_s=(index + 1) * self.cohort_seconds,
                               generated=0, completed=0, on_time_completed=0,
                               failed=0, deadline_misses=0, pending=0,
                               completion_latency_ms_sum=0., max_completion_latency_ms=0.,
                               mature_after_s=(index + 1) * self.cohort_seconds,
                               offloaded=0, local=0, offload_fraction_sum=0.,
                               offload_fraction_min=self.offload[uid],
                               offload_fraction_max=self.offload[uid],
                               drx_config=dict(self.drx[uid]), mixed_drx=False,
                               **OFFLOADED_COUNTER_DEFAULTS)
            # Old in-flight tasks can outlive this telemetry horizon. Their
            # ordinary task counters remain exact; an evicted bucket is not
            # recreated by a late completion and cannot grow memory unbounded.
            while len(rows) > 16:
                del rows[next(iter(rows))]
        row = rows[index]
        row['generated'] += 1
        row['pending'] += 1
        row['mature_after_s'] = max(row['mature_after_s'], row['end_s'] + task['deadline_s'])
        row['offload_fraction_sum'] += self.offload[uid]
        row['offload_fraction_min'] = min(row['offload_fraction_min'], self.offload[uid])
        row['offload_fraction_max'] = max(row['offload_fraction_max'], self.offload[uid])
        row['mixed_drx'] |= row['drx_config'] != self.drx[uid]
        self.cohort_dirty.add(uid)

    def _cohort_count(self, task, key, amount=1):
        index = task.get('cohort_index')
        row = self.cohorts.get(task['uid'], {}).get(index)
        if row is not None:
            row[key] += amount
            self.cohort_dirty.add(task['uid'])

    def _cohort_finish(self, task, outcome, latency=None):
        row = self.cohorts.get(task['uid'], {}).get(task.get('cohort_index'))
        if row is None:
            return
        row['pending'] -= 1
        if outcome == 'completed':
            row['completed'] += 1
            row['on_time_completed'] += latency <= task['deadline_s'] + 1e-12
            row['completion_latency_ms_sum'] += latency * 1000
            row['max_completion_latency_ms'] = max(row['max_completion_latency_ms'], latency * 1000)
        else:
            row['failed'] += 1
        if task.get('offloaded', False):
            row['offloaded_pending'] -= 1
            if outcome == 'completed':
                row['offloaded_completed'] += 1
                row['offloaded_on_time_completed'] += latency <= task['deadline_s'] + 1e-12
                row['offloaded_completion_latency_ms_sum'] += latency * 1000
            else:
                row['offloaded_failed'] += 1
        if row['generated'] != row['completed'] + row['failed'] + row['pending']:
            raise RuntimeError('Task cohort conservation failed')
        if row['offloaded_generated'] != row['offloaded_completed'] + row['offloaded_failed'] + row['offloaded_pending']:
            raise RuntimeError('Offloaded task cohort conservation failed')
        self.cohort_dirty.add(task['uid'])

    def _cohort_reports(self, uid):
        if not self.cohort_slots or uid not in self.profiles:
            return None
        if uid not in self.cohort_cache or uid in self.cohort_dirty:
            result = []
            for row in list(self.cohorts.get(uid, {}).values())[-8:]:
                public = dict(row)
                public['drx_config'] = None if row['mixed_drx'] else dict(row['drx_config'])
                public['offload_fraction'] = row['offload_fraction_min'] if row['offload_fraction_min'] == row['offload_fraction_max'] else None
                public['mean_offload_fraction'] = row['offload_fraction_sum'] / row['generated']
                public['mixed_policy'] = row['mixed_drx'] or public['offload_fraction'] is None
                result.append(public)
            self.cohort_cache[uid] = result
            self.cohort_dirty.discard(uid)
        return self.cohort_cache[uid]

    def _enqueue(self, queue, task, stage, amount):
        if self.stage_metrics_enabled:
            timing_stage = 'upload' if stage == 'upload' else (
                'edge_compute' if task.get('offloaded', False) else 'local_compute')
            self._stage_transition(task, timing_stage, self.now)
        if len(queue) >= self.max_queue:
            self._finish(task, 'queue_overflow')
            return
        task.update(stage=stage, remaining=amount)
        queue.append(task['id'])

    def _later(self, task, kind, now):
        if self.stage_metrics_enabled:
            self._stage_transition(task, 'backhaul_to_edge' if kind == 'edge' else 'backhaul_to_result', now)
        task['stage'] = 'backhaul'
        heapq.heappush(self.future, (now + self.backhaul_s, task['id'], kind, task['id']))

    def _serve(self, queue, budget, complete, allowed=None):
        consumed = Counter()
        skipped = 0
        initial_budget = budget
        while queue and budget > 1e-12:
            task = self.tasks[queue[0]]
            if allowed is not None and not allowed(task):
                queue.rotate(-1)
                skipped += 1
                if skipped >= len(queue):
                    break
                continue
            skipped = 0
            used = min(task['remaining'], budget)
            task['remaining'] -= used
            budget -= used
            consumed[task['uid']] += used
            if self.stage_metrics_enabled:
                timing = self.task_timings[task['id']]
                # Work/rate is the active fraction of this queue's real slot
                # budget. Unused fractions remain slot-boundary quantization
                # or FIFO waiting, not fabricated extra service.
                timing['service_s'][timing['stage']] += used / initial_budget * self.dt
            if task['remaining'] <= 1e-12:
                queue.popleft()
                complete(task)
        return consumed

    def _ul_allowed(self, task):
        uid = task['uid']
        connection = self.state.connections[uid]
        paper = getattr(self.state, 'paper', None)
        reference = paper.raw.get(uid, {}).get(connection.serving_cell_id, {}) if paper else {}
        return self.users[uid].ue_radio_available and connection.connection_state == 'connected' and not reference.get('no_path', False)

    def _upload_block_reason(self, task):
        """Read the same gate as _ul_allowed without touching its decisions."""
        uid = task['uid']
        connection = self.state.connections[uid]
        if not self.users[uid].ue_radio_available:
            return 'radio_unavailable'
        if connection.connection_state != 'connected':
            return 'not_connected'
        paper = getattr(self.state, 'paper', None)
        reference = paper.raw.get(uid, {}).get(connection.serving_cell_id, {}) if paper else {}
        return 'no_path' if reference.get('no_path', False) else None

    def _inject(self, task):
        if self.stage_metrics_enabled:
            self._stage_transition(task, 'downlink', self.now)
        ue = self.users[task['uid']]
        lc = ue.logical_channels[0]
        task.update(stage='downlink', result_remaining=task['result_bytes'])
        self.result_arrivals[ue.ue_id] += task['result_bytes']
        admitted = ue.packet_ledger.add(lc, task['result_bytes'], self.slot,
                                      packet_size=200, service_task_id=task['id'])
        ue.queue_bytes = sum(c.queue_bytes for c in ue.logical_channels)
        if admitted != task['result_bytes']:
            self._finish(task, 'result_overflow')

    def delivered(self, task_id, amount, slot):
        task = self.tasks.get(task_id)
        if task is None:
            return
        if task['stage'] != 'downlink' or amount > task['result_remaining']:
            raise RuntimeError('Invalid task result delivery')
        task['result_remaining'] -= amount
        if not task['result_remaining']:
            self._finish(task, 'completed', slot * self.dt)

    def _finish(self, task, outcome, now=None):
        uid = task['uid']
        if outcome == 'completed':
            latency = max(0., now - task['created'])
            if latency > task['deadline_s'] + 1e-12 and not task['overdue']:
                self.stats[uid]['deadline_misses'] += 1
                self._cohort_count(task, 'deadline_misses')
            self.stats[uid]['completed'] += 1
            self.stats[uid]['completion_latency_ms_sum'] += latency * 1000
            self.stats[uid]['max_completion_latency_ms'] = max(self.stats[uid]['max_completion_latency_ms'], latency * 1000)
            self.stats[uid]['on_time_completed' if latency <= task['deadline_s'] + 1e-12 else 'late_completed'] += 1
        else:
            self.stats[uid]['failed'] += 1
            self.stats[uid][outcome] += 1
        if task.get('offloaded', False):
            self.stats[uid]['offloaded_pending'] -= 1
            if outcome == 'completed':
                self.stats[uid]['offloaded_completed'] += 1
                self.stats[uid]['offloaded_on_time_completed'] += latency <= task['deadline_s'] + 1e-12
                self.stats[uid]['offloaded_completion_latency_ms_sum'] += latency * 1000
            else:
                self.stats[uid]['offloaded_failed'] += 1
        self._cohort_finish(task, outcome, latency if outcome == 'completed' else None)
        event = dict(kind='task_' + outcome, task_id=task['id'], ue_id=uid,
                                time_s=self.now if now is None else now,
                                latency_ms=None if now is None else (now - task['created']) * 1000,
                                created_s=task['created'], deadline_s=task['deadline_s'],
                     cohort_index=task.get('cohort_index'), offloaded=task.get('offloaded'))
        if self.stage_metrics_enabled:
            event['task_stage_timing'] = self._finish_stage_timing(task, outcome, self.now if now is None else now)
        self.events.append(event)
        del self.tasks[task['id']]

    def _stage_transition(self, task, stage, now):
        timing = self.task_timings[task['id']]
        elapsed = now - timing['entered_s']
        if elapsed < -1e-9:
            raise RuntimeError('Task diagnostic stage clock moved backwards')
        timing['stage_s'][timing['stage']] += max(0., elapsed)
        timing.update(stage=stage, entered_s=now)
        self.stage_clock_s = max(self.stage_clock_s, now)

    def _stage_snapshot(self, task, now, *, terminal=False):
        timing = self.task_timings[task['id']]
        elapsed = max(0., now-task['created']) * 1000
        stages = {s: timing['stage_s'][s]*1000 for s in TASK_STAGES}
        stages[timing['stage']] += max(0., now-timing['entered_s'])*1000
        service = {s: timing['service_s'][s]*1000 for s in TASK_SERVICE_STAGES}
        reasons = {s: timing['upload_blocked_s'][s]*1000 for s in UPLOAD_BLOCK_REASONS}
        blocked = sum(reasons.values())
        if not math.isclose(sum(stages.values()), elapsed, abs_tol=1e-6, rel_tol=1e-10):
            raise RuntimeError('Task diagnostic stages do not conserve elapsed time')
        if any(service[s]+(blocked if s=='upload' else 0.) > stages[s]+1e-6
               for s in TASK_SERVICE_STAGES):
            raise RuntimeError('Task diagnostic service exceeds stage residency')
        return dict(schema=TASK_STAGE_SCHEMA, elapsed_ms=elapsed, stage_ms=stages,
                    service_ms=service,
                    wait_or_quantization_ms={s: max(0.,stages[s]-service[s]-(blocked if s=='upload' else 0.))
                                             for s in TASK_SERVICE_STAGES},
                    upload_blocked_ms=blocked, upload_blocked_by_reason_ms=reasons,
                    terminal_stage=timing['stage'] if terminal else None,
                    current_stage=timing['stage'], observed_until_s=now,
                    clock='simulation slot boundaries; active UL/CPU service = actual work / configured rate',
                    downlink_scope='residency includes FIFO waiting, HARQ and radio service; no separate PHY active-service duration inferred')

    def _finish_stage_timing(self, task, outcome, now):
        self.stage_clock_s = max(self.stage_clock_s, now)
        timing = self._stage_snapshot(task, now, terminal=True)
        kind = 'completed' if outcome == 'completed' else 'failed'
        metrics = self.terminal_stage_metrics[task['uid']][kind]
        route = 'offloaded' if task.get('offloaded') is True else (
            'local' if task.get('offloaded') is False else 'unassigned')
        _add_stage_timing(metrics['all'],timing,outcome)
        _add_stage_timing(metrics['routes'].setdefault(route,_stage_aggregate()),timing,outcome)
        del self.task_timings[task['id']]
        return timing

    def _stage_reports(self):
        pending = {uid: dict(all=_stage_aggregate(), by_stage=Counter(), max_age_ms=0.)
                   for uid in self.profiles}
        for task in self.tasks.values():
            snapshot = self._stage_snapshot(task,self.stage_clock_s)
            row = pending[task['uid']]
            _add_stage_timing(row['all'],snapshot)
            row['by_stage'][snapshot['current_stage']] += 1
            row['max_age_ms'] = max(row['max_age_ms'],snapshot['elapsed_ms'])
        reports = {}
        for uid in self.profiles:
            result = dict(schema=TASK_STAGE_SCHEMA, observed_until_s=self.stage_clock_s,
                          scope='Cumulative whole-run task timing; terminal populations and still-pending ages are separate',
                          deadline_semantics=TASK_DEADLINE_SEMANTICS)
            for kind in ('completed','failed'):
                value = self.terminal_stage_metrics[uid][kind]
                result[kind] = _stage_aggregate_report(value['all'])
                result[kind]['by_route'] = {route:_stage_aggregate_report(v)
                                           for route,v in value['routes'].items()}
            value = pending[uid]
            result['pending'] = dict(_stage_aggregate_report(value['all']),
                by_stage={s:value['by_stage'][s] for s in TASK_STAGES},
                max_age_ms=value['max_age_ms'])
            reports[uid] = result
        return reports

    def _global_stage_report(self):
        result = dict(schema=TASK_STAGE_SCHEMA, observed_until_s=self.stage_clock_s,
            scope='Cumulative whole-run task timing; completed, failed and pending are separate populations',
            deadline_semantics=TASK_DEADLINE_SEMANTICS)
        for kind in ('completed','failed'):
            total=_stage_aggregate();routes={}
            for uid in self.users:
                value=self.terminal_stage_metrics[uid][kind]
                _merge_stage_aggregate(total,value['all'])
                for route,v in value['routes'].items():
                    _merge_stage_aggregate(routes.setdefault(route,_stage_aggregate()),v)
            result[kind]=_stage_aggregate_report(total)
            result[kind]['by_route']={route:_stage_aggregate_report(v) for route,v in routes.items()}
        total=_stage_aggregate();stages=Counter();max_age=0.;tasks=[]
        for task in self.tasks.values():
            snapshot=self._stage_snapshot(task,self.stage_clock_s)
            _add_stage_timing(total,snapshot)
            stages[snapshot['current_stage']]+=1
            max_age=max(max_age,snapshot['elapsed_ms'])
            tasks.append(dict(task_id=task['id'],ue_id=task['uid'],created_s=task['created'],
                              offloaded=task.get('offloaded'),task_stage_timing=snapshot))
        result['pending']=dict(_stage_aggregate_report(total),
            by_stage={s:stages[s] for s in TASK_STAGES},max_age_ms=max_age,tasks=tasks)
        return result

    def after_phy(self, trace):
        start = time.perf_counter()
        now = self.slot * self.dt
        if self.stage_metrics_enabled:
            self.stage_clock_s = now
        local_work = Counter()
        ul_bits = Counter()
        for uid, queue in self.local.items():
            local_work.update(self._serve(queue, self.local_rate * self.dt,
                                         lambda task: self._finish(task, 'completed', now)))
        for cid, queue in self.edge.items():
            work = self._serve(queue, self.edge_rate * self.dt,
                               lambda task: self._later(task, 'result', now))
            self.server_stats[cid]['executed_gflop'] += sum(work.values())
        # Uploads use the current radio cell after HO; the chosen MEC server
        # remains pinned for the task, as with IP-session continuity.
        moves = [(cid, tid, self.state.connections[self.tasks[tid]['uid']].serving_cell_id)
                 for cid, queue in self.upload.items() for tid in queue
                 if self.state.connections[self.tasks[tid]['uid']].serving_cell_id != cid]
        for source, tid, target in moves:
            self.upload[source].remove(tid)
            self.upload[target].append(tid)
        for cid, queue in self.upload.items():
            if self.stage_metrics_enabled:
                for tid in queue:
                    task=self.tasks[tid]
                    reason=self._upload_block_reason(task)
                    if reason is not None:
                        self.task_timings[tid]['upload_blocked_s'][reason] += self.dt
            sent = self._serve(queue, self.ul_rate * self.dt,
                               lambda task: self._later(task, 'edge', now),
                               self._ul_allowed)
            ul_bits.update(sent)
            self.server_stats[cid]['uploaded_bits'] += sum(sent.values())
        for task in self.tasks.values():
            if not task['overdue'] and now - task['created'] > task['deadline_s'] + 1e-12:
                task['overdue'] = True
                self.stats[task['uid']]['deadline_misses'] += 1
                self._cohort_count(task, 'deadline_misses')
        rows = {r['ue_id']: r for r in trace.get('ues', [])}
        for uid, ue in self.users.items():
            p = self.power
            row = rows.get(uid, {})
            transmitted = bool(row.get('scheduled') and row.get('ack_source') != 'no_path' and row.get('allocated_re', 0))
            rx = min(1., len(row.get('resource_allocation', {}).get('symbols', [])) / 14.) if transmitted else 0.
            # Older traces expose allocated OFDM symbols under a different key.
            if transmitted and not rx:
                rx = min(1., len(row.get('resource_allocation', {}).get('ofdm_symbols', [])) / 14.)
                if not rx:
                    rx = 1.
            tx = min(1., ul_bits[uid] / (self.ul_rate * self.dt))
            active = ue.ue_radio_available
            if transmitted and not active:
                raise RuntimeError('DL scheduled during UE DRX sleep')
            if (transmitted and not row.get('is_retx')) or tx:
                self.inactivity_until[uid] = self.slot + 1 + round(self.drx[uid]['inactivity_ms'] / self.dt_ms)
            radio = self.radio_states[uid]
            base = p['monitor_w'] if active else p['waking_w'] if radio == 'waking' else p['sleep_w']
            radio_j = (base + max(0., p['receive_w'] - base) * rx + max(0., p['transmit_w'] - base) * tx) * self.dt
            compute_j = (p['cpu_idle_w'] + p['cpu_compute_w'] * local_work[uid] / (self.local_rate * self.dt)) * self.dt
            self.energy[uid]['radio_j'] += radio_j
            self.energy[uid]['compute_j'] += compute_j
            # Transition energy is a discrete impulse, included in slot power.
            total = sum(self.energy[uid][k] for k in ('radio_j', 'compute_j', 'transition_j'))
            previous = self.energy[uid].get('total_j', 0.)
            self.energy[uid]['total_j'] = total
            self.last_power[uid] = (total - previous) / self.dt
            self.active_slots[uid] += active
            self.radio_states[uid] = 'receive' if rx else 'transmit' if tx else radio
        self._reports()
        trace['ue_services'] = self.reports
        if self.events:
            trace['ue_service_events'] = list(self.events)
        self.wall_s += time.perf_counter() - start

    def _reports(self):
        pending = Counter(task['uid'] for task in self.tasks.values())
        offloaded_pending = Counter(task['uid'] for task in self.tasks.values() if task.get('offloaded', False))
        self.reports = {}
        stage_reports = self._stage_reports() if self.stage_metrics_enabled else {}
        for uid in self.users:
            s = self.stats[uid]
            if s['generated'] != s['completed'] + s['failed'] + pending[uid]:
                raise RuntimeError('Task conservation failed')
            if (s['offloaded_pending'] != offloaded_pending[uid] or
                    s['offloaded_generated'] != s['offloaded_completed'] + s['offloaded_failed'] + offloaded_pending[uid]):
                raise RuntimeError('Offloaded task conservation failed')
            metrics = dict(s, generated=s['generated'], completed=s['completed'], failed=s['failed'], pending=pending[uid],
                           success_ratio=s['on_time_completed'] / s['generated'] if s['generated'] else None,
                           mean_completion_latency_ms=s['completion_latency_ms_sum'] / s['completed'] if s['completed'] else None,
                           offloaded_mean_completion_latency_ms=s['offloaded_completion_latency_ms_sum'] / s['offloaded_completed'] if s['offloaded_completed'] else None,
                           offloaded_success_ratio=s['offloaded_on_time_completed'] / s['offloaded_generated'] if s['offloaded_generated'] else None)
            self.reports[uid] = dict(ue_energy_total_j=self.energy[uid].get('total_j', 0.),
                                    ue_power_w=self.last_power[uid], ue_radio_state=self.radio_states[uid],
                                    ue_active_fraction=self.active_slots[uid] / max(1, self.slot),
                                    ue_task_metrics=metrics if uid in self.profiles else None,
                                    ue_task_cohorts=self._cohort_reports(uid),
                                    ue_service_policy=dict(drx=dict(self.drx[uid]), offload_fraction=self.offload[uid]))
            if self.stage_metrics_enabled:
                self.reports[uid]['ue_task_stage_metrics'] = stage_reports[uid] if uid in self.profiles else None

    def summary(self):
        result = dict(enabled=True, scope='illustrative UE radio + CPU energy; conservative connected DRX; abstract shared UL + FIFO compute + actual NR/SYS DL task results',
                    power_calibration='scenario_assumptions_not_device_measurements',
                    per_ue=deepcopy(self.reports), energy_components=deepcopy(self.energy),
                    servers={cid: dict(s, queued=len(self.edge[cid]), upload_queued=len(self.upload[cid])) for cid, s in self.server_stats.items()},
                    module_wall_s=self.wall_s, pending_tasks=len(self.tasks))
        if self.stage_metrics_enabled:
            result['task_stage_metrics'] = self._global_stage_report()
        return result

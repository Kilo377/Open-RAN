"""Attach real recorded runs or atomic live telemetry to the contract graph.

Recorded inputs are streamed once, reduced to bounded UI frames and never
modified. Importing this module needs only the Python standard library.
"""
from collections import deque
from functools import lru_cache
from copy import deepcopy
import gzip
import heapq
import json
import math
from pathlib import Path
from uuid import uuid4
import time

from smo.contracts import PARAMETERS, load_json
from smo.telemetry import finite_json
from smo.population import task_members, urllc_members

APP_LOGS = {
    'ue_energy_rf': 'energy_rf_events', 'ap_energy_rf': 'ap_rf_events',
    'throughput_scheduler_mlp': 'throughput_mlp_events',
    'latency_scheduler_rf': 'latency_rf_events', 'latency_urllc_rf': 'urllc_rf_events',
    'mec_offload_budget_qgbdt': 'budget_qgbdt_events',
}


def records(folder, name):
    for suffix, opener in (('.jsonl.gz', gzip.open), ('.jsonl', open)):
        path = Path(folder)/(name+suffix)
        if path.exists():
            with opener(path, 'rt') as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
            return


@lru_cache(maxsize=1)
def energy_action_profiles():
    metadata=load_json(Path(__file__).resolve().parents[1]/'ric/near_rt/pools/chongyu/x01.json')
    return [(profile,value)for profile in metadata['rcp']['ue_drx_configs'].get('profiles',[])for value in metadata['rcp']['ue_offload_fraction']['values']]


def proposal(row):
    """Keep actual logged suggestions separate from execution receipts."""
    result = {k: deepcopy(row[k]) for k in ('slot_id', 'ue_id', 'cell_id', 'selection_version',
        'action', 'action_name', 'reason', 'status', 'p', 'scheduler_mode', 'ue_scheduler_weight', 'implementation_id', 'instance_id',
        'ue_ids', 'snapshot', 'target_ue_ids', 'proposed_offload_fraction', 'proposed_drx', 'drx_deployment') if k in row}
    if row.get('implementation_id') == 'ue_energy_rf' and type(row.get('action')) is int:
        actions=energy_action_profiles()
        if 0<=row['action']<len(actions):
            drx,offload=actions[row['action']]
            result['proposed_drx']=deepcopy(drx)
            result['proposed_offload_fraction']=offload
        result.setdefault('drx_deployment','proposal_only; use executed ue_service_policy for actual state')
    return result


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def metric(value=None, *, entities=None, source=None, window='instantaneous', start=None, end=None):
    return dict(value=value, by_entity=entities or {}, status='observed' if value is not None or entities else 'unavailable',
        source=source, window=window, window_start_slot=start, sampled_at_slot=end)


def task_profile(config, uid):
    return config.get('configs', {}).get('phy', {}).get('rt_sys', {}).get('ue_services', {}).get('mec', {}).get('tasks_by_ue', {}).get(uid)


class FrameBuilder:
    def __init__(self, config=None, window_ms=100, *, population=False):
        self.config = config or {}
        self.population = population
        self.dt = self.config.get('manifest', {}).get('slot_duration_ms', .5)
        self.window_slots = max(1, round(window_ms/self.dt))
        self.history = deque()
        self.energy_last = {}
        self.energy_epochs = {}
        self.executed = {}
        self.proposals = {}
        self.last_applied = {}
        self.selection = {}
        self.enabled_since = {}

    def observe_selection(self, row):
        if row.get('status') != 'applied':
            return
        key = (row['cell_id'], row['instance_id'])
        if row.get('operation') == 'added':
            self.selection[key] = row['selection_version']
            self.enabled_since[key] = row['slot_id']
        elif row.get('operation') == 'removed':
            self.selection.pop(key, None)
            self.enabled_since.pop(key, None)

    def observe_receipt(self, row):
        if row.get('status') != 'applied':
            return
        # A rejected/queued request never changes executed state.
        for cid, after in row.get('after', {}).items():
            self.executed[cid] = {k: deepcopy(v) for k, v in after.items()
                if k in ('scheduler_mode', 'ue_scheduler_weights', 'ue_service_policies', 'cell_sleep_mode')}
        for p in row.get('provenance', []):
            ident = p.get('implementation_id')
            if ident:
                self.last_applied[ident] = dict(slot=row.get('apply_slot', row.get('slot_id')),
                    target_cell_ids=row.get('target_cell_ids', []), selection_version=p.get('selection_version'),
                    fields=[f['field'] for f in row.get('fields', [])], status='applied')

    def observe_proposal(self, ident, row):
        if row.get('kind') == 'decision' or 'implementation_id' in row:
            scope = row.get('ue_id', row.get('cell_id', '*'))
            self.proposals.setdefault(ident, {})[scope] = proposal(row)

    @staticmethod
    def counters(users, services, ap):
        by_ue = {}
        for u in users:
            p = u.get('packet', {})
            by_ue[u['ue_id']] = dict(delivered=p.get('delivered_bytes'),
                failures=p.get('deadline_failed_packets'), overflows=p.get('overflow_packets'),
                completed=p.get('completed_packets'), latency_sum=p.get('completion_latency_ms_sum'),
                offloaded=(services.get(u['ue_id'], {}).get('ue_task_metrics') or {}).get('offloaded_generated'),
                energy_j=services.get(u['ue_id'], {}).get('ue_energy_total_j'))
        return dict(ues=by_ue, ap={cid: a.get('ap_energy_total_j') for cid, a in ap.items()})

    def observe_energy_counters(self, counters):
        # A counter reset or missing observation invalidates every window that
        # straddles it, even if the counter later overtakes its old baseline.
        # Epochs make that check O(UEs) per frame instead of scanning the window.
        by_ue = counters['ues']
        for uid in self.energy_epochs.keys() | by_ue.keys():
            cur = by_ue.get(uid, {}).get('energy_j')
            epoch = self.energy_epochs.get(uid, 0)
            prev = self.energy_last.get(uid)
            valid_cur = number(cur) and cur >= 0
            valid_prev = number(prev) and prev >= 0
            if uid in self.energy_epochs and (not valid_cur or not valid_prev or cur < prev):
                epoch += 1
            self.energy_epochs[uid] = epoch
            self.energy_last[uid] = cur
            if uid in by_ue:
                by_ue[uid]['energy_epoch'] = epoch

    def frame(self, slot, users, services, ap, bindings, *, weights=None, catalog=(), state_source='recorded_execution'):
        slot = int(slot)
        if self.history and slot < self.history[-1][0]:
            raise ValueError('A new simulation requires a new FrameBuilder')
        counters = self.counters(users, services, ap)
        self.observe_energy_counters(counters)
        if not self.history or slot != self.history[-1][0]:
            self.history.append((slot, counters))
        # Keep the point immediately before the requested counter-delta window.
        while len(self.history) > 2 and self.history[1][0] <= slot-self.window_slots:
            self.history.popleft()
        old_slot, old = self.history[0]
        elapsed = (slot-old_slot)*self.dt/1000

        def delta(uid, key):
            cur = counters['ues'].get(uid, {}).get(key)
            prev = old['ues'].get(uid, {}).get(key)
            if key == 'energy_j' and counters['ues'].get(uid, {}).get('energy_epoch') != old['ues'].get(uid, {}).get('energy_epoch'):
                return None
            return cur-prev if elapsed > 0 and number(cur) and number(prev) and cur >= prev else None

        def total(values):
            values = list(values)
            return sum(values) if values and all(number(v) for v in values) else None

        def ratio(a, b):
            return a/b if number(a) and number(b) and b > 0 else None

        kpis = {}
        observed_ids = [u['ue_id'] for u in users]
        configured = self.config.get('manifest', {}).get('ues') or self.config.get('configs', {}).get('ue', {}).get('ues', [])
        ids = [u['ue_id'] for u in configured] if self.population and configured else observed_ids
        urllc = urllc_members(self.config) if self.population else [u for u in ('ue_2', 'ue_6') if u in ids]
        task_ids = task_members(self.config, services) if self.population else ['ue_0', 'ue_4']
        energy_ids = ids if self.population else ['ue_0', 'ue_4']
        complete_urllc = bool(urllc) and set(urllc) <= set(observed_ids) if self.population else len(urllc) == 2
        # Preserve missing members as missing aggregates, rather than silently
        # reporting a subset as the complete target group.
        def put(key, values, aggregate=None, window='instantaneous'):
            observed = {k: v for k, v in values.items() if v is not None}
            kpis['kpi:'+key] = metric(aggregate, entities=observed, source=state_source,
                window=window, start=old_slot if window == 'counter_delta' else None, end=slot)

        rates = {uid: ratio(delta(uid, 'delivered'), elapsed)*8/1e6 if delta(uid, 'delivered') is not None else None for uid in ids}
        put('network_goodput_mbps', rates, total(rates.values()), 'counter_delta')
        put('per_ue_goodput_mbps', rates, ratio(total(rates.values()), len(ids)) if self.population else None, 'counter_delta')
        energies = {uid: services.get(uid, {}).get('ue_energy_total_j') for uid in energy_ids}
        put('ue_energy_j', energies, total(energies.values()), 'cumulative')
        ue_powers = {uid: ratio(delta(uid, 'energy_j'), elapsed) for uid in energy_ids}
        put('ue_power_w', ue_powers, total(ue_powers.values()), 'counter_delta')
        powers = {cid: ratio(a.get('ap_energy_total_j')-old['ap'].get(cid), elapsed)
                  if number(a.get('ap_energy_total_j')) and number(old['ap'].get(cid)) and a['ap_energy_total_j'] >= old['ap'][cid] else None for cid, a in ap.items()}
        put('ap_power_w', powers, total(powers.values()), 'counter_delta')
        failure_rates = {uid: ratio(total((delta(uid, 'failures'), delta(uid, 'overflows'))), elapsed) for uid in urllc}
        put('urllc_failure_events_per_s', failure_rates, total(failure_rates.values()) if complete_urllc else None, 'counter_delta')
        failures = {u['ue_id']: (u.get('packet', {}).get('deadline_failed_packets'), u.get('packet', {}).get('overflow_packets'), u.get('packet', {}).get('admitted_packets')) for u in users if u['ue_id'] in urllc}
        fnum = total(total(v[:2]) for v in failures.values())
        fden = total(total(v[1:]) for v in failures.values())
        put('urllc_failure_fraction', {uid: ratio(total(v[:2]), total(v[1:])) for uid, v in failures.items()}, ratio(fnum, fden) if complete_urllc else None, 'cumulative')
        for key, subset in (('urllc_packet_latency_ms', urllc), ('packet_latency_ms', ids)):
            by = {uid: ratio(delta(uid, 'latency_sum'), delta(uid, 'completed')) for uid in subset}
            mean = ratio(total(delta(uid, 'latency_sum') for uid in subset), total(delta(uid, 'completed') for uid in subset))
            put(key, by, mean, 'counter_delta')
            kpis['kpi:'+key]['sample_count'] = total(delta(uid, 'completed') for uid in subset)
        for key, counter_sum, count in (
            ('mec_offloaded_latency_ms', 'offloaded_completion_latency_ms_sum', 'offloaded_completed'),
            ('task_latency_ms', 'completion_latency_ms_sum', 'completed'),
            ('task_success_ratio', 'on_time_completed', 'generated')):
            tasks = {uid: services.get(uid, {}).get('ue_task_metrics') or {} for uid in task_ids}
            values = {uid: ratio(t.get(counter_sum), t.get(count)) for uid, t in tasks.items()}
            if key != 'task_success_ratio':
                # Older service reports only serialized the per-UE mean.
                mean_name = 'offloaded_mean_completion_latency_ms' if key.startswith('mec_offloaded') else 'mean_completion_latency_ms'
                for uid, t in tasks.items():
                    if values[uid] is None and number(t.get(count)) and t[count] > 0:
                        values[uid] = t.get(mean_name)
                sums = [values[uid]*t[count] if values[uid] is not None else 0 if t.get(count) == 0 else None for uid, t in tasks.items()]
            else:
                for uid, t in tasks.items():
                    if values[uid] is None and number(t.get('success_ratio')):
                        values[uid] = t['success_ratio']
                sums = [values[uid]*t[count] if values[uid] is not None else 0 if t.get(count) == 0 else None for uid, t in tasks.items()]
            put(key, values, ratio(total(sums), total(t.get(count) for t in tasks.values())), 'cumulative')
            kpis['kpi:'+key]['sample_count'] = total(t.get(count) for t in tasks.values())
        queues = {u['ue_id']: u.get('queue_bytes') for u in users}
        hol = {u['ue_id']: u.get('packet', {}).get('hol_delay_ms') for u in users}
        overflow = {u['ue_id']: u.get('packet', {}).get('overflow_bytes') for u in users}
        complete_population = set(observed_ids) == set(ids)
        put('queue_bytes', queues, total(queues.values()) if complete_population else None)
        put('hol_delay_ms', hol, max(hol.values()) if complete_population and hol and all(number(v) for v in hol.values()) else None)
        put('queue_overflow_bytes', overflow, total(overflow.values()) if complete_population else None, 'cumulative')
        active = {uid: services.get(uid, {}).get('ue_active_fraction') for uid in energy_ids}
        put('ue_active_fraction', active, ratio(total(active.values()), len(active)), 'cumulative')
        avail = {cid: row.get('ap_available_fraction') for cid, row in ap.items()}
        put('ap_available_fraction', avail, ratio(total(avail.values()), len(avail)), 'cumulative')
        profile = task_profile(self.config, 'ue_4')
        p = services.get('ue_4', {}).get('ue_service_policy', {}).get('offload_fraction')
        expected = (1000/profile['interval_ms']*profile['compute_gflop']*p if profile and profile.get('interval_ms', 0) > 0 and number(p) else None)
        realized = (ratio(delta('ue_4', 'offloaded'), elapsed)*profile['compute_gflop'] if profile and delta('ue_4', 'offloaded') is not None else None)
        put('mec_expected_gflops', {'ue_4': expected}, expected)
        kpis['kpi:mec_expected_gflops']['source'] = 'analytic_configured_workload_from_executed_fraction'
        put('mec_realized_gflops', {'ue_4': realized}, realized, 'counter_delta')
        if self.population:
            expected_by, realized_by = {}, {}
            for uid in task_ids:
                profile = task_profile(self.config, uid)
                p = services.get(uid, {}).get('ue_service_policy', {}).get('offload_fraction')
                expected_by[uid] = (1000/profile['interval_ms']*profile['compute_gflop']*p
                    if profile and profile.get('interval_ms', 0) > 0 and number(p) else None)
                realized_by[uid] = (ratio(delta(uid, 'offloaded'), elapsed)*profile['compute_gflop']
                    if profile and delta(uid, 'offloaded') is not None else None)
            put('mec_expected_gflops', expected_by, total(expected_by.values()))
            kpis['kpi:mec_expected_gflops']['source'] = 'analytic_configured_workload_from_executed_fraction'
            put('mec_realized_gflops', realized_by, total(realized_by.values()), 'counter_delta')

        param_values = {p: {} for p in PARAMETERS}
        membership = {u['ue_id']: u.get('cell_id') for u in users}
        for cid, row in self.executed.items():
            for name in ('scheduler_mode', 'cell_sleep_mode'):
                if row.get(name) is not None:
                    param_values[name][cid] = row[name]
            param_values['ue_scheduler_weights'].update({uid: w for uid, w in row.get('ue_scheduler_weights', {}).items()
                if membership.get(uid) == cid})
            for uid, policy in row.get('ue_service_policies', {}).items():
                if membership.get(uid) != cid:
                    continue
                for field, key in (('ue_offload_fraction', 'offload_fraction'), ('ue_drx_configs', 'drx')):
                    if key in policy:
                        param_values[field][uid] = deepcopy(policy[key])
        for uid, row in services.items():
            policy = row.get('ue_service_policy') or {}
            for field, key in (('ue_offload_fraction', 'offload_fraction'), ('ue_drx_configs', 'drx')):
                if key in policy:
                    param_values[field][uid] = deepcopy(policy[key])
        if weights:
            param_values['ue_scheduler_weights'].update({k: v for k, v in weights.items() if v is not None})
        for cid, row in ap.items():
            if row.get('ap_policy') is not None:
                param_values['cell_sleep_mode'][cid] = row['ap_policy']
        if self.population:
            param_values['ue_offload_fraction'] = {uid:p for uid,p in param_values['ue_offload_fraction'].items() if uid in task_ids}
        states = dict(kpis)
        for p, values in param_values.items():
            unique = list({json.dumps(v, sort_keys=True): v for v in values.values()}.values())
            states['rcp:'+p] = metric(unique[0] if len(unique) == 1 else None, entities=values, source=state_source, end=slot)
        enabled = {i for groups in bindings.values() for names in groups.values()
                   for i in (names if isinstance(names, list) else [names]) if i}
        availability = {a['instance_id']: a for a in catalog}
        for ident in APP_LOGS:
            rows = list(self.proposals.get(ident, {}).values())
            cells = [cid for cid, groups in bindings.items() if any(ident in (names if isinstance(names, list) else [names]) for names in groups.values())]
            for cid, version in availability.get(ident, {}).get('selection_versions', {}).items():
                key = (cid, ident)
                if self.selection.get(key) != version:
                    known_before = key in self.selection
                    self.selection[key] = version
                    self.enabled_since[key] = slot if known_before else -1
            def current(row):
                if ident not in enabled:
                    return False
                cid = row.get('cell_id')
                if cid and cid not in cells:
                    return False
                if cid and row.get('selection_version') is not None:
                    token = self.selection.get((cid, ident))
                    return token is not None and row['selection_version'] == token
                # Global suggestions do not serialize selection tokens. A
                # strict later-slot check avoids treating an old same-slot
                # suggestion as new after OFF/ON.
                since = max((self.enabled_since.get((c, ident), -1) for c in cells), default=-1)
                return row.get('slot_id', -1) > since
            states['xapp:'+ident] = dict(enabled=ident in enabled, status='ON' if ident in enabled else 'OFF',
                active_cells=cells, current_proposals=deepcopy([r for r in rows if current(r)]),
                last_proposals=deepcopy(rows), proposed_at_slot=max((r.get('slot_id', 0) for r in rows), default=None),
                last_applied=deepcopy(self.last_applied.get(ident)),
                available=availability.get(ident, {}).get('available'),
                unavailable_reason=availability.get(ident, {}).get('unavailable_reason'), sampled_at_slot=slot)
        return finite_json(dict(slot=slot, time_ms=slot*self.dt, window_ms=(slot-old_slot)*self.dt,
            states=states, state_source=state_source))


def trace_users(row):
    result = []
    for u in row['ues']:
        packet = {k[len('packet_'):]: v for k, v in u.items() if k.startswith('packet_')}
        # PHY queue_bytes is captured before the slot's delivery. The packet
        # ledger is attached after delivery, matching the live publisher.
        result.append(dict(ue_id=u['ue_id'], cell_id=u['cell_id'],
            queue_bytes=packet.get('pending_bytes',u.get('queue_bytes')), packet=packet))
    return result


def replay(folder, max_frames=180, *, population=False, window_index=None):
    folder = Path(folder).resolve()
    config = load_json(folder/'config_snapshot.json')
    status = load_json(folder/'run_status.json')
    if not status.get('completed'):
        raise ValueError('Replay requires a completed run. Use --state for an active simulation.')
    total_slots = status['last_completed_slot']
    builder = FrameBuilder(config, population=population)
    stride = max(1, math.ceil(total_slots/max_frames))
    # Keep a bounded set of exact decision frames. A stride-only replay can
    # otherwise miss every captured proposal (e.g. stride 9 vs period 200).
    preview_slots=sorted({r['slot']for r in records(folder,'xapp_decision_previews')
        if type(r.get('slot'))is int and 1<=r['slot']<=total_slots})
    decision_slots=set(preview_slots[::max(1,math.ceil(len(preview_slots)/max_frames))])
    if preview_slots:decision_slots.add(preview_slots[-1])
    def suggestions(ident, name):
        for r in records(folder, name):
            if r.get('kind') == 'decision':
                yield (r['slot_id'], ident, proposal(r))
    # Retain only compact executed RCPs and provenance; full receipt records stay
    # in their original archives. No physical-layer arrays are loaded.
    def receipts():
        for r in records(folder, 'control_receipts'):
            if r.get('status') == 'applied':
                if window_index is not None:window_index.observe_receipt(r)
                after = {cid: {k: v for k, v in state.items() if k in ('scheduler_mode', 'ue_scheduler_weights', 'ue_service_policies', 'cell_sleep_mode')}
                         for cid, state in r.get('after', {}).items()}
                yield (r.get('apply_slot', r.get('slot_id', 0)), None,
                    dict(status='applied', after=after, provenance=r.get('provenance', []),
                         fields=r.get('fields', []), target_cell_ids=r.get('target_cell_ids', []),
                         apply_slot=r.get('apply_slot', r.get('slot_id'))))
    def selections():
        for r in records(folder, 'xapp_selection'):
            yield (r['slot_id'], '@selection', r)
    pending = iter(heapq.merge(receipts(), selections(), *(suggestions(ident, name) for ident, name in APP_LOGS.items()), key=lambda e: e[0]))
    event = next(pending, None)
    frames = []
    for row in records(folder, 'rt_sys_trace'):
        slot = row['slot_id']
        while event is not None and event[0] <= slot:
            _, ident, data = event
            if ident == '@selection':
                builder.observe_selection(data)
            elif ident:
                # The slim proposal has no kind field, by construction.
                builder.observe_proposal(ident, dict(kind='decision', **data))
            else:
                builder.observe_receipt(data)
            event = next(pending, None)
        services = row.get('ue_services', {})
        ap = row.get('ap_services', {}).get('cells', {})
        # All rows advance the counter baseline; only bounded UI frames remain.
        users = trace_users(row)
        if window_index is not None:
            window_index.observe(slot,users,services,ap,row.get('ue_service_events', []))
        frame = builder.frame(slot, users, services, ap, row.get('xapp_bindings', {}))
        if slot == 1 or slot % stride == 0 or slot == total_slots or slot in decision_slots:
            frames.append(frame)
    if not frames or frames[-1]['slot'] != total_slots:
        raise ValueError('Trace is incomplete; refusing to present it as a complete replay')
    return dict(mode='recorded', name=folder.name, stage='completed', source_sha256=config['source_sha256'],
        total_slots=total_slots, slot_duration_ms=builder.dt, frames=frames, read_only=True,
        note='Recorded execution telemetry; not a live run or delayed E2 measurement. No effect sizes fitted from this observational run.')


class LiveSource:
    def __init__(self, path, *, population=False):
        self.path = Path(path).resolve()
        config_path = self.path.parent/'config_snapshot.json'
        self.config = load_json(config_path) if config_path.exists() else {}
        self.population = population
        self.builder = FrameBuilder(self.config, population=population)
        self.run_id = None
        self.generation = uuid4().hex
        self.frame = None
        self.mtime = None

    def read(self):
        stamp = self.path.stat().st_mtime_ns
        data = load_json(self.path)
        run_id = data.get('xapp_test', {}).get('run_id')
        if run_id != self.run_id or self.frame and data.get('slot', 0) < self.frame['slot']:
            self.builder = FrameBuilder(self.config, population=self.population)
            self.run_id = run_id
            self.generation = uuid4().hex
        if stamp != self.mtime:
            control = data.get('control', {})
            for r in control.get('history', control.get('current_outcomes', [])):
                self.builder.observe_receipt(r)
            for cid, state in ((c['cell_id'], c) for c in data.get('cell_metrics', [])):
                self.builder.executed.setdefault(cid, {}).update(
                    {k: v for k, v in {**state.get('effective', {}), **state.get('scheduler', {})}.items()
                     if k in ('scheduler_mode', 'cell_sleep_mode')})
            x = data.get('xapp_test', {})
            for p in x.get('last_decisions', []):
                if p.get('implementation_id') in APP_LOGS:
                    self.builder.observe_proposal(p['implementation_id'], p)
            telemetry = data.get('graph_telemetry', {})
            self.frame = self.builder.frame(data.get('slot', 0), data.get('ues', []),
                telemetry.get('ue_services', {}), telemetry.get('ap_services', {}),
                x.get('active_sets', control.get('xapp_active_sets', {})),
                weights=telemetry.get('ue_scheduler_weights'), catalog=x.get('catalog', []),
                state_source='live_executed_telemetry')
            self.mtime = stamp
        return dict(mode='live' if data.get('stage') == 'simulating' else 'snapshot',
            run_id=self.run_id or self.generation,
            name=self.path.parent.name, stage=data.get('stage'), total_slots=data.get('total_slots'),
            slot_duration_ms=self.builder.dt, frames=[self.frame], read_only=True,
            wall_age_s=max(0, time.time()-data.get('updated_unix_s', self.path.stat().st_mtime)),
            note='Read-only execution telemetry. Missing legacy service fields stay unavailable; no values are fabricated.')

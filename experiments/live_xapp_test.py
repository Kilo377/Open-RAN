"""Run-local commands and observed experiment segments for a registered xApp.

The GUI only submits requests. Selection, arbitration and RAN execution remain
in the simulator. This module also has a stdlib-only client for the GUI venv.
"""
from collections import deque
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

APP = 'live_scheduler_test'
ROOT = Path(__file__).resolve().parents[1]
ARGUMENTS = ('scene', 'cells', 'ues', 'seed', 'slots', 'path_update_ms', 'platform',
    'device', 'rt_device', 'algebra_backend', 'config', 'deployment_id', 'traffic_load',
    'initial_queue_bytes', 'ue_speed_mps', 'feedback_mode', 'csi_error_nmse_db',
    'harq_combining', 'recording_mode', 'nr_execution', 'diagnostic_window_slots', 'channel_cache_root')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    os.replace(temporary, path)


def configure(configs, manifest, args):
    if not getattr(args, 'live_xapp_test', False):
        return False
    if getattr(args, 'phase', 'integration') != 'integration' or args.control_mode != 'baseline':
        raise ValueError('Live xApp testing requires an integration baseline with no other controllers')
    if getattr(args, 'xapp_test_compare_slots', 200) <= 0:
        raise ValueError('Comparison slot count must be positive')
    if getattr(args, 'xapp_test_cell', '*') not in ['*'] + [c['cell_id'] for c in manifest['cells']]:
        raise ValueError('Unknown xApp test cell')
    from ric.near_rt.xapp_library import IMPLEMENTATIONS
    cells = [c['cell_id'] for c in manifest['cells']]
    period = max(1, math.ceil(20 / manifest['slot_duration_ms']))
    apps = [dict(type=APP, name=APP, cell_ids=cells)]
    # Command-based mobility experiments cannot promise configuration withdrawal.
    apps += [dict(type='xapp_instance', instance_id=ident, implementation_id=ident,
        cell_ids=cells, slot_duration_ms=manifest['slot_duration_ms'], initial_bindings={})
        for ident, spec in sorted(IMPLEMENTATIONS.items()) if not spec['commands']]
    if getattr(args, 'xapp_test_app', APP) not in [APP] + [a['instance_id'] for a in apps[1:]]:
        raise ValueError('Unknown or command-based live test xApp')
    paper = configs.setdefault('phy', {}).setdefault('rt_sys', {}).setdefault('paper_interfaces', {})
    paper.update(enabled=True, rule_observations=True)
    paper.setdefault('connection_events', False)
    paper.setdefault('beam_reports', False)
    configs['ric'].update(rapps=[], xapps=apps, freeze_learning=True,
        run_rapps_on_start=False, xapp_interval_slots=period,
        rapp_interval_slots=math.ceil(1000 / manifest['slot_duration_ms']))
    return True


class LiveXAppSession:
    def __init__(self, simulator, args, output, state_path=None):
        from ric.near_rt.rule_based_v1.leases import RuleLeases
        self.simulator, self.args, self.output = simulator, args, Path(output)
        self.run_id = uuid.uuid4().hex
        self.manager = simulator.controller_manager
        self.apps = self.manager.instance_selection.apps()
        self.app = self.apps[getattr(args, 'xapp_test_app', APP)]
        self.dt = float(simulator.clock.slot_duration_ms) if hasattr(simulator.clock, 'slot_duration_ms') else float(args._live_slot_ms)
        self.period = self.manager.xapp_interval_slots
        self.settle_slots = self.period + max(0, simulator._config_delay_slots)
        self.manager.rule_leases = RuleLeases()
        from ric.near_rt.rule_based_v1.telemetry import executed_configs
        self.baseline = executed_configs(simulator.state)
        self.manager.rule_leases.baseline = deepcopy(self.baseline)
        self.receipts, self.segments = deque(maxlen=30), []
        self.pending_activation = None
        self.previous_packet = (0, 0.)
        self.command_dir = Path(state_path).with_suffix('.commands') if state_path else None
        if self.command_dir:
            self.command_dir.mkdir(exist_ok=True)
        self.events = self.output / 'xapp_test_events.jsonl'
        self.spec_path = self.output / 'xapp_test_spec.json'
        values = {k: str(v) if isinstance(v, Path) else v for k in ARGUMENTS
                  if (v := getattr(args, k, None)) is not None}
        atomic_json(self.spec_path, dict(arguments=values,
            slots=min(args.slots, getattr(args, 'xapp_test_compare_slots', 200))))
        self.change(getattr(args, 'xapp_test_start', 'off') == 'on', getattr(args, 'xapp_test_cell', '*'), 0)

    def unavailable_reason(self, app):
        if not app.contract.available:
            return 'Backend does not provide the declared KPI/action contract'
        required = set(app.contract.required_kpis)
        paper = getattr(self.simulator.state, 'paper', None)
        if app.name != APP and not paper:
            return 'Extended measured reports are disabled'
        if required & {'requested_dl_rate_bps', 'sla_status'} and any(
                getattr(u, 'paper_requested_dl_rate_bps', None) is None for u in self.simulator.state.ues):
            return 'Explicit service requirements are not configured for all UEs'
        if 'mobility_events' in required and not getattr(paper, 'events_enabled', False):
            return 'Connection event model is disabled'
        if 'beam_reference_reports' in required and not paper.config.get('beam_reports', False):
            return 'Beam reference reports are disabled'
        return None

    def change(self, enabled, cell, slot, app_id=None):
        from models.policy import PolicyUpdate
        known = set(self.baseline)
        if cell != '*' and cell not in known:
            raise ValueError('Unknown target cell')
        selection = self.manager.instance_selection
        before = deepcopy(selection.bindings or {})
        app = self.apps.get(app_id or self.app.name)
        if app is None:
            raise ValueError('Unregistered live test xApp')
        if enabled:
            reason = self.unavailable_reason(app)
            if reason:
                raise ValueError(reason)
            if any(ident != app.name for groups in before.values() for ident in groups.values()):
                raise ValueError('Disable the active xApp and wait for parameter restoration before switching')
        new = {cid: {app.category_id: app.name} for cid in (sorted(known) if cell == '*' else [cell])} if enabled else {}
        if new == before and self.segments:
            return 'unchanged'
        # A previous withdrawal must finish before its baseline is reused.
        if enabled:
            from ric.near_rt.rule_based_v1.telemetry import executed_configs
            current = executed_configs(self.simulator.state)
            for cid in set(new) - set(before):
                fields = {f for a in self.apps.values() for f in a.controlled_fields}
                for key in fields:
                    if current[cid].get(key) != self.baseline[cid].get(key):
                        raise ValueError('Wait for the previous disable to restore controlled parameters')
        accepted = self.manager.policy_store.apply_update(PolicyUpdate(
            source_name='live_xapp_ui', xapp_bindings=new), slot_id=slot)
        if accepted is False:
            raise ValueError('Selection policy was rejected')
        # Invalidate queued old-generation requests before the next slot applies them.
        selection.sync(slot, None)
        if enabled:self.app = app
        self.segments.append(dict(label=('ON / ' + cell) if enabled else 'OFF / baseline',
            app_id=app.name,
            enabled=enabled, cell=cell, start_slot=slot, first_counted_slot=slot + self.settle_slots + 1,
            slots=0, skipped_slots=0, delivered_bytes=0, queue_sum=0, hol_sum=0.,
            decodes=0, nacks=0, completed_packets=0, completion_delay_sum_ms=0.))
        return 'selection_admitted; awaiting normal control tick and delayed execution'

    def record_receipt(self, request, status, reason):
        event = dict(request_id=request.get('request_id'), operation=request.get('operation'),
            app=request.get('app'), received_at_slot=self.simulator.state.slot_id,
            status=status, reason=reason)
        self.receipts.append(event)
        with self.events.open('a') as stream:
            stream.write(json.dumps(event) + '\n')

    def cancel_activation(self, reason):
        if self.pending_activation:
            self.record_receipt(self.pending_activation, 'cancelled', reason)
            self.pending_activation = None

    def activate(self, request):
        """One click: withdraw the old binding, restore, then admit the new one."""
        app = self.apps[request['app']]
        if request['cell'] != '*' and request['cell'] not in self.baseline:
            raise ValueError('Unknown target cell')
        reason = self.unavailable_reason(app)
        if reason:raise ValueError(reason)
        self.cancel_activation('Replaced by a newer button click')
        slot = self.simulator.state.slot_id
        desired = {cid: {app.category_id: app.name} for cid in
            (sorted(self.baseline) if request['cell'] == '*' else [request['cell']])}
        if (self.manager.instance_selection.bindings or {}) == desired:
            return self.change(True, request['cell'], slot, app.name)
        self.change(False, '*', slot)
        self.pending_activation = {**request, 'deadline_slot': slot + 3 * self.settle_slots + self.period}
        return 'Restoring previous parameters; activation will continue automatically'

    def advance_activation(self):
        request = self.pending_activation
        if not request:return
        try:
            reason = self.change(True, request['cell'], self.simulator.state.slot_id, request['app'])
        except ValueError as error:
            if 'previous disable' in str(error) and self.simulator.state.slot_id <= request['deadline_slot']:
                return
            self.record_receipt(request, 'rejected', 'Could not complete switch: ' + str(error))
        else:
            self.record_receipt(request, 'accepted', reason)
        self.pending_activation = None

    def before_step(self):
        if not self.command_dir:
            self.advance_activation()
            return
        for path in sorted(self.command_dir.glob('*.json'))[:32]:
            request, status, reason = {}, 'rejected', ''
            try:
                if path.stat().st_size > 8192:
                    raise ValueError('Command too large')
                request = json.loads(path.read_text())
                if not isinstance(request, dict) or set(request) != {'run_id', 'request_id', 'operation', 'app', 'cell'}:
                    raise ValueError('Malformed command')
                if request['run_id'] != self.run_id or request['app'] not in self.apps:
                    raise ValueError('Stale run or unregistered app')
                if request['operation'] not in ('enable', 'disable', 'activate'):
                    raise ValueError('Unsupported operation')
                if request['operation'] == 'activate':
                    reason = self.activate(request)
                    status = 'pending' if self.pending_activation else 'accepted'
                else:
                    reason = self.change(request['operation'] == 'enable', request['cell'], self.simulator.state.slot_id, request['app'])
                    self.cancel_activation('Cancelled by an explicit enable/disable request')
                    status = 'accepted'
            except (OSError, ValueError, TypeError, KeyError) as exc:
                reason = str(exc)
            self.record_receipt(request if isinstance(request, dict) else {}, status, reason)
            path.unlink(missing_ok=True)
        self.advance_activation()

    def after_step(self, adapter):
        state = self.simulator.state
        packet = [u.packet_ledger.summary() for u in state.ues]
        completed = sum(p['completed_packets'] for p in packet)
        delay = sum(p['completion_latency_ms_sum'] for p in packet)
        prior_count, prior_delay = self.previous_packet
        self.previous_packet = completed, delay
        segment = self.segments[-1]
        if state.slot_id < segment['first_counted_slot']:
            segment['skipped_slots'] += 1
            return
        rows = adapter.trace['ues']
        decoded = [r for r in rows if r.get('actual_decode')]
        segment['slots'] += 1
        segment['delivered_bytes'] += sum(r.get('delivered_bytes', 0) for r in rows)
        segment['queue_sum'] += sum(u.queue_bytes for u in state.ues)
        segment['hol_sum'] += max((p['hol_delay_ms'] for p in packet), default=0.)
        segment['decodes'] += len(decoded)
        segment['nacks'] += sum(not r['ack'] for r in decoded)
        segment['completed_packets'] += completed - prior_count
        segment['completion_delay_sum_ms'] += delay - prior_delay

    def snapshot(self):
        from ric.near_rt.xapp_library import IMPLEMENTATIONS
        catalog = []
        for ident, app in self.apps.items():
            descriptor = app.descriptor()
            reason = self.unavailable_reason(app)
            catalog.append({key: descriptor[key] for key in (
                'instance_id', 'implementation_id', 'category_id', 'description', 'writable_actions',
                'active_cell_ids', 'required_kpis', 'min_action_ms') if key in descriptor} |
                dict(registered=True, available=reason is None, unavailable_reason=reason,
                    selection_versions=dict(app.tokens)))
        for ident, spec in sorted(IMPLEMENTATIONS.items()):
            if ident not in self.apps:
                catalog.append(dict(instance_id=ident, implementation_id=ident,
                    category_id=spec['category_id'], description=spec['description'],
                    registered=False, available=False, active_cell_ids=[],
                    unavailable_reason='Handover commands require a separate experiment; configuration withdrawal cannot undo them'))
        segments = []
        for item in self.segments[-12:]:
            n, count = item['slots'], item['completed_packets']
            segments.append({**item, 'duration_ms': n * self.dt,
                'throughput_mbps': item['delivered_bytes'] * 8 / (n * self.dt * 1000) if n else None,
                'mean_queue_bytes': item['queue_sum'] / n if n else None,
                'mean_max_hol_ms': item['hol_sum'] / n if n else None,
                'mean_completion_delay_ms': item['completion_delay_sum_ms'] / count if count else None,
                'nack_fraction': item['nacks'] / item['decodes'] if item['decodes'] else None})
        return dict(run_id=self.run_id, app=self.app.descriptor(), catalog=catalog,
            pending_activation=deepcopy(self.pending_activation),
            selected_cells=sorted(self.app.active_cells), period_ms=self.period * self.dt,
            settle_ms=self.settle_slots * self.dt, segments=segments,
            last_decisions=deepcopy(self.app.last_decisions), receipts=list(self.receipts),
            baseline=deepcopy(self.baseline), command_dir=str(self.command_dir) if self.command_dir else None,
            comparison_spec=str(self.spec_path), simulation_python=sys.executable,
            comparison_note='Live segments are observational; matched A/B restarts both arms with the same seed, traffic and propagation.')


def submit(state_path, data, operation, cell='*', app=APP):
    experiment = data['xapp_test']
    if data.get('stage') != 'simulating':
        raise ValueError('This run has finished; start a new live run to toggle xApps')
    directory = Path(state_path).with_suffix('.commands')
    if not directory.is_dir():
        raise ValueError('No live command channel')
    request_id = uuid.uuid4().hex
    atomic_json(directory / f'{time.time_ns():020d}_{request_id}.json',
        dict(run_id=experiment['run_id'], request_id=request_id,
             operation=operation, app=app, cell=cell))
    return request_id


def start_comparison(state_path, data, cell='*', app=APP):
    state_path = Path(state_path)
    result = state_path.with_suffix('.comparison.json')
    lock = result.with_suffix('.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            prior = json.loads(lock.read_text())
            os.kill(int(prior['pid']), 0)
        except ProcessLookupError:
            lock.unlink(missing_ok=True)
            return start_comparison(state_path, data, cell, app)
        except (OSError, ValueError, KeyError):
            pass
        raise ValueError('A matched A/B test is already running; see its progress below')
    job_dir = state_path.parent / ('comparison_' + uuid.uuid4().hex[:10])
    log = None
    try:
        job_dir.mkdir()
        atomic_json(result, dict(stage='queued', app_id=app, output=str(job_dir)))
        log = (job_dir / 'worker.log').open('w')
        process = subprocess.Popen([data['xapp_test']['simulation_python'],
            str(ROOT / 'experiments/compare_live_xapp.py'), '--spec',
            data['xapp_test']['comparison_spec'], '--output', str(job_dir),
            '--result', str(result), '--cell', cell, '--app', app], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        os.write(fd, json.dumps(dict(pid=process.pid)).encode())
    except Exception:
        lock.unlink(missing_ok=True)
        atomic_json(result, dict(stage='failed', error='Could not start comparison; see worker log', output=str(job_dir)))
        raise
    finally:
        os.close(fd)
        if log is not None:log.close()
    return process.pid

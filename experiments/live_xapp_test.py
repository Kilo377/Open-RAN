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
    period = max(1, math.ceil(20 / manifest['slot_duration_ms']))
    configs['ric'].update(rapps=[], xapps=[dict(type=APP, name=APP,
        cell_ids=[c['cell_id'] for c in manifest['cells']])], freeze_learning=True,
        run_rapps_on_start=False, xapp_interval_slots=period,
        rapp_interval_slots=math.ceil(1000 / manifest['slot_duration_ms']))
    return True


class LiveXAppSession:
    def __init__(self, simulator, args, output, state_path=None):
        from ric.near_rt.rule_based_v1.leases import RuleLeases
        self.simulator, self.args, self.output = simulator, args, Path(output)
        self.run_id = uuid.uuid4().hex
        self.manager = simulator.controller_manager
        self.app = self.manager.instance_selection.apps()[APP]
        self.dt = float(simulator.clock.slot_duration_ms) if hasattr(simulator.clock, 'slot_duration_ms') else float(args._live_slot_ms)
        self.period = self.manager.xapp_interval_slots
        self.settle_slots = self.period + max(0, simulator._config_delay_slots)
        self.manager.rule_leases = RuleLeases()
        self.baseline = {cid: {key: deepcopy(getattr(runtime.scheduler, key))
            for key in self.app.controlled_fields}
            for cid, runtime in simulator.state.managed_cells.items()}
        self.manager.rule_leases.baseline = deepcopy(self.baseline)
        self.receipts, self.segments = deque(maxlen=30), []
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

    def change(self, enabled, cell, slot):
        from models.policy import PolicyUpdate
        known = set(self.baseline)
        if cell != '*' and cell not in known:
            raise ValueError('Unknown target cell')
        selection = self.manager.instance_selection
        before = deepcopy(selection.bindings or {})
        new = {cid: {self.app.category_id: APP} for cid in (sorted(known) if cell == '*' else [cell])} if enabled else {}
        if new == before and self.segments:
            return 'unchanged'
        # A previous withdrawal must finish before its baseline is reused.
        if enabled:
            for cid in set(new) - set(before):
                for key, value in self.baseline[cid].items():
                    if getattr(self.simulator.state.managed_cells[cid].scheduler, key) != value:
                        raise ValueError('Wait for the previous disable to restore the scheduler')
        accepted = self.manager.policy_store.apply_update(PolicyUpdate(
            source_name='live_xapp_ui', xapp_bindings=new), slot_id=slot)
        if accepted is False:
            raise ValueError('Selection policy was rejected')
        # Invalidate queued old-generation requests before the next slot applies them.
        selection.sync(slot, None)
        self.segments.append(dict(label=('ON / ' + cell) if enabled else 'OFF / baseline',
            enabled=enabled, cell=cell, start_slot=slot, first_counted_slot=slot + self.settle_slots + 1,
            slots=0, skipped_slots=0, delivered_bytes=0, queue_sum=0, hol_sum=0.,
            decodes=0, nacks=0, completed_packets=0, completion_delay_sum_ms=0.))
        return 'selection_admitted; awaiting normal control tick and delayed execution'

    def before_step(self):
        if not self.command_dir:
            return
        for path in sorted(self.command_dir.glob('*.json'))[:32]:
            request, status, reason = {}, 'rejected', ''
            try:
                if path.stat().st_size > 8192:
                    raise ValueError('Command too large')
                request = json.loads(path.read_text())
                if not isinstance(request, dict) or set(request) != {'run_id', 'request_id', 'operation', 'app', 'cell'}:
                    raise ValueError('Malformed command')
                if request['run_id'] != self.run_id or request['app'] != APP:
                    raise ValueError('Stale run or unregistered app')
                if request['operation'] not in ('enable', 'disable'):
                    raise ValueError('Unsupported operation')
                reason = self.change(request['operation'] == 'enable', request['cell'], self.simulator.state.slot_id)
                status = 'accepted'
            except (OSError, ValueError, TypeError, KeyError) as exc:
                reason = str(exc)
            event = dict(request_id=request.get('request_id') if isinstance(request, dict) else None,
                operation=request.get('operation') if isinstance(request, dict) else None,
                received_at_slot=self.simulator.state.slot_id, status=status, reason=reason)
            self.receipts.append(event)
            with self.events.open('a') as stream:
                stream.write(json.dumps(event) + '\n')
            path.unlink(missing_ok=True)

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
        segments = []
        for item in self.segments[-12:]:
            n, count = item['slots'], item['completed_packets']
            segments.append({**item, 'duration_ms': n * self.dt,
                'throughput_mbps': item['delivered_bytes'] * 8 / (n * self.dt * 1000) if n else None,
                'mean_queue_bytes': item['queue_sum'] / n if n else None,
                'mean_max_hol_ms': item['hol_sum'] / n if n else None,
                'mean_completion_delay_ms': item['completion_delay_sum_ms'] / count if count else None,
                'nack_fraction': item['nacks'] / item['decodes'] if item['decodes'] else None})
        return dict(run_id=self.run_id, app=self.app.descriptor(),
            selected_cells=sorted(self.app.active_cells), period_ms=self.period * self.dt,
            settle_ms=self.settle_slots * self.dt, segments=segments,
            last_decisions=deepcopy(self.app.last_decisions), receipts=list(self.receipts),
            baseline=deepcopy(self.baseline), command_dir=str(self.command_dir) if self.command_dir else None,
            comparison_spec=str(self.spec_path), simulation_python=sys.executable,
            comparison_note='Live segments are observational; matched A/B restarts both arms with the same seed, traffic and propagation.')


def submit(state_path, data, operation, cell='*'):
    experiment = data['xapp_test']
    if data.get('stage') != 'simulating':
        raise ValueError('This run has finished; start a new live run to toggle xApps')
    directory = Path(state_path).with_suffix('.commands')
    if not directory.is_dir():
        raise ValueError('No live command channel')
    request_id = uuid.uuid4().hex
    atomic_json(directory / f'{time.time_ns():020d}_{request_id}.json',
        dict(run_id=experiment['run_id'], request_id=request_id,
             operation=operation, app=APP, cell=cell))
    return request_id


def start_comparison(state_path, data, cell='*'):
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
            return start_comparison(state_path, data, cell)
        except (OSError, ValueError, KeyError):
            pass
        raise ValueError('A matched A/B test is already running; see its progress below')
    job_dir = state_path.parent / ('comparison_' + uuid.uuid4().hex[:10])
    log = None
    try:
        job_dir.mkdir()
        atomic_json(result, dict(stage='queued', output=str(job_dir)))
        log = (job_dir / 'worker.log').open('w')
        process = subprocess.Popen([data['xapp_test']['simulation_python'],
            str(ROOT / 'experiments/compare_live_xapp.py'), '--spec',
            data['xapp_test']['comparison_spec'], '--output', str(job_dir),
            '--result', str(result), '--cell', cell], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        os.write(fd, json.dumps(dict(pid=process.pid)).encode())
    except Exception:
        lock.unlink(missing_ok=True)
        atomic_json(result, dict(stage='failed', error='Could not start comparison; see worker log', output=str(job_dir)))
        raise
    finally:
        os.close(fd)
        if log is not None:log.close()
    return process.pid

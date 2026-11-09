"""Serial, resumable matched frozen-xApp evaluation; no automatic training."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
POPULATION = {'ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8'}
APP_IDS = {'x01': 'ue_energy_rf', 'x02': 'ap_energy_rf',
           'x08': 'throughput_scheduler_mlp', 'x12_mec': 'latency_scheduler_rf'}
MODE_NAMES = {'x01': 'energy', 'x02': 'ap', 'x08': 'throughput', 'x12_mec': 'latency'}
SUMMARY_KEYS = {'x01': 'ue_energy_xapp', 'x02': 'ap_xapp',
                'x08': 'throughput_xapp', 'x12_mec': 'latency_xapp'}
EVENT_LOGS = {'x01':'energy_rf_events', 'x02':'ap_rf_events',
              'x08':'throughput_mlp_events', 'x12_mec':'latency_rf_events'}
FORMATS = {'x01': 'chongyu.online-energy-rf', 'x02': 'chongyu.online-ap-rf',
           'x08': 'chongyu.online-throughput-mlp', 'x12_mec': 'chongyu.online-energy-rf'}
MODES = ('energy', 'latency', 'budget', 'throughput', 'urllc', 'ap')


def load(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(root=ROOT):
    """Exactly the runtime's source/config rule, without importing NumPy."""
    root = Path(root)
    digest = hashlib.sha256()
    for name in ('core', 'phy', 'ran', 'ric', 'state', 'models', 'traffic', 'experiments', 'configs'):
        for path in sorted((root / name).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.json', '.yaml', '.yml'):
                digest.update(str(path.relative_to(root)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def _path(root, value):
    p = Path(value)
    return (p if p.is_absolute() else Path(root) / p).resolve()


def load_models(model_dir, root=ROOT, manifest_path=None):
    root = Path(root).resolve(); model_dir = _path(root, model_dir)
    if not model_dir.is_relative_to(root / 'outputs'):
        raise ValueError('Frozen model directory must be under this project outputs/')
    manifest = _path(model_dir, manifest_path or 'models.json')
    if not manifest.is_relative_to(model_dir):
        raise ValueError('Model manifest must stay inside the model directory')
    entries = load(manifest).get('models')
    if not isinstance(entries, dict) or set(entries) != set(APP_IDS):
        raise ValueError('models.json must contain exactly x01, x02, x08 and x12_mec')
    result = {}
    for app, entry in entries.items():
        if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
            raise ValueError('Model entries require path and explicit sha256')
        p = _path(model_dir, entry['path'])
        if not p.is_relative_to(model_dir) or not p.is_file():
            raise ValueError('Every frozen artifact must be a file inside the model directory')
        digest = sha256(p)
        if entry.get('sha256') != digest:
            raise ValueError('Frozen model SHA-256 mismatch: ' + app)
        value = load(p)
        rows, fitted, fits = len(value.get('x', [])), value.get('trained_samples'), value.get('fit_count')
        if (value.get('format') != FORMATS[app] or type(fitted) is not int or
                not 0 < fitted <= rows or type(fits) is not int or fits < 1):
            raise ValueError('A genuinely trained JSON artifact is required: ' + app)
        result[app] = dict(path=str(p), sha256=digest, stamp=dict(samples=rows,
            trained_samples=fitted, fit_count=fits, format=value['format'],
            feature_names=value.get('feature_names'), fit_parameters=value.get('fit_parameters')))
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--model-manifest', type=Path, default=Path('models.json'))
    p.add_argument('--ap-services-profile', type=Path, default=Path('configs/rt_sys/ap_services_reference.json'))
    p.add_argument('--duration-s', type=float, default=30.)
    p.add_argument('--seed', type=int, default=55)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--reuse-matched-baselines', action='store_true',
        help='Reuse the exact matched native1x and AP-enabled2x OFF recordings; six physical runs for eight logical cases')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--min-free-gb', type=float, default=1.)
    p.add_argument('--estimated-case-gb', type=float, default=.8)
    p.add_argument('--heartbeat-s', type=float, default=10.)
    p.add_argument('--termination-grace-s', type=float, default=15.)
    return p.parse_args(argv)


def build_plan(args, root=ROOT):
    root = Path(root).resolve(); output = _path(root, args.output)
    if not output.is_relative_to(root / 'outputs') or output == root / 'outputs':
        raise ValueError('A dedicated queue directory under project outputs/ is required')
    duration = args.duration_s
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Finite positive simulation duration required')
    slots = duration * 2000
    if not math.isclose(slots, round(slots), rel_tol=0, abs_tol=1e-8):
        raise ValueError('Duration must align to the 0.5 ms slot')
    if type(args.seed) is not int or not 0 <= args.seed <= 2**32-1:
        raise ValueError('A valid nonnegative network seed is required')
    for name in ('min_free_gb', 'estimated_case_gb'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Finite nonnegative disk budgets required')
    for name in ('heartbeat_s', 'termination_grace_s'):
        if not math.isfinite(getattr(args, name)) or not 0 < getattr(args, name) <= 60:
            raise ValueError('Heartbeat and termination grace must be in (0,60] seconds')
    model_dir = _path(root, args.model_dir)
    manifest = _path(model_dir, args.model_manifest)
    models = load_models(model_dir, root, args.model_manifest)
    profile = _path(root, args.ap_services_profile)
    if not profile.is_relative_to(root) or load(profile).get('ap_services', {}).get('enabled') is not True:
        raise ValueError('Explicit enabled project-local AP profile required')
    cases = []
    for app in APP_IDS:
        for mode in ('off', 'frozen'):
            name = app + '_' + mode; folder = output / 'cases' / name
            cmd = ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast',
                '--duration-s', str(duration), '--seed', str(args.seed), '--traffic-load',
                '2' if app in ('x02', 'x08') else '1', '--xapp-pool', 'chongyu',
                '--ue-services', 'drx-mec', '--record-compression', 'gzip',
                '--direct-conflict-mitigation', 'priority', '--throughput-xapp-policy', 'throughput',
                '--channel-cache-root', str(output / 'channel_cache' / ('seed_' + str(args.seed)))]
            if app in ('x02', 'x08'):
                cmd += ['--ap-services-profile', str(profile)]
            for kind in MODES:
                cmd += ['--' + kind + '-xapp-mode', 'frozen' if mode == 'frozen' and kind == MODE_NAMES[app] else 'off']
            if mode == 'frozen':
                cmd += ['--' + MODE_NAMES[app] + '-xapp-model', models[app]['path']]
            cmd += ['--output', str(folder)]
            cases.append(dict(name=name, app=app, mode=mode, traffic_load=2 if app in ('x02', 'x08') else 1,
                duration_s=float(duration), slots=round(slots), seed=args.seed, output=str(folder), command=cmd))
    if args.reuse_matched_baselines:
        def normalized(command):
            command = list(command); index = command.index('--output')
            return command[:index] + command[index+2:]
        for alias, original in (('x08_off','x02_off'), ('x12_mec_off','x01_off')):
            case = next(c for c in cases if c['name']==alias)
            source = next(c for c in cases if c['name']==original)
            if (case['mode'] != source['mode'] or case['mode'] != 'off' or
                    normalized(case['command']) != normalized(source['command'])):
                raise ValueError('Requested baseline reuse is not exactly configuration-matched')
            case['reused_from'] = original; case['output'] = source['output']
            case['command'] = list(source['command'])
    return dict(schema='oran.serial-xapp-validation.v1', project_root=str(root), output=str(output),
        source_sha256=source_fingerprint(root), model_manifest=str(manifest),
        manifest_sha256=sha256(manifest), profile=str(profile), profile_sha256=sha256(profile),
        models=models, cases=cases, duration_s=float(duration), slot_duration_ms=.5,
        seed=args.seed, serial=True, frozen_models=True, automatic_extra_runs=False,
        reuse_matched_baselines=args.reuse_matched_baselines,
        logical_case_count=len(cases), physical_run_count=sum('reused_from' not in c for c in cases),
        metrics_scope='Matched per-xApp pairs retain native1x for X01/MEC and explicit2x for X02/X08; no retraining or stress-profile change')


def verify_inputs(plan, root=ROOT):
    if source_fingerprint(root) != plan['source_sha256']:
        raise ValueError('Executed source/config changed after the queue was declared')
    if sha256(plan['model_manifest']) != plan['manifest_sha256'] or sha256(plan['profile']) != plan['profile_sha256']:
        raise ValueError('Declared model manifest or AP profile changed')
    for app, model in plan['models'].items():
        if sha256(model['path']) != model['sha256']:
            raise ValueError('Frozen input artifact changed: ' + app)


@contextmanager
def queue_lock(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    stream = (output / '.queue.lock').open('a+')
    try:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another process owns this queue') from error
        stream.seek(0); stream.truncate(); stream.write(str(os.getpid()) + '\n'); stream.flush()
        yield
    finally:
        stream.close()


def prepare_queue(plan, resume=False):
    output = Path(plan['output']); plan_path = output / 'queue-plan.json'
    if plan_path.exists():
        if not resume: raise FileExistsError('Queue exists; explicit --resume is required')
        if load(plan_path) != plan: raise ValueError('Resume parameters/source/models differ from the immutable queue plan')
        state = load(output / 'queue-status.json')
        if state.get('plan_sha256') != sha256(plan_path):
            raise ValueError('Queue plan checksum changed')
        if sha256(output/'input-models-manifest.json') != plan['manifest_sha256']:
            raise ValueError('Archived input-model manifest changed')
        return state
    if resume: raise ValueError('No original queue plan exists to resume')
    if any(p.name != '.queue.lock' for p in output.iterdir()):
        raise FileExistsError('Fresh queue directory must be empty')
    atomic_json(plan_path, plan)
    shutil.copy2(plan['model_manifest'], output / 'input-models-manifest.json')
    state = dict(schema=plan['schema'], plan_sha256=sha256(plan_path), status='prepared', completed=False,
        started_at_utc=datetime.now(timezone.utc).isoformat(), queue_pid=os.getpid(), active_case=None,
        completed_cases=[], attempts=[], failures=[], progress_scope='Actual child status; wall time is not simulated time')
    atomic_json(output / 'queue-status.json', state)
    return state


def _records(folder, stem):
    choices = [p for p in (Path(folder)/(stem+'.jsonl.gz'), Path(folder)/(stem+'.jsonl')) if p.is_file()]
    if len(choices) != 1: raise ValueError('One unambiguous recorded stream required: ' + stem)
    opener = gzip.open if choices[0].suffix == '.gz' else open
    with opener(choices[0], 'rt') as stream:
        for line in stream: yield json.loads(line)


def audit_case(case, plan):
    """Bounded streaming verification; never loads the complete raw trace."""
    folder = Path(case['output']); status = load(folder/'run_status.json')
    cfg = load(folder/'config_snapshot.json'); summary = load(folder/'final_summary.json')
    if not status.get('completed') or not status.get('replay', {}).get('audits_passed'):
        raise ValueError('Completed simulation and audited automatic HTML required')
    if status.get('source_sha256') != plan['source_sha256'] or cfg.get('source_sha256') != plan['source_sha256']:
        raise ValueError('Recorded source does not match the frozen queue source')
    manifest = cfg['manifest']
    if (manifest['total_slots'] != case['slots'] or summary['total_slots'] != case['slots'] or
            manifest['slot_duration_ms'] != .5 or manifest['rng_seed'] != case['seed'] or
            {u['ue_id'] for u in manifest['ues']} != POPULATION or len(manifest['cells']) != 3):
        raise ValueError('Recorded deployment/seed/completed horizon differs from the case')
    apps = cfg['configs']['ric'].get('xapps', [])
    expected = [] if case['mode'] == 'off' else [APP_IDS[case['app']]]
    if [a.get('implementation_id') for a in apps] != expected:
        raise ValueError('Unexpected active controller in recorded case')
    frozen = None
    if expected:
        model = plan['models'][case['app']]; learning = summary.get(SUMMARY_KEYS[case['app']], {})
        if (apps[0]['parameters']['mode'] != 'frozen' or
                str(Path(apps[0]['parameters']['model_path']).resolve()) != model['path'] or
                learning.get('mode') != 'frozen' or any(learning.get(k) != model['stamp'][k] for k in ('samples', 'trained_samples', 'fit_count'))):
            raise ValueError('Learner was not frozen with its exact declared artifact/history')
        selected = learning.get('mlp_selected_actions', learning.get('rf_selected_actions', 0))
        if type(selected) is not int or selected < 0:
            raise ValueError('Frozen model choice counter is invalid')
        frozen = dict(model_sha256=model['sha256'], samples=learning['samples'],
            trained_samples=learning['trained_samples'], fit_count=learning['fit_count'],
            selected_actions=selected, prediction_batches=learning.get('prediction_batches'),
            status=learning.get('status'), status_reason=learning.get('status_reason'),
            invalidated_samples=learning.get('invalidated_samples'),
            no_offloaded_demand_windows=learning.get('no_offloaded_demand_windows'),
            windows=learning.get('windows'))
    packets = summary.get('packet_metrics', {})
    if set(packets) != POPULATION: raise ValueError('All six actual packet ledgers are required')
    for user in packets.values():
        if user['initial_bytes'] + user['offered_bytes'] != user['delivered_bytes'] + user['pending_bytes'] + user['overflow_bytes']:
            raise ValueError('Actual packet-byte conservation failed')
    # Reconstruct terminal outcomes from enriched task events. Public task
    # telemetry retains only eight cohort buckets, so it cannot prove 30s means.
    sys.path.insert(0, plan['project_root'])
    from experiments.run_ue_energy_study import mature_cohorts, aggregate_cohorts
    from experiments.run_latency_energy_study import aggregate_offloaded_cohorts
    from experiments.task_event_metrics import TaskEventCollector
    history = TaskEventCollector(case['duration_s'], cohort_ms=100, start_s=.1,
        target_uids=('ue_0', 'ue_4'))
    observed_cohorts = {u: {} for u in ('ue_0', 'ue_4')}; exogenous = hashlib.sha256(); count = 0
    for row in _records(folder, 'rt_sys_trace'):
        count += 1
        if row['slot_id'] != count or {u['ue_id'] for u in row['ues']} != POPULATION:
            raise ValueError('Every ordered six-UE slot must be recorded exactly once')
        arrivals = row.get('traffic_arrivals', {})
        if set(arrivals) != POPULATION: raise ValueError('Every exogenous traffic input must be recorded')
        exogenous.update(json.dumps([count, [[u, arrivals[u]['exogenous_bytes']] for u in sorted(POPULATION)]], separators=(',', ':')).encode())
        for event in row.get('ue_service_events', []): history.observe(event)
        for uid in observed_cohorts:
            observed = row.get('ue_services', {}).get(uid, {})
            observed_cohorts[uid].update(mature_cohorts(observed, min(case['duration_s'], count*.0005)))
    if count != case['slots']: raise ValueError('Recorded trace horizon is incomplete')
    exact_history = history.summary()
    # Missing event streams must not masquerade as an empty/zero-latency task
    # population. Reconstructed full-horizon counters must match the runtime.
    task_counters = ('generated','completed','on_time_completed','failed','pending',
        'offloaded_generated','offloaded_completed','offloaded_on_time_completed',
        'offloaded_failed','offloaded_pending')
    for uid, reconstructed in exact_history['per_ue'].items():
        cumulative = summary.get('ue_services', {}).get('per_ue', {}).get(uid, {}).get('ue_task_metrics', {})
        whole = reconstructed['whole_run']
        if any(cumulative.get(key) != whole[key] for key in task_counters):
            raise ValueError('Recorded task-event counters differ from final cumulative task ledger: '+uid)
        for key in ('completion_latency_ms_sum','offloaded_completion_latency_ms_sum'):
            value = cumulative.get(key)
            if type(value) not in (int,float) or not math.isclose(value,whole[key],rel_tol=1e-10,abs_tol=1e-7):
                raise ValueError('Recorded task-event latency sums differ from final cumulative ledger: '+uid)
    cohorts = {uid: {(round(c['start_s'],9),round(c['end_s'],9)): c
        for c in exact_history['per_ue'][uid]['cohorts']} for uid in observed_cohorts}
    task_keys = {uid: sorted(k for k in rows if k[0] >= .1-1e-9) for uid, rows in cohorts.items()}
    selected_cohorts = [cohorts[u][k] for u in cohorts for k in task_keys[u]]
    tasks = aggregate_cohorts(selected_cohorts)
    offloaded = aggregate_offloaded_cohorts(selected_cohorts)
    task_details = {u: dict(intervals_s=[list(k) for k in task_keys[u]],
        generated_by_interval=[[k[0],k[1],cohorts[u][k]['generated']] for k in task_keys[u]],
        all_tasks=aggregate_cohorts([cohorts[u][k] for k in task_keys[u]]),
        offloaded_tasks=aggregate_offloaded_cohorts([cohorts[u][k] for k in task_keys[u]])) for u in cohorts}
    actual_controls = dict(applied=0, rejected=0, cells=set(), fields=set())
    if expected:
        for receipt in _records(folder, 'control_receipts'):
            provenance = receipt.get('provenance', [])
            if isinstance(provenance, dict): provenance = [provenance]
            if not any(p.get('implementation_id') == expected[0] for p in provenance): continue
            if receipt.get('status') in ('applied', 'rejected'):
                actual_controls[receipt['status']] += 1
            if receipt.get('status') == 'applied':
                actual_controls['cells'].update(receipt.get('after', {}))
                actual_controls['fields'].update(f['field'] for f in receipt.get('fields', []))
        kinds, statuses, reasons = Counter(), Counter(), Counter()
        for event in _records(folder, EVENT_LOGS[case['app']]):
            kinds[str(event.get('kind','unspecified'))] += 1
            if event.get('status') is not None: statuses[str(event['status'])] += 1
            if event.get('reason') is not None: reasons[str(event['reason'])] += 1
        frozen['event_counts'] = dict(kinds)
        frozen['event_status_counts'] = dict(statuses)
        frozen['event_reason_counts'] = dict(reasons)
        frozen['controller_evaluable'] = bool(frozen['selected_actions'] and actual_controls['applied'])
        frozen['effect_attribution_verified'] = frozen['controller_evaluable']
        frozen['coverage_scope'] = 'Recorded frozen model choices and actual applied receipts; full forecast-by-forecast replay is not performed by this queue'
        frozen['unverified_reasons'] = ([] if frozen['selected_actions'] else ['no_frozen_model_choice']) + ([] if actual_controls['applied'] else ['no_actual_applied_control_receipt'])
    actual_controls['cells'] = sorted(actual_controls['cells']); actual_controls['fields'] = sorted(actual_controls['fields'])
    html = folder/'network-replay.html'
    if not html.is_file(): raise ValueError('Automatic self-contained replay HTML is missing')
    inventory = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    energy = summary.get('ue_services', {}).get('per_ue', {})
    if set(energy) != POPULATION: raise ValueError('All six UE-service energy counters are required')
    metrics = dict(network_goodput_mbps=sum(u['delivered_bytes'] for u in packets.values())*8/case['duration_s']/1e6,
        network_ue_energy_j=sum(u['ue_energy_total_j'] for u in energy.values()) if energy else None,
        target_ue_energy_j=sum(energy[u]['ue_energy_total_j'] for u in cohorts) if set(cohorts)<=set(energy) else None,
        network_ap_energy_j=summary.get('ap_services', {}).get('network_ap_energy_total_j'),
        mature_target_all_tasks=tasks, mature_target_offloaded_tasks=offloaded,
        mature_target_cohort_count=len(selected_cohorts), per_target_task_metrics=task_details,
        whole_run_target_task_metrics={u:exact_history['per_ue'][u]['whole_run'] for u in cohorts},
        task_event_history=exact_history,
        task_event_cumulative_ledger_reconstruction_verified=True,
        public_cohort_diagnostic=dict(eligible_observed_cohorts_by_ue={u:len([k for k in rows if k[0]>=.1-1e-9]) for u,rows in observed_cohorts.items()},
            scope='Secondary limited-history telemetry; not authoritative for final long-horizon completion statistics'),
        task_scoring_start_s=.1, tasks_evaluable=bool(selected_cohorts and offloaded['completed']),
        metrics_scope='Whole-run energy/goodput; primary task means reconstruct all task-event outcomes for mature creation cohorts starting at >=100ms, including completions after public telemetry eviction. Pending/failed tasks remain explicit; no efficacy threshold applied.')
    return dict(case=case['name'], source_sha256=plan['source_sha256'], slots=count,
        duration_s=case['duration_s'], exogenous_slot_inputs_sha256=exogenous.hexdigest(),
        model_frozen=frozen, actual_controls=actual_controls, metrics=metrics,
        recorded_file_sha256=inventory, automatic_html_audits_passed=True)


def _audit_path(output, name):
    return Path(output)/'audits'/(name+'.json')


def _check_saved_audit(case, entry, output):
    path = _audit_path(output, case['name'])
    if sha256(path) != entry['audit_sha256']: raise ValueError('Saved completed-case audit changed')
    audit = load(path)
    actual_files = {str(p.relative_to(Path(case['output'])))
        for p in Path(case['output']).rglob('*') if p.is_file()}
    if actual_files != set(audit['recorded_file_sha256']):
        raise ValueError('Completed recording file inventory changed')
    for name, digest in audit['recorded_file_sha256'].items():
        if sha256(Path(case['output'])/name) != digest:
            raise ValueError('Completed raw/HTML recording changed; refusing to rewrite history')
    return audit


def _alive(pid):
    if type(pid) is not int or pid <= 0: return False
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


def _archive_partial(case, output):
    folder = Path(case['output']); log = Path(output)/'logs'/(case['name']+'.log')
    if not folder.exists() and not log.exists(): return None
    target = Path(output)/'attempts'/(case['name']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    target.mkdir(parents=True)
    if folder.exists(): shutil.move(str(folder), str(target/'case'))
    if log.exists(): shutil.move(str(log), str(target/'child.log'))
    return str(target)


def pair_summaries(plan, audits):
    results = {}
    for app in APP_IDS:
        a, b = (audits.get(app+'_'+mode) for mode in ('off', 'frozen'))
        if a is None or b is None: continue
        ca, cb = (load(Path(c['output'])/'config_snapshot.json') for c in plan['cases'] if c['name'] in (app+'_off', app+'_frozen'))
        if ca['manifest'] != cb['manifest'] or any(ca['configs'][k] != cb['configs'][k] for k in ('ue', 'phy', 'sim', 'cell', 'beamforming')):
            raise ValueError('Matched pair deployment/business/physics differs: '+app)
        if a['exogenous_slot_inputs_sha256'] != b['exogenous_slot_inputs_sha256']:
            raise ValueError('Actual exogenous traffic differs within a matched pair')
        if any(a['metrics']['per_target_task_metrics'][u]['intervals_s'] != b['metrics']['per_target_task_metrics'][u]['intervals_s'] for u in ('ue_0', 'ue_4')):
            raise ValueError('Paired mature task creation intervals differ')
        if any(a['metrics']['per_target_task_metrics'][u]['generated_by_interval'] != b['metrics']['per_target_task_metrics'][u]['generated_by_interval'] for u in ('ue_0','ue_4')):
            raise ValueError('Paired mature task creation counts differ; routed counts may legitimately differ')
        key = {'x01':'target_ue_energy_j', 'x02':'network_ap_energy_j',
               'x08':'network_goodput_mbps', 'x12_mec':'mature_target_offloaded_tasks'}[app]
        av, bv = a['metrics'][key], b['metrics'][key]
        if app == 'x12_mec': av, bv = av['completed_task_mean_latency_ms'], bv['completed_task_mean_latency_ms']
        gain = None if av is None or bv is None or av <= 0 else (bv/av-1 if app=='x08' else 1-bv/av)
        results[app] = dict(primary_metric=key, bare=av, frozen=bv, relative_improvement=gain,
            evaluable=gain is not None, physical_evaluable=True,
            controller_evaluable=b['model_frozen']['controller_evaluable'],
            effect_attribution_verified=b['model_frozen']['effect_attribution_verified'],
            bare_metrics=a['metrics'], frozen_metrics=b['metrics'],
            matched_manifest_physics_and_exogenous_inputs=True, no_effectiveness_gate=True)
    return results


def run_queue(args, plan):
    output = Path(plan['output']); stop = {'signal': None, 'disk': None}; child = None
    with queue_lock(output):
        state = prepare_queue(plan, args.resume)
        old_active = state.get('active_case')
        if isinstance(old_active, dict) and _alive(old_active.get('child_pid')):
            raise RuntimeError('A recorded previous child is still alive; wait or stop it before resuming')
        for directory in ('cases', 'logs', 'audits', 'attempts'): (output/directory).mkdir(exist_ok=True)
        def write():
            state['queue_pid'] = os.getpid(); state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(output/'queue-status.json', state)
        def request_stop(signum, frame):
            stop['signal'] = signum
            if child is not None and child.poll() is None:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        previous = {s:signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        audits = {}
        try:
            verify_inputs(plan, Path(plan['project_root']))
            for entry in state['completed_cases']:
                case = next(c for c in plan['cases'] if c['name']==entry['name'])
                audits[case['name']] = _check_saved_audit(case, entry, output)
            state.update(status='running', completed=False, active_case=None); write()
            for case in plan['cases']:
                if case['name'] in audits: continue
                if stop['signal'] is not None: break
                verify_inputs(plan, Path(plan['project_root']))
                folder = Path(case['output']); run_status = folder/'run_status.json'
                if case.get('reused_from'):
                    source_name = case['reused_from']
                    if source_name not in audits: raise ValueError('Original audited baseline must finish before reuse')
                    source_case = next(c for c in plan['cases'] if c['name']==source_name)
                    if case['output'] != source_case['output'] or case['command'] != source_case['command']:
                        raise ValueError('Reused baseline output/command diverged from its original')
                    audit = deepcopy(audits[source_name])
                    audit.update(case=case['name'], reused_from=source_name,
                        source_baseline_audit_sha256=sha256(_audit_path(output,source_name)),
                        reuse_scope='Identical physical inputs and all-OFF controls; this is one shared recording, not an independent replication')
                elif args.resume and run_status.is_file() and load(run_status).get('completed'):
                    audit = audit_case(case, plan)
                else:
                    free = shutil.disk_usage(output).free
                    needed = (args.min_free_gb+args.estimated_case_gb)*1024**3
                    if free < needed:
                        state.update(status='blocked_disk_space', active_case=None,
                            disk=dict(free_bytes=free, minimum_reserve_gib=args.min_free_gb,
                                estimated_next_case_gib=args.estimated_case_gb, next_case=case['name']))
                        write(); return 75
                    archived = _archive_partial(case, output)
                    if archived: state['attempts'].append(dict(case=case['name'], archived=archived))
                    started = time.time(); started_mono = time.monotonic()
                    with (output/'logs'/(case['name']+'.log')).open('xb') as log:
                        child = subprocess.Popen(case['command'], cwd=plan['project_root'], stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
                        state['active_case'] = dict(name=case['name'], child_pid=child.pid,
                            started_unix_s=started, elapsed_wall_s=0., progress={})
                        write(); last = 0.; term_at = None
                        while child.poll() is None:
                            now = time.monotonic()
                            if stop['signal'] is not None or stop['disk'] is not None:
                                if term_at is None: term_at = now
                                if now-term_at >= args.termination_grace_s:
                                    try: os.killpg(child.pid, signal.SIGKILL)
                                    except ProcessLookupError: pass
                            if now-last >= args.heartbeat_s:
                                available = shutil.disk_usage(output).free
                                if available < args.min_free_gb*1024**3 and stop['disk'] is None:
                                    stop['disk'] = dict(free_bytes=available,
                                        minimum_reserve_gib=args.min_free_gb, next_case=case['name'],
                                        scope='Runtime disk guard stopped an incomplete attempt; resume archives it before a fresh simulation')
                                    try: os.killpg(child.pid, signal.SIGTERM)
                                    except ProcessLookupError: pass
                                progress = {}
                                if run_status.is_file():
                                    try:
                                        r = load(run_status)
                                        progress = {k:r[k] for k in ('stage','last_completed_slot','simulated_seconds','simulation_wall_s_so_far','completed') if k in r}
                                    except (OSError, ValueError): progress={'status_read':'temporarily_unavailable'}
                                state['active_case'].update(elapsed_wall_s=now-started_mono,
                                    progress=progress, free_bytes=available)
                                write(); last=now
                            time.sleep(min(.5, args.heartbeat_s))
                        code = child.wait(); child = None
                    if stop['disk'] is not None:
                        state.update(status='blocked_disk_space', completed=False, active_case=None,
                            disk=stop['disk']); write(); return 75
                    if stop['signal'] is not None:
                        state.update(status='interrupted', completed=False, active_case=None,
                            interruption_signal=stop['signal']); write(); return 128+stop['signal']
                    if code: raise RuntimeError('Simulation child failed: '+case['name']+' exit '+str(code))
                    state['active_case'].update(stage='auditing_completed_recordings'); write()
                    verify_inputs(plan, Path(plan['project_root']))
                    audit = audit_case(case, plan)
                verify_inputs(plan, Path(plan['project_root']))
                audit_path = _audit_path(output, case['name']); atomic_json(audit_path, audit)
                audits[case['name']] = audit
                state['completed_cases'].append(dict(name=case['name'], output=case['output'],
                    audit=str(audit_path), audit_sha256=sha256(audit_path),
                    reused_from=case.get('reused_from'), physical_run='reused_from' not in case,
                    completed_at_utc=datetime.now(timezone.utc).isoformat()))
                state['active_case'] = None
                atomic_json(output/'comparison-summary.json', dict(completed=False, pairs=pair_summaries(plan,audits)))
                write()
            verify_inputs(plan, Path(plan['project_root']))
            if stop['signal'] is not None:
                state.update(status='interrupted', completed=False, active_case=None); write(); return 128+stop['signal']
            atomic_json(output/'comparison-summary.json', dict(completed=True, pairs=pair_summaries(plan,audits),
                scope='Descriptive paired fixed-seed 30s-horizon evidence; not independent replicated confidence intervals. Frozen model and actual control coverage retained in per-case audits.'))
            state.update(status='complete', completed=True, active_case=None); write(); return 0
        except BaseException as error:
            if child is not None and child.poll() is None:
                try: os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=args.termination_grace_s)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                except ProcessLookupError: pass
            state.setdefault('failures', []).append(dict(case=(state.get('active_case') or {}).get('name'),
                type=type(error).__name__, message=str(error), at_utc=datetime.now(timezone.utc).isoformat()))
            state.update(status='failed', completed=False, active_case=None); write(); raise
        finally:
            for signum, handler in previous.items(): signal.signal(signum, handler)


def main(argv=None):
    args = parse_args(argv); plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False)); return 0
    return run_queue(args, plan)


if __name__ == '__main__': raise SystemExit(main())

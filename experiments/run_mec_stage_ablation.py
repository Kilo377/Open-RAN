"""Short matched native MEC scheduler ablation with exact task-stage diagnostics.

Runs OFF, three predeclared fixed scheduler actions, then a frozen RF serially.
This is a held-out diagnostic, not training or proof of RF superiority.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_xapp_long_validation import atomic_json, load, sha256, source_fingerprint
from experiments.run_latency_energy_study import artifact_stamp, jsonl, aggregate_offloaded_cohorts
from experiments.run_ue_energy_study import aggregate_cohorts
from experiments.run_throughput_mlp_study import actual_packet_metrics
from experiments.task_event_metrics import TaskEventCollector
from ric.near_rt.pools.chongyu.latency_scheduler_rf import ACTIONS, FEATURES
from ric.near_rt.pools.chongyu.online_rf import OnlineEnergyRF

ARMS = ('off', 'fixed05', 'fixed1', 'fixed2', 'rf')
FIXED = {'fixed05': 1, 'fixed1': 2, 'fixed2': 3}
POPULATION = {'ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8'}
TARGETS = ('ue_0', 'ue_4')
APP = 'latency_scheduler_rf'
STAGES = ('admission', 'upload', 'backhaul_to_edge', 'edge_compute',
          'backhaul_to_result', 'downlink', 'local_compute')
SERVICE = ('upload', 'edge_compute', 'local_compute')
SCHEMA = 'oran.mec-stage-timing.v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration-s', type=float, default=.8,
        help='Short diagnostic only: 0.3–1.6 s in complete 100 ms cohorts')
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--min-free-gb', type=float, default=.5)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    for key in ('output', 'model_path'):
        path = getattr(args, key)
        path = (path if path.is_absolute() else ROOT/path).resolve()
        if not path.is_relative_to(ROOT/'outputs') or path == ROOT/'outputs':
            raise ValueError('Model and dedicated experiment output must remain under project outputs/')
        setattr(args, key, path)
    if (not math.isfinite(args.duration_s) or not .3 <= args.duration_s <= 1.6 or
            not math.isclose(args.duration_s/.1, round(args.duration_s/.1), abs_tol=1e-8)):
        raise ValueError('Short diagnostic duration must be 0.3–1.6 s in whole 100 ms task cohorts')
    if not 0 <= args.seed <= 2**31-1:
        raise ValueError('Require a nonnegative supported simulation seed')
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < .25:
        raise ValueError('Require at least 0.25 GiB disk reserve')
    if args.model_path.is_relative_to(args.output):
        raise ValueError('Keep the read-only frozen model outside the new experiment directory')
    return args


def command_for(args, arm):
    if arm not in ARMS:
        raise ValueError('Unknown MEC diagnostic arm')
    mode = 'off' if arm == 'off' else 'frozen' if arm == 'rf' else 'fixed'
    command = ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast',
        '--duration-s', str(args.duration_s), '--seed', str(args.seed), '--traffic-load', '1',
        '--ue-services', 'drx-mec', '--mec-stage-metrics', '--xapp-pool', 'chongyu', '--record-compression', 'gzip',
        '--latency-xapp-mode', mode, '--latency-xapp-model', str(args.model_path),
        '--latency-xapp-window-ms', '100', '--channel-cache-root',
        str(ROOT/'channel_cache/mec_stage_ablation'/('seed_'+str(args.seed))),
        '--output', str(args.output/arm)]
    if arm in FIXED:
        command += ['--latency-xapp-fixed-action', str(FIXED[arm])]
    return command


def build_plan(args):
    learner = OnlineEnergyRF.load(args.model_path, FEATURES)
    if not learner.trained:
        raise ValueError('A genuinely fitted frozen MEC RF model is required')
    stamp = artifact_stamp(args.model_path)
    files = ['configs/rt_sys/agx_munich_6ue_depth2_fast.json',
        'configs/rt_sys/ue_services.json', 'scripts/orin/run-six-ue-cuda-nr.sh',
        'scripts/orin/activate-gpu.sh', 'scripts/orin/activate.sh']
    return dict(schema='oran.mec-stage-ablation.v1', source_sha256=source_fingerprint(ROOT),
        duration_s=args.duration_s, slots=round(args.duration_s*2000), seed=args.seed,
        model=dict(path=str(args.model_path), sha256=sha256(args.model_path), stamp=stamp),
        config_and_launcher_sha256={name:sha256(ROOT/name) for name in files},
        arms=list(ARMS), fixed_actions=FIXED, task_scoring_start_s=.1,
        stage_schema=SCHEMA, traffic='native 1x; no URLLC overlay; unchanged UE/radio/deadlines',
        pairing_scope='Short native diagnostic only (<=1.6 s): at most 32 births per UE with empty initial task queues and the native 50 ms task interval. Longer runs require separating exogenous births from endogenous admission/route changes.',
        interpretation='Short single-seed fixed-policy ablation and frozen-model diagnostic; no training, no claim that RF outperforms fixed policies.')


def verify_inputs(args, plan):
    if source_fingerprint(ROOT) != plan['source_sha256']:
        raise ValueError('Source changed; recorded data retained')
    if sha256(args.model_path) != plan['model']['sha256']:
        raise ValueError('Frozen model changed; recorded data retained')
    if any(sha256(ROOT/name) != digest for name,digest in plan['config_and_launcher_sha256'].items()):
        raise ValueError('Pinned configuration or launcher changed')


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid stage diagnostic '+name)
    return float(value)


def validate_timing(timing, *, created_s, observed_s, terminal):
    if not isinstance(timing, dict) or timing.get('schema') != SCHEMA:
        raise ValueError('A recorded v1 task-stage timing event is required')
    if set(timing.get('stage_ms', {})) != set(STAGES):
        raise ValueError('Incomplete task-stage residency vector')
    stages = {s:number(timing['stage_ms'][s], s) for s in STAGES}
    elapsed = number(timing.get('elapsed_ms'), 'elapsed_ms')
    if (not math.isclose(sum(stages.values()), elapsed, abs_tol=1e-6, rel_tol=1e-10) or
            not math.isclose(elapsed, (observed_s-created_s)*1000, abs_tol=1e-6, rel_tol=1e-10) or
            not math.isclose(number(timing.get('observed_until_s'), 'observed_until_s'), observed_s, abs_tol=1e-9)):
        raise ValueError('Task-stage time does not conserve elapsed simulation time')
    if timing.get('current_stage') not in STAGES or (
            timing.get('terminal_stage') not in STAGES if terminal else timing.get('terminal_stage') is not None):
        raise ValueError('Invalid task terminal/current stage')
    for key in ('service_ms', 'wait_or_quantization_ms'):
        if set(timing.get(key, {})) != set(SERVICE):
            raise ValueError('Incomplete active-service or waiting vector')
        for stage in SERVICE:
            number(timing[key][stage], key+'/'+stage)
    blocked = number(timing.get('upload_blocked_ms'), 'upload_blocked_ms')
    reasons = timing.get('upload_blocked_by_reason_ms')
    if not isinstance(reasons, dict) or not math.isclose(sum(number(v, str(k)) for k,v in reasons.items()), blocked, abs_tol=1e-6):
        raise ValueError('Upload-blocked reasons do not conserve blocked time')
    for stage in SERVICE:
        parts = timing['service_ms'][stage]+timing['wait_or_quantization_ms'][stage]+(blocked if stage=='upload' else 0)
        if not math.isclose(parts, stages[stage], abs_tol=1e-6, rel_tol=1e-10):
            raise ValueError('Stage service/wait/blocked decomposition mismatch')
    return timing


def timing_group(timings):
    count = len(timings)
    return dict(count=count,
        elapsed_mean_ms=sum(t['elapsed_ms'] for t in timings)/count if count else None,
        stage_mean_ms={s:sum(t['stage_ms'][s] for t in timings)/count if count else None for s in STAGES},
        service_mean_ms={s:sum(t['service_ms'][s] for t in timings)/count if count else None for s in SERVICE},
        wait_or_quantization_mean_ms={s:sum(t['wait_or_quantization_ms'][s] for t in timings)/count if count else None for s in SERVICE},
        upload_blocked_mean_ms=sum(t['upload_blocked_ms'] for t in timings)/count if count else None,
        terminal_stage_counts=dict(Counter(t['terminal_stage'] for t in timings if t['terminal_stage'] is not None)),
        current_stage_counts=dict(Counter(t['current_stage'] for t in timings)))


def task_stage_metrics(history, summary):
    exact = history.summary()
    eligible = {u:{c['index'] for c in exact['per_ue'][u]['cohorts']} for u in TARGETS}
    selected = {(u,tid):t for (u,tid),t in history.tasks.items() if t['cohort_index'] in eligible[u]}
    births = sorted(([u,tid,t['created_s'],t['deadline_s'],t['cohort_index'],t['offloaded']]
        for (u,tid),t in selected.items()), key=lambda row:(row[0],str(row[1])))
    global_pending = summary.get('ue_services', {}).get('task_stage_metrics', {}).get('pending', {})
    pending_rows = global_pending.get('tasks')
    if not isinstance(pending_rows, list):
        raise ValueError('Final pending task-stage identities are required')
    pending = {}
    for row in pending_rows:
        key = (row['ue_id'],row['task_id'])
        if key in pending:
            raise ValueError('Repeated pending task timing')
        task = history.tasks.get(key)
        if task is None or 'terminal' in task or row['created_s'] != task['created_s'] or row['offloaded'] != task['offloaded']:
            raise ValueError('Pending stage report disagrees with recorded task identity')
        validate_timing(row['task_stage_timing'], created_s=task['created_s'], observed_s=history.duration_s, terminal=False)
        pending[key] = row['task_stage_timing']
    if set(pending) != {key for key,t in history.tasks.items() if 'terminal' not in t}:
        raise ValueError('Pending task-stage population differs from task history')
    routes = {}
    for name, route in [('offloaded',True),('local',False),('unassigned',None)]:
        tasks = [(key,t) for key,t in selected.items() if t['offloaded'] is route]
        completed, failed, still_pending = [], [], []
        for key,task in tasks:
            event = task.get('terminal')
            if event is None:
                still_pending.append(pending[key]); continue
            timing = validate_timing(event.get('task_stage_timing'), created_s=task['created_s'],
                observed_s=event['time_s'], terminal=True)
            (completed if event['kind']=='task_completed' else failed).append(timing)
        routes[name] = dict(generated=len(tasks), completed=timing_group(completed),
            failed=timing_group(failed), pending=timing_group(still_pending),
            failure_stage_counts=dict(Counter(t['terminal_stage'] for t in failed)))
    cohorts = [c for u in TARGETS for c in exact['per_ue'][u]['cohorts']]
    per_ue = summary.get('ue_services', {}).get('per_ue', {})
    for uid in TARGETS:
        runtime = per_ue.get(uid, {}).get('ue_task_metrics', {})
        whole = exact['per_ue'][uid]['whole_run']
        for key in ('generated','completed','on_time_completed','failed','pending','offloaded_generated',
                    'offloaded_completed','offloaded_on_time_completed','offloaded_failed','offloaded_pending'):
            if runtime.get(key) != whole[key]:
                raise ValueError('Task event history differs from final cumulative counters')
        for key in ('completion_latency_ms_sum','offloaded_completion_latency_ms_sum'):
            if not math.isclose(number(runtime.get(key),key),whole[key],abs_tol=1e-6,rel_tol=1e-10):
                raise ValueError('Task-event latency sums differ from final cumulative counters')
        diagnostics=per_ue.get(uid,{}).get('ue_task_stage_metrics',{})
        if (diagnostics.get('schema') != SCHEMA or any(diagnostics.get(kind,{}).get('count') != whole[kind]
                for kind in ('completed','failed','pending'))):
            raise ValueError('Final per-UE task-stage counts differ from recorded task history')
    return dict(all_tasks=aggregate_cohorts(cohorts), offloaded_tasks=aggregate_offloaded_cohorts(cohorts),
        per_route=routes, mature_birth_identity_sha256=fingerprint(births),
        mature_creation_cohorts={u:[c['index'] for c in exact['per_ue'][u]['cohorts']] for u in TARGETS},
        whole_run_pending_by_ue={u:per_ue[u].get('ue_task_stage_metrics', {}).get('pending') for u in TARGETS},
        pending_scope='Mature pending ages remain separate from completion means; whole-run final pending includes right-censored cohorts.',
        history=exact)


def _provenance(receipt):
    values = receipt.get('provenance', [])
    values = [values] if isinstance(values, dict) else values
    return any(v.get('implementation_id') == APP for v in values)


def controller_audit(folder, cfg, summary, arm, model):
    apps = cfg['configs']['ric'].get('xapps', [])
    if arm == 'off':
        if apps:
            raise ValueError('OFF arm contains an active xApp')
        return dict(mode='off')
    mode = 'frozen' if arm == 'rf' else 'fixed'
    if len(apps) != 1 or apps[0].get('implementation_id') != APP:
        raise ValueError('Expected exactly the MEC scheduler xApp')
    params = apps[0]['parameters']; learning = summary.get('latency_xapp', {})
    if (params.get('mode') != mode or learning.get('mode') != mode or
            Path(params.get('model_path','')).resolve() != Path(model['path']).resolve()):
        raise ValueError('Scheduler mode or model differs from the declared arm')
    if any(learning.get(k) != model['stamp'][k] for k in ('samples','trained_samples','fit_count')):
        raise ValueError('Frozen or fixed arm changed the measured training history')
    if arm in FIXED and params.get('fixed_action') != FIXED[arm]:
        raise ValueError('Fixed scheduler arm uses the wrong action')
    events = list(jsonl(folder,'latency_rf_events'))
    if any(e.get('kind')=='rf_fit' for e in events):
        raise ValueError('Diagnostic evaluation must never fit a model')
    decisions = [e for e in events if e.get('kind')=='decision']
    confirmed = [e for e in events if e.get('kind')=='executed_policy_confirmed']
    receipts = [r for r in jsonl(folder,'control_receipts') if _provenance(r)]
    if any(r.get('status')=='rejected' for r in receipts):
        raise ValueError('Rejected scheduler control retained; diagnostic arm is invalid')
    applied = [r for r in receipts if r.get('status')=='applied']
    if not decisions or not confirmed or not applied:
        raise ValueError('Actual scheduler decisions, execution confirmation and applied receipts are required')
    for decision in decisions:
        action = decision.get('action')
        if type(action) is not int or not 0 <= action < len(ACTIONS):
            raise ValueError('Invalid recorded scheduler action')
        if (decision.get('scheduler_mode') != ACTIONS[action]['scheduler_mode'] or
                decision.get('ue_scheduler_weight') != ACTIONS[action]['weight']):
            raise ValueError('Recorded action does not match its physical scheduler mapping')
        if arm in FIXED and (action != FIXED[arm] or decision.get('reason') != 'predeclared_fixed_scheduler_action'):
            raise ValueError('Fixed policy was not actually chosen with its declared reason')
    for event in confirmed:
        cid = event['cell_id']; snapshot = event['executed_config']; action = event['action']
        prior = [e for e in decisions if e.get('cell_id')==cid and e['action']==action
                 and e['slot_id'] <= event['produced_slot']]
        if not prior or any(event.get('owners',{}).get(k) != apps[0].get('instance_id',APP)
                            for k in ('scheduler_mode','ue_scheduler_weights')):
            raise ValueError('Confirmation lacks earlier decision or executed control ownership')
        latest = max(prior,key=lambda e:e['slot_id'])
        if (snapshot.get('scheduler_mode') != ACTIONS[action]['scheduler_mode'] or
                any(snapshot.get('ue_scheduler_weights',{}).get(u) != ACTIONS[action]['weight'] for u in latest['ue_ids'])):
            raise ValueError('Confirmed scheduler differs from the selected action')
        if not any(r.get('apply_slot',r.get('slot_id',math.inf)) <= event['produced_slot']
                   and r.get('after',{}).get(cid)==snapshot for r in applied):
            # Receipt snapshots can contain extra executed fields. Compare the
            # two RCP fields rather than rejecting additional truthful telemetry.
            if not any(r.get('apply_slot',r.get('slot_id',math.inf)) <= event['produced_slot']
                       and all(r.get('after',{}).get(cid,{}).get(k)==snapshot.get(k)
                               for k in ('scheduler_mode','ue_scheduler_weights')) for r in applied):
                raise ValueError('Confirmation lacks an earlier matching applied scheduler receipt')
    prediction_delta = learning.get('prediction_batches',0)-model['stamp'].get('prediction_batches',0)
    if arm=='rf' and prediction_delta <= 0:
        raise ValueError('Frozen RF arm never used the fitted forest')
    return dict(mode=mode, decisions=len(decisions), confirmed=len(confirmed), applied_receipts=len(applied),
        action_counts=dict(Counter(str(e['action']) for e in decisions)),
        decision_reasons=dict(Counter(e['reason'] for e in decisions)), prediction_batches_this_run=prediction_delta,
        genuine_frozen_rf_used=arm=='rf' and prediction_delta>0, measured_history_unchanged=True,
        execution_ack_and_ownership_verified=True)


def audit_case(folder, arm, args, plan):
    status=load(folder/'run_status.json'); cfg=load(folder/'config_snapshot.json'); summary=load(folder/'final_summary.json')
    if (not status.get('completed') or not status.get('replay',{}).get('audits_passed') or
            not (folder/'network-replay.html').is_file()):
        raise ValueError('Completed simulation and audited automatic HTML required')
    if cfg.get('source_sha256') != plan['source_sha256'] or status.get('source_sha256') != plan['source_sha256']:
        raise ValueError('Recorded source differs from pinned diagnostic source')
    manifest=cfg['manifest']
    if (manifest['total_slots'] != plan['slots'] or manifest['slot_duration_ms'] != .5 or
            manifest['rng_seed'] != args.seed or cfg['configs']['sim']['rng_seed'] != args.seed or
            summary['total_slots'] != plan['slots'] or 'traffic_profile' in manifest):
        raise ValueError('Native duration, seed or traffic configuration changed')
    if cfg['configs']['phy']['rt_sys'].get('ue_services',{}).get('stage_metrics') is not True:
        raise ValueError('Explicit opt-in MEC stage metrics are required for every diagnostic arm')
    controller=controller_audit(folder,cfg,summary,arm,plan['model'])
    packets=actual_packet_metrics(folder,cfg,summary)
    history=TaskEventCollector(args.duration_s,cohort_ms=100,start_s=.1,target_uids=TARGETS)
    external=hashlib.sha256(); count=0
    for row in jsonl(folder,'rt_sys_trace'):
        count+=1
        if row.get('slot_id') != count or {u['ue_id'] for u in row['ues']} != POPULATION:
            raise ValueError('Every ordered six-UE trace slot must be recorded')
        arrivals=row.get('traffic_arrivals',{})
        if set(arrivals) != POPULATION:
            raise ValueError('Missing actual per-slot exogenous arrivals')
        values=[arrivals[u].get('exogenous_bytes') for u in sorted(POPULATION)]
        if any(type(v) is not int or v<0 for v in values):
            raise ValueError('Invalid exogenous arrival counter')
        external.update(json.dumps([count,values],separators=(',',':')).encode())
        for event in row.get('ue_service_events',[]):
            history.observe(event)
            if event.get('kind','').startswith('task_') and event.get('kind') != 'task_generated':
                validate_timing(event.get('task_stage_timing'),created_s=event['created_s'],
                    observed_s=event['time_s'],terminal=True)
    if count != plan['slots']:
        raise ValueError('Incomplete recorded simulation horizon')
    task_metrics=task_stage_metrics(history,summary)
    if not task_metrics['all_tasks']['generated']:
        raise ValueError('No mature target task population; cannot score this diagnostic')
    physics=dict(manifest={k:v for k,v in manifest.items() if k not in ('created_at','created_at_utc')},
        configs={k:cfg['configs'][k] for k in ('cell','ue','phy','sim')})
    return dict(arm=arm,source_sha256=plan['source_sha256'],model_sha256=plan['model']['sha256'],
        physical_inputs_sha256=fingerprint(physics),exogenous_slot_inputs_sha256=external.hexdigest(),
        slots=count,duration_s=args.duration_s,seed=args.seed,
        network_goodput_mbps=sum(u['goodput_mbps'] for u in packets.values()),
        tasks=task_metrics,controller=controller,automatic_html_audits_passed=True,
        files_sha256={str(p.relative_to(folder)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()})


def compare(cases):
    baseline=next((c for c in cases if c['arm']=='off'),None)
    if baseline is None:
        raise ValueError('OFF baseline is required before policy comparison')
    for case in cases:
        for key in ('source_sha256','model_sha256','physical_inputs_sha256','exogenous_slot_inputs_sha256'):
            if case[key] != baseline[key]:
                raise ValueError('Unmatched source/model/physical configuration/exogenous arrivals')
        if (case['tasks']['mature_birth_identity_sha256'] != baseline['tasks']['mature_birth_identity_sha256'] or
                case['tasks']['mature_creation_cohorts'] != baseline['tasks']['mature_creation_cohorts']):
            raise ValueError('Mature task birth/route identities differ between policies')
        a=baseline['tasks']['offloaded_tasks']['completed_task_mean_latency_ms']
        b=case['tasks']['offloaded_tasks']['completed_task_mean_latency_ms']
        case['vs_off']=dict(offloaded_completed_mean_latency_change_ms=b-a if a is not None and b is not None else None,
            offloaded_completed_mean_latency_reduction_pct=100*(1-b/a) if a is not None and a>0 and b is not None else None,
            scope='Completion-only offloaded mean; generated/on-time/failed/pending counts remain explicit.')
    return cases


def export_html(output,cases):
    def fmt(value): return '—' if value is None else f'{value:.3f}'
    primary=[]; stages=[]
    for case in cases:
        tasks=case['tasks'];off=tasks['offloaded_tasks'];all_tasks=tasks['all_tasks']
        primary.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (
            case['arm'],off['generated'],off['completed'],off['failed'],off['pending'],
            fmt(off['completed_task_mean_latency_ms']),fmt(all_tasks['on_time_rate']),
            fmt(case['network_goodput_mbps'])))+'</tr>')
        for route in ('offloaded','local'):
            r=tasks['per_route'][route];c=r['completed']
            stages.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (
                case['arm'],route,c['count'],fmt(c['elapsed_mean_ms']),
                *[fmt(c['stage_mean_ms'][s]) for s in STAGES],
                json.dumps(r['failure_stage_counts'],sort_keys=True),
                json.dumps(r['pending']['current_stage_counts'],sort_keys=True)))+'</tr>')
    (output/'comparison.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MEC scheduler stage ablation</title>
<style>body{font:14px/1.5 system-ui;margin:32px;color:#253846;background:#f7f9fb}h1{font-size:26px}h2{font-size:19px;margin-top:28px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%}th,td{padding:10px;text-align:right;border-bottom:1px solid #dce3e8;white-space:nowrap}th:first-child,td:first-child{text-align:left}p{max-width:1100px;color:#586775}</style>
<h1>MEC scheduler stage ablation</h1><p>Native six-UE network · unchanged radio resources, task demand and deadlines. OFF is round robin; fixed05/fixed1/fixed2 enable latency-aware scheduling at target UE weights 0.5/1/2. RF uses the unchanged frozen model. This short, single-seed diagnostic does not establish RF superiority over fixed policies.</p>
<p>Duration per policy: '''+html.escape(str(cases[0].get('duration_s','see results.json')))+''' s · simulation seed: '''+html.escape(str(cases[0].get('seed','see results.json')))+'''.</p>
<p>Mature task creation cohorts begin at 100 ms and have reached their original deadlines by run end. Completion means exclude failed and pending tasks; these counts remain explicit. Stage times are elapsed residency, not solely active service.</p>
<div class="scroll"><table><thead><tr><th>Policy</th><th>Offloaded generated</th><th>Completed</th><th>Failed</th><th>Pending</th><th>Offloaded mean E2E (ms)</th><th>All-task on-time fraction</th><th>Goodput (Mbps)</th></tr></thead><tbody>'''+''.join(primary)+'''</tbody></table></div>
<h2>Completed-task mean stage residency (ms)</h2><div class="scroll"><table><thead><tr><th>Policy</th><th>Route</th><th>Completed</th><th>E2E</th>'''+''.join('<th>'+html.escape(s.replace('_',' '))+'</th>' for s in STAGES)+'''<th>Failed by terminal stage</th><th>Mature pending by stage</th></tr></thead><tbody>'''+''.join(stages)+'''</tbody></table></div>
<p>Failed terminal-stage counts and pending ages are separate populations. results.json contains active upload/compute service, waiting or quantization, upload-blocked time, whole-run pending diagnostics and full provenance. Each arm also retains its original automatic network-replay.html.</p></html>''')


def launch(args,arm,plan):
    folder=args.output/arm; log=args.output/(arm+'.log')
    if folder.exists() or log.exists():
        raise ValueError('Existing partial recording/log retained; choose a fresh experiment directory')
    if shutil.disk_usage(args.output).free < args.min_free_gb*1024**3:
        raise RuntimeError('Disk reserve reached; completed data retained')
    command=command_for(args,arm)
    with log.open('x') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                atomic_json(args.output/'status.json',dict(stage='running',arm=arm,child_pid=process.pid,
                    command=command,source_sha256=plan['source_sha256'],heartbeat_unix=time.time()))
                if shutil.disk_usage(args.output).free < 256*1024**2:
                    raise RuntimeError('Runtime disk floor reached; partial recording retained')
                time.sleep(5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            verify_inputs(args,plan)
    if process.returncode:
        raise RuntimeError('Diagnostic arm failed; retained log '+str(log))


def main(argv=None):
    args=parse_args(argv);plan=build_plan(args)
    if args.dry_run:
        print(json.dumps(dict(plan=plan,commands=[command_for(args,arm) for arm in ARMS]),indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.study.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'plan.json'
        if path.exists():
            if not args.resume or load(path)!=plan:
                raise ValueError('Existing diagnostic requires --resume and identical plan/source/model/configuration')
        else:
            if any(p.name!='.study.lock' for p in args.output.iterdir()):
                raise ValueError('Existing recording without its pinned plan retained; choose a fresh output directory')
            atomic_json(path,plan)
        cases=[]
        try:
            for arm in ARMS:
                verify_inputs(args,plan)
                folder=args.output/arm; audit_path=args.output/(arm+'.audit.json')
                saved = None
                if folder.exists():
                    if not args.resume:
                        raise ValueError('Existing completed recording requires explicit --resume')
                    case=audit_case(folder,arm,args,plan)
                    saved = load(audit_path) if audit_path.exists() else None
                else:
                    if audit_path.exists():
                        raise ValueError('Saved audit without completed recording retained')
                    launch(args,arm,plan)
                    case=audit_case(folder,arm,args,plan)
                candidate=compare(cases+[case])
                if saved is not None and saved!=case:
                    raise ValueError('Completed raw recording differs from its immutable saved audit')
                cases=candidate;atomic_json(audit_path,case)
                atomic_json(args.output/'results.json',dict(status='running',plan=plan,cases=cases))
            verify_inputs(args,plan);export_html(args.output,cases)
            atomic_json(args.output/'results.json',dict(status='complete',plan=plan,cases=cases))
            atomic_json(args.output/'status.json',dict(stage='complete',source_sha256=plan['source_sha256'],model_sha256=plan['model']['sha256']))
        except BaseException as error:
            atomic_json(args.output/'status.json',dict(stage='failed',error_type=type(error).__name__,error=str(error),completed_arms=[c['arm'] for c in cases]))
            raise
    print(json.dumps(dict(output=str(args.output),arms=list(ARMS),duration_s=args.duration_s,seed=args.seed)))


if __name__=='__main__':
    main()

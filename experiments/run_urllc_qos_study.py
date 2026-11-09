"""Matched RR / fixed deadline-QoS / genuinely online RF pilot and validation.

Burst pressure is the primary experiment; native is retained as its reference.
An alternating scenario preserves live state across 0.8 s native/burst phases.
No failed case is discarded, and frozen evaluation never calls fit.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_latency_energy_study import jsonl
from experiments.run_throughput_mlp_study import actual_packet_metrics, arrival_audit, sha256
from experiments.run_xapp_long_validation import atomic_json, load, source_fingerprint

from experiments.urllc_control_audit import (
    AUDIT_VERSION, classify_stale_rejections, is_urllc_receipt, rejection_audit_metadata)
from experiments.urllc_scenarios import PROFILE, PROFILES, SCENARIOS, ALTERNATING_CYCLE_S, profile_for

CONTROLLER_AUDIT_VERSION = AUDIT_VERSION

URLLC = ('ue_2', 'ue_6')
RUN_OUTPUT = None


def atomic_copy_model(source, target):
    temporary=target.with_name(target.name+'.tmp')
    temporary.write_bytes(Path(source).read_bytes())
    temporary.replace(target)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('pilot', 'train', 'evaluate', 'all'), default='all')
    p.add_argument('--pilot-seconds', type=float, default=.8)
    p.add_argument('--train-seconds', type=float, default=30.)
    p.add_argument('--train-native-seconds', type=float, default=10.)
    p.add_argument('--eval-seconds', type=float, default=3.2,
        help='Per evaluation case; default covers native/burst/native/burst in the alternating scenario')
    p.add_argument('--eval-scenarios', choices=SCENARIOS, nargs='+', default=['burst', 'native', 'alternating'],
        help='Pressure first, native reference, then continuous-state adaptation; all policies share each scenario')
    p.add_argument('--train-seed', type=int, default=179)
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[55])
    p.add_argument('--min-free-gb', type=float, default=1.)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    for key in ('pilot_seconds', 'train_seconds', 'train_native_seconds', 'eval_seconds'):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0 or not math.isclose(value/.08, round(value/.08), abs_tol=1e-8):
            raise ValueError('Durations must be positive multiples of the 80 ms traffic cycle')
    if args.train_native_seconds >= args.train_seconds or not math.isclose((args.train_seconds-args.train_native_seconds)/.08,round((args.train_seconds-args.train_native_seconds)/.08),abs_tol=1e-8):
        raise ValueError('Training must include both native and burst episodes')
    if len(set(args.eval_scenarios)) != len(args.eval_scenarios):
        raise ValueError('Evaluation scenarios must be unique')
    if ('alternating' in args.eval_scenarios and
            not math.isclose(args.eval_seconds/ALTERNATING_CYCLE_S,
                             round(args.eval_seconds/ALTERNATING_CYCLE_S), abs_tol=1e-8)):
        raise ValueError('Alternating evaluation must include complete 1.6 s native/burst cycles')
    if (not math.isfinite(args.min_free_gb) or args.min_free_gb < .25
            or any(not 0 <= s <= 2**32-1 for s in args.eval_seeds+[args.train_seed])
            or args.train_seed in args.eval_seeds or len(set(args.eval_seeds)) != len(args.eval_seeds)):
        raise ValueError('Require separate train/test seeds, unique evaluation seeds and a disk reserve')
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT/'outputs') or args.output == ROOT/'outputs':
        raise ValueError('Choose a dedicated experiment output directory')
    return args


def command_for(output, name, scenario, arm, seconds, seed, model):
    profile_for(scenario)  # Fail closed instead of treating a typo as native.
    if arm not in ('rr', 'train', 'rf', 'qos1', 'qos2', 'qos4'):
        raise ValueError('Unknown URLLC policy: '+str(arm))
    mode = 'off' if arm == 'rr' else 'online' if arm == 'train' else 'frozen' if arm == 'rf' else 'fixed'
    cmd = ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast',
        '--duration-s', str(seconds), '--seed', str(seed), '--traffic-load', '2',
        '--urllc-scenario', scenario,
        '--xapp-pool', 'chongyu', '--ue-services', 'drx-mec', '--record-compression', 'gzip',
        '--urllc-xapp-policy-version', 'qos_v2', '--urllc-xapp-mode', mode,
        '--urllc-xapp-model', str(model), '--urllc-xapp-window-ms', '100',
        '--channel-cache-root', str(ROOT/'channel_cache'/'urllc_qos'/('seed_'+str(seed))),
        '--output', str(output/name)]
    if arm.startswith('qos'):
        cmd += ['--urllc-xapp-fixed-action', str({'qos1': 1, 'qos2': 2, 'qos4': 3}[arm])]
    if arm == 'train':
        cmd += ['--urllc-xapp-balanced-start-windows', '40']
    return cmd


def cases_for(args, phase):
    if phase == 'train':
        return [('training/native', 'native', 'train', args.train_native_seconds, args.train_seed),
            ('training/burst', 'burst', 'train', args.train_seconds-args.train_native_seconds, args.train_seed)]
    arms = ('rr', 'qos1', 'qos2', 'qos4') if phase == 'pilot' else ('rr', 'qos1', 'qos2', 'qos4', 'rf')
    seconds = args.pilot_seconds if phase == 'pilot' else args.eval_seconds
    scenarios = ('burst', 'native') if phase == 'pilot' else args.eval_scenarios
    return [(f'{phase}/{scenario}_seed{seed}_{arm}', scenario, arm, seconds, seed)
        for seed in args.eval_seeds for scenario in scenarios for arm in arms]


def plan_for(args):
    return dict(source_sha256=source_fingerprint(), train_seed=args.train_seed,
        evaluation_seeds=args.eval_seeds, evaluation_scenarios=list(args.eval_scenarios),
        profile_sha256=sha256(PROFILE),
        profiles_sha256={name: sha256(path) for name,path in PROFILES.items() if path is not None},
        pilot_seconds=args.pilot_seconds, train_seconds=args.train_seconds,
        train_native_seconds=args.train_native_seconds, eval_seconds=args.eval_seconds)


def cohort_latency(folder, cfg):
    """Deadline-capped latency across ALL mature offered URLLC packets.

    Failed/pending packets receive their original deadline; this is an explicit
    bounded delay cost, not an estimate of their unobserved completion latency.
    """
    stop = cfg['manifest']['total_slots']; dt = cfg['manifest']['slot_duration_ms']
    rows = {u: {} for u in URLLC}
    for e in jsonl(folder, 'packet_events'):
        uid = e['ue_id']
        if uid not in rows:
            continue
        if e['kind'] in ('arrival', 'initial') and (stop-e['slot_id'])*dt > 10:
            rows[uid][e['packet_id']] = 10.
        elif e['kind'] == 'delivery' and e.get('completed') and e['packet_id'] in rows[uid]:
            rows[uid][e['packet_id']] = min(10., float(e['latency_ms']))
    vals = [v for r in rows.values() for v in r.values()]
    return dict(mature_packet_count=len(vals), deadline_capped_delay_cost_ms=sum(vals)/len(vals) if vals else None,
        scope='all mature offered packets; failed/pending cost=10ms, NOT measured completion latency')


def verify_qos_sample(row):
    from ric.near_rt.pools.chongyu.online_urllc_qos_rf import TARGETS
    if row['target_order'] != TARGETS or row['original_latency_budgets_ms'] != {'ue_2': 10, 'ue_6': 10}:
        raise ValueError('QoS target schema or original deadlines changed')
    action = row['action']; duration = row['duration_s']
    if type(action) is not int or not 0 <= action < 4 or row['features'][-4:] != [float(i == action) for i in range(4)]:
        raise ValueError('QoS measured action encoding mismatch')
    if not math.isfinite(duration) or duration < .1-1e-12:
        raise ValueError('Invalid QoS measured duration')
    for name in ('delivered_bits_by_ue','offered_bits_by_ue'):
        values=row[name]
        if set(values) != {'ue_0','ue_1','ue_2','ue_4','ue_6','ue_8'} or any(type(v) is not int or v < 0 for v in values.values()):
            raise ValueError('Incomplete or invalid measured network counters')
    for name in ('deadline_failed_packets_by_ue','overflow_packets_by_ue','completed_packets_by_ue',
            'admitted_packets_by_ue','on_time_completed_packets_by_ue','late_completed_packets_by_ue'):
        values=row[name]
        if set(values) != set(URLLC) or any(type(v) is not int or v < 0 for v in values.values()):
            raise ValueError('Incomplete or invalid measured URLLC counters')
    latency=row['completion_latency_ms_by_ue']
    if set(latency) != set(URLLC) or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in latency.values()):
        raise ValueError('Invalid measured completion latency sums')
    for u in URLLC:
        if row['completed_packets_by_ue'][u] != row['on_time_completed_packets_by_ue'][u]+row['late_completed_packets_by_ue'][u]:
            raise ValueError('Measured completion counters disagree')
    for values in row['snapshot'].values():
        if values['scheduler_mode'] != ('round_robin' if action == 0 else 'qos_deadline'):
            raise ValueError('QoS measured action mode mismatch')
        if any(w != ((1., 1., 2., 4.)[action] if u in URLLC else 1.) for u,w in values['ue_scheduler_weights'].items()):
            raise ValueError('QoS measured UE weight mismatch')
    failures = [(row['deadline_failed_packets_by_ue'][u]+row['overflow_packets_by_ue'][u])/duration for u in URLLC]
    targets = [sum(failures), sum(row['completion_latency_ms_by_ue'][u] for u in URLLC)/duration,
        sum(row['completed_packets_by_ue'][u] for u in URLLC)/duration,
        sum(row['delivered_bits_by_ue'].values())/duration/1e6, *failures]
    if len(row['targets']) != len(targets) or any(not math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-12) for a,b in zip(row['targets'],targets)):
        raise ValueError('QoS target differs from measured packet counter arithmetic')


def training_audit(folder, cfg, summary, model, previous=None):
    from experiments.run_scheduler_conflict_study import _f32, _verify_raw_packet_windows
    from ric.near_rt.pools.chongyu.online_urllc_qos_rf import OnlineUrllcQosRF
    from ric.near_rt.pools.chongyu.latency_urllc_qos_rf import FEATURES
    learner = OnlineUrllcQosRF.load(model,FEATURES)
    before=previous or dict(x=[],y=[],fit_count=0)
    apps=cfg['configs']['ric'].get('xapps',[])
    if (len(apps) != 1 or apps[0].get('implementation_id') != 'latency_urllc_rf'
            or apps[0]['parameters'].get('policy_version') != 'qos_v2' or apps[0]['parameters'].get('mode') != 'online'):
        raise ValueError('Training requires the isolated actual online QoS RF controller')
    declared=summary.get('urllc_xapp',{})
    if any(declared.get(k) != v for k,v in dict(samples=learner.sample_count,trained_samples=learner.trained_samples,fit_count=learner.fit_count).items()):
        raise ValueError('Training artifact and actual learner summary disagree')
    events = list(jsonl(folder,'urllc_rf_events'))
    samples = [e for e in events if e['kind'] == 'measured_sample']
    confirmations = [e for e in events if e['kind'] == 'executed_policy_confirmed']
    fits = [e for e in events if e['kind'] == 'rf_fit']
    checkpoints = [e for e in events if e['kind'] == 'model_checkpoint']
    last_end = -1
    for row in samples:
        verify_qos_sample(row)
        start,end = row['produced_start_slot'],row['produced_end_slot']
        if start < max(0,last_end) or end <= start or end > cfg['manifest']['total_slots'] or not math.isclose(row['duration_s'],(end-start)*cfg['manifest']['slot_duration_ms']/1000,abs_tol=1e-12):
            raise ValueError('QoS sample has overlapping or invalid clock interval')
        eligible = [c for c in confirmations if c['produced_slot'] <= start]
        c = eligible[-1] if eligible else None
        if (c is None or c['action'] != row['action'] or c['executed_snapshot'] != row['snapshot']
                or (start-c['produced_slot'])*cfg['manifest']['slot_duration_ms'] < 20-1e-9):
            raise ValueError('QoS label lacks held executed action and wash-in')
        membership=row['membership'];covered=set()
        if set(membership) != {'ue_0','ue_1','ue_2','ue_4','ue_6','ue_8'}:
            raise ValueError('Incomplete measured membership')
        for cid,snapshot in row['snapshot'].items():
            owned=c['owners'].get(cid,{})
            expected={u for u,cell in membership.items() if cell == cid}
            if (owned.get('scheduler_mode') != 'latency_urllc_rf'
                    or set(snapshot['ue_scheduler_weights']) != expected
                    or (expected and owned.get('ue_scheduler_weights') != 'latency_urllc_rf')):
                raise ValueError('Measured sample lacks full executed ownership')
            covered.update(expected)
        if covered != set(membership):
            raise ValueError('Measured sample omits a served UE')
        last_end=end
    _verify_raw_packet_windows(folder,samples,cfg)
    retained = [r for r in samples if r['retained_for_training']]
    if learner._x != before['x']+[_f32(r['features']) for r in retained] or learner._y != before['y']+[r['targets'] for r in retained]:
        raise ValueError('Saved RF training rows differ from actual measured online windows')
    if learner.fit_count != before['fit_count']+len(fits) or len(checkpoints) != len(fits) or not fits or learner.trained_samples != fits[-1]['samples']:
        raise ValueError('Saved RF fitted prefix lacks real fits/checkpoints')
    counts = Counter(x[-4:].index(1.) for x in learner._x[:learner.trained_samples])
    if learner.trained_samples < 40 or any(counts[i] < 10 for i in range(4)):
        raise ValueError('Saved fitted RF has insufficient actual per-arm coverage')
    for index,(fit,cp) in enumerate(zip(fits,checkpoints),before['fit_count']+1):
        path=Path(cp['path'])
        if fit['fit_count'] != index or cp['fit_count'] != index or sha256(path) != cp['sha256']:
            raise ValueError('Online fit checkpoint sequence or hash mismatch')
        value=load(path)
        if value['fit_count'] != index or value['trained_samples'] != fit['samples'] or value['x'] != learner._x[:fit['samples']] or value['y'] != learner._y[:fit['samples']]:
            raise ValueError('Checkpoint does not preserve the measured fit prefix')
    return dict(actual_measured_rows=len(retained), prior_measured_rows=len(before['x']), fitted_rows=learner.trained_samples,
        fitted_action_counts=dict(counts), fits=len(fits), checkpoints=len(checkpoints),
        raw_packet_counter_arithmetic_verified=True, saved_measured_prefix_verified=True,
        latest_model_sha256=sha256(model), label_scope='outcome_windows_including_carryover_not_cohort_probabilities')


def controller_audit(folder, cfg, summary, arm, model):
    if arm == 'rr':
        if cfg['configs']['ric'].get('xapps'):
            raise ValueError('Native RR must have no active xApp')
        return dict(mode='off')
    events = list(jsonl(folder, 'urllc_rf_events'))
    decisions = [e for e in events if e['kind'] == 'decision']
    confirmed = [e for e in events if e['kind'] == 'executed_policy_confirmed']
    samples = [e for e in events if e['kind'] == 'measured_sample']
    receipts = list(jsonl(folder, 'control_receipts'))
    relevant = [r for r in receipts if is_urllc_receipt(r)]
    # Classify only proven stale-membership atomic no-ops. No receipt is removed:
    # every confirmation below still needs an earlier matching applied ACK in
    # the complete original raw recording, and every other rejection is fatal.
    classified = classify_stale_rejections(receipts, decisions)
    applied = [r for r in relevant if r.get('status') == 'applied']
    if not decisions or not confirmed or not applied:
        raise ValueError('URLLC action lacks actual inference/request, executed ownership or applied ACK')
    from experiments.run_scheduler_conflict_study import _confirmation_actuated
    if any(not _confirmation_actuated(e, receipts) for e in confirmed):
        raise ValueError('Observed ownership lacks an earlier matching applied action')
    if arm.startswith('qos') and any(e['action'] != {'qos1':1,'qos2':2,'qos4':3}[arm] or e['status'] != 'fixed' for e in decisions):
        raise ValueError('Fixed ablation did not execute its predeclared action')
    result = dict(decisions=len(decisions), confirmed=len(confirmed), applied_receipts=len(applied),
        measured_windows=len(samples), action_counts=dict(Counter(str(e['action']) for e in decisions)),
        decision_status_counts=dict(Counter(e['status'] for e in decisions)),
        summary=summary.get('urllc_xapp'), actual_action_ack_and_ownership_verified=True,
        **rejection_audit_metadata(classified))
    if arm == 'rf':
        from ric.near_rt.pools.chongyu.online_urllc_qos_rf import OnlineUrllcQosRF
        from ric.near_rt.pools.chongyu.latency_urllc_qos_rf import FEATURES
        learner = OnlineUrllcQosRF.load(model, FEATURES)
        if not all(e['status'] == 'rf' for e in decisions):
            raise ValueError('Frozen evaluation must use RF on every decision')
        max_diff = 0.
        for e in decisions:
            probes = [e['features'][:-4]+[float(i == j) for i in range(4)] for j in range(4)]
            pred = learner.predict(probes)
            recorded = e['prediction_audit']['raw_predictions']
            max_diff = max(max_diff, max(abs(a-b) for p,q in zip(pred, recorded) for a,b in zip(p,q)))
            if e['action'] != e['prediction_audit']['selected_action']:
                raise ValueError('Recorded selected action disagrees with the RF prediction audit')
        if max_diff > 1e-10:
            raise ValueError('Frozen model prediction does not replay from the recorded state')
        result['frozen_prediction_replay_max_abs_error'] = max_diff
    return result


def audit_case(folder, arm, model, previous=None):
    cfg = load(folder/'config_snapshot.json'); summary = load(folder/'final_summary.json')
    status = load(folder/'run_status.json')
    if not status.get('completed') or not status.get('replay', {}).get('audits_passed'):
        raise ValueError('Simulation must complete with audited automatic HTML')
    if cfg['source_sha256'] != status['source_sha256']:
        raise ValueError('Source snapshot mismatch')
    users = actual_packet_metrics(folder, cfg, summary)
    offered = sum(users[u]['mature_offered_packets'] for u in URLLC)
    failed = sum(users[u]['mature_failed_packets'] for u in URLLC)
    delay = cohort_latency(folder, cfg)
    if delay['mature_packet_count'] != offered:
        raise ValueError('URLLC cohort denominator mismatch')
    physics=dict(manifest=cfg['manifest'],cell=cfg['configs']['cell'],ue=cfg['configs']['ue'],phy=cfg['configs']['phy'])
    physics['manifest'] = {k:v for k,v in physics['manifest'].items() if k not in ('created_at','created_at_utc')}
    result = dict(folder=str(folder), source_sha256=cfg['source_sha256'],
        matched_physical_inputs_sha256=hashlib.sha256(json.dumps(physics,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        simulated_s=cfg['manifest']['total_slots']*cfg['manifest']['slot_duration_ms']/1000,
        network_goodput_mbps=sum(u['goodput_mbps'] for u in users.values()),
        urllc=dict(mature_offered_packets=offered, mature_failed_packets=failed,
            failure_fraction=failed/offered if offered else None,
            worst_ue_failure_fraction=max(users[u]['mature_failure_fraction'] for u in URLLC), **delay),
        per_ue=users, arrivals=arrival_audit(folder, users),
        controller=controller_audit(folder, cfg, summary, arm, model),
        simulation_wall_s=load(folder/'performance.json')['simulation_s'],
        model_statistics=summary.get('model_statistics'), html=str(folder/'network-replay.html'))
    if arm == 'train':
        result['online_training_audit']=training_audit(folder,cfg,summary,model,previous)
    if arm == 'rf':
        result['frozen_model_sha256']=sha256(model)
    return result


def compare(cases):
    grouped = {}
    for c in cases:
        group = grouped.setdefault((c['scenario'], c['seed']), {})
        if c['arm'] in group:
            raise ValueError('Duplicate arm in a paired scenario')
        group[c['arm']] = c
    for group in grouped.values():
        base = group.get('rr')
        if base is None:
            raise ValueError('Every paired scenario requires its own RR reference')
        for c in group.values():
            if (c['source_sha256'] != base['source_sha256'] or c['arrivals']['exogenous_slot_inputs_sha256'] != base['arrivals']['exogenous_slot_inputs_sha256']
                    or c.get('matched_physical_inputs_sha256') != base.get('matched_physical_inputs_sha256')):
                raise ValueError('Paired arms have different source or exogenous arrivals')
            for uid in URLLC:
                if c['per_ue'][uid]['mature_offered_inputs_sha256'] != base['per_ue'][uid]['mature_offered_inputs_sha256']:
                    raise ValueError('Paired mature packet cohort mismatch')
            c['vs_rr'] = dict(failure_change_pp=100*(c['urllc']['failure_fraction']-base['urllc']['failure_fraction']),
                network_goodput_change_pct=100*(c['network_goodput_mbps']/base['network_goodput_mbps']-1))
    return cases


def export_html(output, cases):
    body = []
    for c in cases:
        url = c['urllc']
        per = c['per_ue']
        body.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (
            c['scenario'], c['seed'], c['arm'], f"{url['failure_fraction']*100:.3f}%",
            f"{per['ue_2']['mature_failure_fraction']*100:.3f}%", f"{per['ue_6']['mature_failure_fraction']*100:.3f}%",
            f"{url['deadline_capped_delay_cost_ms']:.3f}", f"{c['network_goodput_mbps']:.3f}",
            url['mature_offered_packets']))+'</tr>')
    (output/'comparison.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>URLLC Scheduler Validation</title><style>body{font:16px system-ui;margin:40px;color:#182b3a;background:#f7f9fc}table{border-collapse:collapse;background:white;width:100%}th,td{padding:12px;text-align:right;border-bottom:1px solid #dae1e8}th:first-child,td:first-child{text-align:left}h1{font-size:26px}p{max-width:1000px;color:#526273}</style>
<h1>URLLC Scheduler Validation</h1><p>6 moving UEs · 3 cells · 10 ms URLLC deadline · Fast system model. Burst pressure is the primary experiment; native CBR is its reference. Alternating runs repeat 0.8 s native / 0.8 s equal-mean burst with queues, channel and controller state preserved. Paired policies use identical per-slot exogenous arrivals and mature packet cohorts.</p>
<table><thead><tr><th>Scenario</th><th>Seed</th><th>Policy</th><th>URLLC failure</th><th>UE2 failure</th><th>UE6 failure</th><th>Capped delay cost (ms)</th><th>Network goodput (Mbps)</th><th>Mature packets</th></tr></thead><tbody>'''+''.join(body)+'''</tbody></table><p>Failure = unique overdue/lost/overflow packets divided by all mature offered URLLC packets, pooled by packet count. Capped delay is an explicit cost: unsuccessful or pending packets receive 10 ms; successful completion delay is capped at 10 ms. It is not physical end-to-end latency. Short pilots do not establish carrier-grade reliability.</p></html>''')


def main():
    global RUN_OUTPUT
    RUN_OUTPUT = None
    args = parse_args(); output = args.output
    model = output/'models/urllc-qos-rf.json'
    phases = ('pilot', 'train', 'evaluate') if args.phase == 'all' else (args.phase,)
    plan = plan_for(args)
    if args.dry_run:
        print(json.dumps(dict(plan=plan, commands=[command_for(output,*c[:3],c[3],c[4],model)
            for phase in phases for c in cases_for(args,phase)]), indent=2)); return
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.study.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        RUN_OUTPUT = output
        if (output/'plan.json').exists():
            if not args.resume or load(output/'plan.json') != plan:
                raise ValueError('Existing study requires --resume and identical pinned inputs')
        else:
            atomic_json(output/'plan.json',plan)
        model.parent.mkdir(parents=True, exist_ok=True)
        all_cases = []
        for phase in phases:
            results = []
            for name, scenario, arm, seconds, seed in cases_for(args,phase):
                if source_fingerprint() != plan['source_sha256']:
                    raise ValueError('Source changed while study was running')
                if shutil.disk_usage(output).free < args.min_free_gb*1024**3:
                    raise ValueError('Disk reserve reached; completed data retained')
                if arm == 'rf':
                    stamp=load(output/'training-model-stamp.json')
                    if sha256(model) != stamp['sha256']:
                        raise ValueError('Frozen model differs from the pinned training checkpoint')
                folder = output/name; folder.parent.mkdir(parents=True,exist_ok=True)
                audit_path = folder.with_name(folder.name+'.audit.json')
                before_path=folder.with_name(folder.name+'.model-before.json')
                after_path=folder.with_name(folder.name+'.model-after.json')
                if audit_path.exists() and args.resume:
                    if arm == 'train':
                        saved=load(audit_path)
                        if (saved['source_sha256'] != plan['source_sha256']
                                or sha256(after_path) != saved['episode_model_sha256']):
                            raise ValueError('Completed training audit source differs')
                        prior=load(before_path) if before_path.exists() else None
                        audit=audit_case(folder,arm,after_path,previous=prior)
                        audit['episode_model_sha256']=sha256(after_path)
                    else:
                        audit = audit_case(folder,arm,model)
                else:
                    if folder.exists():
                        raise ValueError('Incomplete case retained; choose a fresh study directory: '+str(folder))
                    if arm == 'train' and scenario == 'native' and model.exists():
                        raise ValueError('Fresh online training requires a new measured model path')
                    previous=load(model) if arm == 'train' and model.exists() else None
                    if previous is not None:
                        atomic_copy_model(model,before_path)
                    before = sha256(model) if model.is_file() else None
                    command = command_for(output,name,scenario,arm,seconds,seed,model)
                    started=time.time()
                    running=dict(stage='running',case=name,command=command,started_unix=started,
                        parent_pid=os.getpid(),source_sha256=plan['source_sha256'])
                    atomic_json(output/'status.json',running)
                    with folder.with_name(folder.name+'.log').open('w') as stream:
                        proc = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        try:
                            while proc.poll() is None:
                                free=shutil.disk_usage(output).free
                                if free < .25*1024**3:
                                    raise RuntimeError('Runtime disk floor reached; partial case preserved')
                                atomic_json(output/'status.json',dict(running,child_pid=proc.pid,
                                    heartbeat_unix=time.time(),wall_s=time.time()-started,free_disk_bytes=free))
                                time.sleep(5)
                        finally:
                            if proc.poll() is None:
                                os.killpg(proc.pid,signal.SIGTERM)
                                try:proc.wait(timeout=15)
                                except subprocess.TimeoutExpired:
                                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    if proc.returncode:
                        raise RuntimeError('Case failed; see '+str(folder.with_name(folder.name+'.log')))
                    if arm == 'rf' and sha256(model) != before:
                        raise ValueError('Frozen evaluation changed model bytes')
                    audit = audit_case(folder,arm,model,previous=previous)
                    if arm == 'train':
                        atomic_copy_model(model,after_path)
                        audit['episode_model_sha256']=sha256(after_path)
                audit.update(name=name,scenario=scenario,arm=arm,seed=seed)
                atomic_json(audit_path,audit); results.append(audit)
            if phase == 'train':
                payload = load(model)
                if payload['trained_samples'] < 40 or payload['fit_count'] < 1:
                    raise ValueError('Actual online training did not reach fitted coverage')
                atomic_json(output/'training-model-stamp.json',dict(sha256=sha256(model),
                    samples=len(payload['x']),trained_samples=payload['trained_samples'],fit_count=payload['fit_count']))
            else:
                compare(results); atomic_json(output/(phase+'-results.json'),results)
                all_cases.extend(results)
                export_html(output,results)
        atomic_json(output/'status.json',dict(stage='complete',completed_phases=phases,
            source_sha256=plan['source_sha256'],model_sha256=sha256(model) if model.is_file() else None))
        print(json.dumps(dict(output=str(output),completed_phases=phases),indent=2))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if RUN_OUTPUT is not None:
            atomic_json(RUN_OUTPUT/'status.json',dict(stage='failed',error_type=type(error).__name__,
                error=str(error),failed_unix=time.time()))
        raise

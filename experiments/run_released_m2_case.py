"""Evaluate the released X02/X08 pair with simultaneous, disjoint controls.

The four matched recordings distinguish standalone target harm from a violated
operator contract. Models are frozen; this driver never fits or saves a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_ap_energy_study import (
    APP, case_data, ap_control_evidence, model_stamp as ap_stamp,
    readiness as ap_readiness,
)
from experiments.run_scheduler_conflict_study import (
    X08, X08_FEATURES, X08_TARGETS, controller_evidence as scheduler_evidence,
    jsonl,
)
from experiments.run_throughput_mlp_study import model_stamp as mlp_stamp
from experiments.run_xapp_long_validation import load, sha256, source_fingerprint
from experiments.run_ue_energy_study import POPULATION, mature_cohorts
from experiments.run_mec_budget_pressure_study import task_metrics
from experiments.task_event_metrics import TaskEventCollector

ARMS = ('base', 'x02', 'x08', 'both')
MODEL_HASHES = {
    'x02': '686c00d6995f363211ec556623052c9ef807f004908c150aebd9e7e5a9e17935',
    'x08': 'cad8b33db9a2d2f0fb622cb116456d29fc6f1545afa95b759477b07d73eb21fe',
}
MODEL_ROWS = {'x02': 250, 'x08': 254}
APP_IDS = {'x02': APP, 'x08': X08}
RCP = {APP: {'cell_sleep_mode'}, X08: {'scheduler_mode', 'ue_scheduler_weights'}}
PRIMARY_CONTRACTS = {
    'x02': dict(metric='network_ap_energy_j', comparison='<=', base_fraction=.95),
    'x08': dict(metric='network_goodput_mbps', comparison='>=', base_fraction=1.05),
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def parse_args(argv=None, root=ROOT):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('outputs/trained_xapps/20261109'))
    p.add_argument('--model-manifest', type=Path, default=Path('models.json'))
    p.add_argument('--ap-services-profile', type=Path, default=Path('configs/rt_sys/ap_services_reference.json'))
    p.add_argument('--duration-s', type=float, default=3.)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--min-free-gb', type=float, default=.25)
    p.add_argument('--estimated-case-gb', type=float, default=.08)
    p.add_argument('--heartbeat-s', type=float, default=10.)
    p.add_argument('--termination-grace-s', type=float, default=15.)
    p.add_argument('--max-parallel', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv); root = Path(root).resolve()
    for key in ('output', 'model_dir', 'ap_services_profile'):
        path = getattr(args, key)
        setattr(args, key, (path if path.is_absolute() else root/path).resolve())
    args.model_manifest = (args.model_dir/args.model_manifest).resolve()
    if (not args.output.is_relative_to(root/'outputs') or args.output == root/'outputs' or
            not args.model_dir.is_relative_to(root/'outputs') or
            args.model_dir.is_relative_to(args.output) or args.output.is_relative_to(args.model_dir)):
        raise ValueError('Use a dedicated outputs/ study separate from read-only installed models')
    if not args.model_manifest.is_relative_to(args.model_dir):
        raise ValueError('Model manifest must remain in the installed model directory')
    if args.ap_services_profile != root/'configs/rt_sys/ap_services_reference.json':
        raise ValueError('This predeclared case uses the explicit AP reference profile')
    if (not math.isfinite(args.duration_s) or args.duration_s < .3 or
            not math.isclose(args.duration_s/.1, round(args.duration_s/.1), abs_tol=1e-8)):
        raise ValueError('Use at least 0.3 s in whole 100 ms observation windows')
    if not 0 <= args.seed <= 2**32-1 or args.seed == 55:
        raise ValueError('Use a supported fresh seed distinct from training seed55')
    for key in ('heartbeat_s', 'termination_grace_s'):
        if not math.isfinite(getattr(args, key)) or not 0 < getattr(args, key) <= 60:
            raise ValueError('Heartbeat and termination grace must be in (0,60] seconds')
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < .25:
        raise ValueError('Retain at least 0.25 GiB disk reserve')
    if not math.isfinite(args.estimated_case_gb) or args.estimated_case_gb <= 0:
        raise ValueError('Positive per-case disk estimate required')
    return args


def released_models(args):
    entries = load(args.model_manifest)['models']; result = {}
    for app in ('x02', 'x08'):
        entry = entries[app]; path = (args.model_dir/entry['path']).resolve()
        if not path.is_relative_to(args.model_dir) or sha256(path) != MODEL_HASHES[app] or entry['sha256'] != MODEL_HASHES[app]:
            raise ValueError('Expected exact latest released frozen model: '+app)
        payload = load(path); stamp = (ap_stamp if app == 'x02' else mlp_stamp)(path)
        if stamp['samples'] != MODEL_ROWS[app] or stamp['trained_samples'] != MODEL_ROWS[app] or stamp['fit_count'] <= 0:
            raise ValueError('Latest release must retain every measured/fitted row: '+app)
        if app == 'x02':
            ready = ap_readiness(path)
        else:
            # The historical pilot-study readiness helper caps rows at48. The
            # current runtime loader validates all254 rows without refitting.
            from ric.near_rt.pools.chongyu.online_throughput_mlp import OnlineThroughputMLP
            learner = OnlineThroughputMLP.load(path, X08_FEATURES, X08_TARGETS)
            counts = Counter(row[-4:].index(1.) for row in payload['x'][:stamp['trained_samples']])
            ready = dict(ready=bool(learner.optimizer_verified and stamp['optimizer_verified'] and
                all(counts[i] >= 6 for i in range(4))), fitted_action_counts=dict(counts))
        if not ready['ready']:
            raise ValueError('Measured fitted action coverage is insufficient: '+app)
        result[app] = dict(path=str(path), sha256=MODEL_HASHES[app], stamp=stamp, readiness=ready)
    return result


def build_plan(args, root=ROOT):
    root = Path(root).resolve(); models = released_models(args)
    from phy.rt_sys.scenario import build_case
    _, expected_configs, _ = build_case('munich', 3, 6, args.seed, round(args.duration_s*2000),
        device='cuda', config_path=root/'configs/rt_sys/agx_munich_6ue_depth2_fast.json',
        # run_rt_sys parses --traffic-load as float. Preserve that type when
        # deriving the exact serialized input hash (e.g. burst bytes4400.0).
        traffic_load=2.0, xapp_pool='chongyu')
    if load(args.ap_services_profile)['ap_services'].get('enabled') is not True:
        raise ValueError('Explicit enabled AP profile is required')
    cases = []
    for arm in ARMS:
        folder = args.output/'cases'/arm
        command = ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast',
            '--fast-profile', 'system', '--duration-s', str(args.duration_s), '--seed', str(args.seed),
            '--traffic-load', '2', '--ue-services', 'drx-mec', '--xapp-pool', 'chongyu',
            '--record-compression', 'gzip', '--direct-conflict-mitigation', 'priority',
            '--throughput-xapp-policy', 'throughput', '--ap-services-profile', str(args.ap_services_profile),
            '--ap-xapp-window-ms', '100', '--channel-cache-root',
            str(args.output/'channel_cache'/('seed_'+str(args.seed)))]
        for kind in ('energy', 'latency', 'budget', 'throughput', 'urllc', 'ap'):
            active = (kind == 'ap' and arm in ('x02', 'both')) or (kind == 'throughput' and arm in ('x08', 'both'))
            command += ['--'+kind+'-xapp-mode', 'frozen' if active else 'off']
        for app, kind in (('x02', 'ap'), ('x08', 'throughput')):
            if arm in (app, 'both'): command += ['--'+kind+'-xapp-model', models[app]['path']]
        command += ['--output', str(folder)]
        cases.append(dict(name=arm, output=str(folder), command=command, duration_s=args.duration_s,
            slots=round(args.duration_s*2000), seed=args.seed))
    inputs = (args.model_manifest, args.ap_services_profile, root/'configs/rt_sys/ue_services.json',
              root/'configs/rt_sys/agx_munich_6ue_depth2_fast.json')
    return dict(schema='oran.released-m2-plan.v1', project_root=str(root), output=str(args.output),
        source_sha256=source_fingerprint(root), models=models, duration_s=args.duration_s,
        slots=round(args.duration_s*2000), seed=args.seed, slot_duration_ms=.5,
        ap_profile=str(args.ap_services_profile), traffic_load=2, ue_services='native_drx_mec',
        expected_ue_inputs_sha256=fingerprint(expected_configs['ue']),
        primary_contracts=deepcopy(PRIMARY_CONTRACTS), standalone_harm_threshold_fraction=.02,
        declared_rcp={app:sorted(fields) for app, fields in RCP.items()}, cases=cases,
        input_files={str(path):sha256(path) for path in inputs},
        launcher_sha256={p:sha256(root/p) for p in ('scripts/orin/run-six-ue-cuda-nr.sh',
            'scripts/orin/activate-gpu.sh', 'scripts/orin/activate.sh')},
        frozen_models=True, automatic_training=False,
        scope='Matched finite-run evaluation of released AP-energy and throughput controllers. Both disjoint RCPs execute concurrently. QoS costs are observed without blocking deployment; no replicated significance claim.')


def concurrent_control_evidence(folder, cells):
    """Find actual after-state snapshots retaining both acknowledged controls."""
    scheduler = {}; witnesses = {}; count = 0; empty_acks = Counter()
    for receipt in jsonl(folder, 'control_receipts'):
        if receipt.get('status') != 'applied': continue
        provenance = receipt.get('provenance', [])
        if isinstance(provenance, dict): provenance = [provenance]
        owners = {p.get('implementation_id') for p in provenance}
        fields = {f['field']: f['value'] for f in receipt.get('fields', [])}
        after = receipt.get('after', {})
        if X08 in owners and 'scheduler_mode' in fields:
            for cid, actual in after.items():
                weights = actual.get('ue_scheduler_weights')
                # The runtime removes a per-UE weight field for an empty
                # serving cell. Its actual mode-only ACK is still meaningful;
                # no UE weights exist there to request or actuate. Occupied
                # snapshots must retain the complete requested UID bundle.
                empty = weights == {} and actual.get('ue_service_policies') == {}
                full_bundle = ('ue_scheduler_weights' in fields and isinstance(weights, dict) and
                    all(fields['ue_scheduler_weights'].get(u) == v for u, v in weights.items()))
                empty_mode_only = empty and fields.get('ue_scheduler_weights', {}) == {}
                if (actual.get('scheduler_mode') == fields['scheduler_mode'] and
                        (full_bundle or empty_mode_only)):
                    scheduler[cid] = dict(scheduler_mode=actual['scheduler_mode'], ue_scheduler_weights=weights)
                    if empty_mode_only and 'ue_scheduler_weights' not in fields: empty_acks[cid] += 1
        for cid, actual in after.items():
            prior = scheduler.get(cid)
            if (prior is not None and actual.get('ap_policy_owner') == APP and
                    actual.get('cell_sleep_mode') == actual.get('ap_policy') and
                    all(actual.get(key) == value for key, value in prior.items())):
                count += 1; witnesses.setdefault(cid, receipt.get('apply_slot', receipt.get('slot_id')))
    return dict(verified=set(witnesses) == set(cells), cells=sorted(witnesses),
        first_actual_joint_snapshot_slots=witnesses, actual_joint_snapshot_count=count,
        empty_cell_mode_only_ack_counts=dict(empty_acks),
        scope='Actual applied-ACK after states retain X02-owned AP policy and the acknowledged X08 scheduler mode plus all applicable UE weights in the same cell. Mode-only ACKs are accepted exclusively for snapshots with explicit empty UE-policy and weight maps. Controller prediction replays are audited separately.')


def controller_descriptors(summary, expected):
    # Native all-OFF runs omit xapp_library entirely. Missing descriptors are
    # valid only with no expected controllers; ON arms still require exact RCPs.
    descriptors = {d['implementation_id']:set(d['writable_actions']) for d in
        summary.get('xapp_library', {}).get('instances', []) if d['implementation_id'] in expected}
    if descriptors != {app:RCP[app] for app in expected}:
        raise ValueError('Actual controller writable RCPs differ from the declared disjoint controls')
    return descriptors


def audit_case(case, plan):
    folder = Path(case['output']); arm = case['name']
    cfg = load(folder/'config_snapshot.json'); summary = load(folder/'final_summary.json')
    data = case_data(folder); manifest = cfg['manifest']; audit = data['audit']
    if audit['source_sha256'] != plan['source_sha256']:
        raise ValueError('Actual source differs from pinned M2 source')
    if (manifest['total_slots'] != plan['slots'] or summary['total_slots'] != plan['slots'] or
            manifest['rng_seed'] != plan['seed'] or manifest['slot_duration_ms'] != .5 or
            {u['ue_id'] for u in manifest['ues']} != POPULATION or len(manifest['cells']) != 3 or
            any(u['trajectory']['positions_m'][0] == u['trajectory']['positions_m'][-1] for u in manifest['ues'])):
        raise ValueError('Actual seed/horizon/six moving UEs/three cells differ from the declared case')
    if summary.get('rt_sys', {}).get('model') != 'fast' or summary['rt_sys'].get('fast_profile') != 'system':
        raise ValueError('Expected actual fast SYSTEM execution')
    rt = cfg['configs']['phy']['rt_sys']; expected_service = load(Path(plan['project_root'])/'configs/rt_sys/ue_services.json')
    expected_service['enabled'] = True; expected_service['task_cohort_ms'] = 100
    if rt['ue_services'] != expected_service or rt['ap_services'] != load(plan['ap_profile'])['ap_services']:
        raise ValueError('Recorded service/AP physics differ from the declared profiles')
    if fingerprint(cfg['configs']['ue']) != plan['expected_ue_inputs_sha256']:
        raise ValueError('Actual six-UE traffic and demand must be the declared 2x profile')
    apps = cfg['configs']['ric'].get('xapps', []); expected = set()
    if arm in ('x02', 'both'): expected.add(APP)
    if arm in ('x08', 'both'): expected.add(X08)
    if {a.get('implementation_id') for a in apps} != expected or len(apps) != len(expected):
        raise ValueError('Unexpected active xApp')
    if cfg['configs']['ric'].get('direct_conflict_mitigation', 'priority') != 'priority':
        raise ValueError('Disjoint M2 controls must not use random action discard')
    for app in apps:
        key = next(k for k, value in APP_IDS.items() if value == app['implementation_id'])
        if (app['parameters']['mode'] != 'frozen' or
                str(Path(app['parameters']['model_path']).resolve()) != plan['models'][key]['path']):
            raise ValueError('Unexpected controller mode or frozen artifact')
    evidence = {}
    if APP in expected: evidence['x02'] = ap_control_evidence(folder, cfg, summary, plan['models']['x02']['stamp'])
    if X08 in expected:
        evidence['x08'] = scheduler_evidence(folder, summary, X08, plan['models']['x08']['stamp'],
            combined=False, model_path=plan['models']['x08']['path'])
    if not all(e['valid'] for e in evidence.values()):
        raise ValueError('Required frozen inference and actual deployed control evidence failed')
    if any(row.get('reason') == 'direct_conflict_random_selection' for row in jsonl(folder, 'rule_decisions')):
        raise ValueError('Random direct-conflict discard cannot establish concurrent M2')
    descriptors = controller_descriptors(summary, expected)
    concurrent = concurrent_control_evidence(folder, summary['ap_services']['cells']) if arm == 'both' else None
    if arm == 'both' and not concurrent['verified']:
        raise ValueError('No actual simultaneous AP/scheduler snapshots across all cells')
    history = TaskEventCollector(plan['duration_s'], cohort_ms=100, start_s=.1)
    policies = {uid:{} for uid in ('ue_0', 'ue_4')}; slots = 0
    for row in jsonl(folder, 'rt_sys_trace'):
        slots += 1
        for event in row.get('ue_service_events', []): history.observe(event)
        for uid in policies:
            policies[uid].update(mature_cohorts(row.get('ue_services', {}).get(uid, {}), row['slot_id']*.0005))
    mec = task_metrics(history, policies, summary, rt['ue_services']['mec']['tasks_by_ue']['ue_4'])
    energy = summary['ue_services']['per_ue']
    if set(energy) != POPULATION: raise ValueError('Six UE energy ledgers are required')
    urllc = dict(data['traffic_classes']['urllc'])
    for key in ('pending_packets', 'overflow_packets'):
        urllc[key] = sum(data['users'][u][key] for u in ('ue_2', 'ue_6'))
    metrics = dict(network_ap_energy_j=data['ap_energy']['network_ap_energy_j'],
        network_goodput_mbps=data['network']['goodput_mbps'],
        network_ue_energy_j=sum(e['ue_energy_total_j'] for e in energy.values()),
        ue4_energy_j=energy['ue_4']['ue_energy_total_j'], urllc=urllc,
        network_packet_metrics=data['network'], per_ue_packet_metrics=data['users'],
        traffic_classes=data['traffic_classes'], mec_tasks=mec['tasks'], mec_task_details=mec)
    physical = dict(manifest=manifest, configs={k:cfg['configs'][k] for k in ('ue', 'phy', 'sim', 'cell', 'beamforming')})
    return dict(arm=arm, case=arm, case_relative_path='cases/'+arm, audit_relative_path='audits/'+arm+'.json',
        source_sha256=plan['source_sha256'], model_sha256=MODEL_HASHES, slots=slots,
        duration_s=plan['duration_s'], seed=plan['seed'], physical_inputs_sha256=fingerprint(physical),
        exogenous_slot_inputs_sha256=data['arrival_audit']['exogenous_slot_inputs_sha256'],
        task_birth_inputs_sha256=mec['task_birth_inputs_sha256'],
        urllc_mature_inputs_sha256={u:data['users'][u]['mature_offered_inputs_sha256'] for u in ('ue_2', 'ue_6')},
        controller_evidence=evidence, controller_evidence_verified=True,
        declared_rcp={app:sorted(fields) for app, fields in descriptors.items()},
        concurrent_control_evidence=concurrent, metrics=metrics, ap_energy_ledger_audit=data['ap_energy'],
        automatic_html_audits_passed=True,
        files_sha256={str(p.relative_to(folder)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()})


def compare(arms, plan, completed=False):
    if not arms: return dict(schema='oran.released-m2-comparison.v1', completed=False, arms={})
    if 'base' not in arms or set(arms)-set(ARMS): raise ValueError('Matched Base must be present')
    if completed and set(arms) != set(ARMS): raise ValueError('Four completed arms required')
    base = arms['base']
    if base['source_sha256'] != plan['source_sha256'] or base['model_sha256'] != MODEL_HASHES:
        raise ValueError('Comparison source/models differ from the pinned release plan')
    for arm, data in arms.items():
        for key in ('source_sha256', 'model_sha256', 'physical_inputs_sha256',
                    'exogenous_slot_inputs_sha256', 'task_birth_inputs_sha256', 'urllc_mature_inputs_sha256'):
            if data[key] != base[key]: raise ValueError('Unmatched M2 inputs: '+arm+'/'+key)
        if not data['controller_evidence_verified']: raise ValueError('Unverified deployed model: '+arm)
    metrics = {arm:data['metrics'] for arm, data in arms.items()}
    for values in metrics.values():
        for key in ('network_ap_energy_j', 'network_goodput_mbps'):
            if type(values[key]) not in (int, float) or not math.isfinite(values[key]) or values[key] <= 0:
                raise ValueError('Positive finite AP energy and goodput are required')
    b = metrics['base']; primary = deepcopy(PRIMARY_CONTRACTS)
    for contract in primary.values(): contract['threshold'] = contract['base_fraction']*b[contract['metric']]
    def meets(app, arm):
        c = primary[app]; value = metrics[arm][c['metric']]; threshold = c['threshold']
        return value <= threshold+1e-10 if app == 'x02' else value >= threshold-1e-10
    individual = {app:meets(app, app) for app in ('x02', 'x08') if app in metrics}
    joint = {app:meets(app, 'both') for app in ('x02', 'x08')} if 'both' in metrics else {}
    disjoint = bool('both' in arms and arms['both']['concurrent_control_evidence']['verified'] and
        arms['both']['declared_rcp'] == {app:sorted(fields) for app, fields in RCP.items()} and
        all(arms['both']['controller_evidence'].get(k, {}).get('valid') for k in ('x02', 'x08')))
    harm = {}
    if set(ARMS) <= set(metrics):
        harm = dict(ap_energy_increase_fraction=metrics['both']['network_ap_energy_j']/metrics['x02']['network_ap_energy_j']-1,
            goodput_loss_fraction=1-metrics['both']['network_goodput_mbps']/metrics['x08']['network_goodput_mbps'])
    both_effective = len(individual) == 2 and all(individual.values())
    conflict = bool(completed and disjoint and both_effective and not all(joint.values()))
    diagnostic_harm = any(v > .02+1e-12 for v in harm.values())
    classification = ('incomplete' if not completed else 'invalid_concurrent_control_evidence' if not disjoint else
        'controllers_not_individually_effective' if not both_effective else 'operational_m2_contract_violation' if conflict else
        'standalone_target_harm_but_primary_contracts_met' if diagnostic_harm else 'no_detected_m2_conflict')
    return dict(schema='oran.released-m2-comparison.v1', completed=completed, duration_s=plan['duration_s'],
        seed=plan['seed'], source_sha256=plan['source_sha256'], model_sha256=MODEL_HASHES,
        matched_inputs_verified=True, controller_evidence_verified=True, disjoint_controls_verified=disjoint,
        individually_effective=individual, joint_contracts=joint, primary_contracts=primary,
        contracts=dict(ap_energy_ratio_max=.95, goodput_ratio_min=1.05),
        operational_m2_conflict_observed=conflict, standalone_target_harm=harm,
        standalone_target_harm_observed=diagnostic_harm, standalone_harm_threshold_fraction=.02,
        classification=classification, arms=arms, scope=plan['scope'],
        interpretation='Operational M2 requires both standalone 5% contracts to hold and at least one to fail under actual concurrent disjoint controls. More than 2% deterioration versus a standalone arm is a separate diagnostic. QoS costs and non-additivity alone do not establish conflict.')


def main(argv=None):
    args = parse_args(argv); plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, plan=plan), indent=2)); return 0
    from experiments.released_case_queue import run_queue
    return run_queue(args, plan, audit_case, compare)


if __name__ == '__main__':
    raise SystemExit(main())

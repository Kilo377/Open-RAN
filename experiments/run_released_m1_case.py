"""Frozen-release M1: matched native Base / X-01 / X-13 / random arbitration.

This is an evaluation, with no fitting or action tuning. Unequal overlapping
valid requests prove a direct conflict; post-arbitration target violations are
a separate measured outcome. Other QoS costs never invalidate the experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_xapp_long_validation import load, sha256, source_fingerprint, _records
from experiments.run_mec_budget_conflict_study import (
    ENERGY, BUDGET, arbitration_evidence, controller_evidence, model_stamp,
    training_readiness,
)
from experiments.run_mec_budget_pressure_study import task_metrics
from experiments.run_ue_energy_study import mature_cohorts
from experiments.task_event_metrics import TaskEventCollector
from ric.near_rt.pools.chongyu.online_rf import OnlineEnergyRF
from ric.near_rt.pools.chongyu.online_qgbdt import OnlineQGBDT
from ric.near_rt.pools.chongyu.ue_energy_rf import FEATURES as ENERGY_FEATURES
from ric.near_rt.pools.chongyu.mec_offload_budget_qgbdt import FEATURES as BUDGET_FEATURES

ARMS = ('base', 'x01', 'x13', 'both_random')
APPS = {'x01': ENERGY, 'x13': BUDGET}
ENABLED = {'base': (), 'x01': ('x01',), 'x13': ('x13',), 'both_random': ('x01', 'x13')}
POPULATION = {'ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8'}
TARGETS = ('ue_0', 'ue_4')
CONTRACTS = dict(energy_ratio_max=.95, budget_gflops_max=.75,
    energy_population=list(TARGETS), budget_population=['ue_4'],
    qos_costs_are_validity_gates=False,
    energy_scope='Whole-run modeled UE0+UE4 radio/CPU energy relative to matched Base',
    budget_scope='UE4 expected offered offloaded compute, averaged over mature creation cohorts',
    cohort_scope='100ms cohorts starting at >=100ms with complete deadline horizon; all terminal events through run end')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def parse_args(argv=None, root=ROOT):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('outputs/trained_xapps/20261109'))
    p.add_argument('--release-manifest', type=Path,
        default=Path('artifacts/xapps/chongyu/20261109/release.json'))
    p.add_argument('--service-config', type=Path, default=Path('configs/rt_sys/ue_services.json'))
    p.add_argument('--duration-s', type=float, default=3.)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--mitigation-seed', type=int, default=1701)
    p.add_argument('--min-free-gb', type=float, default=.25)
    p.add_argument('--estimated-case-gb', type=float, default=.06)
    p.add_argument('--heartbeat-s', type=float, default=10.)
    p.add_argument('--termination-grace-s', type=float, default=15.)
    p.add_argument('--max-parallel', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv); root = Path(root).resolve()
    for key in ('output', 'model_dir', 'release_manifest', 'service_config'):
        path = getattr(args, key)
        setattr(args, key, (path if path.is_absolute() else root/path).resolve())
    if (not args.output.is_relative_to(root/'outputs') or args.output == root/'outputs' or
            not args.model_dir.is_relative_to(root/'outputs') or
            args.model_dir.is_relative_to(args.output) or args.output.is_relative_to(args.model_dir)):
        raise ValueError('Use separate dedicated evaluation and read-only model directories under outputs/')
    if not args.release_manifest.is_relative_to(root/'artifacts') or not args.service_config.is_relative_to(root/'configs'):
        raise ValueError('Use the project release manifest and native service configuration')
    if (not math.isfinite(args.duration_s) or args.duration_s < .3 or
            not math.isclose(args.duration_s/.1, round(args.duration_s/.1), abs_tol=1e-8)):
        raise ValueError('Duration must cover at least 0.3s in whole 100ms cohorts')
    if not 0 <= args.seed <= 2**32-1 or args.seed in (55, 137, 138):
        raise ValueError('Use a fresh evaluation seed distinct from historical calibration/training')
    if not 0 <= args.mitigation_seed <= 2**32-1 or args.mitigation_seed == args.seed:
        raise ValueError('Use an independent supported mitigation seed')
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < .25:
        raise ValueError('Retain at least 0.25GiB disk reserve')
    if not math.isfinite(args.estimated_case_gb) or args.estimated_case_gb <= 0:
        raise ValueError('A positive estimated recording size is required')
    for key in ('heartbeat_s', 'termination_grace_s'):
        if not math.isfinite(getattr(args, key)) or not 0 < getattr(args, key) <= 60:
            raise ValueError('Heartbeat and termination grace must be in (0,60]s')
    return args


def release_models(args):
    """The published manifest is authoritative; never substitute an older RF."""
    release = load(args.release_manifest)
    if (release.get('schema') != 'oran.chongyu-trained-model-release.v1' or
            release.get('release') != '20261109' or not release.get('complete') or
            release.get('pool') != 'chongyu'):
        raise ValueError('The complete 20261109 chongyu model release is required')
    models = {}
    for app, ident in APPS.items():
        entry = release['models'][app]
        path = (args.model_dir/entry['path']).resolve()
        published = (args.release_manifest.parent/entry['path']).resolve()
        if (not path.is_relative_to(args.model_dir) or
                not published.is_relative_to(args.release_manifest.parent) or
                entry.get('implementation_id') != ident or entry.get('status') != 'trained_and_verified'):
            raise ValueError('Unexpected released model path or implementation: '+app)
        if sha256(path) != entry['sha256'] or sha256(published) != entry['sha256']:
            raise ValueError('Released model checksum mismatch: '+app)
        stamp = model_stamp(path)
        if (stamp['samples'] != entry['measured_rows'] or stamp['trained_samples'] != entry['fitted_rows'] or
                stamp['format'] != entry['format'] or stamp['fit_count'] <= 0):
            raise ValueError('Frozen learner history differs from its release: '+app)
        learner = (OnlineEnergyRF.load(path, ENERGY_FEATURES) if app == 'x01' else
            OnlineQGBDT.load(path, BUDGET_FEATURES))
        if not learner.trained or (app == 'x13' and not training_readiness(path)['ready']):
            raise ValueError('Actual fitted model and measured action coverage are required: '+app)
        models[app] = dict(path=str(path), sha256=entry['sha256'], stamp=stamp,
            implementation_id=ident, algorithm=entry['algorithm'])
    return models


def command_for(args, arm, models=None):
    if arm not in ARMS: raise ValueError('Unknown M1 arm')
    models = models or release_models(args)
    command = ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast',
        '--duration-s', str(args.duration_s), '--seed', str(args.seed), '--traffic-load', '1',
        '--ue-services', 'drx-mec', '--ue-services-config', str(args.service_config),
        '--xapp-pool', 'chongyu', '--record-compression', 'gzip',
        '--direct-conflict-mitigation', 'uniform_random' if arm == 'both_random' else 'priority',
        '--mitigation-seed', str(args.mitigation_seed), '--mitigation-window-ms', '100',
        '--budget-xapp-gflops', '.75', '--budget-xapp-window-ms', '100',
        '--channel-cache-root', str(args.output/'channel_cache'/('seed_'+str(args.seed)))]
    for kind, app in (('energy', 'x01'), ('budget', 'x13'), ('latency', None),
                      ('throughput', None), ('urllc', None), ('ap', None)):
        command += ['--'+kind+'-xapp-mode', 'frozen' if app in ENABLED[arm] else 'off']
        if app in ENABLED[arm]: command += ['--'+kind+'-xapp-model', models[app]['path']]
    return command+['--output', str(args.output/'cases'/arm)]


def build_plan(args, root=ROOT):
    root = Path(root).resolve(); native = load(root/'configs/rt_sys/ue_services.json')
    profile = native['mec']['tasks_by_ue']['ue_4']
    if (load(args.service_config) != native or profile['offload_fraction'] != .5 or
            {k:profile[k] for k in ('interval_ms', 'compute_gflop', 'deadline_ms')} !=
            dict(interval_ms=50, compute_gflop=.05, deadline_ms=100)):
        raise ValueError('M1 requires the unchanged native 1x profile with UE4 initial offload=0.5')
    models = release_models(args)
    launchers = ('scripts/orin/run-six-ue-cuda-nr.sh', 'scripts/orin/activate-gpu.sh', 'scripts/orin/activate.sh')
    return dict(schema='oran.released-m1-plan.v1', project_root=str(root), output=str(args.output),
        source_sha256=source_fingerprint(root), release='20261109', models=models,
        duration_s=args.duration_s, slots=round(args.duration_s*2000), slot_duration_ms=.5,
        seed=args.seed, mitigation_seed=args.mitigation_seed, mitigation_window_ms=100,
        scenario='native_1x_drx_mec', initial_ue4_offload_fraction=.5, traffic_load=1,
        service_config=dict(path=str(args.service_config), sha256=sha256(args.service_config)),
        input_files={str(args.service_config):sha256(args.service_config),
            str(args.release_manifest):sha256(args.release_manifest)},
        launcher_sha256={p:sha256(root/p) for p in launchers},
        contracts=CONTRACTS, frozen_models=True, automatic_training=False,
        max_parallel=args.max_parallel, base_first_shared_channel_cache=True,
        cases=[dict(name=a, output=str(args.output/'cases'/a),
            command=command_for(args, a, models)) for a in ARMS],
        scope='Preregistered native matched-input direct-conflict evaluation. UE0+UE4 energy is primary; UE4 energy is diagnostic. Expected quota is not physical MEC server capacity. Random arbitration may satisfy both goals. QoS costs are retained, not validity gates.')


def energy_controller_evidence(folder, summary, model, require_execution=True):
    """Recognize genuine RF baseline decisions as well as RF-selected changes.

    The runtime only emits these reasons after inference. Where an executed
    mature sample retains its pre-outcome prediction, replay that prediction
    independently using the exact immutable artifact. No labels are fitted.
    """
    proof = controller_evidence(folder, summary, 'energy', model['stamp'], 'frozen', require_execution)
    reasons = Counter(); replayed = 0; learner = OnlineEnergyRF.load(model['path'], ENERGY_FEATURES)
    inferred_reasons = {'rf_minimum_predicted_feasible_power', 'predicted_qos_guard_baseline',
        'no_predicted_energy_gain_baseline'}
    for event in _records(folder, 'energy_rf_events'):
        if event.get('mode') != 'frozen': raise ValueError('Energy event came from a non-frozen controller')
        if event.get('kind') == 'decision':
            reason = event.get('reason'); reasons[reason] += 1
            if (reason not in inferred_reasons or event.get('trained_samples') != model['stamp']['trained_samples']):
                raise ValueError('Energy decision lacks actual trained-model inference')
        prediction = event.get('prediction_before_outcome')
        if event.get('kind') == 'measured_sample' and prediction is not None:
            actual = learner.predict([event['features']])[0]
            if (len(prediction) != len(actual) or any(type(v) not in (int, float) or
                    not math.isfinite(v) or not math.isclose(v, a, rel_tol=1e-10, abs_tol=1e-10)
                    for v, a in zip(prediction, actual))):
                raise ValueError('Recorded pre-outcome RF prediction differs from frozen-model replay')
            replayed += 1
    proof['model_dependent_decision_reasons'] = dict(reasons)
    proof['independent_pre_outcome_prediction_replays'] = replayed
    proof['genuine_model_inference'] = bool(sum(reasons.values()))
    descriptor = next((a for a in summary.get('xapp_library', {}).get('instances', [])
        if a.get('implementation_id') == ENERGY), {})
    proof['valid_control_evidence'] = bool(proof['genuine_model_inference'] and
        descriptor.get('active_cell_ids') and
        proof['frozen_sample_fit_counts_unchanged'] and
        {'ue_offload_fraction', 'ue_drx_configs'} <= set(proof['declared_rcps']) and
        (proof['executed_and_confirmed'] or not require_execution))
    proof['inference_scope'] = 'Pinned source model-dependent decision branch; available pre-outcome predictions independently replayed. Overridden requests remain distinct from executed policies.'
    return proof


def audit_case(case, plan):
    folder = Path(case['output']); arm = case['name']; cfg = load(folder/'config_snapshot.json')
    summary = load(folder/'final_summary.json'); status = load(folder/'run_status.json'); manifest = cfg['manifest']
    if (not status.get('completed') or not status.get('replay', {}).get('audits_passed') or
            not (folder/'network-replay.html').is_file()):
        raise ValueError('Completed recording and audited automatic HTML are required')
    if cfg.get('source_sha256') != plan['source_sha256'] or status.get('source_sha256') != plan['source_sha256']:
        raise ValueError('Executed source differs from the pinned M1 source')
    if (manifest['total_slots'] != plan['slots'] or summary['total_slots'] != plan['slots'] or
            manifest['slot_duration_ms'] != .5 or manifest['rng_seed'] != plan['seed'] or
            {u['ue_id'] for u in manifest['ues']} != POPULATION or len(manifest['cells']) != 3):
        raise ValueError('Recorded network/horizon differs from the M1 plan')
    service = cfg['configs']['phy']['rt_sys']['ue_services']
    expected = load(plan['service_config']['path']); expected['task_cohort_ms'] = 100
    if service != expected: raise ValueError('Executed native service inputs differ')
    ric = cfg['configs']['ric']; apps = ric.get('xapps', [])
    if len(apps) != len(ENABLED[arm]) or {a.get('implementation_id') for a in apps} != {APPS[a] for a in ENABLED[arm]}:
        raise ValueError('Unexpected active controller set')
    mitigation = 'uniform_random' if arm == 'both_random' else 'priority'
    if (ric.get('direct_conflict_mitigation', 'priority') != mitigation or
            ric.get('mitigation_seed') != plan['mitigation_seed'] or ric.get('mitigation_window_ms') != 100):
        raise ValueError('Executed random mitigation differs from its preregistered configuration')
    for app in ENABLED[arm]:
        parameters = next(a['parameters'] for a in apps if a['implementation_id'] == APPS[app])
        model = plan['models'][app]
        if (parameters.get('mode') != 'frozen' or
                Path(parameters.get('model_path', '')).resolve() != Path(model['path']) or
                sha256(model['path']) != model['sha256']):
            raise ValueError('Controller did not load the exact frozen released artifact')
        if app == 'x01' and parameters.get('deploy_drx', False):
            raise ValueError('M1 keeps native DRX locked; only offload actions may differ')
        if app == 'x13' and (parameters.get('budget_gflops') != .75 or parameters.get('window_ms') != 100 or
                parameters.get('task_profile') != {k:service['mec']['tasks_by_ue']['ue_4'][k]
                    for k in ('interval_ms', 'compute_gflop', 'deadline_ms')}):
            raise ValueError('Budget controller uses an undeclared task quota')
    arbitration = None
    if arm == 'both_random':
        arbitration = arbitration_evidence(folder)
        for row in _records(folder, 'rule_decisions'):
            if (row.get('reason') == 'direct_conflict_random_selection' and row.get('ue_id') == 'ue_4' and
                    row.get('mitigation_seed') != plan['mitigation_seed']):
                raise ValueError('Arbitration evidence uses an unexpected seed')
        if arbitration['errors'] or (arbitration['unique_random_epochs'] and not arbitration['valid']):
            raise ValueError('Random requests/draws/executed UID policies failed independent verification')
        if not arbitration['all_selection_epochs'] or not arbitration['executed_matching_receipts']:
            raise ValueError('Combined arm lacks actual request-to-executed UE4 arbitration evidence')
        arbitration['structural_m1_observed'] = bool(arbitration['valid'])
        arbitration['evidence_error_free'] = True
    controllers = {}
    for app in ENABLED[arm]:
        # A contender can validly infer yet lose every random epoch. Do not
        # invent execution for it; require receipts whenever it actually won.
        required = (arm != 'both_random' or bool(arbitration['runtime_winner_receipts'].get(APPS[app])))
        controllers[app] = (energy_controller_evidence(folder, summary, plan['models'][app], required)
            if app == 'x01' else controller_evidence(folder, summary, 'budget',
                plan['models'][app]['stamp'], 'frozen', required))
        if (not controllers[app]['valid_control_evidence'] or not controllers[app]['genuine_model_inference'] or
                summary['ue_energy_xapp' if app == 'x01' else 'budget_xapp'].get('mode') != 'frozen'):
            raise ValueError('Frozen model inference/control evidence is incomplete: '+app)
    history = TaskEventCollector(plan['duration_s'], cohort_ms=100, start_s=.1, target_uids=TARGETS)
    policies = {uid:{} for uid in TARGETS}; external = hashlib.sha256(); count = 0
    for row in _records(folder, 'rt_sys_trace'):
        count += 1
        if row.get('slot_id') != count or {u['ue_id'] for u in row['ues']} != POPULATION:
            raise ValueError('Every ordered six-UE trace slot must be recorded')
        arrivals = row.get('traffic_arrivals', {})
        if set(arrivals) != POPULATION: raise ValueError('Recorded exogenous inputs are incomplete')
        values = [arrivals[u].get('exogenous_bytes') for u in sorted(POPULATION)]
        if any(type(v) is not int or v < 0 for v in values): raise ValueError('Invalid exogenous byte inputs')
        external.update(fingerprint([count, values]).encode())
        for event in row.get('ue_service_events', []): history.observe(event)
        for uid in TARGETS:
            policies[uid].update(mature_cohorts(row.get('ue_services', {}).get(uid, {}), count*.0005))
    if count != plan['slots']: raise ValueError('Recorded trace does not cover the requested duration')
    metrics = task_metrics(history, policies, summary, service['mec']['tasks_by_ue']['ue_4'])
    packet = summary['packet_metrics']; energy = summary['ue_services']['per_ue']
    if set(packet) != POPULATION or set(energy) != POPULATION:
        raise ValueError('All six final packet/energy ledgers are required')
    for uid, value in packet.items():
        if value['initial_bytes']+value['offered_bytes'] != value['delivered_bytes']+value['pending_bytes']+value['overflow_bytes']:
            raise ValueError('Packet bytes do not conserve actual arrivals/delivery')
        joules = energy[uid]['ue_energy_total_j']
        if type(joules) not in (int, float) or not math.isfinite(joules) or joules <= 0:
            raise ValueError('Invalid modeled UE energy ledger')
    metrics.update(network_goodput_mbps=sum(p['delivered_bytes'] for p in packet.values())*8/plan['duration_s']/1e6,
        ue4_goodput_mbps=packet['ue_4']['delivered_bytes']*8/plan['duration_s']/1e6,
        network_ue_energy_j=sum(p['ue_energy_total_j'] for p in energy.values()),
        target_ue_energy_j=sum(energy[u]['ue_energy_total_j'] for u in TARGETS),
        ue4_energy_j=energy['ue_4']['ue_energy_total_j'],
        per_ue_packet_qos={uid:dict(value, goodput_mbps=value['delivered_bytes']*8/plan['duration_s']/1e6,
            ue_energy_j=energy[uid]['ue_energy_total_j']) for uid, value in packet.items()},
        task_event_cumulative_ledger_reconstruction_verified=True,
        task_scoring_start_s=.1, energy_scope='Whole run; modeled radio plus CPU energy')
    physical = dict(manifest=manifest, configs={k:cfg['configs'][k]
        for k in ('ue', 'phy', 'sim', 'cell', 'beamforming') if k in cfg['configs']})
    return dict(arm=arm, case=arm, case_relative_path='cases/'+arm, audit_relative_path='audits/'+arm+'.json',
        source_sha256=plan['source_sha256'], model_sha256={a:m['sha256'] for a,m in plan['models'].items()},
        duration_s=plan['duration_s'], seed=plan['seed'], slots=count,
        physical_inputs_sha256=fingerprint(physical), exogenous_slot_inputs_sha256=external.hexdigest(),
        task_birth_inputs_sha256=metrics.pop('task_birth_inputs_sha256'),
        controller_evidence=controllers, arbitration_evidence=arbitration, metrics=metrics,
        automatic_html_audits_passed=True,
        files_sha256={str(p.relative_to(folder)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()})


def compare(arms, plan, completed=False):
    if not arms: return dict(completed=False, arms={})
    if 'base' not in arms or not set(arms) <= set(ARMS): raise ValueError('Matched Base and declared M1 arms required')
    if completed and set(arms) != set(ARMS): raise ValueError('All four completed arms are required')
    base = arms['base']; model_hashes = {a:m['sha256'] for a,m in plan['models'].items()}
    for arm, case in arms.items():
        for key in ('source_sha256', 'model_sha256', 'physical_inputs_sha256',
                    'exogenous_slot_inputs_sha256', 'task_birth_inputs_sha256'):
            if case[key] != base[key]: raise ValueError('Unmatched M1 inputs: '+arm+'/'+key)
        if case['source_sha256'] != plan['source_sha256'] or case['model_sha256'] != model_hashes:
            raise ValueError('Audited source/model differs from the preregistered plan')
        if case['metrics']['mature_birth_counts'] != base['metrics']['mature_birth_counts']:
            raise ValueError('Mature task birth cohorts differ between policies')
        if set(case['controller_evidence']) != set(ENABLED[arm]) or any(
                not p['valid_control_evidence'] or not p['genuine_model_inference']
                for p in case['controller_evidence'].values()):
            raise ValueError('Actual frozen controller inference/control evidence is missing')
        if not case['automatic_html_audits_passed']: raise ValueError('Automatic HTML is not audited')
    base_energy = base['metrics']['target_ue_energy_j']
    if base_energy <= 0: raise ValueError('A positive matched Base energy is required')
    goals = {}
    for arm, case in arms.items():
        metrics = case['metrics']; ratio = metrics['target_ue_energy_j']/base_energy
        energy_met = ratio <= CONTRACTS['energy_ratio_max']+1e-12
        budget_met = metrics['expected_mec_gflops'] <= CONTRACTS['budget_gflops_max']+1e-12
        goals[arm] = dict(target_energy_ratio=ratio, target_energy_saving_pct=100*(1-ratio),
            energy_goal_met=energy_met, budget_goal_met=budget_met, goals_both_met=energy_met and budget_met,
            budget_all_cohorts_met=metrics['expected_budget_violating_cohorts'] == 0,
            ue4_energy_ratio_diagnostic=metrics['ue4_energy_j']/base['metrics']['ue4_energy_j'])
    arbitration = arms.get('both_random', {}).get('arbitration_evidence')
    if arbitration is not None and (arbitration.get('errors') or not arbitration.get('evidence_error_free')):
        raise ValueError('Random arbitration audit has evidence errors')
    observed = bool(arbitration and arbitration.get('structural_m1_observed'))
    violations = None if 'both_random' not in goals else dict(
        energy=not goals['both_random']['energy_goal_met'], budget=not goals['both_random']['budget_goal_met'])
    isolated = dict(x01=goals.get('x01', {}).get('energy_goal_met'),
        x13=goals.get('x13', {}).get('budget_goal_met'))
    return dict(schema='oran.released-m1-comparison.v1', completed=completed,
        duration_s=plan['duration_s'], seed=plan['seed'], source_sha256=plan['source_sha256'],
        release=plan['release'], model_sha256=model_hashes,
        model_stamp={a:m['stamp'] for a,m in plan['models'].items()}, scenario=plan['scenario'],
        mitigation_seed=plan['mitigation_seed'], mitigation_window_ms=100, contracts=CONTRACTS,
        matched_inputs_verified=True, controller_evidence_verified=True, arms=arms,
        goal_results=goals, isolated_target_goals=isolated,
        isolated_apps_meet_primary_goals=all(v is True for v in isolated.values()),
        structural_m1_observed=observed,
        classification='m1_direct_conflict_observed' if observed else 'no_observed_m1' if arbitration else 'pending_combined_arm',
        actual_rcp_intersection=['ue_offload_fraction'], overlap_ue='ue_4',
        arbitration_evidence=arbitration, post_mitigation_goal_violations=violations,
        scope=plan['scope'], interpretation='Different valid overlapping UE4 offload requests establish structural M1. Uniform random selection can meet both target goals; structural conflict and post-mitigation goal violations are separate. All packet/task QoS, failed/pending outcomes, energy and goodput remain reported tradeoffs. Single-seed paired evaluation does not establish general statistical superiority.')


def main(argv=None):
    args = parse_args(argv); plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return 0
    from experiments.released_case_queue import run_queue
    return run_queue(args, plan, audit_case, compare)


if __name__ == '__main__':
    raise SystemExit(main())

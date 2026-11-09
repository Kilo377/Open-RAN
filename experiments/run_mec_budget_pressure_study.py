"""Matched Base/rules/frozen-QGBDT evaluation of an explicit UE4 offload quota.

Only the scenario's initial UE4 offload probability changes from the native
configuration. This is an expected offered-workload quota, not server admission
or a claim of physical edge congestion. No model fitting occurs in this driver.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_xapp_long_validation import (
    atomic_json, load, sha256, source_fingerprint, queue_lock, _records,
    _alive, _archive_partial,
)
from experiments.run_mec_budget_conflict_study import (
    cohort_demand, controller_evidence, model_stamp, training_readiness,
)
from experiments.run_ue_energy_study import aggregate_cohorts, mature_cohorts
from experiments.run_latency_energy_study import aggregate_offloaded_cohorts
from experiments.task_event_metrics import TaskEventCollector
from ric.near_rt.pools.chongyu.mec_offload_budget_qgbdt import FEATURES
from ric.near_rt.pools.chongyu.online_qgbdt import OnlineQGBDT

ARMS = ('base', 'rules', 'qgbdt')
MODES = {'base': 'off', 'rules': 'rules', 'qgbdt': 'frozen'}
APP = 'mec_offload_budget_qgbdt'
POPULATION = {'ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8'}
TARGETS = ('ue_0', 'ue_4')
MODEL_SHA256 = '005bff7a80541cddf133aac13b684d021583e34357e9f4befe1c0db01728463f'


def fingerprint(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def parse_args(argv=None, root=ROOT):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--expected-model-sha256', default=MODEL_SHA256)
    p.add_argument('--service-config', type=Path,
        default=Path('configs/rt_sys/ue_services_mec_offload_first.json'))
    p.add_argument('--duration-s', type=float, default=10.)
    p.add_argument('--seed', type=int, default=211)
    p.add_argument('--min-free-gb', type=float, default=.25)
    p.add_argument('--estimated-case-gb', type=float, default=.22)
    p.add_argument('--heartbeat-s', type=float, default=10.)
    p.add_argument('--termination-grace-s', type=float, default=15.)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    root = Path(root).resolve()
    for key in ('output', 'model_path', 'service_config'):
        path = getattr(args, key)
        setattr(args, key, (path if path.is_absolute() else root/path).resolve())
    if any(not getattr(args,k).is_relative_to(root/'outputs') for k in ('output','model_path')):
        raise ValueError('Dedicated study and read-only model must stay under project outputs/')
    if args.output == root/'outputs' or args.model_path.is_relative_to(args.output):
        raise ValueError('Keep the input model outside the dedicated new study directory')
    if not args.service_config.is_relative_to(root/'configs'):
        raise ValueError('Use an explicit project configuration profile')
    if (not math.isfinite(args.duration_s) or args.duration_s < .3 or
            not math.isclose(args.duration_s/.1, round(args.duration_s/.1), abs_tol=1e-8)):
        raise ValueError('Duration must cover at least 0.3 s in whole 100 ms cohorts')
    if not 0 <= args.seed <= 2**32-1 or args.seed in (55,137,138):
        raise ValueError('Use a fresh supported seed distinct from historical calibration/training')
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < .25:
        raise ValueError('Retain at least 0.25 GiB of disk reserve')
    if not math.isfinite(args.estimated_case_gb) or args.estimated_case_gb <= 0:
        raise ValueError('Positive estimated recording space required')
    for key in ('heartbeat_s','termination_grace_s'):
        if not math.isfinite(getattr(args,key)) or not 0 < getattr(args,key) <= 60:
            raise ValueError('Heartbeat and termination grace must be in (0,60] s')
    if len(args.expected_model_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.expected_model_sha256):
        raise ValueError('An exact lowercase frozen-model SHA-256 is required')
    return args


def command_for(args, arm, root=ROOT):
    if arm not in ARMS:
        raise ValueError('Unknown quota pressure arm')
    command = ['bash','scripts/orin/run-six-ue-cuda-nr.sh','--model','fast',
        '--duration-s',str(args.duration_s),'--seed',str(args.seed),'--traffic-load','1',
        '--ue-services','drx-mec','--ue-services-config',str(args.service_config),
        '--xapp-pool','chongyu','--record-compression','gzip',
        '--direct-conflict-mitigation','priority','--budget-xapp-gflops','.75',
        '--budget-xapp-window-ms','100','--channel-cache-root',
        str(args.output/'channel_cache'/('seed_'+str(args.seed)))]
    for kind in ('energy','latency','budget','throughput','urllc','ap'):
        command += ['--'+kind+'-xapp-mode', MODES[arm] if kind=='budget' else 'off']
    if arm != 'base':
        command += ['--budget-xapp-model',str(args.model_path)]
    return command+['--output',str(args.output/'cases'/arm)]


def build_plan(args, root=ROOT):
    root = Path(root).resolve()
    native = load(root/'configs/rt_sys/ue_services.json')
    expected = deepcopy(native)
    expected['mec']['tasks_by_ue']['ue_4']['offload_fraction'] = 1.
    if load(args.service_config) != expected:
        raise ValueError('Pressure profile must differ from native only in UE4 initial offload=1')
    if sha256(args.model_path) != args.expected_model_sha256:
        raise ValueError('Frozen model checksum differs from the declared artifact')
    learner = OnlineQGBDT.load(args.model_path, FEATURES)
    ready = training_readiness(args.model_path)
    if not learner.trained or not ready['ready']:
        raise ValueError('Actual fitted QGBDT and coverage of both feasible actions are required')
    launchers = ['scripts/orin/run-six-ue-cuda-nr.sh','scripts/orin/activate-gpu.sh','scripts/orin/activate.sh']
    return dict(schema='oran.mec-budget-pressure-plan.v1', project_root=str(root), output=str(args.output),
        source_sha256=source_fingerprint(root), duration_s=args.duration_s, seed=args.seed,
        slots=round(args.duration_s*2000), slot_duration_ms=.5, scenario='mec_offload_first',
        budget_gflops=.75, initial_ue4_offload_fraction=1., task_scoring_start_s=.1,
        model=dict(path=str(args.model_path),sha256=sha256(args.model_path),stamp=model_stamp(args.model_path)),
        service_config=dict(path=str(args.service_config),sha256=sha256(args.service_config)),
        launcher_sha256={p:sha256(root/p) for p in launchers},
        cases=[dict(name=a,mode=MODES[a],output=str(args.output/'cases'/a),
            command=command_for(args,a,root)) for a in ARMS],
        serial=True, frozen_model=True, automatic_training=False,
        disk=dict(minimum_reserve_gib=args.min_free_gb,estimated_case_gib=args.estimated_case_gb),
        scope='Three matched policies in an explicit UE4 offload-first scenario; native configuration remains unchanged. Expected UE4 operator quota, not physical MEC capacity. A rules match does not demonstrate learned superiority.')


def verify_inputs(plan):
    root = Path(plan['project_root'])
    if source_fingerprint(root) != plan['source_sha256']:
        raise ValueError('Pinned runtime source/configuration changed')
    for item in (plan['model'],plan['service_config']):
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Pinned model or service profile changed')
    if any(sha256(root/p) != s for p,s in plan['launcher_sha256'].items()):
        raise ValueError('Pinned environment launcher changed')


def task_metrics(history, policies, summary, profile):
    """Use full task events for outcomes and public cohorts only for executed p."""
    exact = history.summary()
    rows = {}
    for uid in TARGETS:
        whole = exact['per_ue'][uid]['whole_run']
        final = summary['ue_services']['per_ue'][uid]['ue_task_metrics']
        for key,value in whole.items():
            if key in ('max_completion_latency_ms','offloaded','local','unassigned'):
                continue
            actual = final.get(key)
            if type(actual) not in (int,float) or not math.isclose(actual,value,abs_tol=1e-7,rel_tol=1e-10):
                raise ValueError('Full task events disagree with the final task ledger: '+uid+'/'+key)
        rows[uid] = []
        for c in exact['per_ue'][uid]['cohorts']:
            k = (round(c['start_s'],9),round(c['end_s'],9))
            recorded = policies[uid].get(k)
            if not recorded or recorded.get('generated') != c['generated']:
                raise ValueError('A mature executed-generation policy is missing or has different births')
            rows[uid].append(dict(c,mean_offload_fraction=recorded['mean_offload_fraction']))
    if not all(rows.values()):
        raise ValueError('Mature task populations are required for both MEC UEs')
    all_rows = [c for uid in TARGETS for c in rows[uid]]
    demand = cohort_demand(rows['ue_4'],profile,.75)
    count = len(rows['ue_4'])
    demand['cohort_count'] = count
    demand['expected_budget_compliant_fraction'] = 1-demand['expected_budget_violating_cohorts']/count
    seconds = sum(c['end_s']-c['start_s'] for c in rows['ue_4'])
    demand['mean_expected_excess_gflops'] = sum(max(0,c['expected_mec_gflops']-.75)*
        (c['end_s']-c['start_s']) for c in demand['cohorts'])/seconds
    births = sorted([uid,str(tid),t['created_s'],t['deadline_s'],t['cohort_index']]
                    for (uid,tid),t in history.tasks.items())
    def outcomes(values):
        return dict(values,late_completed=values['completed']-values['on_time_completed'])
    return dict(demand, tasks=outcomes(aggregate_cohorts(all_rows)),
        ue4_tasks=outcomes(aggregate_cohorts(rows['ue_4'])),
        offloaded_tasks=outcomes(aggregate_offloaded_cohorts(all_rows)),
        per_ue_tasks={uid:outcomes(aggregate_cohorts(rows[uid])) for uid in TARGETS},
        whole_run_tasks={uid:exact['per_ue'][uid]['whole_run'] for uid in TARGETS},
        task_birth_inputs_sha256=fingerprint(births),
        mature_birth_counts={uid:[[c['index'],c['generated']] for c in rows[uid]] for uid in TARGETS},
        task_scope='Mature creation cohorts beginning at >=100ms; every final event outcome reconstructed through run end. Completion means exclude pending/failed; all-task success includes their denominator.')


def audit_case(case, plan):
    folder = Path(case['output']); arm = case['name']
    cfg = load(folder/'config_snapshot.json'); summary = load(folder/'final_summary.json')
    status = load(folder/'run_status.json'); manifest = cfg['manifest']
    if (not status.get('completed') or not status.get('replay',{}).get('audits_passed') or
            not (folder/'network-replay.html').is_file()):
        raise ValueError('Completed full recording and audited automatic HTML are required')
    if cfg.get('source_sha256') != plan['source_sha256'] or status.get('source_sha256') != plan['source_sha256']:
        raise ValueError('Executed source differs from the pinned study')
    if (manifest['total_slots'] != plan['slots'] or summary['total_slots'] != plan['slots'] or
            manifest['slot_duration_ms'] != .5 or manifest['rng_seed'] != plan['seed'] or
            {u['ue_id'] for u in manifest['ues']} != POPULATION or len(manifest['cells']) != 3):
        raise ValueError('Recorded network/horizon differs from the study')
    service = cfg['configs']['phy']['rt_sys']['ue_services']
    expected = load(plan['service_config']['path']); expected['task_cohort_ms'] = 100
    if service != expected:
        raise ValueError('Executed UE service scenario differs from the explicit pressure profile')
    apps = cfg['configs']['ric'].get('xapps',[])
    if arm == 'base':
        if apps: raise ValueError('Base contains an active xApp')
        controller = dict(mode='off',valid_control_evidence=True,genuine_model_inference=False)
    else:
        if (len(apps) != 1 or apps[0].get('implementation_id') != APP or
                apps[0]['parameters']['mode'] != MODES[arm] or
                apps[0]['parameters'].get('budget_gflops') != .75 or
                apps[0]['parameters'].get('window_ms') != 100 or
                apps[0]['parameters'].get('task_profile') != {k:service['mec']['tasks_by_ue']['ue_4'][k]
                    for k in ('interval_ms','compute_gflop','deadline_ms')} or
                Path(apps[0]['parameters']['model_path']).resolve() != Path(plan['model']['path'])):
            raise ValueError('Expected exactly the declared budget controller and model')
        controller = controller_evidence(folder,summary,'budget',plan['model']['stamp'],MODES[arm])
        if not controller['valid_control_evidence']:
            raise ValueError('Budget controller lacks frozen inference/rules and actual effect evidence')
        if arm == 'qgbdt' and not controller['genuine_model_inference']:
            raise ValueError('QGBDT arm did not perform actual model inference')
        if arm == 'rules' and summary['budget_xapp'].get('qgbdt_selected_actions',0):
            raise ValueError('Rules arm unexpectedly used QGBDT model decisions')
    history = TaskEventCollector(plan['duration_s'],cohort_ms=100,start_s=.1,target_uids=TARGETS)
    policies = {uid:{} for uid in TARGETS}; external = hashlib.sha256(); count = 0
    for row in _records(folder,'rt_sys_trace'):
        count += 1
        if row.get('slot_id') != count or {u['ue_id'] for u in row['ues']} != POPULATION:
            raise ValueError('Every ordered six-UE trace slot must be recorded')
        arrivals = row.get('traffic_arrivals',{})
        if set(arrivals) != POPULATION:
            raise ValueError('Recorded exogenous inputs are incomplete')
        values = [arrivals[u].get('exogenous_bytes') for u in sorted(POPULATION)]
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError('Invalid exogenous byte inputs')
        external.update(fingerprint([count,values]).encode())
        for event in row.get('ue_service_events',[]): history.observe(event)
        for uid in TARGETS:
            policies[uid].update(mature_cohorts(row.get('ue_services',{}).get(uid,{}),count*.0005))
    if count != plan['slots']:
        raise ValueError('Recorded trace does not cover the full requested duration')
    metrics = task_metrics(history,policies,summary,service['mec']['tasks_by_ue']['ue_4'])
    packet = summary['packet_metrics']; energy = summary['ue_services']['per_ue']
    if set(packet) != POPULATION or set(energy) != POPULATION:
        raise ValueError('All six final packet/energy ledgers are required')
    for p in packet.values():
        if p['initial_bytes']+p['offered_bytes'] != p['delivered_bytes']+p['pending_bytes']+p['overflow_bytes']:
            raise ValueError('Packet bytes do not conserve actual arrivals/delivery')
    metrics.update(network_goodput_mbps=sum(p['delivered_bytes'] for p in packet.values())*8/plan['duration_s']/1e6,
        ue4_goodput_mbps=packet['ue_4']['delivered_bytes']*8/plan['duration_s']/1e6,
        network_ue_energy_j=sum(p['ue_energy_total_j'] for p in energy.values()),
        ue4_energy_j=energy['ue_4']['ue_energy_total_j'])
    physical = dict(manifest=manifest,configs={k:cfg['configs'][k] for k in
        ('ue','phy','sim','cell','beamforming') if k in cfg['configs']})
    return dict(arm=arm,case=arm,case_relative_path='cases/'+arm,audit_relative_path='audits/'+arm+'.json',
        source_sha256=plan['source_sha256'],model_sha256=plan['model']['sha256'],
        duration_s=plan['duration_s'],seed=plan['seed'],slots=count,
        physical_inputs_sha256=fingerprint(physical),exogenous_slot_inputs_sha256=external.hexdigest(),
        task_birth_inputs_sha256=metrics.pop('task_birth_inputs_sha256'),
        controller=controller,metrics=metrics,automatic_html_audits_passed=True,
        files_sha256={str(p.relative_to(folder)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()})


def compare(arms, plan, completed=False):
    if not arms: return dict(completed=False,arms={})
    base = arms['base']
    for arm,case in arms.items():
        for key in ('source_sha256','model_sha256','physical_inputs_sha256',
                    'exogenous_slot_inputs_sha256','task_birth_inputs_sha256'):
            if case[key] != base[key]: raise ValueError('Unmatched policy comparison: '+arm+'/'+key)
        if case['metrics']['mature_birth_counts'] != base['metrics']['mature_birth_counts']:
            raise ValueError('Mature task birth cohorts differ between policies')
    return dict(schema='oran.mec-budget-pressure-comparison.v1',completed=completed,
        duration_s=plan['duration_s'],seed=plan['seed'],source_sha256=plan['source_sha256'],
        model_sha256=plan['model']['sha256'],model_stamp=plan['model']['stamp'],
        scenario=plan['scenario'],budget_gflops=.75,matched_inputs_verified=True,
        controller_evidence_verified=all(c['controller']['valid_control_evidence'] for c in arms.values()),
        arms=arms,scope=plan['scope'],
        interpretation='Measured expected-workload quota control and QoS/energy tradeoffs. QGBDT matching the analytic guard demonstrates no additional learning benefit in this case.')


def checkpoint_inputs(plan, output):
    shutil.copy2(plan['model']['path'],output/'input-model.json')
    shutil.copy2(plan['service_config']['path'],output/'input-service-profile.json')
    root = Path(plan['project_root']); members = {}
    for directory in ('core','phy','ran','ric','state','models','traffic','experiments','configs','scenarios','scripts'):
        for path in sorted((root/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.json','.yaml','.yml','.sh','.js','.css','.html'):
                members[str(path.relative_to(root))] = sha256(path)
    for filename in ('requirements-ue-energy.txt','requirements.txt','pyproject.toml'):
        if (root/filename).is_file():members[filename]=sha256(root/filename)
    with tarfile.open(output/'source-checkpoint.tgz','w:gz') as archive:
        for name in members: archive.add(root/name,arcname=name,recursive=False)
    atomic_json(output/'input-checkpoint.json',dict(source_sha256=plan['source_sha256'],
        model_sha256=sha256(output/'input-model.json'),
        profile_sha256=sha256(output/'input-service-profile.json'),
        archive_sha256=sha256(output/'source-checkpoint.tgz'),source_files_sha256=members))


def run_queue(args, plan):
    output = Path(plan['output']); child = None; stopped = {'signal':None}; arms = {}
    with queue_lock(output):
        pp = output/'queue-plan.json'
        if pp.exists():
            if not args.resume or load(pp) != plan:
                raise ValueError('Existing study requires --resume with identical source/model/parameters')
            state = load(output/'queue-status.json')
            if state['plan_sha256'] != sha256(pp): raise ValueError('Pinned plan checksum changed')
            checkpoint = load(output/'input-checkpoint.json')
            for filename,key in [('input-model.json','model_sha256'),('input-service-profile.json','profile_sha256'),('source-checkpoint.tgz','archive_sha256')]:
                if sha256(output/filename) != checkpoint[key]: raise ValueError('Input checkpoint changed')
            if _alive((state.get('active_case') or {}).get('child_pid')):
                raise RuntimeError('Previous child is still alive; refusing a duplicate study')
        else:
            if args.resume or any(p.name!='.queue.lock' for p in output.iterdir()):
                raise ValueError('Use a fresh dedicated study directory, or explicitly resume its pinned plan')
            atomic_json(pp,plan); checkpoint_inputs(plan,output)
            state = dict(schema=plan['schema'],plan_sha256=sha256(pp),status='prepared',completed=False,
                completed_cases=[],attempts=[],failures=[],active_case=None)
        for directory in ('cases','audits','logs','attempts'): (output/directory).mkdir(exist_ok=True)
        def write():
            state.update(queue_pid=os.getpid(),updated_at_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(output/'queue-status.json',state)
        def stop(signum,frame):
            stopped['signal']=signum
            if child is not None and child.poll() is None:
                try: os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        handlers={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            verify_inputs(plan)
            for entry in state['completed_cases']:
                case=next(c for c in plan['cases'] if c['name']==entry['name'])
                path=output/'audits'/(case['name']+'.json')
                if sha256(path)!=entry['audit_sha256']: raise ValueError('Completed audit changed')
                audit=load(path)
                actual={str(p.relative_to(Path(case['output']))) for p in Path(case['output']).rglob('*') if p.is_file()}
                if actual!=set(audit['files_sha256']) or any(sha256(Path(case['output'])/n)!=s for n,s in audit['files_sha256'].items()):
                    raise ValueError('Completed raw data changed; refusing to rerun or overwrite')
                arms[case['name']]=audit
            state.update(status='running',completed=False,active_case=None);write()
            for case in plan['cases']:
                if case['name'] in arms: continue
                if stopped['signal'] is not None: break
                verify_inputs(plan); folder=Path(case['output']); rp=folder/'run_status.json'
                if not (args.resume and rp.exists() and load(rp).get('completed')):
                    remaining=len(ARMS)-len(arms)
                    needed=(args.min_free_gb+remaining*args.estimated_case_gb)*1024**3
                    free=shutil.disk_usage(output).free
                    if free<needed:
                        state.update(status='blocked_disk_space',active_case=None,
                            disk=dict(free_bytes=free,required_bytes=round(needed),remaining_cases=remaining))
                        write();return 75
                    archived=_archive_partial(case,output)
                    if archived:state['attempts'].append(dict(case=case['name'],archived=archived))
                    started=time.monotonic(); runtime_stop=False; terminated_at=None
                    with (output/'logs'/(case['name']+'.log')).open('xb') as log:
                        child=subprocess.Popen(case['command'],cwd=plan['project_root'],stdout=log,
                            stderr=subprocess.STDOUT,start_new_session=True)
                        state['active_case']=dict(name=case['name'],child_pid=child.pid,progress={});write()
                        last=0.
                        while child.poll() is None:
                            now=time.monotonic()
                            if now-last>=args.heartbeat_s:
                                available=shutil.disk_usage(output).free
                                if available<args.min_free_gb*1024**3 and not runtime_stop:
                                    runtime_stop=True;terminated_at=now
                                    os.killpg(child.pid,signal.SIGTERM)
                                progress={}
                                if rp.exists():
                                    try:
                                        value=load(rp);progress={k:value[k] for k in ('stage','last_completed_slot','simulated_seconds','simulation_wall_s_so_far','completed') if k in value}
                                    except (ValueError,OSError):pass
                                state['active_case'].update(progress=progress,elapsed_wall_s=now-started,free_bytes=available)
                                write();last=now
                            if (runtime_stop or stopped['signal'] is not None):
                                if terminated_at is None:terminated_at=now
                                if now-terminated_at>args.termination_grace_s:
                                    try:os.killpg(child.pid,signal.SIGKILL)
                                    except ProcessLookupError:pass
                            time.sleep(.5)
                        code=child.wait();child=None
                    if runtime_stop or stopped['signal'] is not None:
                        state.update(status='blocked_disk_space' if runtime_stop else 'interrupted',active_case=None);write()
                        return 75 if runtime_stop else 128+stopped['signal']
                    if code:raise RuntimeError('Simulation failed: '+case['name']+' exit '+str(code))
                state['active_case']=dict(name=case['name'],stage='auditing_completed_recording');write()
                verify_inputs(plan);audit=audit_case(case,plan);arms[case['name']]=audit
                comparison=compare(arms,plan)
                ap=output/'audits'/(case['name']+'.json');atomic_json(ap,audit)
                state['completed_cases'].append(dict(name=case['name'],audit=str(ap),audit_sha256=sha256(ap),output=case['output']))
                state['active_case']=None;atomic_json(output/'comparison-summary.json',comparison);write()
            if stopped['signal'] is not None:
                state.update(status='interrupted',active_case=None);write();return 128+stopped['signal']
            verify_inputs(plan)
            atomic_json(output/'comparison-summary.json',compare(arms,plan,completed=True))
            state.update(status='complete',completed=True,active_case=None);write();return 0
        except BaseException as error:
            if child is not None and child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=args.termination_grace_s)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                except ProcessLookupError:pass
            state['failures'].append(dict(case=(state.get('active_case') or {}).get('name'),
                type=type(error).__name__,message=str(error)))
            state.update(status='failed',completed=False,active_case=None);write();raise
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)


def main(argv=None):
    import json
    args=parse_args(argv);plan=build_plan(args)
    if args.dry_run:
        print(json.dumps(plan,indent=2));return 0
    return run_queue(args,plan)


if __name__=='__main__':
    raise SystemExit(main())

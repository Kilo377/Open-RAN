"""Operator-only v5 case isolation; production needs a separate root-go receipt.

The original worker still performs every admission, run and seal operation.
This driver only owns paths, a two-lane host limit, and immutable provenance.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
sys.dont_write_bytecode=True

BASE=Path('/dev/shm/oran_xapp_improvement_20261123/validation-v5')
GIB=1024**3
PROFILE_PINS={
    'traffic_profile':('traffic.json','92b672a5e61d0e8da9e8631ef415141a94aaa69b959309a3f6484522dc8024d8'),
    'ue_services_config':('ue-services.json','d3c760e525b1df7793a660921109a2bf5878b10acb84ebe16ce3a755b74d8769'),
    'ap_services_profile':('ap-services.json','8c43ca64099220d5de74d447424d38ec25482ea01890aba57b270d889f791035'),
    'base_config':('base.json','60e34598785b9d8e4300214430d388b0905817b27ed018712588e5b56fbe78a7')}
PYTHONS={'agx1':Path('/home/lboroldn/Desktop/oran_slot_sim_sionna_fast/.venv-orin/bin/python'),
         'agx2':Path('/home/lboroldn/Desktop/oran_slot_sim_sionna/.venv-orin/bin/python')}
CATALOGUES={'legacy9':'84033c2a8b31a33bd8760d62977ec46d81a698df20717831396cc1a8690b0dd5',
    'extended24':'9279436f9f4b32536ffa780ffe1a7350c08e355501b9e485bbe07734e578b380',
    'balanced42':'2dcb2568cd66fc9b0640c871a47a8da916b5562c33320c96c88c2d4769c9bd20'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    path.chmod(0o444)


def case_root(worker,lane,ident):
    if worker not in PYTHONS or lane not in('lane-0','lane-1')or not re.fullmatch('[0-9a-f]{32}',ident):
        raise ValueError('Only two registered physical-worker lanes and an owned UUID are allowed')
    return BASE/worker/lane/ident/'source-v5'


def validate_plan(plan,*,require_go=False):
    r=plan['request'];root=case_root(plan['physical_worker_id'],plan['lane_id'],r['id'])
    body=dict(r);pin=body.pop('request_sha256')
    for value in [pin,plan['validation_protocol_sha256'],plan['driver_sha256'],r['source_archive_sha256'],
                  plan['source_members']['sha256']]:
        if not isinstance(value,str)or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('A pinned identity is not a SHA256')
    if (plan.get('schema')!='oran.validation-lane-case-plan.v1'or plan.get('validation_version')!='v5'
            or plan.get('max_validation_lanes_per_host')!=2 or Path(plan['project_root'])!=root
            or root.resolve()!=root or Path(plan['python'])!=PYTHONS[plan['physical_worker_id']]
            or r['worker_id']!=plan['physical_worker_id']or r['request_sha256']!=digest(body)
            or r['stage']not in('dev','holdout')or r['mode']not in('off','frozen')
            or r['seed']not in([341710]if r['stage']=='dev'else[341721,341722])
            or r['slots']!=22440 or r['balanced_start_windows']!=0
            or r['train_stop_utc']!='2026-11-23T09:20:00+00:00'
            or r['deadline_utc']!='2026-11-23T11:40:00+00:00'
            or plan['driver_sha256']!=sha(__file__)):
        raise ValueError('The operator lane case differs from the closed v5 protocol')
    inputs=BASE/'inputs'/plan['validation_protocol_sha256']
    expected={'source_archive':inputs/'source.tgz','source_members':inputs/'source-members.json',
              'protocol':inputs/'protocol-v5-validation.json'}
    for key,path in expected.items():
        if Path(plan[key]['path'])!=path:raise ValueError('An input escaped the registered read-only input scope')
    if plan['protocol']['sha256']!=plan['validation_protocol_sha256']or plan['source_archive']['sha256']!=r['source_archive_sha256']:
        raise ValueError('Source/protocol identities differ')
    for key,(name,pin)in PROFILE_PINS.items():
        if r['profiles'].get(key)!={'path':str(root.parent/'profiles'/name),'sha256':pin}:
            raise ValueError('The actual per-case profile path or original SHA differs')
    if set(r['profiles'])!=set(PROFILE_PINS):raise ValueError('Exactly four profiles are required')
    model=plan.get('model')
    if (r['mode']=='off'and(model is not None or r['initial_model_sha256']is not None)
            or r['mode']=='frozen'and(not model or model['sha256']!=r['initial_model_sha256']
                or Path(model['path'])!=inputs/'cases'/r['id']/'model.json')):
        raise ValueError('Pinned frozen model/OFF semantics differ')
    selection=plan.get('selection')
    if r['stage']=='holdout'and(not selection or selection['sha256']!=r['selection_sha256']
            or Path(selection['path'])!=inputs/'cases'/r['id']/'selection.json'):
        raise ValueError('Holdout needs its exact case-owned joint selection copy')
    if require_go:
        item=plan.get('root_go')
        if not item or Path(item['path'])!=inputs/'cases'/r['id']/'root-go.json'or sha(item['path'])!=item['sha256']:
            raise ValueError('No exact root-go receipt authorizes this request')
        go=json.loads(Path(item['path']).read_text())
        check_root_go(plan,go)
    return root


def check_root_go(plan,go):
    r=plan['request']
    if (go.get('schema')!='oran.validation-lane-root-go.v1'or go.get('production_validation_lanes_authorized')is not True
            or go.get('source_archive_sha256')!=r['source_archive_sha256']
            or go.get('validation_protocol_sha256')!=plan['validation_protocol_sha256']
            or go.get('driver_sha256')!=plan['driver_sha256']or go.get('max_validation_lanes_per_host')!=2
            or go.get('allowed_requests',{}).get(r['id'])!=r['request_sha256']
            or r['stage']=='holdout'and(go.get('holdout_authorized')is not True
                or go.get('selection_sha256')!=r['selection_sha256'])):
        raise ValueError('Root has not authorized these exact lanes/request/final models')


def check_joint_selection(selected,request,protocol,protocol_sha):
    models=selected.get('models',{});variant=selected.get('power_action_set_version')
    locked=datetime.fromisoformat(selected['locked_utc'])
    target='throughput'if request['schema']=='oran.throughput-improvement-job.v1'else'power'
    if (selected.get('validation_version')!='v5'or selected.get('validation_protocol_sha256')!=protocol_sha
            or selected.get('source_archive_sha256')!=request['source_archive_sha256']
            or selected.get('both_models_locked')is not True or selected.get('holdout_results_used')is not False
            or selected.get('development_seeds')!=[341710]or set(models)!={'power','throughput'}
            or any(not re.fullmatch('[0-9a-f]{64}',str(v))for v in models.values())
            or models.get('throughput')!=protocol['retained_throughput_model_sha256']
            or locked.tzinfo is None or locked.utcoffset().total_seconds()!=0
            or locked>min(datetime.now(timezone.utc),datetime.fromisoformat(protocol['model_selection_stop_utc']))
            or selected.get('power_action_catalogue_sha256')!=CATALOGUES.get(variant)
            or request['mode']=='frozen'and request['initial_model_sha256']!=models[target]
            or target=='power'and request.get('action_set_version')!=variant):
        raise ValueError('The exact joint model/action/protocol lock is required before final execution')


def checked_input(item):
    p=Path(item['path'])
    if p.resolve()!=p or not p.is_file()or sha(p)!=item['sha256']:raise ValueError('Registered input changed')
    return p


def extract_exact(archive,pin,members,root):
    if sha(archive)!=pin:raise ValueError('Whole source archive differs')
    with tarfile.open(archive)as stream:
        entries=list(stream);files=[m for m in entries if m.isfile()]
        if any(not(m.isfile()or m.isdir())for m in entries)or len(files)!=len(members)or set(m.name for m in files)!=set(members):
            raise ValueError('The source archive has a different member set or special entries')
        root.mkdir(parents=True,exist_ok=False)
        for m in files:
            relative=Path(m.name);target=root/relative
            if relative.is_absolute()or '..'in relative.parts or not target.resolve().is_relative_to(root):raise ValueError('Unsafe source member')
            data=stream.extractfile(m).read();expected=members[m.name]
            if len(data)!=expected['bytes']or hashlib.sha256(data).hexdigest()!=expected['sha256']:raise ValueError('Source member differs')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);target.chmod(expected['mode'])
    return len(files)


def memory_available():
    return int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))*1024


def execution_protocol(plan):
    p=checked_input(plan['protocol']);protocol=json.loads(p.read_text());execution=protocol.get('execution',{})
    if (protocol.get('schema')!='oran.validation-only-protocol.v5'or protocol.get('version')!='v5'
            or protocol.get('source_archive_sha256')!=plan['request']['source_archive_sha256']
            or protocol.get('development_stop_utc')!='2026-11-23T09:20:00+00:00'
            or protocol.get('model_selection_stop_utc')!='2026-11-23T09:20:00+00:00'
            or protocol.get('total_deadline_utc')!='2026-11-23T11:40:00+00:00'
            or protocol.get('allowed_stages')!=['dev','holdout']or protocol.get('allowed_modes')!=['off','frozen']
            or protocol.get('slots')!=22440 or protocol.get('observation_slots')!=20000
            or execution.get('mode')!='isolated_validation_lanes'or execution.get('max_validation_lanes_per_host')!=2
            or execution.get('driver_sha256')!=plan['driver_sha256']
            or execution.get('minimum_memavailable_before_launch_bytes')!=28*GIB
            or execution.get('minimum_shm_free_bytes')!=8*GIB):
        raise ValueError('The sealed protocol does not authorize two isolated v5 validation lanes')
    probes=execution.get('passed_2000_probe_certificates')
    if not isinstance(probes,dict)or set(probes)!={'agx1','agx2'}or any(not re.fullmatch('[0-9a-f]{64}',str(pin))for pin in probes.values()):
        raise ValueError('Both completed 2000-slot probe certificates must be registered')
    return protocol


def audit_original_seal(case,protocol,protocol_sha):
    """Independent of process state: consume only original sealed provenance."""
    case=Path(case);seal=json.loads((case/'sealed-manifest.json').read_text());files=seal['files']
    state=json.loads((case/'status.json').read_text())
    if state.get('phase')!='complete'or state.get('slot')!=22440 or state.get('sealed')is not True or seal.get('failed'):
        raise ValueError('Lane provenance requires an actual complete original worker seal')
    required={'request.json','operator-driver.json','operator-driver.py','operator-protocol.json',
              'operator-case-plan.json','operator-root-go.json','operator-source-members.json','run/config_snapshot.json','command.json'}
    required|={'operator-profiles/'+name for name,_ in PROFILE_PINS.values()}
    if not required.issubset(files):raise ValueError('Original worker seal is missing lane/profile provenance')
    for name,pin in files.items():
        p=case/name
        if p.resolve().is_relative_to(case.resolve())is False or p.is_symlink()or not p.is_file()or sha(p)!=pin['sha256']or p.stat().st_size!=pin['bytes']:
            raise ValueError('An original sealed lane member changed')
    r=json.loads((case/'request.json').read_text());receipt=json.loads((case/'operator-driver.json').read_text())
    plan=json.loads((case/'operator-case-plan.json').read_text());validate_plan(plan)
    if (plan['request']!=r or not plan.get('root_go')or sha(case/'operator-root-go.json')!=plan['root_go']['sha256']
            or sha(case/'operator-source-members.json')!=plan['source_members']['sha256']):
        raise ValueError('Sealed operator plan/provenance inputs differ')
    check_root_go(plan,json.loads((case/'operator-root-go.json').read_text()))
    if r['stage']=='holdout'and('operator-selection.json'not in files or sha(case/'operator-selection.json')!=r['selection_sha256']):
        raise ValueError('The actual prelocked selection is not in the original seal')
    if r['stage']=='holdout':check_joint_selection(json.loads((case/'operator-selection.json').read_text()),r,protocol,protocol_sha)
    root=case_root(r['worker_id'],receipt.get('lane_id'),r['id']);body=dict(r);pin=body.pop('request_sha256')
    if (case.name!=r['id']or receipt.get('schema')!='oran.validation-lane-driver-receipt.v1'or receipt.get('validation_version')!='v5'
            or receipt.get('physical_worker_id')!=r['worker_id']or receipt.get('case_id')!=r['id']
            or receipt.get('request_sha256')!=digest(body)or r['request_sha256']!=digest(body)
            or receipt.get('source_archive_sha256')!=protocol['source_archive_sha256']or r['source_archive_sha256']!=protocol['source_archive_sha256']
            or receipt.get('project_root')!=str(root)or receipt.get('max_validation_lanes_per_host')!=2
            or receipt.get('original_worker_validate_passed')is not True
            or receipt.get('old_v4_exclusive_lock_inactive')is not True
            or receipt.get('lane_lock_inherited_by_worker')is not True
            or receipt.get('legacy_shared_guard_inherited_by_worker')is not True
            or receipt.get('validation_protocol_sha256')!=protocol_sha or sha(case/'operator-protocol.json')!=protocol_sha
            or json.loads((case/'operator-protocol.json').read_text())!=protocol
            or receipt.get('driver_sha256')!=protocol['execution']['driver_sha256']or sha(case/'operator-driver.py')!=protocol['execution']['driver_sha256']):
        raise ValueError('Sealed lane/source/protocol/driver identity differs')
    paths={}
    for key,(name,pin)in PROFILE_PINS.items():
        path=str(root.parent/'profiles'/name);expected={'path':path,'sha256':pin,'sealed_copy':'operator-profiles/'+name}
        if receipt.get('profiles',{}).get(key)!=expected or r['profiles'].get(key)!={'path':path,'sha256':pin}or sha(case/expected['sealed_copy'])!=pin:
            raise ValueError('Sealed actual profile bytes or original pin differ')
        paths[key]=expected
    command=json.loads((case/'command.json').read_text());argv=command['argv']
    if str(root/'scripts/orin/activate-gpu.sh')not in argv:raise ValueError('Actual command activated a different source')
    for flag in('--output','--live-state','--channel-cache-root'):
        if not Path(argv[argv.index(flag)+1]).is_relative_to(root):raise ValueError('A writable case path was shared or escaped')
    if any(not Path(value).is_relative_to(root)for value in command['cache_environment'].values()):
        raise ValueError('A runtime cache was shared or escaped')
    return {'passed':True,'lane_id':receipt['lane_id'],'project_root':str(root),'profiles':paths,
            'driver_sha256':receipt['driver_sha256'],'validation_protocol_sha256':protocol_sha}


def prepare_case(plan,*,require_go=False):
    root=validate_plan(plan,require_go=require_go);protocol=execution_protocol(plan);r=plan['request']
    archive=checked_input(plan['source_archive']);members_path=checked_input(plan['source_members'])
    members=json.loads(members_path.read_text())
    if len(members)!=1432:raise ValueError('All 1432 sealed source members are required')
    marker=root.parent/'operator-prepared.json';existing=marker.exists()
    if existing:
        if json.loads(marker.read_text())!={'request_sha256':r['request_sha256'],'driver_sha256':plan['driver_sha256'],
                'source_archive_sha256':r['source_archive_sha256'],'validation_protocol_sha256':plan['validation_protocol_sha256']}:
            raise ValueError('A prepared case differs; never repurpose an existing case')
        if any(sha(root/name)!=pin['sha256']for name,pin in members.items()):raise ValueError('Prepared source changed')
    else:
        claims=BASE/'claims';claims.mkdir(parents=True,exist_ok=True)
        write(claims/(r['id']+'.json'),dict(request_sha256=r['request_sha256'],physical_worker_id=r['worker_id'],project_root=str(root)))
        extract_exact(archive,r['source_archive_sha256'],members,root)
        (root/'.venv-orin').symlink_to(Path(plan['python']).parent.parent,target_is_directory=True)
        write(root/'.power-study-owned.json',dict(archive_sha256=r['source_archive_sha256'],root=str(root),scope=str(BASE)))
    for key,(name,pin)in PROFILE_PINS.items():
        source=Path(plan['protocol']['path']).parent/name
        if source.resolve()!=source or sha(source)!=pin:raise ValueError('Actual original profile bytes differ')
        target=root.parent/'profiles'/name
        if existing:
            if sha(target)!=pin:raise ValueError('Prepared actual profile changed')
        else:
            target.parent.mkdir(exist_ok=True);target.write_bytes(source.read_bytes());target.chmod(0o444)
    if plan.get('selection'):
        source=checked_input(plan['selection']);target=root.parent/'selection.json'
        if existing:
            if sha(target)!=sha(source):raise ValueError('Prepared joint lock changed')
        else:target.write_bytes(source.read_bytes());target.chmod(0o444)
    sys.path.insert(0,str(root));module_name=('experiments.throughput_improvement_worker'
        if r['schema']=='oran.throughput-improvement-job.v1'else'experiments.power_improvement_worker')
    if r['schema']not in('oran.power-improvement-job.v1','oran.throughput-improvement-job.v1'):raise ValueError('Unregistered worker')
    worker=importlib.import_module(module_name)
    member=module_name.replace('.','/')+'.py'
    if (Path(worker.__file__).resolve()!=root/member or sha(worker.__file__)!=members[member]['sha256']
            or worker.ROOT!=root or worker.SCOPE!=BASE.parent or worker.TRAIN_STOP!='2026-11-23T09:20:00+00:00'):
        raise ValueError('The original exact v5 worker was not imported from this case source')
    worker.validate(r,root) # Original admission is mandatory; no monkeypatching.
    if r['stage']=='holdout':
        selected=json.loads((root.parent/'selection.json').read_text())
        check_joint_selection(selected,r,protocol,plan['validation_protocol_sha256'])
        policy=importlib.import_module('ric.near_rt.pools.chongyu.ue_energy_rf')
        if (sha(policy.__file__)!=protocol['power_policy_sha256']
                or digest(policy.actions_for_version(selected['power_action_set_version']))!=selected['power_action_catalogue_sha256']):
            raise ValueError('The real sealed action resolver differs from the joint lock')
    job,run,args=worker.command(r,root)
    if existing:
        if json.loads((job/'request.json').read_text())!=r:raise ValueError('Prepared original request changed')
        if(job/'status.json').exists()or(job/'operator-driver.json').exists():raise ValueError('This case has already been launched; reconnect without restarting')
    else:
        run.mkdir(parents=True,exist_ok=False);write(job/'request.json',r)
    if plan.get('model'):
        source=checked_input(plan['model']);target=run/('throughput_mlp_model.json'if 'throughput'in module_name else'energy_rf_model.json')
        if existing:
            if sha(target)!=sha(source):raise ValueError('Prepared candidate changed')
        else:target.write_bytes(source.read_bytes());target.chmod(0o444)
    if not existing:write(marker,dict(request_sha256=r['request_sha256'],driver_sha256=plan['driver_sha256'],
        source_archive_sha256=r['source_archive_sha256'],validation_protocol_sha256=plan['validation_protocol_sha256']))
    return root,worker,job,run,members


def run(plan):
    if sys.platform!='linux':raise RuntimeError('Remote Linux execution only')
    root=validate_plan(plan,require_go=True)
    if (sys.prefix==sys.base_prefix or Path(sys.prefix).resolve()!=Path(plan['python']).parent.parent.resolve()):
        raise ValueError('Use the exact registered virtual environment')
    root,worker,job,run_dir,members=prepare_case(plan,require_go=True);r=plan['request']
    lock_path=BASE/'locks'/(plan['lane_id']+'.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
    # Shared access waits for the old exclusive v4 job to finish, then prevents
    # an old exclusive job racing the two independent validation lanes.
    legacy_lock=BASE.parent/'worker.lock'
    with lock_path.open('a')as lane,legacy_lock.open('r')as legacy:
        while True:
            worker.validate(r,root)
            try:fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(3)
        while True:
            worker.validate(r,root)
            try:fcntl.flock(legacy,fcntl.LOCK_SH|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(3)
        if memory_available()<28*GIB or shutil.disk_usage('/dev/shm').free<8*GIB:
            raise RuntimeError('No new validation process: preserve RAM/shm reserves')
        profiles={}
        for key,(name,pin)in PROFILE_PINS.items():
            source=Path(r['profiles'][key]['path'])
            if sha(source)!=pin:raise ValueError('Actual launch profile changed')
            target=job/'operator-profiles'/name;target.parent.mkdir(exist_ok=True);target.write_bytes(source.read_bytes());target.chmod(0o444)
            profiles[key]=dict(path=str(source),sha256=pin,sealed_copy=str(target.relative_to(job)))
        for name,source in [('operator-protocol.json',Path(plan['protocol']['path'])),('operator-driver.py',Path(__file__)),
                            ('operator-source-members.json',Path(plan['source_members']['path'])),('operator-root-go.json',Path(plan['root_go']['path']))]:
            target=job/name;target.write_bytes(source.read_bytes());target.chmod(0o444)
        if plan.get('selection'):
            target=job/'operator-selection.json';target.write_bytes(checked_input(plan['selection']).read_bytes());target.chmod(0o444)
        receipt=dict(schema='oran.validation-lane-driver-receipt.v1',validation_version='v5',
            physical_worker_id=plan['physical_worker_id'],lane_id=plan['lane_id'],case_id=r['id'],
            request_sha256=r['request_sha256'],source_archive_sha256=r['source_archive_sha256'],
            validation_protocol_sha256=plan['validation_protocol_sha256'],driver_sha256=sha(__file__),
            project_root=str(root),max_validation_lanes_per_host=2,driver_pid=os.getpid(),
            original_worker_validate_passed=True,profiles=profiles,lane_lock_inherited_by_worker=True,
            old_v4_exclusive_lock_inactive=True,legacy_shared_guard_inherited_by_worker=True)
        write(job/'operator-driver.json',receipt);write(job/'operator-case-plan.json',plan)
        with(job/'worker.log').open('xb')as log:
            process=subprocess.Popen([plan['python'],'-m',worker.__name__,'--request',str(job/'request.json')],
                cwd=root,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,
                stdout=log,stderr=log,start_new_session=True,pass_fds=(lane.fileno(),legacy.fileno()))
            write(root.parent/'operator-worker-pid.json',dict(pid=process.pid,case_id=r['id'],lane_id=plan['lane_id']))
            code=process.wait()
        status=json.loads((job/'status.json').read_text());seal=job/'sealed-manifest.json'
        passed=(code==0 and status.get('phase')=='complete'and status.get('sealed')is True
            and status.get('slot')==22440 and not json.loads(seal.read_text()).get('failed'))
        if any(sha(root/name)!=pin['sha256']for name,pin in members.items()):passed=False
        if any(sha(Path(item['path']))!=item['sha256']for item in profiles.values()):passed=False
        result=dict(schema='oran.validation-lane-driver-result.v1',passed=passed,case_id=r['id'],
            physical_worker_id=plan['physical_worker_id'],lane_id=plan['lane_id'],
            request_sha256=r['request_sha256'],driver_sha256=sha(__file__),
            validation_protocol_sha256=plan['validation_protocol_sha256'],worker_returncode=code,
            worker_seal_sha256=sha(seal),all_source_members_rechecked=len(members),
            actual_profile_bytes_rechecked=True,finished_utc=datetime.now(timezone.utc).isoformat())
        write(root.parent/'operator-driver-result.json',result)
        write(root.parent/'operator-driver-result.sha256.json',{'sha256':sha(root.parent/'operator-driver-result.json')})
        if not passed:raise RuntimeError('Original worker or post-run source/profile verification failed')
        return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--remote-run',type=Path);g.add_argument('--prepare-only',type=Path)
    args=parser.parse_args()
    if args.prepare_only:
        if sys.platform!='linux':raise RuntimeError('Remote Linux preparation only')
        plan=json.loads(args.prepare_only.read_text());root,*_=prepare_case(plan)
        print(json.dumps(dict(prepared=True,project_root=str(root),new_simulations=0)))
    else:
        plan=json.loads(args.remote_run.read_text())
        try:print(json.dumps(run(plan)))
        except BaseException as error:
            root=case_root(plan['physical_worker_id'],plan['lane_id'],plan['request']['id'])
            if root.parent.resolve()==root.parent:
                path=root.parent/'operator-driver-error.json'
                if not path.exists():write(path,{'error':str(error),'type':type(error).__name__,'case_id':plan['request']['id']})
            raise

"""Frozen four-arm case queue: warm Base first, then bounded parallel policies.

The callbacks audit real recordings and compute case-specific comparisons.
This module never trains models, changes simulation settings, or deletes runs.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tarfile
import time

from experiments.run_xapp_long_validation import (
    atomic_json, load, sha256, source_fingerprint, queue_lock,
    _alive, _archive_partial,
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify_inputs(plan):
    root=Path(plan['project_root'])
    if source_fingerprint(root)!=plan['source_sha256']:
        raise ValueError('Pinned case source/configuration changed')
    for item in plan['models'].values():
        if sha256(item['path'])!=item['sha256']:
            raise ValueError('Pinned frozen model changed')
    for path,expected in plan.get('input_files',{}).items():
        if sha256(path)!=expected:raise ValueError('Pinned input configuration changed: '+path)
    for path,expected in plan.get('launcher_sha256',{}).items():
        if sha256(root/path)!=expected:raise ValueError('Pinned launcher changed: '+path)


def checkpoint_inputs(plan, output):
    root=Path(plan['project_root']);models={};configs={}
    for name,entry in plan['models'].items():
        target=output/'input-models'/(name+'.json');target.parent.mkdir(exist_ok=True)
        shutil.copy2(entry['path'],target)
        models[name]=dict(file=str(target.relative_to(output)),sha256=sha256(target))
    for index,(path,expected) in enumerate(sorted(plan.get('input_files',{}).items())):
        target=output/'input-configs'/(str(index)+'-'+Path(path).name)
        target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
        configs[str(path)]=dict(file=str(target.relative_to(output)),sha256=sha256(target))
    members={}
    extensions={'.py','.json','.yaml','.yml','.sh','.js','.css','.html'}
    for directory in ('core','phy','ran','ric','state','models','traffic','experiments','configs','scenarios','scripts'):
        for path in sorted((root/directory).rglob('*')):
            if path.is_file() and path.suffix in extensions and '__pycache__' not in path.parts:
                members[str(path.relative_to(root))]=sha256(path)
    requirement=root/'requirements-ue-energy.txt'
    if requirement.is_file():members[requirement.name]=sha256(requirement)
    archive=output/'source-checkpoint.tgz'
    with tarfile.open(archive,'w:gz') as stream:
        for name in members:stream.add(root/name,arcname=name,recursive=False)
    atomic_json(output/'input-checkpoint.json',dict(schema='oran.released-case-inputs.v1',
        created_at_utc=utc(),source_sha256=plan['source_sha256'],models=models,configs=configs,
        source_archive='source-checkpoint.tgz',source_archive_sha256=sha256(archive),
        source_files_sha256=members))


def verify_checkpoint(output, plan):
    checkpoint=load(output/'input-checkpoint.json')
    if checkpoint['source_sha256']!=plan['source_sha256']:
        raise ValueError('Source checkpoint identity changed')
    if sha256(output/checkpoint['source_archive'])!=checkpoint['source_archive_sha256']:
        raise ValueError('Source checkpoint archive changed')
    for name,item in checkpoint['models'].items():
        if sha256(output/item['file'])!=plan['models'][name]['sha256'] or item['sha256']!=plan['models'][name]['sha256']:
            raise ValueError('Frozen model checkpoint changed')
    for source,item in checkpoint['configs'].items():
        if sha256(output/item['file'])!=item['sha256'] or item['sha256']!=plan['input_files'][source]:
            raise ValueError('Input configuration checkpoint changed')
    return checkpoint


def _recording_hashes(audit):
    return audit.get('files_sha256',audit.get('recorded_file_sha256'))


def _verify_saved(case, entry, output):
    path=output/'audits'/(case['name']+'.json')
    if sha256(path)!=entry['audit_sha256']:raise ValueError('Completed case audit changed')
    audit=load(path);expected=_recording_hashes(audit);folder=Path(case['output'])
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    if not expected or actual!=set(expected):raise ValueError('Completed recording file set changed')
    for name,value in expected.items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path)!=value:
            raise ValueError('Completed recording changed: '+case['name']+'/'+name)
    return audit


def _terminate(active, grace):
    for worker in active.values():
        child=worker['child']
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=time.monotonic()+grace
    for worker in active.values():
        child=worker['child']
        try:child.wait(timeout=max(.01,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
        worker['log'].close()


def run_queue(args, plan, audit_case, compare):
    # Plans live as JSON on disk; readiness counters may have integer keys in
    # memory. Compare the same representation on initial launch and resume.
    plan=json.loads(json.dumps(plan,allow_nan=False))
    output=Path(plan['output']);root=Path(plan['project_root'])
    cases={c['name']:c for c in plan['cases']}
    if (len(plan['cases'])!=4 or len(cases)!=4 or list(cases)[0]!='base' or
            any(not re.fullmatch('[a-z][a-z0-9_]*',name) for name in cases)):
        raise ValueError('A four-arm case must start with Base and use unique safe arm names')
    for name,case in cases.items():
        if Path(case['output']).resolve()!=output.resolve()/'cases'/name:
            raise ValueError('Each recording must be in its own cases/<arm> directory')
    if not 1<=args.max_parallel<=3:raise ValueError('Choose one to three parallel ON arms')
    if not math.isfinite(args.min_free_gb) or args.min_free_gb<.25:
        raise ValueError('Retain at least 0.25 GiB of disk reserve')
    if not math.isfinite(args.estimated_case_gb) or args.estimated_case_gb<=0:
        raise ValueError('Positive expected per-case space is required')
    active={};audits={};audit_errors=[];stop={'signal':None};state=None
    with queue_lock(output):
        pp=output/'queue-plan.json'
        if pp.exists():
            if not args.resume or load(pp)!=plan:raise ValueError('Resume requires the identical pinned case plan')
            state=load(output/'queue-status.json')
            if state['plan_sha256']!=sha256(pp):raise ValueError('Saved case plan changed')
            if any(_alive(w.get('child_pid')) for w in state.get('active_cases',[])):
                raise RuntimeError('A previous case child is still running')
            verify_checkpoint(output,plan)
        else:
            if args.resume or any(p.name!='.queue.lock' for p in output.iterdir()):
                raise ValueError('Use a fresh case directory or resume its existing plan')
            verify_inputs(plan);atomic_json(pp,plan);checkpoint_inputs(plan,output)
            state=dict(schema='oran.released-conflict-queue.v1',plan_sha256=sha256(pp),
                status='prepared',completed=False,started_at_utc=utc(),active_cases=[],
                completed_cases=[],attempts=[],failures=[])
        for name in ('cases','audits','logs','attempts'):(output/name).mkdir(exist_ok=True)

        def write():
            state.update(queue_pid=os.getpid(),updated_at_utc=utc())
            atomic_json(output/'queue-status.json',state)

        def refresh():
            rows=[]
            for name,worker in active.items():
                progress={};path=Path(cases[name]['output'])/'run_status.json'
                if path.exists():
                    try:
                        value=load(path)
                        progress={k:value[k] for k in ('stage','last_completed_slot','simulated_seconds','simulation_wall_s_so_far','completed') if k in value}
                    except (ValueError,OSError):pass
                rows.append(dict(name=name,child_pid=worker['child'].pid,progress=progress,
                    elapsed_wall_s=time.monotonic()-worker['started'],stage=worker.get('stage','running')))
            state['active_cases']=rows;state['free_bytes']=shutil.disk_usage(output).free;write()

        def handler(signum,frame):
            stop['signal']=signum

        def save_completed(case):
            name=case['name']
            try:
                audit=audit_case(case,plan)
                audit.setdefault('case_relative_path','cases/'+name)
                audit.setdefault('audit_relative_path','audits/'+name+'.json')
                if not _recording_hashes(audit):
                    raise ValueError('Case audit must pin every recorded file')
            except Exception as error:
                # The simulator succeeded. An independent recording-audit
                # failure must not destroy other expensive healthy workers.
                # Stop scheduling, retain the failed recording, and drain all
                # workers already launched; never accept this case as audited.
                audit_errors.append(error)
                state['failures'].append(dict(case=name,phase='audit',
                    type=type(error).__name__,message=str(error),at_utc=utc()))
                state.update(status='draining_after_audit_failure',completed=False,
                    unlaunched_cases=list(pending))
                refresh()
                return False
            ap=output/'audits'/(name+'.json');atomic_json(ap,audit);audits[name]=audit
            state['completed_cases'].append(dict(name=name,output=case['output'],
                audit=str(ap),audit_sha256=sha256(ap),completed_at_utc=utc()))
            # A successful case still has to pass the matched-input comparison.
            # Comparison, persistence, and global input failures remain fatal.
            atomic_json(output/'comparison-summary.json',compare(audits,plan,completed=False))
            refresh()
            return True

        handlers={s:signal.signal(s,handler) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            verify_inputs(plan)
            for entry in state['completed_cases']:
                audits[entry['name']]=_verify_saved(cases[entry['name']],entry,output)
            pending=[name for name in cases if name not in audits]
            needed=(args.min_free_gb+len(pending)*args.estimated_case_gb)*1024**3
            if pending and shutil.disk_usage(output).free<needed:
                state.update(status='blocked_disk_space',completed=False,active_cases=[],
                    disk=dict(required_bytes=math.ceil(needed),free_bytes=shutil.disk_usage(output).free))
                write();return 75
            state.update(status='running',completed=False,active_cases=[],unlaunched_cases=[]);write()
            last=0.
            while (pending and not audit_errors) or active:
                if stop['signal'] is not None:
                    _terminate(active,args.termination_grace_s);active.clear()
                    state.update(status='interrupted',completed=False,active_cases=[]);write()
                    return 128+stop['signal']
                if shutil.disk_usage(output).free<args.min_free_gb*1024**3:
                    _terminate(active,args.termination_grace_s);active.clear()
                    state.update(status='blocked_disk_space',completed=False,active_cases=[],
                        disk=dict(scope='Runtime reserve; partial recordings retained',free_bytes=shutil.disk_usage(output).free))
                    write();return 75
                capacity=1 if 'base' not in audits else args.max_parallel
                while pending and not audit_errors and len(active)<capacity and ('base' in audits or not active):
                    name=pending.pop(0);case=cases[name];verify_inputs(plan)
                    rp=Path(case['output'])/'run_status.json'
                    if args.resume and rp.exists() and load(rp).get('completed'):
                        save_completed(case)
                        continue
                    archived=_archive_partial(case,output)
                    if archived:state['attempts'].append(dict(name=name,archived=archived))
                    log=(output/'logs'/(name+'.log')).open('xb')
                    try:
                        child=subprocess.Popen(case['command'],cwd=root,stdin=subprocess.DEVNULL,
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    except BaseException:log.close();raise
                    active[name]=dict(child=child,log=log,started=time.monotonic())
                    refresh()
                for name in list(active):
                    worker=active[name];child=worker['child'];code=child.poll()
                    if code is None:continue
                    child.wait();worker['log'].close()
                    if code:raise RuntimeError('Case child failed: '+name+' exit '+str(code))
                    worker['stage']='auditing';refresh();verify_inputs(plan)
                    del active[name]
                    save_completed(cases[name])
                if time.monotonic()-last>=args.heartbeat_s:refresh();last=time.monotonic()
                if (pending and not audit_errors) or active:time.sleep(.25)
            verify_inputs(plan);verify_checkpoint(output,plan)
            if audit_errors:raise audit_errors[0]
            if set(audits)!=set(cases):raise ValueError('Every case requires a complete audited recording')
            atomic_json(output/'comparison-summary.json',compare(audits,plan,completed=True))
            state.update(status='complete',completed=True,active_cases=[]);write();return 0
        except BaseException as error:
            _terminate(active,args.termination_grace_s);active.clear()
            if not any(error is original for original in audit_errors):
                state['failures'].append(dict(type=type(error).__name__,message=str(error),at_utc=utc()))
            state.update(status='failed',completed=False,active_cases=[]);write();raise
        finally:
            for s,old in handlers.items():signal.signal(s,old)

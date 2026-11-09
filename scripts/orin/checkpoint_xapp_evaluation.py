#!/usr/bin/env python3
"""Checkpoint frozen xApp artifacts and group progress without changing a queue.

This is not a simulator-state checkpoint: interrupted simulations restart at
slot zero. It never writes source models, queue plans/status, or case folders.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import tarfile
import time
import uuid

GROUPS = ('core', 'phy', 'ran', 'ric', 'state', 'models', 'traffic', 'experiments', 'configs')
SUFFIXES = ('.py', '.json', '.yaml', '.yml')
KIND = 'frozen_model_and_group_progress'
TERMINAL = {'complete', 'failed', 'blocked_disk_space', 'interrupted'}


def now(): return datetime.now(timezone.utc).isoformat()
def load(path): return json.loads(Path(path).read_text())
def encoded(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(value): return hashlib.sha256(encoded(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def _fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path); _fsync_directory(path.parent)
    finally:
        if temporary.exists(): temporary.unlink()


def source_fingerprint(root):
    root = Path(root); h = hashlib.sha256()
    for group in GROUPS:
        for path in sorted((root / group).rglob('*')):
            if path.is_file() and path.suffix in SUFFIXES:
                h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def archive_fingerprint(path):
    files = {}
    with tarfile.open(path, 'r:*') as archive:
        for item in archive.getmembers():
            name = PurePosixPath(item.name)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe source archive member')
            name = str(name)
            if not item.isfile() or not any(name.startswith(g + '/') for g in GROUPS) or PurePosixPath(name).suffix not in SUFFIXES: continue
            if name in files: raise ValueError('Repeated source archive member')
            files[name] = archive.extractfile(item).read()
    h = hashlib.sha256()
    for group in GROUPS:
        for name in sorted(n for n in files if n.startswith(group + '/')):
            h.update(name.encode()); h.update(files[name])
    return h.hexdigest()


def copy_verified(source, destination, expected):
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    if sha256(source) != expected: raise ValueError('Source checksum changed before checkpoint copy')
    with Path(source).open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush(); os.fsync(dst.fileno())
    _fsync_directory(destination.parent)
    if sha256(destination) != expected or sha256(source) != expected:
        raise ValueError('Checksum changed during checkpoint copy')


@contextmanager
def checkpoint_lock(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root / '.checkpoint.lock').open('a+') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another checkpoint sidecar owns this directory') from error
        stream.seek(0); stream.truncate(); stream.write(str(os.getpid()) + '\n'); stream.flush()
        yield


def _paths(queue, checkpoint_root):
    queue = Path(queue).resolve(); root = Path(checkpoint_root).resolve()
    plan = load(queue / 'queue-plan.json')
    if Path(plan['output']).resolve() != queue:
        raise ValueError('Queue plan belongs to another output directory')
    if root == queue or not root.is_relative_to(Path(plan['project_root']).resolve() / 'outputs'):
        raise ValueError('Checkpoints need a dedicated project outputs directory')
    if root.is_relative_to(Path(plan['model_manifest']).parent.resolve()):
        raise ValueError('Checkpoints must stay outside the original model input directory')
    for case in plan['cases']:
        if root.is_relative_to(Path(case['output']).resolve()):
            raise ValueError('Checkpoints must not change a case recording directory')
    return queue, root, plan


def _live_inputs(queue, plan, expected_plan_sha=None):
    plan_sha = sha256(queue / 'queue-plan.json')
    status = load(queue / 'queue-status.json')
    if expected_plan_sha is not None and plan_sha != expected_plan_sha:
        raise ValueError('Immutable queue plan changed')
    if status.get('plan_sha256') != plan_sha:
        raise ValueError('Queue status does not match its immutable plan')
    if source_fingerprint(plan['project_root']) != plan['source_sha256']:
        raise ValueError('Executed source/config fingerprint changed')
    for path, expected in ((plan['model_manifest'], plan['manifest_sha256']),
                           (plan['profile'], plan['profile_sha256'])):
        if sha256(path) != expected: raise ValueError('Input manifest/profile checksum changed')
    for model in plan['models'].values():
        if sha256(model['path']) != model['sha256']: raise ValueError('Frozen model checksum changed')
    return plan_sha, status


def _model_layout(plan):
    manifest = load(plan['model_manifest']); entries = manifest.get('models', {})
    if set(entries) != set(plan['models']): raise ValueError('Model manifest and plan differ')
    paths = {}; used = {Path(plan['model_manifest']).name}
    for app, entry in entries.items():
        relative = PurePosixPath(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in used:
            raise ValueError('Portable checkpoints require distinct relative model paths')
        used.add(str(relative))
        source = (Path(plan['model_manifest']).parent / str(relative)).resolve()
        if source != Path(plan['models'][app]['path']).resolve() or entry['sha256'] != plan['models'][app]['sha256']:
            raise ValueError('Model manifest path/hash differs from immutable plan')
        paths[app] = str(relative)
    return paths


def _verify_modelset(root, modelset_id=None):
    root = Path(root)
    pointer = load(root / 'latest-modelset.json')
    if modelset_id is not None and pointer['modelset_id'] != modelset_id:
        raise ValueError('Progress refers to another frozen model set')
    ident = pointer['modelset_id']
    if not re.fullmatch(r'modelset-[0-9a-f]{20}', ident): raise ValueError('Invalid model set ID')
    folder = root / 'modelsets' / ident
    if sha256(folder / 'manifest.json') != pointer['manifest_sha256']:
        raise ValueError('Checkpoint manifest checksum changed')
    manifest = load(folder / 'manifest.json')
    if 'modelset-' + digest(manifest['identity'])[:20] != ident:
        raise ValueError('Checkpoint identity changed')
    inventory = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p != folder / 'manifest.json'}
    if inventory != set(manifest['files']): raise ValueError('Checkpoint file inventory changed')
    for name, expected in manifest['files'].items():
        if sha256(folder / name) != expected: raise ValueError('Checkpoint file checksum changed: ' + name)
    plan = load(folder / 'queue-plan.json')
    identity=manifest['identity']; expected_files=manifest['files']
    if (expected_files.get('queue-plan.json')!=identity['queue_plan_sha256'] or
            expected_files.get('source/source.tgz')!=identity['source_archive_sha256'] or
            expected_files.get('ap-profile.json')!=identity['profile_sha256'] or
            plan['source_sha256']!=identity['source_sha256'] or
            plan['manifest_sha256']!=identity['model_manifest_sha256'] or
            plan['profile_sha256']!=identity['profile_sha256']):
        raise ValueError('Checkpoint input identity differs from archived plan')
    manifest_name=manifest['model_manifest_name']
    if expected_files.get('models/'+manifest_name)!=identity['model_manifest_sha256']:
        raise ValueError('Checkpoint model manifest differs from its declared identity')
    models=load(folder/'models'/manifest_name)['models']
    if set(models)!=set(plan['models']) or set(models)!=set(identity['models']):
        raise ValueError('Checkpoint model sets differ')
    for app,entry in models.items():
        relative=manifest['model_layout'][app]; expected=plan['models'][app]['sha256']
        if (entry['path']!=relative or entry['sha256']!=expected or
                identity['models'][app]!=dict(relative_path=relative,sha256=expected) or
                expected_files.get('models/'+relative)!=expected):
            raise ValueError('Checkpoint frozen model identity differs from its declared plan')
    if archive_fingerprint(folder / 'source/source.tgz') != plan['source_sha256']:
        raise ValueError('Archived source fingerprint differs from the original queue')
    return ident, folder, manifest, pointer


def initialize_checkpoint(queue, checkpoint_root, source_archive):
    """Caller holds checkpoint_lock for multi-process protection."""
    queue, root, plan = _paths(queue, checkpoint_root)
    plan_sha, _ = _live_inputs(queue, plan)
    source_archive = Path(source_archive).resolve()
    source_sha = sha256(source_archive)
    if archive_fingerprint(source_archive) != plan['source_sha256']:
        raise ValueError('Source archive does not match the executed queue fingerprint')
    layout = _model_layout(plan)
    identity = dict(queue=str(queue), queue_plan_sha256=plan_sha,
        source_sha256=plan['source_sha256'], source_archive_sha256=source_sha,
        model_manifest_sha256=plan['manifest_sha256'], profile_sha256=plan['profile_sha256'],
        models={app:dict(relative_path=layout[app],sha256=m['sha256']) for app,m in plan['models'].items()})
    ident = 'modelset-' + digest(identity)[:20]
    target = root / 'modelsets' / ident; target.parent.mkdir(parents=True, exist_ok=True)
    pointer_path = root / 'latest-modelset.json'
    if pointer_path.exists():
        found, folder, manifest, pointer = _verify_modelset(root)
        if found != ident or manifest['identity'] != identity:
            raise ValueError('A different immutable checkpoint set already exists')
        return dict(modelset_id=found,modelset_path=str(folder),manifest_sha256=pointer['manifest_sha256'],
            checkpoint_kind=KIND,supports_mid_simulation_resume=False)
    if target.exists(): raise ValueError('Unpublished checkpoint folder already exists; do not overwrite evidence')
    temporary = target.parent / ('.' + ident + '.tmp-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        copies = [('queue-plan.json', queue/'queue-plan.json',plan_sha),
            ('models/'+Path(plan['model_manifest']).name,Path(plan['model_manifest']),plan['manifest_sha256']),
            ('ap-profile.json',Path(plan['profile']),plan['profile_sha256']),
            ('source/source.tgz',source_archive,source_sha)]
        copies += [('models/'+layout[app],Path(m['path']),m['sha256']) for app,m in plan['models'].items()]
        for relative, source, expected in copies: copy_verified(source, temporary/relative,expected)
        manifest = dict(schema='oran.frozen-model-checkpoint.v1',modelset_id=ident,identity=identity,
            checkpoint_kind=KIND,supports_mid_simulation_resume=False,created_at_utc=now(),
            model_manifest_name=Path(plan['model_manifest']).name,model_layout=layout,
            files={name:expected for name,_,expected in copies},
            scope='Frozen JSON weights/history plus reproducible inputs. No live simulator queues, HARQ, RNG, channel arrays or raw case data are saved.')
        atomic_json(temporary/'manifest.json',manifest)
        _live_inputs(queue,plan,plan_sha)
        if sha256(source_archive) != source_sha: raise ValueError('Source archive changed during checkpoint creation')
        os.rename(temporary,target); _fsync_directory(target.parent)
        pointer=dict(modelset_id=ident,manifest_sha256=sha256(target/'manifest.json'),
            checkpoint_kind=KIND,supports_mid_simulation_resume=False)
        atomic_json(pointer_path,pointer)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    _verify_modelset(root,ident)
    return dict(pointer,modelset_path=str(target))


def _receipt_payload(entry, modelset_id, modelset_sha, plan_sha, audit_sha):
    return dict(schema='oran.completed-group-checkpoint.v1',case=entry['name'],
        modelset_id=modelset_id,modelset_manifest_sha256=modelset_sha,
        queue_plan_sha256=plan_sha,case_audit_sha256=audit_sha,
        original_completed_entry=entry,checkpoint_kind=KIND,supports_mid_simulation_resume=False)


def snapshot_progress(queue, checkpoint_root, modelset_id):
    queue,root,plan = _paths(queue,checkpoint_root)
    ident,_,manifest,pointer = _verify_modelset(root,modelset_id)
    plan_sha,status = _live_inputs(queue,plan,manifest['identity']['queue_plan_sha256'])
    cases={c['name'] for c in plan['cases']}; seen=set()
    for entry in status.get('completed_cases',[]):
        name=entry['name']
        if name not in cases or name in seen or not re.fullmatch(r'[a-z0-9_-]+',name):
            raise ValueError('Invalid/repeated completed case')
        seen.add(name); audit=queue/'audits'/(name+'.json')
        if Path(entry['audit']).resolve()!=audit.resolve() or sha256(audit)!=entry['audit_sha256']:
            raise ValueError('Completed group audit checksum changed')
        recorded=load(audit)
        if recorded.get('case')!=name or recorded.get('source_sha256')!=plan['source_sha256']:
            raise ValueError('Completed audit refers to another case/source')
        payload=_receipt_payload(entry,ident,pointer['manifest_sha256'],plan_sha,entry['audit_sha256'])
        receipt=root/'completed'/(name+'.json')
        if receipt.exists():
            if load(receipt)!=payload: raise ValueError('Immutable completed group receipt changed')
        else: atomic_json(receipt,payload)
    snapshot=dict(schema='oran.group-progress-checkpoint.v1',captured_at_utc=now(),
        modelset_id=ident,modelset_manifest_sha256=pointer['manifest_sha256'],
        queue_plan_sha256=plan_sha,checkpoint_kind=KIND,supports_mid_simulation_resume=False,
        queue_status=status,completed_case_count=len(seen),
        scope='Observational queue progress only. Original audits/raw files remain authoritative; incomplete simulations restart from slot zero.')
    atomic_json(root/'latest-progress.json',snapshot)
    return snapshot


def verify_checkpoint(queue, checkpoint_root):
    queue=Path(queue).resolve();root=Path(checkpoint_root).resolve()
    ident,folder,manifest,pointer=_verify_modelset(root)
    plan=load(folder/'queue-plan.json')
    if Path(manifest['identity']['queue']).resolve()!=queue: raise ValueError('Checkpoint belongs to another queue')
    count=0
    for receipt in sorted((root/'completed').glob('*.json')):
        value=load(receipt);entry=value['original_completed_entry'];name=value['case']
        if receipt.name!=name+'.json': raise ValueError('Completed receipt filename differs from its case')
        expected=_receipt_payload(entry,ident,pointer['manifest_sha256'],manifest['identity']['queue_plan_sha256'],entry['audit_sha256'])
        if value!=expected: raise ValueError('Completed receipt identity/checksum changed')
        audit=queue/'audits'/(name+'.json')
        if sha256(audit)!=value['case_audit_sha256']: raise ValueError('Original completed-case audit checksum changed')
        count+=1
    progress=root/'latest-progress.json'
    if progress.exists():
        value=load(progress)
        if value['modelset_id']!=ident or value['modelset_manifest_sha256']!=pointer['manifest_sha256'] or value['queue_plan_sha256']!=manifest['identity']['queue_plan_sha256']:
            raise ValueError('Progress refers to another model set/plan')
    return dict(valid=True,modelset_id=ident,models_verified=len(plan['models']),
        completed_receipts_verified=count,raw_case_data_verified=False,
        checkpoint_kind=KIND,supports_mid_simulation_resume=False)


def restore_models(queue, checkpoint_root, destination):
    """Restore weights to a new directory; never replace models in active use."""
    queue=Path(queue).resolve();root=Path(checkpoint_root).resolve();dest=Path(destination).resolve()
    ident,folder,manifest,pointer=_verify_modelset(root)
    plan=load(folder/'queue-plan.json'); original=Path(plan['model_manifest']).parent.resolve()
    project=Path(plan['project_root']).resolve()
    if queue!=Path(manifest['identity']['queue']).resolve(): raise ValueError('Checkpoint belongs to another queue')
    if dest.is_relative_to(original) or dest.is_relative_to(queue) or dest.is_relative_to(root) or not dest.is_relative_to(project/'outputs'):
        raise ValueError('Restore into a separate new project outputs directory, never active inputs or queue records')
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError('Restore destination must be new or empty')
    dest.parent.mkdir(parents=True,exist_ok=True)
    temporary=dest.parent/('.'+dest.name+'.restore-'+uuid.uuid4().hex);temporary.mkdir()
    try:
        names=[manifest['model_manifest_name']]+list(manifest['model_layout'].values())
        for name in names: copy_verified(folder/'models'/name,temporary/name,manifest['files']['models/'+name])
        if dest.exists(): dest.rmdir()
        os.rename(temporary,dest);_fsync_directory(dest.parent)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return dict(restored=True,destination=str(dest),models_restored=len(plan['models']),modelset_id=ident,
        directly_resumes_original_queue=False,
        scope='Original queue resume requires its immutable original model paths and raw/audit files. This command only restores frozen model artifacts to a new directory.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('once','watch','verify','restore-models'))
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--checkpoint-root',type=Path)
    parser.add_argument('--source-archive',type=Path)
    parser.add_argument('--interval-s',type=float,default=60.)
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args(argv)
    root=(args.checkpoint_root or args.queue/'checkpoints').resolve()
    if not math.isfinite(args.interval_s) or not 1<=args.interval_s<=60:
        parser.error('--interval-s must be in [1,60]')
    if args.command in ('once','watch') and args.source_archive is None:
        parser.error('--source-archive is required for checkpoint creation')
    if args.command=='restore-models' and args.destination is None:
        parser.error('--destination is required')
    with checkpoint_lock(root):
        if args.command=='verify': result=verify_checkpoint(args.queue,root)
        elif args.command=='restore-models': result=restore_models(args.queue,root,args.destination)
        else:
            result=initialize_checkpoint(args.queue,root,args.source_archive)
            stopped={'signal':None}
            def stop(signum,frame): stopped['signal']=signum
            previous={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
            try:
                while True:
                    progress=snapshot_progress(args.queue,root,result['modelset_id'])
                    if args.command=='once' or progress['queue_status']['status'] in TERMINAL or stopped['signal'] is not None: break
                    deadline=time.monotonic()+args.interval_s
                    while time.monotonic()<deadline and stopped['signal'] is None:
                        time.sleep(min(1.,max(0.,deadline-time.monotonic())))
                result.update(queue_status=progress['queue_status']['status'],completed_case_count=progress['completed_case_count'])
            finally:
                for signum,handler in previous.items():signal.signal(signum,handler)
        print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())

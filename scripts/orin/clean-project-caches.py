#!/usr/bin/env python3
"""Remove only explicit rebuildable project caches; preserve all run/model data.

Dry-run is the default. outputs/, artifacts/, data/ and checkpoints/ are never
cleanup targets. Project-local ray-tracing channel_cache/ can be regenerated;
channel caches inside archived run outputs are deliberately retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PROTECTED={'outputs','artifacts','data','datasets','checkpoints','results'}


def inventory(root):
    rows=[]
    for name in sorted(PROTECTED):
        p=root/name
        if not p.is_dir()or p.is_symlink():continue
        for base,dirs,files in os.walk(p,followlinks=False):
            for file in sorted(files):
                f=Path(base)/file;s=f.lstat()
                rows.append((str(f.relative_to(root)),s.st_size,s.st_mtime_ns))
    return hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest(),len(rows)


def candidates(root):
    result=[]
    for name in ['.pytest_cache','channel_cache','.cache/drjit','.cache/cuda-orin','.cache/cupy','.cache/matplotlib']:
        p=root/name
        if p.is_dir()and not p.is_symlink()and p.resolve().is_relative_to(root):result.append(p)
    for base,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if d not in PROTECTED|{'.git','.cache','channel_cache','.pytest_cache'}
                 and not d.startswith('.venv')and not(Path(base)/d).is_symlink()]
        if '__pycache__'in dirs:
            result.append(Path(base)/'__pycache__');dirs.remove('__pycache__')
    return sorted(set(result))


def bytes_in(path):
    return sum((Path(base)/f).lstat().st_size for base,dirs,files in os.walk(path,followlinks=False)for f in files)


def active_processes(root):
    output=subprocess.check_output(['ps','-eww','-o','pid=,comm=,args='],text=True,errors='replace')
    found=[]
    for line in output.splitlines():
        parts=line.strip().split(None,2)
        if len(parts)!=3:continue
        pid,comm,args=parts
        executable=args.split(None,1)[0]if args else''
        if int(pid)==os.getpid()or not any('python'in value.lower()for value in(comm,executable)):continue
        cwd=None
        try:
            if Path('/proc').is_dir():cwd=(Path('/proc')/pid/'cwd').resolve(strict=True)
            else:
                listed=subprocess.run(['lsof','-a','-p',pid,'-d','cwd','-Fn'],capture_output=True,text=True,errors='replace')
                cwd=next((Path(r[1:]).resolve()for r in listed.stdout.splitlines()if r.startswith('n')),None)
        except(OSError,RuntimeError):pass
        if str(root)in args or(cwd is not None and cwd.is_relative_to(root)):
            found.append(dict(pid=int(pid),command=args,cwd=str(cwd)if cwd else None))
    return found


def clean(project,apply=False):
    root=Path(project).resolve()
    if not(root/'experiments/run_rt_sys.py').is_file():raise ValueError('Expected an explicit O-RAN project directory')
    active=active_processes(root)
    if apply and active:raise RuntimeError('Project Python processes are active; leave their caches intact: '+str(active))
    before,n=inventory(root);free_before=shutil.disk_usage(root).free
    items=[dict(path=str(p),bytes=bytes_in(p))for p in candidates(root)]
    if apply:
        for item in items:
            path=Path(item['path'])
            if path.is_symlink()or not path.resolve().is_relative_to(root)or any(x in PROTECTED for x in path.relative_to(root).parts):
                raise ValueError('Refusing a protected or redirected cache target')
            shutil.rmtree(path)
    after,count=inventory(root)
    if before!=after or n!=count:raise RuntimeError('Protected data inventory changed; inspect before proceeding')
    return dict(project=str(root),applied=apply,caches=items,cache_bytes=sum(i['bytes']for i in items),
        free_bytes_before=free_before,free_bytes_after=shutil.disk_usage(root).free,
        protected_data_inventory_sha256=after,protected_files=count,protected_data_unchanged=True,active_processes=active)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=ROOT)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args();print(json.dumps(clean(a.project,a.apply),indent=2))

if __name__=='__main__':main()

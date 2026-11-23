#!/usr/bin/env python3
"""Prepare an owned SMO worker copy; reuse an existing AGX environment read-only.

Run on the UI computer with an already authenticated SSH ControlMaster socket.
All source, model, cache and output writes stay in the named new worker folder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
SOURCES=('core','phy','ran','ric','state','models','traffic','configs','experiments',
    'scenarios','smo','scripts/orin','scripts/spark','artifacts/xapps/chongyu/20261117')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('host','socket','remote-root','remote-venv','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,default=20261118);a=p.parse_args()
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    archive=output/'source.tgz'
    with tarfile.open(archive,'w:gz')as t:
        for name in SOURCES:t.add(ROOT/name,arcname=name,filter=lambda m:None if '__pycache__'in m.name or m.name.endswith('.pyc')else m)
    root=Path(a.remote_root)
    if not root.is_absolute()or len(root.parts)<4 or not Path(a.remote_venv).is_absolute():p.error('Use absolute paths and a dedicated worker directory')
    payload=dict(root=str(root),venv=a.remote_venv,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    code='''import json,sys,tarfile
from pathlib import Path
c=json.loads(sys.argv[1]);root=Path(c['root']);marker=root/'.smo-owned-worker.json'
if root.exists() and not marker.exists():raise ValueError('Refusing to overwrite an unowned directory; choose a new folder')
venv=Path(c['venv'])
if not (venv/'bin/python').is_file():raise ValueError('Existing virtual environment unavailable')
root.mkdir(parents=True,exist_ok=True)
tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz').extractall(root)
p=root/'.venv-orin'
if p.is_symlink() and p.resolve()!=venv.resolve():raise ValueError('Worker points to a different environment')
if not p.exists():p.symlink_to(venv,target_is_directory=True)
marker.write_text(json.dumps(c));print(root)
'''
    subprocess.run(['ssh','-S',a.socket,'-o','BatchMode=yes',a.host,
        shlex.join(['python3','-c',code,json.dumps(payload)])],input=archive.read_bytes(),check=True)
    cfg=dict(host=a.host,socket=a.socket,project_root=str(root),python=str(Path(a.remote_venv)/'bin/python'),
        local_output=str(output/'jobs'),seed=a.seed,label='AGX · fast · 6 UEs')
    path=output/'worker-config.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
    print(path)


if __name__=='__main__':main()

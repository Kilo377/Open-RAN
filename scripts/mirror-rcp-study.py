"""Durably mirror sealed per-case checkpoints from an SSH study, including RAM runs."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smo.grid_identify import verify_bridge

REMOTE=r'''
from pathlib import Path
import json,hashlib,tarfile,io,sys,time,os
root=Path(ROOT_PATH);study=root/'outputs/study';sent=set(HAVE)
def packet(name,items):
 data=io.BytesIO()
 with tarfile.open(fileobj=data,mode='w:gz')as t:
  for path,arc in items:t.add(path,arcname=arc)
 b=data.getvalue();header=dict(name=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 sys.stdout.buffer.write(json.dumps(header).encode()+b'\n');sys.stdout.buffer.write(b);sys.stdout.buffer.flush()
packet('_metadata',[(study/'plan.json','plan.json'),(study/'specs','specs'),(root.parent/'source-files.json','deployed-source-files.json')])
deadline=time.time()+120
while not(study/'study_status.json').exists():
 if time.time()>deadline:raise TimeoutError('Runner did not publish progress; inspect its retained launch log')
 time.sleep(1)
while True:
 s=json.loads((study/'study_status.json').read_text())
 for name in s['done']:
  if name not in sent:
   packet(name,[(study/'cases'/name,'cases/'+name),(study/'checkpoints'/(name+'.json'),'checkpoints/'+name+'.json')]);sent.add(name)
 if s.get('failed'):raise RuntimeError(str(s['failed']))
 if s['completed']:
  os.chdir(root);sys.path.insert(0,str(root))
  from experiments.run_rcp_perturbation import verify
  from experiments.run_rcp_grid import dynamics_signature
  p=json.loads((study/'plan.json').read_text());verify(p)
  assert dynamics_signature(root)==p['dynamics_sha256']
  proof=dict(completed_cases=len(s['done']),source_sha256=p['source_sha256'],models_unchanged=True,
   models_sha256={str(Path(k).relative_to(root)):v for k,v in p['frozen_models'].items()},original_cases_retained=True,verified_unix_s=time.time())
  (study/'retention-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
  items=[(study/'study_status.json','study_status.json'),(study/'retention-verification.json','retention-verification.json'),
         (root/'channel_cache/grid_pilot','channel_cache'),(study/'logs','logs')]
  for name in('bridge_verified.json','environment.json'):
   if(study/name).exists():items.append((study/name,name))
  packet('_final',items);break
 time.sleep(3)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True);p.add_argument('--socket',required=True)
    p.add_argument('--remote-root',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bridge-reference',type=Path)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);archives=out/'archives';archives.mkdir(exist_ok=True)
    received_path=out/'received.json';received=json.loads(received_path.read_text())if received_path.exists()else{}
    code='ROOT_PATH='+repr(a.remote_root)+'\nHAVE='+repr([n for n in received if not n.startswith('_')])+'\n'+REMOTE
    proc=subprocess.Popen(['ssh','-S',a.socket,'-o','BatchMode=yes',a.host,'python3','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    proc.stdin.write(code.encode());proc.stdin.close();checked=(out/'bridge-local-verification.json').exists()
    while True:
        header=proc.stdout.readline()
        if not header:break
        h=json.loads(header);name=h['name']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',name)or not 0<h['bytes']<512*1024**2:raise ValueError('Invalid archive frame')
        temp=archives/(name+'.tgz.part');digest=hashlib.sha256();remaining=h['bytes']
        with temp.open('wb')as f:
            while remaining:
                chunk=proc.stdout.read(min(1024**2,remaining))
                if not chunk:raise EOFError('Incomplete SSH archive; existing checkpoints remain retained')
                f.write(chunk);digest.update(chunk);remaining-=len(chunk)
            f.flush();os.fsync(f.fileno())
        if digest.hexdigest()!=h['sha256']:raise ValueError('Archive checksum failed')
        target=archives/(name+'.tgz');temp.replace(target)
        with tarfile.open(target,'r:gz')as t:
            for m in t.getmembers():
                if not(m.isfile()or m.isdir())or m.name.startswith('/')or'..'in Path(m.name).parts:raise ValueError(m.name)
            t.extractall(out)
        if not name.startswith('_'):
            checkpoint=json.loads((out/'checkpoints'/(name+'.json')).read_text())
            for rel,sha in checkpoint['files_sha256'].items():
                if hashlib.sha256((out/'cases'/name/rel).read_bytes()).hexdigest()!=sha:raise ValueError('Raw checkpoint hash differs')
        received[name]=h
        tmp=received_path.with_suffix('.tmp');tmp.write_text(json.dumps(received,indent=2)+'\n');tmp.replace(received_path)
        print(json.dumps(dict(received=name,sealed_cases=sum(not n.startswith('_')for n in received))),flush=True)
        plan=json.loads((out/'plan.json').read_text())
        bridge={c['name']for c in plan['cases']if c.get('role')=='bridge'}
        if a.bridge_reference and not checked and bridge and bridge<=set(received):
            result=verify_bridge(out,a.bridge_reference);path=out/'bridge-local-verification.json';path.write_text(json.dumps(result,indent=2)+'\n')
            flag='bridge_verified.json'if result['passed']else'bridge_failed.json'
            dest=str(Path(a.remote_root)/'outputs/study'/flag)
            command='import sys;from pathlib import Path;p=Path('+repr(dest)+');p.write_bytes(sys.stdin.buffer.read())'
            subprocess.run(['ssh','-S',a.socket,'-o','BatchMode=yes',a.host,'python3','-c',__import__('shlex').quote(command)],input=path.read_bytes(),check=True)
            checked=True
            print(json.dumps(dict(bridge_passed=result['passed'])),flush=True)
            if not result['passed']:raise ValueError('Device bridge failed; grid admission stopped and all evidence retained')
    if proc.wait()!=0:raise RuntimeError('SSH mirror stopped; retained checkpoints can be resumed')
    if '_final'not in received:raise ValueError('Mirror did not receive completed final metadata')


if __name__=='__main__':main()

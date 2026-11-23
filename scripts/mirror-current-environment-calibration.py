"""Durably mirror this bounded P–K campaign; never delete remote raw data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

REMOTE='/dev/shm/oran_smo_selected_20261123/v1/source/outputs/pk_calibration_20261123_v2'
HOSTS=[('agx1','100.116.105.126','/tmp/oran-smo-agx2-20261119.sock'),
       ('agx2','100.72.45.61','/tmp/oran-smo-agx-fast-20261119.sock')]
ESSENTIAL=('case.json','stimulus.json','command.json','simulation.log',
    'run/config_snapshot.json','run/run_status.json','run/control_timeline.json',
    'run/final_summary.json','run/initial_conditions.json',
    'run/rt_sys_trace.jsonl.gz','run/control_receipts.jsonl.gz')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sync(out,host):
    name,address,socket=host;root=out/'workers'/name;root.mkdir(parents=True,exist_ok=True)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','ControlPath='+socket]
    result=subprocess.run([*ssh,'lboroldn@'+address,'cat '+REMOTE+'/status.json'],check=True,stdout=subprocess.PIPE,timeout=30)
    status=json.loads(result.stdout);atomic(root/'status.json',status)
    def transfer(case,essential):
        target=root/'jobs'/case
        target.mkdir(parents=True,exist_ok=True)
        includes=(['--include=/run/','--include=/seal.json']+
            ['--include=/'+name for name in ESSENTIAL]+['--exclude=*'])if essential else []
        subprocess.run(['rsync','-a','--partial','-e',' '.join(ssh),*includes,
            'lboroldn@'+address+':'+REMOTE+'/jobs/'+case+'/',str(target)+'/'],check=True,timeout=600)
        seal=json.loads((target/'seal.json').read_text())
        if not seal.get('completed'):raise ValueError('Unsealed raw evidence')
        names=list(ESSENTIAL) if essential else list(seal['files_sha256'])
        for name in names:
            pin=seal['files_sha256'].get(name)
            path=target/name
            if not path.resolve().is_relative_to(target.resolve()) or sha(path)!=pin:raise ValueError('Mirrored file differs: '+name)
        atomic(target/('analysis-ready.json'if essential else'mirror-verified.json'),dict(
            schema='oran.parameter-measurement-mirror.v1'if essential else'oran.full-raw-mirror.v1',
            verified=True,seal_sha256=sha(target/'seal.json'),file_count=len(names),
            files_sha256={n:seal['files_sha256'][n]for n in names},
            bytes=sum((target/name).stat().st_size for name in names),verified_unix_s=time.time(),
            full_raw_mirror=not essential,remote_raw_deleted=False))
    # Numerical evidence first. Slow PHY-debug transfer cannot delay fitting;
    # originals remain intact and full backup continues after host simulation.
    for case in status['done']:
        target=root/'jobs'/case
        if not any((target/n).exists()for n in('analysis-ready.json','mirror-verified.json')):
            transfer(case,True)
    if status['completed']:
        for case in status['done']:
            if not(root/'jobs'/case/'mirror-verified.json').exists():transfer(case,False)
    core=sum(any((root/'jobs'/c/n).exists()for n in('analysis-ready.json','mirror-verified.json'))for c in status['done'])
    full=sum((root/'jobs'/c/'mirror-verified.json').exists()for c in status['done'])
    return dict(worker=status['worker_id'],phase=status['phase'],done=len(status['done']),failed=status['failed'],
        queued=status['queued'],running=status['running'],completed=status['completed']and full==len(status['done']),
        physical_completed=status['completed'],core_mirrored=core,full_mirrored=full,free=status['free'],
        remote_updated_unix_s=status['updated_unix_s'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    deadline=time.time()+6*3600
    previous={};mutex=threading.Lock()
    def watch(host):
        worker=host[0]
        while time.time()<deadline:
            try:state=sync(a.output,host)
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                state={**previous.get(worker,{}),'worker':worker,'completed':False,'mirror_error':str(exc)}
            with mutex:
                previous[worker]=state
                atomic(a.output/'mirror-status.json',dict(updated_unix_s=time.time(),workers=list(previous.values())))
            if a.once or state['completed']:return
            time.sleep(30)
    threads=[threading.Thread(target=watch,args=(host,)) for host in HOSTS]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    if time.time()>=deadline:raise TimeoutError('Bounded six-hour mirroring horizon ended; retained data remain intact')


if __name__=='__main__':main()

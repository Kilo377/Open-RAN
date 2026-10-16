"""Detach only after a successful current-source response gate; never monitor."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.contracts import source_fingerprint
from scripts.validation.validate_rule_based import write


def main(a):
    out=a.output.resolve();gate=a.gate.resolve()
    if not out.is_relative_to(ROOT/'outputs') or not gate.is_relative_to(ROOT/'outputs'):raise ValueError('Artifacts must stay on Spark')
    receipt=out.parent/(out.name+'_exit.json');log=out.parent/(out.name+'.log')
    if a.detached_worker:
        start=time.time();code=1;error=None
        try:
            code=subprocess.call([sys.executable,str(ROOT/'scripts/validation/validate_xapp_library.py'),
                '--stage','long','--gate',str(gate),'--output',str(out)],cwd=ROOT,stdin=subprocess.DEVNULL)
        except BaseException as exc:error=f'{type(exc).__name__}: {exc}'
        finally:
            write(receipt,dict(process_finished=True,exit_code=code,error=error,started_epoch=start,finished_epoch=time.time(),
                interpretation='Process completion is distinct from the validation.json acceptance result and SHA256 verification.'))
        return
    status=json.loads((gate/'validation.json').read_text());estimate=json.loads((gate/'long_estimate.json').read_text())
    assert status['completed'] and status['passed'] and len(status['responses'])==56,'Response gate not complete/passed'
    assert status['source_sha256']==source_fingerprint(ROOT),'Gate source differs from current source'
    now=time.time();tz=ZoneInfo('Asia/Shanghai')
    result=dict(output=str(out),gate=str(gate),source_sha256=status['source_sha256'],estimate=estimate,
        estimated_finish_earliest=datetime.fromtimestamp(now+estimate['lower_s'],tz).isoformat(),
        estimated_finish_latest=datetime.fromtimestamp(now+estimate['upper_s'],tz).isoformat(),
        raw_data_host='Spark only',active_monitoring=False)
    if a.dry_run:print(json.dumps(result,indent=2));return
    launch=out.parent/(out.name+'_launch.json')
    if out.exists() or launch.exists():raise FileExistsError('Do not overwrite/relaunch an existing matrix')
    with log.open('wb') as stream:
        child=subprocess.Popen([sys.executable,__file__,'--gate',str(gate),'--output',str(out),'--detached-worker'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    result.update(pid=child.pid,launched_epoch=now,log=str(log),exit_receipt=str(receipt))
    write(launch,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--detached-worker',action='store_true')
    main(p.parse_args())

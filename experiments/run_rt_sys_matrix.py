"""Finite integration matrix, distinct from formal generalization evaluation."""
import argparse
from pathlib import Path
import sys
import subprocess
import json
ROOT=Path(__file__).resolve().parents[1]


def cases(slots=1000,update_ms=10):
    return [{'scene':scene,'cells':3,'ues':ues,'seed':seed,'slots':slots,'path_update_ms':update_ms}
            for scene in ('simple_street_canyon','munich') for ues in (12,24) for seed in (1,2,3)]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--slots',type=int,default=1000);p.add_argument('--path-update-ms',type=float,default=10)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--skip-completed',action='store_true')
    args=p.parse_args();plan=cases(args.slots,args.path_update_ms)
    if args.dry_run:print(json.dumps(plan,indent=2));sys.exit(0)
    root=ROOT/'outputs/rt_sys_matrix';root.mkdir(parents=True,exist_ok=True)
    if args.device=='cuda':
        sys.path.insert(0,str(ROOT))
        from scripts.spark.preflight import check
        (root/'preflight.json').write_text(json.dumps(check(),indent=2))
    (root/'run_plan.json').write_text(json.dumps(plan,indent=2))
    for case in plan:
        out=root/f'{case["scene"]}_u{case["ues"]}_s{case["seed"]}_dt{args.path_update_ms}'
        status=out/'run_status.json'
        if args.skip_completed and status.exists():
            saved=out/'config_snapshot.json'
            if saved.exists():
                m=json.loads(saved.read_text())['manifest']
                saved_status=json.loads(status.read_text())
                saved_device=json.loads(saved.read_text())['configs']['phy']['rt_sys']['device']
                if (saved_status.get('completed') and saved_status.get('stage')=='completed' and
                    saved_device==args.device and m['total_slots']==args.slots and m['path_update_ms']==args.path_update_ms):continue
        command=[sys.executable,str(ROOT/'experiments/run_rt_sys.py'),'--output',str(out),'--device',args.device]
        for key,value in case.items(): command.extend(['--'+key.replace('_','-'),str(value)])
        subprocess.run(command,cwd=ROOT,check=True)

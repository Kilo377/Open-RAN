"""One owned fixed-stimulus simulation at a time on an existing AGX worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--worker',required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());root=Path(plan['worker_source_root']);owned=a.plan.parent
    if json.loads((root/'.smo-owned-worker.json').read_text())['sha256']!=plan['worker_source_archive_sha256']:raise ValueError('Pinned worker source differs')
    status=owned/'status.json';entries=[e for e in plan['cases']if e['worker_id']==a.worker]
    summary=dict(worker=a.worker,phase='running',cases=[],simulation_rerun_scope='four_fixed_calibration_conditions_only')
    atomic(status,summary)
    for entry in entries:
        folder=owned/entry['name'];folder.mkdir(exist_ok=True);run=folder/'data/run'
        if (folder/'job.json').exists():raise ValueError('Refusing to overwrite a completed stimulus')
        spec=folder/'stimulus.json';atomic(spec,entry['stimulus'])
        args=['--owned-root',str(root),'--device','cuda','--rt-device','cpu','--scene','munich','--cells','3','--ues','6',
            '--config',str(root/'configs/rt_sys/agx_munich_6ue_depth2_fast.json'),'--algebra-backend','torch',
            '--nr-execution','batched','--recording-mode','experiment','--diagnostic-window-slots','40',
            '--model','fast','--fast-profile','system','--slots','1420','--seed',str(plan['seed']),
            '--traffic-load','1','--ue-services','drx-mec','--xapp-pool','chongyu',
            '--ap-services-profile',str(root/'configs/rt_sys/ap_services_reference.json'),'--record-compression','gzip',
            '--channel-cache-root',str(root/'channel_cache/smo_deployments'/str(plan['seed'])),
            '--control-timeline',str(spec),'--direct-conflict-mitigation','priority',
            '--live-state',str(run/'live.json'),'--output',str(run)]
        command=['bash','-c','source "$1"; shift; exec python "$@"','mec-calibration',
            str(root/'scripts/orin/activate-gpu.sh'),str(owned/'mec_response_stimulus.py'),*args]
        state=dict(name=entry['name'],phase='running',started_unix_s=time.time(),output=str(folder));summary['cases'].append(state)
        with(folder/'simulation.log').open('wb')as log:
            proc=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            state['pid']=proc.pid;atomic(status,summary)
            while proc.poll()is None:
                try:live=json.loads((run/'live.json').read_text());state['slot']=live.get('slot')
                except (OSError,ValueError):pass
                atomic(status,summary);time.sleep(1)
            code=proc.wait()
        state.update(phase='complete'if code==0 else'failed',exit_code=code,elapsed_s=time.time()-state['started_unix_s'])
        if code==0:
            # Verify an actual changed scheduler receipt at the intended slot.
            import gzip
            first=None
            with gzip.open(run/'control_receipts.jsonl.gz','rt')as stream:
                for line in stream:
                    row=json.loads(line)
                    if row.get('status')!='applied':continue
                    if any(before.get('scheduler_mode')!=row.get('after',{}).get(cid,{}).get('scheduler_mode')or
                        before.get('ue_scheduler_weights')!=row.get('after',{}).get(cid,{}).get('ue_scheduler_weights')
                        for cid,before in row.get('before',{}).items()):first=row.get('apply_slot');break
            if first!=561:raise ValueError('Fixed stimulus did not execute at matched slot 561')
            atomic(folder/'job.json',dict(id=entry['name'],phase='complete',seed=plan['seed'],observe_slots=500,app_ids=[],
                worker_id=a.worker,expected_source_archive_sha256=plan['worker_source_archive_sha256'],
                comparison=dict(application_slot=first),kind='fixed_scheduler_response_supplement',
                stimulus=entry['stimulus'],stimulus_driver_sha256=hashlib.sha256((owned/'mec_response_stimulus.py').read_bytes()).hexdigest()))
        atomic(status,summary)
        if code:summary['phase']='failed';atomic(status,summary);return
    summary['phase']='complete';atomic(status,summary)


if __name__=='__main__':main()

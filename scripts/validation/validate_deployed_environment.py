"""Check final reporting/HARQ fixes on GPU, and preserve numerical baseline replay."""
import argparse,csv,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from phy.rt_sys.contracts import source_fingerprint
from scripts.validation.validate_environment_repairs import audit

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    root=ROOT/'outputs/rt_environment_repairs_20261013/deployed_checks'
    root.mkdir(parents=True,exist_ok=True)
    fields=('slot_id','ue_id','queue_bytes','tb_bytes','delivered_bytes','ack','sinr_db','mcs_index','reference_tb_ack','packet_pending_bytes')
    def run(spec):
        scene,mode=spec;folder=root/f'{scene}_{mode}'
        if not args.audit_only:
            if folder.exists():raise ValueError(f'Refusing overwrite: {folder}')
            folder.mkdir()
        cmd=[sys.executable,str(ROOT/'experiments/run_rt_sys.py'),'--scene',scene,'--cells','3','--ues','4',
             '--seed','402','--slots','40','--control-mode',mode,'--selection-mode','fixed_plan','--output',str(folder)]
        if not args.audit_only:
            with (folder/'process.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status=json.loads((folder/'run_status.json').read_text())
        assert status['completed'] and status['spark_verified'] and status['source_sha256']==source_fingerprint(ROOT)
        if mode=='baseline':
            result=audit(folder)
            old=ROOT/'outputs/rt_environment_repairs_20261013/final_baseline'/f'{scene}_u4_load1.0_slots40'
            a=list(csv.DictReader((old/'ue_metrics.csv').open()));b=list(csv.DictReader((folder/'ue_metrics.csv').open()))
            assert len(a)==len(b)
            checked=[f for f in fields if f in a[0] and f in b[0]]
            for x,y in zip(a,b):
                for f in checked:assert x[f]==y[f],(scene,x.get('slot_id'),f,x[f],y[f])
            result.update(numerical_replay=True,replay_fields=checked,prior_source_sha256=json.loads((old/'run_status.json').read_text())['source_sha256'])
        else:
            config=json.loads((folder/'config_snapshot.json').read_text())
            assert config['configs']['ric']['xapps']
            result=dict(case=folder.name,passed=True,source_sha256=status['source_sha256'],control_mode=mode,
                scope='Existing controller execution smoke; no training or performance comparison')
        return result
    plan=[('simple_street_canyon','baseline'),('munich','baseline'),('simple_street_canyon','orchestrated')]
    with ThreadPoolExecutor(max_workers=4) as pool:cases=list(pool.map(run,plan))
    report=dict(passed=True,cases=cases,source_sha256=source_fingerprint(ROOT),
        excluded_diagnostics={'munich_orchestrated':'Partial extra controller stress check stopped; not accepted and not part of environment baseline'})
    (root/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

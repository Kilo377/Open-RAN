"""Finite stationary capacity matrix, with warmup, empty queues and no algorithms."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,subprocess,sys,csv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.contracts import source_fingerprint


def main():
    out=ROOT/'outputs/rt_research_readiness_20261013/final_ran/long_baselines_verified';out.mkdir(parents=True,exist_ok=True)
    plan=[(scene,n,load,slots) for scene in ('simple_street_canyon','munich')
          for n,load,slots in ((4,1.,2000),(24,.1,2000),(24,1.,2000))]
    def run(spec):
        scene,n,load,slots=spec;folder=out/f'{scene}_u{n}_load{load}_slots{slots}'
        if folder.exists():raise ValueError(f'Refusing overwrite {folder}')
        folder.mkdir()
        cmd=[sys.executable,str(ROOT/'experiments/run_rt_sys.py'),'--scene',scene,'--cells','3','--ues',str(n),
            '--slots',str(slots),'--seed','512','--initial-queue-bytes','0','--ue-speed-mps','0',
            '--warmup-ms','200','--traffic-load',str(load),'--feedback-mode','nr_bler_bank','--control-mode','baseline','--output',str(folder)]
        with (folder/'process.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status=json.loads((folder/'run_status.json').read_text());final=json.loads((folder/'final_summary.json').read_text())
        assert status['completed'] and status['spark_verified'] and status['source_sha256']==source_fingerprint(ROOT)
        steady=final['steady_state'];assert steady['evaluated_ms']==slots*.5-200
        assert all(m['initial_bytes']==0 for m in final['packet_metrics'].values())
        for uid,m in final['packet_metrics'].items():
            assert m['initial_bytes']+m['offered_bytes']==m['delivered_bytes']+m['overflow_bytes']+m['pending_bytes']
        assert final['link_reference_statistics']['validated_entries']>0
        for row in csv.DictReader((folder/'cell_metrics.csv').open()):
            assert float(row['average_used_w'])<=float(row['budget_w'])+1e-10
            assert int(row['occupied_re'])<=int(row['grid_re'])
        return dict(case=folder.name,passed=True,source_sha256=status['source_sha256'],elapsed_s=status['elapsed_s'],
            simulated_ms=slots*.5,warmup_ms=200,ue_count=n,stationary=True,load=load,
            never_scheduled=[uid for uid,m in steady['per_ue'].items() if m['admitted_bytes'] and m['scheduled_slots']==0],
            zero_delivery=[uid for uid,m in steady['per_ue'].items() if m['admitted_bytes'] and m['delivered_bytes']==0],
            diagnoses={uid:m['diagnosis'] for uid,m in steady['per_ue'].items()},
            pending_bytes=sum(m['pending_end'] for m in steady['per_ue'].values()),
            overflow_bytes=sum(m['overflow_bytes'] for m in steady['per_ue'].values()),
            deadline_misses=sum(m['deadline_misses'] for m in steady['per_ue'].values()),
            offered_mbps=sum(m['offered_mbps'] for m in steady['per_ue'].values()),
            delivered_mbps=sum(m['delivered_mbps'] for m in steady['per_ue'].values()),
            queue_growth_bytes_per_s=sum(m['queue_growth_bytes_per_s'] for m in steady['per_ue'].values()),
            link_statistics=final['link_reference_statistics'])
    with ThreadPoolExecutor(max_workers=3) as pool:cases=list(pool.map(run,plan))
    report=dict(passed=True,cases=cases,source_sha256=source_fingerprint(ROOT),
        scope='1 second stationary finite capacity validation, not moving-channel generalization or asymptotic stability',
        clocks_unchanged=True)
    (out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

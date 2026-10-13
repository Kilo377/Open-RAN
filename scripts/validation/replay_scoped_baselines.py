"""Replay saved baseline prefixes after a control-scope-only mobility fix."""
import argparse, json, os, shutil, sys, time
from pathlib import Path
from itertools import islice
from concurrent.futures import ThreadPoolExecutor
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.contracts import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--slots',type=int,default=40)
    args=parser.parse_args()
    out=ROOT/'outputs/rt_research_readiness_20261013/baseline_scoped_replay'
    out.mkdir(parents=True,exist_ok=True)
    plan=[(scene,n,load,total) for scene in ('simple_street_canyon','munich')
          for n,load,total in ((4,1.,4000),(24,.1,2000),(24,1.,2000))]
    fields=('handover_margin_db','handover_delay_slots','handover_cost_db','handover_interruption_slots',
            'steering_hysteresis_db','measurement_event_trigger','measurement_event_offset_db',
            'traffic_steering_preference','mobility_bias_db')
    def run(spec):
        scene,n,load,total=spec
        name=f'{scene}_u{n}_load{load}_slots{total}'
        original=args.baseline_root/name
        while not (original/'final_summary.json').exists():time.sleep(10)
        saved=json.loads((original/'config_snapshot.json').read_text())
        c=saved['configs'];manifest=saved['manifest']
        folder=out/name
        if folder.exists():raise ValueError(f'Refusing overwrite {folder}')
        folder.mkdir()
        bank=folder/'link_bler_bank.json';shutil.copy2(original/'link_bler_bank.json',bank)
        c['phy']['rt_sys']['reference']['probability_bank']['bank_file']=str(bank)
        from phy.rt_sys.channel import RtChannelProvider
        from phy.rt_sys.adapter import RtSysPhyAdapter
        from core.factory import build_controller_registry,build_scheduler
        from core.simulator import Simulator
        from core.debug import DebugConfig
        from experiments.scenario_scan_utils import build_runtime_scenario_config
        DebugConfig.SLOT_METRICS=False
        provider=RtChannelProvider(manifest,ROOT/'channel_cache',device='cuda')
        adapter=RtSysPhyAdapter(c['phy']['rt_sys'],channel_provider=provider)
        runtime=build_runtime_scenario_config(scenario_name='scope_replay_'+name,scenario_type='rt_scene',
            sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
        sim=Simulator(runtime,build_controller_registry(c['ric']),build_scheduler(c['sim']),adapter,
            c['cell'],c['ue'],c['beamforming'],str(folder))
        with (original/'rt_sys_trace.jsonl').open() as stream:
            old=[json.loads(line) for line in islice(stream,args.slots)]
        assert len(old)==args.slots
        import numpy as np
        with (folder/'trace.jsonl').open('w') as log:
            for expected in old:
                policies=[tuple(getattr(r.cell,f) for f in fields) for r in sim.state.managed_cells.values()]
                assert all(p==policies[0] for p in policies),'Baseline has unequal policies; old/global equivalence invalid'
                sim.step()
                observed={u['ue_id']:u for u in adapter.trace['ues']}
                for row in expected['ues']:
                    current=observed[row['ue_id']]
                    for field in ('cell_id','scheduled','allocated_re','data_re','layers','mcs','ack',
                                  'tb_bytes','delivered_bytes','queue_bytes','amc_selection_target_met'):
                        assert current[field]==row[field],(name,sim.state.slot_id,row['ue_id'],field)
                    for field in ('sinr_db','bler'):
                        np.testing.assert_allclose(current[field],row[field],rtol=1e-8,atol=1e-8)
                assert {uid:conn.handover_count for uid,conn in sim.state.connections.items()}==expected['handover_counts']
                log.write(json.dumps(adapter.trace)+'\n')
        report=dict(passed=True,case=name,prefix_slots=args.slots,source_sha256=source_fingerprint(ROOT),
                    baseline_source_sha256=saved['source_sha256'],same_channel_cache_key=provider.key==old[0]['cache_key'],
                    uniform_policy_checked_each_slot=True)
        assert report['same_channel_cache_key']
        (folder/'validation.json').write_text(json.dumps(report,indent=2))
        return report
    with ThreadPoolExecutor(max_workers=3) as pool:cases=list(pool.map(run,plan))
    report=dict(passed=True,source_sha256=source_fingerprint(ROOT),cases=cases,
                scope='Saved RT/channel/business baseline prefixes, identical resource/ACK/queue/SINR and handover outcomes; not a new full-duration run')
    (out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()

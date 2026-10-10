"""Dedicated, finite RT/SYS runner. Never falls back to an approximate PHY."""
import argparse
from pathlib import Path
import sys
import json
import time
import resource
import platform
from dataclasses import asdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phy.rt_sys.scenario import build_case


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=['simple_street_canyon','munich'],default='simple_street_canyon')
    p.add_argument('--cells',type=int,default=2); p.add_argument('--ues',type=int,default=4)
    p.add_argument('--seed',type=int,default=1); p.add_argument('--slots',type=int,default=40)
    p.add_argument('--path-update-ms',type=float)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--generate-only',action='store_true')
    return p.parse_args()


def run_case(args):
    manifest,configs,intent=build_case(args.scene,args.cells,args.ues,args.seed,args.slots,args.path_update_ms,args.device,args.config)
    name=f'{args.scene}_c{args.cells}_u{args.ues}_s{args.seed}_dt{manifest["path_update_ms"]}'
    output=(args.output or ROOT/'outputs/rt_sys'/name).resolve()
    if not output.is_relative_to(ROOT): raise ValueError('Outputs must stay in the independent copy')
    if args.dry_run:
        print(json.dumps({'manifest':manifest,'output':str(output),'phy_config':configs['phy']},indent=2)); return
    output.mkdir(parents=True,exist_ok=True)
    snapshot={'manifest':manifest,'configs':configs,'intent':intent,'validation':'local/remote integration; not a generalization result'}
    (output/'config_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    status={'completed':False,'spark_verified':False,'stage':'initializing'}
    def write_status(): (output/'run_status.json').write_text(json.dumps(status,indent=2))
    write_status(); started=time.perf_counter()
    try:
        from phy.rt_sys.channel import RtChannelProvider
        from phy.rt_sys.adapter import RtSysPhyAdapter
        provider=RtChannelProvider(manifest,ROOT/'channel_cache',device=args.device)
        if args.generate_only:
            provider.pregenerate(); status.update(completed=True,stage='channels_generated',cache_key=provider.key)
            return
        from core.factory import build_controller_registry,build_scheduler
        from core.simulator import Simulator
        from core.debug import DebugConfig
        from experiments.scenario_scan_utils import build_runtime_scenario_config
        from ric.non_rt.active_intent_manager import ActiveIntentManager
        from scripts.spark.preflight import environment_info
        environment=environment_info()
        adapter=RtSysPhyAdapter(configs['phy']['rt_sys'],channel_provider=provider)
        registry=build_controller_registry(configs['ric'])
        registry.active_intent_manager=ActiveIntentManager(initial_intent_text=intent)
        runtime=build_runtime_scenario_config(scenario_name=name,scenario_type='rt_scene',sim_config=configs['sim'],cell_config=configs['cell'],ric_config=configs['ric'])
        simulator=Simulator(runtime,registry,build_scheduler(configs['sim']),adapter,configs['cell'],configs['ue'],configs['beamforming'],str(output))
        DebugConfig.SLOT_METRICS=False
        status.update(stage='simulating',cache_key=provider.key); write_status()
        with (output/'rt_sys_trace.jsonl').open('w') as trace:
            import csv
            with (output/'ue_metrics.csv').open('w',newline='') as ufile, (output/'cell_metrics.csv').open('w',newline='') as cfile:
                uw=cw=None
                for _ in range(args.slots):
                    simulator.step()
                    trace.write(json.dumps(simulator.slot_records[-1]['rt_sys'],allow_nan=False)+'\n')
                    for row in adapter.trace['ues']:
                        row={'slot_id':simulator.state.slot_id,**row}
                        if uw is None: uw=csv.DictWriter(ufile,fieldnames=list(row)); uw.writeheader()
                        uw.writerow(row)
                    for cid,row in adapter.trace['cells'].items():
                        row={'slot_id':simulator.state.slot_id,'cell_id':cid,**row}
                        if cw is None: cw=csv.DictWriter(cfile,fieldnames=list(row)); cw.writeheader()
                        cw.writerow(row)
        summary=simulator._build_final_summary()
        summary['rt_sys']={'environment':environment,'cache_key':provider.key,'ideal_csi':True,
            'harq_soft_combining':False,'full_phy_calibrated':False,'cache_hits':provider.hits,'cache_misses':provider.misses}
        simulator.result_writer.write_slot_metrics(simulator.slot_records)
        simulator.result_writer.write_final_summary(summary)
        simulator.result_writer.write_orchestration_epoch_history(simulator.controller_manager.get_orchestration_epoch_history())
        status.update(completed=True,stage='completed',runtime_environment=environment,
            spark_verified=args.device=='cuda' and environment['system']=='Linux' and environment['machine'] in ('aarch64','arm64') and environment['cuda_available'] and
            environment['rt_variant'].startswith('cuda') and 'GB10' in environment['gpu_name'])
    except Exception as exc:
        status.update(stage='failed',error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        status.update(elapsed_s=time.perf_counter()-started,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        status['peak_rss_bytes']=status['peak_rss_native'] if platform.system()=='Darwin' else status['peak_rss_native']*1024
        try:
            import torch
            status['peak_cuda_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        except ImportError: pass
        write_status()


if __name__=='__main__': run_case(parse_args())

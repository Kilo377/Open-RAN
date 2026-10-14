"""Actual moving RT channels and sleep-triggered handover, with measured inputs.

A fixed test xApp sleeps then wakes the initial serving cell. This forces the
existing event/interrupt/association lifecycle without claiming natural mobility
performance or adding new handover actions.
"""
import argparse,json,sys,os
from pathlib import Path
from dataclasses import asdict
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.simulator import Simulator
from core.factory import build_scheduler
from experiments.scenario_scan_utils import build_runtime_scenario_config
from phy.rt_sys.scenario import build_case
from phy.rt_sys.channel import RtChannelProvider
from phy.rt_sys.adapter import RtSysPhyAdapter
from phy.rt_sys.contracts import source_fingerprint
from ric.controller_registry import ControllerRegistry
from ric.near_rt.xapp_base import XAppBase
from models.action import ConfigAction,ControlDecision


class SleepTrigger(XAppBase):
    def __init__(self):super().__init__('sleep_handover_fixture',20,['cell_sleep_mode']);self.target=None;self.inputs=[]
    def run(self,observation,policy_snapshot):
        self.inputs.append(dict(slot=observation.slot_id,metadata=observation.kpi_records))
        mode='deep_sleep' if 40<=observation.slot_id<120 else 'active'
        return ControlDecision(config_action=ConfigAction(target_cell_id=self.target,
            cell_sleep_mode=mode,source_name=self.name))


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);sha=source_fingerprint(ROOT);reports=[]
    for scene in ('simple_street_canyon','munich'):
        folder=out/scene;folder.mkdir()
        manifest,c,_=build_case(scene,num_cells=3,num_ues=4,total_slots=256,
            initial_queue_bytes=1000,ue_speed_mps=1.5,seed=512)
        c['ric'].update(rapps=[],xapps=[])
        provider=RtChannelProvider(manifest,ROOT/'channel_cache',device='cuda')
        adapter=RtSysPhyAdapter(c['phy']['rt_sys'],channel_provider=provider)
        registry=ControllerRegistry();app=SleepTrigger();registry.register_xapp(app)
        runtime=build_runtime_scenario_config(scenario_name='moving_handover_'+scene,scenario_type='rt_scene',
            sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
        sim=Simulator(runtime,registry,build_scheduler(c['sim']),adapter,c['cell'],c['ue'],c['beamforming'],str(folder))
        uid=sim.state.ues[0].ue_id;app.target=sim.state.connections[uid].serving_cell_id
        initial=adapter.current_snapshot.positions.copy();changes=[];flushes=[];count=0;measurement_ages=[]
        (folder/'config_snapshot.json').write_text(json.dumps(dict(manifest=manifest,configs=c,source_sha256=sha),indent=2))
        with (folder/'trace.jsonl').open('w') as stream:
            for _ in range(256):
                sim.step()
                for u in adapter.trace['ues']:
                    assert u['cell_id']==sim.state.connections[u['ue_id']].serving_cell_id
                    assert not u['mac_capacity_guard'] and u['ack_source']!='harq_incompatible_no_tb'
                    count+=u['actual_decode']
                changes.extend(sim.mobility_manager.events)
                for user in sim.state.ues:
                    user.packet_ledger.refresh(sim.state.slot_id)
                    flushes.extend(dict(ue_id=user.ue_id,**e) for e in user.harq.events if e['kind']=='payload_flush')
                    user.harq.events.clear();user.packet_ledger.drain_events()
                for item in sim.state.control.input_observation_trace.get('ue_observations',[]):
                    if item.get('measurement_age_slots') is not None:measurement_ages.append(item['measurement_age_slots'])
                stream.write(json.dumps(dict(slot=sim.state.slot_id,phy=adapter.trace,events=sim.mobility_manager.events,
                    connections={k:v.serving_cell_id for k,v in sim.state.connections.items()},
                    arbitration=asdict(sim.state.control.arbitration_trace)),default=str)+'\n')
        import numpy as np
        displacement=np.linalg.norm(adapter.current_snapshot.positions-initial,axis=1)
        completed=[e for e in changes if e['handover_completed']]
        assert completed and any(e['from_cell']==app.target for e in completed)
        assert min(displacement)>.05
        assert adapter.abstraction.decode_trials==count
        assert adapter.abstraction.statistics['amc_probe_trials']==adapter.abstraction.statistics['bler_probe_trials']==0
        assert adapter.abstraction.harq.stats['incompatible_no_tb']==0
        inputs=[r for batch in app.inputs for r in batch['metadata'] if r['entity']=='ue' and r['name']=='reference_sinr_db']
        assert inputs and all(r['valid'] and r['delivered_at_slot']>=r['produced_at_slot'] for r in inputs)
        assert any(r['age_slots']>0 for r in inputs)
        report=dict(passed=True,scene=scene,source_sha256=sha,displacement_m=displacement.tolist(),
            handovers=completed,harq_flushes=flushes,actual_decode_attempts=count,
            kpi_max_age_slots=max(r['age_slots'] for r in inputs),cache_key=provider.key,
            packet_metrics={u.ue_id:u.packet_ledger.summary() for u in sim.state.ues},scope=__doc__)
        (folder/'validation.json').write_text(json.dumps(report,indent=2));reports.append(report)
    assert source_fingerprint(ROOT)==sha
    (out/'validation.json').write_text(json.dumps(dict(passed=True,cases=reports,source_sha256=sha),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

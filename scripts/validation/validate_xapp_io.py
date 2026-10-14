"""Fixed test policy: measured KPI projection -> supported fields -> actual consumers."""
import argparse,json,sys
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.scenario import build_case
from phy.rt_sys.channel import RtChannelProvider
from phy.rt_sys.adapter import RtSysPhyAdapter
from phy.rt_sys.contracts import source_fingerprint
from core.factory import build_scheduler
from core.simulator import Simulator
from ric.controller_registry import ControllerRegistry
from ric.near_rt.xapp_base import XAppBase
from models.action import ConfigAction,ControlDecision
from experiments.scenario_scan_utils import build_runtime_scenario_config


class FixedPolicy(XAppBase):
    def __init__(self):
        super().__init__('fixed_io',20,['power_cap_dbm','codebook_id','beam_profile_id','cell_sleep_mode',
            'scheduler_policy_hint','slice_weight_adjustment','latency_bias','mobility_bias_db','steering_hysteresis_db',
            'measurement_event_offset_db','traffic_steering_preference'])
        self.inputs=[]

    def run(self,observation,policy_snapshot):
        assert observation.rt_public_projection
        assert not hasattr(observation,'scheduler_true_backlog_bytes_total')
        assert all(not hasattr(u,'cqi') and not hasattr(u,'demand_true_backlog_bytes') for u in observation.ue_observations)
        self.inputs.append(dict(slot=observation.slot_id,kpis=observation.kpi_records))
        latency=observation.slot_id==10
        return ControlDecision(config_action=ConfigAction(target_cell_id='cell_1',source_name=self.name,
            power_cap_dbm=27.,codebook_id=1,beam_profile_id=1,
            cell_sleep_mode='light_sleep' if observation.slot_id==15 else 'active',
            scheduler_policy_hint='latency_bias' if latency else 'slice_bias',latency_bias=1.5 if latency else None,
            slice_weight_adjustment={} if latency else {'urllc':1.5,'embb':.7},
            mobility_bias_db=4.,steering_hysteresis_db=1.5,measurement_event_offset_db=2.,
            traffic_steering_preference='balanced'))


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);sha=source_fingerprint(ROOT)
    reports=[]
    for scene in ('simple_street_canyon','munich'):
        folder=out/scene;folder.mkdir()
        manifest,c,_=build_case(scene,num_cells=2,num_ues=4,total_slots=25,initial_queue_bytes=1000,ue_speed_mps=0)
        c['ric'].update(rapps=[],xapps=[])
        provider=RtChannelProvider(manifest,ROOT/'channel_cache',device='cuda')
        adapter=RtSysPhyAdapter(c['phy']['rt_sys'],channel_provider=provider)
        registry=ControllerRegistry();app=FixedPolicy();registry.register_xapp(app)
        runtime=build_runtime_scenario_config(scenario_name='fixed_xapp_io_'+scene,scenario_type='rt_scene',
            sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
        sim=Simulator(runtime,registry,build_scheduler(c['sim']),adapter,c['cell'],c['ue'],c['beamforming'],str(folder))
        states=[];outcomes=[]
        with (folder/'trace.jsonl').open('w') as log:
            for _ in range(25):
                sim.step();cell=sim.state.managed_cells['cell_1'];r=cell.cell;s=cell.scheduler
                state=dict(slot=sim.state.slot_id,power_cap=r.power_cap_dbm,codebook=r.current_codebook_id,
                    beam=r.current_beam_profile_id,sleep=r.cell_sleep_mode,weights=s.slice_weight_adjustment,
                    hint=s.scheduler_policy_hint,latency=s.latency_bias,mobility=r.mobility_bias_db,
                    hysteresis=r.steering_hysteresis_db,event_offset=r.measurement_event_offset_db)
                states.append(state);outcomes.extend(sim.state.control.actuation_outcomes)
                log.write(json.dumps(dict(state=state,phy=adapter.trace,
                    arbitration=asdict(sim.state.control.arbitration_trace)),default=str)+'\n')
        assert app.inputs and len(app.inputs)==5
        assert any(s['hint']=='latency_bias' and s['latency']==1.5 for s in states)
        assert any(s['sleep']=='light_sleep' for s in states)
        final=states[-1]
        assert final['power_cap']==27 and final['codebook']==final['beam']==1
        assert final['weights']=={'urllc':1.5,'embb':.7} and final['sleep']=='active'
        assert final['mobility']==4 and final['hysteresis']==1.5 and final['event_offset']==2
        assert adapter.abstraction.statistics['amc_probe_trials']==adapter.abstraction.statistics['bler_probe_trials']==0
        (folder/'kpi_inputs.json').write_text(json.dumps(app.inputs,indent=2))
        report=dict(passed=True,scene=scene,states=states,outcomes=outcomes,source_sha256=sha,
            clocks=dict(xapp_slots=sim.controller_manager.xapp_interval_slots,rapp_slots=sim.controller_manager.rapp_interval_slots),
            scope=__doc__)
        (folder/'validation.json').write_text(json.dumps(report,indent=2));reports.append(report)
    assert source_fingerprint(ROOT)==sha
    (out/'validation.json').write_text(json.dumps(dict(passed=True,cases=reports,source_sha256=sha),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

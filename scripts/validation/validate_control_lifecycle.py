"""A1 policy -> owned xApp writes -> arbitration -> delayed scoped RT actuation."""
from pathlib import Path
import argparse
import sys,json
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.contracts import source_fingerprint
from phy.rt_sys.scenario import build_case
from phy.rt_sys.channel import RtChannelProvider
from phy.rt_sys.adapter import RtSysPhyAdapter
from core.factory import build_scheduler
from core.simulator import Simulator
from experiments.scenario_scan_utils import build_runtime_scenario_config
from ric.controller_registry import ControllerRegistry
from ric.near_rt.xapp_base import XAppBase
from ric.non_rt.rapp_base import RAppBase
from models.action import ConfigAction,ControlDecision
from models.policy import PolicyUpdate


class PolicyProbe(RAppBase):
    def __init__(self):super().__init__('policy_probe');self.calls=0
    def run(self,observation,policy_snapshot=None):
        self.calls+=1
        return PolicyUpdate(preferred_scheduler_policy_hint='slice_bias',
            preferred_slice_weight_adjustment={'urllc':2.,'embb':.5},source_name=self.name)


class PowerProbe(XAppBase):
    def __init__(self,name,priority,power):
        super().__init__(name,priority,['power_cap_dbm']);self.power=power
    def run(self,observation,policy_snapshot):
        return ControlDecision(config_action=ConfigAction(target_cell_id='cell_1',
            power_cap_dbm=self.power,effective_from_slot=15,effective_until_slot=20,source_name=self.name))


class SchedulerProbe(XAppBase):
    def __init__(self):super().__init__('policy_consumer',15,['scheduler_policy_hint','slice_weight_adjustment',
        'mobility_bias_db','steering_hysteresis_db','measurement_event_offset_db'])
    def run(self,observation,policy_snapshot):
        return ControlDecision(config_action=ConfigAction(target_cell_id='cell_0',source_name=self.name,
            scheduler_policy_hint=policy_snapshot.preferred_scheduler_policy_hint,
            slice_weight_adjustment=dict(policy_snapshot.preferred_slice_weight_adjustment),
            mobility_bias_db=4.,steering_hysteresis_db=1.5,measurement_event_offset_db=2.))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slots',type=int,default=25)
    parser.add_argument('--feedback-mode',default='nr_direct')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/rt_research_readiness_20261013/final_ran/control_lifecycle_verified')
    args=parser.parse_args()
    if args.slots<25:raise ValueError('Control lifecycle requires at least 25 slots')
    out=args.output.resolve()
    if not out.is_relative_to(ROOT/'outputs'):raise ValueError('Outputs must remain inside copy outputs')
    out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for scene in ('simple_street_canyon','munich'):
        folder=out/scene
        if folder.exists():raise ValueError('Refusing overwrite saved controls')
        folder.mkdir()
        m,c,_=build_case(scene,num_cells=3,num_ues=4,total_slots=args.slots,initial_queue_bytes=1000,ue_speed_mps=0)
        c['ric'].update(rapps=[],xapps=[],run_rapps_on_start=False)
        c['phy']['rt_sys']['feedback_mode']=args.feedback_mode
        provider=RtChannelProvider(m,ROOT/'channel_cache',device='cuda')
        adapter=RtSysPhyAdapter(c['phy']['rt_sys'],channel_provider=provider)
        registry=ControllerRegistry();rapp=PolicyProbe();registry.register_rapp(rapp)
        registry.register_xapp(PowerProbe('low_power_priority',1,25))
        registry.register_xapp(PowerProbe('high_power_priority',20,20))
        registry.register_xapp(SchedulerProbe())
        runtime=build_runtime_scenario_config(scenario_name='lifecycle_'+scene,scenario_type='rt_scene',
            sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
        sim=Simulator(runtime,registry,build_scheduler(c['sim']),adapter,c['cell'],c['ue'],c['beamforming'],str(folder))
        assert sim.controller_manager.xapp_interval_slots==5 and sim.controller_manager.rapp_interval_slots==10
        outcomes=[];dropped=[];states=[]
        with (folder/'trace.jsonl').open('w') as log:
            for _ in range(args.slots):
                sim.step();outcomes.extend(sim.state.control.actuation_outcomes)
                arbitration=sim.state.control.arbitration_trace
                if arbitration is not None:dropped.extend(asdict(x) for x in arbitration.dropped_items_with_reason)
                cells=sim.state.managed_cells
                states.append(dict(slot=sim.state.slot_id,power0=cells['cell_0'].cell.tx_power_dbm,
                    power1=cells['cell_1'].cell.tx_power_dbm,scheduler=cells['cell_0'].scheduler.scheduler_mode,
                    weights=cells['cell_0'].scheduler.slice_weight_adjustment,
                    mobility_bias=cells['cell_0'].cell.mobility_bias_db,
                    hysteresis=cells['cell_0'].cell.steering_hysteresis_db,
                    event_offset=cells['cell_0'].cell.measurement_event_offset_db))
                log.write(json.dumps(dict(slot=sim.state.slot_id,phy=adapter.trace,
                    outcomes=sim.state.control.actuation_outcomes,arbitration=asdict(arbitration)),default=str)+'\n')
        assert rapp.calls==1+args.slots//10,rapp.calls
        assert all(s['power1']==30 for s in states if s['slot']<15)
        assert states[14]['power1']==20 and all(s['power0']==30 for s in states)
        assert states[-1]['weights']=={'urllc':2.,'embb':.5} and states[-1]['scheduler']=='slice_aware'
        assert states[-1]['mobility_bias']==4. and states[-1]['hysteresis']==1.5 and states[-1]['event_offset']==2.
        assert any(x['source']=='low_power_priority' and x['reason']=='superseded_by_higher_priority' for x in dropped),dropped
        assert any(x.get('reason')=='expired_before_actuation' for x in outcomes),outcomes
        report=dict(scene=scene,passed=True,source_sha256=source_fingerprint(ROOT),policy_updates=rapp.calls,
            total_slots=args.slots,simulated_ms=args.slots*.5,
            clocks={'xapp_interval_slots':5,'rapp_interval_slots':10},states=states,outcomes=outcomes,dropped=dropped,
            scope='Actual RT/SYS GPU with selected NR feedback mode; verifies policy, arbitration and actuation, not algorithm performance')
        (folder/'validation.json').write_text(json.dumps(report,indent=2));reports.append(report)
    (out/'validation.json').write_text(json.dumps(dict(passed=True,cases=reports,source_sha256=source_fingerprint(ROOT)),indent=2))
    print(json.dumps(dict(passed=True,scenes=[r['scene'] for r in reports])))
if __name__=='__main__':main()

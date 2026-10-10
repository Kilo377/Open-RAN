"""Real GPU physical responses and delayed ConfigAction closed-loop probes.

Scripted controls isolate causality; these are not algorithm comparisons.
All variants reuse deterministic propagation and offered traffic.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def physical(scene,device):
    import numpy as np
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from phy.nr_reference import lookup_nr_max_prbs
    from models.phy_input import PhyInput,PhyUeInput
    manifest,configs,_=build_case(scene,total_slots=2,device=device)
    provider=RtChannelProvider(manifest,ROOT/'channel_cache',device=device)
    snap=provider.snapshot(1)
    nprb,_=lookup_nr_max_prbs(bandwidth_hz=manifest['bandwidth_hz'],
        subcarrier_spacing_hz=manifest['subcarrier_spacing_hz'],carrier_frequency_hz=manifest['carrier_frequency_hz'])
    ctx=SimpleNamespace(**{k:manifest[k] for k in ('carrier_frequency_hz','bandwidth_hz','subcarrier_spacing_hz','slot_duration_ms')},
        n_prbs_total=nprb,num_ofdm_symbols_per_slot=manifest['num_symbols'],cyclic_prefix_type='normal')
    inputs=[]
    for i,cid in enumerate(snap.cell_ids):
        u=PhyUeInput(f'ue_{i+2}',1000,0,0,0,0,True,'embb','data',100,1,
            allocated_prbs=2,allocated_symbols=manifest['num_symbols'],
            allocated_subcarrier_indices=list(range(24)),allocated_symbol_indices=list(range(manifest['num_symbols'])),
            layer_count=1,requested_bytes=100,grant_bytes_budget=100,target_mcs=3,mcs_table_id='qam64')
        inputs.append(PhyInput(1,manifest['slot_duration_ms'],cid,20,0,0,ue_inputs=[u],runtime_context=ctx))
    adapter=RtSysPhyAdapter(configs['phy']['rt_sys'],channel_provider=provider)
    def execute(plan,channel=snap):
        result=adapter.run_cluster_slot(plan,channel)
        return {'raw_sinr_db':{cid:r.ue_results[0].sinr_db for cid,r in result.items()},
            'trace':deepcopy(adapter.trace)}
    baseline=execute(inputs)
    changed=deepcopy(inputs);changed[1].tx_power_dbm=30
    neighbor_high=execute(changed)
    changed=deepcopy(inputs);changed[1].ue_inputs[0].is_scheduled=False
    silent=execute(changed)
    changed=deepcopy(inputs);changed[1].ue_inputs[0].allocated_subcarrier_indices=list(range(24,48))
    separate=execute(changed)
    solo=deepcopy(inputs);solo[1].ue_inputs[0].is_scheduled=False
    low=execute(solo);solo[0].tx_power_dbm=30;high=execute(solo)
    changed=deepcopy(inputs);changed[0].current_codebook_id=1;changed[0].current_beam_profile_id=1
    beam=execute(changed)
    sleeping=deepcopy(snap);sleeping.metadata['sleep_modes']={'cell_1':'deep_sleep'}
    sleep=execute(inputs,sleeping)
    disconnected=deepcopy(snap);disconnected.cfr[:]=0
    no_path=execute(inputs,disconnected)
    c0,c1=snap.cell_ids
    inter=lambda r:r['trace']['ues'][0]['inter_cell_interference_dbm']
    own_delta=high['raw_sinr_db'][c0]-low['raw_sinr_db'][c0]
    checks={'neighbor_power_interference_10db':abs(inter(neighbor_high)-inter(baseline)-10)<.01,
        'neighbor_silence_reduces_interference':inter(silent)<inter(baseline)-1,
        'resource_separation_matches_silence':abs(separate['raw_sinr_db'][c0]-silent['raw_sinr_db'][c0])<.01,
        'own_power_sinr_10db':abs(own_delta-10)<.01,
        'beam_changes_physical_sinr':abs(beam['raw_sinr_db'][c0]-baseline['raw_sinr_db'][c0])>.01,
        'sleep_zero_power_and_delivery':sleep['trace']['cells'][c1]['average_used_w']==0 and sleep['trace']['ues'][1]['delivered_bytes']==0,
        'no_path_zero_delivery':all(r['delivered_bytes']==0 for r in no_path['trace']['ues'])}
    assert all(checks.values()),checks
    return {'cache_key':provider.key,'checks':checks,'own_power_delta_db':own_delta,
        'variants':dict(baseline=baseline,neighbor_high=neighbor_high,silent=silent,separate=separate,
            own_low=low,own_high=high,beam=beam,sleep=sleep,no_path=no_path)}


def closed_loop(device,slots=40):
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from core.factory import build_controller_registry,build_scheduler
    from core.simulator import Simulator
    from core.debug import DebugConfig
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    from models.action import ConfigAction,ControlDecision,ControlMetadata
    actions={'baseline':{},'power':{'power_cap_dbm':20},'beam':{'codebook_id':1,'beam_profile_id':1},
        'sleep':{'cell_sleep_mode':'deep_sleep'},
        'scheduler':{'scheduler_policy_hint':'latency_bias','latency_bias':2}}
    variants={}
    for name,fields in actions.items():
        manifest,configs,_=build_case(num_cells=3,num_ues=12,total_slots=slots,device=device)
        configs['ric'].update(rapps=[],xapps=[])
        provider=RtChannelProvider(manifest,ROOT/'channel_cache',device=device)
        adapter=RtSysPhyAdapter(configs['phy']['rt_sys'],channel_provider=provider)
        runtime=build_runtime_scenario_config(scenario_name='control_'+name,scenario_type='rt_scene',
            sim_config=configs['sim'],cell_config=configs['cell'],ric_config=configs['ric'])
        sim=Simulator(runtime,build_controller_registry(configs['ric']),build_scheduler(configs['sim']),adapter,
            configs['cell'],configs['ue'],configs['beamforming'],str(ROOT/'outputs/rt_sys_acceptance/control'/name))
        observations=[]
        def update(slot_id,observation):
            observations.append(dict(slot=slot_id,produced_at_slot=observation.produced_at_slot,
                delivered_at_slot=observation.delivered_at_slot))
            # Issue at slot 2; original one-slot control delay applies at 3.
            return ControlDecision(config_actions=[ConfigAction(target_cell_id='cell_0',source_name='acceptance_probe',
                metadata=ControlMetadata(source='acceptance_probe',issued_at_slot=slot_id),
                **(fields if slot_id==2 else {}))])
        sim.controller_manager.update=update
        DebugConfig.SLOT_METRICS=False
        traces=[];arrivals=[];states=[]
        for _ in range(slots):
            sim.step();traces.append(deepcopy(adapter.trace))
            arrivals.append(dict(sim.state.traffic.arrival_bytes_by_ue))
            byuid={row['ue_id']:row for row in adapter.trace['ues']}
            for u in sim.state.ues:
                row=byuid[u.ue_id]
                assert u.queue_bytes==row['queue_bytes']-row['delivered_bytes'],(name,sim.state.slot_id,u.ue_id)
                assert u.harq.pending_retx_bytes==sum(max(0,p.pending_bytes) for p in u.harq.processes)
            states.append({'slot':sim.state.slot_id,'cell_power_dbm':sim.state.managed_cells['cell_0'].cell.tx_power_dbm,
                'scheduler_mode':sim.state.managed_cells['cell_0'].scheduler.scheduler_mode,
                'control_observation':dict(sim.state.control.input_observation_trace)})
        assert all(o['produced_at_slot']<o['slot'] for o in observations[5:])
        variants[name]={'cache_key':provider.key,'arrivals':arrivals,'observations':observations,'states':states,
            'traces':traces,'delivered_bytes':sum(u['delivered_bytes'] for t in traces for u in t['ues'])}
    progress=ROOT/'outputs/rt_sys_acceptance/control_probe_details.json'
    progress.write_text(json.dumps(variants,allow_nan=False))
    baseline=variants['baseline']
    assert all(v['cache_key']==baseline['cache_key'] and v['arrivals']==baseline['arrivals'] for v in variants.values())
    assert variants['power']['traces'][1]['cells']['cell_0']['budget_w']==1
    assert abs(variants['power']['traces'][2]['cells']['cell_0']['budget_w']-.1)<1e-9
    assert all(t['cells']['cell_0']['average_used_w']==0 for t in variants['sleep']['traces'][2:])
    assert any(u['beam_subspace_codebook']==1 for u in variants['beam']['traces'][2]['ues'] if u['cell_id']=='cell_0')
    assert variants['scheduler']['states'][2]['scheduler_mode']=='latency_aware'
    def grants(v):return [[(u['ue_id'],u['allocated_re'],u['layers']) for u in t['ues']] for t in v['traces']]
    assert grants(variants['scheduler'])!=grants(baseline)
    return {'checks':{'same_propagation_and_arrivals':True,'delayed_controls':True,
        'delayed_controller_observations':True,'queue_and_harq_consistency':True,
        'power_beam_sleep_scheduler_responses':True},'variants':variants}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    report={'device':args.device,'scripted_controls':True,
        'physical':{scene:physical(scene,args.device) for scene in ('simple_street_canyon','munich')},
        'closed_loop':closed_loop(args.device),'completed':True}
    target=ROOT/'outputs/rt_sys_acceptance/control_validation.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'completed':True,'device':args.device,
        'physical_checks':{k:v['checks'] for k,v in report['physical'].items()},
        'closed_loop_checks':report['closed_loop']['checks']},indent=2))

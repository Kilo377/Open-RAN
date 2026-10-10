"""Actual RT + Torch + SYS execution with deliberately overlapping MU grants.

This probes the physical interface independently of whether the MAC chooses
MU groups in a short run. Orthogonality/gain claims belong to analytic tests.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate(scene,device):
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.channel import SionnaRayTracer
    from phy.rt_sys.adapter import RtSysPhyAdapter
    import numpy as np
    from models.phy_input import PhyInput,PhyUeInput
    from phy.nr_reference import lookup_nr_max_prbs
    manifest,configs,_=build_case(scene,total_slots=2,device=device)
    provider=RtChannelProvider(manifest,ROOT/'channel_cache',device=device)
    channel=provider.snapshot(1)
    fresh=SionnaRayTracer(manifest,device)
    fresh.validate_geometry()
    recomputed=fresh.generate(0,3,provider.update_slots)['cfr'][1]
    np.testing.assert_allclose(channel.cfr,recomputed,rtol=1e-4,atol=1e-10)
    replay={'rtol':1e-4,'atol':1e-10,
        'max_abs_error':float(np.max(abs(channel.cfr-recomputed))),
        'relative_l2_error':float(np.linalg.norm(channel.cfr-recomputed)/max(np.linalg.norm(channel.cfr),1e-30))}
    adapter=RtSysPhyAdapter(configs['phy']['rt_sys'],channel_provider=provider)
    nprb,_=lookup_nr_max_prbs(bandwidth_hz=manifest['bandwidth_hz'],
        subcarrier_spacing_hz=manifest['subcarrier_spacing_hz'],carrier_frequency_hz=manifest['carrier_frequency_hz'])
    inputs=[]
    for ci,cid in enumerate(channel.cell_ids):
        users=[]
        for i in (ci,ci+2):
            users.append(PhyUeInput(ue_id=f'ue_{i}',queue_bytes=1000,sinr_db=0,
                radio_quality_db=0,bler=0,serving_beam_id=i,is_scheduled=True,
                slice_id='embb',traffic_type='data',latency_budget_ms=100,priority=1,
                allocated_prbs=2,allocated_symbols=manifest['num_symbols'],
                allocated_subcarrier_indices=list(range(24)),
                allocated_symbol_indices=list(range(manifest['num_symbols'])),
                layer_count=1,spatial_group_id=ci,spatial_group_size=2,
                requested_bytes=100,grant_bytes_budget=100,target_mcs=3,mcs_table_id='qam64'))
        context=SimpleNamespace(**{k:manifest[k] for k in ('carrier_frequency_hz','bandwidth_hz','subcarrier_spacing_hz','slot_duration_ms')},
            n_prbs_total=nprb,num_ofdm_symbols_per_slot=manifest['num_symbols'],cyclic_prefix_type='normal')
        inputs.append(PhyInput(1,manifest['slot_duration_ms'],cid,30,0,0,
            ue_inputs=users,runtime_context=context))
    results=adapter.run_cluster_slot(inputs,channel)
    assert all(len(result.ue_results)==2 for result in results.values())
    assert all(len(group['streams'])==2 for group in adapter.trace['precoders'])
    assert all(abs(cell['average_used_w']-cell['budget_w'])<1e-6 for cell in adapter.trace['cells'].values())
    assert all(0<=row['delivered_bytes']<=100 for row in adapter.trace['ues'])
    output=ROOT/'outputs/rt_sys'/f'mu_validation_{scene}_{device}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'actual_rt_sys':True,'device':device,'fixture_grants':True,
        'cache_recompute':replay,'trace':adapter.trace}
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    return {'scene':scene,'device':device,'cache_key':provider.key,
        'scheduled_ues':len(adapter.trace['ues']),'mu_resource_groups':len(adapter.trace['precoders']),
        'delivered_bytes':sum(row['delivered_bytes'] for row in adapter.trace['ues'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    print(json.dumps([validate(scene,args.device) for scene in ('simple_street_canyon','munich')],indent=2))

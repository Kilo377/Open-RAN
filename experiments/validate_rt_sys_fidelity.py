"""Real RT/SYS GPU probes for seeded CSI errors and guarded Chase combining."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def validate(scene):
    import numpy as np
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from phy.rt_sys.abstraction import SionnaAbstraction
    from models.phy_input import PhyInput,PhyUeInput
    m,c,_=build_case(scene,num_cells=1,num_ues=1,total_slots=2)
    snap=RtChannelProvider(m,ROOT/'channel_cache').snapshot(1)
    user=PhyUeInput('ue_0',1000,0,0,0,0,True,'embb','data',100,1,
        allocated_prbs=2,allocated_symbols=14,allocated_subcarrier_indices=list(range(24)),
        allocated_symbol_indices=list(range(14)),layer_count=1,requested_bytes=100,grant_bytes_budget=100,
        target_mcs=3,mcs_table_id='qam64',harq_process_id=0,harq_redundancy_version=0,harq_ndi=0)
    inputs=[PhyInput(1,.5,'cell_0',30,0,0,ue_inputs=[user])]
    class ProbeAbstraction:
        def __init__(self):self.model=SionnaAbstraction('cuda');self.calls=[];self.force_nack=True
        def evaluate(self,values,*args,**kwargs):
            self.calls.append(values.copy());result=self.model.evaluate(values,*args,**kwargs)
            if self.force_nack:result.update(ack=False)
            return result
    model=ProbeAbstraction()
    adapter=RtSysPhyAdapter({**c['phy']['rt_sys'],'harq_combining':'chase'},abstraction=model)
    adapter.run_cluster_slot(inputs,snap);first=copy.deepcopy(adapter.trace)
    size=first['ues'][0]['tb_bytes'];assert size>0
    user.is_retx=True;user.requested_bytes=size;inputs[0].slot_id=2;model.force_nack=False
    adapter.run_cluster_slot(inputs,replace(snap,slot_id=2));second=copy.deepcopy(adapter.trace)
    np.testing.assert_allclose(model.calls[1],2*model.calls[0],rtol=1e-5)
    assert second['ues'][0]['harq_combining']=='chase_combined'
    user.is_retx=False;inputs[0].slot_id=1
    noisy=RtSysPhyAdapter({**c['phy']['rt_sys'],'csi_error_nmse_db':0})
    noisy.run_cluster_slot(inputs,snap);a=copy.deepcopy(noisy.trace)
    noisy.run_cluster_slot(inputs,snap);assert a==noisy.trace
    assert not a['ideal_csi']
    return {'scene':scene,'actual_rt_sys_cuda':True,'chase_input_sinr_ratio':2.,'chase_gain_db':float(10*np.log10(2)),
        'first_ack_forced_false_for_probe_only':True,'repeated_coded_bits_rv':0,'payload_bytes':size,
        'imperfect_csi_replay_exact':True,'ideal_precoder':first['precoders'][0],
        'imperfect_precoder':a['precoders'][0],'limitations':'Chase post-equalizer SINR abstraction; no IR or waveform HARQ proof'}


if __name__=='__main__':
    report={'passed':True,'cases':[validate(scene) for scene in ('simple_street_canyon','munich')]}
    path=ROOT/'outputs/rt_sys_repairs/fidelity_validation.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

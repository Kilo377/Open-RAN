from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS
import numpy as np
import pytest
from phy.rt_sys.tx_contract import data_re, data_mask, nr_budget
from phy.rt_sys.olla import OllaState, HarqFeedbackEvent
from traffic.packet_ledger import PacketLedger


def test_payload_reservations_out_of_order_duplicate_ack_and_release():
    lc=NS(logical_channel_id='l',queue_bytes=400,max_queue_bytes=1000,packet_delay_budget_ms=20)
    ue=NS(ue_id='u',logical_channels=[lc]);ledger=PacketLedger(ue,.5)
    a=ledger.reserve('a',[('l',150)]);b=ledger.reserve('b',[('l',150)])
    assert a!=b and ledger.available('l')==100
    assert ledger.acknowledge('b',2)=={'l':150}
    assert ledger.acknowledge('b',3)=={}
    ledger.release('a');assert ledger.available('l')==250
    ledger.reserve('c',[('l',250)]);ledger.acknowledge('c',4)
    assert lc.queue_bytes==0 and ledger.stats['delivered_bytes']==400
    assert ledger.stats['completed_packets']==2


@pytest.mark.parametrize('prbs,symbols,pattern',[(1,1,None),(4,14,'dense'),(3,5,'sparse')])
def test_shared_pilot_mask_and_sys_budget(prbs,symbols,pattern):
    mask=np.zeros((14,600),bool);mask[0:symbols,12:12+12*prbs]=True
    count=int(data_mask(mask,pattern).sum())
    assert count==data_re(prbs,symbols,pattern)
    q,coded,bits=nr_budget('qam64',3,count)
    assert coded==q*count and bits>0


def test_olla_causality_duplicates_direction_boundaries_and_handover():
    s=OllaState();s.consume(1,'c')
    e=HarqFeedbackEvent('a','c','qam64',8,True,False,2)
    s.pending=[e,e];s.consume(1,'c');assert not s.offsets
    s.consume(2,'c');assert s.offsets['qam64']==pytest.approx(.01)
    s.pending=[HarqFeedbackEvent('b','c','qam64',8,False,False,3)]
    s.consume(3,'c');assert s.offsets['qam64']==pytest.approx(-.08)
    s.pending=[HarqFeedbackEvent('b','c','qam64',8,True,True,4),
               HarqFeedbackEvent('c','c','qam64',27,True,False,4)]
    s.consume(4,'c');assert s.offsets['qam64']==pytest.approx(-.08)
    assert s.stats['first_samples']==3 and s.stats['retx_samples']==1
    s.offsets['qam64']=-9.99
    s.pending=[HarqFeedbackEvent('d','c','qam64',8,False,False,5)]
    s.consume(5,'c');assert s.offsets['qam64']==-10
    s.consume(6,'new');assert s.offsets=={}


def test_real_nr_contract_closed_loop_with_fake_geometry(tmp_path):
    from test_rt_sys import FakeTracer
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from core.simulator import Simulator
    from core.factory import build_controller_registry, build_scheduler
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    m,c,_=build_case(num_cells=1,num_ues=4,total_slots=20)
    c['ric'].update(rapps=[],xapps=[])
    provider=RtChannelProvider(m,tmp_path/'cache',backend=FakeTracer(m))
    adapter=RtSysPhyAdapter({**c['phy']['rt_sys'],'device':'cpu','feedback_mode':'nr_direct'},channel_provider=provider)
    runtime=build_runtime_scenario_config(scenario_name='contract_fixture',scenario_type='rt_scene',sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
    sim=Simulator(runtime,build_controller_registry(c['ric']),build_scheduler(c['sim']),adapter,c['cell'],c['ue'],c['beamforming'],str(tmp_path))
    for _ in range(20):
        sim.step()
        assert not any(u.get('mac_capacity_guard') for u in adapter.trace['ues'])
        assert not any(u.get('ack_source')=='harq_incompatible_no_tb' for u in adapter.trace['ues'])
        for ue in sim.state.ues:ue.packet_ledger.refresh(sim.state.slot_id)
    assert adapter.abstraction.statistics['amc_probe_trials']==0
    assert adapter.abstraction.decode_trials>0


def test_retx_original_rectangle_is_reserved_before_new_transmissions():
    from phy.rt_sys.scheduler import RtSysScheduler
    from phy.rt_sys.tx_contract import TbTxContract
    from models.scheduling import UeGrant
    scheduler=RtSysScheduler(total_prbs_per_slot=8,total_symbols_per_slot=14)
    scheduler._contracts=True
    def contract(uid):return TbTxContract(uid,'c',3,'qam64',1,6,14,'sparse',972,1944,304,30,(('l',30),))
    users={u:NS(ue_id=u,harq=NS(processes=[NS(process_id=0,tx_contract=contract(u),retx_deadline_slot=5,last_tx_slot=1)],events=[])) for u in ('a','b')}
    grants=[UeGrant(u,4,14,is_retx=True,harq_process_id=0,spatial_group_id=1,spatial_group_size=2) for u in users]
    state=NS(slot_id=2,scheduler=NS(packing_strategy='frequency_first'))
    scheduler._apply_spatial_multiplexing(state,grants,users)
    assert len(grants)==1 and grants[0].allocated_prbs==6
    assert grants[0].spatial_group_size==1 and grants[0].spatial_group_id is None
    assert users['b'].harq.events[0]['kind']=='retx_deferred_resource'
    assert not users['a'].harq.events


def test_structural_codec_guard_rejects_bg1_repetition_without_probes():
    from phy.rt_sys.tx_contract import codec_format_supported
    assert not codec_format_supported('qam64_low_se',10,data_re(24,14,'dense')*2)
    assert codec_format_supported('qam64_low_se',10,data_re(16,14,'dense')*2)
    from phy.rt_sys.calibration import NRWaveform
    wave=NRWaveform(3,10,data_re(16,14,'dense')*2,'cpu')
    assert wave.bits>0


def test_immutable_contract_allows_frequency_move_but_rejects_format_change():
    from phy.rt_sys.tx_contract import TbTxContract
    q,coded,bits=nr_budget('qam64',3,data_re(4,14,'sparse'))
    c=TbTxContract('tb','c',3,'qam64',1,4,14,'sparse',data_re(4,14,'sparse'),coded,bits,10,(('l',10),))
    with pytest.raises(FrozenInstanceError):c.mcs=9
    u=NS(target_mcs=3,mcs_table_id='qam64',layer_count=1,allocated_prbs=4,allocated_symbols=14,
         dmrs_pattern_id='sparse',requested_bytes=10,grant_bytes_budget=bits//8,start_prb=0)
    c.validate(u,'c');u.start_prb=20;c.validate(u,'c')
    u.layer_count=2
    with pytest.raises(ValueError,match='before transmission'):c.validate(u,'c')
    u.layer_count=1
    with pytest.raises(ValueError):c.validate(u,'new_cell')


def test_codec_preflight_handles_multidimensional_multicb_lengths():
    from phy.rt_sys.tx_contract import codec_format_supported
    from phy.rt_sys.calibration import NRWaveform
    count=data_re(48,14,'dense')*2
    assert codec_format_supported('qam64',20,count)
    wave=NRWaveform(1,20,count,'cpu')
    assert wave.num_cb>1

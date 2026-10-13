from types import SimpleNamespace as NS
import numpy as np
import pytest
from state.logical_channel_state import LogicalChannelState
from traffic.packet_ledger import PacketLedger
from state.harq_state import HarqState, HarqProcessState
from ran.scheduler import SimpleScheduler
from phy.rt_sys.scheduler import RtSysScheduler


def test_packet_fifo_partial_delivery_latency_overflow_and_byte_ledger():
    lc = LogicalChannelState('lc', queue_bytes=200, max_queue_bytes=350, packet_delay_budget_ms=2)
    ue = NS(ue_id='u0', logical_channels=[lc])
    ledger = PacketLedger(ue, .5)
    ue.packet_ledger = ledger
    ledger.add(lc, 250, 1, packet_size=100)
    # Atomic packets: first 100 admitted, next 100 lost, final 50 admitted.
    assert lc.queue_bytes == 350 and ledger.stats['overflow_bytes'] == 100
    ledger.deliver(lc, 50, 2); lc.queue_bytes -= 50
    ledger.refresh(6)
    assert ue.hol_delay_ms == 3 and ledger.stats['deadline_missed_packets'] == 3
    ledger.deliver(lc, 150, 6); lc.queue_bytes -= 150
    assert ledger.stats['late_completed_packets'] == 1
    assert ue.hol_delay_ms == 2.5
    assert ledger.stats['initial_bytes'] + ledger.stats['offered_bytes'] == ledger.stats['delivered_bytes'] + ledger.stats['pending_bytes'] + ledger.stats['overflow_bytes']
    with pytest.raises(RuntimeError, match='exceeds'):
        ledger.deliver(lc, 999, 7)


def test_multiple_logical_channels_and_hol_is_not_report_age():
    u = NS(ue_id='u', logical_channels=[LogicalChannelState('a', queue_bytes=40), LogicalChannelState('b', queue_bytes=60)])
    u.packet_ledger = PacketLedger(u, .5)
    u.packet_ledger.refresh(100)
    scheduler = SimpleScheduler()
    assert scheduler._estimate_hol_delay_ms(NS(slot_duration_ms=.5), u, NS(queue_age_slots=2, hol_delay_ms=49.)) == 49.
    assert scheduler._estimate_hol_delay_ms(NS(slot_duration_ms=.5), u, None) == 50.


def test_empty_harq_cleanup_is_not_payload_failure():
    empty = HarqProcessState(process_id=0, pending_bytes=0, timeout_slot=5)
    payload = HarqProcessState(process_id=1, pending_bytes=40, timeout_slot=5)
    harq = HarqState(processes=[empty, payload])
    SimpleScheduler()._prune_expired_harq_processes(NS(slot_id=6), NS(harq=harq))
    assert harq.empty_process_cleanup_count == 1 and harq.harq_flush_count == 1
    assert [e['pending_bytes'] for e in harq.events] == [0, 40]
    assert not harq.processes


def test_nr_retransmission_preserves_actual_mcs_and_table():
    p = HarqProcessState(process_id=0, pending_bytes=100, mcs_index=24, mcs_table_id='qam256')
    user = NS(rt_feedback_mode='nr_tb_reference', latency_budget_ms=100, harq=HarqState(processes=[p]))
    s = RtSysScheduler()
    assert s._derive_target_mcs(user, is_retx=True) == 24
    assert s._derive_mcs_cap(user, True, 24) == 24
    assert s._derive_mcs_table_id(user, is_retx=True) == 'qam256'


def reference_state():
    from phy.rt_sys.contracts import ChannelSnapshot
    cell = lambda: NS(tx_power_dbm=30, current_codebook_id=0, current_beam_profile_id=0, cell_sleep_mode='active')
    cells = {cid: NS(cell=cell(), resource_grid=NS(n_prbs_total=2)) for cid in ('c0', 'c1')}
    user = NS(ue_id='u0', serving_beam_id=0, queue_bytes=100, radio_quality_db=500,
              bler=.1, ack=True, scheduled_bytes=100, transport_block_size_bytes=100, post_eq_sinr_db=12.)
    h = np.zeros((1,2,1,4,2,24), np.complex64)
    h[0,0,0,0] = 1e-5; h[0,1,0,0] = 1e-6
    snap = ChannelSnapshot(1, ('u0',), ('c0','c1'), h, np.zeros((1,3)), 'fixture', {'subcarrier_spacing_hz':30000})
    state = NS(ues=[user], managed_cells=cells, cell=cells['c0'].cell, connections={'u0':NS(serving_cell_id='c0')},
               slot_id=1, cluster_scheduled_ue_ids=['u0'], links={})
    return snap, state


def test_rt_measurement_power_interference_beam_and_no_path():
    from phy.rt_sys.measurements import reference_measurements
    snap, state = reference_state()
    a = reference_measurements(snap, state)['u0']
    state.managed_cells['c0'].cell.tx_power_dbm += 10
    b = reference_measurements(snap, state)['u0']
    assert b['rsrp_dbm']-a['rsrp_dbm'] == pytest.approx(10)
    assert b['reference_sinr_db']-a['reference_sinr_db'] == pytest.approx(10)
    state.managed_cells['c1'].cell.cell_sleep_mode = 'deep_sleep'
    assert reference_measurements(snap, state)['u0']['reference_sinr_db'] > b['reference_sinr_db']
    state.managed_cells['c0'].cell.current_codebook_id = 1
    assert reference_measurements(snap, state)['u0']['rsrp_dbm'] < b['rsrp_dbm']
    snap.cfr[0,0] = 0
    assert reference_measurements(snap, state)['u0']['no_path']


def test_reporting_uses_rt_and_excludes_zero_payload_bler_samples():
    from core.measurement_updater import MeasurementUpdater
    from phy.rt_sys.measurements import reference_measurements
    snap, state = reference_state()
    updater = MeasurementUpdater(rng_seed=1, slot_duration_ms=.5, queue_delay_slots=2,
        sinr_window_slots=1, sinr_noise_std_db=0, bler_window_tbs=20, throughput_granularity_slots=10)
    state.rt_reference_measurements = reference_measurements(snap, state)
    updater.update(state)
    first = state.measurement.ue_measurements['u0']
    assert first.sinr_db == pytest.approx(state.rt_reference_measurements['u0']['reference_sinr_db'])
    assert first.data_sinr_db == 12 and first.bler_sample_count == 1
    state.ues[0].radio_quality_db = -500
    state.ues[0].scheduled_bytes = 0
    state.ues[0].ack = False
    state.slot_id = 2
    updater.update(state)
    second = state.measurement.ue_measurements['u0']
    assert first.sinr_db == second.sinr_db and second.bler_sample_count == 1
    state.ues[0].scheduled_bytes = 100
    state.ues[0].reference_tb_ack = True
    state.slot_id = 3
    updater.update(state)
    assert state.measurement.ue_measurements["u0"].bler_window_nacks == 0


@pytest.mark.parametrize('table,mcs', [(1,0),(3,0),(3,8),(3,16)])
def test_low_rate_nr_high_snr_decode_and_deep_outage(table, mcs):
    from phy.rt_sys.calibration import NRWaveform
    w = NRWaveform(table,mcs,324,'cpu')
    errors = w.simulate_fast([40.,-40.], np.ones(324), 4, 113)
    np.testing.assert_array_equal(errors,[0,4])
    np.testing.assert_array_equal(errors,w.simulate([40.,-40.],np.ones(324),4,113))


def test_repetition_mother_prefix_agrees_with_sionna_encoder_and_parity():
    import torch
    from phy.rt_sys.low_rate import LowRateNRCodec
    from sionna.phy.fec.scrambling import Descrambler
    codec = LowRateNRCodec(72,2,648,'cpu')
    u = torch.randint(0,2,(4,72),generator=torch.Generator().manual_seed(42)).float()
    coded = Descrambler(codec.scrambler, binary=True, device='cpu')(codec.encode(u))
    # Undo modulation interleaving and compare first RV0 prefix with official
    # Sionna rate matching of the same CRC and BG2 mother word.
    perm, inverse = codec.mother.generate_out_int(648,2)
    prefix = coded[...,torch.tensor(inverse)]
    official = codec.mother(codec.crc(u))
    torch.testing.assert_close(prefix[...,:official.shape[-1]],official)
    padded = torch.cat([codec.crc(u),torch.zeros(4,codec.mother.k_ldpc-codec.k)],dim=-1)
    word = codec.mother._encode_fast(padded).reshape(4,codec.mother.n_ldpc).numpy()
    assert not (codec.mother.pcm @ word.T % 2).any()


def test_controller_observation_does_not_expose_unfiltered_rt_truth(tmp_path):
    from test_rt_sys import FakeTracer, FixtureAbstraction
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from phy.rt_sys.channel import RtChannelProvider
    from core.simulator import Simulator
    from core.factory import build_controller_registry, build_scheduler
    from core.metrics import build_observation
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    m,c,_=build_case(num_cells=1,num_ues=2,total_slots=5)
    c['ric'].update(rapps=[],xapps=[])
    provider=RtChannelProvider(m,tmp_path/'cache',backend=FakeTracer(m))
    adapter=RtSysPhyAdapter({**c['phy']['rt_sys'],'device':'numpy'},abstraction=FixtureAbstraction(),channel_provider=provider)
    runtime=build_runtime_scenario_config(scenario_name='observer_fixture',scenario_type='rt_scene',sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
    sim=Simulator(runtime,build_controller_registry(c['ric']),build_scheduler(c['sim']),adapter,c['cell'],c['ue'],c['beamforming'],str(tmp_path))
    sim.step()
    uid=sim.state.ues[0].ue_id
    measurement=sim.state.measurement.ue_measurements[uid]
    measurement.data_sinr_db=-17.
    measurement.reference_beam_gain_db=-3.
    sim.state.ues[0].post_eq_sinr_db=999.
    observed=next(u for u in build_observation(sim.state).ue_observations if u.ue_id==uid)
    assert observed.post_eq_sinr_db == -17. and observed.beamforming_gain_db == -3.
    assert observed.post_eq_sinr_source == 'data_measurement'
    assert observed.hol_delay_ms == measurement.hol_delay_ms
    measurement.data_sinr_db=None
    observed=next(u for u in build_observation(sim.state).ue_observations if u.ue_id==uid)
    assert observed.post_eq_sinr_db == measurement.reference_sinr_db
    assert observed.post_eq_sinr_source == 'reference_hypothesis'


def test_fixed_nr_tuple_never_projects_to_another_mcs():
    from phy.rt_sys.reference import NrReferenceAbstraction
    model = object.__new__(NrReferenceAbstraction)
    model.candidates = {'qam64': [0,3]}
    model.selection_criterion = 'wilson_upper'
    attempted = []
    def unsupported(table, mcs, count):
        attempted.append(mcs)
        raise ValueError('Unsupported coderate (r<1/5) test tuple')
    model.waveform = unsupported
    result = model.evaluate(np.ones(24), 'qam64', 3, 1, 1, 'u', fixed=True)
    assert attempted == [3] and result['capacity_bytes'] == 0
    assert result['reference_tb_ack'] is None


@pytest.mark.parametrize('pending', [0,120])
def test_no_tb_clears_empty_reservation_but_retains_retransmission(pending):
    from core.state_updater import StateUpdater
    from models.phy_result import PhyResult, PhyUeResult
    process = HarqProcessState(process_id=1, pending_bytes=pending, is_busy=True,
        tb_id='original', timeout_slot=20, retx_deadline_slot=20, mcs_index=3, mcs_table_id='qam64')
    harq = HarqState(process_id=1, processes=[process], pending_retx_bytes=pending)
    user = NS(ue_id='u', queue_bytes=300, harq=harq, is_retx=bool(pending), is_scheduled=True)
    state = NS(slot_id=5, sim_time_ms=5., slot_duration_ms=1., ues=[user],
        cell=NS(tx_power_dbm=30., current_codebook_id=0, current_beam_profile_id=0),
        traffic=NS(arrival_bytes_total=0, arrival_bytes_by_ue={}),
        scheduler=NS(scheduled_ue_ids=['u'],grants=[]))
    result = PhyResult(slot_id=5, ue_results=[PhyUeResult(ue_id='u',is_scheduled=True,
        is_retx=bool(pending), harq_process_id=1,scheduled_bytes=0,delivered_bytes=0,ack=False)])
    StateUpdater().apply_phy_result(state,result)
    assert harq.pending_retx_bytes == pending and user.queue_bytes == 300
    assert harq.harq_flush_count == 0
    assert len(harq.processes) == bool(pending)
    if pending:
        assert harq.processes[0].tb_id == 'original'

from types import SimpleNamespace
import numpy as np
from phy.rt_sys.mobility import RtMobilityManager
from phy.rt_sys.scheduler import RtSysScheduler
from phy.rt_sys.contracts import ChannelSnapshot
from phy.rt_sys.channel import reconstruct_paths


def test_ttt_transient_reset_and_minimum_dwell():
    adapter=SimpleNamespace(config={'mobility':{'time_to_trigger_ms':4,'min_dwell_ms':10}})
    manager=RtMobilityManager(adapter,slot_duration_ms=.5,rng_seed=1,delay_slots=2)
    cell=SimpleNamespace(handover_margin_db=3,steering_hysteresis_db=1,handover_cost_db=0,cell_sleep_mode='active')
    state=SimpleNamespace(slot_id=1,cell=cell,managed_cells={'c0':SimpleNamespace(cell=cell)})
    conn=SimpleNamespace(serving_cell_id='c0',last_handover_slot=None)
    manager._event_context=(state,SimpleNamespace(ue_id='u0'),conn)
    args=dict(event_trigger='a3_offset',event_offset_db=0,trigger_margin_db=10,
              best_candidate=SimpleNamespace(cell_id='c1'),current_candidate=SimpleNamespace(cell_id='c0'))
    for slot in range(1,9):
        state.slot_id=slot; assert not manager._compute_measurement_event_met(**args)
    state.slot_id=9; assert manager._compute_measurement_event_met(**args)
    conn.last_handover_slot=9;state.slot_id=10;assert not manager._compute_measurement_event_met(**args)
    state.slot_id=28;assert not manager._compute_measurement_event_met(**args)
    state.slot_id=29;assert not manager._compute_measurement_event_met(**args)
    state.slot_id=37;assert manager._compute_measurement_event_met(**args)
    state.slot_id=38;assert not manager._compute_measurement_event_met(**{**args,'trigger_margin_db':2})
    state.slot_id=39;assert not manager._compute_measurement_event_met(**args)


def test_rt_scheduler_uses_channel_independence_without_beam_angle_bonus():
    h=np.zeros((2,1,1,4,2,24),np.complex64);h[0,0,0,0]=1e-5;h[1,0,0,1]=1e-5
    snap=ChannelSnapshot(1,('u0','u1'),('c0',),h,np.zeros((2,3)),'fixture',{})
    state=SimpleNamespace(rt_channel_snapshot=snap,cell=SimpleNamespace(cell_id='c0',current_codebook_id=0,current_beam_profile_id=0),
        resource_grid=SimpleNamespace(n_prbs_total=2),scheduler=SimpleNamespace(max_total_spatial_layers=2),beamforming={})
    users=[SimpleNamespace(ue_id=f'u{i}',serving_beam_id=0,ri=1) for i in range(2)]
    scheduler=RtSysScheduler()
    assert scheduler._compute_spatial_compatibility_score(state,*users)>.99
    h[1]=h[0]; assert scheduler._compute_spatial_compatibility_score(state,*users)==0
    h[1]=0; assert scheduler._compute_spatial_compatibility_score(state,*users)==0


def test_path_cache_retains_absolute_gain_delay_and_doppler():
    cir=np.ones((1,1,1,2,1),np.complex64)*(.3+.4j)
    arrays={'cir0':cir,'tau':np.ones((1,1,1),np.float32)*1e-6,'doppler':np.ones((1,1,1),np.float32)*100}
    m={'num_symbols':2,'fft_size':4,'subcarrier_spacing_hz':30000,'slot_duration_ms':.5}
    h=reconstruct_paths(arrays,m,.01)
    assert h.shape==(1,1,1,2,2,4)
    np.testing.assert_allclose(abs(h),.5,rtol=1e-6)
    expected=(.3+.4j)*np.exp(2j*np.pi*(100*(.01+.00025)-30000*1e-6))
    np.testing.assert_allclose(h[0,0,0,0,1,3],expected,rtol=1e-5)



def test_csi_keyed_replay_and_zero_link():
    from phy.rt_sys.fidelity import estimate_csi
    h=np.ones((2,1,2,4,2,24),np.complex64)*1e-5;h[1]=0
    snap=ChannelSnapshot(1,('u0','u1'),('c0',),h,np.zeros((2,3)),'fixture',{})
    assert estimate_csi(snap,1,None) is h
    a=estimate_csi(snap,7,-10);np.testing.assert_array_equal(a,estimate_csi(snap,7,-10))
    assert not np.array_equal(a,h);assert not a[1].any()
    # Changing traversal order cannot change keyed UE samples.
    other=ChannelSnapshot(1,('u1','u0'),('c0',),h[::-1],np.zeros((2,3)),'fixture',{})
    np.testing.assert_array_equal(a,estimate_csi(other,7,-10)[::-1])


def test_estimated_combiner_is_evaluated_on_true_signal_and_interference():
    from phy.rt_sys.numerics import Algebra
    a=Algebra();d=np.array([[[1.],[0.]]],complex);others=np.array([[[0.],[2.]]],complex)
    optimal=a.lmmse(d,others,1.)
    wrong=a.lmmse(d,others,1.,np.array([[[1.],[1.]]]),np.zeros_like(others))
    np.testing.assert_allclose(optimal,[1.]);np.testing.assert_allclose(wrong,[1/6])


def test_chase_repetition_gain_and_incompatible_tb_reset():
    from phy.rt_sys.fidelity import ChaseCombiner
    c=ChaseCombiner('chase')
    u=SimpleNamespace(ue_id='u0',harq_process_id=1,harq_ndi=0,target_mcs=3,mcs_table_id='qam64',
        requested_bytes=17,harq_redundancy_version=0,is_retx=False)
    x=np.array([[[2.],[0.],[4.]]])
    first,reason=c.combine(u,'c0',1,x);assert reason=='new_tb'
    u.is_retx=True;second,reason=c.combine(u,'c0',2,x)
    assert reason=='chase_combined';np.testing.assert_equal(second,2*x)
    u.harq_redundancy_version=1;third,reason=c.combine(u,'c0',3,x)
    assert reason=='reset_incompatible_coded_bits';np.testing.assert_equal(third,x)
    u.harq_ndi=1;assert c.combine(u,'c0',4,x)[1]=='reset_incompatible_coded_bits'
    c.acknowledge(u);assert not c.pending
    import pytest
    with pytest.raises(ValueError):ChaseCombiner('ir')


def test_compact_quota_failure_is_atomic_and_keeps_existing_evidence(tmp_path):
    from phy.rt_sys.cache_storage import publish,compact_bytes
    import pytest
    folder=tmp_path/'key';folder.mkdir();existing=folder/'paths_000000000.npz';existing.write_bytes(b'old')
    temp=folder/'new.tmp';temp.write_bytes(b'12345');target=folder/'paths_000000020.npz'
    with pytest.raises(RuntimeError,match='quota'):publish(temp,target,7)
    assert existing.read_bytes()==b'old';assert not temp.exists() and not target.exists()
    temp.write_bytes(b'12345');publish(temp,target,8);assert compact_bytes(tmp_path)==8


def test_heldout_protocol_and_long_trajectory():
    from experiments.rt_sys_protocol import plan,validate
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import position_at
    jobs=plan(True);assert validate(jobs)['jobs']==12
    m,_,_=build_case(num_ues=1,total_slots=200000,deployment_id=0)
    route=m['ues'][0]['trajectory'];assert len(route['positions_m'])>2
    speeds=[np.linalg.norm(position_at(route,(a+b)/2)[1]) for a,b in zip(route['times_s'],route['times_s'][1:])]
    np.testing.assert_allclose(speeds,1.5)
    jobs[-1]['seed']=1
    import pytest
    with pytest.raises(AssertionError):validate(jobs)


def test_frozen_selection_memory_is_not_mutated_or_persisted(tmp_path):
    from ric.non_rt.xapp_selection_rapp import XAppSelectionRApp
    app=XAppSelectionRApp(freeze_learning=True,persistent_memory_enabled=True,memory_store_path=str(tmp_path/'model.json'))
    app._store_success_case(SimpleNamespace());app._store_failure_case(SimpleNamespace())
    app._record_retrieval_usage(retrieved_success_cases=[],retrieved_failure_cases=[],selection_query_key=SimpleNamespace())
    app._persist_memory_state()
    assert not app.success_memory and not app.failure_memory
    assert not (tmp_path/'model.json').exists()
    assert not app._baseline_training_active("intent_hrl")


def test_closed_loop_enables_natural_mu_and_counts_shared_resources_once(tmp_path):
    from test_rt_sys import FakeTracer,FixtureAbstraction
    from core.simulator import Simulator
    from core.factory import build_controller_registry,build_scheduler
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.scenario import build_case
    class Orthogonal(FakeTracer):
        def generate(self,*args):
            data=super().generate(*args);h=data['cfr'];h[:]=0
            for i in range(2):h[:,i,:,0,i]=1e-5
            return data
    m,c,_=build_case(num_cells=1,num_ues=2,total_slots=5)
    c['ric'].update(rapps=[],xapps=[])
    c['ue']['ues'][1]['slice_id']=c['ue']['ues'][0]['slice_id']
    provider=RtChannelProvider(m,tmp_path/'cache',backend=Orthogonal(m))
    adapter=RtSysPhyAdapter({**c['phy']['rt_sys'],'device':'numpy'},abstraction=FixtureAbstraction(),channel_provider=provider)
    runtime=build_runtime_scenario_config(scenario_name='mu_fixture',scenario_type='rt_scene',sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
    sim=Simulator(runtime,build_controller_registry(c['ric']),build_scheduler(c['sim']),adapter,c['cell'],c['ue'],c['beamforming'],str(tmp_path))
    groups=0
    for _ in range(5):
        sim.step();groups+=sum(len(g['streams'])>1 for g in adapter.trace['precoders'])
        grid=sim.state.resource_grid
        assert grid.allocated_resource_elements==adapter.trace['cells']['cell_0']['occupied_re']
        assert 0<=grid.utilization<=1
    assert groups>0


def test_scheduled_zero_sinr_streams_count_as_erasures_not_free_re():
    from test_rt_sys import FixtureAbstraction,phy,ue,channel
    from phy.rt_sys.adapter import RtSysPhyAdapter
    class Capture(FixtureAbstraction):
        def evaluate(self,values,*args,**kwargs):
            self.values=values
            return super().evaluate(values,*args,**kwargs)
    model=Capture();adapter=RtSysPhyAdapter({'device':'numpy'},abstraction=model)
    h=channel(np.array([[[[1e-5,0],[0,0]]]]),('u0',),('c0',))
    h.cfr[...,1,0]=0
    result=adapter.run_cluster_slot([phy('c0',[ue('u0',layers=2)])],h)['c0'].ue_results[0]
    assert np.count_nonzero(model.values)==result.effective_data_re_count*2
    assert np.min(model.values[model.values>0])==1e-30


def test_protocol_rejects_hidden_warm_start(tmp_path,monkeypatch):
    import pytest
    import experiments.rt_sys_protocol as protocol
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    root=tmp_path/'outputs'/'proof';(root/'fold0').mkdir(parents=True)
    saved=root/'fold0'/'trained_memory.json';saved.write_text('{"success_cases":[]}')
    with pytest.raises(ValueError,match='fresh protocol'):
        protocol.execute(protocol.plan(True),root,'cuda',True)
    assert saved.read_text()=='{"success_cases":[]}'

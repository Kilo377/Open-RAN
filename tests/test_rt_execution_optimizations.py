"""Implementation optimizations preserve PHY results, including imperfect CSI."""
from dataclasses import asdict
from copy import deepcopy
import numbers
import numpy as np
import pytest

from phy.rt_sys.adapter import RtSysPhyAdapter
from phy.rt_sys.efficiency import configure_torch_threads, diagnostic_power_sums
from phy.rt_sys.numerics import Algebra


def assert_nested(a,b,rtol=1e-10,atol=1e-10):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:assert_nested(a[k],b[k],rtol,atol)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested(x,y,rtol,atol)
    elif isinstance(a,numbers.Real):np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)
    else:assert a==b


@pytest.mark.parametrize('nmse',[None,-10.])
@pytest.mark.parametrize('options',[
    {'ideal_csi_fast_path':True},
    {'packed_phy_diagnostics':True},
    {'ideal_csi_fast_path':True,'packed_phy_diagnostics':True},
])
def test_optimized_cluster_preserves_results_and_diagnostics(nmse,options):
    from test_rt_sys import channel,ue,phy,FixtureAbstraction
    rng=np.random.default_rng(432)
    h=(rng.normal(size=(4,2,2,4))+1j*rng.normal(size=(4,2,2,4)))*1e-5
    snapshot=channel(h,('u0','u1','u2','u3'),('c0','c1'))
    inputs=[phy('c0',[ue('u0'),ue('u1')]),phy('c1',[ue('u2'),ue('u3')])]
    cfg={'device':'numpy','csi_error_nmse_db':nmse}
    original=RtSysPhyAdapter(cfg,abstraction=FixtureAbstraction())
    optimized=RtSysPhyAdapter({**cfg,**options},abstraction=FixtureAbstraction())
    before=original.run_cluster_slot(deepcopy(inputs),snapshot)
    after=optimized.run_cluster_slot(deepcopy(inputs),snapshot)
    assert_nested({k:asdict(v) for k,v in before.items()},{k:asdict(v) for k,v in after.items()})
    assert_nested(original.trace,optimized.trace)


def test_packed_power_diagnostics_keep_addition_order():
    algebra=Algebra('numpy')
    parts=[np.array([1e20,1e-20],dtype=np.float32),np.array([1,2],dtype=np.float32),np.array([3,4],dtype=np.float32)]
    inter,intra=diagnostic_power_sums(algebra,parts[:2],parts[2:],2)
    expected=np.zeros(2)
    for part in parts[:2]:expected+=part
    np.testing.assert_array_equal(inter,expected)
    np.testing.assert_array_equal(intra,parts[2])


@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_invalid_torch_thread_counts_are_rejected_before_import(value):
    with pytest.raises(ValueError,match='positive integer'):configure_torch_threads({'torch_num_threads':value})


def test_retained_cfr_does_not_change_host_reconstruction():
    from phy.rt_sys.channel import reconstruct_paths
    arrays={'cir0':np.full((1,1,1,2,3),1e-5+2e-6j,dtype=np.complex64),
            'tau':np.full((1,1,3),1e-8,dtype=np.float32),'doppler':np.full((1,1,3),12.,dtype=np.float32)}
    manifest={'num_symbols':2,'fft_size':24,'slot_duration_ms':.5,'subcarrier_spacing_hz':30000}
    host,device=reconstruct_paths(arrays,manifest,.003,return_device_array=True)
    np.testing.assert_array_equal(host,reconstruct_paths(arrays,manifest,.003))
    np.testing.assert_array_equal(device,host)


def test_reference_cache_invalidates_when_report_beam_identity_changes(tmp_path):
    from test_paper_interfaces import make_sim
    sim=make_sim(tmp_path,cells=1,slots=3,config={'beam_reports':True})
    sim.phy_adapter.config['reuse_reference_within_slot']=True
    paper=sim.state.paper;paper.prepare_reference()
    count=paper.reference_cache_stats['calculations']
    paper.prepare_reference();assert paper.reference_cache_stats['calculations']==count
    next(iter(sim.state.connections.values())).serving_beam_id+=1
    paper.prepare_reference();assert paper.reference_cache_stats['calculations']==count+1
    sim.phy_adapter.current_snapshot.metadata['subcarrier_spacing_hz']*=2
    paper.prepare_reference();assert paper.reference_cache_stats['calculations']==count+2


@pytest.mark.parametrize('mode',['none','chase'])
@pytest.mark.parametrize('worker_threads',[1,2])
def test_parallel_nr_preserves_codewords_noise_llrs_and_invalid_retransmissions(mode,worker_threads):
    from phy.rt_sys.direct import DirectNrAbstraction
    import torch
    scalar=DirectNrAbstraction('cpu',config={'harq_combining':mode})
    parallel=DirectNrAbstraction('cpu',config={'harq_combining':mode,'nr_execution':'parallel',
        'nr_parallel_workers':2,'nr_worker_threads':worker_threads,'nr_pin_workers':worker_threads==2})
    try:
        for slot in (1,2,3):
            requests=[dict(sinr=np.resize([.02,.3,1.5,10.],336 if slot==2 and i==0 else (1944 if i==3 else 324)),
                table='qam64',mcs=3,seed=512,slot=slot,ue_id=f'u{i}',
                harq_context=dict(process_id=0,ndi=0,is_retx=slot>1,cell_id='c')) for i in range(4)]
            assert parallel.evaluate_many(requests)==[scalar.evaluate(**r) for r in requests]
            for key,a in scalar.harq.pending.items():
                b=parallel.harq.pending[key]
                assert torch.equal(a['bits'],b['bits'])
                if mode=='chase':assert torch.equal(a['llr'],b['llr'])
        assert scalar.decode_trials==parallel.decode_trials==11
        assert parallel.harq.stats['incompatible_no_tb']==1
    finally:parallel.close()


def test_cupy_retained_cfr_and_packed_diagnostics_match_host_path():
    cp=pytest.importorskip('cupy')
    try:cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:pytest.skip('CUDA unavailable')
    from dataclasses import replace
    from test_rt_sys import channel,ue,phy,FixtureAbstraction
    rng=np.random.default_rng(432)
    h=((rng.normal(size=(4,2,2,4))+1j*rng.normal(size=(4,2,2,4)))*1e-5).astype(np.complex64)
    snapshot=channel(h,('u0','u1','u2','u3'),('c0','c1'))
    retained=replace(snapshot)
    retained.device_cfr=cp.asarray(snapshot.cfr);retained.device_backend='cupy'
    assert 'device_cfr' not in asdict(retained)
    assert replace(retained).device_cfr is None
    inputs=[phy('c0',[ue('u0'),ue('u1')]),phy('c1',[ue('u2'),ue('u3')])]
    cfg={'device':'numpy','algebra_backend':'cupy'}
    a=RtSysPhyAdapter(cfg,abstraction=FixtureAbstraction())
    b=RtSysPhyAdapter({**cfg,'retain_channel_on_gpu':True,'ideal_csi_fast_path':True,
        'packed_phy_diagnostics':True},abstraction=FixtureAbstraction())
    x=a.run_cluster_slot(deepcopy(inputs),snapshot)
    y=b.run_cluster_slot(deepcopy(inputs),retained)
    assert_nested({k:asdict(v) for k,v in x.items()},{k:asdict(v) for k,v in y.items()})
    # GPU and host norm reductions differ by float32 rounding; PHY above is exact.
    assert_nested(a.trace,b.trace,rtol=1e-6,atol=1e-6)


def test_parallel_unsupported_codec_keeps_scalar_no_tb_result(monkeypatch):
    from phy.rt_sys.direct import DirectNrAbstraction
    a=DirectNrAbstraction('cpu')
    b=DirectNrAbstraction('cpu',config={'nr_execution':'parallel','nr_parallel_workers':2})
    def unavailable(*args):raise ValueError('Unsupported coderate (r<1/5)')
    monkeypatch.setattr(a,'waveform',unavailable);monkeypatch.setattr(b,'waveform',unavailable)
    requests=[dict(sinr=np.ones(324),table='qam64',mcs=3,seed=512,slot=1,ue_id='u0',
        harq_context=dict(process_id=0,ndi=0,is_retx=False,cell_id='c'))]
    try:assert b.evaluate_many(requests)==[a.evaluate(**r) for r in requests]
    finally:b.close()

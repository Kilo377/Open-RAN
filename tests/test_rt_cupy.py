"""Real Orin CUDA linear algebra must agree with independent CPU references."""
import numpy as np
import pytest


@pytest.fixture
def cp():
    cupy=pytest.importorskip('cupy')
    try:
        available=cupy.cuda.runtime.getDeviceCount()>0
    except cupy.cuda.runtime.CUDARuntimeError:
        available=False
    if not available:pytest.skip('CuPy CUDA unavailable')
    return cupy


def test_cupy_rzf_and_lmmse_match_numpy(cp):
    from phy.rt_sys.numerics import Algebra
    rng=np.random.default_rng(42)
    cpu,gpu=Algebra('numpy'),Algebra('cupy')
    h=(rng.normal(size=(12,2,4))+1j*rng.normal(size=(12,2,4)))*1e-5
    h[0]=0  # No serving path still emits the allocated physical beam.
    basis=np.eye(4,dtype=complex)
    w=cpu.rzf(h,basis,1e-10)
    actual=gpu.rzf(gpu.array(h),gpu.array(basis),1e-10)
    assert isinstance(actual,cp.ndarray) and actual.device.id==0
    np.testing.assert_allclose(gpu.numpy(actual),w,rtol=3e-4,atol=3e-5)
    np.testing.assert_allclose(np.sum(abs(gpu.numpy(actual))**2,axis=-2),1,atol=1e-5)
    desired=h[...,:1]
    others=h[...,1:]
    for estimated in (False,True):
        reference=cpu.lmmse(desired,others,1e-11,desired*.9 if estimated else None,others*.9 if estimated else None)
        result=gpu.lmmse(gpu.array(desired),gpu.array(others),1e-11,
            gpu.array(desired*.9) if estimated else None,gpu.array(others*.9) if estimated else None)
        assert isinstance(result,cp.ndarray)
        np.testing.assert_allclose(gpu.numpy(result),reference,rtol=3e-4,atol=1e-5)


@pytest.mark.parametrize('path_count',[0,5])
def test_cupy_preserves_absolute_path_gain_delay_and_doppler(cp,path_count):
    from phy.rt_sys.channel import reconstruct_paths
    rng=np.random.default_rng(19)
    shape=(3,2,2,4,path_count)
    arrays={'cir0':((rng.normal(size=shape)+1j*rng.normal(size=shape))*1e-5).astype(np.complex64),
        'tau':rng.uniform(0,1e-6,size=(3,2,path_count)).astype(np.float32),
        'doppler':rng.uniform(-120,120,size=(3,2,path_count)).astype(np.float32)}
    manifest={'num_symbols':14,'fft_size':64,'slot_duration_ms':.5,'subcarrier_spacing_hz':30000}
    expected=reconstruct_paths(arrays,manifest,.013)
    actual=reconstruct_paths(arrays,manifest,.013,'cupy')
    assert actual.dtype==np.complex64
    np.testing.assert_allclose(actual,expected,rtol=3e-4,atol=1e-9)


@pytest.mark.parametrize('mode',['overlap','separate','silent','no_path'])
def test_cupy_cluster_matches_cpu_and_preserves_power(cp,mode):
    from test_rt_sys import channel,ue,phy,FixtureAbstraction
    from phy.rt_sys.adapter import RtSysPhyAdapter
    h=np.array([[[[1e-5,2e-6]],[[3e-6,1e-6]]],
                [[[2e-6,1e-6]],[[1e-6,1e-5]]]])
    if mode=='no_path':h[:]=0
    snap=channel(h)
    inputs=[phy('c0',[ue('u0')]),phy('c1',[ue('u1',scheduled=mode!='silent',start=1 if mode=='separate' else 0)])]
    cpu=RtSysPhyAdapter({'device':'cpu'},abstraction=FixtureAbstraction())
    gpu=RtSysPhyAdapter({'device':'cpu','algebra_backend':'cupy'},abstraction=FixtureAbstraction())
    expected=cpu.run_cluster_slot(inputs,snap)
    actual=gpu.run_cluster_slot(inputs,snap)
    for cid in actual:
        np.testing.assert_allclose([u.sinr_db for u in actual[cid].ue_results],
            [u.sinr_db for u in expected[cid].ue_results],atol=2e-3)
        assert [u.delivered_bytes for u in actual[cid].ue_results]==[u.delivered_bytes for u in expected[cid].ue_results]
        assert gpu.trace['cells'][cid]['average_used_w']==pytest.approx(cpu.trace['cells'][cid]['average_used_w'])

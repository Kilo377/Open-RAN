"""Exact coordinate evaluation and single-producer RT cache regression checks."""
from dataclasses import replace
from types import SimpleNamespace as N
import multiprocessing
import json
import time
import numpy as np
import pytest
from phy.rt_sys.channel import reconstruct_paths,geometry_blocks
from phy.rt_sys.sampled_channel import PathCoefficients
from phy.rt_sys.contracts import ChannelSnapshot


def fixture(full_axes=False,paths=5):
    rng=np.random.default_rng(43);shape=(2,2,2,4,paths)
    axes=shape if full_axes else (2,2,paths)
    arrays=dict(cir0=((rng.normal(size=shape)+1j*rng.normal(size=shape))*1e-5).astype(np.complex64),
        tau=rng.uniform(0,5e-6,axes).astype(np.float32),doppler=rng.uniform(-30,30,axes).astype(np.float32))
    manifest=dict(num_symbols=14,fft_size=64,subcarrier_spacing_hz=30000,slot_duration_ms=.5)
    elapsed=.0035;dense=reconstruct_paths(arrays,manifest,elapsed)
    snap=ChannelSnapshot(7,('u0','u1'),('c0','c1'),dense,np.zeros((2,3)),'fixture',{})
    sampled=replace(snap,cfr=None,path_sampler=PathCoefficients(arrays,manifest).snapshot(elapsed))
    return arrays,manifest,snap,sampled


@pytest.mark.parametrize('full_axes',[False,True])
@pytest.mark.parametrize('paths',[0,1,5])
def test_requested_coordinates_match_dense_path_sum(full_axes,paths):
    arrays,manifest,dense,sampled=fixture(full_axes,paths)
    sampled.validate();s=np.array([0,6,13,3,8]);f=np.array([0,17,32,55,63])
    for ui in range(2):
        for ci in range(2):
            np.testing.assert_allclose(sampled.samples(ui,ci,s,f),dense.samples(ui,ci,s,f),rtol=3e-5,atol=1e-10)
            np.testing.assert_allclose(sampled.grid(ui,ci,s,f),dense.grid(ui,ci,s,f),rtol=3e-5,atol=1e-10)
    links=[(0,0),(0,1),(1,0),(1,1)]
    batch=sampled.samples_for_links(links,s,f)
    for link in links:np.testing.assert_allclose(batch[link],dense.samples(*link,s,f),rtol=3e-5,atol=1e-10)
    assert sampled.grid_shape==dense.cfr.shape and sampled.cfr is None
    stats=sampled.path_sampler.coefficients.stats
    before=dict(stats);sampled.samples(1,1,s,f);assert stats==before
    for bad in [([-1],[0]),([0],[64]),([14],[0])]:
        with pytest.raises(ValueError,match='outside'):sampled.samples(0,0,*bad)


def test_sampled_snapshot_tracks_doppler_and_rejects_invalid_paths():
    from phy.rt_sys.fidelity import estimate_csi
    arrays,manifest,dense,sampled=fixture()
    with pytest.raises(ValueError,match='ideal CSI'):estimate_csi(sampled,1,-10)
    next_=PathCoefficients(arrays,manifest).snapshot(.004)
    assert not np.array_equal(next_.paired(0,0,[0],[32]),sampled.samples(0,0,[0],[32]))
    arrays['cir0'][0,0,0,0,0]=np.nan
    with pytest.raises(ValueError,match='Invalid'):PathCoefficients(arrays,manifest)


@pytest.mark.parametrize('ret',[False,True])
@pytest.mark.parametrize('full_axes',[False,True])
def test_reference_beam_and_subband_gains_keep_original_band_and_symbols(ret,full_axes):
    from phy.rt_sys.paper_radio import probe_gains
    _,_,dense,sampled=fixture(full_axes)
    for snap in (dense,sampled):snap.metadata.update(tx_rows=2,tx_cols=2)
    cell=N(ret_enabled=ret,ret_tilt_deg=7,ret_codebook_id=0,current_codebook_id=0,current_beam_profile_id=0)
    for reports in [False,True]:
        a=probe_gains(dense,0,0,cell,2,reports,0);b=probe_gains(sampled,0,0,cell,2,reports,0)
        np.testing.assert_allclose(a[0],b[0],rtol=1e-6);assert a[1]==b[1]
        for x,y in zip(a[2],b[2]):
            assert x[:2]==y[:2];np.testing.assert_allclose(x[2],y[2],rtol=1e-6)


def test_reference_moments_preserve_coherent_cancellation_and_doppler():
    arrays=dict(cir0=np.array([1.,-1.],np.complex64).reshape(1,1,1,1,2),
        tau=np.zeros((1,1,2),np.float32),doppler=np.zeros((1,1,2),np.float32))
    manifest=dict(num_symbols=14,fft_size=64,subcarrier_spacing_hz=30000,slot_duration_ms=.5)
    sampler=PathCoefficients(arrays,manifest).snapshot(.0035)
    np.testing.assert_allclose(sampler.mean_reference_power(0,0,np.ones((1,1)),2,20),0.,atol=1e-15)
    arrays['doppler'][0,0,1]=28
    sampler=PathCoefficients(arrays,manifest).snapshot(.0035)
    dense=reconstruct_paths(arrays,manifest,.0035)[0,0,0,0,:,20:44]
    expected=(abs(dense)**2).mean(axis=0).reshape(2,12).mean(axis=1)
    actual=sampler.mean_reference_power(0,0,np.ones((1,1)),2,20)[0]
    np.testing.assert_allclose(actual,expected,rtol=1e-6)


def test_dynamic_resource_patterns_sinr_energy_and_scheduler_match():
    from test_rt_sys import FixtureAbstraction,phy,ue
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from phy.rt_sys.scheduler import RtSysScheduler
    _,_,dense,sampled=fixture()
    context=N(n_prbs_total=2,num_ofdm_symbols_per_slot=14,slot_duration_ms=.5,
        carrier_frequency_hz=3.5e9,bandwidth_hz=10e6,subcarrier_spacing_hz=30000,cyclic_prefix_type='normal')
    plans=[phy('c0',[ue('u0')]),phy('c1',[ue('u1')])]
    for i,p in enumerate(plans):
        p.slot_id=7;p.runtime_context=context;u=p.ue_inputs[0]
        u.allocated_symbol_indices=[1,3,11] if i==0 else [1,3,9,11]
        u.allocated_symbols=len(u.allocated_symbol_indices)
        u.allocated_subcarrier_indices=[0,2,4,6,8,11,12,15,18,21,23]
        u.allocated_prbs=2;u.resource_block_indices=[0,1]
    results=[];traces=[]
    for snapshot in (dense,sampled):
        a=RtSysPhyAdapter(dict(device='numpy',feedback_mode='fast',fast_profile='system'),abstraction=FixtureAbstraction())
        results.append(a.run_cluster_slot(plans,snapshot));traces.append(a.trace)
    for cid in results[0]:
        a=results[0][cid].ue_results[0];b=results[1][cid].ue_results[0]
        assert a.sinr_db==pytest.approx(b.sinr_db,abs=1e-4)
        assert a.delivered_bytes==b.delivered_bytes and a.effective_data_re_count==b.effective_data_re_count
    assert traces[0]['cells']==traces[1]['cells']
    users=[N(ue_id=f'u{i}',serving_beam_id=0,ri=1) for i in range(2)]
    state=N(cell=N(cell_id='c0',current_codebook_id=0,current_beam_profile_id=0),
        resource_grid=N(n_prbs_total=2),scheduler=N(max_total_spatial_layers=2),beamforming={})
    scores=[]
    for snapshot in (dense,sampled):
        state.rt_channel_snapshot=snapshot
        scores.append(RtSysScheduler()._compute_spatial_compatibility_score(state,*users))
    assert scores[0]==pytest.approx(scores[1],abs=1e-6)


class CountingTracer:
    def __init__(self,folder):self.folder=folder
    def generate_paths(self,block):
        time.sleep(.05)
        with (self.folder/'calls').open('a') as stream:stream.write(str(block)+'\n')
        return {'cir0':np.ones((1,1,1,1,1),np.complex64)}


def produce(folder,barrier,queue):
    from phy.rt_sys.cache_storage import ensure_paths
    try:
        barrier.wait(timeout=10)
        a,created,_=ensure_paths(folder,10,CountingTracer(folder),100000)
        queue.put((created,bool(a['cir0'].all())))
    except BaseException as e:queue.put(str(e))


def test_concurrent_foreground_and_prefetch_generate_once(tmp_path):
    # Use processes and real OS locks, rather than mocking lock acquisition.
    root=tmp_path/'key';root.mkdir();ctx=multiprocessing.get_context('spawn')
    barrier=ctx.Barrier(4);queue=ctx.Queue()
    jobs=[ctx.Process(target=produce,args=(root,barrier,queue)) for _ in range(4)]
    for p in jobs:p.start()
    rows=[queue.get(timeout=30) for _ in jobs]
    for p in jobs:p.join(timeout=10);assert p.exitcode==0
    assert all(isinstance(r,tuple) and r[1] for r in rows),rows
    assert sum(r[0] for r in rows)==1 and (root/'calls').read_text()=='10\n'
    assert len(list(root.glob('paths_*.npz')))==1 and not list(root.glob('*.tmp'))


def test_failed_cache_publish_releases_block_ownership(tmp_path):
    from phy.rt_sys.cache_storage import ensure_paths
    root=tmp_path/'key';root.mkdir();backend=CountingTracer(root)
    with pytest.raises(RuntimeError,match='quota'):ensure_paths(root,0,backend,1)
    assert not list(root.glob('*.tmp')) and not list(root.glob('*.npz'))
    arrays,created,_=ensure_paths(root,0,backend,10000)
    assert created and arrays['cir0'].all()


def test_prefetch_shards_cover_each_block_once_and_skip_foreground_work(tmp_path,monkeypatch):
    from phy.rt_sys.prefetch import _generate
    from phy.rt_sys.cache_storage import ensure_paths
    root=tmp_path/'key';root.mkdir();backend=CountingTracer(root)
    # Foreground has already produced zero and a future block.
    ensure_paths(root,0,backend,100000);ensure_paths(root,20,backend,100000)
    provider=N(key='fixture',root=root,blocks=[0,10,20,30,40],backend=backend)
    monkeypatch.setattr('phy.rt_sys.channel.RtChannelProvider',lambda *a,**k:provider)
    cancel=N(is_set=lambda:False)
    for worker in range(2):
        _generate({},tmp_path,'cpu',100000,'fixture',cancel,worker,2,str(root/f'status{worker}.json'))
    calls=[int(x) for x in (root/'calls').read_text().splitlines()]
    assert sorted(calls)==[0,10,20,30,40]
    statuses=[json.loads((root/f'status{i}.json').read_text()) for i in range(2)]
    assert sum(s['generated_blocks'] for s in statuses)==3
    assert sum(s['skipped_blocks'] for s in statuses)==1


def test_cancelled_prefetch_does_not_report_all_blocks_completed(tmp_path,monkeypatch):
    from phy.rt_sys.prefetch import _generate
    root=tmp_path/'key';root.mkdir();backend=CountingTracer(root)
    provider=N(key='fixture',root=root,blocks=[0,10,20,30,40],backend=backend)
    monkeypatch.setattr('phy.rt_sys.channel.RtChannelProvider',lambda *a,**k:provider)
    status=root/'cancelled.json'
    _generate({},tmp_path,'cpu',100000,'fixture',N(is_set=lambda:True),0,2,str(status))
    row=json.loads(status.read_text())
    assert row['completed'] is False and row['cancelled'] is True
    assert row['assigned_blocks']==2 and row['processed_blocks']==0
    assert not (root/'calls').exists()


def test_ten_ms_cli_applies_motion_cap_and_preserves_child_spec(monkeypatch,capsys):
    from experiments.run_rt_sys import parse_args,run_case,ROOT
    from experiments.live_xapp_test import ARGUMENTS
    args=['runner','--dry-run','--device','cpu','--model','fast','--fast-channel-mode','sampled',
        '--scene','munich','--cells','3','--ues','6','--config',str(ROOT/'configs/rt_sys/agx_munich_6ue_depth2_fast.json'),
        '--path-update-ms','10','--max-geometry-displacement-m','.015']
    monkeypatch.setattr('sys.argv',args);run_case(parse_args());s=json.loads(capsys.readouterr().out)
    assert geometry_blocks(s['manifest'])[1]==20
    assert s['phy_config']['rt_sys']['fast_channel_mode']=='sampled'
    assert {'max_geometry_displacement_m','fast_channel_mode'}<=set(ARGUMENTS)
    monkeypatch.setattr('sys.argv',['runner','--dry-run','--model','full','--fast-channel-mode','sampled'])
    with pytest.raises(ValueError,match='requires'):run_case(parse_args())

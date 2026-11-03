"""Weighted storage against expanded PHY and the installed official EESM."""
from copy import deepcopy
from types import SimpleNamespace as N
import numpy as np
import pytest
from phy.rt_sys.compact_sinr import CompactSinr
from phy.rt_sys.report_copy import copy_report, compiled_copy_report
from phy.rt_sys.numerics import beam_subspace, cached_beam_subspace


@pytest.mark.parametrize('layers,pattern',[(1,None),(2,'dense')])
@pytest.mark.parametrize('silence',[False,True])
def test_compact_phy_preserves_data_order_dmrs_power_and_interference(layers,pattern,silence):
    from test_rt_sys import channel,phy,ue,FixtureAbstraction
    from phy.rt_sys.adapter import RtSysPhyAdapter
    class Capture(FixtureAbstraction):
        supports_compact_sinr=True
        def __init__(self):self.seen={}
        def evaluate(self,sinr,table,mcs,seed,slot,ue_id,fixed=False):
            values=sinr.expand() if isinstance(sinr,CompactSinr) else sinr[sinr>0]
            self.seen[ue_id]=values.copy()
            return super().evaluate(values,table,mcs,seed,slot,ue_id,fixed)
    rng=np.random.default_rng(98)
    snap=channel((rng.normal(size=(2,2,2,4))+1j*rng.normal(size=(2,2,2,4)))*1e-5)
    snap.cfr*=rng.lognormal(0,.5,snap.cfr.shape)
    plans=[phy('c0',[ue('u0',layers=layers)]),phy('c1',[ue('u1')])]
    for p in plans:
        p.runtime_context=N(n_prbs_total=2,num_ofdm_symbols_per_slot=2,slot_duration_ms=.5,
            carrier_frequency_hz=3.5e9,bandwidth_hz=10e6,subcarrier_spacing_hz=30000,cyclic_prefix_type='normal')
        u=p.ue_inputs[0];u.dmrs_pattern_id=pattern
        u.allocated_prbs=2;u.allocated_subcarrier_indices=list(range(24))
    outputs=[]
    for compact in (False,True):
        abstraction=Capture()
        a=RtSysPhyAdapter(dict(device='numpy',feedback_mode='fast',fast_profile='system',compact_sinr=compact),abstraction=abstraction)
        a.paper=N(state=N(managed_cells={c:N(cell=N(prb_power_w=[0,.001] if silence else [.0002,.0008],ret_enabled=False)) for c in ['c0','c1']}))
        result=a.run_cluster_slot(deepcopy(plans),snap)
        outputs.append((abstraction.seen,result,a.trace))
    for uid in outputs[0][0]:
        np.testing.assert_array_equal(outputs[0][0][uid],outputs[1][0][uid])
    assert outputs[0][2]['cells']==outputs[1][2]['cells']
    for a,b in zip(outputs[0][2]['ues'],outputs[1][2]['ues']):
        for key in ['sinr_db','inter_cell_interference_dbm','intra_cell_interference_dbm']:
            assert a[key]==pytest.approx(b[key],abs=1e-10)
        assert (a['data_re'],a['delivered_bytes'])==(b['data_re'],b['delivered_bytes'])


def test_compact_probability_matches_official_eesm_and_bler():
    pytest.importorskip('sionna')
    from phy.rt_sys.fast import FastAbstraction
    from sionna.sys import EESM
    model=FastAbstraction('cpu')
    oracle=EESM(device='cpu',precision='double')
    rng=np.random.default_rng(11);compact=[];expanded=[]
    for table,mcs in [(1,1),(1,5),(1,11),(1,21),(2,11),(3,16)]:
        for shift in [-7,-1,2,8]:
            inverse=rng.integers(0,31,2052)
            values=rng.lognormal(shift,1.4,31)
            x=CompactSinr(values,np.bincount(inverse,minlength=31),inverse)
            wave=model.waveform(table,mcs,len(x))
            compact.append((wave,x));expanded.append((wave,x.expand()))
    expected=model.probability_many(expanded)
    actual=model.probability_many(compact)
    for (wave,x),a,b in zip(compact,expected,actual):
        assert a[2]==b[2]
        if a[0] is not None:
            assert a[0]==pytest.approx(b[0],abs=1e-5)
            # The float32 expanded oracle loses relative precision near zero.
            # Bound that difference in dB and independently check double EESM.
            assert abs(10*np.log10(a[1]/b[1]))<.01
            table,mcs=model.equivalent(wave)
            dense=model.torch.as_tensor(x.expand().astype(np.float32)).reshape(1,1,-1,1,1)
            reference=oracle(dense,mcs_index=model.torch.tensor([[mcs]]),mcs_table_index=table).item()
            assert b[1]==pytest.approx(reference,rel=1e-6,abs=1e-8)


def test_compact_actual_nr_fallback_preserves_codeword_and_feedback():
    pytest.importorskip('sionna')
    from phy.rt_sys.fast import FastAbstraction
    dense=FastAbstraction('cpu');compact=FastAbstraction('cpu')
    for slot in [1,2]:
        values=np.array([.01,.02]) if slot==1 else np.array([100.,200.])
        inverse=np.arange(324)%2
        x=CompactSinr(values,np.bincount(inverse),inverse)
        kw=dict(table='qam64',mcs=1,seed=501,slot=slot,ue_id='u',
                harq_context=dict(process_id=1,ndi=1,is_retx=slot>1,cell_id='c0'))
        a=dense.evaluate_many([dict(sinr=x.expand(),**kw)])[0]
        b=compact.evaluate_many([dict(sinr=x,**kw)])[0]
        assert a['actual_decode'] and b['actual_decode']
        assert a['ack']==b['ack']
        assert a['harq_reference']['codeword_id']==b['harq_reference']['codeword_id']


def test_strong_link_weighted_eesm_avoids_legacy_float32_underflow():
    pytest.importorskip('sionna')
    from phy.rt_sys.fast import FastAbstraction
    from sionna.sys import EESM
    model=FastAbstraction('cpu')
    x=CompactSinr(np.array([700.]),np.array([2052]),np.zeros(2052,dtype=int))
    wave=model.waveform(3,16,len(x))
    legacy=model.probability_many([(wave,x.expand())])[0]
    selected=model.probability_many([(wave,x)])[0]
    table,mcs=model.equivalent(wave)
    oracle=EESM(device='cpu',precision='double')
    dense=model.torch.as_tensor(x.expand()).reshape(1,1,-1,1,1)
    expected=oracle(dense,mcs_index=model.torch.tensor([[mcs]]),mcs_table_index=table).item()
    # EESM of a constant field equals that constant; exp(-700/beta) underflows
    # in the old float32 path, incorrectly selecting the 1000 clamp instead.
    assert legacy[1]==pytest.approx(1000.)
    assert selected[1]==pytest.approx(700.)
    assert expected==pytest.approx(700.)
    assert legacy[0]==selected[0]==0.


def test_compiled_report_copy_and_cached_basis_are_isolated():
    original={'ue':{'u':dict(prbs=[1,2],metadata=(2.,-0.,None))},'cell':{'c':dict(power=1.)}}
    a=compiled_copy_report(original);b=copy_report(original)
    assert a==b
    a['ue']['u']['prbs'].append(8)
    assert original==b
    for codebook in range(3):
        for profile in range(2):
            for beam in range(4):
                np.testing.assert_array_equal(cached_beam_subspace(4,codebook,profile,beam),beam_subspace(4,codebook,profile,beam))
    with pytest.raises(ValueError):cached_beam_subspace(4,0,0,0)[0,0]=0


def test_grant_fast_copy_preserves_contract_and_list_isolation():
    from phy.rt_sys.control import limit_logical_allocations
    from models.scheduling import LogicalChannelGrant
    grant=LogicalChannelGrant('lc',scheduled_bytes=100,prioritized_bytes=70,
                             used_pbr_credit_bytes=50,constraint_hits=['cap'])
    old=limit_logical_allocations([grant],40)
    new=limit_logical_allocations([grant],40,fast_copy=True)
    assert new==old
    new[0].constraint_hits.append('changed')
    assert grant.constraint_hits==['cap'] and old[0].constraint_hits==['cap']


@pytest.mark.parametrize('seed',[1,2,3])
def test_compact_full_grid_mixed_mu_patterns_and_guards(seed):
    from test_rt_sys import phy,ue,FixtureAbstraction
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from phy.rt_sys.contracts import ChannelSnapshot
    class Capture(FixtureAbstraction):
        supports_compact_sinr=True
        def __init__(self):self.seen={}
        def evaluate(self,sinr,table,mcs,seed,slot,ue_id,fixed=False):
            values=sinr.expand() if isinstance(sinr,CompactSinr) else sinr[sinr>0]
            self.seen[ue_id]=values.copy()
            return super().evaluate(values,table,mcs,seed,slot,ue_id,fixed)
    rng=np.random.default_rng(seed);uids=tuple('u'+str(i) for i in range(6));cids=('c0','c1','c2')
    h=(rng.normal(size=(6,3,2,4,14,512))+1j*rng.normal(size=(6,3,2,4,14,512)))*1e-5
    snap=ChannelSnapshot(1,uids,cids,h,np.zeros((6,3)),'mixed_fixture',{'normalized':False})
    plans=[];cells={}
    for ci,cid in enumerate(cids):
        members=[]
        for j in range(2):
            u=ue(uids[2*ci+j],layers=j+1)
            start=0 if j==0 else 4
            u.allocated_prbs=24;u.allocated_symbols=10;u.start_symbol=start
            u.allocated_subcarrier_indices=list(range(288));u.allocated_symbol_indices=list(range(start,start+10))
            u.dmrs_pattern_id='dense' if ci==1 else None
            members.append(u)
        p=phy(cid,members)
        p.runtime_context=N(n_prbs_total=24,num_ofdm_symbols_per_slot=14,slot_duration_ms=.5,
            carrier_frequency_hz=3.5e9,bandwidth_hz=10e6,subcarrier_spacing_hz=30000,cyclic_prefix_type='normal')
        plans.append(p)
        powers=rng.uniform(0,.001,24);powers[ci::5]=0
        cells[cid]=N(cell=N(prb_power_w=powers.tolist(),ret_enabled=False))
    rows=[]
    for compact in [False,True]:
        abstraction=Capture()
        a=RtSysPhyAdapter(dict(device='numpy',feedback_mode='fast',fast_profile='system',compact_sinr=compact),abstraction=abstraction)
        a.paper=N(state=N(managed_cells=cells))
        a.run_cluster_slot(deepcopy(plans),snap);rows.append((abstraction.seen,a.trace))
    for uid in uids:np.testing.assert_array_equal(rows[0][0][uid],rows[1][0][uid])
    assert rows[0][1]['cells']==rows[1][1]['cells']
    for a,b in zip(rows[0][1]['ues'],rows[1][1]['ues']):
        assert a['sinr_db']==pytest.approx(b['sinr_db'],abs=1e-10)
        assert (a['mu_data_re'],a['su_data_re'])==(b['mu_data_re'],b['su_data_re'])


def test_compact_zero_path_remains_an_outage():
    from test_rt_sys import channel,phy,ue,FixtureAbstraction
    from phy.rt_sys.adapter import RtSysPhyAdapter
    class NeverDecode(FixtureAbstraction):
        supports_compact_sinr=True
        def evaluate(self,*a,**kw):pytest.fail('Outage must not manufacture a transmission')
    a=RtSysPhyAdapter(dict(device='numpy',feedback_mode='fast',fast_profile='system'),abstraction=NeverDecode())
    a.run_cluster_slot([phy('c0',[ue('u0')])],channel(np.zeros((1,1,2,4)),('u0',),('c0',)))
    assert a.trace['ues'][0]['delivered_bytes']==0
    assert not a.trace['ues'][0]['feedback_available']

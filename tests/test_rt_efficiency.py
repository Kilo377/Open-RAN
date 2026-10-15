import numpy as np
import pytest
from phy.rt_sys.efficiency import resource_patterns
from phy.rt_sys.direct import DirectNrAbstraction


def test_vectorized_patterns_keep_original_order_and_per_prb_silence():
    rng=np.random.default_rng(513);ns,nf=7,37
    for count in (0,1,4,24,70):
        entries=[dict(mask=rng.random((ns,nf))>.7,
            power=(rng.random((ns,nf))>.3) if i%2 else float(i%3)) for i in range(count)]
        expected={}
        for s in range(ns):
            for f in range(nf):
                key=tuple(i for i,e in enumerate(entries) if e['mask'][s,f] and
                    (e['power'][s,f] if isinstance(e['power'],np.ndarray) else e['power'])>0)
                if key:expected.setdefault(key,[]).append((s,f))
        actual=resource_patterns(entries,ns,nf)
        assert list(actual.items())==list(expected.items())


@pytest.mark.parametrize('mode',['none','chase'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_nr_batch_keeps_bits_noise_feedback_and_chase(mode,device):
    import torch
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable on this test host')
    scalar=DirectNrAbstraction(device,config={'harq_combining':mode})
    batched=DirectNrAbstraction(device,config={'harq_combining':mode})
    for slot in (1,2,3):
        requests=[]
        for i in range(5):
            requests.append(dict(sinr=np.resize([.02,.3,1.5,10.],324 if i<4 else 336),
                table='qam64',mcs=3,seed=512,slot=slot,ue_id=f'u{i}',
                harq_context=dict(process_id=0,ndi=0,is_retx=slot>1,cell_id='c')))
        expected=[scalar.evaluate(**r) for r in requests]
        actual=batched.evaluate_many(requests)
        assert actual==expected
        for key,a in scalar.harq.pending.items():
            b=batched.harq.pending[key]
            assert torch.equal(a['bits'],b['bits'])
            if mode=='chase':torch.testing.assert_close(a['llr'],b['llr'],rtol=1e-6,atol=1e-6)
    assert scalar.decode_trials==batched.decode_trials==15
    assert batched.harq.stats['batched_tbs']==12
    assert batched.statistics['amc_probe_trials']==batched.statistics['bler_probe_trials']==0


def test_reference_reuse_invalidates_when_rf_changes(tmp_path,monkeypatch):
    from test_paper_interfaces import make_sim
    import phy.rt_sys.paper_environment as module
    sim=make_sim(tmp_path,cells=1,slots=3)
    sim.phy_adapter.config['reuse_reference_within_slot']=True
    paper=sim.state.paper;calls=[];original=module.references
    def counted(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(module,'references',counted)
    paper.prepare_reference();assert not calls
    sim.state.cell.tx_power_dbm-=3
    paper.prepare_reference();assert len(calls)==1
    paper.prepare_reference();assert len(calls)==1
    sim.state.cell.current_codebook_id=1
    paper.prepare_reference();assert len(calls)==2
    sim.step();assert len(calls)==3


def test_compact_recording_preserves_tb_contracts_without_mutating_trace():
    from experiments.rt_diagnostics import compact_trace
    raw={'precoders':[{'weights_real':[1.]}],'ues':[{'tb_id':'t','ack':True}],
         'paper_interfaces':{'runtime_config':{}},'cells':{'c':{'budget_w':1}}}
    compact=compact_trace(raw)
    assert 'precoders' not in compact and raw['precoders']
    assert compact['ues']==raw['ues'] and compact['cells']==raw['cells']

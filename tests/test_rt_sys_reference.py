import numpy as np
import pytest
from phy.rt_sys.reference import NrReferenceAbstraction,reference_seed


def test_reference_streams_are_independent_and_order_invariant():
    a=NrReferenceAbstraction('cpu',config={'decoder_cache_size':2})
    values=np.ones((1,324,1))*10**(.5/10)
    first=a.evaluate(values,'qam64',3,71,5,'u0',fixed=True)
    a.evaluate(values,'qam64',3,71,5,'u1',fixed=True)
    replay=a.evaluate(values,'qam64',3,71,5,'u0',fixed=True)
    assert first==replay
    assert first['ack_seed']!=first['bler_probe_seed']
    assert first['reference_tb_errors']==int(not first['ack'])
    assert first['bler_probe_samples']==64


def test_actual_crc_feedback_is_independent_of_sys_bler_sampling(monkeypatch):
    a=NrReferenceAbstraction('cpu')
    w=a.waveform(1,3,324)
    ack_seed=reference_seed(7,2,'u0',1,3,324,'transmitted_tb')
    def simulation(snrs,profile,samples,seed):
        assert len(profile)==324 and np.all(profile==100.)
        return np.array([1 if seed==ack_seed else 0])
    monkeypatch.setattr(w,'simulate_fast',simulation)
    r=a.evaluate(np.ones((1,324,1))*100,'qam64',3,7,2,'u0')
    assert r['tbler']==0 and not r['ack'] and r['reference_tb_errors']==1
    assert r['ack_source']=='actual_nr_crc_and_bit_check'


def test_reference_does_not_hide_unknown_decoder_errors(monkeypatch):
    a=NrReferenceAbstraction('cpu')
    def broken(*args):raise ValueError('unexpected LDPC implementation failure')
    monkeypatch.setattr(a,'waveform',broken)
    with pytest.raises(ValueError,match='unexpected LDPC'):
        a.evaluate(np.ones((1,324,1)),'qam64',3,1,1,'u0')


def test_reference_rejects_invalid_probe_budget():
    with pytest.raises(ValueError,match='AMC probes'):
        NrReferenceAbstraction('cpu',config={'selection_samples':0})


def test_unvalidated_default_sys_requires_explicit_diagnostic_opt_in():
    from phy.rt_sys.adapter import RtSysPhyAdapter
    with pytest.raises(ValueError,match='diagnostic opt-in'):
        RtSysPhyAdapter({'device':'cpu','feedback_mode':'sys_abstraction'})

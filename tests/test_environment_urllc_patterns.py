"""Production phased arrivals and packet conservation; no network runs."""
from copy import deepcopy
import random

import pytest

from experiments import environment_urllc_patterns as patterns
from experiments.environment_validation_plan import CANDIDATES, profile
from experiments.environment_refinement_plan import sampled_load
from traffic.arrival_model import TrafficProfile, build_arrival_model


@pytest.mark.parametrize('pattern',['burst4','mixed4','mixed5','rare6'])
def test_registered_windows_keep_exact_c3_bytes_packet_sizes_and_counts(pattern):
    parent=profile(CANDIDATES[2])
    original=sampled_load(parent['ues'])
    changed=sampled_load(patterns.arrivals(parent,pattern))
    for uid,mean in patterns.MEAN_PACKET_BYTES.items():
        for name,count in [('before',2000),('pilot_after',2000),('validation_after',4000)]:
            a,b=original[uid][name],changed[uid][name]
            assert b['bytes']==a['bytes']==count*mean
            assert b['packets']==a['packets']==count
            assert b['packet_sizes_bytes']==a['packet_sizes_bytes']==[mean]
    if pattern=='burst4':
        assert changed==original
        assert patterns.metadata(pattern)['parent_trace_unchanged'] is True
    else:
        assert patterns.metadata(pattern)['parent_trace_unchanged'] is False
        assert changed!=original


@pytest.mark.parametrize('pattern,peak,low_on,low_off',[('mixed4',4,2,1),('mixed5',5,5,4),('rare6',6,44,5)])
def test_phased_high_starts_global_slot1_and_low_uses_local_slot0(pattern,peak,low_on,low_off):
    parent=profile(CANDIDATES[2]);overrides=patterns.arrivals(parent,pattern)
    for uid,mean in patterns.MEAN_PACKET_BYTES.items():
        fields=overrides[uid]
        assert fields['packet_size_bytes']==mean and fields['burst_size_bytes']==0
        period=2000 if pattern=='rare6' else 400
        assert sum(p['duration_slots']for p in fields['arrival_phases'])==period
        sampler=build_arrival_model(TrafficProfile(ue_id=uid,**fields));rng=random.Random(0)
        samples=[sampler.sample_bytes(slot_id=i,slot_duration_s=.0005,rng=rng)for i in range(1,2*period+1)]
        assert samples[:40]==[peak*mean]*40
        assert samples[40:40+low_on]==[mean]*low_on
        assert samples[40+low_on:40+low_on+low_off]==[0]*low_off
        assert sum(samples[:period])==period*mean and samples[:period]==samples[period:]
        assert sum(x//mean for x in samples[:40])==40*peak
        assert sum(x//mean for x in samples[40:period])==period-40*peak
        if pattern=='rare6':
            assert patterns.metadata(pattern)['cycle_slots']==2000
            assert patterns.metadata(pattern)['high_phase']['packets_per_cycle']==240
            assert patterns.metadata(pattern)['low_phase']['packets_per_cycle']==1760


def test_factory_rejects_other_parent_means_and_does_not_mutate_parent():
    parent=profile(CANDIDATES[2]);preserved=deepcopy(parent)
    mixed=patterns.arrivals(parent,'mixed4');mixed['ue_2']['arrival_phases'][0]['arrival_rate_bps']=0
    assert parent==preserved
    with pytest.raises(ValueError):patterns.arrivals(parent,'invented')
    parent['ues']['ue_2']['packet_size_bytes']=200
    with pytest.raises(ValueError,match='exact unchanged C3'):
        patterns.arrivals(parent,'mixed5')

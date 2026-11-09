from copy import deepcopy
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from experiments.urllc_traffic_profile import apply_profile
from traffic.arrival_model import build_arrival_model, TrafficProfile
from traffic.traffic_engine import TrafficEngine
from state.logical_channel_state import LogicalChannelState


def test_equal_actual_integer_byte_mean_over_full_burst_cycle():
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root/'configs/rt_sys/urllc_equal_mean_burst_2x.json').read_text())
    for uid, expected in [('ue_2', 175), ('ue_6', 137)]:
        model = build_arrival_model(TrafficProfile(ue_id=uid, **profile['ues'][uid]))
        values = [model.sample_bytes(slot_id=i, slot_duration_s=.0005, rng=random.Random(42)) for i in range(1, 1601)]
        assert sum(values) == expected * 1600
        assert max(values) == expected * 4
        assert values.count(0) == 1200
        packet_size = profile['ues'][uid]['packet_size_bytes']
        assert packet_size == expected
        assert sum(math.ceil(v/packet_size) for v in values) == 1600


def test_overlay_reaches_real_logical_channel_and_preserves_other_configuration(tmp_path):
    cfg = {'ue': {'ues': [dict(ue_id='ue_2', latency_budget_ms=10,
        arrival_model='cbr', arrival_rate_bps=2800000,
        logical_channels=[dict(logical_channel_id='data', arrival_model='cbr', arrival_rate_bps=2800000)])]},
        'phy': {'unchanged': True}}
    p = tmp_path/'profile.json'
    p.write_text(json.dumps(dict(profile_id='test', ues={'ue_2': dict(arrival_model='poisson', packet_size_bytes=200)})))
    manifest = {}
    apply_profile(manifest, cfg, p)
    assert cfg['ue']['ues'][0]['logical_channels'][0]['arrival_model'] == 'poisson'
    assert cfg['ue']['ues'][0]['arrival_rate_bps'] == 2800000
    assert cfg['ue']['ues'][0]['latency_budget_ms'] == 10
    assert cfg['phy'] == {'unchanged': True}
    assert manifest['traffic_profile']['sha256']


@pytest.mark.parametrize('extra', [dict(latency_budget_ms=1), dict(arrival_rate_bps=float('nan')),
    dict(packet_size_bytes=True), dict(arrival_model='unknown')])
def test_invalid_or_nonarrival_override_does_not_mutate_case(tmp_path, extra):
    cfg = {'ue': {'ues': [dict(ue_id='ue_2', latency_budget_ms=10)]}}
    before = deepcopy(cfg)
    p = tmp_path/'profile.json'
    p.write_text(json.dumps(dict(profile_id='test', ues={'ue_2': extra})))
    with pytest.raises(ValueError):
        apply_profile({}, cfg, p)
    assert cfg == before


def alternating_profile():
    root = Path(__file__).resolve().parents[1]
    path = root/'configs/rt_sys/urllc_alternating_2x.json'
    return path, json.loads(path.read_text())


@pytest.mark.parametrize('uid, size', [('ue_2', 175), ('ue_6', 137)])
def test_alternating_phase_boundaries_and_exact_full_phase_packet_means(uid, size):
    _, profile = alternating_profile()
    model = build_arrival_model(TrafficProfile(ue_id=uid, **profile['ues'][uid]))
    rng = random.Random(123)
    values = [model.sample_bytes(slot_id=i, slot_duration_s=.0005, rng=rng) for i in range(1, 6401)]
    expected = {1: size, 1600: size, 1601: 4*size, 1640: 4*size,
                1641: 0, 3200: 0, 3201: size}
    for slot, value in expected.items():
        assert values[slot-1] == value
    assert values[:3200] == values[3200:]
    for begin in (0, 1600, 3200, 4800):
        phase = values[begin:begin+1600]
        assert sum(phase) == size*1600
        assert sum(math.ceil(v/size) for v in phase) == 1600
    assert values[1600:3200].count(4*size) == 400
    assert values[1600:3200].count(0) == 1200


def test_alternating_reaches_actual_engine_and_packet_ledger_without_phase_resets():
    path, profile = alternating_profile()
    config = {'ue': {'ues': [dict(ue_id=uid, arrival_model='cbr', arrival_rate_bps=size*16000,
        latency_budget_ms=10, packet_size_bytes=size,
        logical_channels=[dict(logical_channel_id='data', arrival_model='cbr',
            arrival_rate_bps=size*16000, packet_size_bytes=size)])
        for uid, size in [('ue_2',175), ('ue_6',137)]]}}
    config['ue']['ues'].append(dict(ue_id='peer', arrival_model='poisson',
        arrival_rate_bps=2000000, packet_size_bytes=200))
    baseline = deepcopy(config)
    before_radio = dict(slot_duration_ms=.5, marker='unchanged channel')
    manifest = deepcopy(before_radio)
    apply_profile(manifest, config, path)
    assert all(manifest[k] == value for k,value in before_radio.items())
    assert all(u['latency_budget_ms'] == 10 for u in config['ue']['ues'][:2])
    def state_for(cfg):
        return SimpleNamespace(slot_id=0, ues=[SimpleNamespace(ue_id=u['ue_id'], queue_bytes=0,
            logical_channels=[LogicalChannelState(logical_channel_id=(u.get('logical_channels') or
                [dict(logical_channel_id='lc_0')])[0]['logical_channel_id'], queue_bytes=0,
                packet_delay_budget_ms=10, max_queue_bytes=None)]) for u in cfg['ue']['ues']])
    state, reference = state_for(config), state_for(baseline)
    engine = TrafficEngine(ue_config=config['ue'], slot_duration_ms=.5, rng_seed=7)
    baseline_engine = TrafficEngine(ue_config=baseline['ue'], slot_duration_ms=.5, rng_seed=7)
    engine.enable_packet_tracking(state)
    counts = {uid: [0,0,0,0] for uid in profile['ues']}
    sizes = {uid: set() for uid in profile['ues']}
    carried = {}
    peer_bytes = 0
    for slot in range(1,6401):
        state.slot_id = reference.slot_id = slot
        arrivals, _ = engine.apply_arrivals(state)
        reference_arrivals, _ = baseline_engine.apply_arrivals(reference)
        assert arrivals['peer'] == reference_arrivals['peer']
        peer_bytes += arrivals['peer']
        for ue in state.ues:
            lc = ue.logical_channels[0]
            ledger = ue.packet_ledger
            if ue.ue_id in counts:
                events = [e for e in ledger.drain_events() if e['kind']=='arrival']
                counts[ue.ue_id][(slot-1)//1600] += len(events)
                sizes[ue.ue_id].update(e['bytes'] for e in events)
                if slot == 1600:
                    carried[ue.ue_id] = ledger.queues['data'][0]['packet_id']
                if slot == 1601:
                    assert ledger.queues['data'][0]['packet_id'] == carried[ue.ue_id]
            if slot != 1600:
                ledger.deliver(lc, lc.queue_bytes, slot)
    assert counts == {'ue_2': [1600]*4, 'ue_6': [1600]*4}
    assert sizes == {'ue_2': {175}, 'ue_6': {137}}
    assert peer_bytes > 0
    # Config/profile records own their nested data and cannot mutate a running model.
    config['ue']['ues'][0]['arrival_phases'][0]['arrival_rate_bps'] = 0
    assert config['ue']['ues'][0]['logical_channels'][0]['arrival_phases'][0]['arrival_rate_bps'] == 2800000
    assert manifest['traffic_profile']['overrides']['ue_2']['arrival_phases'][0]['arrival_rate_bps'] == 2800000
    assert engine._models['ue_2:data'].sample_bytes(slot_id=6401,slot_duration_s=.0005,rng=random.Random()) == 175


@pytest.mark.parametrize('slot_ms', [None, 1, True, float('nan')])
def test_alternating_explicit_slot_duration_requires_a_matching_case(slot_ms):
    path, _ = alternating_profile()
    cfg = {'ue': {'ues': [dict(ue_id=uid) for uid in ('ue_2','ue_6')]}}
    manifest = {} if slot_ms is None else dict(slot_duration_ms=slot_ms)
    before = deepcopy(cfg)
    with pytest.raises(ValueError,match='slot duration'):
        apply_profile(manifest,cfg,path)
    assert cfg == before
    assert 'traffic_profile' not in manifest


def test_alternating_slot_duration_can_come_from_sim_config():
    path, _ = alternating_profile()
    cfg = {'ue': {'ues': [dict(ue_id=uid) for uid in ('ue_2','ue_6')]},
           'sim': {'slot_duration_ms': .5}}
    assert apply_profile({},cfg,path)['slot_duration_ms'] == .5


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(arrival_phases=[]),
    lambda p: p['arrival_phases'][0].update(duration_slots=0),
    lambda p: p['arrival_phases'][0].update(duration_slots=True),
    lambda p: p['arrival_phases'][0].update(arrival_rate_bps=float('nan')),
    lambda p: p['arrival_phases'][0].update(arrival_rate_bps=True),
    lambda p: p['arrival_phases'][0].update(arrival_model='phased'),
    lambda p: p['arrival_phases'][0].update(packet_size_bytes=1),
    lambda p: p['arrival_phases'][0].update(latency_budget_ms=1),
    lambda p: p['arrival_phases'][1].update(duration_slots=1599),
    lambda p: p['arrival_phases'][1].update(on_duration_slots=0),
    lambda p: p['arrival_phases'][1].update(off_duration_slots=-1),
    lambda p: p['arrival_phases'][1].pop('arrival_rate_bps'),
    lambda p: p.update(arrival_model='cbr'),
])
def test_malformed_phase_schedule_rejected_before_any_ue_or_manifest_mutation(tmp_path,mutation):
    _, profile = alternating_profile()
    mutation(profile['ues']['ue_6'])  # UE2 validates first but must not be mutated.
    p = tmp_path/'bad.json';p.write_text(json.dumps(profile))
    cfg = {'ue': {'ues': [dict(ue_id=uid,latency_budget_ms=10) for uid in ('ue_2','ue_6')]}}
    manifest = {'slot_duration_ms':.5}
    before = deepcopy(cfg),deepcopy(manifest)
    with pytest.raises(ValueError): apply_profile(manifest,cfg,p)
    assert (cfg,manifest) == before


def test_direct_phased_sampler_validates_schedule_and_keeps_a_private_copy():
    _, profile = alternating_profile()
    fields = deepcopy(profile['ues']['ue_2'])
    model = build_arrival_model(TrafficProfile(ue_id='ue_2',**fields))
    fields['arrival_phases'][0]['arrival_rate_bps'] = 0
    assert model.sample_bytes(slot_id=1,slot_duration_s=.0005,rng=random.Random()) == 175
    with pytest.raises(ValueError,match='starting at 1'):
        model.sample_bytes(slot_id=0,slot_duration_s=.0005,rng=random.Random())
    with pytest.raises(ValueError):
        build_arrival_model(TrafficProfile(ue_id='ue_2',arrival_model='phased'))

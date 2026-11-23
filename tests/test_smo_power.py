"""UE power is complete-population energy-counter delta per real window."""
import pytest

from smo.runtime import FrameBuilder


def builder(*, window_ms=100):
    config=dict(manifest=dict(slot_duration_ms=1,ues=[dict(ue_id=u)for u in ('a','b')]))
    return FrameBuilder(config,window_ms=window_ms,population=True)


def observe(b,slot,energies,*,reported=('a','b')):
    users=[dict(ue_id=u,cell_id='cell',queue_bytes=0,packet={})for u in reported]
    services={u:dict(ue_energy_total_j=value)for u,value in energies.items()}
    return b.frame(slot,users,services,{},{})['states']


def test_power_uses_window_energy_increment_in_watts_and_retains_cumulative_joules():
    b=builder()
    initial=observe(b,0,dict(a=10,b=20))
    assert initial['kpi:ue_power_w']['value']is None
    first=observe(b,100,dict(a=11,b=22))
    power=first['kpi:ue_power_w']
    assert power['value']==pytest.approx(30)  # (1 + 2) J / 0.1 s
    assert power['by_entity']==pytest.approx(dict(a=10,b=20))
    assert power['window']=='counter_delta' and power['window_start_slot']==0
    assert power['sampled_at_slot']==100
    assert first['kpi:ue_energy_j']['value']==33
    assert first['kpi:ue_energy_j']['window']=='cumulative'
    # A changed window load must not become a since-run-start average.
    second=observe(b,200,dict(a=13,b=26))
    assert second['kpi:ue_power_w']['value']==pytest.approx(60)
    assert second['kpi:ue_power_w']['window_start_slot']==100
    assert second['kpi:ue_energy_j']['value']==39


@pytest.mark.parametrize('offset',[0,1000])
def test_same_power_is_independent_of_old_accumulated_energy(offset):
    b=builder()
    observe(b,0,dict(a=offset,b=offset))
    first=observe(b,100,dict(a=offset+1,b=offset+2))
    second=observe(b,200,dict(a=offset+2,b=offset+4))
    assert first['kpi:ue_power_w']['value']==pytest.approx(30)
    assert second['kpi:ue_power_w']['value']==pytest.approx(30)
    assert second['kpi:ue_energy_j']['value']>first['kpi:ue_energy_j']['value']


@pytest.mark.parametrize('baseline,current',[(None,22),(20,None),(-1,22),(20,float('nan'))])
def test_missing_or_invalid_energy_endpoint_makes_total_power_unknown(baseline,current):
    b=builder();observe(b,0,dict(a=10,b=baseline))
    states=observe(b,100,dict(a=11,b=current))
    assert states['kpi:ue_power_w']['value']is None
    assert states['kpi:ue_power_w']['by_entity']==pytest.approx(dict(a=10))


def test_absent_reported_member_is_not_silently_dropped_from_power_total():
    b=builder();observe(b,0,dict(a=10,b=20))
    states=observe(b,100,dict(a=11,b=22),reported=('a',))
    assert states['kpi:ue_power_w']['value']is None
    assert states['kpi:ue_power_w']['by_entity']==pytest.approx(dict(a=10))


def test_reset_inside_window_is_invalid_even_after_counter_overtakes_baseline():
    b=builder();observe(b,0,dict(a=10,b=20))
    observe(b,50,dict(a=1,b=21))
    states=observe(b,100,dict(a=11,b=22))
    assert states['kpi:ue_power_w']['value']is None
    assert states['kpi:ue_power_w']['by_entity']==pytest.approx(dict(b=20))
    # Once both endpoints and all observations are after the reset, use the
    # valid new counter segment normally rather than invalidating forever.
    recovered=observe(b,150,dict(a=12,b=23))
    assert recovered['kpi:ue_power_w']['value']==pytest.approx(130)


def test_missing_middle_observation_invalidates_window_until_it_has_passed():
    b=builder();observe(b,0,dict(a=10,b=20))
    observe(b,50,dict(a=None,b=21))
    assert observe(b,100,dict(a=11,b=22))['kpi:ue_power_w']['value']is None
    assert observe(b,150,dict(a=12,b=23))['kpi:ue_power_w']['value']is None
    assert observe(b,200,dict(a=13,b=24))['kpi:ue_power_w']['value']==pytest.approx(40)


def test_zero_duration_never_yields_zero_or_fabricated_power():
    b=builder();observe(b,10,dict(a=10,b=20))
    states=observe(b,10,dict(a=11,b=21))
    assert states['kpi:ue_power_w']['value']is None
    assert not states['kpi:ue_power_w']['by_entity']


def test_actual_elapsed_seconds_are_used_for_a_short_warmup_window():
    b=builder(window_ms=100);observe(b,0,dict(a=10,b=20))
    states=observe(b,25,dict(a=10.25,b=20.5))
    assert states['kpi:ue_power_w']['value']==pytest.approx(30)
    assert states['kpi:ue_power_w']['window_start_slot']==0
    assert states['kpi:ue_power_w']['sampled_at_slot']==25

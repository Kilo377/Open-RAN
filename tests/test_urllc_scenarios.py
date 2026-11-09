"""Explicit experiment presets cannot silently alter native or paired loads."""
import argparse
from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments import urllc_scenarios as scenarios


def case():
    users=[dict(ue_id='ue_2',arrival_model='cbr',arrival_rate_bps=2800000,latency_budget_ms=10),
           dict(ue_id='ue_6',arrival_model='cbr',arrival_rate_bps=2200000,latency_budget_ms=10)]
    return {'slot_duration_ms':.5,'ray_solver':{'max_depth':2}}, {
        'ue':{'ues':users},'cell':{'tx_power_dbm':30},'phy':{'unchanged':True}}


def test_general_entry_point_remains_native_and_has_exclusive_scenario_selection():
    parser=argparse.ArgumentParser()
    scenarios.add_scenario_arguments(parser)
    args=parser.parse_args([])
    manifest,configs=case(); before=deepcopy((manifest,configs))
    scenarios.configure_arrivals(manifest,configs,args)
    assert (manifest,configs)==before
    assert parser.parse_args(['--urllc-scenario','burst']).urllc_scenario=='burst'
    with pytest.raises(SystemExit):
        parser.parse_args(['--urllc-scenario','burst','--traffic-profile','other.json'])


@pytest.mark.parametrize('name',scenarios.SCENARIOS)
def test_named_scenarios_preserve_radio_and_original_deadlines(name):
    manifest,configs=case(); before=deepcopy(configs)
    scenarios.apply_named_scenario(manifest,configs,name)
    assert configs['cell']==before['cell'] and configs['phy']==before['phy']
    assert [u['latency_budget_ms'] for u in configs['ue']['ues']]==[10,10]
    assert manifest['ray_solver']=={'max_depth':2}
    assert manifest['urllc_scenario']['name']==name
    models=[u['arrival_model'] for u in configs['ue']['ues']]
    assert models==[{'native':'cbr','burst':'on_off','alternating':'phased'}[name]]*2
    if name=='native':
        assert configs==before and 'traffic_profile' not in manifest


@pytest.mark.parametrize('change',['slot_duration','mean_load','logical_channel_mean','multi_channel'])
def test_pressure_refuses_incompatible_baseline_without_any_mutation(change):
    manifest,configs=case()
    if change=='slot_duration':manifest['slot_duration_ms']=1
    elif change=='mean_load':configs['ue']['ues'][1]['arrival_rate_bps']=1100000
    elif change=='logical_channel_mean':
        configs['ue']['ues'][0]['logical_channels']=[dict(arrival_model='cbr',arrival_rate_bps=1400000)]
    else:configs['ue']['ues'][0]['logical_channels']=[{},{}]
    before=deepcopy((manifest,configs))
    with pytest.raises(ValueError):scenarios.apply_named_scenario(manifest,configs,'burst')
    assert (manifest,configs)==before


def test_programmatic_configuration_cannot_mix_a_named_scenario_and_custom_profile():
    manifest,configs=case(); before=deepcopy((manifest,configs))
    with pytest.raises(ValueError,match='Choose'):
        scenarios.configure_arrivals(manifest,configs,SimpleNamespace(
            urllc_scenario='burst',traffic_profile='other.json'))
    assert (manifest,configs)==before

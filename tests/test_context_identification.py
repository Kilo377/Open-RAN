"""Audited design, actuation and budget guards for multi-context identification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.context_identification import design,make_case,case_command,SCENARIOS
from experiments.rcp_perturbation import validate,vector
from smo.identify import paired_summary

ROOT=Path(__file__).resolve().parents[1]


def test_design_covers_all_rcps_multiple_levels_and_actual_sleep_profiles():
    specs,grids,contrasts=design(ROOT)
    assert len(specs)==28
    assert {g['parameter']for g in grids}|{c['parameter']for c in contrasts}=={
        'ue_offload_fraction','ue_drx_configs','ue_scheduler_weights','scheduler_mode','cell_sleep_mode'}
    assert all(len(g['levels'])>=4 for g in grids)
    assert {s['cell_sleep_mode']for s in specs.values()}=={'active','micro_sleep','light_sleep','deep_sleep'}
    warm=vector(specs['native'],perturbed=False)
    assert all(vector(s,perturbed=False)==warm for s in specs.values())
    assert {g.get('scalar_component')for g in grids if g['parameter']=='ue_drx_configs'}=={'cycle_ms','inactivity_ms'}
    assert len({json.dumps(vector(s,perturbed=True),sort_keys=True)for s in specs.values()})==len(specs)


def test_each_grid_changes_only_its_declared_parameter():
    specs,grids,_=design(ROOT)
    for g in grids:
        center=vector(specs[g['arms'][g['levels'].index(g['reference'])]],perturbed=True)
        for name in g['arms']:
            actual=vector(specs[name],perturbed=True)
            assert {p for p in center if center[p]!=actual[p]} <= {g['parameter']}


def test_pressure_profiles_do_not_change_radio_or_enable_learned_apps():
    case=make_case('burst_2x',123,'native');command=case_command(ROOT,ROOT/'outputs/study',case,ROOT/'spec.json')
    assert command[command.index('--traffic-load')+1]=='2.0'
    assert command[command.index('--urllc-scenario')+1]=='burst'
    assert '--live-xapps'not in command and '--control-mode'not in command
    assert command[command.index('--model')+1]=='fast'
    assert case['scoring_end_ms']-case['scoring_start_ms']==1200
    assert case['expected_task_count']==48 and case['expected_task_cohort_count']==24
    longer=make_case('native_1x',123,'native',duration_s=3.4)
    assert longer['expected_task_count']==112 and longer['cohort_end_ms']==3200


@pytest.mark.parametrize('cycle',[0,3,10.25,400,float('nan')])
def test_invalid_cycle_refused(cycle):
    specs,_,_=design(ROOT)
    with pytest.raises(ValueError):validate({**specs['native'],'drx_cycle_ms':cycle})


def test_sleep_is_not_numerically_encoded_and_cycle_preserves_disabled_ues():
    specs,_,_=design(ROOT)
    s=specs['drx_cycle_ms_80p0'];v=vector(s,perturbed=True)
    assert all(v['ue_drx_configs'][u]['cycle_ms']==80 for u in s['groups']['drx'])
    assert all(v['ue_drx_configs'][u]==s['native_drx'][u] for u in s['groups']['all']if u not in s['groups']['drx'])
    assert vector(specs['ap_deep_sleep'],perturbed=True)['cell_sleep_mode']=='deep_sleep'
    assert paired_summary([1.,2.,3.,4.,5.])['confidence_interval'] is not None

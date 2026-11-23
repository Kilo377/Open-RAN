"""Dimensional checks against canonical coordinates, independent of outcomes."""
from copy import deepcopy

import pytest

from smo.influence_units import projection,align_graph,normalization,frozen_intent_reference
from smo.result_evaluation import evaluate


def scalar(**kw):
    row=dict(parameter='ue_drx_configs',scalar_component='inactivity_ms',kind='range_linear_fit',
        estimate=2.,confidence_interval=[1.,3.],unit='W / ms',x_unit='ms',fit_domain=[1,40])
    row.update(kw);return row


def test_gain_and_input_coordinates_convert_together_without_modifying_original():
    row=scalar(estimate=2000.,confidence_interval=[1000.,3000.],unit='W / s',x_unit='s',fit_domain=[.001,.04]);raw=deepcopy(row)
    p=projection(row,'W')
    assert p['valid'] and p['fields']['estimate']==pytest.approx(2)
    assert p['fields']['confidence_interval']==pytest.approx([1,3])
    assert p['fields']['fit_domain']==pytest.approx([1,40])
    assert row==raw and p['parameter_unit']=='ms'
    p=projection(row,'mW');assert p['fields']['estimate']==pytest.approx(2000)


@pytest.mark.parametrize('changes,reason',[
    ({'unit':'ms / ms'},'incompatible_unit_dimensions'),
    ({'unit':None},'missing_gain_unit'),
    ({'unit':'independent score'},'unknown_unit'),
    ({'unit':'W'},'missing_parameter_unit'),
    ({'x_unit':'s'},'coefficient_and_fit_coordinates_disagree'),
])
def test_unknown_or_inconsistent_units_are_not_inferred(changes,reason):
    result=projection(scalar(**changes),'W')
    assert not result['valid'] and result['reason']==reason


def test_categorical_profiles_are_kpi_differences_not_category_derivatives():
    row=dict(parameter='scheduler_mode',kind='finite_contrast',estimate=.002,unit='s / contrast')
    p=projection(row,'ms');assert p['fields']['estimate']==2 and p['parameter_unit']=='contrast'
    row['unit']='s / weight';assert not projection(row,'ms')['valid']


def test_graph_annotation_preserves_raw_signed_gains():
    row=scalar();g=dict(nodes=[dict(id='kpi:p',kind='kpi',unit='W')],edges=[dict(kind='influences',target='kpi:p',sensitivities=[row])]);raw=deepcopy(g)
    aligned=align_graph(g);assert aligned['edges'][0]['sensitivities'][0]['projection_units']['valid']
    assert g==raw and aligned['edges'][0]['sensitivities'][0]['estimate']==2


def test_risk_and_observed_result_use_the_identical_before_only_reference_scale():
    row=dict(kpi='p',lower=0,upper=9,lower_inclusive=True,upper_inclusive=False,domain=[0,None],anchors=[dict(current=2)])
    b=dict(baseline_reference={'p':10},result_evaluation_scales={})
    frame=dict(states={'kpi:p':dict(value=1000)})
    n=normalization([row],frame,batch=b)['kpis'][0]
    c=dict(rows=[dict(**row,before=dict(value=10,status='above_range'),after=dict(value=8,status='satisfied'))])
    r=evaluate(c,reference_values=b['baseline_reference'],reference_basis='batch_baseline_reference')['metrics'][0]
    assert n['scale']==r['scale']==10 and n['scale_basis']==r['scale_basis']
    assert n['weight']==1
    b['baseline_reference']['p']=0
    assert normalization([row],frame,batch=b)['kpis'][0]['scale']is None


def test_predeployment_scoring_does_not_wait_for_simulation_outcomes():
    row=dict(kpi='p',lower=0,upper=9,anchors=[dict(current=10)])
    n=normalization([row],dict(states={'kpi:p':dict(value=9000)}),batch={})
    assert n['batch_before_reference_pending'] and n['kpis'][0]['scale']==10
    assert not n['observed_after_kpis_used']


@pytest.mark.parametrize('baseline',[0.,.07,.8])
@pytest.mark.parametrize('unit,unit_scale',[('ratio',1.),('fraction',1.),('%',100.)])
def test_urllc_failure_risk_scale_is_percentage_points_even_at_zero(baseline,unit,unit_scale):
    value=baseline*unit_scale
    row=dict(kpi='urllc_failure_fraction',unit=unit,lower=0,upper=.1*unit_scale,
        domain=[0,unit_scale],current=value,anchors=[dict(current=value)])
    frame=dict(states={'kpi:urllc_failure_fraction':dict(value=value)})
    norm=normalization([row],frame)['kpis'][0]
    measured=dict(rows=[dict(**row,before=dict(value=value,status='satisfied'),
        after=dict(value=value,status='satisfied'))])
    result=evaluate(measured)['metrics'][0]
    assert norm['scale']==result['scale']==unit_scale
    assert norm['scale_basis']==result['scale_basis']=='ratio_percentage_points'
    assert norm['reference_value']==value and norm['weight']==1


def test_failure_fraction_does_not_accept_an_event_rate_unit_or_change_event_rate_scaling():
    frame=dict(states={})
    ratio=dict(kpi='urllc_failure_fraction',unit='events/s',current=7)
    bad=normalization([ratio],frame)['kpis'][0]
    assert bad['scale']is None and bad['scale_basis']=='incompatible_ratio_unit'
    events=dict(kpi='urllc_failure_events_per_s',unit='events/s',current=7)
    assert normalization([events],frame)['kpis'][0]['scale']==7
    events['current']=0
    assert normalization([events],frame)['kpis'][0]['scale']is None


def test_frozen_reference_uses_earliest_network_snapshot_not_active_intent_order():
    from itertools import permutations
    anchors=[dict(current=30,intent_id='a',snapshot=dict(time_ms=20,slot=40)),
        dict(current=20,intent_id='z',snapshot=dict(time_ms=10,slot=20)),
        dict(current=25,intent_id='b',snapshot=dict(time_ms=10,slot=20))]
    original=deepcopy(anchors)
    for order in permutations(anchors):
        row=dict(kpi='p',current=999,anchors=list(order))
        assert frozen_intent_reference(row,dict(states={}))==25
        norm=normalization([row],dict(states={}))
        assert norm['kpis'][0]['reference_value']==25 and norm['kpis'][0]['scale']==25
    assert anchors==original


def test_frozen_reference_has_stable_legacy_ties_and_preserves_zero():
    frame=dict(states={'kpi:p':dict(value=99)})
    for anchors in ([dict(current=20),dict(current=10)], [dict(current=10),dict(current=20)]):
        assert frozen_intent_reference(dict(kpi='p',anchors=anchors),frame)==10
    row=dict(kpi='p',current=50,anchors=[dict(current=0,intent_id='early',snapshot=dict(slot=1)),
        dict(current=10,intent_id='late',snapshot=dict(slot=2))])
    assert frozen_intent_reference(row,frame)==0
    assert normalization([row],frame)['kpis'][0]['scale']is None # no invented non-ratio scale
    assert frozen_intent_reference(dict(kpi='p',current=0,anchors=[dict(current=None)]),frame)==0
    assert frozen_intent_reference(dict(kpi='p',current=None,anchors=None),frame)==99


def test_python_and_browser_choose_the_same_frozen_reference():
    import json
    from pathlib import Path
    import shutil
    import subprocess
    node=shutil.which('node')
    if not node:pytest.skip('Node required for cross-language normalization contract')
    frame=dict(states={'kpi:p':dict(value=99)})
    rows=[dict(kpi='p',anchors=[dict(current=30,intent_id='a',snapshot=dict(time_ms=20,slot=40)),
        dict(current=20,intent_id='z',snapshot=dict(time_ms=10,slot=20)),
        dict(current=25,intent_id='b',snapshot=dict(time_ms=10,slot=20))]),
        dict(kpi='p',anchors=[dict(current=20),dict(current=10)]),
        dict(kpi='p',anchors=[dict(current=0,intent_id='zero',snapshot=dict(slot=1)),
            dict(current=10,intent_id='later',snapshot=dict(slot=2))]),
        dict(kpi='p',current=None,anchors=[]),dict(kpi='p',current=0)]
    script="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const d=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(d);
const result=d.rows.map(r=>c.SmoGraph.frozenIntentReference(r,d.frame));
if(before!==JSON.stringify(d))throw Error('Frozen references mutated');
process.stdout.write(JSON.stringify(result));"""
    actual=json.loads(subprocess.check_output([node,'-e',script],text=True,
        cwd=Path(__file__).resolve().parents[1],input=json.dumps(dict(rows=rows,frame=frame))))
    assert actual==[frozen_intent_reference(row,frame) for row in rows]

"""Abstract linear-reference identities, not evidence about a real network.

The synthetic AFTER values below are constructed algebraically from a chosen
joint reference displacement. No simulator, deployment record or learned
outcome label enters this property check.
"""
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import subprocess

import pytest

from smo.influence_units import align_graph,normalization
from smo.intent_quality import assess
from smo.result_evaluation import boundaries,evaluate

ROOT=Path(__file__).resolve().parents[1]
STEP=.125 # exactly representable actuator change; not a fitted coefficient


def abstract_case(specs,case_id):
    """Turn independent, prescribed mathematical KPI displacements into a KG."""
    nodes=[dict(id='xapp:abstract',kind='xapp'),dict(id='rcp:ue_offload_fraction',kind='rcp')]
    edges=[dict(id='control',kind='controls',source='xapp:abstract',target='rcp:ue_offload_fraction')]
    states={'rcp:ue_offload_fraction':dict(by_entity={'u':.5})};after_states={};ranges=[]
    raw_specs=deepcopy(specs)
    for index,spec in enumerate(specs):
        kpi=spec.get('kpi','abstract_power_'+str(index));unit=spec.get('unit','W')
        source_unit='ratio'if kpi=='urllc_failure_fraction'else'W'
        factor=100 if unit=='%' else 1000 if unit=='mW' else 1
        before=spec['before']*factor;delta=spec['delta']*factor
        domain=[v*factor if v is not None else None for v in spec.get('domain',[0,None])]
        low=spec['lower']*factor;high=spec['upper']*factor if spec['upper'] is not None else None
        after=max(domain[0],before+delta)
        if domain[1] is not None:after=min(domain[1],after)
        nodes.append(dict(id='kpi:'+kpi,kind='kpi',unit=unit))
        gain=spec['delta']/STEP
        row=dict(parameter='ue_offload_fraction',kpi=kpi,kind='range_linear_fit',members=['u'],
            parameter_component='mec_workload',population_control_eligible=True,contrast_id='abstract:'+kpi,
            estimate=gain,confidence_interval=[gain,gain],unit=source_unit+' / fraction',x_unit='fraction',fit_domain=[0,1])
        edges.append(dict(id='influence:'+kpi,kind='influences',source='rcp:ue_offload_fraction',target='kpi:'+kpi,sensitivities=[row]))
        states['kpi:'+kpi]=dict(value=before);after_states['kpi:'+kpi]=dict(value=after)
        reference=spec.get('anchor',spec['before'])*factor
        ranges.append(dict(kpi=kpi,unit=unit,domain=domain,lower=low,upper=high,
            lower_inclusive=spec.get('lower_inclusive',True),upper_inclusive=spec.get('upper_inclusive',True),
            current=before,anchors=[dict(intent_id='abstract-intent',current=reference,snapshot=dict(slot=100,time_ms=50))]))
    frame=dict(slot=100,time_ms=50,states=states)
    preview=dict(slot=100,deployment_enabled=False,apps=[dict(app_id='xapp:abstract',status='proposed',
        actions=[dict(parameter='ue_offload_fraction',cell_id='c',ue_id='u',current=.5,proposed=.5+STEP)])])
    group=dict(id='abstract-group',app_ids=['xapp:abstract'])
    comparison=dict(rows=[dict(**r,before=assess(r,frame),after=assess(r,dict(states=after_states)))for r in ranges])
    return dict(id=case_id,graph=align_graph(dict(nodes=nodes,edges=edges)),frame=frame,preview=preview,
        groups=[group],ranges=ranges,normalization=normalization(ranges,frame),comparison=comparison,
        prescribed_displacements=raw_specs)


def generated_cases(count=1000):
    rng=random.Random(20261122)
    cases=[]
    kinds=('increase','decrease','band','point','full_domain')
    for number in range(count):
        specs=[]
        for index in range(1+number%5):
            ratio=index==0 and number%3==0
            if ratio:
                physical_low,physical_high=0,1
                before=rng.randrange(0,17)/16
                delta=rng.randrange(-32,33)/16
                kpi='urllc_failure_fraction';unit='%'if number%2 else'ratio'
                width=1
            else:
                physical_low=2 if (number+index)%7==0 else 0
                physical_high=32 if (number+index)%2 else None
                before=physical_low+rng.randrange(1,121)/4
                if physical_high is not None:before=min(before,physical_high)
                delta=rng.randrange(-160,161)/4
                kpi='abstract_power_'+str(index);unit='mW'if(number+index)%4==0 else'W'
                width=(physical_high-physical_low)if physical_high is not None else 32
            a=physical_low+width*(1+rng.randrange(0,7))/16
            b=physical_low+width*(9+rng.randrange(0,7))/16
            kind=kinds[(number+index)%len(kinds)]
            if kind=='increase':low,high=a,physical_high
            elif kind=='decrease':low,high=physical_low,b
            elif kind=='band':low,high=a,b
            elif kind=='point':low=high=before if number%2 else a
            else:low,high=physical_low,physical_high
            specs.append(dict(kpi=kpi,unit=unit,before=before,delta=delta,lower=low,upper=high,
                domain=[physical_low,physical_high],intended_kind=kind))
        cases.append(abstract_case(specs,'generated:'+str(number)))
    return cases


def strict_cases():
    cases=[]
    for side in ('lower','upper'):
        for other_delta in (-2,0,2):
            spec=dict(before=10,delta=-5 if side=='lower' else 5,lower=5,upper=15,
                lower_inclusive=side!='lower',upper_inclusive=side!='upper')
            other=dict(before=10,delta=other_delta,lower=10,upper=None)
            cases.append(abstract_case([spec,other],f'strict:{side}:{other_delta}'))
    return cases


def run_reference_cases(cases):
    node=shutil.which('node')
    if not node:pytest.skip('Node required for cross-language abstract reference identity')
    payload=[{k:case[k]for k in('graph','frame','preview','groups','ranges','normalization')}for case in cases]
    script="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
for(const f of ['smo/web/app.js','smo/web/goal_obstruction.js','smo/web/intent_obstruction.js'])
 vm.runInContext(fs.readFileSync(f,'utf8'),c);
const cases=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(cases),out=[];
for(const d of cases){const result=c.SmoIntentObstruction.review(d.graph,d.frame,d.groups,d.ranges,d.preview,{normalization:d.normalization});
 const g=result.groups[0];out.push({status:result.status,reason:result.reason||g?.reason||null,
 score:g?.score,reference_effect_score:g?.reference_effect_score,raw_obstruction_strength:g?.raw_obstruction_strength,
 goal_components:g?.goal_components,observed_after_kpis_used:result.observed_after_kpis_used});}
if(before!==JSON.stringify(cases))throw Error('Abstract test inputs mutated');
process.stdout.write(JSON.stringify(out));"""
    return json.loads(subprocess.check_output([node,'-e',script],cwd=ROOT,text=True,input=json.dumps(payload)))


def compare_reference_cases(cases):
    results=run_reference_cases(cases);comparisons=[]
    for case,reference in zip(cases,results):
        observed=evaluate(case['comparison'],ranges=case['ranges'])
        metric=observed['after']['score']
        assert observed['after']['complete'] and reference['status']=='ready', (case['id'],reference)
        assert metric is not None and reference['reference_effect_score']is not None,(case['id'],reference)
        badness=max(0,-metric)/100
        strict_veto=any(m['strict_boundary_regression']for m in observed['metrics'])
        gains=[m['improvement_score']for m in observed['metrics']]
        strict_positive_mean_exception=strict_veto and min(gains)==0 and max(gains)>0
        comparisons.append(dict(id=case['id'],kpi_count=len(case['ranges']),evaluation_metric=metric,
            reference_effect_score=reference['reference_effect_score'],badness=badness,
            reference_badness=reference['raw_obstruction_strength'],
            reference_display_score=reference['score'],
            expected_display_score=100*(badness/(1+badness)),
            primary_absolute_error=abs(badness-reference['raw_obstruction_strength']),
            secondary_absolute_error=abs(metric-reference['reference_effect_score']),
            strict_boundary_regression=strict_veto,strict_positive_mean_exception=strict_positive_mean_exception,
            effective_bound_counts=[len(boundaries(r))for r in case['ranges']],
            physical_saturation_count=sum(g['physical_saturation_applied']for g in reference['goal_components']),
            actual_network_data_used=False))
    return comparisons


def test_abstract_linear_reference_badness_matches_evaluation_for_many_domains_and_goals():
    for row in compare_reference_cases(generated_cases()):
        assert row['primary_absolute_error']<1e-12,row
        assert abs(row['reference_display_score']-row['expected_display_score'])<1e-9,row
        assert row['secondary_absolute_error']<1e-9,row


def test_strict_endpoint_status_is_separate_from_closure_distance_and_risk():
    rows=compare_reference_cases(strict_cases())
    for row in rows:
        assert row['strict_boundary_regression']
        assert row['primary_absolute_error']<1e-12,row
        if row['strict_positive_mean_exception']:
            assert row['evaluation_metric']==0 and row['reference_effect_score']==10
            assert row['secondary_absolute_error']==10
        else:assert row['secondary_absolute_error']<1e-9,row
    assert sum(r['strict_positive_mean_exception']for r in rows)==2


@pytest.mark.parametrize('delta',[2,-2])
def test_old_frozen_anchors_require_shared_scale_for_the_linear_identity(delta):
    case=abstract_case([dict(before=20,anchor=10,delta=delta,lower=20,upper=None)],'old-anchor')
    reference=run_reference_cases([case])[0]
    native=evaluate(case['comparison'],ranges=case['ranges'])
    shared=evaluate(case['comparison'],ranges=case['ranges'],reference_values={'abstract_power_0':10})
    assert reference['reference_effect_score']==pytest.approx(delta*10)
    assert native['after']['score']==pytest.approx(delta*5)
    assert shared['after']['score']==pytest.approx(reference['reference_effect_score'])

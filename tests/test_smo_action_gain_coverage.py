"""A numeric partial reference drive does not certify every action component."""
import json
from pathlib import Path
import subprocess

import pytest


ROOT=Path(__file__).resolve().parents[1]


def project(parameter,rows,base,target):
    script="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const d=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(d);
const result=c.SmoConflictRisk.project(d.edge,d.base,d.target,{states:{}},false);
if(JSON.stringify(d)!==before)throw Error('Projection mutated its input');
process.stdout.write(JSON.stringify(result));"""
    return json.loads(subprocess.check_output(['node','-e',script],cwd=ROOT,text=True,input=json.dumps(dict(
        edge=dict(kind='influences',source='rcp:'+parameter,sensitivities=rows),base=base,target=target))))


def slope(parameter,**changes):
    return {**dict(kind='range_linear_fit',parameter=parameter,contrast_id='known',members=['u'],
        estimate=2,confidence_interval=[1,3],fit_domain=[0,40]),**changes}


def test_unchanged_action_needs_no_calibration_and_has_complete_empty_coverage():
    result=project('ue_offload_fraction',[],{'u':.5},{'u':.5})
    assert result['drive']==0 and result['action_coverage']['complete']
    assert result['action_coverage']['changed_entities']==[]


def test_known_numeric_part_is_retained_but_extra_unmeasured_member_is_explicit():
    result=project('ue_scheduler_weights',[slope('ue_scheduler_weights')],{'u':1,'v':1},{'u':2,'v':2})
    assert result['drive']==2 # Legacy reference projection remains available.
    assert result['action_coverage']==dict(complete=False,changed_entities=['u','v'],covered_entities=['u'],
        uncovered_entities=['v'],uncovered_components={})


def test_only_unmeasured_numeric_member_is_not_a_zero_gain():
    result=project('ue_scheduler_weights',[slope('ue_scheduler_weights')],{'u':1,'v':1},{'u':1,'v':2})
    assert result['drive']is None and not result['action_coverage']['complete']
    assert result['action_coverage']['uncovered_entities']==['v']


def test_measured_competitor_group_fills_only_the_uncovered_member():
    exact=slope('ue_scheduler_weights',members=['u'],estimate=2,confidence_interval=[1,3])
    group=slope('ue_scheduler_weights',members=['u','v'],estimate=6,confidence_interval=[5,7],
        contrast_id='competitors',projection_role='residual_population_reference')
    # Measured dK/d(u,v)=6 and dK/du=2 imply residual dK/dv=4 under
    # the first-order reference. It must reproduce both measured directions.
    for rows in ([exact,group],[group,exact]):
        result=project('ue_scheduler_weights',rows,{'u':1,'v':1},{'u':2,'v':.5})
        assert result['drive']==pytest.approx(0.)
        assert result['action_coverage']['complete']
        component=next(c for c in result['components']if c['calibration_id']=='competitors')
        assert component['residual_members']==['v']
        assert not component['per_ue_causal_derivatives_identified']
        whole=project('ue_scheduler_weights',rows,{'u':1,'v':1},{'u':2,'v':2})
        assert whole['drive']==pytest.approx(6.)
        only_v=project('ue_scheduler_weights',rows,{'u':1,'v':1},{'u':1,'v':.5})
        assert only_v['drive']==pytest.approx(-2.)


def test_group_supplement_does_not_cover_an_unmeasured_population():
    group=slope('ue_scheduler_weights',members=['u','v'],estimate=6,confidence_interval=[5,7],
        projection_role='residual_population_reference')
    result=project('ue_scheduler_weights',[group],{'u':1,'v':1,'new':1},{'u':1,'v':.5,'new':.5})
    assert result['drive']==pytest.approx(-1.5)
    assert result['action_coverage']['uncovered_entities']==['new']
    assert not result['action_coverage']['complete']


def test_overlapping_reference_groups_do_not_invent_a_residual_gain():
    u=slope('ue_scheduler_weights',contrast_id='u')
    uv=slope('ue_scheduler_weights',contrast_id='uv',members=['u','v'],estimate=5)
    whole=slope('ue_scheduler_weights',contrast_id='uvw',members=['u','v','w'],estimate=9,
        projection_role='residual_population_reference')
    result=project('ue_scheduler_weights',[u,uv,whole],
        {'u':1,'v':1,'w':1},{'u':2,'v':2,'w':2})
    assert not result['action_coverage']['complete']
    assert result['action_coverage']['uncovered_entities']==['w']
    assert 'ambiguous_residual_population_basis' in result['notes']


def test_drx_unknown_changed_field_is_not_hidden_by_a_known_scalar_gain():
    base=dict(enabled=True,cycle_ms=40,on_duration_ms=4,inactivity_ms=8)
    result=project('ue_drx_configs',[slope('ue_drx_configs',scalar_component='inactivity_ms')],
        {'u':base},{'u':dict(base,inactivity_ms=7,on_duration_ms=8)})
    assert result['drive']==-2
    assert result['action_coverage']['uncovered_entities']==[]
    assert result['action_coverage']['uncovered_components']=={'u':['on_duration_ms']}
    assert not result['action_coverage']['complete']


def test_full_measured_joint_drx_profile_covers_only_the_matching_fields():
    profile=dict(kind='finite_contrast',parameter='ue_drx_configs',contrast_id='profile',members=['u'],
        estimate=3,confidence_interval=[2,4],**{'from':dict(cycle_ms=40,inactivity_ms=8),'to':dict(cycle_ms=80,inactivity_ms=1)})
    base=dict(enabled=True,cycle_ms=40,on_duration_ms=4,inactivity_ms=8)
    result=project('ue_drx_configs',[profile],{'u':base},{'u':dict(base,cycle_ms=80,inactivity_ms=1)})
    assert result['drive']==3 and result['action_coverage']['complete']
    assert result['action_coverage']['covered_entities']==['u']


def test_categorical_zero_from_unchanged_measured_members_is_not_full_coverage():
    row=dict(kind='finite_contrast',parameter='cell_sleep_mode',contrast_id='light',members=['c'],
        estimate=-3,confidence_interval=[-4,-2],**{'from':'active','to':'light_sleep'})
    result=project('cell_sleep_mode',[row],{'c':'active','new':'active'},{'c':'active','new':'light_sleep'})
    assert result['drive']==0 # Preserve the old partial reference value.
    assert result['action_coverage']['uncovered_entities']==['new']
    assert not result['action_coverage']['complete']


def test_unmeasured_sleep_family_fallback_is_explicitly_incomplete():
    row=dict(kind='finite_contrast',parameter='cell_sleep_mode',contrast_id='micro',members=['c'],
        estimate=-3,confidence_interval=[-4,-2],**{'from':'active','to':'micro_sleep'})
    result=project('cell_sleep_mode',[row],{'c':'active'},{'c':'light_sleep'})
    assert result['drive']==-3 and result['status']=='sleep_family_reference_weight'
    assert result['action_coverage']['unmeasured_profile']
    assert not result['action_coverage']['complete']

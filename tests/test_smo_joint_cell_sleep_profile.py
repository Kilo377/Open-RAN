"""A measured whole-cell profile is one contrast, never per-cell arithmetic."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_smo_goal_obstruction import fixture,strength

ROOT=Path(__file__).resolve().parents[1]
BASE={'cell_0':'active','cell_1':'active','cell_2':'active'}
MIXED={'cell_0':'light_sleep','cell_1':'deep_sleep','cell_2':'light_sleep'}


def edge(*,uniform=True,gain=-4.5):
    rows=[]
    if uniform:
        for mode,value in [('micro_sleep',10),('light_sleep',20),('deep_sleep',30)]:
            rows.append(dict(kind='finite_contrast',parameter='cell_sleep_mode',contrast_id='uniform:'+mode,
                parameter_component='all_cells',members=list(BASE),estimate=value,confidence_interval=[value,value],
                unit='W / contrast',**{'from':'active','to':mode}))
    rows.append(dict(kind='finite_contrast',parameter='cell_sleep_mode',contrast_id='exact_mixed_profile',
        parameter_component='all_cells',members=list(BASE),estimate=gain,confidence_interval=[gain,gain],
        unit='W / contrast',**{'from':deepcopy(BASE),'to':deepcopy(MIXED)}))
    return dict(id='sleep:k',kind='influences',source='rcp:cell_sleep_mode',target='kpi:k',sensitivities=rows)


def project(e,base=BASE,target=MIXED):
    node=shutil.which('node')
    if not node:pytest.skip('Node required for production profile projection')
    script="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const d=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(d);
const r=c.SmoConflictRisk.project(d.edge,d.base,d.target,{states:{'rcp:cell_sleep_mode':{by_entity:d.base}}},false);
if(before!==JSON.stringify(d))throw Error('Reference profile input mutated');
process.stdout.write(JSON.stringify(r));"""
    return json.loads(subprocess.check_output([node,'-e',script],cwd=ROOT,text=True,
        input=json.dumps(dict(edge=e,base=base,target=target))))


def test_exact_whole_profile_takes_precedence_and_is_used_once():
    r=project(edge())
    assert r['drive']==-4.5 and r['direction']==-1
    assert r['calibrations']==['exact_mixed_profile']
    assert len(r['components'])==1 and r['components'][0]['parameter_step']==1
    assert r['status']=='joint_profile_reference_projection'


def test_exact_reverse_changes_sign_once():
    r=project(edge(),MIXED,BASE)
    assert r['drive']==4.5 and r['components'][0]['parameter_step']==-1
    assert r['calibrations']==['exact_mixed_profile']


@pytest.mark.parametrize('base,target',[
    (BASE,dict(BASE,cell_0='light_sleep')),
    (BASE,{c:'light_sleep'for c in BASE}),
    (dict(BASE,cell_3='active'),dict(MIXED,cell_3='active')),
    ({c:v for c,v in BASE.items()if c!='cell_2'},{c:v for c,v in MIXED.items()if c!='cell_2'}),
    (MIXED,dict(MIXED,cell_1='active')),
])
def test_partial_superset_or_different_profiles_never_borrow_the_joint_gain(base,target):
    r=project(edge(),base,target)
    assert 'exact_mixed_profile'not in r['calibrations']
    assert 'joint_sleep_profile_not_applicable'in r['notes']
    if base==BASE and target==dict(BASE,cell_0='light_sleep'):
        assert r['drive']==pytest.approx(20/3)


def test_joint_only_measurement_cannot_identify_a_partial_profile():
    r=project(edge(uniform=False),BASE,dict(BASE,cell_0='light_sleep'))
    assert r['drive']is None and not r['calibrations']


@pytest.mark.parametrize('change',['missing_members','duplicate_members','mismatched_keys','invalid_mode'])
def test_joint_profile_requires_explicit_complete_cell_members_and_valid_modes(change):
    e=edge(uniform=False);row=e['sensitivities'][0]
    if change=='missing_members':row.pop('members')
    elif change=='duplicate_members':row['members']=['cell_0','cell_1','cell_1']
    elif change=='mismatched_keys':row['from'].pop('cell_2')
    elif change=='invalid_mode':row['to']['cell_1']='invented_depth'
    r=project(e)
    assert r['drive']is None and not r['calibrations']


def test_duplicate_equal_contrast_is_not_added_twice_and_conflicts_abstain():
    e=edge(uniform=False);duplicate=deepcopy(e['sensitivities'][0]);duplicate['contrast_id']='second_source'
    e['sensitivities'].append(duplicate)
    r=project(e);assert r['drive']==-4.5 and len(r['components'])==1
    duplicate['estimate']=-9;duplicate['confidence_interval']=[-9,-9]
    r=project(e)
    assert r['drive']is None and 'ambiguous_joint_sleep_profile_reference'in r['notes']


@pytest.mark.parametrize('gain,interval,note',[(-4.5,[-10,10],'uncertain_gain_direction'),(0,[0,0],'observed_zero_reference_weight')])
def test_joint_uncertainty_or_observed_zero_does_not_restore_constituent_estimates(gain,interval,note):
    e=edge(gain=gain);e['sensitivities'][-1]['confidence_interval']=interval
    r=project(e)
    assert r['drive']==gain and r['calibrations']==['exact_mixed_profile']
    assert note in r['notes']


def test_joint_profile_retains_unit_contract_and_changes_real_obstruction_path():
    g,f,c,p,_=fixture();g['edges'].append(edge())
    f['states']['rcp:cell_sleep_mode']['by_entity']=deepcopy(BASE)
    c['groups']=[dict(id='joint',app_ids=['a'])]
    p['apps'][0]['actions']=[dict(parameter='cell_sleep_mode',cell_id=cid,ue_id=None,current='active',proposed=mode)
        for cid,mode in MIXED.items()]
    q=strength(g,f,c,p)['groups'][0]
    assert q['score']==pytest.approx(100*.45/1.45)
    chain=q['actual_outcomes'][0]['goals'][0]['chains'][0]
    assert chain['calibrations']==['exact_mixed_profile']
    assert chain['unit_contracts'][0]['valid']

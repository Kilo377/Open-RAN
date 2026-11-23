"""Nonempty controls cannot precede six fully pinned prospective score grids."""
import json
from pathlib import Path
import pytest

from experiments import precontrol_admission as admission


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return dict(path=str(path),sha256=admission.sha(path))


def prepare(tmp_path,monkeypatch):
    root=tmp_path/'v2';pin='p'*64;source='s'*64;archive='a'*64;blocks=[];origins=[]
    for seed in(20261201,20261202):
        wid='agx'+str(seed%2+1);cases=[]
        for scene in('native_1x','steady_2x','burst_2x'):
            key=scene+'_s'+str(seed);oid='origin:'+key;job='job:'+key
            cases.append(dict(arm='OFF',origin_key=key,id=job,seed=seed))
            write(root/'workers'/wid/'origins'/(key+'.json'),dict(origin_id=oid))
            prov=dict(physical_plan_sha256=pin,source_archive_sha256=archive,scoring_origin_id=oid,scoring_origin_job_id=job,worker_id=wid,seed=seed,after_kpis_used=False)
            inputs=dict(stage='predeployment_inputs',after_values_included=False,cases=[dict(case_id=f'{key}:{arm}:{intent}',group_id=str(arm),intent_id=str(intent),provenance=prov)for arm in range(20)for intent in range(15)])
            refs=[]
            for category,methods in(('original',admission.ORIGINAL_METHODS),('profile',{admission.PROFILE_METHOD})):
                directory=root/category/key;inputref=write(directory/'inputs.json',inputs);scores={}
                for method in methods:
                    module=directory/'engine.js';module.write_text('/* fixed test scoring engine */')
                    body=dict(method=method,after_values_used=False,stage='scores_before_label_join',input_sha256=inputref['sha256'],
                        scores=[dict(case_id=c['case_id'])for c in inputs['cases']],module_sha256={'engine':admission.sha(module)},archived_modules={'engine':str(module)})
                    scores[method]={**write(directory/(method+'.json'),body),'count':300}
                seal=dict(schema='oran.prospective-score-seal.v1',origin_key=key,blocks=[dict(origin_key=key,origin_id=oid,off_job_id=job)],
                    after_values_used=False,nonempty_arm_started_before_scoring=False,gate_opened=False,algorithm_tuned_to_test_labels=False,
                    cases=300,physical_arms=20,intents=15,input_path=inputref['path'],input_sha256=inputref['sha256'],scores=scores)
                refs.append(write(directory/'prediction-seal.json',seal))
            blocks.append(dict(origin_key=key,**refs[0]));origins.append(dict(origin_key=key,origin_id=oid,worker_id=wid,off_job_id=job,original_seal=refs[0],profile_seal=refs[1]))
        write(root/'workers'/wid/'plan.json',dict(physical_plan_sha256=pin,source_sha256=source,source_archive_sha256=archive,cases=cases))
    original=write(root/'predictions-v2/six-block-score-seal.json',dict(blocks=blocks));monkeypatch.setattr(admission,'ORIGINAL_SET_SHA256',original['sha256'])
    write(root/'precontrol-admission.json',dict(schema='oran.precontrol-admission.v1',source_version='v2',physical_plan_sha256=pin,
        source_sha256=source,source_archive_sha256=archive,after_values_used=False,origins=origins))
    return root,pin


def test_all_four_methods_need_pinned_six_origin_scores_before_admission(tmp_path,monkeypatch):
    root,pin=prepare(tmp_path,monkeypatch);proof=admission.validate_full_admission(root,pin)
    assert proof['origins']==6 and len(proof['methods'])==4


@pytest.mark.parametrize('mutation',['missing_gate','missing_origin','score_bytes','input_bytes','wrong_source','old_version','original_set'])
def test_stale_or_partial_prospective_admission_is_rejected(tmp_path,monkeypatch,mutation):
    root,pin=prepare(tmp_path,monkeypatch);path=root/'precontrol-admission.json';gate=admission.read(path)
    if mutation=='missing_gate':path.unlink()
    elif mutation=='missing_origin':gate['origins'].pop();write(path,gate)
    elif mutation=='wrong_source':gate['source_sha256']='different';write(path,gate)
    elif mutation=='old_version':gate['source_version']='v1';write(path,gate)
    elif mutation=='original_set':(root/'predictions-v2/six-block-score-seal.json').write_text('{}')
    else:
        seal=admission.read(gate['origins'][0]['profile_seal']['path'])
        target=seal['input_path']if mutation=='input_bytes'else next(iter(seal['scores'].values()))['path']
        Path(target).write_text('{}')
    with pytest.raises(ValueError):admission.validate_full_admission(root,pin)


def test_changed_physical_provenance_fails_even_with_resealed_input_and_score_files(tmp_path,monkeypatch):
    root,pin=prepare(tmp_path,monkeypatch);gatepath=root/'precontrol-admission.json';gate=admission.read(gatepath)
    ref=gate['origins'][0]['profile_seal'];seal=admission.read(ref['path']);inputs=admission.read(seal['input_path'])
    inputs['cases'][0]['provenance']['source_archive_sha256']='wrong';new=write(Path(seal['input_path']),inputs);seal['input_sha256']=new['sha256']
    for score in seal['scores'].values():
        body=admission.read(score['path']);body['input_sha256']=new['sha256'];score.update(write(Path(score['path']),body))
    gate['origins'][0]['profile_seal']=write(Path(ref['path']),seal);write(gatepath,gate)
    with pytest.raises(ValueError,match='physical source/origin'):admission.validate_full_admission(root,pin)

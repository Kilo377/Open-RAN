"""Fail-closed prospective score admission; no validation outcomes are read."""
import hashlib
import json
from pathlib import Path

ORIGINAL_SET_SHA256='b714292b662b5f12f88949982b9b913f8f002c1a6eff650508b9758db14e6a22'
ORIGINAL_METHODS={'goal_obstruction_v2','net_intent_obstruction_v1','temporal_net_intent_obstruction_v1'}
PROFILE_METHOD='net_profile_gain_obstruction_v1'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(ref,root):
    path=Path(ref['path']).resolve()
    if not path.is_relative_to(root)or not path.is_file()or sha(path)!=ref['sha256']:
        raise ValueError('Precontrol evidence file pin/path differs: '+str(path))
    return read(path)


def _seal(ref,root,expected,methods):
    seal=pinned(ref,root)
    if (seal.get('schema')!='oran.prospective-score-seal.v1'or seal.get('origin_key')!=expected['origin_key']
        or seal.get('after_values_used')is not False or seal.get('nonempty_arm_started_before_scoring')is not False
        or seal.get('gate_opened')is not False or seal.get('algorithm_tuned_to_test_labels')is not False
        or seal.get('cases')!=300 or seal.get('physical_arms')!=20 or seal.get('intents')!=15):
        raise ValueError('Prospective score seal protocol differs')
    blocks=seal.get('blocks',[])
    if len(blocks)!=1 or any(blocks[0].get(k)!=expected[k]for k in('origin_key','origin_id','off_job_id')):
        raise ValueError('Prospective score origin differs')
    inputs=pinned(dict(path=seal['input_path'],sha256=seal['input_sha256']),root)
    if inputs.get('after_values_included')is not False or inputs.get('stage')!='predeployment_inputs' or len(inputs.get('cases',[]))!=300:
        raise ValueError('Prospective inputs contain labels or incomplete cases')
    ids=set();arms={}
    for case in inputs['cases']:
        provenance=case['provenance'];ident=case['case_id'];ids.add(ident)
        for field,wanted in dict(physical_plan_sha256=expected['physical_plan_sha256'],source_archive_sha256=expected['source_archive_sha256'],
            scoring_origin_id=expected['origin_id'],scoring_origin_job_id=expected['off_job_id'],worker_id=expected['worker_id'],seed=expected['seed'],after_kpis_used=False).items():
            if provenance.get(field)!=wanted:raise ValueError('Prospective input physical source/origin differs: '+field)
        arms.setdefault(case['group_id'],set()).add(case['intent_id'])
    if len(ids)!=300 or len(arms)!=20 or any(len(v)!=15 for v in arms.values()):raise ValueError('Frozen 20×15 prospective grid incomplete')
    if not methods<=set(seal.get('scores',{})):raise ValueError('A preregistered method score is missing')
    for method in methods:
        score_ref=seal['scores'][method];scores=pinned(score_ref,root)
        if (score_ref.get('count')!=300 or scores.get('method')!=method or scores.get('after_values_used')is not False
            or scores.get('stage')!='scores_before_label_join' or scores.get('input_sha256')!=seal['input_sha256']
            or len(scores.get('scores',[]))!=300):raise ValueError('Prospective numeric/unknown score record differs')
        if {row['case_id']for row in scores['scores']}!=ids:raise ValueError('Prospective scores do not cover the frozen input grid')
        if not scores.get('module_sha256'):raise ValueError('Archived scoring implementation pins are missing')
        for source,pin in scores.get('module_sha256',{}).items():
            archived=Path(scores['archived_modules'][source]).resolve()
            if not archived.is_relative_to(root)or not archived.is_file()or sha(archived)!=pin:
                raise ValueError('Archived scoring implementation pin differs')
    return ids


def validate_full_admission(artifact,physical_pin):
    """Validate the root's six-origin certificate before allowing any controls."""
    root=Path(artifact).resolve();path=root/'precontrol-admission.json'
    if root.name!='v2'or not path.is_file():raise ValueError('V2 full admission requires precontrol-admission.json')
    gate=read(path)
    if (gate.get('schema')!='oran.precontrol-admission.v1'or gate.get('source_version')!='v2'
        or gate.get('physical_plan_sha256')!=physical_pin or gate.get('after_values_used')is not False):
        raise ValueError('Precontrol admission identity differs')
    original_set=root/'predictions-v2/six-block-score-seal.json'
    if sha(original_set)!=ORIGINAL_SET_SHA256:raise ValueError('Original three-method score set was modified')
    original={b['origin_key']:b for b in read(original_set)['blocks']}
    expected={}
    plans={}
    for worker in(root/'workers').iterdir():
        if not(worker/'plan.json').is_file():continue
        plan=read(worker/'plan.json');plans[worker.name]=plan
        if (plan['physical_plan_sha256']!=physical_pin or plan['source_sha256']!=gate.get('source_sha256')
            or plan['source_archive_sha256']!=gate.get('source_archive_sha256')):
            raise ValueError('Precontrol worker source differs')
        for case in plan['cases']:
            if case['arm']!='OFF':continue
            proof=read(worker/'origins'/(case['origin_key']+'.json'))
            expected[case['origin_key']]=dict(origin_key=case['origin_key'],origin_id=proof['origin_id'],off_job_id=case['id'],
                worker_id=worker.name,seed=case['seed'],physical_plan_sha256=physical_pin,source_archive_sha256=plan['source_archive_sha256'])
    origins=gate.get('origins',[])
    if len(expected)!=6 or len(origins)!=6 or {o['origin_key']for o in origins}!=set(expected):raise ValueError('All six origins must be sealed')
    for item in origins:
        wanted=expected[item['origin_key']]
        if any(item.get(k)!=wanted[k]for k in('origin_key','origin_id','worker_id','off_job_id')):raise ValueError('Admission origin metadata differs')
        if item['original_seal']!={k:original[item['origin_key']][k]for k in('path','sha256')}:raise ValueError('Original method seal reference differs')
        original_ids=_seal(item['original_seal'],root,wanted,ORIGINAL_METHODS)
        profile_ids=_seal(item['profile_seal'],root,wanted,{PROFILE_METHOD})
        if original_ids!=profile_ids:raise ValueError('Profile comparator changed the frozen prospective test cases')
    return dict(path=str(path),sha256=sha(path),source_sha256=gate['source_sha256'],source_archive_sha256=gate['source_archive_sha256'],
        physical_plan_sha256=physical_pin,origins=6,methods=sorted(ORIGINAL_METHODS|{PROFILE_METHOD}))

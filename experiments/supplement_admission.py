"""Strict BEFORE-only admission for the separate two-arm prospective supplement.

Only immutable plans, certified OFF metadata, score inputs and archived scores
are opened. Label-source files are planned path/window metadata only. No
trace, completed control run or AFTER metric is an input to this gate.
"""
from copy import deepcopy
from pathlib import Path

from experiments.precontrol_admission import ORIGINAL_METHODS, ORIGINAL_SET_SHA256, PROFILE_METHOD, pinned, read, sha
from experiments.quantification_supplement import assignment, verify_plan

METHODS = ORIGINAL_METHODS | {PROFILE_METHOD}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(path):
    return dict(path=str(path), sha256=sha(path))


def _materialize(data, case):
    context = data.get('contexts', {}).get(case.get('context_id'), {})
    return {**context, **case, 'options':{**context.get('options', {}), **case.get('options', {})}}


def _input(ref, root, count):
    data = pinned(ref, root)
    cases = data.get('cases', [])
    _require(data.get('stage') == 'predeployment_inputs' and data.get('after_values_included') is False
             and len(cases) == count and len({c['case_id'] for c in cases}) == count,
             'Supplement input grid/BEFORE-only protocol differs')
    return data


def _off_templates(data, origin):
    cases = [c for c in data['cases'] if c.get('app_ids') == []]
    _require(len(cases) == 15 and len({c['intent_id'] for c in cases}) == 15,
             'Parent input must contain the same fifteen OFF intent templates')
    _require(all(c['provenance'].get('job_id') == origin['off_job_id'] for c in cases),
             'Parent OFF template origin differs')
    return {c['intent_id']:_materialize(data, c) for c in cases}


def _check_case(data, case, template, origin, job, plan):
    value = _materialize(data, case)
    expected_id = origin['origin_key']+':'+job['id']+':'+case['intent_id']
    _require(case['case_id'] == expected_id and case.get('group_id') == job['group_id']
             and case.get('app_ids') == job['app_ids'], 'Supplement app/group/job identity differs')
    _require(case.get('scene') == origin['scene'], 'Supplement case scene differs')
    expected_provenance = deepcopy(template['provenance'])
    expected_provenance.update(job_id=job['id'], supplement_plan_sha256=plan['plan_sha256'],
                               grid_partition='prospective_supplement')
    # The batch label may change; physical source/origin and every other field
    # must remain exactly as they were before any supplement was proposed.
    expected_provenance['batch_id'] = value['provenance'].get('batch_id')
    _require(value['provenance'] == expected_provenance, 'Supplement original provenance changed')
    for key, expected in dict(physical_plan_sha256=plan['parent_physical_plan_sha256'],
        source_archive_sha256=plan['source_archive_sha256'], scoring_origin_id=origin['origin_id'],
        scoring_origin_job_id=origin['off_job_id'], worker_id=origin['worker_id'], seed=origin['seed'],
        after_kpis_used=False).items():
        _require(value['provenance'].get(key) == expected, 'Supplement source/origin differs: '+key)
    protocol = deepcopy(template['protocol'])
    protocol.update(primary_validation_eligible=True, actual_arm_origin_guard_required=True,
                    actual_arm_origin_guard_pending=True)
    protocol['reasons'] = [r for r in protocol.get('reasons', []) if r != 'OFF_diagnostic']
    _require(value.get('protocol') == protocol, 'Supplement inherited OFF diagnostic or changed protocol')
    _require(isinstance(value.get('candidate'), dict) and value['candidate'].get('pool') != 'off_diagnostic',
             'Supplement candidate flags still describe OFF')
    _require(isinstance(value.get('independence_group'), str) and value['independence_group']
             and value['independence_group'] != template.get('independence_group'),
             'Supplement reused OFF independence identity')
    mutable = {'case_id', 'group_id', 'app_ids', 'candidate', 'independence_group',
               'provenance', 'protocol', 'context_id'}
    _require({k:v for k,v in value.items() if k not in mutable}
             == {k:v for k,v in template.items() if k not in mutable},
             'Supplement frozen ranges/state/normalization/history changed')
    return value


def _check_scores(ref, root, method, input_ref, case_map, parent_score_ref, label_pin):
    score = pinned(ref, root)
    parent = pinned(parent_score_ref, root)
    rows = score.get('scores', [])
    _require(ref.get('count') == 30 and score.get('method') == method
             and score.get('stage') == 'scores_before_label_join' and score.get('after_values_used') is False
             and score.get('input_sha256') == input_ref['sha256']
             and Path(score.get('input_path', '')).resolve() == Path(input_ref['path']).resolve()
             and score.get('label_sources_sha256') == label_pin
             and len(rows) == 30 and {r['case_id'] for r in rows} == set(case_map),
             'Supplement four-method score grid/input pin differs')
    modules = score.get('module_sha256', {})
    _require(bool(modules) and modules == parent.get('module_sha256'),
             'Supplement scoring implementation differs from sealed parent')
    _require(score.get('engine') == parent.get('engine')
             and score.get('dispatch_method') == parent.get('dispatch_method'),
             'Supplement scoring engine/dispatch differs from sealed parent')
    for source, pin in modules.items():
        archive = score.get('archived_modules', {}).get(source)
        _require(isinstance(archive, str), 'Supplement archived scoring implementation missing')
        path = Path(archive).resolve()
        _require(path.is_relative_to(root) and path.is_file() and sha(path) == pin,
                 'Supplement archived scoring implementation pin differs')
    for row in rows:
        case = case_map[row['case_id']]
        _require(all(row.get(k) == case.get(k) for k in ('intent_id', 'candidate', 'protocol', 'provenance',
                     'cohort_id', 'independence_group', 'scene')), 'Supplement score/input metadata differs')
        groups = row.get('result', {}).get('groups', [])
        identified = len(groups) == 1 and groups[0].get('group_id') == case['group_id'] and groups[0].get('app_ids') == case['app_ids']
        unavailable = not groups and row.get('result', {}).get('status') == 'unavailable'
        _require(identified or unavailable, 'Supplement score result identifies another arm')
    return score


def _seal(ref, root, origin, jobs, plan, original_ref, profile_ref):
    seal = pinned(ref, root)
    _require(seal.get('schema') == 'oran.prospective-supplement-score-seal.v1'
             and seal.get('source_version') == 'v2' and seal.get('origin_key') == origin['origin_key']
             and seal.get('supplement_plan_sha256') == plan['plan_sha256']
             and seal.get('parent_physical_plan_sha256') == plan['parent_physical_plan_sha256']
             and seal.get('source_sha256') == plan['source_sha256']
             and seal.get('source_archive_sha256') == plan['source_archive_sha256']
             and seal.get('cases') == 30 and seal.get('physical_arms') == 2 and seal.get('intents') == 15,
             'Supplement prospective seal identity/grid differs')
    _require(all(seal.get(k) is False for k in ('after_values_used', 'nonempty_arm_started_before_scoring',
                 'gate_opened', 'algorithm_tuned_to_test_labels')), 'Supplement score not sealed before controls')
    _require(seal.get('base_seal') == original_ref and seal.get('profile_parent_seal') == profile_ref,
             'Supplement sealed parent references differ')
    _require(seal.get('cases_manifest') == _ref(root/'supplement/cases-manifest.json'),
             'Supplement score seal used another physical cases manifest')
    blocks = seal.get('blocks', [])
    _require(len(blocks) == 1 and blocks[0].get('arms') == 2 and blocks[0].get('intents') == 15
             and all(blocks[0].get(k) == origin[k] for k in ('origin_key', 'origin_id', 'off_job_id')),
             'Supplement prospective block origin differs')
    parents = {False:pinned(original_ref, root), True:pinned(profile_ref, root)}
    parent_inputs = {variant:_input(dict(path=p['input_path'], sha256=p['input_sha256']), root, 300)
                     for variant,p in parents.items()}
    templates = {variant:_off_templates(d, origin) for variant,d in parent_inputs.items()}
    _require(set(templates[False]) == set(templates[True]), 'Parent profile intent catalog differs')
    method_inputs = seal.get('method_inputs', {})
    _require(set(method_inputs) == METHODS and set(seal.get('scores', {})) == METHODS,
             'All four preregistered supplement methods must be sealed')
    main_ref = dict(path=seal['input_path'], sha256=seal['input_sha256'])
    _require(all(method_inputs[m] == main_ref for m in ORIGINAL_METHODS), 'Original methods use different supplement inputs')
    materialized, ids = {}, None
    for variant, input_ref in ((False,main_ref), (True,method_inputs[PROFILE_METHOD])):
        data = _input(input_ref, root, 30)
        _require(data.get('graphs') == parent_inputs[variant].get('graphs'), 'Supplement graph differs from sealed parent')
        case_map, arms = {}, {}
        by_group = {j['group_id']:j for j in jobs}
        for case in data['cases']:
            _require(case.get('group_id') in by_group and case.get('intent_id') in templates[variant],
                     'Supplement case is outside the registered arm/intent grid')
            job = by_group[case['group_id']]
            value = _check_case(data, case, templates[variant][case['intent_id']], origin, job, plan)
            case_map[case['case_id']] = value
            arms.setdefault(case['group_id'], set()).add(case['intent_id'])
        _require(set(arms) == set(by_group) and all(v == set(templates[variant]) for v in arms.values()),
                 'Supplement two-by-fifteen grid incomplete')
        _require(ids is None or ids == set(case_map), 'Supplement profile changed case IDs')
        ids = set(case_map);materialized[variant] = case_map
        if variant:
            for ident, case in case_map.items():
                other = materialized[False][ident]
                _require(all(case.get(k) == other.get(k) for k in ('ranges', 'candidate', 'protocol', 'provenance',
                             'timing', 'frame', 'preview', 'cohort_id', 'independence_group')),
                         'Supplement profile changed shared BEFORE geometry/flags')
    for method in METHODS:
        parent = parents[method == PROFILE_METHOD]
        _check_scores(seal['scores'][method], root, method, method_inputs[method],
                      materialized[method == PROFILE_METHOD], parent['scores'][method], seal['label_sources']['sha256'])
    labels = pinned(seal['label_sources'], root)
    profile_labels = pinned(seal['profile_label_sources'], root)
    _require(labels.get('after_values_included') is False
             and seal['label_sources']['sha256'] == seal['profile_label_sources']['sha256']
             and labels == profile_labels and set(labels.get('sources', {})) == ids,
             'Supplement planned label-source metadata differs')
    base = parents[False]
    old_labels = pinned(dict(path=str(Path(base['input_path']).with_name('label-sources.json')),
                             sha256=base['label_sources_sha256']), root)['sources']
    by_group = {j['group_id']:j for j in jobs}
    for ident, case in materialized[False].items():
        job = by_group[case['group_id']]
        template = templates[False][case['intent_id']]
        source = deepcopy(old_labels[template['case_id']])
        run = root/'supplement/workers'/job['worker_id']/'jobs'/job['id']/'data/run'
        source.update(run_path=str(run), seal_path=str(run.parents[3]/'checkpoints'/(job['id']+'.json')),
            requires_actual_origin_guard=True, protocol=case['protocol'], candidate=case['candidate'],
            provenance=case['provenance'], independence_group=case['independence_group'])
        _require(labels['sources'][ident] == source and source.get('expected_origin_id') == origin['origin_id'],
                 'Supplement planned label mapping changed origin/window/job')
    return ids


def validate_full_admission(artifact, supplement_pin):
    """Validate six independently sealed origins before the supplement runs."""
    artifact = Path(artifact).resolve();root = artifact.parent
    _require(artifact.name == 'supplement' and root.name == 'v2', 'Separate V2 supplement root required')
    path = artifact/'precontrol-admission.json';gate = read(path)
    plan = verify_plan(read(artifact/'supplement-plan.json'), supplement_pin)
    publication = read(artifact/'source-publication.json')
    runner_pin = sha(artifact/'runner.py')
    expected = dict(source_version='v2', supplement_plan_sha256=supplement_pin,
        parent_physical_plan_sha256=plan['parent_physical_plan_sha256'], source_sha256=plan['source_sha256'],
        source_archive_sha256=plan['source_archive_sha256'], runner_sha256=runner_pin)
    _require(gate.get('schema') == 'oran.supplement-precontrol-admission.v1' and gate.get('after_values_used') is False
             and all(gate.get(k) == v for k,v in expected.items()), 'Supplement admission source/plan/runner differs')
    _require(all(publication.get(k) == v for k,v in expected.items() if k != 'source_version')
             and publication.get('original_source_modified') is False, 'Supplement source publication differs')
    old_set = plan['original_score_set']
    _require(old_set['sha256'] == ORIGINAL_SET_SHA256, 'Original score set pin differs')
    original_blocks = {b['origin_key']:b for b in pinned(old_set, root)['blocks']}
    parent_gate = read(root/'precontrol-admission.json')
    _require(parent_gate.get('schema') == 'oran.precontrol-admission.v1'
             and parent_gate.get('physical_plan_sha256') == plan['parent_physical_plan_sha256']
             and parent_gate.get('after_values_used') is False,
             'Original four-method prospective admission is missing')
    parent_origins = {o['origin_key']:o for o in parent_gate['origins']}
    schedule = assignment(plan, supplement_pin);all_jobs = []
    for worker, jobs in schedule.items():
        worker_plan = read(artifact/'workers'/worker/'plan.json')
        _require(worker_plan.get('worker_id') == worker and worker_plan.get('supplement') == plan
                 and worker_plan.get('supplement_plan_sha256') == supplement_pin
                 and worker_plan.get('runner_sha256') == runner_pin and worker_plan.get('cases') == jobs,
                 'Supplement worker plan/runner/assignments differ')
        all_jobs.extend({**j,'worker_id':worker} for j in jobs)
    manifest = read(artifact/'cases-manifest.json')
    _require(manifest.get('schema') == 'oran.prospective-supplement-cases.v1'
             and manifest.get('source_version') == 'v2' and manifest.get('after_values_used') is False
             and manifest.get('supplement_plan_sha256') == supplement_pin
             and manifest.get('parent_physical_plan_sha256') == plan['parent_physical_plan_sha256'],
             'Supplement cases manifest protocol differs')
    physical_cases = manifest.get('cases', [])
    _require(len(physical_cases) == 12 and len({c['id'] for c in physical_cases}) == 12,
             'Supplement physical case grid incomplete')
    by_id = {j['id']:j for j in all_jobs}
    for case in physical_cases:
        job = by_id.get(case['id'])
        _require(job is not None and all(case.get(k) == v for k,v in job.items()),
                 'Supplement physical case identity differs')
        wanted_run = artifact/'workers'/job['worker_id']/'jobs'/job['id']/'data/run'
        _require(Path(case['run_path']).resolve() == wanted_run
                 and case.get('remote_run_path') == '/dev/shm/oran_quantification_20261122/source-v2/outputs/prospective_supplement/jobs/'+job['id']+'/data/run'
                 and all(case.get(k) == expected[k] for k in ('source_sha256','source_archive_sha256','runner_sha256')),
                 'Supplement physical case path/source differs')
    origins = {o['origin_key']:o for o in plan['origins']}
    admitted = gate.get('origins', [])
    _require(len(admitted) == 6 and {o['origin_key'] for o in admitted} == set(origins),
             'All six supplement origins must be sealed')
    all_ids = set()
    for item in admitted:
        origin = origins[item['origin_key']]
        _require(all(item.get(k) == origin[k] for k in ('origin_key','origin_id','worker_id','off_job_id')),
                 'Supplement admission origin differs')
        proof = pinned(origin['origin_proof'], root)
        _require(proof.get('origin_id') == origin['origin_id'], 'Supplement certified OFF proof differs')
        pinned(origin['before_preview'], root)
        original_ref = {k:original_blocks[origin['origin_key']][k] for k in ('path','sha256')}
        parent = parent_origins[origin['origin_key']]
        _require(parent['original_seal'] == original_ref, 'Original parent score seal was replaced')
        jobs = [j for j in all_jobs if j['origin_key'] == origin['origin_key']]
        ids = _seal(item['prediction_seal'], root, origin, jobs, plan, original_ref, parent['profile_seal'])
        _require(not all_ids.intersection(ids), 'Supplement origins reused score case IDs')
        all_ids.update(ids)
    return dict(path=str(path), sha256=sha(path), **expected, origins=6, methods=sorted(METHODS))

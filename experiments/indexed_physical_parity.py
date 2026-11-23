"""Read-only, prospectively fixed exact comparison for bookkeeping indexing.

No simulator, remote operation, admission, or deployment is performed here.
Every recorded physical value, ordered action/request identity and ACK is
compared.  Metadata exceptions are file-and-pointer-specific and their original
ordered values remain hashed in the proof.  Unknown fields are compared;
unknown files are rejected.  Never expand this policy to rescue a failed trial.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import compute_parity as exact
from experiments.execution_parallel_parity import APP_STREAMS, CORE_STREAMS

OLD_SOURCE = 'cbe114446814007b275856e4bae92a2fe609abf4f4bcbbf8c32394a722a36af0'
NEW_SOURCE = '714eb8d89c9a384c66f9799189599902464cf3c19822f74d470f1b4de42353b5'
OLD_ARCHIVE = '4f915fd649e287b68465c4aed047a7b626959b471d532abd8a659e376e3641e1'
NEW_ARCHIVE = '29d935e0248b78435869275ab9780b92657cf86938689994ae47a7876885edd7'
OLD_RUNNER = '0aa03b21946b0baa600ab25e24502ef4439fe08e503304ce70aee8209b26edb2'
NEW_RUNNER = 'eeaef7bf00be329fe54cf57795e885cb17b5f9960aec06034792ee1b895256cb'
MEASUREMENT = dict(baseline_slot=2040, origin_observe_slots=2000,
                   observe_slots=400, tail_slots=400, total_slots=2840)
ARMS = {'OFF': [], **{a: ['xapp:'+v] for a, v in zip('ABCDEF', exact.MODEL_IDS)}}
MODEL_FILES = dict(zip(exact.MODEL_IDS,('x01','x02','x08','x12_mec','x12_urllc','x13')))
SOURCE_DIFF_SHA = 'c253db61da96b2d3c14607050ba83c3ccdc4bc44225a253876ca73f5e2ea402d'
ORACLE_SHA = '7a2d4711a16d3a1d5c4999676e7f2ce6919d37e1f570e978c82457ff7f992098'
V2_FAILED_SHA = '127a058f2d49b850f3f5503896f8824c0edf596216430e8ef7f3e8987f4c5d95'
WALL_PRODUCER_DIAG_SHA = 'c6b492ce79d719e8bfb73aa981c0de3c8c7dc2f4f0ebff66bb136b7050f4cd9a'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def _require(value, message):
    if not value:
        raise ValueError(message)


def metadata_rules():
    rules = deepcopy(exact.RULES)
    # Keep this finite list prospective; source hashes elsewhere, including
    # training provenance and channel/cache/backend identities, remain exact.
    for file in ('config_snapshot.json', 'run_status.json', 'performance.json'):
        rules.append(dict(file=file, pointer='/source_sha256', type='source_pin',
            reason='The explicit old/new approved source pins are independently verified'))
    rules.append(dict(file='network-replay.html', pointer='/meta/source_sha256', type='source_pin',
        reason='Embedded replay provenance names the independently verified source version'))
    rules.append(dict(file='evaluation_origin_verified.json', pointer='/origin_id', type='origin_pin',
        reason='Each version verifies its independently sealed, same-host OFF origin; all flags/windows stay compared'))
    rules.append(dict(file='config_snapshot.json', pointer='/configs/ric/memory_store_path', type='owned_path',
        reason='Output location only, permitted only when persistent memory is disabled'))
    rules.append(dict(file='xapp_test_spec.json', pointer='/arguments/ap_services_profile', type='profile_path',
        reason='Exact materialized AP profile and its SHA are independently equal'))
    rules.append(dict(file='final_summary.json', pointer='/xapp_library/instances/*/learning/model_path', type='model_path',
        reason='Learning summary path to the independently pinned frozen checkpoint'))
    for file in ('final_summary.json','live.json'):
        rules.append(dict(file=file, pointer='/xapp_test/app/learning/model_path', type='model_path',
            reason='Session learning-summary copy of the independently pinned active frozen checkpoint'))
    rules.append(dict(file='final_summary.json',pointer='/ue_services/module_wall_s',type='number',
        reason='Mechanical v3 schema correction: identical producer accumulates only time.perf_counter hook costs; no simulated service/control equation consumes it'))
    return rules


def policy():
    return dict(schema='oran.indexed-physical-parity-policy.v3',
        allowed_metadata_difference_categories=['instrumentation', 'source', 'output_path'],
        inherited_metadata_rules_from=ref(exact.__file__), metadata_rules=metadata_rules(),
        exact_decimal=True, tolerance=0, rounding=False, arrays='full ordered sequences',
        unknown_fields='compare', unknown_files='fail',
        mandatory_core_jsonl=sorted(CORE_STREAMS), active_app_streams=APP_STREAMS,
        lazy_app_streams='Required for the active app; absent on both allowed otherwise; present on either requires both.',
        mandatory_csv=list(exact.CSV), mandatory_json=[x for x in exact.JSON if x!='evaluation_origin_verified'],
        ON_origin_guard_required=True, OFF_origin_guard_absent_allowed=True,
        model_checkpoints=list(exact.MODEL_IDS), optional_json=list(exact.OPTIONAL_JSON),
        optional_npz=list(exact.OPTIONAL_NPZ), optional_html=['network-replay.html'],
        channel_cache_key_epochs_anchors_backend_and_all_physical_numbers='compare exactly',
        session_nonce_exceptions='Only the three inherited exact UUID4 paths; request/control/UE/TB IDs remain exact',
        full_run_parity_claimed=False, physical_scope='Complete recorded 2840-slot short trajectories only',
        schema_correction=dict(signed_after_first_indexed_OFF=True,original_v2_failed_proof_sha256=V2_FAILED_SHA,
            producer_diagnosis_sha256=WALL_PRODUCER_DIAG_SHA,precise_added_pointer='final_summary.json:/ue_services/module_wall_s',
            prospective_first_trial_claimed=False,physical_number_tolerance_or_threshold_changed=False),
        unrecorded_CIR_CFR_arrays_compared=False, oracle_is_not_long_run_physical_parity=True,
        no_automatic_action=True)


class Audit(exact._Audit):
    def __init__(self, root, source, origin):
        super().__init__(Path(root))
        self.source = source
        self.origin = origin
        self.scalar_metadata_originals={}
        self.rules = [(r, tuple(r['pointer'].strip('/').split('/'))) for r in metadata_rules()]

    def note(self,file,path,value):
        super().note(file,path,value)
        if file=='final_summary.json' and path==('ue_services','module_wall_s'):
            key=file+':'+exact._pointer(path)
            self.scalar_metadata_originals.setdefault(key,[]).append(str(value))

    def transform(self, file, value, path=()):
        for rule, pattern in self.rules:
            if rule['file'] != file or not exact._matches(pattern, path):
                continue
            kind = rule['type']
            if kind == 'source_pin':
                _require(value == self.source, 'Unverified source pin in '+file)
                replacement = 'approved_bookkeeping_source'
            elif kind == 'origin_pin':
                _require(value == self.origin, 'Unverified own OFF origin')
                replacement = 'verified_own_OFF_origin'
            elif kind == 'profile_path':
                _require(type(value) is str and value.endswith('/configs/rt_sys/ap_services_reference.json'),
                         'Unexpected AP profile location')
                replacement = 'byte_identical_AP_profile'
            else:
                if kind == 'number':
                    _require(exact._number(value), 'Wrong numeric metadata type')
                if kind == 'number_or_null':
                    _require(value is None or exact._number(value), 'Wrong nullable numeric metadata')
                if kind == 'bool':
                    _require(type(value) is bool, 'Wrong boolean metadata')
                if kind in ('str','nonce','model_path','log_path','owned_path','model_dir','timeline_path','mesh_path','preview_model_path'):
                    _require(value is None or type(value) is str, 'Wrong string metadata')
                replacement = kind
                if kind == 'nonce':
                    _require(type(value) is str and re.fullmatch('[0-9a-f]{32}', value), 'Invalid session nonce')
                if kind == 'model_path' and value is not None:
                    _require(value in self.model_paths, 'Unpinned model path '+file+exact._pointer(path))
                    replacement += ':'+self.model_paths[value]
                if kind == 'model_dir':
                    replacement += ':'+hashlib.sha256(exact._encoded(self.model_pins)).hexdigest()
                if kind == 'preview_model_path':
                    rel='models/'+path[0]+'.json'
                    _require(type(value) is str and value.endswith('/preview_models/'+rel), 'Unexpected preview model argv')
                    replacement += ':'+sha(self.root/'preview_models'/rel)
                if kind == 'log_path' and value is not None:
                    leaf=Path(value).name.removesuffix('.gz')
                    _require(leaf in {n+'.jsonl' for n in exact.JSONL+exact.OPTIONAL_JSONL}, 'Unknown log path')
                    replacement += ':'+leaf
                if kind == 'timeline_path' and value is not None:
                    replacement += ':xapp_switch_timeline.json'
                if kind == 'mesh_path' and value is not None:
                    leaf=Path(value).name
                    _require(leaf in exact.OPTIONAL_NPZ and (self.root/leaf).is_file(), 'Missing mesh')
                    replacement += ':'+leaf
                if kind == 'owned_path' and value is not None:
                    if path[-1] == 'comparison_spec':
                        _require(Path(value).name=='xapp_test_spec.json' and (self.root/'xapp_test_spec.json').is_file(), 'Missing test spec')
                    if path[-1] == 'command_dir':
                        _require(Path(value).name.endswith('.commands'), 'Unexpected command directory')
                if kind == 'worker_status_list':
                    allowed={'completed','cancelled','assigned_blocks','processed_blocks','generated_blocks','skipped_blocks','lock_wait_s','worker_wall_s'}
                    _require(isinstance(value,list) and all(isinstance(x,dict) and not(set(x)-allowed) for x in value), 'Unknown prefetch metadata')
                    for row in value:
                        for k,v in row.items():
                            _require(type(v) is bool if k in ('completed','cancelled') else exact._number(v), 'Wrong prefetch type')
            self.note(file,path,value)
            return {'__parity_metadata__': replacement, 'is_null': value is None}
        if isinstance(value,dict):
            return {k:self.transform(file,v,path+(k,)) for k,v in value.items()}
        if isinstance(value,list):
            return [self.transform(file,v,path+(str(i),)) for i,v in enumerate(value)]
        return value


def inventory(root, apps):
    root=Path(root)
    result={}
    known=({s+'.jsonl' for s in exact.JSONL+exact.OPTIONAL_JSONL}
           |{s+'.csv' for s in exact.CSV}|{s+'.json' for s in exact.JSON+exact.OPTIONAL_JSON}
           |set(exact.OPTIONAL_NPZ)|{'network-replay.html'}
           |{'xapp_model_checkpoints/'+s+'.json' for s in exact.MODEL_IDS})
    for p in sorted(root.rglob('*')):
        _require(not p.is_symlink(),'Symlink in recording')
        if not p.is_file():continue
        name=p.relative_to(root).as_posix()
        logical=name[:-3] if name.endswith('.gz') and name[:-3].endswith(('.jsonl','.csv')) else name
        _require(logical in known or logical.startswith('preview_models/'), 'Unknown recording '+name)
        _require(logical not in result, 'Duplicate recording '+name)
        result[logical]=p
    _require(all(a in APP_STREAMS for a in apps), 'Unknown controller')
    needed=(CORE_STREAMS|{APP_STREAMS[a] for a in apps}|{s+'.csv' for s in exact.CSV}
            |{s+'.json' for s in exact.JSON if s!='evaluation_origin_verified'}
            |{'xapp_model_checkpoints/'+s+'.json' for s in exact.MODEL_IDS})
    if apps:needed.add('evaluation_origin_verified.json')
    _require(needed<=result.keys(), 'Missing recordings '+str(sorted(needed-result.keys())))
    return result


def verify(checkpoint, plan_path, side):
    checkpoint,plan_path=Path(checkpoint),Path(plan_path)
    seal,plan=exact._read(checkpoint),exact._read(plan_path)
    # Plan hashes use the producer's native JSON number serialization; do not
    # reserialize Decimal values through a different representation.
    native=json.loads(plan_path.read_text());body=deepcopy(native);pin=body.pop('plan_sha256',None)
    _require(hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==pin,
             'Plan digest mismatch')
    source,archive,runner=(OLD_SOURCE,OLD_ARCHIVE,OLD_RUNNER) if side=='reference' else (NEW_SOURCE,NEW_ARCHIVE,NEW_RUNNER)
    _require(seal.get('completed') is True and seal.get('files_sha256'), 'Complete nonempty seal required')
    case=seal['case'];_require(case in plan['cases'] and seal['plan_sha256']==pin,'Case/plan mismatch')
    _require(case.get('worker_id')==seal.get('worker_id')==plan.get('worker_id')
        and isinstance(seal.get('actual_system_hostname'),str) and seal['actual_system_hostname'].strip(),
        'A nonempty actual execution hostname and consistent logical worker are required')
    for key,value in [('source_sha256',source),('source_archive_sha256',archive),('runner_sha256',runner),('worker_id',plan['worker_id'])]:
        _require(seal.get(key)==plan.get(key)==value,'Wrong '+side+' '+key)
    _require(case['app_ids']==ARMS[case['arm']], 'Wrong singleton binding')
    _require({k:case[k] for k in ('baseline_slot','origin_observe_slots','observe_slots')}=={k:MEASUREMENT[k] for k in ('baseline_slot','origin_observe_slots','observe_slots')}
             and case['slots']==MEASUREMENT['total_slots'], 'Short measurement changed')
    job=checkpoint.parent.parent/'jobs'/case['id']
    _require({p.relative_to(job).as_posix() for p in job.rglob('*') if p.is_file()}==set(seal['files_sha256']),
             'The sealed and actual complete job inventories differ')
    for name,pin in seal['files_sha256'].items():
        path=(job/name).resolve()
        _require(path.is_relative_to(job.resolve()) and path.is_file() and sha(path)==pin,'Sealed recording changed: '+name)
    run=job/'data/run';files=inventory(run,case['app_ids']);a=Audit(run,source,seal['origin_id'])
    request=exact._read(job/'data/request.json')
    _require(request.get('case')==case and request.get('plan_sha256')==seal['plan_sha256']
        and all(request.get(k)==seal.get(k) for k in ('source_sha256','source_archive_sha256','runner_sha256','frozen_models','actual_system_hostname')),
        'Actual request/source/model identity differs')
    config,status,hot=(exact._read(files[n+'.json']) for n in ('config_snapshot','run_status','xapp_hot_start'))
    _require(config['source_sha256']==status['source_sha256']==source,'Source recording mismatch')
    _require(status.get('completed') is True and status.get('simulation_completed') is True and status.get('stage')=='completed'
             and status.get('replay',{}).get('audits_passed') is True,'Incomplete or unaudited network')
    _require(config['manifest']['total_slots']==MEASUREMENT['total_slots'] and config['manifest']['rng_seed']==case['seed'],'Actual trajectory/window differs')
    ric=config['configs']['ric'];_require(ric.get('freeze_learning') is True and ric.get('persistent_memory_enabled') is False,'Learning/memory enabled')
    _require(set(hot['models'])==set(exact.MODEL_IDS) and len(seal['frozen_models'])==6,'Six models required')
    for app,item in hot['models'].items():
        name='xapp_model_checkpoints/'+app+'.json';pin=sha(files[name])
        _require(item['sha256']==pin and item['mode']=='frozen' and item['checkpoint']==name,'Frozen model binding mismatch')
        a.model_paths[item['path']]=pin;a.model_pins[app]=pin
    expected_models={app:plan['frozen_models']['artifacts/xapps/chongyu/20261117/models/'+leaf+'.json'] for app,leaf in MODEL_FILES.items()}
    _require(a.model_pins==expected_models and seal['frozen_models']==plan['frozen_models'],'Specific app-to-model pins differ')
    _require({x['implementation_id'] for x in ric['xapps']}==set(exact.MODEL_IDS) and len(ric['xapps'])==6,'Configured six-app roster differs')
    for app in ric['xapps']:
        item=hot['models'][app['implementation_id']]
        _require(app['parameters']['mode']=='frozen' and app['parameters']['model_path']==item['path'],'Config model binding mismatch')
    if hot.get('timeline_source') is not None:
        _require(hot['timeline_source']['sha256']==sha(files['xapp_switch_timeline.json']),'Timeline pin mismatch')
    if case['app_ids']:
        guard=exact._read(files['evaluation_origin_verified.json'])
        _require(guard.get('origin_id')==seal['origin_id'] and guard.get('after_kpis_used') is False
                 and all(guard.get(k) is True for k in ('verified_before_activation','observed_prefix_equal','configuration_equal','models_equal','network_dynamics_code_equal')),
                 'Actual same-version own-OFF guard missing')
    origin_path=checkpoint.parent.parent/'origins'/(case['origin_key']+'.json')
    origin_native=json.loads(origin_path.read_text());origin_body=deepcopy(origin_native);origin_pin=origin_body.pop('origin_id',None)
    _require(hashlib.sha256(json.dumps(origin_body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==origin_pin==seal['origin_id'],
             'Own-OFF origin digest mismatch')
    own_off=next(c for c in native['cases'] if c['origin_key']==case['origin_key'] and not c['app_ids'])
    _require(origin_native['source_job_id']==own_off['id'] and origin_native['seed']==case['seed']
        and origin_native['source_sha256']==source and origin_native['source_archive_sha256']==archive
        and origin_native['baseline_slot']==2040 and origin_native['observe_slots']==2000
        and origin_native['model_sha256']==a.model_pins and origin_native['after_kpis_used'] is False,
        'Wrong same-host source/model OFF origin')
    if case['app_ids']:
        _require(json.loads((job/'data/expected_origin.json').read_text())==origin_native,'Actual control used another OFF origin')
    release=exact._read(files['preview_models/release.json']);manifest=release['files_sha256']
    expected={'preview_models/release.json','preview_models/models.json','preview_models/runtime-arguments.json'}
    _require(isinstance(manifest,dict) and manifest,'Missing preview manifest')
    for name,pin in manifest.items():
        _require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe model release path')
        key='preview_models/'+name;_require(key in files and sha(files[key])==pin,'Preview release changed')
        expected.add(key)
    _require(expected=={k for k in files if k.startswith('preview_models/')},'Unknown preview artifact')
    argv=exact._read(files['preview_models/runtime-arguments.json'])
    _require(set(argv)==set(release.get('models',{}))==set(MODEL_FILES.values()),'Preview six-model roster differs')
    for app,leaf in MODEL_FILES.items():
        entry=release['models'][leaf];tokens=argv[leaf];mode=entry['cli_mode']
        _require(entry['implementation_id']==app and entry['sha256']==a.model_pins[app]
                 and sha(files['preview_models/models/'+leaf+'.json'])==a.model_pins[app], 'Preview checkpoint differs from frozen app model')
        _require(isinstance(tokens,list) and len(tokens)>=4 and tokens[:3]==['--'+mode+'-xapp-mode','frozen','--'+mode+'-xapp-model']
                 and tokens[3].endswith('/preview_models/models/'+leaf+'.json'),'Preview installer argv differs')
    return seal,plan,run,files,a


def compare_pair(reference_checkpoint,candidate_checkpoint,reference_plan,candidate_plan):
    result=dict(passed=False,checks={},streams={},non_metadata_differences=[],metadata_excluded={})
    audits=[]
    try:
        identity_inputs={str(Path(p).resolve()):sha(p) for p in (reference_checkpoint,candidate_checkpoint,reference_plan,candidate_plan)}
        old,new=(verify(cp,p,side) for cp,p,side in [(reference_checkpoint,reference_plan,'reference'),(candidate_checkpoint,candidate_plan,'candidate')])
        os,op,orr,of,oa=old;ns,np,nrr,nf,na=new;audits=[oa,na]
        _require(orr.resolve()!=nrr.resolve(),'Independent recordings required')
        _require(os['worker_id']==ns['worker_id'] and os['case']['seed']==ns['case']['seed'] and os['case']['arm']==ns['case']['arm']
                 and isinstance(os.get('actual_system_hostname'),str) and os['actual_system_hostname'].strip()
                 and os['actual_system_hostname']==ns.get('actual_system_hostname'),'Same-host seed/arm mismatch')
        # Paths and source provenance differ.  Every materialized case input is
        # independently pinned, while all content hashes and physical fields
        # must agree.  No generic path-or-hash heuristic is used.
        fields=('scene','seed','app_ids','slots','baseline_slot','origin_observe_slots','observe_slots','traffic_load',
                'traffic_profile','traffic_profile_sha256','ue_services_config','ue_services_config_sha256',
                'ap_services_profile_sha256','base_config_sha256')
        _require({k:os['case'][k] for k in fields}=={k:ns['case'][k] for k in fields},'Materialized recipe differs')
        _require(oa.model_pins==na.model_pins,'App-to-model binding differs')
        _require(of.keys()==nf.keys(),'Complete file inventories differ')
        total_diff=0
        for name in sorted(of):
            paths=[of[name],nf[name]];counts=[0,0];hashes=[hashlib.sha256(),hashlib.sha256()];different=0
            for index,pair in enumerate(zip_longest(*(exact._records(name,p) for p in paths),fillvalue=exact._MISSING)):
                values=[]
                for i,value in enumerate(pair):
                    if value is exact._MISSING:values.append(value);continue
                    if name=='rt_sys_trace.jsonl':
                        _require(type(value.get('slot_id')) is int and value['slot_id']==index+1,'Missing or unordered absolute slot')
                    value=audits[i].transform(name,value);values.append(value)
                    hashes[i].update(exact._encoded(value)+b'\n');counts[i]+=1
                issues=[('/','record_count_difference')] if any(v is exact._MISSING for v in values) else exact._differences(*values)
                for pointer,reason in issues:
                    different+=1;total_diff+=1
                    if len(result['non_metadata_differences'])<100:
                        result['non_metadata_differences'].append(dict(file=name,record_index=index,pointer=pointer,reason=reason))
            if name=='rt_sys_trace.jsonl':_require(counts==[MEASUREMENT['total_slots']]*2,'Full short trajectories required')
            result['streams'][name]=dict(exact_equal=different==0,differences=different,
                **{side:dict(raw_sha256=sha(paths[i]),records=counts[i],physical_sha256=hashes[i].hexdigest()) for i,side in enumerate(('reference','candidate'))})
        result.update(worker_id=os['worker_id'],arm=os['case']['arm'],seed=os['case']['seed'],passed=total_diff==0,
                      exact_difference_count=total_diff)
        for side,seal,plan,run,cp,pp in [('reference',os,op,orr,reference_checkpoint,reference_plan),('candidate',ns,np,nrr,candidate_checkpoint,candidate_plan)]:
            trace=next(v for k,v in (of if side=='reference' else nf).items() if k=='rt_sys_trace.jsonl')
            result[side]=dict(job_id=seal['case']['id'],source_sha256=seal['source_sha256'],source_archive_sha256=seal['source_archive_sha256'],
                runner_sha256=seal['runner_sha256'],frozen_models=seal['frozen_models'],origin_id=seal['origin_id'],
                checkpoint=ref(cp),plan=ref(pp),trace=ref(trace))
        equal=total_diff==0
        result['checks']={k:equal for k in ('physical_values_equal','traffic_values_equal','action_values_equal','action_ID_order_equal','ACK_receipts_equal','exogenous_data_equal','window_equal')}
        # A receiver may still be copying files.  A complete comparison is only
        # published if every input identity and raw seal pin remains unchanged.
        _require(all(sha(path)==pin for path,pin in identity_inputs.items()),'Checkpoint or plan changed during comparison')
        for seal,cp in ((os,reference_checkpoint),(ns,candidate_checkpoint)):
            job=Path(cp).parent.parent/'jobs'/seal['case']['id']
            _require({p.relative_to(job).as_posix() for p in job.rglob('*') if p.is_file()}==set(seal['files_sha256']),
                     'Job inventory changed during comparison')
            _require(all(sha(job/name)==pin for name,pin in seal['files_sha256'].items()),'Raw recording changed during comparison')
    except (OSError,ValueError,KeyError,TypeError,csv.Error,zipfile.BadZipFile) as e:
        result['passed']=False
        result['checks']={k:False for k in result['checks']}
        result['non_metadata_differences'].append(dict(reason='fail_closed_admission_or_record_error',error=str(e)))
    result['metadata_excluded']={side:a.exclusions() for side,a in zip(('reference','candidate'),audits)}
    result['metadata_differences']={}
    for i,a in enumerate(audits):
        side=('reference','candidate')[i]
        for pointer,values in a.scalar_metadata_originals.items():
            result['metadata_differences'].setdefault(pointer,{})[side]=dict(
                original_recorded_numbers_exact_decimal=values,
                ordered_original_value_sha256=a.hashes[pointer].hexdigest(),count=a.counts[pointer],
                classification='Pure host instrumentation; original v2 failure and producer proof retained')
    return result


def compare_study(reference, candidate, output, *, oracle_proof, source_diff_proof):
    """Roots contain protocol/pilot and runs/workers; no remote IO."""
    reference,candidate,output=Path(reference),Path(candidate),Path(output)
    _require(not output.exists(),'Preserve parity outputs; choose a new version')
    result=dict(schema='oran.indexed-physical-parity.v1',passed=False,measurement=MEASUREMENT,
        allowed_metadata_difference_categories=['instrumentation','source','output_path'],
        old_source_sha256=OLD_SOURCE,new_source_sha256=NEW_SOURCE,old_source_archive_sha256=OLD_ARCHIVE,new_source_archive_sha256=NEW_ARCHIVE,
        oracle_proof=ref(oracle_proof),source_diff_proof=ref(source_diff_proof),comparison_code=ref(__file__),
        comparison_dependencies=[ref(exact.__file__),ref(ROOT/'experiments/execution_parallel_parity.py')],
        policy=policy(),trial_pairs=[],candidate_parallelism={},
        does_not_claim_full_10s_physical_parity=True,unrecorded_CIR_CFR_arrays_compared=False,parity_is_not_independent_sample=True)
    evidence_dir=Path(source_diff_proof).parent
    result['schema_correction']=schema_correction_evidence(evidence_dir)
    proof=json.loads(Path(source_diff_proof).read_text())
    _require(sha(source_diff_proof)==SOURCE_DIFF_SHA and sha(oracle_proof)==ORACLE_SHA,'Unapproved source/oracle proof changed')
    _require(proof['source_sha256']==NEW_SOURCE and proof['baseline_source_sha256']==OLD_SOURCE and proof['all_other_regular_members_equal'] is True,'Wrong source-diff proof')
    _require(proof['oracle_proof']['sha256']==sha(oracle_proof),'Oracle proof mismatch')
    for worker in ('agx1','agx2'):
        plans=[root/'protocol/pilot'/(worker+'.json') for root in (reference,candidate)]
        values=[json.loads(p.read_text()) for p in plans]
        _require(all(p.get('worker_id')==worker and all(c.get('worker_id')==worker for c in p['cases']) for p in values),
                 'Do not duplicate another logical worker under a different plan filename')
        _require(all([c['arm'] for c in p['cases']]==list(ARMS) for p in values),'Each host requires OFF+A..F')
        for a,b in zip(*(p['cases'] for p in values)):
            cps=[root/'runs/workers'/worker/'checkpoints'/(case['id']+'.json') for root,case in zip((reference,candidate),(a,b))]
            result['trial_pairs'].append(compare_pair(*cps,*plans))
        status=candidate/'runs/workers'/worker/'status.json'
        if status.is_file():
            result['candidate_parallelism'][worker]=dict(status=ref(status),configured_upper_bound=values[1]['max_parallel'],
                note='Actual overlap must be shown by case timing intervals; upper bound is not measured concurrency.')
        intervals=[]
        for case in values[1]['cases']:
            path=candidate/'runs/workers'/worker/'case_status'/(case['id']+'.json')
            if path.is_file():
                s=json.loads(path.read_text())
                if s.get('phase')=='complete' and type(s.get('pid')) is int and s.get('started_unix_s') and s.get('wall_s'):
                    intervals.append(dict(case_id=case['id'],pid=s['pid'],start_unix_s=s['started_unix_s'],end_unix_s=s['started_unix_s']+s['wall_s'],case_status=ref(path)))
        if intervals:
            events=sorted([(r['start_unix_s'],1) for r in intervals]+[(r['end_unix_s'],-1) for r in intervals])
            active=peak=0
            for _,delta in events:active+=delta;peak=max(peak,active)
            result['candidate_parallelism'].setdefault(worker,{})['recorded_case_lifecycle_intervals']=intervals
            result['candidate_parallelism'][worker].update(maximum_case_lifecycle_overlap=peak,
                lifecycle_overlap_is_not_a_direct_simultaneous_PID_observation=True,
                direct_simultaneous_process_observation_required_for_operational_adoption=True)
    result['passed']=len(result['trial_pairs'])==14 and all(p['passed'] for p in result['trial_pairs'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');output.chmod(0o444)
    return result


def schema_correction_evidence(directory):
    directory=Path(directory)
    failed=directory/'agx2-OFF-physical-parity-v2.json'
    producer=directory/'UE-services-module-wall-producer-diagnosis-v1.json'
    _require(sha(failed)==V2_FAILED_SHA and sha(producer)==WALL_PRODUCER_DIAG_SHA,'Retain the exact original failure and producer diagnosis')
    return dict(schema='oran.parity-instrumentation-schema-correction.v3',
        recorded_after_first_indexed_OFF=True,prospective_initial_comparator_claimed=False,
        original_v2_failed_proof=ref(failed),identical_source_producer_diagnosis=ref(producer),
        added_pointer_only='final_summary.json:/ue_services/module_wall_s',
        physical_or_business_value_thresholds_changed=False,original_recordings_rewritten=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path);p.add_argument('--candidate',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--oracle-proof',type=Path);p.add_argument('--source-diff-proof',type=Path);p.add_argument('--write-policy',action='store_true')
    a=p.parse_args()
    if a.write_policy:
        _require(not a.output.exists(),'Preserve prospective comparison policy')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(dict(policy=policy(),schema_correction=schema_correction_evidence(a.output.parent),comparison_code=ref(__file__),dependencies=[ref(exact.__file__),ref(ROOT/'experiments/execution_parallel_parity.py')]),indent=2)+'\n')
        a.output.chmod(0o444);print(json.dumps(ref(a.output)));return 0
    if not all((a.reference,a.candidate,a.oracle_proof,a.source_diff_proof)):p.error('Require all study/proof paths')
    result=compare_study(a.reference,a.candidate,a.output,oracle_proof=a.oracle_proof,source_diff_proof=a.source_diff_proof)
    print(json.dumps(dict(passed=result['passed'],output=str(a.output),pairs=len(result['trial_pairs']))))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

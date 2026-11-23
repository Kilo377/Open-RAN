"""Operator-owned runtime recipes; browser requests never select environment paths.

The archived validation source and the current harness snapshot have distinct
identities. Physics/configuration/model bytes are checked independently, both
locally and on the worker, before any simulation is started.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

SCHEMA='oran.smo.runtime-recipe.v1'
PROFILE_NAMES={'traffic_profile':'traffic.json','ue_services_config':'ue-services.json',
    'ap_services_profile':'ap-services.json','base_config':'base.json'}
FIELDS={'schema','id','seed','baseline_slot','origin_observe_slots','tail_slots','traffic_load',
    'dynamics_sha256','dynamics_members','validated_source_sha256','validated_source_archive_sha256',
    'frozen_models','profiles','validated_origin'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_pins(root):
    from smo.evaluation_reference import (_DYNAMICS_GROUPS,_DYNAMICS_EXPERIMENTS,
        _is_dynamics,_files_digest)
    root=Path(root);files={}
    for group in _DYNAMICS_GROUPS:
        for path in (root/group).rglob('*'):
            if path.is_file() and _is_dynamics(str(path.relative_to(root))):
                files[str(path.relative_to(root))]=path.read_bytes()
    for name in _DYNAMICS_EXPERIMENTS:
        path=root/name
        if path.is_file():files[name]=path.read_bytes()
    return dict(sha256=_files_digest(files),members=len(files))


def _hash(value):
    if not isinstance(value,str)or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('Runtime recipe requires a SHA-256 pin')


def _absolute(path):
    p=Path(path)
    if not p.is_absolute()or '..'in p.parts:raise ValueError('Runtime recipe paths must be absolute and normalized')
    return p


def _check_source(recipe,root):
    pins=source_pins(root)
    if pins!={'sha256':recipe['dynamics_sha256'],'members':recipe['dynamics_members']}:
        raise ValueError('Runtime recipe network dynamics differ from the validated source')
    from smo.model_release import expected_model_pins,verify_release
    expected_model_pins(recipe);verify_release(recipe,root)
    for name,pin in recipe['frozen_models'].items():
        p=Path(name)
        if p.is_absolute()or '..'in p.parts:
            raise ValueError('Runtime recipe model must be a frozen relative model path')
        _hash(pin)
        if sha256(Path(root)/p)!=pin:raise ValueError('Runtime recipe frozen model differs')


def _shape(recipe,*,remote=False):
    expected=FIELDS-{'validated_origin'} if remote else FIELDS
    if remote:expected|={'validated_origin_id','recipe_sha256'}
    if not isinstance(recipe,dict)or set(recipe)not in(expected,expected|{'model_release'})or recipe.get('schema')!=SCHEMA:
        raise ValueError('Unknown runtime recipe schema or fields')
    if not re.fullmatch('[a-z][a-z0-9_]{0,40}',recipe['id']):raise ValueError('Invalid runtime recipe ID')
    if type(recipe['seed'])is not int or not 0<=recipe['seed']<2**32:raise ValueError('Invalid runtime recipe seed')
    if any(type(recipe[k])is not int for k in ('baseline_slot','origin_observe_slots','tail_slots','dynamics_members'))or (recipe['baseline_slot'],recipe['origin_observe_slots'],recipe['tail_slots'])!=(2040,2000,400)or not 1<=recipe['dynamics_members']<=100000:
        raise ValueError('Runtime recipe must retain the validated prefix and dynamics members')
    if type(recipe['traffic_load'])not in(int,float)or recipe['traffic_load']!=1.5:
        raise ValueError('Runtime recipe traffic load differs from validation')
    for key in ('dynamics_sha256','validated_source_sha256','validated_source_archive_sha256'):_hash(recipe[key])
    if not isinstance(recipe['frozen_models'],dict)or len(recipe['frozen_models'])!=6:
        raise ValueError('Runtime recipe requires all six frozen models')
    if not isinstance(recipe['profiles'],dict)or set(recipe['profiles'])!=set(PROFILE_NAMES):
        raise ValueError('Runtime recipe requires four pinned profiles')
    for item in recipe['profiles'].values():
        if not isinstance(item,dict)or set(item)!={'path','sha256'}:raise ValueError('Invalid runtime profile')
        _absolute(item['path']);_hash(item['sha256'])
    from smo.model_release import descriptor
    descriptor(recipe)


def load_recipe(path,*,source_root=None,seed=None):
    """Read a local operator file and validate every referenced artifact."""
    path=_absolute(path);recipe=json.loads(path.read_text());_shape(recipe)
    if seed is not None and recipe['seed']!=seed:raise ValueError('Runtime recipe and pool seeds differ')
    for item in recipe['profiles'].values():
        if sha256(item['path'])!=item['sha256']:raise ValueError('Runtime profile SHA-256 mismatch')
    _check_source(recipe,source_root or Path(__file__).resolve().parents[1])
    origin=recipe['validated_origin']
    if not isinstance(origin,dict)or set(origin)!={'path','sha256','run','config_sha256'}:
        raise ValueError('Invalid validated runtime origin reference')
    _absolute(origin['path']);_absolute(origin['run']);_hash(origin['sha256']);_hash(origin['config_sha256'])
    if sha256(origin['path'])!=origin['sha256']or sha256(Path(origin['run'])/'config_snapshot.json')!=origin['config_sha256']:
        raise ValueError('Validated runtime origin artifact differs')
    proof=json.loads(Path(origin['path']).read_text());validate_origin(recipe,proof)
    from smo.evaluation_reference import origin_configuration,comparable
    config=json.loads((Path(origin['run'])/'config_snapshot.json').read_text())
    if not comparable(origin_configuration(config),proof['configuration']):
        raise ValueError('Validated runtime origin configuration differs')
    return deepcopy(recipe),proof


def validate_origin(recipe,proof):
    from smo.evaluation_origin_guard import validate_proof
    validate_proof(proof)
    if (proof['seed'],proof['baseline_slot'],proof['observe_slots'])!=(recipe['seed'],recipe['baseline_slot'],recipe['origin_observe_slots']):
        raise ValueError('Runtime recipe and prepared origin differ')
    if proof['dynamics_sha256']!=recipe['dynamics_sha256']or proof['source_archive_sha256']!=recipe['validated_source_archive_sha256']or proof.get('source_sha256')!=recipe['validated_source_sha256']:
        raise ValueError('Runtime origin source pins differ')
    from smo.model_release import expected_model_pins,validate_origin_variant
    if proof['model_sha256']!=expected_model_pins(recipe):
        raise ValueError('Runtime origin frozen model pins differ')
    validate_origin_variant(recipe,proof)
    if 'validated_origin_id'in recipe and proof['origin_id']!=recipe['validated_origin_id']:
        raise ValueError('Runtime origin identity differs')
    return proof


def worker_recipe(recipe,project_root,proof):
    """Generate only controlled remote paths, never copy local paths to a job."""
    root=_absolute(project_root)
    result={k:deepcopy(v)for k,v in recipe.items()if k!='validated_origin'}
    result['profiles']={k:dict(path=str(root.parent/'environment'/PROFILE_NAMES[k]),sha256=v['sha256'])for k,v in recipe['profiles'].items()}
    result['validated_origin_id']=proof['origin_id'];result['recipe_sha256']=digest(result)
    return result


def validate_worker_recipe(recipe,root,proof,archive_pin):
    """Fail closed before Popen if files, source marker or proof changed."""
    _shape(recipe,remote=True)
    original=deepcopy(recipe);pin=original.pop('recipe_sha256');_hash(pin)
    if digest(original)!=pin:raise ValueError('Runtime recipe digest mismatch')
    root=_absolute(root).resolve()
    for name,item in recipe['profiles'].items():
        expected=root.parent/'environment'/PROFILE_NAMES[name]
        if Path(item['path'])!=expected or expected.resolve()!=expected:
            raise ValueError('Runtime profile is outside the owned environment directory')
        if sha256(expected)!=item['sha256']:raise ValueError('Remote runtime profile SHA-256 mismatch')
    _check_source(recipe,root);validate_origin(recipe,proof)
    from smo.model_release import verify_release
    verify_release(recipe,root,installed=True)
    _hash(archive_pin)
    marker=root/'.smo-owned-worker.json'
    if not marker.is_file()or json.loads(marker.read_text()).get('sha256')!=archive_pin:
        raise ValueError('Worker snapshot differs from the pinned harness archive')
    return recipe

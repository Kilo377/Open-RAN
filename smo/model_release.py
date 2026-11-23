"""Operator-owned release/variant bindings, independent of historical contracts.

This validates files only. It never installs models, selects candidates or runs
inference. A new model release is not implicitly a new calibration dataset.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

MODEL_IDS = {'x01':'ue_energy_rf','x02':'ap_energy_rf','x08':'throughput_scheduler_mlp',
    'x12_mec':'latency_scheduler_rf','x12_urllc':'latency_urllc_rf','x13':'mec_offload_budget_qgbdt'}
VARIANT_MODULE = 'ric/near_rt/pools/chongyu/ue_energy_rf_action_sets.py'
ACTION_SET_VERSIONS = ('legacy9', 'extended24', 'balanced42')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def energy_catalogue(version):
    profiles=[(40.,4.,8.),(20.,4.,1.),(80.,4.,1.)]
    if version not in ACTION_SET_VERSIONS:raise ValueError('Unknown reviewed energy action set')
    if version in ('extended24','balanced42'):
        profiles += [(40.,4.,0.),(80.,4.,0.),(160.,4.,0.),(80.,1.,0.),(160.,1.,0.)]
    if version=='balanced42':profiles += [(10.,8.,0.),(10.,6.,0.),(20.,10.,0.),(20.,8.,0.),(40.,20.,0.),(40.,10.,0.)]
    return [dict(drx=dict(enabled=True,cycle_ms=c,on_duration_ms=o,inactivity_ms=i,
        wakeup_ms=.5,offset_ms=0.),offload_fraction=f)for c,o,i in profiles for f in(0.,.5,1.)]


def catalogue_sha256(version):return digest(energy_catalogue(version))


def descriptor(recipe):
    value=recipe.get('model_release')
    if value is None:return None
    if (not isinstance(value,dict)or set(value)!={'id','manifest_sha256','action_set_version','action_catalogue_sha256'}
            or not isinstance(value['id'],str)or not re.fullmatch('[0-9]{8}',value['id'])
            or value['action_set_version']not in ACTION_SET_VERSIONS
            or not isinstance(value['manifest_sha256'],str)or not re.fullmatch('[0-9a-f]{64}',value['manifest_sha256'])
            or value['action_catalogue_sha256']!=catalogue_sha256(value['action_set_version'])):
        raise ValueError('Invalid pinned model release/energy variant descriptor')
    return value


def expected_model_pins(recipe):
    value=descriptor(recipe);release=value['id']if value else'20261117'
    prefix='artifacts/xapps/chongyu/'+release+'/models/'
    files={prefix+name+'.json'for name in MODEL_IDS}
    if set(recipe['frozen_models'])!=files:raise ValueError('Pin exactly the canonical six models in one release')
    return {MODEL_IDS[Path(name).stem]:pin for name,pin in recipe['frozen_models'].items()}


def verify_release(recipe,root,*,installed=False):
    """Bind release metadata, exact model identity and action semantics together."""
    root=Path(root);value=descriptor(recipe)
    expected=expected_model_pins(recipe)
    if value is None:return None
    bundle=root/('outputs/trained_xapps'if installed else'artifacts/xapps/chongyu')/value['id']
    if not bundle.resolve().is_relative_to(root.resolve()):raise ValueError('Release escapes the project')
    if sha(bundle/'release.json')!=value['manifest_sha256']:raise ValueError('Release manifest SHA differs')
    spec=importlib.util.spec_from_file_location('_smo_release_installer',root/'scripts/orin/install-trained-xapps.py')
    installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)
    release=installer.load_release(bundle)
    if release['release']!=value['id']:raise ValueError('Release ID differs')
    for name,ident in MODEL_IDS.items():
        entry=release['models'][name]
        if entry['implementation_id']!=ident or entry['sha256']!=expected[ident]:
            raise ValueError('Release application/model binding differs')
    entry=release['models']['x01'];parameters=entry.get('runtime_parameters',{})
    if parameters.get('action_set_version','legacy9')!=value['action_set_version']:
        raise ValueError('Energy model release cannot be run under another action set')
    if value['action_set_version']!='legacy9':
        if (parameters.get('deploy_drx')is not True or entry.get('implementation_variant')!=value['action_set_version']
                or entry.get('action_catalogue')!=energy_catalogue(value['action_set_version'])
                or entry.get('action_catalogue_sha256')!=value['action_catalogue_sha256']
                or not(root/VARIANT_MODULE).is_file()):
            raise ValueError('The extended model requires the reviewed extended factory/catalogue')
    return release


def runtime_arguments(recipe,root):
    value=descriptor(recipe)
    if value is None:return []
    return ['--xapp-model-dir',str(Path(root)/'outputs/trained_xapps'/value['id']),
        '--energy-xapp-action-set-version',value['action_set_version']]


def validate_origin_variant(recipe,proof):
    value=descriptor(recipe)
    if value is None:return
    apps=proof['configuration']['configs']['ric'].get('xapps',[])
    energy=[a for a in apps if a.get('implementation_id')=='ue_energy_rf']
    if len(energy)!=1:raise ValueError('Prepared origin lacks the exact energy instance')
    app=energy[0];version=value['action_set_version']
    if (app.get('parameters',{}).get('action_set_version','legacy9')!=version
            or app.get('implementation_variant','legacy9')!=version):
        raise ValueError('Prepared origin used another energy implementation variant')


def validate_preview(recipe,preview):
    expected=expected_model_pins(recipe);apps=preview.get('apps',[])
    if len(apps)!=6 or {a.get('app_id')for a in apps}!={'xapp:'+name for name in expected}:
        raise ValueError('Exactly six real predeployment previews are required')
    for app in apps:
        if app.get('mode')!='cold_activation'or app.get('model_sha256')!=expected[app['app_id'][5:]]:
            raise ValueError('Cold preview belongs to another model release')
    value=descriptor(recipe)
    if value:
        app=next(a for a in apps if a['app_id']=='xapp:ue_energy_rf')
        if (app.get('action_set_version')!=value['action_set_version']
                or app.get('action_catalogue_sha256')!=value['action_catalogue_sha256']
                or app.get('implementation_variant')!=value['action_set_version']):
            raise ValueError('Cold preview belongs to another energy action catalogue/factory')

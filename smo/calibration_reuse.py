"""Reuse fixed-parameter statistics only after proving unchanged network code.

Changing a trained controller does not refit P–K relations. The original
calibration and its provenance remain immutable; a separate bridge verifies
the protected simulator files and actual native/fixed-stimulus reproductions.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

ALLOWED_CONTROLLER_CHANGES=frozenset({
    'ric/near_rt/pools/__init__.py',
    'ric/near_rt/pools/chongyu/latency_scheduler_adaptive.py',
    'ric/near_rt/pools/chongyu/latency_scheduler_ppo.py',
    'experiments/hot_xapps.py','experiments/xapp_preview.py','experiments/online_rf_xapps.py'})
ADDED_CALIBRATION_TOOLS=frozenset({'experiments/current_environment_calibration.py','experiments/fit_current_environment_gains.py'})
GROUPS=('core','phy','ran','ric','state','models','traffic','experiments','configs')
LAUNCHERS=('scripts/orin/run-six-ue-cuda-nr.sh','scripts/orin/activate.sh','scripts/orin/activate-gpu.sh')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(value,reason):
    if not value:raise ValueError('P–K calibration reuse: '+reason)

def checked(item):
    require(isinstance(item,dict) and set(item)=={'path','sha256'},'a pinned evidence file is required')
    require(isinstance(item['sha256'],str) and re.fullmatch('[0-9a-f]{64}',item['sha256']) and sha(item['path'])==item['sha256'],'evidence checksum differs')
    return json.loads(Path(item['path']).read_text())

def source_files(root):
    root=Path(root);files={}
    for name in GROUPS:
        for p in (root/name).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml'):files[str(p.relative_to(root))]=sha(p)
    for name in LAUNCHERS:
        if (root/name).is_file():files[name]=sha(root/name)
    return files

def verify(entry,recipe,base_validator,root=None):
    root=Path(root or Path(__file__).resolve().parents[1])
    bridge=json.loads(Path(entry['calibration_reuse']).read_text())
    require(bridge.get('schema')=='oran.smo.calibration-reuse.v1','unknown bridge schema')
    require(bridge.get('combination_evaluation_data_used')is False and bridge.get('learned_xapps_invoked_for_response_tests')is False,
            'deployed combination outcomes cannot validate P–K statistics')
    reference=checked(bridge['reference_runtime_recipe'])
    from smo.runtime_recipe import load_recipe
    verified_reference,_=load_recipe(bridge['reference_runtime_recipe']['path'],source_root=bridge['reference_source_root'])
    require(reference==verified_reference,'reference runtime differs')
    original_entry=deepcopy(entry);original_entry.pop('calibration_reuse',None)
    historical=base_validator(original_entry,reference)
    require(historical is not None,'original calibration was not valid')
    require(bridge.get('reference_calibration_sha256')==sha(entry['current_calibration']) and
            bridge.get('influences_sha256')==sha(entry['influences']),'original calibration or coefficients changed')
    for field in ('id','seed','baseline_slot','origin_observe_slots','tail_slots','traffic_load'):
        require(recipe.get(field)==reference.get(field),'network recipe changed: '+field)
    require({k:v['sha256'] for k,v in recipe['profiles'].items()}=={k:v['sha256'] for k,v in reference['profiles'].items()},'network profiles changed')
    pins=lambda r:{Path(p).stem:pin for p,pin in r['frozen_models'].items()}
    old,new=pins(reference),pins(recipe)
    require(set(old)==set(new) and {k for k in old if old[k]!=new[k]}=={'x12_mec'},'more than the independently validated X12 actor changed')
    require(bridge.get('current_dynamics_sha256')==recipe['dynamics_sha256'] and
            bridge.get('current_source_archive_sha256')==recipe['validated_source_archive_sha256'] and
            bridge.get('current_origin_sha256')==recipe['validated_origin']['sha256'],'bridge belongs to another current source or origin')
    equivalence=checked(bridge['source_equivalence'])
    before=source_files(bridge['reference_source_root']);after=source_files(root)
    changed={p for p in before.keys()|after.keys() if before.get(p)!=after.get(p)}
    # These two tools were added after the sealed run. They neither execute in
    # the network loop nor refit the retained, independently pinned coefficients.
    require(not (set(before)&ADDED_CALIBRATION_TOOLS),'an existing calibration tool was modified')
    allowed=ALLOWED_CONTROLLER_CHANGES|ADDED_CALIBRATION_TOOLS
    require(changed<=allowed and set(equivalence.get('changed_files',[]))==changed,'protected simulator code changed')
    protected_before={k:v for k,v in before.items() if k not in allowed}
    protected_after={k:v for k,v in after.items() if k not in allowed}
    require(protected_before==protected_after==equivalence.get('protected_files'),'protected code manifest differs')
    from phy.rt_sys.contracts import source_fingerprint
    require(source_fingerprint(root)==recipe['validated_source_sha256'] and
            source_fingerprint(Path(bridge['reference_source_root']))==reference['validated_source_sha256'],'executed source fingerprint differs')
    response=checked(bridge['response_equivalence'])
    require(response.get('schema')=='oran.fixed-response-equivalence.v1' and response.get('learned_xapps_enabled')is False,
            'independent fixed-parameter reproductions are required')
    require(response.get('reference_executed_source_sha256')==reference['validated_source_sha256'] and
            response.get('current_executed_source_sha256')==recipe['validated_source_sha256'],
            'fixed responses used another network source')
    tests=response.get('cases',[])
    required={'native','ue_offload_fraction','ue_drx_configs','scheduler_mode','ue_scheduler_weights','cell_sleep_mode'}
    require({c.get('parameter') for c in tests}==required,'all five RCP paths and a native reference must be checked')
    for c in tests:
        require(c.get('complete')is True and c.get('model_algorithms_enabled')is False and
                c.get('reference_network_sha256')==c.get('current_network_sha256') and
                isinstance(c.get('reference_network_sha256'),str) and re.fullmatch('[0-9a-f]{64}',c['reference_network_sha256']) and
                c.get('contiguous_slots_verified',0)>=4840,'a fixed response changed or was incomplete')
    native=checked(bridge['native_origin_equivalence'])
    require(native.get('all_target_kpis_exactly_equal')is True and native.get('physical_prefix_hashes_equal')is True and
            native.get('checked_slots')==2040 and native.get('current_origin_sha256')==recipe['validated_origin']['sha256'],
            'current native origin differs')
    return dict(historical,calibration_reused_under_unchanged_physics=True,
        current_source_bridge_sha256=sha(entry['calibration_reuse']),
        protected_source_files_verified=len(protected_after),fixed_response_tests_verified=len(tests),
        newly_fitted_coefficients=False,changed_model_ids=['x12_mec'])

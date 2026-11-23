"""Exact network parity plus explicitly authorized render-index equivalence.

This disclosed post-data representation correction never edits recordings or
loosens a network/control/KPI numeric predicate. Complete v4 exact results are
reused only with fresh complete seal verification and exact input pins. The
two render NPZs receive an additional independently verified coordinate-bit,
topology, ordered-incidence and actual RT object/material proof. Their original
byte differences remain visible and never become ignored metadata.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import indexed_physical_parity_v4 as core
from experiments import render_mesh_equivalence_diagnostic as mesh
from experiments.run_indexed_parity_parallel_v1 import build_jobs,encoded,value_sha

CORE4='db8e9bb5bebc80d0b669c2cb63360d72afbb21178891152eba80ffae3f90ac3d'
WRAPPER='f1e3621ab251648e4ecc14bc2631527e58e7eeef4f30852717b74779e01b9156'
DIAGNOSTIC='04d1eb83df3acfc48853d147ccda8f979465fd9b917916cde752a6746292ec97'
AUTHORIZATION='fb9b70bce1510e1e3d91d49f575b3b5f52a879a7049ee43274f41e6571bc2ca7'
V3_RESULT='6a020ec56ee3bb9dfea6ffdc121c767d5e9ec08d5e3f115942b12148b64e17ef'
INDEPENDENT_REVIEW='9574e7dc4c76ebc37485c1b5bcd6b35c099adbaa524a4d09d5d9c31b420a79f3'
MESHES=('live.mesh.npz','replay.mesh.npz')


def require(value,message):
    if not value:raise ValueError(message)


def save(path,value):
    path=Path(path);require(not path.exists(),'Preserve prior output')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    path.chmod(0o444);return core.ref(path)


def policy(diagnostic,authorization):
    return dict(schema='oran.indexed-physical-parity-policy.v5',
        network_numeric_policy=core.policy(),numeric_tolerance=0,rounding=False,
        render_mesh_rule='exact_object_block_vertex_reindexing_v1',render_mesh_files=list(MESHES),
        requires=['raw float32 vertex coordinate-bit multiset, including duplicates and unused vertices',
            'orientation-preserving world triangle multiset, preserving multiplicity',
            'explicit verified global vertex-index bijection preserving every coordinate bit',
            'mapped literal face triples and original per-vertex incident face sequence',
            'complete actual RT object-block local vertices/faces bytes',
            'actual runtime RT geometry/material and scene asset fingerprint equality'],
        diagnostic=core.ref(diagnostic),authorization=core.ref(authorization),
        raw_mesh_encoding_differences_preserved=True,mesh_differences_are_not_ignored_metadata=True,
        original_v3_v4_failures_retained=True,post_data_schema_correction=True,
        prospective_initial_whitelist_completeness_claimed=False,pixel_bitwise_equivalence_claimed=False,
        whole_10s_physical_parity_claimed=False,no_automatic_admission=True)


def merge_verified_pair(result,proof):
    """Only the explicitly proved render encoding difference may be discharged."""
    value=deepcopy(result);issues=value.get('non_metadata_differences',[])
    require(set(value.get('streams',{}))>=set(MESHES),'Both complete render streams are required')
    for issue in issues:
        require(issue.get('file') in MESHES and issue.get('pointer')=='/exact_npy_sha256'
                and issue.get('reason')=='value_difference' and issue.get('record_index') in (0,1),
                'An actual network/control/KPI or unrecognized difference cannot be discharged')
    for name,stream in value['streams'].items():
        if name not in MESHES:require(stream.get('exact_equal') is True,'Non-render stream is not exactly equal: '+name)
    require(len(proof['meshes'])==2 and {m['file']for m in proof['meshes']}==set(MESHES),'Complete mesh proof required')
    for m in proof['meshes']:
        for key in ('geometry_equivalent_under_verified_reindexing','explicit_vertex_bijection_verified',
                    'world_coordinate_bits_preserved','mapped_literal_faces_equal','ordered_vertex_incidence_equal',
                    'topology_and_normal_accumulation_order_verified','object_block_partition_matches_actual_RT_fingerprint'):
            require(m.get(key) is True,'Missing exact mesh invariant: '+key)
    value['original_v4_passed']=result['passed']
    value['original_v4_exact_difference_count']=result.get('exact_difference_count')
    value['mesh_encoding_differences']=[]
    for m in proof['meshes']:
        name=m['file'];stream=value['streams'][name]
        original=[i for i in issues if i['file']==name]
        if original or m['reference']['sha256']!=m['candidate']['sha256']:
            value['mesh_encoding_differences'].append(dict(file=name,reference=m['reference'],candidate=m['candidate'],
                raw_array_bytes_equal=m['raw_array_bytes_equal'],original_v4_exact_issues=original,
                raw_stream_exact_equal=stream['exact_equal'],geometry_semantic_equal=True,
                rule='exact_object_block_vertex_reindexing_v1',not_a_metadata_exclusion=True))
        # Preserve the original byte/array-hash verdict rather than relabel it.
        stream['geometry_equivalent_under_verified_reindexing']=True
        stream['additional_equality_rule']='exact_object_block_vertex_reindexing_v1'
    value.update(passed=True,non_metadata_differences=[],exact_difference_count=0,
        exact_difference_count_scope='Network/control/physical semantics under explicit render-index equivalence; original raw mesh differences are separately retained')
    value['checks']={k:True for k in ('physical_values_equal','traffic_values_equal','action_values_equal',
        'action_ID_order_equal','ACK_receipts_equal','exogenous_data_equal','window_equal')}
    return value


def verify_mesh_and_backend(proof,verified,captures):
    require(proof['RT_cache_key_exact'] and proof['RT_cache_manifest_raw_bytes_exact']
        and proof['RT_geometry_material_association_exact'] and proof['RT_scene_asset_fingerprints_exact'],'Actual RT fingerprint proof incomplete')
    runtime=[]
    for side,version in enumerate(('reference','indexed')):
        seal,plan,run,files,a=verified[side]
        row=captures[(proof['worker_id'],version,proof['arm'])]
        require(row['case_id']==seal['case']['id'] and hashlib.sha256(row['raw_text'].encode()).hexdigest()==row['raw_sha256'],'Wrong runtime cache capture')
        raw=json.loads(row['raw_text']);key=json.loads((run/'initial_conditions.json').read_text())['cache_key']
        require(raw['cache_key']==key==row['cache_key'],'Actual RT cache identity differs')
        require(hashlib.sha256(encoded(dict(schema=2,manifest=raw['manifest'],backend=raw['backend']))).hexdigest()==key,'Runtime fingerprint key is not its canonical content hash')
        runtime.append(raw)
    require(runtime[0]==runtime[1],'Actual RT material/geometry/assets differ')
    fresh=[]
    for item in proof['meshes']:
        name=item['file'];require(name in MESHES,'Unknown mesh attribute stream')
        paths=[v[3][name]for v in verified]
        require(all(core.sha(path)==item[side]['sha256']for side,path in zip(('reference','candidate'),paths)), 'NPZ changed since diagnostic')
        calculated=mesh.diagnose(*paths)
        require(calculated['geometry_equivalent_under_verified_reindexing'],'Fresh explicit geometry/index/incidence verification failed')
        witness=item['explicit_bijection_witness'];require(core.sha(witness['path'])==witness['sha256'],'Bijection witness changed')
        mapping=np.load(witness['path'],allow_pickle=False)
        require(hashlib.sha256(mapping.tobytes()).hexdigest()==calculated['vertex_bijection_sha256']==item['vertex_bijection_sha256'],'Fresh bijection differs from recorded witness')
        arrays=[mesh._load(path)for path in paths]
        geometries=[[],[]]
        for index,version in enumerate(('reference','candidate')):
            v,f=arrays[index];parts=sorted([b[version]for b in item['object_blocks']],key=lambda x:x['vertex_start']);nextv=nextf=0
            for part in parts:
                require(part['vertex_start']==nextv and part['face_start']==nextf,'Object partition has a gap/overlap')
                ev,ef=part['vertex_end_exclusive'],part['face_end_exclusive'];require(ev>nextv and ef>nextf,'Empty object partition')
                local=f[nextf:ef].astype('<i8')-nextv
                require(local.min()>=0 and local.max()<ev-nextv,'Object has cross-block vertex references')
                h=hashlib.sha256();h.update(v[nextv:ev].astype('<f4',copy=False).tobytes());h.update(local.astype('<u4').tobytes())
                require(h.hexdigest()==part['geometry_sha256'],'Actual full object block bytes differ from RT geometry identity')
                geometries[index].append(h.hexdigest());nextv,nextf=ev,ef
            require(nextv==len(v) and nextf==len(f),'Object partition drops vertices or faces')
            require(Counter(geometries[index])==Counter(m['geometry_sha256']for m in runtime[index]['backend']['materials']),'Actual object/material fingerprint multiplicity differs')
        require(geometries[0] and Counter(geometries[0])==Counter(geometries[1]),'Object block identity differs')
        fresh.append(dict(file=name,vertex_bijection_sha256=calculated['vertex_bijection_sha256'],
            literal_faces_and_ordered_incidence_exact=True,actual_object_geometry_and_material_inventory_exact=True))
    return fresh


def run(args):
    output=Path(args.output);require(not output.exists(),'Preserve original parity proofs')
    protocol=Path(args.source_diff_proof).parent
    diagnostic=protocol/'mesh-semantics-diagnostic-v1/complete-mesh-semantics-diagnostic-v1.json'
    authorization=protocol/'comparison-schema-correction-authorization-v5.json'
    independent=protocol.parent/'analysis/actual-mesh-witness-independent-review-v1.json'
    require(core.sha(diagnostic)==DIAGNOSTIC and core.sha(authorization)==AUTHORIZATION,'Pinned diagnostic/authorization changed')
    require(core.sha(independent)==INDEPENDENT_REVIEW,'Independent actual witness review changed')
    require(core.sha(core.__file__)==CORE4 and core.sha(ROOT/'experiments/run_indexed_parity_parallel_v1.py')==WRAPPER,'Original v4 comparator/producer changed')
    require(core.sha(protocol/'indexed-physical-parity-result-v1.json')==V3_RESULT,'Original v3 failure changed')
    require(core.sha(args.source_diff_proof)==core.SOURCE_DIFF_SHA and core.sha(args.oracle_proof)==core.ORACLE_SHA,'Original source/oracle evidence changed')
    data=json.loads(diagnostic.read_text());require(data['all_14_pairs_mesh_equivalent'] and data['pairs_count']==14 and data['mesh_pair_count']==28,'Incomplete complete-mesh proof')
    require(core.sha(mesh.__file__)==data['diagnostic_code']['sha256'],'Mesh diagnostic code changed')
    proofs={(d['worker_id'],d['arm']):d for d in data['pairs']};require(len(proofs)==14,'Duplicate semantic case')
    captures={}
    for ref in data['runtime_RT_fingerprint_captures']:
        require(core.sha(ref['path'])==ref['sha256'],'RT capture changed')
        for row in json.loads(Path(ref['path']).read_text())['rows']:
            key=(row['worker_id'],row['version'],row['arm']);require(key not in captures,'Duplicate actual RT capture');captures[key]=row
    require(len(captures)==28,'Require all actual RT manifests')
    artifact=Path(args.cached_results);start=json.loads((artifact/'execution-start.json').read_text())
    require(start['wrapper']['sha256']==WRAPPER and start['core_code_pins'][str(Path(core.__file__).resolve())]==CORE4,'Untrusted cached result producer')
    regression=artifact/'serial-parallel-regression.json';reg=json.loads(regression.read_text())
    require(reg['passed'] and reg['entire_core_return_value_compared'] and reg['number_tolerance']==0
        and reg['serial_result_sha256']==reg['pooled_result_sha256'],'Original exact serial/parallel regression missing')
    jobs=build_jobs(args.reference,args.candidate);require(len(jobs)==14,'Complete fixed grid required')
    envelopes={}
    for path in artifact.glob('pair-*.json'):
        envelope=json.loads(path.read_text());key=(envelope['job']['worker_id'],envelope['job']['arm']);require(key not in envelopes,'Duplicate cache result')
        require(value_sha(envelope['core_result'])==envelope['core_result_sha256'],'Cached core result changed')
        require(envelope['execution']['core_code_pins']==start['core_code_pins'],'Mixed cached comparator code')
        envelopes[key]=(path,envelope)
    require(len(envelopes)==14,'Require all complete original exact results')
    input_pins={str(diagnostic):DIAGNOSTIC,str(authorization):AUTHORIZATION,str(Path(core.__file__)):CORE4,str(Path(__file__)):core.sha(__file__)}
    result=dict(schema='oran.indexed-physical-parity.v1',passed=False,measurement=core.MEASUREMENT,
        allowed_metadata_difference_categories=['instrumentation','source','output_path'],old_source_sha256=core.OLD_SOURCE,new_source_sha256=core.NEW_SOURCE,
        old_source_archive_sha256=core.OLD_ARCHIVE,new_source_archive_sha256=core.NEW_ARCHIVE,oracle_proof=core.ref(args.oracle_proof),source_diff_proof=core.ref(args.source_diff_proof),
        comparison_code=core.ref(__file__),comparison_dependencies=[core.ref(core.__file__),core.ref(core.exact.__file__),core.ref(mesh.__file__)],
        policy=policy(diagnostic,authorization),trial_pairs=[],mesh_encoding_differences=[],
        render_mesh_semantics=dict(rule='exact_object_block_vertex_reindexing_v1',diagnostic=core.ref(diagnostic),authorization=core.ref(authorization),
            preserves_raw_NPZ_hash_differences=True,network_control_KPI_numeric_policy_unchanged=True,independent_actual_witness_review=core.ref(independent)),
        original_comparisons=dict(v3_complete_failed_result=core.ref(protocol/'indexed-physical-parity-result-v1.json'),v4_fail_closed_execution=core.ref(artifact/'execution-failed.json'),
            serial_parallel_exact_regression=core.ref(regression),cached_complete_exact_pair_results=[]),
        candidate_parallelism=core.ref(protocol/'indexed-resource-parallelism-final-v1.json'),
        does_not_claim_full_10s_physical_parity=True,pixel_bitwise_equivalence_claimed=False,unrecorded_CIR_CFR_arrays_compared=False,
        primary_admission_authorized=False,actual_cached_comparison_execution='spawn ProcessPool4, already complete; no physical/numeric predicate recomputed approximately')
    verified_all=[]
    for job in jobs:
        key=(job['worker_id'],job['arm']);path,envelope=envelopes[key];require(envelope['job']==job,'Cached job/plan/source input revision differs')
        for source,pin in job['input_pins'].items():require(core.sha(source)==pin,'Plan/seal changed since exact comparison')
        raw=envelope['core_result'];require(raw.get('worker_id')==job['worker_id'] and raw.get('arm')==job['arm'] and raw.get('seed')==job['seed'],'Wrong cached physical identity')
        verified=[core.verify(job['args'][i],job['args'][i+2],side)for i,side in enumerate(('reference','candidate'))]
        for i,side in enumerate(('reference','candidate')):
            seal,plan,run,files,a=verified[i]
            require(raw[side]['checkpoint']['sha256']==core.sha(job['args'][i]) and raw[side]['plan']['sha256']==core.sha(job['args'][i+2]),'Cached reference identity changed')
            require(set(raw['streams'])==set(files),'A complete cached stream was added/removed')
            require(all(raw['streams'][name][side]['raw_sha256']==core.sha(file)for name,file in files.items()),'Cached raw stream changed')
        proof=proofs[key];require(proof['reference_case_id']==verified[0][0]['case']['id'] and proof['candidate_case_id']==verified[1][0]['case']['id'] and proof['seed']==job['seed'],'Mesh proof belongs to another case')
        checks=verify_mesh_and_backend(proof,verified,captures)
        merged=merge_verified_pair(raw,proof);merged['render_mesh_semantic_reverification']=checks
        merged['original_complete_v4_exact_result']=core.ref(path);result['trial_pairs'].append(merged)
        result['mesh_encoding_differences'].extend(dict(worker_id=job['worker_id'],arm=job['arm'],seed=job['seed'],**d)for d in merged['mesh_encoding_differences'])
        result['original_comparisons']['cached_complete_exact_pair_results'].append(core.ref(path));verified_all.append(job)
    # No cached result remains valid after even one plan/seal/raw modification.
    for job in verified_all:
        for i,side in enumerate(('reference','candidate')):core.verify(job['args'][i],job['args'][i+2],side)
        for source,pin in job['input_pins'].items():require(core.sha(source)==pin,'Final input revision changed')
    require(all(core.sha(path)==pin for path,pin in input_pins.items()),'Policy/code/proof changed during semantic validation')
    result['passed']=len(result['trial_pairs'])==14 and all(v['passed'] and all(v['checks'].values())for v in result['trial_pairs'])
    published=save(output,result)
    execution=dict(schema='oran.indexed-physical-parity-semantic-execution.v5',created_utc=datetime.now(timezone.utc).isoformat(),completed=True,pid=os.getpid(),fresh_process=True,
        output=published,passed=result['passed'],comparison_code=core.ref(__file__),root_authorization=core.ref(authorization),render_mesh_diagnostic=core.ref(diagnostic),
        cached_exact_producer=core.ref(ROOT/'experiments/run_indexed_parity_parallel_v1.py'),cached_core_comparator=core.ref(core.__file__),cached_full_pairs=14,
        every_plan_seal_model_and_raw_stream_rehashed=True,complete_seals_reverified_before_and_after=True,original_failures_and_raw_encodings_retained=True,
        new_physical_simulations=0,network_numeric_predicate_modified=False,remote_operations=False,primary_admission_authorized=False)
    certificate=save(output.with_name(output.stem+'-execution.json'),execution)
    print(json.dumps(dict(passed=result['passed'],output=published,execution=certificate,pairs=len(result['trial_pairs']))),flush=True)
    return 0 if result['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','oracle-proof','source-diff-proof','cached-results','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    try:return run(args)
    except Exception as error:
        path=Path(args.output).with_name(Path(args.output).stem+'-failed.json')
        if not path.exists():save(path,dict(schema='oran.indexed-physical-parity-semantic-failed.v5',passed=False,error=type(error).__name__+': '+str(error),primary_admission_authorized=False))
        raise


if __name__=='__main__':raise SystemExit(main())

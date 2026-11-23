"""Report-only logical-owner versus compute-host metadata.

No simulator trace, KPI, Evaluation or certificate admission is interpreted.
Existing physical/checksum guards remain responsible for accepting a run.
"""
from copy import deepcopy
from pathlib import Path

from experiments.study_quantification import read,file_sha


def source_refs(methods):
    """Use one frozen method's identical physical label mapping per block."""
    refs=[]
    for block in methods[0]['blocks']if methods else[]:
        score=read(block['score_path']);path=Path(score['input_path']).with_name('label-sources.json')
        pin=score.get('label_sources_sha256')
        if pin:refs.append(dict(path=str(path),sha256=pin))
    return refs


def collect(references):
    rows={}
    for reference in references:
        if file_sha(reference['path'])!=reference['sha256']:raise ValueError('execution_metadata_label_source_changed')
        for source in read(reference['path'])['sources'].values():
            provenance=source['provenance'];job=provenance['job_id'];owner=provenance.get('worker_id')
            checkpoint=Path(source['seal_path']);seal=read(checkpoint)if checkpoint.is_file()else None
            explicit=(seal or{}).get('execution_provenance')
            row=dict(job_id=job,scene=source.get('scene'),seed=provenance.get('seed'),planned_worker=owner,
                logical_owner_worker_id=owner,actual_execution_host=owner,
                host_evidence='recorded_execution_override'if explicit else'planned_equals_actual_without_recorded_override',
                checkpoint_present=seal is not None,checkpoint_completed=(seal or{}).get('completed')is True,
                checkpoint_path=str(checkpoint),checkpoint_sha256=file_sha(checkpoint)if seal else None,
                independently_replicated=False,independence_assessment_basis='derived_from_same_frozen_registered_case_identity',raw_KPI_values_read=False)
            if seal is not None:
                if seal.get('case',{}).get('id')!=job:raise ValueError('execution_checkpoint_identifies_another_job')
                if owner and seal.get('worker_id')not in(None,owner):raise ValueError('execution_logical_owner_differs_from_frozen_plan')
                if not explicit and seal.get('actual_execution_host')not in(None,owner):raise ValueError('execution_host_override_evidence_required')
                if provenance.get('source_archive_sha256')and seal.get('source_archive_sha256')not in(None,provenance['source_archive_sha256']):
                    raise ValueError('execution_source_archive_differs_from_frozen_plan')
                row['source_sha256']=seal.get('source_sha256');row['source_archive_sha256']=seal.get('source_archive_sha256')
                row['actual_execution_wrapper_sha256']=seal.get('runner_sha256')
                row['original_protocol_runner_sha256']=seal.get('original_protocol_runner_sha256',seal.get('runner_sha256'))
            if explicit:
                if (explicit.get('logical_owner_worker_id')!=owner or not explicit.get('actual_execution_host')
                    or seal.get('logical_owner_worker_id')not in(None,owner)
                    or seal.get('actual_execution_host')not in(None,explicit['actual_execution_host'])
                    or explicit.get('execution_wrapper_sha256')!=seal.get('runner_sha256')
                    or explicit.get('original_protocol_runner_sha256')!=seal.get('original_protocol_runner_sha256')):
                    raise ValueError('execution_override_metadata_is_inconsistent')
                if explicit.get('operational_admission_sha256'):
                    if (explicit['actual_execution_host']!=owner or explicit.get('analysis_reserve_s')!=1800
                        or explicit.get('original_protocol_analysis_reserve_s')!=3600 or not explicit.get('deadline_utc')
                        or any(explicit.get(k)is not False for k in('changes_physical_recipe','changes_case_identity','extends_user_deadline'))
                        or explicit.get('kind')not in(None,'operational_reserve_only')or explicit.get('compute_admission_sha256')or explicit.get('parity_proof_sha256')):
                        raise ValueError('operational_execution_override_metadata_is_inconsistent')
                    row.update(execution_kind='operational_reserve_only',execution_kind_basis='derived_from_explicit_same_host_wall_budget_metadata',
                        operational_admission_sha256=explicit['operational_admission_sha256'],analysis_reserve_s=explicit['analysis_reserve_s'],
                        original_protocol_analysis_reserve_s=explicit['original_protocol_analysis_reserve_s'],deadline_utc=explicit['deadline_utc'])
                else:
                    if (explicit.get('kind')!='compute_migrated_supplement'or not explicit.get('compute_admission_sha256')or not explicit.get('parity_proof_sha256')
                        or explicit.get('physical_source_sha256')!=seal.get('source_sha256')
                        or explicit.get('physical_archive_sha256')!=seal.get('source_archive_sha256')
                        or explicit.get('independent_sample')is not False or explicit.get('additional_independent_replicate')is not False):
                        raise ValueError('execution_override_metadata_is_inconsistent')
                    row.update(execution_kind=explicit['kind'],execution_kind_basis='recorded',
                        compute_admission_sha256=explicit['compute_admission_sha256'],parity_proof_sha256=explicit['parity_proof_sha256'])
                row.update(actual_execution_host=explicit['actual_execution_host'],
                    actual_system_hostname=explicit.get('actual_system_hostname'),actual_ssh_endpoint=explicit.get('actual_ssh_endpoint'),
                    actual_execution_wrapper_sha256=explicit['execution_wrapper_sha256'],
                    original_protocol_runner_sha256=explicit['original_protocol_runner_sha256'],
                    recorded_execution_provenance=deepcopy(explicit))
            if job in rows and rows[job]!=row:raise ValueError('execution_metadata_differs_for_one_physical_job')
            rows[job]=row
    ordered=[rows[k]for k in sorted(rows)]
    return dict(schema='oran.report-execution-provenance.v1',rows=ordered,physical_jobs=len(ordered),
        migrated_compute_jobs=sum(r['planned_worker']!=r['actual_execution_host']for r in ordered),
        source_refs=deepcopy(references),only_metadata_read=True,physical_admission_performed=False,
        frozen_scoring_Evaluation_or_label_paths_changed=False,
        interpretation='Logical ownership follows the frozen plan. A recorded execution override shows the actual host and truthful wrapper pins separately. A same-host operational reserve override changes only the recorded wall-time admission reserve and implies no parity experiment. Without an override, planned and actual host are displayed equal; pending jobs are only planned, not verified executions. Derived replication assessments follow the same registered case identity, without inventing missing checkpoint fields.')

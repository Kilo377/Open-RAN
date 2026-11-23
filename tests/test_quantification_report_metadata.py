"""Report-only explanations and compute provenance; no network outcomes."""
from pathlib import Path

import pytest

from experiments import report_execution_provenance as execution
from experiments import report_study_guide as guide
from experiments import study_quantification as study


def fixture(tmp_path,*,migrated=False,pending=False):
    seal_path=tmp_path/'checkpoints/job.json';seal=dict(completed=True,case={'id':'job'},worker_id='agx1',
        source_sha256='physical',source_archive_sha256='archive',runner_sha256='old-runner')
    if migrated:
        seal.update(runner_sha256='actual-wrapper',original_protocol_runner_sha256='old-runner',execution_provenance=dict(
            logical_owner_worker_id='agx1',actual_execution_host='agx2',actual_system_hostname='real-agx2',
            physical_source_sha256='physical',physical_archive_sha256='archive',execution_wrapper_sha256='actual-wrapper',
            original_protocol_runner_sha256='old-runner',kind='compute_migrated_supplement',
            compute_admission_sha256='admission',parity_proof_sha256='parity',independent_sample=False,additional_independent_replicate=False))
    if not pending:study.write(seal_path,seal)
    path=tmp_path/'label-sources.json';study.write(path,{'sources':{'case':dict(provenance={'job_id':'job','worker_id':'agx1','seed':1},
        scene='native',seal_path=str(seal_path),run_path='/must/not/open/data/run')}})
    return [dict(path=str(path),sha256=study.file_sha(path))],seal_path


def test_pending_and_default_execution_display_planned_host_without_claiming_verified_run(tmp_path):
    refs,_=fixture(tmp_path,pending=True);result=execution.collect(refs);row=result['rows'][0]
    assert row['planned_worker']==row['actual_execution_host']=='agx1'and row['checkpoint_completed']is False
    assert result['only_metadata_read']is True and result['physical_admission_performed']is False
    assert result['migrated_compute_jobs']==0


def test_compute_override_retains_logical_owner_truthful_runner_and_metadata_only_io(tmp_path,monkeypatch):
    refs,path=fixture(tmp_path,migrated=True);real=execution.read;opened=[]
    def read_metadata(value):
        assert Path(value)in(Path(refs[0]['path']),path)
        opened.append(str(value));return real(value)
    monkeypatch.setattr(execution,'read',read_metadata);result=execution.collect(refs);row=result['rows'][0]
    assert row['logical_owner_worker_id']=='agx1'and row['actual_execution_host']=='agx2'
    assert row['original_protocol_runner_sha256']=='old-runner'and row['actual_execution_wrapper_sha256']=='actual-wrapper'
    assert row['compute_admission_sha256']=='admission'and row['parity_proof_sha256']=='parity'
    assert result['migrated_compute_jobs']==1 and row['independently_replicated']is False
    assert len(opened)==2
    seal=real(path);seal['runner_sha256']='old-runner';study.write(path,seal)
    with pytest.raises(ValueError,match='inconsistent'):execution.collect(refs)


def test_frozen_source_pin_and_logical_owner_cannot_be_changed_for_reporting(tmp_path):
    refs,path=fixture(tmp_path);seal=study.read(path);seal['worker_id']='agx2';study.write(path,seal)
    with pytest.raises(ValueError,match='logical_owner'):execution.collect(refs)
    study.write(refs[0]['path'],{'sources':{}})
    with pytest.raises(ValueError,match='label_source_changed'):execution.collect(refs)


def test_same_host_operational_reserve_is_derived_without_inventing_kind_or_parity(tmp_path):
    refs,path=fixture(tmp_path,migrated=True);seal=study.read(path)
    seal['execution_provenance']=dict(logical_owner_worker_id='agx1',actual_execution_host='agx1',actual_system_hostname='real-agx1',
        original_protocol_runner_sha256='old-runner',execution_wrapper_sha256='actual-wrapper',operational_admission_sha256='operational',
        analysis_reserve_s=1800,original_protocol_analysis_reserve_s=3600,deadline_utc='2026-11-22T12:00:00+00:00',
        changes_physical_recipe=False,changes_case_identity=False,extends_user_deadline=False)
    study.write(path,seal);result=execution.collect(refs);row=result['rows'][0]
    assert row['execution_kind']=='operational_reserve_only'and row['planned_worker']==row['actual_execution_host']=='agx1'
    assert row['operational_admission_sha256']=='operational'and result['migrated_compute_jobs']==0
    assert 'kind'not in row['recorded_execution_provenance']and 'independent_sample'not in row['recorded_execution_provenance']
    assert 'parity_proof_sha256'not in row and row['execution_kind_basis'].startswith('derived_')
    seal['execution_provenance']['extends_user_deadline']=True;study.write(path,seal)
    with pytest.raises(ValueError,match='operational_.*inconsistent'):execution.collect(refs)


def test_concise_forward_home_explains_scores_pools_and_projector_version_boundary(tmp_path):
    pages=[]
    for comparison in('joint_five_matched','profileG_pairwise_ablation'):
        for partition in guide.PARTITIONS:
            pages.append(dict(comparison=comparison,partition=partition,html=str(tmp_path/'reports'/comparison/partition/'index.html')))
    guide.write_landing_page(tmp_path,title='Forward validation',status='Synthetic fixture only',pages=pages,forward=True)
    page=(tmp_path/'index.html').read_text()
    assert page.count('<article>')==6 and page.count('<a href=')==6
    assert page.index('<h2>Same-G ablation')<page.index('<h2>Five-method comparison')
    for phrase in('not a probability','Zero estimated risk','only two independent','strict target','complete frozen versions','after original outcome availability'):
        assert phrase.lower()in page.lower()
    assert 'same two seeds'in page and 'not independent seed replications'in page


def test_synthetic_preview_marks_figures_and_rare_failure_as_percentage_points(tmp_path):
    pytest.importorskip('matplotlib')
    from experiments.preview_quantification_report import generate
    result=generate(tmp_path/'preview')
    assert result['real_study_data_read']is False and len(result['reports'])==2
    page=Path(result['index']).read_text();assert 'SYNTHETIC PREVIEW'in page and 'Same-G ablation'in page
    for entry in result['reports']:
        output=Path(entry['html']).parent;body=Path(entry['html']).read_text()
        assert '+0.01 percentage-point'in body and 'relative percentage increase from zero is undefined'in body
        assert '0 → 0.010 %'in body
        svg=(output/'figures/risk-outcome-primary.svg').read_text()
        assert 'SYNTHETIC PREVIEW'in svg and 'normalized points'in svg and 'dimensionless'in svg
        summary=study.read(output/'report-summary.json')
        assert any(r['zero_risk_unmet_rows']for r in summary['availability'])
        assert any(r['label_status_counts'].get('incomplete_requested_kpis')for r in summary['availability'])
    assert 'two independent random seeds'in page

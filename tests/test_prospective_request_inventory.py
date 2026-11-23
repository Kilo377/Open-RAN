"""Future-arm eligibility is a remote metadata fact, never a KPI decision."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments import prospective_request_inventory as inventory


def scan(tmp_path,name,*,start,end,requests=(),partial=(),running=(),unknown=()):
    bundle=tmp_path/'bundle.json'
    if not bundle.exists():bundle.write_text('{"sealed":true}')
    code=dict(path=str(bundle),sha256=inventory._sha(bundle));workers=[]
    for parity in range(2):
        rows=[]
        for i in range(132):
            if i%2!=parity:continue
            ident=f'{i:032x}';exists=i in requests;parts=['preserved/'+ident]if i in partial else[];members=['running']if i in running else[]
            rows.append(dict(job_id=ident,origin_key='origin'+str(i%6),partition='main'if i<120 else'supplement',worker_id='agx'+str(parity+1),arm='arm',
                remote_request_path='/owned/jobs/'+ident+'/data/request.json',exists=exists,request_mtime_ns=100000000000 if exists else None,
                request_size_bytes=100 if exists else None,historical_partial_directories=parts,worker_status_memberships=members,
                worker_status_known=i not in unknown,ever_started_evidence=bool(exists or parts or members)))
        worker=dict(worker_id='agx'+str(parity+1),jobs=rows,source_sha256='source');worker['response_sha256']=inventory._digest(worker);workers.append(worker)
    value=dict(schema='oran.prospective-request-scan.v1',local_started_unix_s=start,local_finished_unix_s=end,frozen_code_bundle=code,
        jobs_grid_sha256='same132',expected_source_pins={'agx1':'source','agx2':'source'},inventory_module_sha256='samehelper',workers=workers,
        code_bundle_persisted_before_scan=True,after_kpi_values_read=False,remote_writes_performed=False)
    value['scan_sha256']=inventory._digest(value);path=tmp_path/(name+'.json');path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=inventory._sha(path)),code


def test_only_jobs_absent_from_both_scans_and_all_history_are_eligible(tmp_path):
    before,code=scan(tmp_path,'before',start=1,end=2,requests=(1,),partial=(2,),running=(3,),unknown=(4,))
    after,_=scan(tmp_path,'after',start=5,end=6,requests=(1,5),partial=(2,),running=(3,),unknown=(4,))
    result=inventory.paired_eligibility(before,after,code_bundle=code,scoring_started_unix_s=3,scoring_finished_unix_s=4)
    assert result['eligible_physical_jobs']==127 and result['diagnostic_physical_jobs']==5
    assert f'{5:032x}'not in result['eligible_job_ids']
    assert any(row['job_id']==f'{2:032x}'for row in result['diagnostic_only'])


def test_unchanged_request_absence_does_not_hide_a_completed_or_moved_job(tmp_path):
    before,code=scan(tmp_path,'before',start=1,end=2,partial=(10,),running=(11,))
    after,_=scan(tmp_path,'after',start=5,end=6,partial=(10,),running=(11,))
    result=inventory.paired_eligibility(before,after,code_bundle=code,scoring_started_unix_s=3,scoring_finished_unix_s=4)
    assert result['eligible_physical_jobs']==130


@pytest.mark.parametrize('change',['hash','grid','bundle','time','response'])
def test_changed_evidence_or_unbracketed_scoring_is_rejected(tmp_path,change):
    before,code=scan(tmp_path,'before',start=1,end=2);after,_=scan(tmp_path,'after',start=5,end=6)
    path=Path(after['path']);body=json.loads(path.read_text())
    if change=='hash':body['local_started_unix_s']=100
    elif change=='grid':body['jobs_grid_sha256']='other'
    elif change=='bundle':body['frozen_code_bundle']['sha256']='other'
    elif change=='time':body['local_started_unix_s']=3
    else:body['workers'][0]['jobs'][0]['exists']=True
    if change!='hash':body['scan_sha256']=inventory._digest({k:v for k,v in body.items()if k!='scan_sha256'})
    path.write_text(json.dumps(body));after['sha256']=inventory._sha(path)
    with pytest.raises(ValueError):inventory.paired_eligibility(before,after,code_bundle=code,scoring_started_unix_s=3,scoring_finished_unix_s=4)

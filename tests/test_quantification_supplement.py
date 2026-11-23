"""Independent supplement IDs, source reuse and original-matrix protection."""
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.quantification_supplement import assignment,digest,verify_plan


def plan():
    apps={key:'xapp:'+key for key in'ABCDEF'}
    value=dict(schema='oran.quantification-supplement-plan.v1',source_version='v2',physical_run_count=12,baseline_slot=2040,observe_slots=2000,total_slots=4440,
        heldout_after_kpis_used=False,original_plan_or_seals_changed=False,apps=apps,arms=[dict(id=a,app_ids=[apps[k]for k in a])for a in('BCE','BCDE')],
        group_id_prefix='study-supplement-arm:',origins=[dict(origin_key=f'{scene}_s{seed}',scene=scene,seed=seed,worker_id=('agx1','agx2')[(i+j)%2],origin_id=f'origin{i}{j}')for i,scene in enumerate(('native_1x','steady_2x','burst_2x'))for j,seed in enumerate((20261201,20261202))])
    value['plan_sha256']=digest(value);return value


def test_two_new_arms_are_separate_balanced_and_deterministic():
    p=plan();before=deepcopy(p);grid=assignment(p,p['plan_sha256']);rows=[r for cases in grid.values()for r in cases]
    assert p==before and {key:len(cases)for key,cases in grid.items()}=={'agx1':6,'agx2':6}
    assert len({r['id']for r in rows})==12 and {r['arm']for r in rows}=={'BCE','BCDE'}
    assert all(r['group_id'].startswith('study-supplement-arm:')for r in rows)
    assert assignment(p,p['plan_sha256'])==grid


def test_supplement_cannot_silently_change_the_matrix_or_hash():
    p=plan();pin=p['plan_sha256'];p['arms'].pop()
    with pytest.raises(ValueError,match='digest'):verify_plan(p,pin)
    p['plan_sha256']=digest({k:v for k,v in p.items()if k!='plan_sha256'})
    with pytest.raises(ValueError,match='arms'):verify_plan(p,p['plan_sha256'])


def test_supplement_reuses_normal_hot_launcher_and_original_guard(tmp_path):
    from experiments.quantification_study import command
    p=plan();case=assignment(p,p['plan_sha256'])['agx1'][0];root=Path('/owned/source-v2');out=root/'outputs/prospective_supplement'
    args=command(root,out,case,p)
    assert '--live-xapps'in args and '--evaluation-origin-file'in args
    assert args[args.index('--evaluation-origin-file')+1].startswith(str(out/'jobs'))
    assert '--perturbation-spec'not in args and '--sleep-profile-spec'not in args

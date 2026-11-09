"""Real additive quantile boosting and safe JSON persistence invariants."""
import builtins
import json
import math
import pytest

from ric.near_rt.pools.chongyu.online_qgbdt import OnlineQGBDT, parameters


def data(model,n=16):
    for i in range(n): model.add([i/16,float(i%2)/2], .7+.1*math.sin(i))
    return model


def fixture(tmp_path):
    """Hand-built inference fixture only; not a trained deployment artifact."""
    model=data(OnlineQGBDT(['load','candidate_offload_fraction']),12)
    path=tmp_path/'model.json';model.save(path);p=json.loads(path.read_text())
    p.update(trained_samples=12,fit_count=1,fit_parameters=parameters(137),initial_prediction=.8,
        forest=[dict(children_left=[-1],children_right=[-1],feature=[-2],threshold=[-2.],value=[.01]) for _ in range(32)])
    path.write_text(json.dumps(p));return path,p


def test_untrained_rows_resume_without_sklearn_or_fake_fit(tmp_path,monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith('sklearn'):raise AssertionError('Premature sklearn import')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    model=data(OnlineQGBDT(['load','candidate_offload_fraction']),8)
    assert not model.fit() and model.predict([[.2,.5]]) is None
    path=tmp_path/'rows.json';model.save(path)
    loaded=OnlineQGBDT.load(path,model.feature_names)
    assert loaded.sample_count==8 and loaded.fit_count==0 and loaded.fit_parameters is None
    assert loaded._x==model._x and loaded._y==model._y


def test_real_quantile_additive_predictions_match_sklearn_and_roundtrip(tmp_path,monkeypatch):
    pytest.importorskip('sklearn')
    from sklearn.ensemble import GradientBoostingRegressor
    model=data(OnlineQGBDT(['load','candidate_offload_fraction']))
    assert model.fit()
    reference=GradientBoostingRegressor(**parameters(137)).fit(model._x,model._y)
    candidates=[[.123456789,.5],[.9,0],[.50000001,.5]]
    assert model.predict(candidates)==pytest.approx(reference.predict(candidates).tolist(),abs=1e-12,rel=1e-12)
    assert model.fit_parameters==parameters(137) and len(model._forest)==32
    path=tmp_path/'trained.json';model.save(path);original=path.read_bytes()
    loaded=OnlineQGBDT.load(path,model.feature_names)
    assert loaded.predict(candidates)==pytest.approx(reference.predict(candidates).tolist(),abs=1e-12,rel=1e-12)
    assert path.read_bytes()==original and loaded.trained_samples==16 and loaded.fit_count==1
    loaded.add([.2,.5],.8);loaded.fit()
    assert loaded.trained_samples==17 and loaded.fit_count==2


def test_standard_library_inference_is_additive_not_forest_average(tmp_path,monkeypatch):
    path,_=fixture(tmp_path)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith('sklearn'):raise AssertionError('Inference imported sklearn')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    model=OnlineQGBDT.load(path,['load','candidate_offload_fraction'])
    assert model.predict([[.1,.5]])==pytest.approx([.8+.05*32*.01])


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(version=True),lambda p:p.update(label='wrong'),
    lambda p:p.update(initial_prediction=math.nan),lambda p:p.update(trained_samples=25),
    lambda p:p.update(fit_parameters=dict(parameters(137),alpha=.1)),
    lambda p:p.update(fit_parameters=dict(parameters(137),unknown=1)),
    lambda p:p['fit_parameters'].update(max_depth=True),
    lambda p:p['forest'][0]['children_left'].__setitem__(0,0),
    lambda p:p['forest'][0]['value'].__setitem__(0,[.1]),
    lambda p:p['forest'].pop(),lambda p:p.update(fit_s=-1),
    lambda p:p['x'].append([1.,.5]),
])
def test_tampering_cannot_execute_or_load_invalid_model(tmp_path,mutation):
    path,p=fixture(tmp_path);mutation(p);path.write_text(json.dumps(p))
    with pytest.raises(ValueError):OnlineQGBDT.load(path,['load','candidate_offload_fraction'])


@pytest.mark.parametrize('x,y',[([1.],.5),([True,.5],.5),([1.,.5],math.inf),([1.,math.nan],.5)])
def test_measured_sample_validation_is_atomic(x,y):
    model=OnlineQGBDT(['load','candidate_offload_fraction'])
    with pytest.raises(ValueError):model.add(x,y)
    assert model.sample_count==0


def test_row_cap_duplicate_keys_and_wrong_feature_order(tmp_path):
    model=data(OnlineQGBDT(['load','candidate_offload_fraction']),24)
    with pytest.raises(ValueError):model.add([0.,0.],.5)
    path=tmp_path/'m.json';model.save(path)
    with pytest.raises(ValueError):OnlineQGBDT.load(path,['candidate_offload_fraction','load'])
    path.write_text('{"version":1,"version":1}')
    with pytest.raises(ValueError):OnlineQGBDT.load(path,['load','candidate_offload_fraction'])

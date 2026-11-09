"""Actual fixed MLP/scalers, safe inference, and measured-row persistence."""
import builtins
import json
import math
import pytest

from ric.near_rt.pools.chongyu.online_throughput_mlp import (OnlineThroughputMLP,parameters,mean_loss_estimator_class,_finite_difference_check,VERIFIED_CONVENTIONS,MAX_ROWS)


def data(n=28):
    model=OnlineThroughputMLP(['load','action'],['goodput','failures'])
    for i in range(n):model.add([i/n,float(i%4)],[10+math.sin(i),float(i%3)])
    return model


def fixture(tmp_path):
    """Explicit test-only network, not a fabricated deployment artifact."""
    model=data(24);path=tmp_path/'m.json';model.save(path);p=json.loads(path.read_text())
    p.update(coefs=[[[.1]*8 for _ in range(2)],[[.2,.3] for _ in range(8)]],
        intercepts=[[0.]*8,[.1,.2]],x_scaler={'mean':[0.,0.],'scale':[1.,2.]},
        y_scaler={'mean':[5.,2.],'scale':[2.,3.]},fit_parameters=parameters(173),
        fit_diagnostics={'n_iter':10,'loss':.1,'converged':True},trained_samples=24,fit_count=1)
    p['version']=1;p.pop('optimizer_metadata') # Explicit legacy inspection-only inference fixture.
    path.write_text(json.dumps(p));return path,p


def test_untrained_measured_rows_persist_without_dependency_or_fit(tmp_path,monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith('sklearn'):raise AssertionError('Premature sklearn import')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    model=data(8);assert not model.fit() and model.predict([[.2,1.]]) is None
    path=tmp_path/'rows.json';model.save(path)
    loaded=OnlineThroughputMLP.load(path,model.feature_names,model.target_names)
    assert loaded._x==model._x and loaded._y==model._y and loaded.fit_count==0


def test_actual_sklearn_mlp_and_scalers_match_json_roundtrip(tmp_path):
    pytest.importorskip('sklearn')
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    model=data();assert model.fit()
    sx,sy=StandardScaler(),StandardScaler();x=sx.fit_transform(model._x);y=sy.fit_transform(model._y)
    params=parameters(173);params['hidden_layer_sizes']=tuple(params['hidden_layer_sizes'])
    estimator,proof=mean_loss_estimator_class(len(model.target_names))
    reference=estimator(**params).fit(x,y)
    candidates=[[.123456789,0.],[.7,2.]]
    expected=sy.inverse_transform(reference.predict(sx.transform(candidates))).tolist()
    actual=model.predict(candidates)
    for a,b in zip(actual,expected):assert a==pytest.approx(b,abs=1e-10,rel=1e-10)
    path=tmp_path/'real.json';model.save(path);original=path.read_bytes()
    loaded=OnlineThroughputMLP.load(path,model.feature_names,model.target_names)
    assert loaded.predict(candidates)==actual and path.read_bytes()==original
    assert loaded.trained_samples==28 and loaded.fit_count==1 and loaded.fit_parameters==parameters(173)
    assert loaded.fit_diagnostics['n_iter']>0
    assert loaded.optimizer_verified and loaded.optimizer_metadata==model.optimizer_metadata
    assert json.loads(path.read_text())['version']==2


def test_frozen_standard_library_tanh_inference(tmp_path,monkeypatch):
    path,_=fixture(tmp_path)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith('sklearn'):raise AssertionError('Inference imported sklearn')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    model=OnlineThroughputMLP.load(path,['load','action'],['goodput','failures'])
    h=math.tanh(.1*.5+.1*.5)
    assert model.predict([[.5,1.]])[0]==pytest.approx([(8*h*.2+.1)*2+5,(8*h*.3+.2)*3+2])


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(version=True),lambda p:p.update(target_names=['wrong']),
    lambda p:p['coefs'][0].pop(),lambda p:p['coefs'][1][0].append(1.),
    lambda p:p['coefs'][0][0].__setitem__(0,math.nan),lambda p:p['intercepts'][0].pop(),
    lambda p:p['x_scaler']['scale'].__setitem__(0,0),lambda p:p['y_scaler'].update(extra=1),
    lambda p:p['fit_parameters'].update(hidden_layer_sizes=[8.]),
    lambda p:p['fit_parameters'].update(unknown=True),lambda p:p.update(trained_samples=41),
    lambda p:p['fit_diagnostics'].update(n_iter=301),lambda p:p.update(fit_s=-1),
])
def test_tampered_models_are_rejected(tmp_path,mutation):
    path,p=fixture(tmp_path);mutation(p);path.write_text(json.dumps(p))
    with pytest.raises(ValueError):OnlineThroughputMLP.load(path,['load','action'],['goodput','failures'])


def test_cap_atomic_rows_duplicate_keys_and_schema_order(tmp_path):
    model=data(MAX_ROWS)
    with pytest.raises(ValueError):model.add([.1,0.],[1.,0.])
    empty=OnlineThroughputMLP(['x'],['y'])
    with pytest.raises(ValueError):empty.add([True],[1.])
    assert empty.sample_count==0
    path=tmp_path/'p.json';model.save(path)
    with pytest.raises(ValueError):OnlineThroughputMLP.load(path,['action','load'],model.target_names)
    path.write_text('{"version":1,"version":1}')
    with pytest.raises(ValueError):OnlineThroughputMLP.load(path,model.feature_names,model.target_names)


def test_actual_zero_guard_targets_preserve_raw_sklearn_residuals():
    pytest.importorskip('sklearn')
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    model=OnlineThroughputMLP(['context','action'],['goodput','guard_0','guard_1'])
    for i in range(24):model.add([i/23,float(i%4)],[10+.1*i,0.,0.])
    assert model.fit()
    sx,sy=StandardScaler(),StandardScaler();xs=sx.fit_transform(model._x);ys=sy.fit_transform(model._y)
    p=parameters(173);p['hidden_layer_sizes']=tuple(p['hidden_layer_sizes'])
    estimator,_=mean_loss_estimator_class(len(model.target_names))
    reference=estimator(**p).fit(xs,ys)
    rows=[[.1,0.],[.5,2.],[.9,3.]]
    expected=sy.inverse_transform(reference.predict(sx.transform(rows))).tolist()
    for a,b in zip(model.predict(rows),expected):assert a==pytest.approx(b,abs=1e-10,rel=1e-10)
    assert all(y[1:]==[0.,0.] for y in model._y) # Training labels remain exactly observed zeros.


@pytest.mark.parametrize('outputs',[1,3,9])
@pytest.mark.parametrize('alpha',[0.,1.])
def test_verified_project_estimator_matches_scalar_objective_finite_differences(outputs,alpha):
    pytest.importorskip('sklearn')
    estimator,proof=mean_loss_estimator_class(outputs)
    check=_finite_difference_check(estimator,outputs,alpha)
    assert check['native_relative_error']<1e-7
    assert proof['gradient_convention'] in VERIFIED_CONVENTIONS
    assert proof['consistency_max_relative_error']<=proof['relative_tolerance']==1e-5
    assert {c['outputs'] for c in proof['checks']}=={1,outputs}
    if outputs==1:
        from sklearn.neural_network import MLPRegressor
        assert estimator is MLPRegressor # Single-output training is unchanged.


def test_already_consistent_library_is_not_corrected_twice():
    pytest.importorskip('sklearn')
    coherent,_=mean_loss_estimator_class(3)
    unchanged,proof=mean_loss_estimator_class(3,coherent)
    assert unchanged is coherent
    assert proof['gradient_convention']=='native_mean_output_consistent'
    assert _finite_difference_check(unchanged,3,1.)['native_relative_error']<1e-7


def test_unknown_loss_gradient_convention_fails_closed():
    pytest.importorskip('sklearn')
    coherent,_=mean_loss_estimator_class(3)
    class UnexpectedGradient(coherent):
        def _backprop(self,*args,**kwargs):
            loss,cg,ig=super()._backprop(*args,**kwargs)
            return loss,[v*3 for v in cg],[v*3 for v in ig]
    with pytest.raises(RuntimeError,match='unsupported'):
        mean_loss_estimator_class(3,UnexpectedGradient)


def test_legacy_artifact_inspection_and_resave_cannot_claim_verified_optimizer(tmp_path):
    path,_=fixture(tmp_path);original=path.read_bytes()
    model=OnlineThroughputMLP.load(path,['load','action'],['goodput','failures'])
    assert model.trained and not model.optimizer_verified
    assert model.optimizer_metadata['gradient_convention']=='legacy_unrecorded'
    migrated=tmp_path/'migrated.json';model.save(migrated)
    p=json.loads(migrated.read_text())
    assert p['version']==2 and p['optimizer_metadata']['gradient_convention']=='legacy_unrecorded'
    assert not OnlineThroughputMLP.load(migrated,model.feature_names,model.target_names).optimizer_verified
    assert path.read_bytes()==original


@pytest.mark.parametrize('mutation',[
    lambda p:p['optimizer_metadata'].update(gradient_convention='claim_without_proof'),
    lambda p:p['optimizer_metadata'].update(extra=True),
    lambda p:p['optimizer_metadata'].update(consistency_max_relative_error=1.),
])
def test_optimizer_proof_metadata_is_strictly_validated(tmp_path,mutation):
    path,_=fixture(tmp_path)
    model=OnlineThroughputMLP.load(path,['load','action'],['goodput','failures'])
    migrated=tmp_path/'v2.json';model.save(migrated);p=json.loads(migrated.read_text())
    mutation(p);migrated.write_text(json.dumps(p))
    with pytest.raises(ValueError):OnlineThroughputMLP.load(migrated,model.feature_names,model.target_names)

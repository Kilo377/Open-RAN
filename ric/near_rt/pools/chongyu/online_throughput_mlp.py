"""Measured MLP with a runtime-verified mean-output gradient and safe JSON.

Only a project-local estimator is corrected when the installed library fails
finite differences in the known output-averaging way. No global monkeypatch.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
import warnings

from .online_rf import _integer, _names, _number, _row, _unique_object

FORMAT='chongyu.online-throughput-mlp'
VERSION=2
MIN_ROWS=24
MAX_ROWS=1024
HIDDEN=8


def parameters(seed):
    return dict(hidden_layer_sizes=[HIDDEN],activation='tanh',solver='lbfgs',alpha=1.,
                max_iter=300,max_fun=5000,tol=1e-5,random_state=seed)



OBJECTIVE='half_mean_standardized_squared_error_plus_L2'
GRADIENT_RELATIVE_TOLERANCE=1e-5
VERIFIED_CONVENTIONS=('native_mean_output_consistent','project_corrected_mean_output')


def legacy_optimizer_metadata():
    return dict(objective=OBJECTIVE,gradient_convention='legacy_unrecorded',
                sklearn_version=None,source_sha256=None,relative_tolerance=GRADIENT_RELATIVE_TOLERANCE,
                consistency_max_relative_error=None,checks=[])


def _finite_difference_check(base_class,outputs,alpha):
    """No fitting: compare installed backprop with its reported scalar loss."""
    import numpy as np
    from sklearn.neural_network._multilayer_perceptron import _pack
    rng=np.random.RandomState(19)
    x=np.asarray([[.2,-.1],[.5,.7],[-.3,.4]])
    model=base_class(hidden_layer_sizes=(2,),activation='tanh',solver='lbfgs',alpha=alpha)
    model.n_layers_=3;model.n_outputs_=outputs;model.out_activation_='identity'
    model.coefs_=[rng.normal(0,.2,(2,2)),rng.normal(0,.2,(2,outputs))]
    model.intercepts_=[rng.normal(0,.1,2),rng.normal(0,.1,outputs)]
    y=rng.normal(0,.3,(3,outputs))
    def backprop():
        return model._backprop(x,y,None,[x.copy(),np.empty((3,2)),np.empty((3,outputs))],
            [np.empty((3,2)),np.empty((3,outputs))],
            [np.zeros_like(w) for w in model.coefs_],[np.zeros_like(b) for b in model.intercepts_])
    loss,cg,ig=backprop();native=_pack(cg,ig);numeric=[]
    for parameter in model.coefs_+model.intercepts_:
        for index in np.ndindex(parameter.shape):
            old=parameter[index]
            parameter[index]=old+1e-6;upper=backprop()[0]
            parameter[index]=old-1e-6;lower=backprop()[0]
            parameter[index]=old
            numeric.append((upper-lower)/2e-6)
    numeric=np.asarray(numeric)
    regularizer=_pack([alpha*w/len(x) for w in model.coefs_],[np.zeros_like(b) for b in model.intercepts_])
    corrected=(native-regularizer)/outputs+regularizer
    norm=max(float(np.linalg.norm(numeric)),1e-12)
    return dict(outputs=outputs,alpha=alpha,
        native_relative_error=float(np.linalg.norm(native-numeric)/norm),
        corrected_relative_error=float(np.linalg.norm(corrected-numeric)/norm))


def mean_loss_estimator_class(output_count,base_class=None):
    """Behavioral guard prevents correcting a future already-correct library.

    This checks tiny deterministic fixtures, never the experiment's rows and
    never an optimized fit. Unexpected gradients fail closed.
    """
    import inspect
    import sklearn
    from sklearn.neural_network import MLPRegressor
    from sklearn.neural_network._multilayer_perceptron import LOSS_FUNCTIONS
    base_class=base_class or MLPRegressor
    output_count=_integer(output_count,'output count',1,512)
    try:
        checks=[_finite_difference_check(base_class,n,alpha) for n in sorted({1,output_count}) for alpha in (0.,1.)]
        native_ok=all(c['native_relative_error']<=GRADIENT_RELATIVE_TOLERANCE for c in checks)
        corrected_ok=all(c['corrected_relative_error']<=GRADIENT_RELATIVE_TOLERANCE for c in checks)
        if native_ok:convention='native_mean_output_consistent'
        elif corrected_ok:convention='project_corrected_mean_output'
        else:raise RuntimeError('Installed MLP loss/gradient convention is unsupported')
        source='\n'.join(inspect.getsource(f) for f in (base_class._backprop,base_class._compute_loss_grad,LOSS_FUNCTIONS['squared_error']))
        metadata=dict(objective=OBJECTIVE,gradient_convention=convention,
            sklearn_version=sklearn.__version__,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            relative_tolerance=GRADIENT_RELATIVE_TOLERANCE,
            consistency_max_relative_error=max(c['native_relative_error'] if native_ok else c['corrected_relative_error'] for c in checks),checks=checks)
    except (AttributeError,TypeError,ValueError,OSError) as exc:
        raise RuntimeError('Cannot verify installed MLP numerical convention') from exc
    if native_ok:return base_class,metadata

    class MeanOutputLossMLP(base_class):
        def _backprop(self,x,y,sample_weight,activations,deltas,coef_grads,intercept_grads):
            if sample_weight is not None:raise ValueError('Project MLP correction supports unweighted measured rows only')
            loss,coef_grads,intercept_grads=super()._backprop(x,y,sample_weight,activations,deltas,coef_grads,intercept_grads)
            outputs=self.n_outputs_
            if outputs>1:
                for i in range(len(coef_grads)):
                    regularizer=self.alpha*self.coefs_[i]/len(x)
                    coef_grads[i]=(coef_grads[i]-regularizer)/outputs+regularizer
                    intercept_grads[i]=intercept_grads[i]/outputs
            return loss,coef_grads,intercept_grads
    return MeanOutputLossMLP,metadata


def _optimizer_metadata(value,outputs):
    keys={'objective','gradient_convention','sklearn_version','source_sha256','relative_tolerance','consistency_max_relative_error','checks'}
    if not isinstance(value,dict) or set(value)!=keys or value['objective']!=OBJECTIVE:raise ValueError('Invalid MLP optimizer metadata schema')
    if value['gradient_convention']=='legacy_unrecorded':
        if value!=legacy_optimizer_metadata():raise ValueError('Invalid legacy MLP optimizer metadata')
        return dict(value)
    if value['gradient_convention'] not in VERIFIED_CONVENTIONS:raise ValueError('Unverified MLP optimizer convention')
    if not isinstance(value['sklearn_version'],str) or not value['sklearn_version']:raise ValueError('Missing MLP library version')
    digest=value['source_sha256']
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid MLP source fingerprint')
    if type(value['relative_tolerance']) is not float or value['relative_tolerance']!=GRADIENT_RELATIVE_TOLERANCE:raise ValueError('Invalid MLP gradient tolerance')
    checks=value['checks']
    expected={(n,a) for n in {1,outputs} for a in (0.,1.)}
    if not isinstance(checks,list) or len(checks)!=len(expected):raise ValueError('Missing MLP gradient checks')
    actual=set();errors=[]
    for check in checks:
        if not isinstance(check,dict) or set(check)!={'outputs','alpha','native_relative_error','corrected_relative_error'}:raise ValueError('Invalid MLP gradient check')
        _integer(check['outputs'],'gradient output count',1,512)
        alpha=_number(check['alpha'],'gradient check alpha');actual.add((check['outputs'],alpha))
        for key in ('native_relative_error','corrected_relative_error'):
            if _number(check[key],key)<0:raise ValueError('Negative MLP gradient error')
        errors.append(check['native_relative_error'] if value['gradient_convention']=='native_mean_output_consistent' else check['corrected_relative_error'])
    if actual!=expected or max(errors)>GRADIENT_RELATIVE_TOLERANCE:raise ValueError('MLP gradient checks did not verify the objective')
    maximum=_number(value['consistency_max_relative_error'],'maximum gradient error')
    if maximum<0 or abs(maximum-max(errors))>1e-15:raise ValueError('Inconsistent MLP gradient proof')
    return value


def _matrix(value,rows,cols,name):
    if not isinstance(value,list) or len(value)!=rows:raise ValueError('Invalid '+name+' shape')
    return [_row(row,cols,name) for row in value]


def _scaler(value,width):
    if not isinstance(value,dict) or set(value)!={'mean','scale'}:raise ValueError('Invalid MLP scaler schema')
    result={k:_row(value[k],width,'scaler '+k) for k in value}
    if any(v<=0 for v in result['scale']):raise ValueError('Invalid MLP scaler scale')
    return result


class OnlineThroughputMLP:
    def __init__(self,feature_names,target_names,seed=173):
        self.feature_names=_names(feature_names);self.target_names=_names(target_names)
        self.seed=_integer(seed,'seed',0,2**32-1)
        self._x,self._y=[],[]
        self.coefs,self.intercepts=[],[]
        self.x_scaler=self.y_scaler=self.fit_parameters=self.fit_diagnostics=self.optimizer_metadata=None
        self.trained_samples=self.fit_count=self.prediction_batches=0
        self.fit_s=self.predict_s=0.
        self.artifact_sha256=None

    @property
    def sample_count(self):return len(self._x)

    @property
    def trained(self):return bool(self.coefs)

    @property
    def optimizer_verified(self):
        return self.trained and self.optimizer_metadata is not None and self.optimizer_metadata['gradient_convention'] in VERIFIED_CONVENTIONS

    def add(self,features,targets):
        if self.sample_count>=MAX_ROWS:raise ValueError('MLP measured-row cap reached')
        x=_row(features,len(self.feature_names),'features')
        y=_row(targets,len(self.target_names),'measured targets')
        self._x.append(x);self._y.append(y)

    def fit(self):
        if self.sample_count<MIN_ROWS:return False
        from sklearn.preprocessing import StandardScaler
        from sklearn.exceptions import ConvergenceWarning
        started=time.perf_counter()
        estimator,optimizer_metadata=mean_loss_estimator_class(len(self.target_names))
        sx,sy=StandardScaler(),StandardScaler()
        x=sx.fit_transform(self._x);y=sy.fit_transform(self._y)
        p=parameters(self.seed);p['hidden_layer_sizes']=tuple(p['hidden_layer_sizes'])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always',ConvergenceWarning)
            model=estimator(**p).fit(x,y)
        self.coefs=[v.tolist() for v in model.coefs_]
        self.intercepts=[v.tolist() for v in model.intercepts_]
        self.x_scaler=dict(mean=sx.mean_.tolist(),scale=sx.scale_.tolist())
        self.y_scaler=dict(mean=sy.mean_.tolist(),scale=sy.scale_.tolist())
        self.fit_parameters=parameters(self.seed)
        self.optimizer_metadata=optimizer_metadata
        self.fit_diagnostics=dict(n_iter=int(model.n_iter_),loss=float(model.loss_),
            converged=not any(issubclass(v.category,ConvergenceWarning) for v in recorded))
        self.trained_samples=self.sample_count;self.fit_count+=1
        self.fit_s+=time.perf_counter()-started
        return True

    def predict(self,candidates):
        if not isinstance(candidates,(list,tuple)):raise ValueError('Invalid MLP candidates')
        rows=[_row(v,len(self.feature_names),'candidate') for v in candidates]
        if not self.trained:return None
        started=time.perf_counter();result=[]
        for row in rows:
            value=[(v-m)/s for v,m,s in zip(row,self.x_scaler['mean'],self.x_scaler['scale'])]
            for layer,(weights,bias) in enumerate(zip(self.coefs,self.intercepts)):
                value=[bias[j]+sum(value[i]*weights[i][j] for i in range(len(value))) for j in range(len(bias))]
                if layer==0:value=[math.tanh(v) for v in value]
            result.append(_row([v*s+m for v,m,s in zip(value,self.y_scaler['mean'],self.y_scaler['scale'])],len(self.target_names),'MLP prediction'))
        self.prediction_batches+=1;self.predict_s+=time.perf_counter()-started
        return result

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(format=FORMAT,version=VERSION,feature_names=self.feature_names,target_names=self.target_names,
            seed=self.seed,x=self._x,y=self._y,coefs=self.coefs,intercepts=self.intercepts,
            x_scaler=self.x_scaler,y_scaler=self.y_scaler,fit_parameters=self.fit_parameters,
            fit_diagnostics=self.fit_diagnostics,optimizer_metadata=self.optimizer_metadata,trained_samples=self.trained_samples,fit_count=self.fit_count,
            prediction_batches=self.prediction_batches,fit_s=self.fit_s,predict_s=self.predict_s)
        raw=json.dumps(payload,allow_nan=False,separators=(',',':')).encode()
        temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:temp.write_bytes(raw);os.replace(temp,path)
        finally:temp.unlink(missing_ok=True)
        self.artifact_sha256=hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(cls,path,expected_features,expected_targets):
        path=Path(path)
        if not path.is_file() or path.stat().st_size>4*1024*1024:raise ValueError('Invalid MLP artifact size')
        raw=path.read_bytes();p=json.loads(raw,object_pairs_hook=_unique_object)
        keys={'format','version','feature_names','target_names','seed','x','y','coefs','intercepts','x_scaler','y_scaler',
              'fit_parameters','fit_diagnostics','trained_samples','fit_count','prediction_batches','fit_s','predict_s'}
        if not isinstance(p,dict):raise ValueError('Invalid MLP artifact schema')
        version=p.get('version')
        if type(version) is not int or version not in (1,VERSION):raise ValueError('Invalid MLP format/version')
        if version==2:keys.add('optimizer_metadata')
        if set(p)!=keys:raise ValueError('Invalid MLP artifact schema')
        if p['format']!=FORMAT:raise ValueError('Invalid MLP format/version')
        if p['feature_names']!=_names(expected_features) or p['target_names']!=_names(expected_targets):raise ValueError('MLP feature/target schema mismatch')
        result=cls(p['feature_names'],p['target_names'],p['seed'])
        if not isinstance(p['x'],list) or not isinstance(p['y'],list) or len(p['x'])!=len(p['y']) or len(p['x'])>MAX_ROWS:raise ValueError('Invalid MLP measured rows')
        for x,y in zip(p['x'],p['y']):result.add(x,y)
        if not isinstance(p['coefs'],list) or not isinstance(p['intercepts'],list) or len(p['coefs']) not in (0,2) or len(p['coefs'])!=len(p['intercepts']):raise ValueError('Invalid MLP layers')
        result.trained_samples=_integer(p['trained_samples'],'trained samples',0,result.sample_count)
        result.fit_count=_integer(p['fit_count'],'fit count',0,MAX_ROWS)
        if p['coefs']:
            if result.trained_samples<MIN_ROWS or result.fit_count==0:raise ValueError('Invalid MLP fitted counts')
            expected=parameters(result.seed);actual=p['fit_parameters']
            if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('Invalid MLP fit schema')
            if any(type(actual[k]) is not type(v) or actual[k]!=v for k,v in expected.items()) or any(type(v) is not int for v in actual['hidden_layer_sizes']):raise ValueError('Invalid MLP fit parameters')
            width,targets=len(result.feature_names),len(result.target_names)
            result.coefs=[_matrix(p['coefs'][0],width,HIDDEN,'input weights'),_matrix(p['coefs'][1],HIDDEN,targets,'output weights')]
            result.intercepts=[_row(p['intercepts'][0],HIDDEN,'hidden bias'),_row(p['intercepts'][1],targets,'output bias')]
            result.x_scaler=_scaler(p['x_scaler'],width);result.y_scaler=_scaler(p['y_scaler'],targets)
            d=p['fit_diagnostics']
            if not isinstance(d,dict) or set(d)!={'n_iter','loss','converged'} or type(d['converged']) is not bool:raise ValueError('Invalid MLP fit diagnostics')
            _integer(d['n_iter'],'MLP iterations',1,300)
            if _number(d['loss'],'MLP loss')<0:raise ValueError('Invalid MLP loss')
            result.fit_parameters=dict(actual);result.fit_diagnostics=dict(d)
            result.optimizer_metadata=legacy_optimizer_metadata() if version==1 else _optimizer_metadata(p['optimizer_metadata'],targets)
        elif result.fit_count or result.trained_samples or any(p[k] is not None for k in ('x_scaler','y_scaler','fit_parameters','fit_diagnostics')):
            raise ValueError('Untrained MLP claims fitted metadata')
        if not result.trained and version==2 and p['optimizer_metadata'] is not None:raise ValueError('Untrained MLP claims optimizer verification')
        result.prediction_batches=_integer(p['prediction_batches'],'prediction batches',0,2**63-1)
        for key in ('fit_s','predict_s'):
            value=_number(p[key],key)
            if value<0:raise ValueError('Negative MLP timing')
            setattr(result,key,value)
        result.artifact_sha256=hashlib.sha256(raw).hexdigest()
        return result

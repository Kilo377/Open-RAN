"""Fail closed before hot activation if a reproduced BEFORE prefix differs.

No simulator state is invented/restored from KPI telemetry. The normal seed,
configuration and native control prefix are replayed, then checked before the
first candidate can be selected. Only compact CPU counters are accumulated.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from smo.evaluation_reference import (ORIGIN_SCHEMA,_digest,_exogenous,_model_pins,
    comparable,dynamics_fingerprint,origin_configuration,origin_frame)
from smo.evaluation_reference import prefix_configuration_compatibility
from smo.runtime import FrameBuilder,trace_users
from smo.window_metrics import WindowIndex


def validate_proof(proof):
    if not isinstance(proof,dict)or proof.get('schema')!=ORIGIN_SCHEMA:
        raise ValueError('Unknown prepared evaluation origin')
    original=deepcopy(proof);pin=original.pop('origin_id',None)
    if pin!=_digest(original):raise ValueError('Prepared evaluation origin digest mismatch')
    n=proof.get('observe_slots');baseline=proof.get('baseline_slot')
    if type(n)is not int or not 1<=n<=20000 or baseline!=((n+1+39)//40)*40:
        raise ValueError('Prepared origin and observation window differ')
    if proof.get('after_kpis_used')is not False or proof.get('intent_bounds_rebased')is not False:
        raise ValueError('Prepared origin must contain only unrevised BEFORE evidence')
    if proof.get('before_frame',{}).get('slot')!=baseline:
        raise ValueError('Prepared origin baseline frame differs')
    return proof


class OriginGuard:
    def __init__(self,proof,config,output):
        self.proof=deepcopy(validate_proof(proof));self.output=Path(output);self.verified=False
        self.configuration_compatibility=prefix_configuration_compatibility(config,proof['configuration'],baseline_slot=proof['baseline_slot'])
        if not self.configuration_compatibility['configuration_equal']:
            raise ValueError('Reproduced network configuration differs from the prepared origin')
        if config.get('manifest',{}).get('rng_seed')!=proof['seed']:
            raise ValueError('Reproduced seed differs from the prepared origin')
        if dynamics_fingerprint(Path(__file__).resolve().parents[1])!=proof.get('dynamics_sha256'):
            raise ValueError('Reproduced network dynamics code differs from the prepared origin')
        if _model_pins(self.output)!=proof['model_sha256']:
            raise ValueError('Reproduced frozen model pins differ from the prepared origin')
        self.config=config;self.index=WindowIndex(config,population=True)
        self.builder=FrameBuilder(config,population=True);self.prefix=hashlib.sha256()

    def observe(self,row):
        slot=row['slot_id'];baseline=self.proof['baseline_slot']
        if self.verified:return
        if slot>baseline:raise ValueError('Passed the evaluation origin without verification')
        if row.get('xapp_bindings'):raise ValueError('Candidate activated before evaluation origin verification')
        users=trace_users(row);services=row.get('ue_services',{});ap=row.get('ap_services',{}).get('cells',{})
        self.index.observe(slot,users,services,ap,row.get('ue_service_events',[]))
        frame=self.builder.frame(slot,users,services,ap,{})
        self.prefix.update(json.dumps(_exogenous(row),sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        if slot!=baseline:return
        keys=[k[4:]for k in self.proof['before_frame']['states']if k.startswith('kpi:')]
        window=self.index.window(keys,baseline-self.proof['observe_slots'],baseline,through=baseline)
        window['slot_duration_ms']=self.config['manifest'].get('slot_duration_ms',.5)
        actual=origin_frame(frame,window)
        if self.prefix.hexdigest()!=self.proof['prefix_exogenous_sha256']:
            raise ValueError('Reproduced exogenous traffic prefix differs from the prepared origin')
        if not comparable(actual,self.proof['before_frame']):
            raise ValueError('Reproduced BEFORE state differs from the prepared origin')
        self.verified=True
        (self.output/'evaluation_origin_verified.json').write_text(json.dumps(dict(
            schema='oran.evaluation-origin-verification.v1',origin_id=self.proof['origin_id'],
            baseline_slot=baseline,verified_before_activation=True,observed_prefix_equal=True,
            configuration_equal=True,models_equal=True,network_dynamics_code_equal=True,after_kpis_used=False,
            configuration_metadata_compatibility=self.configuration_compatibility,
            method='native_prefix_reconstruction_with_numeric_observation_check',
            numeric_relative_tolerance=1e-7,numeric_absolute_tolerance=1e-8),indent=2)+'\n')

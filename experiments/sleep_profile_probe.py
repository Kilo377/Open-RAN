"""One fixed per-cell sleep stimulus through normal RIC/E2 execution.

This is an identification instrument, not a trained or optimizing xApp. It
never inspects KPI values, and never rewrites any other radio parameter.
"""
from copy import deepcopy
import json
from pathlib import Path

SCHEMA='oran.sleep-profile-probe.v1'
CELLS=('cell_0','cell_1','cell_2')
MODES=('active','micro_sleep','light_sleep','deep_sleep')


def validate(spec):
    spec=deepcopy(spec)
    if set(spec)!={'schema','profile_id','switch_slot','profile','calibration_plan_sha256'} or spec['schema']!=SCHEMA:
        raise ValueError('Unknown sleep-profile stimulus schema/fields')
    if type(spec['switch_slot']) is not int or spec['switch_slot']<40 or spec['switch_slot']%40:
        raise ValueError('Sleep-profile switch must align with the 40-slot RIC')
    if not isinstance(spec['profile_id'],str) or not spec['profile_id']:
        raise ValueError('Named fixed sleep profile required')
    if set(spec['profile'])!=set(CELLS) or any(v not in MODES for v in spec['profile'].values()):
        raise ValueError('Exactly three valid per-cell sleep modes required')
    pin=spec['calibration_plan_sha256']
    if not isinstance(pin,str) or len(pin)!=64 or any(c not in '0123456789abcdef'for c in pin):
        raise ValueError('Preregistered calibration plan pin required')
    return spec


def configure(configs,manifest,args):
    path=getattr(args,'sleep_profile_spec',None)
    if path is None:return None
    if (getattr(args,'control_mode','baseline')!='baseline' or getattr(args,'phase','integration')!='integration'
        or any(getattr(args,k,None)for k in('live_xapps','live_xapp_test','control_timeline','policy_timeline','perturbation_spec'))
        or configs['ric'].get('xapps') or configs['ric'].get('rapps')):
        raise ValueError('Sleep calibration requires an independent native baseline without other controllers')
    spec=validate(json.loads(Path(path).read_text()))
    if {c['cell_id']for c in manifest['cells']}!=set(CELLS) or manifest['slot_duration_ms']!=.5:
        raise ValueError('Sleep profile requires the three-cell 0.5ms network')
    if manifest['total_slots']<=spec['switch_slot']+400:
        raise ValueError('Leave a transient and observation window after the switch')
    rt=configs['phy']['rt_sys']
    if not rt.get('ap_services',{}).get('enabled') or not rt.get('ue_services',{}).get('enabled'):
        raise ValueError('Explicit AP and UE services required for sleep calibration')
    configs['ric'].update(xapp_interval_slots=40,rapps=[],xapps=[],run_rapps_on_start=False,
        freeze_learning=True,direct_conflict_mitigation='priority')
    rt.setdefault('paper_interfaces',{}).update(enabled=True,rule_observations=True)
    return spec


def build_controller(spec,output):
    from models.action import ConfigAction,ControlDecision
    from ric.near_rt.xapp_base import XAppBase
    from ric.xapp_contract import XAppContract

    class FixedSleepProfile(XAppBase):
        def __init__(self):
            super().__init__('sleep_profile_probe',100,['cell_sleep_mode'],
                contract=XAppContract((),frozenset({'cell_sleep_mode'})))
            self.spec=validate(spec);self.sent=False
            self.path=Path(output)/'sleep_profile_requests.jsonl'

        def run(self,observation,policy_snapshot):
            slot=observation.slot_id
            if self.sent or slot!=self.spec['switch_slot']:return ControlDecision()
            if {c.cell_id for c in observation.cell_observations}!=set(CELLS):
                raise ValueError('Missing observed cell at the fixed intervention tick')
            self.sent=True
            actions=[ConfigAction(source_name=self.name,target_cell_id=cid,cell_sleep_mode=self.spec['profile'][cid],
                effective_from_slot=slot,effective_until_slot=slot+39)for cid in CELLS]
            with self.path.open('a')as stream:
                stream.write(json.dumps(dict(slot=slot,profile_id=self.spec['profile_id'],profile=self.spec['profile'],
                    controlled_fields=['cell_sleep_mode'],calibration_plan_sha256=self.spec['calibration_plan_sha256'],
                    source='fixed_parameter_identification_stimulus_not_trained_xapp'),allow_nan=False)+'\n')
            return ControlDecision(config_actions=actions)

    return FixedSleepProfile()

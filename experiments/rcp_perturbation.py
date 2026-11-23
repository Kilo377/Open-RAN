"""Preregistered fixed RCP stimuli through the normal admission/executor path.

This is an experiment driver, not a learned or KPI-optimizing xApp. All arms
hold the same complete baseline vector except for their named intervention.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

SCHEMA = 'oran.rcp-perturbation-arm.v1'
POPULATION_SCHEMA = 'oran.rcp-population-arm.v1'
USERS = ('ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8')
FIELDS = frozenset(('ue_offload_fraction', 'ue_drx_configs', 'scheduler_mode',
                    'ue_scheduler_weights', 'cell_sleep_mode'))
DRX = dict(enabled=True, cycle_ms=40, on_duration_ms=4, inactivity_ms=8,
           wakeup_ms=.5, offset_ms=0)


def population_memberships(configs, manifest):
    services = configs['phy']['rt_sys']['ue_services']
    ids = sorted(u['ue_id'] for u in manifest['ues'])
    native_drx = {uid:{**services['drx'],**services.get('drx_by_ue',{}).get(uid,{})} for uid in ids}
    groups = dict(all=ids, mec=sorted(services['mec']['tasks_by_ue']),
        drx=sorted(uid for uid,p in native_drx.items() if p['enabled']),
        urllc=sorted(u['ue_id'] for u in configs['ue']['ues'] if u.get('slice_id')=='urllc'))
    offload = {uid:services['mec']['tasks_by_ue'][uid]['offload_fraction'] for uid in groups['mec']}
    return groups,native_drx,offload


def validate_population(spec):
    fields={'schema','arm','switch_ms','scheduler_mode','cell_sleep_mode','groups','native_drx','native_offload',
            'offload_probability','drx_inactivity_ms','mec_weight','urllc_weight'}
    optional={'drx_cycle_ms'}
    if not fields<=set(spec) or set(spec)-fields-optional or not isinstance(spec['arm'],str) or not spec['arm']:
        raise ValueError('Invalid population intervention fields/name')
    if spec['scheduler_mode']not in('round_robin','latency_aware','qos_deadline')or spec['cell_sleep_mode']not in('active','micro_sleep','light_sleep','deep_sleep'):
        raise ValueError('Invalid population scheduler/AP policy')
    groups=spec['groups']
    if not isinstance(groups,dict)or set(groups)!={'all','mec','drx','urllc'}:
        raise ValueError('Complete service-class membership is required')
    for name,ids in groups.items():
        if not isinstance(ids,list)or len(ids)<2 or len(ids)!=len(set(ids))or any(not isinstance(uid,str)for uid in ids):
            raise ValueError('Population interventions require complete multi-member groups')
        if not set(ids)<=set(groups['all']):raise ValueError('Unknown population member')
    if set(groups['mec'])&set(groups['urllc']):raise ValueError('This pilot requires disjoint MEC and URLLC groups')
    if set(spec['native_drx'])!=set(groups['all'])or set(spec['native_offload'])!=set(groups['mec']):
        raise ValueError('Native parameter vectors do not cover their populations')
    for uid,p in spec['native_drx'].items():
        if not isinstance(p,dict)or set(p)!=set(DRX)or type(p['enabled'])is not bool:
            raise ValueError('A complete native DRX profile is required')
        if (uid in groups['drx'])!=p['enabled']:raise ValueError('DRX membership disagrees with native policy')
        if any(type(p[k])not in(int,float)or not math.isfinite(p[k])for k in set(DRX)-{'enabled'}):
            raise ValueError('Invalid native DRX timers')
    for p in spec['native_offload'].values():
        if type(p)not in(int,float)or not math.isfinite(p)or not 0<=p<=1:raise ValueError('Invalid native offload probability')
    for key,low,high in (('offload_probability',0,1),('drx_inactivity_ms',0,40),
                         ('mec_weight',.5,2),('urllc_weight',.5,4 if spec['scheduler_mode']=='qos_deadline' else 2),('switch_ms',20,500)):
        v=spec[key]
        if type(v)not in(int,float)or not math.isfinite(v)or not low<=v<=high:raise ValueError('Invalid population '+key)
    if not math.isclose(spec['switch_ms']/20,round(spec['switch_ms']/20))or not math.isclose(spec['drx_inactivity_ms']/.5,round(spec['drx_inactivity_ms']/.5)):
        raise ValueError('Population stimulus must align to RIC/slot timing')
    if 'drx_cycle_ms' in spec:
        cycle=spec['drx_cycle_ms']
        if type(cycle)not in(int,float)or not math.isfinite(cycle)or not 10<=cycle<=320 or not math.isclose(cycle/.5,round(cycle/.5)):
            raise ValueError('Invalid population DRX cycle')
        if any(cycle<p['on_duration_ms']+p['wakeup_ms'] for u,p in spec['native_drx'].items()if u in groups['drx']):
            raise ValueError('DRX cycle must contain the on-duration and wakeup')
    return deepcopy(spec)


def validate(spec):
    if isinstance(spec,dict)and spec.get('schema')==POPULATION_SCHEMA:
        return validate_population(spec)
    if not isinstance(spec, dict) or spec.get('schema') != SCHEMA:
        raise ValueError('Expected '+SCHEMA)
    allowed = {'schema','arm','switch_ms','scheduler_mode','offload_ue4',
               'drx_ue4_inactivity_ms','weight_ue4','weight_ue6','cell_sleep_mode'}
    if set(spec)-allowed:
        raise ValueError('Unknown perturbation fields')
    if not isinstance(spec.get('arm'), str) or not spec['arm']:
        raise ValueError('A named arm is required')
    if spec.get('scheduler_mode') not in ('round_robin','latency_aware','qos_deadline'):
        raise ValueError('Unsupported fixed scheduler context')
    if spec.get('cell_sleep_mode') not in ('active','micro_sleep'):
        raise ValueError('Only the preregistered native/micro-sleep contrast is supported')
    for key, low, high in (('offload_ue4',0,1),('drx_ue4_inactivity_ms',0,40),
                          ('weight_ue4',.1,4),('weight_ue6',.1,4),('switch_ms',20,500)):
        value = spec.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid perturbation '+key)
    if not math.isclose(spec['switch_ms']/20, round(spec['switch_ms']/20)):
        raise ValueError('Switch must align to the 20ms RIC tick')
    if not math.isclose(spec['drx_ue4_inactivity_ms']/.5, round(spec['drx_ue4_inactivity_ms']/.5)):
        raise ValueError('DRX must align to 0.5ms slots')
    return deepcopy(spec)


def vector(spec, *, perturbed):
    if spec.get('schema')==POPULATION_SCHEMA:
        drx=deepcopy(spec['native_drx']);offload=deepcopy(spec['native_offload'])
        weights=dict.fromkeys(spec['groups']['all'],1.)
        if perturbed:
            offload=dict.fromkeys(spec['groups']['mec'],spec['offload_probability'])
            for uid in spec['groups']['drx']:
                drx[uid]['inactivity_ms']=spec['drx_inactivity_ms']
                if 'drx_cycle_ms' in spec:drx[uid]['cycle_ms']=spec['drx_cycle_ms']
            for group,field in (('mec','mec_weight'),('urllc','urllc_weight')):
                for uid in spec['groups'][group]:weights[uid]=spec[field]
        return dict(ue_drx_configs=drx,ue_scheduler_weights=weights,ue_offload_fraction=offload,
            scheduler_mode=spec['scheduler_mode']if perturbed else'round_robin',
            cell_sleep_mode=spec['cell_sleep_mode']if perturbed else'active')
    drx = {uid:{**DRX,'enabled':uid not in ('ue_2','ue_6')} for uid in USERS}
    weights = dict.fromkeys(USERS,1.)
    # Offload writes are only legal for UEs with configured MEC tasks.
    offload = dict.fromkeys(('ue_0','ue_4'),.5)
    if perturbed:
        drx['ue_4']['inactivity_ms'] = spec['drx_ue4_inactivity_ms']
        weights.update(ue_4=spec['weight_ue4'], ue_6=spec['weight_ue6'])
        offload['ue_4'] = spec['offload_ue4']
    return dict(ue_drx_configs=drx, ue_scheduler_weights=weights, ue_offload_fraction=offload,
        scheduler_mode=spec['scheduler_mode'] if perturbed else 'round_robin',
        cell_sleep_mode=spec['cell_sleep_mode'] if perturbed else 'active')


def configure(configs, manifest, args):
    path = getattr(args,'perturbation_spec',None)
    if path is None:
        return None
    if (getattr(args,'control_mode','baseline') != 'baseline' or
        getattr(args,'phase','integration') != 'integration' or
        getattr(args,'live_xapps',False) or getattr(args,'live_xapp_test',False) or
        getattr(args,'control_timeline',None) or getattr(args,'policy_timeline',None)):
        raise ValueError('RCP perturbations require an independent integration baseline')
    if configs['ric'].get('xapps') or configs['ric'].get('rapps'):
        raise ValueError('Disable all other controllers during RCP identification')
    if {u['ue_id'] for u in manifest['ues']} != set(USERS) or manifest['slot_duration_ms'] != .5:
        raise ValueError('Perturbation pilot requires the released six-UE 0.5ms scenario')
    services = configs['phy']['rt_sys'].get('ue_services',{})
    if not services.get('enabled') or not services.get('mec',{}).get('enabled') or not configs['phy']['rt_sys'].get('ap_services',{}).get('enabled'):
        raise ValueError('Enable the native UE/MEC and explicit AP services for identification')
    spec = validate(json.loads(Path(path).read_text()))
    if spec['schema']==POPULATION_SCHEMA:
        groups,drx,offload=population_memberships(configs,manifest)
        if spec['groups']!=groups or spec['native_drx']!=drx or spec['native_offload']!=offload:
            raise ValueError('Population stimulus membership/native policy differs from the actual scenario')
    if manifest['total_slots']*.5 <= spec['switch_ms']+200:
        raise ValueError('Leave settling and measurement time after the perturbation')
    configs['ric'].update(xapp_interval_slots=40,rapps=[],xapps=[],run_rapps_on_start=False,freeze_learning=True,
                         direct_conflict_mitigation='priority')
    configs['phy']['rt_sys'].setdefault('paper_interfaces',{}).update(enabled=True,rule_observations=True)
    return spec


def build_controller(spec, output):
    from models.action import ConfigAction,ControlDecision
    from ric.near_rt.xapp_base import XAppBase
    from ric.xapp_contract import XAppContract

    class FixedRCPStimulus(XAppBase):
        def __init__(self):
            super().__init__('rcp_perturbation',100,list(FIELDS),
                contract=XAppContract((),FIELDS))
            self.spec = validate(spec)
            self.events = Path(output)/'perturbation_requests.jsonl'
            self.last_decisions = []

        def run(self, observation, policy_snapshot):
            values = vector(self.spec,perturbed=observation.slot_id*.5 >= self.spec['switch_ms'])
            actions = []
            for cell in observation.cell_observations:
                ids = {u.ue_id for u in observation.ue_observations if u.serving_cell_id == cell.cell_id}
                fields = {k:({uid:v for uid,v in val.items() if uid in ids} if isinstance(val,dict) else val)
                          for k,val in values.items()}
                actions.append(ConfigAction(source_name=self.name,target_cell_id=cell.cell_id,**deepcopy(fields)))
            row = dict(slot_id=observation.slot_id,arm=self.spec['arm'],
                phase='intervention' if observation.slot_id*.5 >= self.spec['switch_ms'] else 'common_warmup',
                requested_vector=values, source='fixed_experimental_stimulus_not_xapp_algorithm')
            with self.events.open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            self.last_decisions = [row]
            return ControlDecision(config_actions=actions)

    return FixedRCPStimulus()

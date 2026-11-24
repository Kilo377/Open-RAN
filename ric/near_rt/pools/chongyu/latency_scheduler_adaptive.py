"""X-12 task-latency policy with versioned, public-observation-only inference.

Research v2 adds relative priorities for competing UEs and guarded, trained
action selection. The MAC/PHY, task model, traffic and evaluation are unchanged.
"""
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

from models.action import ConfigAction, ControlDecision
from ric.near_rt.pools.base import PoolXApp

TARGETS=('ue_0','ue_4')
ACTIONS=[dict(name='native',scheduler_mode='round_robin',target_weight=1.,other_weight=1.)]
for mode,pairs in [('latency_aware',[(2.,.5),(2.,1.),(1.,.5),(1.,1.),(.5,1.),(.5,2.),(1.,2.),(2.,2.)]),
                   ('qos_deadline',[(2.,.5),(2.,1.),(1.,.5),(4.,.5)])]:
    for target,other in pairs:
        ACTIONS.append(dict(name=f'{mode}_m{target:g}_o{other:g}',scheduler_mode=mode,target_weight=target,other_weight=other))

CONTEXT_FEATURES=['has_ue0','has_ue4','target_queue_kb','target_hol_max_ms',
    'target_goodput_mbps','target_offered_mbps','reference_sinr_db','data_sinr_db',
    'observed_bler','observed_retx_fraction','retx_sample_count','offloaded_pending',
    'cell_queue_kb','other_queue_kb','other_hol_max_ms','other_goodput_mbps',
    'other_offered_mbps','other_reference_sinr_db','other_observed_bler',
    'urllc_queue_kb','urllc_hol_max_ms','urllc_goodput_mbps','mean_active_fraction',
    'offload_fraction','drx_cycle_ms','drx_inactivity_ms','current_latency_mode',
    'current_qos_mode','current_target_weight','current_other_weight']
ACTION_FEATURES=['candidate_latency_mode','candidate_qos_mode','candidate_target_weight','candidate_other_weight']
FEATURES=CONTEXT_FEATURES+ACTION_FEATURES
JOINT_FEATURES=[f'{uid}:{name}' for uid in TARGETS for name in CONTEXT_FEATURES]+[f'{uid}:{name}' for uid in TARGETS for name in ACTION_FEATURES]
SCHEMA='oran.x12-adaptive-policy.v1'

def number(obj,name,default=0.):
    value=getattr(obj,name,None)
    return float(value) if type(value) in (int,float) and math.isfinite(value) else float(default)

def context(users,cell,all_users):
    others=[u for u in all_users if u.ue_id not in TARGETS]
    urllc=[u for u in others if u.slice_id=='urllc']
    def mean(group,key,default=0.):
        return sum(number(u,key,default) for u in group)/len(group) if group else float(default)
    def total(group,key):return sum(number(u,key) for u in group)
    def largest(group,key):return max((number(u,key) for u in group),default=0.)
    cfg=cell.executed_config;weights=cfg.get('ue_scheduler_weights',{})
    n=len(users)
    policies=[u.ue_service_policy for u in users]
    return [float(any(u.ue_id==uid for u in users)) for uid in TARGETS]+[
        total(users,'queue_bytes')/1000.,largest(users,'hol_delay_ms'),total(users,'throughput_mbps'),
        total(users,'offered_rate_bps')/1e6,mean(users,'reference_sinr_db'),mean(users,'data_sinr_db'),
        mean(users,'bler'),mean(users,'retx_fraction'),total(users,'retx_sample_count'),
        sum(u.ue_task_metrics.get('offloaded_pending',0) for u in users),
        total(all_users,'queue_bytes')/1000.,total(others,'queue_bytes')/1000.,largest(others,'hol_delay_ms'),
        total(others,'throughput_mbps'),total(others,'offered_rate_bps')/1e6,mean(others,'reference_sinr_db'),mean(others,'bler'),
        total(urllc,'queue_bytes')/1000.,largest(urllc,'hol_delay_ms'),total(urllc,'throughput_mbps'),
        mean(users,'ue_active_fraction'),sum(p['offload_fraction'] for p in policies)/n,
        sum(p['drx']['cycle_ms'] for p in policies)/n,sum(p['drx']['inactivity_ms'] for p in policies)/n,
        float(cfg.get('scheduler_mode')=='latency_aware'),float(cfg.get('scheduler_mode')=='qos_deadline'),
        sum(weights.get(u.ue_id,1.) for u in users)/n,
        sum(weights.get(u.ue_id,1.) for u in others)/len(others) if others else 1.]

def feature_row(ctx,index):
    action=ACTIONS[index]
    return list(ctx)+[float(action['scheduler_mode']=='latency_aware'),float(action['scheduler_mode']=='qos_deadline'),action['target_weight'],action['other_weight']]

def joint_feature_row(contexts,vector):
    return list(contexts[0])+list(contexts[1])+feature_row([],vector[0])+feature_row([],vector[1])

def predict(forest,row):
    # Match sklearn tree input conversion. No pickle or sklearn at inference.
    import struct
    row=[struct.unpack('f',struct.pack('f',v))[0] for v in row]
    values=[]
    for tree in forest:
        node=0
        while tree['children_left'][node]!=-1:
            node=tree['children_left'][node] if row[tree['feature'][node]]<=tree['threshold'][node] else tree['children_right'][node]
        values.append(tree['value'][node])
    mean=sum(values)/len(values)
    return mean,math.sqrt(sum((v-mean)**2 for v in values)/len(values))

def load_policy(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>16*1024**2:raise ValueError('Invalid adaptive policy file')
    def invalid_constant(v):raise ValueError('Nonfinite JSON policy value')
    value=json.loads(path.read_text(),parse_constant=invalid_constant)
    if not isinstance(value,dict):raise ValueError('Adaptive policy must be a JSON object')
    if value.get('schema')!=SCHEMA or value.get('actions')!=ACTIONS:raise ValueError('Adaptive policy action schema mismatch')
    if value.get('kind') not in ('fixed','rf','joint_rf','ppo'):raise ValueError('Unsupported adaptive policy')
    if type(value.get('training',False))is not bool:raise ValueError('Training mode must be boolean')
    if type(value.get('refresh_on_membership_change',True))is not bool:raise ValueError('Membership refresh must be boolean')
    finite=lambda x:type(x)in(int,float) and math.isfinite(x)
    if value['kind']=='fixed':
        if not isinstance(value.get('cell_actions'),dict) or any(type(i)is not int or not 0<=i<len(ACTIONS) for i in value['cell_actions'].values()):raise ValueError('Invalid fixed ablation')
    if value['kind'] in ('rf','joint_rf'):
        expected=JOINT_FEATURES if value['kind']=='joint_rf' else FEATURES
        if value.get('feature_names')!=expected or not value.get('forest'):raise ValueError('RF feature/forest mismatch')
        if not isinstance(value['forest'],list) or not 1<=len(value['forest'])<=256:raise ValueError('Invalid forest size')
        for tree in value['forest']:
            if not isinstance(tree,dict) or set(tree)!={'children_left','children_right','feature','threshold','value'}:raise ValueError('Invalid tree schema')
            if any(not isinstance(v,list) for v in tree.values()):raise ValueError('Invalid tree arrays')
            n=len(tree['feature'])
            if not 1<=n<=4095 or any(len(tree[k])!=n for k in ('children_left','children_right','threshold','value')):raise ValueError('Invalid tree shape')
            if any(not finite(v) for v in tree['threshold']+tree['value']):raise ValueError('Nonfinite tree value')
            parents=[0]*n
            for i,(l,r) in enumerate(zip(tree['children_left'],tree['children_right'])):
                if l==r==-1:
                    if tree['feature'][i]!=-2:raise ValueError('Invalid leaf feature')
                    continue
                if type(l)is not int or type(r)is not int or not i<l<n or not i<r<n or l==r or not 0<=tree['feature'][i]<len(expected):raise ValueError('Invalid tree topology')
                parents[l]+=1;parents[r]+=1
            if parents[0]!=0 or any(p!=1 for p in parents[1:]):raise ValueError('Invalid tree reachability')
            stack=[(0,0)]
            while stack:
                node,depth=stack.pop()
                if depth>12:raise ValueError('Tree exceeds supported depth')
                if tree['children_left'][node]!=-1:stack.extend([(tree['children_left'][node],depth+1),(tree['children_right'][node],depth+1)])
        if value['kind']=='joint_rf':
            vectors=value.get('action_vectors')
            if not isinstance(vectors,list) or not 1<=len(vectors)<=64:raise ValueError('Invalid joint action catalogue')
            if any(not isinstance(v,list) or len(v)!=2 or any(type(i)is not int or not 0<=i<len(ACTIONS) for i in v) for v in vectors):raise ValueError('Invalid joint action vector')
            if [0,0] not in vectors or len({tuple(v) for v in vectors})!=len(vectors):raise ValueError('Invalid joint action catalogue')
    if value['kind']=='ppo':
        allowed=value.get('allowed_actions',list(range(len(ACTIONS))))
        if not isinstance(allowed,list) or not allowed or len(set(allowed))!=len(allowed) or any(type(i)is not int or not 0<=i<len(ACTIONS) for i in allowed):raise ValueError('Invalid PPO action mask')
        for key in ('input_mean','input_scale'):
            row=value.get(key)
            if row is not None and (not isinstance(row,list) or len(row)!=len(CONTEXT_FEATURES) or any(not finite(v) or (key=='input_scale' and v<=0) for v in row)):raise ValueError('Invalid PPO normalization')
        network=value.get('network')
        if not isinstance(network,dict) or set(network)!={'hidden','actor','critic'} or not isinstance(network['hidden'],list) or len(network['hidden'])!=2:raise ValueError('Invalid PPO network')
        def layer(item,width,outputs=None):
            if not isinstance(item,dict) or set(item)!={'weight','bias'}:raise ValueError('Invalid PPO layer')
            weights,bias=item['weight'],item['bias']
            if not isinstance(weights,list) or not isinstance(bias,list) or not 1<=len(weights)<=128 or len(weights)!=len(bias) or (outputs is not None and len(weights)!=outputs):raise ValueError('Invalid PPO layer shape')
            if any(not isinstance(row,list) or len(row)!=width for row in weights):raise ValueError('Invalid PPO weight shape')
            if any(not finite(v) or abs(v)>1e6 for row in weights for v in row) or any(not finite(v) or abs(v)>1e6 for v in bias):raise ValueError('Invalid PPO numeric weight')
            return len(bias)
        width=len(CONTEXT_FEATURES)
        for item in network['hidden']:width=layer(item,width)
        layer(network['actor'],width,len(ACTIONS));layer(network['critic'],width,1)
    for k,default,low,high in [('decision_ms',100,20,1000),('startup_hold_ms',260,20,1000),('minimum_predicted_gain',.01,0,.5),('uncertainty_weight',0.,0.,4.)]:
        x=value.get(k,default)
        if type(x)not in (int,float) or not math.isfinite(x) or not low<=x<=high:raise ValueError('Invalid policy '+k)
    return value

class LatencySchedulerAdaptive(PoolXApp):
    implementation_id='latency_scheduler_rf'
    category_id='latency_scheduler'
    version='2.0.0'
    description='X-12: trained contextual policy reducing MEC offloaded task latency'
    required_kpis=('queue_bytes','hol_delay_ms','throughput_mbps','offered_rate_bps','reference_sinr_db',
                  'delivered_bits_total','offered_bits_total','ue_energy_total_j','ue_service_policy',
                  'ue_active_fraction','executed_config')
    writable_actions=('scheduler_mode','ue_scheduler_weights')
    min_action_ms=20

    def validate_parameters(self,p):
        super().validate_parameters(p)
        allowed={'mode','model_path','log_path','window_ms','seed','complete_scheduler_bundle','adaptive_policy_path'}
        if set(p)-allowed or p.get('mode')!='frozen' or not p.get('adaptive_policy_path'):raise ValueError('Explicit frozen adaptive policy required')
        root=Path(__file__).resolve().parents[4]
        for key in ('adaptive_policy_path','model_path','log_path'):
            if p.get(key) and not Path(p[key]).resolve().is_relative_to(root/'outputs'):raise ValueError('Adaptive artifacts must stay under outputs')

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.policy=load_policy(self.parameters['adaptive_policy_path'])
        self.mode='online' if self.policy.get('training',False) else 'frozen'
        self.requires_whole_network=self.policy['kind']=='joint_rf'
        self.model_path=Path(self.parameters['adaptive_policy_path'])
        self.policy_sha256=hashlib.sha256(Path(self.parameters['adaptive_policy_path']).read_bytes()).hexdigest()
        self.log_path=Path(self.parameters['log_path']) if self.parameters.get('log_path') else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            if self.log_path.exists():raise ValueError('Choose a fresh adaptive event log')
        self.current={};self.events=deque(maxlen=512);self.trainer=None

    def choose(self,cid,ctx):
        if self.policy['kind']=='fixed':return self.policy['cell_actions'].get(cid,0),None,'fixed_action_ablation'
        if self.policy['kind']=='rf':
            preds=[predict(self.policy['forest'],feature_row(ctx,i)) for i in range(len(ACTIONS))]
            guard=self.policy.get('uncertainty_weight',0.)
            score=[m+guard*s for m,s in preds]
            allowed=set(self.policy.get('allowed_actions',range(len(ACTIONS))))
            score=[v if i in allowed else 1e9 for i,v in enumerate(score)]
            # Counterfactual target is relative to native, whose true contrast is zero.
            score[0]=0.
            i=min(range(len(score)),key=lambda j:score[j])
            if score[i]>-self.policy.get('minimum_predicted_gain',.01):i=0
            return i,dict(mean=[p[0] for p in preds],std=[p[1] for p in preds],score=score),'rf_relative_latency_contrast'
        from .latency_scheduler_ppo import inference
        return inference(self.policy,ctx,cid)

    def run(self,observation,policy_snapshot):
        begun=time.perf_counter();self.last_decisions=[];actions=[]
        if self.requires_whole_network and self.active_cells!=set(self.allowed_cells):
            self.status='not_evaluable';self.status_reason='joint_policy_requires_all_cells'
            result=ControlDecision();result.config_actions.clear();return result
        cells={c.cell_id:c for c in observation.cell_observations if c.cell_id in self.active_cells}
        membership_changes=set()
        if self.current and self.policy.get('refresh_on_membership_change',True):
            for cid in cells:
                members=tuple(sorted(u.ue_id for u in observation.ue_observations if u.serving_cell_id==cid))
                if not any(uid in TARGETS for uid in members):continue
                old=self.current.get(cid)
                if old is None or old.get('members')!=members:membership_changes.add(cid)
        refresh_joint=self.requires_whole_network and bool(membership_changes)
        available=[]
        for cid,cell in sorted(cells.items()):
            all_users=[u for u in observation.ue_observations if u.serving_cell_id==cid]
            users=[u for u in all_users if u.ue_id in TARGETS and getattr(u,'ue_task_metrics',None) is not None]
            if not users:continue
            users.sort(key=lambda u:u.ue_id)
            if not isinstance(getattr(cell,'executed_config',None),dict):continue
            produced=cell.kpi_metadata.get('executed_config',{}).get('produced_at_slot')
            if type(produced)is not int or produced>observation.slot_id:continue
            if any(u.kpi_metadata.get('ue_energy_total_j',{}).get('produced_at_slot')!=produced for u in users):continue
            old=self.current.get(cid)
            if old and observation.slot_id<old['next_slot'] and not (refresh_joint or cid in membership_changes):continue
            ctx=context(users,cell,all_users)
            if not all(math.isfinite(v) for v in ctx) or len(ctx)!=len(CONTEXT_FEATURES):raise ValueError('Invalid public feature vector')
            available.append((cid,cell,all_users,users,ctx,produced,old))
        if self.policy['kind']=='ppo' and self.policy.get('training',False) and available:
            if self.log_path is None:raise ValueError('A PPO training run requires its own event log')
            if self.trainer is None:
                from .latency_scheduler_ppo import PPOTrainer
                self.trainer=PPOTrainer(self.policy,self.log_path,self.parameters.get('seed',1))
            for cid,cell,all_users,users,ctx,produced,old in available:
                self.trainer.finish_step(cid,ctx,users,observation.slot_id,self.dt)
            # Complete every old-policy transition before updating and sampling.
            self.trainer.maybe_update()
        joint=None
        if self.policy['kind']=='joint_rf' and available:
            groups={u.ue_id:(cid,ctx) for cid,cell,all_users,users,ctx,produced,old in available for u in users}
            if set(groups)!=set(TARGETS):
                # Never invent a missing public state or coordinate a stale half-bundle.
                self.status='not_evaluable';self.status_reason='joint_public_state_not_ready'
                result=ControlDecision();result.config_actions.clear();return result
            contexts=[groups[uid][1] for uid in TARGETS]
            vectors=self.policy['action_vectors']
            preds=[predict(self.policy['forest'],joint_feature_row(contexts,v)) for v in vectors]
            score=[m+self.policy.get('uncertainty_weight',0.)*s for m,s in preds]
            native=vectors.index([0,0]);score[native]=0.
            if groups[TARGETS[0]][0]==groups[TARGETS[1]][0]:
                score=[v if vectors[i][0]==vectors[i][1] else 1e9 for i,v in enumerate(score)]
            best=min(range(len(vectors)),key=lambda i:score[i])
            if score[best]>-self.policy.get('minimum_predicted_gain',.02):best=native
            vector=vectors[best]
            joint={groups[uid][0]:(vector[i],dict(action_vector=vector,mean=[p[0] for p in preds],std=[p[1] for p in preds],score=score)) for i,uid in enumerate(TARGETS)}
            self.status='ready';self.status_reason=None
        for cid,cell,all_users,users,ctx,produced,old in available:
            if joint:
                index,pred=joint[cid];reason='joint_rf_network_latency_contrast'
            elif self.trainer:
                index,pred,reason=self.trainer.choose(cid,ctx,users,observation.slot_id)
            else:index,pred,reason=self.choose(cid,ctx)
            chosen=ACTIONS[index]
            weights={u.ue_id:chosen['target_weight'] if u.ue_id in TARGETS else chosen['other_weight'] for u in all_users}
            action=ConfigAction(source_name=self.name,target_cell_id=cid,scheduler_mode=chosen['scheduler_mode'],ue_scheduler_weights=weights)
            action._library_provenance=self.identity(cid);actions.append(action)
            duration=self.policy.get('startup_hold_ms',260) if old is None else self.policy.get('decision_ms',100)
            self.current[cid]=dict(action=index,next_slot=observation.slot_id+round(duration/self.dt),
                members=tuple(sorted(u.ue_id for u in all_users)))
            row=dict(kind='decision',slot_id=observation.slot_id,produced_slot=produced,action=index,
                scheduler_mode=chosen['scheduler_mode'],ue_scheduler_weights=weights,ue_ids=[u.ue_id for u in users],
                context=ctx,context_names=CONTEXT_FEATURES,prediction=pred,reason=reason,policy_sha256=self.policy_sha256,
                policy_kind=self.policy['kind'],membership_refresh=refresh_joint or cid in membership_changes,
                **self.identity(cid))
            self.last_decisions.append(row);self.events.append(row)
            if self.log_path:
                with self.log_path.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        self.decision_seconds+=time.perf_counter()-begun
        result=ControlDecision(config_actions=actions)
        if not actions:result.config_actions.clear()
        return result

    def reset(self):
        super().reset();self.current.clear()
        if self.trainer:
            self.trainer.pending.clear();self.trainer.rows.clear()

    def summary(self):
        return dict(mode=self.mode,algorithm=self.policy['kind'],policy_sha256=self.policy_sha256,
            objective='mec_offloaded_latency_ms',action_count=len(ACTIONS),decision_s=self.decision_seconds,
            training_updates=self.trainer.updates if self.trainer else 0,
            training_steps=self.trainer.steps if self.trainer else 0,
            artifact_training_steps=self.policy.get('training_steps'),
            artifact_training_rows=self.policy.get('training_metadata',{}).get('training_rows'),
            model_path=self.parameters['adaptive_policy_path'])

    def descriptor(self):return dict(super().descriptor(),learning=self.summary(),target_ue_ids=list(TARGETS))

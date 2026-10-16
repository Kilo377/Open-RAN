"""Measured-feedback alternatives. No extra PHY executions or hidden state."""
from copy import deepcopy
from collections import deque
import math
import numpy as np
from ric.near_rt.rule_based_v1.apps import RuleXApp,clip,finite,rsrp_map,waterfill_caps
from ric.near_rt.rule_based_v1.catalog import DEFAULTS
from .catalog import EXTRA_DEFAULTS


class Algorithm(RuleXApp):
    def __init__(self,category,algorithm,**kwargs):
        params=kwargs.pop('parameters',{}) or {}
        allowed=set(DEFAULTS)|({'min_action_ms'} if algorithm=='legacy_v1' else set(EXTRA_DEFAULTS)|{'min_action_ms'})
        if set(params)-allowed:raise ValueError('Unknown implementation parameters: '+str(sorted(set(params)-allowed)))
        super().__init__(category,parameters={k:v for k,v in params.items() if k in DEFAULTS or k=='min_action_ms'},**kwargs)
        self.algorithm=algorithm
        self.search_samples=deque(maxlen=5000)
        self.search_seen={};self.search_events={}
        if algorithm!='legacy_v1':self.parameters.update({**EXTRA_DEFAULTS,**params})
        self.validate(self.parameters)

    def capture_reports(self,observation):
        if self.algorithm=='a3_rsrp':
            for c in observation.cell_observations:
                if c.cell_id not in self.contract.cell_ids or not getattr(c,'executed_config',None):continue
                cfg=c.executed_config;s=self.state.setdefault(c.cell_id,{})
                signature=(cfg['time_to_trigger_ms'],cfg['steering_hysteresis_db'],tuple(sorted(cfg['cio_offsets_db'].items())),
                    tuple(sorted(cfg.get('neighbor_cell_ids') or [cell.cell_id for cell in observation.cell_observations])))
                if s.get('a3_configuration')!=signature:s.clear();s['a3_configuration']=signature
                for u in observation.ue_observations:
                    if u.serving_cell_id==c.cell_id and not all(self._valid(u,key) for key in self.contract.required_kpis if key!='resource_grid_utilization'):s.pop(u.ue_id,None)
        for c in observation.cell_observations:
            pending=self.state.get(c.cell_id,{}).get('pending')
            if pending and 'target' in pending and getattr(c,'config_produced_at_slot',-1)>pending['issued']:
                cfg=getattr(c,'executed_config',{})
                if all(cfg.get(key)==value for key,value in pending['target'].items()):
                    pending.setdefault('executed',c.config_produced_at_slot)
        if self.algorithm=='coverage_grid':
            for u in observation.ue_observations:
                if u.serving_cell_id not in self.contract.cell_ids or not self._valid(u,'reference_sinr_db'):continue
                produced=u.kpi_metadata['reference_sinr_db']['produced_at_slot']
                if produced>self.search_seen.get(u.ue_id,-1):
                    self.search_samples.append((produced,u.ue_id,u.reference_sinr_db))
                    self.search_seen[u.ue_id]=produced
        if self.algorithm=='window_search':
            for u in observation.ue_observations:
                if not self._valid(u,'mobility_events'):continue
                for e in u.mobility_events:self.search_events[(e['slot_id'],e['ue_id'],e['kind'])]=deepcopy(e)
            self.search_events={key:e for key,e in self.search_events.items() if (observation.slot_id-e['slot_id'])*self.dt<=3000}

    def run(self,observation,policy_snapshot):
        self.capture_reports(observation)
        if self.algorithm!='window_search':return super().run(observation,policy_snapshot)
        # A mobility evaluation includes recently departed UEs. Otherwise the
        # very handover being evaluated would erase its source-cell cohort.
        from models.action import ConfigAction,ControlDecision
        self.last_decisions=[];actions=[]
        try:params=self._policy(policy_snapshot)
        except (ValueError,TypeError) as error:
            self._reason(None,'nonrt_policy_rejected',detail=str(error));return ControlDecision()
        previous=self.parameters;self.parameters=params
        try:
            for c in observation.cell_observations:
                cid=c.cell_id
                if cid not in self.contract.cell_ids:continue
                members=[u for u in observation.ue_observations if all(self._valid(u,k) for k in self.contract.required_kpis) and
                    (u.serving_cell_id==cid or any(e.get('from_cell')==cid for e in u.mobility_events))]
                report=c.kpi_metadata.get('executed_config',{})
                produced=report.get('produced_at_slot',-1)
                if not report.get('valid') or produced<=self.last_report.get(cid,-1):continue
                self.last_report[cid]=produced
                if not members:self._reason(cid,'no_valid_ue_samples');continue
                if (observation.slot_id-self.last_action.get(cid,-10**12))*self.dt<self.min_ms:continue
                a=ConfigAction(target_cell_id=cid,source_name=self.name)
                if self.search(observation,c,c.executed_config,members,a):
                    actions.append(a);self.last_action[cid]=observation.slot_id
                    self._reason(cid,'action_requested',produced,values={k:deepcopy(getattr(a,k)) for k in self.controlled_fields if getattr(a,k) is not None})
            return ControlDecision(config_actions=actions or [ConfigAction(source_name=self.name)])
        finally:self.parameters=previous

    @staticmethod
    def validate(p):
        RuleXApp._validate_parameters(p)
        if any(not finite(x) for x in p.values()):raise ValueError('Nonfinite algorithm parameters')
        if p.get('pi_kp',0)<0 or p.get('pi_ki',0)<0:raise ValueError('Negative PI gain')
        if not 0<p.get('delay_quantile',.95)<=1 or p.get('hol_target',.5)<=0:raise ValueError('Invalid delay target/quantile')
        if p.get('ret_window_ms',400)<400 or p.get('mro_evaluation_ms',1000)<1000:raise ValueError('Evaluation windows cannot be shortened below acceptance defaults')
        if p.get('deficit_horizon_ms',1000)<=0:raise ValueError('Positive deficit horizon required')

    def _policy(self,policy):
        values=(getattr(policy,'rule_based_parameters',{}) or {}).get(self.name,{})
        if not isinstance(values,dict) or set(values)-set(self.parameters):raise ValueError('Invalid instance parameter update')
        result={**self.parameters,**values}
        if getattr(policy,'preferred_power_cap_dbm',None) is not None:
            result['max_power_dbm']=min(result['max_power_dbm'],float(policy.preferred_power_cap_dbm))
        self.validate(result);return result

    def pi(self,state,key,error,current,lo,hi,step,slot):
        old=state.get(key)
        elapsed=0 if old is None else max(0,(slot-old['slot'])*self.dt/1000)
        # Velocity form; freeze the integral when the executed actuator is
        # saturated in the error direction. Never integrate a requested cap.
        integral=self.parameters['pi_ki']*error*elapsed
        if (current>=hi and error>0) or (current<=lo and error<0):integral=0.
        proportional=self.parameters['pi_kp']*(error-(old['error'] if old else 0))
        state[key]=dict(error=error,slot=slot)
        return clip(current+clip(proportional+integral,-step,step),lo,hi)

    @staticmethod
    def family(a,mode):
        a.scheduler_mode=mode
        a.scheduler_policy_hint={'slice_aware':'slice_bias','latency_aware':'latency_bias','round_robin':'balanced'}[mode]

    def _decide(self,o,c,cfg,ues,cells,a,commands):
        if self.algorithm=='legacy_v1':return super()._decide(o,c,cfg,ues,cells,a,commands)
        p=self.parameters;cid=c.cell_id;s=self.state.setdefault(cid,{})
        active=[u for u in ues if u.queue_bytes>0];n=int(cfg['n_prbs'])
        budget=10**((min(cfg['tx_power_dbm'],cfg.get('power_cap_dbm') if cfg.get('power_cap_dbm') is not None else cfg['tx_power_dbm'])-30)/10)
        k=self.algorithm
        if k in ('cqi_proportional','backlog_cqi'):
            scores=[0.]*n
            for u in active:
                for group,q in zip(u.subband_prb_groups,u.subband_cqi):
                    if not finite(q):continue
                    weight=u.queue_bytes if k=='backlog_cqi' else 1.
                    for g in group:
                        if 0<=g<n:scores[g]+=weight*(q+1)
            if sum(scores)<=0:return False
            a.prb_power_w=[budget*x/sum(scores) for x in scores];a.power_group_size_prbs=1;return True
        if k=='sinr_pi':
            error=(p['sinr_target_db']-min(u.reference_sinr_db for u in ues))/5
            unsafe=any(u.hol_delay_ms>u.latency_budget_ms*.7 or
                (finite(getattr(u,'requested_dl_rate_bps',None)) and u.requested_dl_rate_bps>u.throughput_mbps*1e6 and u.queue_bytes>0) for u in ues)
            if unsafe:error=max(error,1.)
            maximum=min(p['max_power_dbm'],cfg.get('nominal_tx_power_dbm',30),cfg.get('power_cap_dbm') if cfg.get('power_cap_dbm') is not None else math.inf)
            value=self.pi(s,'power',error,cfg['tx_power_dbm'],min(p['min_power_dbm'],maximum),maximum,p['power_step_db'],o.slot_id)
            if unsafe:value=min(maximum,max(cfg['tx_power_dbm'],value))
            if abs(value-cfg['tx_power_dbm'])<1e-9:return False
            a.tx_power_dbm=value;return True
        if k=='empty_only':
            if c.cell_sleep_mode=='active' and ues:return False
            return super()._decide(o,c,cfg,ues,cells,a,commands)
        if k=='load_pi':
            if not finite(c.resource_grid_utilization):return False
            offsets=deepcopy(cfg['cio_offsets_db']);changed=False
            for target in sorted(cells):
                if target==cid or not any(target in self._candidate(u,cells,cfg) for u in ues):continue
                if o.slot_id<s.get('cio_settle:'+target,-1):
                    self._reason(cid,'association_settling');continue
                old=offsets.get(target,0.);error=c.resource_grid_utilization-cells[target].resource_grid_utilization-p['load_gap']
                new=self.pi(s,'cio:'+target,error,old,-p['cio_bound_db'],p['cio_bound_db'],p['cio_step_db'],o.slot_id)
                offsets[target]=new;changed|=abs(new-old)>1e-9
                if abs(new-old)>1e-9:
                    s['cio_settle:'+target]=o.slot_id+math.ceil(max(p['association_settle_ms'],cfg['time_to_trigger_ms']+self.dt*10)/self.dt)
            if changed:a.cio_offsets_db=offsets
            return changed
        if k in ('a3_rsrp','sla_max_sinr'):
            for u in ues:
                targets=self._candidate(u,cells,cfg);rs=rsrp_map(u)
                if not targets or not rs or cid not in rs:s.pop(u.ue_id,None);continue
                target=max(targets,key=lambda t:((rs[t]+cfg['cio_offsets_db'].get(t,0) if k=='a3_rsrp' else u.candidate_sinr_db[t]),t))
                if k=='sla_max_sinr':
                    demand=getattr(u,'requested_dl_rate_bps',None)
                    poor=u.hol_delay_ms>u.latency_budget_ms*.5 or (finite(demand) and demand>u.throughput_mbps*1e6 and u.queue_bytes>0)
                    good=poor and u.candidate_sinr_db[target]>u.reference_sinr_db+p['sinr_deadband_db']
                else:
                    good=rs[target]+cfg['cio_offsets_db'].get(target,0)>rs[cid]+cfg['steering_hysteresis_db']
                produced=u.kpi_metadata['candidate_rsrp_dbm']['produced_at_slot']
                previous=s.get(u.ue_id)
                if not good:s.pop(u.ue_id,None);continue
                # A stale gap breaks continuity; time comes from produced reports.
                if not previous or previous[0]!=target or (produced-previous[2])*self.dt>self.dt*5:
                    previous=(target,produced,produced,0)
                count=previous[3]+1;s[u.ue_id]=(target,previous[1],produced,count)
                ready=(produced-previous[1])*self.dt>=cfg['time_to_trigger_ms'] if k=='a3_rsrp' else count>=p['ts_samples']
                if ready and self._command(o,u,target,commands):s.pop(u.ue_id,None)
            return False
        if k in ('equal_share','deficit_airtime'):
            if not active:
                s.pop('credits',None);s.pop('executed',None)
                a.ue_prb_limits={u.ue_id:0 for u in ues};return True
            ids=sorted(u.ue_id for u in active);start=s.get('rotation',0)%len(ids);ids=ids[start:]+ids[:start]
            result={u.ue_id:0 for u in ues}
            if k=='equal_share':
                for j in range(n):result[ids[j%len(ids)]]+=1
                s['rotation']=start+1
            else:
                credits=s.setdefault('credits',{});last=s.setdefault('executed',{})
                for uid in list(last):
                    if uid not in ids:last.pop(uid);credits.pop(uid,None)
                for u in active:
                    total=u.executed_prb_slots_total;epoch=u.resource_counter_epoch
                    produced=u.kpi_metadata['executed_prb_slots_total']['produced_at_slot']
                    old=last.get(u.ue_id);last[u.ue_id]=(epoch,total,produced)
                    if old and old[0]==epoch:
                        elapsed=max(0,produced-old[2]);used=max(0,total-old[1])
                        credits[u.ue_id]=clip(credits.get(u.ue_id,0)+elapsed*n/len(ids)-used,-n*2000,n*2000)
                    else:credits[u.ue_id]=0.
                temporary={uid:credits[uid] for uid in ids}
                for _ in range(n):
                    uid=max(ids,key=lambda uid:(temporary[uid],-ids.index(uid)))
                    result[uid]+=1;temporary[uid]-=1
            a.ue_prb_limits=result;return True
        if k in ('goodput_pi','service_deficit'):
            weights={}
            for sid in sorted({u.slice_id for u in ues}):
                members=[u for u in ues if u.slice_id==sid]
                if any(not finite(getattr(u,'requested_dl_rate_bps',None)) for u in members):
                    self._reason(cid,'missing_service_requirement');return False
                demand=sum(u.requested_dl_rate_bps for u in members if u.queue_bytes>0)
                goodput=sum(u.throughput_mbps*1e6 for u in members)
                if k=='goodput_pi':
                    weights[sid]=self.pi(s,'slice:'+sid,(demand-goodput)/max(demand,1),cfg['slice_weight_adjustment'].get(sid,1.),p['slice_min_weight'],p['slice_max_weight'],.25,o.slot_id)
                else:
                    bits=sum(u.delivered_bits_total for u in members);old=s.get('slice:'+sid)
                    produced=min(u.kpi_metadata['delivered_bits_total']['produced_at_slot'] for u in members)
                    cohort=tuple(sorted(u.ue_id for u in members))
                    if old and old.get('cohort')==cohort:
                        dt=max(0,(produced-old['slot'])*self.dt/1000)
                        deficit=clip(old['deficit']+demand*dt-max(0,bits-old['bits']),0,max(1,demand*p['deficit_horizon_ms']/1000))
                    else:deficit=0.
                    if demand==0:deficit=0.
                    s['slice:'+sid]=dict(slot=produced,bits=bits,deficit=deficit,cohort=cohort)
                    weights[sid]=clip(1+deficit/max(1,demand*.1),p['slice_min_weight'],p['slice_max_weight'])
            self.family(a,'slice_aware');a.slice_weight_adjustment=weights;return True
        if k=='delay_pi':
            value=float(np.quantile([u.hol_delay_ms/max(u.latency_budget_ms,1e-6) for u in ues],p['delay_quantile']))
            self.family(a,'latency_aware')
            a.latency_bias=self.pi(s,'latency',value-p['hol_target'],cfg['latency_bias'] if cfg['latency_bias'] is not None else 1.,.1,4.,.25,o.slot_id)
            return True
        if k=='adaptive_sfr':
            if not active:return False
            qs=[q for u in active for q in u.subband_cqi if finite(q)]
            if not qs:return False
            protected=clip((p['sfr_cqi_threshold']-float(np.quantile(qs,.1)))/15+float(c.resource_grid_utilization),0,1)
            color=sorted(cells).index(cid)%3
            scores=[(1+protected if g%3==color else 1-.75*protected) for g in range(n)]
            a.prb_power_w=[budget*x/sum(scores) for x in scores];a.power_group_size_prbs=1;return True
        if k=='max_min_reference':
            if cfg['ret_enabled']:self._reason(cid,'ret_owns_analog_subspace');return False
            reports={}
            for u in ues:
                for r in u.beam_reference_reports:
                    if r['cell_id']==cid and r['valid']:
                        reports.setdefault((r['codebook_id'],r['beam_profile_id']),{})[u.ue_id]=r['reference_sinr_db']
            scores={key:min(v.values()) for key,v in reports.items() if len(v)==len(ues)}
            if not scores:return False
            current=(cfg['codebook_id'],cfg['beam_profile_id']);best=max(sorted(scores),key=scores.get)
            if best==current or scores[best]<=scores.get(current,-999)+p['beam_gain_db']:return False
            a.codebook_id,a.beam_profile_id=best;return True
        if k in ('coverage_grid','window_search'):
            return self.search(o,c,cfg,ues,a)
        raise AssertionError(k)

    def search(self,o,c,cfg,ues,a):
        p=self.parameters;cid=c.cell_id;s=self.state.setdefault(cid,{})
        ret=self.algorithm=='coverage_grid'
        fields=('ret_enabled','ret_codebook_id','ret_tilt_deg') if ret else ('time_to_trigger_ms','steering_hysteresis_db')
        signature=deepcopy(getattr(o,'executed_configs',{}))
        for values in signature.values():
            for f in fields:values.pop(f,None)
        signature=(signature,tuple(sorted(u.ue_id for u in (ues if ret else o.ue_observations))))
        if s.get('signature')!=signature:
            if 'signature' in s:self._reason(cid,'search_context_changed')
            s.clear();s['signature']=signature;s['index']=0;s['scores']=[]
        pending=s.get('pending');window=p['ret_window_ms'] if ret else p['mro_evaluation_ms']
        produced=[u.kpi_metadata['reference_sinr_db']['produced_at_slot'] for u in ues]
        if pending:
            if any(abs(float(cfg[f])-v)>1e-8 for f,v in pending['target'].items()):
                self._reason(cid,'search_waiting_for_execution');return False
            # Start evaluation from an executed report, not the request time.
            pending.setdefault('executed',getattr(c,'config_produced_at_slot',o.slot_id))
            begin=pending['executed']
            if min(produced)*self.dt<(begin*self.dt+window):return False
            if ret:
                latest=min(produced)
                samples=[value for sample_slot,uid,value in self.search_samples if begin<sample_slot<=latest]
                if not samples:return False
                score=float(np.quantile(samples,.1))
            else:
                events={key:e for key,e in self.search_events.items() if e['slot_id']>begin and e.get('from_cell',e.get('cell_id'))==cid}
                ended=[e for e in events.values() if e['kind'] in ('handover_success','handover_failure')]
                if len(ended)<p['mro_min_attempts']:
                    self._reason(cid,'insufficient_completed_events');return False
                ordered=sorted(ended,key=lambda e:(e['slot_id'],e['ue_id']));last={};ping=0
                for e in ordered:
                    old=last.get(e['ue_id'])
                    if e['kind']=='handover_success':
                        if old and old.get('from_cell')==e.get('to_cell') and old.get('to_cell')==e.get('from_cell') and (e['slot_id']-old['slot_id'])*self.dt<=p['pingpong_ms']:ping+=1
                        last[e['ue_id']]=e
                failures=sum(e['kind']=='handover_failure' for e in ended)
                drops=sum(e['kind']=='connection_drop' for e in events.values())
                score=-(ping+failures+drops)/len(ended)
            s['scores'].append((score,pending['target']));s.pop('pending');s['index']+=1
        if ret:
            candidates=[dict(ret_enabled=True,ret_codebook_id=0,ret_tilt_deg=x) for x in (0.,3.,6.,9.,12.,15.)]
        else:
            if 'candidates' not in s:
                t,h=cfg['time_to_trigger_ms'],cfg['steering_hysteresis_db']
                s['candidates']=[dict(time_to_trigger_ms=t,steering_hysteresis_db=h)]+[dict(time_to_trigger_ms=clip(t+dt,20,480),steering_hysteresis_db=clip(h+dh,.5,12)) for dt,dh in ((20,0),(-20,0),(0,.5),(0,-.5))]
            candidates=s['candidates']
        if s['index']>=len(candidates):
            target=max(s['scores'],key=lambda item:item[0])[1]
            if all(cfg[k]==v for k,v in target.items()):return False
            for key,value in target.items():setattr(a,key,value)
            return True
        target=candidates[s['index']]
        for key,value in target.items():setattr(a,key,value)
        s['pending']=dict(target=target,issued=o.slot_id);return True

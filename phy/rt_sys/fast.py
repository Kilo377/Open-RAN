"""Fast network model with official SYS probabilities and explicit NR fallback.

Retains the full model's MAC MCS, TBS, HARQ identities and OLLA feedback.
No online fitting, MCS projection or change to the physical channel model.
"""
from collections import Counter, defaultdict
from types import SimpleNamespace
import math
import numpy as np
from phy.rt_sys.direct import DirectNrAbstraction
from phy.rt_sys.contracts import keyed_uniform
from phy.rt_sys.tx_contract import nr_budget

class FastReceiver:
    """Retain a TB's identity and its statistical/actual-NR route across retries."""
    def __init__(self,owner):self.owner=owner;self.real=owner.real.harq;self.stats=self.real.stats;self.mode='none'
    @property
    def pending(self):return self.real.pending
    def acknowledge(self,*args):return self.real.acknowledge(*args)
    def transmit_many(self,jobs):
        if self.owner.profile=='reference':return [self.transmit(*r) for r in jobs]
        probabilities=self.owner.probability_many([(r[0],r[1]) for r in jobs])
        return [self.transmit(*r,computed_probability=p) for r,p in zip(jobs,probabilities)]
    def transmit(self,wave,values,seed,slot,uid,context,*,computed_probability=None):
        key=(uid,int(context['process_id']));old=self.pending.get(key)
        signature=(int(context['ndi']),wave.table,wave.mcs,wave.data_re,str(context['cell_id']))
        if context['is_retx'] and (old is None or old['signature']!=signature):raise ValueError('Statistical HARQ contract mismatch')
        prob,eff,reason=computed_probability if computed_probability is not None else self.owner.probability(wave,values)
        use_nr=old.get('feedback_route')=='nr' if context['is_retx'] else prob is None
        if use_nr:
            realwave=self.owner.real.waveform(wave.table,wave.mcs,wave.data_re)
            info=self.real.transmit(realwave,values,seed,slot,uid,context)
            if key in self.pending:self.pending[key]['feedback_route']='nr'
            self.owner.reasons[reason or 'retained_nr_retransmission']+=1
            info.update(feedback_source='actual_nr_fallback',predicted_probability=prob)
            return info
        if prob is None:raise ValueError('SYS coverage lost during a statistical retransmission')
        if not context['is_retx']:
            old=dict(signature=signature,slot=slot,rounds=0,feedback_route='sys',bits=None,llr=None)
            self.pending[key]=old
        old['rounds']+=1;old['slot']=slot
        failed=keyed_uniform(seed,slot,uid,'tb_ack')<prob
        self.stats['statistical_tb_samples']+=1
        self.owner.prob_sum+=prob
        self.real.pending={k:v for k,v in self.pending.items() if slot-v['slot']<=2000}
        return dict(no_tb=False,ack=not failed,errors=int(failed),attempts=old['rounds'],mode='none',
                    reason='fast_sys_bernoulli',feedback_source='fast_sampled',predicted_probability=prob,effective_sinr=eff)

class FastAbstraction(DirectNrAbstraction):
    """Accepted fast model: SYS sampling, exact MAC budget, explicit NR fallback.

    The inherited reference_tb_ack field is the MAC feedback-presence gate;
    actual_decode distinguishes sampled outcomes from actual NR decoding.
    """
    feedback_mode = 'fast'
    def __init__(self,device='cuda',target_bler=.1,config=None):
        if (config or {}).get('harq_combining','none')!='none':
            raise ValueError('fast uses statistical HARQ feedback without soft combining; use full for Chase')
        super().__init__(device,target_bler,{**(config or {}),'nr_execution':'batched'})
        from phy.rt_sys.fast_profile import configure_fast_profile
        cfg={**(config or {}),'feedback_mode':'fast'}
        self.profile=configure_fast_profile(cfg)
        from sionna.sys import PHYAbstraction, EESM
        from sionna.phy.nr.utils import TransportBlockNR
        self.feedback_device='cpu' if self.profile=='efficient' else self.device
        self.sys=PHYAbstraction(device=self.feedback_device)
        if self.profile=='efficient':
            self.eesm=EESM(device='cpu')
            self.transport=TransportBlockNR(device='cpu')
        self.segments={}
        self.real=DirectNrAbstraction(device,target_bler,{**(config or {}),'nr_execution':'batched'})
        self.formats={};self.reasons=Counter();self.aliases={};self.alias_counts=Counter();self.prob_sum=0.;self.harq=FastReceiver(self)
    def waveform(self,table,mcs,count):
        key=(table,mcs,count)
        if key not in self.formats:
            from sionna.phy.nr.utils import MCSDecoderNR
            q,rate=MCSDecoderNR(device='cpu')(mcs,table,1)
            _,coded,bits=nr_budget({1:'qam64',2:'qam256',3:'qam64_low_se'}[table],mcs,count)
            self.formats[key]=SimpleNamespace(table=table,mcs=mcs,data_re=count,q=int(q),rate=float(rate),bits=bits,device=self.device,coding_backend='statistical_sys_no_codec')
        return self.formats[key]
    def equivalent(self,wave):
        tables=self.sys.bler_table['category'][1]['index']
        key=(wave.table,wave.mcs)
        if key not in self.aliases:
            from sionna.phy.nr.utils import MCSDecoderNR
            decoder=MCSDecoderNR(device='cpu');match=None
            for table in (1,2):
                for mcs in tables[table]['MCS']:
                    q,r=decoder(mcs,table,1)
                    if int(q)==wave.q and abs(float(r)-wave.rate)<1e-9:
                        match=(table,mcs);break
                if match:break
            self.aliases[key]=match
        alias=self.aliases[key]
        if alias is None:return None
        table,mcs=alias
        if nr_budget({1:'qam64',2:'qam256'}[table],mcs,wave.data_re)[2]!=wave.bits:raise ValueError('SYS alias changed NR TBS')
        if alias!=key:self.alias_counts[f'{key}->{alias}']+=1
        return alias
    @staticmethod
    def checked_probability(prob,eff):
        prob,eff=float(prob),float(eff)
        if not math.isfinite(prob) or not 0<=prob<=1 or not math.isfinite(eff):return None,None,'nonfinite_official_interpolation'
        return prob,eff,None
    def probability(self,wave,values):
        if self.profile=='efficient':return self.probability_many([(wave,values)])[0]
        alias=self.equivalent(wave)
        if alias is None:return None,None,'missing_equivalent_official_mcs_and_beta_entry'
        table,mcs=alias
        t=self.torch;index=t.tensor([[mcs]],dtype=t.int32,device=self.device)
        tensor=t.as_tensor(np.asarray(values).reshape(1,1,-1,1,1),dtype=t.float32,device=self.device)
        with t.no_grad():
            _,_,eff,p,_=self.sys(index,sinr=tensor,mcs_table_index=table,mcs_category=1)
        return self.checked_probability(p.item(),eff.item())
    def probability_many(self,items):
        """Official EESM and BLER only; omit unused SYS random/TB outputs."""
        if self.profile=='reference':return [self.probability(*item) for item in items]
        t=self.torch;groups=defaultdict(list);result=[None]*len(items)
        for i,(wave,values) in enumerate(items):
            alias=self.equivalent(wave)
            if alias is None:
                result[i]=(None,None,'missing_equivalent_official_mcs_and_beta_entry');continue
            groups[alias[0]].append((i,wave,np.asarray(values).reshape(-1),alias[1]))
        for table,group in groups.items():
            maximum=max(len(values) for _,_,values,_ in group)
            padded=np.zeros((len(group),1,maximum,1,1),dtype=np.float32)
            segments=[]
            for j,(_,wave,values,mcs) in enumerate(group):
                padded[j,0,:len(values),0,0]=values
                key=(table,mcs,wave.data_re)
                if key not in self.segments:
                    q=t.tensor([[wave.q]],dtype=t.int32)
                    rate=t.tensor([[wave.rate]],dtype=t.float32)
                    tb,cb,n=self.transport.transport_block_size(q,rate,q*wave.data_re)
                    if int(tb.item())!=wave.bits:raise ValueError('Cached SYS segmentation changed NR TBS')
                    self.segments[key]=(int(cb.item()),int(n.item()))
                segments.append(self.segments[key])
            index=t.tensor([[row[3]] for row in group],dtype=t.int32)
            tensor=t.as_tensor(padded,dtype=t.float32)
            with t.no_grad():
                eff=self.eesm(tensor,mcs_index=index,mcs_table_index=table,mcs_category=1,per_stream=False)
                cb=t.tensor([[s[0]] for s in segments],dtype=t.int32)
                n=t.tensor([[s[1]] for s in segments],dtype=t.float32)
                bler=self.sys.get_bler(index,table,1,cb,eff)
                one=t.tensor(1.,dtype=bler.dtype)
                p=t.where(t.isinf(bler),bler,one-t.pow(one-bler,n))
                outputs=t.stack((p,eff),dim=-1).numpy().reshape(-1,2)
            for row,(prob,eff) in zip(group,outputs):
                result[row[0]]=self.checked_probability(prob,eff)
        return result
    def evaluate(self,sinr,table,mcs,seed,slot,ue_id,fixed=False,harq_context=None):
        if harq_context is None:raise ValueError('fast requires an explicit transmitted-TB HARQ context')
        result=super().evaluate(sinr,table,mcs,seed,slot,ue_id,fixed=fixed,harq_context=harq_context);info=result.get('harq_reference') or {}
        self.decode_trials=self.harq.stats['actual_tb_decodes'];self.statistics['actual_tb_decodes']=self.decode_trials
        if info.get('feedback_source')=='fast_sampled':
            result.update(ack_source='fast_sys_bernoulli',tbler=info['predicted_probability'],cbler=None,
                          sinr_eff=info['effective_sinr'],bler_estimate_source='sionna_default_tables',
                          effective_sinr_kind='official_eesm',coding_backend='statistical_sys_no_codec')
        result['actual_decode']=info.get('feedback_source')=='actual_nr_fallback' and not info.get('no_tb',False)
        if result['actual_decode']:result['ack_source']='actual_nr_crc_and_bit_check'
        return result
    def model_statistics(self):
        actual=self.harq.stats['actual_tb_decodes'];sampled=self.harq.stats['statistical_tb_samples']
        return dict(model='fast',fast_profile=self.profile,feedback_device=self.feedback_device,actual_nr_decodes=actual,statistical_tb_samples=sampled,
            statistical_fraction=sampled/(actual+sampled) if actual+sampled else None,
            fallback_reasons=dict(self.reasons),equivalent_format_alias_counts=dict(self.alias_counts))
    def close(self):super().close();self.real.close()

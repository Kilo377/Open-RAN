"""Fast network model with official SYS probabilities and explicit NR fallback.

Retains the full model's MAC MCS, TBS, HARQ identities and OLLA feedback.
No online fitting, MCS projection or change to the physical channel model.
"""
from collections import Counter
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
    def transmit_many(self,jobs):return [self.transmit(*r) for r in jobs]
    def transmit(self,wave,values,seed,slot,uid,context):
        key=(uid,int(context['process_id']));old=self.pending.get(key)
        signature=(int(context['ndi']),wave.table,wave.mcs,wave.data_re,str(context['cell_id']))
        if context['is_retx'] and (old is None or old['signature']!=signature):raise ValueError('Statistical HARQ contract mismatch')
        prob,eff,reason=self.owner.probability(wave,values)
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
        from sionna.sys import PHYAbstraction
        self.sys=PHYAbstraction(device=self.device)
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
    def probability(self,wave,values):
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
        if alias is None:return None,None,'missing_equivalent_official_mcs_and_beta_entry'
        table,mcs=alias
        if nr_budget({1:'qam64',2:'qam256'}[table],mcs,wave.data_re)[2]!=wave.bits:raise ValueError('SYS alias changed NR TBS')
        if alias!=key:self.alias_counts[f'{key}->{alias}']+=1
        t=self.torch;index=t.tensor([[mcs]],dtype=t.int32,device=self.device)
        tensor=t.as_tensor(np.asarray(values).reshape(1,1,-1,1,1),dtype=t.float32,device=self.device)
        with t.no_grad():
            _,_,eff,p,_=self.sys(index,sinr=tensor,mcs_table_index=table,mcs_category=1)
        prob=float(p.item());eff=float(eff.item())
        if not math.isfinite(prob) or not 0<=prob<=1 or not math.isfinite(eff):return None,None,'nonfinite_official_interpolation'
        return prob,eff,None
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
        return dict(model='fast',actual_nr_decodes=actual,statistical_tb_samples=sampled,
            statistical_fraction=sampled/(actual+sampled) if actual+sampled else None,
            fallback_reasons=dict(self.reasons),equivalent_format_alias_counts=dict(self.alias_counts))
    def close(self):super().close();self.real.close()

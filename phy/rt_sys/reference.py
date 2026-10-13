"""Actual NR-TB feedback on the post-LMMSE equivalent Gaussian link.

AMC probes, a post-selection BLER estimate, and the transmitted TB use separate
keyed random streams. No default BLER/EESM table supplies a physical outcome.
HARQ retains the MAC lifecycle; this is an RV0 reference attempt, not IR/Chase.
"""
from collections import OrderedDict
from importlib.metadata import version
import math
import numpy as np
from phy.rt_sys.calibration import NRWaveform
from phy.rt_sys.contracts import digest


def reference_seed(seed,slot,ue_id,table,mcs,count,purpose):
    return int(digest([int(seed),int(slot),str(ue_id),int(table),int(mcs),int(count),purpose])[:15],16)


def probability_interval(errors,samples):
    z=1.959963984540054;p=errors/samples;d=1+z*z/samples
    center=(p+z*z/(2*samples))/d
    radius=z*math.sqrt(p*(1-p)/samples+z*z/(4*samples*samples))/d
    return [max(0.,center-radius),min(1.,center+radius)]


def supported_code_constraint(error):
    message=str(error)
    return ('Repetition coding is currently not supported' in message or
            message.startswith('Unsupported coderate (r<1/5)') or
            message.startswith('Unsupported coderate (r>0.95)'))


def reference_sys_class(base):
    class ReferenceSYS(base):
        def get_bler(self,mcs_index,mcs_table_index,mcs_category,cb_size,snr_eff):
            if getattr(self,'reference_probability',None) is None:
                raise RuntimeError('SYS reference probability was not supplied')
            p,num_cb=self.reference_probability
            cb=1. if p==1 else -np.expm1(np.log1p(-p)/num_cb)
            return self.torch_tensor(cb,snr_eff)

        def torch_tensor(self,value,like):
            import torch
            return torch.full_like(like,value)
    return ReferenceSYS


class NrReferenceAbstraction:
    feedback_mode='nr_tb_reference'
    calibration=None

    def __init__(self,device='cuda',target_bler=.1,config=None):
        import torch
        from sionna.sys import PHYAbstraction
        if version('sionna')!='2.2.0':raise RuntimeError('NR reference requires sionna==2.2.0')
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('NR reference CUDA unavailable')
        self.torch=torch;self.device='cuda:0' if device=='cuda' else device
        self.target_bler=float(target_bler);cfg=dict(config or {})
        self.selection_samples=int(cfg.get('selection_samples',64))
        self.bler_samples=int(cfg.get('bler_samples',64))
        self.cache_size=int(cfg.get('decoder_cache_size',128))
        if not 0<self.target_bler<1 or self.selection_samples<16 or self.bler_samples<32 or self.cache_size<1:
            raise ValueError('NR reference requires valid BLER target, >=16 AMC probes, >=32 BLER probes, and a decoder cache')
        self.selection_criterion=cfg.get('selection_criterion', 'wilson_upper')
        if self.selection_criterion not in ('wilson_upper', 'point_estimate'):
            raise ValueError('Unknown AMC selection criterion')
        self.candidates=cfg.get('mcs_candidates',{'qam64':list(range(29)), 'qam256':list(range(28)), 'qam64_low_se':list(range(29))})
        for table,maximum in (('qam64',28),('qam256',27)):
            indices=self.candidates.get(table)
            if not indices or len(set(indices))!=len(indices) or any(not isinstance(k,int) or not 0<=k<=maximum for k in indices):
                raise ValueError(f'Invalid explicit NR AMC candidate set: {table}')
        self.phy=reference_sys_class(PHYAbstraction)(device=self.device)
        self._waveforms=OrderedDict();self._unsupported={};self.calls=0;self.decode_trials=0

    def waveform(self,table,mcs,count):
        key=(table,mcs,count)
        if key in self._unsupported:raise ValueError(self._unsupported[key])
        if key not in self._waveforms:
            try:self._waveforms[key]=NRWaveform(table,mcs,count,self.device)
            except ValueError as exc:
                if not supported_code_constraint(exc):raise
                self._unsupported[key]=str(exc);raise
        self._waveforms.move_to_end(key)
        while len(self._waveforms)>self.cache_size:self._waveforms.popitem(last=False)
        return self._waveforms[key]

    def evaluate(self,sinr,table,mcs,seed,slot,ue_id,fixed=False):
        if table not in ('qam64','qam256','qam64_low_se'):raise ValueError('Unknown NR PDSCH table')
        table_index={'qam64':1,'qam256':2,'qam64_low_se':3}[table];maximum=27 if table_index==2 else 28
        if not 0<=int(mcs)<=maximum:raise ValueError('Invalid NR MCS index')
        values=np.asarray(sinr)[np.asarray(sinr)>0].astype(float)
        if not len(values) or not np.isfinite(values).all():raise ValueError('Reference requires positive finite scheduled SINRs')
        count=len(values);probes=[];excluded=[];wave=None;candidate=None;selection_met=None
        choices=[int(mcs)] if fixed else sorted({int(mcs)}|{k for k in self.candidates.get(table, list(range(maximum+1))) if k<=mcs},reverse=True)
        for choice in choices:
            try:w=self.waveform(table_index,choice,count)
            except ValueError as exc:
                if not supported_code_constraint(exc):raise
                excluded.append({'mcs':choice,'reason':str(exc)});continue
            wave,candidate=w,choice
            if fixed:break
            draw=reference_seed(seed,slot,ue_id,table_index,choice,count,'amc_probe')
            errors=int(w.simulate_fast([0.],values,self.selection_samples,draw)[0]);self.decode_trials+=self.selection_samples
            estimate=errors/self.selection_samples
            upper=probability_interval(errors,self.selection_samples)[1]
            selection_met=(upper if self.selection_criterion=='wilson_upper' else estimate)<=self.target_bler
            probes.append({'mcs':choice,'errors':errors,'samples':self.selection_samples,'seed':draw,'wilson_upper':upper})
            if selection_met:break
        if wave is None:
            # An unsupported low-rate tuple is an explicit transmission outage,
            # not a license to violate the MCS ceiling or silently switch PHY.
            return dict(mcs=mcs, qm=0, rate=0., capacity_bytes=0, sinr_eff=float(values.mean()),
                tbler=1., cbler=1., ack=False, feedback_mode=self.feedback_mode,
                ack_source='unsupported_nr_code_no_tb', unsupported_nr_mcs=excluded,
                amc_probes=probes, amc_selection_target_met=False,
                amc_selection_criterion=self.selection_criterion, bler_probe_samples=0,
                bler_probe_errors=0, reference_tb_ack=None, reference_tb_errors=None,
                bler_wilson_95=[0.,1.], effective_sinr_kind='arithmetic_mean_diagnostic_only')
        # Post-selection estimate is independent of the AMC selection samples.
        estimate_seed=reference_seed(seed,slot,ue_id,table_index,candidate,count,'bler_estimation')
        ack_seed=reference_seed(seed,slot,ue_id,table_index,candidate,count,'transmitted_tb')
        errors=int(wave.simulate_fast([0.],values,self.bler_samples,estimate_seed)[0])
        failed=int(wave.simulate_fast([0.],values,1,ack_seed)[0])
        self.decode_trials+=self.bler_samples+1;self.calls+=1
        probability=errors/self.bler_samples;t=self.torch
        index=t.tensor([[candidate]],dtype=t.int32,device=self.device)
        counts=t.tensor([[count]],dtype=t.int32,device=self.device)
        diagnostic_mean=float(values.mean())
        self.phy.reference_probability=(probability,wave.num_cb)
        try:
            with t.no_grad():
                _,_,_,tbler,cbler=self.phy(index,sinr_eff=t.tensor([[diagnostic_mean]],device=self.device),num_allocated_re=counts,mcs_table_index=table_index,mcs_category=1)
                qm,rate=self.phy._mcs_decoder_fun(index,table_index,1)
                bits,_,_=self.phy._transport_block_info(qm,rate,qm*counts)
        finally:self.phy.reference_probability=None
        if int(bits)!=wave.bits or abs(float(tbler)-probability)>1e-5:
            raise RuntimeError('SYS/NR reference TBS or probability bridge disagrees')
        return {'mcs':candidate,'qm':wave.q,'rate':wave.rate,'capacity_bytes':wave.bits//8,
            'sinr_eff':diagnostic_mean,'tbler':probability,'cbler':float(cbler),'ack':failed==0,
            'mcs_projection':None if candidate==mcs else f'nr_probe_selection:{mcs}->{candidate}',
            'feedback_mode':self.feedback_mode,'ack_source':'actual_nr_crc_and_bit_check',
            'bler_probe_errors':errors,'bler_probe_samples':self.bler_samples,'bler_wilson_95':probability_interval(errors,self.bler_samples),
            'amc_probes':probes,'unsupported_nr_mcs':excluded,'bler_probe_seed':estimate_seed,'ack_seed':ack_seed,
            'amc_selection_target_met':selection_met,'amc_selection_criterion':self.selection_criterion,
            'coding_backend':wave.coding_backend,
            'reference_tb_errors':failed,'reference_tb_ack':failed==0,
            'effective_sinr_kind':'arithmetic_mean_diagnostic_only'}

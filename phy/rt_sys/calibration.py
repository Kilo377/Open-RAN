"""NR-TB curves indexed by the actual coded RE budget, with SYS execution.

NR TBS rounding changes the actual code rate. Indexing just a coarse CBS grid
loses that information. Tables here retain exact MCS/table/RE tuples; an unseen
tuple fails rather than borrowing a differently rate-matched code.
"""
import hashlib
import json
import numpy as np


def curve_key(table,mcs,data_re):
    return f'{int(table)}:{int(mcs)}:{int(data_re)}'


def artifact_digest(data):
    return hashlib.sha256(json.dumps({k:data[k] for k in ('schema','curves','beta','waveform')},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def beta_value(spec,values):
    """Positive power-law EESM calibration; monotone under uniform SNR scaling."""
    if not isinstance(spec,dict):return float(spec)
    x=np.asarray(values,float);mean=float(x.mean());cv=float(x.std()/mean);db=10*np.log10(mean)
    knots=spec['cv_knots'];parameters=spec['parameters']
    logs=[p['log_beta']+p['slope']*(db-p['reference_db'])*np.log(10)/10 for p in parameters]
    logbeta=float(np.interp(cv,knots,logs))
    # Keep the log-sum-exp subtraction numerically resolvable at high SNR.
    return float(np.exp(np.clip(logbeta,np.log(mean)-20,np.log(mean)+12)))


def effective_sinr_value(spec,values):
    if isinstance(spec,dict) and spec.get('kind')=='qam_bicm_mi':
        from phy.rt_sys.mutual_information import effective_sinr
        return effective_sinr(spec,values)
    from scipy.special import logsumexp
    values=np.asarray(values,float);beta=beta_value(spec,values)
    return float(-beta*(logsumexp(-values/beta)-np.log(len(values))))


def load_artifact(path,allow_unvalidated=False):
    from pathlib import Path
    data=json.loads(Path(path).read_text())
    if data.get('schema')!='nr_tb_exact_re_v1':raise ValueError('Unsupported PHY calibration artifact')
    if data.get('digest')!=artifact_digest(data):raise ValueError('PHY calibration artifact digest mismatch')
    validation=data.get('validation',{})
    if not allow_unvalidated and (not validation.get('passed') or validation.get('fitted_digest')!=data['digest']):
        raise ValueError('PHY calibration requires matching independent validation')
    if not allow_unvalidated and 'held_out_rt_lmmse_profiles' not in validation.get('validated_profiles',[]):
        raise ValueError('PHY calibration requires independent actual RT-profile validation, not just artificial-profile pilots')
    if not data['curves']:raise ValueError('Empty PHY calibration domain')
    for key,c in data['curves'].items():
        x=np.asarray(c['snr_db']);y=np.asarray(c['tbler'])
        if len(x)<3 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x)<=0) or np.any(np.diff(y)>1e-9) or np.any((y<0)|(y>1)):
            raise ValueError(f'Invalid monotone NR BLER curve: {key}')
        beta=data['beta'].get(key)
        if isinstance(beta,dict) and beta.get('kind')=='qam_bicm_mi':
            if beta.get('qam_bits') not in (2,4,6,8) or len(beta['cv_knots'])!=len(beta['shift_db']) or not np.isfinite(beta['shift_db']).all() or np.any(np.diff(beta['cv_knots'])<=0):raise ValueError(f'Invalid MIESM parameters: {key}')
            if 'slope' in beta and (len(beta['slope'])!=len(beta['cv_knots']) or not np.isfinite(beta['slope']).all() or np.any(np.asarray(beta['slope'])<=0)):raise ValueError(f'Invalid monotone MIESM slopes: {key}')
        elif isinstance(beta,dict):
            if beta.get('kind')!='snr_variance_power_law' or len(beta['cv_knots'])!=len(beta['parameters']) or np.any(np.diff(beta['cv_knots'])<=0):raise ValueError(f'Invalid adaptive EESM beta: {key}')
            if any(not np.isfinite([p['log_beta'],p['slope'],p['reference_db']]).all() or not 0<=p['slope']<=4 for p in beta['parameters']):raise ValueError(f'Invalid monotone EESM parameters: {key}')
        elif beta is None or not np.isfinite(beta) or beta<=0:raise ValueError(f'Invalid EESM beta: {key}')
    return data


def calibrated_sys_class(base):
    """Keep SYS's NR TBS, segmentation, TBLER and feedback computation."""
    class CalibratedSYS(base):
        def __init__(self,artifact,**kwargs):
            self.artifact=artifact;self._counts=None
            super().__init__(**kwargs)

        def call(self,mcs_index,sinr=None,sinr_eff=None,num_allocated_re=None,**kwargs):
            if self._counts is not None:raise RuntimeError('Calibrated SYS instance is not reentrant')
            self._counts=(sinr>0).sum(dim=(-4,-3,-1)) if sinr is not None else num_allocated_re
            try:return super().call(mcs_index,sinr,sinr_eff,num_allocated_re,**kwargs)
            finally:self._counts=None

        def get_bler(self,mcs_index,mcs_table_index,mcs_category,cb_size,snr_eff):
            import torch
            if int(mcs_category)!=1:raise ValueError('NR calibration artifact covers PDSCH only')
            result=[]
            for mcs,count,snr in zip(mcs_index.flatten().tolist(),self._counts.flatten().tolist(),snr_eff.flatten().tolist()):
                curve=self.artifact['curves'].get(curve_key(mcs_table_index,mcs,count))
                if curve is None:raise ValueError(f'Unvalidated NR coded-resource tuple: table={mcs_table_index}, MCS={mcs}, RE={count}')
                # Lower/upper endpoints are independently simulated. Higher SNR
                # takes the measured high-SNR bound; no URLLC tail claim.
                probability=float(np.interp(10*np.log10(max(snr,1e-30)),curve['snr_db'],curve['tbler']))
                result.append(-np.expm1(np.log1p(-min(probability,1-1e-15))/curve['num_cb']) if probability<1 else 1.)
            return torch.tensor(result,dtype=self.dtype,device=self.device).reshape(snr_eff.shape)
    return CalibratedSYS


def stable_eesm_class(base):
    class StableEESM(base):
        def __init__(self,artifact,**kwargs):
            super().__init__(**kwargs);self.artifact=artifact

        def call(self,sinr,mcs_index,mcs_table_index=1,mcs_category=None,per_stream=False,**kwargs):
            import torch
            # SYS bridge evaluates one UE at a time. Keep the contract explicit.
            if sinr.shape[-2]!=1 or mcs_index.numel()!=1:raise ValueError('Exact-RE EESM bridge expects one UE')
            spec=self.artifact['beta'][curve_key(mcs_table_index,int(mcs_index),int((sinr>0).sum()))]
            if isinstance(spec,dict) and spec.get('kind')=='qam_bicm_mi':
                if per_stream:raise ValueError('MIESM bridge supports a joint TB only')
                eff=effective_sinr_value(spec,sinr[sinr>0].detach().cpu().numpy())
                return torch.full(mcs_index.shape,eff,dtype=sinr.dtype,device=sinr.device)
            beta=beta_value(spec,sinr[sinr>0].detach().cpu().numpy())
            x=sinr.to(torch.float64);mask=x>0
            axes=(-4,-3) if per_stream else (-4,-3,-1)
            count=mask.sum(dim=axes)
            z=torch.where(mask,-x/beta,torch.full_like(x,-torch.inf))
            eff=-beta*(torch.logsumexp(z,dim=axes)-torch.log(count.clamp_min(1)))
            eff=torch.where(count>0,eff,torch.zeros_like(eff))
            return eff.to(sinr.dtype)
    return StableEESM


class NRWaveform:
    """Actual PDSCH TB CRC, LDPC, rate matching and APP QAM demapping."""
    def __init__(self,table,mcs,data_re,device='cuda:0'):
        import torch
        from sionna.phy.nr import TBEncoder,TBDecoder
        from sionna.phy.nr.utils import MCSDecoderNR,TransportBlockNR
        from sionna.phy.mapping import Mapper,Demapper
        self.torch,self.device,self.data_re=torch,device,int(data_re)
        self.table,self.mcs=int(table),int(mcs)
        q,r=MCSDecoderNR(device=device)(mcs,table,1);self.q,self.rate=int(q),float(r)
        bits,cb,num=TransportBlockNR(device=device).transport_block_size(q,r,q*data_re)
        self.bits,self.cb,self.num_cb=int(bits),int(cb),int(num)
        self.coding_backend='sionna_tb_encoder'
        try:
            self.encoder=TBEncoder(self.bits,self.q*data_re,self.rate,self.q,channel_type='PDSCH',device=device)
            self.decoder=TBDecoder(self.encoder,num_bp_iter=20,device=device)
        except ValueError as exc:
            if ('Unsupported coderate (r<1/5)' not in str(exc) and 'Repetition coding is currently not supported' not in str(exc)) or self.num_cb != 1:
                raise
            from phy.rt_sys.low_rate import LowRateNRCodec
            self.low_rate_codec=LowRateNRCodec(self.bits,self.q,self.q*data_re,device)
            self.encoder=self.low_rate_codec.encode;self.decoder=self.low_rate_codec.decode
            self.coding_backend='sionna_bg2_rv0_circular_repetition'
        self.mapper=Mapper('qam',self.q,device=device);self.demapper=Demapper('app','qam',self.q,device=device)

    def simulate(self,snrs,profile,samples,seed,batch=64):
        t=self.torch;profile=np.asarray(profile,float)
        if profile.shape!=(self.data_re,) or np.any(profile<=0):raise ValueError('Positive per-RE profile required')
        gen=t.Generator(device=self.device);gen.manual_seed(seed)
        errors=np.zeros(len(snrs),int)
        gain=t.as_tensor(np.sqrt(profile),dtype=t.float32,device=self.device)
        # Pair identical codewords/noise across SNRs to stabilize fitted curves.
        with t.no_grad():
            for start in range(0,samples,batch):
                n=min(batch,samples-start);u=t.randint(0,2,(n,self.bits),device=self.device,generator=gen).float()
                x=self.mapper(self.encoder(u));z=(t.randn(x.shape,device=self.device,generator=gen)+1j*t.randn(x.shape,device=self.device,generator=gen))/2**.5
                for i,db in enumerate(snrs):
                    no=10**(-db/10);llr=self.demapper(x+z*no**.5/gain,t.as_tensor(no/profile,dtype=t.float32,device=self.device))
                    decoded,valid=self.decoder(llr)
                    errors[i]+=int((t.any(decoded!=u,dim=-1)|~valid.bool()).sum())
        return errors

    def simulate_fast(self,snrs,profile,samples,seed,batch=64,max_coded_elements=3000000):
        """Same waveform, with SNR as an additional decoder batch dimension."""
        t=self.torch;profile=np.asarray(profile,float);snrs=np.asarray(snrs,float)
        if profile.shape!=(self.data_re,) or np.any(profile<=0):raise ValueError('Positive per-RE profile required')
        batch=min(batch,max(1,max_coded_elements//(len(snrs)*self.q*self.data_re)))
        generator=t.Generator(device=self.device);generator.manual_seed(seed)
        errors=np.zeros(len(snrs),int)
        no=10**(-snrs/10)
        scale=t.as_tensor(np.sqrt(no[:,None,None]/profile[None,None,:]),dtype=t.float32,device=self.device)
        variance=t.as_tensor(no[:,None,None]/profile[None,None,:],dtype=t.float32,device=self.device)
        with t.no_grad():
            for start in range(0,samples,batch):
                n=min(batch,samples-start);u=t.randint(0,2,(n,self.bits),device=self.device,generator=generator).float()
                x=self.mapper(self.encoder(u));z=(t.randn(x.shape,device=self.device,generator=generator)+1j*t.randn(x.shape,device=self.device,generator=generator))/2**.5
                decoded,valid=self.decoder(self.demapper(x[None,:,:]+z[None,:,:]*scale,variance))
                failure=t.any(decoded!=u[None,:,:],dim=-1)|~valid.bool()
                errors+=failure.sum(dim=-1).cpu().numpy()
        return errors


def monotone(values):
    """Equal-weight binomial isotonic regression (nonincreasing BLER)."""
    blocks=[]
    for v in values:
        blocks.append([float(v),1])
        while len(blocks)>1 and blocks[-2][0]<blocks[-1][0]:
            b=blocks.pop();a=blocks.pop();n=a[1]+b[1];blocks.append([(a[0]*a[1]+b[0]*b[1])/n,n])
    return [v for mean,n in blocks for v in [mean]*n]

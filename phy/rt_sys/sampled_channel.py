"""Evaluate the same absolute path sum at requested physical grid coordinates.

No interpolation, gain normalization, or additional channel approximation is
introduced here. Resource masks and SYS/NR feedback still use the full grid.
"""
from collections import OrderedDict
import numpy as np


class PathCoefficients:
    def __init__(self, arrays, manifest, statistics=None):
        self.a=np.asarray(arrays['cir0'])
        self.tau=np.asarray(arrays['tau'])
        self.doppler=np.asarray(arrays['doppler'])
        if self.a.ndim!=5 or not all(np.isfinite(x).all() for x in (self.a,self.tau,self.doppler)):
            raise ValueError('Invalid sampled path coefficients')
        u,r,c,t,p=self.a.shape
        for x in (self.tau,self.doppler):
            if x.shape not in ((u,c,p),(u,r,c,t,p)):
                raise ValueError('Sampled delay/Doppler axes differ from CIR')
        self.shape=(u,c,r,t,manifest['num_symbols'],manifest['fft_size'])
        self.symbol_times=np.arange(manifest['num_symbols'],dtype=np.float32)*manifest['slot_duration_ms']/1000/manifest['num_symbols']
        self.frequencies=(np.arange(manifest['fft_size'],dtype=np.float32)-manifest['fft_size']//2)*manifest['subcarrier_spacing_hz']
        self.phases={};self.batch_phases=OrderedDict();self.frequency_moments={};self.stats=statistics if statistics is not None else {}

    def link(self,ui,ci):
        a=self.a[ui,:,ci]
        tau=self.tau[ui,ci][None,None,:] if self.tau.ndim==3 else self.tau[ui,:,ci]
        doppler=self.doppler[ui,ci][None,None,:] if self.doppler.ndim==3 else self.doppler[ui,:,ci]
        return a,tau,doppler

    def phase(self,ui,ci,frequencies):
        key=(ui,ci,frequencies.tobytes())
        if key not in self.phases:
            _,tau,_=self.link(ui,ci)
            self.phases[key]=np.exp(-2j*np.pi*tau[...,None]*self.frequencies[frequencies])
            if len(self.phases)>64:self.phases.pop(next(iter(self.phases)))
        return self.phases[key]

    def snapshot(self,elapsed_s):return PathSamples(self,elapsed_s)


class PathSamples:
    def __init__(self,coefficients,elapsed_s):
        self.coefficients=coefficients;self.shape=coefficients.shape
        self.times=elapsed_s+coefficients.symbol_times
        self.cache=OrderedDict();self.temporal_cache={};self.time_moments={}

    def temporal(self,ui,ci):
        key=(ui,ci)
        if key not in self.temporal_cache:
            a,_,doppler=self.coefficients.link(ui,ci)
            self.temporal_cache[key]=a[...,None]*np.exp(2j*np.pi*doppler[...,None]*self.times)
        return self.temporal_cache[key]

    def indices(self,symbols,frequencies):
        s=np.asarray(symbols,dtype=np.int64);f=np.asarray(frequencies,dtype=np.int64)
        if s.ndim!=1 or f.ndim!=1 or np.any(s<0) or np.any(s>=self.shape[-2]) or np.any(f<0) or np.any(f>=self.shape[-1]):
            raise ValueError('Sample coordinate outside physical channel grid')
        return s,f

    def evaluate(self,ui,ci,s,f,cartesian):
        key=(ui,ci,cartesian,s.tobytes(),f.tobytes())
        if key in self.cache:return self.cache[key]
        phase=self.coefficients.phase(ui,ci,f)
        at=self.temporal(ui,ci)[...,s]
        if cartesian:
            result=np.einsum('rtps,rtpf->rtsf',at,phase,optimize=True)
        else:
            result=np.sum(at*phase,axis=-2)
        result=result.astype(np.complex64)
        if not np.isfinite(result).all():raise ValueError('Nonfinite sampled CFR')
        stats=self.coefficients.stats;name='cartesian_points' if cartesian else 'paired_points'
        stats[name]=stats.get(name,0)+(len(s)*len(f) if cartesian else len(s))
        self.cache[key]=result
        if len(self.cache)>32:self.cache.popitem(last=False)
        return result

    def paired(self,ui,ci,symbols,frequencies):
        s,f=self.indices(symbols,frequencies)
        if len(s)!=len(f):raise ValueError('Paired channel coordinates must have equal lengths')
        return self.evaluate(ui,ci,s,f,False)

    def paired_links(self,links,symbols,frequencies):
        """Batch the same coordinates across the actual victim/transmitter links."""
        s,f=self.indices(symbols,frequencies)
        if len(s)!=len(f):raise ValueError('Paired channel coordinates must have equal lengths')
        links=tuple(links)
        if not links:return {}
        c=self.coefficients
        if not hasattr(self,'batch_temporal'):
            a=c.a.transpose(0,2,1,3,4)
            d=c.doppler[:,:,None,None,:] if c.doppler.ndim==3 else c.doppler.transpose(0,2,1,3,4)
            self.batch_temporal=a[...,None]*np.exp(2j*np.pi*d[...,None]*self.times)
        key=f.tobytes()
        if key not in c.batch_phases:
            tau=c.tau[:,:,None,None,:] if c.tau.ndim==3 else c.tau.transpose(0,2,1,3,4)
            c.batch_phases[key]=np.exp(-2j*np.pi*tau[...,None]*c.frequencies[f])
            if len(c.batch_phases)>8:c.batch_phases.popitem(last=False)
        ui,ci=np.asarray(links).T
        at=self.batch_temporal[ui,ci][...,s]
        values=np.sum(at*c.batch_phases[key][ui,ci],axis=-2).astype(np.complex64)
        if not np.isfinite(values).all():raise ValueError('Nonfinite sampled CFR')
        c.stats['batched_link_points']=c.stats.get('batched_link_points',0)+len(links)*len(s)
        return dict(zip(links,values))

    def cartesian(self,ui,ci,symbols,frequencies):
        s,f=self.indices(symbols,frequencies)
        return self.evaluate(ui,ci,s,f,True)

    def mean_reference_power(self,ui,ci,vectors,nprb,offset):
        """Exact original symbol/12-tone power average, without dense CFR.

        Synthetic arrays share path delays/Dopplers across antennas. The path
        cross-products factor into the same finite frequency and time sums.
        Antenna-specific paths retain the Cartesian reference implementation.
        """
        vectors=np.asarray(vectors);coefficients=self.coefficients
        if vectors.ndim!=2 or vectors.shape[0]!=self.shape[3]:raise ValueError('Reference beam dimensions differ from channel')
        if coefficients.tau.ndim!=3 or coefficients.doppler.ndim!=3:
            h=self.cartesian(ui,ci,np.arange(self.shape[-2]),np.arange(offset,offset+nprb*12))
            signal=np.einsum('rtsf,tk->rksf',h,vectors)
            return np.mean(np.sum(abs(signal)**2,axis=0),axis=1).reshape(vectors.shape[1],nprb,12).mean(axis=-1)
        key=(ui,ci,nprb,offset)
        if key not in coefficients.frequency_moments:
            f=np.arange(offset,offset+nprb*12,dtype=np.int64)
            phase=coefficients.phase(ui,ci,f)[0,0].astype(np.complex128)
            phase=phase.reshape(phase.shape[0],nprb,12)
            coefficients.frequency_moments[key]=np.einsum('pbf,qbf->bpq',phase,phase.conj())/12
        if (ui,ci) not in self.time_moments:
            d=coefficients.doppler[ui,ci]
            phase=np.exp(2j*np.pi*d[:,None]*self.times).astype(np.complex128)
            self.time_moments[ui,ci]=phase@phase.conj().T/self.shape[-2]
        a=coefficients.a[ui,:,ci]
        beam_paths=np.einsum('rtp,tk->rkp',a,vectors)
        cross=np.einsum('rkp,rkq->kpq',beam_paths,beam_paths.conj())
        result=np.einsum('kpq,bpq,pq->kb',cross,coefficients.frequency_moments[key],self.time_moments[ui,ci]).real
        coefficients.stats['reference_power_evaluations']=coefficients.stats.get('reference_power_evaluations',0)+1
        return np.maximum(result,0.)

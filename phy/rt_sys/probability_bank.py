"""Validated local BLER lookup, not EESM or extrapolation across channel shapes.

Each entry covers one ordered per-RE log-SINR bin vector (0.01 dB bins),
NR table/MCS/resource tuple. Calibration and two independent boundary probes
must agree. Feedback modes explicitly select actual NR decode or statistical TB sampling.
"""
from collections import OrderedDict,Counter
from pathlib import Path
import json
import numpy as np
from phy.rt_sys.contracts import digest


def interval(errors,samples):
    from phy.rt_sys.reference import probability_interval
    return probability_interval(errors,samples)


class ProbabilityBank:
    schema='nr_local_profile_bank_v2'

    def __init__(self,config=None):
        cfg=dict(config or {})
        self.samples=int(cfg.get('calibration_samples',256))
        self.validation_samples=int(cfg.get('validation_samples',256))
        self.limit=int(cfg.get('cache_entries',2048))
        self.width=float(cfg.get('bin_width_db',.01))
        if min(self.samples,self.validation_samples)<256 or self.limit<1 or not 0<self.width<=.01:
            raise ValueError('Profile bank requires >=256 independent samples and <=0.01 dB bins')
        self.entries=OrderedDict();self.rejected=set();self.stats=Counter()
        self.path=Path(cfg['bank_file']).resolve() if cfg.get('bank_file') else None
        if self.path:
            root=Path(__file__).resolve().parents[2]
            if not self.path.is_relative_to(root/'outputs'):raise ValueError('Bank must remain inside copy outputs')
            if self.path.exists():self.load()

    def identity(self,wave,values):
        bins=np.rint(10*np.log10(values)/self.width).astype(np.int64)
        spec=dict(table=wave.table,mcs=wave.mcs,data_re=wave.data_re,coding_backend=wave.coding_backend,
                  bin_width_db=self.width,bins_sha256=digest(bins.tolist()),sionna='2.2.0',bp_iterations=20)
        return digest(spec),spec,10**(bins*self.width/10)

    def probe(self,wave,values):
        key,spec,center=self.identity(wave,values)
        if key in self.entries:
            self.stats['hits']+=1;self.entries.move_to_end(key);return self.entries[key]
        if key in self.rejected:
            self.stats['rejected_fallbacks']+=1;return None
        self.stats['misses']+=1
        seed=lambda purpose:int(digest([key,purpose])[:15],16)
        def draw(profile,n,purpose):
            return int(wave.simulate_fast([0.],profile,n,seed(purpose),batch=256,max_coded_elements=12000000)[0])
        # Independent pilot batches permit early stopping only at jointly
        # saturated 0/1 points; interior points retain the full sample budget.
        initial=64;errors=draw(center,initial,'fit');nsamples=initial
        profiles=[center*10**(direction*self.width/20) for direction in (-1,1)]
        held=[draw(profile,initial,['held_out',direction]) for profile,direction in zip(profiles,(-1,1))]
        trials=3*initial
        saturated=all(e==0 for e in [errors,*held]) or all(e==initial for e in [errors,*held])
        if not saturated:
            errors+=draw(center,self.samples-initial,'fit_extension');nsamples=self.samples
            held=[e+draw(profile,self.validation_samples-initial,['held_out_extension',direction])
                  for e,profile,direction in zip(held,profiles,(-1,1))]
            trials+=self.samples-initial+2*(self.validation_samples-initial)
        validation_count=initial if saturated else self.validation_samples
        validations=[dict(direction=direction,errors=e,samples=validation_count,
            probability=e/validation_count,wilson_95=interval(e,validation_count)) for e,direction in zip(held,(-1,1))]
        self.stats['calibration_decode_trials']+=trials
        passed=all(abs(v['probability']-errors/nsamples)<=.08 for v in validations)
        if not passed:
            self.stats['validation_rejections']+=1;self.rejected.add(key);return None
        upper=max(interval(errors,nsamples)[1],*(v['wilson_95'][1] for v in validations))
        entry=dict(key=key,spec=spec,errors=errors,samples=nsamples,probability=errors/nsamples,
            wilson_95=interval(errors,nsamples),conservative_upper=upper,
            validation=dict(passed=True,max_absolute_bler_error=.08,probes=validations,
                independent_seeds=True,adaptive_saturated=saturated,scope='Ordered per-RE bin bounds only; finite-sample validation, not URLLC tails'))
        entry['waveform_metadata']={field:getattr(wave,field) for field in ('table','mcs','data_re','coding_backend','q','rate','bits','num_cb')}
        self.entries[key]=entry;self.stats['validated_entries']+=1
        while len(self.entries)>self.limit:self.entries.popitem(last=False)
        return entry

    def save(self):
        if not self.path:return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(schema=self.schema,bin_width_db=self.width,entries=list(self.entries.values()))
        payload['digest']=digest(payload)
        temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(payload));temp.replace(self.path)

    def load(self):
        d=json.loads(self.path.read_text());sha=d.pop('digest',None)
        if sha!=digest(d) or d.get('schema')!=self.schema or d.get('bin_width_db')!=self.width:
            raise ValueError('Invalid or incompatible validated probability bank')
        for entry in d['entries']:
            if entry['key']!=digest(entry['spec']) or not entry['validation']['passed'] or entry['samples']<64:
                raise ValueError('Unvalidated BLER bank entry')
            if any(p['samples']<64 for p in entry['validation']['probes']):raise ValueError('Insufficient independent bank validation')
            if any(abs(p['probability']-entry['probability'])>.08 for p in entry['validation']['probes']):raise ValueError('Failed BLER bank validation')
            saturated=entry['validation'].get('adaptive_saturated',False)
            if saturated:
                points=[(entry['errors'],entry['samples']),*((p['errors'],p['samples']) for p in entry['validation']['probes'])]
                if not (all(e==0 for e,n in points) or all(e==n for e,n in points)):raise ValueError('Invalid saturated early stopping')
            elif entry['samples']<256 or any(p['samples']<256 for p in entry['validation']['probes']):raise ValueError('Interior BLER requires full independent budgets')
            self.entries[entry['key']]=entry
        while len(self.entries)>self.limit:self.entries.popitem(last=False)

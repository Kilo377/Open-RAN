"""Compare deployment correlation without using an algorithm reward to select it."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import json,sys
from pathlib import Path
from itertools import combinations
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from phy.rt_sys.scenario import build_case
from phy.rt_sys.channel import RtChannelProvider
from phy.rt_sys.numerics import beam_subspace


if __name__=='__main__':
    previous=json.loads((ROOT/'outputs/rt_sys_targeted_repairs/original_scene_catalog.json').read_text())['munich'];reports=[]
    for count in (12,24):
        for label in ('previous','repaired'):
            m,_,_=build_case('munich',3,count,1,2)
            if label=='previous':
                m['cells']=previous['cells']
                for i,u in enumerate(m['ues']):
                    route=np.asarray(previous['routes'][i%len(previous['routes'])]);phase=((i//len(previous['routes']))+.5)/np.ceil(count/len(previous['routes']))
                    start=route[0]+min(.8,phase*.8)*(route[1]-route[0]);direction=route[1]-start;direction/=np.linalg.norm(direction)
                    u['trajectory']={'times_s':[0.,.001],'positions_m':[start.tolist(),(start+direction*.0015).tolist()]}
            provider=RtChannelProvider(m,ROOT/'channel_cache');h=provider.snapshot(1).cfr
            gain=np.mean(abs(h)**2,axis=(2,3,4,5));serving=gain.argmax(axis=1);nt=h.shape[3];vectors={}
            for i,ci in enumerate(serving):
                link=h[i,ci];nsc=288;offset=(link.shape[-1]-nsc)//2;freq=np.linspace(offset,offset+nsc-1,8,dtype=int)
                projected=link[:,:,0,freq].transpose(2,0,1)@beam_subspace(nt,0,0,0)
                cov=np.einsum('frt,frv->tv',projected.conj(),projected);_,v=np.linalg.eigh(cov);vectors[i]=v[:,-1]
            correlations=[float(np.clip(abs(np.vdot(vectors[i],vectors[j]))**2,0,1)) for i,j in combinations(range(count),2) if serving[i]==serving[j]]
            reports.append({'deployment':label,'ues':count,'cache_key':provider.key,'best_cell_counts':np.bincount(serving,minlength=3).tolist(),
                'same_cell_candidate_pairs':len(correlations),'pairs_below_scheduler_correlation_limit':sum(c<.8 for c in correlations),
                'correlation_quantiles':np.quantile(correlations,[0,.5,1]).tolist() if correlations else [],
                'scope':'Full baseline beam subspace and dominant covariance direction at slot 1; candidate pairs, not scheduled MU or a throughput gain'})
    out=ROOT/'outputs/rt_sys_targeted_repairs/munich_deployment_diagnosis.json';out.write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))

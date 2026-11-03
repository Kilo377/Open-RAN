"""Block-updated spatial fields with instantaneous scheduled-stream interference.

RZF spatial fields are retained for a bounded channel epoch. For each actual
power/stream state, an anchor LMMSE combiner produces scalar coupling gains.
Per-slot SINR is then desired power / (actual other-stream power + RX noise).
No queue, MCS, HARQ or control timing is cached.
"""
from collections import Counter, OrderedDict
import numpy as np
from phy.rt_sys.numerics import Algebra, beam_subspace
from phy.rt_sys.efficiency import resource_patterns
from phy.rt_sys.sampling import sample_coordinates


class SystemGainEngine:
    def __init__(self, update_ms=2., slot_ms=.5, compact=False, cache_static=True):
        slots=update_ms/slot_ms
        if not np.isfinite(slots) or slots<1 or not np.isclose(slots,round(slots)):
            raise ValueError('System gain interval must align to positive whole slots')
        self.slots=int(round(slots));self.epoch=None
        self.layouts=OrderedDict();self.fields=OrderedDict();self.combiners=OrderedDict()
        self.compact=compact
        if cache_static:
            from phy.rt_sys.numerics import cached_beam_subspace
            self.beam_subspace=cached_beam_subspace
        else:self.beam_subspace=beam_subspace
        self.stats=Counter();self.a=Algebra('numpy')

    @staticmethod
    def put(cache,key,value,limit=64):
        cache[key]=value;cache.move_to_end(key)
        while len(cache)>limit:cache.popitem(last=False)
        return value

    def layout(self,entries,ns,nf,offset):
        key=(ns,nf,offset,tuple((e['ue'].ue_id,e['layers'],e['ue'].dmrs_pattern_id,e['mask'].tobytes(),
            np.asarray(e['power']>0).tobytes()) for e in entries))
        if key in self.layouts:
            self.layouts.move_to_end(key);self.stats['layout_hits']+=1;return self.layouts[key]
        result=[]
        for pattern,coords in resource_patterns(entries,ns,nf).items():
            reduced,inverse=sample_coordinates(coords,4,offset)
            fulls,fullf=map(np.asarray,zip(*coords));s,f=map(np.asarray,zip(*reduced))
            result.append((pattern,fulls,fullf,s,f,inverse))
        self.stats['layout_builds']+=1
        from phy.rt_sys.compact_sinr import entry_layout
        templates=entry_layout(entries,result,ns,nf) if self.compact else None
        return self.put(self.layouts,key,(result,templates))

    def field(self,entries,pattern,s,f,snapshot,paper):
        controls=[]
        for i in pattern:
            e=entries[i];p=e['p'];cell=paper.state.managed_cells[p.cell_id].cell if paper else None
            controls.append((e['ue'].ue_id,e['ci'],e['layers'],p.current_codebook_id,
                p.current_beam_profile_id,int(e['ue'].serving_beam_id),
                getattr(cell,'ret_enabled',False),getattr(cell,'ret_tilt_deg',0),getattr(cell,'ret_codebook_id',0)))
        key=(tuple(controls),s.tobytes(),f.tobytes())
        if key in self.fields:
            self.stats['field_hits']+=1;return key,self.fields[key]
        nt=snapshot.grid_shape[3];a=self.a;owners=[];weights=[]
        links=sorted({(entries[i]['ui'],entries[j]['ci']) for i in pattern for j in pattern})
        h=snapshot.samples_for_links(links,s,f)
        for ci in sorted({entries[i]['ci'] for i in pattern}):
            group=[i for i in pattern if entries[i]['ci']==ci]
            directions=set();dft=self.beam_subspace(nt,0,0,0)
            for i in group:
                e=entries[i];p=e['p']
                sub=self.beam_subspace(nt,p.current_codebook_id,p.current_beam_profile_id,int(e['ue'].serving_beam_id))
                directions.update(np.argmax(abs(dft.conj().T@sub),axis=0).tolist())
            basis=dft[:,sorted(directions)]
            if paper and getattr(paper.state.managed_cells[snapshot.cell_ids[ci]].cell,'ret_enabled',False):
                from phy.rt_sys.paper_radio import basis as radio_basis
                basis=radio_basis(snapshot,paper.state.managed_cells[snapshot.cell_ids[ci]].cell,entries[group[0]]['ue'].serving_beam_id)
            rows=[];local=[]
            for i in group:
                e=entries[i];hu=np.moveaxis(h[e['ui'],ci],-1,0).astype(np.complex128)
                u,_,_=a.svd(hu);rows.append(a.adj(u)[...,:e['layers'],:]@hu)
                local.extend((i,k) for k in range(e['layers']))
            if len(local)>basis.shape[-1]:raise ValueError('Scheduled MU layers exceed physical beam subspace')
            rows=np.concatenate(rows,axis=-2);power=np.sum(abs(rows)**2,axis=-1)
            scales={symbol:max(float(power[s==symbol].mean()),1e-30) for symbol in set(s.tolist())}
            alpha=np.asarray([scales[x] for x in s])[:,None,None]*getattr(self,'regularization',.01)
            w=a.rzf(rows,basis,alpha)
            owners.extend(local);weights.extend(w[...,j] for j in range(len(local)))
        # [sample, victim stream, transmitting stream, RX antenna]
        channels=np.asarray([[h[entries[i]['ui'],entries[other]['ci']] for other,_ in owners] for i,_ in owners])
        fields=np.einsum('jvrtn,nvt->njvr',channels,np.stack(weights,axis=1),optimize=True)
        value=(tuple((entries[i]['ue'].ue_id,k) for i,k in owners),fields,snapshot.slot_id)
        self.stats['field_builds']+=1
        return key,self.put(self.fields,key,value)

    def fill(self,entries,snapshot,radio,paper,noise,regularization=.01):
        self.regularization=regularization
        epoch=(snapshot.cache_key,snapshot.slot_id//self.slots,snapshot.metadata.get('geometry_block_slot'),
               snapshot.grid_shape,regularization)
        if epoch!=self.epoch:
            self.fields.clear();self.combiners.clear();self.epoch=epoch;self.stats['epochs']+=1
        ns,nf=snapshot.grid_shape[-2:];offset=(nf-radio[0].n_prbs_total*12)//2 if radio else 0
        patterns,templates=self.layout(entries,ns,nf,offset)
        if self.compact:
            for e,(data,inverse,weights,parts,size) in zip(entries,templates):
                e.update(data_mask=data,compact_values=np.zeros(size),compact_parts=parts,
                         compact_inverse=inverse,compact_weights=weights,inter_sum=0.,intra_sum=0.,inter_count=0)
        for pattern,fulls,fullf,s,f,inverse in patterns:
            multiplicity=np.bincount(inverse,minlength=len(s))
            for ci in {entries[i]['ci'] for i in pattern}:
                group=[entries[i] for i in pattern if entries[i]['ci']==ci]
                if len(group)>1:
                    for e in group:e['mu_mask'][fulls,fullf]=True
            field_key,(identities,fields,anchor_slot)=self.field(entries,pattern,s,f,snapshot,paper)
            age=snapshot.slot_id-anchor_slot
            if not 0<=age<self.slots:raise RuntimeError('System spatial field exceeded causal update bound')
            self.stats['max_anchor_age_slots']=max(self.stats['max_anchor_age_slots'],age)
            by_uid={e['ue'].ue_id:e for e in entries}
            owners=[(by_uid[uid],k) for uid,k in identities]
            powers=np.stack([np.broadcast_to(e['power'],(ns,nf))[s,f] for e,k in owners],axis=-1)
            key=(field_key,powers.tobytes(),noise)
            if key in self.combiners:
                gains,noise_gains=self.combiners[key];self.stats['combiner_hits']+=1
            else:
                n=fields.shape[-1];gains=np.empty(fields.shape[:-1]);noise_gains=np.empty(fields.shape[:2])
                for j in range(len(owners)):
                    other=powers.copy();other[:,j]=0
                    covariance=np.einsum('svr,svk,sv->srk',fields[:,j],fields[:,j].conj(),other)+noise*np.eye(n)
                    q=np.linalg.solve(covariance,fields[:,j,j,:,None])[...,0]
                    q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-150)
                    gains[:,j]=abs(np.einsum('sr,svr->sv',q.conj(),fields[:,j]))**2
                    noise_gains[:,j]=np.sum(abs(q)**2,axis=-1)
                self.put(self.combiners,key,(gains,noise_gains));self.stats['combiner_builds']+=1
            raw=np.sum(abs(fields)**2,axis=-1)
            for j,(e,k) in enumerate(owners):
                interference=powers.copy();interference[:,j]=0
                values=powers[:,j]*gains[:,j,j]/np.maximum(np.sum(interference*gains[:,j],axis=-1)+noise*noise_gains[:,j],1e-300)
                if self.compact:
                    e['compact_values'][e['compact_parts'][pattern,k]]=values
                else:e['sinr'][fulls,fullf,k]=values[inverse]
                inter=[v for v,(other,_) in enumerate(owners) if other['ci']!=e['ci']]
                intra=[v for v,(other,_) in enumerate(owners) if other['ci']==e['ci'] and other is not e]
                inter_power=np.sum(powers[:,inter]*raw[:,j,inter],axis=-1)
                intra_power=np.sum(powers[:,intra]*raw[:,j,intra],axis=-1)
                if self.compact:
                    e['inter_sum']+=float(np.dot(inter_power,multiplicity))
                    e['intra_sum']+=float(np.dot(intra_power,multiplicity))
                    e['inter_count']+=len(inverse)
                else:
                    e['inter'].extend(inter_power[inverse].tolist())
                    e['intra'].extend(intra_power[inverse].tolist())
                e['precoder_norms'].append(1.)
        if self.compact:
            from phy.rt_sys.compact_sinr import CompactSinr
            for e in entries:
                e['compact_sinr']=CompactSinr(e['compact_values'],e['compact_weights'],e['compact_inverse'])
                e['inter']=[e['inter_sum']/max(1,e['inter_count'])]
                e['intra']=[e['intra_sum']/max(1,e['inter_count'])]
        return []

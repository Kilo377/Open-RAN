"""Same-codeword NR retransmission and Chase LLR combining on equivalent REs.

Resource count, MCS/table and serving cell must match; incompatible or missing
retransmission contexts produce explicit no-TB outcomes, never a new codeword.
"""
from collections import Counter
import numpy as np
from phy.rt_sys.contracts import digest


class NrHarqReceiver:
    def __init__(self,mode='none',max_contexts=2048):
        if mode not in ('none','chase'):raise ValueError('NR supports same-codeword none/Chase; IR not implemented')
        self.mode=mode;self.pending={};self.limit=max_contexts;self.stats=Counter()

    def transmit(self,wave,values,seed,slot,uid,context):
        t=wave.torch;key=(uid,int(context['process_id']))
        signature=(int(context['ndi']),wave.table,wave.mcs,wave.data_re,str(context['cell_id']))
        old=self.pending.get(key)
        if context['is_retx'] and (old is None or old['signature']!=signature):
            self.stats['incompatible_no_tb']+=1
            return dict(no_tb=True,reason='missing_original_codeword' if old is None else 'incompatible_coded_resource_or_cell')
        if not context['is_retx']:
            generator=t.Generator(device=wave.device)
            generator.manual_seed(int(digest([seed,uid,context['process_id'],context['ndi'],slot,'original_tb'])[:15],16))
            with t.no_grad():bits=t.randint(0,2,(1,wave.bits),generator=generator,device=wave.device).float()
            old=dict(signature=signature,bits=bits,llr=None,rounds=0,slot=slot,
                codeword_id=digest([seed,uid,context['process_id'],context['ndi'],slot,'original_tb']))
            self.pending[key]=old
        generator=t.Generator(device=wave.device)
        generator.manual_seed(int(digest([seed,slot,uid,'nr_tb_noise'])[:15],16))
        variance=t.as_tensor(1/np.asarray(values),dtype=t.float32,device=wave.device)
        with t.no_grad():
            symbols=wave.mapper(wave.encoder(old['bits']))
            z=(t.randn(symbols.shape,generator=generator,device=wave.device)+1j*t.randn(symbols.shape,generator=generator,device=wave.device))/2**.5
            llr=wave.demapper(symbols+z*t.sqrt(variance),variance)
            if self.mode=='chase' and old['llr'] is not None:llr=llr+old['llr']
            decoded,valid=wave.decoder(llr)
            failure=bool((t.any(decoded!=old['bits'],dim=-1)|~valid.bool()).item())
            old['llr']=llr.detach() if self.mode=='chase' else None
        old['rounds']+=1;old['slot']=slot;self.stats['actual_tb_decodes']+=1
        self.stats['chase_combined_attempts']+=self.mode=='chase' and old['rounds']>1
        self.pending={k:v for k,v in self.pending.items() if slot-v['slot']<=2000}
        if len(self.pending)>self.limit:raise RuntimeError('NR HARQ context limit exceeded')
        return dict(no_tb=False,ack=not failure,errors=int(failure),codeword_id=old['codeword_id'],
            attempts=old['rounds'],noise_stream_id=digest([seed,slot,uid,'nr_tb_noise']),mode=self.mode,reason='chase_llr_combined' if self.mode=='chase' and old['rounds']>1 else 'same_codeword_single_attempt')

    def acknowledge(self,uid,process_id):self.pending.pop((uid,int(process_id)),None)

    def transmit_many(self, requests):
        """Batch equal formats, preserving each scalar-shaped keyed RNG draw.

        Requests are independent UEs in one slot. Never concatenate random
        draws or change a codeword/noise stream when group membership changes.
        """
        from collections import defaultdict
        groups=defaultdict(list);results=[None]*len(requests)
        if len({r[4] for r in requests})!=len(requests):
            raise ValueError('Batched receiver requires distinct UEs')
        for i,r in enumerate(requests):
            w,values,seed,slot,uid,context=r
            groups[(w.table,w.mcs,w.data_re,w.device)].append((i,r))
        for group in groups.values():
            if len(group)==1:
                i,r=group[0];results[i]=self.transmit(*r);continue
            wave=group[0][1][0];t=wave.torch;prepared=[]
            for i,(_,values,seed,slot,uid,context) in group:
                key=(uid,int(context['process_id']))
                signature=(int(context['ndi']),wave.table,wave.mcs,wave.data_re,str(context['cell_id']))
                old=self.pending.get(key)
                if context['is_retx'] and (old is None or old['signature']!=signature):
                    self.stats['incompatible_no_tb']+=1
                    results[i]=dict(no_tb=True,reason='missing_original_codeword' if old is None else 'incompatible_coded_resource_or_cell')
                    continue
                if not context['is_retx']:
                    gen=t.Generator(device=wave.device)
                    gen.manual_seed(int(digest([seed,uid,context['process_id'],context['ndi'],slot,'original_tb'])[:15],16))
                    with t.no_grad():bits=t.randint(0,2,(1,wave.bits),generator=gen,device=wave.device).float()
                    old=dict(signature=signature,bits=bits,llr=None,rounds=0,slot=slot,
                        codeword_id=digest([seed,uid,context['process_id'],context['ndi'],slot,'original_tb']))
                    self.pending[key]=old
                gen=t.Generator(device=wave.device)
                gen.manual_seed(int(digest([seed,slot,uid,'nr_tb_noise'])[:15],16))
                # The scalar symbol tensor has shape [1, data_re].
                shape=(1,wave.data_re)
                z=(t.randn(shape,generator=gen,device=wave.device)+1j*t.randn(shape,generator=gen,device=wave.device))/2**.5
                variance=t.as_tensor(1/np.asarray(values),dtype=t.float32,device=wave.device).reshape(1,-1)
                prepared.append((i,uid,seed,slot,old,z,variance))
            if not prepared:continue
            with t.no_grad():
                bits=t.cat([p[4]['bits'] for p in prepared],dim=0)
                symbols=wave.mapper(wave.encoder(bits))
                z=t.cat([p[5] for p in prepared],dim=0)
                variance=t.cat([p[6] for p in prepared],dim=0)
                llr=wave.demapper(symbols+z*t.sqrt(variance),variance)
                if self.mode=='chase':
                    old_llr=t.cat([p[4]['llr'] if p[4]['llr'] is not None else t.zeros_like(llr[j:j+1])
                                   for j,p in enumerate(prepared)],dim=0)
                    llr=llr+old_llr
                decoded,valid=wave.decoder(llr)
                errors=(t.any(decoded!=bits,dim=-1).reshape(-1)|~valid.bool().reshape(-1)).detach().cpu().tolist()
                for j,(i,uid,seed,slot,old,_,_) in enumerate(prepared):
                    old['llr']=llr[j:j+1].detach() if self.mode=='chase' else None
                    old['rounds']+=1;old['slot']=slot;self.stats['actual_tb_decodes']+=1
                    self.stats['chase_combined_attempts']+=self.mode=='chase' and old['rounds']>1
                    results[i]=dict(no_tb=False,ack=not bool(errors[j]),errors=int(errors[j]),
                        codeword_id=old['codeword_id'],attempts=old['rounds'],
                        noise_stream_id=digest([seed,slot,uid,'nr_tb_noise']),mode=self.mode,
                        reason='chase_llr_combined' if self.mode=='chase' and old['rounds']>1 else 'same_codeword_single_attempt')
            self.stats['batched_groups']+=1;self.stats['batched_tbs']+=len(prepared)
        if requests:
            slot=max(r[3] for r in requests)
            self.pending={k:v for k,v in self.pending.items() if slot-v['slot']<=2000}
        if len(self.pending)>self.limit:raise RuntimeError('NR HARQ context limit exceeded')
        return results

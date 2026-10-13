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

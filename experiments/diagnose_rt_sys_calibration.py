"""Separate sparse SYS-table interpolation from NR TB/CB coding differences."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from phy.rt_sys.abstraction import SionnaAbstraction
from sionna.phy.nr.utils import CodedAWGNChannelNR
from sionna.phy.nr import TBEncoder,TBDecoder
from sionna.phy.mapping import Mapper,Demapper

if __name__=='__main__':
    model=SionnaAbstraction('cuda');records=[]
    for table,mcs,snr in [('qam64',28,19),('qam64',28,21),('qam64',10,4),('qam256',24,21)]:
        index={'qam64':1,'qam256':2}[table];data_re=324
        r=model.evaluate(np.ones((1,data_re,1))*10**(snr/10),table,mcs,1,0,'diagnostic',True)
        q,rate=r['qm'],r['rate'];bits=r['capacity_bytes']*8
        _,cb,_=model.transport.transport_block_size(torch.tensor(q,device=model.device),torch.tensor(rate,device=model.device),torch.tensor(q*data_re,device=model.device))
        channel=CodedAWGNChannelNR(q,int(cb),rate,device=model.device)
        torch.manual_seed(773)
        u,v=channel(512,snr-10*np.log10(q*rate));cb_bler=float(torch.any(u!=v,dim=-1).float().mean())
        enc=TBEncoder(bits,q*data_re,rate,q,channel_type='PDSCH',device=model.device);dec=TBDecoder(enc,device=model.device)
        mapper=Mapper('qam',q,device=model.device);demapper=Demapper('app','qam',q,device=model.device)
        torch.manual_seed(991);errors=0
        with torch.no_grad():
            for _ in range(8):
                u=torch.randint(0,2,(64,bits),device=model.device).float();x=mapper(enc(u));no=10**(-snr/10)
                y=x+(torch.randn_like(x.real)+1j*torch.randn_like(x.real))*(no/2)**.5
                decoded,_=dec(demapper(y,torch.tensor(no,device=model.device)))
                errors+=int(torch.any(u!=decoded,dim=-1).sum())
        record=dict(table=table,mcs=mcs,snr_db=snr,cbs=int(cb),sys_bler=r['tbler'],actual_cb_bler=cb_bler,actual_tb_bler=errors/512)
        print(json.dumps(record),flush=True);records.append(record)
    out=ROOT/'outputs/rt_sys_calibration_repair/diagnosis.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(records,indent=2)+'\n')

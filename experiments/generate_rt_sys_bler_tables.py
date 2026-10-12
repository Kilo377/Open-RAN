"""Generate custom SYS NR code-block BLER tables using the official PHY model.

Table generation is fitting data. Validate with independent TBEncoder seeds
using calibrate_rt_sys_phy.py --bler-table before claiming calibration. EESM
beta and frequency selective validation remain separate requirements.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--mcs',type=int,nargs='+',default=[3,10,20])
    p.add_argument('--table',type=int,choices=[1,2],default=1)
    p.add_argument('--cbs',type=int,nargs='+',default=[100,500,1000,2000,4000,8000])
    p.add_argument('--snr-db',type=float,nargs='+',default=list(range(-5,31)))
    p.add_argument('--batches',type=int,default=16);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'outputs'):raise ValueError('Generated tables must remain in copy outputs/')
    output.parent.mkdir(parents=True,exist_ok=True)
    from phy.rt_sys.abstraction import SionnaAbstraction
    model=SionnaAbstraction(args.device)
    model.phy.new_bler_table(args.snr_db,args.cbs,{'category':{1:{'index':{args.table:{'MCS':args.mcs}}}}},
        filename=str(output),batch_size=args.batch_size,max_mc_iter=args.batches,early_stop=False)

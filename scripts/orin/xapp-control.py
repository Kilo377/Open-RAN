#!/usr/bin/env python3
"""Submit one atomic, run-scoped activation request; no simulation restart."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.hot_xapps import submit_many

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--cell',default='*')
    p.add_argument('operation',choices=['enable','disable','set','disable_all'])
    p.add_argument('apps',nargs='*',help='IDs such as x01 x13 or implementation names')
    a=p.parse_args();data=json.loads(a.state.read_text())
    request=submit_many(a.state,data,a.operation,a.apps,a.cell)
    print(json.dumps(dict(request_id=request,status='submitted',note='Wait for the simulator admission and actual execution receipts')))

if __name__=='__main__':main()

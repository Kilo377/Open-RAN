"""Export an existing completed run without rerunning its simulation."""
from pathlib import Path
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.replay import export_replay

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--mesh',type=Path)
    parser.add_argument('--window-ms',type=float)
    parser.add_argument('--max-frames',type=int,default=500)
    args=parser.parse_args()
    import json
    print(json.dumps(export_replay(args.run,args.output,mesh_path=args.mesh,window_ms=args.window_ms,max_frames=args.max_frames),indent=2))
if __name__=='__main__':main()

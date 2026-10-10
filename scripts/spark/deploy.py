"""Transfer the independent copy when a real SSH destination is supplied.

No remote execution or scanning occurs without explicit CLI arguments.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import re
ROOT=Path(__file__).resolve().parents[2]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True,help='SSH config alias or user@LAN-address')
    p.add_argument('--port',type=int,default=22);p.add_argument('--identity',type=Path)
    p.add_argument('--update',action='store_true',help='Update an existing deployment owned by this project')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.@:-]+',args.host) or args.host.startswith('-'):
        raise ValueError('Invalid SSH destination')
    ssh=['ssh','-p',str(args.port)]
    if args.identity:ssh+=['-i',str(args.identity.expanduser().resolve())]
    marker='.oran_rt_sys_copy'
    if args.update:
        remote='test -f "$HOME/oran_slot_sim_sionna/.oran_rt_sys_copy"'
    else:
        remote='test ! -e "$HOME/oran_slot_sim_sionna" && mkdir "$HOME/oran_slot_sim_sionna" && touch "$HOME/oran_slot_sim_sionna/.oran_rt_sys_copy"'
    rsync=['rsync','-az','--exclude=.venv*','--exclude=.cache','--exclude=.local-llvm','--exclude=outputs','--exclude=channel_cache',
           '--exclude=__pycache__','--exclude=.pytest_cache','--exclude=.git','-e',shlex.join(ssh),str(ROOT)+'/',args.host+':oran_slot_sim_sionna/']
    if args.dry_run:
        print(shlex.join(ssh+[args.host,remote]));print(shlex.join(rsync))
    else:
        subprocess.run(ssh+[args.host,remote],check=True)
        subprocess.run(rsync,check=True)

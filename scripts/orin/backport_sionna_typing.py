"""Opt-in Python 3.10 annotation compatibility for Sionna RT 2.2.0.

JetPack 6 GPU wheels target Python 3.10. Upstream Sionna declares Python 3.11+;
this deployment validates fast/NR explicitly and changes annotation imports only.
"""
import hashlib
from importlib.metadata import distribution
import json
import sys


def main():
    if sys.version_info >= (3,11):
        print('Python 3.11+: no typing backport needed');return
    package=distribution('sionna-rt')
    if package.version!='2.2.0':raise RuntimeError('Backport is scoped to Sionna RT 2.2.0')
    edits={
        'sionna/rt/path_solvers/paths.py':('from typing import Literal, Self','from typing import Literal\nfrom typing_extensions import Self'),
        'sionna/rt/rcs/scattering_points.py':('from typing import Callable, Self, Tuple','from typing import Callable, Tuple\nfrom typing_extensions import Self'),
    }
    rows=[]
    for name,(old,new) in edits.items():
        p=package.locate_file(name);text=p.read_text();before=hashlib.sha256(p.read_bytes()).hexdigest()
        if old in text:
            if text.count(old)!=1:raise RuntimeError('Ambiguous annotation import: '+name)
            p.write_text(text.replace(old,new))
        elif new not in text:raise RuntimeError('Unexpected installed source: '+name)
        rows.append(dict(file=name,before=before,after=hashlib.sha256(p.read_bytes()).hexdigest()))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

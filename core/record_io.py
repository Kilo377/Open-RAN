"""Lossless run-record compression, independent of simulation clocks."""
import gzip
from pathlib import Path


def open_record(path, mode='r', *, compression='none', newline=None):
    path=Path(path)
    if mode not in ('r','w') or compression not in ('none','gzip'):
        raise ValueError('Record IO requires r/w and none/gzip')
    if path.suffix!='.gz':
        compressed=path.with_name(path.name+'.gz')
        if path.exists() and compressed.exists():raise ValueError('Ambiguous plain and gzip records; use a fresh output directory')
        if mode=='w' and ((compression=='gzip' and path.exists()) or (compression=='none' and compressed.exists())):
            raise FileExistsError('Alternate record format exists; use a fresh output directory')
    if mode=='r':
        if not path.exists():path=path.with_name(path.name+'.gz')
        zipped=path.suffix=='.gz'
    else:
        zipped=compression=='gzip'
        if zipped and path.suffix!='.gz':path=path.with_name(path.name+'.gz')
    if zipped:return gzip.open(path,mode+'t',encoding='utf-8',newline=newline,compresslevel=1)
    return path.open(mode,encoding='utf-8',newline=newline)

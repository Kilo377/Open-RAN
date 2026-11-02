"""Atomic, bounded compact cache writes; historical caches are never evicted."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import os
import time


@contextmanager
def cache_lock(root):
    with (Path(root)/'.storage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def compact_bytes(root):
    return sum(p.stat().st_size for p in Path(root).glob('*/paths_*.npz'))


def publish(tmp,path,quota_bytes):
    """Check quota under one lock, including all concurrent writers."""
    try:
        with cache_lock(path.parent.parent):
            if path.exists():return
            if compact_bytes(path.parent.parent)+tmp.stat().st_size>quota_bytes:
                raise RuntimeError('Compact channel cache quota exceeded; inspect cache inventory before cleanup')
            os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def ensure_paths(root,block,backend,quota_bytes,load=True):
    """One producer per physical block, including foreground/prefetch races.

    OS locks release on process failure. Publication remains atomic and quota
    checked; waiting consumers recheck existence after acquiring ownership.
    """
    import numpy as np
    root=Path(root);path=root/f'paths_{block:09d}.npz'
    started=time.perf_counter()
    with (root/f'.paths_{block:09d}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        waited=time.perf_counter()-started
        if path.exists():
            return (dict(np.load(path,allow_pickle=False)) if load else None),False,waited
        arrays=backend.generate_paths(block)
        tmp=path.with_suffix(f'.{os.getpid()}.tmp')
        try:
            with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
            publish(tmp,path,quota_bytes)
        finally:tmp.unlink(missing_ok=True)
        return arrays if load else None,True,waited

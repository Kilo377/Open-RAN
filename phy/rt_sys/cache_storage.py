"""Atomic, bounded compact cache writes; historical caches are never evicted."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import os


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

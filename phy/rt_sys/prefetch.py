"""Prepare deterministic geometry blocks alongside the causal network loop."""
import json
import os
from pathlib import Path
import time
import traceback


def _generate(manifest,cache_root,device,quota_bytes,expected_key,cancel):
    import numpy as np
    import drjit as dr
    dr.set_thread_count(1)
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.cache_storage import publish
    started=time.perf_counter();provider=None;generated=0
    try:
        provider=RtChannelProvider(manifest,Path(cache_root),device=device,quota_bytes=quota_bytes)
        if provider.key!=expected_key:raise RuntimeError('RT prefetch scene fingerprint differs from foreground')
        # The foreground produces block zero while this worker loads the scene.
        for block in provider.blocks[1:]:
            if cancel.is_set():break
            path=provider.root/f'paths_{block:09d}.npz'
            if path.exists():continue
            arrays=provider.backend.generate_paths(block)
            tmp=path.with_suffix(f'.{os.getpid()}.tmp')
            with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
            publish(tmp,path,quota_bytes);generated+=1
        status=dict(completed=True,generated_blocks=generated,worker_wall_s=time.perf_counter()-started)
    except BaseException:
        status=dict(completed=False,generated_blocks=generated,error=traceback.format_exc())
        raise
    finally:
        if provider is not None:
            path=provider.root/'prefetch_status.json';tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(status,indent=2));os.replace(tmp,path)


class PathPrefetch:
    def __init__(self,provider):
        if not provider.compact or provider.device!='cpu':raise ValueError('RT prefetch requires compact CPU ray tracing')
        import multiprocessing
        context=multiprocessing.get_context('spawn');self.cancel=context.Event()
        self.process=context.Process(target=_generate,args=(provider.manifest,provider.root.parent,
            provider.device,provider.quota_bytes,provider.key,self.cancel),daemon=True)
        self.process.start()

    def check(self):
        if self.process.exitcode not in (None,0):raise RuntimeError(f'RT prefetch worker failed with exit code {self.process.exitcode}')

    def close(self):
        self.cancel.set();self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate();self.process.join(timeout=5)
            raise RuntimeError('RT prefetch worker failed to stop')
        self.check()

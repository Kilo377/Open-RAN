"""Prepare deterministic geometry blocks alongside the causal network loop."""
import json
import os
from pathlib import Path
import time
import traceback
import uuid


def _generate(manifest,cache_root,device,quota_bytes,expected_key,cancel,worker_index=0,workers=1,status_path=None):
    import drjit as dr
    dr.set_thread_count(1)
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.cache_storage import ensure_paths
    started=time.perf_counter();provider=None;generated=0;skipped=0;waited=0.
    try:
        provider=RtChannelProvider(manifest,Path(cache_root),device=device,quota_bytes=quota_bytes)
        if provider.key!=expected_key:raise RuntimeError('RT prefetch scene fingerprint differs from foreground')
        # The foreground produces block zero while this worker loads the scene.
        for block in provider.blocks[1+worker_index::workers]:
            if cancel.is_set():break
            path=provider.root/f'paths_{block:09d}.npz'
            if path.exists():skipped+=1;continue
            _,created,wait=ensure_paths(provider.root,block,provider.backend,quota_bytes,load=False)
            generated+=int(created);skipped+=int(not created);waited+=wait
        status=dict(completed=True,generated_blocks=generated,skipped_blocks=skipped,lock_wait_s=waited,worker_wall_s=time.perf_counter()-started)
    except BaseException:
        status=dict(completed=False,generated_blocks=generated,error=traceback.format_exc())
        raise
    finally:
        if provider is not None:
            path=Path(status_path) if status_path else provider.root/'prefetch_status.json';tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(status,indent=2));os.replace(tmp,path)


class PathPrefetch:
    def __init__(self,provider,workers=1):
        if not provider.compact or provider.device!='cpu':raise ValueError('RT prefetch requires compact CPU ray tracing')
        if type(workers) is not int or not 1<=workers<=8:raise ValueError('RT prefetch workers must be an integer in 1..8')
        import multiprocessing
        context=multiprocessing.get_context('spawn');self.cancel=context.Event()
        self.processes=[];self.status_paths=[]
        token=uuid.uuid4().hex
        for i in range(workers):
            status=provider.root/f'prefetch_status_{token}_{i}.json';self.status_paths.append(status)
            process=context.Process(target=_generate,args=(provider.manifest,provider.root.parent,
                provider.device,provider.quota_bytes,provider.key,self.cancel,i,workers,str(status)),daemon=True)
            self.processes.append(process);process.start()

    def check(self):
        for process in self.processes:
            if process.exitcode not in (None,0):raise RuntimeError(f'RT prefetch worker failed with exit code {process.exitcode}')

    def statistics(self):
        rows=[json.loads(p.read_text()) for p in self.status_paths if p.exists()]
        return dict(completed=len(rows)==len(self.processes) and all(r.get('completed') for r in rows),
            workers=len(self.processes),generated_blocks=sum(r.get('generated_blocks',0) for r in rows),
            skipped_blocks=sum(r.get('skipped_blocks',0) for r in rows),
            lock_wait_s=sum(r.get('lock_wait_s',0) for r in rows),
            worker_wall_s=max((r.get('worker_wall_s',0) for r in rows),default=0),
            worker_wall_sum_s=sum(r.get('worker_wall_s',0) for r in rows),
            worker_statuses=rows)

    def close(self):
        self.cancel.set();deadline=time.monotonic()+30
        for process in self.processes:process.join(timeout=max(0,deadline-time.monotonic()))
        alive=[p for p in self.processes if p.is_alive()]
        for process in alive:process.terminate()
        for process in alive:process.join(timeout=5)
        if alive:raise RuntimeError('RT prefetch worker failed to stop')
        self.check()

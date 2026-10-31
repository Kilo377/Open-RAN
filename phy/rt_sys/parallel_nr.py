"""Process-local NR codecs; the simulation process owns every HARQ context.

Only independent UE attempts within one slot run concurrently. Keyed RNG,
20 BP iterations, codewords and returned LLRs retain scalar semantics.
"""
from collections import OrderedDict


def initialize_worker(threads,workers,pin_workers):
    if pin_workers:
        import multiprocessing
        import os
        cpus=sorted(os.sched_getaffinity(0))
        index=(multiprocessing.current_process()._identity[0]-1)%workers
        os.sched_setaffinity(0,cpus[index*threads:(index+1)*threads])
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


_waveforms=OrderedDict()


def decode_attempt(job):
    from phy.rt_sys.calibration import NRWaveform
    from phy.rt_sys.nr_harq import NrHarqReceiver
    import torch
    key=(job['table'],job['mcs'],job['data_re'])
    if key not in _waveforms:_waveforms[key]=NRWaveform(*key,device='cpu')
    wave=_waveforms[key];_waveforms.move_to_end(key)
    while len(_waveforms)>128:_waveforms.popitem(last=False)
    receiver=NrHarqReceiver(job['mode'])
    context_key=(job['uid'],int(job['context']['process_id']))
    old=job['old']
    if old is not None:
        old=dict(old)
        for name in ('bits','llr'):
            if old[name] is not None:old[name]=torch.from_numpy(old[name])
        receiver.pending[context_key]=old
    info=receiver.transmit(wave,job['values'],job['seed'],job['slot'],job['uid'],job['context'])
    updated=receiver.pending.get(context_key)
    if updated is not None:
        updated=dict(updated)
        for name in ('bits','llr'):
            if updated[name] is not None:updated[name]=updated[name].detach().numpy()
    return info,updated,dict(receiver.stats)


class ParallelNrReceiver:
    def __init__(self,workers,threads=1,pin_workers=False):
        if type(workers) is not int or workers<1:raise ValueError('nr_parallel_workers must be a positive integer')
        if type(threads) is not int or threads<1:raise ValueError('nr_worker_threads must be a positive integer')
        if type(pin_workers) is not bool:raise ValueError('nr_pin_workers must be boolean')
        if pin_workers:
            import os
            if not hasattr(os,'sched_getaffinity'):raise ValueError('Worker CPU affinity requires Linux')
            if workers*threads>len(os.sched_getaffinity(0)):raise ValueError('Pinned workers exceed available CPU cores')
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(threads,workers,pin_workers))

    def transmit_many(self,receiver,requests):
        import torch
        if len({r[4] for r in requests})!=len(requests):raise ValueError('Parallel receiver requires distinct UEs')
        jobs=[]
        for wave,values,seed,slot,uid,context in requests:
            if wave.device!='cpu':raise ValueError('Process-parallel NR requires CPU waveforms')
            old=receiver.pending.get((uid,int(context['process_id'])))
            if old is not None:
                old=dict(old)
                for name in ('bits','llr'):
                    if old[name] is not None:old[name]=old[name].detach().numpy()
            jobs.append(dict(table=wave.table,mcs=wave.mcs,data_re=wave.data_re,values=values,
                seed=seed,slot=slot,uid=uid,context=context,old=old,mode=receiver.mode))
        futures=[self.pool.submit(decode_attempt,job) for job in jobs]
        results=[]
        for job,future in zip(jobs,futures):
            info,updated,stats=future.result()
            if updated is not None:
                for name in ('bits','llr'):
                    if updated[name] is not None:updated[name]=torch.from_numpy(updated[name])
                receiver.pending[(job['uid'],int(job['context']['process_id']))]=updated
            receiver.stats.update(stats);results.append(info)
        if requests:
            slot=max(r[3] for r in requests)
            receiver.pending={k:v for k,v in receiver.pending.items() if slot-v['slot']<=2000}
        if len(receiver.pending)>receiver.limit:raise RuntimeError('NR HARQ context limit exceeded')
        receiver.stats['parallel_slots']+=bool(requests)
        receiver.stats['parallel_tbs']+=len(requests)
        return results

    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)

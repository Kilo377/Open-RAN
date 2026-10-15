"""One actual NR decode per transmitted attempt, with no Monte Carlo probes.

The MAC supplies MCS. No target-BLER selection or probability prediction is
claimed. Controller BLER is estimated by the existing real ACK/NACK window.
"""
import numpy as np
from phy.rt_sys.reference import NrReferenceAbstraction, reference_seed, supported_code_constraint


class DirectNrAbstraction(NrReferenceAbstraction):
    feedback_mode = 'nr_direct'

    def __init__(self, device='cuda', target_bler=.1, config=None):
        # Reuse only codec/SYS/HARQ construction; probe budgets are inapplicable.
        cfg = dict(config or {})
        cfg.update(selection_samples=64, bler_samples=64)
        super().__init__(device, target_bler, cfg)
        self.selection_samples = self.bler_samples = 0
        self.information_screening = False
        self.statistics.update(amc_probe_trials=0, bler_probe_trials=0, calibration_decode_trials=0)

    def _probe(self, *args, **kwargs):
        raise RuntimeError('nr_direct must never run a Monte Carlo probe')

    def evaluate_many(self, requests):
        if getattr(self,'_batch_feedback',None) is not None:
            raise RuntimeError('NR batch is not reentrant')
        if any(not r.get('harq_context') for r in requests):
            raise ValueError('Batched actual NR requires explicit HARQ context')
        jobs=[]
        for r in requests:
            index={'qam64':1,'qam256':2,'qam64_low_se':3}[r['table']]
            values=np.asarray(r['sinr']);values=values[values>0].astype(float)
            if not len(values) or not np.isfinite(values).all():raise ValueError('Invalid batch SINR')
            wave=self.waveform(index,int(r['mcs']),len(values))
            jobs.append((wave,values,r['seed'],r['slot'],r['ue_id'],r['harq_context']))
        feedback=self.harq.transmit_many(jobs)
        self._batch_feedback={r['ue_id']:info for r,info in zip(requests,feedback)}
        try:return [self.evaluate(**r) for r in requests]
        finally:self._batch_feedback=None

    def _transmitted_feedback(self,wave,values,seed,slot,ue_id,ack_seed,harq_context,estimate_probe):
        if getattr(self,'_batch_feedback',None) is not None:
            info=self._batch_feedback.pop(ue_id)
            return (None if info['no_tb'] else info['errors']),info
        return super()._transmitted_feedback(wave,values,seed,slot,ue_id,ack_seed,harq_context,estimate_probe)

    def evaluate(self, sinr, table, mcs, seed, slot, ue_id, fixed=False, harq_context=None):
        index = {'qam64': 1, 'qam256': 2, 'qam64_low_se': 3}.get(table)
        if index is None or not 0 <= int(mcs) <= (27 if index == 2 else 28):
            raise ValueError('Invalid NR MCS/table')
        values = np.asarray(sinr)
        values = values[values > 0].astype(float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('Direct NR requires positive finite allocated SINRs')
        result = dict(mcs=int(mcs), qm=0, rate=0., capacity_bytes=0,
                      sinr_eff=float(values.mean()), tbler=1., cbler=1., ack=False,
                      feedback_mode=self.feedback_mode, reference_tb_ack=None, reference_tb_errors=None,
                      amc_probes=[], amc_selection_target_met=None,
                      amc_selection_criterion='mac_mcs_no_target_bler_guarantee',
                      mcs_projection='mac_mcs_actual_nr_no_probes',
                      bler_probe_samples=0, bler_probe_errors=0, bler_probe_seed=None,
                      bler_wilson_95=[0., 1.], unsupported_nr_mcs=[],
                      bler_estimate_source='actual_tb_error_indicator_not_probability',
                      effective_sinr_kind='arithmetic_mean_diagnostic_only')
        try:
            wave = self.waveform(index, int(mcs), len(values))
        except ValueError as exc:
            if not supported_code_constraint(exc):
                raise
            self.statistics['unsupported_no_tb'] = self.statistics.get('unsupported_no_tb', 0) + 1
            return dict(result, ack_source='unsupported_nr_code_no_tb',
                        unsupported_nr_mcs=[dict(mcs=int(mcs), reason=str(exc))])
        # Retain the SYS transport-block-size contract, without asking SYS for BLER.
        from phy.rt_sys.tx_contract import nr_budget
        q, coded, bits = nr_budget(table, int(mcs), len(values))
        if bits != wave.bits or q != wave.q or coded != wave.q * wave.data_re:
            raise RuntimeError('Direct NR/SYS transport block sizes disagree')
        draw = reference_seed(seed, slot, ue_id, index, int(mcs), len(values), 'transmitted_tb')
        failed, info = self._transmitted_feedback(wave, values, seed, slot, ue_id, draw,
                                                harq_context, None)
        if failed is None:
            return dict(result, ack_source='harq_incompatible_no_tb', harq_reference=info,
                        coding_backend=wave.coding_backend)
        self.calls += 1
        self.decode_trials += 1
        self.statistics['actual_tb_decodes'] = self.decode_trials
        # A realized 0/1 block error, not a model probability. MeasurementUpdater
        # separately accumulates transmitted ACK/NACK for the controller BLER.
        return dict(result, qm=wave.q, rate=wave.rate, capacity_bytes=wave.bits // 8,
                    tbler=float(failed), cbler=float(failed), ack=not failed,
                    ack_source='actual_nr_crc_and_bit_check', ack_seed=draw,
                    reference_tb_ack=not failed, reference_tb_errors=int(failed),
                    harq_reference=info, coding_backend=wave.coding_backend)

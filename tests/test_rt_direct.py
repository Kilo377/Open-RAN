import numpy as np
import pytest

from phy.rt_sys.direct import DirectNrAbstraction


def test_direct_never_probes_and_matches_actual_reference_tb(monkeypatch):
    from phy.rt_sys.reference import NrReferenceAbstraction
    from phy.rt_sys.calibration import NRWaveform
    from phy.rt_sys.probability_bank import ProbabilityBank
    reference = NrReferenceAbstraction('cpu')
    vector = np.ones(324) * .5
    expected = reference.evaluate(vector, 'qam64', 3, 17, 4, 'u', fixed=True)
    direct = DirectNrAbstraction('cpu')
    calls = []
    original = NRWaveform.simulate_fast

    def only_actual(self, snrs, profile, samples, seed, **kwargs):
        assert samples == 1
        calls.append(samples)
        return original(self, snrs, profile, samples, seed, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('A probability probe was invoked')

    monkeypatch.setattr(NRWaveform, 'simulate_fast', only_actual)
    monkeypatch.setattr(ProbabilityBank, 'probe', forbidden)
    monkeypatch.setattr(NrReferenceAbstraction, '_probe', forbidden)
    actual = direct.evaluate(vector, 'qam64', 3, 17, 4, 'u')
    assert actual['ack'] == expected['ack']
    assert actual['capacity_bytes'] == expected['capacity_bytes']
    assert actual['mcs'] == 3 and actual['bler_probe_samples'] == 0
    assert actual['amc_probes'] == [] and actual['amc_selection_target_met'] is None
    assert actual['tbler'] == float(not actual['ack'])
    assert calls == [1] and direct.decode_trials == direct.calls == 1
    assert not hasattr(direct, 'bank')


def test_direct_same_word_chase_and_incompatible_resources_do_not_probe(monkeypatch):
    from phy.rt_sys.calibration import NRWaveform
    direct = DirectNrAbstraction('cpu', config={'harq_combining': 'chase'})
    # Physical HARQ must go through retained bits/LLRs, not simulate a new word.
    monkeypatch.setattr(NRWaveform, 'simulate_fast', lambda *a, **k: pytest.fail('New-word simulator'))
    context = dict(process_id=1, ndi=1, is_retx=False, cell_id='c0')
    first = direct.evaluate(np.ones(324)*.1, 'qam64', 3, 501, 1, 'u', harq_context=context)
    context['is_retx'] = True
    bad = direct.evaluate(np.ones(336)*100, 'qam64', 3, 501, 2, 'u', harq_context=context)
    assert bad['capacity_bytes'] == 0 and bad['ack_source'] == 'harq_incompatible_no_tb'
    assert direct.decode_trials == 1
    second = direct.evaluate(np.ones(324)*100, 'qam64', 3, 501, 3, 'u', harq_context=context)
    assert first['harq_reference']['codeword_id'] == second['harq_reference']['codeword_id']
    assert second['ack'] and second['harq_reference']['reason'] == 'chase_llr_combined'
    assert direct.decode_trials == 2
    assert direct.statistics['amc_probe_trials'] == direct.statistics['bler_probe_trials'] == 0


def test_direct_adapter_uses_physical_harq_and_exposes_mode():
    from phy.rt_sys.adapter import RtSysPhyAdapter
    adapter = RtSysPhyAdapter({'device': 'cpu', 'feedback_mode': 'nr_direct', 'harq_combining': 'chase'})
    assert adapter.combiner.mode == 'none'  # no double SINR + LLR combining
    assert adapter.abstraction.harq.mode == 'chase'


def test_direct_mac_selects_mcs_instead_of_executing_reference_ceiling(monkeypatch):
    from types import SimpleNamespace
    from phy.rt_sys.scheduler import RtSysScheduler
    from ran.scheduler import SimpleScheduler
    scheduler = object.__new__(RtSysScheduler)
    monkeypatch.setattr(SimpleScheduler, '_derive_target_mcs', lambda *a, **kw: 4)
    user = SimpleNamespace(rt_feedback_mode='nr_direct', latency_budget_ms=100)
    assert scheduler._derive_target_mcs(user, is_retx=False) == 4
    monkeypatch.setattr(scheduler, '_get_active_harq_process', lambda ue: SimpleNamespace(mcs_index=2))
    assert scheduler._derive_target_mcs(user, is_retx=True) == 2

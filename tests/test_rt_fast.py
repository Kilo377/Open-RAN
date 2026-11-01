"""Full/fast selection, exact NR budgets, and honest sampled feedback."""
import json
import numpy as np
import pytest
from phy.rt_sys.fast import FastAbstraction
from phy.rt_sys.contracts import has_tb_feedback
from experiments.rt_link_statistics import LinkStatistics


def context(retx=False):return dict(process_id=1,ndi=1,is_retx=retx,cell_id='c0')

def test_fast_supported_format_never_constructs_an_actual_codec(monkeypatch):
    model=FastAbstraction('cpu',.1)
    monkeypatch.setattr(model.real,'waveform',lambda *a:pytest.fail('Unexpected actual NR codec'))
    result=model.evaluate(np.ones(324)*100,'qam64',11,7,1,'u',harq_context=context())
    assert result['ack'] and not result['actual_decode']
    assert result['ack_source']=='fast_sys_bernoulli'
    assert model.decode_trials==0 and model.model_statistics()['statistical_tb_samples']==1
    assert result['mcs']==11 and result['capacity_bytes']>0


def test_fast_low_se_alias_preserves_physical_format_and_tb_size(monkeypatch):
    model=FastAbstraction('cpu',.1)
    monkeypatch.setattr(model.real,'waveform',lambda *a:pytest.fail('Equivalent format must use SYS'))
    result=model.evaluate(np.ones(324)*100,'qam64_low_se',16,7,1,'u',harq_context=context())
    assert result['mcs']==16 and result['ack'] and not result['actual_decode']
    wave=model.waveform(3,16,324);table,mcs=model.aliases[(3,16)]
    other=model.waveform(table,mcs,324)
    assert wave.q==other.q and wave.rate==other.rate and wave.bits==other.bits


def test_fast_missing_table_falls_back_and_reuses_the_original_codeword():
    model=FastAbstraction('cpu',.1)
    first=model.evaluate(np.ones(324)*.01,'qam64',1,501,1,'u',harq_context=context())
    second=model.evaluate(np.ones(324)*100,'qam64',1,501,2,'u',harq_context=context(True))
    assert first['actual_decode'] and second['actual_decode'] and second['ack']
    assert first['harq_reference']['codeword_id']==second['harq_reference']['codeword_id']
    assert model.decode_trials==2 and model.model_statistics()['statistical_tb_samples']==0
    assert model.model_statistics()['fallback_reasons']


def test_fast_rejects_incompatible_retransmission_and_soft_combining(monkeypatch):
    with pytest.raises(ValueError,match='soft combining'):FastAbstraction('cpu',.1,{'harq_combining':'chase'})
    model=FastAbstraction('cpu',.1)
    monkeypatch.setattr(model,'probability',lambda *a:(1.,.1,None))
    model.evaluate(np.ones(324),'qam64',11,7,1,'u',harq_context=context())
    with pytest.raises(ValueError,match='contract mismatch'):
        model.evaluate(np.ones(336),'qam64',11,7,2,'u',harq_context=context(True))
    monkeypatch.setattr(model,'probability',lambda *a:(None,None,'unavailable'))
    with pytest.raises(ValueError,match='coverage lost'):
        model.evaluate(np.ones(324),'qam64',11,7,3,'u',harq_context=context(True))


def test_sampled_feedback_is_counted_without_claiming_actual_decoding():
    stats=LinkStatistics()
    rows=[dict(ue_id='u',actual_decode=False,feedback_available=True,ack=False,is_retx=False,resource_mode='su'),
          dict(ue_id='v',actual_decode=True,ack=True,is_retx=False,resource_mode='su')]
    stats.observe(1,rows)
    result=stats.snapshot()['network']
    assert result['first_tx_attempts']==2 and result['first_tx_bler']==.5
    assert result['transmission_attempts']==2 and result['sampled_attempts']==1
    assert result['actual_decode_attempts']==1 and result['actual_nacks']==0
    assert has_tb_feedback(dict(ack_source='official_sys_default_bernoulli',actual_decode=False))
    assert not has_tb_feedback(dict(ack_source='no_path',scheduled=True))


def test_live_command_client_remains_importable_without_scientific_packages():
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,'-S','-c',
        'from experiments.live_xapp_test import submit, atomic_json'],cwd=root,check=True)


@pytest.mark.parametrize('model,feedback',[('full','nr_direct'),('fast','fast')])
def test_cli_selects_model_and_duration_without_starting_a_simulation(monkeypatch,capsys,model,feedback):
    from experiments.run_rt_sys import parse_args,run_case
    monkeypatch.setattr('sys.argv',['run_rt_sys','--dry-run','--device','cpu','--model',model,'--duration-s','10'])
    run_case(parse_args());result=json.loads(capsys.readouterr().out)
    assert result['model']==model and result['phy_config']['rt_sys']['feedback_mode']==feedback
    assert result['manifest']['total_slots']==20000 and result['automatic_html']


def test_duration_and_legacy_mode_arguments_are_unambiguous(monkeypatch):
    from experiments.run_rt_sys import parse_args,run_case
    for arguments in [('--slots','40','--duration-s','1'),('--model','fast','--feedback-mode','nr_direct')]:
        monkeypatch.setattr('sys.argv',['run_rt_sys',*arguments])
        with pytest.raises(SystemExit):parse_args()
    monkeypatch.setattr('sys.argv',['run_rt_sys','--dry-run','--duration-s','.0001'])
    with pytest.raises(ValueError,match='whole time slots'):run_case(parse_args())

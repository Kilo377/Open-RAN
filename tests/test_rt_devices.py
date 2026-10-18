"""The RT backend may run on CPU while PHY/SYS remain on CUDA."""
import json
from pathlib import Path
import pytest
from experiments.run_rt_sys import parse_args, run_case


@pytest.mark.parametrize('rt_device', [None, 'cpu'])
def test_dry_run_records_separate_rt_and_phy_devices(monkeypatch, capsys, rt_device):
    argv=['run_rt_sys', '--dry-run', '--device', 'cuda', '--slots', '2']
    if rt_device:argv+=['--rt-device',rt_device]
    monkeypatch.setattr('sys.argv',argv)
    run_case(parse_args())
    result=json.loads(capsys.readouterr().out)
    config=result['phy_config']['rt_sys']
    assert config['device']=='cuda'
    assert config['rt_device']==(rt_device or 'cuda')
    assert ('_rtcpu_phycuda' in Path(result['output']).name)==(rt_device=='cpu')

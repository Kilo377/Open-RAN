from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from experiments.run_rt_sys import configure_mec_stage_metrics


def case(enabled=True):
    return dict(phy=dict(rt_sys=dict(ue_services=dict(enabled=enabled,mec=dict(enabled=enabled)))))


def test_unspecified_diagnostics_preserve_the_complete_existing_config():
    cfg=case();before=deepcopy(cfg)
    configure_mec_stage_metrics(cfg,NS())
    assert cfg==before


def test_explicit_enable_and_disable_reach_the_actual_service_configuration():
    cfg=case()
    configure_mec_stage_metrics(cfg,NS(mec_stage_metrics=True))
    assert cfg['phy']['rt_sys']['ue_services']['stage_metrics'] is True
    configure_mec_stage_metrics(cfg,NS(mec_stage_metrics=False))
    assert cfg['phy']['rt_sys']['ue_services']['stage_metrics'] is False


@pytest.mark.parametrize('value',[True,'true',1])
def test_bad_or_unavailable_diagnostic_request_is_atomic(value):
    cfg=case(False);before=deepcopy(cfg)
    with pytest.raises(ValueError):configure_mec_stage_metrics(cfg,NS(mec_stage_metrics=value))
    assert cfg==before

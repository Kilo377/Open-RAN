from types import SimpleNamespace
from unittest.mock import patch

import pytest

from phy.rt_sys.scheduler import RtSysScheduler
from ran.scheduler import SimpleScheduler


@pytest.mark.parametrize('target,retx,latency,expected', [
    (26, True, 100, 20), (18, False, 10, 16), (0, False, 100, 3),
])
def test_rt_target_respects_cap_and_sys_calibration(target, retx, latency, expected):
    scheduler = RtSysScheduler()
    user = SimpleNamespace(latency_budget_ms=latency)
    with patch.object(SimpleScheduler, '_derive_target_mcs', return_value=target):
        actual = scheduler._derive_target_mcs(user, is_retx=retx,
            scheduler_mode='slice_aware', allocated_prbs=12, allocated_symbols=14)
    assert actual == expected
    assert actual <= scheduler._derive_mcs_cap(user, retx, actual)

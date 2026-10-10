"""Existing MAC policy with the MCS tables calibrated by SYS default EESM."""
from ran.scheduler import SimpleScheduler


class RtSysScheduler(SimpleScheduler):
    def _derive_target_mcs(self, ue, **kwargs):
        target = super()._derive_target_mcs(ue, **kwargs)
        cap = super()._derive_mcs_cap(ue, kwargs['is_retx'], target)
        # The legacy MAC can emit target > cap for retransmissions or after
        # slice-aware boosts. Reconcile before constructing the grant/TB.
        # SYS table 1 starts at MCS 3; table 2 is selected only at high MCS.
        return max(3, min(target, cap if cap is not None else 27))

    def _derive_mcs_table_id(self, *args, **kwargs):
        selected = super()._derive_mcs_table_id(*args, **kwargs)
        # Default SYS EESM beta parameters cover NR tables 1 and 2 only.
        # URLLC/retransmission grants use table 1; never fabricate table-3 beta.
        return 'qam64' if selected == 'qam64_low_se' else selected

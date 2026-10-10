"""Existing MAC policy with the MCS tables calibrated by SYS default EESM."""
from ran.scheduler import SimpleScheduler


class RtSysScheduler(SimpleScheduler):
    def _derive_mcs_table_id(self, *args, **kwargs):
        selected = super()._derive_mcs_table_id(*args, **kwargs)
        # Default SYS EESM beta parameters cover NR tables 1 and 2 only.
        # URLLC/retransmission grants use table 1; never fabricate table-3 beta.
        return 'qam64' if selected == 'qam64_low_se' else selected

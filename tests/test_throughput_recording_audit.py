"""Arrival evidence survives the simulator's pre-recording shallow snapshot."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.rt_diagnostics import compact_trace
from experiments.run_rt_sys import attach_recorded_arrivals
from experiments.run_throughput_mlp_study import POPULATION, arrival_audit


class TestThroughputRecordingAudit(unittest.TestCase):
    def test_actual_shallow_snapshot_and_both_recording_modes_preserve_arrivals(self):
        live = dict(slot_id=1, ues=[], precoders={'diagnostic_only': True})
        snapshot = dict(live)  # Actual Simulator.step ordering.
        live['traffic_arrivals'] = {u: dict(exogenous_bytes=3 if u=='ue_0' else 0,
                                           mec_result_bytes=2 if u=='ue_0' else 0) for u in POPULATION}
        self.assertNotIn('traffic_arrivals', snapshot)
        before = deepcopy(live)
        attach_recorded_arrivals(snapshot, live)
        self.assertEqual(live, before)  # Recording does not alter runtime data.
        users = {u: dict(offered_bytes=5 if u=='ue_0' else 0) for u in POPULATION}
        for mode in ('audit', 'experiment'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                row = snapshot if mode=='audit' else compact_trace(snapshot)
                self.assertEqual(row['traffic_arrivals'], live['traffic_arrivals'])
                if mode=='experiment':self.assertNotIn('precoders', row)
                folder = Path(temporary)
                (folder/'rt_sys_trace.jsonl').write_text(json.dumps(row)+'\n')
                result = arrival_audit(folder, users)
                self.assertEqual(result['exogenous_bytes']['ue_0'], 3)
                self.assertEqual(result['endogenous_mec_result_bytes']['ue_0'], 2)
                self.assertEqual(result['trace_slots'], 1)


if __name__=='__main__':unittest.main()

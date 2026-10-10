import unittest
from types import SimpleNamespace
from typing import Any, cast

from core.simulator import Simulator


class TestNumerologyDerivation(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = cast(Any, Simulator.__new__(Simulator))
        self.simulator.config = SimpleNamespace(slot_duration_ms=1.0)
        self.simulator.scheduler = SimpleNamespace(
            total_prbs_per_slot=0,
            total_symbols_per_slot=0,
        )

    def test_numerology_scales_with_bandwidth_and_scs(self) -> None:
        base = self.simulator._derive_numerology_config(
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=20e6,
            subcarrier_spacing_hz=30e3,
            cyclic_prefix_type="normal",
        )
        wider_bw = self.simulator._derive_numerology_config(
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=40e6,
            subcarrier_spacing_hz=30e3,
            cyclic_prefix_type="normal",
        )
        lower_scs = self.simulator._derive_numerology_config(
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=20e6,
            subcarrier_spacing_hz=15e3,
            cyclic_prefix_type="normal",
        )

        self.assertGreater(wider_bw["n_prbs_total"], base["n_prbs_total"])
        self.assertGreater(lower_scs["n_prbs_total"], base["n_prbs_total"])
        self.assertGreater(base["n_ofdm_symbols_per_slot"], lower_scs["n_ofdm_symbols_per_slot"])
        self.assertEqual(base["n_prbs_total"], 51)
        self.assertEqual(wider_bw["n_prbs_total"], 106)
        self.assertEqual(lower_scs["n_prbs_total"], 106)
        self.assertEqual(base["nr_frequency_range"], "FR1")
        self.assertEqual(base["nr_numerology_index"], 1)
        self.assertGreaterEqual(base["fft_size"], int(base["n_prbs_total"]) * 12)
        self.assertEqual(sum(base["usable_subcarrier_mask"]), int(base["n_prbs_total"]) * 12)

    def test_extended_cp_reduces_symbols_and_syncs_scheduler_budget(self) -> None:
        normal_cp = self.simulator._derive_numerology_config(
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=20e6,
            subcarrier_spacing_hz=30e3,
            cyclic_prefix_type="normal",
        )
        extended_cp = self.simulator._derive_numerology_config(
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=20e6,
            subcarrier_spacing_hz=30e3,
            cyclic_prefix_type="extended",
        )

        self.assertLess(extended_cp["n_ofdm_symbols_per_slot"], normal_cp["n_ofdm_symbols_per_slot"])

        self.simulator._sync_scheduler_budget_with_numerology(normal_cp)
        self.assertEqual(
            self.simulator.scheduler.total_prbs_per_slot,
            int(normal_cp["n_prbs_total"]),
        )
        self.assertEqual(
            self.simulator.scheduler.total_symbols_per_slot,
            int(normal_cp["n_ofdm_symbols_per_slot"]),
        )


if __name__ == "__main__":
    unittest.main()

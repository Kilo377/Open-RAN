import unittest

from phy.mcs_profiles import (
    estimate_mcs_required_sinr_db,
    get_mcs_profile,
    select_mcs_index_from_sinr,
)


class TestMcsProfiles(unittest.TestCase):
    def test_qam64_table_matches_3gpp_style_shape(self) -> None:
        low = get_mcs_profile(0, "qam64")
        mid = get_mcs_profile(10, "qam64")
        high = get_mcs_profile(23, "qam64")

        self.assertEqual(low.modulation_name, "qpsk")
        self.assertAlmostEqual(low.code_rate, 120 / 1024.0, places=5)
        self.assertEqual(mid.modulation_name, "16qam")
        self.assertEqual(high.modulation_name, "64qam")

    def test_sinr_threshold_and_selected_mcs_are_monotonic(self) -> None:
        threshold_mid = estimate_mcs_required_sinr_db(10, "qam64")
        threshold_high = estimate_mcs_required_sinr_db(23, "qam64")
        low_index = select_mcs_index_from_sinr(-1.0, mcs_table_id="qam64", max_mcs_index=27)
        mid_index = select_mcs_index_from_sinr(8.0, mcs_table_id="qam64", max_mcs_index=27)
        high_index = select_mcs_index_from_sinr(18.0, mcs_table_id="qam64", max_mcs_index=27)

        self.assertGreater(threshold_high, threshold_mid)
        self.assertGreater(mid_index, low_index)
        self.assertGreater(high_index, mid_index)


if __name__ == "__main__":
    unittest.main()

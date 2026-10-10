import unittest

from phy.nr_reference import (
    derive_nr_frequency_range,
    derive_nr_numerology_index,
    lookup_nr_max_prbs,
)


class TestNrReference(unittest.TestCase):
    def test_frequency_range_and_numerology_index(self) -> None:
        self.assertEqual(derive_nr_frequency_range(3.5e9), "FR1")
        self.assertEqual(derive_nr_frequency_range(28.0e9), "FR2")
        self.assertEqual(derive_nr_numerology_index(15e3), 0)
        self.assertEqual(derive_nr_numerology_index(30e3), 1)
        self.assertEqual(derive_nr_numerology_index(120e3), 3)

    def test_common_3gpp_style_nrb_points(self) -> None:
        self.assertEqual(
            lookup_nr_max_prbs(
                bandwidth_hz=20e6,
                subcarrier_spacing_hz=30e3,
                carrier_frequency_hz=3.5e9,
            )[0],
            51,
        )
        self.assertEqual(
            lookup_nr_max_prbs(
                bandwidth_hz=40e6,
                subcarrier_spacing_hz=30e3,
                carrier_frequency_hz=3.5e9,
            )[0],
            106,
        )
        self.assertEqual(
            lookup_nr_max_prbs(
                bandwidth_hz=100e6,
                subcarrier_spacing_hz=120e3,
                carrier_frequency_hz=28.0e9,
            )[0],
            66,
        )


if __name__ == "__main__":
    unittest.main()

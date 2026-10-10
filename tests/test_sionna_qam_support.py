import unittest

from phy.mcs_profiles import get_mcs_profile
from phy.sionna_engine import SionnaEngine


class TestSionnaQamSupport(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SionnaEngine({})
        self.numpy = self.engine._get_numpy()

    def test_square_qam_round_trip_supports_high_order_modulation(self) -> None:
        test_vectors = {
            "64qam": self.numpy.array(
                [
                    0, 0, 0, 0, 0, 0,
                    0, 1, 1, 1, 0, 1,
                    1, 1, 0, 0, 1, 1,
                    1, 0, 1, 1, 1, 0,
                ],
                dtype=self.numpy.int32,
            ),
            "256qam": self.numpy.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 1, 0, 1, 0, 1, 1,
                    1, 1, 0, 1, 0, 1, 1, 0,
                    1, 0, 1, 1, 1, 1, 0, 1,
                ],
                dtype=self.numpy.int32,
            ),
        }

        for modulation, bits in test_vectors.items():
            with self.subTest(modulation=modulation):
                symbols = self.engine._modulate(bits, modulation)
                recovered_bits = self.engine._demodulate(symbols, modulation)
                self.assertListEqual(recovered_bits.tolist(), bits.tolist())

    def test_qam64_table_uses_64qam_for_high_mcs(self) -> None:
        high_se_profile = get_mcs_profile(23, "qam64")

        self.assertEqual(high_se_profile.modulation_name, "64qam")
        self.assertEqual(high_se_profile.bits_per_symbol, 6)

    def test_transport_block_segmentation_respects_sionna_codeblock_limit(self) -> None:
        engine = SionnaEngine(
            {
                "num_bits_per_block": 65536,
                "max_sionna_codeblock_info_bits": 8448,
            }
        )

        codeblock_sizes = engine._split_transport_block_info_bits(65536)

        self.assertGreater(len(codeblock_sizes), 1)
        self.assertGreaterEqual(sum(codeblock_sizes), 65536)
        self.assertTrue(all(codeblock_size <= 8448 for codeblock_size in codeblock_sizes))
        self.assertTrue(all(codeblock_size % 32 == 0 for codeblock_size in codeblock_sizes))


if __name__ == "__main__":
    unittest.main()

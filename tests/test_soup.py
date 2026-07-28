from __future__ import annotations

import unittest
from unittest import mock

from bff.soup import Soup, test_selfrep


class SoupValidationTests(unittest.TestCase):
    def test_rejects_population_sizes_unsafe_for_epoch_pairing(self):
        for population in (0, 1, 3):
            with self.subTest(population=population), self.assertRaises(ValueError):
                Soup(population)


class SelfReplicationDiagnosticTests(unittest.TestCase):
    def test_rejects_63_byte_candidate_before_calling_c(self):
        lib = mock.Mock()

        with self.assertRaisesRegex(ValueError, "exactly 64 bytes"):
            test_selfrep(b"A" * 63, lib, trials=1)

        lib.evaluate.assert_not_called()

    def test_accepts_64_byte_candidate(self):
        lib = mock.Mock()
        lib.evaluate.return_value = 0

        exact_rate, median_run = test_selfrep(b"A" * 64, lib, trials=1)

        lib.evaluate.assert_called_once()
        self.assertEqual(exact_rate, 0.0)
        self.assertGreaterEqual(median_run, 0.0)

    def test_rejects_65_byte_candidate_before_calling_c(self):
        lib = mock.Mock()

        with self.assertRaisesRegex(ValueError, "exactly 64 bytes"):
            test_selfrep(b"A" * 65, lib, trials=1)

        lib.evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()

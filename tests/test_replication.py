from __future__ import annotations

import unittest
from unittest import mock

from bff.bff import PAPER_REPLICATOR
from bff.replication import (
    CandidateSeed,
    ReplicationEvidence,
    ReplicationTracker,
    count_exact_tape,
    count_marker_carriers,
    discover_candidates,
    verify_candidate,
)
from bff.soup import Soup


class FunctionalVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = Soup(2, seed=1).lib

    def test_paper_replicator_passes_both_multigeneration_assays(self):
        evidence = verify_candidate(PAPER_REPLICATOR, self.lib)

        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence.score, 64)
        self.assertEqual(evidence.confirmation_score, 64)
        self.assertEqual(evidence.contexts, 13)
        self.assertEqual(evidence.generations, 5)
        self.assertEqual(evidence.marker_successes, 13)
        self.assertEqual(evidence.confirmation_successes, 13)
        self.assertEqual(len(evidence.marker), 16)
        self.assertIn(evidence.marker, PAPER_REPLICATOR)

    def test_inert_and_arbitrary_tapes_are_not_verified(self):
        self.assertIsNone(verify_candidate(bytes(64), self.lib))
        self.assertIsNone(verify_candidate(bytes(range(64)), self.lib))

    def test_invalid_assay_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_candidate(PAPER_REPLICATOR, self.lib, generations=0)
        with self.assertRaises(ValueError):
            verify_candidate(
                PAPER_REPLICATOR,
                self.lib,
                marker_min_length=17,
                marker_max_length=16,
            )
        with self.assertRaises(ValueError):
            verify_candidate(PAPER_REPLICATOR, self.lib, seed=7, confirmation_seed=7)
        with self.assertRaises(ValueError):
            verify_candidate(
                PAPER_REPLICATOR,
                self.lib,
                seed=0,
                confirmation_seed=1 << 64,
            )


class CandidateDiscoveryTests(unittest.TestCase):
    def test_marker_carriers_are_counted_once_per_tape(self):
        repeated = b"ABCD" * 16
        raw = repeated * 3
        self.assertEqual(count_marker_carriers(raw, b"ABCD"), 3)
        self.assertEqual(count_exact_tape(raw, repeated), 3)

    def test_cross_tape_marker_and_unaligned_exact_match_are_excluded(self):
        left = b"A" * 63 + b"X"
        right = b"Y" + b"B" * 63
        raw = left + right
        self.assertEqual(count_marker_carriers(raw, b"XY"), 0)
        self.assertEqual(count_exact_tape(raw, raw[1:65]), 0)

    def test_carrier_counters_reject_partial_tapes(self):
        for counter, needle in (
            (count_marker_carriers, b"A"),
            (count_exact_tape, b"A" * 64),
        ):
            with self.subTest(counter=counter), self.assertRaises(ValueError):
                counter(b"partial", needle)

    def test_discovery_uses_carrier_count_and_diversifies_exact_tapes(self):
        first = b"ABCD" * 16
        second = b"WXYZ" * 16
        seeds = discover_candidates(
            first * 3 + second * 2,
            marker_length=4,
            top_markers=8,
        )

        self.assertEqual(len(seeds), 2)
        self.assertEqual(
            sorted(seed.sample_carriers for seed in seeds),
            [2, 3],
        )
        self.assertTrue(all(len(seed.representatives) == 1 for seed in seeds))

    def test_discovery_rotates_past_excluded_negative_representatives(self):
        common = b"Q" * 16
        tapes = [common + bytes([value]) * 48 for value in range(1, 5)]
        first = discover_candidates(
            b"".join(tapes),
            top_markers=1,
            representatives_per_marker=2,
        )
        self.assertEqual(len(first), 1)

        second = discover_candidates(
            b"".join(tapes),
            top_markers=1,
            representatives_per_marker=2,
            excluded_representatives=set(first[0].representatives),
        )
        self.assertEqual(len(second), 1)
        self.assertTrue(
            set(first[0].representatives).isdisjoint(second[0].representatives)
        )


class ReplicationTrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = Soup(2, seed=2).lib

    def test_verified_marker_prevalence_and_growth_are_tracked(self):
        background = bytes(range(64))
        tracker = ReplicationTracker()
        first = tracker.observe(
            PAPER_REPLICATOR * 2 + background * 2,
            self.lib,
            epoch=100,
            discover=True,
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].carrier_count, 2)
        self.assertEqual(first[0].carrier_share, 0.5)
        self.assertEqual(first[0].exact_count, 2)
        self.assertEqual(first[0].growth_per_100_epochs, 0.0)

        second = tracker.observe(
            PAPER_REPLICATOR * 4,
            self.lib,
            epoch=200,
            discover=False,
        )
        self.assertEqual(second[0].carrier_count, 4)
        self.assertEqual(second[0].carrier_share, 1.0)
        self.assertAlmostEqual(second[0].growth_per_100_epochs, 0.5)
        self.assertEqual(second[0].peak_share, 1.0)

    def test_capped_cached_markers_can_leave_and_reenter_without_reassay(self):
        def evidence(byte: int, marker_length: int = 16) -> ReplicationEvidence:
            representative = bytes([byte]) * 64
            return ReplicationEvidence(
                representative=representative,
                score=64,
                confirmation_score=64,
                marker=bytes([byte]) * marker_length,
                marker_successes=13,
                confirmation_successes=13,
                contexts=13,
                generations=5,
                left_support=(True,) * 64,
                right_support=(True,) * 64,
            )

        a, b, c = evidence(65), evidence(66), evidence(67)
        tracker = ReplicationTracker(max_markers=2)
        for item in (a, b, c):
            tracker._tested[item.representative] = None
            tracker._verified[item.representative] = (item, 10)

        tracker._register(a, epoch=10, current_share=0.6, verified_epoch=10)
        tracker._register(b, epoch=10, current_share=0.5, verified_epoch=10)
        tracker._register(
            evidence(65, marker_length=8),
            epoch=10,
            current_share=0.9,
            verified_epoch=10,
        )
        self.assertEqual(set(tracker._markers), {a.marker, b.marker})
        tracker._register(c, epoch=10, current_share=0.4, verified_epoch=10)
        self.assertEqual(set(tracker._markers), {a.marker, b.marker})

        tracker._register(c, epoch=20, current_share=0.7, verified_epoch=10)
        self.assertEqual(set(tracker._markers), {a.marker, c.marker})
        self.assertEqual(tracker._markers[c.marker].first_verified_epoch, 10)

        seeds = (CandidateSeed(
            marker=b.marker,
            sample_carriers=4,
            sample_size=4,
            representatives=(b.representative,),
        ),)
        with (
            mock.patch("bff.replication.discover_candidates", return_value=seeds),
            mock.patch(
                "bff.replication.verify_candidate",
                side_effect=AssertionError("cached positive was re-assayed"),
            ) as verifier,
        ):
            tracker._discover(b.representative * 4, self.lib, epoch=30)
        verifier.assert_not_called()

        restored = tracker._markers[b.marker]
        self.assertEqual(restored.first_verified_epoch, 10)
        self.assertEqual(restored.peak_share, 1.0)
        self.assertEqual(set(tracker._markers), {b.marker, c.marker})

    def test_nested_marker_replaces_an_extinct_incumbent(self):
        def evidence(marker: bytes) -> ReplicationEvidence:
            return ReplicationEvidence(
                representative=b"A" * 64,
                score=64,
                confirmation_score=64,
                marker=marker,
                marker_successes=13,
                confirmation_successes=13,
                contexts=13,
                generations=5,
                left_support=(True,) * 64,
                right_support=(True,) * 64,
            )

        long_marker = b"A" * 16
        short_marker = b"A" * 8
        tracker = ReplicationTracker(max_markers=2)
        tracker._register(
            evidence(long_marker),
            epoch=10,
            current_share=0.5,
            verified_epoch=10,
        )
        tracker._markers[long_marker].last_share = 0.0
        tracker._register(
            evidence(short_marker),
            epoch=20,
            current_share=0.7,
            verified_epoch=20,
        )

        self.assertNotIn(long_marker, tracker._markers)
        self.assertIn(short_marker, tracker._markers)
        self.assertEqual(tracker._archives[long_marker], (10, 0.5))


if __name__ == "__main__":
    unittest.main()

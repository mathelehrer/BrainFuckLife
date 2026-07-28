from __future__ import annotations

import dataclasses
import io
import math
import os
import pathlib
import queue
import tempfile
import time
import unittest

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(pathlib.Path(tempfile.gettempdir()) / "brainfucklife-matplotlib"),
)

from bff.live import (  # noqa: E402 - backend environment precedes imports
    Dashboard,
    RunConfig,
    SimulationWorker,
    _put_latest,
    classify_tape,
    collect_snapshot,
    marker_offsets,
    tape_glyph,
    tape_id,
)
from bff.bff import PAPER_REPLICATOR  # noqa: E402
from bff.replication import (  # noqa: E402
    ReplicationHistoryPoint,
    ReplicationTracker,
    VerifiedMarkerSummary,
)
from bff.soup import Soup  # noqa: E402


class FakeSoup:
    def __init__(self, tapes: list[bytes], epoch: int = 7):
        self.n = len(tapes)
        self.epoch = epoch
        self.buf = np.frombuffer(b"".join(tapes), dtype=np.uint8).copy()


class RunConfigTests(unittest.TestCase):
    def test_accepts_supported_configuration(self):
        RunConfig(n=8, seed=0, top_k=8).validate()
        RunConfig(n=8, seed=(1 << 64) - 1, top_k=1).validate()

    def test_rejects_values_that_are_unsafe_for_the_c_epoch_loop(self):
        invalid = (
            RunConfig(n=0),
            RunConfig(n=1),
            RunConfig(n=6),
            RunConfig(seed=-1),
            RunConfig(seed=1 << 64),
            RunConfig(mutation_prob=-0.1),
            RunConfig(mutation_prob=1.1),
            RunConfig(epochs=-1),
            RunConfig(top_k=0),
            RunConfig(top_k=17),
            RunConfig(refresh_s=0),
            RunConfig(verify_interval_s=0),
            RunConfig(replication_threshold=47),
            RunConfig(replication_threshold=65),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()


class SnapshotTests(unittest.TestCase):
    def test_tape_categories_and_glyphs_are_redundant(self):
        tape = bytes([0, ord("<"), ord("+"), ord("."), ord("["), 255])
        self.assertEqual(classify_tape(tape).tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual([tape_glyph(byte) for byte in tape], ["0", "<", "+", ".", "[", ""])

    def test_marker_offsets_include_overlapping_matches(self):
        self.assertEqual(marker_offsets(b"AAAA", b"AA"), (0, 1, 2))

    def test_raw_tape_identity_is_stable_and_distinguishes_inert_bytes(self):
        a = b" " * 64
        b = b"\xff" * 64
        self.assertEqual(tape_id(a), tape_id(a))
        self.assertNotEqual(tape_id(a), tape_id(b))

    def test_snapshot_counts_exact_tapes_with_deterministic_ties(self):
        a, b, c = b"A" * 64, b"B" * 64, b"C" * 64
        soup = FakeSoup([b, a, c, b, a])
        snapshot = collect_snapshot(
            soup,
            top_k=3,
            elapsed_s=1.0,
            epochs_per_s=20.0,
            ops_per_interaction=3.0,
        )
        self.assertEqual(snapshot.population, 5)
        self.assertEqual(snapshot.unique_tapes, 3)
        self.assertEqual([item.tape for item in snapshot.top], [a, b, c])
        self.assertEqual([item.count for item in snapshot.top], [2, 2, 1])
        self.assertAlmostEqual(sum(item.share for item in snapshot.top), 1.0)

    def test_latest_value_queue_drops_stale_snapshots(self):
        target: queue.Queue = queue.Queue(maxsize=1)
        soup = FakeSoup([b"A" * 64, b"B" * 64])
        first = collect_snapshot(
            soup,
            top_k=2,
            elapsed_s=0,
            epochs_per_s=0,
            ops_per_interaction=0,
        )
        soup.epoch += 1
        second = collect_snapshot(
            soup,
            top_k=2,
            elapsed_s=1,
            epochs_per_s=1,
            ops_per_interaction=1,
        )
        _put_latest(target, first)
        _put_latest(target, second)
        self.assertIs(target.get_nowait(), second)
        with self.assertRaises(queue.Empty):
            target.get_nowait()

    def test_observation_does_not_change_seeded_trajectory(self):
        baseline = Soup(8, seed=23)
        baseline_ops = [baseline.step() for _ in range(5)]
        baseline_state = baseline.buf.tobytes()

        observed = Soup(8, seed=23)
        observed_ops = []
        for _ in range(5):
            observed_ops.append(observed.step())
            collect_snapshot(
                observed,
                top_k=8,
                elapsed_s=0,
                epochs_per_s=0,
                ops_per_interaction=0,
            )

        self.assertEqual(observed_ops, baseline_ops)
        self.assertEqual(observed.buf.tobytes(), baseline_state)

    def test_functional_verification_does_not_change_soup_rng(self):
        raw = PAPER_REPLICATOR * 8
        baseline = Soup(8, seed=41)
        baseline.buf[:] = np.frombuffer(raw, dtype=np.uint8)
        baseline_ops = baseline.step()
        baseline_state = baseline.buf.tobytes()

        observed = Soup(8, seed=41)
        observed.buf[:] = np.frombuffer(raw, dtype=np.uint8)
        collect_snapshot(
            observed,
            top_k=8,
            elapsed_s=0,
            epochs_per_s=0,
            ops_per_interaction=0,
            replication_tracker=ReplicationTracker(),
            discover_replicators=True,
            verification_enabled=True,
        )
        observed_ops = observed.step()

        self.assertEqual(observed_ops, baseline_ops)
        self.assertEqual(observed.buf.tobytes(), baseline_state)


class WorkerTests(unittest.TestCase):
    def _run_to_completion(self, config: RunConfig):
        worker = SimulationWorker(config)
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive)
        self.assertIsNone(worker.failure)
        return worker.snapshots.get_nowait()

    def test_finite_worker_publishes_final_snapshot(self):
        snapshot = self._run_to_completion(
            RunConfig(n=8, seed=5, epochs=3, top_k=8, refresh_s=0.001, validate_core=False)
        )
        self.assertTrue(snapshot.final)
        self.assertEqual(snapshot.final_reason, "completed")
        self.assertEqual(snapshot.epoch, 3)
        self.assertEqual(snapshot.population, 8)
        self.assertEqual(sum(item.count for item in snapshot.top), 8)

    def test_zero_epoch_run_is_terminal(self):
        snapshot = self._run_to_completion(
            RunConfig(n=8, seed=5, epochs=0, top_k=8, validate_core=False)
        )
        self.assertTrue(snapshot.final)
        self.assertEqual(snapshot.final_reason, "completed")
        self.assertEqual(snapshot.epoch, 0)

    def test_same_seed_produces_same_final_population_summary(self):
        config = RunConfig(n=8, seed=99, epochs=5, top_k=8, refresh_s=0.001, validate_core=False)
        first = self._run_to_completion(config)
        second = self._run_to_completion(config)
        self.assertEqual(
            [(item.tape, item.count) for item in first.top],
            [(item.tape, item.count) for item in second.top],
        )
        self.assertAlmostEqual(first.complexity, second.complexity)

    def test_stop_wakes_a_paused_worker(self):
        worker = SimulationWorker(
            RunConfig(n=8, seed=5, epochs=1_000_000, top_k=8, refresh_s=60, validate_core=False)
        )
        worker.pause()
        worker.start()
        time.sleep(0.02)
        worker.stop()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive)
        self.assertIsNone(worker.failure)
        snapshot = worker.snapshots.get_nowait()
        self.assertTrue(snapshot.final)
        self.assertEqual(snapshot.final_reason, "stopped")

    def test_verification_cadence_is_independent_of_gui_refresh(self):
        snapshot = self._run_to_completion(RunConfig(
            n=8,
            seed=5,
            epochs=20,
            top_k=8,
            refresh_s=60,
            verify_interval_s=1e-6,
            validate_core=False,
        ))
        self.assertGreater(snapshot.replication_scans, 1)


class DashboardTests(unittest.TestCase):
    @staticmethod
    def _marker(*, share: float, count: int, growth: float = 0.0):
        marker = PAPER_REPLICATOR[-16:]
        return VerifiedMarkerSummary(
            marker=marker,
            marker_id="marker01",
            representative=PAPER_REPLICATOR,
            representative_id="tape0001",
            score=64,
            confirmation_score=64,
            marker_successes=13,
            contexts=13,
            generations=5,
            carrier_count=count,
            carrier_share=share,
            exact_count=count,
            growth_per_100_epochs=growth,
            first_verified_epoch=10,
            peak_share=share,
        )

    def test_history_follows_raw_identity_across_a_rank_change(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        a, b = b"A" * 64, b"B" * 64
        fillers = [bytes([value]) * 64 for value in range(1, 6)]
        first = FakeSoup([a, a, a, b, b, *fillers[:3]], epoch=1)
        second = FakeSoup([b, b, b, b, a, a, *fillers[:2]], epoch=2)

        figure = plt.figure(figsize=(8, 6), layout="constrained")
        dashboard = Dashboard(figure, top_k=1, max_history=3)
        dashboard.update(collect_snapshot(
            first,
            top_k=1,
            elapsed_s=1,
            epochs_per_s=1,
            ops_per_interaction=1,
        ))
        dashboard.update(collect_snapshot(
            second,
            top_k=1,
            elapsed_s=2,
            epochs_per_s=1,
            ops_per_interaction=1,
        ))
        plt.close(figure)

        self.assertAlmostEqual(dashboard.identity_history[a][0], 3 / 8)
        self.assertTrue(math.isnan(dashboard.identity_history[a][1]))
        self.assertTrue(math.isnan(dashboard.identity_history[b][0]))
        self.assertAlmostEqual(dashboard.identity_history[b][1], 4 / 8)

    def test_dashboard_renders_to_png_with_agg(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        soup = FakeSoup([b"A" * 64, b"[" * 64, bytes(64), b"A" * 64])
        snapshot = collect_snapshot(
            soup,
            top_k=4,
            elapsed_s=1.0,
            epochs_per_s=10.0,
            ops_per_interaction=4.0,
            transition_epoch=7,
            final=True,
            final_reason="completed",
        )
        figure = plt.figure(figsize=(12, 8), layout="constrained")
        dashboard = Dashboard(figure, top_k=4, max_history=3)
        dashboard.update(snapshot)
        figure.canvas.draw()
        output = io.BytesIO()
        figure.savefig(output, format="png")
        plt.close(figure)
        self.assertGreater(len(output.getvalue()), 10_000)

    def test_verified_marker_history_renders_and_view_can_toggle(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        soup = FakeSoup([PAPER_REPLICATOR] * 2 + [bytes(range(64))] * 2, epoch=10)
        base = collect_snapshot(
            soup,
            top_k=4,
            elapsed_s=1,
            epochs_per_s=10,
            ops_per_interaction=4,
        )
        first = dataclasses.replace(
            base,
            replication_markers=(self._marker(share=0.5, count=2),),
            verification_enabled=True,
            replication_scans=1,
        )
        second = dataclasses.replace(
            first,
            epoch=20,
            replication_markers=(self._marker(share=0.75, count=3, growth=2.5),),
        )

        figure = plt.figure(figsize=(12, 8), layout="constrained")
        dashboard = Dashboard(figure, top_k=4, max_history=3)
        combined = dataclasses.replace(
            second,
            replication_history=(
                ReplicationHistoryPoint(
                    epoch=10,
                    markers=first.replication_markers,
                ),
                ReplicationHistoryPoint(
                    epoch=20,
                    markers=second.replication_markers,
                ),
            ),
        )
        dashboard.update(combined)
        figure.canvas.draw()

        marker = PAPER_REPLICATOR[-16:]
        self.assertEqual(list(dashboard.marker_history[marker]), [0.5, 0.75])
        self.assertGreater(len(dashboard.tape_ax.patches), 0)
        self.assertEqual(dashboard.toggle_view(), "exact")
        self.assertEqual(dashboard.toggle_view(), "verified")
        output = io.BytesIO()
        figure.savefig(output, format="png")
        plt.close(figure)
        self.assertGreater(len(output.getvalue()), 10_000)


if __name__ == "__main__":
    unittest.main()

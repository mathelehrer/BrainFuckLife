#!/usr/bin/env python3
"""Realtime visualization for the BFF primordial soup.

The simulation runs on a worker thread while Matplotlib remains on the main
thread.  The worker publishes immutable, bounded snapshots; rendering can drop
stale frames without changing the simulation's random-number stream.

Usage:
    python3 live.py --n 4096 --seed 4
    python3 live.py --n 4096 --seed 4 --top 8 --refresh 0.5
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import heapq
import queue
import sys
import threading
import time
from typing import Literal

import numpy as np

try:  # Support both ``python bff/live.py`` and ``python -m bff.live``.
    from .replication import (
        DEFAULT_SCORE_THRESHOLD,
        ReplicationHistoryPoint,
        ReplicationTracker,
        VerifiedMarkerSummary,
    )
    from .soup import (
        COMPRESSOR,
        DEFAULT_STEPS,
        Soup,
        TAPE,
        cross_check,
        high_order_entropy,
    )
except ImportError:  # pragma: no cover - exercised by the documented CLI
    from replication import (
        DEFAULT_SCORE_THRESHOLD,
        ReplicationHistoryPoint,
        ReplicationTracker,
        VerifiedMarkerSummary,
    )
    from soup import (
        COMPRESSOR,
        DEFAULT_STEPS,
        Soup,
        TAPE,
        cross_check,
        high_order_entropy,
    )


_OPS = b"<>{}-+.,[]"
_CATEGORY_NAMES = ("zero", "head move", "arithmetic", "copy", "loop", "inert")
_CATEGORY_COLORS = ("#1a1a19", "#2a78d6", "#eb6834",
                    "#1baf7a", "#4a3aa7", "#e8e7e1")
_CLASS_TABLE = np.full(256, 5, dtype=np.uint8)
_CLASS_TABLE[0] = 0
for _byte in b"<>{}":
    _CLASS_TABLE[_byte] = 1
for _byte in b"+-":
    _CLASS_TABLE[_byte] = 2
for _byte in b".,":
    _CLASS_TABLE[_byte] = 3
for _byte in b"[]":
    _CLASS_TABLE[_byte] = 4


@dataclasses.dataclass(frozen=True, slots=True)
class TapeSummary:
    """One exact 64-byte tape and its current population prevalence."""

    tape: bytes
    count: int
    share: float
    tape_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class Snapshot:
    """Immutable state transferred from the simulation to the GUI."""

    epoch: int
    elapsed_s: float
    epochs_per_s: float
    complexity: float
    ops_per_interaction: float
    population: int
    unique_tapes: int
    top: tuple[TapeSummary, ...]
    sampling_s: float
    replication_markers: tuple[VerifiedMarkerSummary, ...] = ()
    replication_history: tuple[ReplicationHistoryPoint, ...] = ()
    verification_enabled: bool = False
    replication_scans: int = 0
    transition_epoch: int | None = None
    final: bool = False
    final_reason: Literal["completed", "threshold", "stopped"] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class RunConfig:
    n: int = 4096
    seed: int = 4
    mutation_prob: float = 1 / 4096
    epochs: int = 20_000
    stepcount: int = DEFAULT_STEPS
    top_k: int = 8
    refresh_s: float = 0.5
    transition_threshold: float = 1.0
    stop_at: float = 0.0
    validate_core: bool = True
    verify_replication: bool = True
    verify_interval_s: float = 0.25
    replication_threshold: int = DEFAULT_SCORE_THRESHOLD

    def validate(self) -> None:
        if self.n < 2 or self.n & (self.n - 1):
            raise ValueError("n must be a power of two greater than one")
        if not 0 <= self.seed <= (1 << 64) - 1:
            raise ValueError("seed must fit in an unsigned 64-bit integer")
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if not 0.0 <= self.mutation_prob <= 1.0:
            raise ValueError("mutation probability must be between 0 and 1")
        if self.stepcount <= 0:
            raise ValueError("stepcount must be positive")
        if not 1 <= self.top_k <= 16:
            raise ValueError("top must be between 1 and 16")
        if self.refresh_s <= 0:
            raise ValueError("refresh must be positive")
        if self.transition_threshold <= 0:
            raise ValueError("transition threshold must be positive")
        if self.stop_at < 0:
            raise ValueError("stop-at must be non-negative")
        if self.verify_interval_s <= 0:
            raise ValueError("verify-every must be positive")
        if not DEFAULT_SCORE_THRESHOLD <= self.replication_threshold <= TAPE:
            raise ValueError("replication threshold must be between 48 and 64")


def tape_id(tape: bytes) -> str:
    """Short stable identity for raw bytes hidden by the semantic rendering."""
    return hashlib.blake2s(tape, digest_size=4, person=b"BFFLIVE").hexdigest()


def classify_tape(tape: bytes) -> np.ndarray:
    """Map raw bytes to zero/opcode-family/inert display categories."""
    raw = np.frombuffer(tape, dtype=np.uint8)
    return _CLASS_TABLE[raw]


def tape_glyph(byte: int) -> str:
    """Visible glyph for semantically meaningful bytes."""
    if byte == 0:
        return "0"
    return chr(byte) if byte in _OPS else ""


def marker_offsets(tape: bytes, marker: bytes) -> tuple[int, ...]:
    """Return every in-tape marker start, including overlapping matches."""
    offsets: list[int] = []
    start = 0
    while marker and start <= len(tape) - len(marker):
        offset = tape.find(marker, start)
        if offset < 0:
            break
        offsets.append(offset)
        start = offset + 1
    return tuple(offsets)


def collect_snapshot(
    soup: Soup,
    *,
    top_k: int,
    elapsed_s: float,
    epochs_per_s: float,
    ops_per_interaction: float,
    replication_tracker: ReplicationTracker | None = None,
    discover_replicators: bool = False,
    observe_replicators: bool = True,
    verification_enabled: bool = False,
    transition_epoch: int | None = None,
    final: bool = False,
    final_reason: Literal["completed", "threshold", "stopped"] | None = None,
) -> Snapshot:
    """Collect one internally consistent snapshot between complete epochs."""
    started = time.perf_counter()
    raw_soup = soup.buf.tobytes()
    counts = collections.Counter(
        raw_soup[offset:offset + TAPE]
        for offset in range(0, len(raw_soup), TAPE)
    )
    top_items = heapq.nsmallest(
        top_k,
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    top = tuple(
        TapeSummary(raw, count, count / soup.n, tape_id(raw))
        for raw, count in top_items
    )
    complexity = high_order_entropy(raw_soup)
    if replication_tracker is None:
        markers: tuple[VerifiedMarkerSummary, ...] = ()
    elif observe_replicators or discover_replicators:
        markers = replication_tracker.observe(
            raw_soup,
            soup.lib,
            soup.epoch,
            discover=discover_replicators,
        )
    else:
        markers = replication_tracker.latest
    sampling_s = time.perf_counter() - started
    return Snapshot(
        epoch=soup.epoch,
        elapsed_s=elapsed_s,
        epochs_per_s=epochs_per_s,
        complexity=complexity,
        ops_per_interaction=ops_per_interaction,
        population=soup.n,
        unique_tapes=len(counts),
        top=top,
        sampling_s=sampling_s,
        replication_markers=markers,
        replication_history=(
            replication_tracker.history
            if replication_tracker is not None
            else ()
        ),
        verification_enabled=verification_enabled,
        replication_scans=(
            replication_tracker.scan_count
            if replication_tracker is not None
            else 0
        ),
        transition_epoch=transition_epoch,
        final=final,
        final_reason=final_reason,
    )


def _put_latest(target: queue.Queue[Snapshot], snapshot: Snapshot) -> None:
    """Publish without allowing rendering to backpressure the simulation."""
    while True:
        try:
            target.put_nowait(snapshot)
            return
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass


class SimulationWorker:
    """Own and advance exactly one Soup instance on a background thread."""

    def __init__(self, config: RunConfig):
        config.validate()
        self.config = config
        self.snapshots: queue.Queue[Snapshot] = queue.Queue(maxsize=1)
        self.errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        self.failure: BaseException | None = None
        self.done = threading.Event()
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._thread = threading.Thread(
            target=self._run,
            name="bff-soup",
            daemon=True,
        )

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _snapshot(
        self,
        soup: Soup,
        *,
        started: float,
        batch_started: float,
        batch_epochs: int,
        batch_ops: int,
        transition_epoch: int | None,
        replication_tracker: ReplicationTracker | None,
        discover_replicators: bool = False,
        observe_replicators: bool = True,
        final: bool = False,
        final_reason: Literal["completed", "threshold", "stopped"] | None = None,
    ) -> Snapshot:
        now = time.perf_counter()
        batch_elapsed = max(now - batch_started, 1e-12)
        epochs_per_s = batch_epochs / batch_elapsed if batch_epochs else 0.0
        interactions = batch_epochs * (soup.n / 2)
        ops_per_interaction = batch_ops / interactions if interactions else 0.0
        return collect_snapshot(
            soup,
            top_k=self.config.top_k,
            elapsed_s=now - started,
            epochs_per_s=epochs_per_s,
            ops_per_interaction=ops_per_interaction,
            replication_tracker=replication_tracker,
            discover_replicators=discover_replicators,
            observe_replicators=observe_replicators,
            verification_enabled=self.config.verify_replication,
            transition_epoch=transition_epoch,
            final=final,
            final_reason=final_reason,
        )

    def _run(self) -> None:
        soup: Soup | None = None
        last_snapshot: Snapshot | None = None
        try:
            cfg = self.config
            soup = Soup(
                cfg.n,
                seed=cfg.seed,
                mutation_prob=cfg.mutation_prob,
                stepcount=cfg.stepcount,
            )
            if cfg.validate_core:
                cross_check(soup.lib)
            replication_tracker = (
                ReplicationTracker(score_threshold=cfg.replication_threshold)
                if cfg.verify_replication
                else None
            )

            started = time.perf_counter()
            batch_started = started
            last_sample_finished = started
            last_verification_finished = started
            batch_epochs = 0
            batch_ops = 0
            transition_epoch: int | None = None
            pause_started: float | None = None

            last_snapshot = self._snapshot(
                soup,
                started=started,
                batch_started=batch_started,
                batch_epochs=0,
                batch_ops=0,
                transition_epoch=None,
                replication_tracker=replication_tracker,
                final=cfg.epochs == 0,
                final_reason="completed" if cfg.epochs == 0 else None,
            )
            _put_latest(self.snapshots, last_snapshot)

            while soup.epoch < cfg.epochs and not self._stop.is_set():
                if not self._resume.wait(timeout=0.1):
                    if pause_started is None:
                        pause_started = time.perf_counter()
                    continue
                if pause_started is not None:
                    paused_s = time.perf_counter() - pause_started
                    batch_started += paused_s
                    last_sample_finished += paused_s
                    last_verification_finished += paused_s
                    pause_started = None
                if self._stop.is_set():
                    break

                batch_ops += soup.step()
                batch_epochs += 1
                now = time.perf_counter()
                verification_due = bool(
                    replication_tracker is not None
                    and now - last_verification_finished >= cfg.verify_interval_s
                )
                if verification_due:
                    replication_tracker.observe(
                        soup.buf.tobytes(),
                        soup.lib,
                        soup.epoch,
                        discover=True,
                    )
                    last_verification_finished = time.perf_counter()
                    now = last_verification_finished
                due = (
                    soup.epoch == 1
                    or soup.epoch == cfg.epochs
                    or now - last_sample_finished >= cfg.refresh_s
                )
                if not due:
                    continue

                discover_replicators = bool(
                    replication_tracker is not None
                    and soup.epoch >= cfg.epochs
                    and not verification_due
                )
                last_snapshot = self._snapshot(
                    soup,
                    started=started,
                    batch_started=batch_started,
                    batch_epochs=batch_epochs,
                    batch_ops=batch_ops,
                    transition_epoch=transition_epoch,
                    replication_tracker=replication_tracker,
                    discover_replicators=discover_replicators,
                    observe_replicators=not verification_due,
                )
                if discover_replicators:
                    last_verification_finished = time.perf_counter()
                transition_crossed = False
                if (
                    transition_epoch is None
                    and last_snapshot.complexity >= cfg.transition_threshold
                ):
                    transition_epoch = soup.epoch
                    transition_crossed = True
                    last_snapshot = dataclasses.replace(
                        last_snapshot,
                        transition_epoch=transition_epoch,
                    )

                threshold_stop = cfg.stop_at > 0 and last_snapshot.complexity >= cfg.stop_at
                completed = soup.epoch >= cfg.epochs
                if (
                    replication_tracker is not None
                    and (transition_crossed or threshold_stop)
                    and not (discover_replicators or verification_due)
                ):
                    verification_started = time.perf_counter()
                    markers = replication_tracker.observe(
                        soup.buf.tobytes(),
                        soup.lib,
                        soup.epoch,
                        discover=True,
                    )
                    last_verification_finished = time.perf_counter()
                    last_snapshot = dataclasses.replace(
                        last_snapshot,
                        replication_markers=markers,
                        replication_history=replication_tracker.history,
                        replication_scans=replication_tracker.scan_count,
                        sampling_s=(
                            last_snapshot.sampling_s
                            + last_verification_finished
                            - verification_started
                        ),
                    )
                if threshold_stop or completed:
                    last_snapshot = dataclasses.replace(
                        last_snapshot,
                        final=True,
                        final_reason="threshold" if threshold_stop else "completed",
                    )
                _put_latest(self.snapshots, last_snapshot)

                batch_started = time.perf_counter()
                last_sample_finished = batch_started
                batch_epochs = 0
                batch_ops = 0
                if threshold_stop:
                    break

            if self._stop.is_set() and soup is not None:
                if last_snapshot is None or last_snapshot.epoch != soup.epoch:
                    last_snapshot = self._snapshot(
                        soup,
                        started=started,
                        batch_started=batch_started,
                        batch_epochs=batch_epochs,
                        batch_ops=batch_ops,
                        transition_epoch=transition_epoch,
                        replication_tracker=replication_tracker,
                    )
                _put_latest(
                    self.snapshots,
                    dataclasses.replace(
                        last_snapshot,
                        final=True,
                        final_reason="stopped",
                    ),
                )
        except BaseException as exc:  # transfer worker failures to the GUI
            self.failure = exc
            try:
                self.errors.put_nowait(exc)
            except queue.Full:
                pass
        finally:
            self.done.set()


class Dashboard:
    """Matplotlib figure that renders a sequence of snapshots."""

    def __init__(
        self,
        figure,
        *,
        top_k: int,
        transition_threshold: float = 1.0,
        max_history: int = 2_000,
        show_verified: bool = True,
        replication_threshold: int = DEFAULT_SCORE_THRESHOLD,
    ):
        from matplotlib.colors import ListedColormap

        self.figure = figure
        self.top_k = top_k
        self.transition_threshold = transition_threshold
        self.replication_threshold = replication_threshold
        self.epochs: collections.deque[int] = collections.deque(maxlen=max_history)
        self.identity_history: dict[bytes, collections.deque[float]] = {}
        self.identity_ids: dict[bytes, str] = {}
        self.identity_colors: dict[bytes, object] = {}
        self.identity_peaks: dict[bytes, float] = {}
        self.marker_epochs: collections.deque[int] = collections.deque(
            maxlen=max_history
        )
        self.marker_history: dict[bytes, collections.deque[float]] = {}
        self.marker_colors: dict[bytes, object] = {}
        self.marker_meta: dict[bytes, VerifiedMarkerSummary] = {}
        self.marker_peaks: dict[bytes, float] = {}
        self._max_history = max_history
        self._max_identities = max(32, top_k * 8)
        self._max_marker_histories = max(32, top_k * 8)
        self._next_color = 0
        self._next_marker_color = 0
        self.complexity: collections.deque[float] = collections.deque(maxlen=max_history)
        self.ops: collections.deque[float] = collections.deque(maxlen=max_history)
        self._last_epoch: int | None = None
        self._last_marker_epoch: int | None = None
        self._last_snapshot: Snapshot | None = None
        self._run_state = "starting"
        self.view_mode: Literal["verified", "exact"] = (
            "verified" if show_verified else "exact"
        )
        self._cmap = ListedColormap(_CATEGORY_COLORS)

        grid = figure.add_gridspec(3, 1, height_ratios=(1.55, 1.0, 1.0))
        self.tape_ax = figure.add_subplot(grid[0])
        self.rank_ax = figure.add_subplot(grid[1])
        self.metric_ax = figure.add_subplot(grid[2])
        self.ops_ax = self.metric_ax.twinx()
        self.draw_waiting()

    def set_run_state(self, state: str) -> None:
        self._run_state = state
        if self._last_snapshot is not None:
            self._draw_title(self._last_snapshot)
            self.figure.canvas.draw_idle()

    def toggle_view(self) -> Literal["verified", "exact"]:
        self.view_mode = "exact" if self.view_mode == "verified" else "verified"
        if self._last_snapshot is not None:
            self._draw_population_view(self._last_snapshot)
            self._draw_title(self._last_snapshot)
            self.figure.canvas.draw_idle()
        return self.view_mode

    def draw_waiting(self) -> None:
        self.tape_ax.clear()
        self.tape_ax.text(
            0.5,
            0.5,
            "Initializing soup and validating the C core…",
            ha="center",
            va="center",
            transform=self.tape_ax.transAxes,
        )
        self.tape_ax.set_axis_off()
        self.rank_ax.set_axis_off()
        self.metric_ax.set_axis_off()
        self.ops_ax.set_axis_off()
        self.figure.suptitle("BFF primordial soup — verified replication watch")

    def draw_error(self, message: str) -> None:
        self.tape_ax.clear()
        self.rank_ax.clear()
        self.metric_ax.clear()
        self.ops_ax.clear()
        self.tape_ax.text(
            0.5,
            0.5,
            message,
            color="#b42318",
            ha="center",
            va="center",
            wrap=True,
            transform=self.tape_ax.transAxes,
        )
        self.tape_ax.set_axis_off()
        self.rank_ax.set_axis_off()
        self.metric_ax.set_axis_off()
        self.ops_ax.set_axis_off()
        self.figure.suptitle("BFF simulation stopped with an error")

    def update(self, snapshot: Snapshot) -> None:
        self._last_snapshot = snapshot
        if snapshot.epoch != self._last_epoch:
            self._record_identity_history(snapshot)
            self.complexity.append(snapshot.complexity)
            self.ops.append(snapshot.ops_per_interaction)
            self.epochs.append(snapshot.epoch)
            self._last_epoch = snapshot.epoch
        self._record_marker_history(snapshot)

        self._draw_population_view(snapshot)
        self._draw_metrics(snapshot)

        self._draw_title(snapshot)

    def _draw_title(self, snapshot: Snapshot) -> None:
        state = snapshot.final_reason or self._run_state
        verification = (
            f"tracked markers {len(snapshot.replication_markers)} · "
            f"scans {snapshot.replication_scans}"
            if snapshot.verification_enabled
            else "verification off"
        )
        self.figure.suptitle(
            f"BFF soup · epoch {snapshot.epoch:,} · {state} · "
            f"{snapshot.epochs_per_s:,.1f} epochs/s · "
            f"{snapshot.unique_tapes:,}/{snapshot.population:,} unique · "
            f"{verification} · "
            f"sample {snapshot.sampling_s * 1_000:.0f} ms"
        )

    def _draw_population_view(self, snapshot: Snapshot) -> None:
        if self.view_mode == "verified":
            self._draw_verified_tapes(snapshot)
            self._draw_marker_history(snapshot)
        else:
            self._draw_exact_tapes(snapshot)
            self._draw_identity_history(snapshot)

    def _record_identity_history(self, snapshot: Snapshot) -> None:
        from matplotlib import colormaps

        repeated = [item for item in snapshot.top if item.count > 1]
        active = {item.tape for item in repeated}
        history_length = len(self.epochs)

        for item in repeated:
            if item.tape in self.identity_history:
                continue
            if len(self.identity_history) >= self._max_identities:
                evictable = [tape for tape in self.identity_history if tape not in active]
                if not evictable:
                    break
                victim = min(evictable, key=lambda tape: self.identity_peaks[tape])
                self.identity_history.pop(victim)
                self.identity_ids.pop(victim)
                self.identity_colors.pop(victim)
                self.identity_peaks.pop(victim)
            self.identity_history[item.tape] = collections.deque(
                [float("nan")] * history_length,
                maxlen=self._max_history,
            )
            self.identity_ids[item.tape] = item.tape_id
            self.identity_colors[item.tape] = colormaps["tab10"](self._next_color % 10)
            self.identity_peaks[item.tape] = 0.0
            self._next_color += 1

        current = {item.tape: item.share for item in repeated}
        for tape, series in self.identity_history.items():
            share = current.get(tape, float("nan"))
            series.append(share)
            if not np.isnan(share):
                self.identity_peaks[tape] = max(self.identity_peaks[tape], share)

    def _record_marker_history(self, snapshot: Snapshot) -> None:
        from matplotlib import colormaps

        points = snapshot.replication_history
        if not points and snapshot.replication_markers:
            points = (
                ReplicationHistoryPoint(snapshot.epoch, snapshot.replication_markers),
            )
        for point in points:
            if self._last_marker_epoch is not None and point.epoch <= self._last_marker_epoch:
                continue
            history_length = len(self.marker_epochs)
            current = {item.marker: item for item in point.markers}
            for marker, item in current.items():
                if marker not in self.marker_history:
                    if len(self.marker_history) >= self._max_marker_histories:
                        evictable = [
                            tracked
                            for tracked in self.marker_history
                            if tracked not in current
                        ]
                        if not evictable:
                            continue
                        victim = min(
                            evictable,
                            key=lambda tracked: self.marker_peaks[tracked],
                        )
                        self.marker_history.pop(victim)
                        self.marker_colors.pop(victim)
                        self.marker_meta.pop(victim)
                        self.marker_peaks.pop(victim)
                    self.marker_history[marker] = collections.deque(
                        [float("nan")] * history_length,
                        maxlen=self._max_history,
                    )
                    self.marker_colors[marker] = colormaps["tab10"](
                        self._next_marker_color % 10
                    )
                    self.marker_peaks[marker] = item.peak_share
                    self._next_marker_color += 1
                self.marker_meta[marker] = item
                self.marker_peaks[marker] = max(
                    self.marker_peaks[marker],
                    item.peak_share,
                )

            for marker, series in self.marker_history.items():
                item = current.get(marker)
                share = item.carrier_share if item is not None else float("nan")
                series.append(share)
                if not np.isnan(share):
                    self.marker_peaks[marker] = max(
                        self.marker_peaks[marker],
                        share,
                    )
            self.marker_epochs.append(point.epoch)
            self._last_marker_epoch = point.epoch

    def _draw_tape_rows(self, tapes: list[bytes]) -> None:
        ax = self.tape_ax
        matrix = np.stack([classify_tape(tape) for tape in tapes])
        ax.imshow(
            matrix,
            cmap=self._cmap,
            vmin=-0.5,
            vmax=5.5,
            interpolation="nearest",
            aspect="auto",
        )
        for row, tape in enumerate(tapes):
            for column, byte in enumerate(tape):
                glyph = tape_glyph(byte)
                if glyph:
                    ax.text(
                        column,
                        row,
                        glyph,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7,
                        fontweight="bold",
                    )
        ax.set_xticks(range(0, TAPE, 8))
        ax.set_xlabel("byte position")
        ax.set_xlim(-0.5, TAPE - 0.5)
        ax.set_ylim(len(tapes) - 0.5, -0.5)

    def _add_tape_legend(self, *, marker: bool = False) -> None:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=color, label=label)
            for color, label in zip(_CATEGORY_COLORS, _CATEGORY_NAMES)
        ]
        if marker:
            handles.append(Patch(
                facecolor="none",
                edgecolor="#f2b134",
                linewidth=2,
                label="tracking marker",
            ))
        self.tape_ax.legend(
            handles=handles,
            ncol=len(handles),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            frameon=False,
            fontsize=8,
        )

    def _draw_exact_tapes(
        self,
        snapshot: Snapshot,
        *,
        title: str | None = None,
    ) -> None:
        ax = self.tape_ax
        ax.clear()
        ax.set_axis_on()
        rows = list(snapshot.top)
        self._draw_tape_rows([item.tape for item in rows])

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([
            f"#{rank}  {item.count:,} ({item.share:.2%})  {item.tape_id}"
            for rank, item in enumerate(rows, 1)
        ])
        qualifier = "all tapes are unique" if rows and rows[0].count == 1 else "raw-byte identity"
        ax.set_title(
            title or f"Top exact 64-byte tapes — {qualifier}",
            loc="left",
        )
        self._add_tape_legend()

    def _draw_verified_tapes(self, snapshot: Snapshot) -> None:
        from matplotlib.patches import Rectangle

        if not snapshot.replication_markers:
            if not snapshot.verification_enabled:
                status = "Functional verification disabled — showing exact-tape diagnostic"
            elif snapshot.replication_scans == 0:
                status = "Awaiting first functional verification scan — showing exact tapes"
            else:
                status = (
                    "No verified replicator candidate yet — showing exact-tape diagnostic"
                )
            self._draw_exact_tapes(snapshot, title=status)
            return

        ax = self.tape_ax
        ax.clear()
        ax.set_axis_on()
        rows = list(snapshot.replication_markers[:self.top_k])
        self._draw_tape_rows([item.representative for item in rows])
        for row, item in enumerate(rows):
            for offset in marker_offsets(item.representative, item.marker):
                ax.add_patch(Rectangle(
                    (offset - 0.5, row - 0.5),
                    len(item.marker),
                    1.0,
                    fill=False,
                    edgecolor="#f2b134",
                    linewidth=2.0,
                ))

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([
            (
                f"#{rank}  marker {item.marker_id} · "
                f"{item.carrier_count:,} ({item.carrier_share:.2%}) · "
                f"Δ {item.growth_per_100_epochs * 100:+.2f} pp/100e\n"
                f"scores {item.score}/64 + {item.confirmation_score}/64 · "
                f"marker {item.marker_successes}/{item.contexts} ctx · "
                f"exact ×{item.exact_count}"
            )
            for rank, item in enumerate(rows, 1)
        ], fontsize=8)
        ax.set_title(
            "Verified replicator candidates — representative tape and "
            "replicated tracking marker",
            loc="left",
        )
        self._add_tape_legend(marker=True)

    def _draw_marker_history(self, snapshot: Snapshot) -> None:
        from matplotlib.ticker import PercentFormatter

        ax = self.rank_ax
        ax.clear()
        ax.set_axis_on()
        current = {item.marker: item for item in snapshot.replication_markers}
        plotted = [item.marker for item in snapshot.replication_markers[:self.top_k]]
        historical = sorted(
            (marker for marker in self.marker_history if marker not in current),
            key=lambda marker: self.marker_peaks[marker],
            reverse=True,
        )
        plotted.extend(historical[:max(0, self.top_k - len(plotted))])

        for marker in plotted:
            item = current.get(marker) or self.marker_meta[marker]
            current_text = (
                f"now {item.carrier_share:.2%}"
                if marker in current
                else "no longer tracked"
            )
            ax.plot(
                self.marker_epochs,
                self.marker_history[marker],
                linewidth=2.0,
                color=self.marker_colors[marker],
                label=(
                    f"{item.marker_id} · {current_text} · "
                    f"peak {self.marker_peaks[marker]:.2%}"
                ),
            )
        if snapshot.transition_epoch is not None:
            ax.axvline(
                snapshot.transition_epoch,
                color="#52514e",
                linestyle="--",
                linewidth=1,
            )
        ax.set_title("Carrier prevalence of verified-representative markers", loc="left")
        ax.set_xlabel(
            "epoch · verified by two independent 13-context × 5-generation "
            f"assays (score ≥ {self.replication_threshold}/64 twice)",
            fontsize=9,
        )
        ax.set_ylabel("share of soup")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.25)
        if plotted:
            ax.legend(
                ncol=min(len(plotted), 3),
                frameon=False,
                fontsize=8,
                loc="upper left",
            )
        else:
            if snapshot.verification_enabled and snapshot.replication_scans == 0:
                message = "Verification scan pending"
            elif snapshot.verification_enabled:
                message = "No candidate has passed functional verification"
            else:
                message = "Functional verification is disabled"
            ax.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#52514e",
            )
        ax.text(
            1.0,
            1.01,
            "presence-only, non-exclusive; no lineage proof; gaps = outside tracked set",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            color="#52514e",
        )

    def _draw_identity_history(self, snapshot: Snapshot) -> None:
        from matplotlib.ticker import PercentFormatter

        ax = self.rank_ax
        ax.clear()
        active = [item for item in snapshot.top if item.count > 1]
        active_by_tape = {item.tape: item for item in active}
        plotted = [item.tape for item in active]
        historical = sorted(
            (tape for tape in self.identity_history if tape not in active_by_tape),
            key=lambda tape: self.identity_peaks[tape],
            reverse=True,
        )
        plotted.extend(historical[:max(0, self.top_k - len(plotted))])

        for tape in plotted:
            series = self.identity_history.get(tape)
            if series is None:
                continue
            current = active_by_tape.get(tape)
            suffix = f"{current.share:.2%}" if current is not None else "outside top"
            ax.plot(
                self.epochs,
                series,
                linewidth=1.7,
                color=self.identity_colors[tape],
                label=f"{self.identity_ids[tape]} ({suffix})",
            )
        if snapshot.transition_epoch is not None:
            ax.axvline(snapshot.transition_epoch, color="#52514e", linestyle="--", linewidth=1)
        ax.set_title("Dominant exact-tape identities", loc="left")
        ax.set_xlabel("epoch")
        ax.set_ylabel("share of soup")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.25)
        if plotted:
            ax.legend(ncol=min(len(plotted), 4), frameon=False, fontsize=8, loc="upper left")
        else:
            ax.text(
                0.5,
                0.5,
                "No repeated exact tape yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#52514e",
            )
        ax.text(
            1.0,
            1.01,
            "gaps: outside displayed top set",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            color="#52514e",
        )

    def _draw_metrics(self, snapshot: Snapshot) -> None:
        ax = self.metric_ax
        ops_ax = self.ops_ax
        ax.clear()
        ops_ax.clear()
        ax.set_axis_on()
        ops_ax.set_axis_on()
        ops_ax.yaxis.tick_right()
        ops_ax.yaxis.set_label_position("right")
        ops_ax.spines["right"].set_visible(True)
        complexity_line, = ax.plot(
            self.epochs,
            self.complexity,
            color="#2a78d6",
            linewidth=1.8,
            label="high-order entropy",
        )
        ops_line, = ops_ax.plot(
            self.epochs,
            self.ops,
            color="#eb6834",
            linewidth=1.5,
            label="ops / interaction",
        )
        ax.axhline(
            self.transition_threshold,
            color="#898781",
            linestyle=":",
            linewidth=1,
        )
        if snapshot.transition_epoch is not None:
            ax.axvline(snapshot.transition_epoch, color="#52514e", linestyle="--", linewidth=1)
            ops_ax.axvline(snapshot.transition_epoch, color="#52514e", linestyle="--", linewidth=1)
        ax.set_title("Population structure and execution", loc="left")
        ax.set_xlabel("epoch")
        ax.set_ylabel("high-order entropy", color="#2a78d6")
        ops_ax.set_ylabel("ops / interaction", color="#eb6834")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(
            [complexity_line, ops_line],
            [complexity_line.get_label(), ops_line.get_label()],
            frameon=False,
            fontsize=8,
            loc="upper left",
        )


def _interactive_backend_available(matplotlib_module) -> bool:
    backend = str(matplotlib_module.get_backend()).lower()
    non_interactive = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    return backend not in non_interactive and "inline" not in backend


def run_live(config: RunConfig) -> None:
    """Open the interactive dashboard and run until the window closes."""
    import matplotlib

    if not _interactive_backend_available(matplotlib):
        raise RuntimeError(
            f"Matplotlib backend {matplotlib.get_backend()!r} is non-interactive; "
            "run from a desktop session with a GUI backend"
        )

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    figure = plt.figure(figsize=(13.5, 8.8), layout="constrained")
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.075, 1.0, 0.90))
    dashboard = Dashboard(
        figure,
        top_k=config.top_k,
        transition_threshold=config.transition_threshold,
        show_verified=config.verify_replication,
        replication_threshold=config.replication_threshold,
    )
    worker = SimulationWorker(config)

    view_ax = figure.add_axes((0.64, 0.012, 0.13, 0.04))
    pause_ax = figure.add_axes((0.79, 0.012, 0.09, 0.04))
    stop_ax = figure.add_axes((0.89, 0.012, 0.09, 0.04))
    view_button = Button(
        view_ax,
        "Show exact" if dashboard.view_mode == "verified" else "Show verified",
    )
    pause_button = Button(pause_ax, "Pause")
    stop_button = Button(stop_ax, "Stop")

    def toggle_view(_event) -> None:
        mode = dashboard.toggle_view()
        view_button.label.set_text(
            "Show verified" if mode == "exact" else "Show exact"
        )

    def toggle_pause(_event) -> None:
        if worker.paused:
            worker.resume()
            dashboard.set_run_state("running")
            pause_button.label.set_text("Pause")
        else:
            worker.pause()
            dashboard.set_run_state("paused")
            pause_button.label.set_text("Resume")

    def stop_simulation(_event) -> None:
        worker.stop()
        dashboard.set_run_state("stopping")
        pause_button.set_active(False)
        stop_button.label.set_text("Stopping…")
        stop_button.set_active(False)

    def close(_event) -> None:
        worker.stop()

    view_button.on_clicked(toggle_view)
    pause_button.on_clicked(toggle_pause)
    stop_button.on_clicked(stop_simulation)
    figure.canvas.mpl_connect("close_event", close)

    timer = figure.canvas.new_timer(interval=100)

    def poll() -> bool:
        latest: Snapshot | None = None
        finished = False
        while True:
            try:
                latest = worker.snapshots.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            dashboard.update(latest)
            if latest.final:
                finished = True
                pause_button.set_active(False)
                stop_button.set_active(False)
                timer.stop()

        try:
            exc = worker.errors.get_nowait()
        except queue.Empty:
            exc = None
        if exc is not None:
            finished = True
            dashboard.draw_error(f"{type(exc).__name__}: {exc}")
            view_button.set_active(False)
            pause_button.set_active(False)
            stop_button.set_active(False)
            print(f"live simulation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            timer.stop()

        if latest is not None or exc is not None:
            figure.canvas.draw_idle()
        return not finished

    timer.add_callback(poll)
    timer.start()
    dashboard.set_run_state("running")
    worker.start()
    try:
        plt.show()
    finally:
        timer.stop()
        worker.stop()
        worker.join(timeout=5.0)
        plt.close(figure)
    if worker.is_alive:
        raise RuntimeError("simulation worker did not stop within five seconds")
    if worker.failure is not None:
        raise RuntimeError("simulation worker failed") from worker.failure


def _parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096, help="number of tapes (power of two)")
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--mutation", type=float, default=1 / 4096)
    parser.add_argument("--no-mutation", action="store_true")
    parser.add_argument("--top", type=int, default=8, help="rows to display (1-16)")
    parser.add_argument("--refresh", type=float, default=0.5, help="seconds between snapshots")
    parser.add_argument(
        "--verify-every",
        type=float,
        default=0.25,
        help="seconds between bounded functional-verification scans",
    )
    parser.add_argument(
        "--replication-threshold",
        type=int,
        default=DEFAULT_SCORE_THRESHOLD,
        help="minimum score in both assays (48-64)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="disable functional replication detection and start in exact-tape view",
    )
    parser.add_argument("--transition", type=float, default=1.0, help="complexity marker")
    parser.add_argument(
        "--stop-at",
        type=float,
        default=0.0,
        help="stop at this complexity (0 keeps running)",
    )
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args(argv)
    config = RunConfig(
        n=args.n,
        seed=args.seed,
        mutation_prob=0.0 if args.no_mutation else args.mutation,
        epochs=args.epochs,
        top_k=args.top,
        refresh_s=args.refresh,
        verify_interval_s=args.verify_every,
        replication_threshold=args.replication_threshold,
        verify_replication=not args.no_verify,
        transition_threshold=args.transition,
        stop_at=args.stop_at,
        validate_core=not args.skip_check,
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return config


def main(argv: list[str] | None = None) -> int:
    config = _parse_args(argv)
    print(
        f"live BFF soup: {config.n} tapes · seed {config.seed} · "
        f"mutation {config.mutation_prob:.6g} · {COMPRESSOR} · "
        + (
            f"verification every {config.verify_interval_s:g}s"
            if config.verify_replication
            else "verification disabled"
        )
    )
    try:
        run_live(config)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

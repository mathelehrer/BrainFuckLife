"""Bounded functional replication detection for live BFF populations.

The positional score follows CuBFF's current ``CheckSelfRep`` execution model:
13 independent food contexts, five generations per context, and positional
agreement in at least four contexts.  This module uses the more conservative
score threshold of 48 described in the 2026 follow-up paper and requires an
independent confirmation batch before a candidate is called verified.

A verified representative is tracked in the soup with a replicated byte
marker.  The marker is a prevalence proxy, not proof that those bytes are an
autonomous self-replicating core.
"""
from __future__ import annotations

import collections
import ctypes
import dataclasses
import hashlib

import numpy as np

try:
    from .soup import DEFAULT_STEPS, TAPE
except ImportError:  # pragma: no cover - direct script imports
    from soup import DEFAULT_STEPS, TAPE


MASK64 = (1 << 64) - 1
DEFAULT_CONTEXTS = 13
DEFAULT_GENERATIONS = 5
DEFAULT_SCORE_THRESHOLD = 48
DEFAULT_MIN_AGREEMENT = 4


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateSeed:
    marker: bytes
    sample_carriers: int
    sample_size: int
    representatives: tuple[bytes, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ReplicationEvidence:
    representative: bytes
    score: int
    confirmation_score: int
    marker: bytes
    marker_successes: int
    confirmation_successes: int
    contexts: int
    generations: int
    left_support: tuple[bool, ...]
    right_support: tuple[bool, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class VerifiedMarkerSummary:
    marker: bytes
    marker_id: str
    representative: bytes
    representative_id: str
    score: int
    confirmation_score: int
    marker_successes: int
    contexts: int
    generations: int
    carrier_count: int
    carrier_share: float
    exact_count: int
    growth_per_100_epochs: float
    first_verified_epoch: int
    peak_share: float


@dataclasses.dataclass(frozen=True, slots=True)
class ReplicationHistoryPoint:
    epoch: int
    markers: tuple[VerifiedMarkerSummary, ...]


@dataclasses.dataclass(slots=True)
class _Assay:
    score: int
    left_support: tuple[bool, ...]
    right_support: tuple[bool, ...]
    noises: tuple[bytes, ...]
    chains: tuple[tuple[tuple[bytes, bytes], ...], ...]


@dataclasses.dataclass(slots=True)
class _MarkerState:
    evidence: ReplicationEvidence
    first_verified_epoch: int
    last_epoch: int | None = None
    last_share: float = 0.0
    peak_share: float = 0.0


def splitmix64(seed: int) -> int:
    """The stateless 64-bit mixer used by the reference implementation."""
    value = (seed + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def stable_id(data: bytes) -> str:
    return hashlib.blake2s(data, digest_size=4, person=b"BFFREPL").hexdigest()


def _food(seed: int, context: int) -> bytes:
    local_seed = splitmix64(seed)
    return bytes(
        splitmix64(local_seed ^ splitmix64((context + 1) * TAPE + index)) & 0xFF
        for index in range(TAPE)
    )


def _evaluate_pair(left: bytes, right: bytes, lib, stepcount: int) -> tuple[bytes, bytes]:
    tape = np.frombuffer(left + right, dtype=np.uint8).copy()
    ptr = tape.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    lib.evaluate(ptr, stepcount)
    return bytes(tape[:TAPE]), bytes(tape[TAPE:])


def _score_final_states(
    program: bytes,
    chains: tuple[tuple[tuple[bytes, bytes], ...], ...],
    min_agreement: int,
) -> tuple[int, tuple[bool, ...], tuple[bool, ...]]:
    final_left = [chain[-1][0] for chain in chains]
    final_right = [chain[-1][1] for chain in chains]
    left_support = tuple(
        sum(tape[position] == program[position] for tape in final_left) >= min_agreement
        for position in range(TAPE)
    )
    right_support = tuple(
        collections.Counter(tape[position] for tape in final_right).most_common(1)[0][1]
        >= min_agreement
        for position in range(TAPE)
    )
    return min(sum(left_support), sum(right_support)), left_support, right_support


def _run_assay(
    program: bytes,
    lib,
    *,
    seed: int,
    contexts: int,
    generations: int,
    stepcount: int,
    min_agreement: int,
) -> _Assay:
    noises: list[bytes] = []
    branches: list[tuple[tuple[bytes, bytes], ...]] = []
    for context in range(contexts):
        noise = _food(seed, context)
        noises.append(noise)
        parent = program
        chain: list[tuple[bytes, bytes]] = []
        for _ in range(generations):
            left, child = _evaluate_pair(parent, noise, lib, stepcount)
            chain.append((left, child))
            parent = child
        branches.append(tuple(chain))
    chains = tuple(branches)
    score, left_support, right_support = _score_final_states(
        program,
        chains,
        min_agreement,
    )
    return _Assay(
        score=score,
        left_support=left_support,
        right_support=right_support,
        noises=tuple(noises),
        chains=chains,
    )


def _marker_successes(marker: bytes, assay: _Assay) -> int:
    successes = 0
    for noise, chain in zip(assay.noises, assay.chains):
        if marker in noise:
            continue
        if all(marker in left and marker in right for left, right in chain):
            successes += 1
    return successes


def _find_tracking_marker(
    program: bytes,
    first: _Assay,
    confirmation: _Assay,
    *,
    min_length: int,
    max_length: int,
    min_successes: int,
) -> tuple[bytes, int, int] | None:
    for length in range(min(max_length, len(program)), min_length - 1, -1):
        best: tuple[bytes, int, int] | None = None
        seen: set[bytes] = set()
        for offset in range(len(program) - length + 1):
            marker = program[offset:offset + length]
            if marker in seen:
                continue
            seen.add(marker)
            primary_successes = _marker_successes(marker, first)
            heldout_successes = _marker_successes(marker, confirmation)
            if min(primary_successes, heldout_successes) < min_successes:
                continue
            candidate = (marker, primary_successes, heldout_successes)
            if best is None or (
                min(primary_successes, heldout_successes),
                primary_successes + heldout_successes,
                marker,
            ) > (
                min(best[1], best[2]),
                best[1] + best[2],
                best[0],
            ):
                best = candidate
        if best is not None:
            return best
    return None


def verify_candidate(
    program: bytes,
    lib,
    *,
    contexts: int = DEFAULT_CONTEXTS,
    generations: int = DEFAULT_GENERATIONS,
    score_threshold: int = DEFAULT_SCORE_THRESHOLD,
    min_agreement: int = DEFAULT_MIN_AGREEMENT,
    marker_min_length: int = 8,
    marker_max_length: int = 16,
    stepcount: int = DEFAULT_STEPS,
    seed: int = 0x243F6A8885A308D3,
    confirmation_seed: int = 0x13198A2E03707344,
) -> ReplicationEvidence | None:
    """Verify a candidate with two assays and a replicated tracking marker.

    In addition to meeting ``score_threshold`` in both noise batches, an
    accepted program must contain an 8--16 byte substring supported in at
    least ``min_agreement`` contexts in both batches.  A context supports the
    marker when it is absent from that context's food and present in both
    output halves after every generation.
    """
    if len(program) != TAPE:
        raise ValueError(f"candidate must be exactly {TAPE} bytes")
    if not 1 <= min_agreement <= contexts:
        raise ValueError("min_agreement must be within the context count")
    if not 1 <= score_threshold <= TAPE:
        raise ValueError("score_threshold must be between 1 and 64")
    if generations <= 0:
        raise ValueError("generations must be positive")
    if stepcount <= 0:
        raise ValueError("stepcount must be positive")
    if not 1 <= marker_min_length <= marker_max_length <= TAPE:
        raise ValueError("marker lengths must satisfy 1 <= min <= max <= 64")
    if (seed & MASK64) == (confirmation_seed & MASK64):
        raise ValueError("confirmation_seed must differ from seed")

    first = _run_assay(
        program,
        lib,
        seed=seed,
        contexts=contexts,
        generations=generations,
        stepcount=stepcount,
        min_agreement=min_agreement,
    )
    if first.score < score_threshold:
        return None
    confirmation = _run_assay(
        program,
        lib,
        seed=confirmation_seed,
        contexts=contexts,
        generations=generations,
        stepcount=stepcount,
        min_agreement=min_agreement,
    )
    if confirmation.score < score_threshold:
        return None

    marker_result = _find_tracking_marker(
        program,
        first,
        confirmation,
        min_length=marker_min_length,
        max_length=marker_max_length,
        min_successes=min_agreement,
    )
    if marker_result is None:
        return None
    marker, successes, confirmation_successes = marker_result
    return ReplicationEvidence(
        representative=program,
        score=first.score,
        confirmation_score=confirmation.score,
        marker=marker,
        marker_successes=successes,
        confirmation_successes=confirmation_successes,
        contexts=contexts,
        generations=generations,
        left_support=first.left_support,
        right_support=first.right_support,
    )


def _sample_tapes(raw_soup: bytes, sample_limit: int) -> np.ndarray:
    population = len(raw_soup) // TAPE
    if population == 0:
        return np.empty((0, TAPE), dtype=np.uint8)
    tapes = np.frombuffer(raw_soup, dtype=np.uint8).reshape(population, TAPE)
    sample_size = min(population, sample_limit)
    if sample_size == population:
        return tapes
    indices = (np.arange(sample_size, dtype=np.uint64) * population // sample_size).astype(np.intp)
    return tapes[indices]


def discover_candidates(
    raw_soup: bytes,
    *,
    marker_length: int = 16,
    sample_limit: int = 8192,
    top_markers: int = 16,
    representatives_per_marker: int = 2,
    min_sample_carriers: int = 2,
    excluded_representatives: set[bytes] | None = None,
) -> tuple[CandidateSeed, ...]:
    """Find repeated markers by distinct sampled carrier tapes, not occurrences."""
    if len(raw_soup) % TAPE:
        raise ValueError("soup buffer length is not a multiple of the tape size")
    if not 1 <= marker_length <= TAPE:
        raise ValueError("invalid marker length")
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive")
    if top_markers <= 0 or representatives_per_marker <= 0:
        raise ValueError("candidate limits must be positive")
    if min_sample_carriers <= 0:
        raise ValueError("min_sample_carriers must be positive")
    sample = _sample_tapes(raw_soup, sample_limit)
    if len(sample) < min_sample_carriers:
        return ()
    window_count = TAPE - marker_length + 1
    windows = np.lib.stride_tricks.sliding_window_view(sample, marker_length, axis=1)
    hashes = np.zeros((len(sample), window_count), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for index in range(marker_length):
            hashes *= np.uint64(257)
            hashes += windows[:, :, index].astype(np.uint64) + np.uint64(1)

    ordered = np.sort(hashes, axis=1)
    distinct = np.ones_like(ordered, dtype=bool)
    distinct[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    unique_hashes, carrier_counts = np.unique(ordered[distinct], return_counts=True)
    eligible = np.flatnonzero(carrier_counts >= min_sample_carriers)
    if not len(eligible):
        return ()
    order = eligible[np.lexsort((
        unique_hashes[eligible],
        -carrier_counts[eligible].astype(np.int64),
    ))]

    seeds: list[CandidateSeed] = []
    carrier_signatures: set[bytes] = set()
    representative_signatures: set[tuple[bytes, ...]] = set()
    excluded = excluded_representatives or set()
    sample_bytes = [bytes(row) for row in sample]
    for selected in order:
        target_hash = unique_hashes[selected]
        matches = hashes == target_hash
        carrier_mask = matches.any(axis=1)
        carrier_signature = np.packbits(carrier_mask).tobytes()
        if carrier_signature in carrier_signatures:
            continue
        carrier_signatures.add(carrier_signature)
        raw_counts: dict[bytes, set[int]] = collections.defaultdict(set)
        for row in np.flatnonzero(carrier_mask):
            column = int(np.argmax(matches[row]))
            marker = bytes(sample[row, column:column + marker_length])
            raw_counts[marker].add(int(row))
        if not raw_counts:
            continue
        marker, carriers = max(
            raw_counts.items(),
            key=lambda item: (len(item[1]), item[0]),
        )
        if len(carriers) < min_sample_carriers:
            continue
        representatives: list[bytes] = []
        for row in sorted(carriers):
            tape = sample_bytes[row]
            if tape not in excluded and tape not in representatives:
                representatives.append(tape)
            if len(representatives) >= representatives_per_marker:
                break
        if not representatives:
            continue
        signature = tuple(representatives)
        # A duplicated exact tape contributes up to 49 equally frequent
        # windows.  Keep one seed for that carrier set so it cannot crowd all
        # other candidate carrier sets out of the bounded verification budget.
        if signature in representative_signatures:
            continue
        representative_signatures.add(signature)
        seeds.append(CandidateSeed(
            marker=marker,
            sample_carriers=len(carriers),
            sample_size=len(sample),
            representatives=signature,
        ))
        if len(seeds) >= top_markers:
            break
    return tuple(seeds)


def count_marker_carriers(raw_soup: bytes, marker: bytes) -> int:
    """Count tapes containing a marker, excluding cross-boundary matches."""
    if len(raw_soup) % TAPE:
        raise ValueError("soup buffer length is not a multiple of the tape size")
    if not marker or len(marker) > TAPE:
        return 0
    count = 0
    search_from = 0
    while True:
        position = raw_soup.find(marker, search_from)
        if position < 0:
            return count
        tape_start = position - position % TAPE
        if position - tape_start <= TAPE - len(marker):
            count += 1
            search_from = tape_start + TAPE
        else:
            search_from = position + 1


def count_exact_tape(raw_soup: bytes, tape: bytes) -> int:
    if len(raw_soup) % TAPE:
        raise ValueError("soup buffer length is not a multiple of the tape size")
    if len(tape) != TAPE:
        return 0
    count = 0
    search_from = 0
    while True:
        position = raw_soup.find(tape, search_from)
        if position < 0:
            return count
        if position % TAPE == 0:
            count += 1
            search_from = position + TAPE
        else:
            search_from = position + 1


class ReplicationTracker:
    """Discover, verify, and track a bounded set of replicating markers."""

    def __init__(
        self,
        *,
        sample_limit: int = 8192,
        top_markers: int = 16,
        max_new_tests: int = 8,
        max_markers: int = 8,
        tested_cache_size: int = 4096,
        score_threshold: int = DEFAULT_SCORE_THRESHOLD,
        history_size: int = 2_000,
    ):
        if min(
            sample_limit,
            top_markers,
            max_new_tests,
            max_markers,
            tested_cache_size,
            history_size,
        ) <= 0:
            raise ValueError("replication tracker limits must be positive")
        if not 1 <= score_threshold <= TAPE:
            raise ValueError("score_threshold must be between 1 and 64")
        self.sample_limit = sample_limit
        self.top_markers = top_markers
        self.max_new_tests = max_new_tests
        self.max_markers = max_markers
        self.tested_cache_size = tested_cache_size
        self.score_threshold = score_threshold
        self._history: collections.deque[ReplicationHistoryPoint] = (
            collections.deque(maxlen=history_size)
        )
        self._latest: tuple[VerifiedMarkerSummary, ...] = ()
        self._markers: dict[bytes, _MarkerState] = {}
        self._tested: collections.OrderedDict[bytes, None] = collections.OrderedDict()
        self._verified: collections.OrderedDict[
            bytes,
            tuple[ReplicationEvidence, int],
        ] = (
            collections.OrderedDict()
        )
        self._archives: collections.OrderedDict[bytes, tuple[int, float]] = (
            collections.OrderedDict()
        )
        self.scan_count = 0

    @property
    def marker_count(self) -> int:
        return len(self._markers)

    @property
    def latest(self) -> tuple[VerifiedMarkerSummary, ...]:
        return self._latest

    @property
    def history(self) -> tuple[ReplicationHistoryPoint, ...]:
        return tuple(self._history)

    def _remember_test(self, representative: bytes) -> bool:
        if representative in self._tested:
            self._tested.move_to_end(representative)
            return False
        self._tested[representative] = None
        if len(self._tested) > self.tested_cache_size:
            forgotten, _ = self._tested.popitem(last=False)
            self._verified.pop(forgotten, None)
        return True

    def _register(
        self,
        evidence: ReplicationEvidence,
        epoch: int,
        current_share: float,
        verified_epoch: int | None = None,
    ) -> None:
        existing = self._markers.get(evidence.marker)
        if existing is not None:
            if min(evidence.score, evidence.confirmation_score) > min(
                existing.evidence.score,
                existing.evidence.confirmation_score,
            ):
                existing.evidence = evidence
            return
        # Nested markers count substantially the same carrier set and make a
        # single outbreak appear as several independent tracks. Preserve an
        # active incumbent, but retire an extinct one so a derived signature
        # can become visible on the next scan.
        related = [
            (marker, state)
            for marker, state in self._markers.items()
            if evidence.marker in marker or marker in evidence.marker
        ]
        if any(state.last_share > 0 for _, state in related):
            return
        for marker, state in related:
            self._markers.pop(marker)
            self._archive(marker, state)
        if len(self._markers) >= self.max_markers:
            victim_marker, victim = min(
                self._markers.items(),
                key=lambda item: (item[1].last_share, item[1].peak_share),
            )
            if current_share <= victim.last_share and victim.last_share > 0:
                return
            self._markers.pop(victim_marker)
            self._archive(victim_marker, victim)
        archived = self._archives.get(evidence.marker)
        if archived is not None:
            self._archives.move_to_end(evidence.marker)
        self._markers[evidence.marker] = _MarkerState(
            evidence=evidence,
            first_verified_epoch=(
                archived[0]
                if archived is not None
                else verified_epoch if verified_epoch is not None else epoch
            ),
            last_epoch=epoch,
            last_share=current_share,
            peak_share=max(current_share, archived[1]) if archived is not None else current_share,
        )

    def _archive(self, marker: bytes, state: _MarkerState) -> None:
        self._archives[marker] = (
            state.first_verified_epoch,
            state.peak_share,
        )
        self._archives.move_to_end(marker)
        if len(self._archives) > self.tested_cache_size:
            self._archives.popitem(last=False)

    def _discover(self, raw_soup: bytes, lib, epoch: int) -> None:
        self.scan_count += 1
        seeds = discover_candidates(
            raw_soup,
            sample_limit=self.sample_limit,
            top_markers=self.top_markers,
            excluded_representatives=(
                set(self._tested).difference(self._verified)
            ),
        )
        tests = 0
        population = len(raw_soup) // TAPE
        for seed in seeds:
            for representative in seed.representatives:
                cached_entry = self._verified.get(representative)
                if cached_entry is not None:
                    cached, verified_epoch = cached_entry
                    self._verified.move_to_end(representative)
                    self._tested.move_to_end(representative)
                    current_share = (
                        count_marker_carriers(raw_soup, cached.marker) / population
                        if population else 0.0
                    )
                    self._register(
                        cached,
                        epoch,
                        current_share,
                        verified_epoch=verified_epoch,
                    )
                    continue
                if tests >= self.max_new_tests:
                    continue
                if not self._remember_test(representative):
                    continue
                tests += 1
                evidence = verify_candidate(
                    representative,
                    lib,
                    score_threshold=self.score_threshold,
                )
                if evidence is not None:
                    self._verified[representative] = (evidence, epoch)
                    current_share = (
                        count_marker_carriers(raw_soup, evidence.marker) / population
                        if population else 0.0
                    )
                    self._register(
                        evidence,
                        epoch,
                        current_share,
                        verified_epoch=epoch,
                    )

    def observe(
        self,
        raw_soup: bytes,
        lib,
        epoch: int,
        *,
        discover: bool,
    ) -> tuple[VerifiedMarkerSummary, ...]:
        if discover:
            self._discover(raw_soup, lib, epoch)
        population = len(raw_soup) // TAPE
        if population == 0:
            return ()
        summaries: list[VerifiedMarkerSummary] = []
        for marker, state in self._markers.items():
            evidence = state.evidence
            carrier_count = count_marker_carriers(raw_soup, marker)
            share = carrier_count / population
            if state.last_epoch is None or epoch == state.last_epoch:
                growth = 0.0
            else:
                growth = (share - state.last_share) * 100 / (epoch - state.last_epoch)
            state.last_epoch = epoch
            state.last_share = share
            state.peak_share = max(state.peak_share, share)
            summaries.append(VerifiedMarkerSummary(
                marker=marker,
                marker_id=stable_id(marker),
                representative=evidence.representative,
                representative_id=stable_id(evidence.representative),
                score=evidence.score,
                confirmation_score=evidence.confirmation_score,
                marker_successes=min(
                    evidence.marker_successes,
                    evidence.confirmation_successes,
                ),
                contexts=evidence.contexts,
                generations=evidence.generations,
                carrier_count=carrier_count,
                carrier_share=share,
                exact_count=count_exact_tape(raw_soup, evidence.representative),
                growth_per_100_epochs=growth,
                first_verified_epoch=state.first_verified_epoch,
                peak_share=state.peak_share,
            ))
        summaries.sort(
            key=lambda item: (-item.carrier_share, -item.peak_share, item.marker_id)
        )
        result = tuple(summaries)
        self._latest = result
        if result or self._history:
            point = ReplicationHistoryPoint(epoch=epoch, markers=result)
            if self._history and self._history[-1].epoch == epoch:
                self._history[-1] = point
            else:
                self._history.append(point)
        return result

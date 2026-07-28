#!/usr/bin/env python3
"""Is the emergent program actually a self-replicator?

After the soup's phase transition one program family dominates. That alone
does not prove it replicates — it could just be what the soup decayed into.
The decisive test is the paper's "seeded" experiment (Section 2.2): drop a
few copies into a *fresh* soup of pure noise and see whether they take over.

This script:
  1. runs a soup until it transitions, and extracts the dominant tape;
  2. injects it into 1% of a fresh random soup and watches it spread;
  3. repeats with a random tape as a control, which should go nowhere.

Usage:
    python3 takeover.py                 # seed 4 transitions at epoch ~8750
    python3 takeover.py --seed 4 --n 4096
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from soup import Soup, dominant_kmer, render, TAPE


def fraction_carrying(tapes: np.ndarray, kmer: bytes) -> float:
    """Fraction of tapes containing `kmer` anywhere."""
    hits = sum(1 for row in tapes if kmer in bytes(row))
    return hits / len(tapes)


def grow(label: str, probe_tape: bytes, kmer: bytes, n: int, seed: int,
         epochs: int, every: int) -> float:
    """Seed a fresh noise soup with `probe_tape` at 1% and watch `kmer` spread."""
    fresh = Soup(n, seed=seed)
    inject = max(1, n // 100)
    arr = fresh.tapes()
    rng = np.random.default_rng(seed)
    for i in rng.choice(n, inject, replace=False):
        arr[i] = np.frombuffer(probe_tape, dtype=np.uint8)

    print(f"\n{label}")
    print(f"  |{render(probe_tape)}|")
    print(f"  seeded into {inject}/{n} tapes ({inject / n:.1%})")
    print(f"  {'epoch':>6} {'carrying kmer':>14} {'complexity':>11}")
    start = fraction_carrying(fresh.tapes(), kmer)
    print(f"  {0:>6} {start:>13.2%} {fresh.complexity():>11.3f}")

    for _ in range(epochs):
        fresh.step()
        if fresh.epoch % every == 0:
            frac = fraction_carrying(fresh.tapes(), kmer)
            print(f"  {fresh.epoch:>6} {frac:>13.2%} {fresh.complexity():>11.3f}")
    return fraction_carrying(fresh.tapes(), kmer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=20000)
    ap.add_argument("--grow-epochs", type=int, default=120)
    ap.add_argument("--every", type=int, default=20)
    args = ap.parse_args()

    # --- 1. evolve a soup until it transitions -------------------------
    print(f"evolving soup (n={args.n}, seed={args.seed}) until transition ...")
    soup = Soup(args.n, seed=args.seed)
    for _ in range(args.max_epochs):
        soup.step()
        if soup.epoch % 50 == 0 and soup.complexity() >= 1.0:
            break
    c = soup.complexity()
    if c < 1.0:
        print(f"no transition within {args.max_epochs} epochs "
              f"(complexity {c:.3f}); try another --seed")
        sys.exit(1)
    print(f"transition at epoch {soup.epoch}, complexity {c:.3f}")

    # let it settle so the winning variant is clear
    for _ in range(400):
        soup.step()

    winner = soup.most_common(1)[0][0]

    # Probe for the winner's spread with a 16-mer taken *from the winner
    # itself* — the most widespread one. Picking the soup's globally most
    # common 16-mer instead would be wrong: render() draws every non-opcode
    # byte as a space, so two cores can look identical on screen while their
    # filler bytes differ, and the probe would then match nothing.
    tapes = soup.tapes()
    counts = {}
    for i in range(TAPE - 16 + 1):
        cand = winner[i:i + 16]
        counts[cand] = fraction_carrying(tapes, cand)
    kmer = max(counts, key=counts.get)
    print(f"\ndominant tape's most widespread 16-byte core: |{render(kmer)}|  "
          f"(in {counts[kmer]:.1%} of tapes)")

    # --- 2. does it take over a fresh soup? ----------------------------
    got = grow("REPLICATOR — dominant tape injected into fresh noise:",
               winner, kmer, args.n, args.seed + 1000,
               args.grow_epochs, args.every)

    # --- 3. control: a random tape should not spread -------------------
    rng = np.random.default_rng(args.seed)
    ctrl_tape = rng.integers(0, 256, TAPE, dtype=np.uint8).tobytes()
    ctrl_kmer = ctrl_tape[16:32]
    ctrl = grow("CONTROL — a random tape injected the same way:",
                ctrl_tape, ctrl_kmer, args.n, args.seed + 1000,
                args.grow_epochs, args.every)

    print(f"\nresult: replicator core went 1.0% -> {got:.1%} of the soup, "
          f"random control 1.0% -> {ctrl:.1%}")
    if got > 4 * ctrl:
        print("=> the emergent program reproduces itself; the control decays.")
        print("   The core never reaches 100% because the family keeps "
              "drifting — variants shift position and mutate, so an exact "
              "16-byte probe undercounts them. The complexity column is the "
              "better measure of how completely the soup has been taken over.")


if __name__ == "__main__":
    main()

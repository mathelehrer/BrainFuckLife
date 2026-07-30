#!/usr/bin/env python3
"""Watch a running soup and log how its population of tapes evolves.

Runs a :class:`soup.Soup` exactly as ``soup.py`` does, but instead of
printing a final summary it snapshots the soup as it goes: every 10th
epoch, the 100 most frequently occurring tapes are rendered to plain
ASCII (opcodes and zeros, the way ``soup.render`` prints them - see
``soup.py``) and appended to the data file, one tape per line. There are
no epoch markers or counts in the file, just tape after tape after tape -
so a run's whole history is one flat stream of 100-tape snapshots that
can be diffed or skimmed epoch block by epoch block.

Early on almost every tape in the soup is still unique, so the "100 most
frequent" are simply 100 of the original random tapes - there has not
been time for the soup's interactions to produce duplicates yet. As
replicators emerge, the same handful of tapes start dominating those
snapshots instead.

The file is flushed after every snapshot, so it is safe to tail it (or
interrupt the run with Ctrl-C) while the soup is still evolving.

Usage:
    python3 soup_watcher.py                       # 1024 tapes, 20000 epochs
    python3 soup_watcher.py --n 4096 --epochs 100000
    python3 soup_watcher.py --out "soup evolution.csv" --every 10 --top 100
"""
from __future__ import annotations

import argparse
import pathlib
import time

from soup import Soup, render

HERE = pathlib.Path(__file__).parent


def watch(soup: Soup, epochs: int, every: int, top: int, out_path: pathlib.Path,
         report_every: int = 100) -> None:
    """Step ``soup`` for ``epochs`` epochs, snapshotting it along the way.

    Every ``every`` epochs, the ``top`` most frequently occurring tapes
    are appended to ``out_path`` as plain ASCII, one per line - no epoch
    marker, no count, just the tapes themselves.
    """
    t0 = time.time()
    with open(out_path, "w") as f:
        print(f"watching soup: {soup.n} tapes x 64 bytes | snapshotting the "
              f"top {top} tapes every {every} epochs -> {out_path}")
        print(f"{'epoch':>7} {'complexity':>11} {'elapsed':>8}")
        for _ in range(epochs):
            soup.step()
            if soup.epoch % every == 0:
                for tape, _count in soup.most_common(top):
                    f.write(render(tape) + "\n")
                f.flush()
                if soup.epoch % report_every == 0:
                    print(f"{soup.epoch:>7} {soup.complexity():>11.4f} "
                          f"{time.time() - t0:>7.1f}s")
    print(f"\nstopped at epoch {soup.epoch} after {time.time() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1024, help="number of tapes (power of 2)")
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mutation", type=float, default=1 / 4096,
                    help="per-byte, per-epoch mutation probability")
    ap.add_argument("--no-mutation", action="store_true",
                    help="pure self-modification, no background noise")
    ap.add_argument("--every", type=int, default=10,
                    help="epochs between snapshots")
    ap.add_argument("--top", type=int, default=100,
                    help="how many of the most frequent tapes to store per snapshot")
    ap.add_argument("--out", type=str, default="soup evolution.csv",
                    help="data file the snapshots are appended to")
    args = ap.parse_args()

    mut = 0.0 if args.no_mutation else args.mutation
    soup = Soup(args.n, seed=args.seed, mutation_prob=mut)
    out_path = HERE / args.out

    try:
        watch(soup, args.epochs, args.every, args.top, out_path)
    except KeyboardInterrupt:
        print(f"\ninterrupted at epoch {soup.epoch}")


if __name__ == "__main__":
    main()

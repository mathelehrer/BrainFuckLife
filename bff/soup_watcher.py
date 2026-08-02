#!/usr/bin/env python3
"""Watch a running soup and log how its population of tapes evolves.

Runs a :class:`soup.Soup` exactly as ``soup.py`` does, but instead of
printing a final summary it snapshots the soup as it goes: every 10th
epoch, the 100 most frequently occurring tapes are appended to the data
file, one tape per row. There are no epoch markers or counts in the file,
just tape after tape after tape - so a run's whole history is one flat
stream of 100-tape snapshots that can be diffed or skimmed epoch block by
epoch block.

The file is a ``;``-delimited csv of **bytes**: a header row ``c0;..;c63``
then, per tape, 64 integers - the ascii code of each cell as ``soup.render``
prints it (an opcode's own character, ``'0'`` for a zero byte, ``' '`` for
anything else). It is written this way, rather than as the plain ASCII the
tapes render to, so it can be read straight into Blender: geometry nodes
have no string attribute type, so ``Import CSV`` keeps only numeric columns
and the tapes have to arrive as numbers (see ``BlenderMathAnim``'s
``SoupWatcherModifier``, which turns each byte back into its character).
Storing the *rendered* code rather than the raw tape byte keeps zeros
showing as ``'0'`` and non-opcodes blank, exactly as the ASCII form did.

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
    python3 soup_watcher.py --out soup_evolution_bytes.csv --every 10 --top 100
"""
from __future__ import annotations

import argparse
import pathlib
import time

from soup import Soup, TAPE, render

HERE = pathlib.Path(__file__).parent


def watch(soup: Soup, epochs: int, every: int, top: int, out_path: pathlib.Path,
         report_every: int = 100) -> None:
    """Step ``soup`` for ``epochs`` epochs, snapshotting it along the way.

    Every ``every`` epochs, the ``top`` most frequently occurring tapes
    are appended to ``out_path`` - no epoch marker, no count, just the
    tapes themselves, each a row of 64 ``;``-delimited byte codes (see the
    module docstring). A ``c0;..;c63`` header is written once, first.
    """
    t0 = time.time()
    with open(out_path, "w") as f:
        f.write(";".join(f"c{c}" for c in range(TAPE)) + "\n")
        print(f"watching soup: {soup.n} tapes x {TAPE} bytes | snapshotting the "
              f"top {top} tapes every {every} epochs -> {out_path}")
        print(f"{'epoch':>7} {'complexity':>11} {'elapsed':>8}")
        for _ in range(epochs):
            soup.step()
            if soup.epoch % every == 0:
                # for tape, _count in soup.most_common(top):
                #     f.write(";".join(str(ord(ch)) for ch in render(tape)) + "\n")
                for i in range(top):
                    tape = soup.tapes()[i]
                    f.write(";".join(str(ord(ch)) for ch in render(tape)) + "\n")
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
    ap.add_argument("--out", type=str, default="soup_evolution_bytes.csv",
                    help="byte csv the snapshots are written to")
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

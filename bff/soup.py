#!/usr/bin/env python3
"""BFF primordial soup — reproducing the emergence of self-replicators.

The experiment from arXiv:2406.19108 Section 2.1, in full:

    Fill N tapes of 64 bytes with uniform random noise. Repeatedly pick two
    at random, concatenate them into 128 bytes, run that as a BFF program,
    split the result back into two 64-byte tapes, return them to the soup.

That is the entire rule. There is no fitness function, no selection, no
goal. Programs change only because they overwrite themselves and each other.
After some thousands of epochs the soup undergoes a sharp phase transition
from noise to self-replicating code.

Usage:
    python3 soup.py                     # 1024 tapes, run until transition
    python3 soup.py --n 8192 --epochs 50000
    python3 soup.py --no-mutation       # self-modification only, no noise
    python3 soup.py --seed 7 --csv run.csv

Progress is measured with the paper's "high-order entropy": the Shannon
entropy of the soup's bytes minus its compressed size in bits per byte.
Uniform noise scores ~0 (incompressible and high entropy); a soup taken over
by copies of one program scores well above 1 (compressible but its bytes are
still varied). The transition is unmistakable.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import math
import pathlib
import subprocess
import sys
import time
import zlib

import numpy as np

HERE = pathlib.Path(__file__).parent
TAPE = 64
MEM = 128
DEFAULT_STEPS = 1 << 13

try:
    import brotli  # the compressor the paper actually used (brotli -q2)

    def _compressed_size(buf: bytes) -> int:
        return len(brotli.compress(buf, quality=2))

    COMPRESSOR = "brotli -q2"
except ImportError:
    def _compressed_size(buf: bytes) -> int:
        return len(zlib.compress(buf, 6))

    COMPRESSOR = "zlib -6 (brotli not installed; same shape, slight offset)"


def _load_lib() -> ctypes.CDLL:
    """Compile soup.c on first use, then load it."""
    src = HERE / "soup.c"
    so = HERE / "libsoup.so"
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        cmd = ["gcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(src)]
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so))
    lib.evaluate.restype = ctypes.c_size_t
    lib.evaluate.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.run_epoch.restype = ctypes.c_uint64
    lib.run_epoch.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_size_t,
    ]
    lib.seed_rng.argtypes = [ctypes.c_uint64]
    lib.randomize.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    return lib


def high_order_entropy(buf: bytes) -> float:
    """Shannon entropy (bits/byte) minus compressed size (bits/byte)."""
    if not buf:
        return 0.0
    counts = np.bincount(np.frombuffer(buf, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(buf)
    shannon = float(-(p * np.log2(p)).sum())
    compressed = _compressed_size(buf) * 8 / len(buf)
    return shannon - compressed


class Soup:
    def __init__(self, n_programs=1024, seed=1, mutation_prob=1 / 4096,
                 stepcount=DEFAULT_STEPS):
        if n_programs < 2 or n_programs & (n_programs - 1):
            raise ValueError("n_programs should be a power of two greater than one")
        self.lib = _load_lib()
        self.n = n_programs
        self.stepcount = stepcount
        self.mut_num = int(round(mutation_prob * (1 << 30)))
        self.lib.seed_rng(ctypes.c_uint64(seed))

        self.buf = np.zeros(n_programs * TAPE, dtype=np.uint8)
        self._ptr = self.buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self.lib.randomize(self._ptr, self.buf.size)

        self._idx = np.zeros(n_programs, dtype=np.uint32)
        self._idx_ptr = self._idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        self.epoch = 0

    def step(self) -> int:
        ops = self.lib.run_epoch(self._ptr, self.n, self._idx_ptr,
                                 self.mut_num, self.stepcount)
        self.epoch += 1
        return int(ops)

    def complexity(self) -> float:
        return high_order_entropy(self.buf.tobytes())

    def tapes(self) -> np.ndarray:
        return self.buf.reshape(self.n, TAPE)

    def most_common(self, k=3):
        rows = [bytes(r) for r in self.tapes()]
        return collections.Counter(rows).most_common(k)


def render(tape: bytes) -> str:
    """Opcodes and zeros only, the way the paper prints tapes."""
    ops = b"<>{}-+.,[]"
    return "".join(chr(c) if c in ops else ("0" if c == 0 else " ") for c in tape)


def longest_common_substring(a: bytes, b: bytes) -> int:
    """Length of the longest run of bytes common to both (64x64 DP)."""
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def test_selfrep(prog: bytes, lib, trials=16, stepcount=DEFAULT_STEPS):
    """How well `prog` copies itself into a tape of random noise.

    This is the paper's autocatalytic reaction S + F -> 2S: put the candidate
    in slot A, uniform noise in slot B, and see what lands in B. Only slot B
    counts — a program that does nothing leaves slot A untouched, which would
    otherwise read as a perfect replication.

    Returns (exact_rate, median_copied_run):

      exact_rate        fraction of trials where B came out byte-identical
                        to A. This is the strictest possible test and it
                        *under*-reports: the paper notes that a replicator is
                        usually shorter than the 64-byte window, so one that
                        copies itself at an offset other than 64 replicates
                        perfectly well while never producing an exact
                        full-tape copy.
      median_copied_run length of the longest byte run shared by A-in and
                        B-out, median over trials. This catches the offset
                        replicators the exact test misses; 64 means a
                        complete copy, and anything above ~8 is far beyond
                        what random tapes share.
    """
    if len(prog) != TAPE:
        raise ValueError(f"prog must be exactly {TAPE} bytes")

    rng = np.random.default_rng(12345)
    exact = 0
    runs = []
    for _ in range(trials):
        food = rng.integers(0, 256, TAPE, dtype=np.uint8).tobytes()
        tape = np.frombuffer(prog + food, dtype=np.uint8).copy()
        ptr = tape.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        lib.evaluate(ptr, stepcount)
        out_b = bytes(tape[TAPE:])
        if out_b == prog:
            exact += 1
        runs.append(longest_common_substring(prog, out_b))
    return exact / trials, float(np.median(runs))


def dominant_kmer(tapes: np.ndarray, k=16, top=3):
    """Most frequent k-byte substrings in the soup.

    Because replicators are substrings that shift position, counting whole
    tapes badly undercounts them. Counting k-mers finds the actual replicating
    core regardless of where it sits on the tape.
    """
    counts = collections.Counter()
    for row in tapes:
        rb = bytes(row)
        for i in range(len(rb) - k + 1):
            counts[rb[i:i + k]] += 1
    return counts.most_common(top)


def cross_check(lib) -> None:
    """Verify the C core agrees with the readable bff.py on random tapes."""
    if __package__:
        from . import bff
    else:
        sys.path.insert(0, str(HERE))
        import bff

    rng = np.random.default_rng(0)
    for _ in range(200):
        data = rng.integers(0, 256, MEM, dtype=np.uint8)
        py = bytearray(data.tobytes())
        bff.evaluate(py, DEFAULT_STEPS)
        c = data.copy()
        lib.evaluate(c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                     DEFAULT_STEPS)
        if bytes(c) != bytes(py):
            raise AssertionError("C core disagrees with bff.py")

    # and on the paper's replicator
    py = bytearray(bff.PAPER_REPLICATOR + bytes(TAPE))
    bff.evaluate(py, DEFAULT_STEPS)
    assert bytes(py[TAPE:]) == bff.PAPER_REPLICATOR, "replicator check failed"


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
    ap.add_argument("--every", type=int, default=100, help="epochs between reports")
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--stop-at", type=float, default=1.0,
                    help="stop once complexity exceeds this (0 = never)")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    mut = 0.0 if args.no_mutation else args.mutation
    soup = Soup(args.n, seed=args.seed, mutation_prob=mut)

    if not args.skip_check:
        print("cross-checking C core against bff.py ...", end=" ", flush=True)
        cross_check(soup.lib)
        print("ok")

    print(f"soup: {args.n} tapes x {TAPE} bytes | mutation {mut:.6g} "
          f"| seed {args.seed} | steps {DEFAULT_STEPS}")
    print(f"complexity metric: Shannon - compressed, using {COMPRESSOR}")
    print()
    print(f"{'epoch':>7} {'complexity':>11} {'ops/inter':>10} {'top tape':>9}  {'elapsed':>8}")

    rows = []
    t0 = time.time()
    transition_at = None
    for _ in range(args.epochs):
        ops = soup.step()
        if soup.epoch % args.every == 0 or soup.epoch == 1:
            c = soup.complexity()
            top_count = soup.most_common(1)[0][1]
            per_inter = ops / (soup.n / 2)
            rows.append((soup.epoch, c, per_inter, top_count))
            print(f"{soup.epoch:>7} {c:>11.4f} {per_inter:>10.1f} "
                  f"{top_count:>9} {time.time() - t0:>7.1f}s")
            if transition_at is None and c >= args.stop_at > 0:
                transition_at = soup.epoch
                break

    print()
    if transition_at:
        print(f"*** state transition: complexity crossed {args.stop_at} "
              f"at epoch {transition_at} ***")
    else:
        print(f"no transition within {args.epochs} epochs "
              f"(the paper sees one in ~40% of runs within 16k epochs)")

    print("\nmost common whole tapes:")
    for tape, count in soup.most_common(3):
        exact, run = test_selfrep(tape, soup.lib)
        print(f"  x{count:<5} |{render(tape)}|")
        print(f"         exact self-copy {exact:.0%} of trials, "
              f"median {run:.0f}/64 bytes copied into the food tape")

    print("\nmost common 16-byte substrings (the replicating core):")
    for kmer, count in dominant_kmer(soup.tapes(), k=16):
        print(f"  x{count:<5} |{render(kmer)}|")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("epoch,complexity,ops_per_interaction,top_tape_count\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]:.6f},{r[2]:.3f},{r[3]}\n")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

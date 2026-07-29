#!/usr/bin/env python3
"""Sweep ``soup.py`` over a range of seeds, several runs at a time.

Runs

    ./soup.py --n 131072 --seed S --epochs 100000

for every seed in a range, ten at a time by default, and appends a digest of
each run to ``experiments_auto.txt`` in the format of the hand-kept
``experiments.txt``: the command, the last row of the progress table, the
transition verdict, and the two "most common" blocks.

    ./scan_auto.py                      # seeds 0..1000, 10 jobs
    ./scan_auto.py --last 20 --jobs 4   # a short trial run
    ./scan_auto.py                      # again: picks up where it left off

**This takes days, not hours.** A run that never transitions does the full
100000 epochs, which is about five and a half hours under ten-way load; a run
that transitions early is proportionally shorter. Ten at a time over 1001
seeds comes to something like five days, and rather more if few of them
transition. The scan is therefore written to be interrupted and resumed:
every completed run is appended and flushed immediately, and starting the
script again skips the seeds already in the output file. Ctrl-C kills the
running children and leaves the file valid.

Entries land in the order the runs *finish*, not in seed order - a run that
transitions after 2000 epochs is done hours before one that goes the
distance, and holding results back to sort them would mean writing nothing
for hours at a time. Every entry begins with its own command line, so the
seed is never in doubt.
"""

import argparse
import pathlib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent

# A completed entry starts with the command that produced it; a failed one is
# commented out with a "#", so that a resumed scan retries it rather than
# taking the failure for an answer.


def build_library() -> None:
    """Compile ``libsoup.so`` once, before any worker starts.

    ``soup.py`` builds it on demand when the shared object is missing or older
    than ``soup.c``. Ten workers starting at once would all find it missing and
    all run gcc onto the same path at the same time, and the loser of that race
    loads a half-written library. Doing it here, once, in a single process, is
    the whole fix.
    """
    sys.path.insert(0, str(HERE))
    import soup                      # noqa: E402  (path has to be set first)
    soup._load_lib()


def command(seed: int, n: int, epochs: int) -> list[str]:
    """The command line for one run - what is run and what is recorded."""
    return ["./soup.py", "--n", str(n), "--seed", str(seed),
            "--epochs", str(epochs)]


def digest(output: str) -> str:
    """Cut one run's output down to what ``experiments.txt`` keeps.

    That is: the *last* row of the progress table - the state the soup ended
    in - then the transition verdict, then the two "most common" blocks. The
    header, the per-epoch rows leading up to the last one and the cross-check
    line are dropped.

    :return: the digest, or the whole output if it could not be parsed, so
        that a run whose format has drifted is recorded rather than lost.
    """
    lines = output.splitlines()

    # the progress table: a header of column names, then rows, then a blank
    header = next((i for i, line in enumerate(lines)
                   if line.split()[:2] == ["epoch", "complexity"]), None)
    verdict = next((i for i, line in enumerate(lines)
                    if line.startswith("***") or line.startswith("no transition")),
                   None)
    tail = next((i for i, line in enumerate(lines)
                 if line.startswith("most common whole tapes:")), None)
    if header is None or verdict is None or tail is None:
        return output.strip()

    rows = [line for line in lines[header + 1:verdict] if line.strip()]
    last = rows[-1] if rows else "(no epoch completed)"
    body = "\n".join(lines[tail:]).rstrip()
    return "%s\n\n%s\n\n%s" % (last, lines[verdict], body)


def done_commands(path: pathlib.Path) -> set[str]:
    """The command lines already recorded in *path*, for resuming a scan.

    Matched whole rather than by seed, so that re-running the same seeds at a
    different ``--n`` or ``--epochs`` into the same file does the work again
    instead of taking the old answer for the new question.
    """
    if not path.exists():
        return set()
    with path.open() as handle:
        return {line.strip() for line in handle
                if line.startswith("./soup.py ")}


class Scan:
    """The runs, the output file, and the progress reporting."""

    def __init__(self, args):
        self.args = args
        self.out = pathlib.Path(args.out)
        if not self.out.is_absolute():
            self.out = HERE / self.out
        self.lock = threading.Lock()      # guards the file and the counters
        self.live: dict[int, subprocess.Popen] = {}
        self.stopping = False
        self.finished = 0
        self.failed = 0
        self.started_at = time.time()
        self.total = 0

    # ----------------------------------------------------------------
    def run_one(self, seed: int) -> tuple[int, bool, float]:
        """Run one seed to completion and append its digest.

        :return: ``(seed, ok, seconds)``.
        """
        argv = command(seed, self.args.n, self.args.epochs)
        began = time.time()
        with self.lock:
            if self.stopping:
                return seed, False, 0.0
            process = subprocess.Popen(argv, cwd=HERE, text=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            self.live[seed] = process
        try:
            output, errors = process.communicate()
        finally:
            with self.lock:
                self.live.pop(seed, None)

        took = time.time() - began
        ok = process.returncode == 0
        if ok:
            entry = "%s\n\n%s\n" % (" ".join(argv), digest(output))
        elif self.stopping:
            return seed, False, took     # killed by us; nothing to record
        else:
            # commented out, so a resumed scan retries this seed instead of
            # taking the failure for an answer
            note = (errors or output or "").strip().splitlines()
            entry = ("# FAILED %s (exit %s)\n#   %s\n"
                     % (" ".join(argv), process.returncode,
                        "\n#   ".join(note[-5:]) or "no output"))
        self.append(entry)
        return seed, ok, took

    # ----------------------------------------------------------------
    def append(self, entry: str) -> None:
        """Append one entry and flush, so an interrupted scan loses nothing."""
        with self.lock:
            with self.out.open("a") as handle:
                handle.write(entry + "\n")
                handle.flush()

    def report(self, seed: int, ok: bool, took: float) -> None:
        """One line per finished run, with an estimate of what is left."""
        with self.lock:
            self.finished += 1
            self.failed += 0 if ok else 1
            done, failed = self.finished, self.failed
        elapsed = time.time() - self.started_at
        # the rate so far, which is what the remaining runs will go at too
        left = (self.total - done) * elapsed / done if done else 0.0
        print("[%4d/%4d] seed %-4d %-6s %6.0fs | elapsed %s | eta %s%s"
              % (done, self.total, seed, "ok" if ok else "FAILED", took,
                 hours(elapsed), hours(left),
                 "" if not failed else " | %d failed" % failed),
              file=sys.stderr, flush=True)

    def stop(self, grace: float = 5.0) -> None:
        """Kill every running child, for Ctrl-C.

        SIGTERM first, then SIGKILL for whatever is still up. A soup spends
        most of its time inside the C core, and python cannot run a signal
        handler until that call returns - so a run with a big tape can take a
        few seconds to notice it has been asked to stop, and one that is wedged
        would never notice at all.
        """
        with self.lock:
            self.stopping = True
            children = list(self.live.values())
        for process in children:
            process.terminate()
        deadline = time.time() + grace
        for process in children:
            while process.poll() is None and time.time() < deadline:
                time.sleep(0.05)
            if process.poll() is None:
                process.kill()


def hours(seconds: float) -> str:
    """Seconds as h:mm, which is the scale this scan works on."""
    return "%d:%02d" % (int(seconds) // 3600, int(seconds) % 3600 // 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", type=int, default=0, help="first seed")
    parser.add_argument("--last", type=int, default=1000,
                        help="last seed, inclusive")
    parser.add_argument("--n", type=int, default=131072, help="tapes per soup")
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--jobs", type=int, default=10,
                        help="runs in parallel. Ten is not ten times one run: "
                             "the soup is memory-bound, so ten together go at "
                             "about six and a half times the speed of one")
    parser.add_argument("--out", type=str, default="experiments_auto.txt")
    parser.add_argument("--redo", action="store_true",
                        help="run every seed again, even ones already in the "
                             "output file")
    args = parser.parse_args()

    scan = Scan(args)
    wanted = list(range(args.first, args.last + 1))
    if not args.redo:
        already = done_commands(scan.out)
        recorded = {seed: " ".join(command(seed, args.n, args.epochs))
                    for seed in wanted}
        skipped = [seed for seed in wanted if recorded[seed] in already]
        wanted = [seed for seed in wanted if recorded[seed] not in already]
        if skipped:
            print("resuming: %d of %d seeds already in %s"
                  % (len(skipped), len(skipped) + len(wanted), scan.out.name),
                  file=sys.stderr)
    scan.total = len(wanted)
    if not wanted:
        print("nothing to do", file=sys.stderr)
        return 0

    print("compiling libsoup.so ...", file=sys.stderr, flush=True)
    build_library()

    print("%d seeds, %d at a time, into %s"
          % (len(wanted), args.jobs, scan.out.name), file=sys.stderr, flush=True)

    interrupted = False
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(scan.run_one, seed): seed for seed in wanted}
        try:
            for future in as_completed(futures):
                scan.report(*future.result())
        except KeyboardInterrupt:
            interrupted = True
            print("\ninterrupted - stopping the running soups; %s keeps every "
                  "run that finished" % scan.out.name, file=sys.stderr)
            scan.stop()
            for future in futures:
                future.cancel()

    print("%s: %d finished, %d failed, %s elapsed"
          % ("interrupted" if interrupted else "done",
             scan.finished - scan.failed, scan.failed,
             hours(time.time() - scan.started_at)), file=sys.stderr)
    return 1 if interrupted or scan.failed else 0


if __name__ == "__main__":
    sys.exit(main())

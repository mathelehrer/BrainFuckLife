#!/usr/bin/env python3
"""BFF — the self-modifying Brainfuck variant from Agüera y Arcas et al. 2024.

"Computational Life: How Well-formed, Self-replicating Programs Emerge from
Simple Interaction", arXiv:2406.19108.

The difference from plain Brainfuck (see ../bf.py) is that there is no
separate data tape, no input and no output. Code and data are the *same*
byte array, so a program can read and rewrite its own instructions. The
input/output commands are replaced by copies between two data heads.

    tape      one byte per cell; for soup experiments it is two 64-byte
              programs concatenated, so 128 bytes
    pc        instruction pointer, reads tape[pc]
    head0     data head 0
    head1     data head 1

    <   head0 -= 1              >   head0 += 1
    {   head1 -= 1              }   head1 += 1
    -   tape[head0] -= 1        +   tape[head0] += 1
    .   tape[head1] = tape[head0]        (copy head0 -> head1)
    ,   tape[head0] = tape[head1]        (copy head1 -> head0)
    [   if tape[head0] == 0: jump forward past the matching ]
    ]   if tape[head0] != 0: jump back after the matching [

Instructions are dispatched on their **literal ASCII byte values** — `<` is
0x3C, `[` is 0x5B and so on. Every other byte is a no-op, so only 10 of the
256 possible byte values do anything and byte 0 is the "true zero" that ends
loops. That sparseness is the point: a uniformly random 64-byte tape carries
only ~2.5 live instructions, which is what makes the pre-life soup quiet
enough for a replicator to be a dramatic event.

Execution stops when the pc runs off either end of the tape, when a bracket
has no match, or after `stepcount` characters have been read (2**13 by
default) — whichever comes first. Heads wrap around the tape; `+`/`-` wrap
mod 256.

This module is a direct port of `bff.inc.h` from the authors' reference
implementation (github.com/paradigms-of-intelligence/cubff), specifically
the `bff_noheads` variant, which is the one described in the paper's
Section 2. It is written to be read rather than to be fast; `soup.py` uses
an equivalent C core for the actual experiments and cross-checks it against
this one.

Run as a script it takes one argument: either --selfrep, which runs the
paper's predefined self-replicating program, or the path to a file holding a
program of your own. Either way the program is executed against a blank tape.
With no argument this text is printed.
"""
from __future__ import annotations

# The 10 live opcodes. Every other byte value is a no-op.
OPS = frozenset(b"<>{}-+.,[]")  # {43,44,45,46,60,62,91,93,123,125}

TAPE_SIZE = 64  # one program
MEM_SIZE = 2 * TAPE_SIZE  # two programs concatenated = the executed tape
MAX_STEPS = 1 << 13  # 2**13 characters read before we give up


def evaluate(tape: bytearray, stepcount: int = MAX_STEPS,
             trace: list | None = None) -> int:
    """Run `tape` in place. Returns the number of non-no-op ops executed.

    `tape` is modified in place — that *is* the experiment. If `trace` is a
    list, one dict per executed character is appended to it.
    """
    n = len(tape)
    mask = n - 1
    if n & mask: # bitwise and (standard power of two test)
        raise ValueError("tape length must be a power of two")

    # bff_noheads: both heads and the pc start at the left edge. (The C code
    # writes `2 * kSingleTapeSize` here and relies on the mask below turning
    # it into 0; we start at 0 directly, which is the same thing.)
    pc = 0
    head0 = 0
    head1 = 0

    nskip = 0  # characters read that turned out to be no-ops
    i = 0
    while i < stepcount:
        head0 &= mask # wrap around the tape head0%n  -1->111111111->mask
        head1 &= mask

        cmd = tape[pc]
        did_something = True

        if cmd == 0x3C:  # <
            head0 -= 1
        elif cmd == 0x3E:  # >
            head0 += 1
        elif cmd == 0x7B:  # {
            head1 -= 1
        elif cmd == 0x7D:  # }
            head1 += 1
        elif cmd == 0x2B:  # +
            tape[head0] = (tape[head0] + 1) & 0xFF
        elif cmd == 0x2D:  # -
            tape[head0] = (tape[head0] - 1) & 0xFF
        elif cmd == 0x2E:  # .   copy head0 -> head1
            tape[head1] = tape[head0]
        elif cmd == 0x2C:  # ,   copy head1 -> head0
            tape[head0] = tape[head1]
        elif cmd == 0x5B:  # [
            if tape[head0] == 0:
                # Scan forward for the matching ']', landing *on* it; the
                # pc += 1 at the bottom of this loop then steps past it.
                depth = 1
                pc += 1
                while pc < n and depth > 0:
                    if tape[pc] == 0x5D:
                        depth -= 1
                    elif tape[pc] == 0x5B:
                        depth += 1
                    pc += 1
                pc -= 1
                if depth != 0:
                    pc = n  # unmatched -> run off the end, stop
        elif cmd == 0x5D:  # ]
            if tape[head0] != 0:
                # Scan backward for the matching '[', landing on it; the
                # pc += 1 below re-enters the loop body.
                depth = 1
                pc -= 1
                while pc >= 0 and depth > 0:
                    if tape[pc] == 0x5D:
                        depth += 1
                    elif tape[pc] == 0x5B:
                        depth -= 1
                    pc -= 1
                pc += 1
                if depth != 0:
                    pc = -1  # unmatched -> stop
        else:
            did_something = False

        if trace is not None:
            # `tape` is the state *after* this character was executed, which is
            # what a timeline wants to plot.
            trace.append({
                "step": i, "pc": pc, "cmd": chr(cmd) if cmd in OPS else None,
                "head0": head0 & mask, "head1": head1 & mask,
                "op": did_something, "tape": bytes(tape),
            })

        if not did_something:
            nskip += 1

        i += 1
        if pc < 0:
            break
        pc += 1
        if pc >= n:
            break

    return i - nskip


def run_pair(a: bytes, b: bytes, stepcount: int = MAX_STEPS):
    """Concatenate two 64-byte programs, execute, split the result again.

    This is the soup's entire interaction rule:
        split(exec(AB)) -> A' + B'
    Returns (a_out, b_out, n_ops).
    """
    if len(a) != TAPE_SIZE or len(b) != TAPE_SIZE:
        raise ValueError(f"both programs must be {TAPE_SIZE} bytes")
    tape = bytearray(a) + bytearray(b)
    n_ops = evaluate(tape, stepcount)
    return bytes(tape[:TAPE_SIZE]), bytes(tape[TAPE_SIZE:]), n_ops


def render(tape: bytes) -> str:
    """Show a tape the way the paper's figures do: opcodes and zeros only."""
    return "".join(
        chr(c) if c in OPS else ("0" if c == 0 else " ")
        for c in tape
    )


# The self-replicator the authors found emerging at epoch 2354 of their
# case-study run (paper Figure 12 / replicator_run.tex). Note the tail is the
# head reversed. Padding is spaces (0x20), which are no-ops.
PAPER_REPLICATOR = (b"[[{.>]-]" + b" " * 48 + b"]-]>.{[[")


def main() -> None:
    """Run one program against a blank tape and report whether it replicated.

    Usage::

        python3 bff.py                # print the module docstring and exit
        python3 bff.py --self-rep # run PAPER_REPLICATOR
        python3 bff.py PROGRAM_FILE   # run the program stored in a file

    The second option, `PROGRAM_FILE`, is the "bring your own tape" mode: the
    file is read as raw bytes, so it may contain any of the 256 byte values
    and not just printable opcodes. It is normalised to exactly one program of
    TAPE_SIZE (64) bytes — trailing newlines are stripped first, so a file
    written by an ordinary text editor does not end up with a stray 0x0A; then
    the tape is right-padded with spaces (0x20, a no-op) if it is short and
    truncated if it is long. Truncation is silent, so a file of more than 64
    bytes loses its tail without warning.

    In both modes the program becomes tape A and tape B is 64 zero bytes — the
    "food" tape. The pair is concatenated and executed as one 128-byte tape
    (see `run_pair`), which is exactly the interaction two programs undergo in
    the soup, except that here the partner is blank. Printing `B' out` next to
    `A in` therefore answers the question the paper cares about: did the
    program copy itself into the empty half? `replicated: True` means B' is
    byte-for-byte the input program.

    This is why PAPER_REPLICATOR needs its mirrored tail. Running only the
    head, `[[{.>]-]`, writes the head *reversed* into the far end of B and
    stops there, reporting `replicated: False`; it is the full head-plus-
    reversed-tail tape that reproduces itself.
    """
    import sys
    args = sys.argv[1:]
    if args and args[0] == "--self-rep":
        prog = PAPER_REPLICATOR
    elif args:
        prog = open(args[0], "rb").read().rstrip(b"\n")
        prog = prog.ljust(TAPE_SIZE, b" ")[:TAPE_SIZE] # fill with spaces and truncate at TAPE_SIZE
    else:
        print(__doc__)
        # print(main.__doc__)
        return

    food = bytes(TAPE_SIZE)  # a tape of zeros
    a, b, n_ops = run_pair(prog, food)
    print(f"A  in : |{render(prog)}|")
    print(f"B  in : |{render(food)}|")
    print(f"ops   : {n_ops}")
    print(f"A' out: |{render(a)}|")
    print(f"B' out: |{render(b)}|")
    print(f"replicated: {b == prog}")


if __name__ == "__main__":
    main()

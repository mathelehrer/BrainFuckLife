# brainfuck

A minimal Brainfuck interpreter, no dependencies beyond Python 3.

## Usage

```sh
python3 bf.py examples/hello.bf
python3 bf.py -c ',[.,]' < input.txt   # inline program
```

Programs that use `,` (read input) pull from stdin.

## Files

- `bf.py` — the interpreter
- `examples/hello.bf` — prints "Hello World!"
- `examples/cat.bf` — echoes stdin back out
- `viz/trace.py` — traces a program's execution (one row per command, with
  runs of consecutive `+`/`-` collapsed into a single step) and dumps JSON
  with, per step, the touched cell, its value, current loop depth, and any
  output produced: `python3 viz/trace.py examples/hello.bf > trace.json`
- `bff/` — **BFF**, the self-modifying Brainfuck variant from Agüera y Arcas
  et al. 2024 (arXiv:2406.19108), plus the primordial-soup experiment in
  which self-replicating programs emerge from random noise. See
  `bff/README.md`.
- `viz/hello_trace.html` — a standalone, open-in-any-browser visualization
  of `examples/hello.bf`'s run: a heatmap with cells across and time down
  the page, a loop-depth stripe on the left, and a zoomed detail table for
  the printing phase

## How `examples/hello.bf` works

```
++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]
>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.
```

The program has two phases: build up ASCII values via nested loops, then
print them out, tweaking each cell by a few `+`/`-` on the way.

**Phase 1 — multiplication loop.** `++++++++` sets cell 0 to 8, then the
outer loop runs 8 times. Each pass sets cell 1 to 4 and uses an inner loop
(which runs 4 times, once per unit in cell 1) to add fixed amounts to cells
2–5: `+2`, `+3`, `+3`, `+1` per inner iteration. Over all 8 outer passes that
accumulates:

| cell | value              | used for |
|------|--------------------|----------|
| 2    | 8 × (2×4+1) = 72   | `H`      |
| 3    | 8 × (3×4+1) = 104  | `e`/`l`/`o`/`o`/`r`/`l`/`d` (reused) |
| 4    | 8 × (3×4−1) = 88   | `W`      |
| 5    | 8 × (1×4) = 32     | space/`!` (reused) |
| 6    | 8 (iteration count)| newline  |

(The `+1`/`+1`/`-1` after each inner loop and the `>+` on cell 6 come from
`>+>+>->>+` at the end of the outer loop body; `[<]<-` then walks the
pointer back to cell 0 and decrements the outer counter.)

**Phase 2 — print with adjustments.** The pointer ends at cell 0. From
there the remaining code walks across cells 2–6, printing each one and
nudging it by a small amount immediately before printing so the *same*
cell can be reused for several letters:

- `>>.` → cell 2 = 72 → `H`
- `>---.` → cell 3: 104−3=101 → `e`
- `+++++++.` → cell 3: 101+7=108 → `l`
- `.` → cell 3 unchanged (108) → `l`
- `+++.` → cell 3: 108+3=111 → `o`
- `>>.` → cell 5 = 32 → ` `
- `<-.` → cell 4: 88−1=87 → `W`
- `<.` → cell 3 unchanged (111) → `o`
- `+++.` → cell 3: 111+3=114 → `r`
- `------.` → cell 3: 114−6=108 → `l`
- `--------.` → cell 3: 108−8=100 → `d`
- `>>+.` → cell 5: 32+1=33 → `!`
- `>++.` → cell 6: 8+2=10 → `\n`

That prints `Hello World!\n`, reusing just two "scratch" cells (3 and 5)
for six of the letters instead of dedicating one cell per character.

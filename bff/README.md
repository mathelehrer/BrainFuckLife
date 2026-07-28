# BFF — self-replicating programs from a soup of noise

A reproduction of the primordial-soup experiment from **"Computational Life:
How Well-formed, Self-replicating Programs Emerge from Simple Interaction"**,
Agüera y Arcas, Alakuijala, Evans, Laurie, Mordvintsev, Niklasson, Randazzo,
Versari — [arXiv:2406.19108](https://arxiv.org/abs/2406.19108). The paper
itself is not redistributed here; get it from arXiv.

## The language

BFF ("Brainfuck Family") makes Brainfuck **embodied**: there is no separate
data tape, no input and no output. Code and data are the same byte array, so
a program can read and rewrite its own instructions while running. The `,`
and `.` I/O commands become copies between two data heads.

The instruction pointer `pc`, and both heads `head0` and `head1`, all index
the same `tape`.

| op  | effect                     | op  | effect                     |
|-----|----------------------------|-----|----------------------------|
| `<` | `head0 -= 1`               | `>` | `head0 += 1`               |
| `{` | `head1 -= 1`               | `}` | `head1 += 1`               |
| `-` | `tape[head0] -= 1`         | `+` | `tape[head0] += 1`         |
| `,` | `tape[head0] = tape[head1]` | `.` | `tape[head1] = tape[head0]` |
| `[` | if `tape[head0] == 0` jump past matching `]` | `]` | if `tape[head0] != 0` jump back after matching `[` |

Execution stops when the `pc` runs off either end of the tape, when a
bracket has no match, or after 2^13 characters have been read. Heads wrap
around the tape; `+`/`-` wrap mod 256.

### Encoding: it really is plain ASCII

A natural guess is that such a dense soup must use a compact opcode encoding
— 4 bits per instruction, or `byte % 16`. **It does not.** Dispatch is on the
literal ASCII values: `<` is 0x3C, `[` is 0x5B, and so on. Confirmed against
the authors' reference implementation
([cubff](https://github.com/paradigms-of-intelligence/cubff), `bff.inc.h`),
whose `GetOpKind` is a plain `switch` over those characters, with byte 0 as
the "true zero" that terminates loops and *every other byte a no-op*.

The sparseness is the whole point, not an oversight. Only 10 of 256 byte
values do anything, so a uniformly random 64-byte tape carries only about
2.5 live instructions. That is what keeps the "pre-life" soup quiet enough
that the arrival of a replicator is a sharp, visible phase transition rather
than noise. A denser encoding would have random tapes computing furiously
from epoch zero.

> In the talk, Agüera y Arcas says he "reduced it from eight instructions to
> seven" — that describes an earlier personal variant and does not match the
> published language, which has ten. His remark that "only one in 32 or so"
> random bytes are valid instructions is a loose reading of 10/256 ≈ 1/26.
> The paper and cubff agree with each other; this implementation follows
> them, specifically the `bff_noheads` variant of the paper's Section 2,
> where both heads and the `pc` start at 0.

## The experiment

    Fill N tapes of 64 bytes with uniform random noise.
    Repeatedly: pick two at random, concatenate into 128 bytes, run that as
    a BFF program, split the result back into two tapes, return to the soup.

That is all of it. No fitness function, no selection, no goal — programs
change only by overwriting themselves and each other. The paper adds a
background mutation rate of 1/4096 per byte per epoch by default, but shows
the transition also happens with mutation switched off entirely.

Progress is tracked with the paper's **high-order entropy**: the Shannon
entropy of the soup's bytes minus its compressed size in bits per byte.
Uniform noise scores ≈ 0 (high entropy, incompressible); a soup taken over by
copies of one program scores well above 1 (compressible, but its bytes are
still varied).

## Files

- `Youtube.md` — planning notes for turning this into a video: what is worth
  animating in 3D, what should stay flat, and the two open items that block a
  shootable script.
- `BFF_walkthrough.ipynb` — **start here.** An illustrated walkthrough of the
  whole thing: the language and why its encoding matters, the paper's
  replicator traced byte by byte, and an interactive soup you can start and
  watch cross over. Ships with outputs, so the diagrams are visible without
  running anything.
- `bff.py` — the interpreter, written to be read. A direct port of cubff's
  `bff.inc.h`, including its exact bracket-scanning and termination rules.
- `soup.c` — the same interpreter plus the epoch loop in C, ~100× faster,
  which is what makes reaching the transition practical.
- `soup.py` — the experiment driver: builds `soup.c` on first use, runs
  epochs, measures complexity, and tests the dominant tapes for actual
  self-replication.

## Usage

```sh
python3 bff.py --selfrep          # run the paper's replicator against a blank tape
python3 soup.py                   # 1024 tapes, run until the transition
python3 soup.py --n 4096 --seed 3 --csv run.csv
python3 soup.py --no-mutation     # self-modification only, no background noise
python3 takeover.py --seed 4      # seeded takeover test, with a random control
```

For the notebook, use the workspace venv:

```sh
cd bff && ../../.venv/bin/python -m jupyter lab BFF_walkthrough.ipynb
```

`jupyterlab`, `ipykernel`, `ipywidgets` and `brotli` are declared in the
workspace `pyproject.toml` and pinned in `uv.lock`, so `uv sync` keeps them.
If the venv is ever rebuilt from scratch:

```sh
uv sync
```

`soup.py` cross-checks the C core against `bff.py` on 200 random tapes at
startup (skip with `--skip-check`).

## Validation

The paper's case study reports a replicator that emerged at epoch 2354,
shown byte-by-byte in `replicator_run.tex`:

```
[[{.>]-]                                                ]-]>.{[[
```

(the tail is the head reversed; the padding is spaces, which are no-ops).
Running it against a tape of zeros reproduces the paper's figure exactly —
slot A is left unchanged and slot B comes out a perfect copy:

```
$ python3 bff.py --selfrep
A  in : |[[{.>]-]                                                ]-]>.{[[|
B  in : |0000000000000000000000000000000000000000000000000000000000000000|
A' out: |[[{.>]-]                                                ]-]>.{[[|
B' out: |[[{.>]-]                                                ]-]>.{[[|
replicated: True
```

`soup.py`'s self-replication test uses this same criterion (the paper's
autocatalytic reaction `S + F -> 2S`) and checks **only** slot B — an inert
program leaves slot A untouched, which would otherwise read as a perfect
replication.

## Results

Six runs at 4096 tapes, 40k epochs, default mutation. One transitioned
(seed 4, at epoch 8750) — in the same ballpark as the paper's ~40% within
16k epochs at the much larger soup size of 2^17.

Seed 4 reproduces the paper's characteristic curve: complexity rises to
~0.5 over the first ~750 epochs, decays back to ~0.1 for thousands of
epochs of "pre-life", then jumps to >3 within a single 250-epoch window.

```
   8400      0.1086      318.9        # noise
   8800      3.1322      135.1        # life
   9200      3.8435      328.7
```

The emergent replicator has the same reversed-tail shape as the paper's:

```
    [      <,, }             ]]             } ,,<      [   , , ,
```

`takeover.py` confirms it is genuinely self-reproducing rather than just
what the soup decayed into. Injected into 1% of a *fresh* noise soup it
spreads to ~42% within 60 epochs and drives complexity from 0.04 to 4.02,
while a random tape injected identically decays from 1% to 0.6% and leaves
complexity at 0.23.

### Two traps worth knowing about

Both cost real debugging time here, and both silently produce
plausible-looking wrong answers:

1. **Exact whole-tape matching badly undercounts replicators.** The paper
   warns that a replicator is usually shorter than the 64-byte window, so
   one copying itself at an offset ≠ 64 replicates perfectly while never
   producing an exact full-tape copy. The dominant tapes above score 0% on
   an exact test; counting k-mers instead finds them immediately.
2. **`render()` is not injective.** It draws every non-opcode byte as a
   space, so two cores can look byte-identical on screen while their filler
   bytes differ completely. Any probe or comparison must run on raw bytes,
   never on the rendered string.
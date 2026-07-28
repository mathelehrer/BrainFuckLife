# Video notes — turning BFF into a YouTube piece

Planning notes for a video on Agüera y Arcas et al., *Computational Life*
(arXiv:2406.19108), built on the code in this folder. Written to be picked up
cold: the numbers you would otherwise have to re-derive are included inline.

---

## 1 · The through-line

Everything should serve one claim:

> **A program that copies itself is more stable than one that doesn't — so noise
> plus a self-modifying language is enough to get life.**

The emotional payload is the moment a machine builds a copy of itself. Budget the
best animation there, not on the statistics.

A useful framing for the script, from the talk: this is the second law of
thermodynamics producing something *more* complex rather than less, because a
cycle can be more stable than a fixed point. Replicators are not favoured by any
selection rule here — they simply persist, and everything else gets overwritten.

---

## 2 · Worth animating — tier 1

### 2.1 The replication itself, as a physical machine

**This is the video.** Everything else supports it.

- 128 cells in a row, a visible seam at position 64.
- Three gantries riding the rail: `pc` (executing), `head0` (reads),
  `head1` (writes).
- When `.` fires, a byte physically flies from head0's cell to head1's cell.
- Run it and the right half — dead black zeros — fills in with a mirror of the left.

**Hard numbers:** tape B is a complete, byte-perfect copy at **step 255**. The
program keeps running to the 8192-character limit, so cut at ~280 steps. All of
this is already traceable via `bff.evaluate(tape, trace=[...])`, which records
`pc`, `head0`, `head1`, the executed opcode and a full tape snapshot per step.

### 2.2 The interaction loop as mechanics

Two 64-cell tapes fly in → snap together into 128 → the machine runs → they split
and drop back into the pool. Loop it.

Establishes the entire experimental rule in ten wordless seconds, and it re-uses
well as a transition throughout the video.

### 2.3 The mirror fold

```
[[{.>]-]                                                ]-]>.{[[
```

Literally fold the tape at its midpoint and show the tail landing on the head
reversed. Two-second beat, makes the structure legible — and it pays off twice,
because the replicators that emerge in our own runs have the same shape:

```
    [      <,, }             ]]             } ,,<      [   , , ,
```

### 2.4 Sparsity, fast

256 tiles, 10 light up. Then a random 64-byte tape with ~2.5 live instructions.

Five seconds, and it earns the audience's surprise later when the soup goes
berserk. It also sets up the point that **the ASCII encoding is load-bearing** —
under a dense 4-bit encoding every random tape would compute furiously from epoch
zero and there would be no quiet baseline to transition *away* from.

---

## 3 · Worth animating — tier 2

### 3.1 Tape-over-time as terrain

The one place this data is *honestly* 3D: position × time × byte value. Extrude the
notebook's §2 heatmap into a height-field and fly along the time axis. Loops become
visible as periodic ridges.

### 3.2 Byte genealogy

The paper's tracer-token idea: tag every byte with where it came from, then follow
descent. A handful of ancestral bytes end up owning the whole soup.

**This is the differentiator** — nobody has animated it well, and it makes the
"ancestry" claim visual rather than asserted.

---

## 4 · Keep these flat

Opinionated, and worth holding to:

- **The phase-transition plot.** A 2D line chart, animated by drawing in, synced to
  the soup. Do not perspective it, do not make it a 3D ribbon. It is the *evidence*
  — it should read as evidence.
- **The 0-D soup wall.** 4096 tapes paired at random have **no spatial structure**.
  Rendering them as a 3D grid invents adjacency that does not exist and quietly
  lies to the viewer. Use a flat wall of strips with colour sweeping through at the
  transition.

---

## 5 · Build this first: the 2D spatial soup

If you want genuine spatial drama, implement the paper's 2D variant — tapes on a
grid (they used 240×135) interacting **only with neighbours**.

There, adjacency is real, so:

- replicators spread as a visible **wavefront** rather than instantly;
- rival lineages fight over **territory**;
- the takeover takes ~√n epochs instead of ~log n, so it is watchable.

That is the footage that justifies 3D camera work. It is a modest change to
`soup.py`: replace the global shuffle with neighbour selection inside `run_epoch`.

> **Differentiate:** the authors already published a 2D soup video (linked in their
> §2.4). Aim somewhere they did not — colour by **lineage** rather than by raw byte
> value, so competing families read as distinct territories.

---

## 6 · Fitting the Blender pipeline

### 6.1 Bake the trace to a data texture

The cleanest way to drive this from geometry nodes without per-frame Python:

1. Encode the run as a **PNG**: row = execution step, column = tape position,
   red channel = byte value.
2. In GN, index the row by frame number and read byte values per point.
3. Instance a cube per point; drive colour and height from the sampled value.

Scrubbable, fast to render, no Python in the frame loop. The same trick works for
the soup — one image per epoch, or a single tall strip.

Store `pc` / `head0` / `head1` per frame in a small JSON sidecar and drive the three
gantries from that.

### 6.2 Existing tools that fit

- **`BMorphText`** — natural fit for showing a tape *mutating* across epochs. Morph
  the glyph string as bytes change, so the audience sees code drift rather than a
  jump cut.
- **Geometry nodes** — a tape is a 128-point line with a `byte` attribute; instance
  cubes, colour from a ramp. Prefer assembling existing GN modifiers over building
  new ones from scratch.

### 6.3 Colour

The notebook already uses a CVD-validated 4-family palette (checked with the
palette validator; worst all-pairs CVD ΔE 9.2, normal-vision 16.3):

| family | colour | ops |
|:--|:--|:--|
| head moves | `#2a78d6` blue | `< > { }` |
| arithmetic | `#eb6834` orange | `+ -` |
| copies | `#1baf7a` aqua | `. ,` |
| loops | `#4a3aa7` violet | `[ ]` |
| true zero | `#1a1a19` near-black | byte 0 |
| inert data | `#e8e7e1` pale grey | the other 245 |

Reusing it keeps notebook figures and rendered shots visually consistent. Note the
first pick for the fourth slot (magenta) hard-failed against orange — do not swap
hues without re-running the validator.

---

## 7 · Suggested structure

| # | Beat | Visual |
|:-:|:--|:--|
| 1 | The question — can life start from noise? | cold open on the transition, no explanation |
| 2 | Brainfuck in 60s | glyphs, one head, hello world |
| 3 | Why it *cannot* self-replicate | two separate tapes, a wall between them |
| 4 | The one change: one tape, two heads | the wall dissolves |
| 5 | The opcode space is empty | §2.4 sparsity beat |
| 6 | The rule | §2.2 interaction loop |
| 7 | **The replicator copies itself** | §2.1 — the centrepiece |
| 8 | How it works, instruction by instruction | *needs work — see §8* |
| 9 | Now do it 4096 times | soup wall + complexity plot in sync |
| 10 | The transition | epoch 8750, complexity 0.10 → 3.27 in one window |
| 11 | Is it really alive? | seeded takeover vs control |
| 12 | Where it generalises | Forth, Z80, 8080 — and the SUBLEQ failure |

---

## 8 · Open items

Two things stand between these notes and a shootable video:

1. **A trace exporter** — dump a run to the data-texture PNG plus the
   `pc`/`head0`/`head1` JSON sidecar described in §6.1. Straightforward; the trace
   data already exists.
2. **Reverse-engineer the replicator's mechanism.** We have verified *that*
   `[[{.>]-]…]-]>.{[[` copies itself, but **not** *how*, instruction by instruction.
   Beat 8 in the structure above cannot be written without it — and it is the part
   of the story no one else has explained clearly, so it is worth the effort.

Both are good next sessions.

---

## 9 · Numbers worth quoting on screen

| fact | value |
|:--|:--|
| live opcodes | 10 of 256 (~3.9%) |
| live instructions on a random 64-byte tape | ~2.5 |
| step at which the paper's replicator completes a copy | 255 |
| our transition (4096 tapes, seed 4) | epoch **8750** |
| complexity across that transition | 0.10 → 3.27 |
| runs that transitioned | 1 of 6 seeds within 40k epochs |
| paper's rate | ~40% within 16k epochs at 2¹⁷ tapes |
| seeded takeover | 1% → **59.4%** of soup, complexity 0.03 → 3.46 |
| random control, same procedure | 1% → 0.6%, complexity 0.03 → 0.21 |

Caveat worth keeping honest on screen: our 1-in-6 rate is at 4096 tapes, not the
paper's 2¹⁷, so it is not directly comparable to their ~40%.

---

## 10 · Sources

- Paper LaTeX source: `../AgueraYArcas/arXiv-2406.19108v2/`
- Reference implementation: [paradigms-of-intelligence/cubff](https://github.com/paradigms-of-intelligence/cubff)
- Authors' 2D soup video: `youtube.com/watch?v=07NoZwvgJ_M`
- Implementation and validation notes: `README.md` in this folder
- Interactive walkthrough: `BFF_walkthrough.ipynb`

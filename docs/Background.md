# Background: BFF, computational life, and replicator tracking

This document records the scientific and implementation background behind the
BFF experiment in this repository. It separates claims made by the original
2024 paper, later interpretations presented by Blaise Agüera y Arcas, the
functional detector in the CuBFF reference implementation, and the additional
checks used by this repository's live visualization.

The central distinction is between four events:

1. a potentially replicating program first appears;
2. that program can repeatedly reproduce functional structure;
3. descendants or related signatures increase in the population; and
4. the entire soup undergoes an aggregate state transition.

These events need not occur at the same epoch. A replicator can appear and go
extinct, a detector can observe it only after it has already spread, and an
entropy transition measures population structure rather than replication
directly.

## Source and claim boundaries

| Layer | What it supplies |
|---|---|
| [2024 Computational Life paper](https://arxiv.org/abs/2406.19108v2) | The BFF language, primordial-soup experiment, high-order entropy metric, traced examples, and the original emergence results. |
| [ALife 2025 presentation](https://www.youtube.com/watch?v=M2iX6HQOoLg) | A later interpretation in terms of symbiogenesis, merger ancestry, gelation, and broader claims about life and intelligence. |
| [2026 follow-up paper](https://arxiv.org/abs/2607.01483) | A functional self-replication score and experiments that test whether BFF interaction is an unusually effective or compositionally necessary search process. |
| [CuBFF reference implementation](https://github.com/paradigms-of-intelligence/cubff) | Executable language variants, simulation kernels, and the current `CheckSelfRep` assay mechanics. |
| This repository | A readable Python/C reproduction, conservative held-out verification, marker-based prevalence tracking, and a realtime dashboard. |

The later presentation and paper revise the interpretation of the original
result. They should not be read as if all claims appeared in, or were
established by, the 2024 paper.

## The research question

Many artificial-life systems begin with a hand-written ancestor that already
knows how to reproduce. BFF instead asks what can happen before such an
ancestor exists:

- initialize a population with uniformly random programs;
- provide no explicit fitness score, dedicated whole-program reproduction
  instruction, or prewritten reproduction routine;
- allow programs to execute on, and overwrite, themselves and one another;
- observe whether functional self-replicators appear and spread.

The 2024 paper treats the appearance and population-level dominance of
self-replicators as an operational transition from "pre-life" to "life"
dynamics. This is an experimental convention, not a proposed universal
definition of biological life.

There is no explicit fitness landscape in the experiment, but that does not
mean there is no selection. Tape slots and execution opportunities are
limited. A structure that makes persistent copies can occupy slots that would
otherwise contain structures that do not copy. Differential persistence and
amplification therefore arise from the interaction rule itself.

The original paper's narrower claim is that self-replicators can emerge under
these conditions through interaction and self-modification, both with and
without independent background mutation. The stronger claims in the later
talk - for example, that life is a functional phase of matter or that
symbiogenesis is the general source of evolutionary novelty - are theoretical
interpretations that go beyond what the BFF experiments directly establish.

## Historical context

BFF sits within a longer line of work on self-reproduction, autocatalysis, and
artificial life:

- von Neumann studied the logical requirements for a self-reproducing
  automaton, including the separation between a description and machinery
  able to copy and interpret it.
- Tierra and Avida allow assembly-like digital organisms to mutate and
  compete, but begin with a designed self-reproducing ancestor.
- Fontana's algorithmic chemistry studies interacting programs as an
  artificial chemistry capable of forming autocatalytic organizations.
- Earlier "prevolutionary" models ask how selection-like dynamics can exist
  before conventional heredity and reproduction are fully established.

What distinguishes the BFF setup is its attempt to begin with uniformly random
byte strings, without deliberately seeding a replicator, and obtain replicators
through the ordinary execution and modification rules of the substrate. The
initial population is not exhaustively screened, so the experiment does not
assume that a randomly sampled replicator is impossible.

## The BFF computational substrate

BFF is a family of Brainfuck-derived languages. The variant reproduced here
places code and data in the same mutable byte array. Its state consists of:

- an instruction pointer, `pc`;
- two data heads, `head0` and `head1`; and
- one byte array addressed by all three.

Because instructions and data share memory, a copy or arithmetic operation
can rewrite instructions that have not yet executed. This is the mechanism of
self-modification; there is no separate genome or protected code segment.

The repository implements ten instructions using their literal ASCII byte
values:

| Byte | Effect |
|---|---|
| `<`, `>` | Move `head0` left or right. |
| `{`, `}` | Move `head1` left or right. |
| `-`, `+` | Decrement or increment the byte under `head0`, modulo 256. |
| `.` | Copy the byte under `head0` to `head1`. |
| `,` | Copy the byte under `head1` to `head0`. |
| `[`, `]` | Conditionally scan to the matching delimiter using the byte under `head0`. |

All other byte values are no-ops. Byte zero is the false value used by the
loop instructions. Only 10 of 256 uniformly sampled values are active
instructions, so a random 64-byte tape initially contains about 2.5 active
instructions on average.

For a two-tape interaction, the data heads wrap around the 128-byte combined
memory. The instruction pointer does not wrap. Execution ends when it leaves
the combined memory, execution attempts to follow an unmatched loop delimiter,
or `2^13` evaluator steps have run. A no-op consumes an evaluator step; the
execution-activity metric reported by this repository excludes no-ops.

Exact semantics are implemented readably in
[`bff/bff.py`](../bff/bff.py) and independently in C in
[`bff/soup.c`](../bff/soup.c). The experiment cross-checks the Python and C
interpreters by default on random 128-byte tapes and on the known example
replicator; `--skip-check` disables this startup check.

## The primordial-soup experiment

The original paper's main configuration usually contains `2^17` tapes. Each
tape is 64 bytes initialized independently from a uniform byte distribution.
An ordered pair is concatenated, executed as 128 bytes, and split back into
two 64-byte population slots:

```text
A + B -> split(exec(AB)) = A' + B'
```

Order matters because `exec(AB)` and `exec(BA)` can behave differently. The
number of population slots remains fixed; programs gain representation by
overwriting existing slots, not by allocating new ones.

In this repository, one epoch performs the following operations:

1. shuffle all tape indexes;
2. pair adjacent indexes, so each tape interacts once;
3. execute each concatenated pair and write both results back; and
4. independently replace selected bytes with random values.

There are therefore `N/2` interactions per local epoch. The default local
mutation probability is `1/4096` per byte per epoch, approximately `0.0244%`.
The `--no-mutation` option isolates changes produced by program execution and
self-modification.

The current local mutation is applied after pair execution. This differs from
the [pinned CuBFF kernel](https://github.com/paradigms-of-intelligence/cubff/blob/f212e849027c98fcf4b242eccfb5fed435223e23/common_language.h#L157-L192),
which mutates each concatenated pair before evaluating it. Reproduction claims
should therefore identify the implementation and parameters used rather than
assuming every BFF experiment has an identical update order.

## Immediate and functional self-replication

The idealized immediate autocatalytic reaction is:

```text
S + F -> S + S
```

Here `S` survives execution and rewrites food tape `F` into another copy.
This simple equality is useful for a hand-inspected example, but the 2024
paper explicitly warns that a functional replicator is often smaller than
its 64-byte carrier. It may also copy at a different offset, reverse or
alternate its representation, preserve irrelevant carrier bytes, or
participate in a multi-step autocatalytic process.

Consequently, several notions must remain separate:

| Term | Meaning in this repository |
|---|---|
| Tape or carrier | One complete 64-byte population unit. |
| Representative | One complete carrier selected for functional testing. |
| Exact identity | Equality of all 64 raw bytes. |
| Marker or signature | An 8-16 byte substring used to track related carriers. |
| Functional score | Repeatable positional structure after a multigeneration assay. |

Exact whole-tape frequency can substantially undercount replicating code. At
the same time, a common substring is not automatically an autonomous
replicator. It may be inert structure copied by machinery elsewhere on the
carrier.

The text renderer is also intentionally lossy: it prints opcode bytes and
zero while displaying most other bytes as spaces. Two tapes that look the
same in a figure can differ in raw bytes. Detection and identity tracking must
operate on the raw byte strings, not rendered text.

### The sanitized example replicator

The repository includes a 64-byte example that the paper extracted from one
run and sanitized by replacing non-coding bytes with spaces:

```text
[[{.>]-]                                                ]-]>.{[[
```

The replacement preserves this example's behavior, but the displayed tape is
not a raw emergent carrier. Its active head and mirrored tail form a
palindrome-like copying loop. When paired with a zero tape, it leaves one full
representative in each half. It is useful for interpreter validation but is
separate from the detailed provenance-traced outbreak discussed below and
should not be treated as the only shape a functional BFF replicator can take.

### Buffer contract

The C evaluator assumes its pointer references at least 128 bytes and processes
the first 128 as two complete 64-byte tapes. Passing a short candidate directly
would allocate less than C expects and misalign the food boundary; passing a
long candidate would also put food at the wrong boundary while output remains
split at byte 64. Both [`test_selfrep()`](../bff/soup.py) and the newer
functional verifier therefore enforce an exact 64-byte candidate. A shorter
replicating sequence must be tested within an explicitly constructed 64-byte
carrier rather than passed directly to the C evaluator.

## Measuring the population transition

The 2024 paper introduces **high-order entropy**. Conceptually, for a byte
string `x` of length `n`:

```text
H_high(x) = H_byte(x) - K(x) / n
```

`H_byte` is ordinary bytewise Shannon entropy and `K` is Kolmogorov
complexity. Since Kolmogorov complexity is uncomputable, the paper
approximates it with compressed size using Brotli quality 2. The local
implementation computes Shannon entropy minus compressed bits per byte,
preferring Brotli quality 2 and falling back to zlib level 6.

The intended contrast is:

- independent random bytes have high Shannon entropy but compress poorly, so
  high-order entropy approaches zero; and
- repeated structured tapes remain byte-diverse but compress well, producing
  positive high-order entropy.

This metric is a fast population-structure proxy, not a replicator detector.
Compressor overhead, population size, and compressor choice affect finite-run
values. Brotli and zlib results should not be treated as numerically
interchangeable.

The dashboard also reports non-no-op operations per interaction. This
measures execution activity, not fitness. Likewise, `--stop-at` is a
high-order-entropy threshold; `--epochs` is the epoch limit.

## What the 2024 experiments reported

In 1,000 main BFF runs using the paper's approximately `0.024%` mutation
rate, about 40% crossed the reported transition within 16,384 epochs.
Transitions occurred at varied times rather than only near initialization.
Increasing mutation generally accelerated transition, but runs with zero
background mutation transitioned at roughly comparable frequency.

Additional controls helped distinguish interaction-driven construction from
lucky initialization:

- only 3 of 1,000 random soups transitioned within 128 epochs;
- when one known replicator was seeded, only about one fifth transitioned
  within 128 epochs, showing that a singleton replicator is often destroyed;
  and
- zero-mutation runs with a fixed interaction schedule still transitioned
  frequently.

These results support the paper's interpretation that interaction and
self-modification are important in its tested configurations. However, the
transition was measured primarily through high-order entropy and detailed
analysis of one provenance-traced run, not exhaustive functional
classification of every tape.

In that case study, provenance tokens attached to bytes allowed the authors to
trace copied material back through the soup. The first outbreak encountered a
"zero-poisoning" period, after which a more zero-tolerant replicator spread.
This traced outbreak is distinct from the sanitized reverse-copy example
above. The paper's prose and figure captions differ by one epoch when naming
the traced replicator's first appearance, so this document does not assign one
exact emergence epoch.

The paper also reports emergence in spatial BFF, selected Forth variants, and
Z80 and Intel 8080 substrates. It reports a counterexample in SUBLEQ-like
systems: hand-written replicators can exist, but spontaneous emergence was
not observed in the experiments. The result is therefore not that every
computational substrate inevitably generates life.

## The later functional detector

The original paper notes that exhaustive substring-level detection is hard
and does not specify the assay used by this dashboard. A later detector
appears in CuBFF and is described in the 2026 follow-up paper.

The [current CuBFF `CheckSelfRep` implementation, pinned here](https://github.com/paradigms-of-intelligence/cubff/blob/f212e849027c98fcf4b242eccfb5fed435223e23/common_language.h#L203-L277),
uses:

- 13 deterministic random-food contexts;
- five executions per context;
- the previous right-hand output as the next generation's parent;
- the same context-specific food tape at every generation; and
- a maximum of 8,192 evaluator steps per execution.

Only the final generation is scored. For each position in the left half, the
original candidate byte must appear in at least four contexts. For each
position in the right half, any byte value must agree across at least four
contexts. The score is the smaller of the two supported-position counts, so
it ranges from 0 to 64.

The chain tests whether reproduced structure remains functional for several
generations. Multiple contexts test robustness to different partners. For a
two-phase or inverting replicator, the odd number of executions means the
parent entering the final execution can be back in the original phase; the
right output only needs to be consistent across contexts. The positional
comparison intentionally rejects replicators that only produce shifted output.

The 2026 paper classifies a candidate at a heuristic threshold of at least
48/64. It emphasizes that 64 would be too strict because behaviorally
irrelevant bytes need not be reproduced exactly. It also describes a
near-shift-invariant score-56 construction that eventually destroys its copy
mechanism, demonstrating that a high finite score is evidence rather than a
proof of indefinite replication.

### Paper/source discrepancy

The prose of the 2026 paper describes nine final tapes and agreement in at
least three, while its pseudocode writes `i in [0..9]`, conventionally ten
values. The current source is unambiguous: it uses 13 contexts and support in
at least four. The paper and CuBFF research scripts use a threshold of 48,
while the pinned source's general CLI constant is currently
[`5`](https://github.com/paradigms-of-intelligence/cubff/blob/f212e849027c98fcf4b242eccfb5fed435223e23/common.h#L55-L57).
This repository deliberately combines the current 13-context assay mechanics
with the paper's conservative 48/64 threshold and exposes only thresholds from
48 through 64.

## Conservative verification added here

[`bff/replication.py`](../bff/replication.py) adds constraints that are not
part of the 2024 experiment or the base CuBFF assay:

1. Candidate discovery searches a deterministic sample of at most 8,192
   carriers for exact 16-byte markers appearing in at least two distinct
   carriers.
2. A complete representative must score at least 48 in the primary assay.
3. It must independently pass the same threshold against a second deterministic
   food batch generated from a different seed.
4. For a context to support its 8-16 byte marker, the marker must be absent
   from that context's food and present in both output halves after every
   generation. At least four contexts must support it in each batch.
5. Once verified, the marker is counted across the full soup at most once per
   carrier, with matches crossing a 64-byte boundary excluded.

These choices reduce opportunistic false positives and produce a stable
population signal, but they create false-negative modes. A candidate with no
16-byte marker shared by another sampled carrier is invisible to default
discovery, so a unique first replicator will usually be missed until some
signature has spread. A shifted, unusually short, context-dependent, or
multi-member autocatalytic system can also be missed. The selected marker is
evidence of persistent copied structure, not necessarily the minimal copying
mechanism.

## Interpreting the realtime visualization

[`bff/live.py`](../bff/live.py) separates exact tape identity from verified
marker prevalence:

- the upper panel shows verified 64-byte representatives, their replicated
  marker, functional scores, exact representative count, and current carrier
  prevalence;
- the middle panel shows how often each verified marker occurs in the soup,
  including growth and lifetime peak; and
- the lower panel shows high-order entropy and execution activity.

The exact-tape view remains available because it is useful for inspecting
population convergence, but it is not the functional classifier.

The simulation worker exclusively owns and mutates the soup. Functional
assays are read-only and use a separate deterministic noise generator, so
enabling verification does not consume the soup RNG or change a seeded soup
trajectory. By default, verification runs every 0.25 seconds independently of
the 0.5-second GUI snapshot cadence. Both intervals are configurable. Bounded
history is retained so skipped display frames do not erase a rapid outbreak.

The graph's marker count is deliberately described as **carrier prevalence**:

- carriers are not individually functionally assayed;
- marker tracks can overlap and their percentages must not be summed;
- marker presence does not prove ancestry or descent;
- a marker is not proven to be an autonomous replicating core; and
- first verification time depends on wall-clock scan cadence and can vary
  across machines even when the underlying seeded trajectory is unchanged.

One validation of this working-tree revision used Brotli, the default mutation
rate and 8,192-step execution limit, startup core checking, and:

```sh
uv run ./bff/live.py --n 4096 --seed 4 --epochs 9200 \
  --refresh 0.5 --verify-every 0.25 --replication-threshold 48
```

Two verified markers were first observed around epoch 8,551 at about 1%
carrier prevalence. One tracked outbreak reached approximately 73% near epoch
8,645. The exact first-observation epoch is not a cross-machine reproduction
target because verification is scheduled by wall clock. The run nevertheless
shows that the retained history captures accumulation and decline in this
seeded trajectory; it is one repository observation, not an estimate of
general emergence probability.

## How the later work changes the interpretation

The ALife presentation places BFF in a broader theory of symbiogenesis and
gelation. It distinguishes inanimate, viral, and cellular replicators,
proposes that mergers of previously copied structures create more capable
wholes, and interprets ancestry-depth interventions as evidence that such
composition is necessary for the population transition. Relevant sections
include the [BFF setup](https://www.youtube.com/watch?v=M2iX6HQOoLg&t=918s),
[transition and compressibility](https://www.youtube.com/watch?v=M2iX6HQOoLg&t=1103s),
[zero-mutation question](https://www.youtube.com/watch?v=M2iX6HQOoLg&t=1442s),
and [symbiogenesis discussion](https://www.youtube.com/watch?v=M2iX6HQOoLg&t=1696s).

The 2026 follow-up introduces evidence that narrows this interpretation:

- BFF found a first self-replicator after about `5.0 x 10^6` programs tested
  on average in the reported experiment.
- Uniform random 64-byte sampling was slower, at about `2.9 x 10^7` programs.
- Sampling from a manually chosen distribution balancing operators and
  no-ops was faster, at about `4.5 x 10^5` programs.
- A distribution additionally favoring byte value 64 was faster still, at
  about `9.4 x 10^4` programs.
- Restricting merger depth or width did not prevent first replicators from
  appearing, although strong merger restrictions prevented them from taking
  over the soup.

The follow-up authors therefore conclude that pairwise BFF interaction is not
an unusually powerful search operator for finding the first replicator, and
that compositionality is not necessary for discovery in their tested setup.
It may still be important for ecological amplification and takeover.

This motivates a more precise interpretation of the experiment:

- **origin:** how a replication-capable program first enters program space;
- **function:** whether it repeatedly reproduces under varied contexts;
- **ecology:** whether it survives destructive interactions and increases;
- **transition:** whether its spread reorganizes the population enough to
  change aggregate metrics.

The live dashboard observes function, ecology, and transition. It does not
reconstruct the causal origin or full ancestry of the first replicator.

## Open questions

The combined work leaves several technical questions open:

- Which language and environment properties control the density and minimum
  length of functional replicators?
- Which candidate-generation process is most efficient when compute,
  changed bytes, or tested programs are held constant?
- What mechanisms allow a rare replicator to survive long enough to take
  over rather than go extinct?
- When do copied fragments form a causally integrated replicating unit rather
  than a correlated marker or parasitic dependency?
- How should lineage be reconstructed when code shifts position, reverses,
  mutates inert bytes, or participates in autocatalytic sets?
- Can open-ended novelty continue after takeover, or does the population
  settle into a small set of short replication strategies?
- Which conclusions remain stable across well-mixed, spatial, and
  resource-limited environments?

## References

### Primary BFF sources

1. Blaise Agüera y Arcas, Jyrki Alakuijala, James Evans, Ben Laurie,
   Alexander Mordvintsev, Eyvind Niklasson, Ettore Randazzo, and Luca Versari.
   ["Computational Life: How Well-formed, Self-replicating Programs Emerge
   from Simple Interaction"](https://arxiv.org/abs/2406.19108v2), 2024.
   [PDF](https://arxiv.org/pdf/2406.19108v2) and
   [DOI](https://doi.org/10.48550/arXiv.2406.19108).
2. Charlotte Knierim, Luca Versari, Robert Obryk, Blaise Agüera y Arcas, and
   Rif A. Saurous. ["BFF: Simple explanations for complex
   phenomena"](https://arxiv.org/abs/2607.01483), 2026.
   [DOI](https://doi.org/10.48550/arXiv.2607.01483).
3. Paradigms of Intelligence.
   [`cubff` reference implementation](https://github.com/paradigms-of-intelligence/cubff).
   The detector mechanics cited above are pinned to
   [commit `f212e849`](https://github.com/paradigms-of-intelligence/cubff/blob/f212e849027c98fcf4b242eccfb5fed435223e23/common_language.h#L203-L277).
   Relevant detector changes include the
   [13-context update](https://github.com/paradigms-of-intelligence/cubff/commit/d727d1f1b0878a029c8584da288ca94ba906d05b)
   and
   [five-generation hardening](https://github.com/paradigms-of-intelligence/cubff/commit/82f0c64e23acd6c9168e77581fa85e398be439a3).
4. Blaise Agüera y Arcas.
   ["What If Intelligence Didn't Evolve? It 'Was There' From the
   Start!"](https://www.youtube.com/watch?v=M2iX6HQOoLg), ALife 2025
   presentation hosted by Machine Learning Street Talk.

### Historical and conceptual context

5. John von Neumann. [*Theory of Self-Reproducing
   Automata*](https://archive.org/details/theoryofselfrepr00vonn_0), edited by
   Arthur W. Burks, 1966.
6. Mark A. Bedau et al. ["Open Problems in Artificial
   Life"](https://doi.org/10.1162/106454600300103683), *Artificial Life* 6(4),
   2000.
7. Walter Fontana. ["Algorithmic Chemistry: A Model for Functional
   Self-Organization"](https://digital.library.unt.edu/ark:/67531/metadc1197717/),
   1990.
8. Martin A. Nowak and Hisashi Ohtsuki. ["Prevolutionary dynamics and the
   origin of evolution"](https://doi.org/10.1073/pnas.0806714105), *PNAS*
   105(39), 2008.
9. Steen Rasmussen et al. ["The Coreworld: Emergence and evolution of
   cooperative structures in a computational
   chemistry"](https://doi.org/10.1016/0167-2789%2890%2990070-6), *Physica D*,
   1990.
10. Charles Ofria and Claus O. Wilke. ["Avida: A Software Platform for
    Research in Computational Evolutionary
    Biology"](https://doi.org/10.1162/106454604773563612), *Artificial Life*
    10(2), 2004.
11. Urban Müller. [Original Brainfuck
    distribution](https://aminet.net/package.php?package=dev/lang/brainfuck-2.lha),
    1993.

### Repository implementation

- [`bff/bff.py`](../bff/bff.py): readable BFF interpreter and known
  replicator demonstration.
- [`bff/soup.c`](../bff/soup.c): C interpreter and epoch loop.
- [`bff/soup.py`](../bff/soup.py): experiment driver and population metrics.
- [`bff/replication.py`](../bff/replication.py): functional verification and
  marker tracking.
- [`bff/live.py`](../bff/live.py): realtime visualization.
- [`bff/README.md`](../bff/README.md): commands and implementation notes.

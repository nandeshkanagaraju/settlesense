# gen/ — the independent generator path

## Do not import from `settlesense/`.

Not "prefer not to". Do not. A test AST-scans imports in both directions and
fails the build on one.

The reason is the whole evaluation. If the generator and the engine share a UTR
parser, a date parser, an amount normalizer or a matching utility, then a bug in
that shared code cancels itself out: the generator writes the same wrong thing
the engine reads, the metric comes back clean, and the number means nothing.
Independence is what makes the measurement a measurement.

This applies to *any* shared helper, including one that looks too small to
matter. Duplicate it here instead. The duplication is the point.

The rule runs the other way too: `settlesense/` never imports from `gen/`.

## Build order

Clean chains first → verify ground truth is correct → then layer noise.

A generator that mislabels its own truth corrupts every downstream metric
silently, and it does so in the direction that makes results look good. Ground
truth is self-checked before any noise work begins.

## Freeze

This path is frozen at Gate 2 (M1F) and the commit hash is published in
`GENERATOR_MANIFEST.json` at the repo root. After the freeze, engine development
begins and this directory does not change.

Truth files carry `generator_commit: null` — the hash cannot exist before the
commit that creates it, and truth files are never rewritten afterwards.

## Withheld noise types

Two noise types are generated from day one but are never used during engine
development, and are reported separately as the unknown-unknowns result:

- garbled narration with transposed characters
- split settlement across two batches

## Determinism

- D3: no module-level `random`. A single seeded `random.Random(seed)` is passed
  explicitly down the call chain.
- D13: every generated date is in 2026.
- D1: money is `Decimal`, quantized to 2 dp, `ROUND_HALF_UP`.

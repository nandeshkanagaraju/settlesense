## Results

**Three sets, three different rules, reported separately and never blended.**
A single combined figure would hide which one the engine was built against.

| Set | Seeds | Rule |
|---|---|---|
| **Dev** | 42 | Run freely while building. Tuned against — an upper bound, not a generalisation claim. |
| **AI evaluation** | 1000–1019 | Results may be inspected and the hypothesis loop adjusted — *that is what it is for*. What may not change is **which seeds are in it**. The range was declared before M7 and is never revised. |
| **Holdout** | 999 | **Run ONCE, at the end.** Recorded whatever it says. |

The AI evaluation set is **not** a second holdout. Iterating against it is
permitted and expected; the pre-registration constrains set membership, not
whether the results are looked at. The holdout is the only set with a
look-once rule, and it is the only one whose number carries an
untuned-generalisation claim.

### Dev set — seed 42 (`make eval`)

Built against. Inspected freely throughout M2–M5. These numbers are tuned-on
and should be read as an upper bound, not as a generalisation claim.

`as_of=2026-11-30` · `calendar_v1` · `config_hash=12e9a009b59251d8`

| Population A — `ReconciliationCase` (n=5,026) | |
|---|---|
| Case match rate | **0.989654** |
| Deterministic residual | **52** |
| Residual false-match rate | **0.000000** |
| Gross-exposure match rate (₹ `expected_gross`) | 0.989100 |
| Expected-net cash reconciled (₹ `expected_net`) | ₹72,204,883.74 |
| Unresolved expected-net cash | ₹800,722.94 |
| Evidence coverage | 1.000000 |

| Population B — batch↔bank links (n=39 batches) | |
|---|---|
| Batch link rate | **0.948718** (37/39) |
| Batch false-link rate | **0.000000** |
| Injected noise recovered | 15/17 · 0.882353 (0.875000 excluding the known truth defect) |
| Category precision on unresolved batches | 1.000000 (2/2) |

| Population C — row-grain variances (n=29 rows) | |
|---|---|
| Recall | **1.000000** |
| Precision | **1.000000** |
| Value | ₹1,330,088.02 |

Population A divides by payment count, B by batch count, C by row count. They
are three denominators and are never averaged together; `batch_net_total` is
not comparable to case `expected_gross`.

**All 52 residuals are one half of an ambiguous duplicate pair** — 26 pairs,
both halves flagged, because the engine cannot know which was injected. That
is the entire surface M7 has to work on, and it is deliberately not resolved by
rules: guessing would trade a zero false-match rate for coverage.

**Baselines — no ranking claimed.** The naive baseline links *fewer* here, but
that is one run on one dataset and is reported, not concluded.

| Baseline | Linked | False links | |
|---|---|---|---|
| `naive` | 32 | 0 | amount + date window only, no identifiers |
| `deterministic_only` | 37 | 0 | P1–P9, no model calls |
| `settlesense` | 37 | 0 | identical until M7 lands; reported so the AI delta is visible later |
| `llm_only` | — | — | **not run** — needs a recorded fixture set; never falls back to the network |

**Analyst time is a derived estimate, not a measurement**, and is attributed
separately: 4,974 deterministic resolutions × 4 min = 19,896 min. The AI layer
has resolved 0. The two are never added.

### Held-out set — seed 999 (`make eval-holdout`)

**NOT YET RUN.** Reserved for a single run at the end, after M7. Whatever it
prints is what gets recorded here, including if it is worse. It also carries
the two withheld noise types (`garbled_narration`, `split_settlement`), which
the engine has never seen.

---

## Reproducing

```
make gen           # dev dataset,      seed 42
make gen-holdout   # held-out dataset, seed 999, includes withheld noise types
make eval          # all baselines against the DEV set (seed 42)
make eval-holdout  # the HELD-OUT set (seed 999) - run ONCE, at the end
make eval-set      # regenerate the AI evaluation set (seeds 1000-1019)
make bench         # throughput scaling table -> reports/bench.md
make test          # no network, deterministic, under 60 seconds
make check         # ruff + mypy + determinism guard tests
```

Targets that have no implementation yet refuse and exit non-zero. They never
pass silently.

## Generator independence

The adversarial generator was frozen at the commit recorded in
[`GENERATOR_MANIFEST.json`](GENERATOR_MANIFEST.json) **before engine development
began**. It is a separate code path: `gen/` imports nothing from `settlesense/`
and `settlesense/` imports nothing from `gen/`, enforced by an AST scan in both
directions. Rates, row types and the variance taxonomy are restated inside
`gen/` rather than shared, because a bug in a shared helper would cancel itself
out and the measurement would mean nothing.

Results reproduce with new seeds. Truth files carry `generator_commit: null` —
the hash cannot exist before the commit that creates it, and truth files are
never rewritten afterwards; the real hash lives only in the manifest.

The freeze is enforced, not just announced: the manifest records a content hash
of `gen/`, and `tests/test_generator_freeze.py` fails if that path changes.
Editing the generator after the freeze requires an explicit re-freeze with a
stated reason.

Two noise types — `garbled_narration` and `split_settlement` — are **withheld**
from engine tuning, gated behind `--include-withheld`, and reported separately
as the unknown-unknowns result.

## Modules built

Recorded here at submission: which Level 2 modules were built and which were
cut, without apology. Cutting a conditional module is a planned outcome, not
an incomplete submission.

## The AI evaluation set — declared before it was run

**Seeds 1000–1019 inclusive. All twenty. No seed will be excluded for any
reason discovered after generation.**

This section was written and committed BEFORE the twenty datasets were
generated, and before any result from them was seen. That ordering is the
whole point: a range chosen after inspecting per-seed outcomes is not an
evaluation set, it is a selection, and no amount of care afterwards can undo
the choice. Committing the declaration first is what makes the claim
checkable — `git log` shows this text predating the numbers.

| Seed range | Role | Used for |
|---|---|---|
| 42 | dev | Building M2–M5. Inspected freely. Tuned-on, so reported as an upper bound. |
| 999 | holdout | Includes the two withheld noise types. Run ONCE, at the end. Not yet run. |
| **1000–1019** | **AI evaluation** | **M7 onward. The measurement surface.** Results may be inspected and iterated against; the *membership* is frozen. |

**This set is not a second holdout.** Looking at its per-seed results and
adjusting the hypothesis loop is the intended use. The pre-registration below
constrains one thing only — which seeds are in it — because that is the choice
that cannot be audited after the fact. Everything else is open.

**Why twenty and why these.** The deterministic layer leaves one kind of work:
ambiguous duplicate pairs, ~26 per seed on the dev set. Twenty seeds gives
roughly 520 independent decisions — enough that a per-category accuracy figure
is not dominated by a handful of cases, and small enough to run inside a test
budget. The specific numbers 1000–1019 carry no meaning beyond being the next
round block after the dev and holdout seeds; they were fixed by writing them
here, not by trying alternatives.

**Pre-declared expectation.** Roughly 26 `DUPLICATE_CANDIDATE` pairs per seed,
so roughly 520 decisions in total. A seed producing a wildly different count
signals a noise rate interacting with something seed-dependent, and is a
finding to investigate — **not** grounds to drop the seed. If any seed is ever
excluded, the exclusion and its reason are recorded here in the same commit
that excludes it.

### What it turned out to be — measured after the declaration above

Recorded in a later commit than the declaration, deliberately, so the order is
visible in `git log`. Nothing here was used to revise anything above.

| | |
|---|---|
| Decisions (one per ambiguous pair) | **507** against a pre-declared ~520 |
| Cases flagged (both halves of each pair) | 1,014 |
| Pairs per seed | mean 25.35, range 18–34 |
| Seeds excluded | **0** |

**On stability.** "~26 ± a few" is not testable, because *a few* is not a
number. Duplicate pairs are rare independent events over ~5,000 chains, so the
count is Poisson-ish and its expected spread is √mean = 5.03. Observed spread
is 4.25 — a dispersion of **0.84×**, slightly *narrower* than chance alone
produces. The extremes sit at −1.46σ (seed 1005, 18 pairs) and +1.72σ (seed
1009, 34 pairs). No outlier, nothing to investigate.

**Two exact invariants hold on all twenty seeds**, and they are asserted:
`cases == 5000 + pairs` (each pair adds exactly one repeat purchase, hence one
new case) and `residual == 2 × pairs` (the engine flags both halves, because it
cannot know which was injected).

**A negative result worth stating.** Zero natural batch-amount collisions
across all twenty seeds — 780 batches. Path B's abstention rule therefore still
has never fired on generated data, and its 6-of-8 precision remains conditioned
on low batch density exactly as `LIMITATIONS.md` says. Twenty seeds agreeing
that collisions do not occur is *not* evidence that Path B handles them; it is
evidence that this generator cannot produce the case that would test it.

**Storage.** The twenty datasets are ~146MB and are NOT committed. They are
defined by (frozen generator commit, seed) and regenerate byte-identically;
`EVAL_SET_MANIFEST.json` records a content hash per seed, which is a
checkable claim rather than 2,880 files nobody will read. Regenerate with
`make eval-set`.

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md).

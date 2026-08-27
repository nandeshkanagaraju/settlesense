# Limitations

What this project does not establish, stated before anyone has to ask.

## The engine mislabels split settlements, and nothing was changed about it

The held-out set (seed 999) was run once, after M8, and produced **52 false
matches** — a residual false-match rate of **1.0456%** against PDD 7.3's **1%**
budget, and **₹897,396.86** of gross exposure against a **₹500** budget. The dev
set produced zero of both.

**All 52 are `split_settlement`**, one of the two noise types withheld from
engine development. The failure mode is worse than missing them: the engine
CONFIRMS them under a plausible wrong category — `T_PLUS_N_TIMING` 48 times and
`PARTIAL_CAPTURE` 4 times. A payment spread across two batches does settle late
and does settle partially, so each wrong answer is locally consistent with the
evidence in front of it. There was no `SPLIT_SETTLEMENT` rule to reach for, so
it reached for the nearest rule it had.

**Nothing was adjusted in response.** No threshold, no tolerance, no weight, no
new rule. Tuning after seeing held-out results is how a held-out set stops being
held out, and a 1.0456% that became 0.9% by hand would be a worse number than
the one recorded here. The breach stands in the README as measured.

**What this changes about every other number in this project.** The dev set's
`0.000000` false-match rate should be read as *zero on noise the engine was
built against*. It is not a generalisation claim, and the holdout is the only
figure here that carries one. Population B and C both IMPROVED on unseen data —
batch link rate 0.949 → 0.974, noise recovery 0.882 → 0.950, Population C
perfect on both — so the failure is specific to one unseen category rather than
a broad collapse.

**What would fix it, and why it is not done here.** A `SPLIT_SETTLEMENT` pass —
subset-sum over settlement lines against a payment's expected total — is the
obvious rule, and PDD 6.2 anticipates exactly this migration. Building it now
would be tuning against the holdout. It belongs in a version whose evaluation
uses a seed nobody has looked at.

## The duplicate-pair task is fingerprinted, and the fingerprint is worthless

**Every injected duplicate carries an `-R###` suffix on its invoice number.**
Across seed 42: 26 of 5,053 ledger rows carry it, and those 26 are exactly the
26 rows truth marks as the injected duplicate. Zero false positives, zero false
negatives. One regex resolves the entire AI-eligible residual.

**It is not used, and must not be.** It is an artifact of how `gen/noise.py`
mints a unique invoice number for the row it injects, not a property real
merchant data would have. A verifier that tie-broke on it would score ~100% and
would generalise to nothing. The structural verifier is deliberately blind to
invoice numbers.

**What remains genuinely does not distinguish the two rows.** Excluding the
fingerprint, the halves of a pair share customer, gross, order date, SKU and
payment count; settlement-chain length differs in 2 of 26 pairs, and order-id
ordering is a coin flip (10/26). So the honest outcome is that the verifier
rejects nearly every nomination — which is what it does.

**Consequence for any AI number on this dataset.** The ceiling for a perfect
model is 27 of 507 pre-registered decisions (5.3%), measured with an oracle
that always nominates correctly. That figure is a property of the DATASET, not
of any model, and no model result on this data should be read as evidence about
model capability.

## The real-model sample is 40 decisions, not 507

A live sample WAS recorded — 40 decisions against OpenAI `gpt-4o-2024-08-06` —
and it is the only evidence here about how a real model behaves on this task.
The 507-decision figures come from stand-in clients and are properties of the
verifier and the data, not of any model.

**40 is not enough to characterise a model.** It was chosen to demonstrate the
verifier accepting and rejecting real output, not to measure accuracy. The
20/20 on the confirmable stratum should be read as "the model found the
evidence where it exists on these twenty", not as a precision estimate: the
error bars on n=20 are wide, and the sample is stratified rather than random.

**Two defects in this project were found only by real model output**, which is
the strongest argument for having recorded anything at all:

- The prompt never said what `candidate_id` meant, so the model returned the
  pair id `"ORD_A-ORD_B"` in all 40 cases. Synthetic clients had always
  supplied a single row id, so nothing had ever exercised the requirement.
- The verifier dispatched to the arithmetic or structural path by *whether an
  assertion was present*. The model attaches an assertion to every claim, so
  every duplicate hypothesis was routed to the arithmetic path and rejected for
  failing the field grammar — including the ones nominating the right row.
  Dispatch is now by category.

Both were rejections for the wrong reason, and both looked like the
architecture working until the numbers were read carefully.

## Standing rules earned the hard way

Each of these was added after a defect of that exact shape reached a green test
suite. They are recorded here rather than in a code comment because each one
generalises past the file that produced it.

**A text scanner matches its own documentation. Every guard walks the AST and
skips docstrings.**

Six occurrences here. A clock-guard flagged `telemetry.py` for a docstring
naming `date.today()` while explaining why it is forbidden; an AI-cost guard
flagged the paragraph explaining why `0.000s` is not printed; a holdout-seed
guard flagged the comment reading "never 999"; a truth-leak guard flagged
`types.py` for naming `true_category` in a comment about scoring; a chart guard
flagged the docstring saying "Altair, not st.line_chart"; and a
never-writes guard flagged "the choice was to fill them in or to delete the
app". Each was a green-to-red failure on prose, which is the cheap direction -
the expensive one is a scanner that matches nothing and reports green forever.
The fix in every case was to parse rather than grep.

**A default that silently widens a query is the same class of defect as a count
derived by subtraction. Where a function can answer several questions, make the
caller name which one.**

Three instances so far, all with the same signature — a plausible answer to a
question nobody asked, indistinguishable from the right answer until a case came
along where the two diverged:

| Where | The convenient thing | What it hid |
|---|---|---|
| Population counts | a count derived by subtraction | a category that belonged to neither side |
| File reading | empty and missing collapsed to "no rows" | a clean report for a day whose statement never arrived |
| `get_queue` | `status_filter` defaulting to every status | a convergence test that counted "all" while meaning "residual", reporting 11 where the answer was 2 |

The third passed for the wrong reason and would have kept passing: a single-shot
run confirms nothing, so "all" and "residual" coincide, and only a real
multi-day run separates them. `status_filter` is now a required argument, and
`ALL_STATUSES` exists so that asking for everything is a sentence someone wrote
on purpose.

## Dataset

The data is synthetic, produced by an adversarial generator on a separate code
path that shares no normalization, identifier-parsing, date-handling or matching
utility with the engine. It is not merchant data and carries no claim to
reproduce a real settlement distribution.

### The generator was re-frozen once

The generator was re-frozen once, before any engine code existed, to correct six
dataset defects found by the M1 guard suite. No engine had been run against the
superseded dataset.

The original freeze (`m1f-generator-freeze`, commit `6ff7172`) is left in place;
the current one is `m1f-generator-freeze-2`, and `GENERATOR_MANIFEST.json`
records both through its `supersedes` field. The six corrections were: a
hardcoded `true_variance_amount` that made every money-weighted metric
unmeasurable; `UNEXPLAINED` and orphan credits absent from the dev dataset;
`ROUNDING_DIFFERENCE` producible by no injector, leaving SDD 4.2 pass P9
unreachable; a docstring describing a defect rather than the code; and a
predicate that silently skipped all-digit UTRs.

The freeze exists to stop the generator being tuned against engine results. That
guarantee is about ordering, not about dates: what matters is that the generator
was frozen **before the engine**, not that it was frozen early. At the time of
the re-freeze no matching code existed and nothing had been measured.

**From M3 onward there is no further re-freeze.** A generator defect found after
engine development begins is recorded here and lived with, because by then a
correction cannot be distinguished from tuning. `tests/test_generator_freeze.py`
enforces this: once `settlesense/matching/` stops being a stub, a third freeze
generation fails the build.

## Calendar

The working-day calendar (`config/calendar_v1.yaml`) uses a **synthetic holiday
set for the 2026 simulated timeline**. It is not the RBI bank-holiday schedule
and does not attempt to be. The evaluation measures whether the engine handles a
holiday correctly, not whether it knows India's actual holiday list. Inventing a
real-looking holiday list would be a false claim of accuracy. Swapping in real
dates is a config change requiring no code change.

## Fuzzy UTR Path B is conditioned on low batch density

Path B — the scoring path used when no UTR fragment survives in the narration
— accepted **6 of 8** candidates on seed 42 with **zero false accepts**.

That result is conditioned on **low batch density**: 39 batches across 3
merchants over ~90 days, where amount-plus-date is close to unique. It should
not be read as a general precision figure for amount-and-date matching.

A production merchant settling daily with recurring price points would
generate frequent same-amount collisions, and Path B's precision would
degrade — possibly sharply, because its entire discriminating power is those
two fields. The **abstention rule is the safeguard**: when two batches share
an amount inside the date window, Path B refuses rather than picking. It fired
correctly on the one same-amount case constructed for it
(`test_two_batches_same_amount_same_window_no_fragment_abstains`).

Note what that sentence admits. The collision case had to be **constructed**,
because no two batches in the seed-42 dataset share an amount at all — the
safeguard is tested, but it has never fired on real generated data. Whether it
fires at the right rate under realistic density is not established here.
Seeds 1000–1019 were checked for naturally occurring collisions. **Zero, across
all twenty seeds and 780 batches.** So the safeguard still has never fired on
generated data, and this limitation is confirmed rather than resolved: twenty
seeds agreeing that collisions do not occur is not evidence that Path B handles
them, only that this generator cannot produce the case that would test it.

## Baselines

## Thresholds

The safety budgets in `config/thresholds.yaml` — residual false-match rate,
gross-exposure false-match value, cost per 1,000 rows — are **project safety
thresholds for a synthetic evaluation**. They are not a claim about acceptable
production loss. Real deployment would require risk, compliance and
business-owner sign-off on every one of them.

## What this does not establish

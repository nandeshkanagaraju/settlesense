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

## The held-out set has no throughput figure. The harness that failed to take one has been fixed

`eval/run_eval.py` collected no telemetry — it emitted accuracy only. The M5a
`StageTimer` existed, was tested, and was used by `make bench`, and **nobody had
connected it to the evaluation runner.** Nothing failed: the runner emitted
accuracy, the tests checked accuracy, and a missing measurement is invisible to
a suite that never asks for it. So seed 999 recorded Populations A, B and C and
**no records/second**.

It surfaced only when a report of the holdout run was asked for and throughput
was on the list. By then the set was spent.

**The runner is now wired** — `make eval` writes `reports/eval/throughput.md`
alongside the results, and `tests/test_timing.py` 16–22 hold it there. Timings
go to a **separate file**, never into `results.json`, for the reason SDD 8.1
keeps telemetry out of `ReconciliationResult`: that artifact is compared byte
for byte against a committed golden and a duration differs on every run. Test 18
asserts the payload is identical with instrumentation on and off; test 17
booby-traps `perf_counter` so "the uninstrumented path reads no clock" is proven
by the run succeeding rather than inferred from a small number.

**Seed 999 still has no throughput figure and will not get one.** Producing it
means a second run. Test 22 asserts `reports/eval-holdout/throughput.md` does
not exist, so the row cannot be quietly filled in later — the only way that file
could appear is the run that must not happen.

**The separation held in practice, not only in tests.** `reports/eval/results.json`
was byte-identical before and after the commit that wired the timer in — checked
against a copy taken beforehand, not inferred from the test that asserts it.
That is the M5a property doing its job on a real change: instrumentation was
added to the path that produces the artifact, and the artifact did not move.

Two things the wiring exposed immediately, which is the argument for having
done it:

- **The first headline was wrong by 36%.** Dividing cases by *every* stage
  included `metrics` and `baselines` — the code that scores the run against
  truth — and reported 8,732 cases/s where the pipeline had done 13,621. That
  bills the engine for the cost of measuring it. Both figures are printed now,
  each labelled with what it contains.
- **`run_baselines` runs the deterministic engine a second time**, rather than
  reusing what `evaluate` already computed. Left in place — a baseline is
  supposed to be an independent re-run — but it was invisible until something
  timed it.

Recorded rather than quietly omitted, because a reader comparing the dev and
holdout tables will notice one row present in one and absent in the other.

## A result can mean something other than what it looks like: the 47-row recording that was not bought

**The class first, because it generalises past this instance.** A result whose
*surface* matches the story you expected, while its *cause* is different, is the
hardest kind to catch from the number alone. Nothing about it looks wrong. It
agrees with the hypothesis. It only fails if somebody asks *why* it came out
that way — and the number itself never prompts that question.

**The instance.** `--simulate-outage` found that the AI layer had never run
against the exception store (below). The obvious fix was to record fixtures for
the store's 47 `DUPLICATE_CANDIDATE` rows against the live model, ~₹25 at the
measured ₹0.4845/decision, and publish the outcome. A near-zero confirmation
count was expected and would have been publishable: M7 abstains on 480 of 507,
so "structural evidence is absent in the store path too" is a coherent finding.

**Why it was not done.** A store row carries exactly ONE evidence id — the
settlement batch — so the prompt built from it lists a single id and then
instructs the model to pick *"the single row you believe is the duplicate
entry"* from that list. A choice from a list of one. Checked before spending,
with the oracle, which nominates the truth-correct row by construction and is
therefore the ceiling no model can beat:

```
ORACLE over 47 store rows -> confirmed: 0
   47  NO_HYPOTHESIS
distinct evidence ids across the 47 rows: 18   (all batch ids)
evidence ids that are order ids in truth:  0
```

**`NO_HYPOTHESIS`, not `ALL_REJECTED`, is the whole finding.** A rejection would
mean the verifier examined a claim and refused it — a statement about evidence.
`NO_HYPOTHESIS` means no claim could be formed at all. And with zero of the 18
evidence ids scoreable against truth, no nomination on that path could have been
graded in either direction.

So "0 of 47 confirmed" would have read as *the evidence does not distinguish the
candidates* while actually meaning *the prompt contained no question*. Same
number, same shape, different cause — and ₹25 spent to produce it.

**What was done instead cost nothing.** The store's pairs and M7's dataset-derived
pairs describe the same duplicates: 22 of 22 prompts were already recorded. The
gap was a read-time join. `settlesense/ai/pairing.py` groups rows on (gross
amount, settlement batch), replays the existing fixture, and writes the verdict
onto both rows — 22 pairs, 1 confirmed, 0 false confirms, no denominator moved.

`tests/test_m10_store_path.py` keeps the oracle measurement, so the reasoning
that avoided the spend stays checkable rather than surviving only in this file.

## The AI layer and the exception store operate on disjoint objects

Found while building M10, and it changes what the outage demo can claim.

**CLOSED as of 2026-08-28, by a read-time join rather than by recording — see
the section above.** The entry is kept rather than deleted because how it was
found is worth more than the fact that it is closed: it surfaced from
`--simulate-outage`'s own probe on its first run, reporting that 0 of 53
AI-eligible rows had a recording. A guard written to stop a demo faking an
outage is what exposed that a whole layer was unwired.

**The M7 hypothesis loop was measured over dataset-derived pair exceptions** —
`duplicate_exceptions(dataset)`, ids of the form `dup-ORD_A-ORD_B`, evidence
being the two order ids. 26 of them on the dev seed, all fixture-backed, and
every AI number in this README rests on that set.

**The M6 store persists engine outcomes** — Population A/B/C exceptions with
hash ids, evidence being a batch or case id. 47 of its rows are categorised
`DUPLICATE_CANDIDATE`.

**The two sets have zero ids in common, and zero recorded fixtures in common.**
All 53 AI-eligible residual rows in the demo store miss the replay cache. So
the AI layer has never run against the store, `resolved_by = AI_VERIFIED`
appears on no persisted row, and `PENDING_AI_UNAVAILABLE` had no writer at all
until M10 added one.

**What this does and does not invalidate.** The outage path is real and
measured: `OutageLLMClient` raises before any cache lookup, `ModelUnavailable`
and `FixtureMissError` are unrelated types taking provably different paths, and
the 53 rows genuinely go to `PENDING_AI_UNAVAILABLE`. What is NOT established
is what a *healthy* run would have concluded for those rows — nobody has
recorded a response for any of them. `probe()` prints exactly that on every
`--simulate-outage` run rather than letting a screenshot imply otherwise.

**How it was closed, and the option that was rejected.** Persisting the
pair-exceptions into the store would have added rows and moved the Population
A/B/C denominators every published number divides by; recording against the
store's own ids would have bought 47 non-answers. The read-time join does
neither. The M7 figure is untouched and the store-path result is published
beside it as a second measurement, never as a revision.

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

That result is conditioned on **low batch density**: 39 batches across three
merchant profiles, whose settled dates span **20 days** (2026-09-01 to
2026-09-21) inside a 91-day configured window, where amount-plus-date is close
to unique. This sentence previously said "3 merchants over ~90 days" — the 90
was the configured window rather than the batches, and the correction makes the
density HIGHER than stated, which cuts against the engine. It should
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

Every figure below is read from a committed artifact, and
`tests/test_limitations.py` asserts each one against its source. A limitations
section that drifted from the measurements would be the most misleading page in
the repository.

**Synthetic data only.** One generator, three merchant *profiles*
(`profile_a/b/c` in `config/mdr_rates.yaml`), **5,026 reconciliation cases**
from **15,779 input rows** on the dev seed. No production merchant data was
used at any point, and no production performance is claimed. Everything here is
a measurement of an engine against a generator, and the generator was written
by the same hand as the engine — the separation is structural (`gen/` and
`settlesense/` share no utility, and a test enforces it), not organisational.

**Batch density is low, and lower than this file previously said.** The fuzzy
UTR section above described *"39 batches across 3 merchants over ~90 days"*.
The 90 comes from the configured simulation window (2026-09-01 … 2026-11-30);
the **batches themselves span 20 days**, 2026-09-01 to 2026-09-21. That is
denser than stated and the correction cuts *against* the engine: amount-plus-date
is unique here across a shorter window than the sentence implied. Path B — the
path used when no UTR fragment survives — accepted **6 of 8** candidates with
**zero false links**, one ambiguous and one abstained. A merchant settling daily
with recurring price points would collide often, and Path B's entire
discriminating power is those two fields. The abstention rule is the safeguard,
it fires correctly on a constructed collision, and it has **never fired on
generated data** — across seed 42 and seeds 1000–1019, 780 batches, no two share
an amount at all.

**The held-out run breached both pre-declared thresholds.** Residual
false-match rate **0.010456** against `config/thresholds.yaml`'s **0.01**, and
gross exposure **₹897,396.86** against **₹500.00**. All 52 failures are
`split_settlement`, one of the two noise types withheld from engine
development. That is **one blind category, not a collapse**: on the same unseen
data Population B improved (batch link rate 0.948718 → 0.974359; noise recovery
0.882353 → 0.950000 on the counting basis) and Population C was perfect on both
sets (precision and recall 1.000000; 29/29 dev, 26/26 holdout). The dev set's `0.000000` false-match rate
should be read as *zero on noise the engine was built against* — it carries no
generalisation claim, and the holdout is the only figure here that does.

**The AI surface is one category.** Everything measured about the model concerns
`DUPLICATE_CANDIDATE`. It says where a model helped in *this* workflow on *this*
data. It does not generalise to reconciliation broadly, and — the direction
that is easier to forget — it does not establish that a model could not help
elsewhere in the pipeline. No other stage was tried, so no other stage was ruled
out.

**n is small, and smaller on the store path.** 507 dataset-derived decisions
across 20 seeds (27 oracle-confirmable, 0 oracle false confirms), 40 recorded
real-model decisions, and **22 store-path pairs of which 1 confirmed**. One of
22 is not a rate. The error bars on it are wide enough that it should be read as
*the wiring works and produced a correct confirmation*, not as a success
percentage — which is why it is published beside M7's number rather than merged
into it.

**The duplicate task carries a generator artifact.** Every injected duplicate
has an `-R###` suffix on its invoice number: **26 of 5,053 ledger rows**, and
those 26 are exactly the 26 truth marks. One regex resolves the entire
AI-eligible residual. It is deliberately unused and the verifier is blind to
invoice numbers — but it cuts both ways, and the second way is rarely said. The
task as posed is **easier to cheat** than a real one, *and possibly harder to
solve honestly*: excluding the fingerprint, the halves of a pair share customer,
gross, order date, SKU and payment count, settlement-chain length differs in 2
of 26, and order-id ordering is a coin flip. A real duplicate would usually
leave more trace than this one does.

**A truth defect was found and deliberately not corrected.** `BAT_16A0609791AB`
is labelled `ROUNDING_DIFFERENCE` in `truth_42.json` with a declared
`true_variance_amount` of **-0.08**, while the batch total and its bank credit
are **both ₹344,959.64** — an observable difference of exactly **₹0.00**. There
is nothing in the data to detect and an engine reporting "clean" is right. The
generator was frozen before the engine existed and was correctly **not
re-frozen** for this, so Population B noise recovery is reported **both ways**:
**0.882353** counting the batch as a miss and **0.875000** excluding it. The two
diverge on the dev set, which is where the defect lives, and coincide at
**0.950000** on the holdout, where `defect_batches_excluded` is empty because
the batch is not in that dataset. Printing both turns a hidden asterisk into
evidence the ground truth was audited, and lets a reader pick the number they
believe.

**The Tally export has never been imported by an accounting system.** It is
schema-validated against a bundled XSD written here, which catches this
project's own malformed documents and says nothing about whether Tally accepts
them. Labelled on the face of every batch: *schema-validated Tally-compatible
XML; not tested against a live Tally instance.* No live instance was available
and none was used.

**Cost comes from 66 recorded decisions, not a sustained run.** 40 in
`fixtures/llm_manifest.json` (₹19.37) and 26 in `fixtures/llm_manifest_dev.json`
(₹12.61), both against `gpt-4o-2024-08-06`. Two recordings agreeing to 0.2% per
decision is a measurement; it is not a load test, a month of traffic, or a
figure that survives a model or price change. **Live latency was never
captured** — the recorder read the API's token counts and no clock — so every
timing figure in this repository is **replay** off the local cache, labelled as
replay wherever it appears. There is no per-decision API latency number here
and none can be recovered from the fixtures.

**Throughput is one machine.** `arm · 8 cores · 8 GiB · Python 3.14.7 ·
macOS-26.6-arm64-arm-64bit-Mach-O`, stated in `reports/bench.md`'s header and
quoted into the README verbatim rather than retyped. Median of three, never the
best run. Nothing here establishes behaviour on a different CPU, a different
Python, or under contention. The same caveat applies to the suite itself: it
runs at **103.2s against SDD 7's 120s budget** on that machine, and a slower one
may breach it — the budget is a wall-clock assertion, so it is a property of the
machine as much as of the code.

**Cross-platform reproducibility was never run.** The determinism claims are
verified by re-running on one machine, not by two machines agreeing.

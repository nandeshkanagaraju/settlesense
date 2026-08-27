# SettleSense throughput — deterministic pipeline

`arm · 8 cores · 8 GiB · Python 3.14.7 · macOS-26.6-arm64-arm-64bit-Mach-O`

Median of 3 repetitions, never the best run. Dev seed (42) — the holdout is never benchmarked. No model calls.

## Scaling

| Records | Cases | Input rows | Ingest (s) | Engine (s) | Pipeline (s) | Cases/s | Rows/s | Peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 502 | 1,626 | 0.028 | 0.013 | 0.041 | 12,269 | 39,740 | 1.8 |
| 5,000 | 5,026 | 15,779 | 0.239 | 0.117 | 0.356 | 14,118 | 44,324 | 19.4 |
| 25,000 | 25,123 | 78,594 | 1.187 | 0.870 | 2.057 | 12,215 | 38,214 | 94.9 |
| 100,000 | 100,506 | 313,869 | 4.719 | 4.359 | 9.078 | 11,072 | 34,576 | 379.7 |

Pipeline = ingest + engine. Dataset generation is excluded from the timed region and reported separately below; it is scaffolding for the benchmark, not part of the system under test.

**Reading the files costs more than reconciling them.** Ingest is 52 to 67% of pipeline time at every size measured — at 100,000 records, 4.719s of parsing against 4.359s of matching. That is the reverse of what a reconciliation system suggests, and it is where an optimisation would go: the engine is not the constraint.

**100,000 records: attempted and measured**, because 25,000 completed in 2.057s — inside the 120s budget that gates it.

## Per-stage, at the largest size measured

Stages are listed in PASS ORDER, so P2 preceding P8 is readable straight down the column.

| Stage | Seconds | Records in | Records/s |
|---|---:|---:|---:|
| P7a duplicates confirmed | 0.099 | 101,007 | 1,024,133 |
| P7b duplicate pairing | 0.220 | 101,007 | 458,397 |
| P1 build cases | 2.349 | 100,506 | 42,783 |
| P3/P4/P6/P7b/P9 case classification | 1.294 | 100,506 | 77,649 |
| batch profile derivation | 0.176 | 106,392 | 603,382 |
| P2 exact batch<->bank | 0.002 | 39 | 22,848 |
| P2b full-UTR within tolerance | 0.003 | 39 | 15,292 |
| P8 fuzzy UTR | 0.001 | 14 | 22,629 |
| unresolved batch categorisation | 0.000 | 2 | 92,486 |
| row-grain variance assembly | 0.001 | 39 | 54,785 |

`P3/P4/P6/P7b/P9 case classification` is ONE row because those five passes are fused in a single walk over each case. Splitting them would mean restructuring the hot loop so a report could have more rows. The batch-grain passes below it are genuinely separate phases and are timed separately.

`P9` rounding categorisation does not appear as its own stage: it runs inline inside P2b and P8 on links those passes created, which is what makes it a categorisation rather than a matching pass.

READ THE BATCH-GRAIN ROWS AGAINST THEIR OWN DENOMINATOR. Population B is batches, and the generator emits roughly the same number of them at every size, so `records_in` on those rows barely moves while the records column above grows 200-fold. Their records/s figures are not comparable with the case-grain rows and are not a scaling signal.

## AI stage — priced against the residual, not the volume

| Records | Cases | Deterministic residual | Residual share |
|---:|---:|---:|---:|
| 500 | 502 | 4 | 0.80% |
| 5,000 | 5,026 | 52 | 1.03% |
| 25,000 | 25,123 | 246 | 0.98% |
| 100,000 | 100,506 | 1,012 | 1.01% |

**This ratio is the architectural argument.** The expensive stage is sized by the residual column, not the cases column: roughly one case in a hundred reaches it. Deterministic passes carry the volume, and the model is not on the hot path — which the per-stage table above shows directly, since every row in it is a rule.


### Deterministic pipeline against AI stage, at 5,000 records

**Deterministic pipeline: 5,026 cases in 0.356s, zero model calls.** The AI stage runs on the 52 residual cases — 1.03% of the workload, of which 26 are AI-eligible duplicate pairs and the rest abstain without a model call at all. **Model cost scales with ambiguity, not volume.** Building the rules layer properly is what keeps the expensive stage small.

**Seconds — 26 dev-seed decisions replayed in 0.122s, 4.7 ms each.** THIS IS CACHE-REPLAY TIME, NOT API LATENCY. Replay is what every test and every `make eval` executes, so it is the honest figure for what running this system costs in time — but it is not what a live call would take. Live latency was never captured: the recorder took the API's token counts and read no clock. That is a gap in the recorder, and it is stated rather than filled with a plausible number.

**Rupees — MEASURED from the API's own usage figures, not estimated from prompt length.** 41,232 input and 26,035 output tokens across 66 recorded decisions against `gpt-4o-2024-08-06` = $0.363430 (₹31.98), which is **₹0.4845 per decision**.

**Per 1,000 rows: ₹0.80**, and the basis is stated because the number is meaningless without it — the dev dataset's 26 AI-eligible pairs were recorded with NO sampling, so ₹12.61 is the complete model spend for those 15,779 input rows. Against PDD 7.3's ₹50 ceiling.

**WHERE THE PRE-SPEND ESTIMATE WENT WRONG, AND IT IS NOT WHERE IT LOOKS.** The projection made before any model was called was ₹40.98 per 1,000 rows against the ₹0.80 measured here — off by 51x. But the pricing model was nearly right: it projected ₹0.4083 per decision against ₹0.4845 measured, only 16% low. THE WHOLE ERROR WAS THE DECISION COUNT: the estimate assumed 507 decisions per dataset, and a dataset of this size produces 26. Which is the architectural claim restated as a cost bug — the deterministic layer had already removed the work the estimate was pricing.

Two independent recordings agree on the per-decision figure: ₹0.4842 across the 40 evaluation-set decisions and ₹0.4850 across the 26 dev-seed ones — different seeds, different pairs, 0.2% apart.

The verified hypothesis loop is also measured against three stand-in clients in reports/ai/ai_loop.json: an oracle that always nominates correctly establishes a ceiling of 27 of 507 decisions, which no real model can exceed, and an adversarial client confirms zero.

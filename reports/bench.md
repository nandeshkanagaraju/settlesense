# SettleSense throughput — deterministic pipeline

`arm · 8 cores · 8 GiB · Python 3.14.7 · macOS-26.6-arm64-arm-64bit-Mach-O`

Median of 3 repetitions, never the best run. Dev seed (42) — the holdout is never benchmarked. No model calls.

## Scaling

| Records | Cases | Input rows | Ingest (s) | Engine (s) | Pipeline (s) | Cases/s | Rows/s | Peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 502 | 1,626 | 0.027 | 0.013 | 0.040 | 12,581 | 40,751 | 1.8 |
| 5,000 | 5,026 | 15,779 | 0.211 | 0.119 | 0.330 | 15,233 | 47,822 | 19.4 |
| 25,000 | 25,123 | 78,594 | 1.131 | 0.796 | 1.927 | 13,035 | 40,778 | 94.9 |
| 100,000 | 100,506 | 313,869 | 4.551 | 3.974 | 8.525 | 11,790 | 36,818 | 379.7 |

Pipeline = ingest + engine. Dataset generation is excluded from the timed region and reported separately below; it is scaffolding for the benchmark, not part of the system under test.

**100,000 records: attempted and measured**, because 25,000 completed in 1.927s — inside the 120s budget that gates it.

## Per-stage, at the largest size measured

Stages are listed in PASS ORDER, so P2 preceding P8 is readable straight down the column.

| Stage | Seconds | Records in | Records/s |
|---|---:|---:|---:|
| P7a duplicates confirmed | 0.092 | 101,007 | 1,102,133 |
| P7b duplicate pairing | 0.219 | 101,007 | 461,974 |
| P1 build cases | 2.181 | 100,506 | 46,092 |
| P3/P4/P6/P7b/P9 case classification | 1.240 | 100,506 | 81,025 |
| batch profile derivation | 0.153 | 106,392 | 695,310 |
| P2 exact batch<->bank | 0.002 | 39 | 24,177 |
| P2b full-UTR within tolerance | 0.002 | 39 | 16,631 |
| P8 fuzzy UTR | 0.001 | 14 | 24,364 |
| unresolved batch categorisation | 0.000 | 2 | 103,675 |
| row-grain variance assembly | 0.001 | 39 | 55,473 |

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

**Seconds and rupees are NOT reported here, because no model has been called.** The verified hypothesis loop IS built (M7) and is measured in reports/ai/ai_loop.json - but with stand-in clients, not a model: an oracle that always nominates correctly establishes a ceiling of 27 of 507 decisions, which no real model can exceed. `fixtures/llm/` holds zero recordings, so there is no timing to report. Printing `0.000s` and `Rs 0` would be indistinguishable in this table from a stage that ran and cost nothing. The harness takes both numbers the moment a fixture set exists, reading them from the replay cache so re-running is free.

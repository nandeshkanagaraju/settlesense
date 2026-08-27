# SettleSense throughput — deterministic pipeline

`arm · 8 cores · 8 GiB · Python 3.14.7 · macOS-26.6-arm64-arm-64bit-Mach-O`

Median of 3 repetitions, never the best run. Dev seed (42) — the holdout is never benchmarked. No model calls.

## Scaling

| Records | Cases | Input rows | Ingest (s) | Engine (s) | Pipeline (s) | Cases/s | Rows/s | Peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 502 | 1,626 | 0.027 | 0.013 | 0.041 | 12,341 | 39,974 | 1.8 |
| 5,000 | 5,026 | 15,779 | 0.211 | 0.115 | 0.326 | 15,405 | 48,364 | 19.4 |
| 25,000 | 25,123 | 78,594 | 1.136 | 0.808 | 1.944 | 12,925 | 40,435 | 94.9 |
| 100,000 | 100,506 | 313,869 | 4.538 | 3.992 | 8.530 | 11,783 | 36,796 | 379.7 |

Pipeline = ingest + engine. Dataset generation is excluded from the timed region and reported separately below; it is scaffolding for the benchmark, not part of the system under test.

**100,000 records: attempted and measured**, because 25,000 completed in 1.944s — inside the 120s budget that gates it.

## Per-stage, at the largest size measured

Stages are listed in PASS ORDER, so P2 preceding P8 is readable straight down the column.

| Stage | Seconds | Records in | Records/s |
|---|---:|---:|---:|
| P7a duplicates confirmed | 0.088 | 101,007 | 1,153,250 |
| P7b duplicate pairing | 0.195 | 101,007 | 517,054 |
| P1 build cases | 2.201 | 100,506 | 45,656 |
| P3/P4/P6/P7b/P9 case classification | 1.243 | 100,506 | 80,845 |
| batch profile derivation | 0.152 | 106,392 | 700,292 |
| P2 exact batch<->bank | 0.002 | 39 | 22,963 |
| P2b full-UTR within tolerance | 0.002 | 39 | 16,264 |
| P8 fuzzy UTR | 0.001 | 14 | 24,479 |
| unresolved batch categorisation | 0.000 | 2 | 101,482 |
| row-grain variance assembly | 0.001 | 39 | 56,518 |

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

**Seconds and rupees are NOT reported here, because the stage does not exist yet.** The verified hypothesis loop is M7; `settlesense/ai/hypothesis.py` and `verifier.py` are docstring-only stubs and `fixtures/llm/` holds zero recorded responses. Printing `0.000s` and `₹0` would be indistinguishable in this table from a stage that ran and cost nothing. The harness takes those two numbers the moment there is a stage to measure, and it will read them from the replay cache, so re-running costs nothing.

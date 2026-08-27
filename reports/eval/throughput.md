# Evaluation throughput — seed 42

`arm · 8 cores · 8 GiB · Python 3.14.7 · macOS-26.6-arm64-arm-64bit-Mach-O`

**Pipeline: 5,026 cases in 0.358s — 14,051 cases/second.** Ingest plus engine, which is what `bench.md` measures and what a deployment would run.

Whole harness including scoring: 0.564s, 8,918 cases/second. Lower, and correctly so - `metrics` and `baselines` exist to grade the run against truth and have no counterpart in production. Quoting this number as throughput would bill the engine for the measurement.

A SINGLE run, not a median. [`bench.md`](../bench.md) is the headline throughput claim: median of 3 repetitions across 5 dataset sizes. This file says what one `make eval` cost on this machine.

| Stage | Seconds | Records in | Records out | Records/s |
|---|---:|---:|---:|---:|
| ingest+normalize | 0.217 | 15,779 | 15,779 | 72,674 |
| engine (P1-P9) | 0.141 | 15,779 | 5,026 | 112,240 |
| metrics (populations A/B/C) | 0.093 | 5,026 | 5,026 | 53,942 |
| baselines | 0.113 | 15,779 | 15,779 | 139,985 |
| **total** | **0.564** | | | |

`baselines` re-runs the deterministic engine a second time — `run_baselines` calls `run_deterministic_only` again rather than reusing what `evaluate` already computed. That duplication was invisible until this file existed, which is a fair argument for the instrumentation. It is left in place: the baseline table is supposed to be an independent re-run, and the cost is paid by the harness rather than by anything a user waits for.

## Why the held-out set has no such file

Seed 999 was run once, before this wiring existed, and it emitted accuracy only. Producing a throughput figure for it now would mean running it a second time. The harness is fixed for every future evaluation; the holdout keeps its gap, recorded in LIMITATIONS.md rather than quietly filled in.


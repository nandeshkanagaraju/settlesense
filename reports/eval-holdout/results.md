# SettleSense evaluation — seed 999

`as_of=2026-11-30` · `calendar=calendar_v1` · `config_hash=12e9a009b59251d8`

## Headline (PDD 8.3)

> Of the 54 exceptions the deterministic engine could not resolve, the verified hypothesis loop correctly explained 0, abstained on 54, and false-matched 0.

## Population A — ReconciliationCase (denominator: payment count)

| Metric | Value |
|---|---|
| Cases | 5027 |
| Case match rate (case count) | 0.989258 |
| Deterministic residual count | 54 |
| Residual false-match rate (case count) | 0.010456 |
| Gross-exposure match rate (₹ expected gross) | 0.988725 |
| Gross-exposure false-match value (₹ expected gross) | 897396.86 |
| Expected-net cash reconciled (₹ expected net) | 71043122.01 |
| Unresolved expected-net cash (₹ expected net) | 807270.62 |
| Evidence coverage (case count) | 1.000000 |

## Population B — batch↔bank links (denominator: batch count)

Never averaged with Population A. `batch_net_total` is not comparable to case `expected_gross`.

| Metric | Value |
|---|---|
| Batches | 39 |
| Batch link rate (batch count) | 0.974359 |
| Batch false-link rate (batch count) | 0.000000 |
| Batch-net linked value (₹ batch net total) | 71509298.36 |
| Batch-net false-link value (₹ batch net total) | 0.00 |
| Batches carrying injected noise | 20 |
| ...recovered to the correct credit | 19 |
| Noise recovery rate, defect counted | 0.950000 |
| Noise recovery rate, defect excluded | 0.950000 |
| Unresolved batches | 1 |
| Category precision on unresolved (batch count) | 1.000000 |

Category precision is computed over UNRESOLVED batches only. truth's `true_category` records what noise was INJECTED; the engine's category records what variance REMAINS. Once P8 recovers a truncated UTR nothing remains and `None` is correct — comparing the two across all batches scored 0.64 and penalised the engine for succeeding on 13 it had recovered. That was a metric defect, caught before it reached a table.

Truth defect excluded above: `none`. That batch is labelled ROUNDING_DIFFERENCE in truth but its batch total and bank credit differ by exactly ₹0.00, so there is nothing detectable. The generator is frozen and was correctly not re-frozen. Both numbers are shown so the reader chooses, rather than inheriting an asterisk.

## Population C — row-grain variances (denominator: row count)

| Metric | Value |
|---|---|
| Row variances found | 26 |
| In truth | 26 |
| Recall (row count) | 1.000000 |
| Precision (row count) | 1.000000 |
| Value (₹ row value) | 1272130.66 |

## Analyst time — a derived estimate, not a measurement

_derived estimate, assumes 4 min/review; attributed separately to rules and to AI, never blended_

| Source | Resolutions | Minutes (derived) |
|---|---|---|
| Deterministic rules | 4973 | 19892 |
| AI-confirmed residuals | 0 | 0 |

On this dataset the saving is attributable to the deterministic engine. The AI layer has not run. The two are never added together.

## Baselines

No ranking is claimed. The naive baseline may link MORE by pairing on amount and date alone; what differs is precision.

| Baseline | Linked | False links | Note |
|---|---|---|---|
| deterministic_only | 38 | 0 | P1-P9, no model calls |
| llm_only | — | — | requires a recorded fixture set or --allow-network; not run by default. See eval/baselines/llm_only.py for what was done to make this baseline strong. |
| naive | 33 | 0 | amount + date window only; no identifiers of any kind |
| settlesense | 38 | 0 | identical to deterministic_only until M7 lands; reported anyway so the AI contribution is visible as a delta later |

"""M5 - the evaluation runner. Emits results.json and a markdown table.

WHICH SEED THIS RUNS ON IS THE MOST IMPORTANT THING IN THIS FILE. `make eval`
points at the DEV set. The holdout has its own target, `make eval-holdout`,
which prints a warning before it runs. If the default pointed at the holdout it
would be run dozens of times during development and would stop being held out -
not through any single bad decision, just through convenience.

The markdown output carries the PDD 8.3 residual-set sentence with real numbers
substituted, and every money row names its basis.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from eval.baselines.deterministic_only import run_deterministic_only
from eval.baselines.naive import naive_amount_only_agreement, run_naive
from eval.metrics import (
    TruthView,
    analyst_minutes_saved,
    assert_no_ambiguous_money_keys,
    batch_outcome_counts,
    outcome_counts,
    population_a,
    population_b,
    population_c,
    residual_set_sentence,
)
from settlesense.config import AppConfig, load_config
from settlesense.core.telemetry import MachineSpec, StageTimer, StageTiming, format_rate
from settlesense.ingest import DayDataset, load_dataset
from settlesense.matching.engine import build_cases, merge_days

BASELINES = ("naive", "det", "llm", "settlesense")
DEFAULT_AS_OF = date(2026, 11, 30)


def input_rows(data_dir: Path) -> int:
    """Data rows across every day*_*.csv, counted from the FILES.

    Counted from the files rather than from the parsed dataset, so a row the
    ingest layer dropped still counts as work the pipeline was handed.

    A MISSING directory and an EMPTY one are different, and this refused to
    tell them apart in its first version: `Path.glob` on a directory that does
    not exist yields nothing rather than raising, so both returned 0 and a
    scaling table would have shown a clean `0 input rows` for a dataset that
    was never generated. Header-only files still return 0 - that is legitimate
    data, and day 1 genuinely has no bank credits because settlement is T+N.

    LIVES HERE, NOT IN bench.py, WHERE IT WAS WRITTEN. Both the benchmark and
    the evaluation runner need an ingest denominator, and bench.py already
    imports from this module - so moving it up the existing dependency edge
    leaves one definition instead of two that could disagree about what a row
    is. `eval.bench._input_rows` still resolves, for the tests that name it.
    """
    if not data_dir.is_dir():
        raise SystemExit(
            f"{data_dir} does not exist. Counting rows in a dataset that was never "
            "generated would report zero, which is indistinguishable from a "
            "dataset that is legitimately empty."
        )
    paths = sorted(data_dir.glob("day*_*.csv"))
    if not paths:
        raise SystemExit(f"{data_dir} exists but holds no day*_*.csv files")
    total = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
        total += max(lines - 1, 0)
    return total


def load_days(data_dir: Path, config: AppConfig) -> DayDataset:
    """Every day{N}_*.csv in one directory, merged."""
    days = sorted(
        {
            int(match.group(1))
            for path in data_dir.glob("day*_*.csv")
            if (match := re.match(r"day(\d+)_", path.name))
        }
    )
    if not days:
        raise SystemExit(f"no day*_*.csv found in {data_dir}")
    return merge_days([load_dataset(data_dir, day, config) for day in days])


def evaluate(
    dataset: DayDataset,
    config: AppConfig,
    truth: TruthView,
    as_of: date,
    minutes_per_review: int,
    timings: list[StageTiming] | None = None,
) -> dict[str, Any]:
    """Run the deterministic system and compute all three populations.

    `timings` is a COLLECTOR, not a return value, and defaults to None - which
    reads no clock at all. Nothing timed here reaches `payload`: durations vary
    between runs and results.json is compared byte for byte against a committed
    golden, so a seconds field in there would fail the comparison every time it
    was right. Same separation as SDD 8.1, applied to the artifact instead of
    the object.
    """
    with StageTimer(timings, "engine (P1-P9)", dataset.row_count()) as timer:
        result = run_deterministic_only(dataset, config, as_of)
        timer.records_out = len(result.cases)

    with StageTimer(timings, "metrics (populations A/B/C)", len(result.cases)) as timer:
        cases_by_id = {fact.case.case_id: fact.case for fact in build_cases(dataset, config)}
        pop_a = population_a(result, cases_by_id, truth)
        pop_b = population_b(result, truth)
        pop_c = population_c(result, truth)
        minutes = analyst_minutes_saved(result, minutes_per_review, ai_confirmed_residuals=0)
        timer.records_out = pop_a.case_count

    payload: dict[str, Any] = {
        "seed": truth.seed,
        "as_of": as_of.isoformat(),
        "calendar_version": result.calendar_version,
        "config_hash": result.config_hash,
        "population_a_case_count_denominator": pop_a.as_dict(),
        "population_b_batch_count_denominator": pop_b.as_dict(),
        "population_c_row_count_denominator": pop_c.as_dict(),
        "analyst_time": minutes.as_dict(),
        "case_status_counts": outcome_counts(result.cases),
        "batch_status_counts": batch_outcome_counts(result.batch_links),
        "residual_set_sentence": residual_set_sentence(
            residual_count=pop_a.deterministic_residual_count,
            explained=0,  # the AI layer does not exist until M7
            abstained=pop_a.deterministic_residual_count,
            false_matched=0,
        ),
    }
    # The guard runs on the ACTUAL payload, not on a schema someone maintains
    # alongside it. A metric added later is checked without anyone remembering.
    assert_no_ambiguous_money_keys(payload)
    return payload


def run_baselines(
    dataset: DayDataset,
    config: AppConfig,
    truth: TruthView,
    as_of: date,
    selected: Sequence[str],
    timings: list[StageTiming] | None = None,
) -> dict[str, Any]:
    """Each baseline's headline figures. NO RANKING IS ASSERTED.

    The naive baseline may link MORE than the engine - amount-plus-date pairs
    anything plausible. That is the point of including it, and the table
    reports link count and false-link count side by side so volume and
    precision are visible as two different numbers.
    """
    with StageTimer(timings, "baselines", dataset.row_count()):
        return _run_baselines(dataset, config, truth, as_of, selected)


def _run_baselines(
    dataset: DayDataset, config: AppConfig, truth: TruthView, as_of: date, selected: Sequence[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "naive" in selected:
        links = run_naive(dataset, config, as_of)
        wrong = [link for link in links if truth.batch_credit(link.batch_id) != link.bank_txn_id]
        out["naive"] = {
            "note": "amount + date window only; no identifiers of any kind",
            "batches_linked": len(links),
            "false_links": len(wrong),
            "batch_amount_uniqueness": str(naive_amount_only_agreement(dataset)),
        }
    if "det" in selected or "settlesense" in selected:
        result = run_deterministic_only(dataset, config, as_of)
        pop_b = population_b(result, truth)
        entry = {
            "batches_linked": pop_b.linked_count,
            "false_links": pop_b.false_link_count,
            "batch_count": pop_b.batch_count,
        }
        if "det" in selected:
            out["deterministic_only"] = dict(entry, note="P1-P9, no model calls")
        if "settlesense" in selected:
            out["settlesense"] = dict(
                entry,
                note=(
                    "identical to deterministic_only until M7 lands; reported "
                    "anyway so the AI contribution is visible as a delta later"
                ),
            )
    if "llm" in selected:
        out["llm_only"] = {
            "note": (
                "requires a recorded fixture set or --allow-network; not run by "
                "default. See eval/baselines/llm_only.py for what was done to "
                "make this baseline strong."
            ),
            "skipped": True,
        }
    return out


def to_markdown(payload: dict[str, Any], baselines: dict[str, Any]) -> str:
    """The results table. Every money row names its basis in the row label."""
    a = payload["population_a_case_count_denominator"]
    b = payload["population_b_batch_count_denominator"]
    c = payload["population_c_row_count_denominator"]
    t = payload["analyst_time"]

    lines = [
        f"# SettleSense evaluation — seed {payload['seed']}",
        "",
        f"`as_of={payload['as_of']}` · `calendar={payload['calendar_version']}` "
        f"· `config_hash={payload['config_hash']}`",
        "",
        "## Headline (PDD 8.3)",
        "",
        f"> {payload['residual_set_sentence']}",
        "",
        "## Population A — ReconciliationCase (denominator: payment count)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cases | {a['case_count']} |",
        f"| Case match rate (case count) | {a['case_match_rate_case_count']} |",
        f"| Deterministic residual count | {a['deterministic_residual_count']} |",
        f"| Residual false-match rate (case count) | {a['residual_false_match_rate_case_count']} |",
        f"| Gross-exposure match rate (₹ expected gross) "
        f"| {a['gross_exposure_match_rate_expected_gross']} |",
        f"| Gross-exposure false-match value (₹ expected gross) "
        f"| {a['gross_exposure_false_match_value_expected_gross']} |",
        f"| Expected-net cash reconciled (₹ expected net) "
        f"| {a['expected_net_cash_reconciled_expected_net']} |",
        f"| Unresolved expected-net cash (₹ expected net) "
        f"| {a['unresolved_expected_net_cash_expected_net']} |",
        f"| Evidence coverage (case count) | {a['evidence_coverage_case_count']} |",
        "",
        "## Population B — batch↔bank links (denominator: batch count)",
        "",
        "Never averaged with Population A. `batch_net_total` is not comparable "
        "to case `expected_gross`.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Batches | {b['batch_count']} |",
        f"| Batch link rate (batch count) | {b['batch_link_rate_batch_count']} |",
        f"| Batch false-link rate (batch count) | {b['batch_false_link_rate_batch_count']} |",
        f"| Batch-net linked value (₹ batch net total) "
        f"| {b['batch_net_linked_value_batch_net_total']} |",
        f"| Batch-net false-link value (₹ batch net total) "
        f"| {b['batch_net_false_link_value_batch_net_total']} |",
        f"| Batches carrying injected noise | {b['injected_noise_batch_count']} |",
        f"| ...recovered to the correct credit | {b['injected_noise_recovered_batch_count']} |",
        f"| Noise recovery rate, defect counted | {b['noise_recovery_rate_counting_defect']} |",
        f"| Noise recovery rate, defect excluded | {b['noise_recovery_rate_excluding_defect']} |",
        f"| Unresolved batches | {b['unresolved_batch_count']} |",
        f"| Category precision on unresolved (batch count) "
        f"| {b['category_precision_on_unresolved_batch_count']} |",
        "",
        "Category precision is computed over UNRESOLVED batches only. truth's "
        "`true_category` records what noise was INJECTED; the engine's category "
        "records what variance REMAINS. Once P8 recovers a truncated UTR nothing "
        "remains and `None` is correct — comparing the two across all batches "
        "scored 0.64 and penalised the engine for succeeding on 13 it had "
        "recovered. That was a metric defect, caught before it reached a table.",
        "",
        f"Truth defect excluded above: `{', '.join(b['defect_batches_excluded']) or 'none'}`. "
        "That batch is labelled ROUNDING_DIFFERENCE in truth but its batch total and "
        "bank credit differ by exactly ₹0.00, so there is nothing detectable. The "
        "generator is frozen and was correctly not re-frozen. Both numbers are shown "
        "so the reader chooses, rather than inheriting an asterisk.",
        "",
        "## Population C — row-grain variances (denominator: row count)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Row variances found | {c['row_count']} |",
        f"| In truth | {c['truth_row_count']} |",
        f"| Recall (row count) | {c['row_variance_recall_row_count']} |",
        f"| Precision (row count) | {c['row_variance_precision_row_count']} |",
        f"| Value (₹ row value) | {c['row_variance_value_row_value']} |",
        "",
        "## Analyst time — a derived estimate, not a measurement",
        "",
        f"_{t['label']}_",
        "",
        "| Source | Resolutions | Minutes (derived) |",
        "|---|---|---|",
        f"| Deterministic rules | {t['deterministic_resolutions']} "
        f"| {t['minutes_saved_deterministic_derived_estimate']} |",
        f"| AI-confirmed residuals | {t['ai_confirmed_residuals']} "
        f"| {t['minutes_saved_ai_derived_estimate']} |",
        "",
        "On this dataset the saving is attributable to the deterministic engine. "
        "The AI layer has not run. The two are never added together.",
        "",
        "## Baselines",
        "",
        "No ranking is claimed. The naive baseline may link MORE by pairing on "
        "amount and date alone; what differs is precision.",
        "",
        "| Baseline | Linked | False links | Note |",
        "|---|---|---|---|",
    ]
    for name in sorted(baselines):
        entry = baselines[name]
        if entry.get("skipped"):
            lines.append(f"| {name} | — | — | {entry['note']} |")
        else:
            lines.append(
                f"| {name} | {entry['batches_linked']} | {entry['false_links']} | {entry['note']} |"
            )
    return "\n".join(lines) + "\n"


def throughput_markdown(
    payload: dict[str, Any], timings: Sequence[StageTiming], machine: MachineSpec
) -> str:
    """`throughput.md` - a SEPARATE artifact, never folded into results.json.

    Kept apart for the reason SDD 8.1 keeps telemetry out of
    ReconciliationResult: results.json is compared byte for byte against a
    committed golden, and a duration is different on every run. Two files means
    the accuracy artifact stays comparable and the timing artifact stays honest,
    with no strip step between them.

    This is ONE run, not a median. reports/bench.md takes the median of three
    repetitions across five dataset sizes and is the figure to quote; this one
    says what this particular evaluation cost. Stated up front because the two
    tables look alike and a reader would otherwise compare them directly.

    THE HEADLINE IS THE PIPELINE, NOT THE TOTAL. `metrics` and `baselines`
    score the run against truth - they are the measuring instrument, and a
    production deployment has neither. Dividing cases by the total put the
    harness's own cost into a number labelled throughput and reported 8,732
    where the pipeline had done 13,621; the first version of this file did
    exactly that. Both are printed now, each labelled with what it includes.
    """
    total = sum(timing.seconds for timing in timings)
    pipeline_stages = {"ingest+normalize", "engine (P1-P9)"}
    pipeline = sum(timing.seconds for timing in timings if timing.stage in pipeline_stages)
    cases = payload["population_a_case_count_denominator"]["case_count"]
    lines = [
        f"# Evaluation throughput — seed {payload['seed']}",
        "",
        f"`{machine.describe()}`",
        "",
        f"**Pipeline: {cases:,} cases in {pipeline:.3f}s — {cases / pipeline:,.0f} cases/second.** "
        f"Ingest plus engine, which is what `bench.md` measures and what a "
        f"deployment would run."
        if pipeline > 0
        else "",
        "",
        f"Whole harness including scoring: {total:.3f}s, {cases / total:,.0f} cases/second. "
        f"Lower, and correctly so - `metrics` and `baselines` exist to grade the "
        f"run against truth and have no counterpart in production. Quoting this "
        f"number as throughput would bill the engine for the measurement."
        if total > 0
        else "",
        "",
        "A SINGLE run, not a median. [`bench.md`](../bench.md) is the headline "
        "throughput claim: median of 3 repetitions across 5 dataset sizes. This "
        "file says what one `make eval` cost on this machine.",
        "",
        "| Stage | Seconds | Records in | Records out | Records/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for timing in timings:
        lines.append(
            f"| {timing.stage} | {timing.seconds:.3f} | {timing.records_in:,} "
            f"| {timing.records_out:,} | {format_rate(timing.records_per_second)} |"
        )
    lines += [
        f"| **total** | **{total:.3f}** | | | |",
        "",
        "`baselines` re-runs the deterministic engine a second time — "
        "`run_baselines` calls `run_deterministic_only` again rather than "
        "reusing what `evaluate` already computed. That duplication was "
        "invisible until this file existed, which is a fair argument for the "
        "instrumentation. It is left in place: the baseline table is supposed "
        "to be an independent re-run, and the cost is paid by the harness "
        "rather than by anything a user waits for.",
        "",
        "## Why the held-out set has no such file",
        "",
        "Seed 999 was run once, before this wiring existed, and it emitted "
        "accuracy only. Producing a throughput figure for it now would mean "
        "running it a second time. The harness is fixed for every future "
        "evaluation; the holdout keeps its gap, recorded in LIMITATIONS.md "
        "rather than quietly filled in.",
        "",
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SettleSense evaluation.")
    parser.add_argument("--data", type=Path, required=True, help="directory of day*_*.csv")
    parser.add_argument("--truth", type=Path, required=True, help="truth_<seed>.json")
    parser.add_argument("--baselines", default="all", help=f"all|{'|'.join(BASELINES)}")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=DEFAULT_AS_OF)
    parser.add_argument(
        "--minutes-per-review",
        type=int,
        default=None,
        help="the analyst-time ASSUMPTION; read from config when omitted",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)

    # M5a StageTimer, wired here rather than left as a gap. It collects into a
    # local list that only throughput.md ever reads.
    timings: list[StageTiming] = []
    with StageTimer(timings, "ingest+normalize", input_rows(args.data)) as timer:
        dataset = load_days(args.data, config)
        timer.records_out = dataset.row_count()

    truth = TruthView.from_payload(json.loads(args.truth.read_text(encoding="utf-8")))
    selected = BASELINES if args.baselines == "all" else tuple(args.baselines.split(","))
    minutes = (
        args.minutes_per_review or config.thresholds.reporting.assumed_review_minutes_per_exception
    )

    payload = evaluate(dataset, config, truth, args.as_of, minutes, timings)
    payload["baselines"] = run_baselines(dataset, config, truth, args.as_of, selected, timings)
    assert_no_ambiguous_money_keys(payload)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "results.md").write_text(
        to_markdown(payload, payload["baselines"]), encoding="utf-8"
    )
    (args.out / "throughput.md").write_text(
        throughput_markdown(payload, timings, MachineSpec.current()), encoding="utf-8"
    )
    print(payload["residual_set_sentence"])

    pipeline = sum(
        timing.seconds
        for timing in timings
        if timing.stage in {"ingest+normalize", "engine (P1-P9)"}
    )
    cases = payload["population_a_case_count_denominator"]["case_count"]
    if pipeline > 0:
        print(f"\n{cases:,} cases in {pipeline:.3f}s — {cases / pipeline:,.0f} cases/second")
        print("(ingest + engine; scoring excluded, see throughput.md)")
    print(f"wrote results.json, results.md and throughput.md to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

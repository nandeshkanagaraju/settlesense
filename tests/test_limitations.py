"""Every figure in "What this does not establish" is checked against its artifact.

WHY THIS FILE EXISTS. A limitations section is the page a sceptical reader
trusts most, and the one nothing else in a test suite touches. Its numbers are
copied from artifacts by hand and then live forever: the engine can improve, a
threshold can move, a dataset can be regenerated, and the disclosure goes on
reporting whatever it reported. A limitations section that has drifted from the
measurements is worse than none at all, because it is read as an audit.

Writing it surfaced TWO figures already published here that did not match:

  "39 batches across 3 merchants over ~90 days" - the 90 is the configured
  simulation window; the BATCHES span 20 days. The correction cuts against the
  engine, which is the direction that matters: amount-plus-date is unique over
  a shorter window than the sentence implied.

  Noise recovery was described as coinciding on dev and diverging on the
  holdout. It is the reverse. The defect batch lives in the dev set, so dev
  reports 0.882353 counting and 0.875000 excluding; the holdout has no defect
  and both read 0.950000.

Neither would have been caught by reading. Both are asserted below.
"""

from __future__ import annotations

import json
import pathlib
import re
from decimal import Decimal
from typing import Any

import pytest

pytestmark = pytest.mark.hygiene

REPO = pathlib.Path(__file__).resolve().parents[1]
LIMITATIONS = REPO / "LIMITATIONS.md"
SECTION_HEADING = "## What this does not establish"

POP_A = "population_a_case_count_denominator"
POP_B = "population_b_batch_count_denominator"
POP_C = "population_c_row_count_denominator"


def section() -> str:
    """The section under test, isolated so a figure elsewhere cannot satisfy it."""
    text = LIMITATIONS.read_text(encoding="utf-8")
    assert SECTION_HEADING in text, f"{SECTION_HEADING} is gone from LIMITATIONS.md"
    body = text.split(SECTION_HEADING, 1)[1]
    return body.split("\n## ", 1)[0]


def prose() -> str:
    """The section with newlines collapsed, for phrase matching.

    Markdown wraps at 79 columns, so a phrase the author wrote as one sentence
    is stored across two lines. Matching against the raw text would fail on
    wrapping - which is presentation, not content - and the pressure would be
    to write worse prose to satisfy a test. Bold markers survive the collapse,
    so `**0.01**` is still checkable.
    """
    return re.sub(r"\s+", " ", section())


def results(name: str) -> dict[str, Any]:
    """The eval payload. `Any` because the tree mixes strings, ints and lists -
    rates are strings so Decimal round-trips exactly, counts are ints."""
    payload: dict[str, Any] = json.loads(
        (REPO / "reports" / name / "results.json").read_text(encoding="utf-8")
    )
    return payload


def test_the_section_is_not_empty() -> None:
    """It was an empty heading for most of this project's life."""
    body = section().strip()
    assert len(body) > 2_000, f"only {len(body)} characters; the heading is still a promise"
    print(f"\n  {len(body):,} characters, {len(body.splitlines())} lines")


# ===========================================================================
# Dataset shape
# ===========================================================================


def test_the_dataset_figures_match_the_dataset() -> None:
    """Cases, input rows, batches, merchant profiles, and the batch date span.

    The span is computed from the batches themselves rather than read from the
    calendar, because that is exactly the distinction the published "~90 days"
    got wrong.
    """
    from eval.run_eval import input_rows, load_days
    from settlesense.config import load_config
    from settlesense.matching.engine import build_cases

    config = load_config(REPO / "config")
    dataset = load_days(REPO / "data" / "dev", config)
    flat = prose()

    cases = build_cases(dataset, config)
    assert f"{len(cases):,}" in flat, f"the case count is not {len(cases):,}"
    assert f"{input_rows(REPO / 'data' / 'dev'):,}" in flat

    batches = dataset.settlement_batches
    assert f"{len(batches)} batches" in flat, f"there are {len(batches)} batches"

    dates = sorted(batch.settled_event_date for batch in batches)
    span = (dates[-1] - dates[0]).days
    assert f"span {span} days" in flat, f"the batches span {span} days, not what the text says"
    assert f"{dates[0]}" in flat and f"{dates[-1]}" in flat

    profiles = sorted({fact.case.merchant_profile for fact in cases})
    assert len(profiles) == 3, profiles
    assert "`profile_a/b/c`" in flat, "the three profiles are not named as profiles"

    ledger = len(dataset.ledger_rows)
    fingerprinted = [
        row for row in dataset.ledger_rows if re.search(r"-R\d{3}", str(row.invoice_no))
    ]
    assert f"{len(fingerprinted)} of {ledger:,} ledger rows" in flat, (
        f"the fingerprint count is {len(fingerprinted)} of {ledger:,}"
    )
    print(
        f"\n  {len(cases):,} cases, {input_rows(REPO / 'data' / 'dev'):,} rows, "
        f"{len(batches)} batches over {span} days, {len(profiles)} profiles, "
        f"{len(fingerprinted)} fingerprinted of {ledger:,}"
    )


# ===========================================================================
# The holdout breach, and what did NOT break with it
# ===========================================================================


def test_the_holdout_breach_figures_match_the_artifact_and_the_config() -> None:
    """Both breaches, both budgets, and the two populations that improved."""
    import yaml

    holdout = results("eval-holdout")
    dev = results("eval")
    flat = prose()

    rate = holdout[POP_A]["residual_false_match_rate_case_count"]
    exposure = Decimal(holdout[POP_A]["gross_exposure_false_match_value_expected_gross"])
    assert rate in flat, f"the holdout false-match rate is {rate}"
    assert f"₹{exposure:,.2f}" in flat, f"the exposure is ₹{exposure:,.2f}"

    thresholds = yaml.safe_load((REPO / "config" / "thresholds.yaml").read_text(encoding="utf-8"))
    # NOT `flat` - that name holds the prose, and shadowing it here made the
    # next assertion search the config for a sentence from the README.
    configured = json.dumps(thresholds)
    assert '"0.01"' in configured and '"500.00"' in configured, "the thresholds moved"
    assert "**0.01**" in flat and "**₹500.00**" in flat, "the budgets are not quoted as configured"
    assert Decimal(rate) > Decimal("0.01") and exposure > Decimal("500.00"), (
        "the run no longer breaches, so this section describes something else"
    )

    for key, label in (
        ("batch_link_rate_batch_count", "batch link"),
        ("noise_recovery_rate_counting_defect", "noise recovery"),
    ):
        before, after = dev[POP_B][key], holdout[POP_B][key]
        assert Decimal(after) > Decimal(before), f"{label} did not improve: {before} -> {after}"
        assert f"{before} → {after}" in flat, f"{label} is not recorded as {before} → {after}"

    for name, payload in (("dev", dev), ("holdout", holdout)):
        assert payload[POP_C]["row_variance_precision_row_count"] == "1.000000", name
        assert payload[POP_C]["row_variance_recall_row_count"] == "1.000000", name
    dev_rows = dev[POP_C]["matched_row_count"]
    holdout_rows = holdout[POP_C]["matched_row_count"]
    assert f"{dev_rows}/{dev_rows} dev, {holdout_rows}/{holdout_rows} holdout" in flat
    print(
        f"\n  {rate} > 0.01 and ₹{exposure:,.2f} > ₹500.00; B improved, "
        f"C perfect {dev_rows}/{dev_rows} and {holdout_rows}/{holdout_rows}"
    )


def test_the_truth_defect_is_described_exactly_as_the_data_has_it() -> None:
    """The DECLARED variance and the OBSERVABLE difference are different numbers.

    Truth says -0.08; the batch total and the bank credit are the same figure,
    so the observable difference is 0.00. Saying only one of those would make
    the defect sound like either a rounding quibble or a fabrication, and it is
    neither - it is an injection that left no trace.
    """
    from eval.metrics import TRUTH_DEFECT_BATCHES
    from eval.run_eval import load_days
    from settlesense.config import load_config

    flat = prose()
    (batch_id,) = sorted(TRUTH_DEFECT_BATCHES)
    assert batch_id in flat, f"{batch_id} is not named"

    truth = json.loads((REPO / "data" / "dev" / "truth_42.json").read_text(encoding="utf-8"))
    entry = next(item for item in truth["batch_links"] if item["batch_id"] == batch_id)
    assert entry["true_category"] == "ROUNDING_DIFFERENCE", entry
    declared = entry["true_variance_amount"]
    assert f"**{declared}**" in flat, f"truth declares {declared}"

    config = load_config(REPO / "config")
    dataset = load_days(REPO / "data" / "dev", config)
    batch = next(b for b in dataset.settlement_batches if b.batch_id == batch_id)
    credit = next(r for r in dataset.bank_rows if str(r.bank_txn_id) == entry["bank_txn_id"])
    observable = batch.net_total - credit.amount
    assert observable == Decimal("0.00"), observable
    assert f"₹{batch.net_total:,.2f}" in flat, "the shared figure is not quoted"
    assert "**₹0.00**" in flat, "the observable difference is not stated"

    dev = results("eval")
    counting = dev[POP_B]["noise_recovery_rate_counting_defect"]
    excluding = dev[POP_B]["noise_recovery_rate_excluding_defect"]
    assert counting != excluding, "the two bases coincide on dev; the text says they diverge"
    assert f"**{counting}** counting" in flat and f"**{excluding}** excluding" in flat
    assert results("eval-holdout")[POP_B]["defect_batches_excluded"] == [], (
        "the holdout now excludes a defect batch; the text says it excludes none"
    )
    print(
        f"\n  {batch_id}: truth {declared}, observable ₹{observable:.2f} on "
        f"₹{batch.net_total:,.2f}; dev {counting} vs {excluding}, holdout excludes none"
    )


# ===========================================================================
# Sample sizes and the AI surface
# ===========================================================================


def test_every_sample_size_matches_its_manifest() -> None:
    """507, 27, 40, 26, 66, and the store path's 22 and 1."""
    flat = prose()
    loop = json.loads((REPO / "reports" / "ai" / "ai_loop.json").read_text(encoding="utf-8"))
    store = json.loads((REPO / "reports" / "ai" / "store_path.json").read_text(encoding="utf-8"))
    stratified = json.loads((REPO / "fixtures" / "llm_manifest.json").read_text(encoding="utf-8"))
    dev_manifest = json.loads(
        (REPO / "fixtures" / "llm_manifest_dev.json").read_text(encoding="utf-8")
    )

    totals = loop["totals"]
    assert f"{totals['decisions_sent']} dataset-derived decisions" in flat
    assert f"({totals['oracle_confirmed']} oracle-confirmable" in flat
    assert f"{totals['oracle_false_confirmed']} oracle false confirms)" in flat
    assert f"across {totals['seeds']} seeds" in flat

    assert f"**{store['pairs_replayed']} store-path pairs of which {store['pairs_confirmed']}" in (
        flat
    ), f"the store path replayed {store['pairs_replayed']} and confirmed {store['pairs_confirmed']}"
    assert f"One of {store['pairs_replayed']} is not a rate" in flat

    recorded = stratified["recorded"] + dev_manifest["recorded"]
    assert f"{recorded} recorded decisions" in flat, f"{recorded} were recorded in total"
    assert f"{stratified['recorded']} in `fixtures/llm_manifest.json`" in flat
    assert f"(₹{stratified['measured_cost_inr']})" in flat
    assert f"{dev_manifest['recorded']} in `fixtures/llm_manifest_dev.json`" in flat
    assert f"(₹{dev_manifest['measured_cost_inr']})" in flat
    assert stratified["model"] == dev_manifest["model"]
    assert f"`{stratified['model']}`" in flat
    print(
        f"\n  {totals['decisions_sent']}/{totals['oracle_confirmed']} loop, "
        f"{store['pairs_replayed']}/{store['pairs_confirmed']} store path, "
        f"{recorded} recorded (₹{stratified['measured_cost_inr']} + "
        f"₹{dev_manifest['measured_cost_inr']})"
    )


def test_the_path_b_figures_are_the_realised_ones() -> None:
    """6 accepted of 8, one ambiguous, one abstained, zero false links."""
    from datetime import date

    from eval.run_eval import load_days
    from settlesense.config import load_config
    from settlesense.matching.engine import fuzzy_verdicts_for
    from settlesense.matching.fuzzy_utr import ScoringPath

    config = load_config(REPO / "config")
    dataset = load_days(REPO / "data" / "dev", config)
    verdicts = [
        v
        for v in fuzzy_verdicts_for(dataset, config, date(2026, 11, 30))
        if v.path is ScoringPath.AMOUNT_DATE
    ]
    accepted = [v for v in verdicts if v.is_accepted]
    flat = prose()

    assert f"**{len(accepted)} of {len(verdicts)}**" in flat, (
        f"Path B accepted {len(accepted)} of {len(verdicts)}"
    )
    assert results("eval")[POP_B]["false_link_count"] == 0
    assert "**zero false links**" in flat
    print(f"\n  Path B: {len(accepted)} of {len(verdicts)} accepted, 0 false links")


# ===========================================================================
# Environment claims
# ===========================================================================


def test_the_machine_line_is_the_bench_report_header_verbatim() -> None:
    """Quoted, not retyped - the same rule the README follows."""
    bench = (REPO / "reports" / "bench.md").read_text(encoding="utf-8")
    machine = next(
        line.strip().strip("`") for line in bench.splitlines() if line.strip().startswith("`arm")
    )
    flat = prose()
    assert machine in flat, f"the machine line is not the header verbatim:\n  {machine}"
    assert "Median of three, never the best run" in flat
    print(f"\n  machine line quoted verbatim: {machine}")


def test_the_suite_budget_claim_matches_the_configured_budget() -> None:
    """The 120s is read from the budget module, not typed into the prose."""
    from tests import budget

    limit = getattr(budget, "BUDGET_SECONDS", None) or getattr(budget, "LIMIT_SECONDS", None)
    assert limit is not None, "the budget module no longer exposes a limit"
    flat = prose()
    assert f"{int(limit)}s budget" in flat, f"the configured budget is {limit}s"
    assert re.search(r"\*\*\d+\.\d+s against SDD 7's", flat), (
        "the realised figure is missing; it is a measurement, not a rounding"
    )
    print(f"\n  budget {int(limit)}s, quoted from tests/budget.py")


@pytest.mark.charter_guard
def test_the_section_makes_the_claims_it_was_asked_to_make() -> None:
    """Coverage by TOPIC, so a rewrite cannot quietly drop one.

    Each phrase names a distinct thing the project does not establish. A
    limitations section that lost one would read exactly as complete.
    """
    flat = prose()
    required = {
        "synthetic data": "No production merchant data",
        "batch density": "settling daily with recurring price points",
        "holdout breach": "breached both pre-declared thresholds",
        "one AI category": "does not generalise to reconciliation broadly",
        "small n": "is not a rate",
        "generator artifact": "easier to cheat",
        "truth defect": "not re-frozen",
        "no live Tally": "never been imported by an accounting system",
        "cost basis": "not a sustained run",
        "no live latency": "Live latency was never captured",
        "one machine": "Throughput is one machine",
        "cross-platform": "Cross-platform reproducibility was never run",
    }
    missing = sorted(topic for topic, phrase in required.items() if phrase not in flat)
    assert not missing, f"the section no longer covers: {missing}"
    print(f"\n  all {len(required)} topics present")

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

    # THE REALISED DURATION IS NO LONGER REQUIRED IN THE PROSE, and that is a
    # deliberate reversal of what this test used to assert.
    #
    # It demanded `**103.2s against SDD 7's ...**` - a realised measurement
    # rather than a rounding, which was the right instinct about roundings and
    # the wrong conclusion. No committed file holds a suite duration, so the
    # assertion mandated a figure a reader could not check, and the figure had
    # already drifted: the prose said 103.2s while a clean-room run measured
    # 98.2s. That is the Rs2.49 shape - a number that reads as a measurement,
    # cannot be traced, and stops being true quietly.
    #
    # So the BUDGET is asserted, because tests/budget.py holds it, and the
    # realised duration is printed by every run instead. What this now checks
    # is that the section says so, rather than quoting a number.
    assert re.search(r"printed by every run", flat), (
        "LIMITATIONS no longer says the realised duration is printed per run. "
        "Either say it, or quote a figure that a committed artifact contains - "
        "not one that only ever existed in prose."
    )
    assert not re.search(r"\*\*\d+\.\d+s against SDD 7's", flat), (
        "a realised suite duration is quoted again. No artifact holds one, so "
        "it cannot be checked and will drift; tests/test_published_figures.py "
        "would reject it as untraceable."
    )
    print(f"\n  budget {int(limit)}s, quoted from tests/budget.py; realised figure printed per run")


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


# ===========================================================================
# The Baselines section — and the one baseline that has never run
# ===========================================================================

BASELINES_HEADING = "## Baselines"


def baselines_section() -> str:
    text = LIMITATIONS.read_text(encoding="utf-8")
    assert BASELINES_HEADING in text, "the Baselines heading is gone"
    return text.split(BASELINES_HEADING, 1)[1].split("\n## ", 1)[0]


def baselines_prose() -> str:
    return re.sub(r"\s+", " ", baselines_section())


def test_the_baselines_section_is_not_empty() -> None:
    body = baselines_section().strip()
    assert len(body) > 2_000, f"only {len(body)} characters; the heading is still a promise"
    print(f"\n  {len(body):,} characters")


@pytest.mark.charter_guard
def test_the_llm_baseline_is_recorded_as_not_run_because_it_was_not_run() -> None:
    """THE CLAIM THIS SECTION TURNS ON, asserted from both artifacts.

    An unrun baseline described as though it ran is the same defect class as
    the store-path gap, and that one survived until `--simulate-outage` probed
    it by accident. So the artifact is the authority: if `skipped` ever goes
    false, this test fails and the prose has to be rewritten rather than
    quietly becoming true.
    """
    flat = baselines_prose()
    for name in ("eval", "eval-holdout"):
        entry = results(name)["baselines"]["llm_only"]
        assert entry.get("skipped") is True, f"{name} now has a real llm_only result"
        assert "batches_linked" not in entry, f"{name} llm_only reports a link count"

    assert "implemented but was not run against recorded model responses" in flat, (
        "the section does not say plainly that it was not run"
    )
    assert "No comparison figure is published for it" in flat
    assert "`llm_only` ran on **neither**" in flat, "the holdout status is not stated"

    # PRESENT TENSE, not conditional. "would show" is how an unrun thing
    # becomes a result in a reader's memory.
    for hedge in ("the baseline shows", "the baseline demonstrates", "outperforms"):
        assert hedge not in flat.lower(), f"the section claims a result: {hedge!r}"
    print("\n  llm_only skipped=True in both artifacts; the section says so in present tense")


def test_the_llm_baseline_test_replays_synthesised_answers_not_model_output() -> None:
    """The distinction the section rests on, checked in the test's own source.

    `test_12` proves the plumbing and nothing about a model, because it builds
    its own fixture answers from the retrieval ranking. If that ever changed to
    real recordings the claim here would be understating the work.
    """
    source = (REPO / "tests" / "test_m5_eval.py").read_text(encoding="utf-8")
    body = source.split("def test_12_llm_only_runs_against_the_replay_client", 1)[1]
    body = body.split("\ndef ", 1)[0]
    assert "select_candidates(row, batches)[0].batch_id" in body, (
        "test_12 no longer synthesises its answers from retrieval; if it now "
        "replays recorded model output, LIMITATIONS understates what was run"
    )
    assert "synthesised by the test from the retrieval ranking" in baselines_prose()
    print("\n  test_12's fixtures are built from select_candidates, not from a model")


def test_the_baseline_link_counts_match_both_artifacts() -> None:
    """Every cell of the results table, from the committed runs."""
    flat = baselines_prose()
    for name, label in (("eval", "dev"), ("eval-holdout", "holdout")):
        payload = results(name)["baselines"]
        for baseline in ("naive", "deterministic_only", "settlesense"):
            entry = payload[baseline]
            linked = entry["batches_linked"]
            total = payload["deterministic_only"]["batch_count"]
            assert f"{linked} / {total}" in flat, f"{label} {baseline} linked {linked} of {total}"
            assert entry["false_links"] == 0, (label, baseline, entry["false_links"])
    print("\n  naive 32/33, deterministic 37/38, settlesense 37/38, all zero false links")


@pytest.mark.charter_guard
def test_the_surprising_ordering_is_reported_as_a_finding_not_smoothed_over() -> None:
    """Naive linked FEWER and was EQUALLY precise. Both halves contradicted.

    The premise was that a weaker baseline would link more and link worse. It
    did neither, and the cause is measured rather than guessed: every batch
    amount in this dataset is unique, so amount-plus-date cannot collide. That
    makes the naive result a restatement of the batch-density limitation, and
    the section has to say so rather than presenting it as a clean win.
    """
    dev = results("eval")["baselines"]
    holdout = results("eval-holdout")["baselines"]
    flat = baselines_prose()

    for payload in (dev, holdout):
        assert payload["naive"]["batches_linked"] < payload["deterministic_only"]["batches_linked"]
        assert payload["naive"]["false_links"] == payload["deterministic_only"]["false_links"] == 0

    uniqueness = dev["naive"]["batch_amount_uniqueness"]
    assert Decimal(uniqueness) == Decimal("1"), uniqueness
    assert f"**{uniqueness}**" in flat, f"batch amount uniqueness is {uniqueness}"
    assert "did not happen, and that is the finding" in flat
    assert "linked **fewer**" in flat and "zero on both sets" in flat
    assert "batch-density limitation restated as" in flat, (
        "the naive result is not connected back to the density limitation"
    )
    print(
        f"\n  naive linked fewer with equal precision; batch_amount_uniqueness "
        f"{uniqueness} explains it"
    )


def test_the_windows_and_retrieval_width_are_the_configured_ones() -> None:
    """5-day naive window, 3-day engine window, top-20 retrieval."""
    import inspect

    import yaml

    from eval.baselines.llm_only import CANDIDATES_PER_ROW
    from eval.baselines.naive import run_naive

    flat = baselines_prose()
    naive_window = inspect.signature(run_naive).parameters["window_days"].default
    thresholds = yaml.safe_load((REPO / "config" / "thresholds.yaml").read_text(encoding="utf-8"))
    engine_window = json.dumps(thresholds)

    assert f"**{naive_window}-day** date window" in flat, f"naive uses {naive_window} days"
    assert f'"date_window_days": {naive_window - 2}' in engine_window or "date_window_days: 3" in (
        (REPO / "config" / "thresholds.yaml").read_text(encoding="utf-8")
    ), "the engine window moved"
    assert f"wider than the engine's {naive_window - 2} days" in flat
    assert f"top-{CANDIDATES_PER_ROW} candidate retrieval" in flat, (
        f"retrieval is top-{CANDIDATES_PER_ROW}"
    )
    print(f"\n  naive {naive_window}d, engine 3d, retrieval top-{CANDIDATES_PER_ROW}")


def test_the_interpretive_counterweight_uses_the_store_path_figure() -> None:
    """14 of 22, read from store_path.json rather than recalled."""
    store = json.loads((REPO / "reports" / "ai" / "store_path.json").read_text(encoding="utf-8"))
    correct = sum(1 for entry in store["per_pair"] if entry["nominated_correctly"])
    total = store["pairs_replayed"]
    flat = baselines_prose()
    # Bold markers stripped as well as whitespace: where the ** falls inside a
    # phrase is an emphasis choice, and asserting it would make the test fail on
    # a rewording that changed nothing about the claim.
    plain = flat.replace("**", "")
    assert f"truth-correct order in {correct} of {total} pairs" in plain, (
        f"the store-path counterweight is {correct} of {total}"
    )
    assert "not a model that is bad at interpretation" in flat
    print(f"\n  interpretive counterweight: {correct} of {total} correct nominations")


@pytest.mark.charter_guard
def test_the_baselines_section_makes_the_claims_it_was_asked_to_make() -> None:
    """Coverage by topic, so a rewrite cannot quietly drop one."""
    flat = baselines_prose()
    required = {
        "not run": "was not run against recorded model responses",
        "what each is": "No identifiers of any kind",
        "realised numbers": "and no ranking claimed",
        "what was done": "no few-shot examples",
        "what was not done": "no prompt iteration against results",
        "the ceiling": "mostly arithmetic and identifier joins",
        "counterweight": "truth-correct order in",
        "holdout status": "ran on **neither**",
    }
    missing = sorted(topic for topic, phrase in required.items() if phrase not in flat)
    assert not missing, f"the Baselines section no longer covers: {missing}"
    print(f"\n  all {len(required)} topics present")

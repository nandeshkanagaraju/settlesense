"""M5 - denominator discipline first, then metrics, baselines, runner.

DENOMINATOR DISCIPLINE COMES FIRST because it is the failure that cannot be
seen in a number. A wrong rate looks like a wrong rate; a rate over the wrong
denominator looks like a correct rate. SDD 3.1 forbids merging the three
populations, and the tests at the top of this file are the only thing standing
between that rule and a plausible-looking table.

NO NETWORK. Every LLM path uses the replay client, and an autouse fixture
disables socket for the whole module - a test that quietly made a real call
would pass for money and could not be run offline.

NO RANKING BETWEEN BASELINES IS ASSERTED. Naive may match MORE by pairing on
amount and date; what differs is precision. Whatever the ordering turns out to
be is a finding, and a test that needed a baseline to lose was not written.
"""

from __future__ import annotations

import json
import re
import socket
import statistics
import subprocess
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from eval.baselines.llm_only import ROWS_PER_CHUNK, build_prompt, run_llm_only, select_candidates
from eval.baselines.naive import run_naive
from eval.metrics import (
    TRUTH_DEFECT_BATCHES,
    TruthView,
    analyst_minutes_saved,
    assert_no_ambiguous_money_keys,
    cost_per_resolved_exception,
    population_a,
    population_b,
    population_c,
    residual_set_sentence,
)
from eval.run_eval import evaluate, load_days, main, run_baselines, to_markdown
from settlesense.ai.client import ReplayLLMClient, prompt_hash
from settlesense.config import AppConfig, load_config
from settlesense.ingest import DayDataset
from settlesense.matching.engine import build_cases, run
from settlesense.types import (
    BankDirection,
    CaseOutcome,
    ExceptionStatus,
    ReconciliationCase,
    ReconciliationResult,
    ResolutionSource,
    money,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "dev"
EVAL_DIR = REPO / "data" / "eval"
AS_OF = date(2026, 11, 30)

EVAL_SEEDS = range(1000, 1020)
RESERVED_SEEDS = (42, 999)


# ---------------------------------------------------------------------------
# 24. No network, for the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """24. socket.socket raises for the duration of every test here.

    autouse, so it cannot be forgotten on a new test. A guard that has to be
    opted into protects only the tests that did not need it.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "a test attempted to open a socket. Every LLM path in this suite uses "
            "ReplayLLMClient; a real call would bill for a test run and make the "
            "suite unrunnable offline."
        )

    monkeypatch.setattr(socket, "socket", refuse)


@pytest.mark.charter_guard
def test_the_network_guard_actually_fires() -> None:
    """FAULT INJECTION for the fixture above. A no-op patch would protect
    nothing while every test reported green."""
    with pytest.raises(RuntimeError, match="attempted to open a socket"):
        socket.socket()


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture(scope="module")
def dataset(config: AppConfig) -> DayDataset:
    return load_days(DATA, config)


@pytest.fixture(scope="module")
def truth() -> TruthView:
    return TruthView.from_payload(json.loads((DATA / "truth_42.json").read_text()))


@pytest.fixture(scope="module")
def payload(dataset: DayDataset, config: AppConfig, truth: TruthView) -> dict[str, Any]:
    return evaluate(dataset, config, truth, AS_OF, minutes_per_review=4)


@pytest.fixture(scope="module")
def result(dataset: DayDataset, config: AppConfig) -> ReconciliationResult:
    return run(dataset, config, AS_OF)


def _case(case_id: str, gross: str, net: str) -> ReconciliationCase:
    return ReconciliationCase(
        case_id=case_id,
        payment_id=f"PAY_{case_id}",
        order_id=f"ORD_{case_id}",
        merchant_profile="profile_a",
        expected_gross=money(gross),
        expected_net=money(net),
        settlement_line_ids=(),
        payment_line_ids=(),
    )


def _outcome(case_id: str, *, confirmed: bool, category: str | None = None) -> CaseOutcome:
    return CaseOutcome(
        case_id=case_id,
        status=ExceptionStatus.CONFIRMED if confirmed else ExceptionStatus.OPEN,
        observed_net=None,
        variance=None,
        category=category,
        batch_id=None,
        bank_row_id=None,
        resolved_by=ResolutionSource.DETERMINISTIC if confirmed else None,
        confidence=None,
    )


def _result(cases: tuple[CaseOutcome, ...]) -> ReconciliationResult:
    return ReconciliationResult(
        cases=cases,
        batch_links=(),
        row_variances=(),
        exceptions=(),
        calendar_version="calendar_v1",
        config_hash="fixture",
    )


def _truth_for(categories: dict[str, str | None]) -> TruthView:
    return TruthView.from_payload(
        {
            "seed": 0,
            "cases": [{"case_id": cid, "true_category": cat} for cid, cat in categories.items()],
            "batch_links": [],
            "row_variances": [],
        }
    )


# ---------------------------------------------------------------------------
# 0a-0f. Denominator discipline
# ---------------------------------------------------------------------------


def test_0a_population_a_divides_by_the_case_count(
    payload: dict[str, Any], result: ReconciliationResult, dataset: DayDataset
) -> None:
    """0a. NOT rows, settlement lines, or bank rows - all of which differ here."""
    a = payload["population_a_case_count_denominator"]
    assert a["case_count"] == len(result.cases)
    others = {
        "ledger_rows": len(dataset.ledger_rows),
        "settlement_lines": len(dataset.settlement_lines),
        "bank_rows": len(dataset.bank_rows),
        "total_rows": dataset.row_count(),
    }
    print(f"\n  Population A denominator={a['case_count']}  other grains={others}")
    for name, count in others.items():
        assert a["case_count"] != count, (
            f"the Population A denominator equals the {name} count; it may be "
            "dividing by the wrong grain"
        )
    assert Decimal(a["case_match_rate_case_count"]) == (
        Decimal(a["confirmed_case_count"]) / Decimal(a["case_count"])
    ).quantize(Decimal("0.000001"))


def test_0b_batch_metrics_live_under_a_separate_key(payload: dict[str, Any]) -> None:
    """0b. Never in one table with A."""
    a = payload["population_a_case_count_denominator"]
    b = payload["population_b_batch_count_denominator"]
    assert set(a) & set(b) == set(), f"keys shared between A and B: {sorted(set(a) & set(b))}"
    assert not any("batch" in key for key in a), (
        f"a batch metric appears inside Population A: {[k for k in a if 'batch' in k]}"
    )


def test_0c_population_c_has_its_own_key_and_row_denominator(payload: dict[str, Any]) -> None:
    """0c."""
    c = payload["population_c_row_count_denominator"]
    assert "row_count" in c and c["row_count"] > 0
    assert c["matched_row_count"] <= c["row_count"]
    assert not any("case" in key or "batch" in key for key in c)


def test_0d_all_three_denominators_are_distinct(payload: dict[str, Any]) -> None:
    """0d. Accidental equality would hide a merge."""
    a = payload["population_a_case_count_denominator"]["case_count"]
    b = payload["population_b_batch_count_denominator"]["batch_count"]
    c = payload["population_c_row_count_denominator"]["row_count"]
    print(f"\n  denominators A={a} B={b} C={c}")
    assert len({a, b, c}) == 3, f"denominators collide: {a}, {b}, {c}"


def test_0e_a_batch_failure_degrades_population_a_by_its_case_count_once(
    dataset: DayDataset, config: AppConfig, truth: TruthView, result: ReconciliationResult
) -> None:
    """0e. Each case in the failed batch counted ONCE - not 1, not 10.

    Built from real rows: an unlinked batch is chosen, the cases settling into
    it are counted, and Population A is recomputed with exactly those forced
    open. The delta must equal how many of them were confirmed before.
    """
    cases_by_id = {f.case.case_id: f.case for f in build_cases(dataset, config)}
    facts = build_cases(dataset, config)

    target = next((link.batch_id for link in result.batch_links if link.bank_row_id is None), None)
    assert target, "no unlinked batch on this fixture; the precondition did not fire"
    in_batch = {f.case.case_id for f in facts if target in f.batch_ids}
    assert in_batch, f"batch {target} contains no cases; the check would be vacuous"

    before = population_a(result, cases_by_id, truth).confirmed_case_count
    degraded = _result(
        tuple(
            _outcome(c.case_id, confirmed=False) if c.case_id in in_batch else c
            for c in result.cases
        )
    )
    after = population_a(degraded, cases_by_id, truth).confirmed_case_count
    was_confirmed = sum(
        1 for c in result.cases if c.case_id in in_batch and c.status is ExceptionStatus.CONFIRMED
    )
    print(
        f"\n  batch {target} holds {len(in_batch)} cases ({was_confirmed} confirmed); "
        f"Population A fell by {before - after}"
    )
    assert before - after == was_confirmed, (
        f"a batch holding {len(in_batch)} cases moved Population A by "
        f"{before - after}; each case must count exactly once"
    )


def test_0f_every_money_key_names_its_basis(payload: dict[str, Any]) -> None:
    """0f. And no key says only "money_weighted"."""
    assert_no_ambiguous_money_keys(payload)
    a = payload["population_a_case_count_denominator"]
    b = payload["population_b_batch_count_denominator"]
    assert any(k.startswith("gross_exposure_") for k in a)
    assert any("expected_net" in k for k in a)
    assert any(k.startswith("batch_net_") for k in b)
    assert "money_weighted" not in json.dumps(payload)


@pytest.mark.charter_guard
def test_the_money_key_guard_fires_at_every_depth() -> None:
    """FAULT INJECTION. Every real metric is nested, so a top-level-only guard
    would protect nothing."""
    for planted in (
        {"money_weighted": 1},
        {"a": {"money_weighted_rate": "0.5"}},
        {"rows": [{"nested": {"money_weighted_value": "1.00"}}]},
    ):
        with pytest.raises(AssertionError, match="money_weighted"):
            assert_no_ambiguous_money_keys(planted)


# ---------------------------------------------------------------------------
# 1-9. Metrics correctness on hand-computed fixtures
# ---------------------------------------------------------------------------


def test_1_case_match_rate_on_a_ten_case_fixture() -> None:
    """1. Seven of ten confirmed == exactly 0.70."""
    cases = tuple(_outcome(f"c{i}", confirmed=i < 7) for i in range(10))
    cases_by_id = {f"c{i}": _case(f"c{i}", "100.00", "100.00") for i in range(10)}
    metrics = population_a(_result(cases), cases_by_id, _truth_for(dict.fromkeys(cases_by_id)))
    assert metrics.case_count == 10 and metrics.confirmed_case_count == 7
    assert metrics.case_match_rate_case_count == Decimal("0.700000")


def test_2_value_weighted_diverges_from_case_rate_under_skew() -> None:
    """2. One Rs100,000 case unmatched, ninety-nine Rs10 cases matched.

    The whole reason PDD 8.4 insists on a named basis: a case rate of 0.99 and
    a gross-exposure rate of 0.0098 describe the SAME run, and quoting only the
    first hides Rs100,000 of unresolved exposure behind small change.

    There is no "money-weighted match rate" key to reach for (0f) - the basis
    is in the name.
    """
    cases = tuple(
        [_outcome("big", confirmed=False)] + [_outcome(f"s{i}", confirmed=True) for i in range(99)]
    )
    cases_by_id: dict[str, ReconciliationCase] = {
        "big": _case("big", "100000.00", "100000.00"),
        **{f"s{i}": _case(f"s{i}", "10.00", "10.00") for i in range(99)},
    }
    metrics = population_a(_result(cases), cases_by_id, _truth_for(dict.fromkeys(cases_by_id)))
    case_rate = metrics.case_match_rate_case_count
    gross_rate = metrics.gross_exposure_match_rate_expected_gross
    print(f"\n  case rate={case_rate}  gross-exposure rate={gross_rate}")
    assert case_rate is not None and gross_rate is not None
    assert case_rate > Decimal("0.98")
    assert gross_rate < Decimal("0.15")
    assert metrics.gross_exposure_total_expected_gross == money("100990.00")


def test_3_false_match_counts_only_confirmed_but_wrong() -> None:
    """3. The denominator is CONFIRMATIONS, not all cases."""
    cases = (
        _outcome("right", confirmed=True, category="PARTIAL_CAPTURE"),
        _outcome("wrong", confirmed=True, category="PARTIAL_CAPTURE"),
        _outcome("open", confirmed=False, category="DUPLICATE_CANDIDATE"),
    )
    cases_by_id = {c.case_id: _case(c.case_id, "10.00", "10.00") for c in cases}
    truth = _truth_for({"right": "PARTIAL_CAPTURE", "wrong": "T_PLUS_N_TIMING", "open": None})
    metrics = population_a(_result(cases), cases_by_id, truth)
    assert metrics.residual_false_match_rate_case_count == Decimal("0.500000"), (
        "one of two CONFIRMATIONS was wrong; the rate must divide by 2, not 3"
    )


def test_4_abstentions_never_count_as_false_matches() -> None:
    """4. An open case disagreeing with truth is over-inclusion, not a false
    match. Merging the two would make a cautious engine look reckless."""
    cases = (
        _outcome("a", confirmed=True, category=None),
        _outcome("b", confirmed=False, category="DUPLICATE_CANDIDATE"),
        _outcome("c", confirmed=False, category="UNEXPLAINED"),
    )
    cases_by_id = {c.case_id: _case(c.case_id, "10.00", "10.00") for c in cases}
    metrics = population_a(
        _result(cases), cases_by_id, _truth_for({"a": None, "b": None, "c": None})
    )
    assert metrics.residual_false_match_rate_case_count == Decimal("0.000000")
    assert metrics.deterministic_residual_count == 2


def test_5_cost_per_resolution_with_zero_resolutions_is_none() -> None:
    """5. None, not ZeroDivisionError and not zero.

    Raising would crash a report over a run that legitimately explained
    nothing; returning 0 would advertise a free AI layer that did no work.
    """
    assert cost_per_resolved_exception(money("500.00"), 0) is None
    assert cost_per_resolved_exception(money("500.00"), 4) == money("125.00")


@pytest.mark.boundary_refusal
def test_5b_a_negative_resolution_count_raises() -> None:
    """FAULT INJECTION."""
    with pytest.raises(ValueError, match=">= 0"):
        cost_per_resolved_exception(money("1.00"), -1)


def test_6_analyst_minutes_states_the_assumption(result: ReconciliationResult) -> None:
    """6. The label carries "derived estimate" and the assumed value."""
    estimate = analyst_minutes_saved(result, minutes_per_review=7).as_dict()
    assert "derived estimate" in estimate["label"]
    assert "7 min/review" in estimate["label"]
    assert estimate["minutes_per_review_assumption"] == 7


def test_7_deterministic_and_ai_savings_are_distinct_keys(result: ReconciliationResult) -> None:
    """7. Separate keys, no blended field, and the split is printed."""
    estimate = analyst_minutes_saved(result, minutes_per_review=4).as_dict()
    assert "minutes_saved_deterministic_derived_estimate" in estimate
    assert "minutes_saved_ai_derived_estimate" in estimate
    blended = [k for k in estimate if any(w in k for w in ("total", "combined", "blended"))]
    assert not blended, f"a blended minutes key exists: {blended}"
    print(
        f"\n  deterministic={estimate['minutes_saved_deterministic_derived_estimate']} min "
        f"| ai={estimate['minutes_saved_ai_derived_estimate']} min"
    )
    assert estimate["minutes_saved_deterministic_derived_estimate"] > 0
    assert estimate["minutes_saved_ai_derived_estimate"] == 0


def test_8_every_money_metric_is_decimal(
    result: ReconciliationResult, dataset: DayDataset, config: AppConfig, truth: TruthView
) -> None:
    """8. `type(...) is Decimal`, swept across all three populations."""
    cases_by_id = {f.case.case_id: f.case for f in build_cases(dataset, config)}
    checked = 0
    for metrics in (
        population_a(result, cases_by_id, truth),
        population_b(result, truth),
        population_c(result, truth),
    ):
        for name in dir(metrics):
            if name.startswith("_"):
                continue
            value = getattr(metrics, name)
            if isinstance(value, Decimal):
                assert type(value) is Decimal, f"{name} is {type(value).__name__}"
                checked += 1
    assert checked >= 12, f"only {checked} Decimal fields swept"


def test_9_the_truth_defect_is_reported_both_ways(payload: dict[str, Any]) -> None:
    """9. Both keys exist, both populated, and the difference is printed."""
    b = payload["population_b_batch_count_denominator"]
    counting = b["noise_recovery_rate_counting_defect"]
    excluding = b["noise_recovery_rate_excluding_defect"]
    assert counting is not None and excluding is not None
    assert b["defect_batches_excluded"] == sorted(TRUTH_DEFECT_BATCHES)
    print(
        f"\n  BAT_16A0609791AB: recovery counting defect={counting} excluding={excluding} "
        f"difference={Decimal(excluding) - Decimal(counting)}"
    )
    assert counting != excluding, "excluding the defect changes nothing; this is theatre"


# ---------------------------------------------------------------------------
# 10-13. Baselines. NO RANKING.
# ---------------------------------------------------------------------------


def test_10_naive_runs_end_to_end_and_no_ranking_is_asserted(
    dataset: DayDataset, config: AppConfig, truth: TruthView, result: ReconciliationResult
) -> None:
    """10. Both produce valid metrics. Whatever the ordering, it is a finding."""
    naive_links = run_naive(dataset, config, AS_OF)
    naive_wrong = [
        link for link in naive_links if truth.batch_credit(link.batch_id) != link.bank_txn_id
    ]
    engine = population_b(result, truth)
    print(
        f"\n  naive:  linked={len(naive_links)} false={len(naive_wrong)}"
        f"\n  engine: linked={engine.linked_count} false={engine.false_link_count}"
    )
    assert naive_links, "the naive baseline linked nothing; it is not a baseline"
    assert engine.linked_count > 0


def test_no_ranking_assertion_exists_in_this_module() -> None:
    """The instruction was explicit: if a test needs a baseline to lose, delete it.

    Scans this file for an assert comparing naive to the engine. A ranking
    assertion would make the headline claim unfalsifiable.
    """
    offenders = [
        line.strip()
        for line in Path(__file__).read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("assert")
        and "naive" in line.lower()
        and any(op in line for op in ("<", ">"))
    ]
    assert not offenders, f"a baseline ranking is asserted: {offenders}"


def test_11_deterministic_only_reproduces_the_committed_numbers(payload: dict[str, Any]) -> None:
    """11. Against the COMMITTED results file, not numbers in a brief.

    A brief's figures are a claim about the code; the committed results are
    what the code produced. Comparing to the latter catches a regression - the
    former only catches a typo.
    """
    committed_path = REPO / "reports" / "eval" / "results.json"
    assert committed_path.exists(), "no committed results to compare against"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    live_a = payload["population_a_case_count_denominator"]
    committed_a = committed["population_a_case_count_denominator"]
    for key in (
        "case_count",
        "confirmed_case_count",
        "deterministic_residual_count",
        "residual_false_match_rate_case_count",
    ):
        assert live_a[key] == committed_a[key], f"{key} drifted from the committed run"
    live_b = payload["population_b_batch_count_denominator"]
    print(
        f"\n  resolved={live_a['confirmed_case_count']} "
        f"residual={live_a['deterministic_residual_count']} "
        f"linked={live_b['linked_count']}/{live_b['batch_count']} "
        f"false_links={live_b['false_link_count']}"
    )
    assert live_b["false_link_count"] == 0
    assert live_a["residual_false_match_rate_case_count"] == "0.000000"


def test_12_llm_only_runs_against_the_replay_client_with_no_network(
    dataset: DayDataset, config: AppConfig, tmp_path: Path
) -> None:
    """12. A full run on recorded fixtures. The socket guard is autouse, so a
    real call raises rather than bills."""
    credits = sorted(
        (r for r in dataset.bank_rows if r.direction is BankDirection.CREDIT),
        key=lambda r: r.bank_txn_id,
    )
    batches = sorted(dataset.settlement_batches, key=lambda b: b.batch_id)
    assert credits, "no credits in the dataset; the baseline would be untested"

    for start in range(0, len(credits), ROWS_PER_CHUNK):
        chunk = credits[start : start + ROWS_PER_CHUNK]
        answers = [
            {
                "bank_txn_id": row.bank_txn_id,
                "batch_id": select_candidates(row, batches)[0].batch_id,
                "confidence": 0.8,
                "reasoning": "closest amount inside the window",
            }
            for row in chunk
        ]
        (tmp_path / f"{prompt_hash(build_prompt(chunk, batches))}.json").write_text(
            # The client protocol became `complete(prompt, schema) -> dict` at
            # M7, so a fixture is {"prompt": ..., "response": {...}} rather than
            # a text blob with usage attached.
            json.dumps({"prompt": "recorded", "response": {"links": answers}})
        )

    client = ReplayLLMClient(fixture_dir=tmp_path)
    outcome = run_llm_only(dataset, config, AS_OF, client)
    print(
        f"\n  llm_only: prompts={outcome.prompts_sent} links={len(outcome.links)} "
        f"parse_failures={outcome.parse_failures}"
    )
    assert outcome.prompts_sent == len(client.calls) > 0
    assert len(outcome.links) == len(credits)
    assert outcome.parse_failures == 0


def test_13_every_baseline_consumes_the_identical_dataset_object(
    dataset: DayDataset, config: AppConfig, truth: TruthView
) -> None:
    """13. The SAME object, by identity - not an equal copy.

    A baseline handed its own reload could differ by a parse decision, and the
    comparison would be measuring ingestion rather than reconciliation.
    """
    seen: list[int] = []
    for call in (
        lambda ds: run_naive(ds, config, AS_OF),
        lambda ds: run(ds, config, AS_OF),
        lambda ds: select_candidates(
            next(r for r in ds.bank_rows if r.direction is BankDirection.CREDIT),
            list(ds.settlement_batches),
        ),
    ):
        seen.append(id(dataset))
        call(dataset)
    assert len(set(seen)) == 1, "the baselines did not all receive one dataset object"
    assert run_baselines(dataset, config, truth, AS_OF, ("naive", "det", "settlesense"))


# ---------------------------------------------------------------------------
# 14-18. The AI evaluation set
# ---------------------------------------------------------------------------

_EVAL_PRESENT = EVAL_DIR.is_dir() and len(list(EVAL_DIR.glob("seed_*"))) == len(EVAL_SEEDS)
needs_eval_data = pytest.mark.skipif(
    not _EVAL_PRESENT,
    reason="data/eval absent (gitignored, ~146MB). Regenerate with `make eval-set`.",
)


def _eval_manifest() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((REPO / "EVAL_SET_MANIFEST.json").read_text(encoding="utf-8"))
    return data


def test_14_twenty_datasets_exist_and_are_recorded() -> None:
    """14. From the COMMITTED manifest, so this runs on a fresh clone where the
    146MB of data is absent."""
    manifest = _eval_manifest()
    assert sorted(int(s) for s in manifest["seeds"]) == list(EVAL_SEEDS)
    assert manifest["seed_range"]["count"] == len(EVAL_SEEDS)
    assert manifest["seed_range"]["excluded"] == []
    for seed, row in manifest["seeds"].items():
        assert row["file_count"] > 0, f"seed {seed} recorded zero files"
    print(f"\n  {len(manifest['seeds'])} datasets recorded, 0 excluded")


def test_15_per_seed_pair_count_is_stable_within_half_the_median() -> None:
    """15. All twenty printed. None may deviate from the median by >50%."""
    pairs = {
        int(seed): row["duplicate_candidate_pairs"]
        for seed, row in _eval_manifest()["seeds"].items()
    }
    median = statistics.median(pairs.values())
    print(f"\n  per-seed DUPLICATE_CANDIDATE pairs (median {median}):")
    for seed in sorted(pairs):
        print(f"    {seed}: {pairs[seed]:>3}   {(pairs[seed] - median) / median:+.1%} from median")
    outliers = {seed: count for seed, count in pairs.items() if abs(count - median) / median > 0.5}
    assert not outliers, (
        f"seed(s) deviate from the median by more than 50%: {outliers}. A noise rate "
        "is interacting with something seed-dependent - investigate before using it."
    )


def test_16_total_pair_count_is_at_least_three_hundred() -> None:
    """16. This is the n the AI claim will rest on.

    Printed because the number bounds what can honestly be claimed: at n=507 a
    single wrong decision moves precision by 0.2 points; at n=26 it moves it by
    4. If this ever falls, the claim must be weakened to match.
    """
    manifest = _eval_manifest()
    total = sum(row["duplicate_candidate_pairs"] for row in manifest["seeds"].values())
    print(f"\n  TOTAL decisions across {len(manifest['seeds'])} seeds: {total}")
    assert total >= 300, (
        f"only {total} decisions. The AI claim rests on this n; below ~300 the error "
        "bars swallow any per-category figure."
    )


@needs_eval_data
def test_17_evaluation_seeds_are_disjoint_from_dev_and_holdout() -> None:
    """17. R1 - disjoint in seed NUMBER and in generated ID space.

    Seed numbers are the easy half. The ID check is the one that matters: every
    id is a hash of a canonical tuple that includes the seed (D10), so an
    overlap would mean the seed is not reaching the hash and two "independent"
    datasets share rows.
    """
    assert set(EVAL_SEEDS).isdisjoint(RESERVED_SEEDS)

    def order_ids(root: Path) -> set[str]:
        return {
            line.split(",")[0]
            for path in sorted(root.glob("day*_ledger.csv"))
            for line in path.read_text(encoding="utf-8").splitlines()[1:]
            if line
        }

    dev = order_ids(DATA)
    assert dev, "no dev order ids read; the check would be vacuous"
    overlaps = {
        seed: len(dev & order_ids(EVAL_DIR / f"seed_{seed}"))
        for seed in EVAL_SEEDS
        if dev & order_ids(EVAL_DIR / f"seed_{seed}")
    }
    print(f"\n  dev order ids: {len(dev)}; evaluation seeds sharing any id: {len(overlaps)}")
    assert not overlaps, (
        f"generated ID space overlaps between the dev seed and {overlaps}. The seed is "
        "not reaching the id hash (D10)."
    )


def test_18_the_seed_range_is_recorded_in_the_readme_before_m7() -> None:
    """18. A range chosen after seeing results is not a held-out range.

    Greps the README, and SEPARATELY checks that the declaring commit precedes
    the commit carrying the numbers - the file alone proves only that the text
    exists now, not that it existed first.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "1000" in readme and "1019" in readme, "the README does not record the range"
    assert re.search(r"before.{0,60}generat", readme, re.IGNORECASE | re.DOTALL), (
        "the README does not say the range was declared before generation"
    )

    declaring = _eval_manifest()["declared_in_commit"]
    log = subprocess.run(
        ["git", "log", "--format=%H %s", "--reverse"], capture_output=True, text=True, cwd=REPO
    ).stdout.splitlines()
    order = [line.split()[0][:7] for line in log]
    assert declaring[:7] in order, f"declaring commit {declaring} not in history"
    generating = next(
        (line.split()[0][:7] for line in log if "generate seeds 1000-1019" in line), None
    )
    assert generating, "no commit generating the seeds found"
    print(f"\n  declared in {declaring[:7]}, generated in {generating}")
    assert order.index(declaring[:7]) < order.index(generating), (
        "the range was declared AFTER the seeds were generated"
    )


# ---------------------------------------------------------------------------
# 19-23. The runner
# ---------------------------------------------------------------------------

DOCUMENTED_NULLS = frozenset(
    {
        "residual_explanation_precision_case_count",  # the AI layer has not run
    }
)


def test_19_results_json_has_every_key_and_no_undocumented_nulls(
    payload: dict[str, Any],
) -> None:
    """19."""
    for section in (
        "population_a_case_count_denominator",
        "population_b_batch_count_denominator",
        "population_c_row_count_denominator",
        "analyst_time",
        "residual_set_sentence",
    ):
        assert section in payload, f"results.json is missing {section}"
    nulls = [
        f"{section}.{key}"
        for section in payload
        if isinstance(payload[section], dict)
        for key, value in payload[section].items()
        if value is None and key not in DOCUMENTED_NULLS
    ]
    print(f"\n  undocumented nulls: {nulls or 'none'}")
    assert not nulls, f"undocumented nulls: {nulls}"


def test_20_two_runs_produce_identical_results(
    dataset: DayDataset, config: AppConfig, truth: TruthView
) -> None:
    """20. D6."""
    first = evaluate(dataset, config, truth, AS_OF, 4)
    second = evaluate(dataset, config, truth, AS_OF, 4)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_21_the_markdown_carries_real_numbers_not_placeholders(payload: dict[str, Any]) -> None:
    """21. No X, Y, Z or TO BE MEASURED survives into the report."""
    markdown = to_markdown(payload, {})
    sentence = payload["residual_set_sentence"]
    assert sentence in markdown
    for placeholder in ("TO BE MEASURED", "TODO", "<hash>", "**X**", "**Y**", "**Z**"):
        assert placeholder not in markdown, f"{placeholder!r} survived into the report"
    assert re.search(r"explained \d+, abstained on \d+, and false-matched \d+", sentence), (
        f"the residual sentence still carries symbols: {sentence!r}"
    )
    print(f"\n  {sentence}")


def test_22_make_eval_defaults_to_the_dev_seed() -> None:
    """22. A default pointing at the holdout burns it.

    Checks the --data path AND the truth file. The path is the requirement as
    written; the truth file is what actually decides which seed runs, so both
    are asserted rather than trusting a directory name to imply a seed.
    """
    lines = (REPO / "Makefile").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("eval:"))
    body = "\n".join(
        line for line in lines[start + 1 :] if line.startswith("\t") or not line.strip()
    ).split("\n\n")[0]
    print(f"\n  make eval recipe:{body}")
    assert "--data data/dev" in body, f"make eval does not point at the dev set: {body!r}"
    assert "holdout" not in body, "make eval references the holdout"
    assert "truth_42.json" in body, "make eval does not use the dev seed's truth"


def test_22b_eval_holdout_is_a_separate_target_that_warns() -> None:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "eval-holdout:" in text
    assert "HELD-OUT" in text and "Record whatever it prints" in text


def test_23_a_dataset_with_no_residuals_still_reports() -> None:
    """23. An empty residual set is a valid outcome, not a crash.

    The sentence must still render, with zeros, rather than raising or
    returning an empty string a reader would take for a missing result.
    """
    clean = _result(tuple(_outcome(f"c{i}", confirmed=True) for i in range(5)))
    cases_by_id = {f"c{i}": _case(f"c{i}", "10.00", "10.00") for i in range(5)}
    metrics = population_a(clean, cases_by_id, _truth_for(dict.fromkeys(cases_by_id)))
    assert metrics.deterministic_residual_count == 0
    assert metrics.residual_abstention_rate_case_count == Decimal("0.000000")
    sentence = residual_set_sentence(0, 0, 0, 0)
    assert "Of the 0 exceptions" in sentence
    print(f"\n  {sentence}")


def test_23b_the_runner_writes_both_artifacts(tmp_path: Path) -> None:
    """The two files the brief names, written where they are asked for."""
    exit_code = main(
        [
            "--data",
            str(DATA),
            "--truth",
            str(DATA / "truth_42.json"),
            "--out",
            str(tmp_path),
            "--config",
            str(REPO / "config"),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.md").exists()
    assert_no_ambiguous_money_keys(json.loads((tmp_path / "results.json").read_text()))


# --------------------------------------------------------------------------
# Beyond the brief: the README quotes numbers. Nothing checked them.
#
# Every figure in the README Results section was typed by hand from a run.
# That is the weakest link in the whole project: the engine can be correct,
# the metrics correct, the artifact correct, and the README still say
# something else - and a green suite would never notice, because no test read
# the README. A wrong number in the one file a reader actually reads is worse
# than a wrong number anywhere in the codebase.
# --------------------------------------------------------------------------

COMMITTED_RESULTS = REPO / "reports" / "eval" / "results.json"


def _flatten_readme_number(text: str) -> str:
    """README prose formatting -> comparable digits.

    The README writes money for humans (`₹72,204,883.74`) and the artifact
    writes it for machines (`72204883.74`). Only the separators differ, so
    they are stripped rather than the comparison being loosened.
    """
    text = text.replace("₹", "").replace("**", "").replace("`", "")
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def _readme_dev_section(readme: str) -> str:
    """The dev-set block only.

    Bounded deliberately: a number found anywhere in a 200-line README proves
    nothing about the table it was supposed to be in.

    Ends at the Throughput section, which M5a added between here and the
    holdout block. Those figures come from reports/bench.md and are checked by
    tests/test_m5a_bench.py against THAT artifact - letting them sit inside
    this range would mean a bench number could satisfy an eval expectation.
    """
    start = readme.index("### Dev set")
    end = readme.index("### Throughput")
    return readme[start:end]


def _readme_expectations(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """(README label, value the artifact says it must quote).

    Built from the payload, never from literals - a hard-coded expectation
    would still pass after the engine changed, which is the failure this whole
    test exists to catch.
    """
    pop_a = payload["population_a_case_count_denominator"]
    pop_b = payload["population_b_batch_count_denominator"]
    pop_c = payload["population_c_row_count_denominator"]
    time = payload["analyst_time"]
    baselines = payload["baselines"]

    expectations: list[tuple[str, str]] = [
        ("ReconciliationCase", str(pop_a["case_count"])),
        ("Case match rate", pop_a["case_match_rate_case_count"]),
        ("Deterministic residual", str(pop_a["deterministic_residual_count"])),
        ("Residual false-match rate", pop_a["residual_false_match_rate_case_count"]),
        ("Gross-exposure match rate", pop_a["gross_exposure_match_rate_expected_gross"]),
        ("Expected-net cash reconciled", pop_a["expected_net_cash_reconciled_expected_net"]),
        ("Unresolved expected-net cash", pop_a["unresolved_expected_net_cash_expected_net"]),
        ("Evidence coverage", pop_a["evidence_coverage_case_count"]),
        ("batch↔bank links", str(pop_b["batch_count"])),
        ("Batch link rate", pop_b["batch_link_rate_batch_count"]),
        ("Batch link rate", f"{pop_b['linked_count']}/{pop_b['batch_count']}"),
        ("Batch false-link rate", pop_b["batch_false_link_rate_batch_count"]),
        (
            "Injected noise recovered",
            f"{pop_b['injected_noise_recovered_batch_count']}/{pop_b['injected_noise_batch_count']}",
        ),
        ("Injected noise recovered", pop_b["noise_recovery_rate_counting_defect"]),
        ("Injected noise recovered", pop_b["noise_recovery_rate_excluding_defect"]),
        ("Category precision on unresolved", pop_b["category_precision_on_unresolved_batch_count"]),
        ("row-grain variances", str(pop_c["row_count"])),
        ("| Recall", pop_c["row_variance_recall_row_count"]),
        ("| Precision", pop_c["row_variance_precision_row_count"]),
        ("| Value", pop_c["row_variance_value_row_value"]),
        ("deterministic resolutions", str(time["deterministic_resolutions"])),
        ("deterministic resolutions", str(time["minutes_saved_deterministic_derived_estimate"])),
        ("| naive", str(baselines["naive"]["batches_linked"])),
        ("| deterministic_only", str(baselines["deterministic_only"]["batches_linked"])),
        ("| settlesense", str(baselines["settlesense"]["batches_linked"])),
    ]
    return expectations


def _readme_disagreements(section: str, payload: dict[str, Any]) -> list[str]:
    """Every disagreement, not the first.

    Returns a list so one wrong digit does not mask five others - a fix-one-
    rerun loop would let a stale README be corrected one number per commit
    while still being wrong in between.
    """
    lines = [_flatten_readme_number(line) for line in section.splitlines()]
    problems: list[str] = []
    for label, expected in _readme_expectations(payload):
        flat_label = _flatten_readme_number(label)
        row = next((line for line in lines if flat_label.lower() in line.lower()), None)
        if row is None:
            problems.append(f"no README row mentions {label!r}")
        elif expected not in row:
            problems.append(f"{label!r} should quote {expected!r}, README says: {row.strip()}")
    return problems


@pytest.mark.hygiene
def test_25_readme_numbers_match_the_committed_artifact() -> None:
    """Every quoted figure in the README traces to reports/eval/results.json.

    This is the test that makes the README a claim rather than an assertion.
    """
    assert COMMITTED_RESULTS.exists(), (
        "reports/eval/results.json is not committed. The README quotes it, so "
        "without the file the numbers are unfalsifiable."
    )
    payload = json.loads(COMMITTED_RESULTS.read_text(encoding="utf-8"))
    section = _readme_dev_section((REPO / "README.md").read_text(encoding="utf-8"))
    problems = _readme_disagreements(section, payload)
    assert not problems, "README disagrees with the artifact:\n  " + "\n  ".join(problems)
    checked = len(_readme_expectations(payload))
    assert checked >= 20, f"only {checked} figures cross-checked - the table is barely covered"
    print(f"\n  README figures cross-checked against results.json: {checked}, 0 disagreements")


@pytest.mark.hygiene
def test_25b_the_readme_cross_check_can_actually_fail() -> None:
    """Fault injection for test 25. Three ways a README goes stale.

    Without this, test 25 passing would be equally consistent with the checker
    inspecting an empty expectation list.
    """
    payload = json.loads(COMMITTED_RESULTS.read_text(encoding="utf-8"))
    section = _readme_dev_section((REPO / "README.md").read_text(encoding="utf-8"))
    assert not _readme_disagreements(section, payload), "precondition: README is currently clean"

    residual = str(payload["population_a_case_count_denominator"]["deterministic_residual_count"])
    mutations = {
        "a digit changed": section.replace(f"| **{residual}** |", "| **51** |"),
        "a row deleted": "\n".join(
            line for line in section.splitlines() if "Case match rate" not in line
        ),
        "money quietly rounded": section.replace("72,204,883.74", "72,204,884.00"),
    }
    for name, mutated in mutations.items():
        assert mutated != section, f"mutation {name!r} changed nothing - it tested nothing"
        problems = _readme_disagreements(mutated, payload)
        assert problems, f"the checker did not notice: {name}"
        print(f"\n  {name}: {problems[0][:90]}")


@pytest.mark.hygiene
def test_25c_the_holdout_was_run_once_and_its_numbers_are_recorded() -> None:
    """The holdout HAS now been run. This test guards the other half of the rule.

    It previously asserted the README said NOT YET RUN and that no artifact was
    tracked. That was the right guard until 2026-08-27, when the single run
    happened; keeping it would have forced a choice between a stale README and
    a failing suite.

    What it guards now: the numbers are recorded, the artifact is committed so
    they are checkable rather than trusted, and the README says plainly that
    nothing was adjusted afterwards. A holdout figure that could be quietly
    improved after the fact is not a holdout figure.
    """
    results = REPO / "reports" / "eval-holdout" / "results.json"
    assert results.exists(), "the holdout was run but its artifact is not committed"
    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["seed"] == 999, payload["seed"]

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("### Held-out set") :]
    assert "NOT YET RUN" not in section, "the README still claims the holdout is unrun"
    assert "RUN ONCE" in section, "the README does not state it was a single run"
    assert "Nothing was adjusted afterwards" in section, (
        "the README does not state that nothing was tuned in response"
    )

    # THE DISAGREEMENT MUST BE REPORTED, not smoothed. The holdout produced a
    # false-match rate the dev set did not, and a README that quoted only the
    # favourable figures would be the exact failure this discipline exists to
    # prevent.
    pop_a = payload["population_a_case_count_denominator"]
    rate = Decimal(pop_a["residual_false_match_rate_case_count"])
    assert rate > 0, "this test assumes the holdout disagreed; it no longer does"
    assert pop_a["residual_false_match_rate_case_count"] in section, (
        f"the README does not quote the holdout false-match rate {rate}"
    )
    assert "split_settlement" in section.lower(), (
        "the README does not name the withheld noise type responsible"
    )
    print(
        f"\n  holdout seed {payload['seed']}: false-match rate {rate}, "
        f"residual {pop_a['deterministic_residual_count']}, quoted in the README"
    )


@pytest.mark.hygiene
def test_25c2_the_holdout_breach_is_recorded_in_limitations() -> None:
    """A budget breach that lives only in a results table is a breach nobody reads.

    PDD 7.3 sets the residual false-match budget at 1%. The holdout returned
    1.0456%. That belongs in LIMITATIONS.md with what was NOT done about it.
    """
    limitations = (REPO / "LIMITATIONS.md").read_text(encoding="utf-8")
    assert "1.0456" in limitations, "the breached rate is not recorded"
    assert "Nothing was adjusted in response" in limitations, (
        "LIMITATIONS does not state that nothing was tuned after the holdout"
    )
    assert "SPLIT_SETTLEMENT" in limitations
    print("\n  the breach and the decision not to tune are both recorded")


@pytest.mark.hygiene
def test_25d_the_three_sets_are_described_with_their_distinct_rules() -> None:
    """Three sets, three rules. Collapsing them into two loses the distinction.

    The AI evaluation set is NOT a holdout: its results may be inspected and
    iterated against, and only its membership is frozen. A README that calls
    both of them held-out would overclaim on one and misuse the other.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for seed_marker in ("42", "999", "1000", "1019"):
        assert seed_marker in readme, f"seed {seed_marker} is not named in the README"
    assert "not a second holdout" in readme.lower(), (
        "the README does not distinguish the AI evaluation set from the holdout"
    )
    assert "Run ONCE" in readme, "the holdout's look-once rule is not stated"
    print("\n  three sets named with distinct rules: dev(42) ai-eval(1000-1019) holdout(999)")

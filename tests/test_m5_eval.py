"""M5 - metrics, baselines, the runner, and the target that must not point at
the holdout.

The single most consequential thing in this module is a Makefile line. If
`make eval` pointed at seed 999 it would be run dozens of times during
development and the holdout would quietly stop being held out - no bad
decision anywhere, just convenience. That is asserted first, because it is the
kind of mistake nothing else would catch.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from eval.baselines.llm_only import (
    CANDIDATES_PER_ROW,
    build_prompt,
    parse_response,
    retrieval_recall,
    select_candidates,
)
from eval.baselines.naive import naive_amount_only_agreement, run_naive
from eval.metrics import (
    TRUTH_DEFECT_BATCHES,
    TruthView,
    analyst_minutes_saved,
    assert_no_ambiguous_money_keys,
    population_a,
    population_b,
    population_c,
    residual_set_sentence,
)
from eval.run_eval import evaluate, load_days, to_markdown
from settlesense.ai.client import (
    RealLLMClient,
    ReplayLLMClient,
    ReplayMissError,
    prompt_hash,
)
from settlesense.config import AppConfig, load_config
from settlesense.ingest import DayDataset
from settlesense.matching.engine import run
from settlesense.types import BankDirection

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
AS_OF = date(2026, 11, 30)


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


# ---------------------------------------------------------------------------
# The Makefile: which seed each target runs
# ---------------------------------------------------------------------------


def _makefile() -> str:
    return (REPO / "Makefile").read_text(encoding="utf-8")


def _target_body(name: str) -> str:
    """The RECIPE lines only - tab-indented, up to the next unindented line.

    Not "everything until the next target line": the comment block documenting
    eval-holdout sits between eval's recipe and eval-holdout's, so a naive
    slice pulled the word "999" out of a COMMENT and failed the assertion that
    eval does not reference the holdout. The test was right; the helper was
    reading the wrong text.
    """
    lines = _makefile().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"{name}:")), None)
    assert start is not None, f"no {name} target in the Makefile"
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("\t") or not line.strip():
            body.append(line)
            continue
        break
    return "\n".join(body)


def test_make_eval_runs_the_dev_seed_not_the_holdout() -> None:
    """THE critical assertion in this file.

    A default target pointing at the holdout would be run dozens of times
    during development, and nothing in any output would say how many. The
    damage is silent and irreversible, so it is checked rather than
    remembered.
    """
    body = _target_body("eval")
    assert "truth_42.json" in body, "make eval does not run the dev seed"
    assert "999" not in body and "holdout" not in body, (
        "make eval references the holdout. The default target must be the DEV "
        "set; the holdout has its own target."
    )


def test_make_eval_holdout_is_separate_and_warns() -> None:
    body = _target_body("eval-holdout")
    assert "truth_999.json" in body and "data/holdout" in body
    assert "HELD-OUT" in body, "eval-holdout does not announce what it is"
    assert "Record whatever it prints" in body, "eval-holdout does not carry the required warning"


def test_the_three_eval_targets_are_distinct() -> None:
    """eval, eval-holdout and eval-ai must not collapse into one another."""
    bodies = {name: _target_body(name) for name in ("eval", "eval-holdout", "eval-ai")}
    assert len({b.strip() for b in bodies.values()}) == 3
    assert "data/eval" in bodies["eval-ai"]


# ---------------------------------------------------------------------------
# Money keys and the three populations
# ---------------------------------------------------------------------------


def test_no_metric_key_says_only_money_weighted(payload: dict[str, Any]) -> None:
    """PDD 8.4. Three bases exist; a key naming none of them is ambiguous."""
    assert_no_ambiguous_money_keys(payload)


@pytest.mark.charter_guard
def test_the_money_key_guard_catches_a_planted_violation() -> None:
    """FAULT INJECTION, including at depth.

    A guard that only inspected the top level would pass a nested violation,
    and every real metric lives nested under its population.
    """
    with pytest.raises(AssertionError, match="money_weighted"):
        assert_no_ambiguous_money_keys({"money_weighted_total": "1.00"})
    with pytest.raises(AssertionError, match="money_weighted"):
        assert_no_ambiguous_money_keys({"population_a": {"money_weighted_rate": "0.5"}})
    with pytest.raises(AssertionError, match="money_weighted"):
        assert_no_ambiguous_money_keys({"rows": [{"money_weighted": 1}]})


def test_every_money_metric_names_its_basis(payload: dict[str, Any]) -> None:
    """Each money key ends in the basis it is measured on."""
    bases = ("expected_gross", "expected_net", "batch_net_total", "row_value")
    money_words = ("value", "cash", "exposure", "total")
    checked = 0
    for section in (
        "population_a_case_count_denominator",
        "population_b_batch_count_denominator",
        "population_c_row_count_denominator",
    ):
        for key, value in payload[section].items():
            if not isinstance(value, str) or not any(w in key for w in money_words):
                continue
            if key.endswith("_count") or "rate" in key:
                continue
            assert key.endswith(bases), f"money key {key!r} does not name its basis"
            checked += 1
    assert checked >= 6, f"only {checked} money keys checked; the sweep is thin"


def test_the_three_denominators_are_reported_separately(payload: dict[str, Any]) -> None:
    """D11. Three objects, three denominators, and they differ."""
    a = payload["population_a_case_count_denominator"]["case_count"]
    b = payload["population_b_batch_count_denominator"]["batch_count"]
    c = payload["population_c_row_count_denominator"]["row_count"]
    print(f"\n  denominators: A={a} B={b} C={c}")
    assert len({a, b, c}) == 3, f"denominators collide: {a}, {b}, {c}"


def test_a_rate_over_an_empty_population_is_none_not_zero(
    config: AppConfig, truth: TruthView
) -> None:
    """None, because a rate over nothing is UNDEFINED.

    Returning 0 would report "0% false matches" for a run that matched
    nothing, which is the most flattering possible reading of no evidence.
    """
    from settlesense.types import ReconciliationResult

    empty = ReconciliationResult(
        cases=(),
        batch_links=(),
        row_variances=(),
        exceptions=(),
        calendar_version="calendar_v1",
        config_hash="x",
    )
    assert population_a(empty, {}, truth).case_match_rate_case_count is None
    assert population_b(empty, truth).batch_link_rate_batch_count is None
    assert population_c(empty, truth).row_variance_precision_row_count is None


# ---------------------------------------------------------------------------
# The truth defect, reported both ways
# ---------------------------------------------------------------------------


def test_the_known_truth_defect_is_reported_both_ways(payload: dict[str, Any]) -> None:
    """BAT_16A0609791AB is labelled ROUNDING_DIFFERENCE in truth but its batch
    and credit differ by exactly Rs0.00 - nothing is detectable.

    Both numbers are printed so a reader chooses, rather than inheriting an
    asterisk. The generator is frozen and was correctly not re-frozen.
    """
    b = payload["population_b_batch_count_denominator"]
    assert b["defect_batches_excluded"] == sorted(TRUTH_DEFECT_BATCHES)
    counting = b["noise_recovery_rate_counting_defect"]
    excluding = b["noise_recovery_rate_excluding_defect"]
    print(f"\n  noise recovery: counting defect={counting}  excluding={excluding}")
    assert counting is not None and excluding is not None
    assert counting != excluding, (
        "both figures are identical, so excluding the defect changes nothing and "
        "the dual reporting is now theatre - re-check that the defect batch is "
        "still in this population"
    )


def test_the_defect_batch_really_has_a_zero_difference(
    dataset: DayDataset, config: AppConfig
) -> None:
    """The claim the dual reporting rests on, verified against the data.

    If this batch ever DID show a difference, the truth label would be
    correct and the exclusion would be unjustified.
    """
    result = run(dataset, config, AS_OF)
    for link in result.batch_links:
        if link.batch_id in TRUTH_DEFECT_BATCHES:
            assert link.linked_amount is not None
            assert link.batch_net_total - link.linked_amount == Decimal("0.00"), (
                f"{link.batch_id} differs by "
                f"{link.batch_net_total - link.linked_amount}; the truth label is "
                "detectable after all and the exclusion is unjustified"
            )
            return
    pytest.fail("the defect batch is not in the result; the exclusion list is stale")


def test_category_precision_is_computed_only_where_a_category_is_a_claim(
    payload: dict[str, Any],
) -> None:
    """The metric bug this file exists partly to prevent recurring.

    truth's `true_category` records what noise was INJECTED. The engine's
    category records what variance REMAINS. Comparing them across all batches
    scored 0.64 and penalised the engine for succeeding on 13 batches it had
    recovered. Precision is now computed over UNRESOLVED batches only.
    """
    b = payload["population_b_batch_count_denominator"]
    assert b["unresolved_batch_count"] < b["batch_count"], (
        "every batch is unresolved; this metric is no longer distinguishing"
    )
    assert b["injected_noise_recovered_batch_count"] > 0
    print(
        f"\n  unresolved={b['unresolved_batch_count']} "
        f"category precision={b['category_precision_on_unresolved_batch_count']}"
        f"\n  noise recovered={b['injected_noise_recovered_batch_count']}"
        f"/{b['injected_noise_batch_count']}"
    )


# ---------------------------------------------------------------------------
# Analyst minutes: the assumption, and the attribution
# ---------------------------------------------------------------------------


@pytest.mark.boundary_refusal
def test_analyst_minutes_requires_the_assumption(dataset: DayDataset, config: AppConfig) -> None:
    """FAULT INJECTION. No default, because a default lets the number be quoted
    without the premise it rests on."""
    import inspect

    signature = inspect.signature(analyst_minutes_saved)
    assert signature.parameters["minutes_per_review"].default is inspect.Parameter.empty, (
        "minutes_per_review has a default; the assumption can now be omitted"
    )
    result = run(dataset, config, AS_OF)
    with pytest.raises(TypeError, match="must be an int"):
        analyst_minutes_saved(result, "4")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        analyst_minutes_saved(result, 0)


def test_analyst_minutes_are_attributed_not_blended(dataset: DayDataset, config: AppConfig) -> None:
    """Rules and AI are reported separately, and there is no combined field.

    On this dataset the deterministic engine resolves ~99% and the AI layer has
    not run. One blended figure would read as a claim about the AI.
    """
    result = run(dataset, config, AS_OF)
    estimate = analyst_minutes_saved(result, minutes_per_review=4)
    as_dict = estimate.as_dict()
    assert "derived estimate" in as_dict["label"]
    assert "4 min/review" in as_dict["label"]
    assert as_dict["minutes_saved_ai_derived_estimate"] == 0, "the AI layer has not run"
    assert as_dict["minutes_saved_deterministic_derived_estimate"] > 0
    combined = [k for k in as_dict if "total" in k or "combined" in k or "blended" in k]
    assert not combined, f"a combined minutes field exists: {combined}"
    print(f"\n  {as_dict['label']}")


# ---------------------------------------------------------------------------
# The headline sentence
# ---------------------------------------------------------------------------


def test_the_residual_sentence_carries_real_numbers(payload: dict[str, Any]) -> None:
    sentence = payload["residual_set_sentence"]
    residual = payload["population_a_case_count_denominator"]["deterministic_residual_count"]
    assert str(residual) in sentence
    assert "false-matched 0" in sentence
    print(f"\n  {sentence}")


@pytest.mark.boundary_refusal
def test_the_sentence_refuses_to_overstate_the_residual_set() -> None:
    """FAULT INJECTION. Outcomes cannot exceed the surface they came from."""
    with pytest.raises(ValueError, match="exceed the residual"):
        residual_set_sentence(residual_count=10, explained=8, abstained=5, false_matched=0)


# ---------------------------------------------------------------------------
# Baselines - no ranking asserted anywhere
# ---------------------------------------------------------------------------


def test_the_naive_baseline_runs_and_reports_both_volume_and_precision(
    dataset: DayDataset, config: AppConfig, truth: TruthView
) -> None:
    """NO RANKING IS ASSERTED. Naive may link more; what differs is precision.

    Both counts are printed side by side so the distinction is a number rather
    than an argument.
    """
    links = run_naive(dataset, config, AS_OF)
    wrong = [link for link in links if truth.batch_credit(link.batch_id) != link.bank_txn_id]
    engine_linked = sum(1 for b in run(dataset, config, AS_OF).batch_links if b.bank_row_id)
    print(
        f"\n  naive: linked={len(links)} false={len(wrong)}"
        f"\n  engine: linked={engine_linked}"
        f"\n  batch amount uniqueness: {naive_amount_only_agreement(dataset)}"
    )
    assert links, "the naive baseline linked nothing; it is not a baseline"


def test_no_test_in_this_file_asserts_a_baseline_ranking() -> None:
    """The instruction was explicit: if a test needs a baseline to lose, delete it.

    Scans this module for comparisons between baseline link counts. A ranking
    assertion would make the headline claim unfalsifiable, which is worse than
    having no baseline at all.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("assert")
        and "naive" in line
        and any(op in line for op in ("<", ">"))
    ]
    assert not forbidden, f"a baseline ranking is asserted: {forbidden}"


def test_deterministic_and_settlesense_are_equal_until_m7(
    dataset: DayDataset, config: AppConfig, truth: TruthView
) -> None:
    """Stated rather than hidden. They become a before-and-after once M7 lands.

    Calls run_baselines directly rather than skipping when `evaluate` has no
    baselines key - a skip here would silently stop checking the claim the
    results table makes in prose.
    """
    from eval.run_eval import run_baselines

    baselines = run_baselines(dataset, config, truth, AS_OF, ("det", "settlesense"))
    det = baselines["deterministic_only"]
    full = baselines["settlesense"]
    assert det["batches_linked"] == full["batches_linked"]
    assert det["false_links"] == full["false_links"]
    assert "until M7" in full["note"], "the equality is not explained in the output"


# ---------------------------------------------------------------------------
# LLM baseline - strong, and testable without a network
# ---------------------------------------------------------------------------


def test_candidate_retrieval_is_deterministic_and_bounded(
    dataset: DayDataset,
) -> None:
    batches = list(dataset.settlement_batches)
    row = next(r for r in dataset.bank_rows if r.direction is BankDirection.CREDIT)
    first = select_candidates(row, batches)
    second = select_candidates(row, list(reversed(batches)))
    assert [b.batch_id for b in first] == [b.batch_id for b in second]
    assert len(first) == min(CANDIDATES_PER_ROW, len(batches))


def test_retrieval_recall_is_measured_so_a_miss_is_attributable(
    dataset: DayDataset, truth: TruthView
) -> None:
    """Without this a low score is unattributable: the model may have been
    wrong, or may never have been shown the right answer."""
    links = {bid: truth.batch_credit(bid) for bid in truth.batch_links}
    recall = retrieval_recall(dataset, links)
    print(f"\n  candidate retrieval recall (top {CANDIDATES_PER_ROW}): {recall}")
    assert recall is not None
    assert recall > Decimal("0.5"), (
        f"retrieval recall is {recall}; the LLM baseline is being handed candidate "
        "lists that rarely contain the answer, which would make it a strawman"
    )


def test_the_prompt_states_the_domain_rules_and_permits_abstention(
    dataset: DayDataset,
) -> None:
    """A baseline denied domain knowledge is a strawman."""
    batches = list(dataset.settlement_batches)
    rows = [r for r in dataset.bank_rows if r.direction is BankDirection.CREDIT][:2]
    prompt = build_prompt(rows, batches)
    for required in ("UTR", "Rs1.00", "null", "working days", "JSON"):
        assert required in prompt, f"the prompt never mentions {required!r}"
    assert "wrong link is worse than no link" in prompt
    assert all(row.bank_txn_id in prompt for row in rows)


def test_a_null_batch_id_parses_as_an_abstention_not_a_failure() -> None:
    """Collapsing an abstention into a parse failure would score the baseline
    down for doing the right thing."""
    links, failures = parse_response(
        '[{"bank_txn_id":"BNK_1","batch_id":null,"confidence":0.2,"reasoning":"no evidence"}]'
    )
    assert failures == 0
    assert len(links) == 1 and links[0].batch_id is None


def test_the_parser_survives_fenced_and_malformed_output() -> None:
    fenced, failures = parse_response(
        '```json\n[{"bank_txn_id":"BNK_1","batch_id":"BAT_1","confidence":0.9}]\n```'
    )
    assert failures == 0 and len(fenced) == 1
    broken, broken_failures = parse_response("not json at all")
    assert broken == [] and broken_failures == 1


@pytest.mark.boundary_refusal
def test_a_replay_miss_raises_and_never_reaches_the_network(tmp_path: Path) -> None:
    """FAULT INJECTION for SDD 7. A silent fallback would bill for a test run."""
    client = ReplayLLMClient(fixture_dir=tmp_path)
    with pytest.raises(ReplayMissError, match="no recorded response"):
        client.complete("a prompt nobody recorded")


def test_a_recorded_fixture_replays(tmp_path: Path) -> None:
    prompt = "hello"
    (tmp_path / f"{prompt_hash(prompt)}.json").write_text(
        json.dumps({"text": "[]", "input_tokens": 5, "output_tokens": 2, "model": "test"})
    )
    response = ReplayLLMClient(fixture_dir=tmp_path).complete(prompt)
    assert response.text == "[]" and response.total_tokens == 7


@pytest.mark.boundary_refusal
def test_the_real_client_refuses_to_exist_inside_a_test() -> None:
    """FAULT INJECTION. The suite must be INCAPABLE of billing anyone, which is
    a property of the type rather than of who remembered to check."""
    with pytest.raises(RuntimeError, match="must not be constructed inside a test"):
        RealLLMClient()


# ---------------------------------------------------------------------------
# The runner end to end
# ---------------------------------------------------------------------------


def test_the_markdown_report_carries_the_headline_and_the_bases(
    payload: dict[str, Any],
) -> None:
    markdown = to_markdown(payload, {})
    assert payload["residual_set_sentence"] in markdown
    for basis in ("₹ expected gross", "₹ expected net", "₹ batch net total"):
        assert basis in markdown, f"the table never names the {basis} basis"
    assert "derived estimate" in markdown
    assert "never added together" in markdown


def test_the_report_states_that_no_ranking_is_claimed(payload: dict[str, Any]) -> None:
    assert "No ranking is claimed" in to_markdown(payload, {})


def test_evaluate_is_deterministic(
    dataset: DayDataset, config: AppConfig, truth: TruthView
) -> None:
    """D6."""
    first = evaluate(dataset, config, truth, AS_OF, 4)
    second = evaluate(dataset, config, truth, AS_OF, 4)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_population_a_reports_zero_false_matches_on_the_dev_seed(
    payload: dict[str, Any],
) -> None:
    """The M3 result, now reported through the metrics layer rather than
    re-derived by a test - so the number a reader sees is the number the
    pipeline produces."""
    a = payload["population_a_case_count_denominator"]
    print(
        f"\n  cases={a['case_count']} residual={a['deterministic_residual_count']} "
        f"false_match_rate={a['residual_false_match_rate_case_count']}"
    )
    assert a["residual_false_match_rate_case_count"] == "0.000000"
    assert a["gross_exposure_false_match_value_expected_gross"] == "0.00"

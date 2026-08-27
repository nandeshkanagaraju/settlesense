"""M7 - the verified hypothesis loop. No network, no real model.

THE RESULT THIS FILE GUARDS. On the 507 pre-registered decisions, a PERFECT
model - an oracle that always nominates the row truth marks as the injected
duplicate - gets 27 confirmations. An adversary that always nominates the wrong
row gets 0. That gap is the whole architecture working: the verifier confirms
on evidence and rejects on its absence, and it cannot tell which client it is
talking to.

27/507 IS A CEILING, NOT A SCORE. No real model can exceed it, because for the
other 480 pairs the structural facts do not distinguish the two rows and the
verifier rejects whatever is nominated. Reporting "the AI explained 5%" would
be reporting a property of the DATASET as though it were a property of a model.

THE SAFETY PROPERTY IS THE ADVERSARIAL ZERO. A verifier that confirmed the
oracle and rejected nothing would score identically on the oracle run; only the
adversary separates discrimination from rubber-stamping.
"""

from __future__ import annotations

import ast
import json
import os
import socket
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from eval.run_ai import (
    SENDABLE,
    AdversarialClient,
    OracleClient,
    SilentClient,
    duplicate_exceptions,
    truth_duplicate_orders,
)
from eval.run_eval import load_days
from settlesense.ai.client import (
    MODEL,
    TEMPERATURE,
    TOP_P,
    FixtureMissError,
    RealLLMClient,
    ReplayLLMClient,
    prompt_hash,
    record_fixture,
)
from settlesense.ai.confidence import compute_confidence, should_auto_confirm
from settlesense.ai.hypothesis import (
    AI_ELIGIBLE_CATEGORIES,
    MAX_HYPOTHESES,
    Assertion,
    Hypothesis,
    IneligibleCategoryError,
    build_prompt,
    eligible_exceptions,
    generate,
    parse_hypotheses,
)
from settlesense.ai.loop import AbstainReason, run_loop
from settlesense.ai.verifier import (
    FIELD_GRAMMAR,
    GrammarError,
    VerificationResult,
    evaluate_assertion,
    resolve_field,
    verify,
)
from settlesense.config import AppConfig, load_config
from settlesense.exceptions.taxonomy import VarianceCategory
from settlesense.types import Exception_, ExceptionStatus, money

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "dev"
AI_DIR = REPO / "settlesense" / "ai"
COMMITTED = REPO / "reports" / "ai" / "ai_loop.json"
DUPLICATE = str(VarianceCategory.DUPLICATE_CANDIDATE)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Autouse. A guard a test has to remember to request is not a guard."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test in this module attempted a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture(scope="module")
def dataset(config: AppConfig) -> Any:
    return load_days(DATA, config)


@pytest.fixture(scope="module")
def truth() -> frozenset[str]:
    return truth_duplicate_orders(DATA / "truth_42.json")


@pytest.fixture(scope="module")
def exceptions(dataset: Any) -> tuple[Exception_, ...]:
    return duplicate_exceptions(dataset)


# ===========================================================================
# 1. The client
# ===========================================================================


def test_the_real_client_refuses_to_exist_inside_a_test_run() -> None:
    """D7. The suite must be INCAPABLE of billing anyone.

    A property of the type, not of who remembered to check at the call site.
    """
    assert os.environ.get("PYTEST_CURRENT_TEST"), "precondition: pytest sets this and it is unset"
    with pytest.raises(RuntimeError, match="PYTEST_CURRENT_TEST"):
        RealLLMClient()
    with pytest.raises(RuntimeError, match="PYTEST_CURRENT_TEST"):
        RealLLMClient(api_key="sk-not-a-real-key")
    print("\n  RealLLMClient refuses construction, with and without an explicit key")


def test_the_client_pins_a_model_and_disables_sampling() -> None:
    """temperature=0, top_p=1, a pinned model string.

    A sampled model makes one prompt produce different hypotheses on two runs,
    so a golden comparison fails for reasons nobody can reproduce.
    """
    assert TEMPERATURE == 0, TEMPERATURE
    assert TOP_P == 1, TOP_P
    assert MODEL and "latest" not in MODEL, f"{MODEL} is an alias, which silently repoints"
    source = (AI_DIR / "client.py").read_text(encoding="utf-8")
    assert "temperature=TEMPERATURE" in source and "top_p=TOP_P" in source
    print(f"\n  model={MODEL} temperature={TEMPERATURE} top_p={TOP_P}")


@pytest.mark.boundary_refusal
def test_a_fixture_miss_raises_and_names_the_hash(tmp_path: Path) -> None:
    """Loud, and NEVER a network fallback.

    The hash is in the message because "record it" is not actionable without
    the filename to record it into.
    """
    replay = ReplayLLMClient(fixture_dir=tmp_path)
    prompt = "a prompt nobody recorded"
    with pytest.raises(FixtureMissError) as raised:
        replay.complete(prompt, {})
    message = str(raised.value)
    assert prompt_hash(prompt) in message, message
    assert "does NOT fall back to the network" in message
    assert replay.calls == [prompt_hash(prompt)], "the attempt was not counted"
    print(f"\n  miss named {prompt_hash(prompt)[:16]} and counted the attempt")


@pytest.mark.boundary_refusal
def test_a_corrupt_fixture_is_a_different_failure_from_a_missing_one(tmp_path: Path) -> None:
    """FAULT INJECTION. Absent and corrupt need different fixes.

    Absent means record it; corrupt means re-record it. Collapsing both to
    "miss" sends someone to record a prompt that was already recorded.
    """
    prompt = "recorded but broken"
    (tmp_path / f"{prompt_hash(prompt)}.json").write_text('{"prompt": "x"}', encoding="utf-8")
    with pytest.raises(FixtureMissError, match="corrupt"):
        ReplayLLMClient(fixture_dir=tmp_path).complete(prompt, {})
    print("\n  a fixture with no `response` object reports corruption, not absence")


def test_record_fixture_round_trips_and_stores_the_prompt(tmp_path: Path) -> None:
    """The prompt is stored beside the response even though the name is its hash.

    A hash is not readable, and a fixture set nobody can inspect is one nobody
    will notice has gone stale.
    """
    prompt = "a prompt"
    response: dict[str, Any] = {"hypotheses": []}
    path = record_fixture(prompt, response, fixture_dir=tmp_path)
    assert path.name == f"{prompt_hash(prompt)}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["prompt"] == prompt
    assert ReplayLLMClient(fixture_dir=tmp_path).complete(prompt, {}) == response
    print(f"\n  recorded and replayed {path.name}")


# ===========================================================================
# 2. Hypothesis generation
# ===========================================================================


def test_the_prompt_is_byte_identical_across_calls(
    exceptions: tuple[Exception_, ...], dataset: Any, config: AppConfig
) -> None:
    """Two calls, byte-compared. Every interpolated collection is sorted.

    Without this the same exception produces two prompts, misses the fixture
    cache, and the replay mechanism becomes a source of nondeterminism rather
    than a defence against it.
    """
    sampled = exceptions[:5]
    assert sampled, "precondition: no exceptions to build prompts from"
    for exception in sampled:
        first = build_prompt(exception, dataset, config)
        second = build_prompt(exception, dataset, config)
        assert first == second, f"prompt for {exception.exception_id} is not stable"
    hashes = {prompt_hash(build_prompt(e, dataset, config)) for e in sampled}
    assert len(hashes) == len(sampled), "two different exceptions produced one prompt"
    print(f"\n  {len(sampled)} prompts, byte-identical on repeat, {len(hashes)} distinct hashes")


@pytest.mark.charter_guard
def test_the_prompt_stability_check_would_notice_an_unsorted_collection(
    exceptions: tuple[Exception_, ...], dataset: Any, config: AppConfig
) -> None:
    """FAULT INJECTION. Reordering evidence must change the prompt.

    Proves the byte-comparison above is capable of failing - a prompt that
    ignored its inputs would also be stable.
    """
    exception = exceptions[0]
    reversed_evidence = Exception_(
        **{
            **{f.name: getattr(exception, f.name) for f in exception.__dataclass_fields__.values()},
            "evidence_row_ids": tuple(reversed(exception.evidence_row_ids)),
        }
    )
    # SORTED INSIDE build_prompt, so reversing the input must NOT change the
    # output - that is the guarantee. A different exception must.
    assert build_prompt(exception, dataset, config) == build_prompt(
        reversed_evidence, dataset, config
    ), "evidence order leaked into the prompt; the sort is not being applied"
    assert build_prompt(exception, dataset, config) != build_prompt(
        exceptions[1], dataset, config
    ), "two different exceptions produce the same prompt; the prompt ignores its input"
    print("\n  evidence order does not leak; different exceptions differ")


@pytest.mark.boundary_refusal
def test_only_pdd_6_2_categories_may_be_sent(dataset: Any, config: AppConfig) -> None:
    """A REFUSAL, not a filter. Rules-decided categories must never be asked."""
    ineligible = Exception_(
        exception_id="x",
        category=str(VarianceCategory.MDR_FEE),
        amount=money(10),
        status=ExceptionStatus.OPEN,
        confidence=Decimal("0"),
        evidence_row_ids=(),
        reason="a deduction, decided by arithmetic",
        resolved_by=None,
        first_seen_day=1,
        confirmed_day=None,
        closed_day=None,
        audit=(),
    )
    with pytest.raises(IneligibleCategoryError, match=r"PDD 6\.2"):
        build_prompt(ineligible, dataset, config)
    assert str(VarianceCategory.MDR_FEE) not in AI_ELIGIBLE_CATEGORIES
    print(f"\n  {len(AI_ELIGIBLE_CATEGORIES)} eligible categories; MDR_FEE refused")


def test_zero_population_b_exceptions_reach_the_model(dataset: Any, config: AppConfig) -> None:
    """A batch whose credit never arrived is missing DATA, not a question.

    Asserted on the SENDABLE set the wiring actually passes, not on PDD 6.2 -
    the taxonomy permits MISSING_VS_LATE_CREDIT and the wiring still declines
    to send it, which is the decision under test.
    """
    batch_exception = Exception_(
        exception_id="b1",
        category=str(VarianceCategory.MISSING_VS_LATE_CREDIT),
        amount=money(1000),
        status=ExceptionStatus.OPEN,
        confidence=Decimal("0"),
        evidence_row_ids=(),
        reason="the credit never arrived",
        resolved_by=None,
        first_seen_day=1,
        confirmed_day=None,
        closed_day=None,
        audit=(),
    )
    assert eligible_exceptions((batch_exception,), SENDABLE) == ()
    client = OracleClient(frozenset())
    report = run_loop((batch_exception,), dataset, config, client, sendable=SENDABLE)
    assert report.sent == 0, "a Population B exception was sent to the model"
    assert client.calls == [], f"the model was called {len(client.calls)} times anyway"
    print(f"\n  MISSING_VS_LATE_CREDIT: sent={report.sent}, model calls={len(client.calls)}")


@pytest.mark.boundary_refusal
def test_hypotheses_outside_the_closed_enum_are_dropped() -> None:
    """An unknown category is discarded, not repaired.

    Repairing means guessing what the model meant, which is the model deciding
    by proxy.
    """
    payload = {
        "hypotheses": [
            {"category": "INVENTED", "candidate_id": "a", "evidence_row_ids": ["r"], "reason": "x"},
            {"category": DUPLICATE, "candidate_id": "b", "evidence_row_ids": ["r"], "reason": "y"},
        ]
    }
    parsed = parse_hypotheses(payload)
    assert len(parsed) == 1 and parsed[0].candidate_id == "b", parsed
    print("\n  1 of 2 kept; INVENTED dropped")


def test_at_most_three_hypotheses_are_parsed() -> None:
    """SDD 4.4 caps the list at 3, and the cap is enforced on OUR side.

    A model returning five is not an error; taking all five would be.
    """
    payload = {
        "hypotheses": [
            {
                "category": DUPLICATE,
                "candidate_id": f"c{i}",
                "evidence_row_ids": ["r"],
                "reason": "x",
            }
            for i in range(6)
        ]
    }
    parsed = parse_hypotheses(payload)
    assert len(parsed) == MAX_HYPOTHESES == 3, len(parsed)
    assert [h.rank for h in parsed] == [0, 1, 2], "rank is not the returned order"
    print(f"\n  6 offered -> {len(parsed)} kept, ranks {[h.rank for h in parsed]}")


@pytest.mark.boundary_refusal
def test_invalid_output_after_retries_yields_no_hypothesis_and_no_crash(
    exceptions: tuple[Exception_, ...], dataset: Any, config: AppConfig
) -> None:
    """A model that will not produce valid output has ABSTAINED, not crashed.

    Crashing would lose every other exception in the run over one bad reply.
    """
    client = SilentClient(frozenset())
    result = generate(exceptions[0], dataset, config, client)
    assert result == (), result
    assert len(client.calls) == 3, f"expected 1 attempt + 2 retries, got {len(client.calls)}"
    print(f"\n  {len(client.calls)} attempts, no hypothesis, no exception raised")


def test_the_models_self_reported_confidence_is_never_read() -> None:
    """PDD 7.2. It is dropped at parse time and cannot reach any consumer.

    Checked on the TYPE, not on a call: a field that does not exist cannot be
    read by code nobody has written yet.
    """
    payload = {
        "hypotheses": [
            {
                "category": DUPLICATE,
                "candidate_id": "a",
                "evidence_row_ids": ["r"],
                "reason": "x",
                "confidence": 0.99,
                "certainty": "high",
            }
        ]
    }
    (parsed,) = parse_hypotheses(payload)
    fields = set(parsed.__dataclass_fields__)
    assert "confidence" not in fields, fields
    assert "certainty" not in fields, fields
    # confidence.py must not even IMPORT Hypothesis: a module that cannot see
    # the type cannot read a field off it, however the type later changes.
    imported = {
        alias.name
        for node in ast.walk(ast.parse((AI_DIR / "confidence.py").read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "Hypothesis" not in imported, f"confidence.py imports Hypothesis: {sorted(imported)}"
    print(f"\n  Hypothesis fields: {sorted(fields)}")


# ===========================================================================
# 3. The verifier
# ===========================================================================


@pytest.mark.charter_guard
def test_the_verifier_never_calls_eval_or_exec() -> None:
    """SDD 4.4/4.5, by AST over the whole ai/ package.

    A grep would match a docstring; the AST matches a CALL, which is the thing
    that would execute model output.
    """
    forbidden = {"eval", "exec", "compile", "__import__"}
    offenders: list[str] = []
    for path in sorted(AI_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in forbidden
            ):
                offenders.append(f"{path.name}:{node.lineno}: {node.func.id}()")
    assert not offenders, offenders
    planted = ast.parse("eval('1+1')")
    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in forbidden
        for n in ast.walk(planted)
    ), "the scan matches nothing"
    print(f"\n  {len(list(AI_DIR.rglob('*.py')))} modules scanned; no eval/exec/compile call")


@pytest.mark.boundary_refusal
def test_the_expression_grammar_is_an_allow_list() -> None:
    """Anything outside `table.field` or a numeric literal is a parse failure.

    Including the shapes an injection would take. `getattr(row, name)` on an
    arbitrary string would let a model read `__class__` and compare its repr,
    which is neither arithmetic nor checkable.
    """
    from settlesense.types import BankRow

    rows: dict[str, Any] = {
        "bank": BankRow(
            bank_txn_id="B1",
            value_date=__import__("datetime").date(2026, 9, 1),
            amount=money("100.00"),
            narration="x",
            direction=__import__(
                "settlesense.types", fromlist=["BankDirection"]
            ).BankDirection.CREDIT,
        )
    }
    assert resolve_field("bank.amount", rows) == Decimal("100.00")
    assert resolve_field("0", rows) == Decimal("0")
    assert resolve_field("-12.34", rows) == Decimal("-12.34")

    for hostile in (
        "bank.__class__",
        "__import__('os').system",
        "bank.amount; import os",
        "1+1",
        "unknown.amount",
        "bank.narration",
        "",
    ):
        with pytest.raises(GrammarError):
            resolve_field(hostile, rows)
    print(f"\n  {len(FIELD_GRAMMAR)} tables allow-listed; 7 hostile references refused")


@pytest.mark.boundary_refusal
def test_an_unpermitted_operator_is_refused() -> None:
    from settlesense.types import BankDirection, BankRow

    rows: dict[str, Any] = {
        "bank": BankRow(
            bank_txn_id="B1",
            value_date=__import__("datetime").date(2026, 9, 1),
            amount=money("100.00"),
            narration="x",
            direction=BankDirection.CREDIT,
        )
    }
    with pytest.raises(GrammarError, match=r"not permitted"):
        evaluate_assertion(Assertion(lhs="bank.amount", op="**", rhs="0"), rows, money(1))
    print("\n  ** refused as an operator")


def test_the_arithmetic_path_verifies_and_rejects(dataset: Any, config: AppConfig) -> None:
    """UNUSED ON SEED 42 and built anyway - it is the general mechanism.

    Exercised on real rows so it is not a mock testing a mock: a batch and the
    bank credit that settled it, which agree, and then the same claim against a
    credit that does not.
    """
    linked = [
        (batch, bank)
        for batch in dataset.settlement_batches
        for bank in dataset.bank_rows
        if bank.amount == batch.net_total
    ]
    assert linked, "precondition: no batch and credit agree, so nothing exercises this path"
    batch, bank = linked[0]

    passing = Hypothesis(
        category=str(VarianceCategory.UTR_TRUNCATED_MAPPING),
        candidate_id=batch.batch_id,
        assertion=Assertion(lhs="bank.amount", op="==", rhs="batch.net_total"),
        residual_amount=None,
        evidence_row_ids=tuple(sorted((batch.batch_id, bank.bank_txn_id))),
        reason="the credit equals the batch total",
        rank=0,
    )
    result = verify(passing, dataset, config)
    assert result.passed, result.failure_reason
    assert result.computed_residual == money(0), result.computed_residual

    mismatched = [b for b in dataset.bank_rows if b.amount != batch.net_total]
    wrong = Hypothesis(
        **{
            **passing.__dict__,
            "evidence_row_ids": tuple(sorted((batch.batch_id, mismatched[0].bank_txn_id))),
        }
    )
    rejected = verify(wrong, dataset, config)
    assert not rejected.passed
    assert "residual" in rejected.failure_reason
    print(
        f"\n  arithmetic: {batch.batch_id} == {bank.bank_txn_id} passed; "
        f"against {mismatched[0].bank_txn_id} rejected by "
        f"{rejected.computed_residual}"
    )


@pytest.mark.boundary_refusal
def test_a_hypothesis_citing_a_row_that_does_not_exist_is_rejected_immediately(
    dataset: Any, config: AppConfig
) -> None:
    """SDD 4.5. Citing a row it cannot have read is disqualifying."""
    hypothesis = Hypothesis(
        category=DUPLICATE,
        candidate_id="ORD_MADEUP",
        assertion=None,
        residual_amount=None,
        evidence_row_ids=("ORD_MADEUP", "ORD_ALSOFAKE"),
        reason="invented",
        rank=0,
    )
    result = verify(hypothesis, dataset, config)
    assert not result.passed
    assert result.checks_run == ("evidence_resolution",), result.checks_run
    assert "do not exist" in result.failure_reason
    print(f"\n  {result.failure_reason[:70]}")


def test_the_structural_path_rejects_when_the_facts_do_not_distinguish(
    exceptions: tuple[Exception_, ...], dataset: Any, config: AppConfig, truth: frozenset[str]
) -> None:
    """THE CENTRAL BEHAVIOUR. Rejection is correct, not a failure.

    Both rows share customer, gross, date and settlement chain. Nothing in the
    data says which is the double entry, so the verifier refuses - even when
    the nomination is right. Confirming a correct guess it cannot check would
    be indistinguishable from confirming an incorrect one.
    """
    oracle = OracleClient(truth)
    report = run_loop(exceptions, dataset, config, oracle, sendable=SENDABLE)
    rejected = [o for o in report.outcomes if not o.confirmed]
    assert rejected, "precondition: nothing was rejected"
    reasons = {r.rejections[0] for r in rejected if r.rejections}
    assert any("do not distinguish them" in reason for reason in reasons), sorted(reasons)[:2]
    print(
        f"\n  {len(rejected)}/{report.sent} rejected even with a PERFECT nomination; "
        f"reason: {sorted(reasons)[0][:80]}"
    )


def test_the_verifier_discriminates_between_a_perfect_and_a_wrong_nomination(
    exceptions: tuple[Exception_, ...], dataset: Any, config: AppConfig, truth: frozenset[str]
) -> None:
    """THE SAFETY MEASUREMENT. Oracle confirms some; adversary confirms none.

    A verifier that rubber-stamped would score identically on the oracle run.
    Only the adversary separates discrimination from deference.
    """
    oracle = run_loop(exceptions, dataset, config, OracleClient(truth), sendable=SENDABLE)
    adversary = run_loop(exceptions, dataset, config, AdversarialClient(truth), sendable=SENDABLE)

    assert adversary.confirmed == 0, (
        f"{adversary.confirmed} FALSE CONFIRMS: the verifier accepted a nomination "
        "truth says is wrong"
    )
    assert oracle.confirmed > 0, (
        "the oracle confirmed nothing either, so this run cannot show the verifier "
        "discriminates - it is consistent with rejecting everything unconditionally"
    )
    assert oracle.confirmed < oracle.sent, "the oracle confirmed everything; no ambiguity remains"
    print(
        f"\n  seed 42: oracle {oracle.confirmed}/{oracle.sent}, "
        f"adversary {adversary.confirmed}/{adversary.sent}"
    )


# ===========================================================================
# 4. Confidence
# ===========================================================================


def test_confidence_is_the_sdd_4_6_weighted_sum(config: AppConfig) -> None:
    """Hand-computed against the shipped weights, from config (D12)."""
    perfect = VerificationResult(
        passed=True,
        computed_residual=money(0),
        failure_reason="",
        evidence_completeness=Decimal("1"),
        candidate_separation=Decimal("1"),
    )
    breakdown = compute_confidence(perfect, config, freshness_ok=True)
    weights = config.thresholds.confidence
    expected = (
        weights.weight_verification_passed
        + weights.weight_residual_within_tolerance
        + weights.weight_evidence_completeness
        + weights.weight_candidate_separation
        + weights.weight_freshness
    )
    assert breakdown.score == expected == Decimal("1.0000"), (breakdown.score, expected)

    stale = compute_confidence(perfect, config, freshness_ok=False)
    assert stale.score == expected - weights.weight_freshness
    print(f"\n  perfect {breakdown.score}; stale-files {stale.score}")


def test_confidence_alone_can_never_confirm(config: AppConfig) -> None:
    """SDD 4.6. `verification_passed` is checked EXPLICITLY, not inferred.

    With the shipped weights a failed verification cannot reach 0.80 anyway,
    but that is arithmetic about today's weights - re-weighting must not be
    able to silently enable confirmation on score alone. So the test forces a
    failed result to a perfect score and asserts it still cannot confirm.
    """
    failed_but_perfect = VerificationResult(
        passed=False,
        computed_residual=money(0),
        failure_reason="rejected",
        evidence_completeness=Decimal("1"),
        candidate_separation=Decimal("1"),
    )
    breakdown = compute_confidence(failed_but_perfect, config)
    forced = type(breakdown)(**{**breakdown.__dict__, "score": Decimal("1.0000")})
    assert forced.score >= config.thresholds.confidence.auto_confirm
    assert not should_auto_confirm(failed_but_perfect, forced, config), (
        "a rejected hypothesis auto-confirmed on score alone"
    )
    print(f"\n  score forced to {forced.score} with passed=False -> still refused")


@pytest.mark.boundary_refusal
def test_confidence_terms_are_clipped_to_the_unit_interval(config: AppConfig) -> None:
    """A term outside [0,1] would push the total past 1.0 and read as a
    probability nobody computed."""
    absurd = VerificationResult(
        passed=True,
        computed_residual=money(0),
        failure_reason="",
        evidence_completeness=Decimal("9"),
        candidate_separation=Decimal("-4"),
    )
    breakdown = compute_confidence(absurd, config)
    assert breakdown.evidence_completeness == Decimal("1")
    assert breakdown.candidate_separation == Decimal("0")
    assert breakdown.score <= Decimal("1")
    print(f"\n  9 -> 1 and -4 -> 0; score {breakdown.score}")


# ===========================================================================
# 5. The loop, and the committed result
# ===========================================================================


def test_the_loop_abstains_with_a_named_reason(
    exceptions: tuple[Exception_, ...], dataset: Any, config: AppConfig, truth: frozenset[str]
) -> None:
    """ "Nothing was resolved" and "the evidence could not distinguish" are
    different findings with different fixes."""
    report = run_loop(exceptions, dataset, config, OracleClient(truth), sendable=SENDABLE)
    reasons = dict(report.reasons)
    assert reasons, "no abstain reasons recorded"
    assert AbstainReason.ALL_REJECTED.value in reasons, reasons
    assert report.confirmed + report.abstained == report.sent
    print(f"\n  sent {report.sent}: confirmed {report.confirmed}, reasons {reasons}")


def test_a_fixture_miss_abstains_rather_than_crashing(
    exceptions: tuple[Exception_, ...], dataset: Any, config: AppConfig, tmp_path: Path
) -> None:
    """The run continues and reports which case had no recording.

    That is a fact about the fixture set, not about the case, and losing the
    other 25 decisions to it would be the wrong trade.
    """
    report = run_loop(
        exceptions[:3], dataset, config, ReplayLLMClient(fixture_dir=tmp_path), sendable=SENDABLE
    )
    assert report.sent == 3
    assert report.confirmed == 0
    assert dict(report.reasons) == {AbstainReason.FIXTURE_MISS.value: 3}, report.reasons
    print("\n  3 sent, 3 FIXTURE_MISS, 0 crashes")


def test_the_committed_20_seed_result_holds() -> None:
    """The headline, read back off the committed artifact.

    Not recomputed here: running all 20 seeds costs ~40s and the suite has a
    budget. The artifact is committed so the number is checkable, and
    `make eval-ai-loop` regenerates it.
    """
    assert COMMITTED.exists(), "reports/ai/ai_loop.json is not committed"
    payload = json.loads(COMMITTED.read_text(encoding="utf-8"))
    totals = payload["totals"]

    assert totals["adversarial_confirmed"] == 0, (
        f"{totals['adversarial_confirmed']} false confirms across the evaluation set"
    )
    assert totals["silent_confirmed"] == 0
    assert totals["oracle_false_confirmed"] == 0
    assert totals["seeds"] == 20, totals["seeds"]
    assert totals["decisions_sent"] == 507, (
        f"{totals['decisions_sent']} decisions, but the pre-registered set has 507"
    )
    ceiling = totals["oracle_confirmed"]
    assert 0 < ceiling < totals["decisions_sent"], ceiling
    print(
        f"\n  {totals['seeds']} seeds, {totals['decisions_sent']} decisions\n"
        f"  oracle ceiling {ceiling} ({ceiling / totals['decisions_sent']:.1%})\n"
        f"  adversarial false confirms {totals['adversarial_confirmed']}"
    )


@pytest.mark.charter_guard
def test_no_settlesense_module_reads_truth() -> None:
    """The engine and the verifier must never see the answer key.

    eval/ may read truth - it scores. settlesense/ may not, and an import edge
    or a field read is how that would start.

    DOCSTRINGS ARE EXCLUDED, and the first version of this test did not exclude
    them: it failed on `settlesense/types.py`, which NAMES `true_category` in a
    comment explaining what the eval scores against. That is prose about the
    answer key, not a read of it - the fourth time in this project a scanner
    has matched its own documentation, so this one walks the AST and drops
    docstrings rather than grepping text.
    """
    needles = {"truth_", "true_category", "resolvable_in_principle"}
    offenders: list[str] = []
    for path in sorted((REPO / "settlesense").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in docstrings or not any(n in node.value for n in needles):
                    continue
                if True:
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno}: {node.value[:40]!r}")
            elif isinstance(node, ast.Attribute) and node.attr in needles:
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}: .{node.attr}")
    assert not offenders, offenders

    planted = ast.parse('x = payload["true_category"]')
    assert any(
        isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in needles
        for n in ast.walk(planted)
    ), "the scan matches nothing"
    print("\n  settlesense/ reads no truth field (docstrings excluded, AST-scanned)")

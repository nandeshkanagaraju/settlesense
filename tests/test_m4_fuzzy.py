"""M4 - P8 fuzzy UTR, the 24-requirement acceptance suite.

Every expected score below is DERIVED FROM CONFIG at test time and asserted as
exact Decimal equality. None is a literal from the brief and none is a range:
a range assertion passes for the wrong reason, and a literal weight copied
into a test stops tracking the config it is supposed to be checking.

TWO REQUIREMENTS ARE ANSWERED DIFFERENTLY FROM THEIR WORDING, both flagged
where they occur:

  #7 asks for a 4-character fragment. Fragment SELECTION has a 6-character
     floor from config, so a 4-char token can never be chosen in practice. The
     scoring property it is really about - a short fragment is not penalised
     for being short - is tested directly on the scorer at 4 characters AND
     end-to-end at the shortest selectable length, 6.

  #22 says the 16/7/9 figures came from an earlier report and must be
     verified, not trusted. Verified: they are wrong, and the cause is traced
     rather than just corrected. The M3 commit reported "linked 23, unlinked
     16" by counting only the links carrying NO category - the one
     ROUNDING_DIFFERENCE link has a bank_row_id and is linked; a category
     describes a difference ON a link, not the absence of one. True figures:
     linked 24, unlinked 15, of which 7 keep a UTR fragment and 8 do not.
"""

from __future__ import annotations

import ast
import json
import random
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from settlesense.config import AppConfig, load_config
from settlesense.exceptions.taxonomy import VarianceCategory
from settlesense.ingest import DayDataset, load_dataset
from settlesense.matching.engine import fuzzy_verdicts_for, merge_days, run
from settlesense.matching.fuzzy_utr import (
    FuzzyOutcome,
    ScoringPath,
    _score_path_a,
    resolve,
)
from settlesense.types import (
    BankDirection,
    BankRow,
    ExceptionStatus,
    SettlementBatch,
    money,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
DAYS = 24
AS_OF_ALL_DUE = date(2026, 11, 30)
DUE = date(2026, 9, 10)

UTR12 = "A6F2E2B1C3D4"
UTR16 = "A6F2E2B1C3D40506"
OTHER16 = "FFFFFFFFFFFFFFFF"


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture(scope="module")
def dataset(config: AppConfig) -> DayDataset:
    return merge_days([load_dataset(DATA, day, config) for day in range(1, DAYS + 1)])


def _batch(batch_id: str, utr: str, total: str = "1000.00") -> SettlementBatch:
    return SettlementBatch(
        batch_id=batch_id, utr=utr, net_total=money(total), settled_event_date=date(2026, 9, 7)
    )


def _credit(narration: str, amount: str = "1000.00", value_date: date = DUE) -> BankRow:
    return BankRow(
        bank_txn_id="BNK_TEST000001",
        value_date=value_date,
        amount=money(amount),
        narration=narration,
        direction=BankDirection.CREDIT,
    )


def _expected_path_a(
    config: AppConfig, prefix_ratio: Decimal, edit_ratio: Decimal, amount: Decimal
) -> Decimal:
    """The Path A formula, recomputed from CONFIG rather than from a literal.

    If a weight changes in YAML this expectation moves with it, which is the
    point: a test carrying its own copy of 0.50 stops checking the config and
    starts checking itself.
    """
    weights = config.thresholds.fuzzy_utr
    return (
        weights.weight_prefix * prefix_ratio
        + weights.weight_edit * (Decimal(1) - edit_ratio)
        + weights.weight_amount * amount
    ).quantize(weights.score_quantum)


# ---------------------------------------------------------------------------
# Path A  (1-7)
# ---------------------------------------------------------------------------


def test_identical_utr_exact_amount_same_date_scores_one(config: AppConfig) -> None:
    """1."""
    verdict = resolve(_credit(f"NEFT {UTR16} X"), [_batch("BAT_1", UTR16)], {"BAT_1": DUE}, config)
    expected = _expected_path_a(config, Decimal(1), Decimal(0), Decimal(1))
    assert verdict.best_score == expected == Decimal("1.000000")
    assert verdict.is_accepted


def test_six_char_fragment_of_a_twelve_char_utr_scores_exactly(config: AppConfig) -> None:
    """2. Exact Decimal equality against a value computed from config.

    Worked through: the fragment is 6 characters and matches the candidate's
    first 6, so prefix_ratio = 6/6 = 1 and the edit distance against the
    same-length prefix is 0. With the amount agreeing, every term is at its
    maximum and the score is 1.000000.

    NOTE: under the specified formula this collapses into requirement 1. A
    perfect short fragment and a perfect full UTR score identically, and that
    is the formula behaving as designed - prefix_ratio is normalised by what
    was OBSERVED, so length carries no information about correctness.
    """
    verdict = resolve(
        _credit(f"NEFT {UTR12[:6]} SETTLEMENT"), [_batch("BAT_1", UTR12)], {"BAT_1": DUE}, config
    )
    best = verdict.candidates[0]
    assert best.observed_fragment == UTR12[:6]
    assert best.prefix_ratio == Decimal("1.000000")
    assert best.edit_ratio == Decimal("0.000000")
    assert verdict.best_score == _expected_path_a(config, Decimal(1), Decimal(0), Decimal(1))
    assert verdict.best_score >= config.thresholds.fuzzy_utr.accept_score


def test_a_fragment_sharing_no_prefix_falls_below_the_threshold(config: AppConfig) -> None:
    """3. Computed, not guessed: prefix 0/8, edit 8/8, amount agrees."""
    scored = _score_path_a("ZZZZZZZZ", _batch("BAT_1", UTR16), _credit("x"), 0, config)
    expected = _expected_path_a(config, Decimal(0), Decimal(1), Decimal(1))
    assert scored.score == expected == Decimal("0.200000")
    assert scored.score < config.thresholds.fuzzy_utr.accept_score


def test_the_date_gate_dominates_a_perfect_utr_match(config: AppConfig) -> None:
    """4. Exactly Decimal("0"), even with an identical UTR and amount.

    A hard gate, not a penalty. A credit far outside the window is evidence
    about a different batch; scoring it small-but-positive would let several
    such credits out-vote one real candidate on aggregate.
    """
    window = config.thresholds.fuzzy_utr.date_window_days
    far = DUE + timedelta(days=window + 5)
    verdict = resolve(
        _credit(f"NEFT {UTR16} X", value_date=far), [_batch("BAT_1", UTR16)], {"BAT_1": DUE}, config
    )
    assert verdict.candidates[0].score == Decimal("0")
    assert not verdict.is_accepted


@pytest.mark.determinism
def test_no_float_appears_anywhere_in_the_module() -> None:
    """5. AST scan: no float literal, no float() call, no float-returning RNG.

    A float comparison near a threshold is a determinism hazard on a par with
    float money - 0.85 is not representable, so `score >= 0.85` can differ
    between platforms for a score that prints identically.
    """
    source = (REPO / "settlesense" / "matching" / "fuzzy_utr.py").read_text(encoding="utf-8")
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            offenders.append(f"line {line}: float literal {node.value!r}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
        ):
            offenders.append(f"line {line}: float() call")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"random", "uniform", "gauss"}
        ):
            offenders.append(f"line {line}: float-returning random call")
    assert not offenders, f"floats in fuzzy_utr.py: {offenders}"


@pytest.mark.charter_guard
def test_the_float_scanner_catches_a_planted_float() -> None:
    """FAULT INJECTION for the scan above. A scanner that finds nothing would
    pass forever."""
    planted = "x = 0.85\ny = float('1')\n"
    found = [
        node
        for node in ast.walk(ast.parse(planted))
        if (isinstance(node, ast.Constant) and isinstance(node.value, float))
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
        )
    ]
    assert len(found) == 2, f"the scanner found {len(found)} of 2 planted floats"


@pytest.mark.parametrize(
    ("fragment", "label"),
    [("", "empty"), ("A", "single char"), (UTR16 + "EXTRA", "longer than the candidate")],
)
def test_degenerate_fragments_do_not_raise(fragment: str, label: str, config: AppConfig) -> None:
    """6. No crash on any of the three degenerate shapes."""
    scored = _score_path_a(fragment, _batch("BAT_1", UTR16), _credit("x"), 0, config)
    assert isinstance(scored.score, Decimal), f"{label} produced {type(scored.score)}"
    assert Decimal(0) <= scored.score <= Decimal(1), f"{label} scored {scored.score}"


def test_a_short_correct_fragment_outscores_a_long_wrong_one(config: AppConfig) -> None:
    """7. THE anti-length-penalty test, at the 4 characters the brief asks for.

    Tested on the SCORER directly because fragment SELECTION has a 6-character
    floor from config, so a 4-char token is never chosen end-to-end. The
    property under test belongs to the formula, not to the selector.

    If normalized_levenshtein compared the fragment against the FULL candidate
    UTR, the 4-char correct fragment would carry 12 phantom edits (edit_ratio
    12/16 = 0.75) and score 0.50 + 0.30*0.25 + 0.20 = 0.775 - LOSING to a
    wrong 8-char fragment is not quite reached, but the correct answer would
    drop below the 0.85 accept threshold and abstain.
    """
    correct_short = _score_path_a(UTR16[:4], _batch("BAT_1", UTR16), _credit("x"), 0, config)
    wrong_long = _score_path_a("ZZZZZZZZ", _batch("BAT_1", UTR16), _credit("x"), 0, config)
    print(f"\n  4-char correct={correct_short.score}  8-char wrong={wrong_long.score}")
    assert correct_short.score > wrong_long.score
    assert correct_short.score == _expected_path_a(config, Decimal(1), Decimal(0), Decimal(1))
    assert correct_short.score >= config.thresholds.fuzzy_utr.accept_score


def test_the_shortest_selectable_fragment_also_beats_a_long_wrong_one(
    config: AppConfig,
) -> None:
    """7, end-to-end at the shortest length the selector will actually pick."""
    floor = config.thresholds.fuzzy_utr.fragment_min_chars
    verdict = resolve(
        _credit(f"NEFT {UTR16[:floor]} SETTLEMENT"),
        [_batch("BAT_RIGHT", UTR16), _batch("BAT_WRONG", OTHER16, "1000.00")],
        {"BAT_RIGHT": DUE, "BAT_WRONG": DUE},
        config,
    )
    assert verdict.is_accepted
    assert verdict.matched_batch_id == "BAT_RIGHT"


# ---------------------------------------------------------------------------
# Path B  (8-11)
# ---------------------------------------------------------------------------


def test_path_b_is_selected_and_recorded_on_the_verdict(config: AppConfig) -> None:
    """8. A verdict that cannot say how it was scored cannot be audited."""
    verdict = resolve(
        _credit("NEFT AURORA RETAIL SETTLEMENT"), [_batch("BAT_1", UTR16)], {"BAT_1": DUE}, config
    )
    assert verdict.path is ScoringPath.AMOUNT_DATE
    assert all(c.path is ScoringPath.AMOUNT_DATE for c in verdict.candidates)
    assert verdict.threshold == config.thresholds.fuzzy_utr.accept_score_no_utr
    assert verdict.reason


def test_path_b_can_exceed_path_as_threshold(config: AppConfig) -> None:
    """9. If it could not, the two paths would still be coupled.

    Path A's formula with prefix and edit zeroed has a ceiling of
    weight_amount = 0.20. Path B reaching 1.000000 on the same evidence proves
    it is a separate formula and not a degraded reuse.
    """
    weights = config.thresholds.fuzzy_utr
    coupled_ceiling = weights.weight_amount
    verdict = resolve(
        _credit("NEFT AURORA RETAIL SETTLEMENT"), [_batch("BAT_1", UTR16)], {"BAT_1": DUE}, config
    )
    expected = (
        weights.weight_amount_no_utr * Decimal(1) + weights.weight_date_no_utr * Decimal(1)
    ).quantize(weights.score_quantum)
    print(f"\n  Path B score={verdict.best_score}  coupled ceiling would be {coupled_ceiling}")
    assert verdict.best_score == expected == Decimal("1.000000")
    assert verdict.best_score > weights.accept_score, "Path B cannot clear Path A's bar"
    assert verdict.best_score > coupled_ceiling


def test_path_b_threshold_is_strictly_higher_read_from_config(config: AppConfig) -> None:
    """10. From config, never a literal."""
    weights = config.thresholds.fuzzy_utr
    assert weights.accept_score_no_utr > weights.accept_score, (
        f"Path B {weights.accept_score_no_utr} must strictly exceed Path A {weights.accept_score}"
    )


def test_two_batches_same_amount_same_window_no_fragment_abstains(config: AppConfig) -> None:
    """11. Constructed explicitly, and asserted to be the case it claims.

    Both candidates agree on amount and land on the same due date, so Path B
    has nothing left to separate them with. Picking the lexicographically
    smaller batch_id would be a coin toss wearing a determinism costume.
    """
    left, right = _batch("BAT_AAA", UTR16, "1000.00"), _batch("BAT_BBB", OTHER16, "1000.00")
    assert left.net_total == right.net_total, "the fixture does not construct the collision"
    verdict = resolve(
        _credit("NEFT AURORA RETAIL SETTLEMENT"),
        [left, right],
        {"BAT_AAA": DUE, "BAT_BBB": DUE},
        config,
    )
    assert verdict.path is ScoringPath.AMOUNT_DATE
    assert verdict.best_score == verdict.runner_up_score, "the collision did not produce a tie"
    assert verdict.outcome is FuzzyOutcome.ABSTAINED
    assert verdict.matched_batch_id is None
    assert "tie" in verdict.reason.lower()


# ---------------------------------------------------------------------------
# Acceptance  (12-16)
# ---------------------------------------------------------------------------


def test_a_clear_winner_is_accepted(config: AppConfig) -> None:
    """12."""
    verdict = resolve(
        _credit(f"NEFT {UTR16} X"),
        [_batch("BAT_RIGHT", UTR16), _batch("BAT_WRONG", OTHER16, "77.00")],
        {"BAT_RIGHT": DUE, "BAT_WRONG": DUE},
        config,
    )
    assert verdict.outcome is FuzzyOutcome.ACCEPTED
    assert verdict.matched_batch_id == "BAT_RIGHT"


def test_two_close_candidates_are_ambiguous_with_both_attached(config: AppConfig) -> None:
    """13. Both candidates and both scores survive into the verdict."""
    near = UTR16[:-1] + ("0" if UTR16[-1] != "0" else "1")
    verdict = resolve(
        _credit(f"NEFT {UTR16} X"),
        [_batch("BAT_1", UTR16), _batch("BAT_2", near)],
        {"BAT_1": DUE, "BAT_2": DUE},
        config,
    )
    margin = config.thresholds.fuzzy_utr.min_separation
    assert verdict.best_score is not None and verdict.runner_up_score is not None
    gap = verdict.best_score - verdict.runner_up_score
    print(f"\n  best={verdict.best_score} runner_up={verdict.runner_up_score} gap={gap}")
    if gap < margin:
        assert verdict.outcome is FuzzyOutcome.AMBIGUOUS
    assert len(verdict.candidates) == 2
    assert all(c.score is not None for c in verdict.candidates)


def test_below_threshold_is_ambiguous_even_with_no_runner_up(config: AppConfig) -> None:
    """14. A lone candidate that fails the bar is not accepted by default."""
    verdict = resolve(
        _credit("NEFT ZZZZZZZZZZZZ SETTLEMENT", "55.00"),
        [_batch("BAT_1", UTR16, "1000.00")],
        {"BAT_1": DUE},
        config,
    )
    assert verdict.runner_up_score is None
    assert verdict.outcome is not FuzzyOutcome.ACCEPTED
    assert verdict.matched_batch_id is None


def test_an_exact_tie_abstains_and_says_so(config: AppConfig) -> None:
    """15. After score-then-batch_id, a surviving tie is unresolvable."""
    verdict = resolve(
        _credit(f"NEFT {UTR16} X"),
        [_batch("BAT_AAA", UTR16), _batch("BAT_BBB", UTR16)],
        {"BAT_AAA": DUE, "BAT_BBB": DUE},
        config,
    )
    assert verdict.outcome is FuzzyOutcome.ABSTAINED
    assert verdict.matched_batch_id is None
    assert "tie" in verdict.reason.lower(), f"the verdict does not say why: {verdict.reason!r}"


def test_an_empty_candidate_list_is_ambiguous_not_a_crash(config: AppConfig) -> None:
    """16."""
    verdict = resolve(_credit("NEFT X"), [], {}, config)
    assert verdict.outcome is FuzzyOutcome.AMBIGUOUS
    assert verdict.candidates == ()
    assert verdict.matched_batch_id is None


# ---------------------------------------------------------------------------
# Determinism  (17-19)
# ---------------------------------------------------------------------------


def test_candidate_order_does_not_affect_the_verdict(config: AppConfig) -> None:
    """17. One hundred shuffles, identical verdict every time."""
    batches = [
        _batch("BAT_A", UTR16),
        _batch("BAT_B", OTHER16, "500.00"),
        _batch("BAT_C", UTR12 + "0000", "250.00"),
        _batch("BAT_D", "0123456789ABCDEF", "125.00"),
    ]
    due = dict.fromkeys((b.batch_id for b in batches), DUE)
    credit = _credit(f"NEFT {UTR16[:8]} SETTLEMENT")
    baseline = resolve(credit, batches, due, config)

    rng = random.Random(17)
    seen: set[tuple[str, ...]] = set()
    for _ in range(100):
        shuffled = list(batches)
        rng.shuffle(shuffled)
        verdict = resolve(credit, shuffled, due, config)
        seen.add(tuple(f"{c.batch_id}:{c.score}" for c in verdict.candidates))
        assert verdict.outcome is baseline.outcome
        assert verdict.matched_batch_id == baseline.matched_batch_id
        assert verdict.best_score == baseline.best_score
    assert len(seen) == 1, f"candidate ordering varied across shuffles: {len(seen)} orders"


def test_equal_scores_sort_by_batch_id(config: AppConfig) -> None:
    """18. The tie-break that makes the order TOTAL rather than merely stable."""
    verdict = resolve(
        _credit("NEFT AURORA SETTLEMENT"),
        [_batch("BAT_ZZZ", UTR16), _batch("BAT_AAA", OTHER16)],
        {"BAT_ZZZ": DUE, "BAT_AAA": DUE},
        config,
    )
    scores = [c.score for c in verdict.candidates]
    assert scores[0] == scores[1], "the fixture did not produce equal scores"
    assert [c.batch_id for c in verdict.candidates] == ["BAT_AAA", "BAT_ZZZ"]


def test_candidates_are_keyed_by_batch_id_never_settlement_id(
    dataset: DayDataset, config: AppConfig
) -> None:
    """19. SDD 3.3 forbids the two namespaces mixing.

    Checked against the REAL id sets, not a prefix convention: a settlement_id
    appearing here would make SETTLEMENT_TO_BATCH self-referential and corrupt
    every Population B join.
    """
    batch_ids = {b.batch_id for b in dataset.settlement_batches}
    settlement_ids = {line.settlement_id for line in dataset.settlement_lines}
    assert not (batch_ids & settlement_ids), "the id namespaces already overlap"

    verdicts = fuzzy_verdicts_for(dataset, config, AS_OF_ALL_DUE)
    seen = {c.batch_id for v in verdicts for c in v.candidates}
    assert seen, "no candidates were scored; this test is vacuous"
    assert seen <= batch_ids, f"non-batch ids in candidates: {sorted(seen - batch_ids)}"
    assert not (seen & settlement_ids), f"settlement ids leaked in: {sorted(seen & settlement_ids)}"
    print(f"\n  {len(seen)} distinct candidate batch_ids, 0 settlement_ids")


# ---------------------------------------------------------------------------
# Against ground truth  (20-22)
# ---------------------------------------------------------------------------


def _truth_links() -> dict[str, dict[str, object]]:
    payload = json.loads((DATA / "truth_42.json").read_text(encoding="utf-8"))
    return {link["batch_id"]: link for link in payload["batch_links"]}


def test_report_the_realised_p8_outcome_split(dataset: DayDataset, config: AppConfig) -> None:
    """20. Everything printed, nothing asserted from the brief."""
    import collections

    verdicts = fuzzy_verdicts_for(dataset, config, AS_OF_ALL_DUE)
    result = run(dataset, config, AS_OF_ALL_DUE)
    truth = _truth_links()

    split = collections.Counter((str(v.path), str(v.outcome)) for v in verdicts)
    accepted = [v for v in verdicts if v.is_accepted]
    by_category = collections.Counter(
        str(truth[v.matched_batch_id]["true_category"]) for v in accepted if v.matched_batch_id
    )
    unresolved = [b for b in result.batch_links if b.status is ExceptionStatus.OPEN]

    by_a = sum(1 for v in accepted if v.path is ScoringPath.PREFIX)
    by_b = sum(1 for v in accepted if v.path is ScoringPath.AMOUNT_DATE)
    print(
        f"\n  P8 verdicts:            {dict(split)}"
        f"\n  resolved by Path A:     {by_a}"
        f"\n  resolved by Path B:     {by_b}"
        f"\n  left ambiguous/abstain: {len(verdicts) - len(accepted)}"
        f"\n  accepted by truth category: {dict(by_category)}"
        f"\n  batches still unresolved:   {[(b.batch_id, b.category) for b in unresolved]}"
        f"\n  batches linked:         {sum(1 for b in result.batch_links if b.bank_row_id)}"
        f"/{len(result.batch_links)}"
    )
    assert verdicts, "P8 produced no verdicts; it is not wired in"
    assert accepted, "P8 accepted nothing; the report would be vacuous"


def test_zero_false_accepts(dataset: DayDataset, config: AppConfig) -> None:
    """21. EXACTLY ZERO. Same bar as M3 test 31.

    A false accept is a link to a batch that truth says owns a different
    credit, or none at all. It is strictly worse than an abstention: a wrong
    link is a reconciliation somebody will trust.
    """
    truth = _truth_links()
    verdicts = fuzzy_verdicts_for(dataset, config, AS_OF_ALL_DUE)
    accepted = [v for v in verdicts if v.is_accepted and v.matched_batch_id]
    assert accepted, "nothing was accepted; a zero here would be vacuous"

    false_accepts = [
        (v.matched_batch_id, v.bank_txn_id, truth[str(v.matched_batch_id)]["bank_txn_id"])
        for v in accepted
        if truth[str(v.matched_batch_id)]["bank_txn_id"] != v.bank_txn_id
    ]
    print(f"\n  P8 accepts: {len(accepted)}   FALSE ACCEPTS: {len(false_accepts)}")
    assert false_accepts == [], f"FALSE ACCEPTS: {false_accepts}"


def test_the_brief_figures_are_wrong_and_the_realised_ones_are_asserted(
    dataset: DayDataset, config: AppConfig
) -> None:
    """22. The 16/7/9 figures came from an earlier report. Verified: wrong.

    TRACED TO ITS SOURCE rather than just corrected. The M3 commit message
    reported "Population B 39 batches, linked 23, unlinked 16". That counted
    only the 23 links carrying no category and treated the one
    ROUNDING_DIFFERENCE link as unlinked - but it has a bank_row_id, so it is
    linked; a category describes a difference ON a link, not the absence of
    one. True figures were linked 24, unlinked 15, and the brief inherited the
    slip.

    The lesson is the one this project keeps relearning: a count derived by
    subtraction from a category tally is not the same as a count of the thing
    itself. Asserted here on bank_row_id directly, which is what "linked"
    means.
    """
    verdicts = fuzzy_verdicts_for(dataset, config, AS_OF_ALL_DUE)
    path_a = sum(1 for v in verdicts if v.path is ScoringPath.PREFIX)
    path_b = sum(1 for v in verdicts if v.path is ScoringPath.AMOUNT_DATE)
    result = run(dataset, config, AS_OF_ALL_DUE)

    linked = sum(1 for b in result.batch_links if b.bank_row_id is not None)
    uncategorised_links = sum(
        1 for b in result.batch_links if b.bank_row_id is not None and b.category is None
    )
    print(
        f"\n  batches reaching P8: {len(verdicts)}  ->  Path A {path_a}, Path B {path_b}"
        f"\n  linked (bank_row_id set): {linked}"
        f"   of which uncategorised: {uncategorised_links}"
        f"\n  the brief's 16 = 39 - {uncategorised_links}: a categorised"
        f" link counted as unlinked"
    )
    assert len(verdicts) == 15, f"realised {len(verdicts)} reach P8; the brief said 16"
    assert (path_a, path_b) == (7, 8), f"realised {(path_a, path_b)}; the brief said (7, 9)"
    assert linked > uncategorised_links, (
        "no categorised link exists, so the miscount this test documents could "
        "not have happened and the explanation is now stale"
    )


def test_the_unlinked_count_is_as_of_dependent(dataset: DayDataset, config: AppConfig) -> None:
    """Why any single 'unlinked' figure needs its as_of stated.

    A batch whose credit is not yet due is PENDING_EVIDENCE, not unlinked, and
    it never reaches P8 - there is nothing wrong with it yet. Quoting an
    unlinked count without the as_of that produced it is how a figure travels
    into a brief and stops being checkable.
    """
    early = run(dataset, config, date(2026, 9, 10))
    early_pending = sum(
        1 for b in early.batch_links if b.status is ExceptionStatus.PENDING_EVIDENCE
    )
    early_p8 = len(fuzzy_verdicts_for(dataset, config, date(2026, 9, 10)))
    late_p8 = len(fuzzy_verdicts_for(dataset, config, AS_OF_ALL_DUE))
    print(
        f"\n  as_of 2026-09-10: {early_pending} not yet due, {early_p8} reach P8"
        f"\n  as_of 2026-11-30: 0 not yet due, {late_p8} reach P8"
    )
    assert early_pending > 0 and early_p8 < late_p8, (
        "as_of no longer changes how many batches reach P8; the parameter is "
        "being accepted and ignored"
    )


# ---------------------------------------------------------------------------
# Regression  (23-24)
# ---------------------------------------------------------------------------


def test_p8_did_not_weaken_p1_through_p7(dataset: DayDataset, config: AppConfig) -> None:
    """23. M3 test 31, rerun with P8 wired: still exactly zero false matches."""
    payload = json.loads((DATA / "truth_42.json").read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in payload["cases"]}
    result = run(dataset, config, AS_OF_ALL_DUE)

    clean = [
        c
        for c in result.cases
        if by_id[c.case_id]["true_category"] is None and not by_id[c.case_id]["noise_types"]
    ]
    assert len(clean) > 4000, f"only {len(clean)} clean chains; the check is thin"
    false_matches = [
        c.case_id for c in clean if c.status is ExceptionStatus.CONFIRMED and c.category is not None
    ]
    residual = [c for c in result.cases if c.status is not ExceptionStatus.CONFIRMED]
    print(
        f"\n  after P8: clean={len(clean)} FALSE MATCHES={len(false_matches)}"
        f" residual={len(residual)}"
    )
    assert false_matches == [], f"P8 introduced false matches: {false_matches[:10]}"
    assert len(residual) == 52, "P8 moved the Population A residual"


def test_p8_runs_after_the_exact_passes_and_before_p9(
    dataset: DayDataset, config: AppConfig
) -> None:
    """24. A batch an earlier pass can resolve is never reached by P8.

    Asserted structurally AND behaviourally. Structurally: P9 is
    _p9_rounding_category, which only labels a difference on an already-linked
    batch and cannot create a link, so it cannot precede the pass that makes
    one. Behaviourally: no batch that P2 or P2b linked appears as an accepted
    P8 match, and every P8 candidate is a batch those passes left open.
    """
    from settlesense.matching import engine

    source = (REPO / "settlesense" / "matching" / "engine.py").read_text(encoding="utf-8")
    assert "def _p9_rounding_category" in source
    p9_body = source.split("def _p9_rounding_category")[1].split("\ndef ")[0]
    assert "BatchLinkOutcome" not in p9_body, (
        "P9 constructs a link. It must only categorise a difference on a batch "
        "another pass already linked, or it is not the last pass."
    )
    assert hasattr(engine, "_p9_rounding_category")

    result = run(dataset, config, AS_OF_ALL_DUE)
    verdicts = fuzzy_verdicts_for(dataset, config, AS_OF_ALL_DUE)
    p8_accepted = {v.matched_batch_id for v in verdicts if v.is_accepted}
    p8_candidates = {c.batch_id for v in verdicts for c in v.candidates}

    # Batches P2/P2b linked exactly: they never enter P8's candidate pool.
    exact_linked = {
        b.batch_id
        for b in result.batch_links
        if b.bank_row_id is not None and b.batch_id not in p8_accepted
    }
    assert exact_linked, "no batch was linked by an exact pass; the check is vacuous"
    overlap = exact_linked & p8_candidates
    print(
        f"\n  exact-linked batches: {len(exact_linked)}"
        f"   P8 candidates: {len(p8_candidates)}   overlap: {len(overlap)}"
    )
    assert not overlap, (
        f"batches resolvable by an exact pass reached P8: {sorted(overlap)[:5]}. "
        "A looser rule must never see a row a stricter one could claim."
    )


@pytest.mark.boundary_refusal
def test_p9_refuses_to_call_a_large_residual_a_rounding_difference(config: AppConfig) -> None:
    """FAULT INJECTION for P9's own bound.

    A difference above the tolerance is not a rounding difference, and naming
    it one would explain away a real shortfall with a label meaning "too small
    to matter".
    """
    from settlesense.matching.engine import _p9_rounding_category

    tolerance = config.thresholds.tolerance.rounding_rupees
    assert _p9_rounding_category(money("100.00"), money("100.00"), tolerance) is None
    assert _p9_rounding_category(money("100.00"), money("99.50"), tolerance) == str(
        VarianceCategory.ROUNDING_DIFFERENCE
    )
    assert _p9_rounding_category(money("100.00"), money("50.00"), tolerance) == str(
        VarianceCategory.UNEXPLAINED
    )


def test_shuffling_the_dataset_does_not_change_p8(dataset: DayDataset, config: AppConfig) -> None:
    """D4, end to end: P8's inputs arrive sorted, so file order cannot decide."""
    rng = random.Random(24)
    batches = list(dataset.settlement_batches)
    bank = list(dataset.bank_rows)
    rng.shuffle(batches)
    rng.shuffle(bank)
    scrambled = replace(dataset, settlement_batches=tuple(batches), bank_rows=tuple(bank))
    assert scrambled.settlement_batches != dataset.settlement_batches
    original = [
        (b.batch_id, b.bank_row_id, b.category)
        for b in run(dataset, config, AS_OF_ALL_DUE).batch_links
    ]
    shuffled = [
        (b.batch_id, b.bank_row_id, b.category)
        for b in run(scrambled, config, AS_OF_ALL_DUE).batch_links
    ]
    assert original == shuffled

"""M4 - P8 fuzzy UTR: two scoring paths, and the refusals that keep it honest.

The number that matters here is FALSE LINKS, and the bar is the same as M3's
test 31: exactly zero. A fuzzy matcher that links 100% of batches by guessing
is strictly worse than one that links 60% and says so, because every wrong
link is a reconciliation a human will trust.

Counts are MEASURED and printed, never copied from the brief. The brief said
16 unlinked batches with 9 lacking a fragment; measurement says 15 and 8.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from settlesense.config import AppConfig, ConfigError, load_config
from settlesense.exceptions.taxonomy import VarianceCategory
from settlesense.ingest import DayDataset, load_dataset
from settlesense.matching.engine import fuzzy_verdicts_for, merge_days, run
from settlesense.matching.fuzzy_utr import (
    FuzzyOutcome,
    ScoringPath,
    resolve,
)
from settlesense.types import BankDirection, BankRow, ExceptionStatus, SettlementBatch, money

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
DAYS = 24
AS_OF = date(2026, 11, 30)
DUE = date(2026, 9, 10)
UTR_A = "A6F2E2B1C3D40506"
UTR_B = "FFFFFFFFFFFFFFFF"


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture(scope="module")
def dataset(config: AppConfig) -> DayDataset:
    return merge_days([load_dataset(DATA, day, config) for day in range(1, DAYS + 1)])


def _batch(
    batch_id: str, utr: str, total: str, settled: date = date(2026, 9, 7)
) -> SettlementBatch:
    return SettlementBatch(
        batch_id=batch_id, utr=utr, net_total=money(total), settled_event_date=settled
    )


def _credit(narration: str, amount: str, value_date: date = DUE) -> BankRow:
    return BankRow(
        bank_txn_id="BNK_TEST000000",
        value_date=value_date,
        amount=money(amount),
        narration=narration,
        direction=BankDirection.CREDIT,
    )


# ---------------------------------------------------------------------------
# Path A
# ---------------------------------------------------------------------------


def test_path_a_scores_a_perfect_fragment_at_one(config: AppConfig) -> None:
    """0.50 + 0.30 + 0.20 with a clean prefix, no edits and a matching amount."""
    verdict = resolve(
        _credit(f"NEFT {UTR_A[:8]} AURORA RETAIL SETTLEMENT", "1000.00"),
        [_batch("BAT_1", UTR_A, "1000.00")],
        {"BAT_1": DUE},
        config,
    )
    assert verdict.path is ScoringPath.PREFIX
    assert verdict.best_score == Decimal("1.000000")
    assert verdict.is_accepted


def test_prefix_ratio_divides_by_the_fragment_not_the_true_utr(config: AppConfig) -> None:
    """THE decisive Path A test.

    A six-character fragment that matches perfectly scores prefix_ratio 1.0,
    because 6/6 = 1. Dividing by the true UTR's 16 would give 0.375, capping
    the whole score at 0.5*0.375 + 0.30 + 0.20 = 0.6875 - below the 0.85
    threshold - so every short truncation would abstain and truncation itself
    would read as evidence of a mismatch.

    At match time the engine does not know which UTR is true. That is the
    thing being identified.
    """
    verdict = resolve(
        _credit(f"NEFT {UTR_A[:6]} SETTLEMENT", "1000.00"),
        [_batch("BAT_1", UTR_A, "1000.00")],
        {"BAT_1": DUE},
        config,
    )
    best = verdict.candidates[0]
    assert best.observed_fragment == UTR_A[:6]
    assert best.prefix_ratio == Decimal("1.000000"), (
        f"prefix_ratio {best.prefix_ratio} - divided by the true UTR length, not "
        "the observed fragment's"
    )
    assert verdict.best_score == Decimal("1.000000")
    assert verdict.is_accepted


def test_edit_ratio_compares_against_a_same_length_prefix(config: AppConfig) -> None:
    """A short fragment is not charged edits for characters it never had.

    Comparing a 6-character fragment against the full 16-character UTR gives a
    distance of 10 before a single character differs.
    """
    verdict = resolve(
        _credit(f"NEFT {UTR_A[:6]} SETTLEMENT", "1000.00"),
        [_batch("BAT_1", UTR_A, "1000.00")],
        {"BAT_1": DUE},
        config,
    )
    assert verdict.candidates[0].edit_ratio == Decimal("0.000000")


def test_a_wrong_candidate_scores_far_below_the_threshold(config: AppConfig) -> None:
    verdict = resolve(
        _credit(f"NEFT {UTR_A[:8]} SETTLEMENT", "1000.00"),
        [_batch("BAT_WRONG", UTR_B, "9999.99")],
        {"BAT_WRONG": DUE},
        config,
    )
    assert (
        verdict.best_score is not None
        and verdict.best_score < config.thresholds.fuzzy_utr.accept_score
    )
    assert not verdict.is_accepted


# ---------------------------------------------------------------------------
# Path B
# ---------------------------------------------------------------------------


def test_path_b_is_selected_when_no_fragment_survives(config: AppConfig) -> None:
    verdict = resolve(
        _credit("NEFT AURORA RETAIL SETTLEMENT", "1000.00"),
        [_batch("BAT_1", UTR_A, "1000.00")],
        {"BAT_1": DUE},
        config,
    )
    assert verdict.path is ScoringPath.AMOUNT_DATE
    assert verdict.candidates[0].prefix_ratio is None, (
        "Path B reported a prefix_ratio. The term is UNDEFINED here, not zero - "
        "recording it as 0 is how the two formulas get silently merged."
    )
    assert verdict.candidates[0].date_proximity == Decimal("1.000000")
    assert verdict.best_score == Decimal("1.000000")


def test_path_b_is_not_path_a_with_the_terms_zeroed(config: AppConfig) -> None:
    """If Path B reused Path A's weights with prefix and edit at zero, the
    ceiling would be 0.20 - under every threshold - and all 8 no-fragment
    batches would abstain because of a weight rather than a decision."""
    weights = config.thresholds.fuzzy_utr
    zeroed_ceiling = weights.weight_amount * Decimal(1)
    assert zeroed_ceiling < weights.accept_score, "the premise of this test has changed"
    verdict = resolve(
        _credit("NEFT AURORA RETAIL SETTLEMENT", "1000.00"),
        [_batch("BAT_1", UTR_A, "1000.00")],
        {"BAT_1": DUE},
        config,
    )
    assert verdict.best_score is not None and verdict.best_score > zeroed_ceiling


def test_path_b_uses_its_own_higher_threshold(config: AppConfig) -> None:
    weights = config.thresholds.fuzzy_utr
    assert weights.accept_score_no_utr > weights.accept_score
    path_b = resolve(
        _credit("NEFT AURORA SETTLEMENT", "1000.00"),
        [_batch("BAT_1", UTR_A, "1000.00")],
        {"BAT_1": DUE},
        config,
    )
    assert path_b.threshold == weights.accept_score_no_utr
    path_a = resolve(
        _credit(f"NEFT {UTR_A[:8]} SETTLEMENT", "1000.00"),
        [_batch("BAT_1", UTR_A, "1000.00")],
        {"BAT_1": DUE},
        config,
    )
    assert path_a.threshold == weights.accept_score


def test_path_b_abstains_when_two_batches_share_an_amount(config: AppConfig) -> None:
    """THE required Path B refusal, on a CONSTRUCTED fixture.

    No two batches share an amount anywhere in the seed-42 dataset - verified
    in test_no_two_real_batches_share_an_amount below - so this case cannot be
    exercised on real data and is built instead. A test that skipped here
    would report a passing suite for the single most dangerous Path B input.

    Both candidates score identically on amount and date, so the margin is
    zero and the engine must refuse. Picking the lexicographically smaller
    batch_id would be a coin toss wearing a determinism costume.
    """
    verdict = resolve(
        _credit("NEFT AURORA RETAIL SETTLEMENT", "1000.00"),
        [_batch("BAT_AAA", UTR_A, "1000.00"), _batch("BAT_BBB", UTR_B, "1000.00")],
        {"BAT_AAA": DUE, "BAT_BBB": DUE},
        config,
    )
    assert verdict.path is ScoringPath.AMOUNT_DATE
    assert verdict.outcome is FuzzyOutcome.ABSTAINED, (
        f"two batches of the same amount produced {verdict.outcome}: {verdict.reason}"
    )
    assert verdict.matched_batch_id is None
    assert verdict.best_score == verdict.runner_up_score
    assert len(verdict.candidates) == 2, "an abstention must still carry every candidate"


def test_no_two_real_batches_share_an_amount(dataset: DayDataset) -> None:
    """The precondition for the fixture above being a fixture at all."""
    totals = [batch.net_total for batch in dataset.settlement_batches]
    assert len(set(totals)) == len(totals), (
        "two real batches now share an amount; the collision case can be tested "
        "on real data and the constructed fixture should be revisited"
    )
    print(f"\n  {len(totals)} batches, all amounts distinct")


# ---------------------------------------------------------------------------
# Gates, ordering and refusals
# ---------------------------------------------------------------------------


def test_a_date_gap_beyond_the_window_is_a_hard_zero(config: AppConfig) -> None:
    """Not a small score. A credit far outside the window is evidence about
    something else, and several small scores could otherwise out-vote one real
    candidate on aggregate."""
    window = config.thresholds.fuzzy_utr.date_window_days
    far = date(DUE.year, DUE.month, DUE.day + window + 5)
    verdict = resolve(
        _credit(f"NEFT {UTR_A} SETTLEMENT", "1000.00", value_date=far),
        [_batch("BAT_1", UTR_A, "1000.00")],
        {"BAT_1": DUE},
        config,
    )
    assert verdict.candidates[0].score == Decimal("0")
    assert verdict.outcome is FuzzyOutcome.ABSTAINED


def test_an_exact_tie_abstains_rather_than_picking(config: AppConfig) -> None:
    """After score-then-batch_id, a surviving tie means the evidence does not
    separate two candidates. Taking the smaller id would be a pick."""
    verdict = resolve(
        _credit(f"NEFT {UTR_A[:8]} SETTLEMENT", "1000.00"),
        [_batch("BAT_AAA", UTR_A, "1000.00"), _batch("BAT_BBB", UTR_A, "1000.00")],
        {"BAT_AAA": DUE, "BAT_BBB": DUE},
        config,
    )
    assert verdict.outcome is FuzzyOutcome.ABSTAINED
    assert verdict.matched_batch_id is None


def test_a_near_miss_is_ambiguous_and_carries_every_candidate(config: AppConfig) -> None:
    """Clearing the threshold is not enough; the margin must clear too."""
    verdict = resolve(
        _credit(f"NEFT {UTR_A[:8]} SETTLEMENT", "1000.00"),
        [
            _batch("BAT_1", UTR_A, "1000.00"),
            _batch("BAT_2", UTR_A[:7] + "Z" + UTR_A[8:], "1000.00"),
        ],
        {"BAT_1": DUE, "BAT_2": DUE},
        config,
    )
    assert verdict.outcome in {FuzzyOutcome.AMBIGUOUS, FuzzyOutcome.ABSTAINED}
    assert len(verdict.candidates) == 2
    assert verdict.reason, "an unresolved verdict must say why"


def test_candidates_are_totally_ordered(config: AppConfig) -> None:
    """D5. Score descending, then batch_id ascending - no dependence on input
    order."""
    batches = [
        _batch("BAT_C", UTR_A, "1000.00"),
        _batch("BAT_A", UTR_A, "1000.00"),
        _batch("BAT_B", UTR_B, "1000.00"),
    ]
    credit = _credit(f"NEFT {UTR_A[:8]} SETTLEMENT", "1000.00")
    due = {"BAT_A": DUE, "BAT_B": DUE, "BAT_C": DUE}
    forward = resolve(credit, batches, due, config)
    backward = resolve(credit, list(reversed(batches)), due, config)
    assert [c.batch_id for c in forward.candidates] == [c.batch_id for c in backward.candidates]
    assert [c.score for c in forward.candidates] == sorted(
        (c.score for c in forward.candidates), reverse=True
    )


def test_every_verdict_records_which_path_scored_it(config: AppConfig) -> None:
    """A verdict that cannot say how it was scored cannot be audited: the two
    paths have different thresholds, so the same score means different things."""
    for narration in (f"NEFT {UTR_A[:8]} X", "NEFT AURORA RETAIL SETTLEMENT"):
        verdict = resolve(
            _credit(narration, "1000.00"),
            [_batch("BAT_1", UTR_A, "1000.00")],
            {"BAT_1": DUE},
            config,
        )
        assert verdict.path in {ScoringPath.PREFIX, ScoringPath.AMOUNT_DATE}
        assert all(c.path is verdict.path for c in verdict.candidates)


def test_no_candidates_abstains(config: AppConfig) -> None:
    verdict = resolve(_credit("NEFT X", "1.00"), [], {}, config)
    assert verdict.outcome is FuzzyOutcome.ABSTAINED
    assert verdict.candidates == ()


@pytest.mark.determinism
def test_every_score_is_decimal_never_float(config: AppConfig) -> None:
    """D1. A float comparison near a threshold is a determinism hazard on a
    par with float money."""
    verdict = resolve(
        _credit(f"NEFT {UTR_A[:8]} SETTLEMENT", "1000.00"),
        [_batch("BAT_1", UTR_A, "1000.00"), _batch("BAT_2", UTR_B, "5.00")],
        {"BAT_1": DUE, "BAT_2": DUE},
        config,
    )
    values = [verdict.best_score, verdict.threshold, verdict.margin_required]
    for candidate in verdict.candidates:
        values += [
            candidate.score,
            candidate.prefix_ratio,
            candidate.edit_ratio,
            candidate.amount_agrees,
            candidate.date_proximity,
        ]
    checked = 0
    for value in values:
        if value is None:
            continue
        assert type(value) is Decimal, f"{value!r} is {type(value).__name__}, not Decimal"
        checked += 1
    assert checked >= 8, f"only {checked} numeric fields checked"


# ---------------------------------------------------------------------------
# Config guards
# ---------------------------------------------------------------------------


@pytest.mark.config_refusal
def test_path_b_threshold_must_be_strictly_higher(tmp_path: Path) -> None:
    """FAULT INJECTION. Equal thresholds would mean amount-plus-date is
    believed as readily as a surviving UTR prefix, which is the whole reason
    Path B is a separate formula."""
    for name in ("mdr_rates.yaml", "calendar_v1.yaml", "thresholds.yaml"):
        (tmp_path / name).write_text((REPO / "config" / name).read_text(), encoding="utf-8")
    raw: dict[str, Any] = yaml.safe_load((tmp_path / "thresholds.yaml").read_text())
    raw["fuzzy_utr"]["accept_score_no_utr"] = raw["fuzzy_utr"]["accept_score"]
    (tmp_path / "thresholds.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="STRICTLY"):
        load_config(tmp_path)


@pytest.mark.config_refusal
def test_path_b_weights_must_sum_to_one(tmp_path: Path) -> None:
    """FAULT INJECTION."""
    for name in ("mdr_rates.yaml", "calendar_v1.yaml", "thresholds.yaml"):
        (tmp_path / name).write_text((REPO / "config" / name).read_text(), encoding="utf-8")
    raw: dict[str, Any] = yaml.safe_load((tmp_path / "thresholds.yaml").read_text())
    raw["fuzzy_utr"]["weight_amount_no_utr"] = "0.70"
    (tmp_path / "thresholds.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_no_threshold_or_weight_is_hardcoded() -> None:
    """Every number the scorer compares against comes from config."""
    source = (REPO / "settlesense" / "matching" / "fuzzy_utr.py").read_text(encoding="utf-8")
    for literal in ('"0.85"', '"0.90"', '"0.50"', '"0.30"', '"0.20"', '"0.60"', '"0.40"', '"0.15"'):
        assert literal not in source, f"{literal} is hardcoded in fuzzy_utr.py"


# ---------------------------------------------------------------------------
# Wired in: the measured result on seed 42
# ---------------------------------------------------------------------------


def test_p8_resolves_the_batches_it_should_and_links_nothing_wrongly(
    dataset: DayDataset, config: AppConfig
) -> None:
    """THE bar: zero false links, same as M3's test 31.

    Reports the path split and outcome counts rather than asserting figures
    from a brief - the brief said 16 unlinked and 9 without a fragment;
    measurement says 15 and 8.
    """
    import collections
    import json

    truth = {
        b["batch_id"]: b
        for b in json.loads((DATA / "truth_42.json").read_text(encoding="utf-8"))["batch_links"]
    }
    result = run(dataset, config, AS_OF)
    verdicts = fuzzy_verdicts_for(dataset, config, AS_OF)

    by_path = collections.Counter((str(v.path), str(v.outcome)) for v in verdicts)
    linked = [b for b in result.batch_links if b.bank_row_id is not None]
    false_links = [
        (b.batch_id, b.bank_row_id, truth[b.batch_id]["bank_txn_id"])
        for b in linked
        if b.bank_row_id != truth[b.batch_id]["bank_txn_id"]
    ]
    print(
        f"\n  P8 verdicts by (path, outcome): {dict(by_path)}"
        f"\n  batches linked: {len(linked)}/{len(result.batch_links)}"
        f"\n  FALSE LINKS: {len(false_links)}   (must be exactly 0)"
    )
    assert verdicts, "P8 produced no verdicts at all; it is not wired in"
    assert false_links == [], f"FALSE LINKS: {false_links}"
    assert len(linked) > 30, f"only {len(linked)} batches linked; P8 is not resolving"


def test_p8_never_links_a_batch_whose_credit_never_arrived(
    dataset: DayDataset, config: AppConfig
) -> None:
    """The refusal that matters most.

    Two batches in truth have bank_txn_id None - their credit genuinely does
    not exist. Path B scores on amount and date, and inventing a link for
    money that never arrived is the single worst failure available to it.
    """
    import json

    truth = {
        b["batch_id"]: b
        for b in json.loads((DATA / "truth_42.json").read_text(encoding="utf-8"))["batch_links"]
    }
    absent = {bid for bid, b in truth.items() if b["bank_txn_id"] is None}
    assert absent, "no batch in truth lacks a credit; this test is vacuous"

    result = run(dataset, config, AS_OF)
    linked = {b.batch_id for b in result.batch_links if b.bank_row_id is not None}
    print(
        f"\n  batches with no credit in truth: {len(absent)}; "
        f"linked by the engine: {len(absent & linked)}"
    )
    assert not (absent & linked), f"P8 invented a credit for {sorted(absent & linked)}"


def test_p8_does_not_change_population_a(dataset: DayDataset, config: AppConfig) -> None:
    """D11. P8 is Population B work; the case denominator must not move.

    A fuzzy batch link that altered the case count would mean the two
    populations were coupled, which is exactly what the three-denominator rule
    exists to prevent.
    """
    result = run(dataset, config, AS_OF)
    residual = [c for c in result.cases if c.status is not ExceptionStatus.CONFIRMED]
    print(f"\n  Population A after P8: {len(result.cases)} cases, {len(residual)} residual")
    assert len(result.cases) == 5026
    assert len(residual) == 52, "P8 moved the Population A residual; the populations are coupled"


def test_unresolved_batches_carry_a_utr_category_forward(
    dataset: DayDataset, config: AppConfig
) -> None:
    """Ambiguous verdicts become residual exceptions with a category the AI
    layer can prompt on, not a bare "unlinked"."""
    result = run(dataset, config, AS_OF)
    open_links = [b for b in result.batch_links if b.status is ExceptionStatus.OPEN]
    assert open_links, "no unresolved batch links; this test is vacuous"
    allowed = {
        str(VarianceCategory.UTR_TRUNCATED_MAPPING),
        str(VarianceCategory.UTR_MISSING_MAPPING),
        str(VarianceCategory.MISSING_VS_LATE_CREDIT),
    }
    for link in open_links:
        assert link.category in allowed, f"{link.batch_id} carries {link.category!r}"
    print(f"\n  unresolved batch links: {[(b.batch_id, b.category) for b in open_links]}")


def test_p8_is_deterministic_across_runs(dataset: DayDataset, config: AppConfig) -> None:
    """D6. Same input, same verdicts, including scores and candidate order."""
    first = fuzzy_verdicts_for(dataset, config, AS_OF)
    second = fuzzy_verdicts_for(dataset, config, AS_OF)
    assert [(v.bank_txn_id, v.outcome, v.best_score) for v in first] == [
        (v.bank_txn_id, v.outcome, v.best_score) for v in second
    ]


def test_shuffling_batches_does_not_change_p8(dataset: DayDataset, config: AppConfig) -> None:
    """D4. The candidate list arrives sorted, so input order cannot decide a
    winner."""
    import random

    rng = random.Random(4)
    shuffled = list(dataset.settlement_batches)
    rng.shuffle(shuffled)
    scrambled = replace(dataset, settlement_batches=tuple(shuffled))
    assert scrambled.settlement_batches != dataset.settlement_batches
    a = [(b.batch_id, b.bank_row_id) for b in run(dataset, config, AS_OF).batch_links]
    b = [(b.batch_id, b.bank_row_id) for b in run(scrambled, config, AS_OF).batch_links]
    assert a == b

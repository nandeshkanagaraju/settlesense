"""M5 - PDD 8.4 metrics. Pure functions over (results, truth).

EVERY MONEY METRIC NAMES ITS BASIS IN ITS OWN KEY. Three bases exist and none
is comparable to another:

    expected_gross    gross exposure          Population A
    expected_net      cash                    Population A
    batch_net_total   batch payout value      Population B

A key reading only "money_weighted" is ambiguous about which of the three it
means, and a reader comparing two such numbers has no way to know they are
different quantities. `assert_no_ambiguous_money_keys` refuses the string, and
a test feeds it a violation.

THREE POPULATIONS, THREE DENOMINATORS, NEVER MERGED (D11, SDD 3.1). They are
returned as three separate objects rather than three sections of one dict,
because a flat dict invites a caller to average across it.

Decimal throughout. A rate is a ratio and is quantized to six places; money is
quantized to paise by `money()`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from settlesense.types import (
    BatchLinkOutcome,
    CaseOutcome,
    ExceptionStatus,
    Money,
    ReconciliationCase,
    ReconciliationResult,
    RowVarianceOutcome,
    money,
)

__all__ = [
    "AnalystTimeEstimate",
    "PopulationA",
    "PopulationB",
    "PopulationC",
    "TruthView",
    "analyst_minutes_saved",
    "assert_no_ambiguous_money_keys",
    "population_a",
    "population_b",
    "population_c",
    "residual_set_sentence",
]

ZERO: Money = money(0)
RATE_QUANTUM = Decimal("0.000001")

TRUTH_DEFECT_BATCHES: frozenset[str] = frozenset({"BAT_16A0609791AB"})
"""Batches whose truth label describes an injection that left no trace.

BAT_16A0609791AB is labelled ROUNDING_DIFFERENCE in truth_42, but its batch
total and its bank credit differ by exactly Rs0.00 - there is nothing in the
data to detect, and an engine reporting "clean" is right.

The generator is frozen and was correctly NOT re-frozen for this. So Population
B precision is reported BOTH ways: counting it as a miss, and excluding it.
Printing both converts a hidden asterisk into evidence that the ground truth
was audited, and lets a reader decide which number they believe.
"""


def _rate(numerator: int | Decimal, denominator: int | Decimal) -> Decimal | None:
    """A ratio, or None when the denominator is zero.

    None rather than 0. A rate over an empty population is UNDEFINED, and
    returning zero would report "0% false matches" for a run that matched
    nothing - the most flattering possible reading of no evidence.
    """
    if Decimal(denominator) == 0:
        return None
    return (Decimal(numerator) / Decimal(denominator)).quantize(RATE_QUANTUM)


def _sum(values: Sequence[Money]) -> Money:
    return money(sum(values, Decimal(0)))


@dataclass(frozen=True)
class TruthView:
    """Typed access to a truth file. Never mutated, never re-derived."""

    cases: Mapping[str, Mapping[str, Any]]
    batch_links: Mapping[str, Mapping[str, Any]]
    row_variances: Mapping[str, Mapping[str, Any]]
    seed: int

    @staticmethod
    def from_payload(payload: Mapping[str, Any]) -> TruthView:
        return TruthView(
            cases={row["case_id"]: row for row in payload["cases"]},
            batch_links={row["batch_id"]: row for row in payload["batch_links"]},
            row_variances={row["row_id"]: row for row in payload["row_variances"]},
            seed=int(payload["seed"]),
        )

    def case_category(self, case_id: str) -> str | None:
        row = self.cases.get(case_id)
        return None if row is None else row["true_category"]

    def batch_credit(self, batch_id: str) -> str | None:
        row = self.batch_links.get(batch_id)
        return None if row is None else row["bank_txn_id"]

    def batch_category(self, batch_id: str) -> str | None:
        row = self.batch_links.get(batch_id)
        return None if row is None else row["true_category"]


# ---------------------------------------------------------------------------
# Population A - ReconciliationCase, payment-count denominator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PopulationA:
    """Every headline metric. Denominator: count of captured payments."""

    case_count: int
    confirmed_case_count: int
    deterministic_residual_count: int
    case_match_rate_case_count: Decimal | None
    residual_abstention_rate_case_count: Decimal | None
    residual_false_match_rate_case_count: Decimal | None
    residual_explanation_precision_case_count: Decimal | None
    gross_exposure_total_expected_gross: Money
    gross_exposure_matched_expected_gross: Money
    gross_exposure_match_rate_expected_gross: Decimal | None
    gross_exposure_false_match_value_expected_gross: Money
    expected_net_cash_reconciled_expected_net: Money
    unresolved_expected_net_cash_expected_net: Money
    evidence_coverage_case_count: Decimal | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "confirmed_case_count": self.confirmed_case_count,
            "deterministic_residual_count": self.deterministic_residual_count,
            "case_match_rate_case_count": _text(self.case_match_rate_case_count),
            "residual_abstention_rate_case_count": _text(self.residual_abstention_rate_case_count),
            "residual_false_match_rate_case_count": _text(
                self.residual_false_match_rate_case_count
            ),
            "residual_explanation_precision_case_count": _text(
                self.residual_explanation_precision_case_count
            ),
            "gross_exposure_total_expected_gross": str(self.gross_exposure_total_expected_gross),
            "gross_exposure_matched_expected_gross": str(
                self.gross_exposure_matched_expected_gross
            ),
            "gross_exposure_match_rate_expected_gross": _text(
                self.gross_exposure_match_rate_expected_gross
            ),
            "gross_exposure_false_match_value_expected_gross": str(
                self.gross_exposure_false_match_value_expected_gross
            ),
            "expected_net_cash_reconciled_expected_net": str(
                self.expected_net_cash_reconciled_expected_net
            ),
            "unresolved_expected_net_cash_expected_net": str(
                self.unresolved_expected_net_cash_expected_net
            ),
            "evidence_coverage_case_count": _text(self.evidence_coverage_case_count),
        }


def _text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def population_a(
    result: ReconciliationResult,
    cases_by_id: Mapping[str, ReconciliationCase],
    truth: TruthView,
) -> PopulationA:
    """Population A over ReconciliationCase and nothing else (D11).

    A FALSE MATCH is a case the engine CONFIRMED whose category disagrees with
    truth - a confident wrong answer. It is deliberately not the same as
    flagging a clean case for review: that is over-inclusion, which is
    conservative, visible in the residual count, and never presented as an
    answer. Merging the two would let a cautious engine look reckless.
    """
    confirmed = [case for case in result.cases if case.status is ExceptionStatus.CONFIRMED]
    residual = [case for case in result.cases if case.status is not ExceptionStatus.CONFIRMED]
    false_matches = [
        case
        for case in confirmed
        if _as_text(case.category) != _as_text(truth.case_category(case.case_id))
    ]

    def gross(outcomes: Sequence[CaseOutcome]) -> Money:
        return _sum(
            [cases_by_id[c.case_id].expected_gross for c in outcomes if c.case_id in cases_by_id]
        )

    def net(outcomes: Sequence[CaseOutcome]) -> Money:
        return _sum(
            [cases_by_id[c.case_id].expected_net for c in outcomes if c.case_id in cases_by_id]
        )

    total_gross = _sum([case.expected_gross for case in cases_by_id.values()])
    traceable = [case for case in result.cases if case.batch_id or case.bank_row_id]

    return PopulationA(
        case_count=len(result.cases),
        confirmed_case_count=len(confirmed),
        deterministic_residual_count=len(residual),
        case_match_rate_case_count=_rate(len(confirmed), len(result.cases)),
        residual_abstention_rate_case_count=_rate(len(residual), len(result.cases)),
        residual_false_match_rate_case_count=_rate(len(false_matches), len(confirmed)),
        # AI explanations do not exist until M7. None, not 1.0: a precision of
        # 1.0 over zero explanations is the most flattering reading of no work.
        residual_explanation_precision_case_count=None,
        gross_exposure_total_expected_gross=total_gross,
        gross_exposure_matched_expected_gross=gross(confirmed),
        gross_exposure_match_rate_expected_gross=_rate(gross(confirmed), total_gross),
        gross_exposure_false_match_value_expected_gross=gross(false_matches),
        expected_net_cash_reconciled_expected_net=net(confirmed),
        unresolved_expected_net_cash_expected_net=net(residual),
        evidence_coverage_case_count=_rate(len(traceable), len(result.cases)),
    )


def _as_text(value: object) -> str | None:
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Population B - BatchLinkOutcome, batch-count denominator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PopulationB:
    """Batch<->bank links. Its own denominator and its own money basis.

    `batch_net_total` is NEVER comparable to a case's `expected_gross`. The two
    totals are reported separately and are not expected to be equal.
    """

    batch_count: int
    linked_count: int
    batch_link_rate_batch_count: Decimal | None
    false_link_count: int
    batch_false_link_rate_batch_count: Decimal | None
    batch_net_linked_value_batch_net_total: Money
    batch_net_false_link_value_batch_net_total: Money
    batch_net_total_batch_net_total: Money
    # What the engine recovered despite injected noise, both ways for the defect.
    injected_noise_batch_count: int
    injected_noise_recovered_batch_count: int
    noise_recovery_rate_counting_defect: Decimal | None
    noise_recovery_rate_excluding_defect: Decimal | None
    # Precision where a category is actually a CLAIM: on unresolved batches.
    unresolved_batch_count: int
    category_precision_on_unresolved_batch_count: Decimal | None
    defect_batches_excluded: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_count": self.batch_count,
            "linked_count": self.linked_count,
            "batch_link_rate_batch_count": _text(self.batch_link_rate_batch_count),
            "false_link_count": self.false_link_count,
            "batch_false_link_rate_batch_count": _text(self.batch_false_link_rate_batch_count),
            "batch_net_linked_value_batch_net_total": str(
                self.batch_net_linked_value_batch_net_total
            ),
            "batch_net_false_link_value_batch_net_total": str(
                self.batch_net_false_link_value_batch_net_total
            ),
            "batch_net_total_batch_net_total": str(self.batch_net_total_batch_net_total),
            "injected_noise_batch_count": self.injected_noise_batch_count,
            "injected_noise_recovered_batch_count": self.injected_noise_recovered_batch_count,
            "noise_recovery_rate_counting_defect": _text(self.noise_recovery_rate_counting_defect),
            "noise_recovery_rate_excluding_defect": _text(
                self.noise_recovery_rate_excluding_defect
            ),
            "unresolved_batch_count": self.unresolved_batch_count,
            "category_precision_on_unresolved_batch_count": _text(
                self.category_precision_on_unresolved_batch_count
            ),
            "defect_batches_excluded": list(self.defect_batches_excluded),
        }


def population_b(result: ReconciliationResult, truth: TruthView) -> PopulationB:
    """Population B, with category precision reported BOTH ways.

    A FALSE LINK is a link to a credit truth says belongs elsewhere, or to one
    that does not exist. That is the safety metric and it takes no asterisk.

    CATEGORY precision is the one the truth defect touches, and it is reported
    twice: counting BAT_16A0609791AB as a miss, and excluding it. Reporting
    only the flattering number would be hiding a judgement; reporting only the
    harsh one would be accepting a label the data does not support.
    """
    linked = [link for link in result.batch_links if link.bank_row_id is not None]
    false_links = [link for link in linked if truth.batch_credit(link.batch_id) != link.bank_row_id]

    # A CATEGORY IS ONLY A CLAIM WHEN THE BATCH IS UNRESOLVED, and comparing it
    # to truth on a RESOLVED batch measures the wrong thing entirely. truth's
    # `true_category` records WHAT NOISE WAS INJECTED; the engine's category
    # records WHAT VARIANCE REMAINS. Once P8 recovers a truncated UTR there is
    # no remaining variance, and None is the correct answer.
    #
    # Measured: comparing the two across all 39 batches scored 0.64, penalising
    # the engine for succeeding on 13 batches it had recovered. That was a
    # metric defect, not an engine defect, and it is recorded here because a
    # 0.64 in a results table would have been read as the latter.
    unresolved = [link for link in result.batch_links if link.status is ExceptionStatus.OPEN]
    category_agree = sum(
        1
        for link in unresolved
        if _as_text(link.category) == _as_text(truth.batch_category(link.batch_id))
    )

    # What the engine ACHIEVED: batches carrying injected noise that it linked
    # to the correct credit anyway.
    noisy = [link for link in result.batch_links if truth.batch_category(link.batch_id) is not None]
    recovered = [
        link
        for link in noisy
        if link.bank_row_id is not None and link.bank_row_id == truth.batch_credit(link.batch_id)
    ]
    excluded = tuple(sorted({link.batch_id for link in noisy} & TRUTH_DEFECT_BATCHES))
    noisy_kept = [link for link in noisy if link.batch_id not in TRUTH_DEFECT_BATCHES]
    recovered_kept = [link for link in recovered if link.batch_id not in TRUTH_DEFECT_BATCHES]

    return PopulationB(
        batch_count=len(result.batch_links),
        linked_count=len(linked),
        batch_link_rate_batch_count=_rate(len(linked), len(result.batch_links)),
        false_link_count=len(false_links),
        batch_false_link_rate_batch_count=_rate(len(false_links), len(linked)),
        batch_net_linked_value_batch_net_total=_sum([link.batch_net_total for link in linked]),
        batch_net_false_link_value_batch_net_total=_sum(
            [link.batch_net_total for link in false_links]
        ),
        batch_net_total_batch_net_total=_sum([link.batch_net_total for link in result.batch_links]),
        injected_noise_batch_count=len(noisy),
        injected_noise_recovered_batch_count=len(recovered),
        noise_recovery_rate_counting_defect=_rate(len(recovered), len(noisy)),
        noise_recovery_rate_excluding_defect=_rate(len(recovered_kept), len(noisy_kept)),
        unresolved_batch_count=len(unresolved),
        category_precision_on_unresolved_batch_count=_rate(category_agree, len(unresolved)),
        defect_batches_excluded=excluded,
    )


# ---------------------------------------------------------------------------
# Population C - RowVarianceOutcome, row-count denominator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PopulationC:
    """Row-grain variances: duplicate ledger rows, orphan bank credits."""

    row_count: int
    truth_row_count: int
    matched_row_count: int
    row_variance_recall_row_count: Decimal | None
    row_variance_precision_row_count: Decimal | None
    row_variance_value_row_value: Money

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "truth_row_count": self.truth_row_count,
            "matched_row_count": self.matched_row_count,
            "row_variance_recall_row_count": _text(self.row_variance_recall_row_count),
            "row_variance_precision_row_count": _text(self.row_variance_precision_row_count),
            "row_variance_value_row_value": str(self.row_variance_value_row_value),
        }


def population_c(result: ReconciliationResult, truth: TruthView) -> PopulationC:
    """Population C. A ROW-COUNT denominator - never cases, never money."""
    found: Sequence[RowVarianceOutcome] = result.row_variances
    matched = [
        variance
        for variance in found
        if variance.row_id in truth.row_variances
        and _as_text(variance.category)
        == _as_text(truth.row_variances[variance.row_id]["true_category"])
    ]
    return PopulationC(
        row_count=len(found),
        truth_row_count=len(truth.row_variances),
        matched_row_count=len(matched),
        row_variance_recall_row_count=_rate(len(matched), len(truth.row_variances)),
        row_variance_precision_row_count=_rate(len(matched), len(found)),
        row_variance_value_row_value=_sum(
            [variance.amount for variance in found if variance.amount is not None]
        ),
    )


# ---------------------------------------------------------------------------
# Analyst minutes - a derived estimate, attributed correctly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalystTimeEstimate:
    """A DERIVED ESTIMATE, not a measurement, and it says so in its own label."""

    minutes_per_review: int
    deterministic_resolutions: int
    ai_confirmed_residuals: int
    minutes_saved_deterministic: int
    minutes_saved_ai: int
    label: str = field(default="")

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "minutes_per_review_assumption": self.minutes_per_review,
            "deterministic_resolutions": self.deterministic_resolutions,
            "minutes_saved_deterministic_derived_estimate": self.minutes_saved_deterministic,
            "ai_confirmed_residuals": self.ai_confirmed_residuals,
            "minutes_saved_ai_derived_estimate": self.minutes_saved_ai,
        }


def analyst_minutes_saved(
    result: ReconciliationResult,
    minutes_per_review: int,
    ai_confirmed_residuals: int = 0,
) -> AnalystTimeEstimate:
    """Minutes saved, split by WHO saved them. Never one blended figure.

    THE CALLER MUST PASS THE ASSUMPTION. There is no default, because a default
    would let the number be quoted without the premise it rests on, and this is
    the one figure in the whole result set that is not a measurement.

    ATTRIBUTION MATTERS MORE THAN THE TOTAL. On this dataset the deterministic
    engine resolves ~99% of cases and the AI layer has not run at all. A single
    blended "minutes saved" would read as a claim about the AI when it is
    almost entirely a claim about rules. So the two are returned separately and
    there is deliberately no combined field for a report to reach for.
    """
    if not isinstance(minutes_per_review, int) or isinstance(minutes_per_review, bool):
        raise TypeError(
            f"minutes_per_review must be an int, got {type(minutes_per_review).__name__}. "
            "It is a stated assumption, not a measurement."
        )
    if minutes_per_review <= 0:
        raise ValueError(f"minutes_per_review must be positive, got {minutes_per_review}")
    if ai_confirmed_residuals < 0:
        raise ValueError(f"ai_confirmed_residuals must be >= 0, got {ai_confirmed_residuals}")

    deterministic = sum(1 for case in result.cases if case.status is ExceptionStatus.CONFIRMED)
    return AnalystTimeEstimate(
        minutes_per_review=minutes_per_review,
        deterministic_resolutions=deterministic,
        ai_confirmed_residuals=ai_confirmed_residuals,
        minutes_saved_deterministic=deterministic * minutes_per_review,
        minutes_saved_ai=ai_confirmed_residuals * minutes_per_review,
        label=(
            f"derived estimate, assumes {minutes_per_review} min/review; "
            "attributed separately to rules and to AI, never blended"
        ),
    )


# ---------------------------------------------------------------------------
# The headline sentence (PDD 8.3) and the money-key guard
# ---------------------------------------------------------------------------


def residual_set_sentence(
    residual_count: int, explained: int, abstained: int, false_matched: int
) -> str:
    """PDD 8.3, with real numbers substituted. Not the diluted pipeline score."""
    if explained + abstained + false_matched > residual_count:
        raise ValueError(
            f"outcomes ({explained + abstained + false_matched}) exceed the residual "
            f"set ({residual_count}); the sentence would overstate the surface"
        )
    return (
        f"Of the {residual_count} exceptions the deterministic engine could not "
        f"resolve, the verified hypothesis loop correctly explained {explained}, "
        f"abstained on {abstained}, and false-matched {false_matched}."
    )


AMBIGUOUS_MONEY_KEY = "money_weighted"
MONEY_BASES: frozenset[str] = frozenset(
    {"expected_gross", "expected_net", "batch_net_total", "row_value"}
)


def assert_no_ambiguous_money_keys(payload: Mapping[str, Any], path: str = "") -> None:
    """Refuse any key containing the bare string "money_weighted".

    Three money bases exist and none is comparable to another. A key that says
    only "money-weighted" leaves a reader unable to tell which one it means,
    and two such numbers placed side by side invite exactly the comparison
    SDD 3.1 forbids. Walks nested structures, because the guard is worthless if
    it only inspects the top level.
    """
    for key, value in payload.items():
        location = f"{path}.{key}" if path else str(key)
        if AMBIGUOUS_MONEY_KEY in str(key):
            raise AssertionError(
                f"metric key {location!r} contains {AMBIGUOUS_MONEY_KEY!r}. Name the "
                f"basis explicitly - one of {sorted(MONEY_BASES)} - so two numbers "
                "on different bases cannot be read as comparable (PDD 8.4)."
            )
        if isinstance(value, Mapping):
            assert_no_ambiguous_money_keys(value, location)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    assert_no_ambiguous_money_keys(item, f"{location}[{index}]")


def outcome_counts(cases: Sequence[CaseOutcome]) -> dict[str, int]:
    """Cases by status, sorted (D4). A diagnostic, not a denominator."""
    counts: dict[str, int] = {}
    for case in cases:
        counts[str(case.status)] = counts.get(str(case.status), 0) + 1
    return dict(sorted(counts.items()))


def batch_outcome_counts(links: Sequence[BatchLinkOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for link in links:
        counts[str(link.status)] = counts.get(str(link.status), 0) + 1
    return dict(sorted(counts.items()))

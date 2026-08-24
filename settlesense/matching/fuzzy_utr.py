"""M4 - P8 fuzzy UTR. Two scoring paths, because half the damage removes the UTR.

SDD 4.3 puts this INSIDE the deterministic layer deliberately. If fuzzy
matching sat outside the baseline, the residual set would be inflated by
everything a prefix could have resolved and the model would look better than
it is.

TWO PATHS, NOT ONE WITH TERMS ZEROED. Measured on seed 42: of 15 batches that
survive P1-P7 unlinked, 7 keep a UTR fragment and 8 keep none. Reusing Path
A's formula with prefix and edit set to zero caps every no-UTR batch at 0.20,
under any threshold - so more than half the population would abstain because
of a weight rather than because of a decision. Path B is its own formula with
its own weights and its own, strictly higher, threshold.

  Path A  0.50*prefix_ratio + 0.30*(1 - edit_ratio) + 0.20*amount_agrees
  Path B  0.60*amount_agrees + 0.40*date_proximity

WHY PATH B'S BAR IS HIGHER. Amount and date alone are weaker evidence than a
surviving prefix: two payouts of the same size in the same window are
indistinguishable on it. config's loader RAISES if accept_score_no_utr is not
strictly greater than accept_score, so the ordering cannot be flattened by
editing YAML.

Decimal throughout (D1). Every ratio is an exact Decimal division quantized to
score_quantum, so scoring is reproducible across platforms - a float
comparison near a threshold is a determinism hazard on a par with float money.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from Levenshtein import distance as _levenshtein

from settlesense.config import AppConfig
from settlesense.exceptions.taxonomy import VarianceCategory
from settlesense.normalize import extract_utr_candidates, normalize_utr
from settlesense.types import BankRow, Money, SettlementBatch, money

__all__ = [
    "CandidateScore",
    "FuzzyOutcome",
    "FuzzyVerdict",
    "ScoringPath",
    "resolve",
]

ZERO = Decimal(0)
ONE = Decimal(1)


class ScoringPath(StrEnum):
    """Which formula produced a score. Recorded on every verdict.

    A verdict that cannot say how it was scored cannot be audited: the two
    paths have different thresholds, so "score 0.88, accepted" is true under
    one and false under the other, and a reviewer has no way to tell which.
    """

    PREFIX = "prefix"  # Path A - a UTR fragment survived
    AMOUNT_DATE = "amount_date"  # Path B - no fragment; amount and date only


class FuzzyOutcome(StrEnum):
    ACCEPTED = "accepted"
    AMBIGUOUS = "ambiguous"  # candidates exist, none separable -> residual
    ABSTAINED = "abstained"  # exact tie, or nothing in the window at all


@dataclass(frozen=True)
class CandidateScore:
    """One batch scored against one bank credit, with its components shown.

    The component terms are kept, not just the total. A reviewer asked to
    confirm a fuzzy link needs to see that it scored 0.94 on a nine-character
    prefix rather than 0.94 on an amount that two batches share.
    """

    batch_id: str
    score: Decimal
    path: ScoringPath
    prefix_ratio: Decimal | None
    edit_ratio: Decimal | None
    amount_agrees: Decimal
    date_proximity: Decimal | None
    observed_fragment: str | None
    date_gap_days: int

    def sort_key(self) -> tuple[Decimal, str]:
        """TOTAL order (D5): score descending, then batch_id ascending.

        The batch_id tie-break is what makes the order total rather than
        merely deterministic-in-practice. Two candidates scoring identically
        would otherwise be ordered by whatever the input sequence happened to
        be, and the "winner" would depend on file order.
        """
        return (-self.score, self.batch_id)


@dataclass(frozen=True)
class FuzzyVerdict:
    """The outcome of scoring one bank credit against candidate batches."""

    bank_txn_id: str
    outcome: FuzzyOutcome
    path: ScoringPath
    matched_batch_id: str | None
    best_score: Decimal | None
    runner_up_score: Decimal | None
    threshold: Decimal
    margin_required: Decimal
    candidates: tuple[CandidateScore, ...]  # ALL of them, sorted, always
    reason: str

    @property
    def is_accepted(self) -> bool:
        return self.outcome is FuzzyOutcome.ACCEPTED

    @property
    def category(self) -> VarianceCategory | None:
        """The taxonomy category an unresolved verdict carries forward.

        Path A failing means a UTR was present and could not be mapped;
        Path B failing means there was no UTR to map. Different problems,
        different categories, and the AI layer prompts differently for each.
        """
        if self.is_accepted:
            return None
        if self.path is ScoringPath.PREFIX:
            return VarianceCategory.UTR_TRUNCATED_MAPPING
        return VarianceCategory.UTR_MISSING_MAPPING


def _quantize(value: Decimal, config: AppConfig) -> Decimal:
    return value.quantize(config.thresholds.fuzzy_utr.score_quantum)


def _common_prefix_length(left: str, right: str) -> int:
    length = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        length += 1
    return length


def _amount_agrees(credit: Money, batch_total: Money, config: AppConfig) -> Decimal:
    """1 if the amounts agree, 0 otherwise - judged at P9's tolerance.

    NOT strict equality. A sub-rupee residual is already a named, categorised,
    non-substantive difference in this system (ROUNDING_DIFFERENCE, P9), and
    three of the fifteen unlinked batches carry one. Scoring those as
    "amount does not match" would make the fuzzy layer contradict P9's own
    definition of when two amounts are the same number.
    """
    tolerance = config.thresholds.tolerance.rounding_rupees
    return ONE if abs(credit - batch_total) <= tolerance else ZERO


def _date_proximity(gap_days: int, window_days: int) -> Decimal:
    """1 on the due date, decaying linearly to 0 at the window edge.

    Linear rather than a step, so a credit landing on the due date is
    distinguishable from one landing at the edge of what is acceptable. A step
    would score them identically and hand the tie-break to the amount alone.
    """
    if window_days <= 0:  # pragma: no cover - rejected by the config loader
        return ZERO
    distance = min(abs(gap_days), window_days)
    return (Decimal(window_days - distance) / Decimal(window_days)).quantize(Decimal("0.000001"))


def _score_path_a(
    fragment: str,
    batch: SettlementBatch,
    credit: BankRow,
    gap_days: int,
    config: AppConfig,
) -> CandidateScore:
    """0.50*prefix_ratio + 0.30*(1 - edit_ratio) + 0.20*amount_agrees."""
    weights = config.thresholds.fuzzy_utr
    utr = normalize_utr(batch.utr)

    # DIVIDED BY THE OBSERVED FRAGMENT'S LENGTH, never the true UTR's. At match
    # time the engine does not know which UTR is true - that is the thing being
    # identified - so dividing by 16 would score every short fragment low for
    # being short and make truncation itself look like evidence of a mismatch.
    prefix_ratio = (
        Decimal(_common_prefix_length(fragment, utr)) / Decimal(len(fragment)) if fragment else ZERO
    )

    # Compared against the candidate's SAME-LENGTH prefix, so a six-character
    # fragment is not charged ten edits for the ten characters it never had.
    comparable = utr[: len(fragment)]
    edit_ratio = (
        Decimal(_levenshtein(fragment, comparable)) / Decimal(max(len(fragment), len(comparable)))
        if fragment or comparable
        else ZERO
    )
    agrees = _amount_agrees(credit.amount, batch.net_total, config)

    score = (
        weights.weight_prefix * prefix_ratio
        + weights.weight_edit * (ONE - edit_ratio)
        + weights.weight_amount * agrees
    )
    return CandidateScore(
        batch_id=batch.batch_id,
        score=_quantize(score, config),
        path=ScoringPath.PREFIX,
        prefix_ratio=_quantize(prefix_ratio, config),
        edit_ratio=_quantize(edit_ratio, config),
        amount_agrees=agrees,
        date_proximity=None,
        observed_fragment=fragment,
        date_gap_days=gap_days,
    )


def _score_path_b(
    batch: SettlementBatch,
    credit: BankRow,
    gap_days: int,
    config: AppConfig,
) -> CandidateScore:
    """0.60*amount_agrees + 0.40*date_proximity. No prefix, no edit term."""
    weights = config.thresholds.fuzzy_utr
    agrees = _amount_agrees(credit.amount, batch.net_total, config)
    proximity = _date_proximity(gap_days, weights.date_window_days)
    score = weights.weight_amount_no_utr * agrees + weights.weight_date_no_utr * proximity
    return CandidateScore(
        batch_id=batch.batch_id,
        score=_quantize(score, config),
        path=ScoringPath.AMOUNT_DATE,
        prefix_ratio=None,  # undefined, NOT zero - the terms do not apply here
        edit_ratio=None,
        amount_agrees=agrees,
        date_proximity=proximity,
        observed_fragment=None,
        date_gap_days=gap_days,
    )


def _best_fragment(narration: str, utrs: Sequence[str], minimum: int) -> str | None:
    """The narration token sharing the longest leading run with any candidate UTR.

    POSITIVE definition: a run of at least `minimum` leading characters shared
    with some candidate IS a UTR fragment. There is no stop-list of merchant
    words - the same rule the generator was fixed for once already. A merchant
    name shares no leading run with hex, so it never qualifies; a six-character
    truncation does.

    Returning None is what selects Path B, so this function decides which
    formula runs and is deliberately conservative about claiming a fragment.
    """
    best: tuple[int, str] | None = None
    for token in extract_utr_candidates(narration):
        for utr in utrs:
            shared = _common_prefix_length(token, utr)
            if shared >= minimum and (best is None or shared > best[0]):
                best = (shared, token)
    return None if best is None else best[1]


def resolve(
    bank_row: BankRow,
    candidate_batches: Sequence[SettlementBatch],
    due_dates: dict[str, date],
    config: AppConfig,
) -> FuzzyVerdict:
    """Score one bank credit against candidate BATCHES, keyed by batch_id.

    Candidates are batches, never settlement lines: a settlement_id names one
    line and a batch_id names the payout that hits the bank, and SDD 3.3
    forbids the two sharing a namespace. Scoring lines here would make
    SETTLEMENT_TO_BATCH self-referential.

    ACCEPT only if best >= threshold AND (best - runner_up) >= min_separation.
    Failing either is AMBIGUOUS with every candidate and score attached, never
    a pick. An exact tie after the full tie-break chain is ABSTAINED - the
    engine does not choose between two things it cannot tell apart.
    """
    weights = config.thresholds.fuzzy_utr
    utrs = [normalize_utr(batch.utr) for batch in candidate_batches]
    fragment = _best_fragment(bank_row.narration, utrs, weights.fragment_min_chars)
    path = ScoringPath.PREFIX if fragment else ScoringPath.AMOUNT_DATE
    threshold = weights.accept_score if fragment else weights.accept_score_no_utr

    scores: list[CandidateScore] = []
    for batch in candidate_batches:
        due = due_dates.get(batch.batch_id, batch.settled_event_date)
        gap = (bank_row.value_date - due).days

        # HARD GATE, before any scoring. A credit outside the window is not
        # weak evidence for this batch, it is evidence about something else,
        # and a small non-zero score would let several of them out-vote a
        # single real candidate on aggregate.
        if abs(gap) > weights.date_window_days:
            scores.append(
                CandidateScore(
                    batch_id=batch.batch_id,
                    score=ZERO,
                    path=path,
                    prefix_ratio=None,
                    edit_ratio=None,
                    amount_agrees=ZERO,
                    date_proximity=None,
                    observed_fragment=fragment,
                    date_gap_days=gap,
                )
            )
            continue
        if fragment:
            scores.append(_score_path_a(fragment, batch, bank_row, gap, config))
        else:
            scores.append(_score_path_b(batch, bank_row, gap, config))

    ranked = tuple(sorted(scores, key=lambda candidate: candidate.sort_key()))
    if not ranked:
        return FuzzyVerdict(
            bank_txn_id=bank_row.bank_txn_id,
            outcome=FuzzyOutcome.ABSTAINED,
            path=path,
            matched_batch_id=None,
            best_score=None,
            runner_up_score=None,
            threshold=threshold,
            margin_required=weights.min_separation,
            candidates=(),
            reason="no candidate batches were offered",
        )

    best = ranked[0]
    runner_up = ranked[1].score if len(ranked) > 1 else None
    margin = best.score if runner_up is None else best.score - runner_up

    if best.score <= ZERO:
        outcome, reason = (
            FuzzyOutcome.ABSTAINED,
            (
                f"every candidate scored zero (best {best.batch_id} at {best.score}); "
                "nothing in the window agrees on amount or prefix"
            ),
        )
    elif runner_up is not None and best.score == runner_up:
        # An EXACT tie survived score-then-batch_id ordering, meaning two
        # batches are indistinguishable on the evidence. Taking the
        # lexicographically smaller batch_id would be a coin toss wearing a
        # determinism costume.
        outcome, reason = (
            FuzzyOutcome.ABSTAINED,
            (
                f"exact tie at {best.score} between {ranked[0].batch_id} and "
                f"{ranked[1].batch_id}; the evidence does not separate them"
            ),
        )
    elif best.score < threshold:
        outcome, reason = (
            FuzzyOutcome.AMBIGUOUS,
            (f"best score {best.score} is below the {path} threshold {threshold}"),
        )
    elif margin < weights.min_separation:
        outcome, reason = (
            FuzzyOutcome.AMBIGUOUS,
            (
                f"best {best.score} clears {threshold} but leads the runner-up by "
                f"{margin}, under the required {weights.min_separation}"
            ),
        )
    else:
        outcome, reason = (
            FuzzyOutcome.ACCEPTED,
            (f"{path} score {best.score} >= {threshold}, leading by {margin}"),
        )

    return FuzzyVerdict(
        bank_txn_id=bank_row.bank_txn_id,
        outcome=outcome,
        path=path,
        matched_batch_id=best.batch_id if outcome is FuzzyOutcome.ACCEPTED else None,
        best_score=best.score,
        runner_up_score=runner_up,
        threshold=threshold,
        margin_required=weights.min_separation,
        candidates=ranked,
        reason=reason,
    )


def money_of(value: Decimal) -> Money:
    """Quantize a score-domain Decimal to money. Kept explicit so a score,
    which is a ratio, is never mistaken for an amount."""
    return money(value)

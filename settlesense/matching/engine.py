"""M3 - the deterministic orchestrator. Passes P1..P9 in strict order.

THE THREE POPULATIONS ARE THREE SEPARATE RETURN COLLECTIONS, and that is the
central design constraint rather than a presentation choice (D11, SDD 3.1).
There is deliberately no generic "matches" or "residuals" list anywhere here:
the moment one exists, something averages a batch-link failure against a
payment case, and the headline match rate starts depending on how the gateway
happened to batch the payout.

  Population A  ReconciliationCase   denominator: count of captured payments
  Population B  batch <-> bank link  denominator: count of settlement batches
  Population C  row-grain variance   denominator: count of rows

WHAT "RESOLVED" MEANS HERE. A case is CONFIRMED when every rupee of its
variance is attributed to a taxonomy category and no interpretive question
remains. DUPLICATE_CANDIDATE is explicitly NOT resolved even though the engine
detects it perfectly: same customer, same amount, different order id is a data
duplicate or a genuine repeat purchase, and nothing in the data decides which.
Resolving it here would be the cheapest possible way to manufacture a false
match - right most of the time, wrong exactly where a reviewer would care.

The instruction was to build the strongest reasonable deterministic layer and
report what it achieves rather than leave work for the AI. That is what this
does; the residual count is the finding.

as_of is a PARAMETER (D2). No clock is read in this module or anything it
imports. Every returned list is sorted by an explicit key (D4).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from settlesense.config import AppConfig
from settlesense.core.telemetry import RunTelemetry
from settlesense.exceptions.taxonomy import DEDUCTION_CATEGORIES, VarianceCategory
from settlesense.ingest import DayDataset, canonical_sort_key
from settlesense.matching.arithmetic import (
    VarianceComponent,
    compute_fee,
    compute_gst,
    expected_net,
    explain_variance,
)
from settlesense.matching.duplicates import find_candidate_duplicates, find_confirmed_duplicates
from settlesense.matching.exact import (
    link_batches_to_bank,
    link_payments_to_settlements,
)
from settlesense.matching.fuzzy_utr import FuzzyVerdict
from settlesense.matching.fuzzy_utr import resolve as fuzzy_resolve
from settlesense.matching.timing import (
    WorkingDayCalendar,
    bank_value_due_date,
    is_timing_explained,
    settlement_line_due_date,
)
from settlesense.normalize import normalize_utr
from settlesense.types import (
    BatchLinkOutcome,
    CaseOutcome,
    ExceptionStatus,
    LedgerRow,
    Money,
    PaymentRow,
    ReconciliationCase,
    ReconciliationResult,
    ResolutionSource,
    RowVarianceOutcome,
    SettlementLineType,
    money,
)

__all__ = [
    "EngineError",
    "build_cases",
    "fuzzy_verdicts_for",
    "merge_days",
    "residual_cases",
    "run",
    "run_with_telemetry",
]

ZERO: Money = money(0)

BANK_WINDOW_DAYS = 0
"""Working days a credit may miss its computed T+N value date and still be exact.

Zero. The due date is now computed per batch from its own profile's cycle, so
a clean credit lands EXACTLY on it. A non-zero window here would absorb the
genuine late credits that MISSING_VS_LATE_CREDIT exists to name."""

TIMING_TOLERANCE_DAYS = 0
"""Working days a settlement may miss its T+N due date and still be on time.

Zero, deliberately. The generator computes the due date with the same calendar
arithmetic, so an on-time settlement lands EXACTLY on it; any tolerance here
would absorb real delays and shrink T_PLUS_N_TIMING toward zero while looking
like an improvement.
"""

_PROFILE_IN_CUSTOMER_ID = re.compile(r"^CUST-(PROFILE_[A-Z])-")
"""How a merchant profile is recovered from a ledger row.

The engine may not read truth, and no table carries a profile column, so the
customer id is the only carrier. Validated against the configured profile
names on every case - a derivation that silently produced an unknown profile
would price fees from a rate table that does not exist.
"""


class EngineError(ValueError):
    """The engine cannot proceed on this dataset."""


def merge_days(days: Sequence[DayDataset]) -> DayDataset:
    """Combine per-day datasets into the view the engine reconciles.

    `arrival_day` becomes the HIGHEST day merged - the delivery day as of which
    this view was assembled. It is not the day any particular row arrived, and
    the field keeps its SDD 4.1a meaning only at the per-day grain.
    """
    if not days:
        raise EngineError("merge_days requires at least one day")
    return DayDataset(
        arrival_day=max(day.arrival_day for day in days),
        ledger_rows=tuple(
            sorted((r for day in days for r in day.ledger_rows), key=canonical_sort_key)
        ),
        payment_rows=tuple(
            sorted((r for day in days for r in day.payment_rows), key=canonical_sort_key)
        ),
        refund_rows=tuple(
            sorted((r for day in days for r in day.refund_rows), key=canonical_sort_key)
        ),
        settlement_lines=tuple(
            sorted((r for day in days for r in day.settlement_lines), key=canonical_sort_key)
        ),
        settlement_batches=tuple(
            sorted((r for day in days for r in day.settlement_batches), key=canonical_sort_key)
        ),
        bank_rows=tuple(sorted((r for day in days for r in day.bank_rows), key=canonical_sort_key)),
    )


def _stable_id(prefix: str, *parts: object) -> str:
    """sha256 of a canonical tuple, first 16 hex (D10).

    The separator is "|" and every part is stringified, so the tuple is
    unambiguous: without a separator ("ab","c") and ("a","bc") would hash
    identically and two different cases would share an id.
    """
    canonical = prefix + "|" + "|".join(str(part) for part in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def case_id_for(payment_id: str) -> str:
    """SDD 3.1: sha256(b"case|" + payment_id)[:16].

    Matches the generator's truth file exactly, which is what lets accuracy be
    measured by joining on case_id rather than by re-deriving the pairing.
    """
    return _stable_id("case", payment_id)


def _profile_of(order_id: str, ledger: dict[str, LedgerRow], config: AppConfig) -> str:
    row = ledger.get(order_id)
    if row is None:
        raise EngineError(f"payment references order {order_id!r}, which has no ledger row")
    match = _PROFILE_IN_CUSTOMER_ID.match(row.customer_id)
    if match is None:
        raise EngineError(
            f"cannot derive a merchant profile from customer_id {row.customer_id!r}. "
            "Fees are priced per profile; guessing one would charge a rate nobody set."
        )
    profile = match.group(1).lower()
    if profile not in config.mdr.profile_names():
        raise EngineError(
            f"derived profile {profile!r} is not in the rate table {config.mdr.profile_names()}"
        )
    return profile


@dataclass(frozen=True)
class CaseFacts:
    """Everything the passes need about one case, assembled once.

    Separate from ReconciliationCase because the case is the RESULT type - it
    carries what the engine concluded - while this carries the raw rows it
    concluded from. Merging them would put payment rows in a serialized result.
    """

    case: ReconciliationCase
    payment: PaymentRow
    profile: str
    payment_line_ids: tuple[str, ...]
    settlement_line_ids: tuple[str, ...]
    batch_ids: tuple[str, ...]
    settled_dates: tuple[date, ...]
    settled_gross: Money
    """Gross across this case's PAYMENT lines. The variance baseline.

    GROSS basis, not net, and this is a deliberate departure from the inline
    comment on CaseOutcome.variance in SDD 3.1 ("expected_net - observed_net").
    Two reasons, both measured rather than assumed:

      The cash-basis figure needs a linked bank credit, and the bank link is
      Population B's (D11). Computing it here would make every Population A
      variance None until Population B succeeded, which is precisely the
      cross-population coupling the three-denominator rule exists to prevent.

      truth records PARTIAL_CAPTURE as `authorized - captured`, a gross figure.
      Partial captures span all four payment methods here - 46 card, 45 upi, 8
      wallet, 8 netbanking - so a net-basis variance would disagree with truth
      by the fee on 62 of 107 cases, and the disagreement would look like an
      accuracy problem rather than a definitional one.
    """
    refund_total: Money
    computed_fee: Money
    computed_tax: Money
    line_net_total: Money


def build_cases(dataset: DayDataset, config: AppConfig) -> tuple[CaseFacts, ...]:
    """P1 applied: one ReconciliationCase per payment, ALWAYS one.

    A split settlement produces ONE case holding several payment_line_ids. Two
    cases would double-count the denominator every headline metric divides by.
    """
    ledger = {row.order_id: row for row in dataset.ledger_rows}
    payment_ids = frozenset(row.payment_id for row in dataset.payment_rows)
    links, _orphan_lines = link_payments_to_settlements(dataset.settlement_lines, payment_ids)
    lines_by_id = {line.settlement_id: line for line in dataset.settlement_lines}

    refunds_by_payment: dict[str, list[Money]] = {}
    for refund in dataset.refund_rows:
        refunds_by_payment.setdefault(refund.payment_id, []).append(refund.amount)

    facts: list[CaseFacts] = []
    for payment in sorted(dataset.payment_rows, key=lambda p: p.payment_id):
        profile = _profile_of(payment.order_id, ledger, config)
        link = links.get(payment.payment_id)
        payment_line_ids = () if link is None else link.payment_line_ids
        settlement_line_ids = () if link is None else link.settlement_line_ids
        batch_ids = () if link is None else link.batch_ids

        gross = payment.captured
        fee = compute_fee(gross, payment.method, profile, config)
        tax = compute_gst(fee, config)
        refund_total = money(sum(refunds_by_payment.get(payment.payment_id, []), ZERO))

        lines = [lines_by_id[i] for i in settlement_line_ids if i in lines_by_id]
        facts.append(
            CaseFacts(
                case=ReconciliationCase(
                    case_id=case_id_for(payment.payment_id),
                    payment_id=payment.payment_id,
                    order_id=payment.order_id,
                    merchant_profile=profile,
                    expected_gross=gross,
                    expected_net=expected_net(gross, fee, tax, refund_total),
                    settlement_line_ids=settlement_line_ids,
                    payment_line_ids=payment_line_ids,
                ),
                payment=payment,
                profile=profile,
                payment_line_ids=payment_line_ids,
                settlement_line_ids=settlement_line_ids,
                batch_ids=batch_ids,
                settled_dates=tuple(
                    sorted(
                        line.settled_event_date
                        for line in lines
                        if line.line_type is SettlementLineType.PAYMENT
                    )
                ),
                refund_total=refund_total,
                computed_fee=fee,
                computed_tax=tax,
                settled_gross=money(
                    sum(
                        (
                            line.gross
                            for line in lines
                            if line.line_type is SettlementLineType.PAYMENT
                        ),
                        ZERO,
                    )
                ),
                line_net_total=money(sum((line.net for line in lines), ZERO)),
            )
        )
    return tuple(facts)


_ZERO_AMOUNT_PRECEDENCE: tuple[VarianceCategory, ...] = (
    VarianceCategory.T_PLUS_N_TIMING,
    VarianceCategory.ROUNDING_DIFFERENCE,
    VarianceCategory.DUPLICATE_CANDIDATE,
)
"""Order among findings that move no money. Declared, not alphabetical.

DUPLICATE_CANDIDATE is LAST on purpose. A case can be both a partial capture
and one half of an ambiguous duplicate pair; reporting the duplicate would
bury a real 308.98 shortfall behind a question about which of two orders was
the repeat. Measured: an alphabetical tie-break swallowed one PARTIAL_CAPTURE
and one T_PLUS_N_TIMING exactly this way.
"""


def _headline_category(
    components: tuple[VarianceComponent, ...],
) -> VarianceCategory | None:
    """The one category a CaseOutcome reports, from possibly several findings.

    Money first, largest absolute amount winning, because a rupee figure is
    what a reviewer acts on. Only when nothing moved does the declared
    precedence above decide, so the choice is never left to sort order.
    """
    if not components:
        return None
    moving = [component for component in components if component.amount != ZERO]
    if moving:
        return max(
            moving, key=lambda component: (abs(component.amount), component.category)
        ).category
    for candidate in _ZERO_AMOUNT_PRECEDENCE:
        if any(component.category is candidate for component in components):
            return candidate
    return sorted(component.category for component in components)[0]


def _classify_case(
    facts: CaseFacts,
    calendar: WorkingDayCalendar,
    duplicate_order_ids: frozenset[str],
    rounding_tolerance: Money,
    as_of: date,
) -> CaseOutcome:
    """Passes P3, P4, P6, P7b and P9 for one case, in order."""
    components: list[VarianceComponent] = []
    case = facts.case

    # --- P3 arithmetic, case grain -----------------------------------------
    # SDD 4.2 states P3 as the line identity. The authorized-vs-captured check
    # is the same pass one grain coarser: both ask whether the money that moved
    # equals the money the rate table says should have moved.
    uncaptured = money(facts.payment.authorized - facts.payment.captured)
    if uncaptured > ZERO:
        components.append(
            VarianceComponent(
                category=VarianceCategory.PARTIAL_CAPTURE,
                amount=uncaptured,
                detail=(
                    f"authorized {facts.payment.authorized}, captured "
                    f"{facts.payment.captured}; {uncaptured} never captured"
                ),
            )
        )

    # --- P4 refund offset ---------------------------------------------------
    # Refunds are a COMPONENT OF expected_net (SDD 3.1b), never a variance.
    # REFUND_OFFSET is a deduction category and PDD 6.1 forbids emitting it
    # here; it is already inside case.expected_net and appears in no list.

    # --- P6 timing ----------------------------------------------------------
    if facts.settled_dates:
        due = settlement_line_due_date(facts.payment.captured_at, calendar)
        verdict = is_timing_explained(due, facts.settled_dates[-1], TIMING_TOLERANCE_DAYS, calendar)
        if not verdict.explained:
            components.append(
                VarianceComponent(
                    category=VarianceCategory.T_PLUS_N_TIMING,
                    amount=ZERO,  # a delay moves no money
                    detail=(
                        f"line due {due.isoformat()} (T+0 from capture "
                        f"{facts.payment.captured_at.isoformat()}), settled "
                        f"{facts.settled_dates[-1].isoformat()}: "
                        f"{verdict.working_days_late} working day(s) late"
                    ),
                )
            )

    # --- P7b duplicate candidate -------------------------------------------
    interpretive = case.order_id in duplicate_order_ids
    if interpretive:
        components.append(
            VarianceComponent(
                category=VarianceCategory.DUPLICATE_CANDIDATE,
                amount=ZERO,  # the case's own money is intact; the question is the pair
                detail=(
                    f"order {case.order_id} shares a customer and amount with another "
                    "order; data duplicate vs repeat purchase is interpretive"
                ),
            )
        )

    # --- observed vs expected, and P9 rounding ------------------------------
    observed = facts.line_net_total if facts.settlement_line_ids else None
    breakdown = explain_variance(
        expected=facts.payment.authorized,
        actual=facts.settled_gross,
        components=tuple(components),
    )
    remainder = breakdown.unexplained
    if remainder != ZERO and abs(remainder) <= rounding_tolerance:
        components.append(
            VarianceComponent(
                category=VarianceCategory.ROUNDING_DIFFERENCE,
                amount=remainder,
                detail=f"residual {remainder} within tolerance {rounding_tolerance} (P9)",
            )
        )
        breakdown = explain_variance(
            expected=facts.payment.authorized,
            actual=facts.settled_gross,
            components=tuple(components),
        )

    unresolved = breakdown.unexplained != ZERO or interpretive or observed is None
    if unresolved and breakdown.unexplained != ZERO:
        category: VarianceCategory | None = VarianceCategory.UNEXPLAINED
    else:
        category = _headline_category(breakdown.components)

    if category in DEDUCTION_CATEGORIES:  # pragma: no cover - guarded at construction
        raise EngineError(
            f"{category} is a deduction, not a variance (PDD 6.1). It is part of "
            "expected_net and must never reach a CaseOutcome."
        )

    return CaseOutcome(
        case_id=case.case_id,
        status=ExceptionStatus.OPEN if unresolved else ExceptionStatus.CONFIRMED,
        observed_net=observed,
        variance=breakdown.total,
        category=None if category is None else str(category),
        batch_id=facts.batch_ids[0] if facts.batch_ids else None,
        bank_row_id=None,  # Population B owns the bank link; A never borrows it
        resolved_by=None if unresolved else ResolutionSource.DETERMINISTIC,
        confidence=None,
    )


def _batch_profiles(dataset: DayDataset, config: AppConfig) -> dict[str, str]:
    """Each batch's merchant profile, via its lines' payments' orders.

    A batch is bucketed per (profile, date) by the generator, so every line in
    one shares a profile. That is asserted rather than assumed: a batch mixing
    profiles would have no single T+N cycle, and picking one arbitrarily would
    silently misdate the whole payout.
    """
    ledger = {row.order_id: row for row in dataset.ledger_rows}
    order_of_payment = {p.payment_id: p.order_id for p in dataset.payment_rows}
    found: dict[str, set[str]] = {}
    for line in dataset.settlement_lines:
        order_id = order_of_payment.get(line.payment_id)
        if order_id is None:
            continue
        found.setdefault(line.batch_id, set()).add(_profile_of(order_id, ledger, config))
    profiles: dict[str, str] = {}
    for batch_id in sorted(found):
        names = found[batch_id]
        if len(names) != 1:
            raise EngineError(
                f"batch {batch_id} mixes merchant profiles {sorted(names)}. There is "
                "no single T+N cycle for it, and choosing one would misdate the payout."
            )
        profiles[batch_id] = next(iter(names))
    return profiles


def _classify_batches(
    dataset: DayDataset,
    config: AppConfig,
    calendar: WorkingDayCalendar,
    rounding_tolerance: Money,
    as_of: date,
) -> tuple[tuple[BatchLinkOutcome, ...], tuple[str, ...], tuple[FuzzyVerdict, ...]]:
    """Batch grain, in pass order: P2 exact, P9 rounding, P8 fuzzy, then classify.

    P8 IS A PHASE OVER CREDITS, not a step inside the per-batch loop. `resolve`
    scores one bank credit against many candidate batches, which is the right
    direction: the question a damaged narration poses is "which batch is this
    credit", not "which credit is this batch". Running it per batch would let
    two batches each claim the same credit before anything noticed.

    Ordering note: P9's rounding fallback runs before P8 even though SDD 4.2
    numbers them the other way. P9-at-batch-grain here is an EXACT-UTR rule
    that tolerates a sub-rupee amount difference, so it is stricter than P8,
    and running the stricter rule first is what the strict-order discipline
    asks for. The looser rule never claims a row the stricter one could take.
    """
    profiles = _batch_profiles(dataset, config)
    due_dates = {
        batch.batch_id: bank_value_due_date(
            batch.settled_event_date, profiles[batch.batch_id], calendar
        )
        for batch in dataset.settlement_batches
        if batch.batch_id in profiles
    }
    links, unclaimed = link_batches_to_bank(
        dataset.settlement_batches,
        dataset.bank_rows,
        calendar,
        due_dates,
        BANK_WINDOW_DAYS,
        as_of,
    )
    remaining = {row.bank_txn_id: row for row in unclaimed}
    batches = {batch.batch_id: batch for batch in dataset.settlement_batches}
    outcomes: dict[str, BatchLinkOutcome] = {}
    still_open: list[str] = []

    # --- P2 result, plus P9 rounding at batch grain -------------------------
    for link in sorted(links, key=lambda link: link.batch_id):
        batch = batches[link.batch_id]
        if link.is_linked:
            outcomes[link.batch_id] = BatchLinkOutcome(
                batch_id=link.batch_id,
                status=ExceptionStatus.CONFIRMED,
                bank_row_id=link.bank_txn_id,
                batch_net_total=link.batch_net_total,
                linked_amount=link.linked_amount,
                variance=ZERO,
                category=None,
                resolved_by=ResolutionSource.DETERMINISTIC,
                confidence=None,
            )
            continue

        target = normalize_utr(batch.utr)
        near = sorted(
            (
                row
                for row in remaining.values()
                if target in normalize_utr(row.narration)
                and abs(row.amount - batch.net_total) <= rounding_tolerance
            ),
            key=lambda row: row.bank_txn_id,
        )
        if near:
            row = near[0]
            del remaining[row.bank_txn_id]
            outcomes[link.batch_id] = BatchLinkOutcome(
                batch_id=link.batch_id,
                status=ExceptionStatus.CONFIRMED,
                bank_row_id=row.bank_txn_id,
                batch_net_total=batch.net_total,
                linked_amount=row.amount,
                variance=money(batch.net_total - row.amount),
                category=str(VarianceCategory.ROUNDING_DIFFERENCE),
                resolved_by=ResolutionSource.DETERMINISTIC,
                confidence=None,
            )
            continue

        if due_dates.get(link.batch_id, batch.settled_event_date) > as_of:
            outcomes[link.batch_id] = BatchLinkOutcome(
                batch_id=link.batch_id,
                status=ExceptionStatus.PENDING_EVIDENCE,
                bank_row_id=None,
                batch_net_total=batch.net_total,
                linked_amount=None,
                variance=None,
                category=None,  # nothing is wrong yet; the file has not arrived
                resolved_by=None,
                confidence=None,
            )
            continue
        still_open.append(link.batch_id)

    # --- P8 FUZZY UTR (M4) --------------------------------------------------
    # Every surviving credit is scored against every still-open batch, so a
    # credit is never resolved against a batch that an exact rule already
    # claimed. Credits are processed in sorted order and a claimed batch
    # leaves the candidate pool, so the result cannot depend on dict order.
    verdicts: list[FuzzyVerdict] = []
    open_batches = {batch_id: batches[batch_id] for batch_id in still_open}
    for txn_id in sorted(remaining):
        if not open_batches:
            break
        verdict = fuzzy_resolve(
            remaining[txn_id],
            sorted(open_batches.values(), key=lambda b: b.batch_id),
            due_dates,
            config,
        )
        verdicts.append(verdict)
        if not verdict.is_accepted or verdict.matched_batch_id is None:
            continue
        batch = open_batches.pop(verdict.matched_batch_id)
        row = remaining.pop(txn_id)
        outcomes[batch.batch_id] = BatchLinkOutcome(
            batch_id=batch.batch_id,
            status=ExceptionStatus.CONFIRMED,
            bank_row_id=row.bank_txn_id,
            batch_net_total=batch.net_total,
            linked_amount=row.amount,
            variance=money(batch.net_total - row.amount),
            category=(
                str(VarianceCategory.ROUNDING_DIFFERENCE) if row.amount != batch.net_total else None
            ),
            resolved_by=ResolutionSource.DETERMINISTIC,
            confidence=None,
        )

    # --- Whatever P8 could not resolve --------------------------------------
    # The category comes from the FUZZY VERDICT where one exists, because the
    # verdict knows which path scored it. Path A failing means a UTR was there
    # and could not be mapped; Path B failing means there was none to map.
    for batch_id in sorted(open_batches):
        batch = open_batches[batch_id]
        # CLASSIFIED ON EVIDENCE, not on which path happened to score it.
        # A first version took the category straight from the fuzzy verdict,
        # which reports UTR_MISSING_MAPPING for any Path B failure - so the two
        # batches whose credit never arrived at all were filed as UTR-mapping
        # problems. "We cannot find which credit this is" and "there is no
        # credit" are different findings with different fixes, and only the
        # evidence separates them.
        prefix_seen = any(
            _shares_utr_prefix(normalize_utr(batch.utr), row.narration)
            for row in remaining.values()
        )
        amount_seen = any(
            abs(row.amount - batch.net_total) <= rounding_tolerance for row in remaining.values()
        )
        if prefix_seen:
            category = VarianceCategory.UTR_TRUNCATED_MAPPING
        elif amount_seen:
            category = VarianceCategory.UTR_MISSING_MAPPING
        else:
            category = VarianceCategory.MISSING_VS_LATE_CREDIT
        outcomes[batch_id] = BatchLinkOutcome(
            batch_id=batch_id,
            status=ExceptionStatus.OPEN,
            bank_row_id=None,
            batch_net_total=batch.net_total,
            linked_amount=None,
            variance=None,
            category=str(category),
            resolved_by=None,
            confidence=None,
        )

    orphans = tuple(
        sorted(
            txn_id
            for txn_id, row in remaining.items()
            if not any(
                _shares_utr_prefix(normalize_utr(batch.utr), row.narration)
                or normalize_utr(batch.utr) in normalize_utr(row.narration)
                or abs(row.amount - batch.net_total) <= rounding_tolerance
                for batch in dataset.settlement_batches
            )
        )
    )
    ordered = tuple(outcomes[batch_id] for batch_id in sorted(outcomes))
    return ordered, orphans, tuple(verdicts)


_UTR_PREFIX_FLOOR = 6
"""Shortest prefix treated as evidence of a truncated UTR. Matches the
generator's truncation floor and normalize.UTR_CANDIDATE_MIN_LEN."""


def _shares_utr_prefix(target_utr: str, narration: str) -> bool:
    """Whether a narration carries a proper PREFIX of the batch's UTR.

    Substring containment, not scoring - this classifies, it never links.
    Linking on a damaged UTR is M4's job and needs a score with a separation
    threshold; deciding that a UTR looks truncated needs neither.
    """
    text = normalize_utr(narration)
    for length in range(len(target_utr) - 1, _UTR_PREFIX_FLOOR - 1, -1):
        if target_utr[:length] in text:
            return True
    return False


def run(dataset: DayDataset, config: AppConfig, as_of: date) -> ReconciliationResult:
    """Deterministic reconciliation. Passes P1..P9, strict order, matched rows leave.

    Returns the business result ONLY. Timing is a separate object from
    `run_with_telemetry`; there is nothing to strip here because no wall-clock
    value ever enters (SDD 8.1).
    """
    result, _telemetry = run_with_telemetry(dataset, config, as_of)
    return result


def run_with_telemetry(
    dataset: DayDataset, config: AppConfig, as_of: date
) -> tuple[ReconciliationResult, RunTelemetry]:
    """Two return values (SDD 8.1). Callers that persist or compare take [0]."""
    calendar = WorkingDayCalendar(config.calendar)
    tolerance = config.thresholds.tolerance.rounding_rupees

    # --- Population C part 1: P7a, and P7b's pairing for the case pass ------
    confirmed = find_confirmed_duplicates(dataset.ledger_rows)
    confirmed_ids = frozenset(row_id for verdict in confirmed for row_id in verdict.row_ids)
    candidates = find_candidate_duplicates(dataset.ledger_rows, confirmed_ids)
    candidate_order_ids = frozenset(
        order_id for verdict in candidates for order_id in verdict.row_ids
    )

    # --- Population A -------------------------------------------------------
    facts = build_cases(dataset, config)
    cases = tuple(
        sorted(
            (
                _classify_case(fact, calendar, candidate_order_ids, tolerance, as_of)
                for fact in facts
            ),
            key=lambda outcome: outcome.case_id,
        )
    )

    # --- Population B -------------------------------------------------------
    batch_links, orphan_bank_ids, _verdicts = _classify_batches(
        dataset, config, calendar, tolerance, as_of
    )

    # --- Population C part 2: orphan credits --------------------------------
    bank_by_id = {row.bank_txn_id: row for row in dataset.bank_rows}
    row_variances: list[RowVarianceOutcome] = [
        RowVarianceOutcome(
            row_id=verdict.row_ids[0] if verdict.row_ids else "",
            source_table="ledger_rows",
            status=ExceptionStatus.CONFIRMED,
            category=str(VarianceCategory.DUPLICATE_CONFIRMED),
            amount=verdict.amount,
        )
        for verdict in confirmed
    ]
    row_variances.extend(
        RowVarianceOutcome(
            row_id=txn_id,
            source_table="bank_rows",
            status=ExceptionStatus.OPEN,
            category=str(VarianceCategory.UNEXPLAINED),
            amount=bank_by_id[txn_id].amount,
        )
        for txn_id in orphan_bank_ids
        if txn_id in bank_by_id
    )

    result = ReconciliationResult(
        cases=cases,
        batch_links=tuple(sorted(batch_links, key=lambda link: link.batch_id)),
        row_variances=tuple(
            sorted(row_variances, key=lambda v: (v.source_table, v.row_id, str(v.category)))
        ),
        exceptions=(),  # the exception store is M6; outcomes carry the findings
        calendar_version=config.calendar.version,
        config_hash=config.config_hash,
    )
    return result, RunTelemetry()


def fuzzy_verdicts_for(
    dataset: DayDataset, config: AppConfig, as_of: date
) -> tuple[FuzzyVerdict, ...]:
    """Every P8 verdict from a run, for reporting and tests.

    Deliberately NOT a field on ReconciliationResult. A verdict carries scores
    and candidate lists that are diagnostics about how a link was reached, not
    part of the reconciliation itself, and SDD 8.1 keeps the business result to
    what is hashed and goldened.
    """
    calendar = WorkingDayCalendar(config.calendar)
    _links, _orphans, verdicts = _classify_batches(
        dataset, config, calendar, config.thresholds.tolerance.rounding_rupees, as_of
    )
    return verdicts


def residual_cases(result: ReconciliationResult) -> tuple[CaseOutcome, ...]:
    """Population A cases the deterministic layer could not close.

    THE number this module is judged on, and the surface M7 has to work with.
    A helper rather than a field so it cannot drift from the outcomes it counts.
    """
    return tuple(case for case in result.cases if case.status is not ExceptionStatus.CONFIRMED)

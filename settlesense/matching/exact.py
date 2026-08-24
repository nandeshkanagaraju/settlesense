"""M3 - P1 exact payment link, P2 exact batch<->bank link, P4 refund offset.

The exact passes run FIRST and claim greedily, which is the whole point of a
strict pass order: a row that an exact rule can account for must never be left
for a looser rule to explain differently. P1 taking a row that P3 could also
have taken is not a conflict to resolve, it is the ordering working.

Each pass is a pure function of its inputs. `as_of` is a parameter (D2); no
clock is read anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from settlesense.matching.timing import WorkingDayCalendar, is_timing_explained
from settlesense.normalize import normalize_utr
from settlesense.types import (
    BankDirection,
    BankRow,
    Money,
    RefundRow,
    SettlementBatch,
    SettlementLine,
    SettlementLineType,
)

__all__ = [
    "BankLink",
    "PaymentLink",
    "RefundLink",
    "link_batches_to_bank",
    "link_payments_to_settlements",
    "link_refunds",
]


@dataclass(frozen=True)
class PaymentLink:
    """P1. One payment and the settlement lines that reference it."""

    payment_id: str
    payment_line_ids: tuple[str, ...]  # PAYMENT lines only, sorted
    settlement_line_ids: tuple[str, ...]  # every line touching the payment, sorted
    batch_ids: tuple[str, ...]  # sorted, deduplicated

    @property
    def is_split(self) -> bool:
        """SDD 3.2: >= 2 PAYMENT lines is the structural condition for a split.

        Counted on payment_line_ids, NEVER settlement_line_ids. A payment with
        one settlement and one refund has two settlement lines and is not a
        split; conflating the two would report every refunded payment as one.
        """
        return len(self.payment_line_ids) >= 2


def link_payments_to_settlements(
    lines: Sequence[SettlementLine],
    payment_ids: frozenset[str],
) -> tuple[dict[str, PaymentLink], tuple[SettlementLine, ...]]:
    """P1. `settlement.payment_id == payment.payment_id`, exact.

    Returns (links by payment_id, lines referencing no known payment). The
    second value is not a diagnostic afterthought: a settlement line for a
    payment that does not exist is an orphan, and dropping it silently would
    remove money from the batch-side conservation identity.
    """
    grouped: dict[str, list[SettlementLine]] = {}
    orphans: list[SettlementLine] = []
    for line in lines:
        if line.payment_id in payment_ids:
            grouped.setdefault(line.payment_id, []).append(line)
        else:
            orphans.append(line)

    links: dict[str, PaymentLink] = {}
    for payment_id in sorted(grouped):
        group = grouped[payment_id]
        links[payment_id] = PaymentLink(
            payment_id=payment_id,
            payment_line_ids=tuple(
                sorted(
                    line.settlement_id
                    for line in group
                    if line.line_type is SettlementLineType.PAYMENT
                )
            ),
            settlement_line_ids=tuple(sorted(line.settlement_id for line in group)),
            batch_ids=tuple(sorted({line.batch_id for line in group})),
        )
    return links, tuple(sorted(orphans, key=lambda line: line.settlement_id))


@dataclass(frozen=True)
class BankLink:
    """P2. One batch and the bank credit that settled it, if found."""

    batch_id: str
    bank_txn_id: str | None
    batch_net_total: Money
    linked_amount: Money | None
    matched_on_utr: bool
    matched_on_amount: bool
    within_window: bool
    detail: str

    @property
    def is_linked(self) -> bool:
        return self.bank_txn_id is not None


def link_batches_to_bank(
    batches: Sequence[SettlementBatch],
    bank_rows: Sequence[BankRow],
    calendar: WorkingDayCalendar,
    due_dates: Mapping[str, date],
    window_days: int,
    as_of: date,
) -> tuple[tuple[BankLink, ...], tuple[BankRow, ...]]:
    """P2. Keyed on batch_id: normalized UTR equal, amount equal, date in window.

    ALL THREE must hold. UTR alone would link a batch to a credit of the wrong
    amount; amount alone is not unique across 39 batches sharing round figures.
    Requiring all three is what makes a P2 link safe enough to count as exact,
    and what leaves the ambiguous cases for M4 rather than resolving them badly.

    Returns (links for every batch, bank rows claimed by nobody). An unclaimed
    credit is an orphan and becomes a Population C row variance - it is money
    in the account with no batch to explain it, which is a finding, not noise.

    Only CREDITS are considered. A debit is not a settlement payout, and
    matching one would invert the sign of the link.
    """
    available = {row.bank_txn_id: row for row in bank_rows if row.direction is BankDirection.CREDIT}
    # Index by normalized UTR. Bank narrations are free text, so the UTR is
    # sought among the narration's candidate tokens rather than a fixed field.
    claimed: set[str] = set()
    links: list[BankLink] = []

    for batch in sorted(batches, key=lambda b: b.batch_id):
        target_utr = normalize_utr(batch.utr)
        # The batch's OWN T+N due date, per its merchant profile. A blanket
        # window would be wrong in both directions at once: too tight for
        # profile_c at T+3 and too loose for profile_b at T+1, so it would miss
        # real links and accept late ones as exact in the same run.
        due = due_dates.get(batch.batch_id, batch.settled_event_date)
        found: BankRow | None = None
        for txn_id in sorted(available):
            if txn_id in claimed:
                continue
            row = available[txn_id]
            if target_utr not in normalize_utr(row.narration):
                continue
            if row.amount != batch.net_total:
                continue
            verdict = is_timing_explained(due, row.value_date, window_days, calendar)
            if not verdict.explained:
                continue
            found = row
            break

        if found is not None:
            claimed.add(found.bank_txn_id)
            links.append(
                BankLink(
                    batch_id=batch.batch_id,
                    bank_txn_id=found.bank_txn_id,
                    batch_net_total=batch.net_total,
                    linked_amount=found.amount,
                    matched_on_utr=True,
                    matched_on_amount=True,
                    within_window=True,
                    detail=f"exact UTR, amount and date within T+{window_days}",
                )
            )
            continue

        # Report WHY it failed. "Unlinked" alone cannot distinguish a truncated
        # UTR (M4 may still resolve it) from a credit that never arrived.
        utr_present = any(
            target_utr in normalize_utr(row.narration)
            for txn_id, row in available.items()
            if txn_id not in claimed
        )
        amount_present = any(
            row.amount == batch.net_total
            for txn_id, row in available.items()
            if txn_id not in claimed
        )
        links.append(
            BankLink(
                batch_id=batch.batch_id,
                bank_txn_id=None,
                batch_net_total=batch.net_total,
                linked_amount=None,
                matched_on_utr=utr_present,
                matched_on_amount=amount_present,
                within_window=False,
                detail=(
                    f"no exact match: utr_seen={utr_present} amount_seen={amount_present}"
                    f" as_of={as_of.isoformat()}"
                ),
            )
        )

    unclaimed = tuple(
        sorted(
            (row for txn_id, row in available.items() if txn_id not in claimed),
            key=lambda row: row.bank_txn_id,
        )
    )
    return tuple(links), unclaimed


@dataclass(frozen=True)
class RefundLink:
    """P4. A refund row matched to its REFUND settlement line by refund_id."""

    refund_id: str
    payment_id: str
    amount: Money
    settlement_id: str | None
    amount_agrees: bool

    @property
    def is_linked(self) -> bool:
        return self.settlement_id is not None


def link_refunds(
    refunds: Sequence[RefundRow],
    lines: Sequence[SettlementLine],
) -> tuple[tuple[RefundLink, ...], tuple[SettlementLine, ...]]:
    """P4. Exact-amount refund matched by `refund_id`.

    The line's net is NEGATIVE (SDD 3.3), so agreement is checked against its
    absolute value. Comparing signed-to-unsigned would fail every refund and
    report 298 phantom variances.
    """
    by_refund_id = {
        line.refund_id: line
        for line in lines
        if line.line_type is SettlementLineType.REFUND and line.refund_id is not None
    }
    links: list[RefundLink] = []
    matched: set[str] = set()
    for refund in sorted(refunds, key=lambda r: r.refund_id):
        line = by_refund_id.get(refund.refund_id)
        if line is not None:
            matched.add(line.settlement_id)
        links.append(
            RefundLink(
                refund_id=refund.refund_id,
                payment_id=refund.payment_id,
                amount=refund.amount,
                settlement_id=None if line is None else line.settlement_id,
                amount_agrees=line is not None and abs(line.net) == refund.amount,
            )
        )
    unmatched_lines = tuple(
        sorted(
            (
                line
                for line in lines
                if line.line_type is SettlementLineType.REFUND and line.settlement_id not in matched
            ),
            key=lambda line: line.settlement_id,
        )
    )
    return tuple(links), unmatched_lines

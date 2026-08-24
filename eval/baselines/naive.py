"""M5 baseline - amount plus date window, no identifiers at all.

The floor. It never looks at a UTR, a payment id, or a narration: for each bank
credit it takes the batch whose net_total matches within tolerance and whose
date is nearest. That is what a spreadsheet does.

IT MAY WELL MATCH MORE THAN THE ENGINE DOES, and that is not a defect in the
baseline or a bug in the test. Pairing on amount and date alone links anything
plausible, including things that are wrong; the engine abstains where evidence
is ambiguous. The comparison that matters is PRECISION, not volume, and no
test here asserts a ranking between baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from settlesense.config import AppConfig
from settlesense.ingest import DayDataset
from settlesense.types import BankDirection, Money, money

__all__ = ["NaiveLink", "run_naive"]


@dataclass(frozen=True)
class NaiveLink:
    batch_id: str
    bank_txn_id: str
    amount_difference: Money
    date_gap_days: int


def run_naive(
    dataset: DayDataset, config: AppConfig, as_of: date, window_days: int = 5
) -> tuple[NaiveLink, ...]:
    """Link batches to credits on amount and date proximity only.

    Greedy, in sorted batch order, and a claimed credit leaves the pool - so
    the output does not depend on dict iteration order (D4). A wider window
    than the engine's on purpose: this baseline exists to show what happens
    when you accept weaker evidence, so tightening it to flatter the comparison
    would defeat the point.
    """
    tolerance = config.thresholds.tolerance.rounding_rupees
    available = {
        row.bank_txn_id: row
        for row in dataset.bank_rows
        if row.direction is BankDirection.CREDIT and row.value_date <= as_of
    }
    links: list[NaiveLink] = []

    for batch in sorted(dataset.settlement_batches, key=lambda b: b.batch_id):
        candidates = [
            row
            for row in available.values()
            if abs(row.amount - batch.net_total) <= tolerance
            and abs((row.value_date - batch.settled_event_date).days) <= window_days
        ]
        if not candidates:
            continue
        # Nearest date, then smallest amount difference, then id: a TOTAL order,
        # so "the best candidate" is never decided by input sequence.
        best = min(
            candidates,
            key=lambda row: (
                abs((row.value_date - batch.settled_event_date).days),
                abs(row.amount - batch.net_total),
                row.bank_txn_id,
            ),
        )
        del available[best.bank_txn_id]
        links.append(
            NaiveLink(
                batch_id=batch.batch_id,
                bank_txn_id=best.bank_txn_id,
                amount_difference=money(best.amount - batch.net_total),
                date_gap_days=(best.value_date - batch.settled_event_date).days,
            )
        )
    return tuple(links)


def naive_case_outcomes(dataset: DayDataset) -> int:
    """How many payments a naive reading would call reconciled.

    Every payment with any settlement line, with no arithmetic check at all.
    Reported so the volume-versus-precision difference is visible as a number
    rather than as an argument: this counts a partial capture as fine, because
    a line exists.
    """
    settled = {line.payment_id for line in dataset.settlement_lines}
    return sum(1 for payment in dataset.payment_rows if payment.payment_id in settled)


def naive_amount_only_agreement(dataset: DayDataset) -> Decimal:
    """Share of batches whose net_total is unique across the dataset.

    The naive baseline's whole discriminating power. Near 1.0 means amount
    alone nearly identifies a batch HERE, which is a property of this dataset's
    low batch density and not a general one - the same condition LIMITATIONS.md
    records for fuzzy Path B.
    """
    totals = [batch.net_total for batch in dataset.settlement_batches]
    if not totals:
        return Decimal(0)
    unique = sum(1 for total in totals if totals.count(total) == 1)
    return (Decimal(unique) / Decimal(len(totals))).quantize(Decimal("0.000001"))

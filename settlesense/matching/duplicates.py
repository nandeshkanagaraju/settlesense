"""M3 - P7a DUPLICATE_CONFIRMED and P7b DUPLICATE_CANDIDATE.

The two are deliberately not one rule with a confidence score. They differ in
KIND, not in degree:

  P7a is an INGESTION ARTEFACT. Byte-identical content on a distinct source
  line means the same row was delivered twice. There is nothing to interpret;
  the second copy is not a second sale. Deterministic, resolved here.

  P7b is a QUESTION. Same customer, same amount, different order id is either
  a data duplicate or a genuine repeat purchase, and the data cannot tell you
  which. It is emitted to the residual set with BOTH rows attached and is not
  auto-classified in either direction. Guessing here would be the single
  easiest way to manufacture a false match, because the guess is right most of
  the time and wrong exactly where it matters.

Every decision returns a DuplicateVerdict naming the rule that fired, never a
bare bool. "True" cannot be audited, cannot appear in an evidence link, and
cannot tell a reviewer which of two very different rules produced it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from settlesense.exceptions.taxonomy import VarianceCategory
from settlesense.types import LedgerRow, Money

__all__ = [
    "DuplicateRule",
    "DuplicateVerdict",
    "find_candidate_duplicates",
    "find_confirmed_duplicates",
]


class DuplicateRule(StrEnum):
    """Which rule produced a verdict. Named so a verdict can be audited."""

    BYTE_IDENTICAL_DISTINCT_LINE = "byte_identical_distinct_line"
    SAME_CUSTOMER_SAME_GROSS = "same_customer_same_gross"
    NO_RULE_FIRED = "no_rule_fired"


@dataclass(frozen=True)
class DuplicateVerdict:
    """The outcome of a duplicate check, with the rule that decided it."""

    rule: DuplicateRule
    category: VarianceCategory | None
    row_ids: tuple[str, ...]  # sorted; every row involved, not just the later one
    amount: Money | None
    detail: str

    @property
    def is_duplicate(self) -> bool:
        return self.rule is not DuplicateRule.NO_RULE_FIRED

    @property
    def is_resolved_here(self) -> bool:
        """P7a resolves. P7b does not - it is a question for the AI layer."""
        return self.category is VarianceCategory.DUPLICATE_CONFIRMED


def _content_key(row: LedgerRow) -> tuple[str, ...]:
    """Every field of a ledger row, as text. The 'byte-identical' comparison.

    All fields, not a chosen subset: a subset would call two rows identical
    while they differed in the very column that distinguished them, and which
    column that is cannot be known in advance.
    """
    return (
        row.order_id,
        row.invoice_no,
        str(row.gross),
        row.order_date.isoformat(),
        row.customer_id,
        row.sku,
    )


def find_confirmed_duplicates(rows: Sequence[LedgerRow]) -> tuple[DuplicateVerdict, ...]:
    """P7a. Byte-identical content appearing more than once.

    Returns one verdict per EXTRA copy, not one per group. Two identical rows
    means one duplicate; three means two. Counting groups would report 27
    duplicated order ids as 27 excess rows even if one of them appeared three
    times, and Population C's denominator is a row count.

    The first occurrence in sorted order is treated as the original. That is
    arbitrary but must be deterministic - `rows` arrives sorted by the full
    field tuple from ingest, so the choice does not vary between runs (D4).
    """
    groups: dict[tuple[str, ...], list[LedgerRow]] = {}
    for row in rows:
        groups.setdefault(_content_key(row), []).append(row)

    verdicts: list[DuplicateVerdict] = []
    for key in sorted(groups):
        members = groups[key]
        if len(members) < 2:
            continue
        original = members[0]
        for copy in members[1:]:
            verdicts.append(
                DuplicateVerdict(
                    rule=DuplicateRule.BYTE_IDENTICAL_DISTINCT_LINE,
                    category=VarianceCategory.DUPLICATE_CONFIRMED,
                    row_ids=(original.order_id,),
                    amount=copy.gross,
                    detail=(
                        f"ledger row {copy.order_id} is byte-identical to an earlier "
                        f"row on a distinct source line ({len(members)} copies total); "
                        "an ingestion artefact, not a second sale"
                    ),
                )
            )
    return tuple(verdicts)


def find_candidate_duplicates(
    rows: Sequence[LedgerRow],
    confirmed_order_ids: frozenset[str] = frozenset(),
) -> tuple[DuplicateVerdict, ...]:
    """P7b. Same (customer_id, gross), different order_id.

    NOT resolved. Both rows are attached so a reviewer - or M7 - sees the pair
    rather than a claim about it.

    `confirmed_order_ids` excludes rows already resolved by P7a. Without that
    exclusion a byte-identical pair would fire BOTH rules, since identical rows
    trivially share a customer and an amount, and the same rupees would be
    counted once as confirmed and once as a candidate.
    """
    groups: dict[tuple[str, str], list[LedgerRow]] = {}
    for row in rows:
        if row.order_id in confirmed_order_ids:
            continue
        groups.setdefault((row.customer_id, str(row.gross)), []).append(row)

    verdicts: list[DuplicateVerdict] = []
    for key in sorted(groups):
        members = groups[key]
        distinct_orders = sorted({row.order_id for row in members})
        if len(distinct_orders) < 2:
            continue
        verdicts.append(
            DuplicateVerdict(
                rule=DuplicateRule.SAME_CUSTOMER_SAME_GROSS,
                category=VarianceCategory.DUPLICATE_CANDIDATE,
                row_ids=tuple(distinct_orders),
                amount=members[0].gross,
                detail=(
                    f"customer {key[0]} has {len(distinct_orders)} orders at {key[1]}: "
                    f"{', '.join(distinct_orders)}. A data duplicate and a genuine "
                    "repeat purchase are indistinguishable here; left for the AI layer"
                ),
            )
        )
    return tuple(verdicts)


def classify(row: LedgerRow, others: Sequence[LedgerRow]) -> DuplicateVerdict:
    """Single-row convenience. Always names a rule, including when none fired.

    Returning NO_RULE_FIRED rather than None keeps every call site handling one
    shape, and keeps "we checked and it was clean" distinguishable from "we
    did not check".
    """
    identical = [other for other in others if _content_key(other) == _content_key(row)]
    if identical:
        return DuplicateVerdict(
            rule=DuplicateRule.BYTE_IDENTICAL_DISTINCT_LINE,
            category=VarianceCategory.DUPLICATE_CONFIRMED,
            row_ids=tuple(sorted({row.order_id, *(o.order_id for o in identical)})),
            amount=row.gross,
            detail=f"{len(identical)} byte-identical row(s) on distinct source lines",
        )
    similar = [
        other
        for other in others
        if other.customer_id == row.customer_id
        and other.gross == row.gross
        and other.order_id != row.order_id
    ]
    if similar:
        return DuplicateVerdict(
            rule=DuplicateRule.SAME_CUSTOMER_SAME_GROSS,
            category=VarianceCategory.DUPLICATE_CANDIDATE,
            row_ids=tuple(sorted({row.order_id, *(o.order_id for o in similar)})),
            amount=row.gross,
            detail="same customer and amount, different order id; interpretive",
        )
    return DuplicateVerdict(
        rule=DuplicateRule.NO_RULE_FIRED,
        category=None,
        row_ids=(row.order_id,),
        amount=None,
        detail="no duplicate rule matched",
    )

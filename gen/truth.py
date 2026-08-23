"""M1 part B - the ground-truth edge set, its self-check, and the truth writer.

INDEPENDENT PATH. Nothing here imports from settlesense/. The taxonomy and the
edge types are restated rather than shared, for the same reason the rates are.

Truth is a TYPED EDGE SET with per-type cardinality (SDD 3.2), not a flat
partner map. The earlier "no row may be the true partner of two rows" invariant
was wrong: one settlement batch legitimately contains hundreds of settlement
lines. Each edge type is therefore checked against its OWN declared cardinality
and nothing else.

There are FIVE edge types. SDD 3.2 declares exactly five. A sixth would have to
be a dispute edge, and chargebacks are out of v1 scope (SDD 3.0).

Nothing is written unless every assertion passes. A truth file produced from a
dataset that does not balance is worse than no truth file at all: it makes every
downstream metric confidently wrong.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from gen.lifecycle import (
    REQUIRED_YEAR,
    CleanDataset,
    SettlementLineType,
    WorkingCalendar,
    bank_txn_id_for,
    money,
)
from gen.profiles import MerchantProfile

if TYPE_CHECKING:  # noise.py imports VarianceCategory from here; keep it one-way
    from gen.noise import NoiseAnnotation, NoiseLedger

__all__ = [
    "EdgeType",
    "Truth",
    "TruthCase",
    "TruthEdge",
    "TruthSelfCheckError",
    "VarianceCategory",
    "Violation",
    "build_truth",
    "check_cardinality",
    "run_self_check",
    "write_truth",
]

ZERO: Final[Decimal] = Decimal("0.00")


class EdgeType(StrEnum):
    """The five truth relationships. Restated from SDD 3.2, not imported."""

    ORDER_TO_PAYMENT = "order_to_payment"
    PAYMENT_TO_SETTLEMENT = "payment_to_settlement"
    SETTLEMENT_TO_BATCH = "settlement_to_batch"
    BATCH_TO_BANK = "batch_to_bank"
    PAYMENT_TO_REFUND = "payment_to_refund"


class VarianceCategory(StrEnum):
    """The CLOSED variance taxonomy (PDD 6). Restated independently.

    6.1 deterministically derivable - never sent to the model.
    6.2 genuinely interpretive - eligible for the AI layer.
    """

    # --- 6.1 deterministic ---
    MDR_FEE = "MDR_FEE"
    GST_ON_FEE = "GST_ON_FEE"
    ROUNDING_DIFFERENCE = "ROUNDING_DIFFERENCE"
    DUPLICATE_CONFIRMED = "DUPLICATE_CONFIRMED"
    T_PLUS_N_TIMING = "T_PLUS_N_TIMING"
    REFUND_OFFSET = "REFUND_OFFSET"
    PARTIAL_CAPTURE = "PARTIAL_CAPTURE"
    # --- 6.2 interpretive ---
    UTR_TRUNCATED_MAPPING = "UTR_TRUNCATED_MAPPING"
    UTR_MISSING_MAPPING = "UTR_MISSING_MAPPING"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    SPLIT_SETTLEMENT = "SPLIT_SETTLEMENT"
    MISSING_VS_LATE_CREDIT = "MISSING_VS_LATE_CREDIT"
    UNEXPLAINED = "UNEXPLAINED"


@dataclass(frozen=True)
class TruthEdge:
    edge_type: EdgeType
    src_id: str
    dst_id: str


@dataclass(frozen=True)
class Violation:
    """One cardinality breach, named against the edge type it belongs to."""

    edge_type: EdgeType
    detail: str

    def __str__(self) -> str:
        return f"[{self.edge_type.value}] {self.detail}"


# When several noise types land on one target, the reported category is the most
# consequential, chosen by this fixed precedence so the choice is never
# iteration-order dependent (D4). Terminal states outrank recoverable ones.
CATEGORY_PRECEDENCE: Final[tuple[VarianceCategory, ...]] = (
    VarianceCategory.UNEXPLAINED,
    VarianceCategory.MISSING_VS_LATE_CREDIT,
    VarianceCategory.SPLIT_SETTLEMENT,
    VarianceCategory.UTR_MISSING_MAPPING,
    VarianceCategory.UTR_TRUNCATED_MAPPING,
    VarianceCategory.DUPLICATE_CANDIDATE,
    VarianceCategory.DUPLICATE_CONFIRMED,
    VarianceCategory.PARTIAL_CAPTURE,
    VarianceCategory.T_PLUS_N_TIMING,
    VarianceCategory.REFUND_OFFSET,
    VarianceCategory.ROUNDING_DIFFERENCE,
    VarianceCategory.GST_ON_FEE,
    VarianceCategory.MDR_FEE,
)


@dataclass(frozen=True)
class TruthBatchLink:
    """POPULATION B truth: one settlement batch and its bank credit.

    Kept in its own table with its own denominator. A batch-link failure reaches
    Population A only through the cases inside that batch, which the eval finds
    by joining case.batch_ids - never by merging these rows into case metrics
    (D11).
    """

    batch_id: str
    bank_txn_id: str | None  # None means the credit never arrived
    net_total: Decimal
    true_category: VarianceCategory | None
    resolvable_in_principle: bool
    noise_types: tuple[str, ...]


@dataclass(frozen=True)
class TruthCase:
    """Ground truth for one ReconciliationCase - one per captured payment.

    ALWAYS one case per payment. A split settlement produces ONE case holding
    multiple payment_line_ids, never two cases: two would double-count the
    denominator and make the match rate depend on how the gateway happened to
    batch the payout (SDD 3.1).

    `true_category` is null when there is nothing to explain - expected_net
    equals what actually landed, so the case yields no exception. It carries a
    taxonomy value only once a variance exists. That keeps the taxonomy closed
    rather than inventing a CLEAN member for it.

    `deduction_categories` is separate and is NOT the variance category: it
    records which deterministic components explain gross -> expected_net for
    this case. A clean card payment deducts MDR_FEE and GST_ON_FEE and still
    has no variance.
    """

    case_id: str
    payment_id: str
    order_id: str
    merchant_profile: str
    arrival_day: int
    expected_gross: Decimal
    expected_net: Decimal
    settlement_line_ids: tuple[str, ...]  # ALL lines touching this payment
    payment_line_ids: tuple[str, ...]  # PAYMENT lines only; len > 1 => split
    refund_ids: tuple[str, ...]
    batch_ids: tuple[str, ...]
    bank_txn_ids: tuple[str, ...]
    true_category: VarianceCategory | None
    true_variance_amount: Decimal
    resolvable_in_principle: bool
    noise_type: str | None  # the dominant one; see noise_types for all
    noise_types: tuple[str, ...]
    deduction_categories: tuple[VarianceCategory, ...]


@dataclass(frozen=True)
class Truth:
    edges: tuple[TruthEdge, ...]
    cases: tuple[TruthCase, ...]  # Population A
    batch_links: tuple[TruthBatchLink, ...]  # Population B - separate table
    calendar_version: str
    seed: int
    generator_commit: None  # literally null; see write_truth


class TruthSelfCheckError(AssertionError):
    """Raised when the dataset does not support its own truth. Nothing is written."""


def case_id_for(payment_id: str) -> str:
    """sha256(b"case|" + payment_id)[:16] - the formula fixed by SDD 3.1 (D10)."""
    return hashlib.sha256(b"case|" + payment_id.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Building the edge set
# ---------------------------------------------------------------------------


def dominant_category(
    annotations: Sequence[NoiseAnnotation],
) -> VarianceCategory | None:
    """The most consequential category among several, by fixed precedence (D4)."""
    present = {a.category for a in annotations if a.category is not None}
    for candidate in CATEGORY_PRECEDENCE:
        if candidate in present:
            return candidate
    return None


def build_truth(
    dataset: CleanDataset,
    calendar: WorkingCalendar,
    profiles: Mapping[str, MerchantProfile],
    seed: int,
    noise: NoiseLedger | None = None,
) -> Truth:
    """Derive the typed edge set and the per-case truth from a built dataset.

    With `noise`, each case and each batch link is annotated from what the
    injectors recorded doing - truth is derived from the ledger of actions, not
    reverse-engineered from the result.
    """
    by_case: Mapping[str, tuple[NoiseAnnotation, ...]] = noise.by_target("case") if noise else {}
    by_link: Mapping[str, tuple[NoiseAnnotation, ...]] = (
        noise.by_target("batch_link") if noise else {}
    )
    edges: list[TruthEdge] = []

    lines_by_payment: dict[str, list[Any]] = {}
    for line in dataset.settlement_lines:
        lines_by_payment.setdefault(line.payment_id, []).append(line)

    # Linked by construction (see bank_txn_id_for), never by reading narration.
    bank_by_txn = {row.bank_txn_id: row for row in dataset.bank_rows}
    bank_for_batch = {
        batch.batch_id: bank_by_txn[bank_txn_id_for(batch.batch_id)]
        for batch in dataset.batches
        if bank_txn_id_for(batch.batch_id) in bank_by_txn
    }

    # ORDER_TO_PAYMENT (1:1)
    for payment in dataset.payment_rows:
        edges.append(TruthEdge(EdgeType.ORDER_TO_PAYMENT, payment.order_id, payment.payment_id))

    # PAYMENT_TO_SETTLEMENT (1:N) - PAYMENT lines ONLY. Refund lines are reached
    # via PAYMENT_TO_REFUND, never this edge (SDD 3.2).
    for line in dataset.settlement_lines:
        if line.line_type is SettlementLineType.PAYMENT:
            edges.append(
                TruthEdge(EdgeType.PAYMENT_TO_SETTLEMENT, line.payment_id, line.settlement_id)
            )

    # SETTLEMENT_TO_BATCH (N:1) - EVERY v1 line, both types.
    for line in dataset.settlement_lines:
        edges.append(TruthEdge(EdgeType.SETTLEMENT_TO_BATCH, line.settlement_id, line.batch_id))

    # BATCH_TO_BANK (1:1)
    for batch in dataset.batches:
        row = bank_for_batch.get(batch.batch_id)
        if row is not None:
            edges.append(TruthEdge(EdgeType.BATCH_TO_BANK, batch.batch_id, row.bank_txn_id))

    # PAYMENT_TO_REFUND (1:N) - to the RefundRow, and separately to the REFUND
    # line it produced (SDD 3.2).
    for refund in dataset.refund_rows:
        edges.append(TruthEdge(EdgeType.PAYMENT_TO_REFUND, refund.payment_id, refund.refund_id))
    for line in dataset.settlement_lines:
        if line.line_type is SettlementLineType.REFUND and line.refund_id is not None:
            edges.append(TruthEdge(EdgeType.PAYMENT_TO_REFUND, line.payment_id, line.settlement_id))

    # --- per-case truth ----------------------------------------------------
    refunds_by_payment: dict[str, list[Any]] = {}
    for refund in dataset.refund_rows:
        refunds_by_payment.setdefault(refund.payment_id, []).append(refund)

    cases: list[TruthCase] = []
    for chain in dataset.chains:
        payment_id = chain.payment.payment_id
        touching = lines_by_payment.get(payment_id, [])
        payment_lines = [line for line in touching if line.line_type is SettlementLineType.PAYMENT]
        refunds = refunds_by_payment.get(payment_id, [])

        deductions: list[VarianceCategory] = []
        if chain.fee > ZERO:
            deductions.append(VarianceCategory.MDR_FEE)
        if chain.tax > ZERO:
            deductions.append(VarianceCategory.GST_ON_FEE)
        if refunds:
            deductions.append(VarianceCategory.REFUND_OFFSET)

        case_noise = by_case.get(payment_id, ())
        batch_ids = sorted({line.batch_id for line in touching})
        bank_ids = sorted(
            {
                bank_for_batch[batch_id].bank_txn_id
                for batch_id in batch_ids
                if batch_id in bank_for_batch
            }
        )

        cases.append(
            TruthCase(
                case_id=case_id_for(payment_id),
                payment_id=payment_id,
                order_id=chain.payment.order_id,
                merchant_profile=chain.profile_name,
                arrival_day=chain.day,
                expected_gross=chain.expected_gross,
                expected_net=chain.expected_net,
                settlement_line_ids=tuple(sorted(line.settlement_id for line in touching)),
                payment_line_ids=tuple(sorted(line.settlement_id for line in payment_lines)),
                refund_ids=tuple(sorted(refund.refund_id for refund in refunds)),
                batch_ids=tuple(batch_ids),
                bank_txn_ids=tuple(bank_ids),
                # A clean chain lands exactly what it expects, so there is no
                # variance and no category. Noise is what populates these.
                true_category=dominant_category(case_noise),
                true_variance_amount=ZERO,
                resolvable_in_principle=all(a.resolvable for a in case_noise),
                noise_type=(
                    sorted(case_noise, key=_noise_rank)[0].noise_type if case_noise else None
                ),
                noise_types=tuple(sorted({a.noise_type for a in case_noise})),
                deduction_categories=tuple(deductions),
            )
        )

    # Population B, its own table with its own denominator (D11).
    batch_links: list[TruthBatchLink] = []
    for batch in dataset.batches:
        link_noise = by_link.get(batch.batch_id, ())
        row = bank_for_batch.get(batch.batch_id)
        batch_links.append(
            TruthBatchLink(
                batch_id=batch.batch_id,
                bank_txn_id=row.bank_txn_id if row is not None else None,
                net_total=batch.net_total,
                true_category=dominant_category(link_noise),
                resolvable_in_principle=all(a.resolvable for a in link_noise),
                noise_types=tuple(sorted({a.noise_type for a in link_noise})),
            )
        )

    edges.sort(key=lambda edge: (edge.edge_type.value, edge.src_id, edge.dst_id))
    cases.sort(key=lambda case: case.case_id)
    batch_links.sort(key=lambda link: link.batch_id)
    return Truth(
        edges=tuple(edges),
        cases=tuple(cases),
        batch_links=tuple(batch_links),
        calendar_version=calendar.version,
        seed=seed,
        generator_commit=None,
    )


def _noise_rank(annotation: NoiseAnnotation) -> tuple[int, str]:
    order = CATEGORY_PRECEDENCE.index(annotation.category) if annotation.category else 99
    return (order, annotation.noise_type)


# ---------------------------------------------------------------------------
# Cardinality - each edge type against its OWN declared cardinality (SDD 3.2)
# ---------------------------------------------------------------------------


def check_cardinality(edges: Sequence[TruthEdge]) -> list[Violation]:
    """Validate each edge type against its own rule. Empty list means clean.

    There is deliberately NO blanket "no row may partner twice" rule. Many
    settlement lines share one batch, and a rule that forbade it would fail on
    correct data - the most dangerous kind of wrong check.
    """
    violations: list[Violation] = []
    by_type: dict[EdgeType, list[TruthEdge]] = {edge_type: [] for edge_type in EdgeType}
    for edge in edges:
        by_type[edge.edge_type].append(edge)

    for edge_type, group in sorted(by_type.items(), key=lambda item: item[0].value):
        seen: set[tuple[str, str]] = set()
        for edge in group:
            key = (edge.src_id, edge.dst_id)
            if key in seen:
                violations.append(
                    Violation(edge_type, f"duplicate edge {edge.src_id}->{edge.dst_id}")
                )
            seen.add(key)

    # 1. ORDER_TO_PAYMENT - 1:1 in BOTH directions.
    violations += _at_most_once(
        by_type[EdgeType.ORDER_TO_PAYMENT], EdgeType.ORDER_TO_PAYMENT, "src", "order"
    )
    violations += _at_most_once(
        by_type[EdgeType.ORDER_TO_PAYMENT], EdgeType.ORDER_TO_PAYMENT, "dst", "payment"
    )

    # 2. PAYMENT_TO_SETTLEMENT - 1:N. A payment may have many PAYMENT lines
    #    (N >= 2 is the structural condition for SPLIT_SETTLEMENT), but a line
    #    belongs to exactly one payment, so dst must not repeat.
    violations += _at_most_once(
        by_type[EdgeType.PAYMENT_TO_SETTLEMENT],
        EdgeType.PAYMENT_TO_SETTLEMENT,
        "dst",
        "settlement line",
    )

    # 3. SETTLEMENT_TO_BATCH - N:1. MANY LINES PER BATCH IS NORMAL. The only
    #    thing to assert is that each line sits in exactly one batch.
    violations += _at_most_once(
        by_type[EdgeType.SETTLEMENT_TO_BATCH],
        EdgeType.SETTLEMENT_TO_BATCH,
        "src",
        "settlement line",
    )

    # 4. BATCH_TO_BANK - 1:1. A batch with two bank credits IS a generator bug.
    violations += _at_most_once(
        by_type[EdgeType.BATCH_TO_BANK], EdgeType.BATCH_TO_BANK, "src", "batch"
    )
    violations += _at_most_once(
        by_type[EdgeType.BATCH_TO_BANK], EdgeType.BATCH_TO_BANK, "dst", "bank credit"
    )

    # 5. PAYMENT_TO_REFUND - 1:N with NO upper bound. Partial and multiple
    #    refunds are legal. Only the dst side is constrained.
    violations += _at_most_once(
        by_type[EdgeType.PAYMENT_TO_REFUND],
        EdgeType.PAYMENT_TO_REFUND,
        "dst",
        "refund artifact",
    )
    return violations


def _at_most_once(
    edges: Sequence[TruthEdge], edge_type: EdgeType, side: str, label: str
) -> list[Violation]:
    counts: dict[str, int] = {}
    for edge in edges:
        key = edge.src_id if side == "src" else edge.dst_id
        counts[key] = counts.get(key, 0) + 1
    offenders = sorted((key, n) for key, n in counts.items() if n > 1)
    if not offenders:
        return []
    shown = ", ".join(f"{key} x{n}" for key, n in offenders[:5])
    return [
        Violation(
            edge_type,
            f"{len(offenders)} {label}(s) appear more than once on the {side} side: {shown}",
        )
    ]


# ---------------------------------------------------------------------------
# The self-check. Runs BEFORE anything is written.
# ---------------------------------------------------------------------------


def run_self_check(
    dataset: CleanDataset,
    truth: Truth,
    *,
    calendar: WorkingCalendar,
    profiles: Mapping[str, MerchantProfile],
    noise: NoiseLedger | None = None,
) -> None:
    """Raise TruthSelfCheckError unless every assertion holds.

    Collects ALL failures rather than stopping at the first, so one run tells
    you everything that is wrong.

    With `noise`, the check is not RELAXED - it is EXTENDED. Every assertion
    still holds for every un-noised chain; the injected deviations must each be
    claimed by an annotation, and every annotation must correspond to a
    deviation that is actually present. Accounting runs both ways, so noise can
    neither hide a generator bug nor be claimed without having happened.
    """
    problems: list[str] = []
    orphan_credits = noise.orphan_bank_txn_ids() if noise else frozenset()
    unbanked = noise.unbanked_batch_ids() if noise else frozenset()
    dup_confirmed = noise.duplicate_confirmed_order_ids() if noise else frozenset()
    split_payments = noise.split_payment_ids() if noise else frozenset()

    lines_by_batch: dict[str, list[Any]] = {}
    for line in dataset.settlement_lines:
        lines_by_batch.setdefault(line.batch_id, []).append(line)
    batch_by_id = {batch.batch_id: batch for batch in dataset.batches}
    payment_by_id = {row.payment_id: row for row in dataset.payment_rows}
    refund_by_id = {row.refund_id: row for row in dataset.refund_rows}
    profile_of_payment = {chain.payment.payment_id: chain.profile_name for chain in dataset.chains}

    # ---- 1-5. cardinality, per type ---------------------------------------
    problems += [f"cardinality: {violation}" for violation in check_cardinality(truth.edges)]

    # 2b. N >= 2 PAYMENT lines is the structural condition for SPLIT_SETTLEMENT.
    #     Clean chains must never exhibit it - split settlement is WITHHELD noise.
    split_cases = [case for case in truth.cases if len(case.payment_line_ids) >= 2]
    for case in split_cases:
        if case.true_category is not VarianceCategory.SPLIT_SETTLEMENT:
            problems.append(
                f"case {case.case_id}: {len(case.payment_line_ids)} PAYMENT lines but "
                f"category is {case.true_category}; N>=2 marks SPLIT_SETTLEMENT"
            )
    if noise is None and split_cases:
        problems.append(
            f"{len(split_cases)} case(s) have N>=2 PAYMENT lines with no noise applied; "
            f"split settlement is a WITHHELD noise type and cannot occur in clean chains"
        )

    # ---- 6. every clean chain balances to the cent ------------------------
    for chain in dataset.chains:
        want = money(chain.expected_gross - chain.fee - chain.tax - chain.refund_amount)
        if chain.expected_net != want:
            problems.append(
                f"chain {chain.payment.payment_id}: expected_net {chain.expected_net} != "
                f"gross {chain.expected_gross} - fee {chain.fee} - tax {chain.tax} "
                f"- refunds {chain.refund_amount} = {want}"
            )

    # ---- 7. batch net_total == SIGNED sum across ALL line types -----------
    mixed_batches = 0
    for batch in dataset.batches:
        members = lines_by_batch.get(batch.batch_id, [])
        if not members:
            problems.append(f"batch {batch.batch_id}: no member lines")
            continue
        kinds = {line.line_type for line in members}
        if kinds == {SettlementLineType.PAYMENT, SettlementLineType.REFUND}:
            mixed_batches += 1
        summed = money(sum((line.net for line in members), start=ZERO))
        if batch.net_total != summed:
            problems.append(
                f"batch {batch.batch_id}: net_total {batch.net_total} != signed line sum "
                f"{summed} over {len(members)} lines"
            )

    # The regression case for signed-line arithmetic. If no batch mixes a
    # PAYMENT line with a REFUND line, assertion 7 has proved nothing: it would
    # pass on a dataset where refunds never touch a batch at all.
    if mixed_batches == 0:
        problems.append(
            "no batch contains both a PAYMENT and a REFUND line, so the signed-line "
            "arithmetic is never exercised. Assertion 7 would pass vacuously."
        )

    # ---- 7a. bank credit == batch net_total, refund batches included ------
    bank_by_txn = {row.bank_txn_id: row for row in dataset.bank_rows}
    linked = [edge for edge in truth.edges if edge.edge_type is EdgeType.BATCH_TO_BANK]
    expected_links = len(dataset.batches) - len(unbanked)
    if len(linked) != expected_links:
        problems.append(
            f"{len(linked)} BATCH_TO_BANK edges for {len(dataset.batches)} batches minus "
            f"{len(unbanked)} withheld credit(s) = {expected_links} expected; 1:1"
        )
    mixed_checked = 0
    for edge in linked:
        linked_batch = batch_by_id.get(edge.src_id)
        linked_row = bank_by_txn.get(edge.dst_id)
        if linked_batch is None or linked_row is None:
            problems.append(f"BATCH_TO_BANK {edge.src_id}->{edge.dst_id}: endpoint missing")
            continue
        if linked_row.amount != linked_batch.net_total:
            problems.append(
                f"bank {linked_row.bank_txn_id} credits {linked_row.amount} but batch "
                f"{linked_batch.batch_id} nets {linked_batch.net_total}"
            )
        members = lines_by_batch.get(linked_batch.batch_id, [])
        if any(line.line_type is SettlementLineType.REFUND for line in members):
            mixed_checked += 1
    if mixed_checked == 0:
        problems.append("assertion 7a checked no batch containing a refund line")

    # ---- 7b. per-type field invariants (SDD 3.3 table), BOTH shapes -------
    shapes_seen: set[SettlementLineType] = set()
    for line in dataset.settlement_lines:
        shapes_seen.add(line.line_type)
        if line.line_type is SettlementLineType.PAYMENT:
            if line.refund_id is not None:
                problems.append(f"line {line.settlement_id}: PAYMENT must have refund_id None")
            if line.net != money(line.gross - line.fee - line.tax):
                problems.append(f"line {line.settlement_id}: PAYMENT net != +(gross-fee-tax)")
            if line.net <= ZERO:
                problems.append(f"line {line.settlement_id}: PAYMENT net must be positive")
        else:
            if line.refund_id is None:
                problems.append(f"line {line.settlement_id}: REFUND must set refund_id")
            elif line.refund_id not in refund_by_id:
                problems.append(f"line {line.settlement_id}: refund_id not in refunds")
            elif line.net != money(-refund_by_id[line.refund_id].amount):
                problems.append(f"line {line.settlement_id}: REFUND net != -refund.amount")
            if (line.gross, line.fee, line.tax) != (ZERO, ZERO, ZERO):
                problems.append(f"line {line.settlement_id}: REFUND gross/fee/tax must be 0")
    if shapes_seen != {SettlementLineType.PAYMENT, SettlementLineType.REFUND}:
        problems.append(f"assertion 7b saw only {sorted(s.value for s in shapes_seen)}")

    # ---- 7c. PAYMENT_TO_SETTLEMENT points ONLY at PAYMENT lines -----------
    line_type_by_id = {line.settlement_id: line.line_type for line in dataset.settlement_lines}
    wrong = [
        edge
        for edge in truth.edges
        if edge.edge_type is EdgeType.PAYMENT_TO_SETTLEMENT
        and line_type_by_id.get(edge.dst_id) is not SettlementLineType.PAYMENT
    ]
    if wrong:
        problems.append(
            f"{len(wrong)} PAYMENT_TO_SETTLEMENT edge(s) reference a non-PAYMENT line, "
            f"e.g. {wrong[0].src_id}->{wrong[0].dst_id}. A refund line is not a "
            f"settlement of the payment."
        )

    # ---- 8. every truth ID exists in the dataset --------------------------
    known: dict[str, set[str]] = {
        "order": {row.order_id for row in dataset.ledger_rows},
        "payment": set(payment_by_id),
        "refund": set(refund_by_id),
        "line": set(line_type_by_id),
        "batch": set(batch_by_id),
        "bank": set(bank_by_txn),
    }
    endpoints: Mapping[EdgeType, tuple[str, tuple[str, ...]]] = {
        EdgeType.ORDER_TO_PAYMENT: ("order", ("payment",)),
        EdgeType.PAYMENT_TO_SETTLEMENT: ("payment", ("line",)),
        EdgeType.SETTLEMENT_TO_BATCH: ("line", ("batch",)),
        EdgeType.BATCH_TO_BANK: ("batch", ("bank",)),
        # dst is a RefundRow OR the REFUND line it produced (SDD 3.2).
        EdgeType.PAYMENT_TO_REFUND: ("payment", ("refund", "line")),
    }
    for edge in truth.edges:
        src_kind, dst_kinds = endpoints[edge.edge_type]
        if edge.src_id not in known[src_kind]:
            problems.append(
                f"edge {edge.edge_type.value}: src {edge.src_id} is not a known {src_kind}"
            )
        if not any(edge.dst_id in known[kind] for kind in dst_kinds):
            problems.append(
                f"edge {edge.edge_type.value}: dst {edge.dst_id} is not a known "
                f"{' or '.join(dst_kinds)}"
            )
    for case in truth.cases:
        if case.payment_id not in known["payment"]:
            problems.append(f"case {case.case_id}: unknown payment {case.payment_id}")
        if case.order_id not in known["order"]:
            problems.append(f"case {case.case_id}: unknown order {case.order_id}")
        for line_id in case.settlement_line_ids:
            if line_id not in known["line"]:
                problems.append(f"case {case.case_id}: unknown settlement line {line_id}")
        for batch_id in case.batch_ids:
            if batch_id not in known["batch"]:
                problems.append(f"case {case.case_id}: unknown batch {batch_id}")
        for bank_id in case.bank_txn_ids:
            if bank_id not in known["bank"]:
                problems.append(f"case {case.case_id}: unknown bank credit {bank_id}")
        if case.case_id != case_id_for(case.payment_id):
            problems.append(f"case {case.case_id}: case_id is not sha256('case|'+payment_id)[:16]")

    # One case per captured payment. Always one.
    if len(truth.cases) != len(dataset.payment_rows):
        problems.append(
            f"{len(truth.cases)} cases for {len(dataset.payment_rows)} payments; "
            f"Population A is one case per captured payment"
        )
    if len({case.case_id for case in truth.cases}) != len(truth.cases):
        problems.append("duplicate case_id in truth")

    # ---- 9. fee recomputed independently from the profile rate ------------
    for line in dataset.settlement_lines:
        if line.line_type is not SettlementLineType.PAYMENT:
            continue
        profile_name = profile_of_payment.get(line.payment_id)
        payment = payment_by_id.get(line.payment_id)
        if profile_name is None or payment is None:
            continue
        profile = profiles[profile_name]
        want_fee = money(line.gross * profile.rate_for(payment.method))
        want_tax = money(want_fee * profile.gst_rate)
        if line.fee != want_fee:
            problems.append(
                f"line {line.settlement_id}: fee {line.fee} != {want_fee} "
                f"({profile_name}/{payment.method} on gross {line.gross})"
            )
        if line.tax != want_tax:
            problems.append(f"line {line.settlement_id}: tax {line.tax} != {want_tax}")
        if payment.method == "upi" and line.fee != ZERO:
            problems.append(f"line {line.settlement_id}: UPI must be zero-rated")

    # ---- 10. global conservation, BOTH identities -------------------------
    sum_gross = money(sum((case.expected_gross for case in truth.cases), start=ZERO))
    sum_net = money(sum((case.expected_net for case in truth.cases), start=ZERO))
    sum_fee = money(sum((chain.fee for chain in dataset.chains), start=ZERO))
    sum_tax = money(sum((chain.tax for chain in dataset.chains), start=ZERO))
    sum_refunds = money(sum((row.amount for row in dataset.refund_rows), start=ZERO))
    if sum_gross != money(sum_net + sum_fee + sum_tax + sum_refunds):
        problems.append(
            f"case-side conservation: gross {sum_gross} != net {sum_net} + fee {sum_fee} "
            f"+ tax {sum_tax} + refunds {sum_refunds}"
        )

    sum_bank = money(sum((row.amount for row in dataset.bank_rows), start=ZERO))
    sum_batch = money(sum((batch.net_total for batch in dataset.batches), start=ZERO))
    sum_lines = money(sum((line.net for line in dataset.settlement_lines), start=ZERO))
    if sum_batch != sum_lines:
        problems.append(f"batch-side conservation: batches {sum_batch} != signed lines {sum_lines}")
    # Only `unexplainable` may move this identity, and only by an amount the
    # ledger states in advance. Anything else is a generator bug wearing noise
    # as a disguise.
    orphan_total = money(
        sum((r.amount for r in dataset.bank_rows if r.bank_txn_id in orphan_credits), start=ZERO)
    )
    missing_total = money(
        sum((b.net_total for b in dataset.batches if b.batch_id in unbanked), start=ZERO)
    )
    predicted_bank = money(sum_batch + orphan_total - missing_total)
    if sum_bank != predicted_bank:
        problems.append(
            f"batch-side conservation: bank {sum_bank} != batches {sum_batch} "
            f"+ orphan credits {orphan_total} - withheld credits {missing_total} "
            f"= {predicted_bank}"
        )
    if sum_net != sum_lines:
        problems.append(f"the two sides disagree: case net {sum_net} != line total {sum_lines}")

    # ---- noise accounting, BOTH directions --------------------------------
    problems += _account_for_noise(
        dataset, truth, orphan_credits, unbanked, dup_confirmed, split_payments, noise
    )

    # ---- 11. every date value is in 2026 ----------------------------------
    for label, value in _every_date(dataset):
        if value.year != REQUIRED_YEAR:
            problems.append(f"D13: {label} is {value.isoformat()}")

    # ---- 12. arrival_day is a positive int, never a date; no wall clock ---
    # Widened to `object` deliberately. arrival_day is ANNOTATED int, so mypy
    # proves these branches unreachable and would delete the check - but the
    # annotation is exactly what this assertion exists to verify at runtime,
    # against data that may have been built by something that ignored it.
    for case in truth.cases:
        arrival: object = case.arrival_day
        if isinstance(arrival, (date, datetime)):
            problems.append(
                f"case {case.case_id}: arrival_day is a {type(arrival).__name__}, must be an "
                f"int day index. arrival_day is a sequence position, not a point in time."
            )
        elif isinstance(arrival, bool) or not isinstance(arrival, int):
            problems.append(
                f"case {case.case_id}: arrival_day is {type(arrival).__name__}, must be int"
            )
        elif arrival < 1:
            problems.append(f"case {case.case_id}: arrival_day {arrival} is not 1-indexed")

    # ---- 13. settlement respects the profile's T+N against the calendar ---
    for batch in dataset.batches:
        if not calendar.is_working_day(batch.settled_event_date):
            problems.append(
                f"batch {batch.batch_id}: settles {batch.settled_event_date}, not a working day"
            )
    for edge in linked:
        tn_batch = batch_by_id.get(edge.src_id)
        tn_row = bank_by_txn.get(edge.dst_id)
        if tn_batch is None or tn_row is None:
            continue
        members = lines_by_batch.get(tn_batch.batch_id, [])
        names = {
            name
            for line in members
            if (name := profile_of_payment.get(line.payment_id)) is not None
        }
        if len(names) != 1:
            problems.append(f"batch {tn_batch.batch_id}: mixes profiles {sorted(names)}")
            continue
        profile = profiles[next(iter(names))]
        expected_value_date = calendar.add_working_days(
            tn_batch.settled_event_date, profile.settlement_cycle_days
        )
        if tn_row.value_date != expected_value_date:
            problems.append(
                f"bank {tn_row.bank_txn_id}: value_date {tn_row.value_date} != "
                f"{expected_value_date} (batch {tn_batch.settled_event_date} + "
                f"T+{profile.settlement_cycle_days} working days, {calendar.version})"
            )

    if problems:
        raise TruthSelfCheckError(
            f"ground-truth self-check FAILED with {len(problems)} violation(s); "
            f"nothing was written.\n  " + "\n  ".join(sorted(problems)[:50])
        )


def _every_date(dataset: CleanDataset) -> list[tuple[str, date]]:
    dates: list[tuple[str, date]] = []
    dates += [(f"{r.order_id}.order_date", r.order_date) for r in dataset.ledger_rows]
    dates += [(f"{r.payment_id}.captured_at", r.captured_at) for r in dataset.payment_rows]
    dates += [(f"{r.refund_id}.created_at", r.created_at) for r in dataset.refund_rows]
    dates += [
        (f"{line.settlement_id}.settled_event_date", line.settled_event_date)
        for line in dataset.settlement_lines
    ]
    dates += [(f"{b.batch_id}.settled_event_date", b.settled_event_date) for b in dataset.batches]
    dates += [(f"{r.bank_txn_id}.value_date", r.value_date) for r in dataset.bank_rows]
    return dates


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _money_str(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}"


def truth_to_dict(truth: Truth) -> dict[str, Any]:
    """Canonical, JSON-safe form. Decimals become strings; no float appears."""
    return {
        # Literally null. The freeze commit cannot exist before the commit that
        # creates it, and truth files are never rewritten afterwards. The real
        # hash is published once, in GENERATOR_MANIFEST.json, at M1F.
        "generator_commit": None,
        "calendar_version": truth.calendar_version,
        "seed": truth.seed,
        "edge_types": [edge_type.value for edge_type in EdgeType],
        "counts": {
            "edges": len(truth.edges),
            "cases": len(truth.cases),
            "batch_links": len(truth.batch_links),
            **{
                edge_type.value: sum(1 for e in truth.edges if e.edge_type is edge_type)
                for edge_type in EdgeType
            },
        },
        "edges": [
            {"edge_type": edge.edge_type.value, "src_id": edge.src_id, "dst_id": edge.dst_id}
            for edge in truth.edges
        ],
        # POPULATION B. Its own table, its own denominator. Never merged into
        # the case metrics above (D11).
        "batch_links": [
            {
                "batch_id": link.batch_id,
                "bank_txn_id": link.bank_txn_id,
                "net_total": _money_str(link.net_total),
                "true_category": link.true_category.value if link.true_category else None,
                "resolvable_in_principle": link.resolvable_in_principle,
                "noise_types": list(link.noise_types),
            }
            for link in truth.batch_links
        ],
        "cases": [
            {
                "case_id": case.case_id,
                "payment_id": case.payment_id,
                "order_id": case.order_id,
                "merchant_profile": case.merchant_profile,
                "arrival_day": case.arrival_day,
                "expected_gross": _money_str(case.expected_gross),
                "expected_net": _money_str(case.expected_net),
                "settlement_line_ids": list(case.settlement_line_ids),
                "payment_line_ids": list(case.payment_line_ids),
                "refund_ids": list(case.refund_ids),
                "batch_ids": list(case.batch_ids),
                "bank_txn_ids": list(case.bank_txn_ids),
                "true_category": case.true_category.value if case.true_category else None,
                "true_variance_amount": _money_str(case.true_variance_amount),
                "resolvable_in_principle": case.resolvable_in_principle,
                "noise_type": case.noise_type,
                "noise_types": list(case.noise_types),
                "deduction_categories": [c.value for c in case.deduction_categories],
            }
            for case in truth.cases
        ],
    }


def write_truth(truth: Truth, path: Path) -> None:
    """Serialize truth to `path`. Call run_self_check first - this does not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = truth_to_dict(truth)
    _reject_wall_clock(payload, "truth")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reject_wall_clock(node: Any, path: str) -> None:
    """No float, datetime or timestamp may reach the truth file (D2, D1)."""
    if isinstance(node, bool):
        return
    if isinstance(node, float):
        raise TruthSelfCheckError(
            f"{path}: float {node!r} in truth output; money is Decimal-as-string"
        )
    if isinstance(node, (datetime, date)):
        raise TruthSelfCheckError(f"{path}: raw date/datetime object in truth output")
    if isinstance(node, dict):
        for key, value in node.items():
            _reject_wall_clock(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _reject_wall_clock(value, f"{path}[{index}]")


def _account_for_noise(
    dataset: CleanDataset,
    truth: Truth,
    orphan_credits: frozenset[str],
    unbanked: frozenset[str],
    dup_confirmed: frozenset[str],
    split_payments: frozenset[str],
    noise: NoiseLedger | None,
) -> list[str]:
    """Every injected error is claimed, and every claim actually happened.

    One direction alone is not enough. Checking only that annotations are real
    would let an unannotated generator bug pass as noise; checking only that
    deviations are annotated would let the ledger claim errors it never made.
    """
    problems: list[str] = []
    expected_txn = {bank_txn_id_for(b.batch_id): b.batch_id for b in dataset.batches}

    # --- orphan bank credits ------------------------------------------------
    observed_orphans = {
        row.bank_txn_id for row in dataset.bank_rows if row.bank_txn_id not in expected_txn
    }
    for txn_id in sorted(observed_orphans - orphan_credits):
        problems.append(
            f"bank {txn_id}: credit matches no batch and no annotation claims it. "
            f"An unclaimed orphan credit is a generator bug, not noise."
        )
    for txn_id in sorted(orphan_credits - observed_orphans):
        problems.append(f"noise claims orphan credit {txn_id}, but it is not in the dataset")

    # --- batches whose credit never arrived ----------------------------------
    credited = {
        expected_txn[row.bank_txn_id]
        for row in dataset.bank_rows
        if row.bank_txn_id in expected_txn
    }
    observed_unbanked = {b.batch_id for b in dataset.batches} - credited
    for batch_id in sorted(observed_unbanked - unbanked):
        problems.append(
            f"batch {batch_id}: no bank credit and no annotation claims it. Every batch "
            f"is credited unless `unexplainable` withheld it."
        )
    for batch_id in sorted(unbanked - observed_unbanked):
        problems.append(f"noise claims batch {batch_id} was never credited, but a credit exists")

    # --- duplicate ledger rows ------------------------------------------------
    seen: dict[str, int] = {}
    for row in dataset.ledger_rows:
        seen[row.order_id] = seen.get(row.order_id, 0) + 1
    observed_dupes = {order_id for order_id, n in seen.items() if n > 1}
    for order_id in sorted(observed_dupes - dup_confirmed):
        problems.append(
            f"order {order_id} appears {seen[order_id]}x in the ledger with no "
            f"DUPLICATE_CONFIRMED annotation"
        )
    for order_id in sorted(dup_confirmed - observed_dupes):
        problems.append(f"noise claims order {order_id} is duplicated, but it appears once")

    # --- split settlements ----------------------------------------------------
    observed_splits = {case.payment_id for case in truth.cases if len(case.payment_line_ids) >= 2}
    for payment_id in sorted(observed_splits - split_payments):
        problems.append(f"payment {payment_id}: N>=2 PAYMENT lines with no split annotation")
    for payment_id in sorted(split_payments - observed_splits):
        problems.append(f"noise claims payment {payment_id} was split, but it has one PAYMENT line")

    # --- partial captures ------------------------------------------------------
    partial_claimed = (
        {a.target_id for a in noise.of_type("partial_captures")} if noise else frozenset()
    )
    observed_partial = {
        row.payment_id for row in dataset.payment_rows if row.captured < row.authorized
    }
    for payment_id in sorted(observed_partial - partial_claimed):
        problems.append(
            f"payment {payment_id}: captured < authorised with no PARTIAL_CAPTURE annotation"
        )
    for payment_id in sorted(partial_claimed - observed_partial):
        problems.append(
            f"noise claims payment {payment_id} was partially captured, but captured == authorised"
        )

    # --- every annotation points at something that exists ----------------------
    if noise is not None:
        known_cases = {case.payment_id for case in truth.cases}
        known_links = {link.batch_id for link in truth.batch_links}
        known_rows = (
            {r.order_id for r in dataset.ledger_rows}
            | {r.bank_txn_id for r in dataset.bank_rows}
            | {r.refund_id for r in dataset.refund_rows}
            | {line.settlement_id for line in dataset.settlement_lines}
            | {r.payment_id for r in dataset.payment_rows}
        )
        for annotation in noise.annotations:
            pool = {
                "case": known_cases,
                "batch_link": known_links,
                "bank_row": {r.bank_txn_id for r in dataset.bank_rows},
                "ledger_row": known_rows,
            }.get(annotation.target_kind)
            if pool is None:
                problems.append(
                    f"annotation {annotation.noise_type}: unknown target_kind "
                    f"{annotation.target_kind!r}"
                )
            elif annotation.target_id not in pool:
                problems.append(
                    f"annotation {annotation.noise_type} targets {annotation.target_id}, "
                    f"which is not a known {annotation.target_kind}"
                )
    return problems

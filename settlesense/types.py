"""M2 - Money, domain records, the reconciliation unit, the exception model.

Every dataclass here is frozen, and every field annotated `Money` is routed
through `money()` at construction. That routing is driven by the ANNOTATION,
not by a hand-maintained register of money fields: declaring a field `Money`
is what enrols it. A register would drift the first time a field was added
and not listed, and the failure mode - an unquantized Decimal or a float
sitting in a record that everything downstream trusts - is silent.

WHY CONSTRUCTION AND NOT A VALIDATOR CALLED LATER. A validator is a thing
someone has to remember to call. `__post_init__` is not skippable, so
"every money field goes through money()" is a property of the type rather
than a property of the call sites. `test_no_money_field_escapes_quantization`
proves it by walking the annotations rather than trusting this docstring.

SDD 3.1, 3.3, 8.1. Chargebacks are out of v1 scope (SDD 3.0): there is no
DISPUTE line type and no dispute row here, deliberately.

This module NEVER imports settlesense/core/telemetry.py. Wall-clock data does
not enter the business result (SDD 8.1, D6).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, DecimalException
from enum import StrEnum
from typing import Any, Final

__all__ = [
    "CENTS",
    "AuditActor",
    "AuditEntry",
    "BankRow",
    "BatchLinkOutcome",
    "CaseOutcome",
    "ExceptionStatus",
    "Exception_",
    "LedgerRow",
    "Money",
    "PaymentMethod",
    "PaymentRow",
    "PaymentStatus",
    "ReconciliationCase",
    "ReconciliationResult",
    "RefundRow",
    "RowVarianceOutcome",
    "SettlementBatch",
    "SettlementLine",
    "SettlementLineType",
    "money",
]

Money = Decimal
"""Always quantized to CENTS, ROUND_HALF_UP. Never a float (D1)."""

CENTS: Final[Decimal] = Decimal("0.01")
"""The money exponent. One paise. Every Money value carries exactly this scale."""


def money(value: Decimal | int | str) -> Money:
    """Quantize to paise, ROUND_HALF_UP. A float raises TypeError.

    Rejecting float is the whole point (D1). Binary floating point cannot
    represent 0.1, so a float that reached a money field would make totals
    depend on summation order, and the conservation assertions this project
    rests on would fail for reasons unrelated to reconciliation.

    `bool` is rejected explicitly and BEFORE `int`, because bool subclasses
    int: without the guard `money(True)` quietly returns Decimal("1.00").

    str is accepted so a value can be built from config or a literal without a
    Decimal(...) dance. It does NOT make this a CSV parser - external text goes
    through normalize.parse_amount, which knows about currency symbols,
    thousands separators and parenthesised debits.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"money() refuses bool ({value!r}). bool subclasses int, so this would "
            "otherwise silently become a rupee amount."
        )
    if isinstance(value, float):
        raise TypeError(
            f"money() refuses float ({value!r}). Money is Decimal (D1): binary "
            "floating point cannot represent 0.1 exactly, which makes totals "
            "depend on summation order. Use Decimal(str(x)) at the boundary, or "
            "normalize.parse_amount for external text."
        )
    if isinstance(value, Decimal):
        amount = value
    elif isinstance(value, int):
        amount = Decimal(value)
    elif isinstance(value, str):
        try:
            amount = Decimal(value)
        except DecimalException as exc:
            raise ValueError(f"money() could not parse {value!r} as a Decimal") from exc
    else:
        raise TypeError(
            f"money() accepts Decimal, int or str, got {type(value).__name__} ({value!r})"
        )
    if not amount.is_finite():
        raise ValueError(
            f"money() refuses the non-finite value {value!r}. NaN and Infinity "
            "propagate through every subsequent sum and make a conservation "
            "check pass or fail for reasons that have nothing to do with money."
        )
    return amount.quantize(CENTS, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Annotation-driven money enforcement
# ---------------------------------------------------------------------------

_MONEY_ANNOTATIONS: Final[frozenset[str]] = frozenset({"Money", "Money | None"})
"""Field annotations that enrol a field in quantization.

Matched as TEXT because `from __future__ import annotations` makes every
annotation a string. That is a feature here: `Money` and `Decimal` are the
same runtime object, so a resolved annotation could not tell them apart, and
the distinction is meaningful. `Money` means rupees and is quantized to paise.
`Decimal` means a ratio - a confidence, a rate - and must NOT be forced to two
decimal places just because it happens to share a type.
"""

_DECIMAL_ANNOTATIONS: Final[frozenset[str]] = frozenset({"Decimal", "Decimal | None"})
"""Non-money Decimals. Not quantized, but still refused if handed a float."""

_field_cache: dict[type, tuple[tuple[str, bool], ...]] = {}


def _decimal_fields(cls: type) -> tuple[tuple[str, bool], ...]:
    """(field_name, is_money) for every Decimal-ish field. Cached per class.

    Cached because __post_init__ runs on every row of every table - 16k+
    records for a full dataset - and re-walking dataclass metadata each time
    turns record construction into the slowest stage of ingestion.
    """
    cached = _field_cache.get(cls)
    if cached is not None:
        return cached
    found: list[tuple[str, bool]] = []
    for spec in fields(cls):
        annotation = spec.type if isinstance(spec.type, str) else getattr(spec.type, "__name__", "")
        text = annotation.replace("Optional[", "").replace("]", "").strip()
        if text in _MONEY_ANNOTATIONS:
            found.append((spec.name, True))
        elif text in _DECIMAL_ANNOTATIONS:
            found.append((spec.name, False))
    resolved = tuple(found)
    _field_cache[cls] = resolved
    return resolved


class _Record:
    """Base for every frozen domain record. Not itself a dataclass.

    Subclasses are frozen, so __post_init__ writes through object.__setattr__ -
    the standard and only way a frozen dataclass can normalize its own fields.
    """

    __slots__ = ()

    def __post_init__(self) -> None:
        for name, is_money in _decimal_fields(type(self)):
            value = getattr(self, name)
            if value is None:
                continue
            if is_money:
                object.__setattr__(self, name, money(value))
            elif isinstance(value, float | bool):
                raise TypeError(
                    f"{type(self).__name__}.{name} was given "
                    f"{type(value).__name__} ({value!r}); it must be Decimal (D1)."
                )


# ---------------------------------------------------------------------------
# Closed vocabularies (SDD 3.3)
# ---------------------------------------------------------------------------


class PaymentMethod(StrEnum):
    """SDD 3.3 PaymentRow.method. Priced by config/mdr_rates.yaml, per profile."""

    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class PaymentStatus(StrEnum):
    """SDD 3.3 PaymentRow.status."""

    CAPTURED = "captured"
    REFUNDED = "refunded"
    PARTIAL = "partial"
    FAILED = "failed"


class SettlementLineType(StrEnum):
    """SDD 3.3. A batch holds lines of MIXED type; only PAYMENT lines settle
    a payment. A REFUND line references a payment without being a settlement
    of it, which is why PAYMENT_TO_SETTLEMENT links PAYMENT lines only."""

    PAYMENT = "payment"  # credit: a captured payment settling
    REFUND = "refund"  # debit:  a refund deducted from the batch


class BankDirection(StrEnum):
    """SDD 3.3 BankRow.direction."""

    CREDIT = "credit"
    DEBIT = "debit"


class ExceptionStatus(StrEnum):
    """SDD 3 'Exception lifecycle'. Exactly the six states in the normative table.

    CONFIRMED means EXPLAINED. CLOSED means ACTIONED. They were previously used
    interchangeably; they are distinct, and CLOSED has exactly one legal
    predecessor.

    HUMAN_REVIEW appears in the SDD's lifecycle DIAGRAM but not in its state
    table, and it is not a member here. Reading it as a state would give
    ABSTAINED two outgoing edges to CONFIRMED and make the abstention rate
    ambiguous; reading it as the M8 evidence QUEUE - a place abstained
    exceptions wait, not a status they hold - keeps the six-state table intact.
    Recorded as a spec ambiguity; the transition table lands with the store at
    M6 and is where this reading gets tested rather than asserted.
    """

    OPEN = "OPEN"
    PENDING_EVIDENCE = "PENDING_EVIDENCE"
    PENDING_AI_UNAVAILABLE = "PENDING_AI_UNAVAILABLE"
    CONFIRMED = "CONFIRMED"
    ABSTAINED = "ABSTAINED"
    CLOSED = "CLOSED"


class ResolutionSource(StrEnum):
    """Who RESOLVED an exception. SDD 3.1 CaseOutcome.resolved_by.

    Three members, not four. The exporter acts on an exception but never
    resolves one - it emits the accounting entry for a decision already made -
    so it belongs in AuditActor and not here. Keeping one vocabulary for both
    would let `resolved_by = EXPORTER` typecheck, which would read as "the
    exporter explained this variance".
    """

    DETERMINISTIC = "DETERMINISTIC"
    AI_VERIFIED = "AI_VERIFIED"
    HUMAN = "HUMAN"


class AuditActor(StrEnum):
    """Who performed an audited action. SDD 3 AuditEntry.actor - FOUR members.

    A superset of ResolutionSource by exactly one: EXPORTER. SDD 3 names the
    exporter as the sole writer of CLOSED, so it must be able to appear in the
    audit trail while remaining ineligible as a resolver.
    """

    DETERMINISTIC = "DETERMINISTIC"
    AI_VERIFIED = "AI_VERIFIED"
    HUMAN = "HUMAN"
    EXPORTER = "EXPORTER"


# ---------------------------------------------------------------------------
# Row types (SDD 3.3)
#
# Identifier rule: `settlement_id` is payment-level and names one
# SettlementLine. `batch_id` is batch-level and names one SettlementBatch.
# They never share a namespace - test_id_namespaces_disjoint asserts the two
# ID sets do not intersect.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerRow(_Record):
    """A row of the merchant's order/invoice export.

    order_id is NOT unique across the dataset: the duplicate_ledger_rows
    injector emits genuine duplicates, which are Population C variances
    (SDD 3.1). Any code keying a dict on order_id is wrong.
    """

    order_id: str
    invoice_no: str
    gross: Money
    order_date: date
    customer_id: str
    sku: str


@dataclass(frozen=True)
class PaymentRow(_Record):
    payment_id: str
    order_id: str
    method: PaymentMethod
    authorized: Money
    captured: Money
    status: PaymentStatus
    captured_at: date


@dataclass(frozen=True)
class RefundRow(_Record):
    refund_id: str
    payment_id: str
    amount: Money  # POSITIVE here. The sign lives on the REFUND settlement line.
    created_at: date


@dataclass(frozen=True)
class SettlementLine(_Record):
    """A SIGNED LINE in a settlement batch. NOT 'one row per payment'.

    Per-type invariants (SDD 3.3), asserted by test_line_invariants:

      PAYMENT: refund_id is None; gross/fee/tax from the rate table;
               net = +(gross - fee - tax)
      REFUND:  refund_id set; gross/fee/tax all zero; net = -refund.amount
    """

    settlement_id: str  # unique LINE id. Never a batch id.
    batch_id: str  # FK to SettlementBatch.batch_id
    line_type: SettlementLineType
    payment_id: str
    refund_id: str | None  # set iff line_type == REFUND
    gross: Money
    fee: Money  # 0 on REFUND lines
    tax: Money  # 0 on REFUND lines
    net: Money  # SIGNED: + for PAYMENT, - for REFUND
    settled_event_date: date


@dataclass(frozen=True)
class SettlementBatch(_Record):
    """BATCH-level. The payout that hits the bank.

    net_total = sum of every signed line net in the batch (SDD 3.1a). Deducting
    a refund from expected_net while leaving the batch untouched would make the
    cash invariant unsatisfiable, so both sides move together.
    """

    batch_id: str  # THE batch identifier. NEVER named settlement_id.
    utr: str
    net_total: Money
    settled_event_date: date


@dataclass(frozen=True)
class BankRow(_Record):
    bank_txn_id: str
    value_date: date
    amount: Money
    narration: str  # free text; UTR may be truncated, garbled, or absent
    direction: BankDirection


# ---------------------------------------------------------------------------
# The canonical reconciliation unit (SDD 3.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconciliationCase(_Record):
    """THE canonical unit. One case per captured payment - ALWAYS one.

    A split settlement produces ONE case holding MULTIPLE payment_line_ids. It
    never produces two cases: that would double-count the denominator and make
    the match rate depend on how the gateway happened to batch the payout.

    Every headline metric divides by a count of these and by nothing else
    (D11). Population B (batch<->bank) and Population C (row-grain variances)
    have their own denominators and are never merged into this one.
    """

    case_id: str  # sha256(b"case|" + payment_id)[:16], D10
    payment_id: str
    order_id: str
    merchant_profile: str
    expected_gross: Money
    expected_net: Money  # gross - fee - tax - refunds; SDD 3.1b
    settlement_line_ids: tuple[str, ...]  # sorted; ALL lines touching this payment
    payment_line_ids: tuple[str, ...]  # sorted; PAYMENT lines only.
    # len > 1 => SPLIT_SETTLEMENT. Case matching
    # uses THIS field, never settlement_line_ids:
    # a refund line is not a second settlement.


@dataclass(frozen=True)
class CaseOutcome(_Record):
    """Population A. What the engine concluded about one case."""

    case_id: str
    status: ExceptionStatus
    observed_net: Money | None  # None when no bank credit was linked
    variance: Money | None  # expected_net - observed_net
    category: str | None  # taxonomy.VARIANCE_CATEGORIES only. Deduction
    # categories (MDR_FEE, GST_ON_FEE, REFUND_OFFSET)
    # are components of expected_net and are NEVER
    # emitted here. PDD 6.1.
    batch_id: str | None
    bank_row_id: str | None
    resolved_by: ResolutionSource | None
    confidence: Decimal | None  # a ratio, not money: NOT quantized to paise


@dataclass(frozen=True)
class BatchLinkOutcome(_Record):
    """Population B. One per settlement batch. Batch-count denominator.

    Field names follow SDD 8.1. `batch_net_total` and `linked_amount` are
    deliberately not one name: the first is the link's money BASIS and exists
    whether or not a credit arrived, the second is what actually landed and is
    None when nothing did.

    ONE ADDITION TO THE SPEC'S FIELD LIST, made openly. `category` is not in
    SDD 8.1's definition, and it is required for Population B to be scoreable
    at all: gen/truth.py records `TruthBatchLink.true_category`, and the
    taxonomy carries UTR_TRUNCATED_MAPPING, UTR_MISSING_MAPPING and
    MISSING_VS_LATE_CREDIT, every one of which is a batch-link category rather
    than a case category. Without this field a link that failed because the
    UTR was truncated is indistinguishable from one that failed because the
    credit never arrived, and the eval has nothing to compare true_category
    against.
    """

    batch_id: str
    status: ExceptionStatus
    bank_row_id: str | None  # None when unlinked
    batch_net_total: Money  # Population B money basis. NEVER expected_gross.
    linked_amount: Money | None  # bank credit amount when linked
    variance: Money | None  # batch_net_total - linked_amount
    category: str | None  # see the note above; scored against true_category
    resolved_by: ResolutionSource | None
    confidence: Decimal | None


@dataclass(frozen=True)
class RowVarianceOutcome(_Record):
    """Population C. A variance whose subject is neither a payment nor a batch:
    a duplicate ledger row, an orphan bank credit. Row-count denominator.

    Without this population these rows have no home in truth at all and become
    unscoreable - they are neither payments nor batches, so counting them in
    A or B would corrupt whichever denominator absorbed them (SDD 3.1)."""

    row_id: str
    source_table: str  # ledger_rows | bank_rows
    status: ExceptionStatus
    category: str | None
    amount: Money | None


# ---------------------------------------------------------------------------
# Exception model (SDD 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEntry(_Record):
    """Append-only. One per status change or evidence addition. SDD 3.

    `sequence` is the field that makes the lifecycle representable. SDD 3 says
    the store records BOTH transitions when a human resolves an abstention in
    one click - ABSTAINED -> CONFIRMED and CONFIRMED -> CLOSED - and those
    share an arrival_day. Ordering on arrival_day alone cannot distinguish
    them, so the trail could not answer which happened first.

    Both ordering fields are ints, supplied by the caller. There is no
    wall-clock field anywhere here (D2), which is also why `sequence` has to
    exist: a timestamp is the usual way to order two same-day events and is
    exactly what this project forbids.
    """

    exception_id: str
    arrival_day: int  # integer day index, never a timestamp (D2)
    sequence: int  # ordering within a day; caller-supplied
    from_status: ExceptionStatus | None  # None on the opening entry
    to_status: ExceptionStatus
    actor: AuditActor  # includes EXPORTER, unlike resolved_by
    note: str
    evidence_ids: tuple[str, ...]  # sorted


@dataclass(frozen=True)
class Exception_(_Record):
    """One detected, possibly-explained discrepancy. SDD 3.

    Frozen, so a transition produces a new instance rather than mutating one.
    A mutable exception would let a caller assign `status = CLOSED` directly
    and bypass the transition rules the lifecycle section declares. The store
    (M6) is the only writer.
    """

    exception_id: str  # deterministic: sha256(canonical tuple)[:16]
    category: str  # from the closed taxonomy
    amount: Money
    status: ExceptionStatus
    confidence: Decimal  # verification-derived, 0.00-1.00. A ratio, not money.
    evidence_row_ids: tuple[str, ...]  # sorted
    reason: str
    resolved_by: ResolutionSource | None
    first_seen_day: int  # arrival_day, an int - never a timestamp
    confirmed_day: int | None
    closed_day: int | None
    audit: tuple[AuditEntry, ...]


# ---------------------------------------------------------------------------
# The business result (SDD 8.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconciliationResult(_Record):
    """The business result. Serialized, hashed, compared, goldened.

    Contains NO wall-clock data of any kind - no durations, no timestamps, no
    memory figures. Adding a timing field here is a D6 violation, and
    `test_result_has_no_wallclock` walks this type's annotations recursively
    rather than trusting the sentence you just read.

    The three populations are three separate fields precisely so that no
    caller can average them together (D11).
    """

    cases: tuple[CaseOutcome, ...]  # Population A, sorted by case_id
    batch_links: tuple[BatchLinkOutcome, ...]  # Population B, sorted by batch_id
    row_variances: tuple[RowVarianceOutcome, ...]  # Population C, sorted by row_id
    exceptions: tuple[Exception_, ...]  # sorted by exception_id
    calendar_version: str
    config_hash: str


def _money_field_names(cls: type) -> tuple[str, ...]:
    """Every field of `cls` enrolled in quantization. For tests and diagnostics."""
    return tuple(name for name, is_money in _decimal_fields(cls) if is_money)


def _all_record_types() -> tuple[type, ...]:
    """Every frozen record defined here, sorted by name (D4).

    Derived from module contents rather than listed, so a record added without
    being registered is still swept by the guard tests.
    """
    found: list[tuple[str, type]] = []
    for name, value in globals().items():
        if isinstance(value, type) and issubclass(value, _Record) and value is not _Record:
            found.append((name, value))
    return tuple(cls for _, cls in sorted(found))


def _is_money_annotation(annotation: Any) -> bool:
    """Whether a raw annotation string enrols a field in quantization."""
    text = str(annotation).replace("Optional[", "").replace("]", "").strip()
    return text in _MONEY_ANNOTATIONS

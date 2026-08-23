"""M1 part A - clean order -> payment -> settlement -> batch -> bank chains.

INDEPENDENT PATH. Nothing here imports from settlesense/. The row dataclasses
below deliberately restate the engine's domain model instead of sharing it.

No noise is injected in this module. Clean chains come first and their ground
truth is verified before any noise work begins, because a generator that
mislabels its own truth corrupts every downstream metric silently - and it does
so in the direction that makes the results look good.

Grain matters, and two different things are both called "net":

    SettlementLine(PAYMENT).net = + (gross - fee - tax)      per-line
    SettlementLine(REFUND).net  = - refund.amount            per-line
    batch.net_total             = SUM of signed line nets    per-batch
    case expected_net           = gross - fee - tax - refunds  per-payment

The case figure equals the payment line's net plus that payment's refund line
nets. Keeping REFUND a signed batch line is what lets both sides move together,
so `bank_credit.amount == batch.net_total` stays exactly satisfiable (SDD 3.1a).

Chargebacks and disputes are OUT OF SCOPE for v1: there is no dispute row type,
no DISPUTE_DEBIT line type and no dispute edge anywhere in this module.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import yaml

from gen.profiles import MerchantProfile

__all__ = [
    "REQUIRED_YEAR",
    "BankRow",
    "Chain",
    "CleanDataset",
    "GeneratorError",
    "LedgerRow",
    "PaymentRow",
    "PendingSettlementLine",
    "RefundRow",
    "SettlementBatch",
    "SettlementLine",
    "SettlementLineType",
    "WorkingCalendar",
    "assemble_batches",
    "bank_txn_id_for",
    "generate_clean_chain",
    "load_working_calendar",
    "money",
    "verify_clean_dataset",
]

Q2: Final[Decimal] = Decimal("0.01")
ZERO: Final[Decimal] = Decimal("0.00")
REQUIRED_YEAR: Final[int] = 2026  # D13

# A working-day walk that needs more steps than this is a calendar bug, not a
# long weekend. Bounded so a malformed calendar fails instead of hanging.
_MAX_CALENDAR_STEPS: Final[int] = 400

_WEEKDAY_NAMES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
)


class GeneratorError(RuntimeError):
    """Raised when the generator would emit data that violates its own truth."""


def money(value: Decimal | int | str) -> Decimal:
    """Quantize to 2 dp with ROUND_HALF_UP (D1). Floats are not accepted."""
    if isinstance(value, float):  # pragma: no cover - guarded by type checking
        raise TypeError("money() refuses float input; D1 bans floats in money paths")
    return Decimal(value).quantize(Q2, rounding=ROUND_HALF_UP)


def _stable_id(prefix: str, *parts: object) -> str:
    """A deterministic ID: sha256 of a canonical tuple, never uuid4 (D10).

    The prefix keeps namespaces disjoint. `settlement_id` (SET_) and `batch_id`
    (BAT_) must never collide - SDD 3.3 makes that a hard rule, because a shared
    namespace turns SETTLEMENT_TO_BATCH self-referential.
    """
    canonical = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(f"{prefix}|{canonical}".encode()).hexdigest()
    return f"{prefix}_{digest[:12].upper()}"


def _stable_utr(*parts: object) -> str:
    """A 16-character alphanumeric UTR, deterministic in its inputs (D10)."""
    canonical = "|".join(str(part) for part in parts)
    return hashlib.sha256(f"utr|{canonical}".encode()).hexdigest()[:16].upper()


def bank_txn_id_for(batch_id: str) -> str:
    """The bank credit a batch produces, derived from the batch id alone (D10).

    TRUTH-SIDE LINKAGE. The generator knows which credit belongs to which batch
    because it built both; it must never recover that link by parsing the
    narration, because narration is exactly what the noise layer damages.
    Reading truth out of the narration would make ground truth degrade in step
    with the difficulty - the link would vanish precisely when the engine is
    being tested on finding it.
    """
    return _stable_id("BNK", batch_id)


def _check_year(value: date, what: str) -> date:
    if value.year != REQUIRED_YEAR:
        raise GeneratorError(
            f"D13 violation: {what} is {value.isoformat()}, not in {REQUIRED_YEAR}"
        )
    return value


# ---------------------------------------------------------------------------
# Working-day calendar - read from config/calendar_v1.yaml by gen's OWN parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkingCalendar:
    """Working days and T+N, parsed independently of settlesense.config."""

    version: str
    weekly_offs: frozenset[int]
    holidays: frozenset[date]
    window_start: date
    window_end: date
    settlement_cycles: Mapping[str, int]

    def is_working_day(self, day: date) -> bool:
        return day.weekday() not in self.weekly_offs and day not in self.holidays

    def next_working_day(self, day: date) -> date:
        """The first working day on or after `day`."""
        current = day
        for _ in range(_MAX_CALENDAR_STEPS):
            if self.is_working_day(current):
                return _check_year(current, "settlement date")
            current += timedelta(days=1)
        raise GeneratorError(f"no working day within {_MAX_CALENDAR_STEPS} days of {day}")

    def add_working_days(self, start: date, count: int) -> date:
        """`count` working days after `start`, which is first rolled to a working day."""
        if count < 0:
            raise GeneratorError(f"cannot add {count} working days")
        current = self.next_working_day(start)
        remaining = count
        steps = 0
        while remaining > 0:
            current += timedelta(days=1)
            steps += 1
            if steps > _MAX_CALENDAR_STEPS:
                raise GeneratorError(f"working-day walk from {start} exceeded {steps} steps")
            if self.is_working_day(current):
                remaining -= 1
        return _check_year(current, "bank value date")


def load_working_calendar(path: Path) -> WorkingCalendar:
    """Parse config/calendar_v1.yaml with gen's own reader.

    Reading the same config FILE is shared input, not shared code. What the
    hard rule forbids is importing settlesense's loader.
    """
    if not path.is_file():
        raise GeneratorError(f"calendar file not found: {path}")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise GeneratorError(f"calendar file {path} is not a mapping")

    def need(key: str) -> Any:
        if key not in raw:
            raise GeneratorError(f"calendar file {path} is missing required key {key!r}")
        return raw[key]

    weekly_offs: set[int] = set()
    for name in need("weekly_offs"):
        if not isinstance(name, str) or name.lower() not in _WEEKDAY_NAMES:
            raise GeneratorError(f"calendar: {name!r} is not a weekday name")
        weekly_offs.add(_WEEKDAY_NAMES[name.lower()])

    def as_date(value: object, what: str) -> date:
        if isinstance(value, date):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = date.fromisoformat(value)
            except ValueError as exc:
                raise GeneratorError(f"calendar: {what} {value!r} is not an ISO date") from exc
        else:
            raise GeneratorError(f"calendar: {what} {value!r} is not a date")
        return _check_year(parsed, f"calendar {what}")

    window = need("simulation_window")
    holidays = frozenset(as_date(value, "holiday") for value in need("holidays"))

    cycles_raw = need("settlement_cycles")
    if not isinstance(cycles_raw, dict):
        raise GeneratorError("calendar: settlement_cycles must be a mapping")
    cycles: dict[str, int] = {}
    for key, value in cycles_raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise GeneratorError(f"calendar: settlement cycle for {key!r} must be a positive int")
        cycles[str(key)] = value

    return WorkingCalendar(
        version=str(need("version")),
        weekly_offs=frozenset(weekly_offs),
        holidays=holidays,
        window_start=as_date(window["start"], "window start"),
        window_end=as_date(window["end"], "window end"),
        settlement_cycles=MappingProxyType(cycles),
    )


# ---------------------------------------------------------------------------
# Row types. Restated here, deliberately not imported from settlesense.types.
# ---------------------------------------------------------------------------


class SettlementLineType(StrEnum):
    PAYMENT = "payment"  # credit: a captured payment settling
    REFUND = "refund"  # debit:  a refund deducted from the batch


@dataclass(frozen=True)
class LedgerRow:
    order_id: str
    invoice_no: str
    gross: Decimal
    order_date: date
    customer_id: str
    sku: str


@dataclass(frozen=True)
class PaymentRow:
    payment_id: str
    order_id: str
    method: str
    authorized: Decimal
    captured: Decimal
    status: str  # captured | refunded
    captured_at: date


@dataclass(frozen=True)
class RefundRow:
    refund_id: str
    payment_id: str
    amount: Decimal
    created_at: date


@dataclass(frozen=True)
class SettlementLine:
    """A SIGNED line in a settlement batch. Not 'one row per payment'."""

    settlement_id: str  # unique LINE id. Never a batch id.
    batch_id: str
    line_type: SettlementLineType
    payment_id: str
    refund_id: str | None  # set iff line_type == REFUND
    gross: Decimal
    fee: Decimal  # 0 on REFUND lines
    tax: Decimal  # 0 on REFUND lines
    net: Decimal  # SIGNED: + for PAYMENT, - for REFUND
    settled_event_date: date


@dataclass(frozen=True)
class PendingSettlementLine:
    """A line that knows everything except which batch it lands in.

    Batching is a grouping step across chains: a real batch holds many lines
    from many payments, and a refund nets off inside whichever batch settles on
    or after the refund date - often not its own payment's batch. Materialising
    a SettlementLine before that is known would mean inventing a batch_id and
    correcting it later, so the incomplete state gets its own type instead.
    """

    settlement_id: str
    line_type: SettlementLineType
    payment_id: str
    refund_id: str | None
    gross: Decimal
    fee: Decimal
    tax: Decimal
    net: Decimal
    settled_event_date: date
    profile_name: str

    def in_batch(self, batch_id: str, settled_event_date: date) -> SettlementLine:
        return SettlementLine(
            settlement_id=self.settlement_id,
            batch_id=batch_id,
            line_type=self.line_type,
            payment_id=self.payment_id,
            refund_id=self.refund_id,
            gross=self.gross,
            fee=self.fee,
            tax=self.tax,
            net=self.net,
            settled_event_date=settled_event_date,
        )


@dataclass(frozen=True)
class SettlementBatch:
    """BATCH-level. The payout that hits the bank."""

    batch_id: str  # THE batch identifier. Never named settlement_id.
    utr: str
    net_total: Decimal
    settled_event_date: date


@dataclass(frozen=True)
class BankRow:
    bank_txn_id: str
    value_date: date
    amount: Decimal
    narration: str
    direction: str  # credit | debit


@dataclass(frozen=True)
class Chain:
    """One complete payment-grain chain, before batch assignment."""

    profile_name: str
    day: int
    ledger: LedgerRow
    payment: PaymentRow
    refund: RefundRow | None
    payment_line: PendingSettlementLine
    refund_line: PendingSettlementLine | None
    # A split settlement gives ONE case several PAYMENT lines (SDD 3.1). They
    # live here rather than as extra Chains: two chains would be two cases, and
    # the denominator would then depend on how the gateway batched the payout.
    extra_payment_lines: tuple[PendingSettlementLine, ...] = ()

    @property
    def payment_lines(self) -> tuple[PendingSettlementLine, ...]:
        return (self.payment_line, *self.extra_payment_lines)

    @property
    def expected_gross(self) -> Decimal:
        return self.payment.captured

    @property
    def expected_net(self) -> Decimal:
        """gross - fee - tax - refunds, at case grain (SDD 3.1b)."""
        total = sum((line.net for line in self.payment_lines), start=ZERO)
        if self.refund_line is not None:
            total += self.refund_line.net  # already negative
        return money(total)

    @property
    def fee(self) -> Decimal:
        return money(sum((line.fee for line in self.payment_lines), start=ZERO))

    @property
    def tax(self) -> Decimal:
        return money(sum((line.tax for line in self.payment_lines), start=ZERO))

    @property
    def refund_amount(self) -> Decimal:
        return self.refund.amount if self.refund is not None else ZERO


@dataclass(frozen=True)
class CleanDataset:
    """Everything a clean run produces, each list explicitly sorted (D4)."""

    chains: tuple[Chain, ...]
    ledger_rows: tuple[LedgerRow, ...]
    payment_rows: tuple[PaymentRow, ...]
    refund_rows: tuple[RefundRow, ...]
    settlement_lines: tuple[SettlementLine, ...]
    batches: tuple[SettlementBatch, ...]
    bank_rows: tuple[BankRow, ...]


# ---------------------------------------------------------------------------
# Chain generation
# ---------------------------------------------------------------------------


def compute_fee_and_tax(
    gross: Decimal, profile: MerchantProfile, method: str
) -> tuple[
    Decimal,
    Decimal,
]:
    """fee = round(gross * rate, 2); tax = round(fee * gst, 2). Both ROUND_HALF_UP."""
    fee = money(gross * profile.rate_for(method))
    tax = money(fee * profile.gst_rate)
    return fee, tax


def generate_clean_chain(
    rng: random.Random,
    profile: MerchantProfile,
    day: int,
    *,
    sequence: int,
    calendar: WorkingCalendar,
    base_date: date,
    last_capture_date: date,
) -> Chain:
    """Produce one complete, internally consistent chain. No noise.

    `rng` is the single seeded generator, threaded through explicitly (D3).
    `day` is a 1-indexed simulated capture day; `sequence` disambiguates chains
    within a (profile, day) so every derived ID is unique and reproducible.

    `last_capture_date` bounds refund creation. SDD 3.1a puts every REFUND line
    in the batch settling ON OR AFTER the refund date, so a refund created after
    the final batch has no forward batch to net off against. Clamping here keeps
    that rule satisfiable; a refund arriving after the last batch is a Day-N+1
    concern for incremental state, not something clean chains should fabricate.
    """
    if day < 1:
        raise GeneratorError(f"day must be 1-indexed, got {day}")

    order_date = _check_year(base_date + timedelta(days=day - 1), "order date")
    captured_at = order_date  # clean chains capture on the order date

    order_id = _stable_id("ORD", profile.name, day, sequence)
    payment_id = _stable_id("PAY", profile.name, day, sequence)

    gross = money(Decimal(rng.randint(profile.min_gross_paise, profile.max_gross_paise)) / 100)

    method_names, method_weights = profile.weighted_methods()
    method = rng.choices(method_names, weights=method_weights, k=1)[0]
    fee, tax = compute_fee_and_tax(gross, profile, method)

    customer_id = f"CUST-{profile.name.upper()}-{rng.randrange(profile.customer_pool_size):05d}"
    sku = profile.sku_pool[rng.randrange(len(profile.sku_pool))]
    invoice_no = f"INV-{profile.name[-1].upper()}-{day:02d}-{sequence:06d}"

    ledger = LedgerRow(
        order_id=order_id,
        invoice_no=invoice_no,
        gross=gross,
        order_date=order_date,
        customer_id=customer_id,
        sku=sku,
    )

    # A refund is decided before the payment row, because it sets the status.
    refund: RefundRow | None = None
    refund_line: PendingSettlementLine | None = None
    if _bernoulli(rng, profile.refund_rate):
        if _bernoulli(rng, profile.partial_refund_share):
            # Partial: 10%..90% of gross, in whole paise, never exceeding it.
            paise = int((gross * 100).to_integral_value())
            refund_amount = money(Decimal(rng.randint(paise // 10, (paise * 9) // 10)) / 100)
        else:
            refund_amount = gross
        refund_created = min(
            order_date + timedelta(days=rng.randint(0, 3)),
            last_capture_date,
        )
        _check_year(refund_created, "refund created date")
        refund = RefundRow(
            refund_id=_stable_id("RFD", profile.name, day, sequence),
            payment_id=payment_id,
            amount=refund_amount,
            created_at=refund_created,
        )

    payment = PaymentRow(
        payment_id=payment_id,
        order_id=order_id,
        method=method,
        authorized=gross,  # clean chains capture in full; partial capture is noise
        captured=gross,
        status="refunded" if refund is not None else "captured",
        captured_at=captured_at,
    )

    # The gateway cuts the batch on the first working day on or after capture.
    # The T+N cycle is the bank leg, applied at assembly.
    settled_event_date = calendar.next_working_day(captured_at)

    payment_line = PendingSettlementLine(
        settlement_id=_stable_id("SET", profile.name, day, sequence, "payment"),
        line_type=SettlementLineType.PAYMENT,
        payment_id=payment_id,
        refund_id=None,
        gross=gross,
        fee=fee,
        tax=tax,
        net=money(gross - fee - tax),
        settled_event_date=settled_event_date,
        profile_name=profile.name,
    )

    if refund is not None:
        # Refund lines carry no gross, fee or tax (SDD 3.3), and a negative net.
        # settled_event_date is provisional: assembly moves it to the first batch
        # settling on or after the refund date.
        refund_line = PendingSettlementLine(
            settlement_id=_stable_id("SET", profile.name, day, sequence, "refund"),
            line_type=SettlementLineType.REFUND,
            payment_id=payment_id,
            refund_id=refund.refund_id,
            gross=ZERO,
            fee=ZERO,
            tax=ZERO,
            net=money(-refund.amount),
            settled_event_date=calendar.next_working_day(refund.created_at),
            profile_name=profile.name,
        )

    return Chain(
        profile_name=profile.name,
        day=day,
        ledger=ledger,
        payment=payment,
        refund=refund,
        payment_line=payment_line,
        refund_line=refund_line,
    )


def _bernoulli(rng: random.Random, probability: Decimal) -> bool:
    """A Decimal-parameterised coin flip, resolved in integer basis points.

    Comparing against rng.random() would put a float on the decision path; the
    outcome would then depend on binary rounding near the threshold.
    """
    threshold = int((probability * 10_000).to_integral_value(rounding=ROUND_HALF_UP))
    return rng.randrange(10_000) < threshold


# ---------------------------------------------------------------------------
# Batch assembly - the N:1 step
# ---------------------------------------------------------------------------


def assemble_batches(
    chains: Sequence[Chain],
    calendar: WorkingCalendar,
    profiles: Mapping[str, MerchantProfile],
) -> tuple[tuple[SettlementLine, ...], tuple[SettlementBatch, ...], tuple[BankRow, ...]]:
    """Group lines into real multi-line batches, then credit each batch once.

    Batch key is (profile, settlement date), so a batch legitimately holds many
    PAYMENT lines. Each REFUND line is placed in the first batch for its profile
    settling on or after the refund date (SDD 3.1a) - typically a batch full of
    unrelated payments, which is exactly the netting-off behaviour the product
    has to unpick.
    """
    # 1. Batch dates come from PAYMENT lines: a batch exists because payments settle.
    dates_by_profile: dict[str, list[date]] = {}
    for chain in chains:
        dates_by_profile.setdefault(chain.profile_name, []).append(
            chain.payment_line.settled_event_date
        )
    sorted_dates = {
        profile_name: sorted(set(dates)) for profile_name, dates in dates_by_profile.items()
    }

    # 2. Bucket every line under (profile, settlement date).
    buckets: dict[tuple[str, date], list[PendingSettlementLine]] = {}
    for chain in chains:
        key = (chain.profile_name, chain.payment_line.settled_event_date)
        buckets.setdefault(key, []).append(chain.payment_line)

    for chain in chains:
        if chain.refund_line is None:
            continue
        candidates = sorted_dates.get(chain.profile_name, [])
        target = _first_on_or_after(candidates, chain.refund_line.settled_event_date)
        if target is None:
            # Placing it in an EARLIER batch would net a refund off a payout that
            # had already left the bank - backwards in time, and silently wrong
            # rather than loudly wrong. generate_clean_chain clamps refund dates
            # so this is unreachable for clean data; if it fires, that clamp broke.
            raise GeneratorError(
                f"refund {chain.refund_line.refund_id} for {chain.profile_name} settles "
                f"{chain.refund_line.settled_event_date}, after every batch for that "
                f"profile ({candidates[-1] if candidates else 'none'}). A REFUND line "
                f"must land in a batch settling ON OR AFTER the refund date (SDD 3.1a)."
            )
        buckets.setdefault((chain.profile_name, target), []).append(chain.refund_line)

    # 3. Materialise, computing net_total by SUMMATION over signed line nets.
    settlement_lines: list[SettlementLine] = []
    batches: list[SettlementBatch] = []
    bank_rows: list[BankRow] = []

    for profile_name, settled_date in sorted(buckets, key=lambda key: (key[0], key[1])):
        profile = profiles[profile_name]
        pending = sorted(
            buckets[(profile_name, settled_date)],
            key=lambda line: (line.line_type.value, line.settlement_id),
        )
        batch_id = _stable_id("BAT", profile_name, settled_date.isoformat())
        utr = _stable_utr(profile_name, settled_date.isoformat(), batch_id)

        members = tuple(line.in_batch(batch_id, settled_date) for line in pending)
        settlement_lines.extend(members)

        # Summed, never re-derived from gross. Re-deriving would silently paper
        # over a wrong per-line net, which is the bug this data exists to catch.
        net_total = money(sum((line.net for line in members), start=ZERO))
        if net_total <= ZERO:
            raise GeneratorError(
                f"batch {batch_id} ({profile_name} {settled_date}) nets to {net_total}; "
                f"a payout batch must be a positive credit. Refund volume is too "
                f"high relative to payment volume for this profile."
            )

        batches.append(
            SettlementBatch(
                batch_id=batch_id,
                utr=utr,
                net_total=net_total,
                settled_event_date=settled_date,
            )
        )

        value_date = calendar.add_working_days(settled_date, profile.settlement_cycle_days)
        bank_rows.append(
            BankRow(
                bank_txn_id=_stable_id("BNK", batch_id),
                value_date=value_date,
                amount=net_total,
                narration=f"NEFT {utr} {profile.merchant_name} SETTLEMENT",
                direction="credit",
            )
        )

    settlement_lines.sort(key=lambda line: line.settlement_id)
    batches.sort(key=lambda batch: batch.batch_id)
    bank_rows.sort(key=lambda row: row.bank_txn_id)
    return tuple(settlement_lines), tuple(batches), tuple(bank_rows)


def _first_on_or_after(candidates: Sequence[date], target: date) -> date | None:
    for candidate in candidates:  # candidates are sorted
        if candidate >= target:
            return candidate
    return None


def build_clean_dataset(
    rng: random.Random,
    plan: Sequence[tuple[int, MerchantProfile, int]],
    calendar: WorkingCalendar,
    base_date: date,
) -> CleanDataset:
    """Run the whole clean pipeline for a deterministic (day, profile, seq) plan."""
    last_capture_date = base_date + timedelta(days=max(day for day, _, _ in plan) - 1)
    chains = [
        generate_clean_chain(
            rng,
            profile,
            day,
            sequence=sequence,
            calendar=calendar,
            base_date=base_date,
            last_capture_date=last_capture_date,
        )
        for day, profile, sequence in plan
    ]
    profiles = {profile.name: profile for _, profile, _ in plan}
    settlement_lines, batches, bank_rows = assemble_batches(chains, calendar, profiles)

    return CleanDataset(
        chains=tuple(sorted(chains, key=lambda chain: chain.payment.payment_id)),
        ledger_rows=tuple(sorted((c.ledger for c in chains), key=lambda row: row.order_id)),
        payment_rows=tuple(sorted((c.payment for c in chains), key=lambda row: row.payment_id)),
        refund_rows=tuple(
            sorted(
                (c.refund for c in chains if c.refund is not None),
                key=lambda row: row.refund_id,
            )
        ),
        settlement_lines=settlement_lines,
        batches=batches,
        bank_rows=bank_rows,
    )


# ---------------------------------------------------------------------------
# Ground-truth verification. Runs BEFORE any noise work exists.
# ---------------------------------------------------------------------------


def verify_clean_dataset(dataset: CleanDataset) -> list[str]:
    """Return every conservation or structural violation found. Empty is clean.

    Returns rather than raises so a caller can report all failures at once
    instead of one per run.
    """
    problems: list[str] = []
    lines_by_batch: dict[str, list[SettlementLine]] = {}
    for line in dataset.settlement_lines:
        lines_by_batch.setdefault(line.batch_id, []).append(line)

    # --- per-line-type field invariants (SDD 3.3) --------------------------
    for line in dataset.settlement_lines:
        if line.line_type is SettlementLineType.PAYMENT:
            if line.refund_id is not None:
                problems.append(f"{line.settlement_id}: PAYMENT line carries a refund_id")
            if line.net != money(line.gross - line.fee - line.tax):
                problems.append(
                    f"{line.settlement_id}: PAYMENT net {line.net} != gross-fee-tax "
                    f"({line.gross}-{line.fee}-{line.tax})"
                )
            if line.net <= ZERO:
                problems.append(f"{line.settlement_id}: PAYMENT net {line.net} is not positive")
        else:
            if line.refund_id is None:
                problems.append(f"{line.settlement_id}: REFUND line has no refund_id")
            if (line.gross, line.fee, line.tax) != (ZERO, ZERO, ZERO):
                problems.append(
                    f"{line.settlement_id}: REFUND line must have zero gross/fee/tax, "
                    f"got {line.gross}/{line.fee}/{line.tax}"
                )
            if line.net >= ZERO:
                problems.append(f"{line.settlement_id}: REFUND net {line.net} is not negative")

    # --- batch total is the SIGNED SUM of its lines (SDD 3.1a) -------------
    for batch in dataset.batches:
        members = lines_by_batch.get(batch.batch_id, [])
        if not members:
            problems.append(f"{batch.batch_id}: batch has no settlement lines")
            continue
        summed = money(sum((line.net for line in members), start=ZERO))
        if batch.net_total != summed:
            problems.append(
                f"{batch.batch_id}: net_total {batch.net_total} != signed line sum {summed}"
            )
        for line in members:
            if line.settled_event_date != batch.settled_event_date:
                problems.append(
                    f"{line.settlement_id}: settled {line.settled_event_date} but its batch "
                    f"{batch.batch_id} settled {batch.settled_event_date}"
                )

    # --- one bank credit per batch, exactly equal (SDD 3.2 BATCH_TO_BANK) --
    totals_by_batch = {batch.batch_id: batch.net_total for batch in dataset.batches}
    if len(dataset.bank_rows) != len(dataset.batches):
        problems.append(
            f"bank rows ({len(dataset.bank_rows)}) != batches ({len(dataset.batches)}); "
            f"BATCH_TO_BANK is 1:1"
        )
    bank_by_batch: dict[str, BankRow] = {}
    for row in dataset.bank_rows:
        batch_id = _batch_id_for_bank_row(row, dataset.batches)
        if batch_id is None:
            problems.append(f"{row.bank_txn_id}: bank row matches no batch UTR")
            continue
        if batch_id in bank_by_batch:
            problems.append(f"{batch_id}: two bank credits for one batch")
        bank_by_batch[batch_id] = row
        if row.amount != totals_by_batch[batch_id]:
            problems.append(
                f"{row.bank_txn_id}: credit {row.amount} != batch {batch_id} "
                f"net_total {totals_by_batch[batch_id]}"
            )
        if row.direction != "credit":
            problems.append(f"{row.bank_txn_id}: direction is {row.direction!r}, expected 'credit'")

    # --- global conservation, BOTH sides (SDD 3.1b) ------------------------
    sum_gross = money(sum((chain.expected_gross for chain in dataset.chains), start=ZERO))
    sum_net = money(sum((chain.expected_net for chain in dataset.chains), start=ZERO))
    sum_fee = money(sum((chain.fee for chain in dataset.chains), start=ZERO))
    sum_tax = money(sum((chain.tax for chain in dataset.chains), start=ZERO))
    sum_refunds = money(sum((chain.refund_amount for chain in dataset.chains), start=ZERO))
    if sum_gross != money(sum_net + sum_fee + sum_tax + sum_refunds):
        problems.append(
            f"case-side conservation: gross {sum_gross} != net {sum_net} + fee {sum_fee} "
            f"+ tax {sum_tax} + refunds {sum_refunds}"
        )

    sum_bank = money(sum((row.amount for row in dataset.bank_rows), start=ZERO))
    sum_batches = money(sum((batch.net_total for batch in dataset.batches), start=ZERO))
    sum_lines = money(sum((line.net for line in dataset.settlement_lines), start=ZERO))
    if not (sum_bank == sum_batches == sum_lines):
        problems.append(
            f"batch-side conservation: bank {sum_bank} != batches {sum_batches} "
            f"!= signed lines {sum_lines}"
        )
    if sum_net != sum_lines:
        problems.append(
            f"the two sides disagree: case expected_net {sum_net} != signed line total {sum_lines}"
        )

    # --- identifier namespaces are disjoint (SDD 3.3) ----------------------
    line_ids = {line.settlement_id for line in dataset.settlement_lines}
    batch_ids = {batch.batch_id for batch in dataset.batches}
    collisions = sorted(line_ids & batch_ids)
    if collisions:
        problems.append(f"settlement_id and batch_id namespaces overlap: {collisions[:5]}")

    # --- referential integrity and uniqueness ------------------------------
    problems.extend(_check_unique("order_id", (row.order_id for row in dataset.ledger_rows)))
    problems.extend(_check_unique("payment_id", (row.payment_id for row in dataset.payment_rows)))
    problems.extend(_check_unique("refund_id", (row.refund_id for row in dataset.refund_rows)))
    problems.extend(_check_unique("settlement_id", line_ids_iter(dataset)))
    problems.extend(_check_unique("batch_id", (batch.batch_id for batch in dataset.batches)))
    problems.extend(_check_unique("bank_txn_id", (row.bank_txn_id for row in dataset.bank_rows)))

    order_ids = {row.order_id for row in dataset.ledger_rows}
    payment_ids = {row.payment_id for row in dataset.payment_rows}
    refund_ids = {row.refund_id for row in dataset.refund_rows}
    for payment in dataset.payment_rows:
        if payment.order_id not in order_ids:
            problems.append(f"{payment.payment_id}: order_id {payment.order_id} not in ledger")
    for refund in dataset.refund_rows:
        if refund.payment_id not in payment_ids:
            problems.append(f"{refund.refund_id}: payment_id {refund.payment_id} not in payments")
    for line in dataset.settlement_lines:
        if line.payment_id not in payment_ids:
            problems.append(f"{line.settlement_id}: payment_id {line.payment_id} not in payments")
        if line.refund_id is not None and line.refund_id not in refund_ids:
            problems.append(f"{line.settlement_id}: refund_id {line.refund_id} not in refunds")
        if line.batch_id not in batch_ids:
            problems.append(f"{line.settlement_id}: batch_id {line.batch_id} not in batches")

    # --- exactly one REFUND line per RefundRow (SDD 3.1a) ------------------
    refund_line_ids = [
        line.refund_id
        for line in dataset.settlement_lines
        if line.line_type is SettlementLineType.REFUND
    ]
    if len(refund_line_ids) != len(dataset.refund_rows):
        problems.append(
            f"{len(refund_line_ids)} REFUND lines for {len(dataset.refund_rows)} refunds; "
            f"every refund emits exactly one"
        )
    problems.extend(_check_unique("REFUND line refund_id", iter(refund_line_ids)))

    # --- exactly one PAYMENT line per payment (clean chains have no splits) -
    payment_line_counts: dict[str, int] = {}
    for line in dataset.settlement_lines:
        if line.line_type is SettlementLineType.PAYMENT:
            payment_line_counts[line.payment_id] = payment_line_counts.get(line.payment_id, 0) + 1
    for payment_id, count in sorted(payment_line_counts.items()):
        if count != 1:
            problems.append(
                f"{payment_id}: {count} PAYMENT lines. Split settlement is a WITHHELD "
                f"noise type and must not appear in clean chains."
            )

    # --- a REFUND line never settles before its refund exists (SDD 3.1a) ---
    refund_created = {row.refund_id: row.created_at for row in dataset.refund_rows}
    for line in dataset.settlement_lines:
        if line.line_type is not SettlementLineType.REFUND or line.refund_id is None:
            continue
        created = refund_created.get(line.refund_id)
        if created is not None and line.settled_event_date < created:
            problems.append(
                f"{line.settlement_id}: REFUND line settles {line.settled_event_date}, "
                f"before refund {line.refund_id} was created {created}. A refund cannot "
                f"net off a payout that already left the bank."
            )

    # --- refund never exceeds what was captured ----------------------------
    captured = {row.payment_id: row.captured for row in dataset.payment_rows}
    for refund in dataset.refund_rows:
        if refund.payment_id in captured and refund.amount > captured[refund.payment_id]:
            problems.append(
                f"{refund.refund_id}: refund {refund.amount} exceeds captured "
                f"{captured[refund.payment_id]}"
            )

    # --- D13 ---------------------------------------------------------------
    for label, value in _all_dates(dataset):
        if value.year != REQUIRED_YEAR:
            problems.append(f"D13 violation: {label} is {value.isoformat()}")

    return sorted(problems)


def line_ids_iter(dataset: CleanDataset) -> Any:
    return (line.settlement_id for line in dataset.settlement_lines)


def _batch_id_for_bank_row(row: BankRow, batches: Sequence[SettlementBatch]) -> str | None:
    """Recover the batch from the narration's UTR, the way the engine must."""
    for batch in batches:
        if batch.utr in row.narration:
            return batch.batch_id
    return None


def _check_unique(label: str, values: Any) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        return [f"duplicate {label}: {sorted(duplicates)[:5]}"]
    return []


def _all_dates(dataset: CleanDataset) -> list[tuple[str, date]]:
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

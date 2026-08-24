"""M2 - Reading a day's CSV files into typed, normalized records.

ALL file I/O lives here (SDD 2). normalize.py is pure and stays that way:
`test_normalize_is_pure` AST-scans it for file access, and that guard only
works while there is nothing in it to excuse.

This module is where DATASET invariants are enforced - that every date falls
inside the configured simulation window (D13), that a REFUND line carries a
refund id and a PAYMENT line does not. They are checked here, at the boundary,
rather than in normalize.py, because they are properties of this data rather
than properties of parsing.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Final, TypeVar

from settlesense.config import AppConfig
from settlesense.normalize import parse_amount, parse_date
from settlesense.types import (
    BankDirection,
    BankRow,
    LedgerRow,
    PaymentMethod,
    PaymentRow,
    PaymentStatus,
    RefundRow,
    SettlementBatch,
    SettlementLine,
    SettlementLineType,
)

__all__ = ["DayDataset", "IngestError", "canonical_sort_key", "load_dataset"]


class IngestError(ValueError):
    """A day's files could not be read into valid records.

    Every raise below names the file, the row, and the offending value. An
    ingestion error that says only "invalid input" costs more to diagnose than
    the parse cost to prevent.
    """


TABLE_FILES: Final[Mapping[str, str]] = {
    "ledger_rows": "ledger",
    "payment_rows": "payments",
    "refund_rows": "refunds",
    "settlement_lines": "settlements",
    "settlement_batches": "batches",
    "bank_rows": "bank",
}
"""Logical table name -> the `day{N}_<stem>.csv` stem.

The same six-way correspondence tests/test_repo_hygiene.py declares. Written
out rather than derived by stripping suffixes: a rule like "drop the trailing
_rows" produces `settlement_batche` from `settlement_batches`, which this
project has already been bitten by once.
"""

T = TypeVar("T")


@dataclass(frozen=True)
class DayDataset:
    """One arrival day's six tables, typed and sorted.

    `arrival_day` is an int, never a timestamp (SDD 4.1a). It is the delivery
    day, NOT the period the files cover: out-of-order arrival means a day's
    files can carry event dates from an earlier day, and conflating the two is
    how a timing bug becomes a silent false match.

    Every tuple is sorted by its table's primary id. Sorting is by the full
    field tuple rendered as text, with the primary id first in every record, so
    the order is TOTAL. That matters because `ledger_rows.order_id` is not
    unique - the duplicate_ledger_rows injector emits genuine duplicates - and
    a sort keyed on the id alone would leave their relative order to whatever
    the CSV reader happened to yield (D4).
    """

    arrival_day: int
    ledger_rows: tuple[LedgerRow, ...]
    payment_rows: tuple[PaymentRow, ...]
    refund_rows: tuple[RefundRow, ...]
    settlement_lines: tuple[SettlementLine, ...]
    settlement_batches: tuple[SettlementBatch, ...]
    bank_rows: tuple[BankRow, ...]

    def row_count(self) -> int:
        """Total rows across all six tables. Not a reconciliation denominator.

        Population A divides by ReconciliationCase and nothing else (D11).
        This is for ingestion diagnostics only.
        """
        return sum(len(getattr(self, name)) for name in TABLE_FILES)


def canonical_sort_key(record: object) -> tuple[str, ...]:
    """Every field as text, in declaration order.

    The primary id is the first declared field of all six record types, so
    this sorts by primary id and then breaks ties deterministically. The tie
    break is lexical rather than semantic on purpose - it exists to be stable,
    not to be meaningful.
    """
    return tuple(str(getattr(record, spec.name)) for spec in fields(record))  # type: ignore[arg-type]


def _sorted(records: Iterable[T]) -> tuple[T, ...]:
    return tuple(sorted(records, key=canonical_sort_key))


def _read_rows(path: Path, expected: tuple[str, ...]) -> Iterator[tuple[int, Mapping[str, str]]]:
    """Yield (1-based data row number, row). Uses csv, never split(",").

    The mixed_amount_formats injector renders amounts as "1,234.00", which the
    generator quotes. Splitting on commas shifts every later column on those
    rows and produces dates where amounts should be - a corruption that looks
    like a parsing bug in a different module entirely.
    """
    if not path.exists():
        raise IngestError(
            f"{path} does not exist. A missing input file is not an empty table: "
            "returning no rows here would report a clean reconciliation for a "
            "day whose data never arrived."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != expected:
            raise IngestError(
                f"{path.name} header is {list(header)}, expected {list(expected)}. "
                "A reordered or renamed column would otherwise be read positionally "
                "into the wrong field."
            )
        yield from enumerate(reader, start=1)


def _enum(kind: type[T], value: str, path: Path, row: int, column: str) -> T:
    """Coerce to a closed vocabulary. An unknown value raises rather than defaulting."""
    try:
        return kind(value)  # type: ignore[call-arg]
    except ValueError as exc:
        allowed = sorted(member.value for member in kind)  # type: ignore[attr-defined]
        raise IngestError(
            f"{path.name} row {row}: {column}={value!r} is not one of {allowed}. "
            "Defaulting an unrecognised value would let an unknown payment "
            "method be priced as if it were a known one."
        ) from exc


def _guarded(path: Path, row: int, column: str, value: str, parse: Callable[[str], T]) -> T:
    """Run a parser, and name the file, row and column if it raises."""
    try:
        return parse(value)
    except (ValueError, TypeError) as exc:
        raise IngestError(f"{path.name} row {row}: {column}={value!r}: {exc}") from exc


def load_dataset(path: Path, day: int, config: AppConfig) -> DayDataset:
    """Read `day{day}_*.csv` from `path` into typed, normalized, sorted records.

    `config` supplies the simulation window every event date must fall inside.
    That check is the D13 guard with teeth: `test_no_stale_years` greps the
    repository for a 2025 date, which catches one committed by hand but not one
    a future generator run produces, and a date outside the window is a defect
    whether or not its year reads 2026.

    No merchant profile is passed to parse_date. None of the six tables carries
    a profile column, and every date the frozen generator writes is ISO
    yyyy-mm-dd, which is unambiguous by construction. If a source ever emits an
    ambiguous numeric date this raises AmbiguousDateError rather than guessing,
    which is the correct failure: it says a profile rule is missing.
    """
    if day < 1:
        raise IngestError(f"arrival_day is 1-indexed (SDD 4.1a); got {day}")
    if not path.is_dir():
        raise IngestError(f"{path} is not a directory")

    window = (config.calendar.window_start, config.calendar.window_end)

    def when(file: Path, row: int, column: str, value: str) -> date:
        parsed = _guarded(file, row, column, value, parse_date)
        if not window[0] <= parsed <= window[1]:
            raise IngestError(
                f"{file.name} row {row}: {column}={value!r} parses to "
                f"{parsed.isoformat()}, outside the configured simulation window "
                f"{window[0].isoformat()}..{window[1].isoformat()}. Every date in "
                "this project is inside that window (D13)."
            )
        return parsed

    def file_for(table: str) -> Path:
        return path / f"day{day}_{TABLE_FILES[table]}.csv"

    ledger_file = file_for("ledger_rows")
    ledger = [
        LedgerRow(
            order_id=row["order_id"],
            invoice_no=row["invoice_no"],
            gross=_guarded(ledger_file, n, "gross", row["gross"], parse_amount),
            order_date=when(ledger_file, n, "order_date", row["order_date"]),
            customer_id=row["customer_id"],
            sku=row["sku"],
        )
        for n, row in _read_rows(
            ledger_file, ("order_id", "invoice_no", "gross", "order_date", "customer_id", "sku")
        )
    ]

    payments_file = file_for("payment_rows")
    payments = [
        PaymentRow(
            payment_id=row["payment_id"],
            order_id=row["order_id"],
            method=_enum(PaymentMethod, row["method"], payments_file, n, "method"),
            authorized=_guarded(payments_file, n, "authorized", row["authorized"], parse_amount),
            captured=_guarded(payments_file, n, "captured", row["captured"], parse_amount),
            status=_enum(PaymentStatus, row["status"], payments_file, n, "status"),
            captured_at=when(payments_file, n, "captured_at", row["captured_at"]),
        )
        for n, row in _read_rows(
            payments_file,
            ("payment_id", "order_id", "method", "authorized", "captured", "status", "captured_at"),
        )
    ]

    refunds_file = file_for("refund_rows")
    refunds = [
        RefundRow(
            refund_id=row["refund_id"],
            payment_id=row["payment_id"],
            amount=_guarded(refunds_file, n, "amount", row["amount"], parse_amount),
            created_at=when(refunds_file, n, "created_at", row["created_at"]),
        )
        for n, row in _read_rows(refunds_file, ("refund_id", "payment_id", "amount", "created_at"))
    ]

    settlements_file = file_for("settlement_lines")
    settlements = [
        _settlement_line(settlements_file, n, row, when)
        for n, row in _read_rows(
            settlements_file,
            (
                "settlement_id",
                "batch_id",
                "line_type",
                "payment_id",
                "refund_id",
                "gross",
                "fee",
                "tax",
                "net",
                "settled_event_date",
            ),
        )
    ]

    batches_file = file_for("settlement_batches")
    batches = [
        SettlementBatch(
            batch_id=row["batch_id"],
            utr=row["utr"],
            net_total=_guarded(batches_file, n, "net_total", row["net_total"], parse_amount),
            settled_event_date=when(
                batches_file, n, "settled_event_date", row["settled_event_date"]
            ),
        )
        for n, row in _read_rows(
            batches_file, ("batch_id", "utr", "net_total", "settled_event_date")
        )
    ]

    bank_file = file_for("bank_rows")
    bank = [
        BankRow(
            bank_txn_id=row["bank_txn_id"],
            value_date=when(bank_file, n, "value_date", row["value_date"]),
            amount=_guarded(bank_file, n, "amount", row["amount"], parse_amount),
            narration=row["narration"],
            direction=_enum(BankDirection, row["direction"], bank_file, n, "direction"),
        )
        for n, row in _read_rows(
            bank_file, ("bank_txn_id", "value_date", "amount", "narration", "direction")
        )
    ]

    return DayDataset(
        arrival_day=day,
        ledger_rows=_sorted(ledger),
        payment_rows=_sorted(payments),
        refund_rows=_sorted(refunds),
        settlement_lines=_sorted(settlements),
        settlement_batches=_sorted(batches),
        bank_rows=_sorted(bank),
    )


def _settlement_line(
    file: Path,
    number: int,
    row: Mapping[str, str],
    when: Callable[[Path, int, str, str], date],
) -> SettlementLine:
    """One settlement line, with the per-type field invariant enforced.

    SDD 3.3 states the invariant as a table and names test_line_invariants as
    its check. Enforced HERE as well, because a test proves the frozen dataset
    satisfies it once while this refuses to build a record that does not - and
    the engine downstream reads `refund_id is None` to decide whether a line
    settles a payment or deducts from it.
    """
    line_type = _enum(SettlementLineType, row["line_type"], file, number, "line_type")
    refund_id = row["refund_id"] or None

    if line_type is SettlementLineType.REFUND and refund_id is None:
        raise IngestError(
            f"{file.name} row {number}: a REFUND line has no refund_id. The line "
            "would be indistinguishable from a payment settling."
        )
    if line_type is SettlementLineType.PAYMENT and refund_id is not None:
        raise IngestError(
            f"{file.name} row {number}: a PAYMENT line carries refund_id="
            f"{refund_id!r}. Only REFUND lines reference a refund (SDD 3.3)."
        )

    return SettlementLine(
        settlement_id=row["settlement_id"],
        batch_id=row["batch_id"],
        line_type=line_type,
        payment_id=row["payment_id"],
        refund_id=refund_id,
        gross=_guarded(file, number, "gross", row["gross"], parse_amount),
        fee=_guarded(file, number, "fee", row["fee"], parse_amount),
        tax=_guarded(file, number, "tax", row["tax"], parse_amount),
        net=_guarded(file, number, "net", row["net"], parse_amount),
        settled_event_date=when(file, number, "settled_event_date", row["settled_event_date"]),
    )

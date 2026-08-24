"""M2 - reading a day's CSVs into typed records, and refusing to read bad ones.

Two halves. The first loads the real frozen dataset, because a loader that
only ever sees fixtures it wrote itself proves nothing about the artifact the
engine will actually be measured on. The second corrupts a copy of a real day
one field at a time, because a loader that has never refused anything is a
loader whose refusals are untested.
"""

from __future__ import annotations

import csv
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from settlesense.config import AppConfig, load_config
from settlesense.ingest import TABLE_FILES, DayDataset, IngestError, load_dataset
from settlesense.types import Money, SettlementLineType

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
CONFIG = REPO / "config"

# The frozen dataset, from GENERATOR_MANIFEST.json / the M1F freeze.
DAYS = 24
EXPECTED_COUNTS = {
    "ledger_rows": 5053,
    "payment_rows": 5026,
    "refund_rows": 298,
    "settlement_lines": 5324,
    "settlement_batches": 39,
    "bank_rows": 39,
}


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def all_days(config: AppConfig) -> tuple[DayDataset, ...]:
    return tuple(load_dataset(DATA, day, config) for day in range(1, DAYS + 1))


FIXTURE_DAY = 2
"""The day the corruption fixtures copy.

NOT day 1. Settlement is T+N, so day 1 has no bank credits at all and
day1_bank.csv is header-only - every bank-column refusal test would fail on an
IndexError from the fixture rather than passing or failing on the loader.
Day 2 is the first day with all six tables populated.
"""


@pytest.fixture
def a_real_day(tmp_path: Path) -> Path:
    """A writable copy of one real day, for corrupting a field at a time."""
    for stem in TABLE_FILES.values():
        name = f"day{FIXTURE_DAY}_{stem}.csv"
        shutil.copy(DATA / name, tmp_path / name)
    return tmp_path


def _corrupt(directory: Path, stem: str, row_index: int, column: str, value: str) -> None:
    """Rewrite one cell, leaving every other byte alone."""
    path = directory / f"day{FIXTURE_DAY}_{stem}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or ())
        rows = list(reader)
    assert column in header, f"{column} is not a column of {path.name}"
    rows[row_index][column] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# The real dataset
# ---------------------------------------------------------------------------


def test_every_day_of_the_frozen_dataset_loads(all_days: tuple[DayDataset, ...]) -> None:
    assert len(all_days) == DAYS
    totals = {name: sum(len(getattr(day, name)) for day in all_days) for name in EXPECTED_COUNTS}
    assert totals == EXPECTED_COUNTS, (
        f"loaded {totals}, expected {EXPECTED_COUNTS}. A count that drifted means "
        "either the dataset changed after the freeze or rows are being dropped."
    )


def test_arrival_day_is_recorded_as_an_int(all_days: tuple[DayDataset, ...]) -> None:
    """SDD 4.1a: arrival_day is an integer sequence, never a timestamp."""
    for expected, day in enumerate(all_days, start=1):
        assert day.arrival_day == expected
        assert type(day.arrival_day) is int


def test_a_comma_formatted_amount_does_not_shift_the_columns_after_it(
    config: AppConfig,
) -> None:
    """THE test that a naive split(",") loader would fail.

    day1_settlements.csv contains net='8,911.24' - quoted, because the
    mixed_amount_formats injector renders some amounts with thousands
    separators. Splitting on commas puts "24" in the next field and shifts
    every column after it, so settled_event_date would receive an amount
    fragment. The corruption surfaces as a date parse error in a module that
    did nothing wrong, which is the expensive kind of bug.
    """
    day = load_dataset(DATA, 1, config)
    line = next(line for line in day.settlement_lines if line.net == Decimal("8911.24"))
    assert line.settled_event_date.isoformat() == "2026-09-01", (
        "the date column was shifted by a quoted comma"
    )
    assert line.settled_event_date.year == 2026


def test_parenthesised_debits_load_as_negative(all_days: tuple[DayDataset, ...]) -> None:
    """The dataset contains 6 parenthesised amounts. All are REFUND line nets,
    which are signed negative (SDD 3.1a) - so a loader that ignored the
    parentheses would flip a debit into a credit and break batch conservation.
    """
    negatives = [
        line for day in all_days for line in day.settlement_lines if line.net < Decimal("0")
    ]
    assert negatives, "no negative line nets loaded; the sign handling is untested"
    for line in negatives:
        assert line.line_type is SettlementLineType.REFUND
        assert line.refund_id is not None


def test_every_money_value_is_a_quantized_decimal(all_days: tuple[DayDataset, ...]) -> None:
    """Swept across all 15,779 rows rather than a sample.

    `type(...) is Decimal` rather than isinstance: isinstance passes for a
    Decimal subclass, and `assert not isinstance(v, float)` on a value mypy
    already knows is Decimal is unreachable code.
    """
    checked = 0
    for day in all_days:
        for line in day.settlement_lines:
            for value in (line.gross, line.fee, line.tax, line.net):
                assert type(value) is Decimal
                exponent = value.as_tuple().exponent
                assert isinstance(exponent, int) and exponent == -2, (
                    f"{value} is not quantized to paise"
                )
                checked += 1
    assert checked > 20_000, f"swept only {checked} money values"


def test_rows_are_sorted_by_primary_id(all_days: tuple[DayDataset, ...]) -> None:
    for day in all_days:
        assert [r.payment_id for r in day.payment_rows] == sorted(
            r.payment_id for r in day.payment_rows
        )
        assert [r.bank_txn_id for r in day.bank_rows] == sorted(
            r.bank_txn_id for r in day.bank_rows
        )
        assert [r.order_id for r in day.ledger_rows] == sorted(r.order_id for r in day.ledger_rows)


def test_the_sort_is_total_even_where_the_primary_id_repeats(
    all_days: tuple[DayDataset, ...], config: AppConfig
) -> None:
    """order_id is NOT unique - duplicate_ledger_rows emits genuine duplicates.

    A sort keyed on order_id alone leaves their relative order to whatever the
    reader yielded, which is stable in practice and therefore hides the
    problem until something upstream changes. This asserts the full-record key
    instead: reloading must reproduce the identical sequence.
    """
    duplicated = [
        day
        for day in all_days
        if len({r.order_id for r in day.ledger_rows}) != len(day.ledger_rows)
    ]
    assert duplicated, (
        "no day contains a duplicated order_id, so this test is not exercising "
        "the tie-break it exists to check"
    )
    for day in duplicated:
        reloaded = load_dataset(DATA, day.arrival_day, config)
        assert reloaded.ledger_rows == day.ledger_rows


def test_loading_is_deterministic(config: AppConfig) -> None:
    assert load_dataset(DATA, 3, config) == load_dataset(DATA, 3, config)


def test_every_date_falls_inside_the_configured_window(
    all_days: tuple[DayDataset, ...], config: AppConfig
) -> None:
    """D13, checked against config rather than a hardcoded 2026."""
    start, end = config.calendar.window_start, config.calendar.window_end
    checked = 0
    for day in all_days:
        for value in (
            [r.order_date for r in day.ledger_rows]
            + [r.captured_at for r in day.payment_rows]
            + [r.created_at for r in day.refund_rows]
            + [r.value_date for r in day.bank_rows]
        ):
            assert start <= value <= end and value.year == 2026
            checked += 1
    assert checked > 10_000, f"checked only {checked} dates"


def test_row_count_is_not_offered_as_a_denominator(all_days: tuple[DayDataset, ...]) -> None:
    """D11. row_count exists for ingestion diagnostics; Population A divides by
    ReconciliationCase. Asserting they differ keeps the distinction visible."""
    day = all_days[0]
    assert day.row_count() != len(day.payment_rows)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.boundary_refusal
def test_a_missing_file_raises_rather_than_yielding_an_empty_table(
    tmp_path: Path, config: AppConfig
) -> None:
    """FAULT INJECTION. An empty table would report a clean reconciliation for
    a day whose data never arrived - a false negative that looks like success."""
    with pytest.raises(IngestError, match="does not exist"):
        load_dataset(tmp_path, 1, config)


def test_an_empty_table_is_legitimate_but_a_missing_file_is_not(config: AppConfig) -> None:
    """The two look identical downstream and mean opposite things.

    Settlement is T+N, so day 1 genuinely has no bank credits and
    day1_bank.csv is header-only. That is correct data. A file that does not
    exist is a delivery failure. Collapsing both to "no rows" would report a
    clean reconciliation for a day whose statement never arrived, and nothing
    in the output would distinguish it from a day that truly had no credits.
    """
    day_one = load_dataset(DATA, 1, config)
    assert day_one.bank_rows == ()
    assert day_one.payment_rows, "day 1 should still have payments"
    with pytest.raises(IngestError, match="does not exist"):
        load_dataset(DATA, DAYS + 1, config)


@pytest.mark.boundary_refusal
def test_a_renamed_column_raises_rather_than_being_read_positionally(
    a_real_day: Path, config: AppConfig
) -> None:
    """FAULT INJECTION for the header check."""
    path = a_real_day / f"day{FIXTURE_DAY}_bank.csv"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("value_date", "date", 1), encoding="utf-8")
    with pytest.raises(IngestError, match="header is"):
        load_dataset(a_real_day, FIXTURE_DAY, config)


@pytest.mark.boundary_refusal
def test_an_unknown_payment_method_raises_rather_than_defaulting(
    a_real_day: Path, config: AppConfig
) -> None:
    """FAULT INJECTION. Defaulting would let an unpriced method be charged as
    if it were a known one, and the fee would be wrong with no error anywhere."""
    _corrupt(a_real_day, "payments", 0, "method", "crypto")
    with pytest.raises(IngestError, match="method='crypto'"):
        load_dataset(a_real_day, FIXTURE_DAY, config)


@pytest.mark.boundary_refusal
def test_an_unknown_bank_direction_raises(a_real_day: Path, config: AppConfig) -> None:
    """FAULT INJECTION."""
    _corrupt(a_real_day, "bank", 0, "direction", "sideways")
    with pytest.raises(IngestError, match="direction='sideways'"):
        load_dataset(a_real_day, FIXTURE_DAY, config)


@pytest.mark.boundary_refusal
def test_an_unparseable_amount_raises_naming_the_file_row_and_column(
    a_real_day: Path, config: AppConfig
) -> None:
    """FAULT INJECTION. An error that says only "invalid input" costs more to
    diagnose than the parse cost to prevent."""
    _corrupt(a_real_day, "ledger", 4, "gross", "not-a-number")
    with pytest.raises(IngestError) as caught:
        load_dataset(a_real_day, FIXTURE_DAY, config)
    message = str(caught.value)
    assert f"day{FIXTURE_DAY}_ledger.csv" in message and "gross=" in message and "row 5" in message


@pytest.mark.boundary_refusal
def test_a_date_outside_the_simulation_window_raises(a_real_day: Path, config: AppConfig) -> None:
    """FAULT INJECTION for D13 with teeth.

    test_no_stale_years greps the repository for a 2025 date, which catches one
    committed by hand but not one a future generator run produces. 2026-12-25
    is inside 2026 and still wrong, so a year check alone would miss it.
    """
    _corrupt(a_real_day, "bank", 0, "value_date", "2026-12-25")
    with pytest.raises(IngestError, match="outside the configured simulation window"):
        load_dataset(a_real_day, FIXTURE_DAY, config)


@pytest.mark.boundary_refusal
def test_a_2025_date_raises(a_real_day: Path, config: AppConfig) -> None:
    """FAULT INJECTION. D13: a 2025 date anywhere is a defect."""
    _corrupt(a_real_day, "ledger", 0, "order_date", "2025-09-01")
    with pytest.raises(IngestError, match="outside the configured simulation window"):
        load_dataset(a_real_day, FIXTURE_DAY, config)


@pytest.mark.boundary_refusal
def test_a_payment_line_carrying_a_refund_id_raises(a_real_day: Path, config: AppConfig) -> None:
    """FAULT INJECTION for the SDD 3.3 per-type invariant.

    The engine reads `refund_id is None` to decide whether a line settles a
    payment or deducts from it, so this field decides which side of the
    conservation arithmetic the line lands on.
    """
    rows = list(csv.DictReader((a_real_day / f"day{FIXTURE_DAY}_settlements.csv").open(newline="")))
    index = next(i for i, r in enumerate(rows) if r["line_type"] == "payment")
    _corrupt(a_real_day, "settlements", index, "refund_id", "RFD_DEADBEEF0000")
    with pytest.raises(IngestError, match="PAYMENT line carries refund_id"):
        load_dataset(a_real_day, FIXTURE_DAY, config)


@pytest.mark.boundary_refusal
def test_a_refund_line_without_a_refund_id_raises(a_real_day: Path, config: AppConfig) -> None:
    """FAULT INJECTION for the same invariant from the other side."""
    rows = list(csv.DictReader((a_real_day / f"day{FIXTURE_DAY}_settlements.csv").open(newline="")))
    index = next((i for i, r in enumerate(rows) if r["line_type"] == "refund"), None)
    if index is None:
        pytest.skip(f"day {FIXTURE_DAY} contains no REFUND line")
    _corrupt(a_real_day, "settlements", index, "refund_id", "")
    with pytest.raises(IngestError, match="REFUND line has no refund_id"):
        load_dataset(a_real_day, FIXTURE_DAY, config)


@pytest.mark.boundary_refusal
def test_a_float_amount_in_a_csv_cannot_reach_a_record(a_real_day: Path, config: AppConfig) -> None:
    """FAULT INJECTION for D1 at the file boundary.

    Sub-paise text is quantized by design; the guard being tested is that the
    value arrives as Decimal and never as float, whatever the text said.
    """
    _corrupt(a_real_day, "ledger", 0, "gross", "1234.5678")
    day = load_dataset(a_real_day, FIXTURE_DAY, config)
    value: Money = day.ledger_rows[0].gross
    assert type(value) is Decimal
    assert Decimal("1234.56") <= value <= Decimal("1234.57")


@pytest.mark.boundary_refusal
def test_day_zero_raises(config: AppConfig) -> None:
    """FAULT INJECTION. arrival_day is 1-indexed (SDD 4.1a)."""
    with pytest.raises(IngestError, match="1-indexed"):
        load_dataset(DATA, 0, config)


@pytest.mark.boundary_refusal
def test_a_data_path_that_is_not_a_directory_raises(config: AppConfig) -> None:
    """FAULT INJECTION."""
    with pytest.raises(IngestError, match="not a directory"):
        load_dataset(DATA / "day1_bank.csv", 1, config)


def test_an_uncorrupted_copy_still_loads(a_real_day: Path, config: AppConfig) -> None:
    """Guards every refusal test above from the other side.

    If the fixture itself were broken - a bad copy, a rewrite that mangled the
    file - every refusal test would pass for the wrong reason, each one
    "correctly" raising on damage nobody intended.
    """
    day = load_dataset(a_real_day, FIXTURE_DAY, config)
    assert len(day.payment_rows) > 0
    assert day == load_dataset(DATA, FIXTURE_DAY, config)


def test_the_corruption_helper_actually_changes_one_cell(a_real_day: Path) -> None:
    """Guards the helper. A no-op _corrupt would make every refusal test above
    pass only because the real data happened to be fine."""
    before = (a_real_day / f"day{FIXTURE_DAY}_bank.csv").read_text(encoding="utf-8")
    _corrupt(a_real_day, "bank", 0, "direction", "sideways")
    after = (a_real_day / f"day{FIXTURE_DAY}_bank.csv").read_text(encoding="utf-8")
    assert before != after and "sideways" in after

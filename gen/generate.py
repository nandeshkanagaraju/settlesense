"""M1 part A - the generator CLI.

    python -m gen.generate --seed 42 --out data/

Writes SIX tables per simulated day. Chargebacks and disputes are out of v1
scope, so there is no seventh table, no DISPUTE_DEBIT line type, and no dispute
edge anywhere in gen/.

    day{N}_ledger.csv       day{N}_payments.csv     day{N}_refunds.csv
    day{N}_settlements.csv  day{N}_batches.csv      day{N}_bank.csv

A day bundle holds the rows whose OWN event_date falls on that day, which is
what a merchant actually hands over: today's orders, and the bank credit for a
payment captured two days ago. That is also what makes the Day 1 -> Day 2 ->
Day 3 demo real - day 2's file closes an exception day 1 could not.

`--days` is therefore the number of CAPTURE days. Settlement and bank rows
trail behind them by the profile's T+N, so the run emits bundles past --days
until every chain has closed. Truncating instead would leave clean chains
unmatched, and clean chains that do not close cannot verify their own truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from gen.lifecycle import (
    BankRow,
    CleanDataset,
    LedgerRow,
    PaymentRow,
    RefundRow,
    SettlementBatch,
    SettlementLine,
    SettlementLineType,
    WorkingCalendar,
    build_clean_dataset,
    load_working_calendar,
    verify_clean_dataset,
)
from gen.noise import NoiseLedger, NoiseRates, apply_noise
from gen.profiles import PROFILES, MerchantProfile
from gen.truth import (
    EdgeType,
    Truth,
    TruthSelfCheckError,
    build_truth,
    run_self_check,
    write_truth,
)

DEFAULT_CALENDAR: Final[Path] = Path("config/calendar_v1.yaml")
DEFAULT_DAYS: Final[int] = 3
DEFAULT_RECORDS: Final[int] = 5_000

LEDGER_COLUMNS: Final[tuple[str, ...]] = (
    "order_id",
    "invoice_no",
    "gross",
    "order_date",
    "customer_id",
    "sku",
)
PAYMENT_COLUMNS: Final[tuple[str, ...]] = (
    "payment_id",
    "order_id",
    "method",
    "authorized",
    "captured",
    "status",
    "captured_at",
)
REFUND_COLUMNS: Final[tuple[str, ...]] = ("refund_id", "payment_id", "amount", "created_at")
SETTLEMENT_COLUMNS: Final[tuple[str, ...]] = (
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
)
BATCH_COLUMNS: Final[tuple[str, ...]] = ("batch_id", "utr", "net_total", "settled_event_date")
BANK_COLUMNS: Final[tuple[str, ...]] = (
    "bank_txn_id",
    "value_date",
    "amount",
    "narration",
    "direction",
)


def build_plan(
    records: int, days: int, profiles: Sequence[MerchantProfile]
) -> list[tuple[int, MerchantProfile, int]]:
    """Split `records` across days then profiles, deterministically.

    Remainders go to the earliest days and the earliest profiles in sorted
    order, so the split never depends on iteration order (D4).
    """
    if records < 1:
        raise ValueError(f"--records must be at least 1, got {records}")
    if days < 1:
        raise ValueError(f"--days must be at least 1, got {days}")
    if records < days * len(profiles):
        raise ValueError(
            f"--records {records} is too small for {days} days x {len(profiles)} profiles; "
            f"need at least {days * len(profiles)}"
        )

    ordered = sorted(profiles, key=lambda profile: profile.name)
    per_day = [records // days + (1 if index < records % days else 0) for index in range(days)]

    plan: list[tuple[int, MerchantProfile, int]] = []
    for day_index, day_total in enumerate(per_day, start=1):
        share = [
            day_total // len(ordered) + (1 if index < day_total % len(ordered) else 0)
            for index in range(len(ordered))
        ]
        for profile, count in zip(ordered, share, strict=True):
            for sequence in range(count):
                plan.append((day_index, profile, sequence))
    return plan


def _day_index(value: date, base_date: date) -> int:
    return (value - base_date).days + 1


def _fmt(value: object) -> str:
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    table: str = "",
    formats: Mapping[tuple[str, str, str], str] | None = None,
) -> int:
    """Write one CSV with a fixed column order and LF line endings.

    A format override replaces the TEXT of one cell. The row's Decimal is
    unchanged - this is a rendering variation the normalizer must survive, not
    a different number.
    """
    count = 0
    overrides = formats or {}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            row_id = str(row[0])
            cells = [
                overrides.get((table, row_id, column), _fmt(value))
                for column, value in zip(columns, row, strict=True)
            ]
            writer.writerow(cells)
            count += 1
    return count


def _bucket(
    rows: Iterable[Any],
    key: Any,
    base_date: date,
    table: str,
    row_id: Any,
    lag: Any,
) -> dict[int, list[Any]]:
    """Bucket rows by the day they are DELIVERED, which is their event day plus
    any out-of-order-arrival lag. event_date on the row itself never moves."""
    buckets: dict[int, list[Any]] = {}
    for row in rows:
        day = _day_index(key(row), base_date) + lag(table, row_id(row))
        buckets.setdefault(day, []).append(row)
    return buckets


def write_dataset(
    dataset: CleanDataset,
    out_dir: Path,
    base_date: date,
    noise: NoiseLedger | None = None,
) -> dict[str, Any]:
    """Write six CSVs per day bundle. Returns a manifest describing the run.

    Two noise kinds land here rather than in the row model, because neither is a
    change to the data: `mixed_amount_formats` overrides the TEXT of a cell
    while the Decimal behind it is untouched, and `out_of_order_arrival`
    overrides which day bundle DELIVERS a row while its event_date stands.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = dict(noise.format_overrides) if noise else {}
    delays = dict(noise.day_overrides) if noise else {}

    def lag(table: str, row_id: str) -> int:
        return delays.get((table, row_id), 0)

    ledger = _bucket(
        dataset.ledger_rows, lambda r: r.order_date, base_date, "ledger", lambda r: r.order_id, lag
    )
    payments = _bucket(
        dataset.payment_rows,
        lambda r: r.captured_at,
        base_date,
        "payments",
        lambda r: r.payment_id,
        lag,
    )
    refunds = _bucket(
        dataset.refund_rows,
        lambda r: r.created_at,
        base_date,
        "refunds",
        lambda r: r.refund_id,
        lag,
    )
    settlements = _bucket(
        dataset.settlement_lines,
        lambda r: r.settled_event_date,
        base_date,
        "settlements",
        lambda r: r.settlement_id,
        lag,
    )
    batches = _bucket(
        dataset.batches,
        lambda r: r.settled_event_date,
        base_date,
        "batches",
        lambda r: r.batch_id,
        lag,
    )
    bank = _bucket(
        dataset.bank_rows, lambda r: r.value_date, base_date, "bank", lambda r: r.bank_txn_id, lag
    )

    last_day = max(
        [1]
        + [
            day
            for mapping in (ledger, payments, refunds, settlements, batches, bank)
            for day in mapping
        ]
    )

    def ledger_row(row: LedgerRow) -> tuple[object, ...]:
        return (row.order_id, row.invoice_no, row.gross, row.order_date, row.customer_id, row.sku)

    def payment_row(row: PaymentRow) -> tuple[object, ...]:
        return (
            row.payment_id,
            row.order_id,
            row.method,
            row.authorized,
            row.captured,
            row.status,
            row.captured_at,
        )

    def refund_row(row: RefundRow) -> tuple[object, ...]:
        return (row.refund_id, row.payment_id, row.amount, row.created_at)

    def settlement_row(row: SettlementLine) -> tuple[object, ...]:
        return (
            row.settlement_id,
            row.batch_id,
            row.line_type.value,
            row.payment_id,
            row.refund_id,
            row.gross,
            row.fee,
            row.tax,
            row.net,
            row.settled_event_date,
        )

    def batch_row(row: SettlementBatch) -> tuple[object, ...]:
        return (row.batch_id, row.utr, row.net_total, row.settled_event_date)

    def bank_row(row: BankRow) -> tuple[object, ...]:
        return (row.bank_txn_id, row.value_date, row.amount, row.narration, row.direction)

    days: list[dict[str, Any]] = []
    for day in range(1, last_day + 1):
        content_date = base_date + timedelta(days=day - 1)
        counts = {
            "ledger": _write_csv(
                out_dir / f"day{day}_ledger.csv",
                LEDGER_COLUMNS,
                (ledger_row(r) for r in sorted(ledger.get(day, []), key=lambda r: r.order_id)),
                "ledger",
                formats,
            ),
            "payments": _write_csv(
                out_dir / f"day{day}_payments.csv",
                PAYMENT_COLUMNS,
                (payment_row(r) for r in sorted(payments.get(day, []), key=lambda r: r.payment_id)),
                "payments",
                formats,
            ),
            "refunds": _write_csv(
                out_dir / f"day{day}_refunds.csv",
                REFUND_COLUMNS,
                (refund_row(r) for r in sorted(refunds.get(day, []), key=lambda r: r.refund_id)),
                "refunds",
                formats,
            ),
            "settlements": _write_csv(
                out_dir / f"day{day}_settlements.csv",
                SETTLEMENT_COLUMNS,
                (
                    settlement_row(r)
                    for r in sorted(settlements.get(day, []), key=lambda r: r.settlement_id)
                ),
                "settlements",
                formats,
            ),
            "batches": _write_csv(
                out_dir / f"day{day}_batches.csv",
                BATCH_COLUMNS,
                (batch_row(r) for r in sorted(batches.get(day, []), key=lambda r: r.batch_id)),
                "batches",
                formats,
            ),
            "bank": _write_csv(
                out_dir / f"day{day}_bank.csv",
                BANK_COLUMNS,
                (bank_row(r) for r in sorted(bank.get(day, []), key=lambda r: r.bank_txn_id)),
                "bank",
                formats,
            ),
        }
        days.append(
            {
                "arrival_day": day,
                "file_content_date": content_date.isoformat(),
                "rows": counts,
            }
        )

    return {"base_date": base_date.isoformat(), "day_count": last_day, "days": days}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m gen.generate",
        description="Generate a synthetic settlement dataset. All dates are in 2026.",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed for the single random.Random")
    parser.add_argument("--out", type=Path, required=True, help="output directory for CSVs")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="number of capture days")
    parser.add_argument(
        "--records", type=int, default=DEFAULT_RECORDS, help="number of payment chains"
    )
    parser.add_argument(
        "--include-withheld",
        action="store_true",
        default=False,
        help="include the two withheld noise types (M1 part B)",
    )
    parser.add_argument(
        "--calendar", type=Path, default=DEFAULT_CALENDAR, help="path to calendar_v1.yaml"
    )
    return parser.parse_args(argv)


def _check_profiles_against_calendar(calendar: WorkingCalendar) -> None:
    """The generator states T+N independently; disagreement is a loud failure.

    gen/ defines its own settlement cycles on purpose. Asserting they match the
    calendar does not weaken that independence - the code paths remain separate
    - it just refuses to run when the two statements of the same fact drift.
    """
    for profile in PROFILES:
        configured = calendar.settlement_cycles.get(profile.name)
        if configured is None:
            raise SystemExit(
                f"calendar has no settlement cycle for {profile.name!r}; "
                f"known: {sorted(calendar.settlement_cycles)}"
            )
        if configured != profile.settlement_cycle_days:
            raise SystemExit(
                f"settlement cycle drift for {profile.name}: gen/profiles.py says "
                f"T+{profile.settlement_cycle_days}, {calendar.version} says T+{configured}. "
                f"Reconcile them before generating."
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    calendar = load_working_calendar(args.calendar)
    _check_profiles_against_calendar(calendar)

    base_date = calendar.next_working_day(calendar.window_start)
    rng = random.Random(args.seed)
    plan = build_plan(args.records, args.days, PROFILES)

    dataset = build_clean_dataset(rng, plan, calendar, base_date, args.seed)

    problems = verify_clean_dataset(dataset)
    if problems:
        print(
            f"GROUND TRUTH SELF-CHECK FAILED - {len(problems)} violation(s). Nothing was written.",
            file=sys.stderr,
        )
        for problem in problems[:40]:
            print(f"  {problem}", file=sys.stderr)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more", file=sys.stderr)
        return 1

    # Truth is built and self-checked BEFORE a single byte reaches disk. A truth
    # file written from a dataset that does not balance makes every downstream
    # metric confidently wrong, which is worse than having no truth file.
    profiles_by_name = {profile.name: profile for profile in PROFILES}

    # PASS 1 - the clean dataset, with no allowance for noise whatsoever.
    clean_truth = build_truth(dataset, calendar, profiles_by_name, args.seed)
    try:
        run_self_check(dataset, clean_truth, calendar=calendar, profiles=profiles_by_name)
    except TruthSelfCheckError as error:
        print("CLEAN self-check failed:", file=sys.stderr)
        print(str(error), file=sys.stderr)
        return 1

    # PASS 2 - inject noise, then run the SAME check again. It is extended with
    # the noise ledger, not relaxed: every clean chain must still balance, and
    # every injected error must be claimed by an annotation AND observable.
    noisy, ledger = apply_noise(
        dataset,
        rng,
        profiles_by_name,
        NoiseRates(),
        include_withheld=args.include_withheld,
    )
    truth = build_truth(noisy, calendar, profiles_by_name, args.seed, noise=ledger)
    try:
        run_self_check(noisy, truth, calendar=calendar, profiles=profiles_by_name, noise=ledger)
    except TruthSelfCheckError as error:
        print("POST-NOISE self-check failed:", file=sys.stderr)
        print(str(error), file=sys.stderr)
        return 1
    dataset = noisy

    manifest = write_dataset(dataset, args.out, base_date, ledger)
    manifest.update(
        {
            "seed": args.seed,
            "capture_days": args.days,
            "records": args.records,
            "calendar_version": calendar.version,
            "include_withheld": args.include_withheld,
            "noise_counts": dict(ledger.counts),
            "generator_commit": None,  # written only in GENERATOR_MANIFEST.json, after the freeze
            "counts": {
                "chains": len(dataset.chains),
                "ledger_rows": len(dataset.ledger_rows),
                "payment_rows": len(dataset.payment_rows),
                "refund_rows": len(dataset.refund_rows),
                "settlement_lines": len(dataset.settlement_lines),
                "batches": len(dataset.batches),
                "bank_rows": len(dataset.bank_rows),
            },
        }
    )
    truth_path = args.out / f"truth_{args.seed}.json"
    write_truth(truth, truth_path)
    manifest["truth_file"] = truth_path.name
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _report(args, calendar, base_date, dataset, truth, manifest, truth_path, ledger)
    return 0


def _report(
    args: argparse.Namespace,
    calendar: WorkingCalendar,
    base_date: date,
    dataset: CleanDataset,
    truth: Truth,
    manifest: dict[str, Any],
    truth_path: Path,
    ledger: NoiseLedger,
) -> None:
    """Print the self-check result and a summary of what was generated."""
    counts = manifest["counts"]
    total_rows = sum(int(value) for key, value in counts.items() if key != "chains")
    per_batch = [
        sum(1 for line in dataset.settlement_lines if line.batch_id == batch.batch_id)
        for batch in dataset.batches
    ]
    edge_counts = {
        edge_type: sum(1 for edge in truth.edges if edge.edge_type is edge_type)
        for edge_type in EdgeType
    }
    payment_lines = sum(
        1 for line in dataset.settlement_lines if line.line_type is SettlementLineType.PAYMENT
    )
    refund_lines = len(dataset.settlement_lines) - payment_lines
    mixed = sum(
        1
        for batch in dataset.batches
        if any(
            line.batch_id == batch.batch_id and line.line_type is SettlementLineType.REFUND
            for line in dataset.settlement_lines
        )
    )

    print(f"seed {args.seed} | calendar {calendar.version} | base date {base_date.isoformat()}")
    print("")
    print("GROUND-TRUTH SELF-CHECK: PASSED")
    print("  cardinality  1 ORDER_TO_PAYMENT 1:1 .......... ok")
    print("               2 PAYMENT_TO_SETTLEMENT 1:N ..... ok  (PAYMENT lines only)")
    print("               3 SETTLEMENT_TO_BATCH N:1 ....... ok  (many lines per batch is normal)")
    print("               4 BATCH_TO_BANK 1:1 ............. ok")
    print("               5 PAYMENT_TO_REFUND 1:N ......... ok  (no upper bound)")
    print("  balance      6 chains balance to the cent .... ok")
    print(f"               7 batch net == signed line sum .. ok  ({mixed} mixed batches)")
    print("              7a bank credit == batch net ...... ok  (incl. refund batches)")
    print("              7b per-type field invariants ..... ok  (both shapes present)")
    print("              7c PAYMENT_TO_SETTLEMENT purity .. ok  (0 refund-line edges)")
    print("               8 truth IDs exist in dataset .... ok")
    print("               9 fee recomputed from rate ...... ok")
    print("              10 conservation, both identities . ok")
    print("  dates       11 every date in 2026 ............ ok")
    print("              12 arrival_day is a positive int . ok  (no wall clock written)")
    print(f"              13 T+N respected vs {calendar.version} .... ok")
    print("")
    print("GENERATED")
    print(f"  {'table':<22} {'rows':>8}   note")
    print(f"  {'-' * 22} {'-' * 8}   {'-' * 44}")
    print(f"  {'ledger_rows':<22} {counts['ledger_rows']:>8}   one per order")
    print(f"  {'payment_rows':<22} {counts['payment_rows']:>8}   one per capture = Population A")
    print(f"  {'refund_rows':<22} {counts['refund_rows']:>8}   partial and full")
    print(
        f"  {'settlement_lines':<22} {counts['settlement_lines']:>8}   "
        f"{payment_lines} payment + {refund_lines} refund, signed"
    )
    print(
        f"  {'settlement_batches':<22} {counts['batches']:>8}   "
        f"Population B; {min(per_batch)}-{max(per_batch)} lines each"
    )
    print(f"  {'bank_rows':<22} {counts['bank_rows']:>8}   one credit per batch, 1:1")
    print(f"  {'-' * 22} {'-' * 8}")
    print(f"  {'total rows':<22} {total_rows:>8}   across {manifest['day_count']} day bundles")
    print("")
    print("TRUTH EDGES (typed, per-type cardinality)")
    for edge_type in EdgeType:
        print(f"  {edge_type.value:<24} {edge_counts[edge_type]:>8}")
    print(f"  {'-' * 24} {'-' * 8}")
    print(f"  {'total':<24} {len(truth.edges):>8}")
    print("")
    print(
        f"  {len(truth.cases)} ReconciliationCases, "
        f"{sum(1 for c in truth.cases if c.true_category is not None)} with a true variance "
        f"category (clean chains have none)"
    )
    print("  generator_commit: null   (published at M1F in GENERATOR_MANIFEST.json)")
    print(f"  -> {args.out}  +  {truth_path.name}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

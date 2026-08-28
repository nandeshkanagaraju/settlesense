"""M9 - the export runner. Reads the eval artifact, drives the Tally exporter.

WHY THIS LIVES IN eval/ AND NOT IN settlesense/export/. The provenance header
carries the measured residual false-match rate, which is TRUTH-DERIVED - it
comes from `eval/metrics.py`'s Population A, computed against the generator's
ground truth. `settlesense/` may not reach truth and may not read `reports/`;
that boundary is what makes the engine's numbers a measurement rather than a
self-report. So the artifact is read on this side of the fence and the rate is
handed to `build_batch` as an argument, the same way `as_of` is injected rather
than read from a clock.

THE RATE HAS NO DEFAULT ANYWHERE ON THIS PATH. If `results.json` is absent, or
present without the Population A rate, this refuses. A batch exported with the
field blank would be a document asserting that its own accuracy is unknown
while looking exactly like one asserting the accuracy is fine.

Everything written here is a DRY RUN. Nothing transmits.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from eval.run_eval import load_days
from settlesense.config import load_config
from settlesense.exceptions.store import ALL_STATUSES, ExceptionStore
from settlesense.export.tally import (
    ExportError,
    ExportProvenance,
    build_batch,
    close_exported,
    write_dry_run,
)
from settlesense.types import ExceptionStatus
from settlesense.ui.queue import current_categories

RATE_KEY = "residual_false_match_rate_case_count"
POPULATION_A_KEY = "population_a_case_count_denominator"


def provenance_from_results(results_path: Path, dataset_label: str) -> ExportProvenance:
    """Read seed, config_hash and the measured rate out of an eval artifact.

    MISSING AND MALFORMED ARE DIFFERENT, and both refuse. A results.json that
    exists but predates Population A would otherwise reach `.get(...)` and
    return None, and the only thing standing between None and a blank header
    field is that ExportProvenance requires a Decimal.
    """
    if not results_path.is_file():
        raise ExportError(
            f"{results_path} does not exist. The provenance header states the "
            "measured residual false-match rate for this dataset, and there is no "
            "measurement to state. Run `make eval` for the dev set first."
        )
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ExportError(f"{results_path} is not valid JSON: {error}") from error

    population_a = payload.get(POPULATION_A_KEY)
    if not isinstance(population_a, dict) or RATE_KEY not in population_a:
        raise ExportError(
            f"{results_path} has no {POPULATION_A_KEY}.{RATE_KEY}. That figure IS "
            "the header; an export without it would claim provenance it does not have."
        )
    try:
        rate = Decimal(str(population_a[RATE_KEY]))
    except InvalidOperation as error:
        raise ExportError(
            f"{RATE_KEY} is {population_a[RATE_KEY]!r}, which is not a decimal"
        ) from error

    for field in ("seed", "config_hash"):
        if field not in payload:
            raise ExportError(f"{results_path} has no {field!r}")

    return ExportProvenance(
        dataset=dataset_label,
        seed=int(payload["seed"]),
        config_hash=str(payload["config_hash"]),
        residual_false_match_rate=rate,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export CONFIRMED exceptions as Tally XML.")
    parser.add_argument("--db", type=Path, default=Path("reports/ui/state.db"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--data", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/eval/results.json"),
        help="the eval artifact the provenance header is read from",
    )
    parser.add_argument("--dataset", default="dev", help="dataset label for the header")
    parser.add_argument("--out", type=Path, default=Path("reports/export"))
    parser.add_argument(
        "--batch-date",
        default="2026-11-30",
        help="ISO date stamped on every voucher. No clock is read (D2).",
    )
    parser.add_argument(
        "--arrival-day",
        type=int,
        default=24,
        help="the arrival day the CONFIRMED -> CLOSED transition is audited under",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="transition exported exceptions CONFIRMED -> CLOSED. Terminal; off by default.",
    )
    args = parser.parse_args(argv)

    if not args.db.is_file():
        print(f"{args.db} does not exist. Run `make ui-state` first.", file=sys.stderr)
        return 2

    config = load_config(args.config)  # refuses a bad config before anything is written
    batch_date = date.fromisoformat(args.batch_date)

    try:
        provenance = provenance_from_results(args.results, args.dataset)
    except ExportError as error:
        print(f"refusing to export: {error}", file=sys.stderr)
        return 2

    # THE RESOLVING CATEGORY, NOT THE DETECTED ONE. `current_categories` asks
    # the engine what it says about every subject after ALL the files have
    # arrived; the store holds what was true on the day the exception opened.
    # On this dataset those disagree on 274 of 283 confirmed rows, so the
    # difference is not academic - it is the difference between 16 vouchers and
    # 274 wrong ones.
    dataset = load_days(args.data, config)
    by_subject = current_categories(dataset, config, batch_date)

    with ExceptionStore(args.db) as store:
        confirmed = tuple(
            row for row in store.get_queue(ALL_STATUSES) if row.status is ExceptionStatus.CONFIRMED
        )
        resolved = {
            row.exception_id: by_subject.get(store.subject_id(row.exception_id) or "")
            for row in confirmed
            # A subject the store cannot name at all is a different failure from
            # one the engine no longer reports, and build_batch refuses the
            # second. Leaving it out of the mapping is what makes it refuse.
            if store.subject_id(row.exception_id) in by_subject
        }
        try:
            batch = build_batch(confirmed, resolved, provenance, batch_date)
        except ExportError as error:
            print(f"refusing to export: {error}", file=sys.stderr)
            return 2

        path = write_dry_run(batch, args.out)
        closed: tuple[str, ...] = ()
        if args.close:
            closed = close_exported(store, batch, args.arrival_day)

        # THE FIGURES THE README QUOTES, WRITTEN DOWN. The XML carries the 32
        # ledger amounts that sum to the total, and a reader who wants the
        # total has to add them up - so the README's balanced-journal line was
        # a number no committed file contained. It was the third figure this
        # project has published without an artifact behind it, after the
        # Rs2.49 per-1,000-rows cost and the "~90 days" density.
        #
        # Deterministic and tiny. `--close` is excluded from it deliberately:
        # it mutates state a later run needs, so a summary that recorded it
        # would differ between an export and an export-then-close and stop
        # being comparable.
        summary = {
            "batch_date": batch_date.isoformat(),
            "cleared_without_voucher": len(batch.cleared),
            "idempotency_key": batch.idempotency_key,
            "total_credits": f"{batch.total_credits:.2f}",
            "total_debits": f"{batch.total_debits:.2f}",
            "voucher_count": len(batch.lines),
            "xml_file": path.name,
        }
        summary_path = args.out / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", "utf-8")

    print(f"wrote {path}")
    print(f"wrote {summary_path}")
    print(f"  {len(batch.lines)} vouchers, debits {batch.total_debits:,.2f} = credits")
    print(f"  idempotency key {batch.idempotency_key}")
    print(
        f"  provenance: dataset={provenance.dataset} seed={provenance.seed} "
        f"config={provenance.config_hash} residual_false_match_rate={provenance.rate_text}"
    )
    print(f"  {len(batch.cleared)} confirmed exception(s) cleared, no voucher")
    print(f"  {len(closed)} exception(s) transitioned CONFIRMED -> CLOSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

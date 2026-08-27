"""Build the demo state DB the evidence queue reads. Writes; the UI does not.

SEPARATE FROM THE UI ON PURPOSE. The queue is read-only, and the only way to
keep that honest is for the thing that populates the store to live outside it.
`ui/queue.py` opens a database it did not create and refuses if there is none.

The checkpoints are 1, 12 and 24 rather than 1, 2, 3: the dataset spans 24
DELIVERY days - `--days 20` is capture days, and T+N settlement pushes the last
rows out to day 24 - and three checkpoints across that window show the residual
rising and falling, which the first three days cannot.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from settlesense.config import load_config
from settlesense.exceptions.store import ALL_STATUSES, ExceptionStore

DEFAULT_CHECKPOINTS = (1, 12, 24)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the demo state DB.")
    parser.add_argument("--data", type=Path, default=Path("data/dev"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--out", type=Path, default=Path("reports/ui/state.db"))
    parser.add_argument(
        "--days",
        default=",".join(str(day) for day in DEFAULT_CHECKPOINTS),
        help="comma-separated arrival days to run, in order",
    )
    args = parser.parse_args(argv)

    checkpoints = [int(part) for part in args.days.split(",") if part.strip()]
    if not checkpoints:
        raise SystemExit("--days: no arrival days given")

    config = load_config(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        # REBUILT, never appended to. Running twice against an existing store
        # would be a no-op by design (content-hash ingestion), which is correct
        # behaviour and useless for a demo that should be reproducible.
        args.out.unlink()

    with ExceptionStore(args.out) as store:
        for day in checkpoints:
            run = store.run_day(day, args.data, config)
            print(
                f"  day {day:>2}: {len(run.ingested)} files, "
                f"+{len(run.newly_opened)} opened, {len(run.newly_confirmed)} confirmed, "
                f"{len(run.still_residual)} residual"
            )
        total = len(store.get_queue(ALL_STATUSES))
    print(f"wrote {args.out} — {total} tracked exceptions across days {checkpoints}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

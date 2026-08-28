"""Write the evidence queue to a static HTML file. No server, no model.

The page a reviewer screenshots comes from here, and every number on it is
computed by `ui/queue.py` - the same module the Streamlit app reads - so the
screenshot and the app cannot disagree.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from eval.run_eval import load_days
from settlesense.config import load_config
from settlesense.ui.queue import open_store
from settlesense.ui.render import render_page

DEFAULT_AS_OF = date(2026, 11, 30)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the evidence queue to HTML.")
    parser.add_argument("--db", type=Path, default=Path("reports/ui/state.db"))
    parser.add_argument("--data", type=Path, default=Path("data/dev"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--out", type=Path, default=Path("reports/ui/queue.html"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=DEFAULT_AS_OF)
    parser.add_argument("--day", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="render only the N largest. Default is EVERY row: a table that "
        "showed the top 40 reported 2 AI_VERIFIED in its caption and displayed "
        "neither of them, because both rank below 40 by amount.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    dataset = load_days(args.data, config)
    with open_store(args.db) as store:
        page = render_page(store, dataset, config, args.as_of, args.day, args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

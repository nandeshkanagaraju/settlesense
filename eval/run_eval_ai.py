"""M5 - the AI evaluation set runner. Seeds 1000-1019, all twenty.

Reports the residual surface M7 has to work with, aggregated across the twenty
seeds declared in README before any of them was generated.

WHY THIS IS A SEPARATE RUNNER. `run_eval` answers "how did the system do on
this dataset". This answers "how much work is there, and is it the same kind of
work each time" - which is a question about the evaluation set itself, and the
answer has to exist before M7 starts or there is nothing to measure M7 against.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from eval.metrics import TruthView, population_a, population_b, population_c
from eval.run_eval import load_days
from settlesense.config import load_config
from settlesense.matching.duplicates import find_candidate_duplicates, find_confirmed_duplicates
from settlesense.matching.engine import build_cases, fuzzy_verdicts_for, run
from settlesense.matching.fuzzy_utr import ScoringPath

DEFAULT_AS_OF = date(2026, 11, 30)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 20-seed AI evaluation set.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=DEFAULT_AS_OF)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed_dirs = sorted(
        args.eval_dir.glob("seed_*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)
    )
    if not seed_dirs:
        raise SystemExit(
            f"no seed_* directories in {args.eval_dir}. The evaluation set is "
            "gitignored (~146MB); regenerate it with `make eval-set`."
        )

    rows: list[dict[str, Any]] = []
    for seed_dir in seed_dirs:
        seed = int(re.sub(r"\D", "", seed_dir.name))
        truth_path = seed_dir / f"truth_{seed}.json"
        dataset = load_days(seed_dir, config)
        result = run(dataset, config, args.as_of)
        cases_by_id = {fact.case.case_id: fact.case for fact in build_cases(dataset, config)}
        truth = TruthView.from_payload(json.loads(truth_path.read_text(encoding="utf-8")))

        confirmed = find_confirmed_duplicates(dataset.ledger_rows)
        excluded = frozenset(i for v in confirmed for i in v.row_ids)
        pairs = find_candidate_duplicates(dataset.ledger_rows, excluded)
        verdicts = fuzzy_verdicts_for(dataset, config, args.as_of)

        pop_a = population_a(result, cases_by_id, truth)
        pop_b = population_b(result, truth)
        pop_c = population_c(result, truth)
        rows.append(
            {
                "seed": seed,
                "cases": pop_a.case_count,
                "residual": pop_a.deterministic_residual_count,
                "duplicate_candidate_pairs": len(pairs),
                "false_match_rate_case_count": str(pop_a.residual_false_match_rate_case_count),
                "batches_linked": pop_b.linked_count,
                "false_links": pop_b.false_link_count,
                "row_variances": pop_c.row_count,
                "p8_path_a": sum(1 for v in verdicts if v.path is ScoringPath.PREFIX),
                "p8_path_b": sum(1 for v in verdicts if v.path is ScoringPath.AMOUNT_DATE),
                "p8_path_b_unresolved": sum(
                    1 for v in verdicts if v.path is ScoringPath.AMOUNT_DATE and not v.is_accepted
                ),
            }
        )

    pairs_per_seed = [row["duplicate_candidate_pairs"] for row in rows]
    mean = statistics.mean(pairs_per_seed)
    summary = {
        "seeds": [row["seed"] for row in rows],
        "seed_count": len(rows),
        "decisions_one_per_pair": sum(pairs_per_seed),
        "cases_flagged_two_per_pair": 2 * sum(pairs_per_seed),
        "pairs_mean": str(round(Decimal(mean), 2)),
        "pairs_min": min(pairs_per_seed),
        "pairs_max": max(pairs_per_seed),
        "pairs_stdev": str(round(Decimal(statistics.stdev(pairs_per_seed)), 2)),
        "poisson_stdev": str(round(Decimal(math.sqrt(mean)), 2)),
        "total_false_matches_case_count": sum(
            1 for row in rows if row["false_match_rate_case_count"] not in ("0.000000", "None")
        ),
        "total_false_links_batch_count": sum(row["false_links"] for row in rows),
        "per_seed": rows,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ai_eval_set.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"seeds {summary['seeds'][0]}-{summary['seeds'][-1]} ({summary['seed_count']})\n"
        f"decisions (one per ambiguous pair): {summary['decisions_one_per_pair']}\n"
        f"pairs per seed: mean {summary['pairs_mean']} "
        f"range {summary['pairs_min']}-{summary['pairs_max']} "
        f"(sd {summary['pairs_stdev']} vs Poisson {summary['poisson_stdev']})\n"
        f"seeds with any false match: {summary['total_false_matches_case_count']}\n"
        f"false links across all seeds: {summary['total_false_links_batch_count']}\n"
        f"\nwrote {args.out / 'ai_eval_set.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

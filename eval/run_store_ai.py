"""Run the AI layer through the PERSISTED STORE PATH and score it. Zero spend.

A SECOND MEASUREMENT, NEVER A REVISION. M7's figure stands untouched:

    AI layer, dataset-derived decisions (M7): 507 decisions, 27 confirmable,
    zero false confirms.

This produces a different line about a different path, and both are published:

    AI layer, persisted store path: 22 pairs replayed, N confirmed, M abstained,
    zero false confirms. 3 residual rows had no partner and are excluded.

NOTHING IS RECORDED AND NOTHING IS SPENT. Every prompt this replays was already
paid for during the M7 recording session: the store's pairs and the dataset's
pair exceptions describe the SAME duplicates, and 22 of 22 hashes were already
in `fixtures/llm/`. The gap `--simulate-outage` found was wiring, not
recordings - which is only visible once you check, and is why the 47-row
recording was not done (see LIMITATIONS.md).

WHY THIS LIVES IN eval/. Scoring needs truth, and `settlesense/` may not reach
it. The join itself is in `settlesense/ai/pairing.py`; this supplies the
dataset-derived pair exceptions the fixtures were recorded against, and grades
the result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.run_ai import duplicate_exceptions, truth_duplicate_orders
from settlesense.ai.client import MODEL, ReplayLLMClient
from settlesense.ai.hypothesis import generate
from settlesense.ai.pairing import PAIRING_KEY_NOTE, run_store_ai_stage
from settlesense.config import AppConfig, load_config
from settlesense.exceptions.store import ALL_STATUSES, ExceptionStore
from settlesense.ingest import DayDataset
from settlesense.types import Exception_, ExceptionStatus, ResolutionSource

M7_LINE = (
    "AI layer, dataset-derived decisions (M7): 507 decisions, 27 confirmable, zero false confirms."
)
"""Quoted verbatim so the two lines sit together and neither can be mistaken
for a correction of the other."""


def pair_index(dataset: DayDataset) -> dict[tuple[str, ...], Exception_]:
    """Sorted order ids -> the pair exception M7's fixtures were recorded for."""
    return {
        tuple(sorted(exception.evidence_row_ids)): exception
        for exception in duplicate_exceptions(dataset)
    }


def score(
    store: ExceptionStore,
    dataset: DayDataset,
    config: AppConfig,
    truth: frozenset[str],
    arrival_day: int,
) -> dict[str, object]:
    """Replay every pair, write back, and grade against truth.

    FALSE CONFIRMS ARE THE NUMBER THAT MATTERS and are computed from the
    DECISIVE nomination - the hypothesis the verifier actually acted on, not
    rank 0. Scoring rank 0 reported nine phantom false confirms during M7 for
    exactly this reason.
    """
    client = ReplayLLMClient()
    result = run_store_ai_stage(store, dataset, config, client, arrival_day, pair_index(dataset))

    per_pair: list[dict[str, object]] = []
    false_confirms = 0
    for pair, outcome in result.outcomes:
        subject = pair_index(dataset)[pair.order_ids]
        offered = generate(subject, dataset, config, ReplayLLMClient())
        nominated = (
            outcome.hypothesis.candidate_id
            if outcome.hypothesis is not None
            else (offered[0].candidate_id if offered else None)
        )
        correct = bool(nominated and nominated in truth)
        false_confirm = bool(outcome.confirmed and nominated and nominated not in truth)
        false_confirms += false_confirm
        per_pair.append(
            {
                "exception_ids": list(pair.exception_ids),
                "order_ids": list(pair.order_ids),
                "batch_id": pair.batch_id,
                "amount": str(pair.amount),
                "model_gave_a_hypothesis": bool(offered),
                "nominated": nominated,
                "nominated_correctly": correct,
                "confirmed": outcome.confirmed,
                "abstain_reason": (
                    outcome.abstain_reason.value if outcome.abstain_reason else None
                ),
                "false_confirm": false_confirm,
            }
        )

    rows = {row.exception_id: row for row in store.get_queue(ALL_STATUSES)}
    resolvers = {
        source.value: sum(1 for row in rows.values() if row.resolved_by is source)
        for source in ResolutionSource
    }
    resolvers["unresolved"] = sum(1 for row in rows.values() if row.resolved_by is None)
    statuses = {
        status.value: sum(1 for row in rows.values() if row.status is status)
        for status in ExceptionStatus
    }

    confirmed_pairs = sum(1 for entry in per_pair if entry["confirmed"])
    return {
        "model": MODEL,
        "pairing_key_note": PAIRING_KEY_NOTE,
        "m7_line": M7_LINE,
        "store_path_line": (
            f"AI layer, persisted store path: {result.pair_count} pairs replayed, "
            f"{confirmed_pairs} confirmed, {len(per_pair) - confirmed_pairs} abstained, "
            f"{false_confirms} false confirms. {len(result.unpaired)} residual rows "
            "had no partner and are excluded."
        ),
        "pairs_replayed": result.pair_count,
        "pairs_confirmed": confirmed_pairs,
        "pairs_abstained": len(per_pair) - confirmed_pairs,
        "false_confirms": false_confirms,
        "rows_confirmed": len(result.confirmed),
        "rows_abstained": len(result.abstained),
        "unpaired_rows": list(result.unpaired),
        "unpaired_count": len(result.unpaired),
        "resolved_by_counts": resolvers,
        "status_counts": statuses,
        "per_pair": per_pair,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the AI layer over the store path.")
    parser.add_argument("--data", type=Path, default=Path("data/dev"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=Path, default=Path("reports/ui/state.db"))
    parser.add_argument("--days", default="1,12,24")
    parser.add_argument("--arrival-day", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("reports/ai/store_path.json"))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    checkpoints = [int(part) for part in args.days.split(",") if part.strip()]
    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists():
        args.db.unlink()

    with ExceptionStore(args.db) as store:
        for day in checkpoints:
            store.run_day(day, args.data, config)
        dataset = store.cumulative_dataset(args.arrival_day, args.data, config)
        truth = truth_duplicate_orders(args.data / f"truth_{args.seed}.json")
        payload = score(store, dataset, config, truth, args.arrival_day)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{payload['m7_line']}\n{payload['store_path_line']}\n")
    print(f"  pairing key: {PAIRING_KEY_NOTE}")
    print(
        f"  rows written: {payload['rows_confirmed']} confirmed, "
        f"{payload['rows_abstained']} abstained"
    )
    print(f"  resolved_by: {payload['resolved_by_counts']}")
    print(f"  statuses:    {payload['status_counts']}")
    print(f"\nwrote {args.out} (zero model calls, zero spend)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

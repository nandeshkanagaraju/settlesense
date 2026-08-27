"""Record a DELIBERATE SAMPLE of real model responses into fixtures/llm/.

THE ONLY MODULE IN THIS PROJECT THAT TOUCHES A NETWORK. Everything else -
tests, eval, bench - replays from fixtures/llm/ and would raise if it tried.

WHY A SAMPLE AND NOT THE FULL 507. The oracle already bounds the result at 27
of 507: for the other 480 pairs the structural facts do not distinguish the two
rows, so the verifier rejects whatever is nominated and a real model cannot do
better than a perfect one. Recording all 507 would spend money to re-derive a
bound that is already known.

WHAT THE SAMPLE IS FOR. Showing the verifier REJECTING REAL MODEL OUTPUT rather
than output from a synthetic adversarial client. An adversary that is wrong by
construction is a weak demonstration; a real model that is confidently wrong,
rejected by an independent check, is the actual claim this architecture makes.

THE SELECTION RULE IS FIXED HERE, BEFORE ANY MODEL OUTPUT EXISTS:

    Across the AI evaluation set (seeds 1000-1019) in ascending seed order,
    and within each seed by exception_id, take the FIRST 20 pairs the oracle
    CONFIRMS and the FIRST 20 it REJECTS.

WHY THE EVALUATION SET AND NOT SEED 42. The brief asked for ~20 of each from
seed 42. Seed 42 HAS ONLY 2 oracle-confirmable pairs - the 2 where the two
rows' settlement chains differ - so a 20/20 split is arithmetically impossible
there. The 27 confirmable pairs in this project live across the 20 evaluation
seeds, which is the set that exists to be iterated against. Drawing from it is
the only way to fill the confirmable stratum, and it is stated here rather than
quietly shrinking the sample to what seed 42 could supply.

The oracle's verdict is a property of the DATA - whether the two rows have
distinguishable settlement chains - and is computed without any model. So the
sample is stratified by a fact known in advance, not by anything the model said.
Recording the rule in code rather than in a commit message is what makes it
checkable: `test_ai.py` asserts the manifest's rule string matches this one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from eval.run_ai import OracleClient, duplicate_exceptions, truth_duplicate_orders
from eval.run_eval import load_days
from settlesense.ai.client import MODEL, RealLLMClient, record_fixture
from settlesense.ai.hypothesis import HYPOTHESIS_SCHEMA, build_prompt, generate
from settlesense.ai.loop import resolve_exception
from settlesense.config import AppConfig, load_config
from settlesense.ingest import DayDataset
from settlesense.types import Exception_

SAMPLE_PER_STRATUM = 20
SELECTION_RULE = (
    "AI evaluation set (seeds 1000-1019), ascending by seed then exception_id; "
    "the first 20 pairs the oracle CONFIRMS and the first 20 it REJECTS. The "
    "oracle's verdict is computed from the data alone, before any model is "
    "called. Seed 42 was not used because it has only 2 confirmable pairs, so a "
    "20/20 split is impossible there."
)

USD_PER_INR = Decimal("88")
"""One rate, stated once. Cost is reported in both currencies from MEASURED
token counts - never from an estimate of prompt length."""

INPUT_USD_PER_MTOK = Decimal("2.50")
OUTPUT_USD_PER_MTOK = Decimal("10.00")
"""List pricing for the pinned snapshot at time of recording. Recorded in the
manifest so a later reader can tell whether a quoted cost is still current."""


@dataclass(frozen=True)
class Selected:
    """One chosen decision, and the seed whose data it came from.

    The seed is carried because a fixture is keyed by PROMPT hash and a prompt
    embeds row ids - so replaying it requires loading the dataset those rows
    came from. A sample that did not record its source could not be replayed.
    """

    seed: int
    exception: Exception_
    data_dir: str


def stratify(
    eval_dir: Path, config: AppConfig, per_stratum: int = SAMPLE_PER_STRATUM
) -> tuple[list[Selected], list[Selected]]:
    """Split by what the ORACLE does, using no model at all.

    Walks seeds in ascending order and stops as soon as BOTH strata are full,
    so the choice is "the first N", not "N drawn from everything".
    """
    seed_dirs = sorted(
        eval_dir.glob("seed_*"), key=lambda path: int(re.sub(r"\D", "", path.name) or 0)
    )
    if not seed_dirs:
        raise SystemExit(
            f"no seed_* directories in {eval_dir}. The evaluation set is gitignored; "
            "regenerate it with `make eval-set`."
        )

    confirmed: list[Selected] = []
    rejected: list[Selected] = []
    for seed_dir in seed_dirs:
        if len(confirmed) >= per_stratum and len(rejected) >= per_stratum:
            break
        seed = int(re.sub(r"\D", "", seed_dir.name))
        dataset = load_days(seed_dir, config)
        truth = truth_duplicate_orders(seed_dir / f"truth_{seed}.json")
        oracle = OracleClient(truth)
        for exception in sorted(duplicate_exceptions(dataset), key=lambda e: e.exception_id):
            outcome = resolve_exception(exception, dataset, config, oracle)
            bucket = confirmed if outcome.confirmed else rejected
            if len(bucket) < per_stratum:
                bucket.append(Selected(seed, exception, str(seed_dir)))
    return confirmed, rejected


def _score(selected: list[Selected], confirmed_stratum: list[Selected], config: AppConfig) -> int:
    """Replay the RECORDED responses and compare the model with the oracle.

    Uses ReplayLLMClient, so scoring costs nothing and can be re-run. The
    comparison that matters is on the CONFIRMABLE stratum: the model cannot
    beat the oracle, and matching it there is the interesting line.
    """
    from settlesense.ai.client import ReplayLLMClient

    replay = ReplayLLMClient()
    confirmable = {item.exception.exception_id for item in confirmed_stratum}
    datasets: dict[str, DayDataset] = {}
    truths: dict[str, frozenset[str]] = {}

    rows: list[dict[str, object]] = []
    for item in selected:
        dataset = datasets.setdefault(item.data_dir, load_days(Path(item.data_dir), config))
        truth = truths.setdefault(
            item.data_dir,
            truth_duplicate_orders(Path(item.data_dir) / f"truth_{item.seed}.json"),
        )
        # generate() SEPARATELY from resolve_exception: LoopOutcome.hypothesis
        # is None when every hypothesis was rejected, so reading the nomination
        # off the outcome reported "the model said nothing" for cases where it
        # said three things and all three were refused.
        offered = generate(item.exception, dataset, config, replay)
        model = resolve_exception(item.exception, dataset, config, replay)
        oracle = resolve_exception(item.exception, dataset, config, OracleClient(truth))
        # THE DECISIVE NOMINATION is the one the verifier acted on, which is
        # the first hypothesis that PASSED - not necessarily rank 0. Scoring
        # rank 0 reported 9 "false confirms" that were nothing of the kind: the
        # verifier had confirmed a lower-ranked hypothesis nominating the right
        # row, and the comparison was reading a different claim than the one
        # that was acted on.
        nominated = (
            model.hypothesis.candidate_id
            if model.hypothesis is not None
            else (offered[0].candidate_id if offered else None)
        )
        top_ranked = offered[0].candidate_id if offered else None
        rows.append(
            {
                "exception_id": item.exception.exception_id,
                "stratum": "confirmable" if item.exception.exception_id in confirmable else "not",
                "model_nominated_correctly": bool(nominated and nominated in truth),
                "top_ranked_was_correct": bool(top_ranked and top_ranked in truth),
                "model_confirmed": model.confirmed,
                "oracle_confirmed": oracle.confirmed,
                "model_gave_a_hypothesis": bool(offered),
                "nominated_a_real_row": bool(
                    nominated and nominated in item.exception.evidence_row_ids
                ),
                "false_confirm": bool(model.confirmed and nominated and nominated not in truth),
            }
        )

    def tally(key: str, stratum: str | None = None) -> int:
        return sum(1 for r in rows if r[key] and (stratum is None or r["stratum"] == stratum))

    total = len(rows)
    conf_n = sum(1 for r in rows if r["stratum"] == "confirmable")
    print(
        f"\nSCORED {total} recorded decisions ({conf_n} confirmable, {total - conf_n} not)\n"
        f"  model produced a hypothesis      {tally('model_gave_a_hypothesis')}/{total}\n"
        f"  nominated one of the two rows    {tally('nominated_a_real_row')}/{total}\n"
        f"  decisive nomination correct      {tally('model_nominated_correctly')}/{total}\n"
        f"    (its TOP-RANKED guess)         {tally('top_ranked_was_correct')}/{total}\n"
        f"    ...on the confirmable stratum  "
        f"{tally('model_nominated_correctly', 'confirmable')}/{conf_n}\n"
        f"  VERIFIER confirmed (model)       {tally('model_confirmed')}/{total}\n"
        f"  VERIFIER confirmed (oracle)      {tally('oracle_confirmed')}/{total}\n"
        f"  verifier REJECTED real output    {total - tally('model_confirmed')}/{total}\n"
        f"  FALSE CONFIRMS                   {tally('false_confirm')}   <- must be 0"
    )
    Path("reports/ai").mkdir(parents=True, exist_ok=True)
    Path("reports/ai/real_model_sample.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "selection_rule": SELECTION_RULE,
                "totals": {
                    "decisions": total,
                    "confirmable_stratum": conf_n,
                    "model_gave_a_hypothesis": tally("model_gave_a_hypothesis"),
                    "nominated_a_real_row": tally("nominated_a_real_row"),
                    "model_nominated_correctly": tally("model_nominated_correctly"),
                    "top_ranked_was_correct": tally("top_ranked_was_correct"),
                    "model_nominated_correctly_confirmable": tally(
                        "model_nominated_correctly", "confirmable"
                    ),
                    "verifier_confirmed_model": tally("model_confirmed"),
                    "verifier_confirmed_oracle": tally("oracle_confirmed"),
                    "false_confirms": tally("false_confirm"),
                },
                "per_decision": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nwrote reports/ai/real_model_sample.json")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a real-model fixture sample.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="record EVERY duplicate pair in one dataset instead of the stratified "
        "sample; used to populate the M8 evidence-queue demo on the dev seed",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed of --data, for its truth file")
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument(
        "--score",
        action="store_true",
        help="replay the recorded sample through the verifier and compare with the oracle",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="select and print the sample without calling the model or spending anything",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.data is not None:
        # WHOLE-DATASET MODE. No stratification and no sampling: every ambiguous
        # pair in one dataset, so the evidence queue has a recorded response for
        # every row a reviewer can click. Selection is not a question here -
        # "all of them" needs no rule.
        dataset = load_days(args.data, config)
        oracle_truth = truth_duplicate_orders(args.data / f"truth_{args.seed}.json")
        oracle = OracleClient(oracle_truth)
        confirmed: list[Selected] = []
        rejected: list[Selected] = []
        for exception in sorted(duplicate_exceptions(dataset), key=lambda e: e.exception_id):
            outcome = resolve_exception(exception, dataset, config, oracle)
            item = Selected(args.seed, exception, str(args.data))
            (confirmed if outcome.confirmed else rejected).append(item)
    else:
        confirmed, rejected = stratify(args.eval_dir, config)
    selected = [*confirmed, *rejected]
    print(
        f"selection rule: {SELECTION_RULE}\n"
        f"  oracle-confirmed stratum: {len(confirmed)}\n"
        f"  oracle-rejected  stratum: {len(rejected)}\n"
        f"  total to record:          {len(selected)}\n"
    )
    if args.dry_run:
        for item in selected:
            print(f"  seed {item.seed}  {item.exception.exception_id}")
        print("\ndry run: no model was called, nothing was recorded, nothing was spent")
        return 0

    if args.score:
        return _score(selected, confirmed, config)

    client = RealLLMClient()
    datasets: dict[str, DayDataset] = {}
    recorded = 0
    input_tokens = output_tokens = 0
    for index, item in enumerate(selected, start=1):
        dataset = datasets.setdefault(item.data_dir, load_days(Path(item.data_dir), config))
        prompt = build_prompt(item.exception, dataset, config)
        response = client.complete(prompt, HYPOTHESIS_SCHEMA)
        record_fixture(prompt, response)
        recorded += 1
        input_tokens += client.last_usage.get("input_tokens", 0)
        output_tokens += client.last_usage.get("output_tokens", 0)
        print(
            f"  [{index:>2}/{len(selected)}] seed {item.seed} {item.exception.exception_id} "
            f"in={client.last_usage.get('input_tokens', 0)} "
            f"out={client.last_usage.get('output_tokens', 0)}",
            flush=True,
        )

    cost_usd = (
        Decimal(input_tokens) * INPUT_USD_PER_MTOK + Decimal(output_tokens) * OUTPUT_USD_PER_MTOK
    ) / Decimal(1_000_000)
    manifest = {
        "model": MODEL,
        "selection_rule": (
            f"every ambiguous duplicate pair in {args.data} (seed {args.seed}); no "
            "sampling and no stratification - all of them"
            if args.data is not None
            else SELECTION_RULE
        ),
        "sample_per_stratum": SAMPLE_PER_STRATUM,
        "strata": {
            "oracle_confirmed": [
                {"seed": i.seed, "exception_id": i.exception.exception_id} for i in confirmed
            ],
            "oracle_rejected": [
                {"seed": i.seed, "exception_id": i.exception.exception_id} for i in rejected
            ],
        },
        "recorded": recorded,
        "measured_input_tokens": input_tokens,
        "measured_output_tokens": output_tokens,
        "pricing_usd_per_mtok": {
            "input": str(INPUT_USD_PER_MTOK),
            "output": str(OUTPUT_USD_PER_MTOK),
        },
        "measured_cost_usd": str(cost_usd.quantize(Decimal("0.000001"), ROUND_HALF_UP)),
        "measured_cost_inr": str((cost_usd * USD_PER_INR).quantize(Decimal("0.01"), ROUND_HALF_UP)),
    }
    # SEPARATE MANIFESTS. The stratified 40-decision sample is the measured
    # result the README quotes; the whole-dataset dev run exists to populate
    # the evidence queue. One file would let the second silently overwrite the
    # first - which it did, once.
    manifest_name = "llm_manifest_dev.json" if args.data is not None else "llm_manifest.json"
    Path("fixtures").mkdir(exist_ok=True)
    Path(f"fixtures/{manifest_name}").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"\nrecorded {recorded} fixtures against {MODEL}\n"
        f"  measured tokens: {input_tokens:,} in / {output_tokens:,} out\n"
        f"  measured cost:   ${manifest['measured_cost_usd']} "
        f"= Rs {manifest['measured_cost_inr']}\n"
        f"wrote fixtures/{manifest_name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

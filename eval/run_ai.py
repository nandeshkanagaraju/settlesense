"""M7 - the verified hypothesis loop, measured against the 20-seed eval set.

WHY THIS RUNS WITHOUT A MODEL, AND WHY THAT IS THE POINT.

The question "how many residuals can the AI layer explain" has an upper bound
that no model can exceed: the number the VERIFIER would confirm if the model
nominated perfectly every time. This runner measures that bound directly, with
an oracle client that reads truth and always nominates correctly.

If the oracle's confirmed count is zero, then a real model's confirmed count is
also zero - not because the model is weak, but because the verifier cannot
independently check the claim, and confirming anyway is the failure the
architecture exists to prevent. Establishing that BEFORE spending money on
fixtures is the whole reason this exists.

THREE CLIENTS, and the contrast between them is the result:

  ORACLE       always nominates the truth-correct row. The ceiling.
  ADVERSARIAL  always nominates the WRONG row. False-confirms must be 0.
  SILENT       returns nothing schema-valid. Must abstain, never crash.

The adversarial client is the safety measurement. A verifier that confirms the
oracle and rejects the adversary is discriminating; one that confirms both is
rubber-stamping, and only the second client can tell them apart.

NO NETWORK. None of these clients has one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from eval.run_eval import load_days
from settlesense.ai.loop import run_loop
from settlesense.config import AppConfig, load_config
from settlesense.exceptions.taxonomy import VarianceCategory
from settlesense.ingest import DayDataset
from settlesense.matching.duplicates import find_candidate_duplicates, find_confirmed_duplicates
from settlesense.types import Exception_, ExceptionStatus, money

DEFAULT_AS_OF = date(2026, 11, 30)
DUPLICATE = str(VarianceCategory.DUPLICATE_CANDIDATE)

SENDABLE = frozenset({DUPLICATE})
"""What the wiring actually sends, narrower than PDD 6.2.

Population B's unlinked batches are MISSING_VS_LATE_CREDIT, which PDD 6.2 does
list as interpretive - but on this dataset their credit never arrived at all.
That is missing DATA, not an interpretive question, and asking a model about an
absence invites a fabricated explanation for it.
"""


class _StubClient:
    """Base for the three measurement clients. None of them has a network."""

    def __init__(self, truth_duplicates: frozenset[str]) -> None:
        self._truth = truth_duplicates
        self.calls: list[str] = []

    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        del schema
        self.calls.append(prompt)
        pair = sorted(set(re.findall(r"\bORD_[0-9A-F]+\b", prompt)))
        if len(pair) != 2:
            return {"hypotheses": []}
        return {"hypotheses": [self._hypothesis(pair)]}

    def _hypothesis(self, pair: list[str]) -> dict[str, Any]:
        raise NotImplementedError


class OracleClient(_StubClient):
    """Always nominates the truth-correct row. THE CEILING."""

    def _hypothesis(self, pair: list[str]) -> dict[str, Any]:
        correct = next((row for row in pair if row in self._truth), pair[0])
        return {
            "category": DUPLICATE,
            "candidate_id": correct,
            "evidence_row_ids": pair,
            "reason": "oracle: nominated the row truth marks as the injected duplicate",
        }


class AdversarialClient(_StubClient):
    """Always nominates the WRONG row. False-confirms must be zero."""

    def _hypothesis(self, pair: list[str]) -> dict[str, Any]:
        wrong = next((row for row in pair if row not in self._truth), pair[-1])
        return {
            "category": DUPLICATE,
            "candidate_id": wrong,
            "evidence_row_ids": pair,
            "reason": "adversarial: nominated the row truth says is the genuine order",
        }


class SilentClient(_StubClient):
    """Returns nothing schema-valid. Must abstain, never crash."""

    def _hypothesis(self, pair: list[str]) -> dict[str, Any]:
        del pair
        return {
            "category": "NOT_A_CATEGORY",
            "candidate_id": "",
            "evidence_row_ids": [],
            "reason": "",
        }


@dataclass(frozen=True)
class SeedResult:
    """One seed's numbers. Never averaged across seeds without being summed."""

    seed: int
    pairs: int
    sent: int
    oracle_confirmed: int
    oracle_false_confirmed: int
    adversarial_confirmed: int
    silent_confirmed: int
    reasons: tuple[tuple[str, int], ...]


def duplicate_exceptions(dataset: DayDataset) -> tuple[Exception_, ...]:
    """One exception per ambiguous PAIR, not per flagged case.

    The engine flags both halves - 52 cases for 26 pairs - because it cannot
    know which was injected. The DECISION is per pair, so sending 52 prompts
    would ask the same question twice and double every count derived from it.
    """
    confirmed = find_confirmed_duplicates(dataset.ledger_rows)
    excluded = frozenset(row_id for verdict in confirmed for row_id in verdict.row_ids)
    pairs = find_candidate_duplicates(dataset.ledger_rows, excluded)
    return tuple(
        Exception_(
            exception_id=f"dup-{'-'.join(sorted(verdict.row_ids))}",
            category=DUPLICATE,
            amount=money(verdict.amount if verdict.amount is not None else 0),
            status=ExceptionStatus.OPEN,
            confidence=money(0),
            evidence_row_ids=tuple(sorted(verdict.row_ids)),
            reason=verdict.detail,
            resolved_by=None,
            first_seen_day=dataset.arrival_day,
            confirmed_day=None,
            closed_day=None,
            audit=(),
        )
        for verdict in pairs
    )


def truth_duplicate_orders(truth_path: Path) -> frozenset[str]:
    """Order ids truth marks as the INJECTED duplicate.

    Used only to score and to drive the oracle. No engine or verifier code path
    reads truth - `test_ai.py` asserts settlesense/ never imports it.
    """
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    return frozenset(
        case["order_id"]
        for case in payload.get("cases", ())
        if case.get("true_category") == DUPLICATE
    )


def evaluate_seed(seed: int, data_dir: Path, config: AppConfig, as_of: date) -> SeedResult:
    dataset = load_days(data_dir, config)
    truth = truth_duplicate_orders(data_dir / f"truth_{seed}.json")
    exceptions = duplicate_exceptions(dataset)

    oracle = OracleClient(truth)
    adversary = AdversarialClient(truth)
    silent = SilentClient(truth)

    oracle_report = run_loop(exceptions, dataset, config, oracle, sendable=SENDABLE)
    adversary_report = run_loop(exceptions, dataset, config, adversary, sendable=SENDABLE)
    silent_report = run_loop(exceptions, dataset, config, silent, sendable=SENDABLE)

    # A "false confirm" is a confirmation whose nomination truth disagrees
    # with. Counted for the ORACLE too: an oracle that nominates correctly can
    # still be wrong if truth and the pair disagree about membership.
    oracle_false = sum(
        1
        for outcome in oracle_report.outcomes
        if outcome.confirmed
        and outcome.hypothesis is not None
        and outcome.hypothesis.candidate_id not in truth
    )
    return SeedResult(
        seed=seed,
        pairs=len(exceptions),
        sent=oracle_report.sent,
        oracle_confirmed=oracle_report.confirmed,
        oracle_false_confirmed=oracle_false,
        adversarial_confirmed=adversary_report.confirmed,
        silent_confirmed=silent_report.confirmed,
        reasons=oracle_report.reasons,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M7 verified hypothesis loop, measured.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--dev", type=Path, default=None, help="score one dev directory instead")
    parser.add_argument("--out", type=Path, default=Path("reports/ai"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=DEFAULT_AS_OF)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    results: list[SeedResult] = []

    if args.dev is not None:
        seed = int(re.sub(r"\D", "", next(args.dev.glob("truth_*.json")).stem))
        results.append(evaluate_seed(seed, args.dev, config, args.as_of))
    else:
        seed_dirs = sorted(
            args.eval_dir.glob("seed_*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)
        )
        if not seed_dirs:
            raise SystemExit(
                f"no seed_* directories in {args.eval_dir}. The evaluation set is "
                "gitignored; regenerate it with `make eval-set`."
            )
        for seed_dir in seed_dirs:
            seed = int(re.sub(r"\D", "", seed_dir.name))
            results.append(evaluate_seed(seed, seed_dir, config, args.as_of))
            latest = results[-1]
            print(
                f"  seed {latest.seed}: {latest.pairs:>3} pairs  "
                f"oracle {latest.oracle_confirmed:>3}  "
                f"adversarial {latest.adversarial_confirmed:>3}  "
                f"silent {latest.silent_confirmed:>3}",
                flush=True,
            )

    totals = {
        "seeds": len(results),
        "decisions_sent": sum(r.sent for r in results),
        "oracle_confirmed": sum(r.oracle_confirmed for r in results),
        "oracle_false_confirmed": sum(r.oracle_false_confirmed for r in results),
        "adversarial_confirmed": sum(r.adversarial_confirmed for r in results),
        "silent_confirmed": sum(r.silent_confirmed for r in results),
    }
    reasons: Counter[str] = Counter()
    for result in results:
        for reason, count in result.reasons:
            reasons[reason] += count

    payload: dict[str, Any] = {
        "totals": totals,
        "abstain_reasons": dict(sorted(reasons.items())),
        "per_seed": [
            {
                "seed": r.seed,
                "pairs": r.pairs,
                "sent": r.sent,
                "oracle_confirmed": r.oracle_confirmed,
                "adversarial_confirmed": r.adversarial_confirmed,
                "silent_confirmed": r.silent_confirmed,
            }
            for r in results
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ai_loop.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"\nseeds {totals['seeds']}   decisions sent {totals['decisions_sent']}\n"
        f"  oracle (perfect model)      confirmed {totals['oracle_confirmed']}\n"
        f"  adversarial (always wrong)  confirmed {totals['adversarial_confirmed']}"
        "   <- FALSE CONFIRMS, must be 0\n"
        f"  silent (no valid output)    confirmed {totals['silent_confirmed']}\n"
        f"  abstain reasons: {dict(sorted(reasons.items()))}\n"
        f"\nwrote {args.out / 'ai_loop.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

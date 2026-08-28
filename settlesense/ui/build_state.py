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
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from settlesense.ai.client import (
    FIXTURE_DIR,
    FixtureMissError,
    LLMClient,
    ModelUnavailable,
    OutageLLMClient,
    ReplayLLMClient,
)
from settlesense.ai.hypothesis import AI_ELIGIBLE_CATEGORIES, eligible_exceptions, generate
from settlesense.ai.orchestrator import run_ai_stage
from settlesense.config import AppConfig, load_config
from settlesense.exceptions.store import ALL_STATUSES, RESIDUAL_STATES, ExceptionStore
from settlesense.ingest import DayDataset

DEFAULT_CHECKPOINTS = (1, 12, 24)


class ProbeFailed(RuntimeError):
    """`--simulate-outage` was asked for and the client was already unusable."""


def probe(store: ExceptionStore, dataset: DayDataset, config: AppConfig) -> str:
    """Prove the client WORKS before breaking it. Raises if it does not.

    WHY THIS EXISTS. `--simulate-outage` produces a screen full of
    PENDING_AI_UNAVAILABLE rows. So does a broken fixture path, a renamed
    prompt, or a run pointed at the wrong data directory - and the output is
    identical in every case. A demo that cannot tell "I turned the model off"
    from "the model was never on" is a demo that proves nothing, and the
    failure looks like a success.

    WHAT "REACHABLE" MEANS HERE, precisely, because the word would otherwise
    overclaim. This project has no network. The healthy client is
    `ReplayLLMClient`, and reachable means THE REPLAY CACHE ANSWERS: a recorded
    prompt is read off disk and put back through the client, which must return
    the recorded response. That catches every misconfiguration the flag can
    hide - wrong fixture directory, empty cache, a client that raises on
    everything. It is NOT a statement about OpenAI being up.

    AND IT REPORTS COVERAGE SEPARATELY, because the two questions are
    different. "Is the client working" is answered above. "Would the healthy
    client have answered for the rows this outage is about to fail" is answered
    by counting recordings for those specific exceptions, and on the dev store
    that count is ZERO - the M7 loop was measured over dataset-derived pair
    exceptions (`dup-ORD_A-ORD_B`), and the store persists engine outcomes with
    hash ids. Disjoint sets, no recordings in common. The outage is still real
    - `OutageLLMClient` raises before any cache lookup, and ModelUnavailable and
    FixtureMissError take different paths - but what a healthy run would have
    CONCLUDED for those rows is unknown, and the caller is told so rather than
    left to infer it.
    """
    recordings = sorted(FIXTURE_DIR.glob("*.json"))
    if not recordings:
        raise ProbeFailed(
            f"no recordings in {FIXTURE_DIR}. Every row would come back "
            "unavailable and the screen would look identical to a real outage."
        )
    healthy: LLMClient = ReplayLLMClient()
    recorded = json.loads(recordings[0].read_text(encoding="utf-8"))
    try:
        answer = healthy.complete(recorded["prompt"], {})
    except Exception as error:
        raise ProbeFailed(
            f"the healthy client could not answer a prompt that IS recorded "
            f"({recordings[0].name}): {type(error).__name__}: {error}. Simulating "
            "an outage now would show PENDING_AI_UNAVAILABLE rows caused by a "
            "misconfiguration rather than by the outage."
        ) from error
    if answer != recorded["response"]:
        raise ProbeFailed(
            f"the healthy client returned something other than the recorded "
            f"response for {recordings[0].name}; the cache is not serving what it holds"
        )

    candidates = eligible_exceptions(
        tuple(store.get_queue(status_filter=RESIDUAL_STATES)), AI_ELIGIBLE_CATEGORIES
    )
    if not candidates:
        raise ProbeFailed(
            "no AI-eligible residual exceptions exist, so an outage would have "
            "nothing to fail on. The screen would be empty of "
            "PENDING_AI_UNAVAILABLE rows and would look like a healthy run."
        )
    covered = 0
    for candidate in candidates:
        try:
            generate(candidate, dataset, config, ReplayLLMClient())
        except ModelUnavailable as error:  # pragma: no cover - replay cannot raise this
            raise ProbeFailed(f"the healthy client is already unavailable: {error}") from error
        except FixtureMissError:
            continue
        covered += 1
    return (
        f"client OK ({len(recordings)} recordings, {recordings[0].name[:12]} replays); "
        f"{len(candidates)} eligible residual rows, {covered} of them recorded"
        + (
            " - NONE RECORDED, so the outage is real but what a healthy run would "
            "have concluded for these rows is unknown"
            if covered == 0
            else ""
        )
    )


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
    parser.add_argument(
        "--simulate-outage",
        action="store_true",
        help=(
            "run the AI stage with a client that is always down, so the queue "
            "shows PENDING_AI_UNAVAILABLE rows. FAILS LOUDLY if the healthy "
            "client could not have answered in the first place."
        ),
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
        if args.simulate_outage:
            outcome = _simulate_outage(store, args.data, config, checkpoints[-1])
            if outcome != 0:
                return outcome
        total = len(store.get_queue(ALL_STATUSES))
    print(f"wrote {args.out} — {total} tracked exceptions across days {checkpoints}")
    return 0


def _simulate_outage(store: ExceptionStore, data: Path, config: AppConfig, arrival_day: int) -> int:
    """Probe, then break it. Returns a process exit code.

    THE AI STAGE RUNS ONLY UNDER THIS FLAG. Without it this script is exactly
    what it was before M10 and produces a byte-identical store, which is what
    keeps the M8 demo and its committed screenshots meaningful. Adding a model
    stage to the default path would have quietly changed every number in the
    evidence queue.
    """
    dataset = store.cumulative_dataset(arrival_day, data, config)
    try:
        detail = probe(store, dataset, config)
    except ProbeFailed as error:
        # NON-ZERO, and the message names the difference. A demo must not be
        # able to show an outage that was really a misconfiguration.
        print(f"--simulate-outage REFUSED: {error}", file=sys.stderr)
        return 3
    print(f"  probe: {detail}")

    result = run_ai_stage(store, dataset, config, OutageLLMClient(), arrival_day)
    print(
        f"  outage: {result.report.sent} sent, {len(result.pending_unavailable)} "
        f"PENDING_AI_UNAVAILABLE, {len(result.confirmed)} confirmed, "
        f"{len(result.abstained)} abstained"
    )
    if result.confirmed:
        print(
            f"something was confirmed during an outage: {result.confirmed}",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())

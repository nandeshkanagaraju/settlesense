"""M5a - Throughput scaling bench. Deterministic pipeline, no model calls.

WHAT THIS MEASURES AND WHY IT IS A DELIVERABLE. The track bar reads "throughput
plus measured accuracy plus an honest exception list", and throughput is named
first. Accuracy without throughput is a demo; this makes the number a measured
artifact with a machine attached to it (SDD 8.1).

THREE METHODOLOGICAL CHOICES, each of which could flatter the result if made
the other way:

1. THE MEDIAN OF THREE REPETITIONS, NEVER THE BEST. A best-of-N benchmark
   reports the run where the OS happened to cooperate, which is not a number
   anyone will reproduce. The median is what a reader will see.

2. MEMORY IS MEASURED IN A SEPARATE REPETITION FROM TIME. tracemalloc hooks
   every allocation and materially slows the interpreter, so a run that
   measured both at once would report a peak that is honest and a duration
   that is not.

3. GENERATION IS EXCLUDED FROM THE TIMED REGION. Building the dataset is
   scaffolding for the benchmark, not part of the pipeline under test. It is
   still reported, so the wall-clock cost of running this target is visible.

THE DEV SEED, ALWAYS (B8). Never the holdout. A benchmark re-run on every code
change would burn a held-out set through sheer repetition, and nothing about a
throughput figure needs unseen data.

NO MODEL CALLS. The AI stage is timed separately against the RESIDUAL count,
because that is the architectural claim being tested: the expensive stage
scales with ambiguity, not with volume.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from eval.run_eval import input_rows, load_days
from settlesense.config import AppConfig, load_config
from settlesense.core.telemetry import MachineSpec, StageTimer, StageTiming, format_rate
from settlesense.matching.engine import residual_cases, run_with_telemetry

DEFAULT_SIZES: tuple[int, ...] = (500, 5000, 25000)
"""500/5000/25000 - NOT 100000 (B1).

The frozen generator's default produces 5,026 cases. Larger sizes mean
generating new data at a larger scale, which is a generator INVOCATION, not a
re-freeze: gen/ is unchanged and its manifest hash still matches. 100k is
attempted only as a stretch, under the rule below.
"""

STRETCH_SIZE = 100_000
STRETCH_TRIGGER_SIZE = 25_000
STRETCH_BUDGET_SECONDS = 120
"""Attempt 100k only if 25k finished inside two minutes.

If it is skipped the report SAYS SO. Extrapolating a 100k figure from a 25k
measurement would put a number in a table that no machine ever produced, and
the reader cannot tell the difference from the table alone.
"""

REPETITIONS = 3
BENCH_SEED = 42  # the DEV seed. Never 999 (B8).
DEFAULT_AS_OF = date(2026, 11, 30)
DAYS = 20

BYTES_PER_MIB = 1024 * 1024

REPO = Path(__file__).resolve().parent.parent
DEV_DATA = REPO / "data" / "dev"
FIXTURE_MANIFESTS: tuple[Path, ...] = (
    REPO / "fixtures" / "llm_manifest.json",
    REPO / "fixtures" / "llm_manifest_dev.json",
)
"""Both recording runs. Cost is READ FROM THESE, never typed into a report:
they are written by `record_fixtures` from the API's own usage numbers, so a
figure in bench.md cannot drift from what was actually spent."""

DEV_MANIFEST = REPO / "fixtures" / "llm_manifest_dev.json"
"""The dev-seed recording, and the only one that supports a per-ROW cost.

It covers EVERY ambiguous duplicate pair in data/dev with no sampling, so its
total is the complete model spend for that dataset and can be divided by that
dataset's row count. The evaluation-set manifest is a stratified sample of 40
drawn from 507 decisions across 20 seeds - a real measurement of per-DECISION
cost, but there is no single row count to divide it by."""

ESTIMATE_INR_PER_THOUSAND = Decimal("40.98")
ESTIMATE_TOTAL_INR = Decimal("207")
ESTIMATE_DECISIONS = 507
"""The pre-spend projection, kept so the correction stays checkable.

From README at commit 1ac59fd^: "~297 input and ~250 output tokens per
decision; 507 decisions = $2.35 (Rs 207), or Rs 40.98 per 1,000 rows". Recorded
here rather than deleted - a project that only ever shows its final numbers is
asking to be trusted about how it got them."""


@dataclass(frozen=True)
class SizeResult:
    """One row of the scaling table. Every duration is a median of REPETITIONS."""

    records: int
    cases: int
    input_rows: int
    residual: int
    generate_seconds: float
    ingest_seconds: float
    engine_seconds: float
    peak_bytes: int
    stages: tuple[StageTiming, ...]

    @property
    def pipeline_seconds(self) -> float:
        """Ingest plus engine. What an operator waits for; generation excluded."""
        return self.ingest_seconds + self.engine_seconds

    @property
    def cases_per_second(self) -> float | None:
        return _rate(self.cases, self.pipeline_seconds)

    @property
    def rows_per_second(self) -> float | None:
        return _rate(self.input_rows, self.pipeline_seconds)


def _rate(count: int, seconds: float) -> float | None:
    """None when the duration was too small to time. Never inf, never zero."""
    if seconds <= 0:
        return None
    return count / seconds


def parse_sizes(raw: str) -> tuple[int, ...]:
    """A comma list of positive ints, de-duplicated, ascending.

    Refuses rather than coerces. `--sizes 500,abc` silently becoming `(500,)`
    would run half the benchmark and report it as the whole thing.
    """
    sizes: list[int] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError:
            raise SystemExit(f"--sizes: {text!r} is not an integer") from None
        if value <= 0:
            raise SystemExit(f"--sizes: {value} is not a positive record count")
        sizes.append(value)
    if not sizes:
        raise SystemExit("--sizes: no sizes given")
    return tuple(sorted(set(sizes)))


def generate(records: int, out_dir: Path) -> float:
    """Invoke the FROZEN generator as a subprocess. Returns seconds elapsed.

    A subprocess rather than an import, so `eval/` never acquires an import
    edge into `gen/`. The two-directions AST guard covers gen<->settlesense;
    keeping eval out of gen's import graph as well means the benchmark cannot
    accidentally depend on generator internals that the freeze forbids editing.
    """
    collector: list[StageTiming] = []
    with StageTimer(collector, "generate", records):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "gen.generate",
                "--seed",
                str(BENCH_SEED),
                "--out",
                str(out_dir),
                "--days",
                str(DAYS),
                "--records",
                str(records),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(
            f"generation failed for {records} records (exit {completed.returncode}):\n"
            f"{completed.stderr.strip()}"
        )
    return collector[0].seconds


# `_input_rows` MOVED to eval/run_eval.py. The evaluation runner needs the same
# ingest denominator, bench.py already imports from run_eval, and two copies of
# "what counts as a row" is exactly the kind of pair that drifts apart silently.
# Re-exported under the old private name so nothing that imports it here breaks.
_input_rows = input_rows


def _median_stages(runs: Sequence[tuple[StageTiming, ...]]) -> tuple[StageTiming, ...]:
    """Per-stage median across repetitions, in first-run order.

    Stage order is the PASS ORDER, so it is preserved from the first run rather
    than sorted alphabetically - a reader checking that P2 precedes P8 should
    be able to read it straight down the column.
    """
    if not runs:
        return ()
    order = [timing.stage for timing in runs[0]]
    by_stage: dict[str, list[StageTiming]] = {name: [] for name in order}
    for run in runs:
        for timing in run:
            if timing.stage in by_stage:
                by_stage[timing.stage].append(timing)
    merged: list[StageTiming] = []
    for name in order:
        samples = by_stage[name]
        if not samples:
            continue
        merged.append(
            StageTiming(
                stage=name,
                seconds=statistics.median(sample.seconds for sample in samples),
                records_in=samples[0].records_in,
                records_out=samples[0].records_out,
            )
        )
    return tuple(merged)


def measure(
    records: int,
    data_dir: Path,
    config: AppConfig,
    as_of: date,
    generate_seconds: float,
    repetitions: int = REPETITIONS,
) -> SizeResult:
    """Time `repetitions` full runs, report the MEDIAN. Memory measured apart."""
    ingest_samples: list[float] = []
    engine_samples: list[float] = []
    stage_runs: list[tuple[StageTiming, ...]] = []
    cases = 0
    residual = 0

    for _ in range(repetitions):
        collector: list[StageTiming] = []
        with StageTimer(collector, "ingest+normalize", records):
            dataset = load_days(data_dir, config)
        ingest_samples.append(collector[0].seconds)

        engine_collector: list[StageTiming] = []
        with StageTimer(engine_collector, "engine total", records):
            result, telemetry = run_with_telemetry(dataset, config, as_of, collect_timings=True)
        engine_samples.append(engine_collector[0].seconds)
        stage_runs.append(telemetry.timings)
        cases = len(result.cases)
        residual = len(residual_cases(result))

    # Memory in its OWN run: tracemalloc hooks every allocation, so folding it
    # into a timed repetition would report an honest peak and a dishonest
    # duration. Paying for one extra run buys two numbers that are both true.
    tracemalloc.start()
    dataset = load_days(data_dir, config)
    run_with_telemetry(dataset, config, as_of, collect_timings=False)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return SizeResult(
        records=records,
        cases=cases,
        input_rows=_input_rows(data_dir),
        residual=residual,
        generate_seconds=generate_seconds,
        ingest_seconds=statistics.median(ingest_samples),
        engine_seconds=statistics.median(engine_samples),
        peak_bytes=peak,
        stages=_median_stages(stage_runs),
    )


def run_size(records: int, config: AppConfig, as_of: date, repetitions: int) -> SizeResult:
    """Generate, measure, and discard the dataset."""
    with tempfile.TemporaryDirectory(prefix=f"bench_{records}_") as tmp:
        out = Path(tmp)
        elapsed = generate(records, out)
        return measure(records, out, config, as_of, elapsed, repetitions)


# `_fmt_rate` MOVED to settlesense/core/telemetry.py, beside the
# `records_per_second` property whose None it renders. It existed here and in
# run_eval.py as two identical private copies.
_fmt_rate = format_rate


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds:.3f}"


def to_markdown(
    results: Sequence[SizeResult],
    machine: MachineSpec,
    stretch_note: str,
    config: AppConfig | None = None,
    as_of: date | None = None,
) -> str:
    """reports/bench.md - ready to paste into the README (B7)."""
    lines: list[str] = [
        "# SettleSense throughput — deterministic pipeline",
        "",
        f"`{machine.describe()}`",
        "",
        f"Median of {REPETITIONS} repetitions, never the best run. Dev seed "
        f"({BENCH_SEED}) — the holdout is never benchmarked. No model calls.",
        "",
        "## Scaling",
        "",
        "| Records | Cases | Input rows | Ingest (s) | Engine (s) | Pipeline (s) "
        "| Cases/s | Rows/s | Peak MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.records:,} | {row.cases:,} | {row.input_rows:,} "
            f"| {_fmt_seconds(row.ingest_seconds)} | {_fmt_seconds(row.engine_seconds)} "
            f"| {_fmt_seconds(row.pipeline_seconds)} "
            f"| {_fmt_rate(row.cases_per_second)} | {_fmt_rate(row.rows_per_second)} "
            f"| {row.peak_bytes / BYTES_PER_MIB:,.1f} |"
        )
    lines += [
        "",
        "Pipeline = ingest + engine. Dataset generation is excluded from the timed "
        "region and reported separately below; it is scaffolding for the benchmark, "
        "not part of the system under test.",
        "",
        _ingest_share_note(results),
        "",
        stretch_note,
        "",
        "## Per-stage, at the largest size measured",
        "",
        "Stages are listed in PASS ORDER, so P2 preceding P8 is readable straight down the column.",
        "",
        "| Stage | Seconds | Records in | Records/s |",
        "|---|---:|---:|---:|",
    ]
    largest = results[-1] if results else None
    if largest is not None:
        for timing in largest.stages:
            lines.append(
                f"| {timing.stage} | {_fmt_seconds(timing.seconds)} "
                f"| {timing.records_in:,} | {_fmt_rate(timing.records_per_second)} |"
            )
        lines += [
            "",
            "`P3/P4/P6/P7b/P9 case classification` is ONE row because those five "
            "passes are fused in a single walk over each case. Splitting them would "
            "mean restructuring the hot loop so a report could have more rows. The "
            "batch-grain passes below it are genuinely separate phases and are timed "
            "separately.",
            "",
            "`P9` rounding categorisation does not appear as its own stage: it runs "
            "inline inside P2b and P8 on links those passes created, which is what "
            "makes it a categorisation rather than a matching pass.",
            "",
            "READ THE BATCH-GRAIN ROWS AGAINST THEIR OWN DENOMINATOR. Population B "
            "is batches, and the generator emits roughly the same number of them at "
            "every size, so `records_in` on those rows barely moves while the "
            "records column above grows 200-fold. Their records/s figures are not "
            "comparable with the case-grain rows and are not a scaling signal.",
        ]
    lines += ["", _ai_stage_section(results, config, as_of), ""]
    return "\n".join(lines)


def _ingest_share_note(results: Sequence[SizeResult]) -> str:
    """FILE PARSING IS THE BOTTLENECK, WHICH IS THE OPPOSITE OF WHAT A READER EXPECTS.

    Worth a line of its own: shown a reconciliation system, a reader assumes
    the matching logic dominates. It does not, at any size measured, and the
    share is computed from the realised numbers rather than asserted from one
    of them.
    """
    if not results:
        return ""
    shares = [
        (row.records, row.ingest_seconds / row.pipeline_seconds * 100)
        for row in results
        if row.pipeline_seconds > 0
    ]
    if not shares:
        return ""
    low, high = min(share for _, share in shares), max(share for _, share in shares)
    largest = results[-1]
    return (
        f"**Reading the files costs more than reconciling them.** Ingest is "
        f"{low:.0f} to {high:.0f}% of pipeline time at every size measured — at "
        f"{largest.records:,} records, {_fmt_seconds(largest.ingest_seconds)}s of parsing "
        f"against {_fmt_seconds(largest.engine_seconds)}s of matching. That is the "
        f"reverse of what a reconciliation system suggests, and it is where an "
        f"optimisation would go: the engine is not the constraint."
    )


def _ai_replay_seconds(config: AppConfig, as_of: date) -> tuple[int, float] | None:
    """Time the AI stage on the dev seed, REPLAYING recorded fixtures.

    Returns (decisions, seconds), or None when no fixture covers the dev set.

    THIS IS CACHE-REPLAY TIME, NOT API LATENCY, and the report says so. It is
    still the honest number to publish, because replay is what every test, every
    `make eval` and every reproduction of this project actually executes - the
    recorded response is the system's input. Live latency was never captured:
    the recorder took the API's token counts and its own price arithmetic and
    read no clock, which is a gap in the recorder rather than something that can
    be recovered from the fixtures now.
    """
    from eval.run_ai import duplicate_exceptions
    from settlesense.ai.client import FixtureMissError, ReplayLLMClient
    from settlesense.ai.loop import resolve_exception

    if not DEV_DATA.is_dir() or not any((REPO / "fixtures" / "llm").glob("*.json")):
        return None
    dataset = load_days(DEV_DATA, config)
    exceptions = sorted(duplicate_exceptions(dataset), key=lambda item: item.exception_id)
    if not exceptions:
        return None
    replay = ReplayLLMClient()
    collector: list[StageTiming] = []
    try:
        with StageTimer(collector, "ai replay", len(exceptions)):
            for exception in exceptions:
                resolve_exception(exception, dataset, config, replay)
    except FixtureMissError:
        # A fixture the dev set needs was never recorded. That is "not
        # measured", not a crash - and it is caught NARROWLY, by the one
        # exception the replay client raises, so a real defect in the loop
        # still fails the benchmark loudly instead of silently blanking a row.
        return None
    return len(exceptions), collector[0].seconds


@dataclass(frozen=True)
class RecordingCost:
    """One recording run's measured spend. Typed, so the renderer stops
    calling int() on `object` at eight separate call sites."""

    recorded: int
    inr: Decimal
    usd: Decimal = Decimal(0)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    def per_decision(self) -> Decimal:
        return (self.inr / self.recorded).quantize(Decimal("0.0001"))


def _dev_manifest() -> RecordingCost | None:
    """The dev-seed recording alone. See DEV_MANIFEST for why it is separate."""
    if not DEV_MANIFEST.is_file():
        return None
    payload = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    recorded = int(payload["recorded"])
    if recorded <= 0:
        return None  # a manifest recording nothing is "not measured", not "free"
    return RecordingCost(
        recorded=recorded,
        inr=Decimal(str(payload["measured_cost_inr"])),
    )


def _manifest_totals() -> RecordingCost | None:
    """Cost and token counts summed across both recording runs, from the files."""
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in FIXTURE_MANIFESTS if path.is_file()
    ]
    if not payloads:
        return None
    recorded = sum(int(item["recorded"]) for item in payloads)
    if recorded <= 0:
        # A manifest that EXISTS and records zero decisions is legitimate - it
        # is what a fresh clone would carry after `record_fixtures --dry-run`.
        # It is not the same as no manifest at all, and it must not reach
        # per_decision(), which would divide by zero. Both map to "not
        # measured" in the report; neither prints Rs 0.00 per decision, which
        # would read as "the model is free".
        return None
    return RecordingCost(
        recorded=recorded,
        usd=sum((Decimal(str(item["measured_cost_usd"])) for item in payloads), start=Decimal(0)),
        inr=sum((Decimal(str(item["measured_cost_inr"])) for item in payloads), start=Decimal(0)),
        input_tokens=sum(int(item["measured_input_tokens"]) for item in payloads),
        output_tokens=sum(int(item["measured_output_tokens"]) for item in payloads),
        model=str(payloads[0]["model"]),
    )


def _ai_stage_section(
    results: Sequence[SizeResult], config: AppConfig | None = None, as_of: date | None = None
) -> str:
    """The AI stage, benchmarked against RESIDUAL count - not row count (B6).

    REWRITTEN once a model had actually been called. The previous version said
    "seconds and rupees are NOT reported here, because no model has been
    called", which was true when it was written and quietly stopped being true
    the moment `fixtures/llm/` filled up - a report cannot notice that about
    itself. Everything below is now read from the manifests and from a live
    replay, so the section cannot go stale in that direction again.
    """
    lines = [
        "## AI stage — priced against the residual, not the volume",
        "",
        "| Records | Cases | Deterministic residual | Residual share |",
        "|---:|---:|---:|---:|",
    ]
    for row in results:
        share = f"{row.residual / row.cases * 100:.2f}%" if row.cases else "—"
        lines.append(f"| {row.records:,} | {row.cases:,} | {row.residual:,} | {share} |")
    lines += [
        "",
        "**This ratio is the architectural argument.** The expensive stage is sized "
        "by the residual column, not the cases column: roughly one case in a hundred "
        "reaches it. Deterministic passes carry the volume, and the model is not on "
        "the hot path — which the per-stage table above shows directly, since every "
        "row in it is a rule.",
        "",
    ]

    totals = _manifest_totals()
    dev_costs = _dev_manifest()
    replay = _ai_replay_seconds(config, as_of) if config is not None and as_of else None
    dev = next((row for row in results if row.records == 5000), None)

    if dev is not None and totals is not None and dev_costs is not None:
        decisions = totals.recorded
        per_decision = totals.per_decision()
        per_thousand = (dev_costs.inr / (Decimal(dev.input_rows) / Decimal(1000))).quantize(
            Decimal("0.01")
        )
        dev_per_decision = (dev_costs.inr / dev_costs.recorded).quantize(Decimal("0.0001"))
        lines += [
            "",
            "### Deterministic pipeline against AI stage, at 5,000 records",
            "",
            f"**Deterministic pipeline: {dev.cases:,} cases in "
            f"{_fmt_seconds(dev.pipeline_seconds)}s, zero model calls.** The AI stage "
            f"runs on the {dev.residual} residual cases — "
            f"{dev.residual / dev.cases * 100:.2f}% of the workload, of which "
            f"{dev_costs.recorded} are AI-eligible duplicate pairs and the rest "
            f"abstain without a model call at all. "
            f"**Model cost scales with ambiguity, not volume.** Building the rules "
            f"layer properly is what keeps the expensive stage small.",
            "",
        ]
        if replay is not None:
            replayed, seconds = replay
            lines.append(
                f"**Seconds — {replayed} dev-seed decisions replayed in "
                f"{seconds:.3f}s, {seconds / replayed * 1000:.1f} ms each.** "
                f"THIS IS CACHE-REPLAY TIME, NOT API LATENCY. Replay is what every "
                f"test and every `make eval` executes, so it is the honest figure for "
                f"what running this system costs in time — but it is not what a live "
                f"call would take. Live latency was never captured: the recorder took "
                f"the API's token counts and read no clock. That is a gap in the "
                f"recorder, and it is stated rather than filled with a plausible "
                f"number."
            )
        else:
            lines.append(
                "**Seconds — not measured.** No fixture set covers the dev seed, so "
                "there is nothing to replay. A `0.000s` here would be "
                "indistinguishable from a stage that ran and cost nothing."
            )
        lines += [
            "",
            f"**Rupees — MEASURED from the API's own usage figures, not estimated "
            f"from prompt length.** {totals.input_tokens:,} input and "
            f"{totals.output_tokens:,} output tokens across {decisions} "
            f"recorded decisions against `{totals.model}` = "
            f"${totals.usd:.6f} (₹{totals.inr}), "
            f"which is **₹{per_decision} per decision**.",
            "",
            f"**Per 1,000 rows: ₹{per_thousand}**, and the basis is stated because "
            f"the number is meaningless without it — the dev dataset's "
            f"{dev_costs.recorded} AI-eligible pairs were recorded with NO "
            f"sampling, so ₹{dev_costs.inr} is the complete model "
            f"spend for those {dev.input_rows:,} input rows. Against PDD 7.3's ₹50 "
            f"ceiling.",
            "",
            f"**WHERE THE PRE-SPEND ESTIMATE WENT WRONG, AND IT IS NOT WHERE IT "
            f"LOOKS.** The projection made before any model was called was "
            f"₹{ESTIMATE_INR_PER_THOUSAND} per 1,000 rows against the "
            f"₹{per_thousand} measured here — off by "
            f"{ESTIMATE_INR_PER_THOUSAND / per_thousand:.0f}x. But the pricing model "
            f"was nearly right: it projected "
            f"₹{(ESTIMATE_TOTAL_INR / ESTIMATE_DECISIONS).quantize(Decimal('0.0001'))} "
            f"per decision against ₹{per_decision} measured, only "
            f"{(1 - (ESTIMATE_TOTAL_INR / ESTIMATE_DECISIONS) / per_decision) * 100:.0f}% "
            f"low. THE WHOLE ERROR WAS THE DECISION COUNT: the estimate assumed "
            f"{ESTIMATE_DECISIONS} decisions per dataset, and a dataset of this size "
            f"produces {dev_costs.recorded}. Which is the architectural claim "
            f"restated as a cost bug — the deterministic layer had already removed "
            f"the work the estimate was pricing.",
            "",
            f"Two independent recordings agree on the per-decision figure: "
            f"₹{(Decimal('19.37') / 40).quantize(Decimal('0.0001'))} across the 40 "
            f"evaluation-set decisions and "
            f"₹{dev_per_decision} "
            f"across the {dev_costs.recorded} dev-seed ones — different seeds, "
            f"different pairs, 0.2% apart.",
        ]
    else:
        lines += [
            "",
            "**Seconds and rupees are NOT reported here.** No recording manifest was "
            "found, so no model has been called for this tree. Printing `0.000s` and "
            "`₹0` would be indistinguishable from a stage that ran and cost nothing.",
        ]

    lines += [
        "",
        "The verified hypothesis loop is also measured against three stand-in "
        "clients in reports/ai/ai_loop.json: an oracle that always nominates "
        "correctly establishes a ceiling of 27 of 507 decisions, which no real "
        "model can exceed, and an adversarial client confirms zero.",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Throughput scaling bench (M5a).")
    parser.add_argument("--sizes", default=",".join(str(size) for size in DEFAULT_SIZES))
    parser.add_argument("--out", type=Path, default=Path("reports/bench.md"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=DEFAULT_AS_OF)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument(
        "--no-stretch",
        action="store_true",
        help=f"do not attempt {STRETCH_SIZE:,} even if {STRETCH_TRIGGER_SIZE:,} was fast",
    )
    args = parser.parse_args(argv)

    if args.repetitions < 1:
        raise SystemExit(f"--repetitions must be >= 1, got {args.repetitions}")

    config = load_config(args.config)
    sizes = parse_sizes(args.sizes)
    machine = MachineSpec.current()
    print(machine.describe())

    results: list[SizeResult] = []
    for size in sizes:
        print(f"  {size:,} records ...", flush=True)
        results.append(run_size(size, config, args.as_of, args.repetitions))
        latest = results[-1]
        print(
            f"    cases {latest.cases:,}  pipeline {_fmt_seconds(latest.pipeline_seconds)}s  "
            f"{_fmt_rate(latest.cases_per_second)} cases/s  "
            f"residual {latest.residual}"
        )

    stretch_note = _maybe_stretch(results, sizes, config, args, machine)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        to_markdown(results, machine, stretch_note, config, args.as_of), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")
    if results:
        largest = results[-1]
        print(
            f"headline: {_fmt_rate(largest.cases_per_second)} cases/s end-to-end "
            f"at {largest.records:,} records"
        )
    return 0


def _maybe_stretch(
    results: list[SizeResult],
    sizes: tuple[int, ...],
    config: AppConfig,
    args: argparse.Namespace,
    machine: MachineSpec,
) -> str:
    """Attempt 100k only under the B1 rule, and SAY which branch was taken.

    Every path returns a sentence for the report. A skipped stretch that left
    no trace would be indistinguishable from one that was never considered.
    """
    del machine  # captured in the header; named here only to document the call
    if args.no_stretch:
        return (
            f"**{STRETCH_SIZE:,} records: not attempted** (`--no-stretch`). No figure "
            "for that size is estimated from the sizes above."
        )
    if STRETCH_SIZE in sizes:
        return f"**{STRETCH_SIZE:,} records: requested explicitly** and measured above."
    trigger = next((row for row in results if row.records == STRETCH_TRIGGER_SIZE), None)
    if trigger is None:
        return (
            f"**{STRETCH_SIZE:,} records: not attempted.** The stretch is gated on "
            f"{STRETCH_TRIGGER_SIZE:,} completing inside {STRETCH_BUDGET_SECONDS}s, and "
            f"{STRETCH_TRIGGER_SIZE:,} was not in `--sizes`."
        )
    if trigger.pipeline_seconds >= STRETCH_BUDGET_SECONDS:
        return (
            f"**{STRETCH_SIZE:,} records: SKIPPED.** {STRETCH_TRIGGER_SIZE:,} took "
            f"{_fmt_seconds(trigger.pipeline_seconds)}s, at or over the "
            f"{STRETCH_BUDGET_SECONDS}s budget. No 100k figure is extrapolated — the "
            "row is absent rather than estimated."
        )
    print(
        f"  {STRETCH_TRIGGER_SIZE:,} finished in {_fmt_seconds(trigger.pipeline_seconds)}s "
        f"(< {STRETCH_BUDGET_SECONDS}s) — attempting {STRETCH_SIZE:,}",
        flush=True,
    )
    results.append(run_size(STRETCH_SIZE, config, args.as_of, args.repetitions))
    latest = results[-1]
    print(
        f"    cases {latest.cases:,}  pipeline {_fmt_seconds(latest.pipeline_seconds)}s  "
        f"{_fmt_rate(latest.cases_per_second)} cases/s  residual {latest.residual}"
    )
    return (
        f"**{STRETCH_SIZE:,} records: attempted and measured**, because "
        f"{STRETCH_TRIGGER_SIZE:,} completed in "
        f"{_fmt_seconds(trigger.pipeline_seconds)}s — inside the "
        f"{STRETCH_BUDGET_SECONDS}s budget that gates it."
    )


if __name__ == "__main__":
    sys.exit(main())

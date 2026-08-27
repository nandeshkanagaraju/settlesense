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
import statistics
import subprocess
import sys
import tempfile
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from eval.run_eval import load_days
from settlesense.config import AppConfig, load_config
from settlesense.core.telemetry import MachineSpec, StageTimer, StageTiming
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


def _input_rows(data_dir: Path) -> int:
    """Total CSV data rows on disk, header excluded.

    Counted from the FILES rather than from the parsed dataset, so a row the
    ingest layer dropped still counts as work the pipeline was handed.

    A MISSING directory and an EMPTY one are different, and this refused to
    tell them apart in its first version: `Path.glob` on a directory that does
    not exist yields nothing rather than raising, so both returned 0 and the
    scaling table would have shown a clean `0 input rows` for a dataset that
    was never generated. Header-only files still return 0 - that is legitimate
    data, and day 1 genuinely has no bank credits because settlement is T+N.
    """
    if not data_dir.is_dir():
        raise SystemExit(
            f"{data_dir} does not exist. A benchmark over a dataset that was never "
            "generated would report zero rows, which is indistinguishable from a "
            "dataset that is legitimately empty."
        )
    paths = sorted(data_dir.glob("day*_*.csv"))
    if not paths:
        raise SystemExit(f"{data_dir} exists but holds no day*_*.csv files")
    total = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
        total += max(lines - 1, 0)
    return total


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


def _fmt_rate(rate: float | None) -> str:
    return "—" if rate is None else f"{rate:,.0f}"


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds:.3f}"


def to_markdown(results: Sequence[SizeResult], machine: MachineSpec, stretch_note: str) -> str:
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
    lines += ["", _ai_stage_section(results), ""]
    return "\n".join(lines)


def _ai_stage_section(results: Sequence[SizeResult]) -> str:
    """The AI stage, benchmarked against RESIDUAL count - not row count (B6).

    This section reports what has been measured and states plainly what has
    not. M7 is built, but no model has been called and `fixtures/llm/` is
    empty, so there is no timing to report; a zero here would read as
    "instant" rather than "never invoked".
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
        "**Seconds and rupees are NOT reported here, because no model has been "
        "called.** The verified hypothesis loop IS built (M7) and is measured in "
        "reports/ai/ai_loop.json - but with stand-in clients, not a model: an "
        "oracle that always nominates correctly establishes a ceiling of 27 of "
        "507 decisions, which no real model can exceed. `fixtures/llm/` holds "
        "zero recordings, so there is no timing to report. Printing `0.000s` and "
        "`Rs 0` would be indistinguishable in this table from a stage that ran "
        "and cost nothing. The harness takes both numbers the moment a fixture "
        "set exists, reading them from the replay cache so re-running is free.",
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
    args.out.write_text(to_markdown(results, machine, stretch_note), encoding="utf-8")
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

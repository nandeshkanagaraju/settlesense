"""M5a - telemetry, the separation it exists to enforce, and the bench harness.

THE SEPARATION TESTS COME FIRST because they are what the module is for. A
throughput number is easy; keeping wall-clock data out of the business result
is the part that quietly fails. The failure mode is specific and silent: a
`seconds` field lands in ReconciliationResult, two identical runs stop
comparing equal, and someone writes a strip step in the comparator to make the
goldens pass again. At that point the result is no longer a function of the
input and nothing downstream can be trusted, while every test still passes.

So S6 - byte-identical output with instrumentation on and off - is the single
most important assertion in this file. Everything else is scaffolding for it.

NO SLEEPS. Durations are produced by real work in a busy loop, so the suite
does not spend wall-clock time proving that a clock advances.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import subprocess
import sys
import time
import typing
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from eval.bench import (
    BENCH_SEED,
    DEFAULT_SIZES,
    REPETITIONS,
    STRETCH_SIZE,
    SizeResult,
    _median_stages,
    main,
    parse_sizes,
    to_markdown,
)
from eval.run_eval import load_days
from settlesense.config import AppConfig, load_config
from settlesense.core import telemetry as telemetry_module
from settlesense.core.telemetry import MachineSpec, RunTelemetry, StageTimer, StageTiming
from settlesense.matching.engine import run_with_telemetry
from settlesense.types import ReconciliationResult

REPO = Path(__file__).resolve().parent.parent
SETTLESENSE = REPO / "settlesense"
DATA = REPO / "data" / "dev"
AS_OF = date(2026, 11, 30)

TELEMETRY_REL = "settlesense/core/telemetry.py"


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture(scope="module")
def dataset(config: AppConfig) -> Any:
    return load_days(DATA, config)


def _busy(iterations: int = 20_000) -> int:
    """Real work, so a duration is real. A sleep would test the scheduler."""
    return sum(index * index for index in range(iterations))


# ===========================================================================
# S. SEPARATION - the point of the module
# ===========================================================================


def _canonical(result: ReconciliationResult) -> bytes:
    """The bytes a golden comparison would hold.

    Defined HERE rather than imported because no result serializer exists yet
    (goldens are a later gate). What matters for S6 is that some total, stable
    encoding of the result is unchanged by instrumentation; this is one.
    """
    return json.dumps(
        dataclasses.asdict(result), sort_keys=True, default=str, separators=(",", ":")
    ).encode("utf-8")


def test_s6_instrumentation_does_not_change_the_result(dataset: Any, config: AppConfig) -> None:
    """THE ONE THAT MATTERS. Timing must not change a decision (S6).

    Byte-for-byte, not field-by-field: a comparison that walked named fields
    would miss a new field appearing on only one of the two paths, which is
    exactly the regression this guards.
    """
    instrumented, telemetry_on = run_with_telemetry(dataset, config, AS_OF, collect_timings=True)
    plain, telemetry_off = run_with_telemetry(dataset, config, AS_OF, collect_timings=False)

    assert telemetry_on.timings, "precondition: the instrumented run recorded no stages"
    assert telemetry_off.timings == (), "the uninstrumented run recorded timings anyway"

    on_bytes, off_bytes = _canonical(instrumented), _canonical(plain)
    assert on_bytes == off_bytes, "instrumentation changed the serialized result"
    assert instrumented == plain, "instrumented and plain results are not equal"
    print(
        f"\n  identical over {len(on_bytes):,} bytes; "
        f"{len(telemetry_on.timings)} stages timed, {len(telemetry_off.timings)} when off"
    )


@pytest.mark.charter_guard
def test_s6b_the_byte_comparison_can_actually_fail(dataset: Any, config: AppConfig) -> None:
    """FAULT INJECTION for S6. A one-field difference must be caught.

    Without this, S6 passing is equally consistent with `_canonical` returning
    a constant.
    """
    result, _telemetry = run_with_telemetry(dataset, config, AS_OF, collect_timings=True)
    mutated = dataclasses.replace(result, config_hash=result.config_hash + "x")
    assert _canonical(result) != _canonical(mutated), "the serializer ignores a changed field"
    trimmed = dataclasses.replace(result, cases=result.cases[:-1])
    assert _canonical(result) != _canonical(trimmed), "the serializer ignores a dropped case"
    print("\n  serializer distinguishes a changed field and a dropped case")


def test_s1_the_pipeline_returns_two_values(dataset: Any, config: AppConfig) -> None:
    """S1. A tuple, not a result carrying telemetry inside it."""
    returned = run_with_telemetry(dataset, config, AS_OF, collect_timings=True)
    assert isinstance(returned, tuple) and len(returned) == 2, returned
    result, telemetry = returned
    assert isinstance(result, ReconciliationResult)
    assert isinstance(telemetry, RunTelemetry)
    print(f"\n  ({type(result).__name__}, {type(telemetry).__name__})")


def _type_graph(root: type) -> dict[str, Any]:
    """Every annotation reachable from `root`, keyed by dotted path."""
    found: dict[str, Any] = {}
    seen: set[type] = set()

    def walk(node: type, path: str) -> None:
        if not dataclasses.is_dataclass(node) or node in seen:
            return
        seen.add(node)
        hints = typing.get_type_hints(node)
        for field in dataclasses.fields(node):
            annotation = hints[field.name]
            here = f"{path}.{field.name}"
            found[here] = annotation
            stack = [annotation]
            while stack:
                current = stack.pop()
                args = typing.get_args(current)
                if args:
                    stack.extend(args)
                elif isinstance(current, type):
                    walk(current, here)

    walk(root, root.__name__)
    return found


_FORBIDDEN_IN_RESULT = (float, __import__("datetime").datetime)
_TIMING_NAME = re.compile(r"seconds|duration|elapsed|timing|_at$|timestamp|rss|peak", re.I)


def test_s2_the_result_graph_holds_no_float_and_no_timestamp() -> None:
    """S2. Checked over the TRANSITIVE graph, not just the top-level fields.

    `date` is allowed and `float`/`datetime` are not: a settlement's value date
    is business data the result is ABOUT, while a wall-clock instant is a fact
    about the run that produced it. The distinction is the whole of D6.
    """
    graph = _type_graph(ReconciliationResult)
    assert len(graph) > 20, f"the walker only reached {len(graph)} fields - it is not walking"

    offenders = [
        f"{path}: {annotation}"
        for path, annotation in graph.items()
        for member in ([annotation, *typing.get_args(annotation)])
        if member in _FORBIDDEN_IN_RESULT
    ]
    assert not offenders, "wall-clock or float types in the result graph:\n" + "\n".join(offenders)

    named = [path for path in graph if _TIMING_NAME.search(path.rsplit(".", 1)[-1])]
    assert not named, f"fields named like telemetry in the result graph: {named}"
    print(f"\n  {len(graph)} annotations reachable from ReconciliationResult, none of them a clock")


@dataclasses.dataclass(frozen=True)
class _PlantedInner:
    """Module level, not nested in the test: `from __future__ import
    annotations` stringifies hints, and get_type_hints cannot resolve a name
    that only exists in a function's local scope."""

    seconds: float


@dataclasses.dataclass(frozen=True)
class _PlantedOuter:
    rows: tuple[_PlantedInner, ...]
    label: str


@pytest.mark.charter_guard
def test_s2b_the_graph_walker_catches_a_planted_float() -> None:
    """FAULT INJECTION for S2, in both shapes it checks."""
    graph = _type_graph(_PlantedOuter)
    assert "_PlantedOuter.rows.seconds" in graph, graph
    hits = [
        path
        for path, annotation in graph.items()
        for member in ([annotation, *typing.get_args(annotation)])
        if member in _FORBIDDEN_IN_RESULT
    ]
    assert hits == ["_PlantedOuter.rows.seconds"], hits
    assert _TIMING_NAME.search("seconds"), "the name pattern does not match 'seconds'"
    print(f"\n  planted float found at {hits[0]}")


def test_s3_types_does_not_import_telemetry_even_transitively() -> None:
    """S3. Verified by IMPORTING types.py in a clean interpreter.

    An AST scan of types.py would prove only that the import is not written
    there; it would miss telemetry arriving through something types.py imports.
    A fresh process and a look at sys.modules answers the real question.
    """
    source = (SETTLESENSE / "types.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    written = [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and "telemetry" in (node.module or "")
    ]
    assert not written, f"types.py imports telemetry directly: {written}"

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import settlesense.types, sys;print('settlesense.core.telemetry' in sys.modules)",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "False", (
        f"importing settlesense.types pulls in telemetry: {probe.stdout!r}"
    )
    print("\n  fresh interpreter: settlesense.types does not reach core.telemetry")


@pytest.mark.charter_guard
def test_s3b_the_import_probe_would_notice(dataset: Any) -> None:
    """FAULT INJECTION for S3: the probe reports True for a module that DOES."""
    del dataset
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import settlesense.matching.engine, sys;"
            "print('settlesense.core.telemetry' in sys.modules)",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "True", (
        "the engine is known to import telemetry; a probe that cannot see that "
        f"cannot prove types.py does not either. Got {probe.stdout!r}"
    )
    print("\n  probe confirmed positive on engine.py, which does import telemetry")


_STRIP_PATTERN = re.compile(
    r"def\s+\w*(strip|without|drop|scrub|sanit)\w*(timing|telemetry|seconds|clock)\w*", re.I
)


@pytest.mark.charter_guard
def test_s4_nothing_strips_timing_out_of_a_result() -> None:
    """S4. There is no strip step, because there is nothing to strip.

    The brief's instruction was that if a strip step feels necessary, a field
    is in the wrong object. NOTE: golden infrastructure does not exist yet
    (`tests/golden/` is unbuilt), so this asserts the checkable half today -
    that no such helper exists anywhere in the source tree - and S2 above
    asserts the other half, that the result carries nothing worth stripping.
    """
    offenders: list[str] = []
    for package in ("settlesense", "eval"):
        for path in sorted((REPO / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _STRIP_PATTERN.search(line):
                    offenders.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert not offenders, (
        "a timing-strip helper exists - which means a timing field exists "
        "somewhere it should not:\n" + "\n".join(offenders)
    )
    assert _STRIP_PATTERN.search("def strip_timing(result):"), "the strip detector matches nothing"
    print("\n  no strip helper in settlesense/ or eval/; detector verified against a sample")


def test_s5_telemetry_touches_no_database_and_writes_only_reports() -> None:
    """S5. Telemetry goes to reports/. It never reaches the state store."""
    sources = {
        TELEMETRY_REL: (SETTLESENSE / "core" / "telemetry.py").read_text(encoding="utf-8"),
        "eval/bench.py": (REPO / "eval" / "bench.py").read_text(encoding="utf-8"),
    }
    for rel, source in sources.items():
        for banned in ("sqlite3", "settlesense.state", "state_db", ".db"):
            assert banned not in source, f"{rel} references the state store via {banned!r}"

    telemetry_source = sources[TELEMETRY_REL]
    for writer in ("open(", "write_text", "Path("):
        assert writer not in telemetry_source, (
            f"{TELEMETRY_REL} performs I/O ({writer}) - telemetry is data, and the "
            "harness decides where it lands"
        )
    bench_source = sources["eval/bench.py"]
    written = re.findall(r'Path\("([^"]+)"\)', bench_source)
    assert written and all(
        target.startswith("reports/") or target == "config" for target in written
    )
    print(f"\n  telemetry does no I/O; bench default output paths: {written}")


# ===========================================================================
# T. THE TELEMETRY TYPES
# ===========================================================================


def test_t1_stage_timing_carries_the_four_fields_and_a_rate() -> None:
    """T1. Fields, and a throughput that divides by the right thing."""
    timing = StageTiming(stage="P1 build cases", seconds=2, records_in=1000, records_out=900)
    assert (timing.stage, timing.records_in, timing.records_out) == ("P1 build cases", 1000, 900)
    rate = timing.records_per_second
    assert rate is not None and abs(rate - 500) < 1, rate
    print(f"\n  1000 records in 2s -> {rate:.0f}/s (input basis, not the 900 that survived)")


@pytest.mark.boundary_refusal
def test_t1b_records_per_second_guards_division_by_zero() -> None:
    """T1. Zero duration returns NONE - not inf, not 0.0, not a crash.

    All three alternatives put a false statement in a report: a crash loses a
    run over an unmeasurably fast stage, 0.0 reads as slow, inf reads as
    infinitely fast. None renders as a dash.
    """
    for seconds in (0, -1):
        timing = StageTiming(stage="instant", seconds=seconds, records_in=10, records_out=10)
        assert timing.records_per_second is None, (seconds, timing.records_per_second)
    zero_records = StageTiming(stage="empty", seconds=1, records_in=0, records_out=0)
    assert zero_records.records_per_second == 0, "an empty stage did work at zero rate, not None"
    print("\n  seconds=0 -> None; seconds<0 -> None; records=0 -> 0.0 (a real answer)")


def test_t2_machine_spec_captures_cpu_cores_ram_and_python() -> None:
    """T2. Captured automatically (B5), and every field carries something."""
    spec = MachineSpec.current()
    assert spec.cpu and spec.cpu != "unknown", spec.cpu
    assert spec.cores >= 1, spec.cores
    assert spec.ram_bytes > 0, "no RAM figure on a platform that should report one"
    assert re.match(r"^\d+\.\d+", spec.python_version), spec.python_version
    described = spec.describe()
    for fragment in (spec.cpu, str(spec.cores), spec.python_version):
        assert fragment in described, f"{fragment!r} missing from {described!r}"
    print(f"\n  {described}")


@pytest.mark.boundary_refusal
def test_t2b_an_unknown_ram_figure_says_unknown() -> None:
    """FAULT INJECTION for T2. 0 bytes must never render as '0 GiB'."""
    spec = MachineSpec(cpu="cpu", cores=1, ram_bytes=0, python_version="3.11.0", platform="p")
    assert "unknown RAM" in spec.describe(), spec.describe()
    assert "0 GiB" not in spec.describe(), spec.describe()
    print(f"\n  {spec.describe()}")


def test_t3_run_telemetry_shape_and_total() -> None:
    """T3. timings tuple, peak_rss_bytes, machine - and a total that adds up."""
    timings = (
        StageTiming(stage="a", seconds=1, records_in=1, records_out=1),
        StageTiming(stage="b", seconds=2, records_in=1, records_out=1),
    )
    telemetry = RunTelemetry(timings=timings, peak_rss_bytes=4096)
    assert isinstance(telemetry.timings, tuple)
    assert telemetry.peak_rss_bytes == 4096
    assert isinstance(telemetry.machine, MachineSpec)
    assert telemetry.total_seconds() == 3
    assert telemetry.stage("b") is timings[1]
    assert telemetry.stage("missing") is None
    print(f"\n  total {telemetry.total_seconds()}s over {len(telemetry.timings)} stages")


def test_t4_stage_timer_appends_to_the_collector_it_was_given() -> None:
    """T4. NO GLOBAL STATE - two collectors, interleaved, must not mix.

    This is the property the benchmark depends on: three repetitions run in
    one process, and a module-level list would make run two's numbers depend
    on whether run one happened.
    """
    left: list[StageTiming] = []
    right: list[StageTiming] = []
    with StageTimer(left, "outer", 10) as outer:
        _busy()
        with StageTimer(right, "inner", 5) as inner:
            _busy()
            inner.records_out = 3
        outer.records_out = 7

    assert [t.stage for t in left] == ["outer"], left
    assert [t.stage for t in right] == ["inner"], right
    assert left[0].records_out == 7 and right[0].records_out == 3
    assert left[0].seconds > 0 and right[0].seconds > 0
    assert left[0].seconds > right[0].seconds, (
        "the outer stage contains the inner one and must be longer: "
        f"{left[0].seconds} vs {right[0].seconds}"
    )
    print(
        f"\n  outer {left[0].seconds * 1000:.2f}ms encloses inner {right[0].seconds * 1000:.2f}ms"
    )


def test_t4b_records_out_defaults_to_records_in() -> None:
    """T4. A pass-through stage should not have to restate its own count."""
    collected: list[StageTiming] = []
    with StageTimer(collected, "passthrough", 42):
        pass
    assert collected[0].records_in == collected[0].records_out == 42
    print(f"\n  default records_out = {collected[0].records_out}")


@pytest.mark.charter_guard
def test_t4c_a_none_collector_reads_no_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """T4. Instrumentation off means OFF, not cheap.

    Counted rather than asserted in prose. The engine passes None on every
    production call, so this is what makes "timing is not on the hot path" a
    property of the code instead of a claim about how fast perf_counter is.
    """
    calls: list[int] = []
    real = time.perf_counter

    def counted() -> float:
        calls.append(1)
        return real()

    # telemetry.py does `import time`, so the name it resolves at call time is
    # an attribute of this same module object. Patching it here is patching the
    # lookup telemetry actually performs.
    imported_time = telemetry_module.time  # type: ignore[attr-defined]
    assert imported_time is time, "telemetry no longer uses the stdlib time module"
    monkeypatch.setattr(time, "perf_counter", counted)

    with StageTimer(None, "off", 1):
        pass
    assert calls == [], f"a None collector read the clock {len(calls)} times"

    collected: list[StageTiming] = []
    with StageTimer(collected, "on", 1):
        pass
    assert len(calls) == 2, f"expected exactly enter+exit, got {len(calls)}"
    print(f"\n  collector=None: 0 clock reads; collector=list: {len(calls)}")


@pytest.mark.charter_guard
def test_t5_telemetry_is_the_only_clock_in_settlesense() -> None:
    """T5. An independent scan, not a re-run of the determinism guard's list.

    Deliberately duplicated logic: if the guard's exemption were widened, this
    test - which hard-codes the single permitted path - would still fail.

    AST, NOT A TEXT SCAN. A line-based regex flagged telemetry.py's own
    docstring, which names `date.today()` in prose explaining why it is
    forbidden. A scanner that cannot tell code from a comment about code would
    have to be appeased by rewording documentation, and the wording is the part
    worth keeping.
    """
    clock_names = frozenset({"perf_counter", "monotonic", "time_ns", "now", "utcnow", "today"})
    offenders: list[str] = []
    for path in sorted(SETTLESENSE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(REPO))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            called = node.func.attr
            if called not in clock_names:
                continue
            if rel == TELEMETRY_REL and called == "perf_counter":
                continue
            offenders.append(f"{rel}:{node.lineno}: .{called}()")
    assert not offenders, "a clock outside the telemetry module:\n" + "\n".join(offenders)

    planted = ast.parse("import time\nx = time.perf_counter()\n")
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in clock_names
        for node in ast.walk(planted)
    ), "the AST scan matches nothing"
    print(f"\n  AST-scanned settlesense/; the only clock is {TELEMETRY_REL}")


# ===========================================================================
# B. THE BENCH HARNESS
# ===========================================================================


def test_b1_default_sizes_are_500_5000_25000_and_not_100000() -> None:
    """B1. 100k is a STRETCH, gated on 25k being fast - never a default."""
    assert DEFAULT_SIZES == (500, 5000, 25000), DEFAULT_SIZES
    assert STRETCH_SIZE not in DEFAULT_SIZES, "100000 is in the defaults"
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "BENCH_SIZES ?= 500,5000,25000" in makefile, "the Makefile default drifted"
    assert "500,5000,25000,100000" not in makefile, "the Makefile still runs 100k by default"
    print(f"\n  defaults {DEFAULT_SIZES}, stretch {STRETCH_SIZE:,} gated separately")


@pytest.mark.boundary_refusal
def test_b1b_parse_sizes_refuses_rather_than_coerces() -> None:
    """FAULT INJECTION. Half a benchmark reported as a whole one is the risk."""
    assert parse_sizes("5000,500,5000") == (500, 5000), "not sorted and de-duplicated"
    assert parse_sizes(" 500 , 5000 ") == (500, 5000), "whitespace not tolerated"
    for bad in ("500,abc", "500,-1", "500,0", "", ",,"):
        with pytest.raises(SystemExit) as raised:
            parse_sizes(bad)
        assert "--sizes" in str(raised.value), (bad, raised.value)
    print("\n  refused: 500,abc / 500,-1 / 500,0 / empty / commas-only")


def test_b2_the_median_is_reported_not_the_best() -> None:
    """B2. Three runs; the middle one, not the luckiest.

    Fed a deliberately skewed sample - one fast run and two slow - so a
    min()-based implementation gives a different answer from a median() one.
    """
    runs = [
        (StageTiming(stage="P1", seconds=1, records_in=100, records_out=100),),
        (StageTiming(stage="P1", seconds=5, records_in=100, records_out=100),),
        (StageTiming(stage="P1", seconds=9, records_in=100, records_out=100),),
    ]
    (merged,) = _median_stages(runs)
    assert merged.seconds == 5, f"expected the median 5, got {merged.seconds}"
    assert merged.seconds != 1, "the best run was reported"
    assert REPETITIONS == 3, REPETITIONS
    print(f"\n  samples 1/5/9 -> reported {merged.seconds} (best would have been 1)")


def test_b2b_median_stages_preserves_pass_order_not_alphabetical() -> None:
    """B3. P2 must read before P8 down the column; sorting would break that."""
    runs = [
        (
            StageTiming(stage="P2 exact", seconds=1, records_in=1, records_out=1),
            StageTiming(stage="P8 fuzzy", seconds=1, records_in=1, records_out=1),
            StageTiming(stage="P1 build", seconds=1, records_in=1, records_out=1),
        )
    ]
    assert [t.stage for t in _median_stages(runs)] == ["P2 exact", "P8 fuzzy", "P1 build"]
    assert _median_stages([]) == ()
    print("\n  order preserved from the first run, not sorted")


@pytest.fixture(scope="module")
def bench_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """One real bench run at the smallest size. Shared by the B3-B7 tests."""
    out = tmp_path_factory.mktemp("bench") / "bench.md"
    exit_code = main(
        [
            "--sizes",
            "500",
            "--repetitions",
            "1",
            "--no-stretch",
            "--out",
            str(out),
            "--config",
            str(REPO / "config"),
        ]
    )
    assert exit_code == 0
    return out, out.read_text(encoding="utf-8")


def test_b3_every_pass_appears_as_a_named_stage(bench_run: tuple[Path, str]) -> None:
    """B3. Per-stage, by pass name - so a reader can see what is on the path."""
    _path, report = bench_run
    for pass_name in ("P1", "P2 exact", "P2b", "P7a", "P7b", "P8 fuzzy"):
        assert pass_name in report, f"{pass_name} is missing from the stage table"
    assert "case classification" in report
    assert "batch profile derivation" in report
    # P9 is a categorisation that runs inline; the report must SAY so rather
    # than list a stage that does not exist.
    assert "does not appear as its own stage" in report, "P9's absence is unexplained"
    stage_rows = [line for line in report.splitlines() if line.startswith("| P")]
    assert len(stage_rows) >= 6, f"only {len(stage_rows)} pass rows: {stage_rows}"
    print(f"\n  {len(stage_rows)} pass-named stage rows in the report")


def test_b4_peak_memory_is_measured_and_scales(bench_run: tuple[Path, str]) -> None:
    """B4. tracemalloc, and a figure that responds to size."""
    _path, report = bench_run
    assert "Peak MiB" in report
    committed = (REPO / "reports" / "bench.md").read_text(encoding="utf-8")
    peaks = [
        Decimal(match)
        for match in re.findall(r"\|\s*([\d,]+\.\d)\s*\|$", committed, flags=re.MULTILINE)
    ]
    assert len(peaks) >= 3, f"expected a peak per size row, found {peaks}"
    assert peaks == sorted(peaks), f"peak memory did not grow with size: {peaks}"
    assert all(peak > 0 for peak in peaks), peaks
    print(f"\n  peak MiB across sizes: {[str(p) for p in peaks]}")


def test_b5_the_machine_spec_is_in_the_report_header(bench_run: tuple[Path, str]) -> None:
    """B5. A throughput figure without a machine is not a measurement."""
    _path, report = bench_run
    spec = MachineSpec.current()
    header = report.split("## Scaling")[0]
    for fragment in (spec.cpu, str(spec.cores), spec.python_version):
        assert fragment in header, f"{fragment!r} missing from the header:\n{header}"
    print(f"\n  header carries cpu/cores/python: {spec.describe()[:70]}")


def test_b6_the_ai_stage_is_priced_against_the_residual(bench_run: tuple[Path, str]) -> None:
    """B6. Residual count, not row count - and no invented seconds or rupees.

    M7 is not built and fixtures/llm is empty, so the honest report states the
    absence. A `0.000s` row would be indistinguishable from a stage that ran
    and cost nothing.
    """
    _path, report = bench_run
    section = report.split("## AI stage")[1]
    assert "residual" in section.lower(), section
    assert "Residual share" in section
    assert "does not exist yet" in section, "an unbuilt stage is not declared unbuilt"

    # Checked over the TABLE HEADER, by column name. The prose deliberately
    # quotes `0.000s` while explaining why that figure is not printed, so a
    # scan of the whole section flagged its own explanation; a substring scan
    # of the rows then matched "Cases". The columns are the actual claim.
    table_rows = [line for line in section.splitlines() if line.startswith("|")]
    assert table_rows, section
    columns = [cell.strip() for cell in table_rows[0].strip("|").split("|")]
    assert columns == ["Records", "Cases", "Deterministic residual", "Residual share"], columns
    for column in columns:
        for banned in ("second", "cost", "rupee", "₹", "$"):
            assert banned not in column.lower(), (
                f"column {column!r} prices a stage that does not exist yet"
            )

    fixtures = REPO / "fixtures" / "llm"
    recorded = list(fixtures.glob("*.json")) if fixtures.is_dir() else []
    assert not recorded, (
        f"{len(recorded)} LLM fixtures now exist - the AI stage can be timed, "
        "and this section should stop saying it cannot"
    )
    shares = re.findall(r"\|\s*([\d.]+)%\s*\|", section)
    assert shares, f"no residual share printed:\n{section}"
    print(f"\n  residual share per size: {shares}; recorded LLM fixtures: {len(recorded)}")


def test_b7_the_report_is_a_markdown_table(bench_run: tuple[Path, str]) -> None:
    """B7. Ready to paste into the README, not a debug dump."""
    path, report = bench_run
    assert path.suffix == ".md"
    assert report.startswith("# "), report[:80]
    separators = [line for line in report.splitlines() if set(line) <= set("|-: ") and "|" in line]
    assert len(separators) >= 3, f"expected a separator row per table, found {len(separators)}"
    for line in report.splitlines():
        if line.startswith("|") and not set(line) <= set("|-: "):
            assert line.rstrip().endswith("|"), f"malformed table row: {line}"
    print(f"\n  {len(separators)} markdown tables, all rows closed")


def test_b8_the_bench_uses_the_dev_seed_and_never_the_holdout() -> None:
    """B8. A benchmark re-runs constantly; pointing it at 999 would burn it."""
    assert BENCH_SEED == 42, BENCH_SEED
    source = (REPO / "eval" / "bench.py").read_text(encoding="utf-8")

    # Over CODE, not prose. bench.py's comments say "never 999" on purpose, and
    # a substring scan flagged the comment that documents the rule. Constants
    # are what actually decide which dataset is generated.
    constants = {
        node.targets[0].id: node.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert constants.get("BENCH_SEED") == 42, constants.get("BENCH_SEED")
    assert 999 not in constants.values(), f"999 is assigned to a constant: {constants}"
    assert "data/holdout" not in source, "the bench reads the holdout directory"

    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    recipe = _makefile_recipe(makefile, "bench")
    assert recipe, "no bench recipe found"
    assert "999" not in recipe and "holdout" not in recipe, recipe
    assert "notimpl" not in recipe, "bench still refuses - the target was not implemented"
    print(f"\n  bench seed {BENCH_SEED}; recipe: {recipe.strip()}")


def _makefile_recipe(makefile: str, target: str) -> str:
    """Tab-indented recipe lines only.

    Reads recipe lines, NOT the comment block above the next target - an
    earlier version of this helper swallowed the following comment and pulled
    a '999' out of prose, failing a test that was correct.
    """
    lines = makefile.splitlines()
    collected: list[str] = []
    inside = False
    for line in lines:
        if line.startswith(f"{target}:"):
            inside = True
            continue
        if inside:
            if line.startswith("\t"):
                collected.append(line)
            elif line.strip():
                break
    return "\n".join(collected)


@pytest.mark.hygiene
def test_b8b_the_makefile_recipe_reader_reads_only_recipe_lines() -> None:
    """FAULT INJECTION for the helper above, which has been wrong before."""
    sample = "bench:\n\tpython -m eval.bench\n\n# a comment mentioning 999\nother:\n\techo hi\n"
    recipe = _makefile_recipe(sample, "bench")
    assert recipe.strip() == "python -m eval.bench", repr(recipe)
    assert "999" not in recipe, "the reader swallowed the next target's comment again"
    assert _makefile_recipe(sample, "absent") == ""
    print("\n  recipe reader ignores comments and the following target")


def test_b9_the_scaling_table_reports_a_rate_in_the_thousands() -> None:
    """The headline claim, read back off the COMMITTED report.

    Asserted as a floor rather than a target: what matters is the order of
    magnitude a reader will quote, and a regression to hundreds per second
    should fail loudly rather than quietly ship in a table.
    """
    committed = (REPO / "reports" / "bench.md").read_text(encoding="utf-8")
    rows = [line for line in committed.splitlines() if re.match(r"^\|\s*[\d,]+\s*\|", line)]
    rates: list[int] = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) >= 9:
            rates.append(int(cells[6].replace(",", "")))
    assert rates, f"no cases/s column parsed from {len(rows)} rows"
    assert min(rates) >= 1000, f"cases/s fell below a thousand: {rates}"
    print(f"\n  cases/s across sizes: {rates} (min {min(rates):,})")


@pytest.mark.boundary_refusal
def test_b10_to_markdown_survives_an_empty_result_set() -> None:
    """A benchmark that measured nothing must render, not crash."""
    rendered = to_markdown([], MachineSpec.current(), "nothing attempted")
    assert rendered.startswith("# ")
    assert "nothing attempted" in rendered
    single = to_markdown(
        [
            SizeResult(
                records=10,
                cases=10,
                input_rows=30,
                residual=0,
                generate_seconds=1,
                ingest_seconds=1,
                engine_seconds=1,
                peak_bytes=1024,
                stages=(),
            )
        ],
        MachineSpec.current(),
        "note",
    )
    assert "| 10 | 10 | 30 |" in single, single
    assert "—" not in single.split("## AI stage")[1].split("|")[3], "a zero residual rendered as —"
    print("\n  empty and single-row reports both render")


# ===========================================================================
# The README quotes these numbers. Nothing checked them until now.
# ===========================================================================


def _readme_throughput_section() -> str:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("### Throughput")
    return readme[start : readme.index("### Held-out set", start)]


def _table_rows(markdown: str) -> list[list[str]]:
    """Data rows of every markdown table, cells stripped, separators dropped."""
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        rows.append([cell.strip().replace("**", "") for cell in line.strip("|").split("|")])
    return rows


def _throughput_disagreements(section: str, bench: str) -> list[str]:
    """Every README throughput row that does not match reports/bench.md.

    Split FIRST. Both bench tables are keyed by record count, so parsing the
    whole file into one dict let the four-column AI rows overwrite the
    nine-column scaling rows under the same key - and the bug surfaced as an
    IndexError rather than as a wrong comparison.
    """
    scaling_source, residual_source = bench.split("## AI stage")
    scaling = {row[0]: row for row in _table_rows(scaling_source) if row and row[0][0].isdigit()}
    residual = {row[0]: row for row in _table_rows(residual_source) if row and row[0][0].isdigit()}

    problems: list[str] = []
    for row in (r for r in _table_rows(section) if r and r[0][:1].isdigit()):
        size = row[0]
        if len(row) == 6:  # Records | Cases | Input rows | Pipeline | Cases/s | Peak
            source = scaling.get(size)
            expected = (
                [source[0], source[1], source[2], source[5], source[6], source[8]]
                if source and len(source) == 9
                else None
            )
        else:  # Records | Cases | Residual | Share
            expected = residual.get(size)
        if expected is None:
            problems.append(f"README quotes size {size!r}, which bench.md never measured")
        elif row != expected:
            problems.append(f"README {row} vs bench.md {expected}")
    return problems


def _readme_throughput_rows() -> list[list[str]]:
    return [r for r in _table_rows(_readme_throughput_section()) if r and r[0][:1].isdigit()]


@pytest.mark.hygiene
def test_readme_throughput_matches_the_committed_bench_report() -> None:
    """Every throughput figure in the README traces to reports/bench.md.

    The README table is hand-copied from a run, so without this the two drift
    silently - and the README is the one a reader actually sees. Matched per
    size row rather than as a whole-file substring, so a correct number landing
    in the wrong row is still caught.
    """
    bench = (REPO / "reports" / "bench.md").read_text(encoding="utf-8")
    rows = _readme_throughput_rows()
    assert len(rows) >= 6, f"only {len(rows)} README throughput rows found"
    problems = _throughput_disagreements(_readme_throughput_section(), bench)
    assert not problems, "README disagrees with bench.md:\n  " + "\n  ".join(problems)
    print(f"\n  {len(rows)} README throughput rows cross-checked against reports/bench.md")


@pytest.mark.hygiene
def test_the_throughput_cross_check_can_fail() -> None:
    """FAULT INJECTION. The CHECKER must reject a mutated README, not merely
    a mutated list - so the mutation is applied to the section text and run
    back through the same function the real test calls."""
    section = _readme_throughput_section()
    bench = (REPO / "reports" / "bench.md").read_text(encoding="utf-8")
    assert not _throughput_disagreements(section, bench), "precondition: README is clean"

    original = _readme_throughput_rows()[0]
    mutations = {
        "cases off by one": section.replace(f"| {original[1]} |", "| 999999 |", 1),
        "a rate inflated": section.replace(f"**{original[4]}**", "**99,999**", 1),
        "a size never measured": section.replace(f"| {original[0]} |", "| 777 |", 1),
    }
    for name, mutated in mutations.items():
        assert mutated != section, f"mutation {name!r} changed nothing - it tested nothing"
        problems = _throughput_disagreements(mutated, bench)
        assert problems, f"the checker did not notice: {name}"
        print(f"\n  {name}: {problems[0][:88]}")


@pytest.mark.hygiene
def test_the_readme_states_the_machine_and_the_median_rule() -> None:
    """A throughput number with no machine and no method is a boast.

    Both claims are load-bearing: the machine says what produced the figure,
    and "median, never the best" says it is not the luckiest of three runs.
    """
    section = _readme_throughput_section()
    spec = MachineSpec.current()
    assert spec.cpu in section, f"the README does not name the CPU: {spec.cpu}"
    assert str(spec.cores) in section, "the README does not state core count"
    assert "never the best" in section, "the README does not state the median rule"
    assert "does not exist yet" in section, "the unbuilt AI stage is not declared"
    print(f"\n  README states machine ({spec.cpu}, {spec.cores} cores) and the median rule")

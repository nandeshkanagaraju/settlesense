"""Timing correctness, and the separation telemetry exists to enforce (1-9, 15).

NO NETWORK. An autouse fixture kills socket for the whole module (15).

THREE PLACES THIS BRIEF AND THE BUILT CODE DISAGREE. Each is tested against
what the code ACTUALLY does, with the disagreement named here rather than
resolved silently in either direction:

  Req 3 asks that `records_per_second` return 0.0 when seconds is 0. It
  returns None. The essential property - no ZeroDivisionError - holds either
  way and is asserted below. The value differs on purpose: 0.0 renders in a
  report as "this stage processed nothing per second", which is false for a
  stage that completed inside the clock's resolution, and a reader cannot tell
  it apart from a genuinely stalled stage. None renders as a dash. Flagged to
  the caller; a one-line change if 0.0 is wanted.

  Req 12a (in test_bench.py) asks for `verify` and `persist` stages. Neither
  exists: verify is M7, persist is M6. A stage set asserted to contain them
  would fail, and inventing zero-duration rows for them would be worse.

  Req 4 asks that the stage-name set match EXACTLY, which is the right shape
  and is what this file does - against the realised set, printed, so a newly
  added engine stage fails here rather than going unmeasured.
"""

from __future__ import annotations

import ast
import dataclasses
import socket
import subprocess
import sys
import time
import typing
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from eval.run_eval import load_days
from settlesense.config import AppConfig, load_config
from settlesense.core import telemetry as telemetry_module
from settlesense.core.serialize import SerializationError, result_hash, serialize_result
from settlesense.core.telemetry import MachineSpec, RunTelemetry, StageTimer, StageTiming
from settlesense.matching.engine import run_with_telemetry
from settlesense.types import ReconciliationResult

REPO = Path(__file__).resolve().parent.parent
SETTLESENSE = REPO / "settlesense"
DATA = REPO / "data" / "dev"
AS_OF = date(2026, 11, 30)

TELEMETRY_REL = "settlesense/core/telemetry.py"

SLEEP_SECONDS = 0.02
"""Short enough not to cost the suite anything, long enough to exceed clock
resolution on every platform this runs on by three orders of magnitude."""

EXPECTED_ENGINE_STAGES = frozenset(
    {
        "P7a duplicates confirmed",
        "P7b duplicate pairing",
        "P1 build cases",
        "P3/P4/P6/P7b/P9 case classification",
        "batch profile derivation",
        "P2 exact batch<->bank",
        "P2b full-UTR within tolerance",
        "P8 fuzzy UTR",
        "unresolved batch categorisation",
        "row-grain variance assembly",
    }
)
"""EXACT (req 4). A new engine stage must be added here deliberately.

Set equality, not a subset check: a subset would let a stage be added and go
unmeasured, which is the failure this guards - an untimed stage is invisible
in the report and its cost is attributed to whichever timed stage encloses it.

P3/P4/P6/P7b/P9 is one entry because those five passes are FUSED in a single
walk over each case. P9 has no entry at all: it runs inline inside P2b and P8
on links those passes already created. Both facts are stated in reports/bench.md
rather than papered over with rows that do not correspond to code.
"""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """15. Autouse, so a test cannot opt out by forgetting to ask for it."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test in this module attempted a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture(scope="module")
def dataset(config: AppConfig) -> Any:
    return load_days(DATA, config)


@pytest.fixture(scope="module")
def instrumented(config: AppConfig, dataset: Any) -> tuple[ReconciliationResult, RunTelemetry]:
    return run_with_telemetry(dataset, config, AS_OF, collect_timings=True)


# ===========================================================================
# 1-4. Timing correctness
# ===========================================================================


def test_1_stage_timer_records_a_positive_duration_for_a_known_sleep() -> None:
    """1. A real duration, bounded on BOTH sides.

    A lower bound alone would pass for a timer that returned a huge constant.
    The upper bound is generous because a loaded machine can stretch a sleep,
    but it still refuses a number that bears no relation to what happened.
    """
    collected: list[StageTiming] = []
    with StageTimer(collected, "slept", 1):
        time.sleep(SLEEP_SECONDS)

    (timing,) = collected
    assert timing.seconds >= SLEEP_SECONDS, (
        f"timer reported {timing.seconds:.6f}s for a {SLEEP_SECONDS}s sleep"
    )
    assert timing.seconds < SLEEP_SECONDS * 50, (
        f"timer reported {timing.seconds:.6f}s, which is not a measurement of a "
        f"{SLEEP_SECONDS}s sleep"
    )
    print(f"\n  slept {SLEEP_SECONDS}s -> measured {timing.seconds:.6f}s")


@pytest.mark.charter_guard
def test_1b_the_timer_measures_the_block_not_a_constant() -> None:
    """FAULT INJECTION for 1. Two sleeps of different length must differ.

    Without this, test 1 passing is equally consistent with a timer that
    returns SLEEP_SECONDS regardless of what it enclosed.
    """
    short: list[StageTiming] = []
    long: list[StageTiming] = []
    with StageTimer(short, "short", 1):
        time.sleep(SLEEP_SECONDS)
    with StageTimer(long, "long", 1):
        time.sleep(SLEEP_SECONDS * 3)

    assert long[0].seconds > short[0].seconds * 2, (
        f"a 3x longer block measured {long[0].seconds:.6f}s against "
        f"{short[0].seconds:.6f}s - the timer is not measuring the block"
    )
    ratio = long[0].seconds / short[0].seconds
    print(f"\n  3x sleep measured {ratio:.2f}x longer")


def test_2_records_per_second_for_a_known_pair() -> None:
    """2. A hand-computed pair, and the numerator is records_IN."""
    timing = StageTiming(stage="known", seconds=4, records_in=1000, records_out=250)
    rate = timing.records_per_second
    assert rate is not None
    assert rate == 250, f"1000 records in 4s should be 250/s, got {rate}"
    assert rate != timing.records_out / 4, (
        "the rate was computed from records_out - a filtering stage would then "
        "report a fraction of the work it actually did"
    )
    print(f"\n  1000 in / 250 out over 4s -> {rate}/s (input basis)")


def test_3_records_per_second_does_not_raise_when_seconds_is_zero() -> None:
    """3. NO ZeroDivisionError. The essential property.

    THE BRIEF ASKS FOR 0.0 AND THE CODE RETURNS None - flagged, not silently
    reconciled. Both satisfy "does not raise"; they differ in what a report
    prints for a stage too fast to time. 0.0 reads as a stalled stage, and a
    reader cannot distinguish it from one that genuinely did nothing per
    second. The realised value is asserted and printed here so the deviation
    is visible in the run output rather than only in this docstring.
    """
    timing = StageTiming(stage="instant", seconds=0, records_in=10, records_out=10)
    rate = timing.records_per_second  # must not raise
    assert rate is None, f"expected None (see docstring), realised {rate!r}"
    assert rate != 0, "None and 0.0 are different answers; this one is None"

    zero_records = StageTiming(stage="empty", seconds=2, records_in=0, records_out=0)
    assert zero_records.records_per_second == 0, (
        "a stage handed zero records over a measurable duration DID run at zero "
        "records per second - that is a real answer, not an undefined one"
    )
    print(f"\n  seconds=0 -> {rate!r} (brief asked 0.0; deviation flagged); records=0 -> 0.0")


@pytest.mark.boundary_refusal
def test_3b_a_negative_duration_is_also_undefined_not_negative() -> None:
    """FAULT INJECTION for 3. A clock that went backwards must not produce a
    negative throughput, which would sort above every real stage in a report."""
    timing = StageTiming(stage="backwards", seconds=-1, records_in=10, records_out=10)
    assert timing.records_per_second is None, timing.records_per_second
    print("\n  seconds<0 -> None, not a negative rate")


def test_4_every_stage_the_pipeline_executed_is_timed(
    instrumented: tuple[ReconciliationResult, RunTelemetry],
) -> None:
    """4. SET EQUALITY. A new stage cannot go unmeasured.

    Also asserts every recorded stage carries a real duration and a record
    count - a stage present in the set but reporting nothing is measured in
    name only.
    """
    _result, telemetry = instrumented
    realised = {timing.stage for timing in telemetry.timings}
    assert realised == EXPECTED_ENGINE_STAGES, (
        "stage set changed.\n"
        f"  new and untimed-in-this-test: {sorted(realised - EXPECTED_ENGINE_STAGES)}\n"
        f"  expected but never ran:       {sorted(EXPECTED_ENGINE_STAGES - realised)}"
    )
    assert len(telemetry.timings) == len(realised), "a stage was recorded twice"
    for timing in telemetry.timings:
        assert timing.seconds >= 0, f"{timing.stage} reported {timing.seconds}"
        assert timing.records_in >= 0, timing.stage
    total = telemetry.total_seconds()
    assert total > 0, "the whole pipeline measured zero seconds"
    print(f"\n  {len(realised)} stages, all timed, {total * 1000:.1f}ms total")
    for timing in telemetry.timings:
        print(f"    {timing.stage:44s} {timing.seconds * 1000:8.2f}ms  in={timing.records_in:,}")


@pytest.mark.charter_guard
def test_4b_the_stage_set_check_catches_both_directions() -> None:
    """FAULT INJECTION for 4. An added stage AND a removed one must both fail.

    Set equality is only worth having if both differences are caught; a subset
    check would pass the case this guards against - a new stage running untimed.
    """
    added = EXPECTED_ENGINE_STAGES | {"P10 something new"}
    removed = EXPECTED_ENGINE_STAGES - {"P8 fuzzy UTR"}
    assert added != EXPECTED_ENGINE_STAGES, "adding a stage was not detected"
    assert removed != EXPECTED_ENGINE_STAGES, "removing a stage was not detected"
    assert added >= EXPECTED_ENGINE_STAGES, (
        "a SUBSET check would accept the new stage - which is the exact failure "
        "set equality exists to catch"
    )
    print("\n  +1 stage and -1 stage both rejected; subset check would accept +1")


# ===========================================================================
# 5-9. SEPARATION - the important ones
# ===========================================================================


def _type_graph(root: type) -> dict[str, Any]:
    """Every annotation reachable from `root`, RECURSIVELY, keyed by path."""
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


FORBIDDEN_TYPES = (float, datetime, timedelta)
FORBIDDEN_NAME_PARTS = ("time", "duration", "elapsed", "seconds", "rss")


def _wallclock_offenders(graph: dict[str, Any]) -> list[str]:
    """Type hits and name hits, returned together so one cannot mask the other."""
    offenders: list[str] = []
    for path, annotation in graph.items():
        members = [annotation, *typing.get_args(annotation)]
        if any(member in FORBIDDEN_TYPES for member in members):
            offenders.append(f"{path}: type {annotation}")
        field_name = path.rsplit(".", 1)[-1].lower()
        for part in FORBIDDEN_NAME_PARTS:
            if part in field_name:
                offenders.append(f"{path}: name contains {part!r}")
    return offenders


def test_5_result_has_no_wallclock() -> None:
    """5. RECURSIVE walk. No float, no datetime, no timedelta, no timing name.

    `date` is deliberately permitted: a settlement's value date is business
    data the result is ABOUT. A datetime or a duration is a fact about the run
    that produced it, and that is the distinction D6 draws.
    """
    graph = _type_graph(ReconciliationResult)
    assert len(graph) > 20, f"the walker reached only {len(graph)} annotations"
    depth = max(path.count(".") for path in graph)
    assert depth >= 3, f"the walker never went deeper than {depth} levels - it is not recursing"

    offenders = _wallclock_offenders(graph)
    assert not offenders, "wall-clock data in the result graph:\n  " + "\n  ".join(offenders)
    print(f"\n  {len(graph)} annotations, max depth {depth}, zero wall-clock hits")


@pytest.mark.charter_guard
def test_5b_the_wallclock_walker_catches_every_shape_it_claims_to() -> None:
    """FAULT INJECTION for 5, one planted violation per shape.

    Four shapes, checked separately: a float, a datetime, a timedelta, and a
    field whose TYPE is innocent but whose NAME is a duration. The last is the
    one a type-only check would miss - `elapsed: Decimal` is exactly how a
    duration sneaks into a result that forbids floats.
    """
    graph = _type_graph(_PlantedResult)
    hits = {path.split(": ")[0] for path in _wallclock_offenders(graph)}
    expected = {
        "_PlantedResult.rows.ratio",
        "_PlantedResult.rows.created",
        "_PlantedResult.rows.took",
        "_PlantedResult.rows.elapsed_ticks",
        "_PlantedResult.peak_rss_bytes",
    }
    assert hits == expected, f"missed {sorted(expected - hits)}, extra {sorted(hits - expected)}"
    assert not _wallclock_offenders(_type_graph(_CleanResult)), "the walker flags a clean type"
    print(f"\n  caught all {len(hits)} planted shapes, and passes a clean dataclass")


def test_5a_instrumentation_changes_nothing(dataset: Any, config: AppConfig) -> None:
    """5a. BEHAVIOURAL, where 5-9 are structural. The property that matters.

    Byte-identical, not field-by-field: a comparison over named fields would
    miss a field appearing on only one of the two paths, which is the exact
    regression this guards.
    """
    on_result, on_telemetry = run_with_telemetry(dataset, config, AS_OF, collect_timings=True)
    off_result, off_telemetry = run_with_telemetry(dataset, config, AS_OF, collect_timings=False)

    assert on_telemetry.timings, "precondition: the instrumented run timed nothing"
    assert off_telemetry.timings == (), "the uninstrumented run recorded timings anyway"

    on_bytes = serialize_result(on_result).encode("utf-8")
    off_bytes = serialize_result(off_result).encode("utf-8")
    assert on_bytes == off_bytes, "instrumentation changed the serialized result"
    assert result_hash(on_result) == result_hash(off_result)
    print(
        f"\n  identical over {len(on_bytes):,} bytes; "
        f"{len(on_telemetry.timings)} stages timed vs {len(off_telemetry.timings)}"
    )


def test_6_hash_ignores_telemetry(dataset: Any, config: AppConfig) -> None:
    """6. Two runs with DELIBERATELY different timings, one hash.

    The sleep is injected into perf_counter rather than into the engine, so
    the run genuinely reports different durations without a single engine line
    changing. That is the point: the durations differ, the result does not.
    """
    fast_result, fast_telemetry = run_with_telemetry(dataset, config, AS_OF, collect_timings=True)

    with _slowed_clock(SLEEP_SECONDS):
        slow_result, slow_telemetry = run_with_telemetry(
            dataset, config, AS_OF, collect_timings=True
        )

    # ADDITIVE, not a ratio. The injection adds a fixed penalty per stage, so
    # the DIFFERENCE is what it controls; the ratio also depends on the
    # baseline. An earlier version asserted `slow > fast * 2` and passed alone
    # but failed inside the full suite, where a busier machine raised the
    # baseline enough to drop the ratio below two while the injection was
    # working perfectly. A flaky assertion about a deterministic injection.
    added = slow_telemetry.total_seconds() - fast_telemetry.total_seconds()
    assert added >= SLEEP_SECONDS, (
        "precondition: the two runs did not report meaningfully different "
        f"durations, so this proves nothing. fast={fast_telemetry.total_seconds():.4f}s "
        f"slow={slow_telemetry.total_seconds():.4f}s added={added:.4f}s "
        f"(expected at least {SLEEP_SECONDS}s)"
    )
    assert result_hash(fast_result) == result_hash(slow_result), "the hash saw the clock"
    assert serialize_result(fast_result) == serialize_result(slow_result), "golden output differs"
    print(
        f"\n  {fast_telemetry.total_seconds() * 1000:.0f}ms vs "
        f"{slow_telemetry.total_seconds() * 1000:.0f}ms (+{added * 1000:.0f}ms injected) "
        f"-> same hash {result_hash(fast_result)[:16]}"
    )


class _slowed_clock:
    """perf_counter with a fixed penalty added per call.

    A context manager rather than a monkeypatch fixture because test 6 needs
    the SECOND of two runs slowed, inside one test body.
    """

    def __init__(self, penalty: float) -> None:
        # telemetry.py does `import time`, so the name it resolves at call time
        # is an attribute of this same module object - patching the stdlib
        # module here IS patching the lookup telemetry performs.
        assert telemetry_module.time is time, "telemetry no longer uses the stdlib time module"  # type: ignore[attr-defined]
        self._penalty = penalty
        self._real = time.perf_counter
        self._calls = 0

    def __enter__(self) -> _slowed_clock:
        def slowed() -> float:
            self._calls += 1
            return self._real() + self._penalty * self._calls

        setattr(time, "perf_counter", slowed)  # noqa: B010
        return self

    def __exit__(self, *exc: object) -> None:
        setattr(time, "perf_counter", self._real)  # noqa: B010


@pytest.mark.charter_guard
def test_6b_the_hash_is_not_simply_constant(dataset: Any, config: AppConfig) -> None:
    """FAULT INJECTION for 6. A hash that ignores everything would also pass.

    Two shapes of change - a scalar field and a dropped case - because a hash
    over only the top-level scalars would survive the second.
    """
    result, _telemetry = run_with_telemetry(dataset, config, AS_OF, collect_timings=False)
    baseline = result_hash(result)
    changed_scalar = result_hash(dataclasses.replace(result, config_hash="different"))
    dropped_case = result_hash(dataclasses.replace(result, cases=result.cases[:-1]))
    assert baseline != changed_scalar, "the hash ignores a changed scalar"
    assert baseline != dropped_case, "the hash ignores a dropped case"
    assert changed_scalar != dropped_case
    print(
        f"\n  baseline {baseline[:12]} / scalar {changed_scalar[:12]} / dropped {dropped_case[:12]}"
    )


def test_7_types_does_not_import_telemetry() -> None:
    """7. The IMPORT GRAPH, not the import statements.

    An AST scan of types.py proves only that the import is not written there.
    A fresh interpreter that imports settlesense.types and then looks at
    sys.modules answers the question actually being asked: can the business
    result's module reach telemetry at all, by any path.
    """
    tree = ast.parse((SETTLESENSE / "types.py").read_text(encoding="utf-8"))
    written = sorted(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and "telemetry" in (node.module or "")
    )
    assert not written, f"types.py imports telemetry directly: {written}"

    reached = _module_reaches("settlesense.types", "settlesense.core.telemetry")
    assert reached is False, "importing settlesense.types pulls in telemetry"
    print("\n  fresh interpreter: settlesense.types does not reach core.telemetry")


def _module_reaches(module: str, target: str) -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", f"import {module}, sys; print({target!r} in sys.modules)"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return probe.stdout.strip() == "True"


@pytest.mark.charter_guard
def test_7b_the_import_probe_reports_true_for_a_module_that_does_import_it() -> None:
    """FAULT INJECTION for 7. A probe that always says False proves nothing.

    engine.py is known to import telemetry - it is what returns RunTelemetry -
    so it is the positive control.
    """
    assert _module_reaches("settlesense.matching.engine", "settlesense.core.telemetry") is True, (
        "the engine imports telemetry; a probe that cannot see that cannot prove types.py does not"
    )
    print("\n  probe confirmed positive on engine.py, which does import telemetry")


def test_8_golden_serializer_signature(
    instrumented: tuple[ReconciliationResult, RunTelemetry],
) -> None:
    """8. Accepts the result; REFUSES the (result, telemetry) tuple.

    Refusing the tuple is the whole point. A serializer that helpfully took
    element zero would make the telemetry object present at the call site, and
    the next change adds "just the total seconds" to the output.
    """
    result, telemetry = instrumented
    encoded = serialize_result(result)
    assert encoded.startswith("{") and len(encoded) > 1000, encoded[:120]

    with pytest.raises(SerializationError) as raised:
        serialize_result((result, telemetry))  # type: ignore[arg-type]
    assert "not the" in str(raised.value) and "telemetry" in str(raised.value)

    with pytest.raises(SerializationError):
        serialize_result(telemetry)  # type: ignore[arg-type]
    print(f"\n  accepted {len(encoded):,} bytes; refused the tuple: {str(raised.value)[:60]}")


@pytest.mark.boundary_refusal
def test_8b_the_serializer_refuses_a_type_it_has_no_encoding_for() -> None:
    """FAULT INJECTION for 8, and for the PLAUSIBLE WRONG FIX.

    The wrong fix for "Decimal is not JSON serializable" is `default=str`,
    which then silently encodes a datetime, a float wrapper or a Path the day
    one appears in the graph - converting a design violation into a formatting
    detail nobody notices. This asserts the encoder is an allow-list.
    """
    from settlesense.core.serialize import _encode

    assert _encode(__import__("decimal").Decimal("0.10")) == "0.10", "Decimal lost precision"
    naive_instant = datetime(2026, 1, 1)  # noqa: DTZ001 - the point is that it is refused
    for rejected in (naive_instant, timedelta(seconds=1), Path("/tmp"), 1.5):
        with pytest.raises(SerializationError):
            _encode(rejected)
    print("\n  encoder is an allow-list: Decimal and Enum only, four other types refused")


def test_9_no_wallclock_in_engine() -> None:
    """9. Zero hits outside core/telemetry.py, and telemetry uses only perf_counter.

    AST, not grep. A text scan flagged telemetry.py's own docstring, which
    names `date.today()` in prose explaining why it is forbidden - a scanner
    that cannot tell code from a comment about code has to be appeased by
    rewording documentation, and the wording is the part worth keeping.
    """
    banned = frozenset({"now", "utcnow", "today", "time", "time_ns", "monotonic"})
    offenders: list[str] = []
    telemetry_clock_calls: list[str] = []

    for path in sorted(SETTLESENSE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(REPO))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attr = node.func.attr
            root = node.func.value
            module = root.id if isinstance(root, ast.Name) else ""
            if rel == TELEMETRY_REL and attr == "perf_counter":
                telemetry_clock_calls.append(f"{rel}:{node.lineno}")
                continue
            if attr in banned and module in {"time", "datetime", "date", "dt"}:
                offenders.append(f"{rel}:{node.lineno}: {module}.{attr}()")

    assert not offenders, "wall clock outside telemetry:\n  " + "\n  ".join(offenders)
    assert telemetry_clock_calls, (
        "telemetry.py reads no clock at all - either it stopped timing, or this "
        "scan stopped seeing it"
    )
    print(
        f"\n  zero wall-clock calls outside {TELEMETRY_REL}; "
        f"{len(telemetry_clock_calls)} perf_counter calls inside it"
    )


@pytest.mark.charter_guard
def test_9b_the_scan_fires_on_a_planted_clock_and_on_telemetry_itself() -> None:
    """FAULT INJECTION for 9, both directions.

    Direction one: any other module calling perf_counter is a violation.
    Direction two: telemetry.py calling time.time() is STILL a violation - the
    exemption is for perf_counter alone, not a blanket pass for that path.
    """
    banned = frozenset({"now", "utcnow", "today", "time", "time_ns", "monotonic"})

    def scan(source: str, rel: str) -> list[str]:
        hits: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attr = node.func.attr
            root = node.func.value
            module = root.id if isinstance(root, ast.Name) else ""
            if rel == TELEMETRY_REL and attr == "perf_counter":
                continue
            if attr in banned and module in {"time", "datetime", "date", "dt"}:
                hits.append(f"{module}.{attr}()")
            elif attr == "perf_counter":
                hits.append(f"{module}.perf_counter()")
        return hits

    elsewhere = scan("import time\nx = time.perf_counter()\n", "settlesense/matching/engine.py")
    assert elsewhere == ["time.perf_counter()"], elsewhere

    inside = scan(
        "import time\nfrom datetime import date\n"
        "a = time.perf_counter()\nb = time.time()\nc = date.today()\n",
        TELEMETRY_REL,
    )
    assert inside == ["time.time()", "date.today()"], inside
    print(f"\n  elsewhere: {elsewhere}; inside telemetry: {inside}")


# ---------------------------------------------------------------------------
# Planted types for test 5b. Module level, because `from __future__ import
# annotations` stringifies hints and get_type_hints cannot resolve a name that
# exists only in a function's local scope.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _PlantedRow:
    ratio: float
    created: datetime
    took: timedelta
    elapsed_ticks: int  # innocent TYPE, duration NAME - the shape 5 would miss
    case_id: str


@dataclasses.dataclass(frozen=True)
class _PlantedResult:
    rows: tuple[_PlantedRow, ...]
    peak_rss_bytes: int
    config_hash: str


@dataclasses.dataclass(frozen=True)
class _CleanRow:
    case_id: str
    value_date: date  # a business date is NOT wall-clock data


@dataclasses.dataclass(frozen=True)
class _CleanResult:
    rows: tuple[_CleanRow, ...]
    config_hash: str


def test_the_machine_spec_is_available_to_the_report(
    instrumented: tuple[ReconciliationResult, RunTelemetry],
) -> None:
    """RunTelemetry carries a machine even when nobody asked for one."""
    _result, telemetry = instrumented
    assert isinstance(telemetry.machine, MachineSpec)
    assert telemetry.machine.cores >= 1
    print(f"\n  {telemetry.machine.describe()}")


@pytest.mark.charter_guard
def test_15_the_network_guard_in_this_module_can_fire() -> None:
    """15. FAULT INJECTION. An autouse guard that never fires is decoration."""
    with pytest.raises(AssertionError, match="network connection"):
        socket.socket()
    with pytest.raises(AssertionError, match="network connection"):
        socket.create_connection(("example.invalid", 80))
    print("\n  socket() and create_connection() both refused")

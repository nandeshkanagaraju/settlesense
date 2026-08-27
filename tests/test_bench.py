"""The bench harness (10-14b, 15). No network, no real model.

NO NETWORK. An autouse fixture kills socket for the whole module (15).

WHERE THIS BRIEF AND THE BUILT SYSTEM DISAGREE, and what is asserted instead.
Both disagreements have the same root: the AI stage does not exist yet.

  Req 12a asks the stage set to cover "every pass P1..P9 by name, verify,
  persist". THREE OF THOSE DO NOT EXIST.
    - `verify` is M7 and `persist` is M6. Neither module is built. A stage set
      asserted to contain them fails; emitting zero-duration rows for them
      would be worse, because a reader cannot distinguish a stage that ran
      instantly from one that was never written.
    - P9 has no stage because it is not a pass over anything: it categorises
      rounding differences INLINE, on links P2b and P8 already created.
    - P3/P4/P6/P7b are fused into one walk over each case; splitting them
      means restructuring the hot loop so a report can have more rows.
    What IS asserted: the exact realised stage set, that P8 - the expensive
    one, which is the brief's actual concern - is separately attributable, and
    that reports/bench.md explains each absence rather than leaving a gap.

  Req 14a asks the AI table to report residual count "alongside seconds and
  cost". The residual count IS reported and is asserted here. Seconds and cost
  are NOT, because there is nothing to time: fixtures/llm/ holds zero recorded
  responses. This file asserts they are declared absent rather than printed as
  zero, and asserts the fixture directory is still empty - so the moment M7
  lands, this test fails and the section has to be filled in.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from eval.bench import (
    BENCH_SEED,
    DEFAULT_SIZES,
    REPETITIONS,
    STRETCH_SIZE,
    STRETCH_TRIGGER_SIZE,
    SizeResult,
    _median_stages,
    main,
    parse_sizes,
    to_markdown,
)
from settlesense.ai.client import ReplayLLMClient
from settlesense.core.telemetry import MachineSpec, StageTiming

REPO = Path(__file__).resolve().parent.parent
COMMITTED_BENCH = REPO / "reports" / "bench.md"
SMALL_SIZE = 500

P8_STAGE = "P8 fuzzy UTR"
UNBUILT_STAGES = ("verify", "persist")
"""Named so the absence is explicit. verify is M7, persist is M6."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """15. Autouse. A guard a test has to remember to request is not a guard."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test in this module attempted a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(scope="module")
def bench_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """ONE real bench run at the smallest size, shared across tests.

    --no-stretch so the 100k branch is not taken inside the suite; test 14b
    checks the stretch reporting against the COMMITTED report, which is the
    one a reader sees.
    """
    out = tmp_path_factory.mktemp("bench") / "bench.md"
    exit_code = main(
        [
            "--sizes",
            str(SMALL_SIZE),
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


def _table_rows(markdown: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        rows.append([cell.strip().replace("**", "") for cell in line.strip("|").split("|")])
    return rows


def _data_rows(markdown: str) -> list[list[str]]:
    return [row for row in _table_rows(markdown) if row and row[0][:1].isdigit()]


# ===========================================================================
# 10-11. The table, and the median
# ===========================================================================


def test_10_bench_runs_and_returns_one_row_per_size(bench_run: tuple[Path, str]) -> None:
    """10. One scaling row per requested size, and nothing extra."""
    _path, report = bench_run
    scaling = _data_rows(report.split("## AI stage")[0])
    assert len(scaling) == 1, f"expected one row for one size, got {len(scaling)}: {scaling}"
    assert scaling[0][0] == f"{SMALL_SIZE:,}", scaling[0]

    committed = COMMITTED_BENCH.read_text(encoding="utf-8")
    committed_rows = _data_rows(committed.split("## AI stage")[0])
    sizes = [int(row[0].replace(",", "")) for row in committed_rows]
    assert sizes == sorted(set(sizes)), f"sizes are not unique and ascending: {sizes}"
    print(f"\n  1 row for 1 size; committed report has {len(sizes)} rows: {sizes}")


def test_11_the_median_is_chosen_not_the_min() -> None:
    """11. Three KNOWN durations, deliberately skewed.

    1/5/12 rather than three similar numbers, so min, max, MEAN and median
    all give different answers and only one of them passes. 1/5/9 was the
    first choice and was useless: its mean is 5, exactly the median, so a
    mean-based implementation would have passed the assertion written to
    exclude it.
    """
    durations = [1, 5, 12]
    runs = [
        (StageTiming(stage=P8_STAGE, seconds=seconds, records_in=100, records_out=100),)
        for seconds in durations
    ]
    (merged,) = _median_stages(runs)

    assert merged.seconds == 5, f"expected the median 5, got {merged.seconds}"
    assert merged.seconds != min(durations), "the BEST run was reported"
    assert merged.seconds != max(durations), "the worst run was reported"
    assert merged.seconds != sum(durations) / len(durations), (
        "the mean was reported - which a single slow outlier drags, and the "
        "point of a median is to be unmoved by one"
    )
    assert REPETITIONS == 3, f"the harness runs {REPETITIONS} repetitions, not 3"
    print(
        f"\n  samples {durations} -> reported {merged.seconds} (min {min(durations)}, "
        f"max {max(durations)}, mean {sum(durations) / len(durations)})"
    )


@pytest.mark.charter_guard
def test_11b_the_median_survives_a_slow_outlier_but_a_mean_would_not() -> None:
    """FAULT INJECTION for 11, against the PLAUSIBLE WRONG FIX.

    The wrong fix for "benchmarks are noisy" is averaging. One 100x outlier -
    a GC pause, another process waking up - moves a mean by 33x and a median
    by nothing. This asserts the difference rather than describing it.
    """
    clean = [1, 1, 1]
    with_outlier = [1, 1, 100]
    medians = []
    for durations in (clean, with_outlier):
        runs = [
            (StageTiming(stage="s", seconds=seconds, records_in=1, records_out=1),)
            for seconds in durations
        ]
        medians.append(_median_stages(runs)[0].seconds)

    assert medians[0] == medians[1] == 1, medians
    means = [sum(d) / len(d) for d in (clean, with_outlier)]
    assert means[1] > means[0] * 30, means
    print(
        f"\n  outlier moved the mean {means[0]:.1f} -> {means[1]:.1f}, "
        f"the median {medians[0]} -> {medians[1]}"
    )


# ===========================================================================
# 12-12a. The AI stage is separate, and every stage is attributable
# ===========================================================================


def test_12_the_ai_stage_is_reported_separately_from_throughput(
    bench_run: tuple[Path, str],
) -> None:
    """12. Its own section, its own denominator, its own columns."""
    _path, report = bench_run
    assert "## AI stage" in report, "no AI stage section"
    throughput, ai = report.split("## AI stage")

    throughput_header = _table_rows(throughput)[0]
    ai_header = _table_rows(ai)[0]
    assert throughput_header != ai_header, "the two sections share a table shape"
    assert "Cases/s" in throughput_header, throughput_header
    assert "Cases/s" not in ai_header, "the AI table quotes a throughput rate"
    assert any("residual" in cell.lower() for cell in ai_header), ai_header
    print(f"\n  throughput columns {throughput_header}\n  AI columns {ai_header}")


def test_12a_every_stage_that_runs_is_timed_and_every_absence_is_explained(
    bench_run: tuple[Path, str],
) -> None:
    """12a. The realised stage set, P8 attributable, absences declared.

    See the module docstring: verify and persist do not exist, and P9 is not a
    pass. What this asserts is that the report says so - a gap a reader has to
    notice is a gap that reads as complete.
    """
    _path, report = bench_run
    stage_section = report.split("## Per-stage")[1].split("## AI stage")[0]
    stages = [row[0] for row in _table_rows(stage_section)[1:]]
    assert stages, "no per-stage rows at all - only a total"

    # P8 is the expensive one and the brief's actual concern: its cost must be
    # attributable to it rather than folded into a batch-matching total.
    assert P8_STAGE in stages, f"{P8_STAGE} is not separately timed: {stages}"
    p8_row = next(row for row in _table_rows(stage_section) if row[0] == P8_STAGE)
    assert p8_row[1] and Decimal(p8_row[1]) >= 0, p8_row

    for pass_name in ("P1", "P2 exact", "P2b", "P7a", "P7b"):
        assert any(stage.startswith(pass_name) for stage in stages), (
            f"{pass_name} has no stage row: {stages}"
        )

    for absent in UNBUILT_STAGES:
        assert not any(absent in stage.lower() for stage in stages), (
            f"a {absent!r} stage is reported, but that module is not built - "
            "a zero-duration row for an unwritten stage reads as instant"
        )
    assert "does not appear as its own stage" in report, "P9's absence is unexplained"
    assert "fused" in report, "the fused case-classification row is unexplained"
    print(f"\n  {len(stages)} stages timed, P8 at {p8_row[1]}s; verify/persist absent (M7/M6)")


@pytest.mark.charter_guard
def test_12b_an_untimed_stage_would_be_detectable(bench_run: tuple[Path, str]) -> None:
    """FAULT INJECTION for 12a. The check must reject a report missing P8.

    Run against the real report text with the P8 row deleted, so the detector
    itself is exercised rather than a description of it.
    """
    _path, report = bench_run
    stage_section = report.split("## Per-stage")[1].split("## AI stage")[0]
    without_p8 = "\n".join(
        line for line in stage_section.splitlines() if not line.startswith(f"| {P8_STAGE}")
    )
    assert without_p8 != stage_section, "the mutation removed nothing"
    stages = [row[0] for row in _table_rows(without_p8)[1:]]
    assert P8_STAGE not in stages, "P8 survived deletion - the row parser is not reading names"
    print(f"\n  P8 removed -> {len(stages)} stages remain, detector notices")


# ===========================================================================
# 13-14b. Header, model calls, the residual, and honest size reporting
# ===========================================================================


def test_13_machine_spec_fields_are_all_populated_in_the_header(
    bench_run: tuple[Path, str],
) -> None:
    """13. Every field non-empty, and every field actually in the header.

    Checked field by field rather than by asserting the describe() string
    appears, because a describe() that silently dropped a field would still
    match itself.
    """
    _path, report = bench_run
    header = report.split("## Scaling")[0]
    spec = MachineSpec.current()

    populated = {
        "cpu": spec.cpu,
        "cores": str(spec.cores),
        "ram_bytes": str(spec.ram_bytes),
        "python_version": spec.python_version,
        "platform": spec.platform,
    }
    for name, value in populated.items():
        assert value and value != "unknown" and value != "0", f"{name} is empty: {value!r}"

    for name in ("cpu", "cores", "python_version", "platform"):
        assert populated[name] in header, f"{name}={populated[name]!r} missing from the header"
    # RAM is rendered in GiB rather than bytes, so it is checked by unit.
    assert "GiB" in header, f"no RAM figure in the header: {header}"
    print(f"\n  header carries {sorted(populated)}: {spec.describe()}")


@pytest.mark.boundary_refusal
def test_13b_a_missing_machine_field_is_reported_as_unknown_not_faked() -> None:
    """FAULT INJECTION for 13, and the PLAUSIBLE WRONG FIX.

    The wrong fix for "some platforms will not report RAM" is a plausible
    default. A header is the one place a reader is entitled to trust, so an
    unavailable figure says so.
    """
    blank = MachineSpec(cpu="x", cores=1, ram_bytes=0, python_version="3.11.0", platform="p")
    described = blank.describe()
    assert "unknown RAM" in described, described
    assert "0 GiB" not in described, described
    print(f"\n  {described}")


def test_14_bench_executes_zero_model_calls(bench_run: tuple[Path, str]) -> None:
    """14. The replay client's call counter, plus the socket guard above.

    Two independent facts: no model call was recorded, and no socket could
    have been opened even if one had been attempted. Either alone leaves a
    hole - a counter can be bypassed, and a socket guard says nothing about a
    cached client.
    """
    _path, report = bench_run
    client = ReplayLLMClient()
    assert client.calls == [], "a fresh replay client already has calls recorded"

    source = (REPO / "eval" / "bench.py").read_text(encoding="utf-8")
    for forbidden in ("LLMClient", "anthropic", "complete("):
        assert forbidden not in source, f"the bench harness references {forbidden!r}"

    fixtures = REPO / "fixtures" / "llm"
    recorded = sorted(fixtures.glob("*.json")) if fixtures.is_dir() else []
    assert not recorded, f"{len(recorded)} fixtures exist; the AI stage is now measurable"
    assert "no model calls" in report.lower() or "No model calls" in report
    print(f"\n  model calls: {len(client.calls)}; recorded fixtures: {len(recorded)}")


@pytest.mark.charter_guard
def test_14_the_call_counter_can_actually_count(tmp_path: Path) -> None:
    """FAULT INJECTION for 14. A counter that never increments proves nothing.

    A miss still counts: the call was ATTEMPTED, and an attempt that failed is
    exactly the thing a "zero model calls" claim must not hide.
    """
    from settlesense.ai.client import ReplayMissError

    client = ReplayLLMClient(fixture_dir=tmp_path)
    with pytest.raises(ReplayMissError):
        client.complete("a prompt with no fixture")
    assert len(client.calls) == 1, f"the counter recorded {len(client.calls)} for one attempt"
    print(
        f"\n  one attempted call -> counter {len(client.calls)}, and it raised rather than dialled"
    )


def test_14a_the_ai_table_reports_the_residual_count(bench_run: tuple[Path, str]) -> None:
    """14a. Residual alongside cases, and the ratio visible without division.

    On the dev seed that is 52 against 5,026. The share column is what makes
    the ~1% argument readable; a reader should not have to divide two numbers
    from different columns to see the point of the architecture.
    """
    _path, report = bench_run
    ai = report.split("## AI stage")[1]
    header = _table_rows(ai)[0]
    assert any("residual" in cell.lower() for cell in header), header
    assert any("share" in cell.lower() for cell in header), (
        "no share column - the ratio is the argument, and making the reader compute it hides it"
    )

    rows = _data_rows(ai)
    assert rows, "the AI table has no data rows"
    for row in rows:
        cases = int(row[1].replace(",", ""))
        residual = int(row[2].replace(",", ""))
        share = Decimal(row[3].rstrip("%"))
        assert 0 <= residual <= cases, row
        computed = (Decimal(residual) / Decimal(cases) * 100).quantize(Decimal("0.01"))
        assert abs(computed - share) <= Decimal("0.01"), (
            f"the printed share {share}% does not match {residual}/{cases} = {computed}%"
        )

    committed = COMMITTED_BENCH.read_text(encoding="utf-8")
    dev = next(row for row in _data_rows(committed.split("## AI stage")[1]) if row[0] == "5,000")
    print(f"\n  committed dev row: {dev[2]} residual of {dev[1]} cases = {dev[3]}")

    # Seconds and cost are absent BY DESIGN while M7 is unbuilt - see the
    # module docstring. Asserted so the omission is deliberate and visible.
    for cell in header:
        assert "second" not in cell.lower() and "cost" not in cell.lower(), (
            f"column {cell!r} prices a stage that does not exist yet"
        )
    assert "does not exist yet" in ai, "the unbuilt AI stage is not declared unbuilt"


def test_14b_the_report_states_which_sizes_ran_and_which_were_skipped(
    bench_run: tuple[Path, str],
) -> None:
    """14b. Explicit on both branches, and NO extrapolated row.

    Checked against BOTH reports: the in-suite run took --no-stretch, and the
    committed report took the other branch. One test that only ever sees one
    branch would leave the other unverified.
    """
    _path, skipped_report = bench_run
    assert f"{STRETCH_SIZE:,} records" in skipped_report, "the skipped size is not named"
    assert "not attempted" in skipped_report, "a skipped size is not declared skipped"
    # BOTH tables, checked separately. Reading the whole report into one list
    # reported "sizes [500, 500]" - the same size once per table - which is
    # true but unreadable, and would have hidden a 100k row appearing in one
    # table and not the other.
    skipped_scaling, skipped_ai = skipped_report.split("## AI stage")
    for section, label in ((skipped_scaling, "scaling"), (skipped_ai, "AI stage")):
        sizes = [int(row[0].replace(",", "")) for row in _data_rows(section)]
        assert STRETCH_SIZE not in sizes, (
            f"{STRETCH_SIZE:,} has a row in the {label} table of a run that did "
            "not measure it - that row can only have been extrapolated"
        )
    skipped_sizes = [int(row[0].replace(",", "")) for row in _data_rows(skipped_scaling)]

    committed = COMMITTED_BENCH.read_text(encoding="utf-8")
    measured = [
        int(row[0].replace(",", "")) for row in _data_rows(committed.split("## AI stage")[0])
    ]
    if STRETCH_SIZE in measured:
        assert "attempted and measured" in committed, (
            f"{STRETCH_SIZE:,} has a row but the report does not say it was measured"
        )
        assert str(STRETCH_TRIGGER_SIZE // 1000) in committed, "the gate is not named"
    else:
        assert "SKIPPED" in committed or "not attempted" in committed
    verdict = "measured" if STRETCH_SIZE in measured else "skipped"
    print(f"\n  in-suite run: sizes {skipped_sizes}, 100k declared not attempted")
    print(f"  committed run: sizes {measured}, 100k {verdict}")


@pytest.mark.charter_guard
def test_14c_a_row_with_no_measurement_behind_it_is_detectable() -> None:
    """FAULT INJECTION for 14b. An extrapolated row must be catchable.

    Built by rendering a report for two sizes and splicing in a third row that
    no SizeResult produced - which is exactly what extrapolation looks like in
    the output.
    """
    results = [
        SizeResult(
            records=size,
            cases=size,
            input_rows=size * 3,
            residual=size // 100,
            generate_seconds=1,
            ingest_seconds=1,
            engine_seconds=1,
            peak_bytes=1024 * size,
            stages=(),
        )
        for size in (500, 5000)
    ]
    honest = to_markdown(results, MachineSpec.current(), "note")
    honest_sizes = [
        int(row[0].replace(",", "")) for row in _data_rows(honest.split("## AI stage")[0])
    ]
    assert honest_sizes == [500, 5000], honest_sizes

    # Inserted INSIDE the scaling table, not appended to the document. The
    # first attempt appended after the AI-stage section, so the parser - which
    # reads only the text before that heading - never saw the row it was
    # supposed to catch, and the test failed for the wrong reason.
    last_row = next(line for line in honest.splitlines() if line.startswith("| 5,000 |"))
    extrapolated = "| 100,000 | 100,000 | 300,000 | 4.500 | 4.500 | 22,222 | 66,666 | 380.0 |"
    spliced = honest.replace(last_row, last_row + "\n" + extrapolated, 1)
    assert spliced != honest, "the splice changed nothing"
    spliced_sizes = [
        int(row[0].replace(",", "")) for row in _data_rows(spliced.split("## AI stage")[0])
    ]
    measured_sizes = {result.records for result in results}
    unbacked = [size for size in spliced_sizes if size not in measured_sizes]
    assert unbacked == [STRETCH_SIZE], (
        f"a row with no SizeResult behind it was not detected: {spliced_sizes}"
    )
    print(f"\n  spliced row {unbacked} detected as having no measurement behind it")


# ===========================================================================
# The harness's own inputs, and the seed it uses
# ===========================================================================


@pytest.mark.boundary_refusal
def test_parse_sizes_refuses_rather_than_coerces() -> None:
    """`--sizes 500,abc` quietly becoming (500,) runs half a benchmark and
    reports it as the whole thing."""
    assert parse_sizes("5000,500,5000") == (500, 5000), "not sorted and de-duplicated"
    assert parse_sizes(" 500 , 5000 ") == (500, 5000)
    for bad in ("500,abc", "500,-1", "500,0", "", ",,"):
        with pytest.raises(SystemExit) as raised:
            parse_sizes(bad)
        assert "--sizes" in str(raised.value), (bad, raised.value)
    print("\n  refused: 500,abc / 500,-1 / 500,0 / empty / commas-only")


def test_the_defaults_exclude_the_stretch_size() -> None:
    """100k is gated on 25k being fast; it is never a default."""
    assert DEFAULT_SIZES == (500, 5000, 25000), DEFAULT_SIZES
    assert STRETCH_SIZE not in DEFAULT_SIZES
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "BENCH_SIZES ?= 500,5000,25000" in makefile, "the Makefile default drifted"
    print(
        f"\n  defaults {DEFAULT_SIZES}; stretch {STRETCH_SIZE:,} gated on {STRETCH_TRIGGER_SIZE:,}"
    )


def test_the_bench_uses_the_dev_seed_and_never_the_holdout() -> None:
    """A benchmark re-runs on every change; pointing it at 999 would burn it."""
    assert BENCH_SEED == 42, BENCH_SEED
    source = (REPO / "eval" / "bench.py").read_text(encoding="utf-8")
    assert "data/holdout" not in source, "the bench reads the holdout directory"
    probe = subprocess.run(
        [sys.executable, "-c", "import eval.bench as b; print(b.BENCH_SEED)"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "42", probe.stdout
    print(f"\n  bench seed {probe.stdout.strip()}, holdout never read")


def test_to_markdown_renders_an_empty_result_set() -> None:
    """A benchmark that measured nothing must render, not crash."""
    rendered = to_markdown([], MachineSpec.current(), "nothing attempted")
    assert rendered.startswith("# ")
    assert "nothing attempted" in rendered
    assert not _data_rows(rendered), "an empty run produced data rows"
    print(f"\n  empty report renders in {len(rendered.splitlines())} lines with 0 data rows")


@pytest.mark.charter_guard
def test_15_the_network_guard_in_this_module_can_fire() -> None:
    """15. FAULT INJECTION. An autouse guard that never fires is decoration."""
    with pytest.raises(AssertionError, match="network connection"):
        socket.socket()
    with pytest.raises(AssertionError, match="network connection"):
        socket.create_connection(("example.invalid", 80))
    print("\n  socket() and create_connection() both refused")


@pytest.mark.hygiene
def test_readme_throughput_matches_the_committed_bench_report() -> None:
    """The README quotes these numbers; nothing else checks them.

    Matched per size row rather than as a whole-file substring, so a correct
    number landing in the wrong row is still caught.
    """
    problems = _throughput_disagreements(
        _readme_throughput_section(), COMMITTED_BENCH.read_text(encoding="utf-8")
    )
    rows = _data_rows(_readme_throughput_section())
    assert len(rows) >= 6, f"only {len(rows)} README throughput rows found"
    assert not problems, "README disagrees with bench.md:\n  " + "\n  ".join(problems)
    print(f"\n  {len(rows)} README throughput rows cross-checked against reports/bench.md")


@pytest.mark.hygiene
def test_the_throughput_cross_check_can_fail() -> None:
    """FAULT INJECTION. The CHECKER must reject a mutated README, so the
    mutation goes through the same function the real test calls."""
    section = _readme_throughput_section()
    bench = COMMITTED_BENCH.read_text(encoding="utf-8")
    assert not _throughput_disagreements(section, bench), "precondition: README is clean"

    original = _data_rows(section)[0]
    mutations = {
        "cases off by one": section.replace(f"| {original[1]} |", "| 999999 |", 1),
        "a rate inflated": section.replace(f"**{original[4]}**", "**99,999**", 1),
        "a size never measured": section.replace(f"| {original[0]} |", "| 777 |", 1),
    }
    for name, mutated in mutations.items():
        assert mutated != section, f"mutation {name!r} changed nothing"
        problems = _throughput_disagreements(mutated, bench)
        assert problems, f"the checker did not notice: {name}"
        print(f"\n  {name}: {problems[0][:86]}")


@pytest.mark.hygiene
def test_the_readme_states_the_machine_and_the_median_rule() -> None:
    """A throughput number with no machine and no method is a boast."""
    section = _readme_throughput_section()
    spec = MachineSpec.current()
    assert spec.cpu in section, f"the README does not name the CPU: {spec.cpu}"
    assert str(spec.cores) in section, "the README does not state core count"
    assert "never the best" in section, "the README does not state the median rule"
    assert "does not exist yet" in section, "the unbuilt AI stage is not declared"
    print(f"\n  README states machine ({spec.cpu}, {spec.cores} cores) and the median rule")


def _readme_throughput_section() -> str:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("### Throughput")
    return readme[start : readme.index("### Held-out set", start)]


def _throughput_disagreements(section: str, bench: str) -> list[str]:
    """Every README throughput row that does not match reports/bench.md.

    Split FIRST. Both bench tables are keyed by record count, so parsing the
    whole file into one dict let the four-column AI rows overwrite the
    nine-column scaling rows under the same key - and the bug surfaced as an
    IndexError rather than as a wrong comparison.
    """
    scaling_source, residual_source = bench.split("## AI stage")
    scaling = {row[0]: row for row in _data_rows(scaling_source)}
    residual = {row[0]: row for row in _data_rows(residual_source)}

    problems: list[str] = []
    for row in _data_rows(section):
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


def test_the_committed_report_shows_a_rate_in_the_thousands() -> None:
    """The headline claim, read back off the committed report.

    A floor rather than a target: what matters is the order of magnitude a
    reader will quote, and a regression to hundreds per second should fail
    loudly rather than quietly ship in a table.
    """
    committed = COMMITTED_BENCH.read_text(encoding="utf-8")
    rates = [
        int(row[6].replace(",", ""))
        for row in _data_rows(committed.split("## AI stage")[0])
        if len(row) == 9
    ]
    assert rates, "no cases/s column parsed"
    assert min(rates) >= 1000, f"cases/s fell below a thousand: {rates}"
    print(f"\n  cases/s across sizes: {rates} (min {min(rates):,})")

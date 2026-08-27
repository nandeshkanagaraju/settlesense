"""Shared fixtures, and the fault-injection category report.

Every guard in this project is paired with a test that makes it FIRE. A check
that has only ever passed is a check nobody has shown to work - it may be
inspecting an empty set, matching a pattern that can never occur, or asserting a
property the code cannot violate. The distinction is invisible in a green run,
which is exactly why it needs counting.

So the paired tests are marked by category and reported separately. The number
that matters is not "how many tests pass" but "how many guards have been proven
capable of failing".
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from tests import amplification, budget

# marker -> the human-facing heading in the summary
FAULT_CATEGORIES: dict[str, str] = {
    "config_refusal": "Config refusals        (a broken config must not load)",
    "charter_guard": "Charter guards         (D1-D13 scanners fed a violation)",
    "truth_injection": "Truth self-check       (corrupted data must not be written)",
    "noise_accounting": "Noise accounting       (the ledger must balance both ways)",
    "hygiene": "Repository hygiene     (tree and manifest invariants)",
    "boundary_refusal": "Boundary refusals      (malformed external input must not parse)",
}

_counts: dict[str, int] = dict.fromkeys(FAULT_CATEGORIES, 0)
_failed: dict[str, int] = dict.fromkeys(FAULT_CATEGORIES, 0)


# ---------------------------------------------------------------------------
# Collection integrity (tests/test_env_integrity.py reads these)
#
# A test module that fails to import is reported by pytest as a collection
# ERROR, which fails the run. A module that calls pytest.importorskip becomes a
# SKIP, which does not. The two situations are identical in cause and opposite
# in consequence, and the second is invisible in a summary line that says
# "passed". So collection is recorded here rather than trusted.
# ---------------------------------------------------------------------------

COLLECTION_ERRORS: list[str] = []
"""nodeids that failed to collect. Non-empty means a module did not import."""

COLLECTED_PER_FILE: dict[str, int] = {}
"""Test file (repo-relative) -> number of items collected from it."""


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.failed:
        COLLECTION_ERRORS.append(report.nodeid or "<unknown>")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Record what was COLLECTED, before -m or -k deselection removes anything.

    tryfirst is load-bearing. Deselection edits `items` in place, so a hook
    running after it would record the filtered run instead of the collected
    tree - and `make check` runs `-m determinism`, which deselects most of the
    suite. The baseline comparison would then be measured against whatever
    happened to be selected, which is the failure mode this whole file exists
    to prevent: a check whose subject quietly shrank to nothing.
    """
    COLLECTED_PER_FILE.clear()
    root = session.config.rootpath
    for item in items:
        try:
            relative = str(item.path.relative_to(root))
        except ValueError:  # pragma: no cover - an item outside the repo
            relative = str(item.path)
        COLLECTED_PER_FILE[relative] = COLLECTED_PER_FILE.get(relative, 0) + 1


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
    outcome = yield
    report = outcome.get_result()  # type: ignore[attr-defined]
    if report.when != "call":
        return
    # Recorded here rather than parsed back out of --durations output, so the
    # budget check names real tests without depending on a report format.
    _durations.append((report.nodeid, float(report.duration)))
    for marker in FAULT_CATEGORIES:
        if item.get_closest_marker(marker) is None:
            continue
        _counts[marker] += 1
        if not report.passed:
            _failed[marker] += 1


_durations: list[tuple[str, float]] = []
"""(nodeid, seconds) per test, for naming the slowest when the budget fails."""

_session_start: float = 0
"""perf_counter at session start. The ONLY clock read in tests/conftest.py.

Not a D2 concern: it measures how long the suite took and never reaches a
result. Same reasoning as settlesense/core/telemetry.py, which is the only
module in the package permitted a clock.
"""


def pytest_sessionstart(session: pytest.Session) -> None:
    global _session_start
    _session_start = time.perf_counter()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the RUN when the suite exceeds its budget (SDD 7).

    A non-zero exit rather than a warning: a warning about runtime is a warning
    nobody reads on the twentieth green run, and the point of a budget is that
    exceeding it stops something.

    Skipped when the run is a subset - `pytest tests/test_store.py` is not the
    suite, and judging a fraction of it against the whole suite's budget would
    make the check meaningless in exactly the situation a developer is in most
    often.
    """
    if exitstatus != 0 or not _is_full_run(session):
        return
    verdict = budget.judge(time.perf_counter() - _session_start, _slowest_durations())
    if verdict.over:
        session.exitstatus = 1


def _is_full_run(session: pytest.Session | None) -> bool:
    """A whole-suite run, not a filtered one.

    Deliberately generous: the budget binds only when everything ran, so a
    subset can never fail it - and a subset can never satisfy it either, which
    is why the summary line says so.
    """
    # None means the reporter has no session to ask - treated as a subset, so
    # an unknown run is never judged against the whole suite's allowance.
    collected = getattr(session, "testscollected", 0) if session is not None else 0
    return bool(collected) and collected >= _EXPECTED_FULL_RUN


_EXPECTED_FULL_RUN = 500
"""Below this, the run is a subset and the budget does not apply.

A floor rather than an exact count, because the exact count is what
tests/collection_baseline.json is for and duplicating it here would create a
second place to update.
"""


def _slowest_durations() -> tuple[str, ...]:
    """The slowest calls of THIS session, from recorded reports.

    Takes no session argument: the durations are collected by the makereport
    hook above, so asking pytest for them again would be a second source of
    truth for one fact.
    """
    reports = sorted(_durations, key=lambda item: item[1], reverse=True)
    return tuple(f"{name}  {seconds:.2f}s" for name, seconds in reports[: budget.SLOWEST_TO_NAME])


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    _report_interactions(terminalreporter)
    _report_budget(terminalreporter)
    if not any(_counts.values()):
        return
    terminalreporter.write_sep("=", "fault injection by category")
    total = 0
    for marker, heading in FAULT_CATEGORIES.items():
        count, failed = _counts[marker], _failed[marker]
        if not count:
            continue
        total += count
        suffix = f"  ({failed} FAILING)" if failed else ""
        terminalreporter.write_line(f"  {heading}  {count:>3}{suffix}")
    terminalreporter.write_line(f"  {'':<55}{'':>3}")
    terminalreporter.write_line(f"  {'TOTAL guards proven able to fail':<55}{total:>3}")


def _report_budget(terminalreporter: pytest.TerminalReporter) -> None:
    """The realised duration, on EVERY run. Green or red.

    Printed unconditionally: a budget that only speaks when breached gives no
    warning that it is about to be, and the run before the failure looks
    identical to one from a month earlier.
    """
    if not _session_start:
        return
    session = terminalreporter._session
    verdict = budget.judge(time.perf_counter() - _session_start, _slowest_durations())
    terminalreporter.write_sep("=", "suite budget (SDD 7)")
    if not _is_full_run(session):
        terminalreporter.write_line(
            f"  {verdict.seconds:.1f}s for a SUBSET - the {budget.BUDGET_SECONDS}s "
            "budget applies to the whole suite only"
        )
        return
    terminalreporter.write_line(verdict.summary())
    if verdict.over:
        terminalreporter.write_line(verdict.failure_message())


def _report_interactions(terminalreporter: pytest.TerminalReporter) -> None:
    """Print every amplified interaction and what it would have been at
    production rates.

    Reported unconditionally rather than only on failure. An interaction that is
    shrinking - 40 co-occurrences last month, 3 today - is on its way to zero,
    and zero is where the test silently stops testing anything. The number has
    to be visible while it is still non-zero.
    """
    lines = amplification.summary_lines()
    if not lines:
        return
    terminalreporter.write_sep("=", "rate-amplified interactions")
    for line in lines:
        terminalreporter.write_line(line)

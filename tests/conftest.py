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

from collections.abc import Iterator

import pytest

# marker -> the human-facing heading in the summary
FAULT_CATEGORIES: dict[str, str] = {
    "config_refusal": "Config refusals        (a broken config must not load)",
    "charter_guard": "Charter guards         (D1-D13 scanners fed a violation)",
    "truth_injection": "Truth self-check       (corrupted data must not be written)",
    "noise_accounting": "Noise accounting       (the ledger must balance both ways)",
    "hygiene": "Repository hygiene     (tree and manifest invariants)",
}

_counts: dict[str, int] = dict.fromkeys(FAULT_CATEGORIES, 0)
_failed: dict[str, int] = dict.fromkeys(FAULT_CATEGORIES, 0)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
    outcome = yield
    report = outcome.get_result()  # type: ignore[attr-defined]
    if report.when != "call":
        return
    for marker in FAULT_CATEGORIES:
        if item.get_closest_marker(marker) is None:
            continue
        _counts[marker] += 1
        if not report.passed:
            _failed[marker] += 1


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
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

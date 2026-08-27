"""The runtime budget: the decision, the wiring, and the stated figure (SDD 7).

THE MEASUREMENT IS NOT TAKEN HERE. It is taken by `tests/conftest.py` from the
real session and printed on every run, green or red. The first version of this
file ran the whole suite in a subprocess to time it, which DOUBLED every
`make test` - a measurement costing as much as the thing measured, so a
developer waits twice as long to learn a fact about waiting.

What is tested here is `tests/budget.py`'s decision, fed known durations at,
above and below the threshold; that the hook is actually wired, because a
decision function nothing calls is a decision nobody makes; and that the
figure is stated consistently in the three places a reader might look.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests import budget

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.hygiene
def test_the_budget_decision_fires_above_the_threshold_and_not_below() -> None:
    """FAULT INJECTION. Three points, including the boundary.

    The boundary is included because `>` and `>=` are a coin flip when written
    and the difference only shows on a run that lands exactly on the number.
    A run AT the budget is inside it.
    """
    under = budget.judge(seconds=10, budget=120)
    exact = budget.judge(seconds=120, budget=120)
    over = budget.judge(seconds=120.1, budget=120)

    assert not under.over and not exact.over, (under.over, exact.over)
    assert over.over, "a run past the budget was judged acceptable"
    assert under.fraction < exact.fraction < over.fraction
    print(
        f"\n  10s -> {under.over}, 120s -> {exact.over} (at the budget is inside it), "
        f"120.1s -> {over.over}"
    )


@pytest.mark.hygiene
def test_the_failure_message_names_the_slowest_tests() -> None:
    """ "The suite got slower" is a complaint; naming three tests is evidence."""
    verdict = budget.judge(
        seconds=200,
        slowest=("tests/test_a.py::slow  9.00s", "tests/test_b.py::slower  8.00s"),
        budget=120,
    )
    message = verdict.failure_message()
    assert "200.0s" in message and "120s" in message
    for entry in verdict.slowest:
        assert entry in message, f"{entry} is not named in the failure message"
    assert "Do NOT add a fast target" in message

    silent = budget.judge(seconds=200, slowest=(), budget=120)
    assert "no durations reported" in silent.failure_message(), (
        "an empty durations list renders as a blank list, which reads as "
        "'nothing was slow' for a run that was over budget"
    )
    print(f"\n  message names {len(verdict.slowest)} tests; empty case says so explicitly")


@pytest.mark.boundary_refusal
def test_a_negative_duration_is_refused() -> None:
    """A clock that went backwards is a broken measurement, not a fast run."""
    with pytest.raises(ValueError, match="cannot take"):
        budget.judge(seconds=-1)
    print("\n  negative duration refused")


@pytest.mark.hygiene
def test_the_budget_is_actually_wired_into_the_session() -> None:
    """A decision function nothing calls is a decision nobody makes.

    The hooks are checked by name in conftest rather than by triggering a
    failing session: making the real suite exceed its budget to prove the hook
    fires would take two minutes and prove it once.
    """
    conftest = (REPO / "tests" / "conftest.py").read_text(encoding="utf-8")
    for hook in ("pytest_sessionstart", "pytest_sessionfinish", "pytest_terminal_summary"):
        assert f"def {hook}(" in conftest, f"{hook} is not defined - the budget is not measured"
    assert "session.exitstatus = 1" in conftest, (
        "the budget prints but does not FAIL the run - a warning about runtime is "
        "a warning nobody reads on the twentieth green run"
    )
    assert "budget.judge(" in conftest, "conftest does not consult the budget decision"
    assert "_is_full_run" in conftest, (
        "the budget is not gated on a full run, so `pytest tests/test_store.py` "
        "would be judged against the whole suite's allowance"
    )
    print("\n  sessionstart/sessionfinish/terminal_summary wired; failure sets exitstatus")


@pytest.mark.hygiene
def test_a_subset_run_is_not_judged_against_the_whole_suites_budget() -> None:
    """THIS RUN proves it: if you are reading this from a `-k` invocation, the
    summary above said SUBSET rather than passing judgement.

    Asserted structurally, since the running session cannot know in advance
    whether it is a subset.
    """
    conftest = (REPO / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "SUBSET" in conftest, "a partial run is judged silently or not at all"
    assert re.search(r"_EXPECTED_FULL_RUN\s*=\s*\d+", conftest), "no full-run floor is defined"
    print("\n  partial runs report SUBSET instead of a verdict")


@pytest.mark.hygiene
def test_there_is_no_fast_target_to_run_instead() -> None:
    """The absence is the point, and it is asserted rather than remembered.

    A `test-fast` target would be added in good faith by someone in a hurry,
    and from that moment the expensive tests run only when a human remembers
    to. This fails the moment such a target appears, so adding one becomes a
    conversation rather than a convenience.
    """
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    targets = re.findall(r"^([a-zA-Z][\w-]*):", makefile, flags=re.MULTILINE)
    offenders = [
        target
        for target in targets
        if target != "test" and re.search(r"fast|quick|smoke|subset", target, re.I)
    ]
    assert not offenders, (
        f"a fast test target exists: {offenders}. It becomes the one that runs in "
        "the inner loop, and the slow tests then execute only when someone "
        "remembers make check."
    )
    assert "test" in targets, f"there is no test target at all: {targets}"
    print(f"\n  {len(targets)} make targets, none of them a fast subset of `test`")


@pytest.mark.hygiene
def test_the_budget_is_stated_consistently_everywhere() -> None:
    """One number, three places. A stale copy is how a budget stops binding."""
    sdd = (REPO / "SettleSense_SDD.md").read_text(encoding="utf-8")
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    assert f"under **{budget.BUDGET_SECONDS} seconds**" in sdd, "SDD 7 does not state this budget"
    assert f"under {budget.BUDGET_SECONDS}s" in makefile, "the Makefile help text is stale"
    assert f"under {budget.BUDGET_SECONDS} seconds" in readme, "the README is stale"

    for name, text in (("Makefile", makefile), ("README.md", readme)):
        assert "under 60s" not in text and "under 60 seconds" not in text, (
            f"{name} still states the superseded 60s budget"
        )
    assert "was 60 seconds" in sdd, (
        "the SDD raised the budget without recording that it was raised, so the "
        "change reads as though the figure had always been 120"
    )
    print(f"\n  {budget.BUDGET_SECONDS}s stated in SDD 7, Makefile help, and README")


@pytest.mark.hygiene
def test_the_two_invariants_that_matter_are_still_asserted_elsewhere() -> None:
    """Duration is negotiable. These two are not.

    Named here so a future decision to relax the budget cannot be mistaken for
    relaxing the suite's actual guarantees.
    """
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((REPO / "tests").glob("test_*.py"))
    }
    network = [name for name, text in sources.items() if "socket" in text]
    determinism = [
        name
        for name, text in sources.items()
        if "byte-identical" in text or "result_hash" in text or "byte_identical" in text
    ]
    assert len(network) >= 2, f"only {network} guard against network calls"
    assert determinism, "nothing asserts byte-identical results on repeat runs"
    print(
        f"\n  zero-network asserted in {len(network)} modules; byte-identity in {len(determinism)}"
    )

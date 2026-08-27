"""The suite's runtime budget (SDD 7), as a decision that can be unit-tested.

WHY A MODULE AND NOT A TEST. The obvious implementation - a test that runs the
whole suite in a subprocess and times it - DOUBLES every `make test`: the
measurement costs as much as the thing measured, and a developer waits twice
as long to learn a fact about waiting. So the measurement is taken from the
real session by a conftest hook, at no extra cost, and the DECISION lives here
where it can be fed known inputs and shown to fail.

WHY IT FAILS THE RUN RATHER THAN PRINTING A WARNING. A warning about the
budget is a warning nobody reads on the twentieth green run. `pytest_sessionfinish`
sets a non-zero exit status, so `make test` goes red - which is what a budget
is for.

NO `make test-fast`. Deliberately, and asserted. A fast target becomes the one
that runs in the inner loop, and the expensive tests then run only when someone
remembers `make check`. That is the same failure as `-q` swallowing output and
as a guard that inspects an empty set: a check that is not running looks
exactly like one that passes. This project has hit that family three times.

DURATION IS A COMFORT PROPERTY. The two invariants that matter are zero network
calls and byte-identical results on repeat runs. This file must never be the
reason one of those tests is deleted; if the budget and a correctness test ever
conflict, the budget moves.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BUDGET_SECONDS", "SLOWEST_TO_NAME", "BudgetVerdict", "judge"]

BUDGET_SECONDS = 120
"""SDD 7. Raised from 60 at M6.

The original was written at M0, before a single test existed, with no
measurement behind it. It drifted for six modules until the realised suite
reached 58.8s. This figure has evidence behind it - printed on every run.
"""

SLOWEST_TO_NAME = 3
"""How many durations to name when the budget is exceeded.

Naming them turns "the suite got slower" into a decision about specific tests,
which is the difference between evidence and a complaint.
"""


@dataclass(frozen=True)
class BudgetVerdict:
    """What the run cost, and whether that is acceptable."""

    seconds: float
    budget: float
    slowest: tuple[str, ...]

    @property
    def over(self) -> bool:
        return self.seconds > self.budget

    @property
    def fraction(self) -> float:
        return self.seconds / self.budget if self.budget else 0

    def summary(self) -> str:
        """One line, printed on EVERY run - green or red.

        Printed unconditionally on purpose: a budget that only speaks when
        breached gives no warning that it is about to be, and the run before
        the failure looks identical to the run a month earlier.
        """
        verdict = "OVER BUDGET" if self.over else "within budget"
        return f"  {self.seconds:.1f}s of {self.budget:.0f}s ({self.fraction:.0%}) - {verdict}"

    def failure_message(self) -> str:
        named = "\n".join(f"    {line}" for line in self.slowest) or "    (no durations reported)"
        return (
            f"  the suite took {self.seconds:.1f}s against a {self.budget:.0f}s budget (SDD 7).\n"
            f"  The {len(self.slowest)} slowest:\n{named}\n"
            "  Raise the budget DELIBERATELY, in a commit recording the realised\n"
            "  figure - or make one of the above cheaper. Do NOT add a fast target\n"
            "  that skips them: a check that is not running looks exactly like one\n"
            "  that passes."
        )


def judge(
    seconds: float, slowest: tuple[str, ...] = (), budget: float = BUDGET_SECONDS
) -> BudgetVerdict:
    """Decide, given a measured duration.

    `budget` is a parameter so the decision can be tested at, above and below
    the threshold without editing a module constant - a guard whose threshold
    cannot be varied cannot be shown to fire.
    """
    if seconds < 0:
        raise ValueError(f"a run cannot take {seconds}s")
    return BudgetVerdict(seconds=seconds, budget=budget, slowest=tuple(slowest))

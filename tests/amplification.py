"""The rate-amplification rule, implemented once.

A fault injection at production rates proves nothing about rare interactions.

This was learned the hard way. Four tests in this suite passed while the bug
they existed to catch was present, because production rates make the
co-occurrence they depend on vanishingly rare:

    partial_captures (0.020) x duplicate_ledger_rows (0.010) over ~5,000 chains
        = 1.0 expected co-occurrences

One. Seed 42 happened to produce zero, so the test asserted a property about an
empty set and reported success. The same shape recurred three more times: a
120-ordering permutation sweep that never reached the interaction, a dangling
annotation check green at production rates, and a narration pair sweep where
seven of twelve pairs skipped.

THE RULE

  1. Compute the expected co-occurrence at production rates. Below
     AMPLIFICATION_THRESHOLD, the test MUST run amplified.
  2. Amplify until the interaction is guaranteed, and ASSERT IT OCCURRED before
     asserting anything about it. A test that skips because its precondition
     never fired is a failing test, not a neutral one.
  3. Report the realised count. Zero fails.
  4. A production-rate test is a SMOKE CHECK, never the proof.

  Corollary: amplifying a destructive injector to 1.0 can wipe out every
  analysable row - the interaction occurs and leaves nothing to observe. Find
  the rate where both the interaction happens AND survivors remain, then assert
  on the survivor count too.

Why this lives in one module rather than four copies: the arithmetic is the
part that is easy to get subtly wrong, and four subtly different versions of it
would be four different definitions of "rare".
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

# Below this many expected co-occurrences, a production-rate test is measuring
# luck. Ten is chosen so that a Poisson process with this mean has a probability
# of producing zero of about 4.5e-5 - rare enough that a green run is evidence
# rather than a coin toss.
AMPLIFICATION_THRESHOLD: Decimal = Decimal("10")

# Every interaction observed during a run, for the end-of-run report.
OBSERVED: list[Observation] = []


@dataclass(frozen=True)
class Observation:
    """One measured interaction: what fired, how often, and how often it would
    have fired if the test had trusted production rates."""

    label: str
    realised: int
    expected_at_production: Decimal
    survivors: int | None = None

    @property
    def amplification_was_required(self) -> bool:
        return self.expected_at_production < AMPLIFICATION_THRESHOLD


def expected_cooccurrence(population: int, *rates: Decimal) -> Decimal:
    """Expected count of entities hit by EVERY one of `rates`.

    Independent Bernoulli draws over one shared population, which is how the
    injectors actually select. Decimal throughout - a probability on a decision
    path is exactly the float hazard D1 exists for, and this figure decides
    whether a test is trusted.
    """
    if population < 0:
        raise ValueError(f"population must be non-negative, got {population}")
    product = Decimal(population)
    for rate in rates:
        if not isinstance(rate, Decimal):  # pragma: no cover - guarded by mypy
            raise TypeError(f"rates must be Decimal, got {type(rate).__name__}")
        product *= rate
    return product


def record(
    label: str,
    realised: int,
    *,
    expected_at_production: Decimal,
    survivors: int | None = None,
    minimum: int = 1,
) -> None:
    """Register a measured interaction, and FAIL if it did not happen.

    Called by every amplified test before it asserts anything about the
    interaction. A healthy count reaches the end-of-run report, so an
    interaction that is quietly shrinking - 40 co-occurrences last month, 3
    today - is visible while it is still non-zero, rather than only once it has
    reached zero and the test has stopped testing anything.

    `survivors` is for the corollary: an interaction that consumed every
    analysable row has occurred and proved nothing.
    """
    assert realised >= minimum, (
        f"{label}: the interaction fired {realised} time(s), needed >= {minimum}.\n"
        f"At production rates it would be expected {expected_at_production:.2f} "
        "times, which is why this test runs amplified. A test whose precondition "
        "never fired has asserted a property about an empty set - it is failing, "
        "not passing."
    )
    if survivors is not None:
        assert survivors > 0, (
            f"{label}: the interaction fired {realised} time(s) but consumed every "
            "analysable row, leaving nothing to assert on. The amplification is "
            "too high - find the rate where the interaction occurs AND survivors "
            "remain."
        )
    # Registered only AFTER both checks pass. The report is a census of healthy
    # interactions; a failed one is already reported by its own failure, and
    # listing it here would put the suite's own fault injections - which fire on
    # purpose - alongside real measurements.
    OBSERVED.append(
        Observation(
            label=label,
            realised=realised,
            expected_at_production=expected_at_production,
            survivors=survivors,
        )
    )


def summary_lines() -> list[str]:
    """The end-of-run table. Empty when no amplified test ran."""
    if not OBSERVED:
        return []
    width = max(len(o.label) for o in OBSERVED)
    lines: list[str] = []
    for observation in sorted(OBSERVED, key=lambda o: o.label):
        at_production = f"{observation.expected_at_production:.2f}"
        flag = "  amplified" if observation.amplification_was_required else "  (would fire anyway)"
        survivors = "" if observation.survivors is None else f", {observation.survivors} survivors"
        lines.append(
            f"  {observation.label:<{width}}  realised {observation.realised:>4}"
            f"{survivors}   production expectation {at_production:>7}{flag}"
        )
    return lines

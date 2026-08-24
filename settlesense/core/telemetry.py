"""M5a - StageTiming and RunTelemetry. NEVER imported by types.py.

Wall-clock data does not enter the business result (SDD 8.1, D6). The earlier
design put timings inside the result and stripped them before comparison; that
was a patch over a design error, and the rule now is that there is nothing to
strip because the two never met.

`seconds` is a float HERE and only here. Telemetry is never hashed, compared,
goldened, or used in a decision, so the D1 prohibition on floats - which exists
because float sums depend on order - does not apply. A float in
ReconciliationResult would be a D6 violation; a float in this module is the
correct type for a duration.

Populating this fully is M5a. What exists now is the shape the engine returns,
so that "two return values" is a fact about the code rather than a promise.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field

__all__ = ["MachineSpec", "RunTelemetry", "StageTiming"]


@dataclass(frozen=True)
class StageTiming:
    """One pipeline stage. `records_in`/`records_out` make a stage that dropped
    rows visible without reading the business result."""

    stage: str
    seconds: float  # float is fine HERE: telemetry is never compared or hashed
    records_in: int
    records_out: int


@dataclass(frozen=True)
class MachineSpec:
    """Captured automatically so a benchmark states what produced it."""

    python_version: str
    platform: str
    processor: str

    @staticmethod
    def current() -> MachineSpec:
        return MachineSpec(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            processor=platform.processor() or "unknown",
        )


@dataclass(frozen=True)
class RunTelemetry:
    """The second return value. Written to reports/, never to the state DB."""

    timings: tuple[StageTiming, ...] = ()
    peak_rss_bytes: int = 0
    machine: MachineSpec = field(default_factory=MachineSpec.current)

    def total_seconds(self) -> float:
        return sum(timing.seconds for timing in self.timings)

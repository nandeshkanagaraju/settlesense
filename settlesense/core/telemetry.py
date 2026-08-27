"""M5a - StageTiming and RunTelemetry. NEVER imported by types.py.

Wall-clock data does not enter the business result (SDD 8.1, D6). The earlier
design put timings inside the result and stripped them before golden
comparison; that was a patch over a design error, and the rule now is that
there is nothing to strip BECAUSE THE TWO NEVER MEET. If a comparator in this
repo ever needs a strip step, a field is in the wrong object.

`seconds` is a float HERE and only here. Telemetry is never hashed, compared,
goldened, or used in a decision, so the D1 prohibition on floats - which exists
because float sums depend on order - does not apply. A float in
ReconciliationResult would be a D6 violation; a float in this module is the
correct type for a duration.

THE ONLY CLOCK IN settlesense/ (D2)
-----------------------------------
This module is the single place in the package permitted to read a clock, and
it may read only `time.perf_counter()`. D2 forbids `datetime.now()`,
`date.today()` and `time.time()` everywhere, including here, because those are
the calls that make a RESULT depend on when it ran. `perf_counter` is a
monotonic counter with no defined epoch: it cannot be formatted as a date, it
cannot be compared across processes, and nothing derived from it reaches
ReconciliationResult. It measures how long something took, which is the one
question a wall clock is legitimately for.

`tests/test_determinism_guard.py` enforces this by PATH: perf_counter is
exempted for this file alone, and `time.time()` here is still a violation. The
exemption is proven to be narrow by a fault injection in both directions.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field
from types import TracebackType

__all__ = ["MachineSpec", "RunTelemetry", "StageTimer", "StageTiming", "format_rate"]


def format_rate(rate: float | None) -> str:
    """A dash, never a number, when the stage was too fast to time.

    LIVES HERE because it is the other half of `records_per_second` returning
    None: that decision is only worth anything if every renderer honours it,
    and it was honoured by two identical private copies - one in eval/bench.py,
    one in eval/run_eval.py - which is how a pair drifts. One definition, beside
    the property whose contract it implements.

    It also keeps `float` out of eval/run_eval.py's annotations. D6's exemption
    covers telemetry and the benchmark; a third module joining that list should
    be a decision someone makes, not a side effect of needing to print a rate.
    """
    return "—" if rate is None else f"{rate:,.0f}"


@dataclass(frozen=True)
class StageTiming:
    """One pipeline stage. `records_in`/`records_out` make a stage that dropped
    rows visible without reading the business result."""

    stage: str
    seconds: float  # float is fine HERE: telemetry is never compared or hashed
    records_in: int
    records_out: int

    @property
    def records_per_second(self) -> float | None:
        """Throughput, or NONE when the stage was too fast to time.

        None rather than 0.0 or inf. A stage that completed inside the clock's
        resolution has an UNMEASURED rate, and the two obvious alternatives
        both state something false: 0.0 reads as "this stage is slow", inf
        reads as "infinitely fast", and a report that prints either has
        invented a number. None forces the renderer to print a dash.

        `records_in` is the numerator: throughput is what the stage was ASKED
        to process. Using records_out would flatter every filtering stage,
        since a pass that rejects 90% of its input would report a tenth of the
        work it actually did.
        """
        if self.seconds <= 0:
            return None
        return self.records_in / self.seconds


@dataclass(frozen=True)
class MachineSpec:
    """Captured automatically so a benchmark states what produced it.

    A throughput figure without a machine is not a measurement, it is a boast.
    """

    cpu: str
    cores: int
    ram_bytes: int
    python_version: str
    platform: str

    @staticmethod
    def current() -> MachineSpec:
        return MachineSpec(
            cpu=platform.processor() or platform.machine() or "unknown",
            cores=os.cpu_count() or 0,
            ram_bytes=_total_ram_bytes(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
        )

    def describe(self) -> str:
        """One line for a report header."""
        gib = self.ram_bytes / (1024 * 1024 * 1024) if self.ram_bytes else None
        ram = f"{gib:.0f} GiB" if gib else "unknown RAM"
        return (
            f"{self.cpu} · {self.cores} cores · {ram} · "
            f"Python {self.python_version} · {self.platform}"
        )


def _total_ram_bytes() -> int:
    """Physical RAM, or 0 when the platform will not say.

    0 means UNKNOWN and `describe` renders it as such. Guessing a plausible
    default would put an invented number in a report header, which is the one
    place a reader is entitled to trust.
    """
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return 0


@dataclass(frozen=True)
class RunTelemetry:
    """The second return value. Written to reports/, never to the state DB."""

    timings: tuple[StageTiming, ...] = ()
    peak_rss_bytes: int = 0
    machine: MachineSpec = field(default_factory=MachineSpec.current)

    def total_seconds(self) -> float:
        return sum(timing.seconds for timing in self.timings)

    def stage(self, name: str) -> StageTiming | None:
        """One stage by name, or None. Used by tests and the bench renderer."""
        for timing in self.timings:
            if timing.stage == name:
                return timing
        return None


class StageTimer:
    """Times one stage and appends a StageTiming to a collector.

    NO GLOBAL STATE. The collector is passed at construction, so two runs in
    the same process - which is exactly what the benchmark's three repetitions
    are - cannot contaminate each other. A module-level list would make the
    second repetition's numbers depend on whether the first one ran.

    `records_out` defaults to `records_in` because most stages pass their rows
    through; a stage that filters sets `timer.records_out` before exiting.

    A None collector reads NO clock at all - not a cheap one, none. The engine
    passes None on every production call, so "instrumentation is off the hot
    path" is a property of the code rather than an assurance about how small
    perf_counter is.
    """

    __slots__ = ("_collector", "_records_in", "_stage", "_start", "records_out")

    def __init__(self, collector: list[StageTiming] | None, stage: str, records_in: int) -> None:
        self._collector = collector
        self._stage = stage
        self._records_in = records_in
        self._start: float = 0
        self.records_out = records_in

    def __enter__(self) -> StageTimer:
        # The clock is read on ENTRY, not at construction: a timer built early
        # and entered later would otherwise silently bill the gap between them.
        if self._collector is not None:
            self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._collector is not None:
            elapsed = time.perf_counter() - self._start
            self._collector.append(
                StageTiming(
                    stage=self._stage,
                    seconds=elapsed,
                    records_in=self._records_in,
                    records_out=self.records_out,
                )
            )

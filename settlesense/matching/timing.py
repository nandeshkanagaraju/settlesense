"""M3 - working days, T+N settlement due dates, and timing explanation.

SEMANTICS MATCH gen/lifecycle.py EXACTLY, and that is a hard requirement
rather than a nicety. The generator used its own calendar to decide when every
settlement landed, and truth records 71 cases as T_PLUS_N_TIMING on that
basis. If this module counted working days differently by even one, thousands
of clean cases would be classified as late and the 71 real ones would be lost
in the noise. The two implementations are independent by design (gen/ may not
import settlesense/), so `test_timing_matches_the_generator` compares them on
the frozen dataset rather than trusting that they were written to agree.

No clock access anywhere (D2). `as_of` is a parameter at every entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from settlesense.config import CalendarConfig

__all__ = [
    "TimingVerdict",
    "WorkingDayCalendar",
    "bank_value_due_date",
    "is_timing_explained",
    "settlement_due_date",
    "settlement_line_due_date",
]

_MAX_CALENDAR_STEPS = 400
"""Bound on any calendar walk. A malformed calendar - every weekday marked a
weekly off - would otherwise loop forever rather than fail."""


@dataclass(frozen=True)
class WorkingDayCalendar:
    """Working-day arithmetic over a versioned calendar.

    Wraps CalendarConfig rather than re-reading YAML: config loading belongs to
    settlesense/config.py, and a second parser is a second place for the
    holiday list to be wrong.
    """

    config: CalendarConfig

    @property
    def version(self) -> str:
        """Stamped into every ReconciliationResult (SDD 8.1)."""
        return self.config.version

    def is_working_day(self, day: date) -> bool:
        return self.config.is_working_day(day)

    def next_working_day(self, day: date) -> date:
        """The first working day ON OR AFTER `day`. A working day maps to itself."""
        current = day
        for _ in range(_MAX_CALENDAR_STEPS):
            if self.is_working_day(current):
                return current
            current += timedelta(days=1)
        raise ValueError(
            f"no working day within {_MAX_CALENDAR_STEPS} days of {day}. The "
            "calendar marks every day non-working; check weekly_offs."
        )

    def add_working_days(self, start: date, count: int) -> date:
        """`count` working days after `start`, WHICH IS FIRST ROLLED FORWARD.

        The roll-forward is why `add_working_days(saturday, 0)` is the
        following Monday while `add_working_days(monday, 0)` is that Monday.
        Both are correct and they are not in tension: a settlement cannot be
        instructed on a Saturday, so T+0 from a Saturday means "the first day
        this could actually happen".

        Counting BEFORE rolling would let a Saturday capture settle on Monday
        at T+1 and on Monday at T+0 - two different inputs, one output, and a
        day of drift that appears only for weekend captures.
        """
        if count < 0:
            raise ValueError(f"cannot add {count} working days; count must be >= 0")
        current = self.next_working_day(start)
        remaining = count
        steps = 0
        while remaining > 0:
            current += timedelta(days=1)
            steps += 1
            if steps > _MAX_CALENDAR_STEPS:
                raise ValueError(f"working-day walk from {start} exceeded {steps} steps")
            if self.is_working_day(current):
                remaining -= 1
        return current

    def working_days_between(self, start: date, end: date) -> int:
        """Working days from `start` to `end`, signed. Used to size a delay.

        Signed rather than absolute: a settlement landing EARLY is a different
        finding from one landing late, and an absolute value would merge them.
        """
        if end == start:
            return 0
        step = 1 if end > start else -1
        current, count, steps = start, 0, 0
        while current != end:
            current += timedelta(days=step)
            steps += 1
            if steps > _MAX_CALENDAR_STEPS:
                raise ValueError(f"walk from {start} to {end} exceeded {steps} steps")
            if self.is_working_day(current):
                count += step
        return count


def settlement_line_due_date(captured_at: date, calendar: WorkingDayCalendar) -> date:
    """When a captured payment's SETTLEMENT LINE is due to be dated: T+0.

    T+0, not T+N, and getting this wrong is the single largest classification
    error available in this module. The gateway dates a settlement line on the
    first working day on or after capture; the T+N cycle governs the BANK leg,
    from batch to credit. Applying T+N here classifies almost every clean case
    as late - measured, it reported 4841 timing exceptions against a true 71.

    The two legs compose: a payment captured on a Saturday appears on a line
    dated Monday, and its cash lands N working days after that.
    """
    return calendar.add_working_days(captured_at, 0)


def settlement_due_date(captured_at: date, profile: str, calendar: WorkingDayCalendar) -> date:
    """When the CASH for a payment captured on `captured_at` is due, per T+N.

    This is the bank leg. Equal to add_working_days(batch_date, cycle) for a
    clean chain, because add_working_days rolls its start to a working day
    before counting and the batch is already dated on one - so composing the
    two legs and taking the whole cycle from capture give the same date.

    The cycle comes from config, per profile, and an unknown profile raises -
    defaulting to a common cycle would silently classify a whole merchant's
    settlements as on-time or late depending on a number nobody chose.
    """
    cycle = calendar.config.settlement_cycle_for(profile)
    return calendar.add_working_days(captured_at, cycle)


def bank_value_due_date(
    batch_settled_date: date, profile: str, calendar: WorkingDayCalendar
) -> date:
    """When a batch settled on `batch_settled_date` is due to credit the bank.

    Mirrors gen/lifecycle.py exactly: value_date = add_working_days(batch date,
    settlement_cycle_days). Named separately from settlement_due_date because
    the inputs differ - one starts at capture, one at the batch - and a caller
    passing the wrong one gets a plausible date that is silently off by the
    length of a weekend.
    """
    cycle = calendar.config.settlement_cycle_for(profile)
    return calendar.add_working_days(batch_settled_date, cycle)


@dataclass(frozen=True)
class TimingVerdict:
    """Why a settlement date was or was not considered on time.

    Carries the numbers rather than a bare bool, for the same reason
    DuplicateVerdict does: "late" without "by how many working days, against
    what due date" cannot be audited, and evidence links need the figures.
    """

    expected: date
    actual: date
    working_days_late: int
    tolerance_days: int
    explained: bool

    @property
    def is_late(self) -> bool:
        return self.working_days_late > 0

    @property
    def is_early(self) -> bool:
        return self.working_days_late < 0


def is_timing_explained(
    expected_date: date,
    actual_date: date,
    tolerance_days: int,
    calendar: WorkingDayCalendar,
) -> TimingVerdict:
    """Whether `actual_date` is within `tolerance_days` WORKING days of due.

    Working days, not calendar days. A settlement due Friday and landing Monday
    is one working day late, not three, and counting calendar days would report
    every weekend as a delay.
    """
    if tolerance_days < 0:
        raise ValueError(f"tolerance_days must be >= 0, got {tolerance_days}")
    late = calendar.working_days_between(expected_date, actual_date)
    return TimingVerdict(
        expected=expected_date,
        actual=actual_date,
        working_days_late=late,
        tolerance_days=tolerance_days,
        explained=abs(late) <= tolerance_days,
    )


def is_within_arrival_horizon(event_date: date, as_of: date) -> bool:
    """Whether an event could have been observed by `as_of`.

    `as_of` is INJECTED (D2). Nothing in this module reads a clock, which is
    what makes two runs at different simulated dates comparable - and what
    makes `test_as_of_is_honoured` able to change the answer by changing only
    the parameter.
    """
    return event_date <= as_of

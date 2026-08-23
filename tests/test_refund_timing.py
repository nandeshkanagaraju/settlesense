"""A refund can never net off a payout that has already left the bank.

SDD 3.1a places every REFUND line in the batch settling ON OR AFTER
`refund.created_at`. Violating it is not a rounding error - it is a debit
applied to money that is already gone, so the batch total, the bank credit and
the case's expected_net all disagree with reality in a way no downstream pass
can detect. It looks like an ordinary variance.

The original defect had three parts and needed three fixes:

  1. Refund dates were not bounded, so with few capture days a refund could be
     created after EVERY batch had settled.
  2. The batch-selection fallback then silently picked the last batch - i.e. the
     nearest one BACKWARDS in time. Silently wrong, not loudly wrong.
  3. Nothing asserted the rule afterwards, so it could regress unnoticed.

Fixing only (1) leaves the silent fallback armed for whoever next changes the
day count. These tests pin all three independently, so removing any one fails.
"""

from __future__ import annotations

import pathlib
import random
from dataclasses import replace
from datetime import date, timedelta

import pytest

from gen.generate import build_plan
from gen.lifecycle import (
    CleanDataset,
    GeneratorError,
    SettlementLineType,
    WorkingCalendar,
    _first_on_or_after,
    assemble_batches,
    build_clean_dataset,
    load_working_calendar,
    verify_clean_dataset,
)
from gen.profiles import PROFILES

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CALENDAR_PATH = REPO_ROOT / "config" / "calendar_v1.yaml"

SEED = 42
PROFILES_BY_NAME = {profile.name: profile for profile in PROFILES}

# THREE capture days is the adversarial setting, not twenty. The defect needed
# refunds to postdate every batch, which is exactly what a short window causes:
# refunds are created relative to their payment, so with few capture days the
# late ones fall past the last settlement date. A test run only at --days 20
# would have shown 0 violations and proved nothing.
DAYS = 3
RECORDS = 900


@pytest.fixture(scope="module")
def calendar() -> WorkingCalendar:
    return load_working_calendar(CALENDAR_PATH)


def _dataset(calendar: WorkingCalendar, *, days: int, records: int) -> CleanDataset:
    base_date = calendar.next_working_day(calendar.window_start)
    plan = build_plan(records, days, list(PROFILES))
    return build_clean_dataset(random.Random(SEED), plan, calendar, base_date, SEED)


@pytest.fixture(scope="module")
def short_window(calendar: WorkingCalendar) -> CleanDataset:
    """The configuration the defect was found in."""
    return _dataset(calendar, days=DAYS, records=RECORDS)


@pytest.fixture(scope="module")
def production_window(calendar: WorkingCalendar) -> CleanDataset:
    """What `make gen` ships."""
    return _dataset(calendar, days=20, records=5_000)


def _refund_violations(dataset: CleanDataset) -> list[str]:
    """Every REFUND line settling in a batch older than its refund."""
    created = {row.refund_id: row.created_at for row in dataset.refund_rows}
    batch_date = {batch.batch_id: batch.settled_event_date for batch in dataset.batches}
    problems: list[str] = []
    for line in dataset.settlement_lines:
        if line.line_type is not SettlementLineType.REFUND or line.refund_id is None:
            continue
        when_created = created[line.refund_id]
        when_settled = batch_date[line.batch_id]
        if when_settled < when_created:
            problems.append(
                f"{line.settlement_id}: batch {line.batch_id} settled {when_settled} "
                f"but refund {line.refund_id} was created {when_created} "
                f"({(when_created - when_settled).days}d early)"
            )
    return problems


# ===========================================================================
# 1. Zero violations - measured against the BATCH date, not the line date
# ===========================================================================


@pytest.mark.parametrize("window", ["short_window", "production_window"])
def test_no_refund_settles_before_it_exists(window: str, request: pytest.FixtureRequest) -> None:
    """ZERO, not "few".

    Checked against `batch.settled_event_date` rather than the line's own date.
    The line-vs-batch agreement is a separate invariant, and reusing it here
    would make this test pass whenever that one was broken in the same
    direction.
    """
    dataset: CleanDataset = request.getfixturevalue(window)
    violations = _refund_violations(dataset)
    assert not violations, (
        f"{len(violations)} REFUND line(s) settle before their refund exists "
        f"({window}):\n" + "\n".join(violations[:10])
    )


def test_there_were_refunds_to_check(short_window: CleanDataset) -> None:
    """Guards the guard: zero violations over zero refunds is not a result."""
    refund_lines = [
        line
        for line in short_window.settlement_lines
        if line.line_type is SettlementLineType.REFUND
    ]
    assert len(refund_lines) >= 20, (
        f"only {len(refund_lines)} REFUND lines at days={DAYS}; too few to "
        "exercise the late-refund path the defect lived in"
    )
    assert len(refund_lines) == len(short_window.refund_rows)


# ===========================================================================
# 2. The fallback RAISES rather than silently selecting
# ===========================================================================


@pytest.mark.truth_injection
def test_a_refund_after_every_batch_raises(
    short_window: CleanDataset, calendar: WorkingCalendar
) -> None:
    """Force the unreachable path and assert it is loud.

    The clamp makes this unreachable for clean data, which is precisely why the
    fallback must still raise: an unreachable silent fallback is indistinguishable
    from a correct one until someone changes the day count, and then it quietly
    starts producing backwards-in-time refunds again.
    """
    latest = max(batch.settled_event_date for batch in short_window.batches)
    victim = next(chain for chain in short_window.chains if chain.refund_line is not None)
    assert victim.refund_line is not None  # narrowing for mypy

    late = replace(victim.refund_line, settled_event_date=latest + timedelta(days=30))
    chains = tuple(
        replace(chain, refund_line=late) if chain is victim else chain
        for chain in short_window.chains
    )

    with pytest.raises(GeneratorError) as excinfo:
        assemble_batches(chains, calendar, PROFILES_BY_NAME, SEED)

    message = str(excinfo.value)
    assert late.refund_id is not None
    assert late.refund_id in message, "the error must name the offending refund"
    assert "ON OR AFTER" in message, "the error must state the rule it enforces"
    assert victim.profile_name in message, "the error must name the profile whose batches ran out"


@pytest.mark.truth_injection
def test_the_fallback_does_not_silently_pick_an_earlier_batch(
    short_window: CleanDataset, calendar: WorkingCalendar
) -> None:
    """The specific wrong behaviour: choosing the nearest batch BACKWARDS.

    Stated separately from the raise because they are different failures. A
    fallback could be removed and replaced by `return candidates[-1]` and the
    test above would be the only thing standing in the way - this one names the
    outcome rather than the mechanism.
    """
    latest = max(batch.settled_event_date for batch in short_window.batches)
    victim = next(chain for chain in short_window.chains if chain.refund_line is not None)
    assert victim.refund_line is not None

    late = replace(victim.refund_line, settled_event_date=latest + timedelta(days=30))
    chains = tuple(
        replace(chain, refund_line=late) if chain is victim else chain
        for chain in short_window.chains
    )

    try:
        lines, batches, _ = assemble_batches(chains, calendar, PROFILES_BY_NAME, SEED)
    except GeneratorError:
        return  # raising is the correct outcome

    batch_date = {batch.batch_id: batch.settled_event_date for batch in batches}
    placed = next(line for line in lines if line.refund_id == late.refund_id)
    pytest.fail(
        f"no exception raised: refund {late.refund_id} dated "
        f"{late.settled_event_date} was placed in a batch settling "
        f"{batch_date[placed.batch_id]} - a payout that had already left the bank"
    )


# ===========================================================================
# 3. verify_clean_dataset carries the rule, so it cannot regress silently
# ===========================================================================


def test_verify_clean_dataset_passes_on_good_data(short_window: CleanDataset) -> None:
    assert verify_clean_dataset(short_window) == []


@pytest.mark.truth_injection
def test_verify_clean_dataset_catches_a_backdated_refund_line(
    short_window: CleanDataset,
) -> None:
    """Fault injection: move one REFUND line's batch back before its refund.

    Without this, the rule would live only inside `assemble_batches` - enforced
    at construction but never re-checked on the finished artifact. A later noise
    injector that re-batches lines could break it with nothing to notice.
    """
    created = {row.refund_id: row.created_at for row in short_window.refund_rows}
    target = next(
        line
        for line in short_window.settlement_lines
        if line.line_type is SettlementLineType.REFUND and line.refund_id is not None
    )
    assert target.refund_id is not None
    broken_line = replace(target, settled_event_date=created[target.refund_id] - timedelta(days=1))
    broken = replace(
        short_window,
        settlement_lines=tuple(
            broken_line if line is target else line for line in short_window.settlement_lines
        ),
    )

    problems = verify_clean_dataset(broken)
    assert any(target.settlement_id in problem for problem in problems), (
        f"verify_clean_dataset did not flag the backdated refund line "
        f"{target.settlement_id}; problems were: {problems[:5]}"
    )
    assert any("before refund" in problem for problem in problems)


# ===========================================================================
# 4. Same-day edge: ON or after, never before
# ===========================================================================


def test_first_on_or_after_selects_the_same_day() -> None:
    """`>=`, not `>`. A refund created the day a batch settles belongs in THAT batch.

    An off-by-one here pushes every same-day refund into the next batch, which
    is not a violation and so would never be caught by test 1 - it silently
    changes which batch a refund nets off and therefore two batch totals.
    """
    days = [date(2026, 9, 7), date(2026, 9, 10), date(2026, 9, 14)]
    assert _first_on_or_after(days, date(2026, 9, 10)) == date(2026, 9, 10)
    assert _first_on_or_after(days, date(2026, 9, 8)) == date(2026, 9, 10)
    assert _first_on_or_after(days, date(2026, 9, 7)) == date(2026, 9, 7)
    assert _first_on_or_after(days, date(2026, 9, 15)) is None


def test_a_same_day_refund_lands_in_that_batch_not_an_earlier_one(
    short_window: CleanDataset, calendar: WorkingCalendar
) -> None:
    """The end-to-end form: retarget a refund line onto an existing batch date."""
    victim = next(chain for chain in short_window.chains if chain.refund_line is not None)
    assert victim.refund_line is not None
    profile_dates = sorted(
        {
            batch.settled_event_date
            for batch in short_window.batches
            if any(
                line.batch_id == batch.batch_id and line.payment_id == victim.payment.payment_id
                for line in short_window.settlement_lines
            )
        }
    )
    # Any settlement date for this chain's profile will do; use the earliest so
    # an off-by-one would visibly jump forward.
    all_dates = sorted({batch.settled_event_date for batch in short_window.batches})
    same_day = profile_dates[0] if profile_dates else all_dates[0]

    aligned = replace(victim.refund_line, settled_event_date=same_day)
    chains = tuple(
        replace(chain, refund_line=aligned) if chain is victim else chain
        for chain in short_window.chains
    )
    lines, batches, _ = assemble_batches(chains, calendar, PROFILES_BY_NAME, SEED)

    batch_date = {batch.batch_id: batch.settled_event_date for batch in batches}
    placed = next(line for line in lines if line.refund_id == aligned.refund_id)
    landed = batch_date[placed.batch_id]

    assert landed >= same_day, f"same-day refund landed in an EARLIER batch: {landed} < {same_day}"
    assert landed == same_day, (
        f"same-day refund was pushed forward to {landed} instead of settling in "
        f"the {same_day} batch; _first_on_or_after is using `>` where SDD 3.1a "
        f"says ON or after"
    )

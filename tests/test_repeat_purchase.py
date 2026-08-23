"""A repeat purchase is a fresh full capture, never a clone of a partial one.

`duplicate_ledger_rows` emits a genuine repeat purchase - same customer, same
amount, new order_id - so the engine has to distinguish it from an ingestion
artefact using something other than the amount. It builds that repeat by copying
the source payment.

The defect: `partial_captures` runs FIRST in NOISE_ORDER, so the payment being
copied may already have `captured < authorized`. Copying it wholesale carried
that partial-capture state into a brand-new payment_id, annotated only as
DUPLICATE_CANDIDATE. The result was a payment that looks partially captured, is
partially captured arithmetically, and has no PARTIAL_CAPTURE truth anywhere.

Undeclared truth is worse than a missing category. A missing category shows up
as a gap. This shows up as the ENGINE being wrong: it correctly detects a
partial capture, truth says there is none, and the case is scored a false
positive. The generator would be marking correct answers wrong.

Test 3 is the general form and the one that matters: no payment anywhere may
have captured < authorized without an explicit annotation, whatever produced it.
Tests 1 and 2 pin the specific path that broke.
"""

from __future__ import annotations

import pathlib
import random
from dataclasses import fields, replace
from decimal import Decimal

import pytest

from gen.generate import build_plan
from gen.lifecycle import (
    CleanDataset,
    WorkingCalendar,
    build_clean_dataset,
    load_working_calendar,
)
from gen.noise import NoiseLedger, NoiseRates, _noise_id, apply_noise
from gen.profiles import PROFILES
from gen.truth import Truth, VarianceCategory, build_truth

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CALENDAR_PATH = REPO_ROOT / "config" / "calendar_v1.yaml"

SEED = 42
DAYS = 20
RECORDS = 5_000
ZERO = Decimal("0.00")

PROFILES_BY_NAME = {profile.name: profile for profile in PROFILES}

Built = tuple[CleanDataset, NoiseLedger, Truth]


@pytest.fixture(scope="module")
def calendar() -> WorkingCalendar:
    return load_working_calendar(CALENDAR_PATH)


@pytest.fixture(scope="module")
def clean(calendar: WorkingCalendar) -> CleanDataset:
    base_date = calendar.next_working_day(calendar.window_start)
    plan = build_plan(RECORDS, DAYS, list(PROFILES))
    return build_clean_dataset(random.Random(SEED), plan, calendar, base_date, SEED)


def _build(clean: CleanDataset, calendar: WorkingCalendar, **rates: Decimal) -> Built:
    base = NoiseRates(**{field.name: ZERO for field in fields(NoiseRates)})
    dataset, ledger = apply_noise(
        clean,
        random.Random(SEED),
        PROFILES_BY_NAME,
        replace(base, **rates),
        include_withheld=True,
    )
    truth = build_truth(dataset, calendar, PROFILES_BY_NAME, SEED, noise=ledger)
    return dataset, ledger, truth


@pytest.fixture(scope="module")
def shipped(clean: CleanDataset, calendar: WorkingCalendar) -> Built:
    """Production rates: every injector on, as the datasets ship."""
    dataset, ledger = apply_noise(
        clean, random.Random(SEED), PROFILES_BY_NAME, NoiseRates(), include_withheld=True
    )
    return dataset, ledger, build_truth(dataset, calendar, PROFILES_BY_NAME, SEED, noise=ledger)


@pytest.fixture(scope="module")
def saturated(clean: CleanDataset, calendar: WorkingCalendar) -> Built:
    """Rates raised so partial captures and repeats OVERLAP heavily.

    At production rates the overlap is ~1 payment in 5,000 - the defect could
    sit in a shipped dataset for a long time before a repeat happened to clone a
    partial capture. Raising both rates makes the interaction certain rather
    than lucky. The invariant under test does not depend on the rate; only the
    chance of observing a violation does.
    """
    return _build(
        clean,
        calendar,
        partial_captures=Decimal("0.50"),
        duplicate_ledger_rows=Decimal("0.20"),
    )


def _repeat_ids(truth: Truth) -> set[str]:
    """Payments the generator declares to be genuine repeat purchases."""
    return {
        case.payment_id
        for case in truth.cases
        if case.true_category == VarianceCategory.DUPLICATE_CANDIDATE
    }


# ===========================================================================
# 1. Every repeat purchase is a full capture
# ===========================================================================


@pytest.mark.parametrize("scenario", ["shipped", "saturated"])
def test_every_repeat_purchase_captures_in_full(
    scenario: str, request: pytest.FixtureRequest
) -> None:
    dataset, _, truth = request.getfixturevalue(scenario)
    repeats = _repeat_ids(truth)
    assert repeats, f"{scenario}: no repeat purchases generated; nothing was checked"

    partial = [
        f"{row.payment_id}: captured {row.captured} < authorized {row.authorized}"
        for row in dataset.payment_rows
        if row.payment_id in repeats and row.captured < row.authorized
    ]
    assert not partial, (
        f"{len(partial)} repeat purchase(s) are partially captured ({scenario}):\n"
        + "\n".join(partial[:10])
    )


@pytest.mark.parametrize("scenario", ["shipped", "saturated"])
def test_every_repeat_purchase_has_captured_status(
    scenario: str, request: pytest.FixtureRequest
) -> None:
    """`status` must agree with the amounts, or the row contradicts itself."""
    dataset, _, truth = request.getfixturevalue(scenario)
    repeats = _repeat_ids(truth)
    wrong = [
        f"{row.payment_id}: status {row.status!r}"
        for row in dataset.payment_rows
        if row.payment_id in repeats and row.status != "captured"
    ]
    assert not wrong, "repeat purchases with a non-captured status:\n" + "\n".join(wrong[:10])


# ===========================================================================
# 2. No inheritance from a partially-captured source
# ===========================================================================


@pytest.mark.noise_accounting
def test_a_repeat_of_a_partial_capture_is_still_a_full_capture(saturated: Built) -> None:
    """The exact defect: link each repeat back to its source and check.

    The source-to-repeat link is recomputed from the same canonical tuple the
    generator uses, rather than parsed out of the annotation text - the same
    reason truth links a batch to its bank credit by construction.
    """
    dataset, ledger, truth = saturated
    by_id = {row.payment_id: row for row in dataset.payment_rows}
    repeats = _repeat_ids(truth)

    # Which payments were partially captured, per the noise ledger itself.
    partial_sources = {annotation.target_id for annotation in ledger.of_type("partial_captures")}
    assert partial_sources, "no partial captures generated; the interaction is untested"

    examined = 0
    problems: list[str] = []
    for source_id in sorted(partial_sources):
        source = by_id.get(source_id)
        if source is None:
            continue
        for suffix in range(64):  # suffix is a running count; scan a safe range
            repeat_id = _noise_id("PAY", source_id, "repeat", suffix)
            if repeat_id not in repeats:
                continue
            repeat = by_id[repeat_id]
            examined += 1
            if repeat.captured < repeat.authorized:
                problems.append(
                    f"{repeat_id} (repeat of partially-captured {source_id}): "
                    f"captured {repeat.captured} < authorized {repeat.authorized}"
                )
            elif repeat.authorized != source.captured:
                problems.append(
                    f"{repeat_id}: authorized {repeat.authorized} should be the "
                    f"source's CAPTURED amount {source.captured}, not its "
                    f"authorized {source.authorized}"
                )

    assert examined > 0, (
        "no repeat purchase was cloned from a partially-captured payment, so the "
        "inheritance path was never exercised. Raise the saturated rates."
    )
    assert not problems, (
        f"{len(problems)} of {examined} repeat(s) inherited partial-capture state:\n"
        + "\n".join(problems[:10])
    )


def test_a_repeat_purchase_gets_a_new_identity(saturated: Built) -> None:
    """A repeat shares the customer and the amount, never an id (D10).

    If it reused the source's payment_id it would be an ingestion duplicate, not
    a repeat purchase, and the taxonomy distinction the engine must draw would
    not exist in the data.
    """
    dataset, _, truth = saturated
    repeats = _repeat_ids(truth)
    by_id = {row.payment_id: row for row in dataset.payment_rows}
    orders = {row.payment_id: row.order_id for row in dataset.payment_rows}

    source_orders = {orders[pid] for pid in by_id if pid not in repeats and pid in orders}
    shared = sorted(pid for pid in repeats if orders[pid] in source_orders)
    assert not shared, f"repeat purchases reusing a source order_id: {shared[:10]}"


# ===========================================================================
# 3. THE GENERAL FORM - zero unannotated partial captures anywhere
# ===========================================================================


@pytest.mark.parametrize("scenario", ["shipped", "saturated"])
def test_no_unannotated_partial_capture_anywhere_in_the_dataset(
    scenario: str, request: pytest.FixtureRequest
) -> None:
    """Whatever produced it, `captured < authorized` must be declared truth.

    Scanned over EVERY payment row rather than over the repeat purchases,
    because the failure is defined by the data, not by the code path that made
    it. A future injector that copies a payment for some other reason falls into
    exactly the same hole, and this notices without being told about it.
    """
    dataset, _, truth = request.getfixturevalue(scenario)

    annotated = {
        case.payment_id
        for case in truth.cases
        if case.true_category == VarianceCategory.PARTIAL_CAPTURE
        or "partial_captures" in case.noise_types
    }
    unannotated = [
        f"{row.payment_id}: captured {row.captured} < authorized {row.authorized} "
        f"(status {row.status!r})"
        for row in dataset.payment_rows
        if row.captured < row.authorized and row.payment_id not in annotated
    ]
    assert not unannotated, (
        f"{len(unannotated)} partially-captured payment(s) with no truth annotation "
        f"({scenario}):\n" + "\n".join(unannotated[:10])
    )


@pytest.mark.parametrize("scenario", ["shipped", "saturated"])
@pytest.mark.noise_accounting
def test_partial_captures_actually_exist_to_be_checked(
    scenario: str, request: pytest.FixtureRequest
) -> None:
    """Guards the guard: zero unannotated over zero partials proves nothing."""
    dataset, _, _ = request.getfixturevalue(scenario)
    partial = [row for row in dataset.payment_rows if row.captured < row.authorized]
    assert len(partial) >= 10, (
        f"only {len(partial)} partially-captured payments in {scenario}; the "
        "annotation scan has almost nothing to inspect"
    )


@pytest.mark.noise_accounting
def test_every_annotated_partial_capture_really_is_one(shipped: Built) -> None:
    """The other direction: a PARTIAL_CAPTURE label with full capture behind it.

    Two-way accounting, the same rule the noise self-check applies: an
    unannotated deviation fails, and a claim with nothing behind it fails too.
    """
    dataset, _, truth = shipped
    by_id = {row.payment_id: row for row in dataset.payment_rows}
    hollow = [
        f"{case.payment_id}: labelled PARTIAL_CAPTURE but captured "
        f"{by_id[case.payment_id].captured} == authorized "
        f"{by_id[case.payment_id].authorized}"
        for case in truth.cases
        if case.true_category == VarianceCategory.PARTIAL_CAPTURE
        and case.payment_id in by_id
        and by_id[case.payment_id].captured >= by_id[case.payment_id].authorized
    ]
    assert not hollow, "PARTIAL_CAPTURE claimed with no partial capture behind it:\n" + "\n".join(
        hollow[:10]
    )

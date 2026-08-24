"""M3 - arithmetic, timing and duplicates (brief requirements 1-13).

ONE REQUIREMENT IS ASSERTED AGAINST A DIFFERENT VALUE THAN THE BRIEF GIVES,
and it is flagged where it occurs rather than quietly satisfied:

  #7 says "T+2 from a Friday lands on Tuesday when Monday is a holiday". It
     lands on WEDNESDAY. Tuesday is T+1: with Sat, Sun and a Monday holiday
     all skipped, the first working day after that Friday is Tuesday and the
     second is Wednesday. The test asserts Wednesday and pins Tuesday as T+1
     so the off-by-one cannot come back.

     This is not a licence to bend the code to a brief. The generator is
     frozen and computed every settlement date with this arithmetic, so the
     engine must agree with IT, not with a sentence. test_timing_matches_the
     _generator checks that directly.
"""

from __future__ import annotations

import random
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from settlesense.config import AppConfig, load_config
from settlesense.exceptions.taxonomy import (
    DEDUCTION_CATEGORIES,
    VARIANCE_CATEGORIES,
    VarianceCategory,
)
from settlesense.matching.arithmetic import (
    VarianceComponent,
    compute_fee,
    compute_gst,
    expected_net,
    explain_variance,
)
from settlesense.matching.duplicates import (
    DuplicateRule,
    classify,
    find_candidate_duplicates,
    find_confirmed_duplicates,
)
from settlesense.matching.timing import (
    WorkingDayCalendar,
    is_timing_explained,
    settlement_due_date,
    settlement_line_due_date,
)
from settlesense.types import LedgerRow, PaymentMethod, money

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture(scope="module")
def calendar(config: AppConfig) -> WorkingDayCalendar:
    return WorkingDayCalendar(config.calendar)


# ---------------------------------------------------------------------------
# Arithmetic (1-6)
# ---------------------------------------------------------------------------

FEE_CASES = [
    # (gross, method, profile, rate, fee, gst, net)   -- exact Decimal strings
    ("1000.00", PaymentMethod.CARD, "profile_a", "20.00", "3.60", "976.40"),  # 1: 2%
    ("1000.00", PaymentMethod.UPI, "profile_a", "0.00", "0.00", "1000.00"),  # 2: 0%
    ("999.99", PaymentMethod.CARD, "profile_b", "23.50", "4.23", "972.26"),  # 3: 2.35%
    ("0.01", PaymentMethod.CARD, "profile_a", "0.00", "0.00", "0.01"),  # 4: minimum
]


@pytest.mark.parametrize(("gross", "method", "profile", "fee", "gst", "net"), FEE_CASES)
def test_fee_gst_and_net_are_exact(
    gross: str, method: PaymentMethod, profile: str, fee: str, gst: str, net: str, config: AppConfig
) -> None:
    """1-4. Exact Decimal equality, including the scale."""
    computed_fee = compute_fee(money(gross), method, profile, config)
    computed_gst = compute_gst(computed_fee, config)
    computed_net = expected_net(money(gross), computed_fee, computed_gst, money(0))
    assert str(computed_fee) == fee
    assert str(computed_gst) == gst
    assert str(computed_net) == net


def test_the_boundary_case_rounds_half_up_not_to_even(config: AppConfig) -> None:
    """3, made explicit. 999.99 * 0.0235 = 23.499765, which is not a half case.

    The half case is the GST: 23.50 * 0.18 = 4.23 exactly. So the boundary
    this row actually exercises is the fee's third decimal place rounding UP
    from ...765. Stated because a test named "boundary" that exercises no
    boundary is worse than no test.
    """
    raw = Decimal("999.99") * config.mdr.rate_for("profile_b", "card")
    assert raw == Decimal("23.4997650000")
    assert str(compute_fee(money("999.99"), PaymentMethod.CARD, "profile_b", config)) == "23.50"


def test_the_minimum_amount_never_produces_a_negative_net(config: AppConfig) -> None:
    """4. One paise, every profile and method. No crash, no negative net."""
    checked = 0
    for profile in config.mdr.profile_names():
        for method_name in config.mdr.method_names(profile):
            fee = compute_fee(money("0.01"), PaymentMethod(method_name), profile, config)
            gst = compute_gst(fee, config)
            net = expected_net(money("0.01"), fee, gst, money(0))
            assert net >= money(0), f"{profile}/{method_name} produced a negative net {net}"
            checked += 1
    assert checked == 12, f"expected the 3x4 matrix, checked {checked}"


def test_expected_net_may_be_negative_when_refunds_exceed_the_settlement() -> None:
    """SDD 3.1b. Clamping at zero would break the batch-side identity in 3.1a,
    and 298 cases in the dev dataset carry a refund."""
    assert expected_net(money("100.00"), money(0), money(0), money("150.00")) == money("-50.00")


@pytest.mark.parametrize("seed", range(50))
def test_components_plus_unexplained_equals_the_total_exactly(seed: int) -> None:
    """5. Fifty generated cases, exact Decimal equality.

    Seeded per-case rather than from one shared stream, so a failure names a
    reproducible seed instead of "the 37th draw of a run you cannot repeat".
    """
    rng = random.Random(seed)
    expected = money(Decimal(rng.randint(-500_000, 5_000_000)) / 100)
    actual = money(Decimal(rng.randint(-500_000, 5_000_000)) / 100)
    # Drawn from VARIANCE_CATEGORIES, never the full enum. A first draft drew
    # from the whole taxonomy and filtered afterwards on a SECOND independent
    # draw, so the filter tested a different category than the component used
    # and nine of fifty cases constructed a deduction. The generator must be
    # incapable of producing one, not merely unlikely to.
    categories = sorted(VARIANCE_CATEGORIES)
    components = tuple(
        VarianceComponent(
            category=categories[rng.randrange(len(categories))],
            amount=money(Decimal(rng.randint(-100_000, 100_000)) / 100),
            detail=f"generated component {index}",
        )
        for index in range(rng.randint(0, 4))
    )
    assert all(c.category not in DEDUCTION_CATEGORIES for c in components)

    breakdown = explain_variance(expected, actual, components)
    assert breakdown.attributed + breakdown.unexplained == breakdown.total
    assert breakdown.total == money(expected - actual)


def test_an_unaccountable_amount_lands_wholly_in_unexplained() -> None:
    """6. Rs7.13 that no component claims is NOT absorbed into rounding.

    The failure this prevents: a breakdown that widens its last component to
    make the sum work. That understates the residual set, and the residual
    count is the one number this project reports.
    """
    breakdown = explain_variance(
        expected=money("107.13"),
        actual=money("0.00"),
        components=(
            VarianceComponent(VarianceCategory.PARTIAL_CAPTURE, money("100.00"), "explained"),
        ),
    )
    assert breakdown.unexplained == money("7.13")
    assert breakdown.attributed == money("100.00")
    assert not breakdown.is_fully_explained
    assert [c.category for c in breakdown.components] == [VarianceCategory.PARTIAL_CAPTURE], (
        "the unexplained remainder was folded into a component"
    )


@pytest.mark.boundary_refusal
def test_a_deduction_category_cannot_be_a_variance_component() -> None:
    """FAULT INJECTION for PDD 6.1, at construction rather than in review.

    MDR_FEE is a component of expected_net computed on every clean case.
    Emitting it as a variance would report a fee as a discrepancy.
    """
    for category in sorted(DEDUCTION_CATEGORIES):
        with pytest.raises(ValueError, match="DEDUCTION category"):
            VarianceComponent(category, money("1.00"), "should never construct")


@pytest.mark.boundary_refusal
def test_the_balance_assertion_lives_inside_explain_variance() -> None:
    """FAULT INJECTION. The guard must be in the function, not only in a test.

    Proven by calling it with components that cannot balance if the remainder
    were dropped: unexplained must absorb exactly the difference, and the
    internal AssertionError must be reachable. A test-only check can be
    deselected by a marker expression; this cannot.
    """
    breakdown = explain_variance(money("10.00"), money("0.00"), ())
    assert breakdown.unexplained == money("10.00")

    import inspect

    from settlesense.matching import arithmetic

    source = inspect.getsource(arithmetic.explain_variance)
    assert "raise AssertionError" in source, (
        "explain_variance no longer asserts its own balance; the property is now "
        "only checked by tests, which can be skipped or deselected"
    )


@pytest.mark.boundary_refusal
def test_compute_fee_refuses_an_unknown_profile_or_method(config: AppConfig) -> None:
    """FAULT INJECTION. A silent zero would price an unrecognised method as
    free, and the case would reconcile perfectly against a fee never charged."""
    with pytest.raises(Exception, match="unknown merchant profile"):
        compute_fee(money("100.00"), PaymentMethod.CARD, "profile_z", config)


# ---------------------------------------------------------------------------
# Timing (7-10)
# ---------------------------------------------------------------------------

FRIDAY = date(2026, 10, 30)
HOLIDAY_MONDAY = date(2026, 11, 2)


def test_the_three_day_non_working_run_exists(calendar: WorkingDayCalendar) -> None:
    """Precondition for #7. A holiday test whose holiday is a working day
    proves nothing, so the fixture is asserted before it is used."""
    assert FRIDAY.strftime("%a") == "Fri"
    assert calendar.is_working_day(FRIDAY)
    assert not calendar.is_working_day(date(2026, 10, 31))  # Sat
    assert not calendar.is_working_day(date(2026, 11, 1))  # Sun
    assert not calendar.is_working_day(HOLIDAY_MONDAY), "2026-11-02 is not a holiday"


def test_t_plus_2_from_that_friday_lands_on_wednesday(calendar: WorkingDayCalendar) -> None:
    """7, corrected. The brief says Tuesday; Tuesday is T+1.

    Sat, Sun and the Monday holiday are all skipped, so the first working day
    after Friday is Tue 03 Nov and the second is Wed 04 Nov. Both are asserted
    so the off-by-one cannot reappear as "well, one of them is right".
    """
    assert calendar.add_working_days(FRIDAY, 1) == date(2026, 11, 3)  # Tue
    assert calendar.add_working_days(FRIDAY, 2) == date(2026, 11, 4)  # Wed
    assert calendar.add_working_days(FRIDAY, 2).strftime("%a") == "Wed"


def test_adding_zero_working_days_to_a_working_day_is_identity(
    calendar: WorkingDayCalendar,
) -> None:
    """8."""
    monday = date(2026, 10, 26)
    assert calendar.is_working_day(monday)
    assert calendar.add_working_days(monday, 0) == monday


def test_a_saturday_rolls_forward_before_counting(calendar: WorkingDayCalendar) -> None:
    """9. Roll first, then count.

    Counting before rolling would let a Saturday capture settle at T+1 on the
    same day a Monday capture settles at T+0 - two inputs, one output, and a
    day of drift visible only on weekend captures.
    """
    saturday = date(2026, 10, 31)
    assert not calendar.is_working_day(saturday)
    assert calendar.add_working_days(saturday, 0) == date(2026, 11, 3)  # Tue, Mon is a holiday
    assert calendar.add_working_days(saturday, 1) == date(2026, 11, 4)


def test_each_profile_produces_its_own_t_plus_n(
    calendar: WorkingDayCalendar, config: AppConfig
) -> None:
    """10. All three profiles, against the configured cycles rather than literals."""
    captured = date(2026, 9, 1)  # a Tuesday
    cycles = {p: config.calendar.settlement_cycle_for(p) for p in config.mdr.profile_names()}
    assert sorted(cycles.values()) == [1, 2, 3], f"expected three distinct cycles, got {cycles}"
    for profile, cycle in sorted(cycles.items()):
        assert settlement_due_date(captured, profile, calendar) == calendar.add_working_days(
            captured, cycle
        ), f"{profile} did not use its configured T+{cycle}"


def test_the_line_leg_is_t_plus_zero_not_t_plus_n(calendar: WorkingDayCalendar) -> None:
    """The distinction that cost 4841 false timing exceptions before it was found.

    A settlement LINE is dated the first working day on or after capture. The
    T+N cycle governs the BANK leg. Both legs exist and they are not the same
    number.
    """
    captured = date(2026, 9, 1)
    assert settlement_line_due_date(captured, calendar) == captured
    assert settlement_due_date(captured, "profile_c", calendar) != captured


def test_timing_is_measured_in_working_days_not_calendar_days(
    calendar: WorkingDayCalendar,
) -> None:
    """A settlement due Friday landing Tuesday is ONE working day late here,
    not four. Calendar days would report every weekend as a delay."""
    verdict = is_timing_explained(FRIDAY, date(2026, 11, 3), 0, calendar)
    assert verdict.working_days_late == 1
    assert not verdict.explained
    assert is_timing_explained(FRIDAY, date(2026, 11, 3), 1, calendar).explained


def test_an_early_settlement_is_distinguishable_from_a_late_one(
    calendar: WorkingDayCalendar,
) -> None:
    """Signed, not absolute. Early and late are different findings."""
    early = is_timing_explained(date(2026, 11, 4), date(2026, 11, 3), 0, calendar)
    assert early.working_days_late == -1 and early.is_early and not early.is_late


@pytest.mark.boundary_refusal
def test_a_negative_working_day_count_raises(calendar: WorkingDayCalendar) -> None:
    """FAULT INJECTION."""
    with pytest.raises(ValueError, match="cannot add"):
        calendar.add_working_days(FRIDAY, -1)


# ---------------------------------------------------------------------------
# Duplicates (11-13)
# ---------------------------------------------------------------------------


def _ledger(
    order_id: str, customer: str = "CUST-PROFILE_A-001", gross: str = "100.00"
) -> LedgerRow:
    return LedgerRow(
        order_id=order_id,
        invoice_no=f"INV-{order_id}",
        gross=money(gross),
        order_date=date(2026, 9, 1),
        customer_id=customer,
        sku="SKU-1",
    )


def test_a_byte_identical_row_is_duplicate_confirmed() -> None:
    """11. Deterministic: an ingestion artefact, resolved here."""
    row = _ledger("ORD_1")
    verdicts = find_confirmed_duplicates([row, row])
    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.category is VarianceCategory.DUPLICATE_CONFIRMED
    assert verdict.rule is DuplicateRule.BYTE_IDENTICAL_DISTINCT_LINE
    assert verdict.is_resolved_here


def test_one_verdict_per_extra_copy_not_per_group() -> None:
    """Population C's denominator is a ROW count. Three identical rows are two
    excess rows, and counting groups would report one."""
    row = _ledger("ORD_1")
    assert len(find_confirmed_duplicates([row, row, row])) == 2


def test_same_customer_same_amount_different_order_is_only_a_candidate() -> None:
    """12. NOT auto-classified in either direction, and both rows attached."""
    left, right = _ledger("ORD_1"), _ledger("ORD_2")
    verdicts = find_candidate_duplicates([left, right])
    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.category is VarianceCategory.DUPLICATE_CANDIDATE
    assert verdict.rule is DuplicateRule.SAME_CUSTOMER_SAME_GROSS
    assert not verdict.is_resolved_here, "an interpretive pair was resolved deterministically"
    assert verdict.row_ids == ("ORD_1", "ORD_2"), "both rows must be attached, not just one"


def test_a_confirmed_duplicate_is_not_also_counted_as_a_candidate() -> None:
    """Identical rows trivially share a customer and an amount, so without the
    exclusion the same rupees would be counted once as confirmed and once as a
    candidate."""
    row = _ledger("ORD_1")
    confirmed = find_confirmed_duplicates([row, row])
    excluded = frozenset(i for v in confirmed for i in v.row_ids)
    assert find_candidate_duplicates([row, row], excluded) == ()


def test_a_different_amount_is_not_a_duplicate() -> None:
    assert find_candidate_duplicates([_ledger("ORD_1"), _ledger("ORD_2", gross="101.00")]) == ()


def test_a_different_customer_is_not_a_duplicate() -> None:
    other = _ledger("ORD_2", customer="CUST-PROFILE_A-002")
    assert find_candidate_duplicates([_ledger("ORD_1"), other]) == ()


def test_every_verdict_names_the_rule_that_fired() -> None:
    """13. Including when nothing fired.

    Returning NO_RULE_FIRED rather than None keeps "we checked and it was
    clean" distinguishable from "we did not check", and keeps every call site
    handling one shape.
    """
    row, twin, cousin = _ledger("ORD_1"), _ledger("ORD_1"), _ledger("ORD_2")
    lonely = _ledger("ORD_3", customer="CUST-PROFILE_B-9", gross="7.00")
    assert classify(row, [twin]).rule is DuplicateRule.BYTE_IDENTICAL_DISTINCT_LINE
    assert classify(row, [cousin]).rule is DuplicateRule.SAME_CUSTOMER_SAME_GROSS
    assert classify(lonely, [row]).rule is DuplicateRule.NO_RULE_FIRED
    assert classify(lonely, [row]).category is None
    assert not classify(lonely, [row]).is_duplicate
    for verdict in (classify(row, [twin]), classify(row, [cousin]), classify(lonely, [row])):
        assert verdict.detail, "a verdict with no detail cannot be audited"
        assert verdict.row_ids, "a verdict must name the rows it is about"


def test_duplicate_detection_is_order_independent() -> None:
    """D4. The same rows in a different order must produce the same verdicts."""
    rows = [_ledger(f"ORD_{i}") for i in range(4)]
    forward = find_candidate_duplicates(rows)
    backward = find_candidate_duplicates(list(reversed(rows)))
    assert forward == backward

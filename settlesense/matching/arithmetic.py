"""M3 - fee and GST computation, and variance attribution that must balance.

The rule this module exists to enforce: EVERY RUPEE IS ACCOUNTED FOR. A
variance is either attributed to a taxonomy category or it sits in an explicit
unexplained bucket. There is no third place for it to go, and the sum is
asserted inside `explain_variance` rather than only in a test - a test can be
deselected, and an unbalanced attribution silently understates the residual,
which is the one number this project is measured on.

MDR_FEE, GST_ON_FEE and REFUND_OFFSET are DEDUCTION categories (PDD 6.1). They
are components of expected_net, computed on every clean case, and are never
emitted as variances. They appear here as named components of the expected
figure, never as an explanation of a discrepancy.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from settlesense.config import AppConfig
from settlesense.exceptions.taxonomy import DEDUCTION_CATEGORIES, VarianceCategory
from settlesense.types import Money, PaymentMethod, money

__all__ = [
    "VarianceBreakdown",
    "VarianceComponent",
    "compute_fee",
    "compute_gst",
    "expected_net",
    "explain_variance",
]

ZERO: Money = money(0)


def compute_fee(gross: Money, method: PaymentMethod, profile: str, config: AppConfig) -> Money:
    """MDR fee = gross * rate(profile, method), ROUND_HALF_UP to paise.

    An unknown profile or method RAISES rather than defaulting to zero. A
    silent zero would price an unrecognised method as free, and the case would
    reconcile perfectly against a fee that was never charged.
    """
    if gross < ZERO:
        raise ValueError(f"compute_fee expects a non-negative gross, got {gross}")
    rate = config.mdr.rate_for(profile, str(method))
    return money(gross * rate)


def compute_gst(fee: Money, config: AppConfig) -> Money:
    """GST = fee * gst_rate, ROUND_HALF_UP to paise.

    Computed on the ROUNDED fee, not on the unrounded product. The generator
    does the same, and the two must agree exactly: rounding twice and rounding
    once differ by a paise often enough to move a case across the P9 tolerance.
    """
    if fee < ZERO:
        raise ValueError(f"compute_gst expects a non-negative fee, got {fee}")
    return money(fee * config.mdr.gst_rate)


def expected_net(gross: Money, fee: Money, tax: Money, refunds: Money) -> Money:
    """SDD 3.1b: expected_net = gross - fee - tax - refunds. One definition.

    MAY BE NEGATIVE, and that is not an error. A payment refunded for more than
    it settled leaves the batch owing money; 298 cases in the dev dataset carry
    a refund and some land negative. Clamping at zero here would break the
    batch-side conservation identity in SDD 3.1a.
    """
    return money(gross - fee - tax - refunds)


@dataclass(frozen=True)
class VarianceComponent:
    """One attributed slice of a variance."""

    category: VarianceCategory
    amount: Money
    detail: str

    def __post_init__(self) -> None:
        if self.category in DEDUCTION_CATEGORIES:
            raise ValueError(
                f"{self.category} is a DEDUCTION category (PDD 6.1). It is a "
                "component of expected_net, computed on every case, and can never "
                "be an explanation of a variance. Emitting it here would report a "
                "fee as though it were a discrepancy."
            )


@dataclass(frozen=True)
class VarianceBreakdown:
    """A variance, fully attributed. components + unexplained == total, exactly."""

    total: Money
    components: tuple[VarianceComponent, ...]
    unexplained: Money

    @property
    def attributed(self) -> Money:
        return money(sum((component.amount for component in self.components), ZERO))

    @property
    def is_fully_explained(self) -> bool:
        return self.unexplained == ZERO

    def categories(self) -> tuple[VarianceCategory, ...]:
        """Attributed categories, sorted (D4)."""
        return tuple(sorted({component.category for component in self.components}))


def explain_variance(
    expected: Money,
    actual: Money,
    components: tuple[VarianceComponent, ...],
) -> VarianceBreakdown:
    """Attribute `expected - actual` across `components`, with the remainder explicit.

    THE ASSERTION IS INSIDE THIS FUNCTION, not only in a test. An attribution
    that does not sum to the total understates the residual set, and the
    residual count is the single number this project reports. A test asserting
    the same property can be deselected by a marker expression, skipped, or
    simply not run; this cannot.

    The remainder is named `unexplained` and is a real bucket, not a rounding
    slop. A component list that explains ~7 rupees of an 8-rupee variance
    leaves 1 rupee unexplained - it does not quietly widen the last component.
    """
    total = money(expected - actual)
    attributed = money(sum((component.amount for component in components), ZERO))
    unexplained = money(total - attributed)

    balanced = money(attributed + unexplained)
    if balanced != total:
        raise AssertionError(
            f"variance attribution does not balance: attributed {attributed} + "
            f"unexplained {unexplained} = {balanced}, but the total variance is "
            f"{total}. Every rupee must land in a category or in the unexplained "
            "bucket; there is no third place for it to go."
        )

    return VarianceBreakdown(
        total=total,
        components=tuple(
            sorted(components, key=lambda component: (component.category, component.detail))
        ),
        unexplained=unexplained,
    )


def deduction_reference(gross: Money, net: Money) -> Money:
    """SDD 3.1b: deductions_reference = expected_gross - expected_net.

    Covers fees, GST and refunds together. Named rather than inlined because
    it is a REFERENCE figure for reporting, never a variance - the distinction
    that PDD 6.1 exists to protect.
    """
    return money(gross - net)


def as_decimal_sum(values: tuple[Money, ...]) -> Money:
    """Sum with an explicit Decimal zero, quantized once at the end.

    `sum()` with the default start value of int 0 works, but seeds the
    accumulation with a type that is not Money. Being explicit keeps every
    money total inside the Decimal domain from the first term (D1).
    """
    return money(sum(values, Decimal(0)))

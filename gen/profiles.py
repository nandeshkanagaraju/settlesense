"""M1 - the three merchant profiles.

INDEPENDENT PATH. This module does not import from settlesense/, and the values
below are defined here rather than read from config/mdr_rates.yaml on purpose.

The duplication is the point. If the generator and the engine read their rates
through the same loader, a bug in that loader writes the same wrong number the
engine reads back, the fee arithmetic reconciles perfectly, and the metric is
measuring nothing. Two independent statements of the same fact can disagree;
one shared statement cannot. A test outside both trees compares them.

Rates here MUST mirror config/mdr_rates.yaml:

    profile   card     upi      netbanking  wallet   GST    cycle
    a         2.00%    0.00%    1.75%       2.10%    18%    T+2
    b         2.35%    0.00%    1.90%       2.00%    18%    T+1
    c         1.80%    0.00%    1.60%       1.95%    18%    T+3

Profiles differ in MDR, settlement cycle, refund rate, method mix and ticket
size, so results read as generalising rather than tuned to one distribution
(PDD 8.5).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Final

__all__ = [
    "GST_RATE",
    "PROFILES",
    "PROFILE_NAMES",
    "MerchantProfile",
    "profile_by_name",
]

# GST on the MDR fee. 18% for every profile.
GST_RATE: Final[Decimal] = Decimal("0.18")

# The closed set of payment methods. Mirrors SettlementLine.method in the
# engine's domain model, stated independently here.
METHODS: Final[tuple[str, ...]] = ("card", "netbanking", "upi", "wallet")


@dataclass(frozen=True)
class MerchantProfile:
    """One simulated merchant. All money-shaped values are Decimal (D1)."""

    name: str
    merchant_name: str  # as printed in a bank narration
    mdr_rates: Mapping[str, Decimal]
    gst_rate: Decimal
    settlement_cycle_days: int  # T+N, in WORKING days
    refund_rate: Decimal  # fraction of payments that get refunded
    partial_refund_share: Decimal  # of refunds, the fraction that are partial
    method_weights: Mapping[str, int]  # sampling weights, integers
    min_gross_paise: int
    max_gross_paise: int
    customer_pool_size: int
    sku_pool: tuple[str, ...]

    def rate_for(self, method: str) -> Decimal:
        """MDR rate as a fraction of gross. An unknown method raises."""
        if method not in self.mdr_rates:
            raise KeyError(
                f"profile {self.name!r} has no MDR rate for method {method!r}; "
                f"known methods: {self.methods()}"
            )
        return self.mdr_rates[method]

    def methods(self) -> tuple[str, ...]:
        """Method names, explicitly sorted (D4)."""
        return tuple(sorted(self.mdr_rates))

    def weighted_methods(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """(names, weights) in a stable sorted order, for deterministic sampling."""
        names = tuple(sorted(self.method_weights))
        weights = tuple(self.method_weights[name] for name in names)
        return names, weights


def _rates(card: str, netbanking: str, upi: str, wallet: str) -> Mapping[str, Decimal]:
    return MappingProxyType(
        {
            "card": Decimal(card),
            "netbanking": Decimal(netbanking),
            "upi": Decimal(upi),
            "wallet": Decimal(wallet),
        }
    )


PROFILE_A: Final[MerchantProfile] = MerchantProfile(
    name="profile_a",
    merchant_name="AURORA RETAIL",
    mdr_rates=_rates(card="0.0200", netbanking="0.0175", upi="0.0000", wallet="0.0210"),
    gst_rate=GST_RATE,
    settlement_cycle_days=2,  # T+2
    refund_rate=Decimal("0.06"),
    partial_refund_share=Decimal("0.40"),
    method_weights=MappingProxyType({"card": 45, "netbanking": 12, "upi": 35, "wallet": 8}),
    min_gross_paise=10_000,  # Rs 100.00
    max_gross_paise=2_500_000,  # Rs 25,000.00
    customer_pool_size=900,
    sku_pool=("SKU-AR-KITCHEN", "SKU-AR-DECOR", "SKU-AR-LINEN", "SKU-AR-TABLEWARE"),
)

PROFILE_B: Final[MerchantProfile] = MerchantProfile(
    name="profile_b",
    merchant_name="BLUEPEAK FOODS",
    mdr_rates=_rates(card="0.0235", netbanking="0.0190", upi="0.0000", wallet="0.0200"),
    gst_rate=GST_RATE,
    settlement_cycle_days=1,  # T+1
    refund_rate=Decimal("0.03"),
    partial_refund_share=Decimal("0.25"),
    method_weights=MappingProxyType({"card": 25, "netbanking": 8, "upi": 60, "wallet": 7}),
    min_gross_paise=25_000,  # Rs 250.00
    max_gross_paise=800_000,  # Rs 8,000.00
    customer_pool_size=1_600,
    sku_pool=("SKU-BP-MEAL", "SKU-BP-BEVERAGE", "SKU-BP-DESSERT"),
)

PROFILE_C: Final[MerchantProfile] = MerchantProfile(
    name="profile_c",
    merchant_name="CARBON WORKS PVT LTD",
    mdr_rates=_rates(card="0.0180", netbanking="0.0160", upi="0.0000", wallet="0.0195"),
    gst_rate=GST_RATE,
    settlement_cycle_days=3,  # T+3
    refund_rate=Decimal("0.09"),
    partial_refund_share=Decimal("0.55"),
    method_weights=MappingProxyType({"card": 55, "netbanking": 20, "upi": 20, "wallet": 5}),
    min_gross_paise=50_000,  # Rs 500.00
    max_gross_paise=6_000_000,  # Rs 60,000.00
    customer_pool_size=400,
    sku_pool=("SKU-CW-PANEL", "SKU-CW-INVERTER", "SKU-CW-MOUNT", "SKU-CW-CABLE"),
)

PROFILES: Final[tuple[MerchantProfile, ...]] = (PROFILE_A, PROFILE_B, PROFILE_C)
PROFILE_NAMES: Final[tuple[str, ...]] = tuple(profile.name for profile in PROFILES)

_BY_NAME: Final[Mapping[str, MerchantProfile]] = MappingProxyType(
    {profile.name: profile for profile in PROFILES}
)


def profile_by_name(name: str) -> MerchantProfile:
    """Look up a profile. An unknown name raises rather than defaulting."""
    if name not in _BY_NAME:
        raise KeyError(f"unknown merchant profile {name!r}; known: {PROFILE_NAMES}")
    return _BY_NAME[name]


# Structural self-checks. These run at import so a bad edit cannot sit unnoticed
# until a 5,000-record run has already been written to disk.
for _profile in PROFILES:
    if tuple(sorted(_profile.mdr_rates)) != METHODS:
        raise AssertionError(f"{_profile.name}: MDR table must price exactly {METHODS}")
    if tuple(sorted(_profile.method_weights)) != METHODS:
        raise AssertionError(f"{_profile.name}: method weights must cover exactly {METHODS}")
    if _profile.mdr_rates["upi"] != Decimal("0.0000"):
        raise AssertionError(f"{_profile.name}: UPI must be zero-rated")
    if _profile.settlement_cycle_days < 1:
        raise AssertionError(f"{_profile.name}: settlement cycle must be at least T+1")
    if _profile.min_gross_paise >= _profile.max_gross_paise:
        raise AssertionError(f"{_profile.name}: gross range is empty")
    if not _profile.sku_pool:
        raise AssertionError(f"{_profile.name}: SKU pool is empty")
del _profile

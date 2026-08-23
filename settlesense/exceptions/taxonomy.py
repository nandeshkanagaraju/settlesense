"""The closed variance taxonomy (PDD 6).

THIRTEEN members, split into two kinds that must never be conflated:

  DEDUCTION categories describe *components of `expected_net`*. MDR_FEE,
  GST_ON_FEE and REFUND_OFFSET are computed on EVERY case - a clean card
  payment deducts a fee and GST and has no variance at all. They are never
  emitted as a variance.

  VARIANCE categories describe an unexplained difference the engine must
  account for. These, and only these, are what a coverage assertion sweeps.

PDD 6.1 states the consequence directly: a coverage assertion of the form
"every taxonomy category appears in truth" is WRONG, because it demands that
the generator invent a variance out of a fee. Coverage is asserted over
`VARIANCE_CATEGORIES` only.

The two sets are ENUMERATED EXPLICITLY, not derived by subtracting a hardcoded
deduction list from the enum. Subtraction makes the membership of one set an
accident of the other: adding a category and forgetting to classify it would
silently enrol it in whichever set was defined by subtraction, and a coverage
test would then start demanding - or stop demanding - it with no edit anywhere
saying so. `PARTITION_IS_COMPLETE` closes the loop by asserting at import time
that every member is classified exactly once.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = [
    "DEDUCTION_CATEGORIES",
    "VARIANCE_CATEGORIES",
    "VarianceCategory",
]


class VarianceCategory(StrEnum):
    """The closed taxonomy. Adding a member here requires classifying it below."""

    # --- PDD 6.1, deterministically derivable - never sent to the model ---
    MDR_FEE = "MDR_FEE"
    GST_ON_FEE = "GST_ON_FEE"
    ROUNDING_DIFFERENCE = "ROUNDING_DIFFERENCE"
    DUPLICATE_CONFIRMED = "DUPLICATE_CONFIRMED"
    T_PLUS_N_TIMING = "T_PLUS_N_TIMING"
    REFUND_OFFSET = "REFUND_OFFSET"
    PARTIAL_CAPTURE = "PARTIAL_CAPTURE"
    # --- PDD 6.2, genuinely interpretive - eligible for the AI layer ---
    UTR_TRUNCATED_MAPPING = "UTR_TRUNCATED_MAPPING"
    UTR_MISSING_MAPPING = "UTR_MISSING_MAPPING"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    SPLIT_SETTLEMENT = "SPLIT_SETTLEMENT"
    MISSING_VS_LATE_CREDIT = "MISSING_VS_LATE_CREDIT"
    UNEXPLAINED = "UNEXPLAINED"


DEDUCTION_CATEGORIES: Final[frozenset[VarianceCategory]] = frozenset(
    {
        VarianceCategory.MDR_FEE,
        VarianceCategory.GST_ON_FEE,
        VarianceCategory.REFUND_OFFSET,
    }
)
"""Components of `expected_net` (PDD 6.1). Computed on every case, never a variance."""


VARIANCE_CATEGORIES: Final[frozenset[VarianceCategory]] = frozenset(
    {
        VarianceCategory.ROUNDING_DIFFERENCE,
        VarianceCategory.DUPLICATE_CONFIRMED,
        VarianceCategory.T_PLUS_N_TIMING,
        VarianceCategory.PARTIAL_CAPTURE,
        VarianceCategory.UTR_TRUNCATED_MAPPING,
        VarianceCategory.UTR_MISSING_MAPPING,
        VarianceCategory.DUPLICATE_CANDIDATE,
        VarianceCategory.SPLIT_SETTLEMENT,
        VarianceCategory.MISSING_VS_LATE_CREDIT,
        VarianceCategory.UNEXPLAINED,
    }
)
"""The only categories a coverage assertion may sweep (PDD 6.1).

`CaseOutcome.category` and `RowVarianceOutcome.category` draw from this set;
SDD 3.1 says so in the field comment.
"""


# Import-time partition check. Not a test, deliberately: a test can be skipped,
# deselected by a marker expression, or simply not run, and this invariant has
# to hold for anything that imports the module at all. Adding a taxonomy member
# without classifying it fails here, at the point of the mistake.
_classified = DEDUCTION_CATEGORIES | VARIANCE_CATEGORIES
_unclassified = set(VarianceCategory) - _classified
_double = DEDUCTION_CATEGORIES & VARIANCE_CATEGORIES
if _unclassified:
    raise AssertionError(
        f"taxonomy member(s) {sorted(c.value for c in _unclassified)} are in neither "
        "DEDUCTION_CATEGORIES nor VARIANCE_CATEGORIES. Classify explicitly: a "
        "member that is in neither set is invisible to coverage assertions, and "
        "one that is in both is counted twice."
    )
if _double:
    raise AssertionError(
        f"taxonomy member(s) {sorted(c.value for c in _double)} are classified as "
        "BOTH a deduction and a variance. A component of expected_net cannot also "
        "be an unexplained difference."
    )

PARTITION_IS_COMPLETE: Final[bool] = True
"""Set only if the checks above passed. Importing this module proves the split."""

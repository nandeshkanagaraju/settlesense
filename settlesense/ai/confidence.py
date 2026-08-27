"""M7 - confidence, computed from verification outcomes (SDD 4.6, PDD 7.2).

CONFIDENCE IS NEVER THE MODEL'S SELF-REPORT. Every term below is a fact the
verifier established or a property of the run. If the model emits a confidence
field it is dropped at parse time and never reaches `Hypothesis`, so no code
here can read it even by accident - `test_ai.py` asserts that the string
"confidence" does not appear in any parsed hypothesis's fields.

CONFIDENCE ALONE CAN NEVER CONFIRM (SDD 4.6). Auto-confirm requires
`verification_passed` AND `confidence >= auto_confirm`. The first term of the
formula carries weight 0.40, so a rejected hypothesis cannot reach 0.80 even
with every other signal perfect - but that arithmetic is a coincidence of the
weights, not the rule. `should_auto_confirm` checks the flag explicitly, so
re-weighting the formula can never silently enable confirmation-by-score.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from settlesense.ai.verifier import VerificationResult
from settlesense.config import AppConfig

__all__ = ["ConfidenceBreakdown", "compute_confidence", "should_auto_confirm"]

_ONE = Decimal("1")
_ZERO = Decimal("0")
_QUANTUM = Decimal("0.0001")
"""Four places. Confidence is a RATIO, not money - never quantized to paise."""


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """The score AND its terms.

    The breakdown is carried rather than just the total because a bare 0.72
    tells a reviewer nothing about which signal was weak, and the M8 queue has
    to show why a case was not auto-confirmed.
    """

    score: Decimal
    verification_passed: Decimal
    residual_within_tolerance: Decimal
    evidence_completeness: Decimal
    candidate_separation: Decimal
    freshness_ok: Decimal

    def terms(self) -> tuple[tuple[str, Decimal], ...]:
        """Sorted by name, so two runs render identically (D4)."""
        return tuple(
            sorted(
                (
                    ("candidate_separation", self.candidate_separation),
                    ("evidence_completeness", self.evidence_completeness),
                    ("freshness_ok", self.freshness_ok),
                    ("residual_within_tolerance", self.residual_within_tolerance),
                    ("verification_passed", self.verification_passed),
                )
            )
        )


def _clip(value: Decimal) -> Decimal:
    """Clip to [0, 1]. SDD 4.6 says so for separation; applied to every term.

    A term outside [0,1] would let one signal dominate the weighted sum and
    push the total past 1.0, which then reads as a probability nobody computed.
    """
    return max(_ZERO, min(_ONE, value))


def compute_confidence(
    result: VerificationResult, config: AppConfig, freshness_ok: bool = True
) -> ConfidenceBreakdown:
    """The SDD 4.6 weighted sum, with weights from config (D12).

    `freshness_ok` is a PARAMETER, not a clock read: it means "the expected
    files have arrived per the watermark", which the caller knows and this
    module must not go looking for (D2).
    """
    weights = config.thresholds.confidence

    passed = _ONE if result.passed else _ZERO
    within = (
        _ONE
        if result.computed_residual is not None
        and abs(result.computed_residual) <= config.thresholds.tolerance.verifier_rupees
        else _ZERO
    )
    completeness = _clip(result.evidence_completeness)
    separation = _clip(result.candidate_separation)
    fresh = _ONE if freshness_ok else _ZERO

    score = (
        weights.weight_verification_passed * passed
        + weights.weight_residual_within_tolerance * within
        + weights.weight_evidence_completeness * completeness
        + weights.weight_candidate_separation * separation
        + weights.weight_freshness * fresh
    )
    return ConfidenceBreakdown(
        score=score.quantize(_QUANTUM, rounding=ROUND_HALF_UP),
        verification_passed=passed,
        residual_within_tolerance=within,
        evidence_completeness=completeness,
        candidate_separation=separation,
        freshness_ok=fresh,
    )


def should_auto_confirm(
    result: VerificationResult, breakdown: ConfidenceBreakdown, config: AppConfig
) -> bool:
    """BOTH conditions, checked independently (SDD 4.6).

    `result.passed` is read here rather than inferred from the score. With the
    shipped weights a failed verification cannot reach 0.80 anyway, but that is
    arithmetic about today's weights - re-weighting the formula must not be
    able to silently enable confirmation on score alone.
    """
    return bool(result.passed) and breakdown.score >= config.thresholds.confidence.auto_confirm

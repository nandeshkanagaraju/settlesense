"""M7 - the verified hypothesis loop: residual -> generate -> verify -> decide.

THE ORDER IS THE ARCHITECTURE. Rules first; only what they could not close
reaches a model; every model claim is checked by a verifier that does not
consult the model; anything unverified ABSTAINS into the M8 queue.

FIRST PASS WINS (SDD 4.4). Hypotheses are verified in the model's rank order
and the first one that verifies is taken. Not the highest-confidence one:
confidence is computed FROM verification, so ranking by it would mean scoring
every hypothesis before deciding, and a second passing hypothesis is a sign of
ambiguity rather than of a better answer.

ABSTENTION IS A RESULT. `AbstainReason` names why, because "the AI resolved
nothing" and "the evidence could not distinguish the candidates" are different
findings with different fixes, and only the second is a statement about the
dataset.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from settlesense.ai.client import FixtureMissError, LLMClient, ModelUnavailable
from settlesense.ai.confidence import ConfidenceBreakdown, compute_confidence, should_auto_confirm
from settlesense.ai.hypothesis import Hypothesis, eligible_exceptions, generate
from settlesense.ai.verifier import VerificationResult, verify
from settlesense.config import AppConfig
from settlesense.ingest import DayDataset
from settlesense.types import Exception_

__all__ = ["AbstainReason", "LoopOutcome", "LoopReport", "resolve_exception", "run_loop"]


class AbstainReason(StrEnum):
    """Why nothing was confirmed. Distinctions that matter downstream."""

    NO_HYPOTHESIS = "NO_HYPOTHESIS"
    """The model returned nothing schema-valid after its retries."""

    ALL_REJECTED = "ALL_REJECTED"
    """Hypotheses arrived; the verifier could not confirm any of them."""

    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    """A hypothesis verified but scored under auto_confirm - a human decides."""

    FIXTURE_MISS = "FIXTURE_MISS"
    """No recorded response. Never a network fallback; never a crash."""


@dataclass(frozen=True)
class LoopOutcome:
    """What happened to one exception. Confirmed or abstained, never CLOSED."""

    exception_id: str
    confirmed: bool
    hypothesis: Hypothesis | None
    verification: VerificationResult | None
    confidence: ConfidenceBreakdown | None
    abstain_reason: AbstainReason | None
    hypotheses_seen: int
    rejections: tuple[str, ...]
    unavailable: bool = False
    """The model was DOWN. Not an abstention, and not counted as one (M10).

    An abstention is a RESULT: hypotheses arrived, the verifier looked, and
    nothing survived. That is a statement about the case and about the
    evidence. An outage is a statement about the world - the case was never
    examined at all - and the exception goes back in the queue for the next
    run rather than into the human review pile.

    Kept as a separate flag rather than a fifth AbstainReason because
    `LoopReport.abstention_rate` divides by `sent`, and folding outages in
    would let a provider incident inflate a number this project publishes as a
    property of the verifier.
    """

    @property
    def resolution_type(self) -> str:
        if self.unavailable:
            return "PENDING_AI_UNAVAILABLE"
        return "AI_VERIFIED" if self.confirmed else "ABSTAINED"


@dataclass(frozen=True)
class LoopReport:
    """The numbers this module is judged on. Sorted, counted, never averaged."""

    outcomes: tuple[LoopOutcome, ...]
    sent: int
    confirmed: int
    abstained: int
    reasons: tuple[tuple[str, int], ...]
    unavailable: int = 0
    """Cases the model was down for. NEVER added to `abstained` (M10, D11).

    Three outcomes, three counts: confirmed + abstained + unavailable == sent.
    Two of those are decisions and one is a service failure, and a report that
    summed them would say the loop "handled" every case it was handed.
    """

    @property
    def abstention_rate(self) -> str:
        """Abstentions over cases the model actually ANSWERED.

        The denominator excludes outages. A run where the provider was down for
        half the cases has not abstained on them - nothing looked at them - and
        dividing by `sent` would report a verifier that got stricter when in
        fact a service went away.
        """
        examined = self.sent - self.unavailable
        if not examined:
            return "n/a"
        return f"{self.abstained / examined:.4f}"


def resolve_exception(
    exception: Exception_,
    dataset: DayDataset,
    config: AppConfig,
    client: LLMClient,
    freshness_ok: bool = True,
) -> LoopOutcome:
    """One exception, end to end. NEVER raises on a model or fixture problem.

    A fixture miss is an abstention, not a crash: the run continues and reports
    that this case had no recording, which is a fact about the fixture set
    rather than about the case.
    """
    try:
        hypotheses = generate(exception, dataset, config, client)
    except ModelUnavailable:
        # THE CASE IS UNTOUCHED. No hypothesis, no verification, no confidence
        # and NO abstain_reason - an abstention is a decision, and no decision
        # was made here. The orchestrator reads `unavailable` and puts it back
        # in the queue; the next run picks it up exactly as it was.
        return LoopOutcome(
            exception_id=exception.exception_id,
            confirmed=False,
            hypothesis=None,
            verification=None,
            confidence=None,
            abstain_reason=None,
            hypotheses_seen=0,
            rejections=(),
            unavailable=True,
        )
    except FixtureMissError:
        return LoopOutcome(
            exception_id=exception.exception_id,
            confirmed=False,
            hypothesis=None,
            verification=None,
            confidence=None,
            abstain_reason=AbstainReason.FIXTURE_MISS,
            hypotheses_seen=0,
            rejections=(),
        )

    if not hypotheses:
        return LoopOutcome(
            exception_id=exception.exception_id,
            confirmed=False,
            hypothesis=None,
            verification=None,
            confidence=None,
            abstain_reason=AbstainReason.NO_HYPOTHESIS,
            hypotheses_seen=0,
            rejections=(),
        )

    rejections: list[str] = []
    for hypothesis in sorted(hypotheses, key=lambda h: h.rank):
        result = verify(hypothesis, dataset, config)
        if not result.passed:
            rejections.append(result.failure_reason)
            continue
        breakdown = compute_confidence(result, config, freshness_ok)
        if should_auto_confirm(result, breakdown, config):
            return LoopOutcome(
                exception_id=exception.exception_id,
                confirmed=True,
                hypothesis=hypothesis,
                verification=result,
                confidence=breakdown,
                abstain_reason=None,
                hypotheses_seen=len(hypotheses),
                rejections=tuple(rejections),
            )
        # VERIFIED BUT UNDER THRESHOLD is not a rejection of the claim - it is
        # a refusal to act on it automatically. Reported separately so the two
        # are never added together.
        return LoopOutcome(
            exception_id=exception.exception_id,
            confirmed=False,
            hypothesis=hypothesis,
            verification=result,
            confidence=breakdown,
            abstain_reason=AbstainReason.BELOW_THRESHOLD,
            hypotheses_seen=len(hypotheses),
            rejections=tuple(rejections),
        )

    return LoopOutcome(
        exception_id=exception.exception_id,
        confirmed=False,
        hypothesis=None,
        verification=None,
        confidence=None,
        abstain_reason=AbstainReason.ALL_REJECTED,
        hypotheses_seen=len(hypotheses),
        rejections=tuple(rejections),
    )


def run_loop(
    exceptions: tuple[Exception_, ...],
    dataset: DayDataset,
    config: AppConfig,
    client: LLMClient,
    sendable: frozenset[str] | None = None,
    freshness_ok: bool = True,
) -> LoopReport:
    """Every eligible exception, in a deterministic order.

    `sendable` is REQUIRED to be named by the caller when narrowing: passing
    None means PDD 6.2's full interpretive set. The wiring for this dataset
    passes DUPLICATE_CANDIDATE alone, so Population B's unlinked batches cannot
    reach the model - a batch whose credit never arrived is missing data, and
    asking a model about an absence invites a fabricated explanation.
    """
    from settlesense.ai.hypothesis import AI_ELIGIBLE_CATEGORIES

    eligible = eligible_exceptions(
        exceptions, AI_ELIGIBLE_CATEGORIES if sendable is None else sendable
    )
    outcomes = tuple(
        resolve_exception(exception, dataset, config, client, freshness_ok)
        for exception in eligible
    )
    reasons = Counter(
        outcome.abstain_reason.value for outcome in outcomes if outcome.abstain_reason is not None
    )
    confirmed = sum(1 for outcome in outcomes if outcome.confirmed)
    unavailable = sum(1 for outcome in outcomes if outcome.unavailable)
    # ABSTAINED IS COUNTED, NOT DERIVED BY SUBTRACTION. `sent - confirmed` was
    # correct while there were two outcomes and silently absorbs the third: an
    # outage would have been reported as an abstention with no line of this
    # file changing. Counting each and asserting they add up is the same rule
    # the population metrics follow.
    abstained = sum(1 for outcome in outcomes if not outcome.confirmed and not outcome.unavailable)
    assert confirmed + abstained + unavailable == len(eligible), (
        f"{confirmed} + {abstained} + {unavailable} != {len(eligible)} sent"
    )
    return LoopReport(
        outcomes=outcomes,
        sent=len(eligible),
        confirmed=confirmed,
        abstained=abstained,
        reasons=tuple(sorted(reasons.items())),
        unavailable=unavailable,
    )

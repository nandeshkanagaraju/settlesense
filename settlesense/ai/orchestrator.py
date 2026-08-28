"""M10 - the AI stage against the store, and what happens when the model is down.

WHAT THIS ADDS TO M7. `run_loop` decides; it does not persist. This applies
those decisions to the exception store, which is where the difference between
an abstention and an outage stops being a field on a dataclass and becomes a
status somebody has to act on.

THE DETERMINISTIC RESULT IS NOT TOUCHED, AND THAT IS CHECKABLE. Nothing here
runs the matching engine, re-decides a rule outcome, or writes to an exception
the rules already confirmed. It reads the RESIDUAL set - what the rules could
not close - and writes only to rows in it. `test_m10_degradation.py` serializes
the deterministic rows before and after an outage run and compares BYTES, which
is the strongest form of the graceful-degradation claim and the only one that
cannot be satisfied by a summary that happens to match.

DURING AN OUTAGE, ZERO CONFIRMATIONS. Not few. The model produced no answer, so
there is nothing to verify and nothing to confirm - and because verification
happens locally, a partially-arrived answer cannot slip through either: the
loop returns `unavailable` before a hypothesis exists.

PENDING_AI_UNAVAILABLE IS NOT PENDING_EVIDENCE. One says a service failed; the
other says a file has not arrived. They carry different reasons here, render in
different colours (M8), and lead to different actions - go and look at the
model, versus wait for tomorrow's statement. The queue collapsed them once
already, by colour, and the fix is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from settlesense.ai.client import LLMClient
from settlesense.ai.hypothesis import AI_ELIGIBLE_CATEGORIES
from settlesense.ai.loop import LoopReport, run_loop
from settlesense.config import AppConfig
from settlesense.exceptions.store import RESIDUAL_STATES, ExceptionStore
from settlesense.ingest import DayDataset
from settlesense.types import AuditActor, ExceptionStatus, ResolutionSource

__all__ = [
    "OUTAGE_REASON",
    "PENDING_EVIDENCE_REASON",
    "AiStageResult",
    "run_ai_stage",
]

OUTAGE_REASON = "model unavailable after retries; nothing was examined, retry next run"
"""Why a row is PENDING_AI_UNAVAILABLE. Distinct text, deliberately.

An operator reading the queue has to be able to tell, from the row alone,
whether to wait or to go and look at a service. Sharing a phrase with the
evidence-waiting reason would put that distinction entirely in a colour - which
is where it lived until 2026-08-28, and where it was wrong.
"""

PENDING_EVIDENCE_REASON = "settlement credit not yet due; waiting for a later file"
"""The counterpart, quoted here so the two can be compared in one place."""


@dataclass(frozen=True)
class AiStageResult:
    """What the AI stage did. Three disjoint counts and the rows they name."""

    report: LoopReport
    confirmed: tuple[str, ...]
    pending_unavailable: tuple[str, ...]
    abstained: tuple[str, ...]

    @property
    def outage(self) -> bool:
        """Did the model fail on ANY case this run?

        Any, not all. A run where 3 of 10 calls failed is a degraded run, and
        reporting it as healthy because 7 succeeded would hide the 3.
        """
        return bool(self.pending_unavailable)


def run_ai_stage(
    store: ExceptionStore,
    dataset: DayDataset,
    config: AppConfig,
    client: LLMClient,
    arrival_day: int,
    sendable: frozenset[str] | None = None,
) -> AiStageResult:
    """Run the hypothesis loop over the residual set and persist the outcomes.

    `sendable` is passed through to `run_loop`, which requires the caller to
    name the narrowing rather than defaulting to everything.

    ORDER MATTERS AND IS NOT ALPHABETICAL. Rows are processed in the queue's
    own order - `get_queue` sorts by (-amount, exception_id) - so a partial
    outage fails on the same cases every run and the demo is reproducible.
    """
    residual = store.get_queue(status_filter=RESIDUAL_STATES)
    report = run_loop(
        tuple(residual),
        dataset,
        config,
        client,
        sendable=AI_ELIGIBLE_CATEGORIES if sendable is None else sendable,
    )

    by_id = {exception.exception_id: exception for exception in residual}
    confirmed: list[str] = []
    pending: list[str] = []
    abstained: list[str] = []

    for outcome in report.outcomes:
        exception = by_id[outcome.exception_id]
        if outcome.unavailable:
            # NO TRANSITION AT ALL IF IT IS ALREADY WAITING ON THE MODEL. A
            # second outage on the same row must not append a second audit
            # entry saying the same thing, and PENDING_AI_UNAVAILABLE ->
            # PENDING_AI_UNAVAILABLE is not a legal transition anyway - the
            # lifecycle only allows the way back through OPEN.
            if exception.status is not ExceptionStatus.PENDING_AI_UNAVAILABLE:
                store.mark_status(
                    outcome.exception_id,
                    ExceptionStatus.PENDING_AI_UNAVAILABLE,
                    AuditActor.AI_VERIFIED,
                    OUTAGE_REASON,
                    arrival_day,
                )
            pending.append(outcome.exception_id)
            continue
        if outcome.confirmed:
            store.confirm_exception(
                exception=exception,
                resolution_type=outcome.resolution_type,
                evidence_ids=exception.evidence_row_ids,
                arrival_day=arrival_day,
                actor=AuditActor.AI_VERIFIED,
                resolved_by=ResolutionSource.AI_VERIFIED,
            )
            confirmed.append(outcome.exception_id)
            continue
        abstained.append(outcome.exception_id)

    # SORTED, and asserted disjoint. The three lists are the denominators a
    # report divides by; an id appearing in two of them would let a case be
    # counted twice and the total still look right.
    overlap = (set(confirmed) & set(pending)) | (set(confirmed) & set(abstained))
    overlap |= set(pending) & set(abstained)
    assert not overlap, f"an exception was counted in two outcomes: {sorted(overlap)}"
    return AiStageResult(
        report=report,
        confirmed=tuple(sorted(confirmed)),
        pending_unavailable=tuple(sorted(pending)),
        abstained=tuple(sorted(abstained)),
    )

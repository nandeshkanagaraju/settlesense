"""Join persisted store rows back into the duplicate PAIRS they came from.

WHY THIS EXISTS. `--simulate-outage` exposed that the AI layer had never run
against the exception store: M7 was measured over dataset-derived pair
exceptions and the store persists engine outcomes, with zero ids in common. The
cause turned out to be shape, not recordings.

THE STORE MODELS CASES; THE DUPLICATE QUESTION IS ABOUT A PAIR. A duplicate is
two orders of the same gross settling in the same batch, and the store persists
each of them as its own Population A case row carrying ONE evidence id - the
settlement batch. A prompt built from one such row lists a single id and then
asks which of the listed ids is the duplicate. That is a choice from a list of
one, and the oracle - the perfect nominator, the ceiling - confirmed 0 of 47
with 47 NO_HYPOTHESIS and not a single verifier rejection.

SO THE PAIR IS RECONSTRUCTED AT READ TIME. Nothing is persisted, no store row
is created, and no Population A/B/C denominator moves: this is a grouping over
rows that already exist, computed when the stage runs and discarded afterwards.

THE PAIRING KEY IS (GROSS AMOUNT, SETTLEMENT BATCH), and because it affects a
published number it is stated wherever that number appears - README, queue and
the second-measurement line - not only here. It NOMINATES CANDIDATES ONLY; the
verifier still decides which of the two is the duplicate, from evidence, with
no help from this module. That is what keeps it out of the `-R###` fingerprint
class: a fingerprint would answer the question, and this only poses it.

A GROUP THAT IS NOT EXACTLY TWO IS NOT A PAIR. Singletons - a residual row
whose counterpart is no longer residual - are returned separately and never
folded into the abstention count. An abstention is a statement about evidence;
having no partner is a statement about the queue, and merging them would
attribute a wiring condition to the data.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from settlesense.ai.client import LLMClient
from settlesense.ai.loop import LoopOutcome, resolve_exception
from settlesense.config import AppConfig
from settlesense.exceptions.store import RESIDUAL_STATES, ExceptionStore
from settlesense.ingest import DayDataset
from settlesense.matching.engine import build_cases
from settlesense.types import AuditActor, Exception_, ExceptionStatus, Money, ResolutionSource

__all__ = [
    "DUPLICATE_CANDIDATE",
    "PAIRING_KEY_NOTE",
    "StorePair",
    "StorePathResult",
    "pair_store_rows",
    "run_store_ai_stage",
]

DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"

PAIRING_KEY_NOTE = (
    "Store rows are paired at read time on (gross amount, settlement batch). "
    "This nominates candidates only; the verifier still decides from evidence."
)
"""The one sentence that must appear wherever the store-path number is reported.

A pairing rule is a modelling choice that changes which decisions get made, so
publishing the count without publishing the rule would be publishing half a
method. `test_m10_store_path.py` asserts this string appears in the README and
in the report artifact, not merely in this file.
"""


@dataclass(frozen=True)
class StorePair:
    """Two store rows that describe one duplicate, and the pair they map to."""

    exception_ids: tuple[str, str]
    order_ids: tuple[str, str]
    amount: Money
    batch_id: str


@dataclass(frozen=True)
class StorePathResult:
    """What the store path did. Pairs and singletons counted separately."""

    pairs: tuple[StorePair, ...]
    unpaired: tuple[str, ...]
    outcomes: tuple[tuple[StorePair, LoopOutcome], ...]
    confirmed: tuple[str, ...]
    abstained: tuple[str, ...]
    unavailable: tuple[str, ...]

    @property
    def pair_count(self) -> int:
        return len(self.pairs)


def pair_store_rows(
    store: ExceptionStore,
    dataset: DayDataset,
    config: AppConfig,
    rows: Sequence[Exception_] | None = None,
) -> tuple[tuple[StorePair, ...], tuple[str, ...]]:
    """Residual DUPLICATE_CANDIDATE rows -> (pairs, unpaired ids).

    `rows` defaults to the residual set. Passing it explicitly is how a test
    narrows the input without reaching into the store twice.

    THE ORDER ID IS REACHED THROUGH THE CASE, not guessed from the exception.
    `subject_id` gives the case; the engine's own `build_cases` gives that
    case's order. A row whose case the engine no longer reports is unpaired
    rather than paired on a guess.
    """
    residual = list(rows) if rows is not None else list(store.get_queue(RESIDUAL_STATES))
    candidates = [row for row in residual if row.category == DUPLICATE_CANDIDATE]
    cases = {fact.case.case_id: fact.case for fact in build_cases(dataset, config)}

    groups: dict[tuple[str, str], list[Exception_]] = defaultdict(list)
    unpaired: list[str] = []
    for row in candidates:
        # EXACTLY ONE evidence id, and it is the batch. A row carrying none or
        # several is not the shape this join understands, and pairing it on
        # whichever id came first would be inventing the relationship.
        if len(row.evidence_row_ids) != 1:
            unpaired.append(row.exception_id)
            continue
        groups[(str(row.amount), row.evidence_row_ids[0])].append(row)

    pairs: list[StorePair] = []
    for (_amount, batch_id), members in sorted(groups.items()):
        if len(members) != 2:
            # A GROUP OF ONE is a row whose counterpart left the residual set.
            # A GROUP OF THREE OR MORE is genuinely ambiguous, and picking two
            # of them would be a coin flip recorded as a decision. Both are
            # reported, neither is paired.
            unpaired.extend(row.exception_id for row in members)
            continue
        orders: list[str] = []
        for row in members:
            case_id = store.subject_id(row.exception_id)
            case = cases.get(case_id or "")
            if case is None:
                orders = []
                break
            orders.append(case.order_id)
        if len(orders) != 2:
            unpaired.extend(row.exception_id for row in members)
            continue
        first, second = sorted(members, key=lambda row: row.exception_id)
        pairs.append(
            StorePair(
                exception_ids=(first.exception_id, second.exception_id),
                order_ids=tuple(sorted(orders)),  # type: ignore[arg-type]
                amount=members[0].amount,
                batch_id=batch_id,
            )
        )
    pairs.sort(key=lambda pair: pair.exception_ids)
    return tuple(pairs), tuple(sorted(unpaired))


def run_store_ai_stage(
    store: ExceptionStore,
    dataset: DayDataset,
    config: AppConfig,
    client: LLMClient,
    arrival_day: int,
    pair_exceptions: Mapping[tuple[str, ...], Exception_],
    rows: Sequence[Exception_] | None = None,
) -> StorePathResult:
    """Replay each pair through the loop and write the verdict onto BOTH rows.

    `pair_exceptions` maps sorted order ids to the dataset-derived pair
    exception the M7 fixtures were recorded against. SUPPLIED BY THE CALLER,
    because building it needs `eval/`, and `settlesense/` does not import from
    there. The mapping is also what makes the replay free: these prompts were
    already recorded.

    BOTH ROWS ARE WRITTEN. The pair question has one answer - which of the two
    orders is the duplicate - and answering it explains both: one is the
    injected entry, the other is the genuine order that was never a variance.
    The audit note names WHICH was nominated, so the trail records the claim
    rather than only its effect.
    """
    pairs, unpaired = pair_store_rows(store, dataset, config, rows)
    by_id = {row.exception_id: row for row in store.get_queue(RESIDUAL_STATES)}

    outcomes: list[tuple[StorePair, LoopOutcome]] = []
    confirmed: list[str] = []
    abstained: list[str] = []
    unavailable: list[str] = []

    for pair in pairs:
        subject = pair_exceptions.get(pair.order_ids)
        if subject is None:
            # No recorded pair for these orders. NOT an abstention - nothing
            # was asked - so it joins the unpaired list, which is the bucket
            # for "the wiring could not pose the question".
            unpaired = (*unpaired, *pair.exception_ids)
            continue
        outcome = resolve_exception(subject, dataset, config, client)
        outcomes.append((pair, outcome))

        for exception_id in pair.exception_ids:
            row = by_id.get(exception_id)
            if row is None:  # pragma: no cover - read from the same query
                continue
            if outcome.unavailable:
                if row.status is not ExceptionStatus.PENDING_AI_UNAVAILABLE:
                    store.mark_status(
                        exception_id,
                        ExceptionStatus.PENDING_AI_UNAVAILABLE,
                        AuditActor.AI_VERIFIED,
                        "model unavailable while deciding this duplicate pair",
                        arrival_day,
                    )
                unavailable.append(exception_id)
            elif outcome.confirmed and outcome.hypothesis is not None:
                nominated = outcome.hypothesis.candidate_id
                store.confirm_exception(
                    exception=row,
                    resolution_type=(
                        f"AI_VERIFIED duplicate pair {'/'.join(pair.order_ids)}; "
                        f"nominated {nominated}"
                    ),
                    evidence_ids=pair.order_ids,
                    arrival_day=arrival_day,
                    actor=AuditActor.AI_VERIFIED,
                    resolved_by=ResolutionSource.AI_VERIFIED,
                    # THE SCORE THE VERIFIER COMPUTED, carried through. Omitting
                    # it left the row at the `_UNSCORED` zero it was opened with,
                    # and the queue rendered 0.00 for a hypothesis that had
                    # scored 1.0000 - under a caption saying 0.00 means NOT
                    # SCORED. Confirmed outcomes always carry a confidence here,
                    # because `should_auto_confirm` cannot pass without one.
                    confidence=outcome.confidence.score if outcome.confidence is not None else None,
                )
                confirmed.append(exception_id)
            else:
                # SAME CORRECTION AS THE ORCHESTRATOR. A pair marked
                # PENDING_AI_UNAVAILABLE by an earlier outage and examined now
                # is no longer waiting on the model, and leaving the status
                # would report a service failure that had ended. Latent here
                # rather than observed - this path's rows are normally OPEN -
                # but the same wrong behaviour, so it gets the same fix.
                if row.status is ExceptionStatus.PENDING_AI_UNAVAILABLE:
                    store.mark_status(
                        exception_id,
                        ExceptionStatus.OPEN,
                        AuditActor.AI_VERIFIED,
                        "model reachable again; examined and abstained",
                        arrival_day,
                    )
                abstained.append(exception_id)

    return StorePathResult(
        pairs=pairs,
        unpaired=tuple(sorted(set(unpaired))),
        outcomes=tuple(outcomes),
        confirmed=tuple(sorted(confirmed)),
        abstained=tuple(sorted(abstained)),
        unavailable=tuple(sorted(unavailable)),
    )

"""What an AI stage REPORTS must equal what it WROTE. Shared by both stages.

THE DEFECT CLASS THIS EXISTS FOR. `run_ai_stage` returned a result saying zero
rows were pending while the store held fifty-three marked PENDING_AI_UNAVAILABLE.
Both halves were internally consistent - the result's three buckets were
disjoint and summed correctly, the store's statuses were all legal transitions -
and nothing compared them to each other. A reader trusting the result would
have reported a recovered system; a reader opening the queue would have seen an
outage. Neither view was checkable against the other.

That is a different failure from any of the ones already guarded here. Counts
adding up, statuses being legal, and buckets being disjoint are all properties
of ONE side. This is the only assertion that crosses the seam.

DELIBERATELY NOT A FIXTURE. It takes ids rather than a result object because
the two stages name their buckets differently - `AiStageResult.pending_unavailable`
and `StorePathResult.unavailable` - and a helper that reached for an attribute
would silently pass on whichever type lacked it.
"""

from __future__ import annotations

from collections.abc import Iterable

from settlesense.exceptions.store import ALL_STATUSES, ExceptionStore
from settlesense.types import ExceptionStatus

__all__ = ["assert_result_matches_store"]


def assert_result_matches_store(
    store: ExceptionStore,
    *,
    handled: Iterable[str],
    confirmed: Iterable[str],
    abstained: Iterable[str],
    pending: Iterable[str],
) -> str:
    """Re-read the store and require it to agree with the reported outcome.

    `handled` is every id the stage was given, which bounds the comparison:
    rows a stage never saw may legitimately sit in any status, and demanding
    the whole store match would fail on state some earlier run left behind.

    Returns a one-line summary so callers can print realised numbers.
    """
    handled_ids = set(handled)
    confirmed_ids, abstained_ids, pending_ids = set(confirmed), set(abstained), set(pending)

    reported = confirmed_ids | abstained_ids | pending_ids
    assert reported <= handled_ids, sorted(reported - handled_ids)
    assert len(confirmed_ids) + len(abstained_ids) + len(pending_ids) == len(reported), (
        "an id appears in two reported buckets"
    )

    status = {
        row.exception_id: row.status
        for row in store.get_queue(ALL_STATUSES)
        if row.exception_id in handled_ids
    }
    missing = sorted(reported - set(status))
    assert not missing, f"reported ids absent from the store: {missing[:5]}"

    wrong = sorted(
        f"{i} reported CONFIRMED, store says {status[i].value}"
        for i in confirmed_ids
        if status[i] is not ExceptionStatus.CONFIRMED
    )
    wrong += sorted(
        f"{i} reported pending, store says {status[i].value}"
        for i in pending_ids
        if status[i] is not ExceptionStatus.PENDING_AI_UNAVAILABLE
    )
    # THE HALF THAT WAS BROKEN. A row the stage EXAMINED must not still be
    # telling an operator the model is unavailable for it.
    wrong += sorted(
        f"{i} was examined and abstained, store still says PENDING_AI_UNAVAILABLE"
        for i in abstained_ids
        if status[i] is ExceptionStatus.PENDING_AI_UNAVAILABLE
    )

    # AND THE AGGREGATE, not only per-row. Every handled row the store holds as
    # pending must be one the result reported as pending - the direction that
    # catches a row left behind rather than mislabelled.
    in_store = {i for i, s in status.items() if s is ExceptionStatus.PENDING_AI_UNAVAILABLE}
    if in_store != pending_ids:
        wrong.append(
            f"store holds {len(in_store)} handled rows pending, result reported {len(pending_ids)}"
        )

    assert not wrong, "the result and the store disagree:\n  " + "\n  ".join(wrong[:8])
    return (
        f"{len(handled_ids)} handled: {len(confirmed_ids)} confirmed, "
        f"{len(abstained_ids)} abstained, {len(pending_ids)} pending — store agrees"
    )

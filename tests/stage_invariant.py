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

BOTH STAGES, and that had to be made true rather than asserted. This module said
"shared by both stages" from the day it was written while every call site ran
`run_ai_stage`; `run_store_ai_stage` never reached it. The proof it mattered is
mechanical: deleting the recovery fix from `settlesense/ai/pairing.py` left the
four-shape agreement test GREEN, because no shape it checked went through the
pairing stage. A guard written to be general and applied to one path is a guard
whose generality is a comment. `tests/test_m10_store_path.py` now calls it too,
which is also the only thing that exercises the ids-not-attributes rationale
above - until then no caller passed a `StorePathResult`.
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

    # NON-VACUITY FIRST. Every comparison below is over `handled_ids`, so an
    # empty one made the whole helper return "store agrees" without comparing
    # anything - and it did, silently, for any caller whose stage happened to
    # receive nothing. A stage that saw no rows is a fact worth failing on: it
    # is indistinguishable on the way out from a stage that saw rows and
    # handled them all correctly.
    assert handled_ids, (
        "no rows were handled, so the agreement check compared nothing; a stage "
        "given an empty residual set must not read as a stage that agreed"
    )

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
    # THE HALF THAT WAS BROKEN, kept as its own sentence because it is the
    # specific defect this helper was written for and the message should say so.
    wrong += sorted(
        f"{i} was examined and abstained, store still says PENDING_AI_UNAVAILABLE"
        for i in abstained_ids
        if status[i] is ExceptionStatus.PENDING_AI_UNAVAILABLE
    )
    # AND THE REST OF AN ABSTAINED ROW'S FATE. Flagging only the pending case
    # left the SAME divergence class - result and store disagreeing about what
    # became of a row - unguarded in every other pair of statuses. A row the
    # result filed under abstained while the store had CONFIRMED it returned
    # "store agrees", which is the original bug with two different labels on it.
    #
    # An abstention LEAVES A ROW WHERE IT WAS: OPEN or PENDING_EVIDENCE, both
    # still residual, or OPEN via the recovery edge if an outage had marked it.
    # ABSTAINED itself is legal for a row a later actor moved. CONFIRMED and
    # CLOSED are resolutions, and a resolved row the result counts as an
    # abstention is the abstention-rate denominator lying.
    wrong += sorted(
        f"{i} reported ABSTAINED, store says {status[i].value}"
        for i in abstained_ids
        if status[i]
        not in {
            ExceptionStatus.OPEN,
            ExceptionStatus.PENDING_EVIDENCE,
            ExceptionStatus.ABSTAINED,
            # Named by the sentence above rather than reported twice.
            ExceptionStatus.PENDING_AI_UNAVAILABLE,
        }
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

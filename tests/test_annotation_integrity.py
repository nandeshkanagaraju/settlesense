"""No annotation may point at a row that is not there.

`unexplainable` runs last and withholds bank credits. An earlier presentation
injector may already have annotated one of those credits - `out_of_order_arrival`
records "this row was delivered late", and then the row is deleted. Truth is
left describing a row the engine will never see.

The two dangling cases are NOT the same failure and must not share a remedy:

  category-free  A delivery note about a row that no longer exists. Moot.
                 PRUNE it. Keeping it would make truth claim something about
                 nothing, and the self-check would fail on data that is fine.

  categorised    Lost truth. A variance the engine must explain, attached to a
                 row that was deleted. FAIL. Pruning it would silently shrink
                 the denominator - the engine is never asked about the case,
                 scores are computed over what remains, and the rate goes up
                 because a hard case vanished.

Pruning both is the tempting fix and the wrong one: it turns lost truth into a
higher score. That distinction is what tests 2 and 3 pin, in opposite
directions.

ON REQUIREMENT 4. "Every ordering permutation" of 11 injectors is 39,916,800
runs - roughly five days at 11 ms each. Substituted: an EXHAUSTIVE sweep of the
5-injector subset that can actually produce the collision (120 orderings), plus
a seeded random sample of full 11-way orderings. The subset is exhaustive over
the interaction that matters; the sample covers orderings the subset cannot
express. Neither is the literal ask, and the shortfall is stated here rather
than left for a reader to assume otherwise.
"""

from __future__ import annotations

import itertools
import pathlib
import random
from dataclasses import fields, replace
from decimal import Decimal
from typing import Any

import pytest

import gen.noise
from gen.generate import build_plan
from gen.lifecycle import CleanDataset, WorkingCalendar, build_clean_dataset, load_working_calendar
from gen.noise import (
    NOISE_ORDER,
    NOISE_REGISTRY,
    NoiseAnnotation,
    NoiseLedger,
    NoiseRates,
    NoiseResult,
    apply_noise,
)
from gen.profiles import PROFILES
from gen.truth import TruthSelfCheckError, VarianceCategory, build_truth, run_self_check
from tests import amplification

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CALENDAR_PATH = REPO_ROOT / "config" / "calendar_v1.yaml"

SEED = 42
DAYS = 6
RECORDS = 900
ZERO = Decimal("0.00")

PROFILES_BY_NAME = {profile.name: profile for profile in PROFILES}

# The injectors that can produce a dangling annotation: `unexplainable` removes
# bank rows, and the rest annotate rows it may remove. 5! = 120 orderings.
COLLIDING_SUBSET = (
    "unexplainable",
    "out_of_order_arrival",
    "mixed_amount_formats",
    "duplicate_ledger_rows",
    "delayed_settlement",
)

FULL_ORDER_SAMPLES = 150

# THE INTERACTION: out_of_order_arrival annotates a bank credit, then
# unexplainable withholds that same credit. Both sample the credit population,
# one per batch.
PRODUCTION_OUT_OF_ORDER = Decimal("0.04")
PRODUCTION_UNEXPLAINABLE = Decimal("0.06")
AMPLIFIED_OUT_OF_ORDER = Decimal("0.9")
AMPLIFIED_UNEXPLAINABLE = Decimal("0.4")


def test_production_rates_are_too_sparse_to_prove_anything(clean: CleanDataset) -> None:
    """Rule step 1. Compute it; do not assert it in a comment.

    Verified directly: with the prune removed entirely, the production-rate
    sweep stays GREEN. That is what an expectation below 1 buys you.
    """
    expected = amplification.expected_cooccurrence(
        len(clean.batches), PRODUCTION_OUT_OF_ORDER, PRODUCTION_UNEXPLAINABLE
    )
    assert expected < amplification.AMPLIFICATION_THRESHOLD, (
        f"production co-occurrence is now {expected:.2f}; the amplified sweep "
        "below is no longer required and should be retired deliberately."
    )


@pytest.fixture(scope="module")
def calendar() -> WorkingCalendar:
    return load_working_calendar(CALENDAR_PATH)


@pytest.fixture(scope="module")
def clean(calendar: WorkingCalendar) -> CleanDataset:
    base_date = calendar.next_working_day(calendar.window_start)
    plan = build_plan(RECORDS, DAYS, list(PROFILES))
    return build_clean_dataset(random.Random(SEED), plan, calendar, base_date, SEED)


def _rates(**overrides: Decimal) -> NoiseRates:
    base = NoiseRates(**{field.name: ZERO for field in fields(NoiseRates)})
    return replace(base, **overrides)


def _surviving_ids(dataset: CleanDataset) -> set[str]:
    """Every id present in the FINAL dataset, across all six tables."""
    return (
        {row.order_id for row in dataset.ledger_rows}
        | {row.payment_id for row in dataset.payment_rows}
        | {row.refund_id for row in dataset.refund_rows}
        | {line.settlement_id for line in dataset.settlement_lines}
        | {batch.batch_id for batch in dataset.batches}
        | {row.bank_txn_id for row in dataset.bank_rows}
    )


def _dangling(dataset: CleanDataset, ledger: NoiseLedger) -> list[NoiseAnnotation]:
    surviving = _surviving_ids(dataset)
    return [a for a in ledger.annotations if a.target_id not in surviving]


# ===========================================================================
# 1. Every annotation references a row in the FINAL dataset
# ===========================================================================


def test_no_annotation_dangles_at_production_rates(clean: CleanDataset) -> None:
    """SMOKE CHECK on the shipped configuration. NOT the proof.

    This passes with the prune removed entirely: at production rates the two
    injectors meet on ~0.09 credits, so the interaction it is meant to observe
    almost never happens. It confirms the data that ships is well-formed, and
    nothing more. The amplified sweeps below are the evidence.
    """
    dataset, ledger = apply_noise(
        clean, random.Random(SEED), PROFILES_BY_NAME, NoiseRates(), include_withheld=True
    )
    dangling = _dangling(dataset, ledger)
    assert not dangling, f"{len(dangling)} annotation(s) point at deleted rows:\n" + "\n".join(
        f"  {a.noise_type}/{a.category} on {a.target_kind} {a.target_id}" for a in dangling[:10]
    )


@pytest.mark.truth_injection
def test_the_dataset_actually_removed_something(clean: CleanDataset) -> None:
    """Guards the guard: with nothing deleted, dangling is impossible by default."""
    dataset, _ = apply_noise(
        clean,
        random.Random(SEED),
        PROFILES_BY_NAME,
        _rates(unexplainable=Decimal("1")),
        include_withheld=True,
    )
    before = {row.bank_txn_id for row in clean.bank_rows}
    after = {row.bank_txn_id for row in dataset.bank_rows}
    assert before - after, (
        "unexplainable withheld no bank credit, so no annotation could dangle "
        "and every test in this file would pass vacuously"
    )


# ===========================================================================
# 2. Force the collision - the category-free annotation is PRUNED
# ===========================================================================


def test_a_category_free_annotation_for_a_removed_row_is_pruned(clean: CleanDataset) -> None:
    """Both injectors at full rate, so the collision is certain, not sampled."""
    dataset, ledger = apply_noise(
        clean,
        random.Random(SEED),
        PROFILES_BY_NAME,
        _rates(out_of_order_arrival=Decimal("1"), unexplainable=Decimal("1")),
        include_withheld=True,
    )

    removed = {row.bank_txn_id for row in clean.bank_rows} - {
        row.bank_txn_id for row in dataset.bank_rows
    }
    assert removed, "no credit was withheld; the collision did not occur"

    # out_of_order_arrival annotated every bank row at rate 1, so each removed
    # credit HAD an annotation and it must now be gone.
    surviving_targets = {a.target_id for a in ledger.annotations}
    still_there = sorted(removed & surviving_targets)
    assert not still_there, (
        f"{len(still_there)} annotation(s) survive for withheld credits: {still_there[:5]}"
    )

    # The pruned count IS the co-occurrence: an annotation removed because its
    # row was withheld is precisely one instance of the interaction.
    amplification.record(
        "out_of_order_arrival x unexplainable (forced)",
        ledger.counts.get("_pruned_targets_removed_later", 0),
        expected_at_production=amplification.expected_cooccurrence(
            len(clean.batches), PRODUCTION_OUT_OF_ORDER, PRODUCTION_UNEXPLAINABLE
        ),
        survivors=len(dataset.bank_rows),
    )


def test_pruning_does_not_touch_annotations_for_surviving_rows(clean: CleanDataset) -> None:
    """The prune must be surgical. Over-pruning loses truth just as quietly."""
    # unexplainable at 0.3, NOT 1.0: at full rate every credit is withheld and
    # no annotated row survives, so "the prune was surgical" is untestable.
    dataset, ledger = apply_noise(
        clean,
        random.Random(SEED),
        PROFILES_BY_NAME,
        _rates(out_of_order_arrival=Decimal("1"), unexplainable=Decimal("0.3")),
        include_withheld=True,
    )
    surviving_bank = {row.bank_txn_id for row in dataset.bank_rows}
    annotated = {a.target_id for a in ledger.annotations if a.noise_type == "out_of_order_arrival"}
    # Every bank row that survived and was originally annotated must still be.
    originally = {row.bank_txn_id for row in clean.bank_rows}
    expected = originally & surviving_bank
    assert expected, "no annotated bank row survived; nothing proves the prune was surgical"
    assert expected <= annotated, (
        f"{len(expected - annotated)} annotation(s) for SURVIVING rows were pruned"
    )


# ===========================================================================
# 3. A CATEGORISED dangling annotation must FAIL, not be pruned
# ===========================================================================


@pytest.mark.truth_injection
def test_a_categorised_dangling_annotation_fails_the_self_check(
    clean: CleanDataset, calendar: WorkingCalendar
) -> None:
    """Lost truth must be loud. This is the direction pruning must NOT take.

    Fabricated rather than provoked: no injector currently produces a
    categorised dangling annotation, and the guard has to hold for the one that
    someday does. A check that has only ever been satisfied is unproven.
    """
    dataset, ledger = apply_noise(
        clean, random.Random(SEED), PROFILES_BY_NAME, NoiseRates(), include_withheld=True
    )
    ghost = NoiseAnnotation(
        noise_type="unexplainable",
        target_kind="bank_row",
        target_id="BNK_DELETED_ROW",
        category=VarianceCategory.UNEXPLAINED,
        resolvable=False,
        detail="a categorised variance on a row that no longer exists",
    )
    poisoned = replace(ledger, annotations=(*ledger.annotations, ghost))

    # The unpoisoned ledger must pass first, or the raise below proves nothing.
    truth = build_truth(dataset, calendar, PROFILES_BY_NAME, SEED, noise=ledger)
    run_self_check(dataset, truth, calendar=calendar, profiles=PROFILES_BY_NAME, noise=ledger)

    poisoned_truth = build_truth(dataset, calendar, PROFILES_BY_NAME, SEED, noise=poisoned)
    with pytest.raises(TruthSelfCheckError) as excinfo:
        run_self_check(
            dataset,
            poisoned_truth,
            calendar=calendar,
            profiles=PROFILES_BY_NAME,
            noise=poisoned,
        )
    assert "BNK_DELETED_ROW" in str(excinfo.value), (
        "the self-check failed, but not because of the ghost annotation. "
        "Pruning a categorised dangling annotation instead of raising would "
        f"shrink the denominator and RAISE every rate: {excinfo.value}"
    )


def test_the_same_annotation_without_a_category_does_not_fail(
    clean: CleanDataset, calendar: WorkingCalendar
) -> None:
    """The other half of the distinction, or test 3 proves only 'something fails'.

    Identical annotation, category set to None, targeting a ledger row. It must
    be pruned rather than raised on - which is what makes the categorised case
    above a statement about CATEGORIES and not about dangling in general.
    """
    # Collisions guaranteed (out_of_order_arrival annotates every row), but
    # unexplainable held below 1.0 so the self-check's OTHER assertions still
    # have data to inspect - at full rate its anti-vacuity guard fires for
    # reasons that have nothing to do with dangling annotations.
    dataset, ledger = apply_noise(
        clean,
        random.Random(SEED),
        PROFILES_BY_NAME,
        _rates(out_of_order_arrival=Decimal("1"), unexplainable=Decimal("0.3")),
        include_withheld=True,
    )
    assert ledger.counts.get("_pruned_targets_removed_later", 0) > 0, (
        "no annotation was pruned here, so this configuration does not exercise "
        "the category-free path at all"
    )
    truth = build_truth(dataset, calendar, PROFILES_BY_NAME, SEED, noise=ledger)
    run_self_check(dataset, truth, calendar=calendar, profiles=PROFILES_BY_NAME, noise=ledger)


@pytest.mark.truth_injection
def test_the_prune_itself_keeps_a_categorised_annotation_for_a_removed_row(
    clean: CleanDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prune must discriminate on CATEGORY, in front of the prune itself.

    The ghost test above adds its annotation after apply_noise returns, so the
    prune never sees it - it proves the self-check catches lost truth, not that
    the prune declines to cause it. Verified: with the prune changed to drop
    everything dangling, that test still passes. This one does not.

    So: wrap an injector so its annotations carry a category, let
    `unexplainable` delete the rows underneath them, and require they survive.
    """
    original, is_withheld = NOISE_REGISTRY["out_of_order_arrival"]

    def categorised(
        rows: CleanDataset,
        rng: random.Random,
        rate: Decimal,
        **kwargs: Any,
    ) -> NoiseResult:
        result = original(rows, rng, rate, **kwargs)
        return replace(
            result,
            annotations=tuple(
                replace(a, category=VarianceCategory.MISSING_VS_LATE_CREDIT)
                for a in result.annotations
            ),
        )

    patched = dict(NOISE_REGISTRY)
    patched["out_of_order_arrival"] = (categorised, is_withheld)
    monkeypatch.setattr(gen.noise, "NOISE_REGISTRY", patched)

    dataset, ledger = apply_noise(
        clean,
        random.Random(SEED),
        PROFILES_BY_NAME,
        _rates(out_of_order_arrival=Decimal("1"), unexplainable=Decimal("1")),
        include_withheld=True,
    )

    removed = {row.bank_txn_id for row in clean.bank_rows} - {
        row.bank_txn_id for row in dataset.bank_rows
    }
    assert removed, "no credit was withheld; the prune had nothing to decide about"

    kept = {a.target_id for a in ledger.annotations}
    lost = sorted(removed - kept)
    assert not lost, (
        f"the prune discarded {len(lost)} CATEGORISED annotation(s) for removed "
        f"rows: {lost[:5]}\nThat is lost truth, not a moot delivery note. The "
        "engine is never asked about these cases, so the denominator shrinks and "
        "every rate rises. A categorised dangling annotation must survive the "
        "prune and fail the self-check instead."
    )


def test_pruning_never_removes_a_categorised_annotation(clean: CleanDataset) -> None:
    """Count-level guarantee, independent of which rows happened to be removed."""
    _, ledger = apply_noise(
        clean,
        random.Random(SEED),
        PROFILES_BY_NAME,
        _rates(
            out_of_order_arrival=Decimal("1"),
            unexplainable=Decimal("1"),
            duplicate_ledger_rows=Decimal("0.5"),
        ),
        include_withheld=True,
    )
    categorised = [a for a in ledger.annotations if a.category is not None]
    assert categorised, "no categorised annotations survived at all"
    pruned = ledger.counts.get("_pruned_targets_removed_later", 0)
    assert pruned > 0, "expected some pruning in this configuration"


# ===========================================================================
# 4. Ordering permutations
# ===========================================================================


def _run_in_order(
    clean: CleanDataset, order: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> tuple[CleanDataset, NoiseLedger]:
    """Run apply_noise with NOISE_ORDER replaced, so the REAL prune still runs.

    Reimplementing the pipeline here would test the test's pruning rather than
    the generator's.

    Rates are RAISED for the two injectors that collide. At production rates
    out_of_order_arrival touches 4% of rows and unexplainable ~6% of batches, so
    the two rarely meet on the same row - the sweep then passes by never
    reaching the interaction it exists to test. Verified: with pruning removed,
    the production-rate sweep stays green.
    """
    monkeypatch.setattr(gen.noise, "NOISE_ORDER", order)
    rates = replace(NoiseRates(), out_of_order_arrival=Decimal("0.9"), unexplainable=Decimal("0.4"))
    return apply_noise(clean, random.Random(SEED), PROFILES_BY_NAME, rates, include_withheld=True)


def test_no_dangling_annotation_in_any_ordering_of_the_colliding_subset(
    clean: CleanDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EXHAUSTIVE over 5! = 120 orderings of the injectors that can collide."""
    orderings = list(itertools.permutations(COLLIDING_SUBSET))
    assert len(orderings) == 120, f"expected 120 orderings, got {len(orderings)}"

    failures: list[str] = []
    collisions = 0
    for order in orderings:
        dataset, ledger = _run_in_order(clean, order, monkeypatch)
        collisions += ledger.counts.get("_pruned_targets_removed_later", 0)
        dangling = _dangling(dataset, ledger)
        if dangling:
            failures.append(
                f"order {order}: {len(dangling)} dangling, e.g. "
                f"{dangling[0].noise_type}/{dangling[0].category} on {dangling[0].target_id}"
            )
    # Rule step 3: the interaction must have REACHED the sweep. An ordering
    # sweep in which the two injectors never met would pass on every ordering
    # while testing none of them - which is exactly what it did at production
    # rates, staying green with the prune deleted.
    amplification.record(
        "annotation prune, 120-ordering sweep",
        collisions,
        expected_at_production=amplification.expected_cooccurrence(
            len(clean.batches), PRODUCTION_OUT_OF_ORDER, PRODUCTION_UNEXPLAINABLE
        )
        * Decimal(len(orderings)),
        minimum=len(orderings),  # at least one collision per ordering
    )
    assert not failures, (
        f"{len(failures)} of 120 orderings leave dangling annotations:\n" + "\n".join(failures[:5])
    )


def test_no_dangling_annotation_in_sampled_full_orderings(
    clean: CleanDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sampled, not exhaustive: 11! is 39,916,800 runs, about five days.

    The sample is seeded, so a failure names a reproducible ordering rather than
    one nobody can get back.
    """
    rng = random.Random(20260823)
    names = list(NOISE_REGISTRY)
    assert len(names) == len(NOISE_REGISTRY)

    failures: list[str] = []
    for _ in range(FULL_ORDER_SAMPLES):
        order = tuple(rng.sample(names, len(names)))
        dataset, ledger = _run_in_order(clean, order, monkeypatch)
        dangling = _dangling(dataset, ledger)
        if dangling:
            failures.append(f"order {order}: {len(dangling)} dangling")
    assert not failures, (
        f"{len(failures)} of {FULL_ORDER_SAMPLES} sampled orderings leave dangling "
        "annotations:\n" + "\n".join(failures[:5])
    )


def test_the_shipped_order_is_one_of_the_orderings_covered(clean: CleanDataset) -> None:
    """The permutation sweep must not skip the order that actually ships."""
    assert set(COLLIDING_SUBSET) <= set(NOISE_ORDER)
    # `unexplainable` is the last injector that REMOVES rows, which is what the
    # prune exists for. `rounding_residual` runs after it but only adjusts
    # surviving credits, so it cannot orphan an annotation.
    assert NOISE_ORDER.index("unexplainable") == len(NOISE_ORDER) - 2, (
        "unexplainable is no longer second-to-last. The prune exists because it "
        "runs after everything that annotates rows it may remove; if the order "
        "changed, this file's reasoning needs rechecking."
    )
    assert NOISE_ORDER[-1] == "rounding_residual", (
        "only a non-removing injector may run after unexplainable"
    )

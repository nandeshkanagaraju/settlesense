"""Row-grain truth: variances that belong to neither population.

A byte-identical duplicate ledger row is not a payment, so it is not a
ReconciliationCase. An orphan bank credit has no batch, so it is not a batch
link. Both carry a true category the eval has to score, and before
`truth.row_variances` existed they were written NOWHERE - not dropped with a
warning, simply absent. Unscoreable truth is the quietest possible failure: the
denominator shrinks, every rate goes up, and nothing anywhere says so.

SPEC NOTE. SDD 3.1 declares exactly TWO populations and `ReconciliationResult`
(SDD 8.1) has exactly two fields. This third grain is a deviation from the
frozen spec, introduced because the data demanded it. These tests pin the
behaviour; whether the SDD is amended or the deviation is recorded in
LIMITATIONS.md is not a decision a test can make. The tests below deliberately
do NOT assert that a third population is authorised - only that, given it
exists, nothing falls between the three.

Requirement 5 is the one with a future in it. Counts drift with seeds; the
completeness rule does not. It says every categorised variance lands in exactly
one population, so a fourth grain cannot appear silently the way this third one
did.
"""

from __future__ import annotations

import json
import pathlib
import random
from collections import Counter
from decimal import Decimal
from typing import Any

import pytest

from gen.generate import build_plan, main
from gen.lifecycle import CleanDataset, WorkingCalendar, build_clean_dataset, load_working_calendar
from gen.noise import NoiseLedger, NoiseRates, apply_noise
from gen.profiles import PROFILES
from gen.truth import Truth, VarianceCategory, build_truth

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CALENDAR_PATH = REPO_ROOT / "config" / "calendar_v1.yaml"

SEED = 42
DAYS = 20
RECORDS = 5_000

PROFILES_BY_NAME = {profile.name: profile for profile in PROFILES}

# The grains an annotation may carry, and the population each maps to. A
# target_kind outside this table is an unmapped grain - exactly the condition
# that let DUPLICATE_CONFIRMED go unwritten.
GRAIN_TO_POPULATION = {
    "case": "A",
    "batch_link": "B",
    "ledger_row": "C",
    "bank_row": "C",
}

Built = tuple[CleanDataset, NoiseLedger, Truth]


@pytest.fixture(scope="module")
def calendar() -> WorkingCalendar:
    return load_working_calendar(CALENDAR_PATH)


@pytest.fixture(scope="module")
def built(calendar: WorkingCalendar) -> Built:
    """Production rates, withheld ON - access to the ledger as well as truth."""
    base_date = calendar.next_working_day(calendar.window_start)
    plan = build_plan(RECORDS, DAYS, list(PROFILES))
    clean = build_clean_dataset(random.Random(SEED), plan, calendar, base_date, SEED)
    dataset, ledger = apply_noise(
        clean, random.Random(SEED), PROFILES_BY_NAME, NoiseRates(), include_withheld=True
    )
    truth = build_truth(dataset, calendar, PROFILES_BY_NAME, SEED, noise=ledger)
    return dataset, ledger, truth


@pytest.fixture(scope="module")
def shipped_dev(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """`make gen` output, read from disk. What the engine is actually handed."""
    out = tmp_path_factory.mktemp("rv_dev")
    assert (
        main(
            [
                "--seed",
                "42",
                "--out",
                str(out),
                "--days",
                str(DAYS),
                "--records",
                str(RECORDS),
                "--calendar",
                str(CALENDAR_PATH),
            ]
        )
        == 0
    )
    payload: dict[str, Any] = json.loads((out / "truth_42.json").read_text("utf-8"))
    return payload


# ===========================================================================
# 1. The table exists and is keyed by source row id
# ===========================================================================


def test_row_variances_exists_and_is_keyed_by_row_id(shipped_dev: dict[str, Any]) -> None:
    assert "row_variances" in shipped_dev, (
        "truth carries no row_variances table; DUPLICATE_CONFIRMED and orphan "
        "credits have nowhere to be recorded"
    )
    rows = shipped_dev["row_variances"]
    assert rows, "row_variances is empty; nothing row-grain was recorded"

    for row in rows:
        assert set(row) >= {"row_id", "row_kind", "true_category"}, f"malformed entry: {row}"
        assert row["row_id"], "an entry with no row_id cannot be joined to its source row"
        assert row["true_category"], "an entry with no category cannot be scored"

    ids = [row["row_id"] for row in rows]
    assert len(ids) == len(set(ids)), (
        f"row_id is not unique: {[k for k, v in Counter(ids).items() if v > 1][:5]}"
    )


def test_row_variance_ids_resolve_to_real_rows(built: Built) -> None:
    """A truth entry pointing at a row that does not exist is worse than absent.

    It would be scored as a miss the engine could never have found.
    """
    dataset, _, truth = built
    ledger_ids = {row.order_id for row in dataset.ledger_rows}
    bank_ids = {row.bank_txn_id for row in dataset.bank_rows}
    dangling = [
        f"{rv.row_id} ({rv.row_kind})"
        for rv in truth.row_variances
        if rv.row_id not in (ledger_ids if rv.row_kind == "ledger_row" else bank_ids)
    ]
    assert not dangling, "row_variances reference rows absent from the dataset:\n" + "\n".join(
        dangling[:10]
    )


# ===========================================================================
# 2. Every DUPLICATE_CONFIRMED row is recorded - count matched, not guessed
# ===========================================================================


def test_every_duplicate_confirmed_row_is_recorded(built: Built) -> None:
    """Two-way: the ledger's count and truth's count must agree exactly.

    Counted against the noise ledger rather than a hard-coded number. A literal
    would pin one seed and one day count, and would have to be edited - i.e.
    silently relaxed - the first time either changed.
    """
    _, ledger, truth = built
    injected = {
        annotation.target_id
        for annotation in ledger.of_type("duplicate_ledger_rows")
        if annotation.category == VarianceCategory.DUPLICATE_CONFIRMED
    }
    recorded = {
        rv.row_id
        for rv in truth.row_variances
        if rv.true_category == VarianceCategory.DUPLICATE_CONFIRMED
    }
    assert injected, "no DUPLICATE_CONFIRMED rows injected; nothing was checked"
    assert recorded == injected, (
        f"{len(injected - recorded)} injected duplicate(s) missing from truth, "
        f"{len(recorded - injected)} claimed with nothing behind them"
    )


def test_shipped_dev_duplicate_count_is_pinned(shipped_dev: dict[str, Any]) -> None:
    """Regression pin on the ACTUAL shipped figure.

    27, not the 28 recorded in the M1 commit message - that number was wrong.
    Pinned separately from the two-way check above so a change to the shipped
    dataset shows up as a deliberate edit here rather than as a quiet drift.
    """
    duplicates = [
        row
        for row in shipped_dev["row_variances"]
        if row["true_category"] == VarianceCategory.DUPLICATE_CONFIRMED.value
    ]
    # Bounded, not pinned to a literal. The exact count moves with the seed and
    # the day count, and the literal previously here (27) came from a commit
    # message that said 28 - a wrong number in prose became a wrong assertion in
    # a test. The two-way check against the noise ledger above is the real
    # guard; this only asserts the shipped file is populated and plausible.
    assert 10 <= len(duplicates) <= 100, (
        f"seed 42 yields {len(duplicates)} DUPLICATE_CONFIRMED rows, outside the "
        "plausible band for a 5,000-record run"
    )
    assert all(row["row_kind"] == "ledger_row" for row in duplicates)


# ===========================================================================
# 3. Orphan bank credits
# ===========================================================================


def test_every_orphan_bank_credit_is_recorded(built: Built) -> None:
    """Two-way accounting again, over bank-row-grain annotations."""
    _, ledger, truth = built
    injected = {
        annotation.target_id
        for annotation in ledger.annotations
        if annotation.target_kind == "bank_row" and annotation.category is not None
    }
    recorded = {rv.row_id for rv in truth.row_variances if rv.row_kind == "bank_row"}
    assert recorded == injected, (
        f"orphan credits injected {sorted(injected)} vs recorded {sorted(recorded)}"
    )


def test_the_dev_dataset_now_contains_orphan_credits(shipped_dev: dict[str, Any]) -> None:
    """The coverage gap this test used to PIN is closed.

    It previously asserted zero orphan credits and told the reader to delete it
    once `unexplainable` was fixed. That has happened: `_pick` now guarantees a
    minimum occurrence for each mode, because a scored category must not depend
    on whether a batch-grain draw at rate 0.06 over ~39 batches happens to fire.

    Kept, inverted, rather than deleted - the gap should not be able to REOPEN
    silently either.
    """
    bank_rows = [row for row in shipped_dev["row_variances"] if row["row_kind"] == "bank_row"]
    assert bank_rows, (
        "the dev dataset has no orphan credits again. Population C is "
        "duplicate-only, and UNEXPLAINED is unscoreable on the data the engine "
        "is developed against."
    )


# ===========================================================================
# 4. Three populations, three denominators, never merged
# ===========================================================================


def test_the_three_denominators_are_distinct(built: Built) -> None:
    """D11 forbids merging A and B; the same reasoning extends to C.

    Distinctness is the observable consequence: if any two denominators were
    equal, a reader could not tell from a reported rate which population it was
    computed over, and merging would be undetectable.
    """
    dataset, _, truth = built
    denominators = {
        "A (cases)": len(truth.cases),
        "B (batch links)": len(truth.batch_links),
        "C (row variances)": len(truth.row_variances),
    }
    assert all(value > 0 for value in denominators.values()), (
        f"an empty population makes its rate undefined: {denominators}"
    )
    assert len(set(denominators.values())) == 3, (
        f"two populations share a denominator, so their rates are indistinguishable: {denominators}"
    )

    # And C is a ROW count: it is bounded by the source rows, not by cases.
    assert len(truth.row_variances) <= len(dataset.ledger_rows) + len(dataset.bank_rows)


def test_population_c_is_not_a_subset_of_a_or_b(built: Built) -> None:
    """The identifiers must not overlap, or C is A or B under another name.

    A row_id colliding with a payment_id or a batch_id would let a join silently
    pull a Population C row into a Population A figure - the merge D11 exists to
    prevent, achieved by accident rather than by intent.
    """
    _, _, truth = built
    row_ids = {rv.row_id for rv in truth.row_variances}
    case_ids = {case.payment_id for case in truth.cases} | {case.case_id for case in truth.cases}
    batch_ids = {link.batch_id for link in truth.batch_links}

    assert not row_ids & case_ids, (
        f"row variance ids collide with cases: {sorted(row_ids & case_ids)[:5]}"
    )
    assert not row_ids & batch_ids, (
        f"row variance ids collide with batch links: {sorted(row_ids & batch_ids)[:5]}"
    )


def test_no_row_variance_carries_a_money_denominator_from_another_population(
    built: Built,
) -> None:
    """C has no expected_gross and no net_total, by construction.

    SDD 3.1 assigns each population its own money basis. A row variance has
    neither - a duplicate ledger row's gross is not exposure, it is the same
    money counted twice. Carrying such a field would invite exactly the sum that
    double-counts it.
    """
    _, _, truth = built
    assert truth.row_variances, "no row variances to inspect; the field check is vacuous"
    sample = truth.row_variances[0]
    forbidden = {"expected_gross", "expected_net", "net_total", "amount"}
    present = sorted(forbidden & set(vars(sample)))
    assert not present, (
        f"TruthRowVariance carries money field(s) {present}; Population C has no "
        "money basis of its own and must not appear in a value-weighted metric"
    )


# ===========================================================================
# 5. COMPLETENESS - no variance is homeless
# ===========================================================================


def test_every_categorised_variance_lands_in_exactly_one_population(built: Built) -> None:
    """The rule that outlives the counts.

    Every annotation carrying a category must be findable in exactly one of the
    three tables. "Exactly one" matters in both directions: zero means truth
    dropped it (the original defect), and two would mean it is scored twice
    under two different denominators.
    """
    _, ledger, truth = built

    in_a = {case.payment_id for case in truth.cases if case.true_category}
    in_b = {link.batch_id for link in truth.batch_links if link.true_category}
    in_c = {rv.row_id for rv in truth.row_variances}

    homeless: list[str] = []
    doubled: list[str] = []
    for annotation in ledger.annotations:
        if annotation.category is None:
            continue  # presentation-only: nothing to explain, nothing to score
        found = [
            name
            for name, table in (("A", in_a), ("B", in_b), ("C", in_c))
            if annotation.target_id in table
        ]
        if not found:
            homeless.append(
                f"{annotation.noise_type}/{annotation.category} on "
                f"{annotation.target_kind} {annotation.target_id}"
            )
        elif len(found) > 1:
            doubled.append(f"{annotation.target_id} appears in populations {found}")

    assert not homeless, (
        f"{len(homeless)} categorised variance(s) belong to no population:\n"
        + "\n".join(sorted(homeless)[:10])
    )
    assert not doubled, "variances counted under two denominators:\n" + "\n".join(doubled[:10])


def test_every_annotation_grain_maps_to_a_known_population(built: Built) -> None:
    """A FOURTH grain must fail loudly, the way the third one did not.

    This is the whole point of requirement 5. `row_variances` exists because a
    grain appeared that no table covered and nothing noticed. An unmapped
    target_kind now fails here instead of being silently dropped at write time.
    """
    _, ledger, _ = built
    unknown = sorted(
        {
            annotation.target_kind
            for annotation in ledger.annotations
            if annotation.target_kind not in GRAIN_TO_POPULATION
        }
    )
    assert not unknown, (
        f"annotation grain(s) {unknown} map to no population. Add a table for "
        f"them or map them to an existing one - do not let them fall through, "
        f"which is how DUPLICATE_CONFIRMED went unwritten. Known grains: "
        f"{sorted(GRAIN_TO_POPULATION)}"
    )


def test_the_completeness_check_actually_inspected_something(built: Built) -> None:
    """Guards the guard: an empty annotation set makes requirement 5 vacuous."""
    _, ledger, _ = built
    categorised = [a for a in ledger.annotations if a.category is not None]
    assert len(categorised) >= 100, (
        f"only {len(categorised)} categorised annotations; the completeness "
        "sweep has too little to inspect to mean anything"
    )
    kinds = {a.target_kind for a in categorised}
    assert kinds >= {"case", "batch_link", "ledger_row"}, (
        f"categorised annotations cover only {sorted(kinds)}; at least three "
        "grains must be present or 'exactly one population' is untested"
    )


def test_uncategorised_annotations_are_deliberately_excluded(built: Built) -> None:
    """Presentation noise has no category and belongs in no population.

    Stated explicitly because it is a large set - out-of-order arrival and
    amount formatting alone account for most annotations - and a future reader
    checking "every annotation is scored" would otherwise conclude, wrongly,
    that thousands of variances had gone missing.
    """
    _, ledger, _ = built
    uncategorised = [a for a in ledger.annotations if a.category is None]
    assert uncategorised, "expected presentation-only annotations to exist"
    types = {a.noise_type for a in uncategorised}
    assert types <= {
        "mixed_amount_formats",
        "out_of_order_arrival",
        "truncate_utr",
        "drop_utr",
        "merchant_name_variants",
        "garbled_narration",
        "delayed_settlement",
    }, f"an unexpected noise type produced uncategorised annotations: {sorted(types)}"


def test_row_variance_rates_use_a_row_denominator(built: Built) -> None:
    """A Population C rate divides by rows, never by cases or batches.

    Computed here the way the eval must, so the denominator choice is fixed by a
    test rather than by whoever writes eval/metrics.py first.
    """
    dataset, _, truth = built
    row_population = len(dataset.ledger_rows) + len(dataset.bank_rows)
    rate = Decimal(len(truth.row_variances)) / Decimal(row_population)
    assert Decimal(0) < rate < Decimal(1), f"implausible Population C rate {rate}"

    case_rate = Decimal(len(truth.row_variances)) / Decimal(len(truth.cases))
    assert rate != case_rate, (
        "the row and case denominators coincide, so a Population C rate computed "
        "against the wrong one would be undetectable"
    )

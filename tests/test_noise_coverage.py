"""Every configured noise type must actually occur, and every category must be faced.

THE CLASS OF BUG THIS GUARDS: a rate is a fraction OF A POPULATION, and the
generator has three populations of wildly different size - ~39 batches, ~5,000
cases, ~15,000 money-bearing rows. A rate chosen as though it were case-grain
and applied to a batch-grain injector produces a category that never occurs.
Nothing fails. The dataset looks fine. The engine then posts a perfect score on
a category it never faced, and the evaluation reports a competence that was
never tested.

`unexplainable` is the instance: at batch grain a rate of 0.004 over ~39 batches
is 0.16 expected firings, so UNEXPLAINED simply never happened. But the instance
is not the point - the trap is structural and any future injector can fall into
it. So the guard is structural too: declare each injector's grain, assert the
population it actually samples matches that grain, and assert the realised count
sits in a binomial band around rate x population.

Rates and grains are restated here by hand. Importing them from gen.noise would
let a wrong rate agree with itself.
"""

from __future__ import annotations

import json
import math
import pathlib
import random
from dataclasses import fields, replace
from decimal import Decimal
from typing import Any

import pytest

from gen.generate import build_plan, main
from gen.lifecycle import CleanDataset, WorkingCalendar, build_clean_dataset, load_working_calendar
from gen.noise import NOISE_REGISTRY, NoiseRates, apply_noise
from gen.profiles import PROFILES
from gen.truth import VarianceCategory
from tests import amplification

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CALENDAR_PATH = REPO_ROOT / "config" / "calendar_v1.yaml"

# Production parameters. Grain arithmetic depends on population size, so this
# suite must run at the size `make gen` actually ships.
SEED = 42
DAYS = 20
RECORDS = 5_000

PROFILES_BY_NAME = {profile.name: profile for profile in PROFILES}

# --- restated by hand ------------------------------------------------------

# The rate each injector is configured with, stated independently of gen.noise.
CONFIGURED_RATE = {
    "truncate_utr": Decimal("0.25"),
    "drop_utr": Decimal("0.12"),
    "merchant_name_variants": Decimal("0.30"),
    "unexplainable": Decimal("0.06"),
    "garbled_narration": Decimal("0.12"),
    "mixed_amount_formats": Decimal("0.08"),
    "duplicate_ledger_rows": Decimal("0.010"),
    "partial_captures": Decimal("0.020"),
    "delayed_settlement": Decimal("0.015"),
    "out_of_order_arrival": Decimal("0.04"),
    "split_settlement": Decimal("0.010"),
    "rounding_residual": Decimal("0.15"),
}

# Which population each rate multiplies against. THIS IS THE FIELD THE BUG
# LIVES IN. "batch" is ~39 targets; "case" is ~5,000; "row" is ~15,000. A rate
# is meaningless without it.
DECLARED_GRAIN = {
    "truncate_utr": "batch",
    "drop_utr": "batch",
    "merchant_name_variants": "batch",
    "garbled_narration": "batch",
    "unexplainable": "batch",
    "mixed_amount_formats": "row",
    "out_of_order_arrival": "row",
    "duplicate_ledger_rows": "case",
    "partial_captures": "case",
    "delayed_settlement": "case",
    "split_settlement": "case",
    "rounding_residual": "batch",
}

# Order-of-magnitude bounds for each grain, as a fraction of the relevant base
# population. Deliberately loose: this catches a rate applied to the WRONG
# population (a 100x error), not a target list that is merely filtered.
GRAIN_BOUNDS = {
    "batch": (0.5, 3.0),  # vs batch count - unexplainable emits 2 annotations/batch
    "case": (0.5, 1.5),  # vs case count  - some injectors filter to eligible cases
    "row": (1.0, 5.0),  # vs case count  - money-bearing rows across several tables
}

WITHHELD = {"garbled_narration", "split_settlement"}

# The taxonomy split now lives where PDD 6.1 says it does:
# settlesense.exceptions.taxonomy.VARIANCE_CATEGORIES. These local sets are a
# HAND RESTATEMENT of it, kept for the same reason the rates are restated -
# importing the engine's answer and then asserting against it would let a wrong
# classification agree with itself. test_the_restatement_matches_the_taxonomy
# below is where the two independent statements are made to meet.
VARIANCE_CATEGORIES = {
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

# Components of expected_net (PDD 6.1). Computed on EVERY case and never emitted
# as a variance, so a coverage assertion that sweeps them demands the generator
# invent a variance out of a fee. Checked against `deduction_categories`, never
# against `true_category`.
DEDUCTION_CATEGORIES = {
    VarianceCategory.MDR_FEE,
    VarianceCategory.GST_ON_FEE,
    VarianceCategory.REFUND_OFFSET,
}


def _zero_rates() -> NoiseRates:
    return NoiseRates(**{field.name: Decimal("0") for field in fields(NoiseRates)})


@pytest.fixture(scope="module")
def calendar() -> WorkingCalendar:
    return load_working_calendar(CALENDAR_PATH)


@pytest.fixture(scope="module")
def clean(calendar: WorkingCalendar) -> CleanDataset:
    base_date = calendar.next_working_day(calendar.window_start)
    plan = build_plan(RECORDS, DAYS, list(PROFILES))
    return build_clean_dataset(random.Random(SEED), plan, calendar, base_date, SEED)


@pytest.fixture(scope="module")
def populations(clean: CleanDataset) -> dict[str, int]:
    """The target count each injector actually samples, measured at rate 1.0.

    Measured rather than assumed: at rate 1 every target is picked, so the
    annotation count IS the population. That makes the grain observable instead
    of a claim in a comment.
    """
    sizes: dict[str, int] = {}
    for name in NOISE_REGISTRY:
        rates = replace(_zero_rates(), **{name: Decimal("1")})
        _, ledger = apply_noise(
            clean, random.Random(SEED), PROFILES_BY_NAME, rates, include_withheld=True
        )
        sizes[name] = ledger.counts[name]
    return sizes


@pytest.fixture(scope="module")
def realised(clean: CleanDataset) -> dict[str, int]:
    """Realised counts at the CONFIGURED rate, each injector run alone.

    Run alone so one injector's structural changes cannot alter another's
    target pool - this measures the rate, not the interaction.
    """
    counts: dict[str, int] = {}
    for name in NOISE_REGISTRY:
        rates = replace(_zero_rates(), **{name: CONFIGURED_RATE[name]})
        _, ledger = apply_noise(
            clean, random.Random(SEED), PROFILES_BY_NAME, rates, include_withheld=True
        )
        counts[name] = ledger.counts[name]
    return counts


def _generate(out: pathlib.Path, seed: int, *, withheld: bool) -> None:
    argv = [
        "--seed",
        str(seed),
        "--out",
        str(out),
        "--days",
        str(DAYS),
        "--records",
        str(RECORDS),
        "--calendar",
        str(CALENDAR_PATH),
    ]
    if withheld:
        argv.append("--include-withheld")
    assert main(argv) == 0, f"generation failed for seed {seed}"


def _shipped_truth(out: pathlib.Path, seed: int) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((out / f"truth_{seed}.json").read_text("utf-8"))
    return payload


# Category coverage is measured on the datasets AS SHIPPED, via the real CLI -
# not by re-running apply_noise in-process. The generator threads ONE rng
# through chain building and then noise, so a fresh random.Random(seed) handed
# to apply_noise samples a different stream and lands on different targets. An
# in-process reconstruction therefore reports coverage the shipped files do not
# have; this was observed here, showing UNEXPLAINED present when `make gen`
# emits none. What matters is what the engine is handed.
@pytest.fixture(scope="module")
def dev_truth(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """`make gen`: seed 42, withheld OFF. What engine development may see."""
    out = tmp_path_factory.mktemp("cov_dev")
    _generate(out, 42, withheld=False)
    return _shipped_truth(out, 42)


@pytest.fixture(scope="module")
def holdout_truth(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """`make gen-holdout`: seed 999, withheld ON. The scored dataset."""
    out = tmp_path_factory.mktemp("cov_holdout")
    _generate(out, 999, withheld=True)
    return _shipped_truth(out, 999)


# ===========================================================================
# 1. THE POSITIVE CONTROL - every configured noise type actually occurs
# ===========================================================================


@pytest.mark.noise_accounting
def test_every_configured_noise_type_occurs_at_least_once(
    realised: dict[str, int], populations: dict[str, int]
) -> None:
    """A configured injector that never fires is a category nobody is tested on.

    This is the guard the `unexplainable` bug walked past: the rate was set, the
    injector was registered, the pipeline ran it, and it produced nothing. There
    was no assertion that a configured thing must happen.
    """
    table = "\n".join(
        f"  {name:24} rate={CONFIGURED_RATE[name]!s:>6} "
        f"pop={populations[name]:>6} realised={realised[name]:>6}"
        for name in sorted(NOISE_REGISTRY)
    )
    print("\nrealised noise counts (each injector run alone, seed 42):\n" + table)

    silent = sorted(name for name in NOISE_REGISTRY if realised[name] == 0)
    assert not silent, (
        f"configured noise types that never fired: {silent}\n{table}\n"
        "A rate greater than zero that produces nothing means the rate was "
        "chosen against the wrong population. Check the grain."
    )


@pytest.mark.noise_accounting
def test_the_registry_is_not_empty(populations: dict[str, int]) -> None:
    """Guards the guard: iterating an empty registry proves nothing."""
    # Restated by hand, so a new injector must be declared here too - with its
    # rate AND its grain. An injector added to gen/ but not to this file is one
    # whose coverage nobody is asserting.
    assert len(NOISE_REGISTRY) == len(CONFIGURED_RATE), (
        f"registry has {len(NOISE_REGISTRY)} injectors, the restated rate table "
        f"has {len(CONFIGURED_RATE)}: {set(NOISE_REGISTRY) ^ set(CONFIGURED_RATE)}"
    )
    assert set(NOISE_REGISTRY) == set(CONFIGURED_RATE), (
        "the hand-restated rate table has drifted from the registry: "
        f"{set(NOISE_REGISTRY) ^ set(CONFIGURED_RATE)}"
    )


# ===========================================================================
# 2. Realised count sits in a binomial band around rate x population
# ===========================================================================


@pytest.mark.parametrize("name", sorted(CONFIGURED_RATE))
def test_realised_count_matches_rate_times_its_own_population(
    name: str, realised: dict[str, int], populations: dict[str, int]
) -> None:
    """expected = rate x THE POPULATION THAT INJECTOR SAMPLES - not the case count.

    Using the case count for a batch-grain injector is the whole bug: it makes a
    rate of 0.004 look like 20 expected firings when it is really 0.16.
    """
    population = populations[name]
    rate = float(CONFIGURED_RATE[name])
    expected = population * rate
    # Bernoulli per target, so the count is binomial. 3 sigma, floored at 3 so
    # the band stays meaningful for the ~39-target batch grain.
    sigma = math.sqrt(population * rate * (1 - rate))
    slack = max(3.0 * sigma, 3.0)
    low, high = expected - slack, expected + slack

    assert low <= realised[name] <= high, (
        f"{name}: realised {realised[name]} outside [{low:.1f}, {high:.1f}]; "
        f"expected {expected:.1f} = rate {rate} x population {population} "
        f"({DECLARED_GRAIN[name]}-grain)"
    )


# ===========================================================================
# 3. Every taxonomy category is faced at least once
# ===========================================================================


def _categories(truth: dict[str, Any]) -> set[str]:
    seen: set[str] = set()
    for group in ("cases", "batch_links", "row_variances"):
        for row in truth[group]:
            if row.get("true_category"):
                seen.add(row["true_category"])
    return seen


@pytest.mark.noise_accounting
def test_every_variance_category_occurs_with_all_noise_on(
    holdout_truth: dict[str, Any],
) -> None:
    """A category the engine never faces cannot be scored - in either direction.

    It cannot be got right and it cannot be got wrong, so any headline accuracy
    figure silently excludes it while reading as though it covered the taxonomy.
    """
    seen = _categories(holdout_truth)
    missing = sorted(c.value for c in VARIANCE_CATEGORIES if c.value not in seen)
    assert not missing, (
        f"variance categories that never occur, even with every injector on: {missing}\n"
        f"realised: {sorted(seen)}\n"
        "No generator input produces these, so the engine's handling of them is "
        "unmeasured and unmeasurable."
    )


@pytest.mark.noise_accounting
def test_every_non_withheld_category_occurs_in_the_tuned_dataset(
    dev_truth: dict[str, Any],
) -> None:
    """`make gen` is what engine development sees. SPLIT_SETTLEMENT is withheld.

    Withholding a category is a deliberate unknown-unknowns choice. A category
    that is absent by ACCIDENT is the opposite - it looks like coverage.
    """
    seen = _categories(dev_truth)
    expected = {c.value for c in VARIANCE_CATEGORIES} - {VarianceCategory.SPLIT_SETTLEMENT.value}
    missing = sorted(expected - seen)
    assert not missing, (
        f"categories absent from the DEV dataset: {missing}\n"
        f"realised: {sorted(seen)}\n"
        "The engine is developed against this dataset. A category missing here "
        "is one the engine is never exercised on before the held-out run."
    )


def test_deduction_categories_are_recorded_separately(holdout_truth: dict[str, Any]) -> None:
    """MDR_FEE/GST_ON_FEE/REFUND_OFFSET are deductions, never variances.

    They explain gross -> expected_net on a perfectly clean case, so they belong
    in `deduction_categories` and must NOT appear as `true_category`. Asserting
    them as variances would force the generator to invent a variance that is not
    one.
    """
    deductions: set[str] = set()
    for case in holdout_truth["cases"]:
        deductions.update(case["deduction_categories"])
    missing = sorted(c.value for c in DEDUCTION_CATEGORIES if c.value not in deductions)
    assert not missing, f"deduction categories never recorded: {missing}"

    variances = _categories(holdout_truth)
    leaked = sorted(c.value for c in DEDUCTION_CATEGORIES if c.value in variances)
    assert not leaked, f"deduction categories leaked into true_category: {leaked}"


def test_the_restatement_matches_the_engine_taxonomy() -> None:
    """gen-side restatement vs settlesense.exceptions.taxonomy (PDD 6.1).

    Two independent statements of the same classification, made to meet in
    exactly one place. If they disagree, one of them is wrong and every coverage
    assertion built on the wrong one has been sweeping the wrong set - silently,
    and in the flattering direction if a hard category was dropped.
    """
    from settlesense.exceptions.taxonomy import (
        DEDUCTION_CATEGORIES as ENGINE_DEDUCTIONS,
    )
    from settlesense.exceptions.taxonomy import (
        VARIANCE_CATEGORIES as ENGINE_VARIANCES,
    )

    assert {c.value for c in VARIANCE_CATEGORIES} == {c.value for c in ENGINE_VARIANCES}, (
        "the restated variance set disagrees with taxonomy.VARIANCE_CATEGORIES: "
        f"{ {c.value for c in VARIANCE_CATEGORIES} ^ {c.value for c in ENGINE_VARIANCES} }"
    )
    assert {c.value for c in DEDUCTION_CATEGORIES} == {c.value for c in ENGINE_DEDUCTIONS}, (
        "the restated deduction set disagrees with taxonomy.DEDUCTION_CATEGORIES"
    )


def test_the_taxonomy_is_fully_partitioned() -> None:
    """Guards the guard: no category may be silently outside both buckets.

    Without this, dropping a member from VARIANCE_CATEGORIES would make the
    coverage tests pass by checking one category fewer.
    """
    covered = {c.value for c in VARIANCE_CATEGORIES} | {c.value for c in DEDUCTION_CATEGORIES}
    everything = {c.value for c in VarianceCategory}
    assert covered == everything, (
        f"taxonomy members in neither bucket: {sorted(everything - covered)}; "
        f"unknown members claimed: {sorted(covered - everything)}"
    )


# ===========================================================================
# 4. Declared grain matches the population actually sampled
# ===========================================================================


@pytest.mark.parametrize("name", sorted(DECLARED_GRAIN))
def test_injector_grain_matches_its_declaration(
    name: str, populations: dict[str, int], clean: CleanDataset
) -> None:
    """The population an injector samples must match the grain its rate assumes.

    This is the structural form of the check. Counts drift with seeds and
    filters; grain does not. An injector that moves from batch grain to case
    grain changes its effective rate by a factor of ~130, and this is what
    notices.
    """
    base = {
        "batch": len(clean.batches),
        "case": len(clean.chains),
        "row": len(clean.chains),
    }[DECLARED_GRAIN[name]]
    low, high = GRAIN_BOUNDS[DECLARED_GRAIN[name]]
    population = populations[name]

    assert low * base <= population <= high * base, (
        f"{name} is declared {DECLARED_GRAIN[name]}-grain (base {base}) but samples "
        f"{population} targets, outside [{low * base:.0f}, {high * base:.0f}]. "
        f"Its rate of {CONFIGURED_RATE[name]} therefore means "
        f"{population * float(CONFIGURED_RATE[name]):.1f} expected firings, not "
        f"{base * float(CONFIGURED_RATE[name]):.1f}."
    )


def test_batch_and_case_grain_differ_by_two_orders_of_magnitude(clean: CleanDataset) -> None:
    """Why the trap exists at all, asserted rather than asserted in prose.

    If these populations were similar, a rate applied to the wrong one would be
    a small error. They are not: the same numeric rate means ~130x more firings
    at case grain than at batch grain.
    """
    ratio = len(clean.chains) / len(clean.batches)
    assert ratio > 50, (
        f"case:batch population ratio is only {ratio:.1f}; the grain trap this "
        "suite guards depends on them differing sharply"
    )


# ===========================================================================
# The rate-amplification rule, surveyed across every pair of injectors
# ===========================================================================


def test_every_injector_pair_is_classified_for_amplification(
    populations: dict[str, int],
) -> None:
    """Rule step 1, applied to the WHOLE registry rather than to four known pairs.

    The four interactions this suite tests were found one defect at a time.
    This computes the expected co-occurrence for every pair sharing a grain, so
    the next one is visible BEFORE it costs a debugging session.

    It does not demand a test for each - most pairs interact in no interesting
    way. It reports which are too sparse to be worth testing at production
    rates, which is the fact that has to be known before someone writes an
    interaction test and trusts the shipped rates.
    """
    sparse: list[str] = []
    dense: list[str] = []
    names = sorted(CONFIGURED_RATE)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if DECLARED_GRAIN[first] != DECLARED_GRAIN[second]:
                continue  # different populations; they cannot collide row-wise
            expected = amplification.expected_cooccurrence(
                populations[first], CONFIGURED_RATE[first], CONFIGURED_RATE[second]
            )
            line = f"{first} x {second}: {expected:.2f}"
            (sparse if expected < amplification.AMPLIFICATION_THRESHOLD else dense).append(line)

    print("\nsame-grain injector pairs, expected co-occurrence at production rates:")
    for line in sorted(sparse):
        print(f"    SPARSE (must amplify)  {line}")
    for line in sorted(dense):
        print(f"    dense  (may test live) {line}")

    assert sparse or dense, "no same-grain pairs found; the survey inspected nothing"
    assert sparse, (
        "every same-grain pair is now dense enough to test at production rates. "
        "If the rates really were raised that far, the amplified fixtures across "
        "this suite can be retired - deliberately, not by drift."
    )


@pytest.mark.noise_accounting
def test_the_cooccurrence_arithmetic_is_right() -> None:
    """FAULT INJECTION for the shared helper every amplified test now depends on.

    The whole rule rests on this multiplication. If it were wrong - a rate
    squared, a population dropped - every classification above would be wrong in
    the same direction, and the suite would report confident nonsense.
    """
    assert amplification.expected_cooccurrence(1000, Decimal("0.5"), Decimal("0.2")) == Decimal(
        "100.0"
    )
    assert amplification.expected_cooccurrence(39, Decimal("0.04"), Decimal("0.06")) == Decimal(
        "0.0936"
    )
    assert amplification.expected_cooccurrence(0, Decimal("1")) == Decimal("0")
    assert amplification.expected_cooccurrence(500, Decimal("0")) == Decimal("0")
    # Single-rate case is just the marginal expectation.
    assert amplification.expected_cooccurrence(5000, Decimal("0.02")) == Decimal("100.00")
    with pytest.raises(ValueError):
        amplification.expected_cooccurrence(-1, Decimal("0.5"))


@pytest.mark.noise_accounting
def test_record_fails_when_the_interaction_never_fired() -> None:
    """FAULT INJECTION: a zero co-occurrence must FAIL, not pass quietly.

    This is the rule's whole point. Before it existed, four tests reported
    success while asserting properties about empty sets.
    """
    with pytest.raises(AssertionError, match=r"interaction fired"):
        amplification.record(
            "deliberately-empty interaction",
            0,
            expected_at_production=Decimal("0.09"),
        )


@pytest.mark.noise_accounting
def test_record_fails_when_amplification_consumed_every_row() -> None:
    """FAULT INJECTION for the COROLLARY.

    An interaction that fired on every row and left no survivors has happened
    and proved nothing - the state seven of twelve narration pairs were in when
    they silently skipped.
    """
    with pytest.raises(AssertionError, match="consumed every"):
        amplification.record(
            "deliberately-total interaction",
            12,
            expected_at_production=Decimal("0.5"),
            survivors=0,
        )

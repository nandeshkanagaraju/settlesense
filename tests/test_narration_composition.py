"""Narration injectors must compose without destroying each other's damage.

Each narration injector is correct alone. The defect lived in their COMPOSITION.

`merchant_name_variants` identified the merchant name by exclusion - "every
token that is not NEFT and not the UTR". That definition holds only while the
UTR is recognisable. Once `truncate_utr` had shortened it, the leftover prefix
no longer looked like a UTR, was absorbed into the "name", and the unspacing
variant fused the two into a single token:

    NEFT 1B4F5CA6 BLUEPEAK FOODS SETTLEMENT   ->  correct
    NEFT 1B4F5CA6BLUEPEAKFOODS SETTLEMENT     ->  prefix destroyed

Nothing failed. The dataset looked plausible, the row still had a narration, and
`UTR_TRUNCATED_MAPPING` was still claimed in truth. But the prefix the engine was
supposed to recover no longer existed as a token, so the case was not hard - it
was UNSOLVABLE, and an unsolvable case is not a test. It is a guaranteed miss
that lowers the measured score for a reason unrelated to the engine.

The general rule, which is what test 2 pins: never define a token by what it is
not. Exclusion-based identification is only as stable as every other definition
in the system, so it breaks whenever anything upstream changes - silently, and
at a distance from the change.

Test 5 is the one that matters. Everything else checks that the prefix is
PRESENT; test 5 checks that it is still RECOVERABLE - that it actually prefixes
the UTR of the batch truth says it belongs to.
"""

from __future__ import annotations

import itertools
import pathlib
import random
from dataclasses import fields, replace
from decimal import Decimal

import pytest

import gen.noise
from gen.generate import build_plan
from gen.lifecycle import (
    CleanDataset,
    WorkingCalendar,
    bank_txn_id_for,
    build_clean_dataset,
    load_working_calendar,
)
from gen.noise import NoiseLedger, NoiseRates, apply_noise
from gen.profiles import PROFILES
from gen.truth import VarianceCategory
from tests import amplification

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CALENDAR_PATH = REPO_ROOT / "config" / "calendar_v1.yaml"

SEED = 42
DAYS = 8
RECORDS = 1_800
ZERO = Decimal("0.00")

PROFILES_BY_NAME = {profile.name: profile for profile in PROFILES}

# Restated by hand - the whole point is not to infer these.
MERCHANT_NAMES = ("AURORA RETAIL", "BLUEPEAK FOODS", "CARBON WORKS PVT LTD")

# Injectors that rewrite narration text.
NARRATION_NOISE = ("truncate_utr", "drop_utr", "merchant_name_variants", "garbled_narration")

# Injectors that deliberately make the UTR unrecoverable. A row they touched
# cannot be expected to retain a readable prefix, and demanding one would be
# asserting that the noise did not work.
DESTRUCTIVE = frozenset({"drop_utr", "garbled_narration"})

# THE INTERACTION: truncate_utr shortens a UTR, then merchant_name_variants
# rewrites the same narration. Both sample the bank-credit population.
PRODUCTION_TRUNCATE = Decimal("0.25")
PRODUCTION_MERCHANT = Decimal("0.30")
# THE COROLLARY, applied twice.
#
# At 1.0 a destructive partner touches every credit, `_analysable` filters them
# all out, and seven of twelve pairs skipped: the interaction occurred and left
# nothing to observe. Dropping to a symmetric 0.7 fixed the skips but left the
# thinnest pair (drop_utr -> truncate_utr) with ONE analysable row - one seed
# away from zero, which is where a test stops testing.
#
# So the rates are ASYMMETRIC. A destructive injector running first eats the
# UTRs a later truncate would need, so it is held lower while the non-destructive
# partner is pushed higher. Measured, not guessed:
#
#     symmetric 0.70 / 0.70   thinnest pair:  1 row
#     dest 0.35 / safe 0.90   thinnest pair:  4 rows
#     dest 0.25 / safe 0.95   thinnest pair:  5 rows
PAIR_RATE_DESTRUCTIVE = Decimal("0.25")
PAIR_RATE_SAFE = Decimal("0.95")


def test_production_rates_are_borderline(clean: CleanDataset) -> None:
    """Rule step 1, and an honest answer: this one is close to the line.

    truncate x merchant at production is ~2.9 expected co-occurrences over ~39
    credits. Below the threshold of 10, so amplification is required - but not
    by the wide margin the other three have. Computed rather than asserted in
    prose, so if the rates move the classification moves with them.
    """
    expected = amplification.expected_cooccurrence(
        len(clean.batches), PRODUCTION_TRUNCATE, PRODUCTION_MERCHANT
    )
    assert expected < amplification.AMPLIFICATION_THRESHOLD, (
        f"production co-occurrence is now {expected:.2f}; amplification is no "
        "longer required and the fixtures should be simplified deliberately."
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


def _run(clean: CleanDataset, **rates: Decimal) -> tuple[CleanDataset, NoiseLedger]:
    return apply_noise(
        clean, random.Random(SEED), PROFILES_BY_NAME, _rates(**rates), include_withheld=True
    )


def _touched(ledger: NoiseLedger) -> dict[str, set[str]]:
    """batch_id -> the narration noise types applied to its credit."""
    touched: dict[str, set[str]] = {}
    for annotation in ledger.annotations:
        if annotation.noise_type in NARRATION_NOISE:
            touched.setdefault(annotation.target_id, set()).add(annotation.noise_type)
    return touched


def _truncated_prefix(clean_utr: str, narration: str) -> str | None:
    """The surviving prefix of `clean_utr` in `narration`, as a whole token."""
    for token in narration.split():
        if token and clean_utr.startswith(token) and len(token) < len(clean_utr):
            return token
    return None


def _analysable(
    clean: CleanDataset, dataset: CleanDataset, ledger: NoiseLedger
) -> list[tuple[str, str, str]]:
    """(batch_id, true_utr, narration) for credits truncated but not destroyed."""
    original_utr = {batch.batch_id: batch.utr for batch in clean.batches}
    narration_by_batch = {
        bank_txn_id_for(batch.batch_id): batch.batch_id for batch in dataset.batches
    }
    narrations = {
        narration_by_batch[row.bank_txn_id]: row.narration
        for row in dataset.bank_rows
        if row.bank_txn_id in narration_by_batch
    }
    touched = _touched(ledger)
    return [
        (batch_id, original_utr[batch_id], narrations[batch_id])
        for batch_id, kinds in sorted(touched.items())
        if "truncate_utr" in kinds and not (kinds & DESTRUCTIVE) and batch_id in narrations
    ]


# ===========================================================================
# 1. Truncate then vary the name - the prefix stays a separate token
# ===========================================================================


def test_the_truncated_prefix_survives_a_merchant_name_variant(clean: CleanDataset) -> None:
    """The exact composition that broke, at full rate so every credit gets both."""
    dataset, ledger = _run(clean, truncate_utr=Decimal("1"), merchant_name_variants=Decimal("1"))
    rows = _analysable(clean, dataset, ledger)
    amplification.record(
        "truncate_utr x merchant_name_variants",
        len(rows),
        expected_at_production=amplification.expected_cooccurrence(
            len(clean.batches), PRODUCTION_TRUNCATE, PRODUCTION_MERCHANT
        ),
        survivors=len(rows),
    )

    fused: list[str] = []
    for batch_id, utr, narration in rows:
        if _truncated_prefix(utr, narration) is None:
            fused.append(f"{batch_id}: no prefix token of {utr} in {narration!r}")
    assert not fused, (
        f"{len(fused)} of {len(rows)} narration(s) lost the truncated prefix as a "
        "separate token:\n" + "\n".join(fused[:10])
    )


# ===========================================================================
# 2. The merchant name comes from the KNOWN set, never by exclusion
# ===========================================================================


def test_the_narration_merchant_name_derives_from_the_known_set(clean: CleanDataset) -> None:
    """Every rendered name must be a variant of a name we shipped.

    Reconstructed from the three known names rather than by taking whatever
    remains after removing NEFT and the UTR - which is the very definition that
    caused the defect.
    """
    dataset, _ = _run(clean, truncate_utr=Decimal("1"), merchant_name_variants=Decimal("1"))

    permitted: set[str] = set()
    for name in MERCHANT_NAMES:
        words = name.split()
        permitted.add(name)
        permitted.add(name.replace(" ", ""))
        permitted.add(f"{name} PVT")
        if name.endswith("PVT LTD"):
            permitted.add(name.replace(" PVT LTD", ""))
        if len(words) > 1:
            last = words[-1]
            squeezed = last[0] + "".join(c for c in last[1:] if c not in "AEIOU")
            permitted.add(" ".join([*words[:-1], squeezed[:4]]))

    unrecognised = [
        row.narration
        for row in dataset.bank_rows
        if not any(variant in row.narration for variant in permitted)
    ]
    assert not unrecognised, (
        f"{len(unrecognised)} narration(s) carry a merchant name matching no known "
        "variant, so the name was produced by exclusion rather than lookup:\n"
        + "\n".join(repr(n) for n in unrecognised[:10])
    )


def test_a_truncated_prefix_is_never_mistaken_for_the_merchant_name(
    clean: CleanDataset,
) -> None:
    """The mechanism, not just the symptom.

    The bug was that the prefix became part of the "name". Assert the rendered
    name never contains UTR characters by checking the name variant present in
    the narration does not also contain the prefix.
    """
    dataset, ledger = _run(clean, truncate_utr=Decimal("1"), merchant_name_variants=Decimal("1"))
    contaminated: list[str] = []
    for _, utr, narration in _analysable(clean, dataset, ledger):
        prefix = _truncated_prefix(utr, narration)
        if prefix is None:
            continue
        for token in narration.split():
            if token == prefix:
                continue
            if prefix in token:
                contaminated.append(f"token {token!r} absorbed prefix {prefix!r} in {narration!r}")
    assert not contaminated, "the prefix was absorbed into another token:\n" + "\n".join(
        contaminated[:10]
    )


# ===========================================================================
# 3. Every pair of narration injectors, in both orders
# ===========================================================================


@pytest.mark.parametrize(
    ("first", "second"),
    [pair for pair in itertools.permutations(NARRATION_NOISE, 2)],
)
def test_the_prefix_survives_every_ordered_pair(
    first: str, second: str, clean: CleanDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """12 ordered pairs. Order matters here: the defect only fired one way round.

    `truncate_utr` before `merchant_name_variants` broke; the reverse did not,
    because an intact UTR was still recognisable. Testing unordered pairs would
    have had a 50% chance of missing it entirely.

    Rates are asymmetric and below 1.0 - see PAIR_RATE_DESTRUCTIVE. At full rate
    a destructive partner touches every credit and the pair asserts nothing;
    seven of twelve pairs skipped that way in the first draft. Every pair now
    records its realised co-occurrence and fails at zero, so a pair that stops
    reaching the interaction fails instead of going quietly green.

    Every pair is also checked for FUSION, which is meaningful even when the
    pair produces no truncated prefix. No pair is allowed to assert nothing.
    """
    monkeypatch.setattr(gen.noise, "NOISE_ORDER", (first, second))
    rates = {
        name: (PAIR_RATE_DESTRUCTIVE if name in DESTRUCTIVE else PAIR_RATE_SAFE)
        for name in (first, second)
    }
    dataset, ledger = _run(clean, **rates)

    unspaced = {name.replace(" ", "") for name in MERCHANT_NAMES}
    fused = [
        f"token {token!r}"
        for row in dataset.bank_rows
        for token in row.narration.split()
        for name in unspaced
        if name in token and token != name
    ]
    assert not fused, f"{first} -> {second} fused a merchant name into a token:\n" + "\n".join(
        sorted(set(fused))[:5]
    )

    rows = _analysable(clean, dataset, ledger)
    if "truncate_utr" in {first, second}:
        # Only pairs containing truncate_utr can produce a prefix at all. For
        # those, the interaction MUST have been observed - a pair that quietly
        # yields no analysable row is the skip this rule forbids.
        amplification.record(
            f"pair {first} -> {second}",
            len(rows),
            minimum=3,
            expected_at_production=amplification.expected_cooccurrence(
                len(clean.batches), PRODUCTION_TRUNCATE, PRODUCTION_MERCHANT
            ),
            survivors=len(rows),
        )

    broken = [
        f"{batch_id}: {narration!r} has no prefix token of {utr}"
        for batch_id, utr, narration in rows
        if _truncated_prefix(utr, narration) is None
    ]
    assert not broken, f"{first} -> {second} destroyed the prefix:\n" + "\n".join(broken[:5])


def test_at_least_one_ordered_pair_actually_exercised_the_composition(
    clean: CleanDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the guard: every pair skipping would make the sweep meaningless."""
    monkeypatch.setattr(gen.noise, "NOISE_ORDER", ("truncate_utr", "merchant_name_variants"))
    dataset, ledger = _run(clean, truncate_utr=Decimal("1"), merchant_name_variants=Decimal("1"))
    assert len(_analysable(clean, dataset, ledger)) >= 5


# ===========================================================================
# 4. No token fuses a merchant name with a UTR fragment
# ===========================================================================


def test_no_token_fuses_a_merchant_name_and_a_utr_fragment(clean: CleanDataset) -> None:
    """The literal shape of the bug: '1B4F5CA6BLUEPEAKFOODS'.

    Checked over EVERY narration, not only truncated ones - an unspaced variant
    fused to an intact UTR would be the same defect wearing a longer prefix.
    """
    dataset, _ = _run(
        clean,
        truncate_utr=Decimal("1"),
        merchant_name_variants=Decimal("1"),
        drop_utr=Decimal("0.2"),
    )
    unspaced = {name.replace(" ", "") for name in MERCHANT_NAMES}

    fused: list[str] = []
    for row in dataset.bank_rows:
        for token in row.narration.split():
            for name in unspaced:
                if name in token and token != name:
                    fused.append(f"token {token!r} fuses {name!r} with other characters")
    assert not fused, "a merchant name is fused into a larger token:\n" + "\n".join(
        sorted(set(fused))[:10]
    )


# ===========================================================================
# 5. THE POINT - the prefix still identifies its true batch
# ===========================================================================


def test_every_truncated_prefix_still_matches_its_true_batch(clean: CleanDataset) -> None:
    """Present is not enough. It must still PREFIX the UTR truth assigns it.

    This is the difference between a hard case and an impossible one. A prefix
    that no longer matches its own batch cannot be recovered by any engine, so
    UTR_TRUNCATED_MAPPING would be scored as a guaranteed miss - measuring the
    generator, not the engine.
    """
    dataset, ledger = _run(clean, truncate_utr=Decimal("1"), merchant_name_variants=Decimal("1"))
    rows = _analysable(clean, dataset, ledger)
    assert rows, "no truncated credits to check"

    unrecoverable: list[str] = []
    for batch_id, utr, narration in rows:
        prefix = _truncated_prefix(utr, narration)
        if prefix is None:
            unrecoverable.append(f"{batch_id}: no prefix at all in {narration!r}")
        elif not utr.startswith(prefix):
            unrecoverable.append(f"{batch_id}: {prefix!r} does not prefix its true UTR {utr}")
        elif len(prefix) < 6:
            unrecoverable.append(f"{batch_id}: prefix {prefix!r} shorter than the declared 6 chars")
    assert not unrecoverable, (
        f"{len(unrecoverable)} of {len(rows)} truncated credit(s) are UNSOLVABLE, "
        "not merely hard:\n" + "\n".join(unrecoverable[:10])
    )


def test_a_prefix_truth_calls_resolvable_identifies_exactly_one_batch(
    clean: CleanDataset,
) -> None:
    """`resolvable=True` is a claim about the data, and must be true of it.

    truncate_utr sets resolvable when exactly one batch shares the prefix. If
    that were wrong, the eval would count a genuinely ambiguous case as one the
    engine should have resolved - penalising a correct abstention, which SDD 4.3
    makes the right answer when candidates are close.
    """
    dataset, ledger = _run(clean, truncate_utr=Decimal("1"))
    all_utrs = [batch.utr for batch in clean.batches]
    narration_of = {
        batch.batch_id: next(
            row.narration
            for row in dataset.bank_rows
            if row.bank_txn_id == bank_txn_id_for(batch.batch_id)
        )
        for batch in dataset.batches
    }
    utr_of = {batch.batch_id: batch.utr for batch in clean.batches}

    checked = 0
    wrong: list[str] = []
    for annotation in ledger.of_type("truncate_utr"):
        assert annotation.category == VarianceCategory.UTR_TRUNCATED_MAPPING
        batch_id = annotation.target_id
        if batch_id not in narration_of:
            continue
        prefix = _truncated_prefix(utr_of[batch_id], narration_of[batch_id])
        if prefix is None:
            continue
        checked += 1
        sharers = sum(1 for utr in all_utrs if utr.startswith(prefix))
        if annotation.resolvable and sharers != 1:
            wrong.append(f"{batch_id}: claimed resolvable but {sharers} batches share {prefix!r}")
        if not annotation.resolvable and sharers == 1:
            wrong.append(f"{batch_id}: claimed ambiguous but {prefix!r} is unique")
    assert checked > 0, "no truncate_utr annotation could be checked"
    assert not wrong, "resolvable flags disagree with the data:\n" + "\n".join(wrong[:10])

"""A split settlement must price each part on its own gross.

SDD 3.1 makes a split payment ONE ReconciliationCase holding several PAYMENT
lines. The stale-line defect: the case kept fee, tax and expected_net derived
from a single line, so after a split those case-level figures described part A
only. Conservation still closed at the batch level, so nothing failed - the
error was invisible until the engine recomputed a fee from a line's own gross
and disagreed with truth.

That is the shape that makes this dangerous. The engine's arithmetic pass (SDD
4.2 P3) recomputes fee and tax from the rate table for EVERY line. A line whose
fee was derived as "the remainder after part A" cannot survive that: the
remainder is not equal to rate x gross_b except by coincidence. Truth would then
mark a correct engine answer wrong, on exactly the cases the split noise makes
hardest.

So the decisive test here is 4: recompute every fee independently, the way the
engine will, and demand line-by-line agreement. Tests 1-3 check the parts sum
correctly, which the BUGGY version also did.

The rate table is restated by hand. Importing gen.profiles would let a wrong
rate agree with itself.
"""

from __future__ import annotations

import pathlib
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields, replace
from decimal import ROUND_HALF_UP, Decimal

import pytest

from gen.generate import build_plan
from gen.lifecycle import (
    CleanDataset,
    SettlementLine,
    SettlementLineType,
    WorkingCalendar,
    build_clean_dataset,
    load_working_calendar,
)
from gen.noise import NoiseLedger, NoiseRates, apply_noise
from gen.profiles import PROFILES
from gen.truth import Truth, VarianceCategory, build_truth

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CALENDAR_PATH = REPO_ROOT / "config" / "calendar_v1.yaml"

SEED = 42
DAYS = 20
RECORDS = 5_000

Q2 = Decimal("0.01")
ZERO = Decimal("0.00")

# Restated by hand - see module docstring.
RATES: Mapping[str, Mapping[str, str]] = {
    "profile_a": {"card": "0.0200", "upi": "0.0000", "netbanking": "0.0175", "wallet": "0.0210"},
    "profile_b": {"card": "0.0235", "upi": "0.0000", "netbanking": "0.0190", "wallet": "0.0200"},
    "profile_c": {"card": "0.0180", "upi": "0.0000", "netbanking": "0.0160", "wallet": "0.0195"},
}
GST = Decimal("0.18")

PROFILES_BY_NAME = {profile.name: profile for profile in PROFILES}


def q(value: Decimal) -> Decimal:
    return value.quantize(Q2, rounding=ROUND_HALF_UP)


def expected_fee(profile: str, method: str, gross: Decimal) -> Decimal:
    """Exactly what the engine's arithmetic pass will compute (SDD 4.2 P3)."""
    return q(gross * Decimal(RATES[profile][method]))


def expected_tax(fee: Decimal) -> Decimal:
    return q(fee * GST)


@pytest.fixture(scope="module")
def calendar() -> WorkingCalendar:
    return load_working_calendar(CALENDAR_PATH)


@pytest.fixture(scope="module")
def noisy(calendar: WorkingCalendar) -> tuple[CleanDataset, NoiseLedger, Truth]:
    """Withheld noise ON - split_settlement is a withheld type (SDD 5).

    Built in-process rather than via the CLI. That samples a different rng
    stream than a shipped run, which matters for COVERAGE claims but not here:
    any split the injector produces is a valid split to price-check, and this
    gives direct access to profile and method per chain.
    """
    base_date = calendar.next_working_day(calendar.window_start)
    plan = build_plan(RECORDS, DAYS, list(PROFILES))
    clean = build_clean_dataset(random.Random(SEED), plan, calendar, base_date, SEED)
    # Splits only; other injectors would re-price lines for unrelated reasons
    # and blur which arithmetic is under test.
    rates = replace(
        NoiseRates(**{field.name: ZERO for field in fields(NoiseRates)}),
        split_settlement=Decimal("0.05"),
    )
    dataset, ledger = apply_noise(
        clean, random.Random(SEED), PROFILES_BY_NAME, rates, include_withheld=True
    )
    truth = build_truth(dataset, calendar, PROFILES_BY_NAME, SEED, noise=ledger)
    return dataset, ledger, truth


@pytest.fixture(scope="module")
def payment_lines(
    noisy: tuple[CleanDataset, NoiseLedger, Truth],
) -> dict[str, list[SettlementLine]]:
    dataset, _, _ = noisy
    grouped: dict[str, list[SettlementLine]] = defaultdict(list)
    for line in dataset.settlement_lines:
        if line.line_type is SettlementLineType.PAYMENT:
            grouped[line.payment_id].append(line)
    return grouped


@pytest.fixture(scope="module")
def split_payment_ids(payment_lines: dict[str, list[SettlementLine]]) -> list[str]:
    return sorted(pid for pid, lines in payment_lines.items() if len(lines) >= 2)


@pytest.mark.noise_accounting
def test_there_are_splits_to_examine(split_payment_ids: list[str]) -> None:
    """Guards the guard: every assertion below iterates the split set.

    An empty split set makes tests 1-5 pass by checking nothing, which is the
    same failure mode as the grain bug that hid UNEXPLAINED.
    """
    assert len(split_payment_ids) >= 20, (
        f"only {len(split_payment_ids)} split payments; too few to exercise the "
        "per-part pricing path"
    )


# ===========================================================================
# 1. Parts sum to the payment gross, exactly
# ===========================================================================


def test_split_parts_sum_to_the_payment_gross(
    noisy: tuple[CleanDataset, NoiseLedger, Truth],
    payment_lines: dict[str, list[SettlementLine]],
    split_payment_ids: list[str],
) -> None:
    """No paisa created or destroyed by the split itself."""
    dataset, _, _ = noisy
    captured = {row.payment_id: row.captured for row in dataset.payment_rows}
    bad: list[str] = []
    for payment_id in split_payment_ids:
        lines = payment_lines[payment_id]
        total = q(sum((line.gross for line in lines), start=ZERO))
        if total != captured[payment_id]:
            bad.append(
                f"{payment_id}: {len(lines)} parts sum to {total}, captured {captured[payment_id]}"
            )
    assert not bad, "split parts do not sum to the captured amount:\n" + "\n".join(bad[:10])


# ===========================================================================
# 2 & 4. Each part priced on ITS OWN gross - the decisive test
# ===========================================================================


def test_every_split_line_is_priced_on_its_own_gross(
    noisy: tuple[CleanDataset, NoiseLedger, Truth],
    payment_lines: dict[str, list[SettlementLine]],
    split_payment_ids: list[str],
) -> None:
    """Recompute fee and tax the way the engine will, line by line.

    THIS is the test that catches the stale-line bug. A part whose fee was
    handed to it as "the remainder after part A" fails here even though the
    parts still sum correctly, because the remainder is not rate x gross_b.
    """
    dataset, _, _ = noisy
    method_of = {row.payment_id: row.method for row in dataset.payment_rows}
    profile_of = {chain.payment.payment_id: chain.profile_name for chain in dataset.chains}

    bad: list[str] = []
    for payment_id in split_payment_ids:
        profile, method = profile_of[payment_id], method_of[payment_id]
        for line in sorted(payment_lines[payment_id], key=lambda line: line.settlement_id):
            want_fee = expected_fee(profile, method, line.gross)
            want_tax = expected_tax(want_fee)
            want_net = q(line.gross - want_fee - want_tax)
            if (line.fee, line.tax, line.net) != (want_fee, want_tax, want_net):
                bad.append(
                    f"{line.settlement_id} ({profile}/{method}, gross {line.gross}): "
                    f"fee {line.fee} != {want_fee}, tax {line.tax} != {want_tax}, "
                    f"net {line.net} != {want_net}"
                )
    assert not bad, (
        f"{len(bad)} split line(s) disagree with the rate table applied to their "
        "own gross - the engine's arithmetic pass would reject them:\n" + "\n".join(bad[:10])
    )


@pytest.mark.noise_accounting
def test_a_split_part_fee_is_not_the_remainder(
    noisy: tuple[CleanDataset, NoiseLedger, Truth],
    payment_lines: dict[str, list[SettlementLine]],
    split_payment_ids: list[str],
) -> None:
    """Name the wrong implementation directly, not just its symptom.

    Pricing part B as `unsplit_fee - fee_a` is the natural way to make the parts
    sum to the original fee. It is wrong, and the two differ by at most a paisa,
    so it survives every sum-based check. Where they differ at all, the recorded
    fee must be the rate-derived one.
    """
    dataset, _, _ = noisy
    method_of = {row.payment_id: row.method for row in dataset.payment_rows}
    profile_of = {chain.payment.payment_id: chain.profile_name for chain in dataset.chains}
    captured = {row.payment_id: row.captured for row in dataset.payment_rows}

    examined = 0
    for payment_id in split_payment_ids:
        profile, method = profile_of[payment_id], method_of[payment_id]
        lines = sorted(payment_lines[payment_id], key=lambda line: line.settlement_id)
        unsplit_fee = expected_fee(profile, method, captured[payment_id])
        parts_fee = q(sum((line.fee for line in lines), start=ZERO))
        if parts_fee == unsplit_fee:
            continue  # rounding happened to agree; nothing to distinguish here
        examined += 1
        for line in lines:
            assert line.fee == expected_fee(profile, method, line.gross), (
                f"{line.settlement_id}: fee {line.fee} looks like a remainder, not "
                f"{expected_fee(profile, method, line.gross)} = rate x {line.gross}"
            )

    assert examined > 0, (
        "no split produced a fee total differing from the unsplit fee, so the "
        "remainder-vs-rate distinction was never actually exercised"
    )


# ===========================================================================
# 3. Case-level expected_net is the sum over ALL its lines
# ===========================================================================


def test_case_expected_net_equals_the_sum_of_its_line_nets(
    noisy: tuple[CleanDataset, NoiseLedger, Truth], split_payment_ids: list[str]
) -> None:
    """expected_net == sum(PAYMENT nets) + sum(REFUND nets), exact Decimal.

    The brief said "sum of all its PAYMENT line nets". That holds only for a
    split with no refund; SDD 3.1b defines expected_net as
    gross - fee - tax - refunds, and a REFUND line net is already negative. The
    refund term is included here so a split case that ALSO has a refund is
    covered rather than excluded - which is the harder case and the one where a
    stale line would hide longest.
    """
    dataset, _, _ = noisy
    by_payment: dict[str, list[SettlementLine]] = defaultdict(list)
    for line in dataset.settlement_lines:
        by_payment[line.payment_id].append(line)

    bad: list[str] = []
    with_refund = 0
    for chain in dataset.chains:
        payment_id = chain.payment.payment_id
        if payment_id not in split_payment_ids:
            continue
        lines = by_payment[payment_id]
        if any(line.line_type is SettlementLineType.REFUND for line in lines):
            with_refund += 1
        total = q(sum((line.net for line in lines), start=ZERO))
        if chain.expected_net != total:
            bad.append(f"{payment_id}: expected_net {chain.expected_net} != line sum {total}")

    assert not bad, "case expected_net is stale:\n" + "\n".join(bad[:10])
    print(f"\nsplit cases examined: {len(split_payment_ids)}, of which {with_refund} also refund")


def test_case_fee_and_tax_aggregate_every_payment_line(
    noisy: tuple[CleanDataset, NoiseLedger, Truth], split_payment_ids: list[str]
) -> None:
    """The stale-line bug in its original form: case.fee came from part A alone."""
    dataset, _, _ = noisy
    bad: list[str] = []
    for chain in dataset.chains:
        if chain.payment.payment_id not in split_payment_ids:
            continue
        assert len(chain.payment_lines) >= 2, (
            f"{chain.payment.payment_id}: the chain knows only "
            f"{len(chain.payment_lines)} payment line(s) but the dataset holds 2+. "
            "The case never learned about the split, so its fee, tax and "
            "expected_net still describe part A alone."
        )
        want_fee = q(sum((line.fee for line in chain.payment_lines), start=ZERO))
        want_tax = q(sum((line.tax for line in chain.payment_lines), start=ZERO))
        if (chain.fee, chain.tax) != (want_fee, want_tax):
            bad.append(
                f"{chain.payment.payment_id}: case fee {chain.fee}/{want_fee}, "
                f"tax {chain.tax}/{want_tax}"
            )
    assert not bad, "case fee/tax do not aggregate all payment lines:\n" + "\n".join(bad[:10])


# ===========================================================================
# 5 & 6. Truth marks the split, and a refund is NOT a split
# ===========================================================================


def test_truth_marks_split_cases_with_two_payment_line_ids(
    noisy: tuple[CleanDataset, NoiseLedger, Truth], split_payment_ids: list[str]
) -> None:
    """SPLIT_SETTLEMENT and len(payment_line_ids) >= 2 must agree, both ways.

    Checked in both directions: a case flagged SPLIT_SETTLEMENT with one line is
    a mislabel, and a case with two lines and no flag is a missing label. One
    direction alone would let the two drift apart.
    """
    _, _, truth = noisy
    splits = set(split_payment_ids)

    mismatched: list[str] = []
    for case in truth.cases:
        flagged = case.true_category == VarianceCategory.SPLIT_SETTLEMENT
        multi = len(case.payment_line_ids) >= 2
        if case.payment_id in splits and not multi:
            mismatched.append(f"{case.payment_id}: split in data, {len(case.payment_line_ids)} ids")
        if multi and not flagged:
            mismatched.append(f"{case.payment_id}: {len(case.payment_line_ids)} ids, not flagged")
        if flagged and not multi:
            mismatched.append(f"{case.payment_id}: flagged SPLIT_SETTLEMENT with 1 payment line")
    assert not mismatched, "truth and data disagree about splits:\n" + "\n".join(mismatched[:10])


def test_one_case_per_split_payment(
    noisy: tuple[CleanDataset, NoiseLedger, Truth], split_payment_ids: list[str]
) -> None:
    """ALWAYS one case per payment (SDD 3.1). Two would double-count Population A."""
    _, _, truth = noisy
    counts: dict[str, int] = defaultdict(int)
    for case in truth.cases:
        counts[case.payment_id] += 1
    doubled = sorted(pid for pid in split_payment_ids if counts[pid] != 1)
    assert not doubled, f"split payments producing != 1 case: {doubled[:10]}"


def test_a_refund_line_does_not_make_a_case_a_split(
    noisy: tuple[CleanDataset, NoiseLedger, Truth],
) -> None:
    """A REFUND line references a payment without being a settlement of it.

    payment_line_ids counts PAYMENT lines only (SDD 3.1). Counting
    settlement_line_ids instead would mark every refunded case as a split -
    which is why the SDD names the field the case matcher must use.
    """
    dataset, _, truth = noisy
    refunded = {
        line.payment_id
        for line in dataset.settlement_lines
        if line.line_type is SettlementLineType.REFUND
    }
    payment_line_count: dict[str, int] = defaultdict(int)
    for line in dataset.settlement_lines:
        if line.line_type is SettlementLineType.PAYMENT:
            payment_line_count[line.payment_id] += 1

    single_with_refund = [
        case
        for case in truth.cases
        if case.payment_id in refunded and payment_line_count[case.payment_id] == 1
    ]
    assert single_with_refund, "no refunded non-split cases found; nothing was checked"

    bad = [
        f"{case.payment_id}: {len(case.payment_line_ids)} payment_line_ids, "
        f"{len(case.settlement_line_ids)} settlement_line_ids, "
        f"category {case.true_category}"
        for case in single_with_refund
        if len(case.payment_line_ids) != 1
        or case.true_category == VarianceCategory.SPLIT_SETTLEMENT
    ]
    assert not bad, "a refund was mistaken for a split:\n" + "\n".join(bad[:10])

    sample = single_with_refund[0]
    assert len(sample.settlement_line_ids) >= 2, (
        "the refunded case should hold both lines in settlement_line_ids; "
        "otherwise this test cannot distinguish the two fields"
    )

"""Ground truth must know by construction, never by parsing.

The generator builds every batch and every bank credit, so it KNOWS which
credit belongs to which batch. If truth instead recovers that link the way the
engine has to - by finding a UTR in the narration - then the noise layer damages
ground truth and the engine's task in the same stroke. Truth degrades in lockstep
with difficulty: `truncate_utr` blinds the answer key exactly where it blinds the
candidate, the un-recoverable rows silently leave the denominator, and the engine
posts its highest score on precisely the rows it could not see.

That is the circular evaluation this project exists to avoid, and it cannot be
caught by looking at accuracy numbers - a circular measurement looks GOOD. It has
to be caught structurally, which is what these tests do.

The invariant: BATCH_TO_BANK is a function of the construction, so no amount of
narration damage may change it. Not the edge count, not the edge set, not one
edge.
"""

from __future__ import annotations

import ast
import pathlib
import random
from dataclasses import fields, replace
from datetime import date
from decimal import Decimal

import pytest

from gen.generate import build_plan
from gen.lifecycle import (
    CleanDataset,
    WorkingCalendar,
    build_clean_dataset,
    load_working_calendar,
)
from gen.noise import NoiseRates, apply_noise
from gen.profiles import PROFILES
from gen.truth import EdgeType, TruthEdge, build_truth

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CALENDAR_PATH = REPO_ROOT / "config" / "calendar_v1.yaml"
TRUTH_SOURCE = REPO_ROOT / "gen" / "truth.py"

SEED = 42
DAYS = 4
RECORDS = 240

# A narration with no UTR, no merchant name, and nothing else to parse. Every
# heuristic a narration-reading truth builder could use is absent.
BLANK_NARRATION = "CREDIT"

# Injectors that damage narration text WITHOUT removing a bank row. Each is
# presentation-only: the money is untouched and the batch->bank relationship
# still exists, it is merely harder to see.
NARRATION_DAMAGING = ("truncate_utr", "drop_utr", "merchant_name_variants", "garbled_narration")

PROFILES_BY_NAME = {profile.name: profile for profile in PROFILES}


def _calendar() -> WorkingCalendar:
    return load_working_calendar(CALENDAR_PATH)


def _clean_dataset(calendar: WorkingCalendar) -> tuple[CleanDataset, date]:
    rng = random.Random(SEED)
    plan = build_plan(RECORDS, DAYS, list(PROFILES))
    base_date = calendar.next_working_day(calendar.window_start)
    return build_clean_dataset(rng, plan, calendar, base_date, SEED), base_date


def _zero_rates() -> NoiseRates:
    """All injectors off. The baseline every damaged run is compared against."""
    return NoiseRates(**{field.name: Decimal("0") for field in fields(NoiseRates)})


def _batch_to_bank(edges: tuple[TruthEdge, ...]) -> set[tuple[str, str]]:
    return {(e.src_id, e.dst_id) for e in edges if e.edge_type is EdgeType.BATCH_TO_BANK}


@pytest.fixture(scope="module")
def calendar() -> WorkingCalendar:
    return _calendar()


@pytest.fixture(scope="module")
def baseline(calendar: WorkingCalendar) -> tuple[CleanDataset, set[tuple[str, str]]]:
    """The undamaged run: every narration intact, no noise applied."""
    dataset, _ = _clean_dataset(calendar)
    truth = build_truth(dataset, calendar, PROFILES_BY_NAME, SEED)
    links = _batch_to_bank(truth.edges)
    assert links, "baseline produced no BATCH_TO_BANK edges - nothing would be proven"
    return dataset, links


# ===========================================================================
# 1. Every narration blanked - truth is unchanged
# ===========================================================================


def test_truth_survives_total_narration_destruction(
    calendar: WorkingCalendar, baseline: tuple[CleanDataset, set[tuple[str, str]]]
) -> None:
    """The strongest form of the test: remove ALL parseable text, keep all truth.

    This is deliberately harsher than any injector. If truth reads narration at
    all - regex, prefix scan, substring, fuzzy - it cannot survive this, because
    there is nothing left to read. Only construction-side linkage survives.
    """
    dataset, expected = baseline
    blanked = replace(
        dataset,
        bank_rows=tuple(replace(row, narration=BLANK_NARRATION) for row in dataset.bank_rows),
    )
    assert all(row.narration == BLANK_NARRATION for row in blanked.bank_rows)

    truth = build_truth(blanked, calendar, PROFILES_BY_NAME, SEED)
    actual = _batch_to_bank(truth.edges)

    assert actual == expected, (
        f"BATCH_TO_BANK changed when narrations were blanked: "
        f"{len(expected - actual)} link(s) lost, {len(actual - expected)} invented. "
        "Truth is reading the narration instead of recording what it built."
    )


def test_blanking_narration_changes_the_dataset_but_not_truth(
    calendar: WorkingCalendar, baseline: tuple[CleanDataset, set[tuple[str, str]]]
) -> None:
    """Guards the guard: prove the damage was real.

    If `replace` silently failed, or the baseline narrations already held no
    UTR, the test above would pass by damaging nothing.
    """
    dataset, _ = baseline
    assert any(row.narration != BLANK_NARRATION for row in dataset.bank_rows), (
        "baseline narrations were already blank; the damage test proves nothing"
    )
    # And the UTRs really were recoverable before blanking.
    utrs = {batch.utr for batch in dataset.batches}
    found = sum(1 for row in dataset.bank_rows if any(utr in row.narration for utr in utrs))
    assert found > 0, "no baseline narration carried its UTR - nothing was destroyed"


# ===========================================================================
# 2. truncate_utr at 100%
# ===========================================================================


def test_truncate_utr_at_full_rate_leaves_the_link_count_unchanged(
    calendar: WorkingCalendar, baseline: tuple[CleanDataset, set[tuple[str, str]]]
) -> None:
    """Every UTR truncated. The link set must be identical, not merely similar."""
    dataset, expected = baseline
    rates = replace(_zero_rates(), truncate_utr=Decimal("1"))
    noisy, ledger = apply_noise(
        dataset, random.Random(SEED), PROFILES_BY_NAME, rates, include_withheld=True
    )
    assert ledger.counts["truncate_utr"] > 0, "truncate_utr did not fire at rate 1"

    truth = build_truth(noisy, calendar, PROFILES_BY_NAME, SEED, noise=ledger)
    actual = _batch_to_bank(truth.edges)

    assert len(actual) == len(expected), (
        f"link count moved with narration damage: {len(expected)} -> {len(actual)}"
    )
    assert actual == expected, "link SET changed even though the count matched"


# ===========================================================================
# 3. Static guard: truth.py never reads narration
# ===========================================================================


def _truth_ast() -> ast.Module:
    return ast.parse(TRUTH_SOURCE.read_text("utf-8"))


def test_truth_module_imports_no_regex() -> None:
    tree = _truth_ast()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "re" not in imported, "gen/truth.py imports `re`; truth must not parse text"


def test_truth_module_never_reads_the_narration_field() -> None:
    """No attribute access, no keyword, no string mentioning narration.

    Attribute access is the real signal - `row.narration` is the only way to
    reach the text - but the string check also catches a dict-shaped read such
    as `row["narration"]`.
    """
    tree = _truth_ast()
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "narration":
            offenders.append(f"line {node.lineno}: attribute access .narration")
        elif isinstance(node, ast.Constant) and node.value == "narration":
            offenders.append(f"line {node.lineno}: string literal 'narration'")
    assert not offenders, (
        "gen/truth.py reads the narration:\n"
        + "\n".join(offenders)
        + "\nTruth must record what it constructed (bank_txn_id_for), never what "
        "it can parse. Narration is what the noise layer damages."
    )


def test_truth_module_does_not_split_or_substring_search() -> None:
    """`.split()` and `x in y` on text are how a parser gets rebuilt by accident.

    Membership tests against containers are legitimate and everywhere, so this
    checks only the two shapes that indicate text scanning: a `.split()` call,
    and a `in` test whose right side names something narration-like.
    """
    tree = _truth_ast()
    offenders: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"split", "rsplit", "partition", "findall", "search", "match"}
        ):
            offenders.append(f"line {node.lineno}: .{node.func.attr}()")
        if isinstance(node, ast.Compare):
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                if not isinstance(op, ast.In):
                    continue
                name = (
                    comparator.attr
                    if isinstance(comparator, ast.Attribute)
                    else getattr(comparator, "id", "")
                )
                if "narration" in name or "utr" in name.lower():
                    offenders.append(f"line {node.lineno}: `in {name}`")
    assert not offenders, "gen/truth.py scans text:\n" + "\n".join(offenders)


def test_the_static_guard_would_catch_a_parser() -> None:
    """Positive control: the AST checks fire on code that DOES parse narration.

    An AST scan that matches nothing passes for the same reason a build target
    with no tests goes green.
    """
    parser = ast.parse(
        "import re\n"
        "def link(batch, row):\n"
        "    if batch.utr in row.narration:\n"
        "        return row.narration.split()[0]\n"
        "    return re.search('[A-Z]+', row.narration)\n"
    )
    attrs = [n for n in ast.walk(parser) if isinstance(n, ast.Attribute) and n.attr == "narration"]
    splits = [
        n
        for n in ast.walk(parser)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"split", "search"}
    ]
    assert attrs, "the .narration detector matched nothing on a known parser"
    assert splits, "the .split()/.search() detector matched nothing on a known parser"


# ===========================================================================
# 4. Link count invariant across 0%..100% for every narration-damaging type
# ===========================================================================


@pytest.mark.parametrize("noise_type", NARRATION_DAMAGING)
def test_link_count_is_invariant_across_all_rates(
    noise_type: str, calendar: WorkingCalendar, baseline: tuple[CleanDataset, set[tuple[str, str]]]
) -> None:
    """0% to 100% in 10% steps. The link set never moves.

    `unexplainable` is deliberately absent from NARRATION_DAMAGING: it WITHHOLDS
    a bank credit, so the link genuinely ceases to exist and truth must record
    that. The distinction is the point - presentation noise may never move a
    link, structural noise may, and conflating them would either hide this bug
    or forbid a legitimate one.
    """
    dataset, expected = baseline
    moved: list[str] = []
    fired = 0

    for step in range(11):
        rate = Decimal(step) / Decimal(10)
        rates = replace(_zero_rates(), **{noise_type: rate})
        noisy, ledger = apply_noise(
            dataset, random.Random(SEED), PROFILES_BY_NAME, rates, include_withheld=True
        )
        fired += ledger.counts.get(noise_type, 0)
        truth = build_truth(noisy, calendar, PROFILES_BY_NAME, SEED, noise=ledger)
        actual = _batch_to_bank(truth.edges)
        if actual != expected:
            moved.append(
                f"rate {rate}: {len(expected)} -> {len(actual)} links "
                f"({len(expected - actual)} lost, {len(actual - expected)} invented)"
            )

    assert not moved, f"{noise_type} moved BATCH_TO_BANK truth:\n" + "\n".join(moved)
    assert fired > 0, (
        f"{noise_type} never fired across any rate; the invariant held vacuously. "
        "Check the injector's grain - a batch-grain type needs enough batches."
    )

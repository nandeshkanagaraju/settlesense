"""Cross-seed ID independence - the held-out set must share no identifier.

If `_stable_id` omits the seed from its canonical tuple, two datasets built from
different seeds emit byte-identical order/payment/settlement/batch IDs. The
held-out set then stops being held out in the one respect that matters most for
an evaluation: an engine that memorises an ID scores free marks on data it was
never supposed to have seen, and the measurement reports a competence that does
not exist.

The defect is per-entity. A single call site that forgets the seed collides for
exactly one entity type while the other five stay clean, so a test that samples
one table - as the M1 suite did, checking `payment_id` alone - passes while the
dataset is compromised. Every entity is therefore checked here, and the call
sites are additionally audited STATICALLY: a sampled test can only fail on a
code path the sample happened to reach, and a rarely-taken minting branch is
precisely where this defect survives.
"""

from __future__ import annotations

import ast
import csv
import itertools
import pathlib
from collections.abc import Mapping

import pytest

from gen.generate import main
from gen.lifecycle import _stable_id, _stable_utr, bank_txn_id_for

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CALENDAR = REPO_ROOT / "config" / "calendar_v1.yaml"
LIFECYCLE = REPO_ROOT / "gen" / "lifecycle.py"

# Small but complete: every one of the six tables is non-empty at this size,
# including refunds and batches, which are the sparse ones.
DAYS = 4
RECORDS = 240

# entity -> (table, column). Every ID-bearing entity in the v1 schema (SDD 3.0
# names six tables; `utr` is carried on the batch row and checked alongside).
ID_COLUMNS: Mapping[str, tuple[str, str]] = {
    "order_id": ("ledger", "order_id"),
    "payment_id": ("payments", "payment_id"),
    "refund_id": ("refunds", "refund_id"),
    "settlement_id": ("settlements", "settlement_id"),
    "batch_id": ("batches", "batch_id"),
    "bank_txn_id": ("bank", "bank_txn_id"),
    "utr": ("batches", "utr"),
}


def _generate(out: pathlib.Path, seed: int) -> None:
    rc = main(
        [
            "--seed",
            str(seed),
            "--out",
            str(out),
            "--days",
            str(DAYS),
            "--records",
            str(RECORDS),
            "--calendar",
            str(CALENDAR),
            "--include-withheld",
        ]
    )
    assert rc == 0, f"generator self-check failed on seed {seed}"


def _id_sets(out: pathlib.Path) -> dict[str, set[str]]:
    """Every ID the generator wrote, grouped by entity."""
    sets: dict[str, set[str]] = {}
    for entity, (table, column) in ID_COLUMNS.items():
        values: set[str] = set()
        for path in sorted(out.glob(f"day*_{table}.csv")):
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row[column]:
                        values.add(row[column])
        assert values, f"{entity}: no IDs found - the test would pass vacuously"
        sets[entity] = values
    return sets


@pytest.fixture(scope="module")
def seed_42(tmp_path_factory: pytest.TempPathFactory) -> dict[str, set[str]]:
    out = tmp_path_factory.mktemp("ids42")
    _generate(out, 42)
    return _id_sets(out)


@pytest.fixture(scope="module")
def seed_999(tmp_path_factory: pytest.TempPathFactory) -> dict[str, set[str]]:
    out = tmp_path_factory.mktemp("ids999")
    _generate(out, 999)
    return _id_sets(out)


# ===========================================================================
# 1. The two shipped seeds share no identifier, for any entity
# ===========================================================================


def test_dev_and_holdout_id_sets_are_disjoint(
    seed_42: dict[str, set[str]], seed_999: dict[str, set[str]]
) -> None:
    """Exactly empty, not merely small.

    "Mostly disjoint" is not a property - one shared payment_id is one case the
    engine can recognise rather than reconcile. The assertion names the entity
    and shows offending IDs, because the failure mode is a single call site and
    the entity identifies it.
    """
    collisions = {
        entity: sorted(seed_42[entity] & seed_999[entity])
        for entity in ID_COLUMNS
        if seed_42[entity] & seed_999[entity]
    }
    assert not collisions, "\n".join(
        f"{entity}: {len(ids)} shared ID(s) across seeds 42/999, e.g. {ids[:3]}"
        for entity, ids in sorted(collisions.items())
    )


def test_every_entity_was_actually_compared(
    seed_42: dict[str, set[str]], seed_999: dict[str, set[str]]
) -> None:
    """Guards the guard: disjointness over an empty set is vacuously true.

    If a table stopped being written, the disjointness test above would go green
    by comparing nothing at all - the same failure as a build target that passes
    by running no tests.
    """
    for entity in ID_COLUMNS:
        assert seed_42[entity], f"seed 42 wrote no {entity}"
        assert seed_999[entity], f"seed 999 wrote no {entity}"


# ===========================================================================
# 2. Determinism is not the price of independence
# ===========================================================================


def test_same_seed_produces_identical_id_sets(tmp_path: pathlib.Path) -> None:
    """The obvious wrong fix for a cross-seed collision is entropy (D10).

    Salting IDs with anything unstable - a uuid, a clock, an address - makes the
    sets disjoint across seeds AND across runs of the same seed. This test is
    what stops that fix from looking correct.
    """
    first, second = tmp_path / "a", tmp_path / "b"
    _generate(first, 42)
    _generate(second, 42)

    left, right = _id_sets(first), _id_sets(second)
    for entity in ID_COLUMNS:
        assert left[entity] == right[entity], (
            f"{entity}: seed 42 did not reproduce across two runs; "
            f"{len(left[entity] ^ right[entity])} IDs differ"
        )


# ===========================================================================
# 3. Property: all 28 pairs over 8 seeds
# ===========================================================================

PROPERTY_SEEDS = (1, 42, 99, 123, 999, 2026, 31337, 65535)


@pytest.fixture(scope="module")
def property_id_sets(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[int, dict[str, set[str]]]:
    root = tmp_path_factory.mktemp("props")
    result: dict[int, dict[str, set[str]]] = {}
    for seed in PROPERTY_SEEDS:
        out = root / f"seed{seed}"
        _generate(out, seed)
        result[seed] = _id_sets(out)
    return result


def test_all_pairwise_seed_combinations_are_disjoint(
    property_id_sets: dict[int, dict[str, set[str]]],
) -> None:
    """8 seeds -> 28 unordered pairs, each checked for every entity.

    Two seeds agreeing is a bug; two seeds agreeing only for `refund_id` is the
    same bug, reachable through one call site. The pair count is asserted so a
    shrunken seed list cannot quietly weaken the property.
    """
    pairs = list(itertools.combinations(PROPERTY_SEEDS, 2))
    assert len(pairs) == 28, f"expected 28 unordered pairs, got {len(pairs)}"

    failures: list[str] = []
    for left_seed, right_seed in pairs:
        for entity in ID_COLUMNS:
            shared = property_id_sets[left_seed][entity] & property_id_sets[right_seed][entity]
            if shared:
                failures.append(
                    f"seeds {left_seed}/{right_seed} share {len(shared)} {entity}: "
                    f"{sorted(shared)[:3]}"
                )
    assert not failures, "\n".join(failures)


# ===========================================================================
# 4. The seed is in the canonical tuple BY CONSTRUCTION
# ===========================================================================
#
# The brief asked for "_stable_id with identical arguments under two different
# seeds -> different output". That cannot hold and must not: _stable_id is a
# pure function of its arguments with no ambient state, so identical arguments
# returning different digests would be a D10 violation, not evidence of a fix.
# The seed enters AS an argument. Requirement 4 is therefore split: a
# differential proving the seed argument changes the digest, and a static audit
# proving every call site actually passes one.


def test_seed_argument_changes_the_digest() -> None:
    """The seed is load-bearing in the tuple, not decorative."""
    for prefix in ("ORD", "PAY", "RFD", "SET", "BAT"):
        left = _stable_id(prefix, 42, "profile_a", 1, 7)
        right = _stable_id(prefix, 999, "profile_a", 1, 7)
        assert left != right, f"{prefix}: seed is not part of the canonical tuple"
    assert _stable_utr(42, "profile_a", "2026-09-01", "BAT_X") != _stable_utr(
        999, "profile_a", "2026-09-01", "BAT_X"
    )


def test_stable_id_is_pure_in_its_arguments() -> None:
    """The other half of D10: no ambient entropy anywhere in the digest."""
    assert _stable_id("ORD", 42, "profile_a", 1, 7) == _stable_id("ORD", 42, "profile_a", 1, 7)
    assert bank_txn_id_for("BAT_ABC") == bank_txn_id_for("BAT_ABC")


def test_derived_ids_inherit_the_seed() -> None:
    """bank_txn_id takes no seed argument; it must still vary with the seed.

    It is derived from batch_id, which is seeded. That indirection is the one
    call site that reads like an oversight, so it is pinned explicitly rather
    than argued for in a comment.
    """
    batch_42 = _stable_id("BAT", 42, "profile_a", "2026-09-01")
    batch_999 = _stable_id("BAT", 999, "profile_a", "2026-09-01")
    assert batch_42 != batch_999
    assert bank_txn_id_for(batch_42) != bank_txn_id_for(batch_999)


# --- static audit of every call site ---------------------------------------

MINTERS = {"_stable_id", "_stable_utr"}

# Call sites that legitimately take no `seed` argument because they derive from
# an identifier that is already seeded. Each entry must be justified, and the
# derivation is pinned by test_derived_ids_inherit_the_seed above.
DERIVED_FROM_SEEDED_ID = {"batch_id"}


def _call_sites() -> list[tuple[int, str, set[str]]]:
    """(lineno, source, names referenced in args) for every minting call."""
    source = LIFECYCLE.read_text("utf-8")
    tree = ast.parse(source)
    sites: list[tuple[int, str, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name not in MINTERS:
            continue
        referenced = {
            sub.id for arg in node.args for sub in ast.walk(arg) if isinstance(sub, ast.Name)
        }
        sites.append((node.lineno, ast.get_source_segment(source, node) or "", referenced))
    return sorted(sites)


@pytest.mark.charter_guard
def test_the_audit_finds_every_known_call_site() -> None:
    """Guards the guard: an AST scan that matches nothing passes silently."""
    sites = _call_sites()
    assert len(sites) >= 8, f"expected at least 8 minting call sites, found {len(sites)}"


@pytest.mark.charter_guard
def test_every_id_minting_call_site_passes_the_seed() -> None:
    """One missed call site collides for exactly one entity - find it statically.

    Sampling generated data can only fail on a branch the sample reached. A
    minting site behind a rare condition (a refund that only some chains carry,
    a split that only fires at one rate) would slip through every test above.
    This one reads the source instead.
    """
    offenders = [
        f"gen/lifecycle.py:{lineno}: {src}"
        for lineno, src, names in _call_sites()
        if "seed" not in names and not (names & DERIVED_FROM_SEEDED_ID)
    ]
    assert not offenders, (
        "ID minting call sites with no seed in the canonical tuple:\n"
        + "\n".join(offenders)
        + "\nEvery generated ID must be a function of the seed, directly or via "
        "an already-seeded ID listed in DERIVED_FROM_SEEDED_ID."
    )

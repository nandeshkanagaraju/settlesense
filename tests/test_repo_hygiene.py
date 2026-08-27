"""Repository and manifest invariants that were fixed once and must not drift back.

Each of these was a real defect, corrected before the freeze, and each is the
kind that returns silently:

  .DS_Store was committed and had to be removed with `git rm --cached` before
  the freeze. Nothing would have objected to it coming back.

  GENERATOR_MANIFEST.json said `table_count: 7`, copied from the SDD's own
  example, which predates the removal of chargebacks from v1. Six is correct.
  A count that disagrees with the files on disk is a claim nobody checked.

  EdgeType was briefly described as having six values. It has five; the only
  candidate sixth is a dispute edge, and disputes are out of v1 scope (SDD 3.0).
  Inventing one to fill the gap would open a closed taxonomy.

DISCIPLINE. Every guard here is paired with the fault injection that makes it
fire, marked `hygiene` so `make fault-report` counts it. A check you cannot
break on demand is one you know has not complained - which is not the same as
knowing it works. That distinction is invisible in a green run.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
from typing import Any

import pytest

from gen.truth import EdgeType

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "GENERATOR_MANIFEST.json"

# Artifacts that must never be tracked. Editor droppings, caches, and the
# virtualenv - none of them are anyone's source, and all of them are noisy.
MUST_IGNORE = (".DS_Store", "__pycache__", ".pytest_cache", "*.pyc", ".venv")

# Every top-level entry the repository is allowed to track. SDD section 2 fixes
# the layout; data/ and reports/ are generated outputs that section 8 commits;
# the three specification documents and two manifests are named by SDD 5.1 and
# by this project's own drift guard.
DECLARED_TREE = frozenset(
    {
        ".gitignore",
        "EVAL_SET_MANIFEST.json",
        "GENERATOR_MANIFEST.json",
        "LIMITATIONS.md",
        "Makefile",
        "README.md",
        "SPEC_MANIFEST.json",
        "SettleSense_BUILD_PROMPTS.md",
        "SettleSense_PDD.md",
        "SettleSense_SDD.md",
        "config",
        "data",
        "eval",
        # Declared in SDD section 2 from the start; it only became TRACKED at
        # M7, when the first fixtures were recorded into it. An empty
        # directory is invisible to git, so this guard could not fire until
        # there was something in it.
        "fixtures",
        "gen",
        "pyproject.toml",
        "reports",
        "settlesense",
        "tests",
    }
)

FREEZE_TAG = "m1f-generator-freeze-2"

# manifest's logical table name -> the day{N}_<suffix>.csv family it is written
# as. Declared explicitly rather than derived by stripping plurals: a fuzzy
# normaliser can make two different names collide and then agree with itself,
# which is the failure this whole file is about. `settlement_lines` is written
# as day{N}_settlements.csv, and only a stated mapping makes that checkable.
TABLE_FILE_NAMES = {
    "ledger_rows": "ledger",
    "payment_rows": "payments",
    "refund_rows": "refunds",
    "settlement_lines": "settlements",
    "settlement_batches": "batches",
    "bank_rows": "bank",
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _tracked() -> list[str]:
    return [line for line in _git("ls-files").splitlines() if line]


def _manifest() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(MANIFEST_PATH.read_text("utf-8"))
    return payload


# ===========================================================================
# 1. .gitignore covers the known noise
# ===========================================================================


@pytest.mark.parametrize("pattern", MUST_IGNORE)
def test_gitignore_covers(pattern: str) -> None:
    text = (REPO_ROOT / ".gitignore").read_text("utf-8")
    entries = {line.strip().rstrip("/") for line in text.splitlines() if line.strip()}
    assert pattern.rstrip("/") in entries, (
        f"{pattern} is not in .gitignore. It was committed once already and had "
        "to be removed with `git rm --cached` before the freeze."
    )


def test_no_ignored_artifact_is_actually_tracked() -> None:
    """.gitignore does not untrack what is already tracked.

    A pattern added AFTER a file was committed has no effect on it at all - the
    file stays in the index, and `git status` stays clean while lying. That is
    exactly how .DS_Store survived into the tree.
    """
    offenders = [
        path
        for path in _tracked()
        if pathlib.PurePath(path).name == ".DS_Store"
        or "__pycache__" in pathlib.PurePath(path).parts
        or path.endswith(".pyc")
        or path.startswith((".venv/", ".pytest_cache/"))
    ]
    assert not offenders, (
        "ignored artifact(s) are tracked despite .gitignore:\n"
        + "\n".join(offenders[:10])
        + "\nAdding a pattern does not untrack an existing file; use "
        "`git rm --cached`."
    )


@pytest.mark.hygiene
def test_the_tracked_artifact_scan_would_catch_one() -> None:
    """FAULT INJECTION for the scan above.

    The predicate is fed a file list containing each artifact it is supposed to
    reject. Without this, the test passes on a repository that happens to be
    clean and on a predicate that can never be true.
    """
    planted = [
        "gen/.DS_Store",
        "settlesense/__pycache__/config.cpython-314.pyc",
        "tests/stale.pyc",
        ".venv/bin/python",
        ".pytest_cache/CACHEDIR.TAG",
        "gen/noise.py",  # innocent, must NOT be flagged
    ]
    caught = [
        path
        for path in planted
        if pathlib.PurePath(path).name == ".DS_Store"
        or "__pycache__" in pathlib.PurePath(path).parts
        or path.endswith(".pyc")
        or path.startswith((".venv/", ".pytest_cache/"))
    ]
    assert len(caught) == 5, f"the scan caught {caught}, expected all five artifacts"
    assert "gen/noise.py" not in caught, "the scan flags legitimate source files"


# ===========================================================================
# 2. The frozen tree is fully committed
# ===========================================================================


def test_the_frozen_generator_has_no_uncommitted_state() -> None:
    """`git status --porcelain gen/` is empty.

    The manifest hashes the files on DISK. If any of them differ from what was
    committed, the recorded hash describes a state that exists nowhere in
    history, and the freeze names a commit that cannot reproduce it.
    """
    dirty = [line for line in _git("status", "--porcelain", "gen").splitlines() if line]
    assert not dirty, (
        "gen/ has uncommitted changes, so GENERATOR_MANIFEST.json hashes a state "
        "that is not in any commit:\n" + "\n".join(dirty)
    )


def test_the_freeze_tag_points_at_the_manifest_commit_or_later() -> None:
    """The tag must be able to reproduce what the manifest claims."""
    tags = {line.strip() for line in _git("tag").splitlines() if line.strip()}
    assert FREEZE_TAG in tags, f"{FREEZE_TAG} does not exist; tags are {sorted(tags)}"
    recorded = str(_manifest()["generator_commit"])
    contained = _git("merge-base", "--is-ancestor", recorded, FREEZE_TAG) == ""
    assert contained, (
        f"{FREEZE_TAG} does not contain the manifest's generator_commit {recorded[:12]}"
    )


@pytest.mark.hygiene
def test_a_dirty_generator_would_be_detected() -> None:
    """FAULT INJECTION: `git status --porcelain` really does report a change.

    Verified against a path git is definitely watching, without touching gen/ -
    dirtying the frozen tree to prove the frozen tree can be seen as dirty would
    be a poor trade.
    """
    probe = REPO_ROOT / "tests" / ".hygiene_probe"
    probe.write_text("transient\n", encoding="utf-8")
    try:
        reported = _git("status", "--porcelain", "tests")
        assert ".hygiene_probe" in reported, (
            "git status did not report a new file, so the cleanliness check "
            f"above cannot detect one either. Output was: {reported!r}"
        )
    finally:
        probe.unlink()
    assert ".hygiene_probe" not in _git("status", "--porcelain", "tests")


# ===========================================================================
# 3. Nothing outside the declared tree is tracked
# ===========================================================================


def test_only_declared_top_level_paths_are_tracked() -> None:
    """SDD section 2 fixes the layout; anything else is undeclared.

    Stray top-level files are how a scratch script, a downloaded fixture or a
    second copy of a config ends up shipped and later mistaken for canon.
    """
    top_level = {path.split("/", 1)[0] for path in _tracked()}
    undeclared = sorted(top_level - DECLARED_TREE)
    assert not undeclared, (
        f"tracked path(s) outside the declared repository tree: {undeclared}\n"
        "Either add them to SDD section 2 and to DECLARED_TREE deliberately, or "
        "remove them."
    )


@pytest.mark.hygiene
def test_an_undeclared_path_would_be_rejected() -> None:
    """FAULT INJECTION for the allow-list."""
    planted = {"gen", "tests", "scratch.py", "Downloads"}
    undeclared = sorted(planted - DECLARED_TREE)
    assert undeclared == ["Downloads", "scratch.py"], (
        f"the allow-list check found {undeclared}; it must reject both strays "
        "and accept both declared paths"
    )


def test_the_declared_tree_matches_what_is_actually_tracked() -> None:
    """Guards the allow-list from the other side.

    An entry listed here but absent from the repository means the list has
    drifted into wishful thinking, and it would silently permit that path to
    reappear later with no review.
    """
    top_level = {path.split("/", 1)[0] for path in _tracked()}
    missing = sorted(DECLARED_TREE - top_level)
    assert not missing, (
        f"DECLARED_TREE lists path(s) that are not tracked: {missing}. An "
        "allow-list entry for something that does not exist permits it in "
        "advance, unreviewed."
    )


# ===========================================================================
# 4. table_count is 6, and equals what the generator actually writes
# ===========================================================================


def _tables_on_disk(root: pathlib.Path) -> set[str]:
    """The distinct table suffixes in day{N}_<table>.csv, measured not assumed."""
    found: set[str] = set()
    for path in root.glob("day*_*.csv"):
        match = re.fullmatch(r"day\d+_(.+)\.csv", path.name)
        if match:
            found.add(match.group(1))
    return found


def test_manifest_table_count_is_six_not_seven() -> None:
    manifest = _manifest()
    assert manifest["table_count"] == 6, (
        f"table_count is {manifest['table_count']}, not 6. Chargebacks are out "
        "of v1 scope (SDD 3.0), so there is no seventh table. SDD 5.1's example "
        "manifest said 7 until revision 2 and now says 6."
    )
    tables = manifest["tables"]
    assert len(tables) == 6
    assert not any("dispute" in name or "chargeback" in name for name in tables)


def test_manifest_table_count_equals_the_files_the_generator_writes() -> None:
    """The declared count must match the artifact, not just itself.

    `table_count == len(tables)` only proves the manifest is internally tidy. It
    is the day{N}_*.csv files on disk that the engine will read, and a manifest
    that disagrees with them is a claim nobody checked.
    """
    data = REPO_ROOT / "data" / "dev"
    if not data.is_dir() or not any(data.glob("day*_*.csv")):
        pytest.skip("no generated data present; run `make gen` first")

    on_disk = _tables_on_disk(data)
    manifest = _manifest()

    assert len(on_disk) == manifest["table_count"], (
        f"the generator writes {len(on_disk)} table(s) {sorted(on_disk)} but the "
        f"manifest declares {manifest['table_count']}"
    )
    assert sorted(TABLE_FILE_NAMES.values()) == sorted(on_disk), (
        f"the file families on disk {sorted(on_disk)} do not match the declared "
        f"correspondence {sorted(TABLE_FILE_NAMES.values())}"
    )
    declared_names = sorted(str(name) for name in manifest["tables"])
    assert sorted(TABLE_FILE_NAMES) == declared_names, (
        f"the manifest's logical names {declared_names} do not match the "
        f"declared correspondence {sorted(TABLE_FILE_NAMES)}"
    )


@pytest.mark.hygiene
def test_the_table_scan_counts_correctly() -> None:
    """FAULT INJECTION: a seventh table would be seen, a stray file would not."""
    assert _tables_on_disk(REPO_ROOT / "does-not-exist") == set()
    fake = {"day1_ledger.csv", "day1_bank.csv", "day12_disputes.csv", "truth_42.json"}
    parsed = {m.group(1) for name in fake if (m := re.fullmatch(r"day\d+_(.+)\.csv", name))}
    assert parsed == {"ledger", "bank", "disputes"}, parsed
    assert "truth_42.json" not in parsed, "the scan counts non-table files"


# ===========================================================================
# 5. EdgeType has exactly five values
# ===========================================================================


def test_edge_type_has_exactly_five_values() -> None:
    """Five, not six. A dispute edge is out of v1 scope (SDD 3.0).

    The sixth was briefly described as existing. It does not, and inventing one
    to match a description would open a closed taxonomy - the same error as
    adding a CLEAN member so that every case has a category.
    """
    assert len(EdgeType) == 5, f"EdgeType has {len(EdgeType)} values: {[e.value for e in EdgeType]}"
    assert {e.name for e in EdgeType} == {
        "ORDER_TO_PAYMENT",
        "PAYMENT_TO_SETTLEMENT",
        "SETTLEMENT_TO_BATCH",
        "BATCH_TO_BANK",
        "PAYMENT_TO_REFUND",
    }
    assert not any("DISPUTE" in e.name or "CHARGEBACK" in e.name for e in EdgeType)


def test_the_manifest_agrees_with_the_enum() -> None:
    """Two statements of the same fact must not drift apart.

    The manifest is published; the enum is what runs. If they disagree, the
    published claim is the one a reader believes.
    """
    manifest_edges = _manifest()["edge_types"]
    assert sorted(manifest_edges) == sorted(e.value for e in EdgeType), (
        f"manifest declares {sorted(manifest_edges)}, enum defines "
        f"{sorted(e.value for e in EdgeType)}"
    )


@pytest.mark.hygiene
def test_the_edge_count_check_is_not_vacuous() -> None:
    """FAULT INJECTION: prove the assertion distinguishes 5 from 6."""
    import enum

    class SixEdges(enum.StrEnum):
        ORDER_TO_PAYMENT = "order_to_payment"
        PAYMENT_TO_SETTLEMENT = "payment_to_settlement"
        SETTLEMENT_TO_BATCH = "settlement_to_batch"
        BATCH_TO_BANK = "batch_to_bank"
        PAYMENT_TO_REFUND = "payment_to_refund"
        PAYMENT_TO_DISPUTE = "payment_to_dispute"

    assert len(SixEdges) == 6, "the control enum is not actually six-valued"
    assert any("DISPUTE" in e.name for e in SixEdges), "the dispute detector matches nothing"
    assert len(EdgeType) != len(SixEdges), "the real enum has drifted to six values"

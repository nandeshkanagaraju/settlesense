"""M1F - the generator freeze is enforced, not merely announced.

PDD 8.1 publishes the claim that the adversarial generator was frozen at a
recorded commit BEFORE engine development began. A claim like that is worth
nothing if nothing checks it: the whole point of an independent generator is
that it cannot be quietly nudged to make the engine look better.

So the manifest records a content hash of the frozen path, and this test fails
if `gen/` changes afterwards. If a change to `gen/` is genuinely necessary, this
failure is the intended outcome - it forces an explicit re-freeze with a new
commit and a stated reason, rather than a silent edit.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
from typing import Any

import pytest

pytestmark = pytest.mark.determinism

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "GENERATOR_MANIFEST.json"
GEN = REPO_ROOT / "gen"


def _manifest() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(MANIFEST_PATH.read_text("utf-8"))
    return payload


def _frozen_files() -> list[pathlib.Path]:
    return sorted(p for p in GEN.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def _tree_hash() -> str:
    digest = hashlib.sha256()
    for path in _frozen_files():
        digest.update(str(path.relative_to(REPO_ROOT)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def test_manifest_exists_and_names_a_real_commit() -> None:
    manifest = _manifest()
    commit = manifest["generator_commit"]
    assert isinstance(commit, str), "generator_commit must be a real sha in the MANIFEST"
    assert re.fullmatch(r"[0-9a-f]{40}", commit), f"not a git sha: {commit!r}"


def test_generator_tree_matches_the_frozen_hash() -> None:
    """gen/ has not changed since the freeze.

    A failure here is not necessarily a bug - it means someone edited the
    generator after Gate 2. That is allowed only as a deliberate re-freeze:
    commit the change, then regenerate GENERATOR_MANIFEST.json so the recorded
    hash and commit name the new state. What is NOT allowed is the edit going
    unrecorded while the README still advertises the old freeze.
    """
    manifest = _manifest()
    actual = _tree_hash()
    if actual == manifest["gen_tree_sha256"]:
        return

    recorded: dict[str, str] = manifest["gen_files"]
    current = {
        str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in _frozen_files()
    }
    changed = sorted(k for k in recorded.keys() & current.keys() if recorded[k] != current[k])
    added = sorted(current.keys() - recorded.keys())
    removed = sorted(recorded.keys() - current.keys())
    pytest.fail(
        "gen/ has changed since the M1F freeze.\n"
        f"  modified: {changed}\n"
        f"  added:    {added}\n"
        f"  removed:  {removed}\n"
        "If this change is intended, re-freeze explicitly: commit it, then "
        "regenerate GENERATOR_MANIFEST.json and say why in the commit message."
    )


def test_manifest_declares_six_tables_not_seven() -> None:
    """Chargebacks are out of v1 scope, so there is no seventh table (SDD 3.0).

    The SDD's own example manifest in 5.1 says `table_count: 7`, which predates
    the removal of disputes. Six is correct.
    """
    manifest = _manifest()
    assert manifest["table_count"] == 6
    assert len(manifest["tables"]) == 6
    assert "dispute_rows" not in manifest["tables"]


def test_manifest_declares_five_edge_types() -> None:
    manifest = _manifest()
    assert len(manifest["edge_types"]) == 5
    assert not any("dispute" in e for e in manifest["edge_types"])


def test_manifest_records_both_seeds_and_the_calendar_version() -> None:
    manifest = _manifest()
    assert manifest["seeds"] == {"dev": 42, "holdout": 999}
    assert manifest["calendar_version"] == "calendar_v1"


def test_manifest_names_the_withheld_noise_types() -> None:
    manifest = _manifest()
    assert sorted(manifest["withheld_noise_types"]) == [
        "garbled_narration",
        "split_settlement",
    ]


def test_truth_files_never_carry_the_commit_hash() -> None:
    """The hash cannot exist before the commit that creates it (SDD 5.1).

    Truth files are written during generation and are never rewritten
    afterwards, so `generator_commit` is null in every one of them. The real
    hash lives only in the manifest.
    """
    commit = _manifest()["generator_commit"]
    truth_files = sorted(REPO_ROOT.glob("data/**/truth_*.json"))
    if not truth_files:
        pytest.skip("no generated truth files present; run `make gen` first")
    for path in truth_files:
        payload = json.loads(path.read_text("utf-8"))
        assert payload["generator_commit"] is None, f"{path.name} carries a commit hash"
        assert commit not in path.read_text("utf-8"), f"{path.name} leaks the freeze hash"


# ===========================================================================
# The post-M3 rule: no third freeze once the engine exists
# ===========================================================================

ENGINE_DIR = REPO_ROOT / "settlesense" / "matching"


def _engine_is_implemented() -> bool:
    """True once settlesense/matching/ holds more than docstring-only stubs.

    Measured by code, not by a date or a flag: a rule keyed on a calendar
    deadline is one nobody notices passing, and a flag is one that gets set
    wrong. A module with executable statements in it is the observable fact
    that M3 has begun.
    """
    for path in sorted(ENGINE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text("utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # module docstring
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            return True
    return False


def test_no_further_re_freeze_once_the_engine_exists() -> None:
    """LIMITATIONS.md commits to exactly one re-freeze. This is what holds it.

    The generator was re-frozen once, before any engine code existed - a
    correction that cannot be tuning, because there was nothing to tune against.
    After M3 begins that reasoning stops applying: a change to the generator can
    no longer be distinguished from one that flatters a measured result, whatever
    the intent behind it.

    A rule recorded only in prose is the ceremonial version of the discipline it
    describes. So it is checked: with the engine implemented, a third freeze
    generation fails the build and the defect goes in LIMITATIONS.md instead.
    """
    generation = _manifest().get("freeze_generation", 1)
    if not _engine_is_implemented():
        assert generation <= 2, (
            f"freeze generation {generation} before the engine exists. Two is "
            "already the documented maximum; a third needs its own justification "
            "in LIMITATIONS.md."
        )
        return
    assert generation <= 2, (
        f"freeze generation {generation} with settlesense/matching/ implemented. "
        "LIMITATIONS.md commits to no re-freeze after M3 begins: past that point "
        "a generator correction cannot be distinguished from tuning the data to "
        "the result. Record the defect in LIMITATIONS.md and live with it."
    )


def test_the_engine_detector_is_honest() -> None:
    """Guards the guard, in both directions.

    A detector stuck at False would let a third freeze through forever; one
    stuck at True would fire on the stubs that exist today.
    """
    assert ENGINE_DIR.is_dir(), "settlesense/matching/ is missing entirely"
    assert not _engine_is_implemented(), (
        "the engine reads as implemented already. If M3 has genuinely started, "
        "delete this assertion in the same commit that starts it - deliberately, "
        "so the no-re-freeze rule becomes live rather than silently bypassed."
    )

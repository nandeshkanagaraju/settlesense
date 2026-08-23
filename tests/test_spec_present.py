"""The specification must be present, referenced accurately, and never edited.

Three failures this guards, all of which have already happened here:

  ABSENT     SettleSense_BUILD_PROMPTS.md is named normative by SDD 5.1 and was
             missing from the tree for the whole of M0-M1F. Work proceeded
             against a document nobody could open.

  STALE      The repo's SDD lagged the authoritative revision. A brief cited
             "SDD 3.1, Population C"; the repo copy declares two populations, so
             a correct implementation was reported as a spec deviation and a
             real requirement looked like an invention. Neither side was wrong -
             they were reading different documents.

  EDITED     ruff 0.16.4 reformats Python blocks inside Markdown. Unguarded, it
             rewrites 93 lines of the SDD, de-indenting continuation comments
             until "Case matching uses THIS field" refers to nothing.

A dangling section reference is the quiet member of that set. It does not fail
anything; it just sends the next reader to a section that says something else,
or nothing at all.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tomllib

import pytest

pytestmark = pytest.mark.determinism

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "SPEC_MANIFEST.json"

DOCUMENTS = {
    "SDD": REPO_ROOT / "SettleSense_SDD.md",
    "PDD": REPO_ROOT / "SettleSense_PDD.md",
    "BUILD_PROMPTS": REPO_ROOT / "SettleSense_BUILD_PROMPTS.md",
}

SCANNED_DIRS = ("gen", "settlesense", "tests", "eval")

# "SDD 3.1a", "SDD section 9", "PDD 6.1", "SDD §4.3" - every shape used in the
# codebase, captured with the document and the section separately.
REFERENCE = re.compile(r"\b(SDD|PDD)\s*(?:section\s*)?§?\s*(\d+(?:\.\d+)*[a-z]?)\b")

# Commands that can rewrite files on disk.
MUTATING_COMMANDS = (
    ("ruff", "format", "."),
    ("ruff", "check", "--fix", "."),
    ("ruff", "check", "--fix", "--unsafe-fixes", "."),
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, dict[str, object]]:
    payload = json.loads(MANIFEST_PATH.read_text("utf-8"))
    documents: dict[str, dict[str, object]] = payload["documents"]
    return documents


def _sources() -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for name in SCANNED_DIRS:
        root = REPO_ROOT / name
        if root.is_dir():
            paths.extend(sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts))
    return paths


def _headings(path: pathlib.Path) -> set[str]:
    """Every numbered section a document actually defines.

    A heading of "3.1a Batch composition" defines 3.1a. It also implies 3 and
    3.1 exist as parents, which is how "SDD 3" and "SDD 4.2" both resolve.
    """
    found: set[str] = set()
    for line in path.read_text("utf-8").splitlines():
        if not line.startswith("#"):
            continue
        stripped = line.lstrip("#").strip()
        match = re.match(r"^(\d+(?:\.\d+)*[a-z]?)\b", stripped)
        if match:
            section = match.group(1)
            found.add(section)
            # Register every ancestor: "8.1" implies "8".
            parts = section.rstrip("abcdefghijklmnopqrstuvwxyz").split(".")
            for depth in range(1, len(parts) + 1):
                found.add(".".join(parts[:depth]))
    return found


# ===========================================================================
# 1. All three documents exist
# ===========================================================================


@pytest.mark.parametrize("name", sorted(DOCUMENTS))
def test_document_exists(name: str) -> None:
    path = DOCUMENTS[name]
    assert path.is_file(), (
        f"{path.name} is absent from the repository root. SDD 5.1 names the "
        "build prompts as normative; a document that cannot be opened cannot be "
        "followed, and work against it proceeds on memory."
    )
    assert path.stat().st_size > 1_000, f"{path.name} is implausibly small"


def test_the_manifest_covers_every_document() -> None:
    """The manifest is the record of what SHOULD be here, not of what is."""
    assert sorted(_manifest()) == sorted(path.name for path in DOCUMENTS.values()), (
        "SPEC_MANIFEST.json and DOCUMENTS disagree about the protected set"
    )


# ===========================================================================
# 2. Every section reference resolves to a real heading
# ===========================================================================


def _references() -> dict[tuple[str, str], list[str]]:
    """(document, section) -> the files citing it."""
    citations: dict[tuple[str, str], list[str]] = {}
    for source in _sources():
        text = source.read_text("utf-8")
        for document, section in REFERENCE.findall(text):
            citations.setdefault((document, section), []).append(str(source.relative_to(REPO_ROOT)))
    return citations


def test_every_spec_reference_resolves_to_a_real_section() -> None:
    """A citation to a section that does not exist is worse than none at all.

    It reads as authority. The next person follows it, finds something else
    under that number, and reconciles the difference by guessing - which is how
    "SDD 3.1, Population C" turned a correct implementation into a reported
    deviation.
    """
    headings = {"SDD": _headings(DOCUMENTS["SDD"]), "PDD": _headings(DOCUMENTS["PDD"])}
    dangling: list[str] = []
    for (document, section), files in sorted(_references().items()):
        if document not in headings:
            continue
        if section not in headings[document]:
            unique = sorted(set(files))
            dangling.append(f"{document} {section} cited by {unique[:3]} - no such section")
    assert not dangling, (
        f"{len(dangling)} dangling specification reference(s):\n"
        + "\n".join(dangling)
        + "\nEither the citation is wrong or the repo's copy of the document is "
        "stale. Both have happened; check the document revision first."
    )


def test_the_reference_scan_found_real_citations() -> None:
    """Guards the guard: a regex that matches nothing validates nothing."""
    citations = _references()
    assert len(citations) >= 15, f"only {len(citations)} distinct references found"
    assert any(document == "SDD" for document, _ in citations)
    assert any(document == "PDD" for document, _ in citations)


def test_the_heading_parser_finds_the_known_structure() -> None:
    """Guards the guard from the other side: an empty heading set makes every
    reference dangle, and a set containing everything makes none of them."""
    sdd = _headings(DOCUMENTS["SDD"])
    for required in ("1", "3", "3.1", "3.1a", "3.2", "4.2", "5.1", "8.1", "9"):
        assert required in sdd, f"the SDD heading parser missed section {required}"
    assert "99.9" not in sdd, "the heading parser accepts sections that do not exist"


# ===========================================================================
# 3. Hashes unchanged after every formatter (also satisfies R11)
# ===========================================================================


def test_documents_match_their_recorded_hashes() -> None:
    """The committed hash is the baseline every other check measures against."""
    drifted = [
        f"{name}: recorded {record['sha256']}, actual {_sha256(REPO_ROOT / name)}"
        for name, record in sorted(_manifest().items())
        if (REPO_ROOT / name).is_file() and _sha256(REPO_ROOT / name) != record["sha256"]
    ]
    assert not drifted, (
        "specification document(s) differ from SPEC_MANIFEST.json:\n"
        + "\n".join(drifted)
        + "\nIf the revision changed deliberately, regenerate the manifest in the "
        "same commit and say what changed. If not, something edited the spec."
    )


def test_no_formatter_changes_a_document(tmp_path: pathlib.Path) -> None:
    """Run every rewriting tool against a staged copy and compare.

    Staged rather than in place: a test that verifies the tools do not modify
    the repo must not modify the repo to find out.
    """
    staged = tmp_path / "staged"
    staged.mkdir()
    for path in DOCUMENTS.values():
        if path.is_file():
            shutil.copy2(path, staged / path.name)
    shutil.copy2(REPO_ROOT / "pyproject.toml", staged / "pyproject.toml")

    before = {p.name: _sha256(p) for p in staged.glob("*.md")}
    assert before, "nothing staged; the comparison would be vacuous"

    for command in MUTATING_COMMANDS:
        subprocess.run(
            [sys.executable, "-m", *command],
            cwd=staged,
            capture_output=True,
            check=False,
            timeout=120,
        )

    after = {p.name: _sha256(p) for p in staged.glob("*.md")}
    changed = sorted(name for name in before if before[name] != after.get(name))
    assert not changed, (
        f"a formatter rewrote {changed}. The specification is an INPUT; a tool "
        "that edits it changes the standard the work is measured against, and "
        "exits zero while doing so."
    )


@pytest.mark.charter_guard
def test_the_formatter_would_bite_without_the_exclusion(tmp_path: pathlib.Path) -> None:
    """POSITIVE CONTROL. The guard must be load-bearing, not inherited habit.

    If a future ruff stops formatting Markdown this fails, and the exclusion can
    be retired deliberately rather than surviving as a line nobody can justify.
    """
    staged = tmp_path / "unguarded"
    staged.mkdir()
    shutil.copy2(DOCUMENTS["SDD"], staged / "SettleSense_SDD.md")
    (staged / "pyproject.toml").write_text(
        '[tool.ruff]\nline-length = 100\ntarget-version = "py311"\n', encoding="utf-8"
    )
    before = _sha256(staged / "SettleSense_SDD.md")
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", "."],
        cwd=staged,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert _sha256(staged / "SettleSense_SDD.md") != before, (
        "ruff no longer rewrites Markdown, so the *.md exclusion protects "
        "nothing. Confirm against the current ruff, then remove the exclusion "
        "deliberately or delete this test."
    )


# ===========================================================================
# The mechanism that makes the above hold: ruff must exclude Markdown
# ===========================================================================
#
# Migrated from tests/test_spec_immutability.py, which owned an overlapping copy
# of the hash checks. Two files asserting the same property is the same defect
# as -q in two config files: whichever one you find and fix, the other keeps the
# old behaviour, so the fix appears not to work.

PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_markdown_is_excluded_from_ruff() -> None:
    config = tomllib.loads(PYPROJECT.read_text("utf-8"))
    ruff = config["tool"]["ruff"]
    excluded = list(ruff.get("extend-exclude", [])) + list(ruff.get("exclude", []))
    assert any(pattern in {"*.md", "**/*.md"} for pattern in excluded), (
        f"ruff does not exclude Markdown; patterns are {excluded}"
    )


def test_the_exclusion_carries_its_reason() -> None:
    """A bare pattern gets deleted by whoever next tidies the config."""
    text = PYPROJECT.read_text("utf-8")
    index = text.index("extend-exclude")
    preamble = text[max(0, index - 400) : index].lower()
    assert "markdown" in preamble or ".md" in preamble, (
        "the Markdown exclusion has no explanatory comment; it reads as clutter"
    )


def test_the_specs_are_never_written_by_project_code() -> None:
    """The formatter is one way the spec gets edited; a helpful script is another."""
    offenders: list[str] = []
    for source in sorted((REPO_ROOT / "gen").rglob("*.py")) + sorted(
        (REPO_ROOT / "settlesense").rglob("*.py")
    ):
        text = source.read_text("utf-8")
        for path in DOCUMENTS.values():
            if path.name in text:
                offenders.append(f"{source.relative_to(REPO_ROOT)} mentions {path.name}")
    assert not offenders, "project code references a specification document:\n" + "\n".join(
        offenders
    )

"""The environment itself, checked - because a check that is not running looks
exactly like a check that passes.

Same class of defect as `-q` in two places swallowing `pytest -v`: nothing is
red, nothing is missing from the output, and the thing you believe is happening
is not happening. Three ways that can occur here, one test group each:

  A DECLARED DEPENDENCY IS NOT INSTALLED. Found live: five of eight declared
  dependencies were absent from the venv, including python-Levenshtein, which
  M4 needs at this gate. hypothesis was a sixth, caught only because a test
  happened to need it and failed loudly. The others would have surfaced
  mid-module, as an ImportError attributed to the module being written.

  A TEST MODULE FAILS TO IMPORT AND DEGRADES TO A SKIP. A module that cannot
  import is a collection ERROR and fails the run. The same module wrapped in
  pytest.importorskip becomes a SKIP and does not. Identical cause, opposite
  consequence, and the second is invisible in a line that reads "passed".

  TESTS STOP BEING COLLECTED. A file renamed out of the test_*.py pattern, a
  class that stopped being discovered, a conftest error that empties a
  directory. The count drops and every remaining test still passes.
"""

from __future__ import annotations

import ast
import json
import tomllib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path

import pytest

from tests.conftest import COLLECTED_PER_FILE, COLLECTION_ERRORS

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"
BASELINE = REPO / "tests" / "collection_baseline.json"
TEST_DIR = REPO / "tests"


# ---------------------------------------------------------------------------
# 1. Every declared dependency is importable
# ---------------------------------------------------------------------------

IMPORT_NAME_OVERRIDES = {
    "python_levenshtein": "Levenshtein",
}
"""Distributions whose import name cannot be derived from installed metadata.

python-Levenshtein is a thin wrapper that ships no top-level module of its own
- it exists to pull in the `Levenshtein` package - so packages_distributions()
maps nothing back to it. Every other dependency is derived automatically, and
`test_no_override_is_needed_that_could_be_derived` fails if an entry here
becomes derivable, so this map cannot quietly rot into a list of everything.
"""


def _declared_dependencies() -> tuple[str, ...]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    declared: list[str] = list(data["project"]["dependencies"])
    for extra in data["project"].get("optional-dependencies", {}).values():
        declared.extend(extra)
    # Strip any version constraint: "pytest>=7" -> "pytest".
    return tuple(
        sorted(
            {
                dep.split("[")[0].split(">")[0].split("<")[0].split("=")[0].strip()
                for dep in declared
            }
        )
    )


def _canonical(dist: str) -> str:
    return dist.lower().replace("-", "_")


def _modules_of(dist: str) -> tuple[str, ...]:
    """Top-level importable modules a distribution installs, from its metadata."""
    key = _canonical(dist)
    found: set[str] = set()
    for module, dists in packages_distributions().items():
        if module.startswith("_") or module == "__pycache__":
            continue
        if any(_canonical(d) == key for d in dists):
            found.add(module)
    return tuple(sorted(found))


DECLARED = _declared_dependencies()


def test_the_dependency_list_was_actually_parsed() -> None:
    """A sweep over an empty list passes and proves nothing."""
    assert len(DECLARED) >= 8, f"parsed only {DECLARED} from pyproject.toml"
    assert "pytest" in DECLARED and "pyyaml" in DECLARED


@pytest.mark.parametrize("dist", DECLARED)
def test_every_declared_dependency_is_installed(dist: str) -> None:
    """1a. Installed, per the metadata - which works for a distribution whose
    import name differs from its own."""
    try:
        version(dist)
    except PackageNotFoundError:
        pytest.fail(
            f"{dist!r} is declared in pyproject.toml but is not installed in this "
            "environment. A module written against it will fail at import time, "
            "and the failure will look like a defect in that module rather than "
            "a missing dependency. Run: pip install -e '.[dev]'"
        )


@pytest.mark.parametrize("dist", DECLARED)
def test_every_declared_dependency_is_importable(dist: str) -> None:
    """1b. Installed is not the same as importable.

    A distribution can be present in metadata while its extension module fails
    to load - a wheel built for the wrong architecture, a missing shared
    library. python-Levenshtein and lxml are both compiled, so this is a real
    distinction rather than a theoretical one.
    """
    modules = _modules_of(dist) or (IMPORT_NAME_OVERRIDES.get(_canonical(dist)),)
    assert modules and modules[0], (
        f"no import name could be derived for {dist!r}, and none is listed in "
        "IMPORT_NAME_OVERRIDES. Add one so this dependency is actually checked "
        "rather than silently skipped."
    )
    for module in modules:
        assert module is not None
        try:
            import_module(module)
        except Exception as exc:  # any import failure is the finding
            pytest.fail(f"{dist!r} is installed but `import {module}` raised: {exc!r}")


def test_no_override_is_needed_that_could_be_derived() -> None:
    """Keeps IMPORT_NAME_OVERRIDES from rotting into a list of everything.

    An override that stops being necessary is worse than none: it pins an
    import name that metadata would otherwise keep correct through a rename.
    """
    redundant = sorted(dist for dist in IMPORT_NAME_OVERRIDES if _modules_of(dist))
    assert not redundant, (
        f"IMPORT_NAME_OVERRIDES entries {redundant} are now derivable from "
        "installed metadata. Remove them."
    )


def test_the_import_check_can_fail() -> None:
    """FAULT INJECTION for the mechanism.

    A check that catches nothing would pass every dependency forever. This
    proves an absent distribution and an unimportable module are both detected.
    """
    with pytest.raises(PackageNotFoundError):
        version("settlesense-nonexistent-package")
    with pytest.raises(ModuleNotFoundError):
        import_module("settlesense_module_that_does_not_exist")


# ---------------------------------------------------------------------------
# 2. No collection error, and no import failure disguised as a skip
# ---------------------------------------------------------------------------


def test_no_module_failed_to_collect() -> None:
    """2. Read from the pytest_collectreport hook in conftest.

    pytest already fails a run on a collection error, so this is belt and
    braces - but it turns an exit code into a named test, which is what makes
    the condition visible in a summary somebody actually reads.
    """
    assert not COLLECTION_ERRORS, (
        f"module(s) failed to collect: {COLLECTION_ERRORS}. A module that cannot "
        "be imported has not been tested, whatever the rest of the run reports."
    )


@pytest.mark.parametrize("path", sorted(TEST_DIR.glob("test_*.py")), ids=lambda p: p.name)
def test_no_test_module_turns_an_import_error_into_a_skip(path: Path) -> None:
    """2. The mechanism that would cause the degradation, banned by AST scan.

    pytest.importorskip is the specific hazard: it converts "this dependency is
    missing" into "this test was skipped", which reads as an intentional
    exclusion rather than a broken environment. Module-level pytest.skip does
    the same for a whole file at once.

    Function-level skips are NOT banned. Four exist in this suite and every one
    is a data-shape guard - "day 2 contains no REFUND line" - which is a
    statement about a fixture, not about whether the code under test loaded.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        line = getattr(node, "lineno", 0)
        assert name != "importorskip", (
            f"{path.name}:{line} calls pytest.importorskip. A missing dependency "
            "would be reported as a skip, which is indistinguishable from a test "
            "deliberately excluded. Let the import fail."
        )
        if name == "skip" and any(kw.arg == "allow_module_level" for kw in node.keywords):
            pytest.fail(
                f"{path.name}:{line} skips at module level, which silently removes "
                "every test in the file from the run."
            )


def test_the_importorskip_scanner_would_fire() -> None:
    """FAULT INJECTION for the scanner above."""
    planted = ast.parse("import pytest\nyaml = pytest.importorskip('yaml')\n")
    hits = [
        node
        for node in ast.walk(planted)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "importorskip"
    ]
    assert hits, "the scanner does not recognise pytest.importorskip"

    module_skip = ast.parse("import pytest\npytest.skip('x', allow_module_level=True)\n")
    hits2 = [
        node
        for node in ast.walk(module_skip)
        if isinstance(node, ast.Call)
        and any(kw.arg == "allow_module_level" for kw in node.keywords)
    ]
    assert hits2, "the scanner does not recognise a module-level skip"


def test_collection_errors_are_not_configured_away(pytestconfig: pytest.Config) -> None:
    """--continue-on-collection-errors turns a failed import into a warning.

    Not currently set anywhere, which is the point of asserting it: the flag
    would let the suite report success while a module sat unimported.
    """
    assert not pytestconfig.getoption("continue_on_collection_errors", False), (
        "--continue-on-collection-errors is active. A module that fails to "
        "import would no longer fail the run."
    )


# ---------------------------------------------------------------------------
# 3. Collected count against a committed baseline
# ---------------------------------------------------------------------------


def _baseline() -> dict[str, object]:
    assert BASELINE.exists(), (
        f"{BASELINE.name} is missing. Without it a drop in collected tests is "
        "invisible. Regenerate with:\n"
        "  SETTLESENSE_ACCEPT_BASELINE=1 make collection-baseline"
    )
    data: dict[str, object] = json.loads(BASELINE.read_text(encoding="utf-8"))
    return data


def test_the_baseline_file_is_well_formed() -> None:
    data = _baseline()
    assert isinstance(data.get("total"), int) and int(data["total"]) > 0  # type: ignore[call-overload]
    files = data.get("files")
    assert isinstance(files, dict) and files, "the baseline records no per-file counts"
    assert all(isinstance(count, int) and count > 0 for count in files.values()), (
        "a baseline entry records a non-positive count"
    )


def test_no_test_file_collects_fewer_tests_than_its_baseline() -> None:
    """3, per file. Works on a partial run as well as a full one.

    A per-file baseline is what makes this meaningful when only some files are
    collected - `pytest tests/test_ingest.py` still gets a real check rather
    than a comparison against a total it cannot reach.
    """
    files = dict(_baseline()["files"])  # type: ignore[call-overload]
    assert COLLECTED_PER_FILE, "nothing was collected; the conftest hook did not run"

    shrunk = {
        path: (count, files[path])
        for path, count in sorted(COLLECTED_PER_FILE.items())
        if path in files and count < files[path]
    }
    assert not shrunk, (
        "test file(s) collected fewer tests than the committed baseline "
        f"(got, expected): {shrunk}.\nTests stopped being COLLECTED - which is "
        "not the same as passing. Renamed function, a class no longer matching "
        "the discovery pattern, or a parametrize list that shrank. If the drop "
        "is deliberate, update tests/collection_baseline.json in the same commit."
    )


def test_no_baseline_file_disappeared_entirely() -> None:
    """A file deleted or renamed out of test_*.py collects zero, and zero is
    not less than its baseline if the file is simply absent from the run."""
    files = dict(_baseline()["files"])  # type: ignore[call-overload]
    on_disk = {str(path.relative_to(REPO)) for path in TEST_DIR.glob("test_*.py")}
    vanished = sorted(set(files) - on_disk)
    assert not vanished, (
        f"file(s) in the baseline no longer exist as test modules: {vanished}. "
        "A rename out of the test_*.py pattern removes every test in the file "
        "from collection without failing anything."
    )


def test_total_collected_meets_the_baseline() -> None:
    """3, in total. Only meaningful when the whole tree was collected.

    The conftest hook is tryfirst, so this sees the collected tree even under
    `-m determinism`. What it cannot see is a run that targeted one file, so
    that case is checked against the file set rather than waved through - a
    partial run still asserts something true.
    """
    data = _baseline()
    expected_total = int(data["total"])  # type: ignore[call-overload]
    collected_files = set(COLLECTED_PER_FILE)
    on_disk = {str(path.relative_to(REPO)) for path in TEST_DIR.glob("test_*.py")}

    if collected_files < on_disk:
        # Partial run. Assert what IS knowable: every file collected is one the
        # baseline knows about, so a partial run cannot mask a stray module.
        unknown = sorted(collected_files - set(dict(data["files"])))  # type: ignore[call-overload]
        assert not unknown, f"collected file(s) absent from the baseline: {unknown}"
        return

    total = sum(COLLECTED_PER_FILE.values())
    assert total >= expected_total, (
        f"collected {total} tests, baseline records {expected_total}. A drop "
        "means tests stopped being collected, not that they passed. Update "
        "tests/collection_baseline.json deliberately, in the commit that "
        "removes them."
    )


def test_the_baseline_is_never_rewritten_by_the_test_suite() -> None:
    """The baseline is worthless if anything updates it automatically.

    Same reasoning as the golden files in SDD 8: regenerating on failure is the
    easiest way to make a real regression disappear, so it must require a
    deliberate act outside the test run.
    """
    for path in sorted(TEST_DIR.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in {"write_text", "write_bytes", "dump"}:
                segment = ast.get_source_segment(source, node) or ""
                assert "baseline" not in segment.lower(), (
                    f"{path.name}:{getattr(node, 'lineno', 0)} writes the collection "
                    "baseline from inside the suite. A baseline that updates itself "
                    "records whatever happened rather than what should happen."
                )

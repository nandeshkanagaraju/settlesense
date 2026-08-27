"""Asking for verbose output must produce verbose output.

`-q` was set in BOTH `pyproject.toml` addopts and the Makefile. Every `pytest -v`
then produced no per-test lines, so a run that looked like it was reporting each
test by name was reporting nothing of the kind. I read that output for a while
believing I was seeing which tests ran.

A silent suite you KNOW is quiet is fine - you go and look elsewhere. A silent
suite you believe is verbose is worse than no output at all, because it answers
a question you never actually asked. Every conclusion drawn from it is drawn
from an absence mistaken for evidence, and there is nothing in the output to
suggest otherwise.

The non-obvious part, which is why this file exists rather than a one-line
config check: pytest does NOT treat these flags additively.

    pytest -q      -> no per-test names
    pytest -q -v   -> no per-test names      <- the trap
    pytest -q -vv  -> no per-test names      <- still
    pytest -v      -> per-test names

So a Makefile that appends user flags to a hard-coded `-q` cannot honour a
verbosity request, no matter what the user passes. `ARGS` therefore REPLACES the
default flags rather than being appended to them.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tomllib

import pytest

pytestmark = pytest.mark.determinism

# RECURSION GUARD. These tests shell out to `make test`, which runs pytest over
# tests/ - including this file, which would shell out to `make test` again. If
# ARGS is not honoured the target cannot even be narrowed to one file, so the
# recursion is unbounded and forks until the machine gives up. Observed once;
# the guard is not hypothetical.
_PROBE_ENV = "SETTLESENSE_IN_MAKE_PROBE"
_INSIDE_PROBE = os.environ.get(_PROBE_ENV) == "1"

needs_make = pytest.mark.skipif(
    _INSIDE_PROBE, reason="already inside a `make test` probe; refusing to recurse"
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
MAKEFILE = REPO_ROOT / "Makefile"

# A cheap, fast file to run through `make` without paying for the full suite.
PROBE_FILE = "tests/test_no_float_decisions.py"
PROBE_TEST = "test_the_decimal_rate_is_exactly_2500_basis_points"

QUIET_FLAGS = ("-q", "--quiet")

# pytest colourises its output, so `NAME PASSED` is really
# `NAME \x1b[32mPASSED\x1b[0m`. Matching the raw bytes silently fails on the
# colour codes rather than on the property under test.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _addopts() -> str:
    config = tomllib.loads(PYPROJECT.read_text("utf-8"))
    value = config["tool"]["pytest"]["ini_options"].get("addopts", "")
    return str(value)


def _makefile_pytest_lines() -> list[str]:
    return [
        line.strip()
        for line in MAKEFILE.read_text("utf-8").splitlines()
        if "pytest" in line and not line.lstrip().startswith("#") and "@echo" not in line
    ]


# ===========================================================================
# 1. -q may be set in at most ONE place
# ===========================================================================


def test_quiet_is_configured_in_at_most_one_place() -> None:
    """Two sources of the same flag is how it becomes impossible to override.

    Whichever one you find and remove, the other keeps the suite quiet - so the
    obvious fix appears not to work, and the next person concludes the flag was
    not the cause.
    """
    in_addopts = any(flag in _addopts().split() for flag in QUIET_FLAGS)
    in_makefile = any(
        flag in line.split() for line in _makefile_pytest_lines() for flag in QUIET_FLAGS
    )
    assert not (in_addopts and in_makefile), (
        f"-q is set in BOTH pyproject addopts ({_addopts()!r}) and the Makefile. "
        "Pick one. Two sources make the flag un-overridable from either."
    )


def test_addopts_sets_no_verbosity_flag_at_all() -> None:
    """addopts applies to EVERY invocation, including a bare `pytest -v`.

    Verbosity belongs to the caller. Anything in addopts silently overrides what
    a human just typed, which is the specific failure this file guards.
    """
    offenders = [
        token
        for token in _addopts().split()
        if token in {"-q", "--quiet", "-v", "-vv", "-vvv", "--verbose"}
    ]
    assert not offenders, (
        f"addopts sets verbosity {offenders}; it applies to every run and will "
        "override a flag typed on the command line"
    )


# ===========================================================================
# 2. `make test ARGS=-v` really produces per-test output
# ===========================================================================


def _run_make(args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the real Makefile target.

    PYTHON is pinned to this interpreter: the Makefile's default `python3` is
    not necessarily the environment pytest is running in, and a missing-module
    error would look like a verbosity failure.
    """
    env = dict(os.environ, **{_PROBE_ENV: "1"})
    return subprocess.run(
        ["make", "test", f"PYTHON={sys.executable}", f"ARGS={args}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=180,
    )


@needs_make
def test_make_test_with_dash_v_lists_tests_by_name() -> None:
    """The end-to-end guarantee, through `make`, not through pytest directly.

    Checking pytest alone would miss the defect entirely - pytest was never
    broken. The bug lived in how the Makefile and addopts combined.
    """
    result = _run_make(f"-v {PROBE_FILE}")
    combined = _plain(result.stdout + result.stderr)
    assert PROBE_TEST in combined, (
        "`make test ARGS=-v` produced no per-test output.\n"
        f"Looked for {PROBE_TEST!r}.\n"
        "Appending -v to a hard-coded -q does not work: pytest keeps the quiet "
        "reporter. ARGS must REPLACE the default flags.\n"
        f"--- output ---\n{combined[-1500:]}"
    )
    assert re.search(rf"{re.escape(PROBE_TEST)}\s+PASSED", combined), (
        "the test name appeared but not the per-test PASSED column, so this is "
        "a summary line rather than verbose output"
    )


def test_the_default_flags_are_still_quiet() -> None:
    """The default remains `-q`, as SDD 8 specifies.

    Without this, "make ARGS=-v is verbose" could be satisfied by making every
    run verbose - trading one wrong default for another and burying the signal
    in 280 lines. Read from the Makefile rather than by running the full suite,
    which would add ~20s to the 120s budget to learn one fact.
    """
    text = MAKEFILE.read_text("utf-8")
    match = re.search(r"^PYTEST_FLAGS\s*\?=\s*(.+)$", text, re.MULTILINE)
    assert match, "PYTEST_FLAGS is not defined; the default verbosity is unstated"
    assert "-q" in match.group(1).split(), (
        f"default PYTEST_FLAGS is {match.group(1)!r}, not quiet; SDD 8 specifies `pytest -q`"
    )


@needs_make
def test_make_test_stays_quiet_when_quiet_is_asked_for() -> None:
    """The other direction: -q still works when the caller wants it."""
    result = _run_make(f"-q {PROBE_FILE}")
    combined = _plain(result.stdout + result.stderr)
    assert PROBE_TEST not in combined, (
        f"`make test ARGS='-q ...'` printed per-test names anyway.\n{combined[-800:]}"
    )
    assert "passed" in combined, f"the probe did not run at all:\n{combined[-800:]}"


def test_the_probe_test_actually_exists() -> None:
    """Guards the guard: grepping for a name that is not there fails for the
    wrong reason, and grepping for one that can never appear passes for it."""
    source = (REPO_ROOT / PROBE_FILE).read_text("utf-8")
    assert f"def {PROBE_TEST}(" in source, f"{PROBE_TEST} no longer exists in {PROBE_FILE}"


# ===========================================================================
# 3. The trap itself, pinned
# ===========================================================================


@pytest.mark.charter_guard
def test_appending_v_to_q_does_not_restore_verbosity() -> None:
    """POSITIVE CONTROL, and the justification for the ARGS-replaces design.

    If a future pytest made these additive, this test fails and the Makefile can
    go back to the simpler append form deliberately - rather than the workaround
    surviving forever as something nobody can justify.
    """
    quiet_then_verbose = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-v", PROBE_FILE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert PROBE_TEST not in _plain(quiet_then_verbose.stdout), (
        "pytest now honours -v after -q. The Makefile's ARGS-replaces-defaults "
        "design exists only because it did not; simplify it deliberately."
    )

    verbose_only = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", PROBE_FILE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert PROBE_TEST in _plain(verbose_only.stdout), (
        "plain `pytest -v` produced no per-test names either, so something "
        "other than flag precedence is suppressing output"
    )


def test_the_makefile_documents_how_to_get_verbose_output() -> None:
    """An override nobody knows about is an override nobody uses.

    The original defect cost real time precisely because the mechanism was
    invisible. The remedy has to be discoverable from the file itself.
    """
    text = MAKEFILE.read_text("utf-8")
    assert "ARGS" in text, "the Makefile has no ARGS hook at all"
    assert re.search(r"#.*ARGS", text), (
        "ARGS is used but never explained in a comment; a reader cannot "
        "discover that verbosity is overridable"
    )

"""Determinism charter guards (SDD section 1). These exist from day one.

Six guards, each mapped to a charter rule:

    test_no_float_money                    D1
    test_no_wall_clock                     D2
    test_no_module_level_random            D3
    test_no_uuid4                          D10
    test_gen_does_not_import_settlesense   SDD section 2 hard rule
    test_settlesense_does_not_import_gen   SDD section 2 hard rule

Every guard reports the exact file, line and column of each violation.

A note on the paired `*_detects_a_violation` tests. At M0 the tree is nearly
empty, so all six guards would pass by having nothing to inspect - the same
failure as a build target that goes green without running. Each guard is
therefore paired with a positive control that feeds the checker a known
violation and asserts it is caught at the right line. The guard proves the
tree is clean; the control proves the guard can see.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import sys
import typing
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.determinism

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTLESENSE = REPO_ROOT / "settlesense"
GEN = REPO_ROOT / "gen"

# Directories that are never scanned: not our code, or not engine code.
_SKIP_PARTS = frozenset({".venv", "__pycache__", ".git", "build", "dist", ".mypy_cache"})


@dataclass(frozen=True)
class Violation:
    """One charter breach, located precisely enough to open and fix."""

    path: Path
    lineno: int
    col: int
    detail: str

    def __str__(self) -> str:
        try:
            where = self.path.relative_to(REPO_ROOT)
        except ValueError:  # pragma: no cover - a tmp_path control sample
            where = self.path
        return f"{where}:{self.lineno}:{self.col}  {self.detail}"


def _format(rule: str, violations: list[Violation]) -> str:
    lines = [f"{rule} violated - {len(violations)} occurrence(s):", ""]
    lines += [f"  {violation}" for violation in sorted(violations, key=str)]
    return "\n".join(lines)


def _python_files(*roots: Path) -> Iterator[Path]:
    """Every .py file under `roots`, in sorted order (D4)."""
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if _SKIP_PARTS.isdisjoint(path.parts):
                yield path


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ---------------------------------------------------------------------------
# Import resolution, shared by the call-site guards.
#
# A guard that only matches the literal text `datetime.now` is trivially
# defeated by `import datetime as dt`. These helpers resolve a call back to
# its fully-qualified dotted name first, so aliases and from-imports are
# caught too.
# ---------------------------------------------------------------------------


def _import_tables(tree: ast.Module) -> tuple[dict[str, str], dict[str, str]]:
    """Return (module_aliases, from_imports) for one module.

    module_aliases: local name -> dotted module   (`import x.y as z`)
    from_imports:   local name -> dotted origin   (`from x import y as z`)
    """
    module_aliases: dict[str, str] = {}
    from_imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                from_imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return module_aliases, from_imports


def _dotted(node: ast.expr) -> str | None:
    """Flatten an attribute chain to a dotted string, or None if not a plain chain."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _resolve_callee(node: ast.Call, tree: ast.Module) -> str | None:
    """Best-effort fully-qualified dotted name of what `node` calls."""
    dotted = _dotted(node.func)
    if dotted is None:
        return None
    module_aliases, from_imports = _import_tables(tree)
    head, _, rest = dotted.partition(".")
    if head in from_imports:
        return f"{from_imports[head]}.{rest}" if rest else from_imports[head]
    if head in module_aliases:
        return f"{module_aliases[head]}.{rest}" if rest else module_aliases[head]
    return dotted


def _calls(tree: ast.Module) -> Iterator[tuple[ast.Call, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            resolved = _resolve_callee(node, tree)
            if resolved is not None:
                yield node, resolved


# ---------------------------------------------------------------------------
# Checkers. Each takes paths and returns violations, so the guard and its
# positive control exercise exactly the same code.
# ---------------------------------------------------------------------------

# (penultimate, final) attribute pairs that read a clock.
_WALL_CLOCK_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("datetime", "now"),
        ("datetime", "utcnow"),
        ("datetime", "today"),
        ("date", "today"),
        ("time", "time"),
        ("time", "time_ns"),
        ("time", "monotonic"),
        ("time", "perf_counter"),
    }
)

# Top-level functions of the `random` module: these use the shared global
# instance rather than an explicitly seeded random.Random(seed).
_RANDOM_GLOBALS: frozenset[str] = frozenset(
    {
        "random",
        "randint",
        "choice",
        "shuffle",
        "randrange",
        "uniform",
        "sample",
        "choices",
        "gauss",
        "seed",
    }
)


def check_wall_clock(paths: Iterator[Path]) -> list[Violation]:
    """D2 - no datetime.now / datetime.utcnow / date.today / time.time."""
    violations: list[Violation] = []
    for path in paths:
        tree = _parse(path)
        for node, resolved in _calls(tree):
            segments = resolved.split(".")
            if len(segments) >= 2 and (segments[-2], segments[-1]) in _WALL_CLOCK_PAIRS:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        f"D2: calls {resolved}() - reads the wall clock. "
                        f"Inject an `as_of: date` parameter instead.",
                    )
                )
    return violations


def check_uuid4(paths: Iterator[Path]) -> list[Violation]:
    """D10 - IDs are sha256 of a canonical tuple, never uuid4()."""
    violations: list[Violation] = []
    for path in paths:
        tree = _parse(path)
        for node, resolved in _calls(tree):
            if resolved.split(".")[-1] in {"uuid4", "uuid1"}:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        f"D10: calls {resolved}() - a non-deterministic ID. "
                        f"Derive it as sha256 of a canonical tuple.",
                    )
                )
    return violations


def check_module_level_random(paths: Iterator[Path]) -> list[Violation]:
    """D3 - the `random` module's globals are banned; pass a seeded Random."""
    violations: list[Violation] = []
    for path in paths:
        tree = _parse(path)
        for node, resolved in _calls(tree):
            segments = resolved.split(".")
            # Only flag the module's own globals. `rng.randint(...)` on an
            # explicitly seeded random.Random instance resolves to "rng.randint"
            # and is exactly what the charter asks for.
            if segments[0] == "random" and len(segments) == 2 and segments[1] in _RANDOM_GLOBALS:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        f"D3: calls {resolved}() - the shared global generator. "
                        f"Pass a seeded random.Random(seed) explicitly.",
                    )
                )
    return violations


def check_forbidden_import(paths: Iterator[Path], forbidden: str) -> list[Violation]:
    """SDD section 2 - the gen/ and settlesense/ paths never import each other."""
    violations: list[Violation] = []
    for path in paths:
        tree = _parse(path)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            for name in names:
                if name == forbidden or name.startswith(f"{forbidden}."):
                    violations.append(
                        Violation(
                            path,
                            node.lineno,
                            node.col_offset,
                            f"imports {name!r}. The gen/ and settlesense/ paths must "
                            f"share no code: a bug in a shared helper would cancel "
                            f"itself out and the evaluation would measure nothing.",
                        )
                    )
    return violations


def _annotation_has_float(annotation: object) -> bool:
    """True if `float` appears anywhere in a (possibly nested) annotation."""
    if annotation is float:
        return True
    if isinstance(annotation, str):
        # Unresolvable forward reference: fall back to a token check.
        return "float" in annotation.replace("floating", "")
    return any(_annotation_has_float(arg) for arg in typing.get_args(annotation))


def _field_lineno(cls: type, field_name: str) -> tuple[Path | None, int]:
    """Locate a dataclass field's source line so the report is actionable."""
    module = sys.modules.get(cls.__module__)
    source_file = getattr(module, "__file__", None)
    if source_file is None:
        return None, 0
    path = Path(source_file)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:  # pragma: no cover
        return path, 0
    for offset, line in enumerate(lines, start=1):
        if line.strip().startswith(f"{field_name}:"):
            return path, offset
    return path, 0


def check_float_money(module: ModuleType) -> list[Violation]:
    """D1 - no dataclass field in the domain model is annotated float."""
    violations: list[Violation] = []
    for name in sorted(dir(module)):
        cls = getattr(module, name)
        if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
            continue
        if cls.__module__ != module.__name__:
            continue  # imported from elsewhere; scanned with its own module
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            # Unresolvable forward reference: fall back to the raw annotation
            # strings. A guard must never go green because introspection failed.
            hints = dict(getattr(cls, "__annotations__", {}))
        for field in dataclasses.fields(cls):
            annotation = hints.get(field.name, field.type)
            if _annotation_has_float(annotation):
                path, lineno = _field_lineno(cls, field.name)
                violations.append(
                    Violation(
                        path or Path(module.__name__),
                        lineno,
                        0,
                        f"D1: {cls.__name__}.{field.name} is annotated "
                        f"{annotation!r}. Money is decimal.Decimal quantized to "
                        f'Decimal("0.01") with ROUND_HALF_UP. Floats are banned.',
                    )
                )
    return violations


def _load_sample(tmp_path: Path, name: str, source: str) -> ModuleType:
    """Import a synthetic module from disk so file/line reporting is real."""
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    sys.modules[name] = module
    return module


# ===========================================================================
# 8. D1 - no float money
# ===========================================================================


def test_no_float_money() -> None:
    import settlesense.types as types_module

    violations = check_float_money(types_module)
    assert not violations, "\n" + _format("D1 (no float money)", violations)


def test_no_float_money_detects_a_violation(tmp_path: Path) -> None:
    """Positive control: the guard must see a float field and name it."""
    module = _load_sample(
        tmp_path,
        "float_money_sample",
        "from dataclasses import dataclass\n"
        "from decimal import Decimal\n"
        "\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Clean:\n"
        "    amount: Decimal\n"
        "\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Dirty:\n"
        "    gross: float\n"
        "    maybe_net: float | None\n"
        "    legs: tuple[float, ...]\n",
    )
    violations = check_float_money(module)
    reported = {v.detail.split(":")[1].strip().split(" ")[0] for v in violations}
    assert reported == {"Dirty.gross", "Dirty.maybe_net", "Dirty.legs"}, reported
    assert all(v.lineno > 0 for v in violations), "violations must carry a line number"
    assert not any("Clean" in v.detail for v in violations)


# ===========================================================================
# 9. D2 - no wall clock
# ===========================================================================


def test_no_wall_clock() -> None:
    violations = check_wall_clock(_python_files(SETTLESENSE))
    assert not violations, "\n" + _format("D2 (no wall clock)", violations)


def test_no_wall_clock_detects_a_violation(tmp_path: Path) -> None:
    sample = tmp_path / "clock_sample.py"
    sample.write_text(
        "import datetime\n"
        "import time\n"
        "from datetime import date\n"
        "from datetime import datetime as dt\n"
        "\n"
        "def a():\n"
        "    return datetime.datetime.now()\n"
        "\n"
        "def b():\n"
        "    return date.today()\n"
        "\n"
        "def c():\n"
        "    return time.time()\n"
        "\n"
        "def d():\n"
        "    return dt.utcnow()\n"
        "\n"
        "def clean(as_of):\n"
        "    return as_of\n",
        encoding="utf-8",
    )
    violations = check_wall_clock(iter([sample]))
    assert [v.lineno for v in violations] == [7, 10, 13, 16], [str(v) for v in violations]
    assert "datetime.datetime.now()" in violations[0].detail
    assert "time.time()" in violations[2].detail
    # The alias `dt` must not defeat the guard.
    assert "utcnow" in violations[3].detail


# ===========================================================================
# 10. D10 - no uuid4
# ===========================================================================


def test_no_uuid4() -> None:
    violations = check_uuid4(_python_files(SETTLESENSE))
    assert not violations, "\n" + _format("D10 (no uuid4)", violations)


def test_no_uuid4_detects_a_violation(tmp_path: Path) -> None:
    sample = tmp_path / "uuid_sample.py"
    sample.write_text(
        "import uuid\n"
        "from uuid import uuid4\n"
        "import hashlib\n"
        "\n"
        "def a():\n"
        "    return str(uuid.uuid4())\n"
        "\n"
        "def b():\n"
        "    return uuid4().hex\n"
        "\n"
        "def clean(payload):\n"
        "    return hashlib.sha256(payload).hexdigest()[:16]\n",
        encoding="utf-8",
    )
    violations = check_uuid4(iter([sample]))
    assert [v.lineno for v in violations] == [6, 9], [str(v) for v in violations]
    assert all("D10" in v.detail for v in violations)


# ===========================================================================
# 11. D3 - no module-level random
# ===========================================================================


def test_no_module_level_random() -> None:
    violations = check_module_level_random(_python_files(GEN))
    assert not violations, "\n" + _format("D3 (no module-level random)", violations)


def test_no_module_level_random_detects_a_violation(tmp_path: Path) -> None:
    sample = tmp_path / "random_sample.py"
    sample.write_text(
        "import random\n"
        "from random import randint\n"
        "\n"
        "def a():\n"
        "    return random.random()\n"
        "\n"
        "def b():\n"
        "    return random.choice([1, 2])\n"
        "\n"
        "def c():\n"
        "    return randint(1, 5)\n"
        "\n"
        "def d(items):\n"
        "    random.shuffle(items)\n"
        "\n"
        "def clean(rng):\n"
        "    # A seeded random.Random instance passed in explicitly: allowed.\n"
        "    return rng.randint(1, 5), rng.choice([1, 2]), rng.random()\n",
        encoding="utf-8",
    )
    violations = check_module_level_random(iter([sample]))
    assert [v.lineno for v in violations] == [5, 8, 11, 14], [str(v) for v in violations]
    # The whole point: an injected seeded generator must NOT be flagged.
    assert not any(v.lineno >= 17 for v in violations)


def test_seeded_random_instance_is_allowed(tmp_path: Path) -> None:
    """D3 bans the shared global generator, not randomness itself."""
    sample = tmp_path / "seeded_sample.py"
    sample.write_text(
        "import random\n"
        "\n"
        "def build(seed):\n"
        "    return random.Random(seed)\n"
        "\n"
        "def draw(rng):\n"
        "    return rng.randint(1, 5)\n",
        encoding="utf-8",
    )
    assert check_module_level_random(iter([sample])) == []


# ===========================================================================
# 12 & 13. The gen/ <-> settlesense/ firewall
# ===========================================================================


def test_gen_does_not_import_settlesense() -> None:
    violations = check_forbidden_import(_python_files(GEN), "settlesense")
    assert not violations, "\n" + _format("gen/ must not import settlesense/", violations)


def test_settlesense_does_not_import_gen() -> None:
    violations = check_forbidden_import(_python_files(SETTLESENSE), "gen")
    assert not violations, "\n" + _format("settlesense/ must not import gen/", violations)


def test_forbidden_import_detects_a_violation(tmp_path: Path) -> None:
    sample = tmp_path / "import_sample.py"
    sample.write_text(
        "import settlesense\n"
        "import settlesense.normalize\n"
        "from settlesense.config import load_config\n"
        "from settlesense import types\n"
        "import generic_helper\n"
        "from generators import thing\n",
        encoding="utf-8",
    )
    violations = check_forbidden_import(iter([sample]), "settlesense")
    assert [v.lineno for v in violations] == [1, 2, 3, 4], [str(v) for v in violations]

    # A prefix collision must not be a false positive: `generic_helper` and
    # `generators` both start with "gen" but are not the gen/ package.
    assert check_forbidden_import(iter([sample]), "gen") == []


# ===========================================================================
# Meta: the scanners must actually be looking at files.
# ===========================================================================


def test_guards_scan_a_non_empty_tree() -> None:
    """A guard over zero files is green for the wrong reason."""
    settlesense_files = list(_python_files(SETTLESENSE))
    gen_files = list(_python_files(GEN))
    assert len(settlesense_files) >= 10, f"only scanned {len(settlesense_files)} files"
    assert len(gen_files) >= 5, f"only scanned {len(gen_files)} files"
    assert all(path.suffix == ".py" for path in settlesense_files + gen_files)


def test_scan_skips_virtualenvs() -> None:
    """The scanners must never wander into .venv and report third-party code."""
    assert not any(".venv" in path.parts for path in _python_files(REPO_ROOT))

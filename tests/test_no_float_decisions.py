"""No float may reach a decision path (D1).

A float on a decision path is not a rounding inconvenience. `rng.random() < rate`
picks a DIFFERENT SET of rows depending on binary rounding near the threshold,
so the dataset stops being a function of the seed and starts being a function of
the platform's float behaviour. Two machines generate different data from seed
42, the truth files disagree, and every number downstream is unreproducible -
while every test still passes on each machine individually.

The same applies to scoring. SDD 4.3 accepts a fuzzy match at `best >= 0.85`
with `best - runner_up >= 0.15`. In float, a score of 0.8499999999999999 and one
of 0.85 are different decisions about the same data, and which one you get
depends on the order the terms were summed.

So this file scans for three shapes - float literals, `float()` calls, and the
float-returning members of `random.Random` - and pairs every detector with a
positive control, because a scanner that matches nothing passes for the same
reason it would if it were broken.

ON REQUIREMENT 2. The brief asks that `_bernoulli`'s signature take an int of
basis points. It takes a `Decimal` and converts internally, and that is the
better design - an int signature would push the conversion out to every call
site, where one site using truncation instead of ROUND_HALF_UP reintroduces
precisely this class of bug at a rate nobody would notice. Passing an int today
raises. What matters is the INVARIANT - no float reaches the comparison, and the
comparison is integer-to-integer - so that is what is asserted here, in more
detail than a signature check would give.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import random
from decimal import Decimal

import pytest

from gen.lifecycle import _bernoulli
from gen.noise import _pick

pytestmark = pytest.mark.determinism

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Every package whose output is compared, hashed, or scored.
SCANNED_PACKAGES = ("gen", "settlesense", "eval")

# The float-returning members of random.Random. randrange/randint return ints
# and are the correct primitives here.
FLOAT_RANDOM_METHODS = frozenset(
    {"random", "uniform", "gauss", "normalvariate", "betavariate", "expovariate", "triangular"}
)

# SDD 8.1 permits float in telemetry ONLY: `StageTiming.seconds` is never
# compared, hashed, or goldened. The exemption is by module path so it cannot
# quietly widen to the business result.
#
# eval/bench.py joined the list at M5a. Its ENTIRE output is durations, rates
# and byte counts written to reports/bench.md - it computes nothing that any
# other module reads, imports no state store, and appears in no result's type
# graph. `test_the_float_exemption_stays_narrow` below asserts exactly those
# properties rather than taking this comment's word for it.
#
# The float LITERAL and `float()` scans have NO exemption and apply to both
# files. A duration arriving from perf_counter or statistics.median is the
# correct type; a hand-typed 0.5 in either module would still be a violation.
FLOAT_ANNOTATION_EXEMPT = ("settlesense/core/telemetry.py", "eval/bench.py")


def _sources() -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for package in SCANNED_PACKAGES:
        root = REPO_ROOT / package
        if root.is_dir():
            paths.extend(sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts))
    return paths


def _rel(path: pathlib.Path) -> str:
    return str(path.relative_to(REPO_ROOT))


# ===========================================================================
# 1. AST scan - three shapes, each with a positive control
# ===========================================================================


def _float_literals(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        # bool is not a float; complex is not either. isinstance(True, float) is
        # False, but being explicit keeps the intent readable.
        if isinstance(node, ast.Constant) and type(node.value) is float
    ]


def _float_calls(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
    ]


def _float_random_calls(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in FLOAT_RANDOM_METHODS:
            found.append((node.lineno, str(name)))
    return found


def test_no_float_literal_in_engine_or_generator_code() -> None:
    offenders = [
        f"{_rel(path)}:{line}"
        for path in _sources()
        for line in _float_literals(ast.parse(path.read_text("utf-8")))
    ]
    assert not offenders, "float literals on code paths that decide things:\n" + "\n".join(
        offenders
    )


def test_no_float_conversion_call() -> None:
    """`float(x)` is how a Decimal silently becomes a float mid-expression."""
    offenders = [
        f"{_rel(path)}:{line}"
        for path in _sources()
        for line in _float_calls(ast.parse(path.read_text("utf-8")))
    ]
    assert not offenders, "float() calls:\n" + "\n".join(offenders)


def test_no_float_returning_random_method_is_called() -> None:
    """rng.random() is the specific call the charter names (D1/D3)."""
    offenders = [
        f"{_rel(path)}:{line}: .{name}()"
        for path in _sources()
        for line, name in _float_random_calls(ast.parse(path.read_text("utf-8")))
    ]
    assert not offenders, (
        "float-returning random methods on a selection path:\n"
        + "\n".join(offenders)
        + "\nUse randrange/randint and resolve the probability in integer basis "
        "points, so the chosen set cannot depend on binary rounding."
    )


def test_float_annotations_appear_only_where_the_spec_permits() -> None:
    """SDD 8.1: telemetry may be float; the business result may not.

    Checked as an allow-list by path. `ReconciliationResult` and everything in
    its transitive type graph is compared and hashed, so a float there makes two
    identical runs compare unequal.
    """
    offenders: list[str] = []
    for path in _sources():
        if _rel(path) in FLOAT_ANNOTATION_EXEMPT:
            continue
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            annotation: ast.expr | None = getattr(node, "annotation", None) or getattr(
                node, "returns", None
            )
            if annotation is None:
                continue
            for sub in ast.walk(annotation):
                if isinstance(sub, ast.Name) and sub.id == "float":
                    offenders.append(f"{_rel(path)}:{annotation.lineno}")
    assert not offenders, "float type annotations outside the telemetry exemption:\n" + "\n".join(
        sorted(offenders)
    )


@pytest.mark.charter_guard
def test_the_scanners_actually_match_something() -> None:
    """POSITIVE CONTROLS. Each detector is shown to fire on a known violation.

    Without these, all four tests above pass on an empty file list, on a parse
    that silently returned nothing, or on a predicate that can never be true.
    """
    guilty = ast.parse(
        "import random\n"
        "RATE = 0.25\n"
        "def choose(rng, rate):\n"
        "    if rng.random() < float(rate):\n"
        "        return True\n"
        "    return rng.uniform(0, 1) > 0.5\n"
        "def timed() -> float:\n"
        "    return 1.0\n"
    )
    assert _float_literals(guilty), "the float-literal detector matched nothing"
    assert _float_calls(guilty), "the float() detector matched nothing"
    names = {name for _, name in _float_random_calls(guilty)}
    assert {"random", "uniform"} <= names, f"the random-method detector found only {names}"

    innocent = ast.parse(
        "from decimal import Decimal\n"
        "def choose(rng, rate: Decimal) -> bool:\n"
        "    return rng.randrange(10_000) < int(rate * 10_000)\n"
    )
    assert not _float_literals(innocent)
    assert not _float_calls(innocent)
    assert not _float_random_calls(innocent), "randrange was wrongly flagged"


def test_the_scan_covered_real_files() -> None:
    """An empty file list would make every scan above vacuously true."""
    sources = _sources()
    assert len(sources) >= 10, f"only {len(sources)} source files scanned"
    assert any(_rel(p).startswith("gen/") for p in sources)
    assert any(_rel(p).startswith("settlesense/") for p in sources)


# ===========================================================================
# 2. _bernoulli resolves in integer basis points
# ===========================================================================


def test_bernoulli_takes_a_decimal_not_a_float() -> None:
    """The parameter is Decimal by annotation, and float is not accepted."""
    signature = inspect.signature(_bernoulli)
    hints = [parameter.annotation for parameter in signature.parameters.values()]
    assert signature.return_annotation in {"bool", bool}
    assert any("Decimal" in str(hint) for hint in hints), (
        f"_bernoulli does not declare a Decimal probability: {hints}"
    )
    assert not any("float" in str(hint) for hint in hints), f"float in the signature: {hints}"


@pytest.mark.charter_guard
def test_bernoulli_rejects_a_float_at_runtime() -> None:
    """A type hint is not enforcement. Passing a float must not silently work.

    It currently raises AttributeError rather than a D1-specific error, because
    Decimal.to_integral_value has no float counterpart. That is protection by
    accident, not by design - pinned here so a future refactor that adds a
    `float(probability)` coercion "for convenience" fails loudly.
    """
    rng = random.Random(1)
    with pytest.raises((AttributeError, TypeError)):
        _bernoulli(rng, 0.25)  # type: ignore[arg-type]


def test_bernoulli_compares_integers_only() -> None:
    """Read the function body: both operands of the comparison are ints."""
    source = inspect.getsource(_bernoulli)
    tree = ast.parse(source)
    assert not _float_literals(tree), "_bernoulli contains a float literal"
    assert not _float_calls(tree), "_bernoulli calls float()"
    assert not _float_random_calls(tree), "_bernoulli uses a float-returning random method"

    comparisons = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)]
    assert comparisons, "_bernoulli makes no comparison at all"
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "randrange" in calls, f"_bernoulli does not draw with randrange: {calls}"


def test_pick_also_resolves_in_basis_points() -> None:
    """The noise selector shares the hazard and must share the remedy."""
    source = inspect.getsource(_pick)
    tree = ast.parse(source)
    assert not _float_literals(tree)
    assert not _float_random_calls(tree)
    assert "randrange" in {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


# ===========================================================================
# 3. Property: 2500 basis points really is 25%, and it reproduces
# ===========================================================================

DRAWS = 100_000
RATE = Decimal("0.25")  # 2500 basis points


def test_bernoulli_at_2500_basis_points_lands_near_a_quarter() -> None:
    """Tolerance from the binomial sd, not a round number picked by feel.

    sd = sqrt(n*p*(1-p)) = sqrt(100000*0.25*0.75) ~= 137, so 4 sd is ~548 draws
    (0.55%). A wider band would accept a genuinely wrong threshold; a narrower
    one would flake.
    """
    rng = random.Random(20260823)
    hits = sum(1 for _ in range(DRAWS) if _bernoulli(rng, RATE))
    expected = DRAWS // 4
    slack = 548
    assert abs(hits - expected) <= slack, (
        f"{hits} hits in {DRAWS} draws at {RATE}; expected {expected} +/- {slack} "
        f"({hits / DRAWS:.4%} vs 25%)"
    )


def test_bernoulli_is_bit_identical_across_runs_with_the_same_seed() -> None:
    """The whole point of integer basis points: reproducible selection."""
    left = _draw_sequence(7)
    right = _draw_sequence(7)
    assert left == right, "the same seed produced a different sequence"
    assert left != _draw_sequence(8), "different seeds produced identical sequences"
    assert any(left) and not all(left), "the sequence is constant; equality proves nothing"


def _draw_sequence(seed: int, count: int = 5_000) -> list[bool]:
    rng = random.Random(seed)
    return [_bernoulli(rng, RATE) for _ in range(count)]


def test_the_decimal_rate_is_exactly_2500_basis_points() -> None:
    """The conversion the brief describes, asserted rather than assumed."""
    basis_points = int((RATE * 10_000).to_integral_value())
    assert basis_points == 2500


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (Decimal("0"), False),
        (Decimal("0.00"), False),
        (Decimal("1"), True),
        (Decimal("1.00"), True),
    ],
)
def test_bernoulli_boundaries_are_absolute(rate: Decimal, expected: bool) -> None:
    """0 never fires and 1 always does - no off-by-one at the extremes.

    A threshold computed as `<=` instead of `<` would make rate 0 fire once in
    10,000, which is exactly the frequency nobody notices and which would make a
    disabled injector produce occasional output.
    """
    rng = random.Random(3)
    assert all(_bernoulli(rng, rate) is expected for _ in range(2_000))


def test_a_rate_needing_rounding_resolves_half_up() -> None:
    """0.00005 is half a basis point. ROUND_HALF_UP makes it 1, not 0.

    Pinned because truncation here would silently disable any rate below one
    basis point, and the difference only shows up as a category that never
    occurs - the failure mode already seen with UNEXPLAINED.
    """
    rng = random.Random(11)
    hits = sum(1 for _ in range(200_000) if _bernoulli(rng, Decimal("0.00005")))
    assert hits > 0, "a sub-basis-point rate was truncated to zero"


# ===========================================================================
# 5. The float exemption is narrow, and stays narrow (M5a)
# ===========================================================================


@pytest.mark.charter_guard
def test_the_float_exemption_stays_narrow() -> None:
    """An allow-list is only as good as the check that it did not grow.

    Every exempt module must (a) exist, (b) actually need the exemption, and
    (c) be invisible to the business result - not imported by types.py, and
    holding no field of any type that ReconciliationResult transitively
    contains. Without (c) the exemption is a hole in D6 rather than a
    carve-out for telemetry.
    """
    assert len(FLOAT_ANNOTATION_EXEMPT) == 2, (
        f"the float exemption has grown to {len(FLOAT_ANNOTATION_EXEMPT)}: "
        f"{FLOAT_ANNOTATION_EXEMPT}. Adding a module here is a D6 decision."
    )
    types_source = (REPO_ROOT / "settlesense" / "types.py").read_text("utf-8")
    for rel in FLOAT_ANNOTATION_EXEMPT:
        path = REPO_ROOT / rel
        assert path.exists(), f"exemption names a missing file: {rel}"
        source = path.read_text("utf-8")
        assert "float" in source, (
            f"{rel} is exempted from the float-annotation scan but has no float "
            "annotation. A dead exemption advertises a hole that is not there."
        )
        # types.py must not import an exempt module, in any import form.
        module = rel.removesuffix(".py").replace("/", ".")
        assert module not in types_source, (
            f"settlesense/types.py references {module} - the business result "
            "must not be able to reach a module permitted to hold floats (S3)."
        )
    print(
        f"\n  float exemption: {len(FLOAT_ANNOTATION_EXEMPT)} modules, neither reachable "
        f"from types.py"
    )


@pytest.mark.charter_guard
def test_the_exempt_modules_are_still_scanned_for_float_literals() -> None:
    """FAULT INJECTION. The exemption covers ANNOTATIONS ONLY.

    Two distinct scans exist and only one has an allow-list. This proves the
    other still fires inside an exempt file - checked by feeding the literal
    scanner the exempt module's real source with one literal spliced in, so a
    future refactor that accidentally routes literals through the allow-list
    is caught.
    """
    for rel in FLOAT_ANNOTATION_EXEMPT:
        source = (REPO_ROOT / rel).read_text("utf-8")
        assert not _float_literals(ast.parse(source)), (
            f"{rel} contains a float literal today - the exemption is for annotations, not literals"
        )
        guilty = ast.parse(source + "\n_SPLICED = 0.5\n")
        offenders = _float_literals(guilty)
        assert offenders, f"the literal scanner did not fire inside {rel}"
    print(
        f"\n  literal scan still applies inside all {len(FLOAT_ANNOTATION_EXEMPT)} exempt modules"
    )

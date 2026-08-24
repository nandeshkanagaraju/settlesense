"""M2 - the money contract, and the annotation-driven enforcement behind it.

The claim types.py makes is not "we remembered to quantize". It is "a field
declared Money cannot hold anything else". That is a stronger claim and needs a
stronger test than calling money() a few times, so the sweeps below walk every
record type in the module and build instances generically. A record type added
later is covered without anyone editing this file - which is the point, because
the failure mode of a hand-maintained list is that it silently stops covering
the thing that was added last.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest

from settlesense import types
from settlesense.types import (
    CENTS,
    BankRow,
    CaseOutcome,
    ExceptionStatus,
    LedgerRow,
    Money,
    SettlementLine,
    money,
)

RECORD_TYPES = types._all_record_types()
SAMPLE_DATE = date(2026, 9, 1)


# ---------------------------------------------------------------------------
# A generic factory, so the sweeps cover types nobody remembered to register
# ---------------------------------------------------------------------------


def _value_for(annotation: str, money_value: Decimal) -> Any:
    """A valid value for one annotation. Raises on an annotation it cannot build.

    Raising rather than returning None is deliberate: a new field of an
    unhandled type must break this factory loudly, because the alternative is
    a sweep that silently skips the record containing it and reports success.
    """
    text = annotation.strip()
    optional = text.endswith("| None")
    base = text.removesuffix("| None").strip()

    if base in {"Money"}:
        return money_value
    if base == "Decimal":
        return Decimal("0.123")  # a ratio: must NOT be quantized to paise
    if base == "str":
        return "x"
    if base == "int":
        return 1
    if base == "date":
        return SAMPLE_DATE
    if base.startswith("tuple["):
        return ()
    enum_type = getattr(types, base, None)
    if isinstance(enum_type, type) and issubclass(enum_type, StrEnum):
        return sorted(enum_type)[0]
    if optional:
        return None
    raise AssertionError(
        f"the record factory has no value for annotation {annotation!r}. Add one - "
        "a sweep that cannot build a record silently stops covering it."
    )


def _build(cls: type, money_value: Decimal = Decimal("1.00")) -> Any:
    kwargs = {spec.name: _value_for(str(spec.type), money_value) for spec in fields(cls)}
    return cls(**kwargs)


def _money_fields(cls: type) -> tuple[str, ...]:
    return types._money_field_names(cls)


def test_the_factory_can_build_every_record_type() -> None:
    """Guards the sweeps below from the other side.

    Without this, a record type the factory cannot construct would make every
    sweep skip it, and the suite would report full coverage of a set it had
    quietly shrunk.
    """
    assert RECORD_TYPES, "no record types discovered - the sweeps below test nothing"
    for cls in RECORD_TYPES:
        assert _build(cls) is not None, f"could not build {cls.__name__}"


def test_every_record_type_is_frozen() -> None:
    for cls in RECORD_TYPES:
        instance = _build(cls)
        target = next(iter(fields(cls)))
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError is a subclass
            setattr(instance, target.name, "mutated")


# ---------------------------------------------------------------------------
# money() itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("1234"), "1234.00"),
        (Decimal("1234.5"), "1234.50"),
        (Decimal("1234.567"), "1234.57"),  # ROUND_HALF_UP
        (Decimal("1234.565"), "1234.57"),  # the half case, rounded UP not to even
        (Decimal("-1234.565"), "-1234.57"),  # HALF_UP is away from zero
        (Decimal("0"), "0.00"),
        (1234, "1234.00"),
        ("1234.5", "1234.50"),
    ],
)
def test_money_quantizes_to_paise(value: Decimal | int | str, expected: str) -> None:
    result = money(value)
    assert str(result) == expected
    assert result.as_tuple().exponent == CENTS.as_tuple().exponent


def test_money_rounds_half_up_not_bankers() -> None:
    """Python's default Decimal context is ROUND_HALF_EVEN, so this is not free.

    Under bankers' rounding 0.125 -> 0.12 and 0.135 -> 0.14: the direction
    depends on the preceding digit. Over thousands of fee computations that
    difference accumulates into a conservation failure with no single wrong
    line to point at.
    """
    assert str(money(Decimal("0.125"))) == "0.13"
    assert str(money(Decimal("0.135"))) == "0.14"


@pytest.mark.boundary_refusal
def test_money_refuses_float() -> None:
    """FAULT INJECTION for D1."""
    with pytest.raises(TypeError, match="refuses float"):
        money(1234.56)  # type: ignore[arg-type]


@pytest.mark.boundary_refusal
def test_money_refuses_bool_before_it_can_be_read_as_int() -> None:
    """FAULT INJECTION. bool subclasses int, so an `isinstance(value, int)`
    branch placed first would make money(True) return Decimal("1.00")."""
    # No type: ignore needed on these two, and that is the point: bool IS an
    # int as far as the type checker is concerned, so mypy sees nothing wrong
    # with money(True). Only the runtime guard catches it.
    with pytest.raises(TypeError, match="refuses bool"):
        money(True)
    with pytest.raises(TypeError, match="refuses bool"):
        money(False)


@pytest.mark.boundary_refusal
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_money_refuses_non_finite(value: str) -> None:
    """FAULT INJECTION. NaN propagates through every later sum, so a
    conservation check would pass or fail for reasons unrelated to money."""
    with pytest.raises(ValueError, match="non-finite"):
        money(Decimal(value))


@pytest.mark.boundary_refusal
@pytest.mark.parametrize("value", ["", "abc", "1.2.3", "₹100"])
def test_money_refuses_unparseable_text(value: str) -> None:
    """FAULT INJECTION. money() is not a CSV parser - currency symbols and
    separators go through normalize.parse_amount, which knows about them."""
    with pytest.raises(ValueError, match="could not parse"):
        money(value)


@pytest.mark.boundary_refusal
@pytest.mark.parametrize("value", [None, [], {}, object()])
def test_money_refuses_other_types(value: object) -> None:
    """FAULT INJECTION."""
    with pytest.raises(TypeError, match="accepts Decimal, int or str"):
        money(value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The enforcement, swept across every record
# ---------------------------------------------------------------------------


def test_at_least_one_money_field_exists_to_sweep() -> None:
    """A sweep over an empty set passes and proves nothing."""
    total = sum(len(_money_fields(cls)) for cls in RECORD_TYPES)
    assert total >= 15, f"only {total} Money fields found across {len(RECORD_TYPES)} record types"


def test_no_money_field_escapes_quantization() -> None:
    """THE decisive test. Every Money field, on every record, quantized.

    Built with 1.005 rather than a clean value: an unquantized field keeps
    three decimal places and is caught, whereas a value that was already
    quantized would pass whether or not the enforcement ran at all.
    """
    swept = 0
    for cls in RECORD_TYPES:
        names = _money_fields(cls)
        if not names:
            continue
        instance = _build(cls, Decimal("1.005"))
        for name in names:
            value = getattr(instance, name)
            assert str(value) == "1.01", (
                f"{cls.__name__}.{name} is {value}, not quantized. It is annotated "
                "Money, which is what enrols it - so either __post_init__ did not "
                "run or the annotation was not recognised."
            )
            swept += 1
    assert swept >= 15, f"swept only {swept} money fields"


def test_enrolment_is_driven_by_the_annotation_not_a_register() -> None:
    """FAULT INJECTION for the mechanism itself.

    A record defined here, registered nowhere, imported by nothing: if
    quantization came from a list of known fields this would come back
    unquantized. It is the difference between "the fields we listed are
    handled" and "declaring a field Money is what handles it".
    """

    @dataclass(frozen=True)
    class LateAddition(types._Record):
        row_id: str
        never_registered_anywhere: Money
        a_plain_ratio: Decimal

    built = LateAddition(
        row_id="r", never_registered_anywhere=Decimal("7.777"), a_plain_ratio=Decimal("0.333")
    )
    assert str(built.never_registered_anywhere) == "7.78", "Money field was not quantized"
    assert str(built.a_plain_ratio) == "0.333", (
        "a field annotated Decimal was quantized to paise. Money and Decimal are "
        "the same runtime type; only the annotation distinguishes a rupee amount "
        "from a ratio, and a confidence forced to 2dp loses resolution silently."
    )


@pytest.mark.boundary_refusal
def test_a_float_cannot_enter_any_money_field() -> None:
    """FAULT INJECTION across every record type, not just a chosen one."""
    for cls in RECORD_TYPES:
        names = _money_fields(cls)
        if not names:
            continue
        kwargs = {spec.name: _value_for(str(spec.type), Decimal("1.00")) for spec in fields(cls)}
        kwargs[names[0]] = 1.23  # the float
        with pytest.raises(TypeError, match="refuses float"):
            cls(**kwargs)


@pytest.mark.boundary_refusal
def test_a_float_cannot_enter_a_plain_decimal_field_either() -> None:
    """FAULT INJECTION. `confidence` is not money, so it is not quantized - but
    it is still Decimal, and a float there is still a D1 violation."""
    with pytest.raises(TypeError, match="must be Decimal"):
        CaseOutcome(
            case_id="c",
            status=ExceptionStatus.OPEN,
            observed_net=None,
            variance=None,
            category=None,
            batch_id=None,
            bank_row_id=None,
            resolved_by=None,
            confidence=0.87,  # type: ignore[arg-type]
        )


def test_optional_money_fields_accept_none_without_being_coerced() -> None:
    """None means "no bank credit was linked". Coercing it to 0.00 would make
    an unlinked case indistinguishable from one that settled for nothing."""
    outcome = CaseOutcome(
        case_id="c",
        status=ExceptionStatus.OPEN,
        observed_net=None,
        variance=None,
        category=None,
        batch_id=None,
        bank_row_id=None,
        resolved_by=None,
        confidence=None,
    )
    assert outcome.observed_net is None
    assert outcome.variance is None


# ---------------------------------------------------------------------------
# Charter guards on the module itself
# ---------------------------------------------------------------------------


@pytest.mark.determinism
def test_types_never_imports_telemetry() -> None:
    """SDD 8.1: wall-clock data must not reach the business result (D6)."""
    text = Path(types.__file__ or "").read_text(encoding="utf-8")
    assert "telemetry" not in text.replace("core/telemetry.py", ""), (
        "settlesense/types.py references telemetry. ReconciliationResult is "
        "hashed and goldened; a duration field in its transitive type graph "
        "makes two identical reconciliations compare unequal."
    )


@pytest.mark.determinism
def test_result_type_graph_has_no_float_or_timestamp() -> None:
    """SDD 8.1, walked recursively rather than asserted in a docstring."""
    forbidden = re.compile(r"\b(float|datetime|time|timedelta|seconds|duration)\b")
    seen: set[type] = set()
    queue: list[type] = [types.ReconciliationResult]
    checked = 0
    while queue:
        cls = queue.pop()
        if cls in seen:
            continue
        seen.add(cls)
        for spec in fields(cls):
            annotation = str(spec.type)
            assert not forbidden.search(annotation), (
                f"{cls.__name__}.{spec.name}: {annotation!r} puts wall-clock or "
                "float data in the business result (D6)."
            )
            checked += 1
            for name in re.findall(r"\w+", annotation):
                nested = getattr(types, name, None)
                if isinstance(nested, type) and nested in RECORD_TYPES:
                    queue.append(nested)
    assert checked >= 25, f"walked only {checked} annotations; the graph was not traversed"
    assert types.CaseOutcome in seen, "the walk never reached CaseOutcome"
    assert types.AuditEntry in seen, "the walk never reached AuditEntry through Exception_"


@pytest.mark.determinism
def test_no_money_field_is_annotated_as_a_bare_decimal_by_mistake() -> None:
    """Money and Decimal are the same runtime type, so only the annotation
    tells them apart. This asserts the ones that clearly mean rupees say so."""
    for cls, name in (
        (LedgerRow, "gross"),
        (SettlementLine, "net"),
        (BankRow, "amount"),
    ):
        assert name in _money_fields(cls), f"{cls.__name__}.{name} is not enrolled as Money"

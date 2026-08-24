"""M2 acceptance suite (SDD BUILD_PROMPTS, 25 numbered requirements).

Every function gets hostile inputs, not just happy paths. The numbering below
maps 1:1 to the brief so a reader can check coverage without reading the code.

TWO REQUIREMENTS ARE ASSERTED DIFFERENTLY FROM THEIR LITERAL WORDING, and both
are marked where they occur rather than quietly reinterpreted:

  #15 asks for two 12-character tokens "longest first". Equal lengths cannot be
      ordered by length, so what actually decides is the lexical tie-break. A
      third case with genuinely different lengths is added so "longest first"
      is exercised by something.

  #16 asks that a narration with no UTR returns (). True only when nothing in
      the narration is candidate-shaped. "NEFT SETTLEMENT" returns
      ("SETTLEMENT",) - deliberately, because candidates are defined
      positively and there is no stop-list. Both cases are asserted.

tests/test_normalize.py covers the same functions from the implementation
side - purity scans, the merchant-convergence proof, the no-stop-list
argument. This file is the brief's acceptance checklist and is kept separate
so a requirement cannot be quietly dropped while the other file still passes.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from settlesense.config import load_config
from settlesense.ingest import load_dataset
from settlesense.normalize import (
    AmbiguousDateError,
    DateOrder,
    extract_utr_candidates,
    normalize_merchant_name,
    normalize_utr,
    parse_amount,
    parse_date,
)
from settlesense.types import money
from tests.test_ingest import CONFIG, DATA

# ---------------------------------------------------------------------------
# parse_amount (1-10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,234.00", Decimal("1234.00")),  # 1
        ("1234.0", Decimal("1234.00")),  # 2
        ("1234", Decimal("1234.00")),  # 3
        (" ₹1,234.00 ", Decimal("1234.00")),  # 4
        ("(1234.00)", Decimal("-1234.00")),  # 5
        ("1,23,456.00", Decimal("123456.00")),  # 7 - Indian lakh grouping
    ],
)
def test_parse_amount_exact_decimal_equality(raw: str, expected: Decimal) -> None:
    """1-5, 7. Exact equality, not approximate.

    `==` on Decimal compares numeric value, so Decimal("1234.0") would satisfy
    it while carrying the wrong scale. The scale is asserted separately below
    because a value that is numerically right and unquantized still breaks the
    money invariant everything downstream assumes.
    """
    result = parse_amount(raw)
    assert result == expected
    assert str(result) == str(expected), f"{raw!r} parsed to the right value at the wrong scale"


@pytest.mark.boundary_refusal
def test_parse_amount_refuses_empty_and_garbage() -> None:
    """6. Never a silent zero - a parse failure returning 0 is
    indistinguishable from a genuine zero and lands in a conservation sum."""
    with pytest.raises(ValueError):
        parse_amount("")
    with pytest.raises(ValueError):
        parse_amount("abc")


@pytest.mark.boundary_refusal
def test_parse_amount_refuses_none() -> None:
    """6. TypeError rather than ValueError: None is not malformed text, it is
    the wrong kind of thing. str(None) would otherwise become "NONE"."""
    with pytest.raises(TypeError):
        parse_amount(None)  # type: ignore[arg-type]


def test_parse_amount_never_returns_a_float() -> None:
    """8. `type(...) is Decimal`, not isinstance.

    isinstance passes for a Decimal subclass, and `assert not isinstance(x,
    float)` on a value already typed Decimal is unreachable code that mypy
    --strict rejects. This is the strictly stronger check.
    """
    for raw in ("1,234.00", "1234", "(1234.00)", " ₹1,234.00 ", "0.00", "-1.5"):
        result = parse_amount(raw)
        assert type(result) is Decimal, f"{raw!r} produced {type(result).__name__}"


def test_half_up_not_bankers_rounding() -> None:
    """9. Python's default Decimal context is ROUND_HALF_EVEN, so this is not free.

    Under banker's rounding "0.005" gives 0.00 and "0.015" gives 0.02: the
    direction depends on the preceding digit. Both are asserted, because
    "0.005" alone would also pass under a rule that always rounds up.
    """
    assert parse_amount("0.005") == Decimal("0.01")
    assert parse_amount("0.015") == Decimal("0.02")
    assert Decimal("0.005").quantize(Decimal("0.01")) == Decimal("0.00"), (
        "the default context stopped being ROUND_HALF_EVEN; this test no longer "
        "demonstrates the difference it was written to catch"
    )
    assert Decimal("0.005").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) == Decimal("0.01")


@pytest.mark.boundary_refusal
def test_money_refuses_a_float() -> None:
    """10. D1. Note there is no type: ignore here - bool and float pass through
    mypy differently, and 1.1 against `Decimal | int | str` IS flagged, which
    is why the ignore is needed. The runtime guard is what catches it when a
    value arrives untyped from JSON or a CSV."""
    with pytest.raises(TypeError, match="refuses float"):
        money(1.1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# normalize_utr (11-13)
# ---------------------------------------------------------------------------


def test_normalize_utr_strips_and_uppercases() -> None:
    """11."""
    assert normalize_utr("utr-1234 5678") == "UTR12345678"


def test_normalize_utr_of_empty_is_empty() -> None:
    """12. Empty in, empty out - not an error. An absent UTR is a real state
    that drop_utr produces on purpose."""
    assert normalize_utr("") == ""


IDEMPOTENCE_INPUTS = (
    "utr-1234 5678",
    "3d05d31c183f6613",
    "",
    "   ",
    "!!!",
    "NEFT/3D05D31C183F6613/SETTLEMENT",
    "already-normalized",
    "ALREADYNORMALIZED",
    "mIxEd CaSe 123",
    "trailing space ",
    " leading space",
    "\ttab\tseparated\t",
    "new\nline",
    "unicode-₹-sign",
    "1234567890123456",
    "a",
    "-",
    "--__--",
    "UTR: 3D05D31C183F6613 (credit)",
    "ütf8-áccents",
)


@pytest.mark.parametrize("raw", IDEMPOTENCE_INPUTS)
def test_normalize_utr_is_idempotent(raw: str) -> None:
    """13. Twenty inputs, including empty, punctuation-only and non-ASCII.

    Idempotence matters because a UTR can be normalized at ingest and again at
    match time; a function that changed its answer on the second pass would
    make a row match or fail depending on how many times it had been touched.
    """
    once = normalize_utr(raw)
    assert normalize_utr(once) == once


def test_the_idempotence_sweep_covers_twenty_inputs() -> None:
    """A sweep that shrank to three inputs still passes and proves less."""
    assert len(IDEMPOTENCE_INPUTS) == 20
    assert len(set(IDEMPOTENCE_INPUTS)) == 20, "duplicate inputs inflate the count"


# ---------------------------------------------------------------------------
# extract_utr_candidates (14-17)
# ---------------------------------------------------------------------------


def test_one_utr_returns_exactly_that_one() -> None:
    """14. NEFT (4) and CR (2) fall below the candidate floor, so the real UTR
    is the only token left - by a positive length rule, not by naming them."""
    assert extract_utr_candidates("NEFT 3D05D31C183F6613 CR") == ("3D05D31C183F6613",)


def test_two_equal_length_tokens_both_returned_in_a_deterministic_order() -> None:
    """15, asserted as it actually behaves.

    The brief says "longest first", but both tokens are 12 characters, so
    length does not order them - the lexical tie-break does. Asserting the
    exact tuple pins that tie-break, which is the part that would otherwise
    vary with dict or set iteration order (D4).
    """
    candidates = extract_utr_candidates("NEFT ABCDEF123456 ZZZZZZ999999 CR")
    assert candidates == ("ABCDEF123456", "ZZZZZZ999999")
    assert len(candidates) == 2


def test_longest_first_when_the_lengths_actually_differ() -> None:
    """15, the half the equal-length case cannot exercise."""
    assert extract_utr_candidates("NEFT ABCDEF ABCDEF1234567 ABCDEF12 CR") == (
        "ABCDEF1234567",
        "ABCDEF12",
        "ABCDEF",
    )


def test_a_narration_with_no_candidate_shaped_token_returns_empty() -> None:
    """16."""
    assert extract_utr_candidates("NEFT CR TO AC") == ()
    assert extract_utr_candidates("") == ()


def test_no_utr_does_not_mean_empty_when_another_long_token_exists() -> None:
    """16, the honest other half.

    "NEFT SETTLEMENT" has no UTR and does NOT return (). SETTLEMENT is ten
    characters and comes back as a candidate, because candidates are defined
    positively and nothing here knows the word. Filtering it would mean a
    stop-list - "any token that is not NEFT or SETTLEMENT" - which is the exact
    bug already fixed once in the generator, where it broke the moment another
    injector introduced a word nobody had listed.

    M4 scores candidates against known batch UTRs, so a spurious candidate
    costs a comparison. A missing one costs a match.
    """
    assert extract_utr_candidates("NEFT SETTLEMENT") == ("SETTLEMENT",)


def test_candidate_ordering_is_stable_across_one_hundred_runs() -> None:
    """17. Re-derived each call rather than compared to a cached first result,
    so a function returning a memoized value would not pass by accident."""
    narration = "NEFT ABCDEF123456 ZZZZZZ999999 QQQQQQ111 AURORARETAIL SETTLEMENT"
    observed = {extract_utr_candidates(narration) for _ in range(100)}
    assert len(observed) == 1, f"ordering varied across runs: {observed}"
    assert len(next(iter(observed))) == 5, "the sample lost candidates; it tests less than it reads"


# ---------------------------------------------------------------------------
# parse_date (18-21)
# ---------------------------------------------------------------------------


def test_day_first_rule() -> None:
    """18."""
    assert parse_date("01/02/2026", DateOrder.DAY_FIRST) == date(2026, 2, 1)


def test_month_first_rule() -> None:
    """19. Same string, different rule, different date - which is precisely why
    the rule may never be inferred from the data."""
    assert parse_date("01/02/2026", DateOrder.MONTH_FIRST) == date(2026, 1, 2)


@pytest.mark.boundary_refusal
def test_ambiguous_with_no_rule_raises() -> None:
    """20. Both readings are real dates, so choosing one would be a guess.

    AmbiguousDateError is a distinct type because "a rule is missing" is
    actionable and "this text is malformed" is not, and a caller should not
    have to match on message text to tell them apart.
    """
    with pytest.raises(AmbiguousDateError):
        parse_date("01/02/2026")
    assert issubclass(AmbiguousDateError, ValueError)


@pytest.mark.boundary_refusal
def test_impossible_month_raises() -> None:
    """21. And it is NOT an ambiguity error - there is no rule that could
    rescue a thirteenth month."""
    with pytest.raises(ValueError) as caught:
        parse_date("2026-13-01")
    assert not isinstance(caught.value, AmbiguousDateError)


# ---------------------------------------------------------------------------
# Property-based (22-23)
# ---------------------------------------------------------------------------

MONEY = st.decimals(
    min_value=Decimal("-9999999.99"),
    max_value=Decimal("9999999.99"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)


WHOLE_RUPEES = st.decimals(
    min_value=Decimal("-999999"),
    max_value=Decimal("999999"),
    places=0,
    allow_nan=False,
    allow_infinity=False,
)
"""Amplified strategy for the sub-rupee rendering branches.

MEASURED, NOT ASSUMED. Over 200 draws from MONEY, `cents == 0` came up twice -
so the "1234" rendering, which is requirement 3, was being exercised about
1% of the time and one seed away from not at all. Two is below this project's
amplification threshold of ten, where a green run stops being evidence.

This strategy makes cents == 0 certain, so both sub-rupee renderings are
reached on every example rather than when the draw happens to cooperate.
"""

_RENDERINGS_SEEN: dict[str, int] = {}


def _round_trip(quantized: Decimal) -> None:
    """Render `quantized` every lossless way the dataset can, and parse each back.

    The renderings mirror gen/noise.py::_format_variant, which is what actually
    writes them. Round-tripping a value the test formatted by its own private
    rule would only prove the test and the parser agree with each other.
    """
    renderings = {
        "plain": f"{quantized:.2f}",
        "grouped": f"{quantized:,.2f}",
        "padded": f" {quantized:.2f} ",
    }
    if quantized < 0:
        renderings["parenthesised"] = f"({-quantized:,.2f})"
    cents = abs(int((quantized * 100).to_integral_value())) % 100
    if cents % 10 == 0:
        renderings["one_decimal"] = f"{quantized:.1f}"
    if cents == 0:
        renderings["integer"] = f"{quantized:.0f}"

    for shape, text in renderings.items():
        assert parse_amount(text) == quantized, (
            f"{text!r} ({shape}) did not round-trip to {quantized}"
        )
        _RENDERINGS_SEEN[shape] = _RENDERINGS_SEEN.get(shape, 0) + 1


@given(value=MONEY)
@settings(max_examples=200, deadline=None)
def test_every_valid_rendering_round_trips(value: Decimal) -> None:
    """22. Each format the dataset can contain, parsed back to the same Decimal."""
    _round_trip(money(value))


@given(value=WHOLE_RUPEES)
@settings(max_examples=100, deadline=None)
def test_sub_rupee_renderings_round_trip_at_an_amplified_rate(value: Decimal) -> None:
    """22, amplified. The "1234" and "1234.5" shapes, guaranteed on every draw."""
    quantized = money(value)
    assert quantized == quantized.to_integral_value(), "the strategy stopped producing whole rupees"
    _round_trip(quantized)


def test_every_rendering_shape_was_actually_exercised() -> None:
    """Both property tests above are meaningless if a shape never rendered.

    Ordering note: pytest runs tests in file order, so the two @given tests
    have populated the counter by the time this runs. The floor of 10 is this
    project's amplification threshold - below it, a passing run is luck rather
    than evidence.
    """
    expected = {"plain", "grouped", "padded", "parenthesised", "one_decimal", "integer"}
    missing = sorted(expected - set(_RENDERINGS_SEEN))
    assert not missing, f"rendering shape(s) never exercised: {missing}"
    thin = sorted(shape for shape, count in _RENDERINGS_SEEN.items() if count < 10)
    assert not thin, (
        f"rendering shape(s) exercised fewer than 10 times: "
        f"{ {shape: _RENDERINGS_SEEN[shape] for shape in thin} }. Below that "
        "count a green run is luck, not evidence."
    )


@given(raw=st.text(max_size=60))
@settings(max_examples=200, deadline=None)
def test_normalize_merchant_name_is_idempotent(raw: str) -> None:
    """23. Over arbitrary text, not a curated list.

    The trailing-suffix pass is what makes this non-trivial: it loops, so a
    name ending in stacked corporate forms is rewritten more than once, and a
    loop that removed one form per CALL rather than per PASS would be caught
    here and nowhere else.
    """
    once = normalize_merchant_name(raw)
    assert normalize_merchant_name(once) == once


@given(raw=st.text(max_size=60))
@settings(max_examples=200, deadline=None)
def test_normalize_merchant_name_never_invents_characters(raw: str) -> None:
    """23, strengthened. Idempotence alone is satisfied by returning "" always."""
    result = normalize_merchant_name(raw)
    assert set(result) <= set(raw.upper()), "output contains characters absent from the input"


# ---------------------------------------------------------------------------
# Integration (24-25)
# ---------------------------------------------------------------------------

DAY1_ROWS_PER_TABLE = {
    "ledger_rows": 253,
    "payment_rows": 252,
    "refund_rows": 2,
    "settlement_lines": 238,
    "settlement_batches": 3,
    "bank_rows": 0,  # settlement is T+N: day 1 has no credits yet
}

DAY1_ROWS_PER_SOURCE = {
    "gateway": 243,  # settlement_lines + settlement_batches + refund_rows
    "bank": 0,  # bank_rows
    "ledger": 505,  # ledger_rows + payment_rows
}


def test_day1_row_counts_per_table_and_per_source() -> None:
    """24. Both grains, because SDD 3.0 distinguishes them.

    Three source PACKAGES expand into six internal TABLES. Asserting only the
    six would let the gateway/ledger split drift unnoticed, and that split is
    what decides which artefacts a missing delivery takes out together.
    """
    day = load_dataset(DATA, 1, load_config(CONFIG))
    per_table = {name: len(getattr(day, name)) for name in DAY1_ROWS_PER_TABLE}
    assert per_table == DAY1_ROWS_PER_TABLE

    per_source = {
        "gateway": len(day.settlement_lines) + len(day.settlement_batches) + len(day.refund_rows),
        "bank": len(day.bank_rows),
        "ledger": len(day.ledger_rows) + len(day.payment_rows),
    }
    assert per_source == DAY1_ROWS_PER_SOURCE
    assert sum(per_source.values()) == sum(per_table.values()), (
        "the source grouping and the table grouping count different totals"
    )


def test_loading_twice_returns_equal_objects() -> None:
    """25. D6.

    Equality across the whole DayDataset, not a row count. Frozen dataclasses
    compare field by field, so this covers every Decimal, date and enum in all
    six tables - including the sort order, which is where non-determinism
    would actually surface.
    """
    config = load_config(CONFIG)
    first = load_dataset(DATA, 1, config)
    second = load_dataset(DATA, 1, config)
    assert first == second
    assert first is not second, "the loader returned a cached object; equality proves nothing"


def test_loading_twice_is_equal_on_a_day_with_every_table_populated() -> None:
    """25, extended. Day 1 has an empty bank table, so on its own it cannot
    show that bank row ordering is stable. Day 2 populates all six."""
    config = load_config(CONFIG)
    assert load_dataset(DATA, 2, config) == load_dataset(DATA, 2, config)
    assert load_dataset(DATA, 2, config).bank_rows, "day 2 bank table is empty; pick another day"

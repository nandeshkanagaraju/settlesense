"""M2 - the normalization functions, and the purity they claim.

SDD 4.1 requires every normalization to be a pure function with at least one
hostile input. The hostile inputs here are not invented: the amount shapes are
the six the frozen dataset actually contains, and the merchant name forms are
the three the generator actually emits.

The tests that matter most are the ones that would pass under a WRONG
implementation if written carelessly, and each says so where that applies.
"""

from __future__ import annotations

import ast
import inspect
import random
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from gen.noise import _name_variant
from gen.profiles import PROFILES
from settlesense import normalize
from settlesense.normalize import (
    UTR_CANDIDATE_MAX_LEN,
    UTR_CANDIDATE_MIN_LEN,
    UTR_LEN,
    AmbiguousDateError,
    DateOrder,
    extract_utr_candidates,
    normalize_merchant_name,
    normalize_narration,
    normalize_utr,
    parse_amount,
    parse_date,
)

# A real narration from data/day3_bank.csv, unmodified.
REAL_NARRATION = "NEFT 3D05D31C183F6613 AURORA RETAIL SETTLEMENT"
REAL_UTR = "3D05D31C183F6613"


# ---------------------------------------------------------------------------
# normalize_utr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3d05d31c183f6613", REAL_UTR),
        ("  3D05-D31C 183F/6613  ", REAL_UTR),
        ("UTR: 3D05D31C183F6613", "UTR3D05D31C183F6613"),
        ("", ""),
        ("!!!", ""),
    ],
)
def test_normalize_utr(raw: str, expected: str) -> None:
    assert normalize_utr(raw) == expected


def test_normalize_utr_is_idempotent() -> None:
    once = normalize_utr(" 3d05-d31c 183f6613 ")
    assert normalize_utr(once) == once


@pytest.mark.boundary_refusal
def test_normalize_utr_refuses_non_string() -> None:
    """FAULT INJECTION. A None narration would otherwise become "NONE"."""
    with pytest.raises(TypeError, match="expects str"):
        normalize_utr(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_utr_candidates
# ---------------------------------------------------------------------------


def test_the_real_utr_is_among_the_candidates_of_a_real_narration() -> None:
    assert REAL_UTR in extract_utr_candidates(REAL_NARRATION)


def test_a_truncated_utr_is_still_a_candidate() -> None:
    """truncate_utr keeps a 6-10 character prefix. The floor must admit it."""
    for keep in range(UTR_CANDIDATE_MIN_LEN, 11):
        narration = REAL_NARRATION.replace(REAL_UTR, REAL_UTR[:keep])
        assert REAL_UTR[:keep] in extract_utr_candidates(narration), (
            f"a {keep}-character truncated UTR was not offered as a candidate; "
            "M4 would abstain on a row it could have resolved"
        )


def test_a_garbled_utr_is_still_a_candidate() -> None:
    """garbled_narration transposes two characters. Length is unchanged, so the
    token remains plausible - which is the point: it is recoverable by edit
    distance, and dropping it here would put it beyond M4's reach."""
    garbled = REAL_UTR[:4] + REAL_UTR[5] + REAL_UTR[4] + REAL_UTR[6:]
    assert garbled in extract_utr_candidates(REAL_NARRATION.replace(REAL_UTR, garbled))


def test_candidates_are_defined_positively_with_no_stop_list() -> None:
    """THE test that would pass under a wrong implementation if written loosely.

    NEFT is 4 characters and falls below the length floor - excluded by a
    positive rule. SETTLEMENT is 10 characters and IS returned, because
    nothing here knows the word. Asserting its presence is the point: a
    stop-list implementation would drop it, look tidier, and break the moment
    another injector introduced a word nobody listed. That exact bug was fixed
    in the generator once already.
    """
    candidates = extract_utr_candidates(REAL_NARRATION)
    assert "SETTLEMENT" in candidates, (
        "SETTLEMENT was filtered out, which means a stop-list crept in. "
        "Never define a token by what it is not."
    )
    assert "NEFT" not in candidates, "NEFT is 4 chars and below the length floor"


def test_candidate_order_is_total_and_longest_first() -> None:
    """Ties must not fall back to narration order (D4)."""
    narration = "ZZZZZZ AAAAAA LONGERTOKEN"
    assert extract_utr_candidates(narration) == ("LONGERTOKEN", "AAAAAA", "ZZZZZZ")
    shuffled = "AAAAAA LONGERTOKEN ZZZZZZ"
    assert extract_utr_candidates(narration) == extract_utr_candidates(shuffled)


def test_a_long_merchant_token_can_outrank_a_truncated_utr() -> None:
    """Documented consequence, asserted rather than left as a docstring claim.

    This is WHY M4 scores candidates against known batch UTRs instead of
    taking [0]. If this test ever fails because the ranking changed, M4's
    contract changed with it.
    """
    narration = "NEFT 3D05D3 AURORARETAIL SETTLEMENT"
    candidates = extract_utr_candidates(narration)
    assert candidates[0] == "AURORARETAIL"
    assert "3D05D3" in candidates
    assert candidates.index("AURORARETAIL") < candidates.index("3D05D3")


def test_candidates_are_deduplicated() -> None:
    doubled = f"NEFT {REAL_UTR} {REAL_UTR} SETTLEMENT"
    assert extract_utr_candidates(doubled).count(REAL_UTR) == 1


def test_tokens_longer_than_a_utr_are_not_candidates() -> None:
    """A positive bound: a UTR is at most UTR_LEN characters."""
    too_long = "A" * (UTR_CANDIDATE_MAX_LEN + 1)
    assert too_long not in extract_utr_candidates(f"NEFT {too_long} SETTLEMENT")
    assert UTR_CANDIDATE_MAX_LEN == UTR_LEN


def test_a_narration_with_no_utr_yields_no_false_certainty() -> None:
    """drop_utr removes it entirely. Returning () is correct; returning a
    merchant token as though it were a UTR would be a false link."""
    assert extract_utr_candidates("NEFT SETTLEMENT") == ("SETTLEMENT",)
    assert extract_utr_candidates("") == ()


# ---------------------------------------------------------------------------
# parse_amount
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1234.00", "1234.00"),
        ("1,234.00", "1234.00"),  # thousands separated
        ("1234.0", "1234.00"),  # one decimal place
        ("1234", "1234.00"),  # integer
        (" 1234.00 ", "1234.00"),  # whitespace padded
        ("₹1,234.00", "1234.00"),  # rupee sign
        ("(1234.00)", "-1234.00"),  # parenthesised debit
        ("(1,234.00)", "-1234.00"),  # both, as the dataset contains
        ("-1234.00", "-1234.00"),  # signed negative
        ("0.00", "0.00"),
        ("(0.00)", "0.00"),  # never "-0.00"
        ("1,23,456.00", "123456.00"),  # Indian lakh grouping
    ],
)
def test_parse_amount_handles_every_shape_the_dataset_contains(raw: str, expected: str) -> None:
    assert str(parse_amount(raw)) == expected


def test_parse_amount_returns_quantized_money() -> None:
    assert str(parse_amount("1234.5")) == "1234.50"
    assert str(parse_amount("1234.567")) == "1234.57"


@pytest.mark.boundary_refusal
@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "abc",
        "1.2.3",
        "12 34.00",  # inner space: never guess which digits belong together
        "(1234.00",  # unbalanced
        "1234.00)",
        "1234-",
        "NULL",
        "-",
        ".50",  # no integer part
        "1234.",  # no fractional digits
    ],
)
def test_parse_amount_raises_and_never_silently_returns_zero(raw: str) -> None:
    """FAULT INJECTION. A zero returned from a parse failure is
    indistinguishable from a genuine zero and lands in a conservation sum."""
    with pytest.raises(ValueError) as caught:
        parse_amount(raw)
    assert repr(raw) in str(caught.value), "the error must quote the offending input"


@pytest.mark.boundary_refusal
def test_parse_amount_refuses_a_double_negative() -> None:
    """FAULT INJECTION. "(-5.00)" is negative twice; which was meant is not
    recoverable, so resolving it either way would be a guess."""
    with pytest.raises(ValueError, match="negative twice"):
        parse_amount("(-5.00)")


@pytest.mark.boundary_refusal
def test_parse_amount_refuses_non_string() -> None:
    """FAULT INJECTION. A float slipping through here bypasses D1 entirely."""
    with pytest.raises(TypeError, match="expects str"):
        parse_amount(1234.56)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------


def test_iso_parses_without_a_profile() -> None:
    assert parse_date("2026-09-01") == date(2026, 9, 1)
    assert parse_date("  2026-09-01  ") == date(2026, 9, 1)
    assert parse_date("2026/09/01") == date(2026, 9, 1)


def test_a_single_valid_reading_is_deduced_not_guessed() -> None:
    """25 is not a month, so "25/03/2026" has exactly one reading.

    Deducing the only possibility is not guessing. This distinction is the
    whole content of SDD 4.1's "never guessed", and conflating the two would
    make the parser refuse input it can read unambiguously.
    """
    assert parse_date("25/03/2026") == date(2026, 3, 25)
    assert parse_date("03/25/2026") == date(2026, 3, 25)


@pytest.mark.boundary_refusal
def test_two_valid_readings_raise_rather_than_preferring_one() -> None:
    """FAULT INJECTION for the rule SDD 4.1 actually states.

    Both 3 April and 4 March are real. Silently preferring day-first because
    the data is Indian is exactly the guess the spec prohibits.
    """
    with pytest.raises(AmbiguousDateError) as caught:
        parse_date("03/04/2026")
    assert "2026-04-03" in str(caught.value) and "2026-03-04" in str(caught.value)


def test_a_profile_resolves_the_ambiguity_both_ways() -> None:
    assert parse_date("03/04/2026", DateOrder.DAY_FIRST) == date(2026, 4, 3)
    assert parse_date("03/04/2026", DateOrder.MONTH_FIRST) == date(2026, 3, 4)


@pytest.mark.boundary_refusal
def test_a_profile_rule_is_authoritative_and_does_not_fall_back() -> None:
    """FAULT INJECTION for the rule's teeth.

    "25/03/2026" read month-first has no 25th month. The other reading IS
    valid, and substituting it would make the configured rule advisory. A rule
    that yields when inconvenient is not a rule - and worse, it would make the
    parser's output depend on data rather than on configuration.
    """
    with pytest.raises(ValueError, match="month-first"):
        parse_date("25/03/2026", DateOrder.MONTH_FIRST)


@pytest.mark.determinism
@pytest.mark.boundary_refusal
def test_month_name_formats_are_refused_because_they_are_locale_dependent() -> None:
    """FAULT INJECTION for a determinism decision, not just a parsing one.

    strptime's %b resolves month abbreviations through the process locale, so
    "01-MAR-2026" parses on one machine and raises on another with no code
    change between them. Accepting it would put a locale dependency inside a
    system whose results are byte-compared across machines.
    """
    for raw in ("01-MAR-2026", "01 Mar 2026", "MAR 01 2026"):
        with pytest.raises(ValueError, match="matches no allowed format"):
            parse_date(raw)


@pytest.mark.boundary_refusal
@pytest.mark.parametrize(
    "raw",
    ["", "   ", "not a date", "2026-13-01", "32/01/2026", "01/01/26", "1/1/2026/", "2026-02-30"],
)
def test_parse_date_refuses_malformed_input(raw: str) -> None:
    """FAULT INJECTION. A two-digit year is ambiguous about its century and no
    profile rule could resolve it honestly."""
    with pytest.raises(ValueError):
        parse_date(raw)


@pytest.mark.boundary_refusal
def test_both_readings_invalid_raises_a_plain_value_error_not_ambiguity() -> None:
    """FAULT INJECTION. "32/45/2026" is malformed, not ambiguous - a caller
    matching on AmbiguousDateError to mean "a profile rule is missing" would
    otherwise be told to add config that could not help."""
    with pytest.raises(ValueError) as caught:
        parse_date("32/45/2026")
    assert not isinstance(caught.value, AmbiguousDateError)


def test_parse_date_does_not_enforce_2026() -> None:
    """D13 is a dataset invariant, checked in ingest.py where the dataset is.

    Baking a year into a parser makes the parser wrong the moment the
    simulated window moves, and hides a window violation behind a parse error.
    """
    assert parse_date("2027-01-01") == date(2027, 1, 1)


# ---------------------------------------------------------------------------
# narrations and merchant names
# ---------------------------------------------------------------------------


def test_normalize_narration_uppercases_and_collapses() -> None:
    assert normalize_narration("  neft   3d05   aurora  ") == "NEFT 3D05 AURORA"


def test_punctuation_becomes_a_space_and_does_not_fuse_tokens() -> None:
    """Deleting punctuation would turn a slash-delimited narration into one
    unusable token - both an unrecognisable merchant and an impossible UTR."""
    assert normalize_narration("NEFT/3D05D31C183F6613/SETTLEMENT") == (
        "NEFT 3D05D31C183F6613 SETTLEMENT"
    )
    assert "3D05D31C183F6613" in extract_utr_candidates("NEFT/3D05D31C183F6613/SETTLEMENT")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CARBON WORKS PVT LTD", "CARBONWORKS"),
        ("Carbon Works Private Limited", "CARBONWORKS"),
        ("CARBON WORKS", "CARBONWORKS"),
        ("CARBONWORKSPVTLTD", "CARBONWORKS"),  # the second pass
        ("AURORA RETAIL", "AURORARETAIL"),
        ("AURORARETAIL", "AURORARETAIL"),
        ("AURORA RETAIL PVT", "AURORARETAIL"),
    ],
)
def test_normalize_merchant_name(raw: str, expected: str) -> None:
    assert normalize_merchant_name(raw) == expected


def test_every_generator_name_variant_that_can_converge_does() -> None:
    """THE decisive merchant test, run against the generator's real variants.

    _name_variant emits three styles. Two of them - the corporate-suffix form
    and the space-stripped form - are pure renderings of the same name and
    MUST normalize to one string. The third abbreviates a word ("AURORA
    RETAIL" -> "AURORA RTL") and deliberately must NOT converge: recovering it
    is edit-distance work and belongs to M4, and quietly making normalization
    fuzzy would hide a real matching decision inside a string function.
    """
    checked = 0
    for profile in PROFILES:
        base = profile.merchant_name
        suffix_form = _name_variant(base, _FixedRng(1))
        unspaced = _name_variant(base, _FixedRng(2))
        normalized = {normalize_merchant_name(form) for form in (base, suffix_form, unspaced)}
        assert len(normalized) == 1, (
            f"{profile.name}: {base!r}, {suffix_form!r} and {unspaced!r} normalized "
            f"to {sorted(normalized)}. These are renderings of one merchant; a "
            "split here is a silent merchant mismatch, not an error anyone sees."
        )
        abbreviated = _name_variant(base, _FixedRng(0))
        if abbreviated != base:
            assert normalize_merchant_name(abbreviated) not in normalized, (
                "an abbreviated name converged, which means normalization has "
                "become fuzzy. Edit-distance recovery belongs in M4."
            )
        checked += 1
    assert checked == 3, f"checked {checked} profiles, expected 3"


def test_the_trailing_suffix_pass_is_load_bearing() -> None:
    """FAULT INJECTION for the second pass, by running the naive version.

    Dropping PVT/LTD as whole TOKENS is the obvious implementation and handles
    two of the three forms. It leaves "CARBONWORKSPVTLTD" untouched, which
    then fails to match the other two. This asserts the naive result is wrong
    and the real one is right, so the pass cannot be removed as redundant.
    """
    unspaced = "CARBONWORKSPVTLTD"
    suffixes = ("PRIVATE", "LIMITED", "PVT", "LTD")
    naive = "".join(t for t in normalize_narration(unspaced).split() if t not in suffixes)

    assert naive == "CARBONWORKSPVTLTD", "the naive implementation changed; revisit this test"
    assert naive != normalize_merchant_name("CARBON WORKS PVT LTD"), (
        "token-only stripping would have converged by luck; this test proves nothing"
    )
    assert normalize_merchant_name(unspaced) == "CARBONWORKS"


def test_a_name_of_only_corporate_forms_does_not_normalize_to_empty() -> None:
    """Every such name would otherwise collide with every other one."""
    assert normalize_merchant_name("PRIVATE LIMITED") != ""
    assert normalize_merchant_name("PVT LTD") != ""


class _FixedRng(random.Random):
    """Returns a fixed style index, so each _name_variant branch is reachable.

    Subclasses random.Random rather than duck-typing it: _name_variant is
    annotated to take a Random, and a stub that merely happens to have a
    randrange method would drift silently the day it starts calling another
    method of the interface.
    """

    def __init__(self, value: int) -> None:
        super().__init__(0)
        self._value = value

    def randrange(self, *args: int, **kwargs: object) -> int:  # type: ignore[override]
        return self._value


# ---------------------------------------------------------------------------
# Purity (SDD 4.1: "every normalization is a pure function")
# ---------------------------------------------------------------------------

FORBIDDEN_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "pathlib",
        "csv",
        "json",
        "random",
        "time",
        "socket",
        "sqlite3",
        "subprocess",
        "urllib",
        "requests",
        "settlesense.config",
        "settlesense.ingest",
    }
)
FORBIDDEN_CALLS = frozenset({"open", "input", "eval", "exec", "compile"})
FORBIDDEN_ATTRIBUTES = frozenset({"now", "utcnow", "today", "monotonic", "read_text", "read_bytes"})


def _impurities(source: str) -> list[str]:
    """Every impurity in a module's source. Shared by the scan and its injection."""
    found: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS or alias.name in FORBIDDEN_IMPORTS:
                    found.append(f"line {line}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] in FORBIDDEN_IMPORTS or module in FORBIDDEN_IMPORTS:
                found.append(f"line {line}: from {module} import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                found.append(f"line {line}: {func.id}()")
            elif isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_ATTRIBUTES:
                found.append(f"line {line}: .{func.attr}()")
    return found


@pytest.mark.determinism
def test_normalize_is_pure() -> None:
    """No I/O, no clock, no randomness, no config lookup (D2, D3).

    This is the reason load_dataset lives in settlesense/ingest.py. Moving a
    CSV reader in here would force this scan to grow an allow-list, and an
    allow-list of one is how an allow-list of five begins.
    """
    source = Path(inspect.getfile(normalize)).read_text(encoding="utf-8")
    assert not _impurities(source), f"settlesense/normalize.py is not pure: {_impurities(source)}"


@pytest.mark.charter_guard
def test_the_purity_scanner_catches_a_planted_impurity() -> None:
    """FAULT INJECTION for the scanner itself.

    A scanner that returns an empty list for every input passes the test above
    forever. Each planted line is checked individually so that one working
    detector cannot mask three broken ones.
    """
    plants = {
        "import os": "import os\n",
        "from pathlib": "from pathlib import Path\n",
        "open()": "def f():\n    return open('x')\n",
        ".now()": "import datetime\ndef f():\n    return datetime.datetime.now()\n",
        ".today()": "import datetime\ndef f():\n    return datetime.date.today()\n",
        "random": "import random\n",
    }
    undetected = [label for label, source in plants.items() if not _impurities(source)]
    assert not undetected, f"the purity scanner missed: {undetected}"


@pytest.mark.determinism
def test_normalize_does_not_import_from_gen() -> None:
    """The hard rule from SDD 2, asserted at this module specifically."""
    source = Path(inspect.getfile(normalize)).read_text(encoding="utf-8")
    assert "import gen" not in source and "from gen" not in source


def test_every_public_function_is_deterministic_across_repeated_calls() -> None:
    """Purity implies this, but the scan is structural and this is behavioural."""
    samples = (REAL_NARRATION, "  ₹1,234.00  ", "2026-09-01", "CARBON WORKS PVT LTD")
    for _ in range(3):
        assert extract_utr_candidates(REAL_NARRATION) == extract_utr_candidates(REAL_NARRATION)
        assert normalize_narration(samples[0]) == normalize_narration(samples[0])
        assert parse_amount(samples[1]) == Decimal("1234.00")
        assert parse_date(samples[2]) == date(2026, 9, 1)
        assert normalize_merchant_name(samples[3]) == "CARBONWORKS"

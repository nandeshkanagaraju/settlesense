"""M2 - Pure normalization for UTRs, amounts, dates, narrations and names.

PURE MEANS PURE. No file access, no clock, no randomness, no config lookup, no
network. Every function here is a total function of its arguments, which is
what lets the unit tests feed hostile input directly instead of building a
dataset first. `test_normalize_is_pure` AST-scans this module for `open`,
`Path`, `datetime.now`, `date.today`, `time.` and `random.`, so the property is
enforced rather than described.

Reading the day's CSVs is I/O and therefore lives in settlesense/ingest.py.

SDD 4.1. Three rules run through everything below:

  NEVER GUESS. An input with two valid readings raises. Deducing the only
  valid reading is not guessing; picking one of two is.

  NEVER DEFINE A TOKEN BY WHAT IT IS NOT. `merchant_name_variants` in the
  generator was originally written as "every token that is not NEFT or the
  UTR" and broke the moment another injector touched the narration first.
  Nothing here uses a stop-list.

  NEVER SILENTLY RETURN ZERO. An unparseable amount raises with the offending
  text. A zero returned from a parse failure is indistinguishable from a
  genuine zero and lands in a conservation sum as if it were real.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final

from settlesense.types import Money, money

__all__ = [
    "UTR_CANDIDATE_MAX_LEN",
    "UTR_CANDIDATE_MIN_LEN",
    "UTR_LEN",
    "AmbiguousDateError",
    "DateOrder",
    "extract_utr_candidates",
    "normalize_merchant_name",
    "normalize_narration",
    "normalize_utr",
    "parse_amount",
    "parse_date",
]


# ---------------------------------------------------------------------------
# UTR
# ---------------------------------------------------------------------------

UTR_LEN: Final[int] = 16
"""A full UTR. SettlementBatch.utr is 16 characters (SDD 3.3)."""

UTR_CANDIDATE_MIN_LEN: Final[int] = 6
"""Shortest token that could still identify a batch.

A deliberate coupling to the truncation floor the generator uses, and the
reason it is a named constant rather than a literal 6: raising it silently
drops real truncated UTRs from the candidate list, and M4's fuzzy matcher
would then abstain on rows it could have resolved, with nothing in the output
saying why.
"""

UTR_CANDIDATE_MAX_LEN: Final[int] = UTR_LEN
"""A token longer than a full UTR is not a damaged UTR - it is something else.

A POSITIVE bound (a UTR is at most this long), not an exclusion list.
"""

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def normalize_utr(raw: str) -> str:
    """Uppercase, then keep only A-Z and 0-9.

    SDD 4.1 words this as "uppercase, strip non-alphanumerics, collapse
    whitespace". The third clause is subsumed by the second - once every
    non-alphanumeric character is gone there is no whitespace left to collapse
    - so this is two operations, not three. Said explicitly because a reader
    checking the code against the spec will otherwise look for a missing step.

    Callers keep the raw value alongside this one; SDD 4.1 requires both to be
    retained, and evidence links must quote what the bank actually printed.
    """
    if not isinstance(raw, str):
        raise TypeError(f"normalize_utr expects str, got {type(raw).__name__} ({raw!r})")
    return _NON_ALNUM.sub("", raw.upper())


def extract_utr_candidates(narration: str) -> tuple[str, ...]:
    """Every plausible UTR token in a narration, ranked longest-first.

    Returns ALL candidates rather than a best guess. A narration whose UTR was
    truncated to six characters is genuinely ambiguous between batches sharing
    that prefix, and collapsing that to one answer here would hide the
    ambiguity from the layer built to weigh it (M4 fuzzy UTR).

    PLAUSIBLE IS DEFINED POSITIVELY: alphanumeric, case-invariant under
    upper(), and between UTR_CANDIDATE_MIN_LEN and UTR_CANDIDATE_MAX_LEN
    characters. There is deliberately NO stop-list of narration furniture like
    NEFT or SETTLEMENT. Defining a token as "not one of these known words" is
    the exact bug this project already fixed once in the generator: the moment
    another injector adds a word, the definition silently admits it.

    CONSEQUENCE, STATED RATHER THAN HIDDEN: a merchant name rendered without
    spaces - "AURORARETAIL", 12 characters - outranks a 6-character truncated
    UTR under a longest-first rank. That is why M4 SCORES candidates against
    known batch UTRs instead of taking [0]. A caller that takes [0] and treats
    it as the answer has misread this function.

    Ordering is total: length descending, then the token ascending. Two
    candidates of equal length can never tie, so the result does not depend on
    the order tokens happened to appear in the narration (D4).
    """
    if not isinstance(narration, str):
        raise TypeError(
            f"extract_utr_candidates expects str, got {type(narration).__name__} ({narration!r})"
        )
    seen: set[str] = set()
    for token in normalize_narration(narration).split():
        if not token.isalnum():
            continue
        if not UTR_CANDIDATE_MIN_LEN <= len(token) <= UTR_CANDIDATE_MAX_LEN:
            continue
        seen.add(token)
    return tuple(sorted(seen, key=lambda token: (-len(token), token)))


# ---------------------------------------------------------------------------
# Amounts
# ---------------------------------------------------------------------------

_RUPEE_SIGN: Final[str] = "₹"
_BARE_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


def parse_amount(raw: str) -> Money:
    """Parse external amount text to quantized Decimal. Unparseable input raises.

    Handles every shape the frozen dataset actually contains - plain,
    thousands-separated, whitespace-padded, one-decimal, integer, and
    parenthesised negative - plus the rupee sign that SDD 4.1 names.

    THOUSANDS SEPARATORS ARE STRIPPED WITHOUT VALIDATING THEIR PLACEMENT.
    "1,23,456.00" is Indian lakh grouping and is correct; "1,234,567.00" is
    western grouping and is also correct. Enforcing one would reject
    legitimate input from a real Indian bank statement, and the grouping
    carries no information the digits do not.

    MORE THAN TWO DECIMAL PLACES IS ROUNDED, NOT REJECTED, because money() is
    the single definition of what a rupee amount is and it quantizes
    ROUND_HALF_UP. Sub-paise input is not expected from any source in this
    project; if one ever appears, this is where to reconsider.

    A negative may be written EITHER parenthesised OR signed, never both:
    "(-5.00)" is a double negative whose author's intent is unknowable, so it
    raises rather than resolving to one reading.
    """
    if not isinstance(raw, str):
        raise TypeError(f"parse_amount expects str, got {type(raw).__name__} ({raw!r})")

    text = raw.strip().replace(_RUPEE_SIGN, "").strip()
    if not text:
        raise ValueError(
            f"parse_amount cannot parse {raw!r}: the value is empty. Returning 0 "
            "here would be indistinguishable from a genuine zero amount."
        )

    parenthesised = False
    if text.startswith("(") and text.endswith(")"):
        parenthesised = True
        text = text[1:-1].strip()
    elif "(" in text or ")" in text:
        raise ValueError(
            f"parse_amount cannot parse {raw!r}: unbalanced parenthesis. A "
            "parenthesised debit must be wrapped on both sides."
        )

    text = text.replace(",", "").strip()

    if parenthesised and text.startswith("-"):
        raise ValueError(
            f"parse_amount refuses {raw!r}: it is negative twice, once by "
            "parenthesis and once by sign. Which one was meant is not "
            "recoverable from the text."
        )
    if not _BARE_NUMBER.match(text):
        raise ValueError(
            f"parse_amount cannot parse {raw!r}: after removing the currency "
            f"sign, separators and any parenthesis it reads {text!r}, which is "
            "not a bare decimal number."
        )

    value = Decimal(text)
    if parenthesised:
        value = -value
    if value == 0:
        value = abs(value)  # never emit "-0.00"
    return money(value)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


class DateOrder(StrEnum):
    """How a source writes a numeric date whose day and month are both <= 12."""

    DAY_FIRST = "day_first"  # 03/04/2026 -> 3 April
    MONTH_FIRST = "month_first"  # 03/04/2026 -> 4 March


class AmbiguousDateError(ValueError):
    """Raised when a date has two valid readings and no rule to choose between.

    A distinct type because "this date is ambiguous" is actionable - it means a
    profile rule is missing - while "this date is malformed" is not, and a
    caller that wants to distinguish them should not have to match on message
    text.
    """


_UNAMBIGUOUS_FORMATS: Final[tuple[str, ...]] = ("%Y-%m-%d", "%Y/%m/%d")
"""Year-first formats, which cannot be misread.

MONTH-NAME FORMATS ARE DELIBERATELY ABSENT. strptime's %b resolves month
abbreviations through the process locale, so "01-MAR-2026" would parse on one
machine and raise on another with no code change between them. That is exactly
the class of hidden non-determinism the charter exists to prevent, and a
locale-dependent date parser inside a system whose results are byte-compared
would be a defect waiting for a different machine to find. Adding month-name
support means adding an explicit month table here, never %b.
"""

_NUMERIC_DATE = re.compile(r"^(\d{1,2})([/-])(\d{1,2})\2(\d{4})$")
"""DD/MM/YYYY or MM/DD/YYYY, with a matching separator on both sides.

A four-digit year is required. A two-digit year is ambiguous about its century
and there is no profile rule that could resolve it honestly.
"""


def parse_date(raw: str, profile: DateOrder | None = None) -> date:
    """Parse a date from an allow-list of formats. Ambiguity raises.

    `profile` is the RESOLVED day/month rule for the source being read, not a
    merchant profile name. Passing a name would mean looking it up, which would
    make this module read config and stop being pure; the caller resolves the
    name and passes the rule.

    THE DISTINCTION THAT MATTERS. "25/03/2026" has exactly one valid reading -
    there is no 25th month - so this returns 25 March with no profile. That is
    deduction, not guessing. "03/04/2026" has two valid readings, so without a
    profile rule it raises AmbiguousDateError. SDD 4.1 says ambiguity is
    resolved by config and never guessed; silently preferring day-first because
    the data happens to be Indian is precisely the guess it prohibits.

    When a profile IS given it is authoritative and strict: if the reading it
    dictates is not a real date, that raises rather than falling back to the
    other reading. A fallback would make the rule advisory, and a rule that
    quietly yields is not a rule.

    NOT ENFORCED HERE: that the result falls in 2026 (D13). That is a property
    of this dataset, not of date parsing, and belongs where dataset invariants
    are checked - settlesense/ingest.py. Baking a year into a parser makes the
    parser wrong the moment the simulated window moves.
    """
    if not isinstance(raw, str):
        raise TypeError(f"parse_date expects str, got {type(raw).__name__} ({raw!r})")
    text = raw.strip()
    if not text:
        raise ValueError("parse_date cannot parse an empty value")

    for pattern in _UNAMBIGUOUS_FORMATS:
        try:
            # DTZ007 is suppressed below: a naive datetime is correct here and a
            # timezone would be wrong. This parses a calendar DATE - a value date, a
            # settlement date - and .date() discards the time before it can be
            # read. Requiring %z would make every date in the dataset
            # unparseable in exchange for a timezone nothing consults.
            return datetime.strptime(text, pattern).date()  # noqa: DTZ007
        except ValueError:
            continue

    match = _NUMERIC_DATE.match(text)
    if match is None:
        raise ValueError(
            f"parse_date cannot parse {raw!r}: it matches no allowed format. "
            f"Allowed: {list(_UNAMBIGUOUS_FORMATS)} or a numeric D/M/YYYY or "
            "M/D/YYYY with a four-digit year."
        )

    first, _, second, year_text = match.groups()
    year, left, right = int(year_text), int(first), int(second)

    day_first = _maybe_date(year, right, left)
    month_first = _maybe_date(year, left, right)

    if profile is DateOrder.DAY_FIRST:
        return _require(day_first, raw, "day-first")
    if profile is DateOrder.MONTH_FIRST:
        return _require(month_first, raw, "month-first")

    valid = [reading for reading in (day_first, month_first) if reading is not None]
    if not valid:
        raise ValueError(
            f"parse_date cannot parse {raw!r}: neither {left}/{right} nor "
            f"{right}/{left} is a real day and month in {year}."
        )
    if day_first is not None and month_first is not None and day_first != month_first:
        raise AmbiguousDateError(
            f"parse_date refuses {raw!r}: it reads as {day_first.isoformat()} "
            f"day-first and {month_first.isoformat()} month-first, and both are "
            "real dates. Supply the source's DateOrder. SDD 4.1: ambiguous "
            "dates are resolved by configuration, never guessed."
        )
    return valid[0]


def _maybe_date(year: int, month: int, day: int) -> date | None:
    """The date if it exists, else None. Used to count valid readings."""
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _require(reading: date | None, raw: str, rule: str) -> date:
    if reading is None:
        raise ValueError(
            f"parse_date cannot read {raw!r} as {rule}, which is the configured "
            "rule for this source. The other reading is not substituted: a rule "
            "that yields when inconvenient is not a rule."
        )
    return reading


# ---------------------------------------------------------------------------
# Narrations and merchant names
# ---------------------------------------------------------------------------

_PUNCTUATION = re.compile(r"[^A-Z0-9]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_narration(raw: str) -> str:
    """Uppercase, replace punctuation with a space, collapse whitespace.

    PUNCTUATION BECOMES A SPACE RATHER THAN BEING DELETED. Deleting it would
    turn "NEFT/AB12CD34/SETTLEMENT" into one 24-character token that is both an
    unusable UTR candidate and an unrecognisable merchant name. Replacing
    preserves the token boundaries the bank actually printed.
    """
    if not isinstance(raw, str):
        raise TypeError(f"normalize_narration expects str, got {type(raw).__name__} ({raw!r})")
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", raw.upper())).strip()


_LEGAL_SUFFIXES: Final[tuple[str, ...]] = ("PRIVATE", "LIMITED", "PVT", "LTD")
"""Corporate-form words that carry no identifying information.

Ordered longest-first so the trailing-strip loop removes LIMITED before it can
mistake the tail of it for LTD.
"""


def normalize_merchant_name(raw: str) -> str:
    """Uppercase, drop corporate-form words, remove spaces.

    Two forms of the same name must converge, because the generator emits both:
    "CARBON WORKS PVT LTD" and its space-stripped variant "CARBONWORKSPVTLTD"
    are one merchant.

    THIS NEEDS TWO PASSES, AND THE SECOND IS THE ONE THAT IS EASY TO MISS.
    Dropping PVT and LTD as whole TOKENS handles "CARBON WORKS PVT LTD" and
    "CARBON WORKS". It does nothing for "CARBONWORKSPVTLTD", where the words
    are no longer separate tokens - that form would normalize to itself and
    fail to match the other two, which is a silent merchant mismatch rather
    than an error anyone would see. So a second pass strips the same words from
    the END of the joined string, repeatedly.

    Trailing-only, and never to empty. A name whose every word is a corporate
    form ("PRIVATE LIMITED") keeps its last word rather than normalizing to ""
    and colliding with every other such name.

    KNOWN COST: a merchant genuinely ending in these letters loses them. No
    such name exists in this dataset's three profiles, and matching a real
    merchant list is a config problem rather than a parsing one, but the
    tradeoff is real and is recorded rather than assumed away.
    """
    tokens = normalize_narration(raw).split()
    kept = [token for token in tokens if token not in _LEGAL_SUFFIXES]
    if not kept:
        # Every word was a corporate form. Dropping them all leaves "", which
        # collides with every other such name AND with a genuinely empty input.
        # The emptiness guard on the trailing pass below cannot help here: it
        # only refuses to make a name empty, and by then it already is.
        kept = tokens
    joined = "".join(kept)
    stripping = True
    while stripping:
        stripping = False
        for suffix in _LEGAL_SUFFIXES:
            if joined.endswith(suffix) and len(joined) > len(suffix):
                joined = joined[: -len(suffix)]
                stripping = True
                break
    return joined

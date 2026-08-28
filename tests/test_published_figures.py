"""Every number the README and LIMITATIONS publish must exist in an artifact.

THE DEFECT CLASS. A figure that reads as a measurement and has no file behind
it cannot be checked, cannot be reproduced, and stays true-looking forever.
This project has shipped three:

  "about Rs2.49 per 1,000 rows"   - no derivation recorded anywhere
  "~90 days" batch density        - retracted, basis never stated
  "0.861 s wall clock" on the holdout, "about 5,800 cases/s end to end"

The third shows why reading cannot catch these. LIMITATIONS said, in its own
section heading, that the held-out set has no throughput figure and will not
get one - and the README published one for that exact run. Both documents were
internally consistent. Nothing compared either to the artifacts, so nothing
compared them to each other.

MATCHED AS NUMBERS, NOT AS TEXT. Every numeric literal in every artifact is
parsed once into a set, with each value also rounded to 0-6 decimal places, so
a document that writes 0.949 matches an artifact holding 0.948718 and a
document that writes 1.0456% matches a stored ratio of 0.010456. Substring
matching was tried first and was both wrong and slow: it made `2.49` match the
`2` inside any longer number, and re-scanning 11.7MB of truth files for every
figure took 158 seconds - on its own more than the whole suite is allowed.

WHAT THIS DOES NOT ESTABLISH. That the figure came from that artifact - only
that a reader can open a file and find the number. TWO-DIGIT FIGURES ARE OUT OF
SCOPE and are not reliably checkable: `90` occurs as a value somewhere in a
config or a truth file, so the "~90 days" escape above would NOT have been
caught here. `2.49`, `0.861` and `64,286` all would. Three significant digits
is where matching stops being an accident, and claiming more reach than that
would make this module the thing it exists to prevent.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from functools import cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCUMENTS = ("README.md", "LIMITATIONS.md")

ARTIFACT_GLOBS = (
    "reports/*",
    "config/*",
    "fixtures/llm_manifest*.json",
    "*MANIFEST.json",
    "tests/collection_baseline.json",
    "data/*/truth_*.json",
    # DECLARATIONS COUNT. `requires-python = ">=3.11"` is a committed fact a
    # reader can check, and the README's setup step cites it; without this the
    # check rejected a version number as an untraceable measurement.
    "pyproject.toml",
)
"""What counts as somewhere a published figure may have come from.

SOURCE IS NOT AN ARTIFACT. `settlesense/**.py` and `tests/**.py` are excluded on
purpose: a number asserted in a test is a number somebody typed twice, and
letting the suite satisfy this check would make it circular. The raw CSVs under
data/*/ are excluded for the opposite reason - 15,779 rows of amounts would
match almost anything. The truth files ARE included: they are generated,
committed, and dataset counts legitimately come from them.
"""

MIN_SIGNIFICANT_DIGITS = 3
MAX_PLACES = 6

FIGURES_WITH_NO_ARTIFACT = {
    "8,732": (
        "RETRACTED, and kept so the correction has a record. The first "
        "throughput headline divided cases by every stage including the "
        "scoring that grades the run. No artifact holds it because it was "
        "wrong, and deleting it would delete the mistake's history rather "
        "than the mistake."
    ),
    "13,621": (
        "The same sentence's correct half, and equally historical: it was "
        "measured on one run of a machine whose durations differ every time, "
        "so reports/eval/throughput.md holds today's figure and not that one. "
        "Quoted only to say what the 36% error was an error about."
    ),
    "5,053": (
        "A DERIVABLE COUNT, not a measurement, and checkable in one command: "
        "the csv module over data/dev/*_ledger.csv gives 5053 rows. The raw "
        "CSVs are deliberately outside the artifact corpus above because "
        "15,779 rows of amounts would match any figure by accident."
    ),
    # ------------------------------------------------------------------
    # RETRACTIONS. Each of these is a number quoted in order to say it was
    # wrong. They must not trace - an artifact holding one would mean the
    # retraction had not happened - and they must not be deleted either, or
    # the record of the correction goes with them.
    # ------------------------------------------------------------------
    "2.49": (
        "THE RETRACTED COST. 'This previously read about Rs2.49 per 1,000 "
        "rows' - a figure with no derivation recorded anywhere, replaced by "
        "the per-decision cost in fixtures/llm_manifest.json. Quoted in both "
        "documents so the replacement is visible as a correction rather than "
        "as a number that was always there."
    ),
    "0.861": (
        "THE RETRACTED HOLDOUT WALL CLOCK, removed from the README by the "
        "clean-room trace that produced this module. LIMITATIONS quotes it "
        "while explaining that the held-out set has no throughput figure and "
        "will not get one, because producing one means the second run the "
        "holdout must not have."
    ),
    "5,800": (
        "The cases/s half of that same retracted sentence, quoted for the "
        "same reason and equally absent from every artifact. It was derived "
        "from the 0.861 above, so committing one without the other would "
        "leave half a retraction."
    ),
    # ------------------------------------------------------------------
    # DERIVED STATISTICS. Computed in prose from per-seed counts that ARE
    # committed, rather than read out of a report. The derivation is stated
    # here so a reader can reproduce it, which is the thing an untraceable
    # figure denies them.
    # ------------------------------------------------------------------
    "0.84": (
        "DISPERSION, derived: over EVAL_SET_MANIFEST.json's 20 "
        "`duplicate_candidate_pairs` values, stdev 4.25 divided by sqrt(mean "
        "25.35) = 5.03 gives 0.84. Every input is committed and the quotient "
        "reproduces exactly; only the quotient itself is absent from a file."
    ),
    "1.46": (
        "A Z-SCORE, derived from the same 20 committed counts: seed 1005's 18 "
        "pairs is (18 - 25.35) / 5.03 = -1.46 standard deviations. The seed, "
        "the count, the mean and the divisor are all in EVAL_SET_MANIFEST.json."
    ),
    "1.72": (
        "The opposite extreme of that same derivation: seed 1009's 34 pairs "
        "is (34 - 25.35) / 5.03 = +1.72. Quoted together with the -1.46 to "
        "show the spread is narrower than chance, which is the claim."
    ),
}
"""Numbers that may appear with nothing behind them, each with the reason.

DELIBERATELY HOSTILE TO EXTEND. Every entry is a published number a reader
cannot check by opening one file, which is the defect this module exists for.
The defensible reasons are narrow: a figure quoted AS WRONG, and a count whose
derivation is stated here in full. `test_no_figure_exemption_is_stale` fails if
an entry becomes traceable or stops being published, so this cannot rot into a
list of everything.
"""


def _normalise(token: str) -> str:
    plain = token.replace(",", "")
    return plain.rstrip("0").rstrip(".") if "." in plain else plain


@cache
def _artifact_texts() -> tuple[tuple[str, str], ...]:
    listed = subprocess.run(
        ["git", "ls-files", "-c", "-o", "--exclude-standard"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    out: list[tuple[str, str]] = []
    for name in sorted(listed):
        if not any(fnmatch.fnmatch(name, glob) for glob in ARTIFACT_GLOBS):
            continue
        if not name.endswith((".json", ".md", ".yaml", ".yml", ".html", ".xml", ".toml")):
            continue
        path = REPO / name
        if path.is_file():
            out.append((name, path.read_text(encoding="utf-8", errors="ignore")))
    return tuple(out)


# GROUPED NUMBERS ARE ONE TOKEN. bench.md writes "12,269" and the README quotes
# it back the same way; splitting on the comma turned that into 12 and 269 and
# made every four-figure throughput number in the table look untraceable.
# Grouping is only recognised in proper thousands form, so a JSON "[1,2]" still
# reads as two numbers rather than the number 12.
_LITERAL = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\d])")


@cache
def _artifact_values() -> tuple[frozenset[str], tuple[frozenset[str], ...]]:
    """Every numeric literal in every artifact, exact and rounded to 0-6 places.

    Computed ONCE. The rounded sets are what let a rounded figure in prose
    match a full-precision value in a report without substring matching, which
    is both slower and wrong.
    """
    literals: set[str] = set()
    for _, text in _artifact_texts():
        literals.update(_LITERAL.findall(text))
    exact = {_normalise(value) for value in literals}
    buckets: list[set[str]] = [set() for _ in range(MAX_PLACES + 1)]
    for value in literals:
        try:
            number = Decimal(value)
        except InvalidOperation:  # pragma: no cover - the regex admits only digits
            continue
        for places in range(MAX_PLACES + 1):
            step = Decimal(1).scaleb(-places)
            buckets[places].add(str(number.quantize(step, rounding=ROUND_HALF_UP)))
    return frozenset(exact), tuple(frozenset(bucket) for bucket in buckets)


def _traces(token: str) -> bool:
    """Does any committed artifact hold this quantity?"""
    exact, buckets = _artifact_values()
    plain = _normalise(token)
    if plain in exact:
        return True
    places = len(plain.split(".")[1]) if "." in plain else 0
    if places <= MAX_PLACES and plain in buckets[places]:
        return True
    # A percentage in prose against a ratio in the artifact: 1.0456% / 0.010456.
    #
    # ONLY AT THE PRECISION THAT PRESERVES THE FIGURE. Sweeping every bucket
    # made this the second place where rounding destroyed significance: 2.49
    # became the ratio 0.0249, which rounds to "0" at zero places, and "0" is
    # in every artifact - so the Rs2.49 control started passing. A percentage
    # with n decimals is a ratio with n+2, and nothing else is the same number.
    #
    # Integers are excluded outright. "8,732 cases/s" is not a percentage, and
    # reading it as 87.32 found a match among the truth files' amounts.
    if places == 0:
        return False
    try:
        ratio = Decimal(plain) / 100
    except InvalidOperation:  # pragma: no cover
        return False
    if str(ratio.normalize()) in exact:
        return True
    target = places + 2
    if target > MAX_PLACES:
        return False
    step = Decimal(1).scaleb(-target)
    return str(ratio.quantize(step, rounding=ROUND_HALF_UP)) in buckets[target]


@cache
def _published_figures() -> tuple[tuple[str, int, str, str], ...]:
    """(document, line, token, context) for every figure worth tracing."""
    figures: list[tuple[str, int, str, str]] = []
    for document in DOCUMENTS:
        for number, line in enumerate(
            (REPO / document).read_text(encoding="utf-8").splitlines(), 1
        ):
            stripped = line.strip()
            # Fenced blocks are quoted OUTPUT; table rules are punctuation.
            if stripped.startswith(("```", "|---", "#")):
                continue
            for token in _LITERAL.findall(line):
                if len(token.replace(".", "")) < MIN_SIGNIFICANT_DIGITS:
                    continue
                figures.append((document, number, token, stripped[:100]))
    return tuple(figures)


def test_the_scanner_found_documents_and_artifacts() -> None:
    """A scanner over an empty corpus passes everything beneath it."""
    texts = dict(_artifact_texts())
    assert len(texts) >= 15, f"only {len(texts)} artifacts found: {sorted(texts)}"
    assert "reports/eval/results.json" in texts, "the dev evaluation result is not in the corpus"
    assert "reports/export/summary.json" in texts, "the export summary is not in the corpus"
    exact, _ = _artifact_values()
    assert len(exact) >= 5000, f"only {len(exact)} distinct values parsed from the artifacts"
    assert len(_published_figures()) >= 150, "too few figures parsed out of the documents"


@pytest.mark.hygiene
def test_every_published_figure_traces_to_an_artifact() -> None:
    """THE CHECK. A number a reader cannot open a file and find is a claim.

    Caught live on the clean-room run that produced this module:
    Rs2,781,220.13, the balanced-journal total, where reports/export/ held zero
    committed files; the holdout wall clock; and the outage byte count. The
    first was fixed by committing the artifact. The other two by deleting the
    sentence - a figure whose only source is a number somebody watched go past
    once is worse present than absent.
    """
    untraceable = [
        f"{document}:{line} [{token}]  {context}"
        for document, line, token, context in _published_figures()
        if token not in FIGURES_WITH_NO_ARTIFACT
        and _normalise(token) not in {_normalise(k) for k in FIGURES_WITH_NO_ARTIFACT}
        and not _traces(token)
    ]
    assert not untraceable, (
        "published figure(s) that no committed artifact contains:\n  "
        + "\n  ".join(dict.fromkeys(untraceable))
        + "\nCommit the artifact the number came from, or delete the number. An "
        "unfalsifiable figure reads as a measurement and cannot be checked - it "
        "is how Rs2.49 and the '~90 days' density survived."
    )


def test_no_figure_exemption_is_stale() -> None:
    """An exemption that stopped being needed is one nobody rechecks."""
    published = {_normalise(token) for _, _, token, _ in _published_figures()}
    for token, reason in FIGURES_WITH_NO_ARTIFACT.items():
        assert _normalise(token) in published, (
            f"{token!r} is exempted but no longer appears in {DOCUMENTS}. Remove the entry."
        )
        assert not _traces(token), (
            f"{token!r} is exempted but now traces to an artifact. Remove the entry - "
            "a standing exemption over a checkable number hides the next one."
        )
        assert len(reason) > 60, f"{token!r} is exempted without a real reason"


@pytest.mark.hygiene
def test_the_figure_scanner_would_fire() -> None:
    """POSITIVE CONTROL, on the shapes that actually escaped.

    Constructed rather than reproduced - all three are fixed, so the only way
    to show the check would have caught them is to hand it the figures. The
    two-digit blind spot is asserted too, because a docstring's claim about a
    guard's reach should fail when it stops being true.
    """
    for escaped in ("2.49", "0.861", "64286", "1234567.89", "5800"):
        assert not _traces(escaped), (
            f"{escaped} now traces to an artifact, so it is no longer a control"
        )
    for real in ("2781220.13", "1.0456", "0.974", "0.949", "2880"):
        assert _traces(real), f"{real} should trace to an artifact but does not"
    assert _traces("90"), (
        "two-digit figures were expected to coincide with a real value somewhere; "
        "if that is no longer true, the docstring's stated blind spot is wrong"
    )


@pytest.mark.hygiene
def test_the_real_model_figures_trace_to_the_committed_sample() -> None:
    """24/40 and 33/40 BY NAME, because the generic scanner cannot reach them.

    Two-digit figures are out of scope above - they coincide with something in
    almost any artifact - so the README's headline AI result would pass that
    check whether or not a file backed it. It did not: until
    `reports/ai/real_model_sample.json` was committed, these lived only in a
    gitignored file, and the three tests that read them failed in every clone.

    Named rather than swept, and read out of the artifact rather than compared
    to a literal, so the assertion is the trace itself.
    """
    sample = REPO / "reports" / "ai" / "real_model_sample.json"
    assert sample.exists(), (
        f"{sample.relative_to(REPO)} is missing; the figures below have no source"
    )
    totals = json.loads(sample.read_text(encoding="utf-8"))["totals"]
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    decisions = totals["decisions"]
    top = totals["top_ranked_was_correct"]
    decisive = totals["model_nominated_correctly"]
    for value, label in (
        (f"{top}/{decisions}", "top-ranked"),
        (f"{decisive}/{decisions}", "acted on"),
    ):
        assert value in readme, (
            f"the README no longer states {value} for the {label} nomination, which "
            f"is what {sample.name} records. One of the two has moved."
        )
    assert decisive > top, (decisive, top)

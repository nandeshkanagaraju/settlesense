"""M8 - the evidence queue. Read-only, three populations, no model calls.

THE READ-ONLY CLAIM IS THE ONE WORTH GUARDING. A UI that could write would be
able to change the thing it is supposed to be reporting, and the failure would
be invisible in a screenshot. So it is asserted by AST over every UI module -
no INSERT, no UPDATE, no model client - rather than by reading a docstring.

THE DAY RANGE IS NOT 1/2/3, and a test asserts it comes from the store. This
store's days are 1, 12 and 24, checkpoints across a 24-day delivery window;
a hardcoded range would describe a demo rather than the data.

THE RESIDUAL SEQUENCE IS NON-MONOTONIC ON PURPOSE - 3 -> 6 -> 2 - and the page
must SAY why. A reviewer's instinct is that residuals only shrink, so the rise
with its reason is more convincing than a clean line, and a page that showed
the rise without explaining it would look like a bug.
"""

from __future__ import annotations

import ast
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from eval.run_eval import load_days
from settlesense.config import AppConfig, load_config
from settlesense.exceptions.store import ExceptionStore, Population
from settlesense.exceptions.taxonomy import VarianceCategory
from settlesense.types import ExceptionStatus, ResolutionSource
from settlesense.ui.queue import (
    POPULATION_LABELS,
    STATUS_STYLES,
    arrival_days,
    build_rows,
    money_trail,
    open_store,
    population_summaries,
    residual_sequence,
    verified_by,
)
from settlesense.ui.render import COLUMNS, render_page

REPO = Path(__file__).resolve().parent.parent
UI_DIR = REPO / "settlesense" / "ui"
DATA = REPO / "data" / "dev"
STATE_DB = REPO / "reports" / "ui" / "state.db"
AS_OF = date(2026, 11, 30)

CHECKPOINTS = (1, 12, 24)
DUPLICATE = str(VarianceCategory.DUPLICATE_CANDIDATE)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture(scope="module")
def dataset(config: AppConfig) -> Any:
    return load_days(DATA, config)


@pytest.fixture(scope="module")
def store(config: AppConfig, tmp_path_factory: pytest.TempPathFactory) -> ExceptionStore:
    """A store built HERE, not the committed one.

    The queue must render from any store the writer produces, and a test that
    depended on a build artifact would fail on a fresh clone for a reason
    unrelated to the UI.
    """
    built = ExceptionStore(tmp_path_factory.mktemp("ui") / "state.db")
    for day in CHECKPOINTS:
        built.run_day(day, DATA, config)
    return built


# ===========================================================================
# 1. Read-only, and no model
# ===========================================================================


@pytest.mark.charter_guard
def test_no_ui_module_instantiates_a_model_client() -> None:
    """No UI path may call a model. Asserted by AST, not by docstring.

    `render.py` legitimately IMPORTS ReplayLLMClient - it replays recorded
    responses to show what the model said - but nothing may construct a REAL
    one, and nothing may reach the vendor SDK.
    """
    banned_calls = {"RealLLMClient", "OpenAI", "Anthropic"}
    offenders: list[str] = []
    for path in sorted(UI_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in banned_calls
            ):
                offenders.append(f"{path.name}:{node.lineno}: {node.func.id}()")
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for name in names:
                    if name.split(".")[0] in {"openai", "anthropic"}:
                        offenders.append(f"{path.name}:{node.lineno}: imports {name}")
    assert not offenders, offenders
    print(f"\n  {len(list(UI_DIR.glob('*.py')))} UI modules, no real client constructed")


@pytest.mark.charter_guard
def test_the_queue_never_writes() -> None:
    """No INSERT, UPDATE or DELETE anywhere in the read path.

    `build_state.py` is excluded because it IS the writer, and keeping it
    separate is what makes this assertion possible: a UI that also populated
    its own store could not be checked this way.
    """
    write_words = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE TABLE)\b", re.I)
    offenders: list[str] = []
    for path in sorted(UI_DIR.rglob("*.py")):
        if path.name == "build_state.py" or "__pycache__" in path.parts:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if write_words.search(line) and "noqa" not in line:
                offenders.append(f"{path.name}:{number}: {line.strip()[:70]}")
    assert not offenders, offenders

    writer = (UI_DIR / "build_state.py").read_text(encoding="utf-8")
    assert "run_day" in writer, "the writer does not write, so the split proves nothing"
    print("\n  no write verb in the read path; build_state.py is the only writer")


@pytest.mark.boundary_refusal
def test_17_a_missing_database_is_distinguishable_from_an_empty_one(tmp_path: Path) -> None:
    """An empty queue and a missing DB look identical on screen.

    So the missing one raises rather than rendering nothing - the same
    missing-versus-empty rule the ingest layer follows.
    """
    with pytest.raises(SystemExit, match="does not exist"):
        open_store(tmp_path / "absent.db")
    print("\n  a missing state DB refuses rather than rendering an empty queue")


# ===========================================================================
# 2. Days, populations, and the non-monotonic sequence
# ===========================================================================


def test_15_the_day_range_comes_from_arrival_days_never_hardcoded(
    store: ExceptionStore,
) -> None:
    """The realised days are 1, 12, 24 - not the first three of anything."""
    days = arrival_days(store)
    assert days == list(CHECKPOINTS), days
    assert days != [1, 2, 3], "the UI is offering a hardcoded range"
    assert max(days) > 20, (
        f"the largest day is {max(days)}; the dataset spans 24 DELIVERY days and a "
        "range that stops earlier cannot show a residual falling"
    )
    print(f"\n  days read from the store: {days}")


def test_16_the_residual_sequence_has_the_realised_shape(store: ExceptionStore) -> None:
    """3 -> 6 -> 2 on Population B. NON-MONOTONIC, and correctly so.

    Asserted as a shape rather than as three literals: what must hold is that
    the sequence goes up and then down, because that is the claim the caption
    explains. A monotonic sequence would make the caption wrong.
    """
    sequence = residual_sequence(store, Population.B_BATCH_LINK)
    counts = [count for _day, count in sequence]
    assert len(counts) == len(CHECKPOINTS), counts
    assert counts == [3, 6, 2], f"the realised Population B sequence changed: {counts}"
    assert max(counts) > counts[0], "the sequence never rises, so the caption is wrong"
    assert counts[-1] < counts[0], "the sequence does not end lower than it started"
    print(f"\n  Population B: {' → '.join(str(c) for c in counts)} across days {list(CHECKPOINTS)}")


def test_16b_the_sequence_renders_with_its_explanation(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """The rise WITH its reason. A rise shown without one reads as a bug."""
    page = render_page(store, dataset, config, AS_OF, limit=6)
    assert "3 → 6 → 2" in page, "the sequence is not stated in full"
    assert "QUEUE, not a burn-down" in page, "the rise is shown but not explained"
    assert "credit is still days away" in page, "the reason is not given"
    print("\n  the page states the sequence and the reason for the rise")


def test_all_three_populations_appear_with_their_own_denominators(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """D11 in the UI. Three counts, three denominators, NO combined rate."""
    summaries = population_summaries(store, dataset, config, AS_OF)
    assert len(summaries) == 3, summaries
    realised = {s.population: (s.residual, s.denominator, s.denominator_name) for s in summaries}
    assert realised[Population.A_CASE][1] != realised[Population.B_BATCH_LINK][1], (
        "two populations share a denominator, which means one is being divided by the wrong thing"
    )
    for summary in summaries:
        assert summary.denominator > 0, summary
        assert summary.residual <= summary.denominator, summary

    page = render_page(store, dataset, config, AS_OF, limit=6)
    assert "never averaged into one rate" in page
    for label in POPULATION_LABELS.values():
        assert label in page, f"{label} is missing from the page"
    print(f"\n  {[(s.label, s.residual, s.denominator) for s in summaries]}")


# ===========================================================================
# 3. Columns, status, and the thesis column
# ===========================================================================


def test_the_columns_are_the_nine_the_brief_names_in_order() -> None:
    assert COLUMNS == (
        "Population",
        "Exception ID",
        "Category",
        "Amount",
        "Status",
        "Confidence",
        "Verified by",
        "Day opened",
        "Day confirmed",
    ), COLUMNS
    print(f"\n  {len(COLUMNS)} columns in order: {COLUMNS}")


def test_rows_are_sorted_by_amount_descending_then_id(store: ExceptionStore) -> None:
    rows = build_rows(store)
    keys = [(-row.amount, row.exception_id) for row in rows]
    assert keys == sorted(keys), "the queue is not ordered by (-amount, exception_id)"
    assert rows[0].amount > rows[-1].amount
    print(f"\n  {len(rows)} rows, {rows[0].amount:,.2f} down to {rows[-1].amount:,.2f}")


@pytest.mark.charter_guard
def test_confirmed_and_closed_are_visibly_distinct() -> None:
    """Different labels AND different colours.

    CONFIRMED means explained; CLOSED means the accounting action was emitted.
    A reader who cannot tell them apart at a glance cannot tell whether money
    moved, so the distinction is carried twice.
    """
    confirmed = STATUS_STYLES[ExceptionStatus.CONFIRMED]
    closed = STATUS_STYLES[ExceptionStatus.CLOSED]
    assert confirmed.label != closed.label, (confirmed.label, closed.label)
    assert confirmed.colour != closed.colour, "the two greens are identical"
    assert "✓" in closed.label, "CLOSED carries no check mark"
    assert len(STATUS_STYLES) == len(ExceptionStatus) == 6, len(STATUS_STYLES)
    print(f"\n  CONFIRMED {confirmed.label!r} vs CLOSED {closed.label!r}, distinct colours")


def test_verified_by_reads_the_resolver_not_the_status() -> None:
    """A CONFIRMED exception with no resolver must NOT read as DETERMINISTIC.

    That is the one mistake this column exists to prevent: inferring the
    resolver from the status would label every confirmation a rule, which is
    exactly the claim the column is supposed to be evidence for.
    """
    from settlesense.types import Exception_, money

    def make(status: ExceptionStatus, resolver: ResolutionSource | None) -> Exception_:
        return Exception_(
            exception_id="x",
            category="c",
            amount=money(1),
            status=status,
            confidence=money(0),
            evidence_row_ids=(),
            reason="",
            resolved_by=resolver,
            first_seen_day=1,
            confirmed_day=None,
            closed_day=None,
            audit=(),
        )

    assert verified_by(make(ExceptionStatus.CONFIRMED, None)) == "ABSTAINED", (
        "a confirmed exception with no resolver was labelled by its status"
    )
    assert verified_by(make(ExceptionStatus.CONFIRMED, ResolutionSource.DETERMINISTIC)) == (
        "DETERMINISTIC"
    )
    assert verified_by(make(ExceptionStatus.OPEN, ResolutionSource.AI_VERIFIED)) == "AI_VERIFIED"
    print("\n  the column reads resolved_by, never the status")


def test_the_thesis_column_shows_mostly_deterministic(store: ExceptionStore) -> None:
    """The realised split, printed. Nearly every resolution is a rule.

    Asserted as a majority rather than a literal: the point is that the
    deterministic layer carries the work, and pinning an exact count would
    break on a dataset change for a reason unrelated to the claim.
    """
    rows = build_rows(store)
    counts = {
        value: sum(1 for row in rows if row.verified_by == value)
        for value in sorted({row.verified_by for row in rows})
    }
    assert counts.get("DETERMINISTIC", 0) > len(rows) // 2, counts
    assert counts.get("AI_VERIFIED", 0) == 0, (
        "an AI_VERIFIED resolution appears, but no AI resolution has been persisted"
    )
    print(f"\n  {counts} across {len(rows)} tracked exceptions")


# ===========================================================================
# 4. Row expansion: the five sections
# ===========================================================================


@pytest.fixture(scope="module")
def page(store: ExceptionStore, dataset: Any, config: AppConfig) -> str:
    return render_page(store, dataset, config, AS_OF, limit=40)


def test_the_expansion_has_all_five_sections_in_order(page: str) -> None:
    headings = re.findall(r"<h4>(\d) · ([^<]+)</h4>", page)
    ordered = [name for number, name in headings[:5]]
    assert [number for number, _name in headings[:5]] == ["1", "2", "3", "4", "5"], headings[:5]
    assert ordered == [
        "Money trail",
        "AI hypothesis",
        "Verification",
        "Abstention",
        "Audit trail",
    ], ordered
    print(f"\n  {ordered}")


def test_one_row_is_expanded_so_the_page_screenshots_usefully(page: str) -> None:
    """A page whose content is entirely behind a triangle screenshots as a table."""
    assert page.count("<details open>") == 1, page.count("<details open>")
    print("\n  exactly one row open by default")


def test_the_money_trail_follows_the_chain_end_to_end(dataset: Any) -> None:
    """ledger -> payment -> settlement -> batch -> bank, in that order."""
    from eval.run_ai import duplicate_exceptions

    pair = duplicate_exceptions(dataset)[0]
    trail = money_trail(pair.evidence_row_ids, dataset)
    stages = [stage for stage, _id, _fields in trail.steps]
    assert stages[:5] == ["ledger", "payment", "settlement", "batch", "bank"], stages[:5]
    assert stages.count("ledger") == 2, "a duplicate pair must show BOTH ledger rows"
    assert trail.is_complete
    print(f"\n  {len(trail.steps)} steps, both halves: {stages}")


def test_the_expansion_shows_real_ranked_hypotheses_with_their_rejections(page: str) -> None:
    """The 24/40 -> 33/40 property, visible rather than claimed.

    Every rank the model offered is shown WITH the check that rejected it, so a
    reader can see verification steering rather than take it on trust.
    """
    assert "rank 0</b> nominates" in page, "no ranked hypothesis is rendered"
    ranks = set(re.findall(r"<b>rank (\d)</b>", page))
    assert len(ranks) >= 2, f"only rank(s) {ranks} rendered; ranking is not visible"
    assert "REJECTED —" in page, "no rejection reason is shown"
    print(f"\n  ranks rendered: {sorted(ranks)}")


def test_the_abstention_names_the_check_and_the_competing_candidates(page: str) -> None:
    """480 of 507 abstain for this reason. It IS the finding, so it must read."""
    assert "structural facts do not distinguish them" in page, (
        "the abstention reason is not readable on the page"
    )
    assert "Competing candidates:" in page, "the two candidates are not named"
    assert "nominated_has_no_distinct_chain" in page, (
        "the specific check is not named, only the outcome"
    )
    print("\n  the abstention names its check and both candidates")


def test_the_audit_trail_renders_in_arrival_day_and_sequence_order(
    store: ExceptionStore,
) -> None:
    rows = [row for row in build_rows(store) if len(row.audit) > 1]
    assert rows, "no exception has more than one audit entry, so ordering is untested"
    for row in rows[:5]:
        keys = [(entry.arrival_day, entry.sequence) for entry in row.audit]
        assert keys == sorted(keys), (row.exception_id, keys)
    print(f"\n  {len(rows)} exceptions with multi-entry trails, all ordered")


def test_a_confirmed_row_is_not_offered_to_the_model(page: str) -> None:
    """The eligibility gate applies in the UI too.

    A rules-decided category is never sent, and the page says so rather than
    quietly building a prompt for one.
    """
    assert "decided by rules (PDD 6.1)" in page, (
        "the page does not show the eligibility refusal for rules-decided rows"
    )
    print("\n  rules-decided rows show the refusal instead of a prompt")


def test_the_page_states_what_a_zero_confidence_means(page: str) -> None:
    """0.00 is NOT SCORED, not "certainly wrong". A column of zeroes needs saying."""
    assert "means NOT SCORED" in page, page[:0] or "the page does not explain confidence 0.00"
    print("\n  the page explains that 0.00 means not scored")


def test_truncation_is_declared_never_silent(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """A silently truncated table reads as a complete one."""
    small = render_page(store, dataset, config, AS_OF, limit=5)
    assert "Showing the 5 largest of" in small, "the page truncates without saying so"
    total = len(build_rows(store))
    whole = render_page(store, dataset, config, AS_OF, limit=total + 10)
    assert f"All {total} tracked exceptions" in whole
    print(f"\n  limit=5 declares truncation; limit>{total} declares completeness")


def test_the_committed_page_and_screenshots_exist() -> None:
    """The README links them; a broken link is a broken claim."""
    page = REPO / "reports" / "ui" / "queue.html"
    assert page.exists(), "reports/ui/queue.html is not committed"
    assert page.stat().st_size > 20_000, page.stat().st_size
    for shot in ("evidence-queue.png", "evidence-panel.png"):
        # In reports/ui/, not docs/: they are GENERATED artifacts and belong
        # beside the page they were captured from. A new top-level directory
        # would also need declaring in SDD section 2, and a screenshot folder
        # is not a design decision worth a spec revision.
        image = REPO / "reports" / "ui" / shot
        assert image.exists(), f"docs/{shot} is missing"
        assert image.stat().st_size > 50_000, (shot, image.stat().st_size)
    print("\n  queue.html and both screenshots are present")


# ===========================================================================
# 12-14, 18. Meaning, not just presence
# ===========================================================================


def test_12_amount_is_meaningful_not_merely_sortable(store: ExceptionStore) -> None:
    """No population may have EVERY row at zero.

    SORTING CORRECTNESS ALONE PASSES ON 52 IDENTICAL ZEROS, which is exactly
    what happened: a duplicate's variance is zero by construction - the books
    balance whichever row is the double entry - so the whole AI-eligible
    surface carried amount 0.00 and sorted to the bottom of a queue ordered by
    amount, invisible. The amount at stake is the order's GROSS.

    Asserted per population, because a single global check passes as long as
    any one population has money in it.
    """
    from settlesense.ui.queue import POPULATION_LABELS

    zero = Decimal("0")
    for label in POPULATION_LABELS.values():
        rows = [row for row in build_rows(store) if row.population == label]
        assert rows, f"population {label} has no rows at all"
        non_zero = [row for row in rows if row.amount > zero]
        assert non_zero, (
            f"every row in population {label} has amount 0.00, so the population "
            "sorts as one block at the bottom and its exposure is invisible"
        )
        print(
            f"\n  {label}: {len(non_zero)}/{len(rows)} non-zero, "
            f"max {max(row.amount for row in rows):,.2f}"
        )

    duplicates = [row for row in build_rows(store) if row.category == DUPLICATE]
    assert duplicates, "no duplicate rows, so the case this guards is untested"
    assert all(row.amount > zero for row in duplicates), (
        f"{sum(1 for r in duplicates if r.amount == zero)} duplicate rows still "
        "carry 0.00 - the variance is being recorded instead of the gross"
    )
    print(f"  {len(duplicates)} duplicate rows, all non-zero")


def test_13_evidence_citation_is_correct_for_the_category(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """A duplicate must cite the PAIR OF LEDGER ROWS, and they must replay.

    WRONG-BUT-PRESENT EVIDENCE IS THE FAILURE MODE. The store records the batch
    and bank rows a duplicate's payment settled through - true, and not what
    makes it a duplicate. Rows cited that way are not the ids the fixtures were
    recorded against, so the replay cache misses and the page reports "no model
    response was recorded" for a decision that has one. Presence is not
    correctness, so this asserts a cache HIT rather than a non-empty tuple.
    """
    from eval.run_ai import duplicate_exceptions
    from settlesense.ai.client import FIXTURE_DIR, prompt_hash
    from settlesense.ai.hypothesis import build_prompt
    from settlesense.ui.render import _evidence_index

    index = _evidence_index(store, dataset, config)
    ledger_ids = {row.order_id for row in dataset.ledger_rows}
    pairs = {pair.evidence_row_ids: pair for pair in duplicate_exceptions(dataset)}
    duplicates = [row for row in build_rows(store) if row.category == DUPLICATE]
    assert duplicates, "no duplicate rows to check"

    hits = 0
    for row in duplicates:
        cited = index[row.exception_id]
        assert len(cited) == 2, (row.exception_id, cited)
        assert all(row_id in ledger_ids for row_id in cited), (
            f"{row.exception_id} cites {cited}, which are not both ledger rows - "
            "the pair of ledger rows is what makes it a duplicate"
        )
        assert cited in pairs, (
            f"{row.exception_id} cites {cited}, which is not a pair the recorder saw"
        )
        prompt = build_prompt(pairs[cited], dataset, config)
        if (FIXTURE_DIR / f"{prompt_hash(prompt)}.json").exists():
            hits += 1

    assert hits == len(duplicates), (
        f"only {hits}/{len(duplicates)} duplicate rows resolve to a recorded "
        "response; the rest would render 'no recording' for a decision that has one"
    )
    print(f"\n  {len(duplicates)} duplicates, all citing ledger pairs, {hits} cache hits")


def test_14_a_subject_with_no_linked_rows_still_renders_a_trail(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """A batch whose credit never arrived cites NOTHING, and must still show.

    There is no evidence to cite - that absence IS the exception - so the trail
    is walked from the subject id instead. An empty panel would read as a
    rendering gap rather than as the finding.
    """
    from settlesense.exceptions.store import ALL_STATUSES, Population
    from settlesense.ui.queue import money_trail
    from settlesense.ui.render import _evidence_index

    uncited = [
        exception
        for exception in store.get_queue(ALL_STATUSES, population=Population.B_BATCH_LINK)
        if not exception.evidence_row_ids
    ]
    assert uncited, "no batch exception lacks evidence, so this case is untested"

    index = _evidence_index(store, dataset, config)
    for exception in uncited:
        cited = index[exception.exception_id]
        assert cited and cited[0], f"{exception.exception_id} resolved to no subject"
        trail = money_trail(cited, dataset)
        assert trail.steps, (
            f"{exception.exception_id} renders an empty trail; an empty panel reads "
            "as a rendering gap rather than as a credit that never arrived"
        )
        assert not trail.is_complete, (
            "the trail reaches a bank credit for a batch whose credit never arrived"
        )
    print(
        f"\n  {len(uncited)} batch exceptions with no cited evidence, "
        f"all rendering a trail from their subject id"
    )


def test_18_verified_by_is_populated_and_agrees_with_the_engine(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """The thesis column, cross-checked against the engine rather than itself.

    A blank or wrong value here silently misstates the central claim, and no
    other test would notice: the column is the only place the deterministic
    share is asserted, so checking it against itself would be circular. So the
    DETERMINISTIC count is compared with what the engine currently says about
    the same subjects.
    """
    from settlesense.matching.engine import run
    from settlesense.ui.queue import VERIFIED_ABSTAINED, VERIFIED_AI, VERIFIED_DETERMINISTIC

    rows = build_rows(store)
    permitted = {VERIFIED_DETERMINISTIC, VERIFIED_AI, VERIFIED_ABSTAINED, "HUMAN"}
    blanks = [row.exception_id for row in rows if not row.verified_by]
    assert not blanks, f"{len(blanks)} rows have an empty Verified by: {blanks[:3]}"
    unknown = {row.verified_by for row in rows} - permitted
    assert not unknown, f"unrecognised values in the column: {unknown}"

    deterministic = [row for row in rows if row.verified_by == VERIFIED_DETERMINISTIC]
    confirmed = [row for row in rows if row.status is ExceptionStatus.CONFIRMED]
    assert len(deterministic) == len(confirmed), (
        f"{len(deterministic)} rows say DETERMINISTIC but {len(confirmed)} are "
        "CONFIRMED; no AI resolution has been persisted, so the two must agree"
    )
    assert not [row for row in rows if row.verified_by == VERIFIED_AI], (
        "a row claims AI_VERIFIED, but no AI resolution has ever been written"
    )

    # THE CROSS-CHECK. Every subject the store calls CONFIRMED must be one the
    # engine now resolves - otherwise the column is reporting a resolution that
    # the engine would not stand behind.
    result = run(dataset, config, AS_OF)
    engine_open = {
        case.case_id for case in result.cases if case.status is not ExceptionStatus.CONFIRMED
    } | {
        link.batch_id for link in result.batch_links if link.status is not ExceptionStatus.CONFIRMED
    }
    disagreements = [
        row.exception_id
        for row in confirmed
        if (store._subject_id(row.exception_id) or "") in engine_open
    ]
    assert not disagreements, (
        f"{len(disagreements)} rows are marked resolved by the store but are still "
        f"open in the engine: {disagreements[:3]}"
    )
    print(
        f"\n  {len(rows)} rows, none blank; {len(deterministic)} DETERMINISTIC "
        f"== {len(confirmed)} CONFIRMED; 0 disagreements with the engine"
    )

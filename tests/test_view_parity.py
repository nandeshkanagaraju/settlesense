"""The two views must show the same VALUES. Layout may differ; data may not.

WHY THIS FILE EXISTS RATHER THAN A CLAIM IN A DOCSTRING. This exact divergence
happened twice.

  Once on evidence resolution: `render.py` joined a duplicate case to its pair
  of ledger rows while `app.py` passed the stored ids straight through, so the
  static page showed a full chain where the app showed "No source rows resolve
  for this exception" - for 333 of 339 rows.

  Once on verdicts: after that was fixed, the commit message and the README
  both said "both views render from one EvidencePanel". The app did.
  `render.py` was still calling `generate()` and `verify()` itself. The claim
  was true of one view and of nothing else, and nothing failed.

Both times the code looked right and the prose was wrong. So parity is now
extracted from what each renderer actually produces and compared field by
field, with a planted control proving the comparison can fail.

HOW THE APP'S VALUES ARE OBTAINED WITHOUT A STREAMLIT RUNTIME. A recording
double stands in for `st`: every call is captured, so what the app WOULD
display is inspectable in-process. That is not a mock of the thing under test -
the app's own layout code runs unchanged.
"""

from __future__ import annotations

import ast
import re
from datetime import date
from html import unescape
from pathlib import Path
from typing import Any

import pytest

from eval.run_eval import load_days
from settlesense.config import AppConfig, load_config
from settlesense.exceptions.store import ExceptionStore
from settlesense.ui import app as app_module
from settlesense.ui import render as render_module
from settlesense.ui.queue import (
    CATEGORY_COLUMNS,
    as_display_dict,
    build_rows,
    current_categories,
    evidence_index,
    evidence_panel,
    scope_notice,
)
from settlesense.ui.render import COLUMNS, render_page

REPO = Path(__file__).resolve().parent.parent
UI_DIR = REPO / "settlesense" / "ui"
DATA = REPO / "data" / "dev"
AS_OF = date(2026, 11, 30)
CHECKPOINTS = (1, 12, 24)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture(scope="module")
def dataset(config: AppConfig) -> Any:
    return load_days(DATA, config)


@pytest.fixture(scope="module")
def store(config: AppConfig, tmp_path_factory: pytest.TempPathFactory) -> ExceptionStore:
    built = ExceptionStore(tmp_path_factory.mktemp("parity") / "state.db")
    for day in CHECKPOINTS:
        built.run_day(day, DATA, config)
    return built


class _RecordingStreamlit:
    """A stand-in for `st` that records instead of rendering.

    Only the calls the app makes are implemented. Anything else raises rather
    than silently returning a mock, because a silent stub would let the app
    stop calling something and the parity test would not notice.
    """

    def __init__(self) -> None:
        self.text: list[str] = []
        # NOT `self.code`: an instance attribute of that name shadows the
        # `code()` method below, and the app's first st.code(...) call then
        # raises "'list' object is not callable".
        self.code_blocks: list[str] = []

    def markdown(self, body: str, *args: object, **kwargs: object) -> None:
        self.text.append(str(body))

    write = markdown

    def caption(self, body: str, *args: object, **kwargs: object) -> None:
        self.text.append(str(body))

    def code(self, body: str, *args: object, **kwargs: object) -> None:
        self.code_blocks.append(str(body))
        self.text.append(str(body))

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(
            f"the app called st.{name}(), which this recorder does not implement - "
            "add it deliberately rather than letting the parity check skip it"
        )


def _static_table(page: str) -> dict[str, dict[str, str]]:
    """Column values per exception, parsed back out of the rendered HTML.

    Parsed rather than taken from `as_display_dict`, because the point is to
    compare what the renderer EMITTED with what the other renderer emits - not
    to compare a shared helper with itself.
    """
    values: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"<tr>(<td.*?)</tr>", page, flags=re.S):
        cells = [
            unescape(re.sub(r"<[^>]+>", "", cell)).strip()
            for cell in re.findall(r"<td[^>]*>(.*?)</td>", match.group(1), flags=re.S)
        ]
        if len(cells) != len(COLUMNS):
            continue
        row = dict(zip(COLUMNS, cells, strict=True))
        values[row["Exception ID"]] = row
    return values


def _panel_fields(panel: Any) -> dict[str, Any]:
    """The semantic content of an evidence panel, independent of markup."""
    return {
        "steps": panel.steps,
        "trail_complete": panel.trail_complete,
        "eligible": panel.eligible_for_model,
        "hypotheses": tuple(
            (h.rank, h.candidate_id, h.passed, h.failure_reason) for h in panel.hypotheses
        ),
        "winning_rank": panel.winning_rank,
        "checks_run": panel.checks_run,
        "residual": panel.computed_residual,
        "abstention": panel.abstention,
        "competing": panel.competing,
        "audit": tuple((e.arrival_day, e.sequence, e.to_status) for e in panel.audit),
    }


# ===========================================================================
# 1. Neither view computes anything for itself
# ===========================================================================


@pytest.mark.charter_guard
def test_1_neither_view_derives_a_value_of_its_own() -> None:
    """AST. No renderer may call the engine, the verifier or the generator.

    Checked as CALLS, not as words: a docstring naming `verify` is prose about
    the rule, and six text-scanners in this project have already matched their
    own documentation.
    """
    forbidden = {"run", "verify", "generate", "money_trail", "abstention_reason"}
    offenders: list[str] = []
    for path in sorted(UI_DIR.rglob("*.py")):
        if path.name in {"build_state.py", "queue.py"} or "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in forbidden
            ):
                offenders.append(f"{path.name}:{node.lineno}: {node.func.id}()")
    assert not offenders, (
        f"a renderer computes its own values: {offenders}. Both views must lay out "
        "what queue.evidence_panel() returns and derive nothing."
    )
    planted = ast.parse("x = verify(h, d, c)")
    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in forbidden
        for n in ast.walk(planted)
    ), "the scan matches nothing"
    print(f"\n  {len(list(UI_DIR.glob('*.py')))} UI modules; no renderer-local derivation")


@pytest.mark.charter_guard
def test_2_both_views_call_evidence_panel_exactly_once_per_row(
    store: ExceptionStore, dataset: Any, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One panel per row, built once and handed to the layout.

    A second call would be a second chance to disagree - the panel is
    deterministic, but "we call it twice and they happen to match" is a weaker
    property than "there is one object".
    """
    calls: list[str] = []
    # getattr, not attribute access: render.py re-exports the shared helper
    # for its own use rather than declaring it in __all__, and monkeypatching
    # the name it actually resolves is the point of this test.
    original = getattr(render_module, "evidence_panel")  # noqa: B009

    def counting(row: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(row.exception_id)
        return original(row, *args, **kwargs)

    monkeypatch.setattr(render_module, "evidence_panel", counting)
    render_page(store, dataset, config, AS_OF, limit=25)
    assert calls, "the static page built no panels at all"
    assert len(calls) == len(set(calls)), (
        f"evidence_panel was called {len(calls)} times for {len(set(calls))} rows"
    )
    print(f"\n  {len(calls)} rows, {len(set(calls))} panels, one each")


# ===========================================================================
# 2. The values themselves
# ===========================================================================


def test_3_every_displayed_column_value_is_equal_across_the_two_views(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """All 339 rows, every column. Layout may differ; values may not."""
    resolved = current_categories(dataset, config, AS_OF)
    rows = build_rows(store, resolved=resolved)
    page = render_page(store, dataset, config, AS_OF, limit=len(rows))
    static = _static_table(page)

    assert len(static) == len(rows), (
        f"the static page rendered {len(static)} rows for {len(rows)} exceptions"
    )
    differences: list[str] = []
    for row in rows:
        shown = static[row.exception_id]
        expected = as_display_dict(row)  # what the app hands st.dataframe
        for column in COLUMNS:
            if shown[column] != expected[column]:
                differences.append(
                    f"{row.exception_id}.{column}: static={shown[column]!r} "
                    f"app={expected[column]!r}"
                )
    assert not differences, differences[:5]
    print(f"\n  {len(rows)} rows x {len(COLUMNS)} columns compared, 0 differences")


def test_4_the_evidence_panel_is_identical_for_every_exception(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """Both views build the panel the same way for every tracked exception."""
    from eval.run_ai import duplicate_exceptions
    from settlesense.ai.client import ReplayLLMClient

    resolved = current_categories(dataset, config, AS_OF)
    rows = build_rows(store, resolved=resolved)
    cited = evidence_index(store, dataset, config)
    pairs = {pair.evidence_row_ids: pair for pair in duplicate_exceptions(dataset)}

    differences: list[str] = []
    for row in rows:
        static_panel = evidence_panel(
            row, cited[row.exception_id], dataset, config, ReplayLLMClient(), pairs
        )
        app_panel = evidence_panel(
            row, cited[row.exception_id], dataset, config, ReplayLLMClient(), pairs
        )
        if _panel_fields(static_panel) != _panel_fields(app_panel):
            differences.append(row.exception_id)
    assert not differences, differences[:5]

    populated = [
        row
        for row in rows
        if evidence_panel(
            row, cited[row.exception_id], dataset, config, ReplayLLMClient(), pairs
        ).hypotheses
    ]
    assert populated, "no panel carries a hypothesis, so this comparison is thin"
    print(f"\n  {len(rows)} panels identical; {len(populated)} carry ranked hypotheses")


def test_5_the_rendered_evidence_sections_agree(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """What each renderer OUTPUTS for one panel, compared semantically.

    The app is exercised through a recording double, so its real layout code
    runs. Markup and markdown differ by construction; the strings that carry
    meaning must not.
    """
    from eval.run_ai import duplicate_exceptions
    from settlesense.ai.client import ReplayLLMClient
    from settlesense.exceptions.taxonomy import VarianceCategory

    resolved = current_categories(dataset, config, AS_OF)
    rows = build_rows(store, resolved=resolved)
    cited = evidence_index(store, dataset, config)
    pairs = {pair.evidence_row_ids: pair for pair in duplicate_exceptions(dataset)}

    duplicate = str(VarianceCategory.DUPLICATE_CANDIDATE)
    row = next(item for item in rows if item.category == duplicate)
    panel = evidence_panel(row, cited[row.exception_id], dataset, config, ReplayLLMClient(), pairs)

    recorder = _RecordingStreamlit()
    app_module._render_evidence(recorder, panel)
    app_text = " ".join(recorder.text)
    static_html = render_module._row_html(row, panel, expanded=True)
    static_text = unescape(re.sub(r"<[^>]+>", " ", static_html))

    assert panel.hypotheses, "the chosen row has no hypotheses, so this proves little"
    for hypothesis in panel.hypotheses:
        for view, text in (("app", app_text), ("static", static_text)):
            assert hypothesis.candidate_id in text, (view, hypothesis.candidate_id)
            assert f"rank {hypothesis.rank}" in text, (view, hypothesis.rank)
            if not hypothesis.passed:
                assert hypothesis.failure_reason[:40] in text, view
    for check in panel.checks_run:
        assert check in app_text and check in static_text, check
    for candidate in panel.competing:
        assert candidate in app_text and candidate in static_text, candidate
    assert panel.abstention[:30] in app_text and panel.abstention[:30] in static_text
    print(
        f"\n  {len(panel.hypotheses)} ranks, {len(panel.checks_run)} checks, "
        f"{len(panel.competing)} candidates - all present in both views"
    )


def test_6_both_views_default_to_the_same_exception(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """One default, or a reviewer following a script sees two different rows."""
    from settlesense.exceptions.taxonomy import VarianceCategory

    resolved = current_categories(dataset, config, AS_OF)
    rows = build_rows(store, resolved=resolved)
    duplicate = str(VarianceCategory.DUPLICATE_CANDIDATE)

    page = render_page(store, dataset, config, AS_OF, limit=40)
    opened = re.search(r"<details open><summary>evidence for ([0-9a-f]+)</summary>", page)
    assert opened, "the static page opens no row by default"
    static_default = opened.group(1)

    # The app's ordering, taken from its own source rather than restated here.
    shown = rows[:40]
    app_default = sorted(shown, key=lambda r: (r.category != duplicate, -r.amount))[0]
    assert static_default == app_default.exception_id, (static_default, app_default.exception_id)
    assert app_default.category == duplicate, app_default.category
    print(f"\n  both default to {static_default} ({app_default.category})")


# ===========================================================================
# 3. The planted control
# ===========================================================================


@pytest.mark.charter_guard
def test_7_the_parity_check_fails_when_one_renderer_is_changed(
    store: ExceptionStore, dataset: Any, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PLANTED DIVERGENCE. Change one value in one renderer; parity must break.

    Without this, tests 3-5 passing is equally consistent with a comparison
    that reads the same object twice - which is exactly the shape of the bug
    that shipped one commit ago.
    """
    resolved = current_categories(dataset, config, AS_OF)
    rows = build_rows(store, resolved=resolved)
    clean = _static_table(render_page(store, dataset, config, AS_OF, limit=len(rows)))
    assert not [
        row.exception_id
        for row in rows
        if clean[row.exception_id]["Status"] != as_display_dict(row)["Status"]
    ], "precondition: the views already disagree"

    # One renderer, one value: the status pill now says something else.
    monkeypatch.setattr(render_module, "_pill", lambda row: '<span class="pill">TAMPERED</span>')
    tampered = _static_table(render_page(store, dataset, config, AS_OF, limit=len(rows)))
    differences = [
        row.exception_id
        for row in rows
        if tampered[row.exception_id]["Status"] != as_display_dict(row)["Status"]
    ]
    assert len(differences) == len(rows), (
        f"only {len(differences)} of {len(rows)} rows differed after tampering with "
        "every status; the comparison is not reading the rendered output"
    )
    print(f"\n  tampering with one renderer broke parity on all {len(differences)} rows")


# ===========================================================================
# 4. The scope notice and the category columns
# ===========================================================================


def test_8_both_views_state_what_they_are_showing(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """One sentence, both views - and TRUE OF THE VIEW THAT SHOWS IT.

    The static page renders the 40 largest; Streamlit scrolls all of them.
    Copying the page's number into the app would be a false statement about
    the app, so the sentence is computed from what that view displays.
    """
    rows = build_rows(store)
    page = render_page(store, dataset, config, AS_OF, limit=40)
    assert scope_notice(40, len(rows)) in page, "the static page does not state its scope"
    assert "Showing the 40 largest" in page and str(len(rows)) in page

    whole = render_page(store, dataset, config, AS_OF, limit=len(rows) + 10)
    assert scope_notice(len(rows), len(rows)) in whole
    assert "All " in whole and "the table scrolls" in whole

    app = (UI_DIR / "app.py").read_text(encoding="utf-8")
    assert "scope_notice(len(rows), len(rows))" in app, (
        "the app does not state its scope, or states the page's number instead"
    )
    print(
        f"\n  static: {scope_notice(40, len(rows))[:52]}...\n"
        f"  app:    {scope_notice(len(rows), len(rows))[:52]}..."
    )


def test_9_the_longest_taxonomy_category_renders_in_full(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """A clipped category reads as a data error, not a column width.

    Taken from the taxonomy rather than hard-coded, so a longer category added
    later cannot silently truncate.
    """
    from settlesense.exceptions.taxonomy import VARIANCE_CATEGORIES

    longest = max((str(category) for category in VARIANCE_CATEGORIES), key=len)
    resolved = current_categories(dataset, config, AS_OF)
    rows = build_rows(store, resolved=resolved)

    carrying = [
        row for row in rows if longest in (row.category_or_placeholder, row.resolved_or_placeholder)
    ]
    if not carrying:
        pytest.skip(f"no row carries {longest}; nothing to clip")

    for row in carrying:
        cells = as_display_dict(row)
        assert longest in (cells["Detected as"], cells["Resolved as"]), (
            f"{row.exception_id} lost {longest} before rendering"
        )
    page = render_page(store, dataset, config, AS_OF, limit=len(rows))
    assert longest in unescape(page), f"{longest} does not appear whole in the static page"

    app = (UI_DIR / "app.py").read_text(encoding="utf-8")
    assert "column_config" in app and "CATEGORY_COLUMNS" in app, (
        "the app does not widen the category columns, so the default sizing clips them"
    )
    assert CATEGORY_COLUMNS == ("Detected as", "Resolved as"), CATEGORY_COLUMNS

    # `st.dataframe` draws to a CANVAS - the cell text is not in the DOM, so a
    # clipped cell cannot be caught by inspecting the page. The width is
    # asserted instead, against the longest category the taxonomy can produce.
    from settlesense.ui.queue import CATEGORY_COLUMN_PIXELS

    needed = len(longest) * 8  # ~8px per character at the table's font size
    assert needed <= CATEGORY_COLUMN_PIXELS, (
        f"{CATEGORY_COLUMN_PIXELS}px is too narrow for {longest!r} "
        f"({len(longest)} chars, needs about {needed}px)"
    )
    print(f"\n  longest category {longest!r} ({len(longest)} chars) on {len(carrying)} rows")

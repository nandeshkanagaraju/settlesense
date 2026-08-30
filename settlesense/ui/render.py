"""M8 - the evidence queue as a static HTML page. Read-only, no model.

WHY A STATIC PAGE EXISTS ALONGSIDE THE STREAMLIT APP. Legibility matters more
than interactivity here: the queue has to be screenshotted for the README and
recorded for a video, and a file on disk is easier to do both with than a
server that has to be running. Both views read `ui/queue.py`, so they cannot
report different numbers.

PLAIN, NOT STYLED. System fonts, a border, one accent per status. A dashboard
that looks designed invites the reader to admire it; this one is meant to be
read, and the interesting content is the expansion under each row.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from datetime import date
from itertools import pairwise

from settlesense.ai.client import ReplayLLMClient
from settlesense.config import AppConfig
from settlesense.exceptions.store import (
    ExceptionStore,
    Population,
)
from settlesense.ingest import DayDataset
from settlesense.types import ExceptionStatus
from settlesense.ui.queue import (
    PAGE_TITLE,
    SEQUENCE_CAPTION,
    EvidencePanel,
    PopulationSummary,
    QueueRow,
    arrival_days,
    as_display_dict,
    build_rows,
    current_categories,
    evidence_index,
    evidence_panel,
    population_summaries,
    residual_sequence,
    scope_notice,
)

__all__ = ["render_page"]

COLUMNS = (
    "Population",
    "Exception ID",
    "Detected as",
    "Resolved as",
    "Amount",
    "Status",
    "Confidence",
    "Verified by",
    "Day opened",
    "Day confirmed",
)

_CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 24px; max-width: 1180px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 24px 0 6px; }
.sub { color: #57606a; margin: 0 0 18px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #d0d7de; padding: 5px 8px; text-align: left;
         vertical-align: top; }
th { background: #f6f8fa; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pill { display: inline-block; padding: 1px 7px; border-radius: 10px;
        font-size: 12px; font-weight: 600; white-space: nowrap; }
.pops { display: flex; gap: 14px; flex-wrap: wrap; margin: 0 0 8px; }
.pop { border: 1px solid #d0d7de; border-radius: 6px; padding: 8px 12px; }
.pop b { font-size: 17px; }
.pop span { color: #57606a; }
.seq { border: 1px solid #d0d7de; border-radius: 6px; padding: 10px 12px;
       margin: 0 0 16px; }
.caption { color: #57606a; font-size: 12.5px; margin: 6px 0 0; }
.scope { color: #7d4e00; background: #fff8c5; border: 1px solid #eac54f;
         border-radius: 6px; padding: 8px 12px; margin: 10px 0 0;
         font-size: 13px; }
details { margin: 4px 0 0; }
summary { cursor: pointer; color: #0969da; font-size: 12.5px; }
.panel { border-left: 3px solid #d0d7de; margin: 8px 0 12px 4px;
         padding: 2px 0 2px 12px; }
.panel h4 { margin: 10px 0 4px; font-size: 13px; }
.chain { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
         font-size: 12px; white-space: pre-wrap; }
.reject { color: #9a6700; }
.note { color: #57606a; font-size: 12.5px; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _pill(row: QueueRow) -> str:
    style = row.style
    return (
        f'<span class="pill" style="color:{style.colour};background:{style.background}">'
        f"{_esc(style.label)}</span>"
    )


def _sequence_block(sequence: Sequence[tuple[int, int]], label: str) -> str:
    """The residual over time, as a plain bar row plus its reason.

    A CAPTION, NOT A CHART LIBRARY. The number that matters is that it goes up
    before it comes down, and a reviewer's instinct is that residuals only
    shrink - so the rise is stated in words rather than left to be inferred
    from a shape.
    """
    if not sequence:
        return ""
    peak = max(count for _day, count in sequence) or 1
    bars = "".join(
        f'<div style="display:inline-block;text-align:center;margin-right:18px">'
        f'<div style="height:{max(4, round(46 * count / peak))}px;width:34px;'
        f'background:#0969da;border-radius:2px 2px 0 0"></div>'
        f'<div style="font-size:12px;margin-top:3px"><b>{count}</b><br>day {day}</div>'
        f"</div>"
        for day, count in sequence
    )
    arrow = " → ".join(str(count) for _day, count in sequence)
    rises = any(later > earlier for (_, earlier), (_, later) in pairwise(sequence))
    assert rises, "the sequence does not rise, so SEQUENCE_CAPTION would be wrong"
    reason = SEQUENCE_CAPTION
    return (
        f'<div class="seq"><b>{_esc(label)} residual: {_esc(arrow)}</b>'
        f'<div style="margin-top:10px">{bars}</div>'
        f'<p class="caption">{reason}</p></div>'
    )


def _population_block(summaries: Sequence[PopulationSummary]) -> str:
    """Three counts against three denominators, and NO combined rate (D11)."""
    cards = "".join(
        f'<div class="pop"><b>{s.residual}</b> residual<br>'
        f"<span>of {s.denominator:,} {_esc(s.denominator_name)}</span><br>"
        f"<span>{_esc(s.label)} · {s.persisted} tracked</span></div>"
        for s in summaries
    )
    return (
        f'<div class="pops">{cards}</div>'
        '<p class="caption">Three populations, three denominators. They are never '
        "averaged into one rate: a batch is not a payment, and dividing one by the "
        "other would produce a number about nothing.</p>"
    )


def _trail_html(panel: EvidencePanel) -> str:
    if not panel.steps:
        return '<p class="note">No source rows resolve for this exception.</p>'
    lines = "\n".join(
        f"{stage:>11} │ {_esc(row_id):<22} {_esc(fields)}" for stage, row_id, fields in panel.steps
    )
    tail = (
        ""
        if panel.trail_complete
        else '<p class="note">The chain stops before a bank credit — which is the finding, '
        "not a rendering gap.</p>"
    )
    return f'<div class="chain">{lines}</div>{tail}'


def _hypotheses_html(panel: EvidencePanel) -> str:
    """Ranked hypotheses with the check that rejected each. FROM THE PANEL.

    This function used to call `generate` and `verify` itself, which made the
    claim "both views read one EvidencePanel" false for the static page - it
    was true of the app and of nothing else. `tests/test_view_parity.py` now
    fails if either renderer computes a verdict.
    """
    if not panel.eligible_for_model:
        return (
            '<p class="note">Not eligible for the model: decided by rules '
            "(PDD 6.1), so no hypothesis is requested.</p>"
        )
    if panel.no_recording or not panel.hypotheses:
        return (
            '<p class="note">No model response was recorded for this exception. '
            "See <code>fixtures/llm_manifest.json</code>.</p>"
        )
    items = "".join(
        f"<li><b>rank {h.rank}</b> nominates <code>{_esc(h.candidate_id)}</code> — "
        + (
            '<b style="color:#1a7f37">VERIFIED</b>'
            if h.passed
            else f'<span class="reject">REJECTED — {_esc(h.failure_reason)}</span>'
        )
        + f'<br><span class="note">{_esc(h.reason)}</span></li>'
        for h in panel.hypotheses
    )
    header = (
        f'<p class="note">The verifier took <b>rank {panel.winning_rank}</b>.</p>'
        if panel.winning_rank is not None
        else '<p class="note">No hypothesis survived verification.</p>'
    )
    return f"{header}<ol style='margin:4px 0'>{items}</ol>"


def _verification_html(panel: EvidencePanel) -> str:
    """Which check passed or failed, BY NAME, and the residual where one exists."""
    if not panel.verification_ran:
        return (
            '<p class="note">No hypothesis-level check applies: this exception was '
            "settled by the deterministic passes, and the audit trail below is the "
            "record of that.</p>"
        )
    checks = ", ".join(panel.checks_run) or "none"
    residual = (
        "no residual applies to this category"
        if panel.computed_residual is None
        else f"computed residual {panel.computed_residual}"
    )
    detail = (
        f"<br><span class='reject'>{_esc(panel.verification_failure)}</span>"
        if not panel.verification_passed
        else ""
    )
    return (
        f"<p><b>{'PASSED' if panel.verification_passed else 'FAILED'}</b> — checks run: "
        f'<code>{_esc(checks)}</code><br><span class="note">{_esc(residual)}</span>{detail}</p>'
    )


def _audit_html(panel: EvidencePanel) -> str:
    if not panel.audit:
        return '<p class="note">No audit entries.</p>'
    lines = "\n".join(
        f"day {entry.arrival_day:>3} seq {entry.sequence:>3} │ "
        f"{_esc(entry.from_status or '—'):<22} → {_esc(entry.to_status):<18} "
        f"{_esc(entry.actor):<14} {_esc(entry.note)}"
        for entry in panel.audit
    )
    return f'<div class="chain">{lines}</div>'


def _row_html(row: QueueRow, panel: EvidencePanel, expanded: bool = False) -> str:
    """Layout only. Every value comes from `row` or `panel`."""
    cells = as_display_dict(row)
    numeric = {"Amount", "Confidence", "Day opened", "Day confirmed"}
    tds = "".join(
        f'<td class="num">{_esc(cells[name])}</td>'
        if name in numeric
        else (f"<td>{_pill(row)}</td>" if name == "Status" else f"<td>{_esc(cells[name])}</td>")
        for name in COLUMNS
    )
    competing = (
        '<p class="note">Competing candidates: '
        + " vs ".join(f"<code>{_esc(row_id)}</code>" for row_id in panel.competing)
        + ". Neither can be eliminated on the evidence, so neither is chosen.</p>"
        if panel.competing
        else ""
    )
    abstain = (
        f"<p>{_esc(panel.abstention)}</p>{competing}"
        if panel.abstention
        else "<p class='note'>Not abstained — this exception was resolved.</p>"
    )
    body = (
        '<div class="panel">'
        f"<h4>1 · Money trail</h4>{_trail_html(panel)}"
        f"<h4>2 · AI hypothesis</h4>{_hypotheses_html(panel)}"
        f"<h4>3 · Verification</h4>{_verification_html(panel)}"
        f"<h4>4 · Abstention</h4>{abstain}"
        f"<h4>5 · Audit trail</h4>{_audit_html(panel)}"
        "</div>"
    )
    return (
        f"<tr>{tds}</tr>"
        f'<tr><td colspan="{len(COLUMNS)}" style="border-top:none">'
        f"<details{' open' if expanded else ''}>"
        f"<summary>evidence for {_esc(row.exception_id)}</summary>{body}</details>"
        "</td></tr>"
    )


def render_page(
    store: ExceptionStore,
    dataset: DayDataset,
    config: AppConfig,
    as_of: date,
    day: int | None = None,
    limit: int | None = None,
) -> str:
    """The whole page. `limit` bounds the rows RENDERED and says so on the page.

    A silently truncated table reads as a complete one, which is the same
    defect class as a count derived by subtraction.

    NO LIMIT BY DEFAULT, and it used to be 40. The AI layer resolves the two
    largest duplicate pairs it can prove, and those rank 58 and 59 by amount -
    so a 40-row table stated "2 AI_VERIFIED" in its caption and showed none of
    them. The fix is to render everything, NOT to crop to the interesting rows:
    a crop is a selection, and selecting the rows that flatter the result is
    the one move this project has consistently refused. `scope_notice` reads
    the realised counts either way, so the caption stays true whatever a caller
    passes.
    """
    resolved = current_categories(dataset, config, as_of)
    rows = build_rows(store, day, resolved)
    shown = rows if limit is None else rows[:limit]
    summaries = population_summaries(store, dataset, config, as_of)
    days = arrival_days(store)
    evidence = evidence_index(store, dataset, config)
    client = ReplayLLMClient()

    # ONE ROW OPEN BY DEFAULT - the largest AI-eligible one. A page whose
    # interesting content is entirely behind a disclosure triangle screenshots
    # as a plain table, and the expansion is the part that shows the work.

    # PREFER A DUPLICATE PAIR. It is the only category with recorded model
    # responses, so it is the only row whose AI panel shows real ranks - which
    # is the part of the page worth screenshotting.
    from settlesense.exceptions.taxonomy import VarianceCategory

    duplicate = str(VarianceCategory.DUPLICATE_CANDIDATE)
    # A DUPLICATE PAIR, not merely "AI-eligible". Both halves identical at
    # every step is the more legible finding, and it is what the 480-case
    # abstention reason rests on; a batch whose credit never arrived has a
    # one-line trail that shows less.
    expand_id = next((row.exception_id for row in shown if row.category == duplicate), None)
    from eval.run_ai import duplicate_exceptions

    pair_exceptions = {pair.evidence_row_ids: pair for pair in duplicate_exceptions(dataset)}
    panels = {
        row.exception_id: evidence_panel(
            row, evidence[row.exception_id], dataset, config, client, pair_exceptions
        )
        for row in shown
    }
    body = "".join(
        _row_html(row, panels[row.exception_id], row.exception_id == expand_id) for row in shown
    )
    header = "".join(f"<th>{_esc(name)}</th>" for name in COLUMNS)
    verified = ", ".join(
        f"{sum(1 for row in rows if row.verified_by == value)} {value}"
        for value in sorted({row.verified_by for row in rows})
    )
    truncation = f'<p class="caption">{_esc(scope_notice(len(shown), len(rows)))}</p>'
    day_note = f"day {day}" if day is not None else f"all days ({', '.join(str(d) for d in days)})"

    # DERIVED FROM THE STORE, NEVER PASSED IN. A caller who had to remember to
    # set an `outage=True` flag would forget it on exactly the run where the
    # warning matters, and a page carrying it wrongly would be worse than one
    # carrying it not at all. The presence of a PENDING_AI_UNAVAILABLE row IS
    # the condition: nothing else in this project writes that status.
    #
    # WHY A READER NEEDS IT. The outage store covers days 1-12 and its AI path
    # never ran, so its Population A residual is 286 against the full run's 50.
    # Both numbers are correct and they are not comparable, and two screenshots
    # side by side give a reader no way to know that.
    outage_note = ""
    if any(row.status is ExceptionStatus.PENDING_AI_UNAVAILABLE for row in rows):
        span = f"{days[0]}-{days[-1]}" if len(days) > 1 else str(days[0])
        outage_note = (
            f'<p class="scope">Outage run: days {_esc(span)} only, AI path did not '
            "execute. Residual counts are not comparable with the full-run queue.</p>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{_esc(PAGE_TITLE)}</title><style>{_CSS}</style></head><body>
<h1>{_esc(PAGE_TITLE)}</h1>
<p class="sub">Read-only view of the state DB · {_esc(day_note)} ·
calendar {_esc(config.calendar.version)} · config {_esc(config.config_hash)}</p>
{outage_note}
<h2>Populations</h2>
{_population_block(summaries)}

<h2>Residual over time — Population B</h2>
{_sequence_block(residual_sequence(store, Population.B_BATCH_LINK), "Population B")}

<h2>Queue</h2>
<p class="caption"><b>Verified by</b> is the column that carries the thesis:
{_esc(verified)} across {len(rows)} tracked exceptions. Nearly every resolution is a
rule, not a model.</p>
{truncation}
<p class="caption"><b>Confidence 0.00</b> means NOT SCORED, not "certainly wrong":
a deterministic outcome is a rule result rather than a hypothesis, and the
confidence model applies only to hypotheses the verifier assessed.</p>
<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
</body></html>
"""

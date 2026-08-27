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
from typing import Any

from eval.run_ai import duplicate_exceptions
from settlesense.ai.client import FixtureMissError, ReplayLLMClient
from settlesense.ai.hypothesis import generate
from settlesense.ai.verifier import verify
from settlesense.config import AppConfig
from settlesense.exceptions.store import (
    ALL_STATUSES,
    RESIDUAL_STATES,
    ExceptionStore,
    Population,
)
from settlesense.ingest import DayDataset
from settlesense.ui.queue import (
    PopulationSummary,
    QueueRow,
    abstention_reason,
    arrival_days,
    as_display_dict,
    build_rows,
    money_trail,
    population_summaries,
    residual_sequence,
)

__all__ = ["render_page"]

COLUMNS = (
    "Population",
    "Exception ID",
    "Category",
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
    reason = (
        "It rises before it falls, and that is correct: a residual is a QUEUE, not a "
        "burn-down. A later day delivers batches whose credit is still days away, so "
        "arrivals outpace departures in the middle of a run."
        if rises
        else "This run does not show the rise; on the full dataset the sequence is 3 → 6 → 2."
    )
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


def _money_trail_html(row: QueueRow, dataset: DayDataset, evidence: tuple[str, ...]) -> str:
    trail = money_trail(evidence, dataset)
    if not trail.steps:
        return '<p class="note">No source rows resolve for this exception.</p>'
    lines = "\n".join(
        f"{stage:>11} │ {_esc(row_id):<22} {_esc(fields)}" for stage, row_id, fields in trail.steps
    )
    tail = (
        ""
        if trail.is_complete
        else '<p class="note">The chain stops before a bank credit — which is the finding, '
        "not a rendering gap.</p>"
    )
    return f'<div class="chain">{lines}</div>{tail}'


def _ai_html(
    row: QueueRow,
    dataset: DayDataset,
    config: AppConfig,
    evidence: tuple[str, ...],
    client: ReplayLLMClient,
    pair_exceptions: dict[tuple[str, ...], Any] | None = None,
) -> str:
    """The hypothesis, its RANK, and the ranks tried before it.

    Where a lower-ranked hypothesis won, the rejected higher ones are shown
    WITH their reasons. That is the 24/40 -> 33/40 property - verification
    steering rather than only braking - and a claim nobody can see is a claim
    nobody should believe.
    """
    from settlesense.ai.hypothesis import AI_ELIGIBLE_CATEGORIES
    from settlesense.types import Exception_

    # THE FIXTURE IS KEYED BY PROMPT HASH, and a prompt embeds the exception's
    # id, amount and reason. So the lookup has to rebuild the EXACT exception
    # the recorder used - not an equivalent one describing the same pair. A
    # probe assembled from the store row differs in all three fields and misses
    # a recording that exists, which reads on the page as "the model was never
    # asked" about a decision it answered.
    canonical = (pair_exceptions or {}).get(evidence)

    # THE ELIGIBILITY GATE APPLIES HERE TOO. A rules-decided category is never
    # offered to a model, and the UI must not be the place that quietly
    # bypasses that by building a prompt for one. Rendering the refusal is the
    # honest thing to show: it is why most rows have no AI section at all.
    if row.category not in AI_ELIGIBLE_CATEGORIES:
        return (
            f'<p class="note">Not eligible for the model: <code>'
            f"{_esc(row.category_or_placeholder)}</code> is decided by rules "
            "(PDD 6.1), so no hypothesis is requested.</p>"
        )

    probe = Exception_(
        exception_id=row.exception_id,
        category=row.category,
        amount=row.amount,
        status=row.status,
        confidence=row.confidence,
        evidence_row_ids=evidence,
        reason="",
        resolved_by=None,
        first_seen_day=row.day_opened,
        confirmed_day=row.day_confirmed,
        closed_day=None,
        audit=(),
    )
    try:
        offered = generate(canonical or probe, dataset, config, client)
    except FixtureMissError:
        return (
            '<p class="note">No model response was recorded for this exception. '
            "Fixtures exist for a 40-decision sample; see "
            "<code>fixtures/llm_manifest.json</code>.</p>"
        )
    if not offered:
        return '<p class="note">The model returned nothing schema-valid for this exception.</p>'

    parts: list[str] = []
    winner: int | None = None
    for hypothesis in offered:
        result = verify(hypothesis, dataset, config)
        if result.passed and winner is None:
            winner = hypothesis.rank
        verdict = (
            '<b style="color:#1a7f37">VERIFIED</b>'
            if result.passed
            else f'<span class="reject">REJECTED — {_esc(result.failure_reason)}</span>'
        )
        parts.append(
            f"<li><b>rank {hypothesis.rank}</b> nominates "
            f"<code>{_esc(hypothesis.candidate_id)}</code> — {verdict}"
            f'<br><span class="note">{_esc(hypothesis.reason)}</span></li>'
        )
    header = (
        f'<p class="note">The verifier took <b>rank {winner}</b>; '
        f"{winner} higher-ranked hypothes{'is was' if winner == 1 else 'es were'} "
        "rejected first.</p>"
        if winner
        else '<p class="note">No hypothesis survived verification.</p>'
    )
    return f"{header}<ol style='margin:4px 0'>{''.join(parts)}</ol>"


def _verification_html(
    row: QueueRow, dataset: DayDataset, config: AppConfig, evidence: tuple[str, ...]
) -> str:
    """Which check passed or failed, BY NAME, and the residual where one exists."""
    from settlesense.ai.hypothesis import Hypothesis

    if len(evidence) != 2:
        return '<p class="note">No structural claim applies to this exception.</p>'
    probe = Hypothesis(
        category=str(row.category),
        candidate_id=evidence[0],
        assertion=None,
        residual_amount=None,
        evidence_row_ids=evidence,
        reason="",
        rank=0,
    )
    result = verify(probe, dataset, config)
    checks = ", ".join(result.checks_run) or "none"
    residual = (
        "no residual applies to this category"
        if result.computed_residual is None
        else f"computed residual {result.computed_residual}"
    )
    verdict = "PASSED" if result.passed else "FAILED"
    detail = (
        f"<br><span class='reject'>{_esc(result.failure_reason)}</span>"
        if not result.passed
        else ""
    )
    return (
        f"<p><b>{verdict}</b> — checks run: <code>{_esc(checks)}</code><br>"
        f'<span class="note">{_esc(residual)}</span>{detail}</p>'
    )


def _audit_html(row: QueueRow) -> str:
    if not row.audit:
        return '<p class="note">No audit entries.</p>'
    lines = "\n".join(
        f"day {entry.arrival_day:>3} seq {entry.sequence:>3} │ "
        f"{_esc(entry.from_status or '—'):<22} → {_esc(entry.to_status):<18} "
        f"{_esc(entry.actor):<14} {_esc(entry.note)}"
        for entry in row.audit
    )
    return f'<div class="chain">{lines}</div>'


def _row_html(
    row: QueueRow,
    dataset: DayDataset,
    config: AppConfig,
    evidence: dict[str, tuple[str, ...]],
    client: ReplayLLMClient,
    expanded: bool = False,
    pair_exceptions: dict[tuple[str, ...], Any] | None = None,
) -> str:
    cells = as_display_dict(row)
    tds = "".join(
        f'<td class="num">{_esc(cells[name])}</td>'
        if name in {"Amount", "Confidence", "Day opened", "Day confirmed"}
        else (f"<td>{_pill(row)}</td>" if name == "Status" else f"<td>{_esc(cells[name])}</td>")
        for name in COLUMNS
    )
    ev = evidence.get(row.exception_id, ())
    reason = abstention_reason(row)
    competing = (
        '<p class="note">Competing candidates: '
        + " vs ".join(f"<code>{_esc(row_id)}</code>" for row_id in ev)
        + ". Neither can be eliminated on the evidence, so neither is chosen.</p>"
        if len(ev) == 2 and row.status in RESIDUAL_STATES
        else ""
    )
    abstain = (
        f"<h4>4 · Abstention</h4><p>{_esc(reason)}</p>{competing}"
        if reason
        else (
            "<h4>4 · Abstention</h4><p class='note'>Not abstained — this exception "
            "was resolved.</p>"
        )
    )
    panel = (
        '<div class="panel">'
        f"<h4>1 · Money trail</h4>{_money_trail_html(row, dataset, ev)}"
        f"<h4>2 · AI hypothesis</h4>"
        f"{_ai_html(row, dataset, config, ev, client, pair_exceptions)}"
        f"<h4>3 · Verification</h4>{_verification_html(row, dataset, config, ev)}"
        f"{abstain}"
        f"<h4>5 · Audit trail</h4>{_audit_html(row)}"
        "</div>"
    )
    return (
        f"<tr>{tds}</tr>"
        f'<tr><td colspan="{len(COLUMNS)}" style="border-top:none">'
        f"<details{' open' if expanded else ''}>"
        f"<summary>evidence for {_esc(row.exception_id)}</summary>{panel}</details>"
        "</td></tr>"
    )


def _evidence_index(
    store: ExceptionStore, dataset: DayDataset, config: AppConfig
) -> dict[str, tuple[str, ...]]:
    """What each exception's expansion should be built from.

    THREE DIFFERENT ANSWERS, because the engine records evidence appropriate to
    its own finding and that is not always what a reviewer needs to see:

      A DUPLICATE_CANDIDATE case cites the batch and bank rows its payment
      settled through - true, and useless for the question at hand. The pair of
      LEDGER ROWS is what makes it a duplicate, and it is also what the AI
      fixtures were recorded against, so a row cited any other way would miss
      the replay cache and show "no recording" for a decision that has one.

      A batch whose credit never arrived cites nothing, because there is
      nothing to cite. The trail is walked from its subject id instead.

      Everything else uses the evidence the engine recorded.
    """
    from settlesense.exceptions.taxonomy import VarianceCategory
    from settlesense.matching.engine import build_cases

    duplicate = str(VarianceCategory.DUPLICATE_CANDIDATE)
    order_of_case = {fact.case.case_id: fact.case.order_id for fact in build_cases(dataset, config)}
    pair_of_order: dict[str, tuple[str, ...]] = {}
    for pair in duplicate_exceptions(dataset):
        for order_id in pair.evidence_row_ids:
            pair_of_order[order_id] = pair.evidence_row_ids

    index: dict[str, tuple[str, ...]] = {}
    for population in Population:
        for exception in store.get_queue(ALL_STATUSES, population=population):
            subject = store._subject_id(exception.exception_id) or ""
            if exception.category == duplicate:
                order_id = order_of_case.get(subject, "")
                index[exception.exception_id] = pair_of_order.get(
                    order_id, exception.evidence_row_ids
                )
                continue
            index[exception.exception_id] = exception.evidence_row_ids or (subject,)
    return index


def render_page(
    store: ExceptionStore,
    dataset: DayDataset,
    config: AppConfig,
    as_of: date,
    day: int | None = None,
    limit: int = 40,
) -> str:
    """The whole page. `limit` bounds the rows RENDERED and says so on the page.

    A silently truncated table reads as a complete one, which is the same
    defect class as a count derived by subtraction.
    """
    rows = build_rows(store, day)
    shown = rows[:limit]
    summaries = population_summaries(store, dataset, config, as_of)
    days = arrival_days(store)
    evidence = _evidence_index(store, dataset, config)
    client = ReplayLLMClient()

    # ONE ROW OPEN BY DEFAULT - the largest AI-eligible one. A page whose
    # interesting content is entirely behind a disclosure triangle screenshots
    # as a plain table, and the expansion is the part that shows the work.
    from settlesense.ai.hypothesis import AI_ELIGIBLE_CATEGORIES

    # PREFER A DUPLICATE PAIR. It is the only category with recorded model
    # responses, so it is the only row whose AI panel shows real ranks - which
    # is the part of the page worth screenshotting.
    from settlesense.exceptions.taxonomy import VarianceCategory

    duplicate = str(VarianceCategory.DUPLICATE_CANDIDATE)
    expand_id = next(
        (row.exception_id for row in shown if row.category == duplicate),
        next((row.exception_id for row in shown if row.category in AI_ELIGIBLE_CATEGORIES), None),
    )
    pair_exceptions = {pair.evidence_row_ids: pair for pair in duplicate_exceptions(dataset)}
    body = "".join(
        _row_html(
            row,
            dataset,
            config,
            evidence,
            client,
            row.exception_id == expand_id,
            pair_exceptions,
        )
        for row in shown
    )
    header = "".join(f"<th>{_esc(name)}</th>" for name in COLUMNS)
    verified = ", ".join(
        f"{sum(1 for row in rows if row.verified_by == value)} {value}"
        for value in sorted({row.verified_by for row in rows})
    )
    truncation = (
        f'<p class="caption">Showing the {len(shown)} largest of {len(rows)} tracked '
        f"exceptions, sorted by amount. Nothing is hidden by filtering — the rest are "
        f"smaller.</p>"
        if len(shown) < len(rows)
        else f'<p class="caption">All {len(rows)} tracked exceptions.</p>'
    )
    day_note = f"day {day}" if day is not None else f"all days ({', '.join(str(d) for d in days)})"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>SettleSense — evidence queue</title><style>{_CSS}</style></head><body>
<h1>SettleSense — evidence queue</h1>
<p class="sub">Read-only view of the state DB · {_esc(day_note)} ·
calendar {_esc(config.calendar.version)} · config {_esc(config.config_hash)}</p>

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

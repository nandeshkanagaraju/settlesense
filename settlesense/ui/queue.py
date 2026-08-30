"""M8 - the evidence queue's data layer. No Streamlit, no HTML, no model.

WHY THIS IS A SEPARATE MODULE FROM THE VIEW. There are two views - a Streamlit
app and a static HTML page - and a second implementation is a second chance to
disagree. Everything either view shows is computed here, so the page a reviewer
screenshots and the app a reviewer clicks through cannot report different
numbers.

READ-ONLY, AND STRUCTURALLY SO. Nothing here opens a model client, and nothing
here writes. The store is opened, queried and closed; `test_ui.py` asserts by
AST that no UI module imports a client or calls a write method.

THREE POPULATIONS, THREE DENOMINATORS, NEVER MERGED (D11). The summary reports
each population's count against its own denominator and deliberately offers no
combined rate: a single "match rate" over A+B+C would divide payments by
batches and mean nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from settlesense.config import AppConfig
from settlesense.exceptions.store import (
    ALL_STATUSES,
    RESIDUAL_STATES,
    ExceptionStore,
    Population,
)
from settlesense.ingest import DayDataset
from settlesense.matching.engine import run
from settlesense.types import AuditEntry, Exception_, ExceptionStatus, Money, ResolutionSource

__all__ = [
    "PAGE_TITLE",
    "POPULATION_LABELS",
    "STATUS_STYLES",
    "MoneyTrail",
    "PopulationSummary",
    "QueueRow",
    "StatusStyle",
    "arrival_days",
    "build_rows",
    "evidence_index",
    "money_trail",
    "population_summaries",
    "residual_sequence",
    "verified_by",
]


@dataclass(frozen=True)
class StatusStyle:
    """How one status renders. Label and colour, in one place for both views."""

    label: str
    colour: str
    background: str


STATUS_STYLES: dict[ExceptionStatus, StatusStyle] = {
    ExceptionStatus.CONFIRMED: StatusStyle("CONFIRMED", "#1a7f37", "#dafbe1"),
    # CLOSED carries a check mark AND a darker green. Two signals, because
    # CONFIRMED and CLOSED are different states - explained versus actioned -
    # and a reader who cannot tell them apart at a glance cannot tell whether
    # money moved. `test_ui.py` asserts the two labels differ.
    ExceptionStatus.CLOSED: StatusStyle("CLOSED ✓", "#0a4d20", "#b7f0c6"),
    ExceptionStatus.ABSTAINED: StatusStyle("ABSTAINED", "#9a6700", "#fff8c5"),
    ExceptionStatus.OPEN: StatusStyle("OPEN", "#57606a", "#eaeef2"),
    # THE TWO WAITING STATES MUST NOT LOOK ALIKE, and until 2026-08-28 they did:
    # both were #0969da on #ddf4ff, identical in every field. One is data that
    # has not arrived yet; the other is the model being DOWN. Rendering a
    # service failure as a normal waiting state is the one confusion a queue of
    # unresolved items cannot afford, because the operator's response differs -
    # wait, versus go and look at why.
    #
    # PENDING_EVIDENCE MOVED, NOT PENDING_AI_UNAVAILABLE. The M8 build prompt
    # names five statuses and assigns "PENDING_AI_UNAVAILABLE blue"; it says
    # nothing about PENDING_EVIDENCE, which is how PENDING_EVIDENCE came to
    # borrow its neighbour's line in the first place. So the borrower moves and
    # the status the spec actually assigned a colour keeps it.
    ExceptionStatus.PENDING_EVIDENCE: StatusStyle("PENDING_EVIDENCE", "#8250df", "#fbefff"),
    ExceptionStatus.PENDING_AI_UNAVAILABLE: StatusStyle(
        "PENDING_AI_UNAVAILABLE", "#0969da", "#ddf4ff"
    ),
}

POPULATION_LABELS: dict[Population, str] = {
    Population.A_CASE: "A · case",
    Population.B_BATCH_LINK: "B · batch link",
    Population.C_ROW_VARIANCE: "C · row variance",
}

VERIFIED_DETERMINISTIC = "DETERMINISTIC"
VERIFIED_AI = "AI_VERIFIED"
VERIFIED_ABSTAINED = "ABSTAINED"


def verified_by(exception: Exception_) -> str:
    """DETERMINISTIC / AI_VERIFIED / ABSTAINED. The thesis column.

    Read off `resolved_by`, which the store sets only on a real transition, and
    NOT inferred from the status: a CONFIRMED exception with no resolver would
    otherwise be labelled deterministic by default, which is the one mistake
    this column exists to prevent.
    """
    if exception.resolved_by is ResolutionSource.DETERMINISTIC:
        return VERIFIED_DETERMINISTIC
    if exception.resolved_by is ResolutionSource.AI_VERIFIED:
        return VERIFIED_AI
    if exception.resolved_by is ResolutionSource.HUMAN:
        return "HUMAN"
    return VERIFIED_ABSTAINED


@dataclass(frozen=True)
class QueueRow:
    """One line of the queue, in column order."""

    population: str
    exception_id: str
    category: str
    """The category recorded AT DETECTION. Rendered as "Detected as"."""

    resolved_as: str | None
    """The category that closed it, or None if it is still open.

    TWO COLUMNS BECAUSE ONE LIED. A single "Category" column showed rows as
    `UNEXPLAINED` + `CONFIRMED` - a contradiction on its face, and the first
    thing a reviewer checks under "an honest exception list". The cause was not
    a wrong category: it was a STALE one. 274 of 339 rows opened as OPEN /
    UNEXPLAINED on day 1 or 12, were confirmed on a later day by
    DETERMINISTIC_REEVALUATION, and kept the category written at detection.
    Both facts are true and interesting - what it looked like when found, and
    what it turned out to be - so both are shown rather than one overwriting
    the other."""

    amount: Money
    status: ExceptionStatus
    confidence: Decimal
    verified_by: str
    day_opened: int
    day_confirmed: int | None
    audit: tuple[AuditEntry, ...]
    evidence_row_ids: tuple[str, ...] = ()

    @property
    def style(self) -> StatusStyle:
        return STATUS_STYLES[self.status]

    @property
    def resolved_or_placeholder(self) -> str:
        """The resolving category, "cleared" when nothing remained, "—" if open.

        `None` from the engine on a CONFIRMED subject means no variance is
        left, which is a RESULT and not an absence - rendering it as a dash
        alongside genuinely open rows would merge the two.
        """
        if self.status is not ExceptionStatus.CONFIRMED:
            return "—"
        return self.resolved_as or "cleared"

    @property
    def confidence_or_placeholder(self) -> str:
        """ "—" on any deterministic row. Confidence is an AI-path property.

        0.00 on 283 deterministic rows read as "no confidence", which is a
        statement about the resolution rather than about the column. A rule
        outcome has no confidence score, and a dash says so.
        """
        if self.verified_by != VERIFIED_AI:
            return "—"
        return f"{self.confidence:.2f}"

    @property
    def category_or_placeholder(self) -> str:
        """An empty category renders as an em dash, never as blank.

        A blank cell reads as "not loaded"; a dash reads as "none applies",
        which is what a confirmed case with no remaining variance actually means.
        """
        return self.category or "—"


@dataclass(frozen=True)
class PopulationSummary:
    """One population's counts against ITS OWN denominator (D11)."""

    population: Population
    label: str
    denominator: int
    denominator_name: str
    persisted: int
    residual: int

    @property
    def residual_rate(self) -> Decimal:
        """Residual over this population's denominator. NEVER across populations."""
        if not self.denominator:
            return Decimal("0")
        return (Decimal(self.residual) / Decimal(self.denominator)).quantize(Decimal("0.0001"))


def arrival_days(store: ExceptionStore) -> list[int]:
    """The days the store actually holds. NOT a hardcoded range.

    A UI that offered 1/2/3 would be describing a demo rather than the data,
    and this store's days are 1, 12 and 24 - checkpoints across a 24-day
    delivery window, not the first three days of anything.
    """
    rows = store._connection.execute(
        "SELECT DISTINCT arrival_day FROM audit ORDER BY arrival_day"
    ).fetchall()
    return [int(row["arrival_day"]) for row in rows]


def residual_sequence(
    store: ExceptionStore, population: Population | None = None
) -> list[tuple[int, int]]:
    """Residual count as of each arrival day.

    Computed from `first_seen_day` and `confirmed_day` rather than by re-running
    anything: an exception is residual on day D if it had opened by then and had
    not yet been confirmed.

    THE SEQUENCE IS NOT MONOTONIC AND MUST NOT BE PRESENTED AS IF IT WERE. A
    residual is a QUEUE, not a burn-down - a later day delivers batches whose
    credit is still days away, so arrivals outpace departures in the middle of
    a run. The realised Population B sequence is 3 -> 6 -> 2.
    """
    clause: str = ""
    params: list[str] = []
    if population is not None:
        clause = " AND population = ?"
        params = [population.value]
    sequence: list[tuple[int, int]] = []
    for day in arrival_days(store):
        row = store._connection.execute(
            "SELECT COUNT(*) AS n FROM exceptions WHERE first_seen_day <= ? "
            "AND (confirmed_day IS NULL OR confirmed_day > ?) AND closed_day IS NULL" + clause,
            [day, day, *params],
        ).fetchone()
        sequence.append((day, int(row["n"])))
    return sequence


def current_categories(
    dataset: DayDataset, config: AppConfig, as_of: date
) -> dict[str, str | None]:
    """What the engine says about every subject RIGHT NOW, by subject id.

    The store records what was true when an exception opened; this records what
    is true after every file has arrived. The queue shows both, and neither is
    derivable from the other.
    """
    result = run(dataset, config, as_of)
    latest: dict[str, str | None] = {}
    for case in result.cases:
        latest[case.case_id] = case.category
    for link in result.batch_links:
        latest[link.batch_id] = link.category
    for variance in result.row_variances:
        latest[variance.row_id] = variance.category
    return latest


def build_rows(
    store: ExceptionStore,
    day: int | None = None,
    resolved: dict[str, str | None] | None = None,
) -> list[QueueRow]:
    """Every exception, sorted by (-amount, exception_id).

    `day` filters to exceptions that EXISTED on that day - opened by then -
    rather than to those opened exactly then. A reviewer asking "what did the
    queue look like on day 12" means the former.
    """
    rows = [
        QueueRow(
            population=POPULATION_LABELS[population],
            exception_id=exception.exception_id,
            category=exception.category,
            amount=exception.amount,
            status=exception.status,
            confidence=exception.confidence,
            resolved_as=(resolved or {}).get(store.subject_id(exception.exception_id) or ""),
            verified_by=verified_by(exception),
            day_opened=exception.first_seen_day,
            day_confirmed=exception.confirmed_day,
            audit=exception.audit,
            evidence_row_ids=exception.evidence_row_ids,
        )
        for population in Population
        for exception in store.get_queue(ALL_STATUSES, population=population)
        if day is None or exception.first_seen_day <= day
    ]
    return sorted(rows, key=lambda row: (-row.amount, row.exception_id))


def population_summaries(
    store: ExceptionStore, dataset: DayDataset, config: AppConfig, as_of: date
) -> list[PopulationSummary]:
    """Each population against its own denominator. No combined rate exists.

    The denominators come from the ENGINE rather than the store, because the
    store holds exceptions and a denominator is a count of everything - the
    5,026 cases of which 52 are residual, not the 52.
    """
    result = run(dataset, config, as_of)
    denominators = {
        Population.A_CASE: (len(result.cases), "payments"),
        Population.B_BATCH_LINK: (len(result.batch_links), "batches"),
        Population.C_ROW_VARIANCE: (len(result.row_variances), "rows"),
    }
    summaries: list[PopulationSummary] = []
    for population in Population:
        denominator, name = denominators[population]
        summaries.append(
            PopulationSummary(
                population=population,
                label=POPULATION_LABELS[population],
                denominator=denominator,
                denominator_name=name,
                persisted=len(store.get_queue(ALL_STATUSES, population=population)),
                residual=len(store.get_queue(RESIDUAL_STATES, population=population)),
            )
        )
    return summaries


@dataclass(frozen=True)
class MoneyTrail:
    """The chain behind one exception: ledger -> payment -> settlement -> batch -> bank.

    Rendered as ordered (stage, id, fields) triples rather than a nested object,
    because the point is that a reviewer can follow it top to bottom and see
    where it stops.
    """

    steps: tuple[tuple[str, str, str], ...]

    @property
    def is_complete(self) -> bool:
        """Whether the chain reaches a bank credit. A broken chain IS the finding."""
        return any(stage == "bank" for stage, _id, _fields in self.steps)


def money_trail(evidence_row_ids: tuple[str, ...], dataset: DayDataset) -> MoneyTrail:
    """Walk the chain from whatever rows the exception cites.

    Follows the JOIN rather than reading a stored path: an order's relationship
    to a settlement line exists only through its payment, and a duplicate's
    missing chain is exactly what makes it detectable.
    """
    steps: list[tuple[str, str, str]] = []
    ledger = {row.order_id: row for row in dataset.ledger_rows}
    batches = {row.batch_id: row for row in dataset.settlement_batches}

    for order_id in sorted(row_id for row_id in evidence_row_ids if row_id in ledger):
        order = ledger[order_id]
        steps.append(
            ("ledger", order.order_id, f"customer={order.customer_id} gross={order.gross}")
        )
        payments = sorted(
            (row for row in dataset.payment_rows if row.order_id == order_id),
            key=lambda row: row.payment_id,
        )
        for payment in payments:
            steps.append(
                (
                    "payment",
                    payment.payment_id,
                    f"authorized={payment.authorized} captured={payment.captured} "
                    f"status={payment.status}",
                )
            )
            lines = sorted(
                (row for row in dataset.settlement_lines if row.payment_id == payment.payment_id),
                key=lambda row: row.settlement_id,
            )
            for line in lines:
                steps.append(
                    (
                        "settlement",
                        line.settlement_id,
                        f"type={line.line_type} gross={line.gross} fee={line.fee} "
                        f"tax={line.tax} net={line.net}",
                    )
                )
                batch = batches.get(line.batch_id) if line.batch_id else None
                if batch is None:
                    continue
                steps.append(
                    (
                        "batch",
                        batch.batch_id,
                        f"utr={batch.utr} net_total={batch.net_total} "
                        f"settled={batch.settled_event_date}",
                    )
                )
                for bank in sorted(
                    (row for row in dataset.bank_rows if batch.utr and batch.utr in row.narration),
                    key=lambda row: row.bank_txn_id,
                ):
                    steps.append(
                        (
                            "bank",
                            bank.bank_txn_id,
                            f"value_date={bank.value_date} amount={bank.amount} "
                            f"narration={bank.narration}",
                        )
                    )
    # Batch-grain exceptions cite a batch id directly and have no ledger row.
    for batch_id in sorted(row_id for row_id in evidence_row_ids if row_id in batches):
        batch = batches[batch_id]
        steps.append(
            (
                "batch",
                batch.batch_id,
                f"utr={batch.utr} net_total={batch.net_total} settled={batch.settled_event_date}",
            )
        )
    return MoneyTrail(steps=tuple(steps))


def evidence_index(
    store: ExceptionStore, dataset: DayDataset, config: AppConfig
) -> dict[str, tuple[str, ...]]:
    """What each exception's expansion should be built from.

    IN THE SHARED LAYER, not in one renderer. It lived in render.py, and the
    Streamlit app passed `exception.evidence_row_ids` straight to money_trail
    instead - so the static page showed a duplicate's ledger pair while the app
    showed "No source rows resolve for this exception" for the same row. Two
    views disagreeing is the exact failure this package's docstring claims is
    impossible, and it was true only for the numbers, not for the evidence.

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
    from eval.run_ai import duplicate_exceptions
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
            subject = store.subject_id(exception.exception_id) or ""
            if exception.category == duplicate:
                order_id = order_of_case.get(subject, "")
                index[exception.exception_id] = pair_of_order.get(
                    order_id, exception.evidence_row_ids
                )
                continue
            index[exception.exception_id] = exception.evidence_row_ids or (subject,)
    return index


def open_store(db_path: Path) -> ExceptionStore:
    """Open the state DB. The UI never creates one it did not find.

    A UI that silently created an empty database would render an empty queue
    and look identical to a run that reconciled everything.
    """
    if not db_path.exists():
        raise SystemExit(
            f"{db_path} does not exist. The evidence queue reads a state DB that "
            "`make demo-state` builds; it does not create one, because an empty "
            "queue and a missing database look the same on screen."
        )
    return ExceptionStore(db_path)


def abstention_reason(row: QueueRow) -> str:
    """Why this row is still open, NAMED.

    480 of 507 decisions abstain because the structural facts cannot separate
    the two halves of a duplicate pair. That sentence IS the finding, so it is
    surfaced verbatim rather than reduced to "unresolved".
    """
    if row.status not in RESIDUAL_STATES:
        return ""
    if row.status is ExceptionStatus.PENDING_EVIDENCE:
        return (
            "the expected file has not arrived yet - not an error, a question a later day answers"
        )
    last = row.audit[-1] if row.audit else None
    return last.note if last and last.note else "no reason recorded"


def as_display_dict(row: QueueRow) -> dict[str, Any]:
    """One row as plain strings, for either renderer."""
    return {
        "Population": row.population,
        "Exception ID": row.exception_id,
        "Detected as": row.category_or_placeholder,
        "Resolved as": row.resolved_or_placeholder,
        "Amount": f"{row.amount:,.2f}",
        "Status": row.style.label,
        "Confidence": row.confidence_or_placeholder,
        "Verified by": row.verified_by,
        "Day opened": str(row.day_opened),
        "Day confirmed": "—" if row.day_confirmed is None else str(row.day_confirmed),
    }


@dataclass(frozen=True)
class RankedHypothesis:
    """One model claim and the verifier's verdict on it, as DATA not markup.

    In the shared layer because both renderers must show the same ranks with
    the same rejection reasons. The static page rendering them while the
    Streamlit app said "see the static page" made the app's two most important
    sections hollow - and those two sections ARE the architecture.
    """

    rank: int
    candidate_id: str
    reason: str
    passed: bool
    failure_reason: str


@dataclass(frozen=True)
class EvidencePanel:
    """Everything a reviewer needs about one exception, in display order."""

    steps: tuple[tuple[str, str, str], ...]
    trail_complete: bool
    eligible_for_model: bool
    hypotheses: tuple[RankedHypothesis, ...]
    winning_rank: int | None
    no_recording: bool
    verification_ran: bool
    verification_passed: bool
    checks_run: tuple[str, ...]
    computed_residual: Money | None
    verification_failure: str
    abstention: str
    competing: tuple[str, ...]
    audit: tuple[AuditEntry, ...]


def evidence_panel(
    row: QueueRow,
    cited: tuple[str, ...],
    dataset: DayDataset,
    config: AppConfig,
    client: object,
    pair_exceptions: dict[tuple[str, ...], Exception_] | None = None,
) -> EvidencePanel:
    """Assemble the panel once. Both renderers display the same object.

    Takes a CLIENT rather than constructing one: the caller decides whether
    this is a replay client or nothing at all, and this layer never reaches a
    network either way.
    """
    from settlesense.ai.client import FixtureMissError
    from settlesense.ai.hypothesis import AI_ELIGIBLE_CATEGORIES, Hypothesis, generate
    from settlesense.ai.verifier import STRUCTURAL_CATEGORIES, verify

    trail = money_trail(cited, dataset)
    eligible = row.category in AI_ELIGIBLE_CATEGORIES
    hypotheses: list[RankedHypothesis] = []
    winning: int | None = None
    missing = False

    canonical = (pair_exceptions or {}).get(cited)
    if eligible and canonical is not None:
        try:
            offered = generate(canonical, dataset, config, client)  # type: ignore[arg-type]
        except FixtureMissError:
            offered = ()
            missing = True
        for hypothesis in offered:
            result = verify(hypothesis, dataset, config)
            if result.passed and winning is None:
                winning = hypothesis.rank
            hypotheses.append(
                RankedHypothesis(
                    rank=hypothesis.rank,
                    candidate_id=hypothesis.candidate_id,
                    reason=hypothesis.reason,
                    passed=result.passed,
                    failure_reason=result.failure_reason,
                )
            )

    ran = row.category in STRUCTURAL_CATEGORIES and len(cited) == 2
    verdict = (
        verify(
            Hypothesis(
                category=row.category,
                candidate_id=cited[0],
                assertion=None,
                residual_amount=None,
                evidence_row_ids=cited,
                reason="",
                rank=0,
            ),
            dataset,
            config,
        )
        if ran
        else None
    )
    return EvidencePanel(
        steps=trail.steps,
        trail_complete=trail.is_complete,
        eligible_for_model=eligible,
        hypotheses=tuple(hypotheses),
        winning_rank=winning,
        no_recording=missing,
        verification_ran=ran,
        verification_passed=bool(verdict and verdict.passed),
        checks_run=verdict.checks_run if verdict else (),
        computed_residual=verdict.computed_residual if verdict else None,
        verification_failure=verdict.failure_reason if verdict else "",
        abstention=abstention_reason(row),
        competing=cited if len(cited) == 2 and row.status in RESIDUAL_STATES else (),
        audit=row.audit,
    )


def scope_notice(shown: int, total: int) -> str:
    """What the reader is looking at: all of it, or the top N of it.

    ONE SENTENCE, BOTH VIEWS, AND IT MUST BE TRUE OF THE VIEW THAT SHOWS IT.
    Both now render every row, so both say so - but the sentence is computed
    from the realised `shown` and `total` rather than written, because the
    static page DID render only the 40 largest until the AI layer resolved two
    rows that rank 58 and 59. A hardcoded sentence would have gone on saying
    40 after the limit was lifted, and in a view whose purpose is an honest
    exception list a reviewer cannot otherwise tell whether they are seeing
    everything or a filtered subset.
    """
    if shown >= total:
        return (
            f"All {total:,} tracked exceptions, sorted by amount. Nothing is hidden "
            "by filtering; the table scrolls."
        )
    return (
        f"Showing the {shown:,} largest of {total:,} tracked exceptions, sorted by "
        "amount. Nothing is hidden by filtering — the rest are smaller."
    )


CATEGORY_COLUMNS = ("Detected as", "Resolved as")
CATEGORY_COLUMN_PIXELS = 210
"""Wide enough for the longest taxonomy category, with room to spare.

MEASURED, not guessed: `MISSING_VS_LATE_CREDIT` is 22 characters, and
`st.dataframe` draws to a CANVAS - the text is not in the DOM, so a clipped
cell cannot be caught by a test that inspects the page. `test_view_parity.py`
asserts the width is at least the longest category needs; the rendering itself
is checked by eye in reports/ui/streamlit-queue.png.
"""
"""Columns that must render a full taxonomy category, never a clipped one.

A clipped `MISSING_VS_LATE_CRED[` reads as a data error rather than a column
width, and this is the view whose whole purpose is an honest exception list.
"""


PAGE_TITLE = "Abstain — evidence queue"
"""The heading BOTH views render, defined once in the shared layer.

Abstain is the project's display name; the package keeps its original name,
`settlesense`. A second copy of this string in either renderer is a second
chance for the two views to disagree, which is the exact defect class this
module exists to close.
"""

SEQUENCE_CAPTION = (
    "Open batch links rise before they fall: day 12 delivers batches whose credit "
    "is not yet due. A residual is a queue, not a burn-down."
)
"""ALWAYS rendered directly under the chart, in both views.

The caption is the entire reason the chart is there. A rise shown without its
cause reads as a bug, and a reviewer's instinct is that residuals only shrink.
"""

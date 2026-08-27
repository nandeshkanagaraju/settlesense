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
    "POPULATION_LABELS",
    "STATUS_STYLES",
    "MoneyTrail",
    "PopulationSummary",
    "QueueRow",
    "StatusStyle",
    "arrival_days",
    "build_rows",
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
    ExceptionStatus.PENDING_EVIDENCE: StatusStyle("PENDING_EVIDENCE", "#0969da", "#ddf4ff"),
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


def build_rows(store: ExceptionStore, day: int | None = None) -> list[QueueRow]:
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
        "Category": row.category_or_placeholder,
        "Amount": f"{row.amount:,.2f}",
        "Status": row.style.label,
        "Confidence": f"{row.confidence:.2f}",
        "Verified by": row.verified_by,
        "Day opened": str(row.day_opened),
        "Day confirmed": "—" if row.day_confirmed is None else str(row.day_confirmed),
    }

"""M6 - incremental state. SQLite, no wall clock, three populations (SDD 4.7).

WHAT THIS MODULE IS FOR. Reconciliation is not a batch job that runs once. A
batch whose credit has not arrived on day 1 is not an error; it is a question
that day 3's bank file answers. Without persistence there is nowhere to hold
that question, so every run re-derives the world from scratch and the Day 1 ->
Day 2 story - an exception opening, waiting, and closing itself when evidence
lands - cannot be told at all.

NO WALL CLOCK, ANYWHERE (D2, SDD 4.7). Ordering is `(arrival_day, arrival_seq)`,
both integers supplied by the caller. There is no `ingested_at`. A TIMESTAMP,
DATETIME or REAL column in this schema is a D2 violation, and `_assert_schema_
has_no_wallclock_columns` refuses to open a database containing one - enforced
by the module at connect time, not only by a test, because a schema that
drifted in a migration would otherwise be caught a release late.

MONEY IS TEXT, NEVER REAL. SQLite REAL is a float, and a float that round-trips
through a database is a float that decides things (D1). Amounts are stored as
the Decimal's own string and read back through `money()`. The cost is that
SQL cannot sort by amount numerically, so `get_queue` sorts in Python - which
is stated where it happens rather than discovered later by someone whose
₹1,000,000 exception sorted below ₹9.

CONFIRMED IS NOT CLOSED (SDD 3). CONFIRMED means an explanation passed
verification. CLOSED means the accounting action was emitted. This module
writes CONFIRMED and never CLOSED on any reconciliation path; `close_exception`
exists as the single enforcement point for M9 and hard-codes the EXPORTER
actor. Every other route to CLOSED raises.

ALL THREE POPULATIONS PERSIST. Population A cases, Population B batch links and
Population C row variances each get exception rows, each carry their own
lifecycle, and each appear in the queue. Persisting only A would mean the two
batches whose credit has not arrived - the clearest thing a multi-day demo can
show closing - have nowhere to live between days.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from types import TracebackType

from settlesense.config import AppConfig
from settlesense.ingest import DayDataset, load_dataset
from settlesense.matching.engine import build_cases, merge_days, run
from settlesense.types import (
    AuditActor,
    AuditEntry,
    Exception_,
    ExceptionStatus,
    Money,
    ResolutionSource,
    money,
)

__all__ = [
    "ALL_STATUSES",
    "DayRun",
    "ExceptionStore",
    "IllegalTransitionError",
    "IngestResult",
    "MissingFileError",
    "Population",
    "StoreError",
    "as_of_for_arrival_day",
    "exception_id_for",
    "idempotency_key_for",
]


class StoreError(RuntimeError):
    """The store refuses to do this."""


class IllegalTransitionError(StoreError):
    """A status change the lifecycle does not permit (SDD 3)."""


class MissingFileError(StoreError):
    """A file that was expected and is not there.

    DISTINCT FROM AN EMPTY ONE, and that is the whole point of the type. A
    header-only bank table is legitimate data - settlement is T+N, so day 1
    genuinely has no credits. A missing file is a delivery failure. Collapsing
    both to "no rows" reports a clean reconciliation for a day whose statement
    never arrived, and the output is a page of zeroes either way.
    """


class Population(StrEnum):
    """Which denominator an exception belongs to (D11, SDD 3.1).

    Stored explicitly rather than inferred from the subject id, because the
    three populations must never be merged and a metric that has to guess
    which one a row belongs to will eventually guess wrong.
    """

    A_CASE = "A"
    B_BATCH_LINK = "B"
    C_ROW_VARIANCE = "C"


# ---------------------------------------------------------------------------
# The lifecycle (SDD 3). This table IS the rule; nothing else may change status.
# ---------------------------------------------------------------------------

LEGAL_TRANSITIONS: dict[ExceptionStatus, frozenset[ExceptionStatus]] = {
    ExceptionStatus.OPEN: frozenset(
        {
            ExceptionStatus.PENDING_EVIDENCE,
            ExceptionStatus.PENDING_AI_UNAVAILABLE,
            ExceptionStatus.CONFIRMED,
            ExceptionStatus.ABSTAINED,
        }
    ),
    # Both PENDING states return to OPEN when the thing they waited for
    # arrives; they do not shortcut to CONFIRMED. The SDD's diagram draws the
    # return edge into OPEN, and a batch whose credit lands is re-decided by
    # the engine rather than confirmed by the act of waiting. So the store
    # records TWO transitions - PENDING_EVIDENCE -> OPEN -> CONFIRMED - in the
    # same way SDD 3 requires two for a human resolving an abstention.
    ExceptionStatus.PENDING_EVIDENCE: frozenset({ExceptionStatus.OPEN}),
    ExceptionStatus.PENDING_AI_UNAVAILABLE: frozenset({ExceptionStatus.OPEN}),
    ExceptionStatus.CONFIRMED: frozenset({ExceptionStatus.CLOSED}),
    # ABSTAINED -> CONFIRMED only, via the M8 HUMAN_REVIEW queue. The queue is
    # a place, not a status: an exception must not leave ABSTAINED merely by
    # being looked at, or the abstention-rate denominator stops meaning
    # anything.
    ExceptionStatus.ABSTAINED: frozenset({ExceptionStatus.CONFIRMED}),
    ExceptionStatus.CLOSED: frozenset(),  # terminal
}

ALL_STATUSES: frozenset[ExceptionStatus] = frozenset(ExceptionStatus)
"""Every status, and a caller has to ASK for it by name.

`get_queue` has no default filter. A default that silently widens a query is
the same class of defect as a count derived by subtraction: the answer looks
right, and it is the answer to a question nobody asked. This one had already
bitten - a convergence test counted "all" while meaning "residual" and
reported Population B as 11 when the store and the engine both said 2. It
passed for the wrong reason, because a single-shot run confirms nothing and
the two counts coincide.

Wanting every status is legitimate; wanting it BY ACCIDENT is not. Spelling
it `get_queue(ALL_STATUSES)` costs one word and makes the intent reviewable.
"""

RESIDUAL_STATES: frozenset[ExceptionStatus] = frozenset(
    {
        ExceptionStatus.OPEN,
        ExceptionStatus.PENDING_EVIDENCE,
        ExceptionStatus.PENDING_AI_UNAVAILABLE,
    }
)
"""What `reevaluate_open` actually re-evaluates.

SDD 3's table calls all three "residual set". Re-evaluating only literal
status==OPEN would leave a PENDING_EVIDENCE batch waiting forever: it is
precisely the exception whose evidence a later day delivers, and it is the one
the multi-day demo turns on.
"""

_UNSCORED = Decimal("0")
"""Confidence for an exception no hypothesis has scored.

ZERO MEANS NOT SCORED, not "certainly wrong". Deterministic outcomes are rule
results rather than hypotheses and carry no confidence; the confidence model
is M7's. Recording a flattering 1.00 for a rule would put deterministic and
AI-verified resolutions on one scale, which is the comparison the whole
evaluation exists to keep separate.
"""


def as_of_for_arrival_day(arrival_day: int, config: AppConfig) -> date:
    """The date a delivery day corresponds to. NOT a clock read (D2).

    Derived from the configured simulation window, which is data, not a
    reading of now: day 1 is `window_start`, day N is N-1 calendar days after
    it. Delivery days are CALENDAR days rather than working days - the
    generator emits a day 5 and a day 6 for a Saturday and a Sunday, because a
    file drop that lands on a weekend is still a file drop.

    WHY THIS IS THE DEFAULT AND NOT THE CALLER'S PROBLEM. `as_of` decides
    whether a batch's credit is not yet due (PENDING_EVIDENCE) or past due and
    absent (MISSING_VS_LATE_CREDIT). A demo that passed a far-future value -
    the obvious thing to reach for - would turn every not-yet-due settlement
    into MISSING_VS_LATE_CREDIT and silently erase a distinction M3 works to
    draw. The correct value is knowable from the arrival day, so nobody should
    have to remember it.

    An explicit override stays available, and must: a test probing the
    boundary needs to ask what happens at a date the calendar would not
    choose, and a parameter that cannot be varied cannot be shown to matter.
    """
    if arrival_day < 1:
        raise StoreError(f"arrival_day is 1-indexed (SDD 4.1a); got {arrival_day}")
    derived = config.calendar.window_start + timedelta(days=arrival_day - 1)
    if derived > config.calendar.window_end:
        raise StoreError(
            f"arrival_day {arrival_day} derives {derived.isoformat()}, past the "
            f"simulation window end {config.calendar.window_end.isoformat()}. "
            "Every date in this project is inside that window (D13)."
        )
    return derived


def exception_id_for(population: Population, subject_id: str) -> str:
    """sha256 of a canonical tuple, first 16 hex (D10).

    STABLE ACROSS DAYS, which is what makes incremental state work at all: day
    3 must recognise the exception day 1 opened for the same batch, and a
    uuid4 or a row counter would mint a new one every run.
    """
    canonical = f"exception|{population.value}|{subject_id}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def idempotency_key_for(
    exception_id: str, resolution_type: str, evidence_ids: Sequence[str]
) -> str:
    """SDD 4.7: sha256(exception_id | resolution_type | evidence_ids_sorted).

    Evidence is SORTED before hashing, so the same resolution reached with the
    same evidence in a different order is the same key. Without that, a replay
    that happened to enumerate evidence differently would insert a second
    resolution for one decision and double-count it.
    """
    canonical = "|".join([exception_id, resolution_type, *sorted(evidence_ids)])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IngestResult:
    """What happened to one file. `skipped` and `is_empty` are independent."""

    path: str
    content_hash: str
    arrival_day: int
    arrival_seq: int
    row_count: int
    skipped: bool
    is_empty: bool

    @property
    def outcome(self) -> str:
        """A word for a report. Four distinguishable outcomes, not two."""
        if self.skipped:
            return "skipped (already ingested)"
        return "loaded, zero rows" if self.is_empty else f"loaded, {self.row_count} rows"


@dataclass(frozen=True)
class DayRun:
    """What one day's run did. Every list is sorted by an explicit key (D4)."""

    arrival_day: int
    ingested: tuple[IngestResult, ...]
    newly_confirmed: tuple[Exception_, ...]
    newly_opened: tuple[Exception_, ...]
    still_residual: tuple[Exception_, ...]

    @property
    def skipped_count(self) -> int:
        return sum(1 for result in self.ingested if result.skipped)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    content_hash TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    arrival_day  INTEGER NOT NULL,
    arrival_seq  INTEGER NOT NULL,
    row_count    INTEGER NOT NULL,
    is_empty     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS exceptions (
    exception_id     TEXT PRIMARY KEY,
    population       TEXT NOT NULL,
    subject_id       TEXT NOT NULL,
    category         TEXT,
    amount           TEXT NOT NULL,
    status           TEXT NOT NULL,
    confidence       TEXT NOT NULL,
    evidence_row_ids TEXT NOT NULL,
    reason           TEXT NOT NULL,
    resolved_by      TEXT,
    first_seen_day   INTEGER NOT NULL,
    confirmed_day    INTEGER,
    closed_day       INTEGER
);

CREATE TABLE IF NOT EXISTS audit (
    audit_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    exception_id TEXT NOT NULL,
    arrival_day  INTEGER NOT NULL,
    sequence     INTEGER NOT NULL,
    from_status  TEXT,
    to_status    TEXT NOT NULL,
    actor        TEXT NOT NULL,
    note         TEXT NOT NULL,
    evidence_ids TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resolutions (
    idempotency_key TEXT NOT NULL UNIQUE,
    exception_id    TEXT NOT NULL,
    resolution_type TEXT NOT NULL,
    evidence_ids    TEXT NOT NULL,
    arrival_day     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS watermarks (
    name        TEXT PRIMARY KEY,
    arrival_day INTEGER NOT NULL,
    arrival_seq INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exceptions_status ON exceptions(status);
CREATE INDEX IF NOT EXISTS idx_audit_exception ON audit(exception_id, arrival_day, sequence);
"""

_WALLCLOCK_COLUMN_TYPES = frozenset({"TIMESTAMP", "DATETIME", "DATE", "TIME", "REAL", "FLOAT"})
"""Declared types that make this a D2 or D1 violation.

REAL and FLOAT are here for D1, not D2: a money column that is a float decides
things differently on two machines. DATE is here because a business date in
the STATE schema would be an ordering key competing with (arrival_day,
arrival_seq) - business dates live in the row tables, which the store does not
own.
"""


class ExceptionStore:
    """The state DB. The only writer of exception status.

    `Exception_` is frozen so a transition produces a new instance; that is
    deliberate, and this class is what makes it enforceable. A caller cannot
    assign `status = CLOSED` and bypass the lifecycle, because the only way
    status reaches the database is through `_transition`, which consults
    LEGAL_TRANSITIONS first.
    """

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self._connection = sqlite3.connect(str(db_path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()
        self._assert_schema_has_no_wallclock_columns()

    # -- lifecycle of the connection itself ---------------------------------

    def __enter__(self) -> ExceptionStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _assert_schema_has_no_wallclock_columns(self) -> None:
        """Refuse to open a database with a clock in it (SDD 4.7).

        Checked against the LIVE schema rather than the DDL string above, so a
        database created by an older version - or hand-migrated - is caught on
        open rather than trusted because this file looks correct.
        """
        offenders: list[str] = []
        tables = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for table in tables:
            for column in self._connection.execute(f"PRAGMA table_info({table['name']})"):
                declared = str(column["type"]).upper()
                if declared in _WALLCLOCK_COLUMN_TYPES:
                    offenders.append(f"{table['name']}.{column['name']} {declared}")
        if offenders:
            raise StoreError(
                "the state schema contains wall-clock or float columns, which "
                f"D2/D1 forbid: {sorted(offenders)}. Ordering uses "
                "(arrival_day, arrival_seq); money is TEXT."
            )

    # -- files ---------------------------------------------------------------

    def ingest_file(self, path: Path, arrival_day: int, arrival_seq: int) -> IngestResult:
        """Hash the content; if seen, no-op (SDD 4.7).

        MISSING RAISES, EMPTY RETURNS. Both are outcomes, and they are
        different outcomes - see MissingFileError.
        """
        if not path.exists():
            raise MissingFileError(
                f"{path} does not exist. An empty table is legitimate data and "
                "would be ingested with zero rows; a file that is absent is a "
                "delivery failure, and reporting it as 'no rows' would show a "
                "clean reconciliation for a day whose statement never arrived."
            )
        if path.is_dir():
            raise MissingFileError(f"{path} is a directory, not a file")

        content = path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        row_count = _data_row_count(content)

        seen = self._connection.execute(
            "SELECT path, arrival_day, arrival_seq, row_count, is_empty FROM files "
            "WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if seen is not None:
            # TOUCH NOTHING. Not the arrival_day, not the seq. A re-ingested
            # file that updated its own arrival ordering would silently
            # reorder the audit trail of everything that referenced it.
            return IngestResult(
                path=str(path),
                content_hash=content_hash,
                arrival_day=int(seen["arrival_day"]),
                arrival_seq=int(seen["arrival_seq"]),
                row_count=int(seen["row_count"]),
                skipped=True,
                is_empty=bool(seen["is_empty"]),
            )

        self._connection.execute(
            "INSERT INTO files (content_hash, path, arrival_day, arrival_seq, row_count, "
            "is_empty) VALUES (?, ?, ?, ?, ?, ?)",
            (content_hash, str(path), arrival_day, arrival_seq, row_count, int(row_count == 0)),
        )
        self._connection.commit()
        return IngestResult(
            path=str(path),
            content_hash=content_hash,
            arrival_day=arrival_day,
            arrival_seq=arrival_seq,
            row_count=row_count,
            skipped=False,
            is_empty=row_count == 0,
        )

    def file_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS n FROM files").fetchone()
        return int(row["n"])

    # -- exceptions ----------------------------------------------------------

    def upsert_exception(
        self, exception: Exception_, population: Population, subject_id: str
    ) -> bool:
        """Record an exception if it is new. Returns whether it was inserted.

        Never overwrites an existing row: day 3 must not reset the status day
        1 established, and `first_seen_day` is the field that makes an ageing
        exception visible at all.
        """
        existing = self._connection.execute(
            "SELECT exception_id FROM exceptions WHERE exception_id = ?",
            (exception.exception_id,),
        ).fetchone()
        if existing is not None:
            return False
        self._connection.execute(
            "INSERT INTO exceptions (exception_id, population, subject_id, category, amount, "
            "status, confidence, evidence_row_ids, reason, resolved_by, first_seen_day, "
            "confirmed_day, closed_day) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exception.exception_id,
                population.value,
                subject_id,
                exception.category,
                str(exception.amount),
                str(exception.status),
                str(exception.confidence),
                json.dumps(sorted(exception.evidence_row_ids)),
                exception.reason,
                str(exception.resolved_by) if exception.resolved_by is not None else None,
                exception.first_seen_day,
                exception.confirmed_day,
                exception.closed_day,
            ),
        )
        self._connection.commit()
        self.append_audit(
            exception_id=exception.exception_id,
            from_status=None,  # the opening entry has no predecessor
            to_status=exception.status,
            actor=AuditActor.DETERMINISTIC,
            note=exception.reason,
            evidence_ids=exception.evidence_row_ids,
            arrival_day=exception.first_seen_day,
            sequence=self.next_sequence(exception.first_seen_day),
        )
        return True

    def get_exception(self, exception_id: str) -> Exception_ | None:
        row = self._connection.execute(
            "SELECT * FROM exceptions WHERE exception_id = ?", (exception_id,)
        ).fetchone()
        return None if row is None else self._to_exception(row)

    def get_queue(
        self,
        status_filter: ExceptionStatus | Iterable[ExceptionStatus],
        population: Population | None = None,
    ) -> list[Exception_]:
        """Exceptions sorted by (-amount, exception_id).

        `status_filter` IS REQUIRED. There is no default, and `None` is
        refused rather than read as "everything". Where a function can answer
        several questions, the caller names which one - see ALL_STATUSES for
        what a permissive default cost this project once already.

        SORTED IN PYTHON, NOT SQL. Amounts are TEXT because SQLite REAL is a
        float, so `ORDER BY amount DESC` would sort lexically and put ₹9.00
        above ₹1,000,000.00 - a review queue that showed the smallest
        exceptions first while looking correctly ordered.

        exception_id breaks ties, so two equal amounts have a stable order
        across runs (D4).
        """
        # Runtime guard as well as a type hint: an untyped caller (a notebook,
        # a JSON-driven CLI) can still pass None, and a silent "everything" is
        # the exact defect this signature change exists to remove.
        if status_filter is None:
            raise StoreError(
                "get_queue requires an explicit status_filter. Pass "
                "RESIDUAL_STATES for the residual set, a single status, or "
                "ALL_STATUSES to mean every status on purpose. A default that "
                "silently widens a query gives the right-looking answer to a "
                "question nobody asked."
            )
        clauses: list[str] = []
        params: list[str] = []
        statuses = (
            [status_filter] if isinstance(status_filter, ExceptionStatus) else list(status_filter)
        )
        if not statuses:
            return []
        clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
        params.extend(str(status) for status in statuses)
        if population is not None:
            clauses.append("population = ?")
            params.append(population.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(f"SELECT * FROM exceptions{where}", params).fetchall()
        exceptions = [self._to_exception(row) for row in rows]
        return sorted(exceptions, key=lambda exc: (-exc.amount, exc.exception_id))

    def _to_exception(self, row: sqlite3.Row) -> Exception_:
        return Exception_(
            exception_id=str(row["exception_id"]),
            category=str(row["category"]) if row["category"] is not None else "",
            amount=money(Decimal(str(row["amount"]))),
            status=ExceptionStatus(str(row["status"])),
            confidence=Decimal(str(row["confidence"])),
            evidence_row_ids=tuple(json.loads(str(row["evidence_row_ids"]))),
            reason=str(row["reason"]),
            resolved_by=(
                ResolutionSource(str(row["resolved_by"]))
                if row["resolved_by"] is not None
                else None
            ),
            first_seen_day=int(row["first_seen_day"]),
            confirmed_day=(None if row["confirmed_day"] is None else int(row["confirmed_day"])),
            closed_day=None if row["closed_day"] is None else int(row["closed_day"]),
            audit=self.get_audit(str(row["exception_id"])),
        )

    # -- audit ---------------------------------------------------------------

    def append_audit(
        self,
        exception_id: str,
        from_status: ExceptionStatus | None,
        to_status: ExceptionStatus,
        actor: AuditActor,
        note: str,
        evidence_ids: Sequence[str],
        arrival_day: int,
        sequence: int,
    ) -> AuditEntry:
        """APPEND ONLY. Never updated, never deleted (SDD 3).

        There is no update or delete path for this table anywhere in the
        module - not a private one, not a test helper. An audit trail that can
        be edited is a record of what someone wanted to have happened.
        """
        entry = AuditEntry(
            exception_id=exception_id,
            arrival_day=arrival_day,
            sequence=sequence,
            from_status=from_status,
            to_status=to_status,
            actor=actor,
            note=note,
            evidence_ids=tuple(sorted(evidence_ids)),
        )
        self._connection.execute(
            "INSERT INTO audit (exception_id, arrival_day, sequence, from_status, to_status, "
            "actor, note, evidence_ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.exception_id,
                entry.arrival_day,
                entry.sequence,
                str(entry.from_status) if entry.from_status is not None else None,
                str(entry.to_status),
                str(entry.actor),
                entry.note,
                json.dumps(list(entry.evidence_ids)),
            ),
        )
        self._connection.commit()
        return entry

    def get_audit(self, exception_id: str) -> tuple[AuditEntry, ...]:
        """One exception's trail, in (arrival_day, sequence) order.

        `sequence` is why two same-day transitions - ABSTAINED -> CONFIRMED
        and CONFIRMED -> CLOSED - can be told apart at all. Ordering on
        arrival_day alone could not, and a timestamp is what D2 forbids.
        """
        rows = self._connection.execute(
            "SELECT * FROM audit WHERE exception_id = ? ORDER BY arrival_day, sequence, audit_id",
            (exception_id,),
        ).fetchall()
        return tuple(
            AuditEntry(
                exception_id=str(row["exception_id"]),
                arrival_day=int(row["arrival_day"]),
                sequence=int(row["sequence"]),
                from_status=(
                    None if row["from_status"] is None else ExceptionStatus(str(row["from_status"]))
                ),
                to_status=ExceptionStatus(str(row["to_status"])),
                actor=AuditActor(str(row["actor"])),
                note=str(row["note"]),
                evidence_ids=tuple(json.loads(str(row["evidence_ids"]))),
            )
            for row in rows
        )

    def next_sequence(self, arrival_day: int) -> int:
        """The next ordering slot within a day. Monotonic, never reused."""
        row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), -1) AS top FROM audit WHERE arrival_day = ?",
            (arrival_day,),
        ).fetchone()
        return int(row["top"]) + 1

    # -- transitions ---------------------------------------------------------

    def _check_transition(self, current: ExceptionStatus, target: ExceptionStatus) -> None:
        allowed = LEGAL_TRANSITIONS[current]
        if target not in allowed:
            raise IllegalTransitionError(
                f"{current} -> {target} is not a legal transition. "
                f"From {current} the lifecycle permits {sorted(allowed) or 'nothing (terminal)'}. "
                "CLOSED has exactly one predecessor, CONFIRMED (SDD 3)."
            )

    def _transition(
        self,
        exception_id: str,
        target: ExceptionStatus,
        actor: AuditActor,
        note: str,
        evidence_ids: Sequence[str],
        arrival_day: int,
        sequence: int | None = None,
        resolved_by: ResolutionSource | None = None,
    ) -> Exception_:
        """THE ONLY WRITER OF `status`. Every path checks the table first."""
        current = self.get_exception(exception_id)
        if current is None:
            raise StoreError(f"no exception {exception_id!r} to transition")
        self._check_transition(current.status, target)

        confirmed_day = current.confirmed_day
        closed_day = current.closed_day
        if target is ExceptionStatus.CONFIRMED:
            confirmed_day = arrival_day
        if target is ExceptionStatus.CLOSED:
            closed_day = arrival_day

        self._connection.execute(
            "UPDATE exceptions SET status = ?, confirmed_day = ?, closed_day = ?, "
            "resolved_by = COALESCE(?, resolved_by) WHERE exception_id = ?",
            (
                str(target),
                confirmed_day,
                closed_day,
                str(resolved_by) if resolved_by is not None else None,
                exception_id,
            ),
        )
        self._connection.commit()
        self.append_audit(
            exception_id=exception_id,
            from_status=current.status,
            to_status=target,
            actor=actor,
            note=note,
            evidence_ids=evidence_ids,
            arrival_day=arrival_day,
            sequence=self.next_sequence(arrival_day) if sequence is None else sequence,
        )
        found = self.get_exception(exception_id)
        assert found is not None  # just written, in the same transaction
        return found

    def confirm_exception(
        self,
        exception: Exception_,
        resolution_type: str,
        evidence_ids: Sequence[str],
        arrival_day: int,
        actor: AuditActor = AuditActor.DETERMINISTIC,
        resolved_by: ResolutionSource = ResolutionSource.DETERMINISTIC,
        confidence: Decimal | None = None,
    ) -> bool:
        """Confirm once. Replay is a no-op. Returns whether it was NEW.

        SETS confirmed_day AND LEAVES closed_day None. Confirming is
        explaining; emitting the accounting entry is a separate act by a
        separate actor, and conflating them would make "we explained it" and
        "we posted it" the same claim.

        `confidence` DEFAULTS TO None, MEANING "DO NOT WRITE ONE". A rule
        result has no confidence and must keep the `_UNSCORED` zero it was
        opened with; a verified hypothesis has one and it has to survive.

        This parameter did not exist until 2026-08-28, and its absence was a
        silent data-loss bug rather than a missing feature: the M10 store path
        confirmed a duplicate pair the verifier had scored 1.0000 on all five
        components, and the queue rendered 0.00 - under a caption saying 0.00
        means NOT SCORED. Every number on screen was wrong and nothing failed.
        Found by reading a committed screenshot.

        INSERT OR IGNORE on the idempotency key is what makes a re-run of day
        3 harmless: the second attempt inserts nothing, transitions nothing,
        and appends no audit entry.
        """
        key = idempotency_key_for(exception.exception_id, resolution_type, evidence_ids)
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO resolutions (idempotency_key, exception_id, resolution_type, "
            "evidence_ids, arrival_day) VALUES (?, ?, ?, ?, ?)",
            (
                key,
                exception.exception_id,
                resolution_type,
                json.dumps(sorted(evidence_ids)),
                arrival_day,
            ),
        )
        self._connection.commit()
        if cursor.rowcount == 0:
            return False

        current = self.get_exception(exception.exception_id)
        if current is None:
            raise StoreError(f"no exception {exception.exception_id!r} to confirm")
        # A PENDING state returns to OPEN before it can be confirmed - two
        # transitions, both audited, exactly as SDD 3 requires for a human
        # resolving an abstention in one click.
        if current.status in {
            ExceptionStatus.PENDING_EVIDENCE,
            ExceptionStatus.PENDING_AI_UNAVAILABLE,
        }:
            self._transition(
                exception_id=exception.exception_id,
                target=ExceptionStatus.OPEN,
                actor=actor,
                note=f"evidence arrived on day {arrival_day}; reopened for decision",
                evidence_ids=evidence_ids,
                arrival_day=arrival_day,
            )
        # WRITTEN BEFORE THE TRANSITION, so a reader who sees CONFIRMED never
        # sees it beside a stale score. Stored as the Decimal's own string and
        # NOT through `money()`: confidence is a RATIO, not money, and
        # quantizing to paise would round a four-place score to two on the way
        # into a column that reads back unquantized.
        if confidence is not None:
            self._connection.execute(
                "UPDATE exceptions SET confidence = ? WHERE exception_id = ?",
                (str(confidence), exception.exception_id),
            )
            self._connection.commit()
        self._transition(
            exception_id=exception.exception_id,
            target=ExceptionStatus.CONFIRMED,
            actor=actor,
            note=resolution_type,
            evidence_ids=evidence_ids,
            arrival_day=arrival_day,
            resolved_by=resolved_by,
        )
        return True

    def close_exception(
        self, exception_id: str, arrival_day: int, note: str, sequence: int | None = None
    ) -> Exception_:
        """CONFIRMED -> CLOSED. The M9 exporter's single entry point.

        The actor is hard-coded to EXPORTER, not a parameter: SDD 3 names the
        exporter as the sole writer of CLOSED, and a caller able to pass
        `actor=HUMAN` here would make the audit trail claim a person emitted
        an accounting entry that a program emitted.

        No reconciliation path in this module calls this. `test_store.py`
        asserts that, by exercising every other public method and checking
        nothing reaches CLOSED.
        """
        return self._transition(
            exception_id=exception_id,
            target=ExceptionStatus.CLOSED,
            actor=AuditActor.EXPORTER,
            note=note,
            evidence_ids=(),
            arrival_day=arrival_day,
            sequence=sequence,
        )

    def mark_status(
        self,
        exception_id: str,
        target: ExceptionStatus,
        actor: AuditActor,
        note: str,
        arrival_day: int,
        evidence_ids: Sequence[str] = (),
    ) -> Exception_:
        """A checked transition for callers that are not confirming or closing.

        REFUSES CLOSED outright, whatever the predecessor, so the exporter's
        exclusivity does not rest on every caller choosing the right method.
        """
        if target is ExceptionStatus.CLOSED:
            raise IllegalTransitionError(
                "CLOSED is written only by close_exception, which the M9 exporter "
                "calls. Reaching it here would let a reconciliation path emit an "
                "accounting action (SDD 3)."
            )
        return self._transition(
            exception_id=exception_id,
            target=target,
            actor=actor,
            note=note,
            evidence_ids=evidence_ids,
            arrival_day=arrival_day,
        )

    # -- watermarks ----------------------------------------------------------

    def set_watermark(self, name: str, arrival_day: int, arrival_seq: int) -> None:
        self._connection.execute(
            "INSERT INTO watermarks (name, arrival_day, arrival_seq) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET arrival_day = excluded.arrival_day, "
            "arrival_seq = excluded.arrival_seq",
            (name, arrival_day, arrival_seq),
        )
        self._connection.commit()

    def get_watermark(self, name: str) -> tuple[int, int] | None:
        row = self._connection.execute(
            "SELECT arrival_day, arrival_seq FROM watermarks WHERE name = ?", (name,)
        ).fetchone()
        return None if row is None else (int(row["arrival_day"]), int(row["arrival_seq"]))

    # -- re-evaluation -------------------------------------------------------

    def reevaluate_open(
        self, dataset: DayDataset, config: AppConfig, as_of: date, arrival_day: int
    ) -> list[Exception_]:
        """Re-decide every residual exception against the enlarged dataset.

        RETURNS CONFIRMED, NEVER CLOSED. Explaining a variance and emitting the
        accounting entry for it are different acts by different actors.

        `as_of` is READ, not merely threaded: it reaches the engine, which uses
        it to decide whether a batch's credit is not yet due (PENDING_EVIDENCE)
        or past due and absent (MISSING_VS_LATE_CREDIT). A parameter passed
        through every signature and never read is behaviourally identical to
        reading a clock, and D2's scanner would not catch it - so
        `test_store.py` runs the same day at two different as_of values and
        asserts the statuses differ.
        """
        residual = self.get_queue(status_filter=RESIDUAL_STATES)
        if not residual:
            return []

        result = run(dataset, config, as_of)
        resolved_now = _resolved_subject_ids(result)
        newly_confirmed: list[Exception_] = []

        for exception in residual:
            subject = self.subject_id(exception.exception_id)
            if subject is None or subject not in resolved_now:
                continue
            evidence = resolved_now[subject]
            was_new = self.confirm_exception(
                exception=exception,
                resolution_type="DETERMINISTIC_REEVALUATION",
                evidence_ids=evidence,
                arrival_day=arrival_day,
            )
            if was_new:
                confirmed = self.get_exception(exception.exception_id)
                if confirmed is not None:
                    newly_confirmed.append(confirmed)

        return sorted(newly_confirmed, key=lambda exc: (-exc.amount, exc.exception_id))

    def subject_id(self, exception_id: str) -> str | None:
        """The case, batch or row this exception is ABOUT. None if unknown.

        PUBLIC BECAUSE TWO CALLERS OUTSIDE THIS MODULE NEED IT, and both were
        reaching for the underscore-prefixed version. The queue joins the
        store's detected category to the engine's current one, and the M9
        exporter picks a ledger from the same join - neither can be written
        without it, so it is part of the interface whether or not it was
        declared as one.

        READ-ONLY, and it is the only projection of a row this class exposes
        beyond `Exception_`. `subject_id` is not on `Exception_` itself because
        an exception is identified by its own id everywhere else; carrying the
        subject would invite code that matched on it.
        """
        row = self._connection.execute(
            "SELECT subject_id FROM exceptions WHERE exception_id = ?", (exception_id,)
        ).fetchone()
        return None if row is None else str(row["subject_id"])

    # -- the day driver ------------------------------------------------------

    def run_day(
        self,
        arrival_day: int,
        data_dir: Path,
        config: AppConfig,
        as_of: date | None = None,
    ) -> DayRun:
        """Ingest, re-evaluate, match, persist. No clock read anywhere (D2).

        `as_of=None` DERIVES the date from `arrival_day` (see
        `as_of_for_arrival_day`). That is the correct value and it is knowable,
        so it is not something a caller has to remember: a far-future override
        turns every not-yet-due settlement from PENDING_EVIDENCE into
        MISSING_VS_LATE_CREDIT, which reads as "the credit never came" for a
        credit that is not late yet.

        The override remains, and the tests use both: the derived default must
        produce PENDING_EVIDENCE for a not-yet-due batch, and an explicit
        far-future value must produce MISSING_VS_LATE_CREDIT for the same one.
        A parameter that cannot change the answer is a parameter nobody reads,
        which is behaviourally identical to reading a clock - and D2's scanner
        would see no clock call either way.

        The dataset handed to the engine is CUMULATIVE - every day up to and
        including this one - because a settlement line delivered on day 2 pairs
        with a payment delivered on day 1, and reconciling a day in isolation
        would report both halves as exceptions.
        """
        resolved_as_of = as_of_for_arrival_day(arrival_day, config) if as_of is None else as_of
        ingested = self._ingest_day_files(arrival_day, data_dir)
        dataset = self.cumulative_dataset(arrival_day, data_dir, config)

        newly_confirmed = self.reevaluate_open(dataset, config, resolved_as_of, arrival_day)
        newly_opened = self._persist_outcomes(dataset, config, resolved_as_of, arrival_day)

        last_seq = max((result.arrival_seq for result in ingested), default=0)
        self.set_watermark("ingest", arrival_day, last_seq)

        return DayRun(
            arrival_day=arrival_day,
            ingested=tuple(ingested),
            newly_confirmed=tuple(newly_confirmed),
            newly_opened=tuple(newly_opened),
            still_residual=tuple(self.get_queue(status_filter=RESIDUAL_STATES)),
        )

    def _ingest_day_files(self, arrival_day: int, data_dir: Path) -> list[IngestResult]:
        """Every day{N}_*.csv, in sorted order so arrival_seq is deterministic."""
        paths = sorted(data_dir.glob(f"day{arrival_day}_*.csv"))
        if not paths:
            raise MissingFileError(
                f"no day{arrival_day}_*.csv in {data_dir}. A day with no files is "
                "a delivery failure, not a quiet day - a quiet day still delivers "
                "header-only tables."
            )
        return [self.ingest_file(path, arrival_day, seq) for seq, path in enumerate(paths, start=1)]

    def cumulative_dataset(self, arrival_day: int, data_dir: Path, config: AppConfig) -> DayDataset:
        """Every day up to and including this one, merged.

        PUBLIC FOR THE SAME REASON `subject_id` IS. A settlement line delivered
        on day 2 pairs with a payment delivered on day 1, so any stage running
        AFTER the day driver - the M10 AI stage is the first - needs the same
        cumulative view the engine was given, not a fresh guess at which files
        count. The M10 demo runner had reimplemented this glob and got a walrus
        wrong; one implementation is the fix.
        """
        days = sorted(
            {
                int(name)
                for path in data_dir.glob("day*_*.csv")
                if (name := path.name[3:].split("_", 1)[0]).isdigit() and int(name) <= arrival_day
            }
        )
        return merge_days([load_dataset(data_dir, day, config) for day in days])

    def _persist_outcomes(
        self, dataset: DayDataset, config: AppConfig, as_of: date, arrival_day: int
    ) -> list[Exception_]:
        """ALL THREE POPULATIONS (D11). Each keeps its own denominator.

        Confirmed outcomes are not persisted as exceptions at all: an
        exception is a discrepancy that needs work, and a case the rules
        closed on the day it arrived never needed any. Persisting them would
        make the queue's denominator the case count rather than the residual
        count.
        """
        result = run(dataset, config, as_of)
        # THE MONEY AT STAKE, WHICH IS NOT ALWAYS THE VARIANCE. A
        # DUPLICATE_CANDIDATE has variance ZERO by construction - the books
        # balance whichever row is the double entry - but the sum in question
        # is the order's gross, and recording 0.00 sorted the entire AI-eligible
        # surface to the bottom of a queue ordered by amount. The M8 queue is
        # what surfaced it.
        gross_by_case = {
            fact.case.case_id: fact.case.expected_gross for fact in build_cases(dataset, config)
        }
        opened: list[Exception_] = []

        for case in result.cases:
            if case.status is ExceptionStatus.CONFIRMED:
                continue
            at_stake = case.variance
            if at_stake is None or at_stake == money(0):
                at_stake = gross_by_case.get(case.case_id, case.variance)
            opened.extend(
                self._open_if_new(
                    population=Population.A_CASE,
                    subject_id=case.case_id,
                    category=case.category,
                    amount=at_stake,
                    status=case.status,
                    evidence=(case.batch_id, case.bank_row_id),
                    reason=f"case {case.case_id}: {case.category or 'unresolved'}",
                    arrival_day=arrival_day,
                )
            )

        for link in result.batch_links:
            if link.status is ExceptionStatus.CONFIRMED:
                continue
            opened.extend(
                self._open_if_new(
                    population=Population.B_BATCH_LINK,
                    subject_id=link.batch_id,
                    category=link.category,
                    # batch_net_total, NEVER a case amount: Population B's
                    # money basis is its own (D11).
                    amount=link.batch_net_total,
                    status=link.status,
                    evidence=(link.bank_row_id,),
                    reason=f"batch {link.batch_id}: {link.category or 'unlinked'}",
                    arrival_day=arrival_day,
                )
            )

        for variance in result.row_variances:
            if variance.status is ExceptionStatus.CONFIRMED:
                continue
            opened.extend(
                self._open_if_new(
                    population=Population.C_ROW_VARIANCE,
                    subject_id=variance.row_id,
                    category=variance.category,
                    amount=variance.amount,
                    status=variance.status,
                    evidence=(variance.row_id,),
                    reason=f"{variance.source_table} {variance.row_id}: "
                    f"{variance.category or 'unexplained'}",
                    arrival_day=arrival_day,
                )
            )

        return sorted(opened, key=lambda exc: (-exc.amount, exc.exception_id))

    def _open_if_new(
        self,
        population: Population,
        subject_id: str,
        category: str | None,
        amount: Money | None,
        status: ExceptionStatus,
        evidence: Sequence[str | None],
        reason: str,
        arrival_day: int,
    ) -> list[Exception_]:
        exception = Exception_(
            exception_id=exception_id_for(population, subject_id),
            category=category or "",
            amount=money(amount if amount is not None else Decimal(0)),
            status=status,
            confidence=_UNSCORED,
            evidence_row_ids=tuple(sorted(item for item in evidence if item)),
            reason=reason,
            resolved_by=None,
            first_seen_day=arrival_day,
            confirmed_day=None,
            closed_day=None,
            audit=(),  # the opening entry is appended by upsert_exception
        )
        if self.upsert_exception(exception, population, subject_id):
            return [exception]
        return []


def _data_row_count(content: bytes) -> int:
    """CSV data rows, header excluded. Zero is a real answer, not an absence."""
    lines = [line for line in content.decode("utf-8").splitlines() if line.strip()]
    return max(len(lines) - 1, 0)


def _resolved_subject_ids(result: object) -> dict[str, tuple[str, ...]]:
    """Subjects the engine now considers CONFIRMED, with their evidence.

    Keyed by subject id across ALL THREE populations. Case ids, batch ids and
    row ids share no namespace - `test_id_namespaces_disjoint` asserts that -
    so one dict cannot collide across populations.
    """
    resolved: dict[str, tuple[str, ...]] = {}
    cases = getattr(result, "cases", ())
    for case in cases:
        if case.status is ExceptionStatus.CONFIRMED:
            resolved[case.case_id] = tuple(
                sorted(item for item in (case.batch_id, case.bank_row_id) if item)
            )
    for link in getattr(result, "batch_links", ()):
        if link.status is ExceptionStatus.CONFIRMED:
            resolved[link.batch_id] = tuple(sorted(item for item in (link.bank_row_id,) if item))
    for variance in getattr(result, "row_variances", ()):
        if variance.status is ExceptionStatus.CONFIRMED:
            resolved[variance.row_id] = (variance.row_id,)
    return resolved

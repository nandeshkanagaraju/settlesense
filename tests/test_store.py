"""M6 - the state DB: schema, lifecycle, idempotency, and the multi-day story.

THE LIFECYCLE TESTS COME FIRST because CONFIRMED and CLOSED were used
interchangeably before the SDD separated them, and that is the failure this
module has to make impossible rather than merely avoid. CONFIRMED means an
explanation passed verification; CLOSED means the accounting action was
emitted. A store that let a reconciliation path write CLOSED would be claiming
money moved because a rule matched.

THE TRANSITION TABLE IS TESTED EXHAUSTIVELY, all 36 ordered pairs, not the
five interesting ones. A hand-picked list of illegal transitions is a list
someone forgot to extend the day a state was added.

as_of IS ASSERTED TO CHANGE BEHAVIOUR. A parameter threaded through every
signature and never read is behaviourally identical to reading a clock, and
D2's AST scanner cannot tell the difference - it sees no clock call either
way. So the same day is run at two different as_of values and the statuses
must differ.
"""

from __future__ import annotations

import ast
import itertools
import sqlite3
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from settlesense.config import AppConfig, load_config
from settlesense.exceptions.store import (
    ALL_STATUSES,
    LEGAL_TRANSITIONS,
    RESIDUAL_STATES,
    DayRun,
    ExceptionStore,
    IllegalTransitionError,
    MissingFileError,
    Population,
    StoreError,
    as_of_for_arrival_day,
    exception_id_for,
    idempotency_key_for,
)
from settlesense.matching.engine import run
from settlesense.types import (
    AuditActor,
    Exception_,
    ExceptionStatus,
    ResolutionSource,
    money,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "dev"
STORE_SOURCE = REPO / "settlesense" / "exceptions" / "store.py"

EARLY_AS_OF = date(2026, 1, 5)
LATE_AS_OF = date(2026, 11, 30)

LAST_DELIVERY_DAY = 24
"""The dataset spans 24 DELIVERY days, not the 20 passed to `--days`.

`--days 20` is CAPTURE days. Settlement is T+N and the bank credit follows the
batch, so the last rows for a day-20 capture land on day 24. Reconciling at
arrival_day 20 leaves 557 Population A cases open - correctly, because their
settlement genuinely has not been delivered yet - and only day 24 converges on
the 52 the batch engine reports over the whole dataset.

That difference is the module working, not a discrepancy: an incremental store
that already knew about undelivered rows would be reading the future.
"""


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture
def store() -> ExceptionStore:
    return ExceptionStore()


def _exception(
    exception_id: str = "e1",
    status: ExceptionStatus = ExceptionStatus.OPEN,
    amount: str = "100.00",
) -> Exception_:
    return Exception_(
        exception_id=exception_id,
        category="UNEXPLAINED",
        amount=money(Decimal(amount)),
        status=status,
        confidence=Decimal("0"),
        evidence_row_ids=("row-1",),
        reason="test fixture",
        resolved_by=None,
        first_seen_day=1,
        confirmed_day=None,
        closed_day=None,
        audit=(),
    )


def _seed(store: ExceptionStore, **kwargs: object) -> Exception_:
    exception = _exception(**kwargs)  # type: ignore[arg-type]
    store.upsert_exception(exception, Population.A_CASE, subject_id=exception.exception_id)
    stored = store.get_exception(exception.exception_id)
    assert stored is not None
    return stored


# ===========================================================================
# 1. Lifecycle - CONFIRMED is not CLOSED
# ===========================================================================


def test_exception_status_has_exactly_six_members() -> None:
    """HUMAN_REVIEW is the M8 queue, not a status (SDD 3).

    A seventh member would give ABSTAINED two outgoing edges to CONFIRMED and
    make the abstention-rate denominator ambiguous - an exception would leave
    ABSTAINED merely by being looked at.
    """
    members = [status.value for status in ExceptionStatus]
    assert len(members) == 6, members
    assert "HUMAN_REVIEW" not in members
    assert set(LEGAL_TRANSITIONS) == set(ExceptionStatus), (
        "the transition table does not cover every status - an uncovered state "
        "would KeyError at runtime instead of refusing"
    )
    print(f"\n  six statuses: {members}")


@pytest.mark.charter_guard
def test_illegal_transitions_rejected(store: ExceptionStore) -> None:
    """EVERY one of the 36 ordered pairs, not a hand-picked five.

    A curated list of illegal transitions is a list nobody extends when a
    state is added. This enumerates the product, so a new status arrives with
    its whole row and column already under test.
    """
    legal: list[tuple[str, str]] = []
    illegal: list[tuple[str, str]] = []
    for source, target in itertools.product(ExceptionStatus, repeat=2):
        exception = _seed(store, exception_id=f"e-{source}-{target}")
        store._connection.execute(
            "UPDATE exceptions SET status = ? WHERE exception_id = ?",
            (str(source), exception.exception_id),
        )
        allowed = target in LEGAL_TRANSITIONS[source]
        if allowed:
            legal.append((source.value, target.value))
        else:
            illegal.append((source.value, target.value))
            with pytest.raises(IllegalTransitionError) as raised:
                store._transition(
                    exception_id=exception.exception_id,
                    target=target,
                    actor=AuditActor.HUMAN,
                    note="illegal",
                    evidence_ids=(),
                    arrival_day=1,
                )
            assert str(source) in str(raised.value) and str(target) in str(raised.value)

    assert len(legal) + len(illegal) == len(ExceptionStatus) ** 2 == 36
    for pair in (
        ("OPEN", "CLOSED"),
        ("ABSTAINED", "CLOSED"),
        ("PENDING_EVIDENCE", "CLOSED"),
        ("PENDING_AI_UNAVAILABLE", "CLOSED"),
    ):
        assert pair in illegal, f"{pair} must be illegal (SDD 3)"
    assert ("CONFIRMED", "CLOSED") in legal, "CLOSED must have exactly one predecessor"
    closed_predecessors = [src for src, tgt in legal if tgt == "CLOSED"]
    assert closed_predecessors == ["CONFIRMED"], closed_predecessors
    print(
        f"\n  {len(legal)} legal / {len(illegal)} illegal of 36; CLOSED predecessors: "
        f"{closed_predecessors}"
    )


def test_no_self_transition_is_legal() -> None:
    """A status changing to itself is not a transition, and recording one
    would put a meaningless row in an append-only trail."""
    offenders = [status.value for status in ExceptionStatus if status in LEGAL_TRANSITIONS[status]]
    assert not offenders, offenders
    print("\n  no status may transition to itself")


def test_closed_is_terminal() -> None:
    assert LEGAL_TRANSITIONS[ExceptionStatus.CLOSED] == frozenset()
    print("\n  CLOSED has no outgoing edges")


def test_abstained_reaches_closed_only_through_confirmed(store: ExceptionStore) -> None:
    """ABSTAINED -> HUMAN_REVIEW queue -> CONFIRMED -> CLOSED.

    Both transitions are recorded even though a human clicks once, and both
    land on the same arrival_day - which is exactly why AuditEntry.sequence
    has to exist. Ordering on arrival_day alone could not say which came
    first, and a timestamp is what D2 forbids.
    """
    exception = _seed(store, exception_id="abst", status=ExceptionStatus.OPEN)
    store.mark_status(
        exception.exception_id,
        ExceptionStatus.ABSTAINED,
        AuditActor.AI_VERIFIED,
        "no hypothesis",
        1,
    )
    with pytest.raises(IllegalTransitionError):
        store.close_exception(exception.exception_id, arrival_day=2, note="shortcut")

    store.mark_status(
        exception.exception_id, ExceptionStatus.CONFIRMED, AuditActor.HUMAN, "reviewer explained", 2
    )
    closed = store.close_exception(exception.exception_id, arrival_day=2, note="exported")
    assert closed.status is ExceptionStatus.CLOSED

    trail = store.get_audit(exception.exception_id)
    path = [entry.to_status.value for entry in trail]
    assert path == ["OPEN", "ABSTAINED", "CONFIRMED", "CLOSED"], path
    same_day = [entry for entry in trail if entry.arrival_day == 2]
    assert len(same_day) == 2, same_day
    assert same_day[0].sequence < same_day[1].sequence, (
        "two same-day transitions share an ordering slot - the trail cannot say "
        "which happened first"
    )
    assert same_day[1].actor is AuditActor.EXPORTER
    print(f"\n  {' -> '.join(path)}; day-2 sequences {[e.sequence for e in same_day]}")


@pytest.mark.charter_guard
def test_the_exporter_is_the_only_actor_that_writes_closed(store: ExceptionStore) -> None:
    """close_exception hard-codes EXPORTER; mark_status refuses CLOSED outright.

    Two independent barriers. If the actor were a parameter, an audit trail
    could claim a person emitted an accounting entry that a program emitted.
    """
    exception = _seed(store, exception_id="exp")
    store.mark_status(exception.exception_id, ExceptionStatus.CONFIRMED, AuditActor.HUMAN, "ok", 1)

    with pytest.raises(IllegalTransitionError, match="close_exception"):
        store.mark_status(
            exception.exception_id, ExceptionStatus.CLOSED, AuditActor.HUMAN, "sneaking", 1
        )

    closed = store.close_exception(exception.exception_id, arrival_day=1, note="exported")
    entry = closed.audit[-1]
    assert entry.actor is AuditActor.EXPORTER, entry.actor
    assert closed.closed_day == 1
    print(f"\n  mark_status refused CLOSED; close_exception wrote actor={entry.actor}")


@pytest.mark.charter_guard
def test_no_reconciliation_path_ever_writes_closed(config: AppConfig, tmp_path: Path) -> None:
    """The whole driver runs and nothing reaches CLOSED.

    Exercised behaviourally over real data rather than by reading the source:
    a grep would miss CLOSED arriving through a helper.
    """
    with ExceptionStore(tmp_path / "state.db") as store:
        for day in (1, 2, 3):
            store.run_day(day, DATA, config, LATE_AS_OF)
        closed = store.get_queue(status_filter=ExceptionStatus.CLOSED)
        confirmed = store.get_queue(status_filter=ExceptionStatus.CONFIRMED)
    assert closed == [], f"a reconciliation path wrote {len(closed)} CLOSED exceptions"
    assert confirmed, "precondition: nothing was confirmed, so this proves nothing"
    print(f"\n  3 days: {len(confirmed)} CONFIRMED, {len(closed)} CLOSED")


# ===========================================================================
# 2. Schema - no wall clock, no float, the five tables
# ===========================================================================


def _columns(store: ExceptionStore) -> dict[str, dict[str, str]]:
    connection = store._connection
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {
        str(table["name"]): {
            str(col["name"]): str(col["type"]).upper()
            for col in connection.execute(f"PRAGMA table_info({table['name']})")
        }
        for table in tables
    }


def test_the_schema_has_the_five_tables_sdd_names(store: ExceptionStore) -> None:
    """SDD 4.7: files, exceptions, audit, resolutions, watermarks."""
    tables = set(_columns(store))
    expected = {"files", "exceptions", "audit", "resolutions", "watermarks"}
    assert expected <= tables, f"missing {sorted(expected - tables)}"
    assert tables == expected, f"undeclared tables: {sorted(tables - expected)}"
    assert {"content_hash", "arrival_day", "arrival_seq"} <= set(_columns(store)["files"])
    print(f"\n  tables: {sorted(tables)}")


@pytest.mark.charter_guard
def test_state_db_schema_has_no_timing_columns(store: ExceptionStore) -> None:
    """SDD 4.7 names this test. A TIMESTAMP/DATETIME/REAL column is a D2 (or
    D1) violation, and ordering uses (arrival_day, arrival_seq)."""
    offenders: list[str] = []
    for table, columns in _columns(store).items():
        for name, declared in columns.items():
            if declared in {"TIMESTAMP", "DATETIME", "DATE", "TIME", "REAL", "FLOAT"}:
                offenders.append(f"{table}.{name} {declared}")
            if name.endswith("_at") or name in {"created", "updated", "ingested_at"}:
                offenders.append(f"{table}.{name} (wall-clock name)")
    assert not offenders, "wall-clock or float columns in the state DB:\n  " + "\n  ".join(
        offenders
    )

    amounts = _columns(store)["exceptions"]["amount"]
    assert amounts == "TEXT", f"money is {amounts}; SQLite REAL is a float (D1)"
    print(f"\n  {sum(len(c) for c in _columns(store).values())} columns, none a clock; amount TEXT")


@pytest.mark.charter_guard
def test_the_store_refuses_to_open_a_database_with_a_clock_column(tmp_path: Path) -> None:
    """FAULT INJECTION. The guard runs at connect time, not only in a test.

    A schema that drifted in a migration would otherwise be caught a release
    late - and by a test the migration author was not running.
    """
    path = tmp_path / "dirty.db"
    connection = sqlite3.connect(str(path))
    connection.execute("CREATE TABLE files (content_hash TEXT PRIMARY KEY, ingested_at TIMESTAMP)")
    connection.commit()
    connection.close()

    with pytest.raises(StoreError, match="wall-clock"):
        ExceptionStore(path)

    real_path = tmp_path / "real.db"
    connection = sqlite3.connect(str(real_path))
    connection.execute("CREATE TABLE junk (amount REAL)")
    connection.commit()
    connection.close()
    with pytest.raises(StoreError, match="wall-clock or float"):
        ExceptionStore(real_path)
    print("\n  refused a TIMESTAMP column and a REAL column at open time")


@pytest.mark.boundary_refusal
def test_resolutions_idempotency_key_is_unique(store: ExceptionStore) -> None:
    """The UNIQUE constraint SDD 4.7 names, asserted against SQLite itself."""
    indexes = store._connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='resolutions'"
    ).fetchone()
    assert "UNIQUE" in str(indexes["sql"]).upper()

    store._connection.execute("INSERT INTO resolutions VALUES ('k', 'e', 't', '[]', 1)")
    with pytest.raises(sqlite3.IntegrityError):
        store._connection.execute("INSERT INTO resolutions VALUES ('k', 'e2', 't2', '[]', 2)")
    print("\n  duplicate idempotency_key rejected by the database, not by application code")


@pytest.mark.charter_guard
def test_no_clock_is_read_anywhere_in_the_store() -> None:
    """D2, by AST. arrival_day and as_of are parameters."""
    banned = {"now", "utcnow", "today", "time", "time_ns", "monotonic", "perf_counter"}
    offenders = [
        f"{node.func.attr}() at line {node.lineno}"
        for node in ast.walk(ast.parse(STORE_SOURCE.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in banned
    ]
    assert not offenders, offenders
    source = STORE_SOURCE.read_text(encoding="utf-8")
    assert "CURRENT_TIMESTAMP" not in source.upper(), "SQL reads a clock"
    print("\n  store.py reads no clock, in Python or in SQL")


# ===========================================================================
# 3. Files - hash dedupe, and missing is not empty
# ===========================================================================


def test_ingest_file_skips_a_hash_it_has_seen_and_touches_nothing(
    store: ExceptionStore, tmp_path: Path
) -> None:
    """SDD 4.7: hash the file, if seen no-op.

    "Touches nothing" is asserted on the STORED arrival ordering: a re-ingest
    that updated arrival_day would silently reorder the audit trail of
    everything that referenced the file.
    """
    path = tmp_path / "day1_bank.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    first = store.ingest_file(path, arrival_day=1, arrival_seq=1)
    second = store.ingest_file(path, arrival_day=9, arrival_seq=9)

    assert not first.skipped and second.skipped
    assert store.file_count() == 1, "a skipped ingest inserted a row"
    assert (second.arrival_day, second.arrival_seq) == (1, 1), (
        f"re-ingest moved the file's arrival ordering to {(second.arrival_day, second.arrival_seq)}"
    )
    assert first.content_hash == second.content_hash
    print(f"\n  first {first.outcome}; second {second.outcome}; files={store.file_count()}")


@pytest.mark.boundary_refusal
def test_a_changed_file_is_a_new_file(store: ExceptionStore, tmp_path: Path) -> None:
    """FAULT INJECTION for the dedupe. Same name, different bytes, new row.

    Without this, the skip test passes for an implementation that keys on
    PATH - which would ignore a corrected file redelivered under the same name.
    """
    path = tmp_path / "day1_bank.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    first = store.ingest_file(path, 1, 1)
    path.write_text("a,b\n1,3\n", encoding="utf-8")
    second = store.ingest_file(path, 2, 1)

    assert not second.skipped, "a corrected redelivery was mistaken for a duplicate"
    assert first.content_hash != second.content_hash
    assert store.file_count() == 2
    print(f"\n  same path, different bytes -> {store.file_count()} files")


@pytest.mark.boundary_refusal
def test_missing_and_empty_are_different_outcomes(store: ExceptionStore, tmp_path: Path) -> None:
    """An empty table is data; a missing file is a delivery failure."""
    with pytest.raises(MissingFileError) as caught:
        store.ingest_file(tmp_path / "absent.csv", 1, 1)
    missing_message = str(caught.value)

    empty = tmp_path / "day1_bank.csv"
    empty.write_text("bank_txn_id,value_date,amount,narration,direction\n", encoding="utf-8")
    result = store.ingest_file(empty, 1, 1)

    assert result.is_empty and not result.skipped and result.row_count == 0
    assert result.outcome != missing_message
    assert "does not exist" in missing_message
    print(f"\n  missing -> raises; empty -> {result.outcome!r}")


def test_the_four_ingest_outcomes_are_distinguishable(
    store: ExceptionStore, tmp_path: Path
) -> None:
    """skipped and is_empty are INDEPENDENT, so there are four states, not two.

    An empty file that was already ingested is a different report line from an
    empty file arriving for the first time, and both differ from a populated
    one.
    """
    empty = tmp_path / "day1_bank.csv"
    empty.write_text("h\n", encoding="utf-8")
    full = tmp_path / "day1_ledger.csv"
    full.write_text("h\n1\n", encoding="utf-8")

    outcomes = {
        (False, True): store.ingest_file(empty, 1, 1).outcome,
        (False, False): store.ingest_file(full, 1, 2).outcome,
        (True, True): store.ingest_file(empty, 1, 3).outcome,
        (True, False): store.ingest_file(full, 1, 4).outcome,
    }
    assert len(set(outcomes.values())) >= 3, outcomes
    print(f"\n  outcomes: {outcomes}")


# ===========================================================================
# 4. Idempotency and audit
# ===========================================================================


def test_confirm_is_idempotent_and_sets_confirmed_not_closed(store: ExceptionStore) -> None:
    """Second confirm returns False, changes nothing, appends no audit row."""
    exception = _seed(store, exception_id="idem", amount="250.00")
    evidence = ["b2", "b1"]

    first = store.confirm_exception(exception, "REFUND_OFFSET", evidence, arrival_day=3)
    trail_after_first = len(store.get_audit(exception.exception_id))
    second = store.confirm_exception(exception, "REFUND_OFFSET", list(reversed(evidence)), 4)

    assert first is True and second is False
    confirmed = store.get_exception(exception.exception_id)
    assert confirmed is not None
    assert confirmed.status is ExceptionStatus.CONFIRMED
    assert confirmed.confirmed_day == 3, "the replay moved confirmed_day"
    assert confirmed.closed_day is None, "confirming emitted an accounting action"
    assert confirmed.resolved_by is ResolutionSource.DETERMINISTIC
    assert len(store.get_audit(exception.exception_id)) == trail_after_first, (
        "the replay appended an audit entry"
    )
    print(
        f"\n  confirm/replay -> {first}/{second}; confirmed_day={confirmed.confirmed_day} "
        f"closed_day={confirmed.closed_day}"
    )


def test_the_idempotency_key_sorts_its_evidence() -> None:
    """SDD 4.7 hashes evidence_ids SORTED.

    Without sorting, a replay that enumerated evidence in a different order
    would produce a different key, insert a second resolution for one
    decision, and double-count it.
    """
    forward = idempotency_key_for("e1", "T", ["b", "a"])
    backward = idempotency_key_for("e1", "T", ["a", "b"])
    assert forward == backward
    assert forward != idempotency_key_for("e1", "T", ["a", "c"]), "evidence is ignored"
    assert forward != idempotency_key_for("e2", "T", ["a", "b"]), "exception_id is ignored"
    assert forward != idempotency_key_for("e1", "U", ["a", "b"]), "resolution_type is ignored"
    print(f"\n  order-independent, but sensitive to all three inputs: {forward[:16]}")


def test_exception_ids_are_stable_and_population_scoped() -> None:
    """Day 3 must recognise the exception day 1 opened for the same subject."""
    first = exception_id_for(Population.B_BATCH_LINK, "BAT_1")
    assert first == exception_id_for(Population.B_BATCH_LINK, "BAT_1")
    assert first != exception_id_for(Population.A_CASE, "BAT_1"), (
        "the same subject id in two populations collides - one would overwrite "
        "the other's lifecycle"
    )
    assert len(first) == 16, first
    print(f"\n  stable id {first}, distinct per population")


@pytest.mark.hygiene
def test_the_audit_trail_is_append_only(store: ExceptionStore) -> None:
    """No update or delete path exists anywhere in the module.

    Checked by AST over store.py rather than by trying to call something: the
    assertion is that no such code EXISTS, and a runtime probe can only show
    that the paths it happened to try are absent.
    """
    source = STORE_SOURCE.read_text(encoding="utf-8")
    statements = [
        node.value.strip()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    offenders = [
        text
        for text in statements
        if ("UPDATE audit" in text.upper() or "DELETE FROM AUDIT" in text.upper())
    ]
    assert not offenders, f"the audit trail is mutable: {offenders}"

    exception = _seed(store, exception_id="trail")
    store.mark_status(exception.exception_id, ExceptionStatus.ABSTAINED, AuditActor.HUMAN, "n", 1)
    store.mark_status(exception.exception_id, ExceptionStatus.CONFIRMED, AuditActor.HUMAN, "n", 2)
    trail = store.get_audit(exception.exception_id)
    assert [entry.to_status.value for entry in trail] == ["OPEN", "ABSTAINED", "CONFIRMED"]
    assert [(e.arrival_day, e.sequence) for e in trail] == sorted(
        (e.arrival_day, e.sequence) for e in trail
    ), "the trail is not ordered by (arrival_day, sequence)"
    print(f"\n  {len(trail)} entries, ordered, no UPDATE/DELETE path in the module")


def test_the_opening_audit_entry_has_no_predecessor(store: ExceptionStore) -> None:
    exception = _seed(store, exception_id="first")
    opening = store.get_audit(exception.exception_id)[0]
    assert opening.from_status is None, opening.from_status
    assert opening.to_status is ExceptionStatus.OPEN
    print(f"\n  opening entry: {opening.from_status} -> {opening.to_status}")


# ===========================================================================
# 5. The queue
# ===========================================================================


@pytest.mark.hygiene
def test_get_queue_sorts_by_amount_descending_then_id(store: ExceptionStore) -> None:
    """(-amount, exception_id), and NOT lexically.

    The amounts are chosen so a string sort gives a different answer: "9.00"
    sorts above "1000000.00" lexically, which would put the largest exception
    at the bottom of a review queue that looked correctly ordered.
    """
    for exception_id, amount in (("c", "9.00"), ("a", "1000000.00"), ("b", "1000000.00")):
        _seed(store, exception_id=exception_id, amount=amount)

    queue = store.get_queue(ALL_STATUSES)
    order = [(exc.exception_id, str(exc.amount)) for exc in queue]
    assert [item[0] for item in order] == ["a", "b", "c"], order
    assert queue[0].amount > queue[-1].amount

    lexical = sorted(order, key=lambda item: item[1], reverse=True)
    assert [item[0] for item in lexical] != [item[0] for item in order], (
        "the fixture does not distinguish numeric from lexical sorting, so it "
        "cannot prove the queue is ordered by value"
    )
    print(f"\n  numeric {[i[0] for i in order]} vs lexical {[i[0] for i in lexical]}")


def test_the_queue_filters_by_status_and_population(store: ExceptionStore) -> None:
    a = _exception("qa", ExceptionStatus.OPEN, "10.00")
    b = _exception("qb", ExceptionStatus.PENDING_EVIDENCE, "20.00")
    store.upsert_exception(a, Population.A_CASE, "case-1")
    store.upsert_exception(b, Population.B_BATCH_LINK, "BAT_1")

    assert [e.exception_id for e in store.get_queue(ExceptionStatus.OPEN)] == ["qa"]
    assert [
        e.exception_id for e in store.get_queue(ALL_STATUSES, population=Population.B_BATCH_LINK)
    ] == ["qb"]
    assert len(store.get_queue(RESIDUAL_STATES)) == 2
    assert store.get_queue(status_filter=[]) == [], "an empty filter returned everything"
    print("\n  filters: status, population, and an empty list means empty")


# ===========================================================================
# 6. All three populations, and the multi-day story
# ===========================================================================


@pytest.fixture(scope="module")
def full_run(config: AppConfig, tmp_path_factory: pytest.TempPathFactory) -> ExceptionStore:
    """Every delivered day, at a late as_of. Day 24, not 20 - see above."""
    store = ExceptionStore(tmp_path_factory.mktemp("m6") / "full.db")
    store.run_day(LAST_DELIVERY_DAY, DATA, config, LATE_AS_OF)
    return store


def test_the_store_converges_on_the_batch_engines_residual(
    full_run: ExceptionStore, config: AppConfig
) -> None:
    """THE STRONGEST CHECK AVAILABLE ON M6. Two paths, one answer.

    The batch engine reconciles the whole dataset in one call. The store
    ingests, persists, re-evaluates and carries state across days. If those
    disagree, one of them is wrong - and nothing else in this file would say
    which, because every other test checks the store against itself.

    Compared per population, never summed: three denominators (D11).
    """
    dataset = full_run.cumulative_dataset(LAST_DELIVERY_DAY, DATA, config)
    result = run(dataset, config, LATE_AS_OF)

    engine = {
        Population.A_CASE: sum(
            1 for case in result.cases if case.status is not ExceptionStatus.CONFIRMED
        ),
        Population.B_BATCH_LINK: sum(
            1 for link in result.batch_links if link.status is not ExceptionStatus.CONFIRMED
        ),
        Population.C_ROW_VARIANCE: sum(
            1 for row in result.row_variances if row.status is not ExceptionStatus.CONFIRMED
        ),
    }
    # RESIDUAL_STATES, and the signature now REFUSES to be asked vaguely.
    # An earlier version of this test passed for the wrong reason: get_queue
    # defaulted to every status, a single-shot run confirms nothing, so "all"
    # and "residual" coincided. Under a real multi-day run it reported
    # Population B as 11 while the store and engine both said 2.
    stored = {
        pop: len(full_run.get_queue(status_filter=RESIDUAL_STATES, population=pop))
        for pop in Population
    }

    assert stored == engine, f"store {stored} disagrees with the engine {engine}"
    assert all(count > 0 for count in engine.values()), (
        f"a population has no residual at all, so agreement is trivial: {engine}"
    )
    tally = {pop.value: stored[pop] for pop in Population}
    print(f"\n  per population, store == engine: {tally}")


CHECKPOINTS = (1, 12, LAST_DELIVERY_DAY)
"""Three checkpoints instead of 24 runs.

Each run reconciles the CUMULATIVE dataset, so 24 of them would cost more than
the whole suite's budget. What matters is the direction and the endpoint.
"""


@pytest.fixture(scope="module")
def incremental_run(
    config: AppConfig, tmp_path_factory: pytest.TempPathFactory
) -> tuple[ExceptionStore, dict[int, dict[str, int]]]:
    """A genuine multi-day store, plus residual counts at each checkpoint.

    Shared because it is the only fixture where CONFIRMED exceptions coexist
    with residual ones - which is exactly what a single-shot run cannot
    produce, and exactly what the widened-filter guard needs to detect.
    """
    store = ExceptionStore(tmp_path_factory.mktemp("m6inc") / "inc.db")
    counts: dict[int, dict[str, int]] = {}
    for day in CHECKPOINTS:
        store.run_day(day, DATA, config, LATE_AS_OF)
        counts[day] = {
            pop.value: len(store.get_queue(status_filter=RESIDUAL_STATES, population=pop))
            for pop in Population
        }
    return store, counts


def test_the_residual_shrinks_as_evidence_arrives(
    incremental_run: tuple[ExceptionStore, dict[int, dict[str, int]]], config: AppConfig
) -> None:
    """The incremental story, measured at checkpoints rather than every day."""
    _store, counts = incremental_run
    batch_residual = [counts[day][Population.B_BATCH_LINK.value] for day in CHECKPOINTS]
    assert batch_residual[-1] < batch_residual[0], (
        f"Population B did not shrink as bank files arrived: {batch_residual}"
    )
    # NOT asserted monotonic, and the first version of this test wrongly was.
    # The residual is a QUEUE, not a burn-down: day 12 delivers batches whose
    # credit is still days away, so arrivals can outpace departures at any
    # checkpoint. Realised here as [3, 6, 2] - up, then down. What must hold is
    # the ENDPOINT, because that is the claim: once every file has landed, the
    # incremental path agrees with the batch engine.
    final_engine = sum(
        1
        for link in run(
            ExceptionStore().cumulative_dataset(LAST_DELIVERY_DAY, DATA, config),
            config,
            LATE_AS_OF,
        ).batch_links
        if link.status is not ExceptionStatus.CONFIRMED
    )
    assert batch_residual[-1] == final_engine, (
        f"after every file arrived the store holds {batch_residual[-1]} open batches "
        f"and the engine says {final_engine}"
    )
    for day, tallies in counts.items():
        print(f"\n  day {day:>2}: A={tallies['A']:>4} B={tallies['B']:>3} C={tallies['C']:>2}")


def test_all_three_populations_persist(full_run: ExceptionStore) -> None:
    """D11. Three denominators, three sets of rows, never merged.

    Population B is the one that matters for the demo: a batch whose credit
    has not arrived is exactly the exception a later day's bank file closes,
    and if only A persisted there would be nothing to show closing.
    """
    # Every status here, deliberately: this test is about PERSISTENCE, and an
    # exception that was opened and later confirmed is still persisted.
    counts = {
        pop.value: len(full_run.get_queue(ALL_STATUSES, population=pop)) for pop in Population
    }
    for population, count in counts.items():
        assert count > 0, f"population {population} persisted nothing"

    subjects = {
        pop.value: {exc.exception_id for exc in full_run.get_queue(ALL_STATUSES, population=pop)}
        for pop in Population
    }
    assert not subjects["A"] & subjects["B"], "populations A and B share exception ids"
    assert not subjects["B"] & subjects["C"], "populations B and C share exception ids"
    print(f"\n  persisted per population: {counts}")


def test_population_b_holds_the_batches_whose_credit_never_arrived(
    full_run: ExceptionStore,
) -> None:
    """Realised count, printed - not a literal from the brief."""
    batch_exceptions = full_run.get_queue(
        status_filter=RESIDUAL_STATES, population=Population.B_BATCH_LINK
    )
    categories = sorted({exc.category for exc in batch_exceptions})
    assert batch_exceptions, "no Population B exceptions at all"
    for exception in batch_exceptions:
        assert exception.amount > 0, (
            f"{exception.exception_id} has no money basis - Population B's basis "
            "is batch_net_total and exists whether or not a credit arrived"
        )
    print(f"\n  {len(batch_exceptions)} open batch exceptions, categories {categories}")


def test_pending_evidence_is_not_flattened_to_open(config: AppConfig, tmp_path: Path) -> None:
    """PENDING_EVIDENCE vs MISSING_VS_LATE_CREDIT, preserved across days.

    M3 already draws this distinction - not yet due versus past due and absent
    - and the store must carry it rather than collapsing both to OPEN. An
    early as_of is used so batches genuinely are not yet due.
    """
    with ExceptionStore(tmp_path / "pending.db") as store:
        store.run_day(3, DATA, config, EARLY_AS_OF)
        pending = store.get_queue(status_filter=ExceptionStatus.PENDING_EVIDENCE)
        open_now = store.get_queue(status_filter=ExceptionStatus.OPEN)

    assert pending, "nothing is PENDING_EVIDENCE, so the distinction is untested here"
    assert open_now, "nothing is OPEN either - the run did not produce a residual set"
    statuses = {exc.status for exc in pending}
    assert statuses == {ExceptionStatus.PENDING_EVIDENCE}, statuses
    print(f"\n  {len(pending)} PENDING_EVIDENCE and {len(open_now)} OPEN, held apart")


@pytest.mark.charter_guard
def test_as_of_changes_behaviour(config: AppConfig, tmp_path: Path) -> None:
    """A parameter that is never READ is indistinguishable from a clock (D2).

    D2's AST scanner sees no clock call in either case, so it cannot catch
    this. The only way to know as_of matters is to vary it and watch the
    answer change.
    """
    outcomes: dict[str, dict[str, int]] = {}
    for label, as_of in (("early", EARLY_AS_OF), ("late", LATE_AS_OF)):
        with ExceptionStore(tmp_path / f"{label}.db") as store:
            store.run_day(5, DATA, config, as_of)
            outcomes[label] = {
                status.value: len(store.get_queue(status_filter=status))
                for status in ExceptionStatus
            }

    assert outcomes["early"] != outcomes["late"], (
        f"as_of made no difference: {outcomes['early']}. The parameter is "
        "threaded through every signature and never read, which is behaviourally "
        "identical to reading a clock."
    )
    early_pending = outcomes["early"][ExceptionStatus.PENDING_EVIDENCE.value]
    late_pending = outcomes["late"][ExceptionStatus.PENDING_EVIDENCE.value]
    assert early_pending > late_pending, (
        f"an EARLIER as_of should leave MORE not-yet-due evidence pending, got "
        f"early={early_pending} late={late_pending}"
    )
    print(f"\n  early {outcomes['early']}\n  late  {outcomes['late']}")


def test_run_day_ingests_reevaluates_and_confirms_across_days(
    config: AppConfig, tmp_path: Path
) -> None:
    """THE DEMO. An exception opens on an early day and closes on a later one.

    Asserted on realised counts, printed, and on the audit trail of a specific
    exception - so "it confirmed something" cannot be satisfied by a run that
    opened and confirmed the same thing on the same day.
    """
    days: list[DayRun] = []
    with ExceptionStore(tmp_path / "demo.db") as store:
        for day in range(1, 6):
            days.append(store.run_day(day, DATA, config, LATE_AS_OF))

        confirmed_later = [
            exc
            for exc in store.get_queue(status_filter=ExceptionStatus.CONFIRMED)
            if exc.confirmed_day is not None and exc.confirmed_day > exc.first_seen_day
        ]
        assert confirmed_later, (
            "nothing opened on one day and confirmed on a later one - the "
            "incremental story is untested"
        )
        example = confirmed_later[0]
        trail = [(entry.arrival_day, entry.from_status, entry.to_status) for entry in example.audit]
        assert trail[0][1] is None
        assert trail[-1][2] is ExceptionStatus.CONFIRMED
        assert example.closed_day is None, "re-evaluation emitted an accounting action"

    for day_run in days:  # not `run`: that name is the engine entry point here
        print(
            f"\n  day {day_run.arrival_day}: {len(day_run.ingested)} files "
            f"({day_run.skipped_count} skipped), +{len(day_run.newly_opened)} opened, "
            f"{len(day_run.newly_confirmed)} confirmed, {len(day_run.still_residual)} residual"
        )
    print(f"  {len(confirmed_later)} confirmed on a LATER day than first seen")
    print(
        f"  example {example.exception_id}: seen day {example.first_seen_day}, "
        f"confirmed day {example.confirmed_day}, trail {len(trail)} entries"
    )


def test_rerunning_a_day_is_a_no_op(config: AppConfig, tmp_path: Path) -> None:
    """Replay safety, end to end. The second run ingests nothing new.

    This is what the idempotency key buys: a re-run after a crash must not
    double-count resolutions or append a second audit trail.
    """
    with ExceptionStore(tmp_path / "replay.db") as store:
        first = store.run_day(2, DATA, config, LATE_AS_OF)
        queue_after_first = [
            (exc.exception_id, exc.status.value) for exc in store.get_queue(ALL_STATUSES)
        ]
        audit_after_first = sum(len(store.get_audit(exc_id)) for exc_id, _ in queue_after_first)

        second = store.run_day(2, DATA, config, LATE_AS_OF)
        queue_after_second = [
            (exc.exception_id, exc.status.value) for exc in store.get_queue(ALL_STATUSES)
        ]
        audit_after_second = sum(len(store.get_audit(exc_id)) for exc_id, _ in queue_after_second)

    assert first.skipped_count == 0 and second.skipped_count == len(second.ingested), (
        f"re-run skipped {second.skipped_count} of {len(second.ingested)} files"
    )
    assert second.newly_opened == (), f"re-run opened {len(second.newly_opened)} exceptions"
    assert queue_after_first == queue_after_second, "the queue changed on replay"
    assert audit_after_first == audit_after_second, (
        f"replay appended {audit_after_second - audit_after_first} audit entries"
    )
    print(
        f"\n  replay: {second.skipped_count}/{len(second.ingested)} files skipped, "
        f"0 opened, {audit_after_second} audit entries unchanged"
    )


@pytest.mark.boundary_refusal
def test_run_day_refuses_a_day_with_no_files(config: AppConfig, tmp_path: Path) -> None:
    """FAULT INJECTION. A day that delivered nothing is a delivery failure.

    A quiet day still delivers header-only tables, so zero files means the
    drop did not happen - not that nothing happened.
    """
    with ExceptionStore(tmp_path / "empty.db") as store, pytest.raises(MissingFileError) as caught:
        store.run_day(99, DATA, config, LATE_AS_OF)
    assert "day99" in str(caught.value)
    print(f"\n  {str(caught.value)[:80]}")


def test_watermarks_record_arrival_ordering(config: AppConfig, tmp_path: Path) -> None:
    with ExceptionStore(tmp_path / "wm.db") as store:
        assert store.get_watermark("ingest") is None
        store.run_day(1, DATA, config, LATE_AS_OF)
        first = store.get_watermark("ingest")
        store.run_day(2, DATA, config, LATE_AS_OF)
        second = store.get_watermark("ingest")

    assert first is not None and second is not None
    assert second[0] > first[0], (first, second)
    print(f"\n  watermark {first} -> {second}")


def test_the_store_survives_being_reopened(config: AppConfig, tmp_path: Path) -> None:
    """State is state. Closing and reopening must not lose or change it."""
    path = tmp_path / "persist.db"
    with ExceptionStore(path) as store:
        store.run_day(1, DATA, config, LATE_AS_OF)
        before = [
            (exc.exception_id, exc.status.value, str(exc.amount))
            for exc in store.get_queue(ALL_STATUSES)
        ]

    with ExceptionStore(path) as reopened:
        after = [
            (exc.exception_id, exc.status.value, str(exc.amount))
            for exc in reopened.get_queue(ALL_STATUSES)
        ]
        rerun = reopened.run_day(1, DATA, config, LATE_AS_OF)

    assert before == after, "state changed across a close/open cycle"
    assert rerun.skipped_count == len(rerun.ingested), "a reopened store re-ingested files"
    print(
        f"\n  {len(before)} exceptions survived close/open; re-run skipped all "
        f"{rerun.skipped_count} files"
    )


# ===========================================================================
# M6 review fixes
# ===========================================================================


@pytest.mark.boundary_refusal
def test_get_queue_requires_an_explicit_status_filter(store: ExceptionStore) -> None:
    """No default. Every call site names what it is asking for.

    THE DEFECT THIS REMOVES, restated because the fix is only obvious in
    hindsight: a permissive default gave a convergence test the answer to a
    question it had not asked - "all statuses" while it meant "residual" - and
    it passed anyway, because a single-shot run confirms nothing and the two
    counts coincide. Only a real multi-day run separated them, at which point
    it reported Population B as 11 while the store and engine both said 2.
    """
    _seed(store, exception_id="q1")
    with pytest.raises(TypeError):
        store.get_queue()  # type: ignore[call-arg]
    with pytest.raises(StoreError, match="explicit status_filter"):
        store.get_queue(None)  # type: ignore[arg-type]

    assert len(store.get_queue(ALL_STATUSES)) == 1
    assert len(store.get_queue(RESIDUAL_STATES)) == 1
    assert store.get_queue(ExceptionStatus.CLOSED) == []
    print("\n  no-arg raises TypeError, None raises StoreError, ALL_STATUSES works")


@pytest.mark.charter_guard
def test_the_convergence_check_fails_if_the_filter_is_widened(
    incremental_run: tuple[ExceptionStore, dict[int, dict[str, int]]], config: AppConfig
) -> None:
    """FAULT INJECTION for fix 3, against the ORIGINAL BUG.

    Runs the convergence comparison twice over the same store: once with
    RESIDUAL_STATES (correct) and once with ALL_STATUSES (the old default).
    The first must agree with the engine and the second must not.

    A MULTI-DAY store, not the single-shot one - and the first version of this
    guard used the single-shot fixture and failed with "widening the filter
    gives the same answer (2)". That failure was the point restated: a
    single-shot run confirms nothing, so all-statuses and residual COINCIDE,
    and a store where they coincide cannot detect the defect. Only a store
    that has confirmed something across days separates them (11 vs 2).
    """
    full_run, _counts = incremental_run
    dataset = full_run.cumulative_dataset(LAST_DELIVERY_DAY, DATA, config)
    result = run(dataset, config, LATE_AS_OF)
    engine_b = sum(1 for link in result.batch_links if link.status is not ExceptionStatus.CONFIRMED)

    correct = len(
        full_run.get_queue(status_filter=RESIDUAL_STATES, population=Population.B_BATCH_LINK)
    )
    widened = len(
        full_run.get_queue(status_filter=ALL_STATUSES, population=Population.B_BATCH_LINK)
    )

    assert correct == engine_b, (
        f"residual filter disagrees with the engine: {correct} vs {engine_b}"
    )
    assert widened != engine_b, (
        f"widening the filter gives the same answer ({widened}) on this store, so "
        "the test cannot detect the defect it exists to catch"
    )
    print(f"\n  engine {engine_b}; residual filter {correct}; widened filter {widened}")


@pytest.mark.charter_guard
def test_run_day_derives_as_of_and_the_override_still_changes_the_answer(
    config: AppConfig, tmp_path: Path
) -> None:
    """Fix 2, BOTH halves. The default is right AND the parameter is read.

    (a) derived default  -> a not-yet-due batch is PENDING_EVIDENCE
    (b) far-future override -> the same batch is MISSING_VS_LATE_CREDIT

    Both matter. (a) alone would be satisfied by hard-coding a date; (b) alone
    is the R27 property - a parameter threaded through every signature and
    never read is behaviourally identical to reading a clock, and D2's scanner
    sees no clock call either way.

    The SAME batch is followed through both runs, not just aggregate counts: a
    count could match by coincidence across two different sets of batches.
    """
    day = 3
    derived = as_of_for_arrival_day(day, config)
    assert derived == config.calendar.window_start + timedelta(days=day - 1)

    with ExceptionStore(tmp_path / "derived.db") as store:
        store.run_day(day, DATA, config)  # no as_of: derived
        pending = store.get_queue(
            status_filter=ExceptionStatus.PENDING_EVIDENCE, population=Population.B_BATCH_LINK
        )
    assert pending, "the derived as_of produced no PENDING_EVIDENCE batch at all"
    subject = pending[0].exception_id

    with ExceptionStore(tmp_path / "override.db") as store:
        store.run_day(day, DATA, config, as_of=LATE_AS_OF)
        same_batch = store.get_exception(subject)
        overridden_pending = store.get_queue(
            status_filter=ExceptionStatus.PENDING_EVIDENCE, population=Population.B_BATCH_LINK
        )

    assert same_batch is not None, "the same batch is absent under the override"
    assert same_batch.status is not ExceptionStatus.PENDING_EVIDENCE, (
        "the far-future override left the batch PENDING_EVIDENCE - as_of is not being read"
    )
    assert same_batch.category == "MISSING_VS_LATE_CREDIT", same_batch.category
    assert not overridden_pending, (
        f"{len(overridden_pending)} batches are still PENDING_EVIDENCE under a "
        "far-future as_of, where nothing can be not-yet-due"
    )
    print(
        f"\n  day {day} derives {derived.isoformat()}: {len(pending)} PENDING_EVIDENCE batches\n"
        f"  same batch under as_of={LATE_AS_OF.isoformat()}: "
        f"{same_batch.status.value} / {same_batch.category}"
    )


@pytest.mark.boundary_refusal
def test_the_derived_as_of_refuses_a_day_outside_the_window(config: AppConfig) -> None:
    """FAULT INJECTION for the derivation. D13: every date is inside the window."""
    with pytest.raises(StoreError, match="1-indexed"):
        as_of_for_arrival_day(0, config)
    span = (config.calendar.window_end - config.calendar.window_start).days
    assert as_of_for_arrival_day(span + 1, config) == config.calendar.window_end
    with pytest.raises(StoreError, match="simulation window"):
        as_of_for_arrival_day(span + 2, config)
    print(f"\n  window spans {span + 1} days; day 0 and day {span + 2} both refused")

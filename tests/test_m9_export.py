"""M9 - the Tally exporter, and everything CLOSED means. Mostly not built yet.

WHY THIS FILE EXISTS BEFORE THE MODULE DOES. CLOSED is the one status no other
milestone may write: it means the accounting action was emitted, and only the
exporter emits one. The assertions about it were living in the M8 UI suite,
where they read as questions about how a pill renders. They are not. They are
questions about whether anything in this project can claim money moved, and
they belong beside the module that would make that claim.

WHAT IS ASSERTED TODAY. That nothing is CLOSED, that nothing but the exporter
can write CLOSED, and that the two states remain visibly distinct wherever they
are rendered. Those are checkable now and each would silently stop being true
if someone wired a shortcut.

WHAT IS NOT ASSERTED. Anything about XML, schemas or idempotency keys. The
exporter is a docstring-only stub; a test that passed over an unbuilt module
would be worse than no test, because a green run reads as evidence.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from eval.run_eval import load_days
from settlesense.config import AppConfig, load_config
from settlesense.exceptions.store import (
    ALL_STATUSES,
    LEGAL_TRANSITIONS,
    ExceptionStore,
)
from settlesense.types import AuditActor, ExceptionStatus
from settlesense.ui.queue import STATUS_STYLES

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "dev"
EXPORTER = REPO / "settlesense" / "export" / "tally.py"
CHECKPOINTS = (1, 12, 24)
AS_OF = date(2026, 11, 30)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture(scope="module")
def store(config: AppConfig, tmp_path_factory: pytest.TempPathFactory) -> ExceptionStore:
    built = ExceptionStore(tmp_path_factory.mktemp("m9") / "state.db")
    for day in CHECKPOINTS:
        built.run_day(day, DATA, config)
    return built


# ===========================================================================
# 1. Nothing is CLOSED, because nothing can close anything
# ===========================================================================


def test_the_exporter_is_not_built_yet() -> None:
    """The premise every other test here rests on, asserted rather than assumed.

    If this starts failing, the tests below are no longer measuring an absence
    and the file needs real export assertions instead.
    """
    source = EXPORTER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert not functions, f"tally.py now defines {functions}; write real M9 tests"
    assert "Not yet implemented" in source, source[:120]
    print(f"\n  {EXPORTER.relative_to(REPO)} is still a stub: {len(source)} bytes, 0 functions")


def test_nothing_in_a_real_run_is_closed(store: ExceptionStore, config: AppConfig) -> None:
    """Over a full multi-day run: many CONFIRMED, zero CLOSED.

    Not a vacuous check - the run confirms hundreds of exceptions, so the
    absence of CLOSED is a decision the store makes rather than an empty set.
    """
    rows = store.get_queue(ALL_STATUSES)
    statuses = {
        status: sum(1 for row in rows if row.status is status) for status in ExceptionStatus
    }
    assert statuses[ExceptionStatus.CONFIRMED] > 0, (
        "nothing was confirmed either, so 'nothing is CLOSED' proves nothing"
    )
    assert statuses[ExceptionStatus.CLOSED] == 0, (
        f"{statuses[ExceptionStatus.CLOSED]} exceptions are CLOSED, but no exporter "
        "exists to have emitted an accounting action for them"
    )
    assert all(row.closed_day is None for row in rows), "a closed_day is set on some row"
    print(f"\n  {statuses[ExceptionStatus.CONFIRMED]} CONFIRMED, 0 CLOSED, no closed_day set")


@pytest.mark.charter_guard
def test_closed_has_exactly_one_predecessor_and_one_writer() -> None:
    """SDD 3. CONFIRMED is the only way in; the EXPORTER is the only writer.

    Both halves matter. A second predecessor would let an unexplained exception
    be actioned; a second writer would let a reconciliation path claim money
    moved.
    """
    predecessors = [
        source.value
        for source, targets in LEGAL_TRANSITIONS.items()
        if ExceptionStatus.CLOSED in targets
    ]
    assert predecessors == ["CONFIRMED"], predecessors
    assert LEGAL_TRANSITIONS[ExceptionStatus.CLOSED] == frozenset(), "CLOSED is not terminal"

    store_source = (REPO / "settlesense" / "exceptions" / "store.py").read_text(encoding="utf-8")
    assert "AuditActor.EXPORTER," in store_source, (
        "close_exception no longer hard-codes the EXPORTER actor"
    )
    assert AuditActor.EXPORTER in set(AuditActor), AuditActor
    print(f"\n  predecessors {predecessors}; terminal; actor hard-coded to EXPORTER")


@pytest.mark.charter_guard
def test_confirmed_and_closed_are_visibly_distinct() -> None:
    """MOVED HERE FROM THE M8 SUITE. Different labels AND different colours.

    CONFIRMED means explained; CLOSED means the accounting action was emitted.
    A reader who cannot tell them apart at a glance cannot tell whether money
    moved, so the distinction is carried twice - by text and by colour - and
    neither alone is relied on.
    """
    confirmed = STATUS_STYLES[ExceptionStatus.CONFIRMED]
    closed = STATUS_STYLES[ExceptionStatus.CLOSED]
    assert confirmed.label != closed.label, (confirmed.label, closed.label)
    assert confirmed.colour != closed.colour, "the two greens are identical"
    assert confirmed.background != closed.background, "the two backgrounds are identical"
    assert "✓" in closed.label, "CLOSED carries no check mark"
    assert len(STATUS_STYLES) == len(ExceptionStatus) == 6, len(STATUS_STYLES)
    print(f"\n  CONFIRMED {confirmed.label!r} vs CLOSED {closed.label!r}, distinct colours")


@pytest.mark.charter_guard
def test_no_module_outside_the_store_writes_closed() -> None:
    """A grep-by-AST for anyone assigning or transitioning to CLOSED.

    `store.py` legitimately names it - it owns the transition table and
    `close_exception`. `ui/queue.py` names it in a style map, which renders
    rather than writes. Anything else is a shortcut around the exporter.
    """
    permitted = {
        "settlesense/exceptions/store.py",
        "settlesense/ui/queue.py",
        "settlesense/types.py",
    }
    offenders: list[str] = []
    for path in sorted((REPO / "settlesense").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel in permitted:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr == "CLOSED":
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, f"CLOSED is referenced outside the store and the renderer: {offenders}"
    print(f"\n  CLOSED referenced only in {sorted(permitted)}")


def test_metrics_read_confirmed_never_closed() -> None:
    """SDD 3: accuracy metrics read CONFIRMED. Reading CLOSED would measure
    what was exported rather than what was explained, and today would report
    zero for a run that explained thousands."""
    metrics = (REPO / "eval" / "metrics.py").read_text(encoding="utf-8")
    assert "ExceptionStatus.CONFIRMED" in metrics, "metrics do not read CONFIRMED at all"
    assert "ExceptionStatus.CLOSED" not in metrics, (
        "a metric reads CLOSED; accuracy would then measure export progress"
    )
    print("\n  eval/metrics.py reads CONFIRMED and never CLOSED")


def test_the_ui_can_render_closed_even_though_nothing_is(
    store: ExceptionStore, config: AppConfig, tmp_path: Path
) -> None:
    """The renderer must be READY for a status it has never seen.

    A style map missing CLOSED would raise the first time the exporter ran,
    which is the worst possible moment to discover it. Exercised by forcing one
    exception through the legal path in a throwaway copy of the store.
    """
    from settlesense.ui.queue import build_rows

    dataset: Any = load_days(DATA, config)
    del dataset  # loaded only to prove the fixture path is intact

    scratch = ExceptionStore(tmp_path / "scratch.db")
    scratch.run_day(1, DATA, config)

    # An OPEN row, not just any row. PENDING_EVIDENCE cannot go straight to
    # CONFIRMED - the lifecycle sends it back through OPEN first - and picking
    # blindly hit exactly that, which is the transition guard working.
    candidates = [
        row
        for row in scratch.get_queue(ALL_STATUSES)
        if row.status in {ExceptionStatus.OPEN, ExceptionStatus.CONFIRMED}
    ]
    assert candidates, "day 1 produced nothing that can legally reach CONFIRMED"
    target = candidates[0]
    subject = target.exception_id
    if target.status is ExceptionStatus.OPEN:
        scratch.mark_status(
            subject, ExceptionStatus.CONFIRMED, AuditActor.HUMAN, "for this test", 1
        )

    closed = scratch.close_exception(subject, arrival_day=1, note="exported by this test")
    assert closed.status is ExceptionStatus.CLOSED
    rendered = [row for row in build_rows(scratch) if row.exception_id == subject]
    assert rendered and rendered[0].style.label == STATUS_STYLES[ExceptionStatus.CLOSED].label
    print(f"\n  a forced CLOSED renders as {rendered[0].style.label!r}")

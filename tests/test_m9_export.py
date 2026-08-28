"""M9 - the Tally exporter, and everything CLOSED means.

WHY THIS FILE EXISTED BEFORE THE MODULE DID. CLOSED is the one status no other
milestone may write: it means the accounting action was emitted, and only the
exporter emits one. The assertions about it were living in the M8 UI suite,
where they read as questions about how a pill renders. They are not. They are
questions about whether anything in this project can claim money moved, and
they belong beside the module that would make that claim.

WHAT THE EXPORTER TURNED OUT TO NEED, which the build prompt did not say. An
exception stores the category recorded AT DETECTION, and 274 of this store's
283 confirmed rows were opened as UNEXPLAINED and explained by a later day's
files. Posting on `exception.category` would have written 274 vouchers for a
variance nobody claimed existed - and the first run of the exporter refused,
because UNEXPLAINED deliberately has no ledger. The resolving category is an
argument, and most confirmed exceptions have NO entry at all: their variance
cleared. 283 confirmed produce 16 vouchers and 267 clearances here.

EVERY EXPECTED VALUE IS READ, NEVER TYPED. The provenance comes out of the
committed eval artifact; totals come out of the parsed XML rather than the
objects that built it; counts are realised and printed. The dev false-match
rate is 0.000000 today, and typing that here would keep passing in exactly the
circumstance where this suite has a job to do.
"""

from __future__ import annotations

import ast
import json
import re
import socket
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from lxml import etree

from eval.run_eval import load_days
from eval.run_export import provenance_from_results
from settlesense.config import AppConfig, load_config
from settlesense.exceptions.store import (
    ALL_STATUSES,
    LEGAL_TRANSITIONS,
    ExceptionStore,
)
from settlesense.exceptions.taxonomy import VARIANCE_CATEGORIES, VarianceCategory
from settlesense.export.tally import (
    BASIS,
    LEDGERS,
    ExportError,
    ExportProvenance,
    TallyBatch,
    build_batch,
    close_exported,
    to_xml,
    validate,
    write_dry_run,
)
from settlesense.types import AuditActor, Exception_, ExceptionStatus
from settlesense.ui.queue import STATUS_STYLES, current_categories

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "dev"
EXPORTER = REPO / "settlesense" / "export" / "tally.py"
CHECKPOINTS = (1, 12, 24)
AS_OF = date(2026, 11, 30)
BATCH_DATE = date(2026, 11, 30)


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


# ===========================================================================
# 2. The exporter itself, now that there is one
#
# EVERY EXPECTED VALUE IS READ, NEVER TYPED. The provenance comes out of the
# committed eval artifact; the voucher totals come out of the parsed XML rather
# than the objects that built it; the counts are realised and printed. A test
# that compared against a figure copied from a brief would keep passing after
# the thing it describes stopped being true.
# ===========================================================================

RESULTS = REPO / "reports" / "eval" / "results.json"
RATE_KEY = "residual_false_match_rate_case_count"
POPULATION_A_KEY = "population_a_case_count_denominator"


@pytest.fixture(scope="module")
def dataset(config: AppConfig) -> Any:
    return load_days(DATA, config)


@pytest.fixture(scope="module")
def provenance() -> ExportProvenance:
    """Built from the committed artifact, exactly as run_export.py builds it."""
    return provenance_from_results(RESULTS, "dev")


@pytest.fixture(scope="module")
def resolved(store: ExceptionStore, dataset: Any, config: AppConfig) -> dict[str, str | None]:
    by_subject = current_categories(dataset, config, AS_OF)
    return {
        row.exception_id: by_subject.get(store.subject_id(row.exception_id) or "")
        for row in store.get_queue(ALL_STATUSES)
        if store.subject_id(row.exception_id) in by_subject
    }


@pytest.fixture(scope="module")
def confirmed(store: ExceptionStore) -> tuple[Exception_, ...]:
    return tuple(
        row for row in store.get_queue(ALL_STATUSES) if row.status is ExceptionStatus.CONFIRMED
    )


@pytest.fixture(scope="module")
def batch(
    confirmed: tuple[Exception_, ...],
    resolved: dict[str, str | None],
    provenance: ExportProvenance,
) -> TallyBatch:
    return build_batch(confirmed, resolved, provenance, BATCH_DATE)


def test_1_the_generated_xml_validates_against_the_bundled_xsd(batch: TallyBatch) -> None:
    """And the schema is load-bearing: a broken document must FAIL it.

    `to_xml` validates before returning, so a passing call is already the
    assertion. The planted half is what makes it mean anything - without it a
    schema that accepted everything would look identical from here.
    """
    xml = to_xml(batch)
    validate(xml)

    broken = xml.replace("<AMOUNT>", "<AMOUNTS>", 1).replace("</AMOUNT>", "</AMOUNTS>", 1)
    with pytest.raises(ExportError, match="does not validate"):
        validate(broken)
    print(f"\n  {len(xml):,} bytes validate; a renamed element is refused")


def test_the_schema_rejects_money_that_is_not_two_places(batch: TallyBatch) -> None:
    """Requirement 9, asserted from the SCHEMA rather than from the formatter.

    Three places, no places and scientific notation are each rejected. The
    formatter and the schema are two independent statements of the same rule,
    which is the point: a change to one is caught by the other.
    """
    xml = to_xml(batch)
    for bad in ("1234.567", "1234", "1.23E+3"):
        mangled = re.sub(r"<AMOUNT>-?[\d.]+</AMOUNT>", f"<AMOUNT>{bad}</AMOUNT>", xml, count=1)
        with pytest.raises(ExportError, match="does not validate"):
            validate(mangled)

    amounts = re.findall(r"<AMOUNT>(-?[\d.]+)</AMOUNT>", xml)
    assert amounts, "no amounts in the document"
    assert all(re.fullmatch(r"-?\d+\.\d{2}", value) for value in amounts), amounts
    assert not any("E" in value or "e" in value for value in amounts)
    print(f"\n  {len(amounts)} amounts, all exactly 2dp; 3dp/0dp/scientific all refused")


def test_2_and_3_the_key_is_a_function_of_the_confirmed_set(
    confirmed: tuple[Exception_, ...],
    resolved: dict[str, str | None],
    provenance: ExportProvenance,
) -> None:
    """Same set -> same key. Different set -> different key."""
    again = build_batch(confirmed, resolved, provenance, BATCH_DATE)
    original = build_batch(confirmed, resolved, provenance, BATCH_DATE)
    assert again.idempotency_key == original.idempotency_key

    postable = [row for row in confirmed if resolved.get(row.exception_id) is not None]
    assert len(postable) > 1, "cannot drop one and still have a batch"
    fewer = build_batch(
        tuple(row for row in confirmed if row is not postable[0]), resolved, provenance, BATCH_DATE
    )
    assert fewer.idempotency_key != original.idempotency_key
    print(
        f"\n  {len(original.lines)} vouchers -> {original.idempotency_key[:16]}; "
        f"{len(fewer.lines)} vouchers -> {fewer.idempotency_key[:16]}"
    )


def test_20_the_key_changes_when_config_hash_changes_and_the_set_does_not(
    confirmed: tuple[Exception_, ...],
    resolved: dict[str, str | None],
    provenance: ExportProvenance,
) -> None:
    """THE AMENDMENT'S POINT. SDD 4.7's key omitted config_hash.

    Two batches over the identical confirmed set under different configs
    produce different DOCUMENTS - the header states the config hash - so a key
    that could not tell them apart would name both with the same filename and
    the second would overwrite the first. Every provenance field is hashed;
    each is varied here one at a time.
    """
    base = build_batch(confirmed, resolved, provenance, BATCH_DATE)
    variants = {
        "config_hash": replace(provenance, config_hash=provenance.config_hash + "0"),
        "seed": replace(provenance, seed=provenance.seed + 1),
        "dataset": replace(provenance, dataset=provenance.dataset + "-copy"),
        "rate": replace(provenance, residual_false_match_rate=Decimal("0.010456")),
    }
    keys = {"base": base.idempotency_key}
    for name, variant in variants.items():
        other = build_batch(confirmed, resolved, variant, BATCH_DATE)
        assert other.exception_ids == base.exception_ids, "the confirmed set moved; wrong control"
        assert other.idempotency_key != base.idempotency_key, (
            f"changing {name} left the key unchanged, so two different documents "
            "would be written to one filename"
        )
        assert other.filename != base.filename
        keys[name] = other.idempotency_key
    assert len(set(keys.values())) == len(keys), keys
    print(f"\n  identical confirmed set, {len(keys)} distinct keys across {sorted(variants)}")


def test_4_and_5_a_non_confirmed_exception_raises(
    store: ExceptionStore,
    resolved: dict[str, str | None],
    provenance: ExportProvenance,
    confirmed: tuple[Exception_, ...],
) -> None:
    """ABSTAINED and OPEN both refuse. Realised rows, not constructed ones."""
    rows = store.get_queue(ALL_STATUSES)
    for status in (ExceptionStatus.OPEN, ExceptionStatus.PENDING_EVIDENCE):
        intruders = [row for row in rows if row.status is status]
        if not intruders:
            continue
        with pytest.raises(ExportError, match="not CONFIRMED"):
            build_batch((*confirmed[:2], intruders[0]), resolved, provenance, BATCH_DATE)

    abstained = replace(confirmed[0], status=ExceptionStatus.ABSTAINED)
    with pytest.raises(ExportError, match="not CONFIRMED"):
        build_batch((abstained,), resolved, provenance, BATCH_DATE)
    print("\n  OPEN, PENDING_EVIDENCE and ABSTAINED are each refused")


def test_6_exporting_twice_writes_one_file_with_identical_bytes(
    batch: TallyBatch, tmp_path: Path
) -> None:
    first = write_dry_run(batch, tmp_path)
    before = first.read_bytes()
    second = write_dry_run(batch, tmp_path)
    assert first == second, "a second export minted a second filename"
    assert second.read_bytes() == before, "the same batch produced different bytes"
    assert sorted(p.name for p in tmp_path.iterdir()) == [batch.filename]
    assert batch.idempotency_key[:16] in batch.filename
    print(f"\n  two exports, one file: {batch.filename} ({len(before):,} bytes, byte-identical)")


def test_7_debits_and_credits_balance_exactly_in_the_document(batch: TallyBatch) -> None:
    """Summed from the PARSED XML, not from the objects that built it.

    Balancing the dataclasses would restate how they were constructed. The
    document is what an accountant receives, so the document is what is
    checked - and every voucher must balance on its own, not merely in total,
    because two offsetting errors sum to zero just as neatly as no errors.
    """
    root = etree.fromstring(to_xml(batch).encode("utf-8"))
    vouchers = root.findall(".//VOUCHER")
    assert len(vouchers) == len(batch.lines), (len(vouchers), len(batch.lines))

    debits = credits = Decimal("0")
    for voucher in vouchers:
        legs = [Decimal(node.text or "0") for node in voucher.findall(".//AMOUNT")]
        assert len(legs) == 2, f"{len(legs)} legs on one voucher"
        assert sum(legs) == Decimal("0"), (voucher.findtext("VOUCHERNUMBER"), legs)
        debits += -min(legs)
        credits += max(legs)

    assert debits == credits == batch.total_debits, (debits, credits, batch.total_debits)
    print(f"\n  {len(vouchers)} vouchers, each balancing; debits {debits:,.2f} = credits")


@pytest.mark.charter_guard
def test_8_a_dry_run_performs_no_network_io(
    batch: TallyBatch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """socket.socket booby-trapped, then a REAL export driven through it.

    Proven by the export succeeding rather than inferred from reading the
    source. `create_connection` is trapped too - a library that reached the
    network through it would not construct a socket by name.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the exporter opened a socket during a dry run")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    path = write_dry_run(batch, tmp_path)
    assert path.is_file() and path.stat().st_size > 0

    with pytest.raises(AssertionError, match="opened a socket"):
        socket.socket()
    print(f"\n  {path.name} written with socket.socket() booby-trapped; the trap fires")


# ===========================================================================
# 3. The provenance header - the 2026-08-28 amendment
# ===========================================================================


@pytest.mark.charter_guard
def test_18_the_header_carries_the_real_measured_rate(batch: TallyBatch) -> None:
    """Read out of reports/eval/results.json, compared against the document.

    NOT A LITERAL. The dev rate is 0.000000 today; typing that here would keep
    passing if the harness started emitting something else, which is the one
    circumstance where this test has a job to do.
    """
    expected = json.loads(RESULTS.read_text(encoding="utf-8"))
    header = etree.fromstring(to_xml(batch).encode("utf-8")).find(".//SETTLESENSE.PROVENANCE")
    assert header is not None, "no provenance element in the document"

    assert header.get("seed") == str(expected["seed"])
    assert header.get("configHash") == expected["config_hash"]
    assert header.get("residualFalseMatchRate") == expected[POPULATION_A_KEY][RATE_KEY]
    assert header.get("dataset") == "dev"
    assert header.get("voucherCount") == str(len(batch.lines))
    assert header.get("clearedCount") == str(len(batch.cleared))
    print(
        f"\n  header reads seed={header.get('seed')} config={header.get('configHash')} "
        f"rate={header.get('residualFalseMatchRate')}, all three read from results.json"
    )


@pytest.mark.boundary_refusal
def test_19_a_batch_whose_rate_is_unknown_does_not_export(tmp_path: Path) -> None:
    """Four ways the rate can be missing or wrong. All four refuse.

    Fault injection in both directions: the artifact that IS valid must still
    produce a provenance, or these would pass by refusing everything.
    """
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))

    missing_file = tmp_path / "absent.json"
    with pytest.raises(ExportError, match="does not exist"):
        provenance_from_results(missing_file, "dev")

    no_rate = tmp_path / "no_rate.json"
    stripped = {k: v for k, v in payload.items() if k != POPULATION_A_KEY}
    no_rate.write_text(json.dumps(stripped), encoding="utf-8")
    with pytest.raises(ExportError, match=RATE_KEY):
        provenance_from_results(no_rate, "dev")

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExportError, match="not valid JSON"):
        provenance_from_results(corrupt, "dev")

    # A float reaching the provenance (D6) and a percentage passed as 1.05.
    with pytest.raises(ExportError, match="not Decimal"):
        ExportProvenance("dev", 42, "abc", 0.01)  # type: ignore[arg-type]
    with pytest.raises(ExportError, match="not a rate in"):
        ExportProvenance("dev", 42, "abc", Decimal("1.05"))

    good = provenance_from_results(RESULTS, "dev")
    assert good.residual_false_match_rate == Decimal(payload[POPULATION_A_KEY][RATE_KEY])
    print(f"\n  4 refusals; the valid artifact still yields rate={good.rate_text}")


@pytest.mark.charter_guard
def test_the_wrong_rate_fails_the_header_assertion(
    confirmed: tuple[Exception_, ...],
    resolved: dict[str, str | None],
    provenance: ExportProvenance,
) -> None:
    """POSITIVE CONTROL for test 18. A header stating a rate nobody measured
    must be detectable, or test 18 is checking that a string equals itself."""
    expected = json.loads(RESULTS.read_text(encoding="utf-8"))[POPULATION_A_KEY][RATE_KEY]
    wrong = replace(provenance, residual_false_match_rate=Decimal("0.999999"))
    header = etree.fromstring(
        to_xml(build_batch(confirmed, resolved, wrong, BATCH_DATE)).encode("utf-8")
    ).find(".//SETTLESENSE.PROVENANCE")
    assert header is not None
    assert header.get("residualFalseMatchRate") != expected
    print(f"\n  a planted rate of 0.999999 does not match the measured {expected}")


@pytest.mark.charter_guard
def test_21_the_output_is_labelled_precisely_and_never_as_an_integration() -> None:
    """The exact sentence, in the module, in the document, and in the README.

    "Integration" is banned on the export path because it is the word a reader
    reaches for, and it would claim a thing nobody tested. The scan walks the
    AST rather than grepping, so this test's own prose cannot satisfy it - the
    sixth-and-counting instance of a scanner matching its own documentation.
    """
    sentence = "not tested against a live Tally instance"
    source = EXPORTER.read_text(encoding="utf-8")
    assert sentence in source, "the exporter does not state what it is"
    assert sentence in (REPO / "README.md").read_text(encoding="utf-8"), "the README does not"
    assert BASIS.endswith(sentence)

    for path in (EXPORTER, REPO / "eval" / "run_export.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        offenders = [
            f"{path.name}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and "integration" in node.value.lower()
        ]
        assert not offenders, f"the export path calls itself an integration: {offenders}"
    print(f"\n  both files carry {sentence!r}; 'integration' appears in no emitted string")


# ===========================================================================
# 4. What must NOT be posted
# ===========================================================================


@pytest.mark.charter_guard
def test_every_postable_category_has_a_ledger_and_unexplained_has_none() -> None:
    """Coverage by MEMBERSHIP, the shape commit 2a744b0 was written for.

    A category added to the taxonomy later must fail here rather than inherit
    a neighbour's ledger. UNEXPLAINED is absent on purpose and its absence is
    asserted, so nobody 'fixes' the gap by mapping it to a suspense account.
    """
    postable = VARIANCE_CATEGORIES - {VarianceCategory.UNEXPLAINED}
    uncovered = sorted(category.value for category in postable if category not in LEDGERS)
    assert not uncovered, f"variance categories with no ledger: {uncovered}"
    assert VarianceCategory.UNEXPLAINED not in LEDGERS, (
        "UNEXPLAINED has a ledger. Posting an unexplained variance turns 'we do "
        "not know what this is' into a booked accounting fact."
    )
    assert set(LEDGERS) <= set(VarianceCategory), "a ledger maps something not in the taxonomy"
    print(f"\n  {len(postable)} postable categories all mapped; UNEXPLAINED deliberately not")


@pytest.mark.boundary_refusal
def test_an_unexplained_confirmation_refuses_rather_than_posting(
    confirmed: tuple[Exception_, ...], provenance: ExportProvenance
) -> None:
    with pytest.raises(ExportError, match="no ledger mapping"):
        build_batch(
            confirmed[:1],
            {confirmed[0].exception_id: VarianceCategory.UNEXPLAINED.value},
            provenance,
            BATCH_DATE,
        )
    with pytest.raises(ExportError, match="not in the taxonomy"):
        build_batch(confirmed[:1], {confirmed[0].exception_id: "INVENTED"}, provenance, BATCH_DATE)
    print("\n  UNEXPLAINED and an off-taxonomy category are both refused")


def test_a_cleared_variance_produces_no_voucher_and_is_counted(
    batch: TallyBatch, confirmed: tuple[Exception_, ...]
) -> None:
    """THE FINDING THIS MODULE WAS BUILT AROUND, asserted with realised counts.

    Most confirmed exceptions have nothing to post. They opened as UNEXPLAINED
    when their day's files were all that existed, and the variance CLEARED once
    the rest arrived. Exporting on the DETECTED category would have booked a
    voucher for every one of them.
    """
    assert len(batch.lines) + len(batch.cleared) == len(confirmed), (
        len(batch.lines),
        len(batch.cleared),
        len(confirmed),
    )
    assert batch.cleared, "nothing cleared, so this proves nothing about clearing"
    assert not set(batch.exception_ids) & set(batch.cleared)
    detected = {row.exception_id: row.category for row in confirmed}
    posted_under_detected = [
        exception_id
        for exception_id, line in zip(batch.exception_ids, batch.lines, strict=True)
        if line.category == detected[exception_id]
    ]
    print(
        f"\n  {len(confirmed)} confirmed = {len(batch.lines)} vouchers + "
        f"{len(batch.cleared)} cleared; only {len(posted_under_detected)} posted under "
        "the category they were detected as"
    )


@pytest.mark.boundary_refusal
def test_a_subject_the_engine_no_longer_reports_refuses(
    confirmed: tuple[Exception_, ...], provenance: ExportProvenance
) -> None:
    """MISSING IS NOT CLEARED. Absent from the mapping raises; None clears."""
    with pytest.raises(ExportError, match="no resolving category"):
        build_batch(confirmed[:2], {confirmed[0].exception_id: None}, provenance, BATCH_DATE)
    with pytest.raises(ExportError, match="nothing to post"):
        build_batch(
            confirmed[:2],
            {row.exception_id: None for row in confirmed[:2]},
            provenance,
            BATCH_DATE,
        )
    print("\n  a missing subject raises; an all-cleared set refuses to write an empty envelope")


# ===========================================================================
# 5. CLOSED, written here and nowhere else
# ===========================================================================


def test_the_exporter_closes_what_it_exported_and_nothing_else(
    config: AppConfig,
    resolved: dict[str, str | None],
    provenance: ExportProvenance,
    tmp_path: Path,
) -> None:
    """CONFIRMED -> CLOSED for exported rows only, in a throwaway store.

    The module-scoped store is deliberately not used: CLOSED is terminal, and
    closing rows in a shared fixture would leave every later test reading a
    store whose statuses this test moved.
    """
    scratch = ExceptionStore(tmp_path / "scratch.db")
    for day in CHECKPOINTS:
        scratch.run_day(day, DATA, config)
    rows = scratch.get_queue(ALL_STATUSES)
    scratch_confirmed = tuple(row for row in rows if row.status is ExceptionStatus.CONFIRMED)
    scratch_resolved = {
        row.exception_id: resolved.get(row.exception_id)
        for row in scratch_confirmed
        if row.exception_id in resolved
    }
    scratch_batch = build_batch(scratch_confirmed, scratch_resolved, provenance, BATCH_DATE)

    before = sum(1 for row in rows if row.status is ExceptionStatus.CLOSED)
    assert before == 0, "something was already CLOSED before the exporter ran"

    closed = close_exported(scratch, scratch_batch, arrival_day=24)
    assert closed == tuple(sorted(scratch_batch.exception_ids))

    after = scratch.get_queue(ALL_STATUSES)
    now_closed = {row.exception_id for row in after if row.status is ExceptionStatus.CLOSED}
    assert now_closed == set(closed), "the exporter closed something it did not export"
    for row in after:
        if row.exception_id in now_closed:
            assert row.closed_day == 24, row.closed_day
            actors = [entry.actor for entry in row.audit if entry.to_status == "CLOSED"]
            assert actors == [AuditActor.EXPORTER], actors
        else:
            assert row.closed_day is None

    print(
        f"\n  {len(closed)} CONFIRMED -> CLOSED, closed_day=24, actor=EXPORTER; "
        f"{len(after) - len(closed)} rows untouched"
    )

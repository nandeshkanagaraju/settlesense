"""M3 engine - ordering, three-population conservation, determinism, accuracy.

Brief requirements 14-31. Every count below is MEASURED and printed, never a
literal copied from a brief: the M1 sessions established that a figure quoted
from a document and asserted as fact is a test of the document, not the code.

THE HEADLINE NUMBER this file exists to produce is the deterministic residual
case count on seed 42. It is the surface M7 has to work with, so it is
reported explicitly rather than left to be inferred from a pass count.
"""

from __future__ import annotations

import json
import random
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from settlesense.config import AppConfig, load_config
from settlesense.exceptions.taxonomy import DEDUCTION_CATEGORIES, VarianceCategory
from settlesense.ingest import DayDataset, load_dataset
from settlesense.matching.engine import (
    EngineError,
    build_cases,
    merge_days,
    residual_cases,
    run,
    run_with_telemetry,
)
from settlesense.types import (
    ExceptionStatus,
    ReconciliationResult,
    SettlementLineType,
    money,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "dev"
DAYS = 24
AS_OF = date(2026, 9, 30)
ZERO = money(0)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture(scope="module")
def dataset(config: AppConfig) -> DayDataset:
    return merge_days([load_dataset(DATA, day, config) for day in range(1, DAYS + 1)])


@pytest.fixture(scope="module")
def result(dataset: DayDataset, config: AppConfig) -> ReconciliationResult:
    return run(dataset, config, AS_OF)


@pytest.fixture(scope="module")
def truth() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((DATA / "truth_42.json").read_text(encoding="utf-8"))
    return data


# ---------------------------------------------------------------------------
# 14-15  Pass ordering
# ---------------------------------------------------------------------------


def test_an_exact_payment_link_claims_a_row_before_a_looser_rule(
    dataset: DayDataset, config: AppConfig
) -> None:
    """14. A line matchable by P1 is claimed by P1, not left for P3.

    Constructed from real rows rather than described: the case's payment_line
    is reachable by exact payment_id (P1) AND its arithmetic closes (P3). The
    assertion is that the case carries the line as a P1 link - if P3 had
    claimed it first the line would not appear in payment_line_ids at all.
    """
    facts = {f.case.payment_id: f for f in build_cases(dataset, config)}
    lines_by_payment: dict[str, list[str]] = {}
    for line in dataset.settlement_lines:
        if line.line_type is SettlementLineType.PAYMENT:
            lines_by_payment.setdefault(line.payment_id, []).append(line.settlement_id)

    both = [
        payment_id
        for payment_id, ids in lines_by_payment.items()
        if payment_id in facts
        and facts[payment_id].settled_gross == facts[payment_id].case.expected_gross
    ]
    assert len(both) > 1000, f"only {len(both)} rows are claimable by both passes"
    for payment_id in sorted(both)[:200]:
        assert facts[payment_id].payment_line_ids == tuple(sorted(lines_by_payment[payment_id])), (
            f"{payment_id}: P1 did not claim the line it can match exactly"
        )


def test_a_matched_row_never_appears_in_the_residual_set(result: ReconciliationResult) -> None:
    """15. CONFIRMED and residual are disjoint, and together are everything."""
    resolved = {case.case_id for case in result.cases if case.status is ExceptionStatus.CONFIRMED}
    residual = {case.case_id for case in residual_cases(result)}
    assert resolved & residual == set(), "a case is both resolved and residual"
    assert resolved | residual == {case.case_id for case in result.cases}
    assert resolved and residual, "one side is empty; the partition proves nothing"


def test_a_bank_credit_is_claimed_by_at_most_one_batch(result: ReconciliationResult) -> None:
    """Once matched, a row leaves the pool. Two batches claiming one credit
    would double-count the cash that arrived."""
    claimed = [link.bank_row_id for link in result.batch_links if link.bank_row_id is not None]
    assert len(claimed) == len(set(claimed)), "a bank credit was claimed twice"


# ---------------------------------------------------------------------------
# 16-21  Population conservation (D11)
# ---------------------------------------------------------------------------


def test_population_a_conserves_on_case_count(result: ReconciliationResult) -> None:
    """16. NEVER over raw input rows - source tables have different grains."""
    resolved = sum(1 for c in result.cases if c.status is ExceptionStatus.CONFIRMED)
    residual = len(residual_cases(result))
    print(f"\n  Population A: resolved={resolved} residual={residual} total={len(result.cases)}")
    assert resolved + residual == len(result.cases)


def test_population_a_conserves_gross_exposure_exactly(
    result: ReconciliationResult, dataset: DayDataset, config: AppConfig
) -> None:
    """17. Exact Decimal on expected_gross, the Population A money basis."""
    cases = {f.case.case_id: f.case for f in build_cases(dataset, config)}
    resolved = money(
        sum(
            (
                cases[c.case_id].expected_gross
                for c in result.cases
                if c.status is ExceptionStatus.CONFIRMED
            ),
            ZERO,
        )
    )
    residual = money(sum((cases[c.case_id].expected_gross for c in residual_cases(result)), ZERO))
    total = money(sum((case.expected_gross for case in cases.values()), ZERO))
    print(f"  gross exposure: resolved={resolved} + residual={residual} == total={total}")
    assert resolved + residual == total
    assert total > ZERO, "the conservation identity is over an empty sum"


def test_population_b_conserves_separately_on_batch_net_total(result: ReconciliationResult) -> None:
    """18. Its OWN basis. Never expected_gross, never merged with Population A."""
    linked = money(sum((b.batch_net_total for b in result.batch_links if b.bank_row_id), ZERO))
    unlinked = money(
        sum((b.batch_net_total for b in result.batch_links if not b.bank_row_id), ZERO)
    )
    total = money(sum((b.batch_net_total for b in result.batch_links), ZERO))
    print(f"  Population B: linked={linked} + unlinked={unlinked} == total={total}")
    assert linked + unlinked == total
    assert linked > ZERO and unlinked > ZERO, "one side is empty; conservation is vacuous"


def test_population_c_conserves_on_a_row_count_denominator(result: ReconciliationResult) -> None:
    """19. A ROW count, not money and not cases."""
    by_table: dict[str, int] = {}
    for variance in result.row_variances:
        by_table[variance.source_table] = by_table.get(variance.source_table, 0) + 1
    print(f"  Population C: {by_table} total={len(result.row_variances)}")
    assert sum(by_table.values()) == len(result.row_variances)
    assert set(by_table) <= {"ledger_rows", "bank_rows"}
    assert len(by_table) == 2, f"only {sorted(by_table)} represented; C is not exercised"


def test_the_three_denominators_are_distinct_values(result: ReconciliationResult) -> None:
    """20. If any two coincided, a merged metric would be undetectable here."""
    a, b, c = len(result.cases), len(result.batch_links), len(result.row_variances)
    print(f"  denominators: A={a} B={b} C={c}")
    assert len({a, b, c}) == 3, f"denominators collide: A={a} B={b} C={c}"


def test_every_truth_variance_lands_in_exactly_one_population(
    result: ReconciliationResult, truth: dict[str, Any]
) -> None:
    """21. COMPLETENESS. Zero homeless variances.

    Every variance the generator recorded must have a home in A, B or C. One
    that belongs to none is unscoreable - it can never be found and never be
    missed, so it silently improves every rate it is absent from.
    """
    case_ids = {c.case_id for c in result.cases}
    batch_ids = {b.batch_id for b in result.batch_links}
    row_ids = {v.row_id for v in result.row_variances}

    homeless: list[str] = []
    counted = 0
    for case in truth["cases"]:
        if case["true_category"] is None:
            continue
        counted += 1
        if case["case_id"] not in case_ids:
            homeless.append(f"case {case['case_id']}")
    for link in truth["batch_links"]:
        if link["true_category"] is None:
            continue
        counted += 1
        if link["batch_id"] not in batch_ids:
            homeless.append(f"batch {link['batch_id']}")
    for variance in truth["row_variances"]:
        counted += 1
        if variance["row_id"] not in row_ids:
            homeless.append(f"row {variance['row_id']}")

    print(f"  truth variances placed: {counted}, homeless: {len(homeless)}")
    assert counted > 200, f"only {counted} truth variances swept"
    assert not homeless, f"variances with no population: {homeless[:10]}"


def test_no_deduction_category_is_ever_emitted_as_a_variance(result: ReconciliationResult) -> None:
    """22. PDD 6.1, swept across all three populations.

    MDR_FEE, GST_ON_FEE and REFUND_OFFSET are components of expected_net,
    computed on every clean case. Emitting one as a variance would report a
    fee as a discrepancy.
    """
    forbidden = {str(category) for category in DEDUCTION_CATEGORIES}
    assert len(forbidden) == 3
    emitted = (
        {str(c.category) for c in result.cases if c.category}
        | {str(b.category) for b in result.batch_links if b.category}
        | {str(v.category) for v in result.row_variances if v.category}
    )
    print(f"  categories emitted: {sorted(emitted)}")
    assert emitted, "no categories emitted at all; the sweep is vacuous"
    assert not (emitted & forbidden), f"deduction categories emitted: {sorted(emitted & forbidden)}"
    assert emitted <= {str(c) for c in VarianceCategory}, "a category outside the taxonomy"


def test_a_refund_case_is_not_mistaken_for_a_split(dataset: DayDataset, config: AppConfig) -> None:
    """23, the half seed 42 can exercise.

    A payment with one PAYMENT line and one REFUND line has TWO settlement
    lines and is NOT a split. Counting settlement_line_ids instead of
    payment_line_ids would report all 298 refunded cases as splits.
    """
    facts = build_cases(dataset, config)
    refunded = [f for f in facts if len(f.settlement_line_ids) > len(f.payment_line_ids)]
    print(f"  cases with a refund line: {len(refunded)}")
    assert len(refunded) > 100, f"only {len(refunded)} refunded cases; the check is thin"
    for fact in refunded:
        assert len(fact.payment_line_ids) == 1, (
            f"{fact.case.case_id} has {len(fact.payment_line_ids)} payment lines; a refund "
            "line was counted as a second settlement"
        )


def test_a_split_settlement_yields_one_case_with_several_payment_lines(
    dataset: DayDataset, config: AppConfig
) -> None:
    """23, the half seed 42 CANNOT exercise, on a constructed fixture.

    split_settlement is a WITHHELD noise type (GENERATOR_MANIFEST.json), so
    the dev dataset contains zero splits - verified below rather than assumed.
    A test that skipped here would report a passing suite for a rule nothing
    checked, so the fixture is built instead.
    """
    facts = build_cases(dataset, config)
    natural = [f for f in facts if len(f.payment_line_ids) >= 2]
    assert not natural, (
        f"{len(natural)} splits found in the dev dataset; split_settlement is "
        "supposed to be withheld. Re-check GENERATOR_MANIFEST.json."
    )

    from settlesense.matching.exact import link_payments_to_settlements

    original = next(
        line for line in dataset.settlement_lines if line.line_type is SettlementLineType.PAYMENT
    )
    half = money(original.gross / 2)
    from dataclasses import replace

    part_a = replace(original, settlement_id=original.settlement_id + "_A", gross=half, net=half)
    part_b = replace(original, settlement_id=original.settlement_id + "_B", gross=half, net=half)
    links, _ = link_payments_to_settlements([part_a, part_b], frozenset({original.payment_id}))

    assert len(links) == 1, "a split produced more than one case"
    link = links[original.payment_id]
    assert len(link.payment_line_ids) == 2
    assert link.is_split


# ---------------------------------------------------------------------------
# 24-29  Determinism
# ---------------------------------------------------------------------------


def _serialize(result: ReconciliationResult) -> str:
    """Canonical text for byte comparison. Sorted keys, Decimals as strings."""
    return json.dumps(
        {
            "cases": [
                [c.case_id, str(c.status), str(c.observed_net), str(c.variance), str(c.category)]
                for c in result.cases
            ],
            "batch_links": [
                [
                    b.batch_id,
                    str(b.status),
                    str(b.bank_row_id),
                    str(b.batch_net_total),
                    str(b.linked_amount),
                    str(b.variance),
                    str(b.category),
                ]
                for b in result.batch_links
            ],
            "row_variances": [
                [v.row_id, v.source_table, str(v.status), str(v.category), str(v.amount)]
                for v in result.row_variances
            ],
            "calendar_version": result.calendar_version,
            "config_hash": result.config_hash,
        },
        sort_keys=True,
    )


def test_two_runs_serialize_byte_identically(dataset: DayDataset, config: AppConfig) -> None:
    """24. D6."""
    first = _serialize(run(dataset, config, AS_OF))
    second = _serialize(run(dataset, config, AS_OF))
    assert first == second
    assert len(first) > 100_000, "the serialization is too small to be the whole result"


def test_ids_are_identical_across_runs(dataset: DayDataset, config: AppConfig) -> None:
    """25. D10. Deterministic hashes, not uuid4."""
    first = [c.case_id for c in run(dataset, config, AS_OF).cases]
    second = [c.case_id for c in run(dataset, config, AS_OF).cases]
    assert first == second
    assert len(set(first)) == len(first), "case ids are not unique"
    assert all(len(i) == 16 and all(ch in "0123456789abcdef" for ch in i) for i in first)


def test_shuffling_the_input_leaves_the_output_unchanged(
    dataset: DayDataset, config: AppConfig, result: ReconciliationResult
) -> None:
    """26. D4. Every output list sorted by an explicit key.

    Shuffles all six tables with a seeded RNG, so a failure is reproducible.
    """
    from dataclasses import replace

    rng = random.Random(20260827)

    def shuffled(rows: tuple[object, ...]) -> tuple[object, ...]:
        listed = list(rows)
        rng.shuffle(listed)
        return tuple(listed)

    scrambled = replace(
        dataset,
        ledger_rows=shuffled(dataset.ledger_rows),  # type: ignore[arg-type]
        payment_rows=shuffled(dataset.payment_rows),  # type: ignore[arg-type]
        refund_rows=shuffled(dataset.refund_rows),  # type: ignore[arg-type]
        settlement_lines=shuffled(dataset.settlement_lines),  # type: ignore[arg-type]
        settlement_batches=shuffled(dataset.settlement_batches),  # type: ignore[arg-type]
        bank_rows=shuffled(dataset.bank_rows),  # type: ignore[arg-type]
    )
    assert scrambled.ledger_rows != dataset.ledger_rows, "the shuffle changed nothing"
    assert _serialize(run(scrambled, config, AS_OF)) == _serialize(result)


def test_as_of_is_honoured_as_a_parameter(dataset: DayDataset, config: AppConfig) -> None:
    """27. Two as_of dates must produce DIFFERENT timing classifications.

    Writing this test is what showed that as_of was accepted and ignored: it
    was threaded through every signature and changed nothing, which is
    behaviourally identical to reading a clock and no better. A batch whose
    credit is not due until after as_of is NOT a missing credit - it is a
    payout in the future, and calling it missing manufactures an exception out
    of a file that was never late.
    """
    early = run(dataset, config, date(2026, 9, 10))
    late = run(dataset, config, date(2026, 11, 30))

    def pending(outcome: ReconciliationResult) -> int:
        return sum(1 for b in outcome.batch_links if b.status is ExceptionStatus.PENDING_EVIDENCE)

    def missing(outcome: ReconciliationResult) -> int:
        return sum(
            1
            for b in outcome.batch_links
            if b.category == str(VarianceCategory.MISSING_VS_LATE_CREDIT)
        )

    print(
        f"\n  as_of 2026-09-10: not-yet-due={pending(early)} missing={missing(early)}"
        f"\n  as_of 2026-11-30: not-yet-due={pending(late)} missing={missing(late)}"
    )
    assert pending(early) > pending(late), (
        "an earlier as_of must leave MORE batches not yet due. If it does not, "
        "as_of is being accepted and ignored."
    )
    assert pending(late) == 0, "every batch is due by 2026-11-30"
    assert _serialize(early) != _serialize(late), "as_of changed nothing in the result"


@pytest.mark.determinism
def test_the_engine_reads_no_clock() -> None:
    """D2, by AST scan rather than by inspection."""
    import ast

    from settlesense.matching import arithmetic, duplicates, engine, exact, timing

    forbidden = {"now", "utcnow", "today", "monotonic", "time"}
    offenders: list[str] = []
    for module in (arithmetic, duplicates, engine, exact, timing):
        source = Path(module.__file__ or "").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in forbidden
            ):
                offenders.append(f"{Path(module.__file__ or '').name}:{node.lineno}")
    assert not offenders, f"clock access in the engine: {offenders}"


def test_the_engine_never_raises_on_the_full_dataset(result: ReconciliationResult) -> None:
    """28. Reaching this fixture at all is the assertion."""
    assert len(result.cases) > 5000


def test_an_empty_dataset_produces_an_empty_result(config: AppConfig) -> None:
    """29. And an EMPTY table stays distinguishable from a MISSING file."""
    empty = DayDataset(
        arrival_day=1,
        ledger_rows=(),
        payment_rows=(),
        refund_rows=(),
        settlement_lines=(),
        settlement_batches=(),
        bank_rows=(),
    )
    outcome = run(empty, config, AS_OF)
    assert outcome.cases == () and outcome.batch_links == () and outcome.row_variances == ()
    assert outcome.calendar_version == config.calendar.version

    from settlesense.ingest import IngestError

    with pytest.raises(IngestError, match="does not exist"):
        load_dataset(DATA, DAYS + 99, config)


@pytest.mark.boundary_refusal
def test_merging_no_days_raises(config: AppConfig) -> None:
    """FAULT INJECTION. Zero days is not an empty dataset; it is a caller bug."""
    with pytest.raises(EngineError, match="at least one day"):
        merge_days([])


def test_the_result_carries_no_wallclock_and_telemetry_is_separate(
    dataset: DayDataset, config: AppConfig
) -> None:
    """SDD 8.1. Two return values; nothing to strip before comparison."""
    business, telemetry = run_with_telemetry(dataset, config, AS_OF)
    assert _serialize(business) == _serialize(run(dataset, config, AS_OF))
    assert hasattr(telemetry, "timings") and not hasattr(business, "timings")
    text = json.dumps(_serialize(business))
    for banned in ("seconds", "duration", "elapsed", "timestamp"):
        assert banned not in text.lower(), f"{banned!r} reached the business result"


# ---------------------------------------------------------------------------
# 30-31  Accuracy against ground truth
# ---------------------------------------------------------------------------


def test_resolved_cases_agree_with_truth_and_report_precision(
    result: ReconciliationResult, truth: dict[str, Any]
) -> None:
    """30. THE accuracy test. Precision and residual count are printed."""
    by_id = {c["case_id"]: c for c in truth["cases"]}
    resolved = [c for c in result.cases if c.status is ExceptionStatus.CONFIRMED]
    residual = residual_cases(result)
    assert resolved, "nothing was resolved; precision over an empty set is meaningless"

    disagreements = [
        (c.case_id, str(c.category), str(by_id[c.case_id]["true_category"]))
        for c in resolved
        if str(c.category) != str(by_id[c.case_id]["true_category"])
    ]
    precision = Decimal(len(resolved) - len(disagreements)) / Decimal(len(resolved))
    print(
        f"\n  DETERMINISTIC PRECISION: {len(resolved) - len(disagreements)}/{len(resolved)}"
        f" = {precision * 100:.3f}%"
        f"\n  RESIDUAL CASES: {len(residual)} of {len(result.cases)}"
        f" ({Decimal(len(residual)) / Decimal(len(result.cases)) * 100:.2f}%)"
    )
    assert not disagreements, f"resolved cases disagreeing with truth: {disagreements[:10]}"


def test_zero_false_matches_on_clean_chains(
    result: ReconciliationResult, truth: dict[str, Any]
) -> None:
    """31. EXACTLY ZERO. Everything downstream measures against this.

    A false match is a case the engine RESOLVED against truth that says
    otherwise - a confident wrong answer. It is not the same as flagging a
    clean case for review, which is over-inclusion: conservative, visible in
    the residual count, and never presented as an answer.

    Both are counted, and the second is printed rather than hidden, because
    over-inclusion is a real cost even though it is not a false match.
    """
    by_id = {c["case_id"]: c for c in truth["cases"]}
    clean = [
        c
        for c in result.cases
        if by_id[c.case_id]["true_category"] is None and not by_id[c.case_id]["noise_types"]
    ]
    assert len(clean) > 4000, f"only {len(clean)} clean chains; the check is thin"

    false_matches = [
        c.case_id for c in clean if c.status is ExceptionStatus.CONFIRMED and c.category is not None
    ]
    over_flagged = [c.case_id for c in clean if c.status is not ExceptionStatus.CONFIRMED]
    print(
        f"\n  clean chains: {len(clean)}"
        f"\n  FALSE MATCHES: {len(false_matches)}   (must be exactly 0)"
        f"\n  over-flagged for review: {len(over_flagged)}"
        f" ({Decimal(len(over_flagged)) / Decimal(len(clean)) * 100:.2f}% of clean)"
    )
    assert false_matches == [], f"FALSE MATCHES on clean chains: {false_matches[:10]}"


def test_every_residual_is_an_ambiguous_duplicate_pair(
    result: ReconciliationResult,
    dataset: DayDataset,
    config: AppConfig,
    truth: dict[str, Any],
) -> None:
    """What the residual set actually consists of, asserted not assumed.

    The deterministic layer leaves exactly one kind of work: pairs of orders
    sharing a customer and an amount, where a data duplicate and a genuine
    repeat purchase are indistinguishable. The engine flags BOTH halves,
    because it cannot know which one was injected without reading the
    generator's invoice-suffix convention (-R027), and reading that would be
    fitting to the fixture rather than reconciling.

    PAIR MEMBERSHIP IS TAKEN FROM THE ENGINE'S OWN P7b OUTPUT, not inferred
    from truth. Truth annotates only the injected repeat, so two residual
    cases - the ORIGINALS of pairs that were also late or partially captured -
    carry neither the duplicate category nor the duplicate noise type. Reading
    pair membership from truth reported them as unexplained residuals when
    they are nothing of the kind.
    """
    from settlesense.matching.duplicates import (
        find_candidate_duplicates,
        find_confirmed_duplicates,
    )

    confirmed = find_confirmed_duplicates(dataset.ledger_rows)
    excluded = frozenset(i for v in confirmed for i in v.row_ids)
    pairs = find_candidate_duplicates(dataset.ledger_rows, excluded)
    paired_orders = frozenset(order for v in pairs for order in v.row_ids)
    assert paired_orders, "P7b found no pairs; this test would be vacuous"

    order_of_case = {f.case.case_id: f.case.order_id for f in build_cases(dataset, config)}
    residual = residual_cases(result)
    strays = [c.case_id for c in residual if order_of_case[c.case_id] not in paired_orders]

    truth_flagged = sum(1 for c in truth["cases"] if c["true_category"] == "DUPLICATE_CANDIDATE")
    print(
        f"\n  residual={len(residual)}  ambiguous pairs found={len(pairs)}"
        f"  truth DUPLICATE_CANDIDATE={truth_flagged}"
        f"\n  every residual is one half of a pair: {not strays}"
    )
    assert not strays, f"residual cases that are NOT part of a duplicate pair: {strays[:5]}"
    assert len(residual) == truth_flagged * 2, (
        f"expected both halves of {truth_flagged} pairs, got {len(residual)}"
    )
    assert len(pairs) == truth_flagged, (
        f"engine found {len(pairs)} pairs, truth annotated {truth_flagged} repeats"
    )

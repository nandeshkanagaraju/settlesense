"""M7 - deterministic verification of every hypothesis. Never eval() (SDD 4.5).

THE CORE INVARIANT (PDD 7.1). No exception is ever confirmed on the strength of
model output alone. This module re-derives everything itself and rejects when
it cannot.

TWO PATHS, BECAUSE ONE IS NOT ENOUGH FOR THIS DATASET.

  ARITHMETIC. The hypothesis carries an assertion; the verifier recomputes both
  sides with Decimal and checks the residual against tolerance. This is the
  general mechanism and M8 must be able to show it working - so it is built and
  tested here even though the seed-42 residual never exercises it.

  STRUCTURAL. The hypothesis carries a claim about record structure; the
  verifier checks it against the data WITHOUT the model's help.
  DUPLICATE_CANDIDATE is the entire residual on this dataset and has no
  arithmetic at all: no residual to recompute, no tolerance to check. A
  verifier built only for arithmetic could not verify the one category it will
  ever see.

REJECTION IS THE DEFAULT, AND IT IS NOT A FAILURE. If the structural facts do
not distinguish the candidates, this rejects rather than deferring to the
model. A verifier that cannot independently check a claim has not verified it,
and confirming on the model's say-so is exactly the failure this architecture
exists to prevent.

NO eval(), NO exec(). Assertions evaluate through a small allow-listed grammar:
a field reference resolves to a Decimal read off a typed record, and the
operator comes from a fixed tuple. `eval("bank.amount == settlement.net")`
would work, and would also run whatever else the model chose to write.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from settlesense.ai.hypothesis import Assertion, Hypothesis
from settlesense.config import AppConfig
from settlesense.exceptions.taxonomy import VarianceCategory
from settlesense.ingest import DayDataset
from settlesense.types import Money, money

__all__ = [
    "FIELD_GRAMMAR",
    "GrammarError",
    "VerificationResult",
    "evaluate_assertion",
    "resolve_field",
    "verify",
]


@dataclass(frozen=True)
class VerificationResult:
    """SDD 4.5: (passed, computed_residual, failure_reason).

    `failure_reason` is populated on EVERY rejection and names the check that
    failed. A verifier returning a bare False teaches nobody anything, and the
    reason is what a reviewer reads in the M8 queue.

    `evidence_completeness` and `candidate_separation` are carried here because
    confidence.py needs them and must not recompute them from the model's
    output - they are facts this module established.
    """

    passed: bool
    computed_residual: Money | None
    failure_reason: str
    checks_run: tuple[str, ...] = ()
    evidence_completeness: Decimal = Decimal("0")
    candidate_separation: Decimal = Decimal("0")


class GrammarError(ValueError):
    """A field reference or operator outside the allow-list."""


FIELD_GRAMMAR: dict[str, tuple[str, ...]] = {
    "ledger": ("gross",),
    "payment": ("authorized", "captured"),
    "refund": ("amount",),
    "settlement": ("gross", "fee", "tax", "net"),
    "batch": ("net_total",),
    "bank": ("amount",),
}
"""The ONLY readable fields. Money-valued, on typed records.

An allow-list rather than getattr on anything: `getattr(row, name)` would let a
model read `__class__` or a method and compare its repr, which is neither
arithmetic nor checkable.
"""

_REFERENCE = re.compile(r"^(?P<table>[a-z_]+)\.(?P<field>[a-z_]+)$")
_LITERAL = re.compile(r"^-?\d+(\.\d+)?$")


def resolve_field(reference: str, rows: dict[str, Any]) -> Decimal:
    """`table.field` or a numeric literal -> Decimal. Nothing else parses.

    Literals are permitted so an assertion can compare against zero without a
    special case; they are matched by a strict regex, so "0; import os" is a
    parse failure rather than a clever string.
    """
    if _LITERAL.match(reference):
        return Decimal(reference)

    match = _REFERENCE.match(reference)
    if match is None:
        raise GrammarError(
            f"{reference!r} is not a permitted reference. Use `table.field` from "
            f"{sorted(FIELD_GRAMMAR)} or a numeric literal."
        )
    table, field = match.group("table"), match.group("field")
    if table not in FIELD_GRAMMAR:
        raise GrammarError(f"unknown table {table!r}; permitted: {sorted(FIELD_GRAMMAR)}")
    if field not in FIELD_GRAMMAR[table]:
        raise GrammarError(
            f"{table}.{field} is not readable; permitted on {table}: {sorted(FIELD_GRAMMAR[table])}"
        )
    row = rows.get(table)
    if row is None:
        raise GrammarError(f"the hypothesis references {table}.{field} but no {table} row resolved")
    value = getattr(row, field, None)
    if not isinstance(value, Decimal):
        raise GrammarError(f"{table}.{field} is {type(value).__name__}, not a Decimal")
    return value


_OPERATORS: dict[str, Any] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def evaluate_assertion(
    assertion: Assertion, rows: dict[str, Any], tolerance: Money
) -> tuple[bool, Money]:
    """Recompute both sides with Decimal. Returns (holds, residual).

    The residual is lhs - rhs regardless of operator, because that is the
    number a reviewer needs: "this failed" is less useful than "this failed by
    Rs 4,312.00". For an equality the tolerance applies to |residual|; for an
    inequality the operator is the whole test and the residual is informational.
    """
    operator = _OPERATORS.get(assertion.op)
    if operator is None:
        raise GrammarError(f"operator {assertion.op!r} is not permitted")
    left = resolve_field(assertion.lhs, rows)
    right = resolve_field(assertion.rhs, rows)
    residual = money(left - right)
    if assertion.op == "==":
        return abs(residual) <= tolerance, residual
    return bool(operator(left, right)), residual


def verify(hypothesis: Hypothesis, dataset: DayDataset, config: AppConfig) -> VerificationResult:
    """The one entry point. Resolve evidence, then take the matching path."""
    tolerance = config.thresholds.tolerance.verifier_rupees
    index = _row_index(dataset)

    resolved = {row_id: index[row_id] for row_id in hypothesis.evidence_row_ids if row_id in index}
    missing = sorted(set(hypothesis.evidence_row_ids) - set(resolved))
    completeness = (
        Decimal(len(resolved)) / Decimal(len(hypothesis.evidence_row_ids))
        if hypothesis.evidence_row_ids
        else Decimal("0")
    )
    if missing:
        # IMMEDIATE REJECT (SDD 4.5). A hypothesis citing a row that is not in
        # the dataset is citing something it cannot have read.
        return VerificationResult(
            passed=False,
            computed_residual=None,
            failure_reason=f"evidence rows do not exist in the dataset: {missing}",
            checks_run=("evidence_resolution",),
            evidence_completeness=completeness,
        )
    if not hypothesis.evidence_row_ids:
        return VerificationResult(
            passed=False,
            computed_residual=None,
            failure_reason="the hypothesis cites no evidence, so there is nothing to check",
            checks_run=("evidence_resolution",),
        )

    if hypothesis.is_structural:
        return _verify_structural(hypothesis, resolved, dataset, completeness)
    return _verify_arithmetic(hypothesis, resolved, tolerance, completeness)


def _verify_arithmetic(
    hypothesis: Hypothesis,
    resolved: dict[str, Any],
    tolerance: Money,
    completeness: Decimal,
) -> VerificationResult:
    """Recompute the assertion. Unused on seed 42; the general mechanism."""
    assertion = hypothesis.assertion
    assert assertion is not None  # is_structural was False
    rows = _by_table(resolved)
    try:
        holds, residual = evaluate_assertion(assertion, rows, tolerance)
    except GrammarError as error:
        return VerificationResult(
            passed=False,
            computed_residual=None,
            failure_reason=f"assertion did not parse: {error}",
            checks_run=("grammar",),
            evidence_completeness=completeness,
        )
    if not holds:
        return VerificationResult(
            passed=False,
            computed_residual=residual,
            failure_reason=(
                f"{assertion.lhs} {assertion.op} {assertion.rhs} is false; "
                f"residual {residual} exceeds tolerance {tolerance}"
            ),
            checks_run=("grammar", "assertion"),
            evidence_completeness=completeness,
        )
    # The model's own residual, if it offered one, must AGREE with ours. It is
    # cross-checked, never trusted: a claim that happens to be arithmetically
    # true about the wrong rows is still wrong.
    if hypothesis.residual_amount is not None and money(hypothesis.residual_amount) != residual:
        return VerificationResult(
            passed=False,
            computed_residual=residual,
            failure_reason=(
                f"the hypothesis claims residual {hypothesis.residual_amount} but the "
                f"data gives {residual}"
            ),
            checks_run=("grammar", "assertion", "residual_agreement"),
            evidence_completeness=completeness,
        )
    return VerificationResult(
        passed=True,
        computed_residual=residual,
        failure_reason="",
        checks_run=("grammar", "assertion", "residual_agreement"),
        evidence_completeness=completeness,
        candidate_separation=Decimal("1"),
    )


def _verify_structural(
    hypothesis: Hypothesis,
    resolved: dict[str, Any],
    dataset: DayDataset,
    completeness: Decimal,
) -> VerificationResult:
    """Check the category's precondition against the data, without the model."""
    del resolved  # the structural path re-reads from the dataset itself
    if hypothesis.category != str(VarianceCategory.DUPLICATE_CANDIDATE):
        return VerificationResult(
            passed=False,
            computed_residual=None,
            failure_reason=(
                f"no structural check is defined for {hypothesis.category}; a "
                "category with neither an arithmetic assertion nor a structural "
                "precondition cannot be verified at all"
            ),
            checks_run=("structural_dispatch",),
            evidence_completeness=completeness,
        )
    return _verify_duplicate_candidate(hypothesis, dataset, completeness)


def _verify_duplicate_candidate(
    hypothesis: Hypothesis, dataset: DayDataset, completeness: Decimal
) -> VerificationResult:
    """DUPLICATE_CANDIDATE: the model nominates WHICH row; we check the rest.

    THE FIRST CHECKS ARE FACTS ABOUT THE PAIR, and every one is true of BOTH
    rows by construction - that is what makes the pair ambiguous. They
    establish only that the hypothesis is talking about a real duplicate pair.

    THE CHECK THAT DECIDES is the last: the nominated row must have no distinct
    settlement chain of its own. A row separately captured, settled and
    credited is a real order however much it resembles its neighbour; a row
    with no chain is the double entry.

    WHEN THAT CHECK CANNOT SEPARATE THE TWO, THIS REJECTS. It does not fall
    back to the nomination, and it does not tie-break on anything the generator
    happens to leave behind - see LIMITATIONS.md on the invoice suffix, which
    identifies the injected row perfectly and means nothing.
    """
    ledger = {row.order_id: row for row in dataset.ledger_rows}
    cited = sorted(row_id for row_id in hypothesis.evidence_row_ids if row_id in ledger)
    checks: list[str] = ["both_rows_exist"]

    if len(cited) != 2:
        return VerificationResult(
            passed=False,
            computed_residual=None,
            failure_reason=(
                "a duplicate claim needs exactly two ledger rows as evidence; "
                f"{len(cited)} resolved from {list(hypothesis.evidence_row_ids)}"
            ),
            checks_run=tuple(checks),
            evidence_completeness=completeness,
        )

    first, second = (ledger[row_id] for row_id in cited)
    if first.customer_id != second.customer_id:
        return VerificationResult(
            passed=False,
            computed_residual=None,
            failure_reason=(
                f"the two rows belong to different customers ({first.customer_id} vs "
                f"{second.customer_id}), so they are not a duplicate pair at all"
            ),
            checks_run=tuple([*checks, "same_customer"]),
            evidence_completeness=completeness,
        )
    checks.append("same_customer")

    if first.gross != second.gross:
        return VerificationResult(
            passed=False,
            computed_residual=money(first.gross - second.gross),
            failure_reason=(
                f"the two rows differ in gross ({first.gross} vs {second.gross}); a "
                "duplicate is the same amount twice"
            ),
            checks_run=tuple([*checks, "same_gross"]),
            evidence_completeness=completeness,
        )
    checks.append("same_gross")

    if first.order_id == second.order_id:
        return VerificationResult(
            passed=False,
            computed_residual=None,
            failure_reason="the hypothesis cites one row twice, not a pair",
            checks_run=tuple([*checks, "distinct_order_id"]),
            evidence_completeness=completeness,
        )
    checks.append("distinct_order_id")

    nominated = hypothesis.candidate_id
    if nominated not in {first.order_id, second.order_id}:
        return VerificationResult(
            passed=False,
            computed_residual=None,
            failure_reason=(
                f"the nominated duplicate {nominated!r} is not one of the two rows {cited}"
            ),
            checks_run=tuple([*checks, "nomination_in_pair"]),
            evidence_completeness=completeness,
        )
    checks.append("nomination_in_pair")

    other = second.order_id if nominated == first.order_id else first.order_id
    nominated_chain = _chain_length(nominated, dataset)
    other_chain = _chain_length(other, dataset)
    checks.append("nominated_has_no_distinct_chain")

    if nominated_chain > 0 and nominated_chain == other_chain:
        # THE HONEST REJECTION, and on this dataset it is every pair. Both rows
        # were captured and settled identically, so nothing here says which is
        # the double entry. Confirming would be repeating the model's guess
        # with extra steps.
        return VerificationResult(
            passed=False,
            computed_residual=None,
            failure_reason=(
                f"both rows carry an identical settlement chain ({nominated_chain} "
                "lines each), so the structural facts do not distinguish them. The "
                "nomination cannot be checked independently, and confirming it "
                "would be deferring to the model."
            ),
            checks_run=tuple(checks),
            evidence_completeness=completeness,
            candidate_separation=Decimal("0"),
        )
    if nominated_chain >= other_chain:
        return VerificationResult(
            passed=False,
            computed_residual=None,
            failure_reason=(
                f"the nominated row {nominated} has {nominated_chain} settlement lines "
                f"against {other_chain} for {other}; a double entry does not settle "
                "more than the order it duplicates"
            ),
            checks_run=tuple(checks),
            evidence_completeness=completeness,
        )
    return VerificationResult(
        passed=True,
        computed_residual=money(0),
        failure_reason="",
        checks_run=tuple(checks),
        evidence_completeness=completeness,
        candidate_separation=Decimal("1"),
    )


def _chain_length(order_id: str, dataset: DayDataset) -> int:
    """Settlement lines reachable from one order. The structural discriminator.

    Counted through payments rather than read off a field, because the field
    does not exist: an order's relationship to a settlement line is the join,
    and the join is what a duplicate fails to have.
    """
    payments = {row.payment_id for row in dataset.payment_rows if row.order_id == order_id}
    if not payments:
        return 0
    return sum(1 for line in dataset.settlement_lines if line.payment_id in payments)


def _row_index(dataset: DayDataset) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for ledger_row in dataset.ledger_rows:
        index[ledger_row.order_id] = ledger_row
    for payment in dataset.payment_rows:
        index[payment.payment_id] = payment
    for refund in dataset.refund_rows:
        index[refund.refund_id] = refund
    for line in dataset.settlement_lines:
        index[line.settlement_id] = line
    for batch in dataset.settlement_batches:
        index[batch.batch_id] = batch
    for bank in dataset.bank_rows:
        index[bank.bank_txn_id] = bank
    return index


def _by_table(resolved: dict[str, Any]) -> dict[str, Any]:
    """Group resolved rows by grammar table name.

    The LAST row of each type wins, deliberately: an assertion naming
    `settlement.net` when two settlement lines were cited is ambiguous, and the
    arithmetic path is for single-row comparisons. A hypothesis needing two
    lines of one type has to say so through evidence the structural path reads.
    """
    from settlesense.types import (
        BankRow,
        LedgerRow,
        PaymentRow,
        RefundRow,
        SettlementBatch,
        SettlementLine,
    )

    mapping = {
        LedgerRow: "ledger",
        PaymentRow: "payment",
        RefundRow: "refund",
        SettlementLine: "settlement",
        SettlementBatch: "batch",
        BankRow: "bank",
    }
    tables: dict[str, Any] = {}
    for _row_id, row in sorted(resolved.items()):
        name = mapping.get(type(row))
        if name is not None:
            tables[name] = row
    return tables

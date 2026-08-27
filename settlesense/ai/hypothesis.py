"""M7 - structured hypothesis generation against the closed taxonomy (SDD 4.4).

THE MODEL NOMINATES; IT NEVER DECIDES. Everything here produces a CLAIM that
`verifier.py` then checks against the data without the model's help. Nothing in
this module confirms anything, and the hypothesis type carries no field that
could be mistaken for a verdict.

DETERMINISTIC PROMPTS. The same exception must produce a byte-identical prompt
every time, so every interpolated collection is sorted by an explicit key and
no dict is rendered in insertion order. Without that, two runs send two
different prompts, miss the fixture cache, and the whole replay mechanism
becomes a source of nondeterminism rather than a defence against it.

THE ELIGIBILITY GATE IS A REFUSAL, NOT A FILTER. PDD 6.2 lists the categories
that are genuinely interpretive. Anything else raises rather than being quietly
skipped: a category silently dropped is a category nobody notices stopped being
sent, and a category wrongly sent invites a fabricated explanation for what is
actually missing data. Population B's unlinked batches are the live example -
"the credit never arrived" is missing data, not an interpretive question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from settlesense.ai.client import LLMClient
from settlesense.config import AppConfig
from settlesense.exceptions.taxonomy import VarianceCategory
from settlesense.ingest import DayDataset
from settlesense.types import Exception_

__all__ = [
    "AI_ELIGIBLE_CATEGORIES",
    "HYPOTHESIS_SCHEMA",
    "MAX_HYPOTHESES",
    "Assertion",
    "Hypothesis",
    "IneligibleCategoryError",
    "build_prompt",
    "generate",
    "parse_hypotheses",
]

MAX_HYPOTHESES = 3
"""SDD 4.4: up to 3 ranked hypotheses per exception."""

MAX_RETRIES = 2
"""Invalid JSON after 2 retries -> no hypothesis, and NO crash.

A model that cannot produce schema-valid output is a model that has not
answered. That is an abstention, which is a legitimate outcome; crashing would
lose every other exception in the run over one bad reply.
"""

AI_ELIGIBLE_CATEGORIES: frozenset[str] = frozenset(
    {
        str(VarianceCategory.UTR_TRUNCATED_MAPPING),
        str(VarianceCategory.UTR_MISSING_MAPPING),
        str(VarianceCategory.DUPLICATE_CANDIDATE),
        str(VarianceCategory.SPLIT_SETTLEMENT),
        str(VarianceCategory.MISSING_VS_LATE_CREDIT),
        str(VarianceCategory.UNEXPLAINED),
    }
)
"""PDD 6.2, exactly. Nothing else may be sent to a model.

MISSING_VS_LATE_CREDIT is listed by the PDD and is therefore permitted here,
but on this dataset it is deliberately NOT sent: see `eligible_exceptions`.
The distinction matters - the taxonomy says a human might reason about it, and
the wiring says the evidence to reason FROM does not exist yet.
"""

_ALLOWED_OPS = ("==", "<=", ">=", "<", ">", "!=")


@dataclass(frozen=True)
class Assertion:
    """An arithmetic claim, as data. Never a string to be evaluated.

    Three fields rather than one expression, because a parsed structure cannot
    smuggle a function call. `eval()` on "bank.amount == settlement.net_total"
    would work and would also run whatever else the model chose to write.
    """

    lhs: str
    op: str
    rhs: str

    @staticmethod
    def from_payload(payload: object) -> Assertion | None:
        if not isinstance(payload, dict):
            return None
        lhs, op, rhs = payload.get("lhs"), payload.get("op"), payload.get("rhs")
        if not isinstance(lhs, str) or not isinstance(rhs, str) or op not in _ALLOWED_OPS:
            return None
        return Assertion(lhs=lhs, op=op, rhs=rhs)


@dataclass(frozen=True)
class Hypothesis:
    """One ranked claim about one exception (SDD 4.4).

    NO CONFIDENCE FIELD. If the model emits one it is dropped at parse time and
    never reaches this type, so no downstream code can read it by accident.
    Confidence is computed by `confidence.py` from verification outcomes -
    PDD 7.2 - and a self-reported number in the same object would eventually be
    used because it was there.
    """

    category: str
    candidate_id: str
    assertion: Assertion | None
    residual_amount: Decimal | None
    evidence_row_ids: tuple[str, ...]
    reason: str
    rank: int

    @property
    def is_structural(self) -> bool:
        """No arithmetic assertion means the structural path must verify it.

        DUPLICATE_CANDIDATE is the whole residual on this dataset and has no
        arithmetic to recompute: there is no residual and no tolerance. A
        verifier built only for arithmetic could not verify the one category it
        will ever see.
        """
        return self.assertion is None


HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "maxItems": MAX_HYPOTHESES,
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": sorted(AI_ELIGIBLE_CATEGORIES)},
                    "candidate_id": {"type": "string"},
                    "assertion": {
                        "type": "object",
                        "properties": {
                            "lhs": {"type": "string"},
                            "op": {"type": "string", "enum": list(_ALLOWED_OPS)},
                            "rhs": {"type": "string"},
                        },
                        "required": ["lhs", "op", "rhs"],
                    },
                    "residual_amount": {"type": "string"},
                    "evidence_row_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["category", "candidate_id", "evidence_row_ids", "reason"],
            },
        }
    },
    "required": ["hypotheses"],
}


class IneligibleCategoryError(ValueError):
    """An exception outside PDD 6.2 was offered to the model."""


def eligible_exceptions(
    exceptions: tuple[Exception_, ...], sendable: frozenset[str] = AI_ELIGIBLE_CATEGORIES
) -> tuple[Exception_, ...]:
    """The exceptions a model may see, sorted by (-amount, exception_id).

    A FILTER WITH A NAMED SET, not a default. Callers pass the set they mean;
    on this dataset the wiring passes DUPLICATE_CANDIDATE alone, so Population
    B's unlinked batches cannot reach the model even though PDD 6.2 lists
    MISSING_VS_LATE_CREDIT as interpretive. A batch whose credit never arrived
    is missing DATA - there is nothing to interpret, and asking anyway invites
    a fabricated explanation for an absence.
    """
    return tuple(
        sorted(
            (exc for exc in exceptions if exc.category in sendable),
            key=lambda exc: (-exc.amount, exc.exception_id),
        )
    )


def build_prompt(exception: Exception_, dataset: DayDataset, config: AppConfig) -> str:
    """A byte-identical prompt for the same exception, every time.

    EVERY interpolated collection is sorted. An unsorted set or a dict rendered
    in insertion order would produce a different prompt on a different run,
    which misses the fixture cache and turns the replay mechanism - the thing
    that makes this reproducible - into a source of nondeterminism.

    Decimals are rendered with `str`, never through float.
    """
    if exception.category not in AI_ELIGIBLE_CATEGORIES:
        raise IneligibleCategoryError(
            f"{exception.category!r} is not in PDD 6.2's interpretive set "
            f"{sorted(AI_ELIGIBLE_CATEGORIES)}. Rules decide it, and sending it "
            "to a model invites an explanation for something already explained."
        )

    evidence = sorted(exception.evidence_row_ids)
    rows = _evidence_rows(evidence, dataset)
    lines = [
        "You are reconciling merchant settlement data. Decide nothing; propose",
        "at most three ranked hypotheses that a deterministic verifier will",
        "check against the data without your help.",
        "",
        f"EXCEPTION {exception.exception_id}",
        f"  category   {exception.category}",
        f"  amount     {exception.amount}",
        f"  reason     {exception.reason}",
        f"  first seen day {exception.first_seen_day}",
        "",
        "EVIDENCE ROWS (sorted by id):",
    ]
    lines.extend(rows or ["  (none resolvable in the loaded dataset)"])
    lines += [
        "",
        f"TOLERANCE {config.thresholds.tolerance.verifier_rupees}",
        f"CALENDAR {config.calendar.version}",
        "",
        "Return JSON matching the tool schema. Rank best first.",
        "",
        "`candidate_id` MUST BE EXACTLY ONE of the evidence ids listed above -",
        "the single row you believe is the duplicate entry. Not both, not a",
        "concatenation of them, not a new identifier. A candidate_id that is not",
        "one of those ids is rejected without being read.",
        "",
        "Every evidence_row_id must be an id that appears above. If the rows do",
        "not distinguish the candidates, say so in `reason` and return fewer",
        "hypotheses - a guess the verifier cannot check will be rejected.",
    ]
    return "\n".join(lines)


def _evidence_rows(evidence_ids: list[str], dataset: DayDataset) -> list[str]:
    """One rendered line per resolvable id, in sorted id order."""
    index = _row_index(dataset)
    rendered: list[str] = []
    for row_id in evidence_ids:
        row = index.get(row_id)
        if row is None:
            continue
        fields = ", ".join(
            f"{name}={value}"
            for name, value in sorted(_public_fields(row).items())
            if value is not None
        )
        rendered.append(f"  {row_id}: {type(row).__name__}({fields})")
    return rendered


def _public_fields(row: object) -> dict[str, object]:
    import dataclasses

    if not dataclasses.is_dataclass(row):
        return {}
    return {f.name: getattr(row, f.name) for f in dataclasses.fields(row)}


def _row_index(dataset: DayDataset) -> dict[str, object]:
    """Every row by its own id. Built once per prompt, sorted where it matters."""
    # Distinct loop names per table: reusing `row` makes mypy infer the dict's
    # value type from the first assignment and reject every table after it.
    index: dict[str, object] = {}
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


def parse_hypotheses(payload: object) -> tuple[Hypothesis, ...]:
    """Validate against SDD 4.4 BEFORE use. Invalid items are dropped, not fixed.

    A hypothesis with an unknown category, a malformed assertion or a
    non-Decimal residual is discarded rather than repaired: repairing it means
    guessing what the model meant, which is the model deciding by proxy.

    Ranking is the order the model returned. That is its one privilege - it may
    say which claim it thinks is best, and the verifier still checks them in
    that order and rejects them all if none survives.
    """
    if not isinstance(payload, dict):
        return ()
    raw = payload.get("hypotheses")
    if not isinstance(raw, list):
        return ()

    parsed: list[Hypothesis] = []
    for rank, item in enumerate(raw[:MAX_HYPOTHESES]):
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        candidate_id = item.get("candidate_id")
        reason = item.get("reason")
        evidence = item.get("evidence_row_ids")
        if category not in AI_ELIGIBLE_CATEGORIES:
            continue
        if not isinstance(candidate_id, str) or not isinstance(reason, str):
            continue
        if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
            continue
        residual = _decimal_or_none(item.get("residual_amount"))
        if item.get("residual_amount") is not None and residual is None:
            continue  # a residual that will not parse is not a residual
        parsed.append(
            Hypothesis(
                category=str(category),
                candidate_id=candidate_id,
                assertion=Assertion.from_payload(item.get("assertion")),
                residual_amount=residual,
                # SORTED: the same evidence in a different order is the same
                # claim, and downstream keys hash this tuple.
                evidence_row_ids=tuple(sorted(evidence)),
                reason=reason,
                rank=rank,
            )
        )
    return tuple(parsed)


def _decimal_or_none(value: object) -> Decimal | None:
    """str -> Decimal. Never float: a float residual decides things (D1)."""
    if value is None:
        return None
    if isinstance(value, float | bool):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def generate(
    exception: Exception_,
    dataset: DayDataset,
    config: AppConfig,
    client: LLMClient,
    max_retries: int = MAX_RETRIES,
) -> tuple[Hypothesis, ...]:
    """Up to 3 ranked hypotheses. NEVER raises on a bad reply.

    A model that will not produce schema-valid output after `max_retries` has
    not answered, which is an abstention - a legitimate outcome. Crashing would
    lose every other exception in the run over one malformed reply.
    """
    prompt = build_prompt(exception, dataset, config)
    for _attempt in range(max_retries + 1):
        try:
            payload = client.complete(prompt, HYPOTHESIS_SCHEMA)
        except json.JSONDecodeError:
            continue
        hypotheses = parse_hypotheses(payload)
        if hypotheses:
            return hypotheses
    return ()

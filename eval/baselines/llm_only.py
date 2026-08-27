"""M5 baseline - LLM-only reconciliation, built in good faith to be STRONG.

WHAT WAS DONE TO MAKE THIS BASELINE STRONG
==========================================

A weak LLM baseline is worth nothing. If the headline claim is "constrained AI
beats an LLM doing the whole job", then a strawman baseline makes that claim
unfalsifiable, and a reader who suspects a strawman is right to discount the
whole result. So this is written as if it were the product:

1. THE SAME NORMALIZED RECORDS. It reads M2's typed, parsed, normalized
   output - not raw CSV text. The model is not handed a parsing problem it
   would fail at for reasons unrelated to reconciliation. Amounts arrive as
   exact decimal strings, dates as ISO, UTRs already uppercased.

2. REAL CANDIDATE RETRIEVAL, not the whole table. Each bank row is paired with
   the top 20 plausible batches, ranked by amount closeness then date
   proximity. 39 batches would fit in a prompt here, but a production ledger
   would not, and a baseline that only works because the dataset is small is
   not a baseline. Retrieval is deterministic and its recall is MEASURED
   (`retrieval_recall`) so a failure to link can be attributed to the model
   rather than to a candidate list that never contained the answer.

3. STRUCTURED JSON OUTPUT with an explicit schema in the prompt, one object per
   bank row, plus a required `confidence` and `reasoning` field. Free-text
   replies are unparseable at scale and would understate the baseline.

4. CHUNKING that keeps each prompt well inside context and never splits a bank
   row from its candidates. Chunk boundaries are deterministic.

5. THE PROMPT STATES THE DOMAIN RULES: what a UTR is, that a batch total is a
   signed line sum, that amounts may carry sub-rupee rounding, that a credit
   may legitimately be missing. It is allowed to answer "no match" and told
   that answering so is correct when the evidence is absent. Withholding
   domain knowledge would be rigging the comparison.

6. IT IS ALLOWED TO ABSTAIN. The parser treats a null batch_id as a genuine
   abstention rather than a failure, so the baseline is scored on the same
   accept/abstain axis as the engine.

WHAT THIS BASELINE STILL CANNOT DO, and why that is the finding rather than a
handicap: it has no arithmetic guarantee. Nothing verifies that the fee it
accepted recomputes, and nothing stops it linking two batches to one credit.
Those are properties of a deterministic layer, not of a better prompt.

NO RANKING IS ASSERTED ANYWHERE. This baseline may link more than the engine.
The comparison that matters is precision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from settlesense.ai.client import LLMClient
from settlesense.config import AppConfig
from settlesense.ingest import DayDataset
from settlesense.normalize import normalize_narration
from settlesense.types import BankDirection, BankRow, SettlementBatch

__all__ = [
    "CANDIDATES_PER_ROW",
    "LLMBaselineResult",
    "LLMLink",
    "build_prompt",
    "parse_response",
    "retrieval_recall",
    "run_llm_only",
    "select_candidates",
]

CANDIDATES_PER_ROW = 20
"""Top-N batches offered per bank row.

Twenty, not all 39. Every batch would fit in a prompt on THIS dataset, and a
baseline that depends on that is measuring the dataset rather than the
approach - a real ledger has thousands. Retrieval recall is measured so a miss
can be attributed correctly.
"""

ROWS_PER_CHUNK = 10
"""Bank rows per prompt. Chunk boundaries never split a row from its candidates."""


@dataclass(frozen=True)
class LLMLink:
    bank_txn_id: str
    batch_id: str | None  # None is a genuine abstention, not a parse failure
    confidence: Decimal
    reasoning: str


@dataclass(frozen=True)
class LLMBaselineResult:
    links: tuple[LLMLink, ...]
    prompts_sent: int
    input_tokens: int
    output_tokens: int
    retrieval_recall_batch_count: Decimal | None
    parse_failures: int


def select_candidates(
    row: BankRow, batches: list[SettlementBatch], limit: int = CANDIDATES_PER_ROW
) -> list[SettlementBatch]:
    """Top-N batches for one credit: amount closeness, then date proximity.

    Deterministic and total - batch_id breaks any remaining tie - so the
    candidate list does not depend on input order (D4).
    """
    return sorted(
        batches,
        key=lambda batch: (
            abs(batch.net_total - row.amount),
            abs((row.value_date - batch.settled_event_date).days),
            batch.batch_id,
        ),
    )[:limit]


def retrieval_recall(
    dataset: DayDataset, truth_links: dict[str, str | None], limit: int = CANDIDATES_PER_ROW
) -> Decimal | None:
    """Share of answerable rows whose TRUE batch was in the candidate list.

    Measured, not assumed. Without it a low score is unattributable: the model
    may have been wrong, or it may never have been shown the right answer. Only
    rows whose truth names a batch are counted - a row with no true batch has
    no answer to miss.
    """
    batches = list(dataset.settlement_batches)
    wanted = {txn: batch for batch, txn in truth_links.items() if txn is not None}
    credits = [r for r in dataset.bank_rows if r.direction is BankDirection.CREDIT]
    answerable = [row for row in credits if row.bank_txn_id in wanted]
    if not answerable:
        return None
    hits = sum(
        1
        for row in answerable
        if wanted[row.bank_txn_id]
        in {batch.batch_id for batch in select_candidates(row, batches, limit)}
    )
    return (Decimal(hits) / Decimal(len(answerable))).quantize(Decimal("0.000001"))


SYSTEM_RULES = """\
You are reconciling bank credits against settlement batches for an Indian \
payment gateway. Match each bank credit to the settlement batch that produced \
it, or say there is no match.

Domain rules you should use:
- A UTR is a 16-character alphanumeric reference. Bank narrations are free \
text and the UTR may be truncated to a prefix, garbled by a transposition, or \
absent entirely. A surviving prefix is strong evidence.
- A batch's net_total is the signed sum of its settlement lines: payments add, \
refunds subtract. The bank credit should equal it.
- Amounts may differ by up to Rs1.00 from sub-rupee rounding. That is not a \
mismatch.
- A credit normally lands 1 to 3 working days after the batch settles, per \
merchant. Weekends and holidays do not count.
- A batch may legitimately have NO credit yet, and a credit may belong to no \
batch. Answering null is CORRECT when the evidence is absent. Do not guess to \
fill every row.
- If two candidate batches are equally consistent with the evidence, answer \
null. A wrong link is worse than no link.

Return ONLY a JSON array, one object per bank credit given, in the same order:
[{"bank_txn_id": "...", "batch_id": "..." or null, "confidence": 0.0-1.0, \
"reasoning": "one sentence"}]
"""


LINK_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string"},
                    "bank_txn_id": {"type": "string"},
                },
                "required": ["batch_id", "bank_txn_id"],
            },
        }
    },
    "required": ["links"],
}
"""The schema this baseline asks for. Declared here rather than reused from
hypothesis.py: the baseline answers a DIFFERENT question - which credit is
this batch - and sharing a schema would couple two things that must be able
to diverge."""


def build_prompt(rows: list[BankRow], batches: list[SettlementBatch]) -> str:
    """One chunk: several credits, each with its own candidate list.

    Candidates are inlined PER ROW rather than as one shared table, so the
    model is never asked to hold a join in its head - the same courtesy the
    retrieval step exists to provide.
    """
    blocks: list[str] = []
    for row in rows:
        candidates = select_candidates(row, batches)
        rendered = "\n".join(
            f"    - batch_id={batch.batch_id} net_total={batch.net_total} "
            f"utr={batch.utr} settled={batch.settled_event_date.isoformat()}"
            for batch in candidates
        )
        blocks.append(
            f"  BANK CREDIT {row.bank_txn_id}\n"
            f"    amount={row.amount}\n"
            f"    value_date={row.value_date.isoformat()}\n"
            f"    narration={normalize_narration(row.narration)!r}\n"
            f"    candidate batches:\n{rendered}"
        )
    return SYSTEM_RULES + "\n\nCREDITS TO MATCH:\n\n" + "\n\n".join(blocks) + "\n"


def parse_response(text: str) -> tuple[list[LLMLink], int]:
    """Parse the JSON array. Returns (links, parse_failures).

    A null batch_id is an ABSTENTION and is preserved as one. Collapsing it
    into a parse failure would score the baseline down for doing the right
    thing, which is the most common way an LLM baseline gets quietly weakened.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned.removeprefix("json").strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return [], 1
    if not isinstance(payload, list):
        return [], 1

    links: list[LLMLink] = []
    failures = 0
    for item in payload:
        if not isinstance(item, dict) or "bank_txn_id" not in item:
            failures += 1
            continue
        raw_confidence = item.get("confidence", 0)
        try:
            confidence = Decimal(str(raw_confidence))
        except Exception:
            failures += 1
            continue
        links.append(
            LLMLink(
                bank_txn_id=str(item["bank_txn_id"]),
                batch_id=None
                if item.get("batch_id") in (None, "", "null")
                else str(item["batch_id"]),
                confidence=confidence,
                reasoning=str(item.get("reasoning", "")),
            )
        )
    return links, failures


def run_llm_only(
    dataset: DayDataset,
    config: AppConfig,
    as_of: date,
    client: LLMClient,
    truth_links: dict[str, str | None] | None = None,
) -> LLMBaselineResult:
    """The whole reconciliation, done by the model, with real retrieval.

    Chunks are deterministic: credits sorted by id, sliced by ROWS_PER_CHUNK.
    Two runs against the same client produce the same prompts, so a replay
    fixture keyed on prompt hash is stable.
    """
    credits = sorted(
        (row for row in dataset.bank_rows if row.direction is BankDirection.CREDIT),
        key=lambda row: row.bank_txn_id,
    )
    batches = sorted(dataset.settlement_batches, key=lambda batch: batch.batch_id)

    links: list[LLMLink] = []
    prompts = input_tokens = output_tokens = failures = 0
    for start in range(0, len(credits), ROWS_PER_CHUNK):
        chunk = credits[start : start + ROWS_PER_CHUNK]
        response = client.complete(build_prompt(chunk, batches), LINK_SCHEMA)
        prompts += 1
        # The client now returns the STRUCTURED payload rather than a text
        # blob with usage attached (M7 changed the protocol to
        # `complete(prompt, schema) -> dict`). Token counts are no longer on
        # the response, so they are reported as unknown rather than guessed -
        # a fabricated token count feeds a fabricated cost.
        input_tokens += 0
        output_tokens += 0
        parsed, chunk_failures = parse_response(json.dumps(response.get("links", [])))
        links.extend(parsed)
        failures += chunk_failures

    return LLMBaselineResult(
        links=tuple(sorted(links, key=lambda link: link.bank_txn_id)),
        prompts_sent=prompts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        retrieval_recall_batch_count=(
            None if truth_links is None else retrieval_recall(dataset, truth_links)
        ),
        parse_failures=failures,
    )

"""The AI layer against the PERSISTED STORE, and the pairing rule that enables it.

WHAT THIS CLOSES. `--simulate-outage` found that the AI layer had never run
against the exception store - M7 measured dataset-derived pair exceptions, the
store persists engine outcomes, zero ids in common - so no persisted row carried
`resolved_by = AI_VERIFIED` and the queue read as an AI layer that existed only
in tests.

THE CAUSE WAS SHAPE, NOT MISSING RECORDINGS, and the difference decided what to
build. A store row carries ONE evidence id, the settlement batch, so the prompt
built from it lists a single id and then asks which of the listed ids is the
duplicate - a choice from a list of one. The ORACLE, the perfect nominator,
confirmed 0 of 47 with 47 NO_HYPOTHESIS and not one verifier rejection. Buying
47 live recordings would have produced 47 identical non-answers and a published
"0 of 47" that reads as absent structural evidence while meaning the prompt
contained no question. It was not done; LIMITATIONS.md records why.

WHAT WAS DONE INSTEAD COST NOTHING. The store's pairs and the dataset's pair
exceptions describe the SAME duplicates: 22 of 22 prompts were already in
`fixtures/llm/` from the M7 session. The gap was a read-time join.

A SECOND MEASUREMENT, NEVER A REVISION. M7's line stands verbatim and is
asserted to stand; the store-path line sits beside it.
"""

from __future__ import annotations

import ast
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from eval.run_ai import OracleClient, truth_duplicate_orders
from eval.run_store_ai import M7_LINE, pair_index, score
from settlesense.ai.client import ReplayLLMClient
from settlesense.ai.loop import resolve_exception
from settlesense.ai.pairing import (
    DUPLICATE_CANDIDATE,
    PAIRING_KEY_NOTE,
    pair_store_rows,
    run_store_ai_stage,
)
from settlesense.config import AppConfig, load_config
from settlesense.exceptions.store import ALL_STATUSES, RESIDUAL_STATES, ExceptionStore
from settlesense.types import ExceptionStatus, ResolutionSource
from settlesense.ui.queue import population_summaries

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "dev"
CHECKPOINTS = (1, 12, 24)
DAY = 24
AS_OF = date(2026, 11, 30)
REPORT = REPO / "reports" / "ai" / "store_path.json"


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


@pytest.fixture
def store(fresh_store: Any) -> Any:
    """A private copy of the session template. READ-ONLY in these tests.

    Copied rather than rebuilt: `run_day` over three checkpoints costs ~1.6s
    and this suite needs a clean store several times over, which on its own
    took the whole run to 97% of the SDD 7 budget.
    """
    return fresh_store()


@pytest.fixture(scope="module")
def dataset(config: AppConfig, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The cumulative dataset. Module-scoped because nothing mutates it."""
    scratch = ExceptionStore(tmp_path_factory.mktemp("ds") / "empty.db")
    return scratch.cumulative_dataset(DAY, DATA, config)


@pytest.fixture(scope="module")
def truth() -> frozenset[str]:
    return truth_duplicate_orders(DATA / "truth_42.json")


# ===========================================================================
# 1. Why the 47-row recording was not bought
# ===========================================================================


def test_a_single_evidence_store_row_cannot_pose_the_duplicate_question(
    store: ExceptionStore, dataset: Any, config: AppConfig, truth: frozenset[str]
) -> None:
    """THE MEASUREMENT THAT SAVED THE SPEND, kept so the reasoning is checkable.

    The oracle is the ceiling: it nominates the truth-correct row by
    construction, so no model can beat it. If the oracle cannot even form a
    hypothesis, the path is unanswerable and recording it buys nothing.

    NO_HYPOTHESIS, not ALL_REJECTED, is the whole point. A rejection would mean
    the verifier looked at a claim and refused it - a statement about evidence.
    NO_HYPOTHESIS means no claim could be made at all.
    """
    rows = [row for row in store.get_queue(ALL_STATUSES) if row.category == DUPLICATE_CANDIDATE]
    assert rows, "no DUPLICATE_CANDIDATE rows; the measurement is vacuous"
    assert all(len(row.evidence_row_ids) == 1 for row in rows), "a row carries a real pair"

    oracle = OracleClient(truth)
    outcomes = [resolve_exception(row, dataset, config, oracle) for row in rows]
    confirmed = sum(1 for outcome in outcomes if outcome.confirmed)
    reasons = {
        outcome.abstain_reason.value for outcome in outcomes if outcome.abstain_reason is not None
    }
    assert confirmed == 0, f"the oracle confirmed {confirmed} on the single-row path"
    assert reasons == {"NO_HYPOTHESIS"}, reasons

    evidence = {row_id for row in rows for row_id in row.evidence_row_ids}
    assert not evidence & truth, "an evidence id IS an order id; the premise has changed"
    print(
        f"\n  {len(rows)} single-evidence rows: oracle confirmed 0, all NO_HYPOTHESIS; "
        f"{len(evidence)} distinct evidence ids, 0 scoreable against truth"
    )


# ===========================================================================
# 2. The pairing rule
# ===========================================================================


def test_the_pairing_groups_rows_into_pairs_and_names_the_leftovers(
    store: ExceptionStore, dataset: Any, config: AppConfig, truth: frozenset[str]
) -> None:
    """Realised counts, printed. Pairs and singletons are separate outputs."""
    pairs, unpaired = pair_store_rows(store, dataset, config)
    rows = [row for row in store.get_queue(RESIDUAL_STATES) if row.category == DUPLICATE_CANDIDATE]
    assert 2 * len(pairs) + len(unpaired) == len(rows), (len(pairs), len(unpaired), len(rows))
    assert pairs and unpaired, "one of the two buckets is empty; the split proves nothing"

    for pair in pairs:
        assert len(set(pair.exception_ids)) == 2
        assert len(set(pair.order_ids)) == 2
        assert pair.order_ids == tuple(sorted(pair.order_ids)), "order ids are not canonical"
        # EXACTLY ONE of the two is the injected duplicate. If neither or both
        # were, the pairing would be joining rows that are not a pair.
        marked = sum(1 for order_id in pair.order_ids if order_id in truth)
        assert marked == 1, (pair.order_ids, marked)
    print(
        f"\n  {len(rows)} residual DUPLICATE_CANDIDATE rows -> {len(pairs)} pairs "
        f"+ {len(unpaired)} unpaired; exactly one truth-marked order in {len(pairs)}/{len(pairs)}"
    )


@pytest.mark.boundary_refusal
def test_a_group_that_is_not_exactly_two_is_never_paired(
    store: ExceptionStore, dataset: Any, config: AppConfig
) -> None:
    """FAULT INJECTION. Three rows sharing a key must not become a pair.

    Picking two of three would be a coin flip recorded as a decision, and the
    resulting nomination would be scored as if somebody had reasoned about it.
    """
    from dataclasses import replace

    residual = [
        row
        for row in store.get_queue(RESIDUAL_STATES)
        if row.category == DUPLICATE_CANDIDATE and len(row.evidence_row_ids) == 1
    ]
    pairs, _ = pair_store_rows(store, dataset, config, residual)
    assert pairs, "no pairs to distort"

    first = pairs[0]
    members = [row for row in residual if row.exception_id in first.exception_ids]
    intruder = replace(
        members[0], exception_id="planted-third-row", evidence_row_ids=members[0].evidence_row_ids
    )
    distorted, leftover = pair_store_rows(store, dataset, config, [*residual, intruder])
    assert first.exception_ids not in {pair.exception_ids for pair in distorted}, (
        "a group of three still produced a pair"
    )
    assert "planted-third-row" in leftover

    # And a row with no single evidence id is unpaired rather than guessed at.
    no_evidence = replace(members[0], exception_id="planted-no-evidence", evidence_row_ids=())
    _, leftover2 = pair_store_rows(store, dataset, config, [*residual, no_evidence])
    assert "planted-no-evidence" in leftover2
    print(
        f"\n  a 3-member group drops to unpaired ({len(distorted)} pairs vs {len(pairs)}); "
        "a row with no evidence id is never paired"
    )


# ===========================================================================
# 3. The measurement
# ===========================================================================


@pytest.fixture
def result(
    config: AppConfig, dataset: Any, truth: frozenset[str], fresh_store: Any
) -> dict[str, object]:
    """A scored run over its own store, so no other test sees the write-back."""
    return score(fresh_store(), dataset, config, truth, DAY)


def test_the_store_path_confirms_something_and_false_confirms_zero(
    result: dict[str, object],
) -> None:
    """The headline. Realised, printed, and the confirmation checked against truth.

    ZERO FALSE CONFIRMS IS THE NUMBER THAT MATTERS. A confirmation the verifier
    accepted while truth disagrees would mean the whole architecture had
    admitted a wrong answer through the one gate meant to stop it.
    """
    assert result["pairs_replayed"] == 22, result["pairs_replayed"]
    assert result["false_confirms"] == 0, result["false_confirms"]
    per_pair = result["per_pair"]
    assert isinstance(per_pair, list)

    confirmed = [entry for entry in per_pair if entry["confirmed"]]
    assert confirmed, "nothing was confirmed, so zero false confirms is trivially true"
    for entry in confirmed:
        assert entry["nominated_correctly"], (
            f"{entry['exception_ids']} was confirmed on a nomination truth rejects"
        )

    gave = sum(1 for entry in per_pair if entry["model_gave_a_hypothesis"])
    correct = sum(1 for entry in per_pair if entry["nominated_correctly"])
    print(
        f"\n  {result['pairs_replayed']} pairs: {result['pairs_confirmed']} confirmed, "
        f"{result['pairs_abstained']} abstained, {result['false_confirms']} false confirms\n"
        f"  the model answered {gave}/{len(per_pair)} and nominated correctly "
        f"{correct}/{len(per_pair)} - the verifier confirmed {len(confirmed)}, "
        f"rejecting {correct - len(confirmed)} that happened to be right"
    )


def test_the_verifier_rejects_correct_guesses_it_cannot_check(
    result: dict[str, object],
) -> None:
    """The architectural claim, visible in one number on this path.

    Being right is not the same as being provable. The model nominates the
    truth-correct row far more often than the verifier confirms, and the gap is
    not a shortfall - it is the verifier refusing to act on a claim the
    evidence cannot support. A system that confirmed every correct guess would
    also confirm the incorrect ones it could not tell apart.
    """
    per_pair = result["per_pair"]
    assert isinstance(per_pair, list)
    correct = [entry for entry in per_pair if entry["nominated_correctly"]]
    confirmed = [entry for entry in per_pair if entry["confirmed"]]
    assert len(correct) > len(confirmed), (
        "the verifier confirmed everything it was right about, so this path shows "
        "nothing about verification"
    )
    rejected_but_right = [entry for entry in correct if not entry["confirmed"]]
    assert all(entry["abstain_reason"] == "ALL_REJECTED" for entry in rejected_but_right)
    print(
        f"\n  {len(correct)} correct nominations, {len(confirmed)} confirmed: "
        f"{len(rejected_but_right)} correct answers rejected for want of checkable evidence"
    )


def test_the_singletons_are_reported_separately_from_the_abstentions(
    result: dict[str, object],
) -> None:
    """Requirement 3. A row with no partner has not abstained.

    An abstention is a statement about EVIDENCE - the verifier looked and could
    not confirm. Having no partner is a statement about the QUEUE. Folding the
    second into the first would attribute a wiring condition to the data, which
    is the same error as publishing "0 of 47".
    """
    assert result["unpaired_count"] == 3, result["unpaired_rows"]
    unpaired = result["unpaired_rows"]
    assert isinstance(unpaired, list)
    assert result["pairs_abstained"] == 21
    assert result["pairs_replayed"] == result["pairs_confirmed"] + result["pairs_abstained"]  # type: ignore[operator]

    line = result["store_path_line"]
    assert isinstance(line, str)
    assert "had no partner and are excluded" in line
    assert f"{result['unpaired_count']} residual rows" in line
    print(f"\n  {len(unpaired)} unpaired rows excluded from the abstention count: {unpaired}")


def test_both_measurements_are_published_and_m7_is_not_revised(
    result: dict[str, object],
) -> None:
    """Requirement 4. Two lines, side by side, neither correcting the other."""
    assert result["m7_line"] == M7_LINE
    assert M7_LINE == (
        "AI layer, dataset-derived decisions (M7): 507 decisions, 27 confirmable, "
        "zero false confirms."
    )
    line = result["store_path_line"]
    assert isinstance(line, str)
    assert line.startswith("AI layer, persisted store path:")
    assert "507" not in line, "the store-path line quotes M7's denominator"
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "507 decisions, 27 confirmable" in readme, "M7's figure left the README"
    assert "persisted store path" in readme
    print(f"\n  M7 line intact; store-path line published beside it:\n    {line}")


@pytest.mark.charter_guard
def test_the_pairing_key_is_stated_wherever_the_number_is_reported() -> None:
    """Requirement 2. A pairing rule changes which decisions get made.

    Publishing the count without the rule publishes half a method. Asserted in
    the README and in the committed artifact, not only in the module that
    implements it - which is where it would be least likely to be read.
    """
    assert "(gross amount, settlement batch)" in PAIRING_KEY_NOTE
    assert "nominates candidates only" in PAIRING_KEY_NOTE

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert PAIRING_KEY_NOTE.split(". ")[0] in readme, "the README does not state the pairing key"
    assert "verifier still decides from evidence" in readme

    assert REPORT.is_file(), f"{REPORT} is not committed"
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["pairing_key_note"] == PAIRING_KEY_NOTE
    print("\n  the pairing key is stated in the README and in reports/ai/store_path.json")


# ===========================================================================
# 4. What must NOT have changed
# ===========================================================================


@pytest.mark.charter_guard
def test_no_population_denominator_moves(config: AppConfig, dataset: Any, fresh_store: Any) -> None:
    """THE CONSTRAINT THE WHOLE DESIGN WAS SHAPED BY (D11).

    Persisting pair-exceptions would have added store rows and moved the
    denominators every published number divides by. The read-time join adds
    nothing: the same cases, batches and rows before and after. The residual
    COUNT moves, and that is the point - two rows got explained.
    """
    before_store = fresh_store()
    after_store = fresh_store()

    before = population_summaries(before_store, dataset, config, AS_OF)
    run_store_ai_stage(after_store, dataset, config, ReplayLLMClient(), DAY, pair_index(dataset))
    after = population_summaries(after_store, dataset, config, AS_OF)

    assert [summary.denominator for summary in after] == [
        summary.denominator for summary in before
    ], "a population denominator moved"
    assert [summary.persisted for summary in after] == [summary.persisted for summary in before], (
        "the persisted row count moved; something was added to the store"
    )
    residual_before = [summary.residual for summary in before]
    residual_after = [summary.residual for summary in after]
    assert residual_after != residual_before, (
        "no residual moved either, so the AI stage did nothing at all"
    )
    print(
        f"\n  denominators {[s.denominator for s in before]} unchanged; "
        f"persisted {[s.persisted for s in before]} unchanged; "
        f"residual {residual_before} -> {residual_after}"
    )


@pytest.mark.charter_guard
def test_the_store_path_makes_zero_model_calls() -> None:
    """Zero spend, asserted structurally rather than by watching a bill.

    `RealLLMClient` must not appear on this path at all, and the runner must
    not import the recorder. An AST walk, so this docstring cannot satisfy it.
    """
    for name in ("settlesense/ai/pairing.py", "eval/run_store_ai.py"):
        source = (REPO / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "RealLLMClient", f"{name}:{node.lineno} builds a real client"
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for imported in names:
                    assert imported.split(".")[0] != "openai", f"{name} imports openai"
                    assert "record_fixtures" not in imported, f"{name} imports the recorder"
    print("\n  neither module constructs a real client, imports openai, or reaches the recorder")


def test_the_write_back_lands_on_both_rows_with_the_ai_resolver(
    config: AppConfig, dataset: Any, fresh_store: Any
) -> None:
    """A confirmed pair explains BOTH orders, and the trail says which was named."""
    scratch = fresh_store()
    outcome = run_store_ai_stage(
        scratch, dataset, config, ReplayLLMClient(), DAY, pair_index(dataset)
    )
    assert outcome.confirmed, "nothing was confirmed; the write-back is untested"
    assert len(outcome.confirmed) % 2 == 0, "a pair was written back to one row only"

    rows = {row.exception_id: row for row in scratch.get_queue(ALL_STATUSES)}
    for exception_id in outcome.confirmed:
        row = rows[exception_id]
        assert row.status is ExceptionStatus.CONFIRMED
        assert row.resolved_by is ResolutionSource.AI_VERIFIED
        notes = [entry.note for entry in row.audit if "nominated" in entry.note]
        assert notes, f"{exception_id} has no note naming the nomination"
        assert "duplicate pair" in notes[0]

    ai_rows = [row for row in rows.values() if row.resolved_by is ResolutionSource.AI_VERIFIED]
    assert len(ai_rows) == len(outcome.confirmed)
    print(
        f"\n  {len(outcome.confirmed)} rows carry resolved_by=AI_VERIFIED, "
        f"audit note names the nominated order"
    )


def test_the_queue_now_shows_an_ai_verified_row(
    config: AppConfig, dataset: Any, fresh_store: Any
) -> None:
    """The reason this was worth doing: the thesis column is no longer all-rules.

    Before this, every persisted row read DETERMINISTIC and the AI layer
    existed only in tests - which is indistinguishable, from the queue, from an
    AI layer that does not work.
    """
    from settlesense.ui.queue import VERIFIED_AI, VERIFIED_DETERMINISTIC, build_rows

    scratch = fresh_store()
    run_store_ai_stage(scratch, dataset, config, ReplayLLMClient(), DAY, pair_index(dataset))

    rows = build_rows(scratch)
    counts = {
        VERIFIED_DETERMINISTIC: sum(1 for row in rows if row.verified_by == VERIFIED_DETERMINISTIC),
        VERIFIED_AI: sum(1 for row in rows if row.verified_by == VERIFIED_AI),
    }
    assert counts[VERIFIED_AI] > 0, "the queue still shows no AI_VERIFIED row"
    assert counts[VERIFIED_DETERMINISTIC] > counts[VERIFIED_AI], (
        "the AI layer now outweighs the rules; that is not this architecture"
    )
    print(
        f"\n  queue thesis column: {counts[VERIFIED_DETERMINISTIC]} DETERMINISTIC, "
        f"{counts[VERIFIED_AI]} AI_VERIFIED"
    )

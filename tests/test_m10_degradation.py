"""M10 - what happens when the model is down, and what must NOT happen.

THE CLAIM THIS FILE EXISTS TO CHECK. A model outage must not perturb a single
rule-resolved case. That is the whole architectural argument - rules first, the
model only on what rules could not close - and it is worthless as a sentence.
So it is checked as BYTES: the deterministic rows are serialized before and
after an outage run and compared, through the same canonical serializer M5a
used to prove telemetry did not leak into results.json. A matching count is a
weaker claim and would pass if two rows had swapped statuses.

ZERO, NOT FEW. Nothing is confirmed during an outage, and the count is asserted
as exactly 0 and printed. "Few" is what you write when you have not checked.

THE TWO WAITING STATES ARE DIFFERENT THINGS. PENDING_EVIDENCE means a file has
not arrived; PENDING_AI_UNAVAILABLE means a service failed. Different colours
since 2a744b0, different reasons here, and different next actions - wait, or go
and look at why the model is down. They are asserted distinct in both.

WHAT THIS SUITE FOUND, recorded because it changes what the demo can claim: the
store's AI-eligible residual rows have ZERO recorded fixtures. M7 was measured
over dataset-derived pair exceptions (`dup-ORD_A-ORD_B`); the store persists
engine outcomes under hash ids. The two sets are disjoint. The outage is still
real - `OutageLLMClient` raises before any cache lookup, and the two failures
take provably different paths - but what a healthy run would have CONCLUDED for
those rows is unknown, and `probe()` says so out loud rather than letting the
screenshot imply otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from settlesense.ai.client import (
    MAX_ATTEMPTS,
    FixtureMissError,
    ModelUnavailable,
    OutageLLMClient,
    ReplayLLMClient,
    prompt_hash,
    retry_until_unavailable,
)
from settlesense.ai.loop import AbstainReason, resolve_exception, run_loop
from settlesense.ai.orchestrator import (
    OUTAGE_REASON,
    PENDING_EVIDENCE_REASON,
    run_ai_stage,
)
from settlesense.config import AppConfig, load_config
from settlesense.core.serialize import result_hash, serialize_result
from settlesense.exceptions.store import (
    ALL_STATUSES,
    RESIDUAL_STATES,
    ExceptionStore,
    as_of_for_arrival_day,
)
from settlesense.matching.engine import run
from settlesense.types import ExceptionStatus, ResolutionSource
from settlesense.ui.build_state import ProbeFailed, probe
from settlesense.ui.build_state import main as build_state_main
from settlesense.ui.queue import STATUS_STYLES

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "dev"
CHECKPOINTS = (1, 12, 24)
OUTAGE_DAY = 24


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


def _store(config: AppConfig, path: Path, days: tuple[int, ...] = CHECKPOINTS) -> ExceptionStore:
    """Build a store over `days`. Used only where the day range is NOT the
    template's - everything on 1/12/24 takes a copy via `fresh_store`, because
    rebuilding cost ~1.6s a time and took the suite to 97% of the budget."""
    built = ExceptionStore(path)
    for day in days:
        built.run_day(day, DATA, config)
    return built


def _deterministic_rows(store: ExceptionStore) -> str:
    """Canonical JSON over every row the RULES decided. The comparison subject.

    Rows the model could touch are excluded by construction - this reads only
    exceptions whose `resolved_by` is DETERMINISTIC - so a difference here can
    only mean the outage reached something it had no business reaching.

    Sorted by id, not by amount: the ordering must not depend on a value the
    thing under test could change.
    """
    rows = [
        {
            "exception_id": row.exception_id,
            "status": row.status.value,
            "category": row.category,
            "amount": str(row.amount),
            "confidence": str(row.confidence),
            "resolved_by": row.resolved_by.value if row.resolved_by else None,
            "first_seen_day": row.first_seen_day,
            "confirmed_day": row.confirmed_day,
            "closed_day": row.closed_day,
            "evidence_row_ids": list(row.evidence_row_ids),
        }
        for row in store.get_queue(ALL_STATUSES)
        if row.resolved_by is ResolutionSource.DETERMINISTIC
    ]
    rows.sort(key=lambda entry: str(entry["exception_id"]))
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


class FlakyClient:
    """Fails the first `failures` calls, then serves recordings. For test 16."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls: list[str] = []
        self._replay = ReplayLLMClient()

    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(prompt_hash(prompt))
        if len(self.calls) <= self.failures:
            raise ModelUnavailable("flaky", attempts=MAX_ATTEMPTS)
        return self._replay.complete(prompt, schema)


# ===========================================================================
# 1. The retry policy, reachable at last
# ===========================================================================


@pytest.mark.boundary_refusal
def test_10_and_15_timeout_http_error_and_bad_json_all_end_in_model_unavailable() -> None:
    """SDD 4.9's three failures, each retried to MAX_ATTEMPTS then given up on.

    TESTED THROUGH THE EXTRACTED HELPER, because D7 forbids constructing
    RealLLMClient inside a test run - so while the policy lived inside
    `complete` the suite could not reach it at all and "it retries twice" was a
    claim backed by reading.
    """
    failures: dict[str, Exception] = {
        "timeout": TimeoutError("read timed out"),
        "http": ConnectionError("502 Bad Gateway"),
        "bad json": json.JSONDecodeError("Expecting value", "not json", 0),
        "empty message": RuntimeError("the model returned an empty message"),
    }
    for name, error in failures.items():
        attempts = 0

        def failing(error: Exception = error) -> dict[str, Any]:
            nonlocal attempts
            attempts += 1
            raise error

        with pytest.raises(ModelUnavailable) as caught:
            retry_until_unavailable(failing)
        assert attempts == MAX_ATTEMPTS, f"{name} was tried {attempts} times"
        assert caught.value.attempts == MAX_ATTEMPTS
        assert str(caught.value).endswith(f"(after {MAX_ATTEMPTS} attempt(s))")

    # FAULT INJECTION THE OTHER WAY: a call that succeeds on the last attempt
    # must NOT be reported as unavailable, or the policy would be "always fail".
    tries = 0

    def recovers() -> dict[str, Any]:
        nonlocal tries
        tries += 1
        if tries < MAX_ATTEMPTS:
            raise TimeoutError("still down")
        return {"hypotheses": []}

    assert retry_until_unavailable(recovers) == {"hypotheses": []}
    assert tries == MAX_ATTEMPTS
    print(f"\n  {len(failures)} failure kinds, each {MAX_ATTEMPTS} attempts; recovery on the last")


@pytest.mark.charter_guard
def test_an_outage_is_not_a_fixture_miss_and_a_fixture_miss_is_not_an_outage(
    config: AppConfig, fresh_store: Any
) -> None:
    """The two paths must not converge. This is the load-bearing distinction.

    A missing recording is a fact about this repository; an outage is a fact
    about the world. If FixtureMissError were caught as unavailable, every
    unrecorded prompt would report as a service failure and the outage numbers
    would measure the fixture set.
    """
    store = fresh_store()
    dataset = store.cumulative_dataset(OUTAGE_DAY, DATA, config)
    residual = store.get_queue(status_filter=RESIDUAL_STATES)
    subject = next(row for row in residual if row.category == "DUPLICATE_CANDIDATE")

    outage = resolve_exception(subject, dataset, config, OutageLLMClient())
    assert outage.unavailable is True
    assert outage.abstain_reason is None, "an outage recorded an abstention reason"
    assert outage.resolution_type == "PENDING_AI_UNAVAILABLE"

    miss = resolve_exception(subject, dataset, config, ReplayLLMClient())
    assert miss.unavailable is False
    assert miss.abstain_reason is AbstainReason.FIXTURE_MISS
    assert miss.resolution_type == "ABSTAINED"

    assert not issubclass(ModelUnavailable, FixtureMissError)
    assert not issubclass(FixtureMissError, ModelUnavailable)
    print(
        f"\n  same exception {subject.exception_id}: outage -> "
        f"{outage.resolution_type}, no recording -> {miss.resolution_type}"
    )


# ===========================================================================
# 2. The outage run itself
# ===========================================================================


@pytest.mark.charter_guard
def test_11_12_13_an_outage_confirms_exactly_zero_and_pends_the_rest(
    config: AppConfig, fresh_store: Any
) -> None:
    """Requirements 11, 12 and 13 together, on one realised run.

    They belong together because each is only meaningful with the others: zero
    confirmations is trivial if nothing was sent, and "rows are pending" is
    trivial if the deterministic rows moved too.
    """
    store = fresh_store()
    dataset = store.cumulative_dataset(OUTAGE_DAY, DATA, config)
    before_confirmed = {
        row.exception_id
        for row in store.get_queue(ALL_STATUSES)
        if row.status is ExceptionStatus.CONFIRMED
    }
    client = OutageLLMClient()
    result = run_ai_stage(store, dataset, config, client, OUTAGE_DAY)

    assert result.report.sent > 0, "nothing was sent, so zero confirmations proves nothing"
    assert result.report.confirmed == 0, f"{result.report.confirmed} confirmed during an outage"
    assert len(result.confirmed) == 0
    assert result.report.unavailable == result.report.sent
    assert result.report.abstained == 0, "an outage was counted as an abstention"

    rows = {row.exception_id: row for row in store.get_queue(ALL_STATUSES)}
    for exception_id in result.pending_unavailable:
        assert rows[exception_id].status is ExceptionStatus.PENDING_AI_UNAVAILABLE, (
            f"{exception_id} is {rows[exception_id].status}"
        )
    after_confirmed = {
        row.exception_id for row in rows.values() if row.status is ExceptionStatus.CONFIRMED
    }
    assert after_confirmed == before_confirmed, "the confirmed set moved during an outage"
    assert all(row.status is not ExceptionStatus.CLOSED for row in rows.values())
    print(
        f"\n  {result.report.sent} sent, {result.report.confirmed} confirmed (exactly 0), "
        f"{len(result.pending_unavailable)} PENDING_AI_UNAVAILABLE, "
        f"{len(before_confirmed)} deterministic confirmations untouched"
    )


@pytest.mark.charter_guard
def test_24_the_deterministic_result_is_byte_identical_with_and_without_the_outage(
    config: AppConfig, fresh_store: Any
) -> None:
    """BYTES, not a count. The strongest form of the degradation claim.

    Two comparisons, because they answer different questions:

      the ENGINE result - `serialize_result`, the same canonical serializer the
      M5a work used to prove telemetry never reached results.json. An outage
      must not change what the rules computed.

      the STORE's deterministic rows - status, category, amount, resolver,
      days, evidence. This is the one that could actually have failed: the AI
      stage writes to the store, and a bug that marked the wrong rows would
      leave the engine untouched and the store wrong.
    """
    healthy = fresh_store()
    broken = fresh_store()
    dataset = healthy.cumulative_dataset(OUTAGE_DAY, DATA, config)
    as_of = as_of_for_arrival_day(OUTAGE_DAY, config)

    before_engine = serialize_result(run(dataset, config, as_of))
    before_rows = _deterministic_rows(broken)
    assert before_rows == _deterministic_rows(healthy), "the two stores started out different"
    assert len(json.loads(before_rows)) > 0, "no deterministic rows; the comparison is vacuous"

    result = run_ai_stage(broken, dataset, config, OutageLLMClient(), OUTAGE_DAY)
    assert result.outage, "no outage occurred, so this compares a run against itself"

    after_engine = serialize_result(run(dataset, config, as_of))
    after_rows = _deterministic_rows(broken)

    assert after_engine == before_engine, "the engine result changed across an outage"
    assert result_hash(run(dataset, config, as_of)) == result_hash(run(dataset, config, as_of))
    assert after_rows == before_rows, (
        "a deterministic row moved during a model outage. The rules layer must be "
        "unreachable from the AI stage; this is the claim the architecture rests on."
    )
    assert after_rows == _deterministic_rows(healthy), "drift against the untouched store"

    # FAULT INJECTION: the comparison must be able to FAIL. Without this, a
    # serializer that returned a constant would pass every assertion above.
    victim = json.loads(after_rows)
    victim[0]["status"] = "CLOSED"
    assert json.dumps(victim, sort_keys=True, separators=(",", ":")) != after_rows

    print(
        f"\n  {len(json.loads(after_rows))} deterministic rows, byte-identical "
        f"({len(after_rows):,} bytes); engine result identical too"
    )


def test_16_a_partial_outage_loses_nothing(config: AppConfig, fresh_store: Any) -> None:
    """3 of N fail; the rest are processed; the three come back next run.

    The failing rows are the FIRST three in queue order, which is deterministic
    - `get_queue` sorts by (-amount, exception_id) - so this run is reproducible
    rather than depending on which call the flakiness happened to land on.
    """
    store = fresh_store()
    dataset = store.cumulative_dataset(OUTAGE_DAY, DATA, config)
    client = FlakyClient(failures=3)
    result = run_ai_stage(store, dataset, config, client, OUTAGE_DAY)

    assert len(result.pending_unavailable) == 3, result.pending_unavailable
    processed = len(result.confirmed) + len(result.abstained)
    assert processed == result.report.sent - 3
    assert (
        len(result.confirmed) + len(result.abstained) + len(result.pending_unavailable)
        == result.report.sent
    )

    tracked = {row.exception_id for row in store.get_queue(ALL_STATUSES)}
    assert set(result.pending_unavailable) <= tracked, "a pending row vanished from the store"
    print(
        f"\n  {result.report.sent} sent: {processed} processed, "
        f"{len(result.pending_unavailable)} pending, 0 lost"
    )


def test_14_and_17_a_later_day_picks_the_pending_rows_back_up(
    config: AppConfig, tmp_path: Path
) -> None:
    """Recovery, and the database is re-runnable afterwards.

    PENDING_AI_UNAVAILABLE is in RESIDUAL_STATES, so `reevaluate_open` reaches
    it. It does NOT shortcut to CONFIRMED - the lifecycle sends it back through
    OPEN first - which is why the audit trail is checked for both transitions
    rather than only the destination.
    """
    store = _store(config, tmp_path / "recover.db", days=(1, 12))
    dataset = store.cumulative_dataset(12, DATA, config)
    result = run_ai_stage(store, dataset, config, OutageLLMClient(), 12)
    pending = set(result.pending_unavailable)
    assert pending, "nothing went pending, so recovery proves nothing"

    run_day = store.run_day(24, DATA, config)
    assert run_day.newly_confirmed, "day 24 confirmed nothing at all"

    rows = {row.exception_id: row for row in store.get_queue(ALL_STATUSES)}
    recovered = [
        exception_id
        for exception_id in pending
        if rows[exception_id].status is ExceptionStatus.CONFIRMED
    ]
    assert recovered, "no PENDING_AI_UNAVAILABLE row was ever picked back up"
    # THE ROUTE, not just the destination. PENDING_AI_UNAVAILABLE -> CONFIRMED
    # is not a legal transition; the lifecycle sends it back through OPEN, and
    # a recovery that appeared to skip that step would mean something wrote the
    # status directly.
    hops = [str(entry.to_status) for entry in rows[recovered[0]].audit]
    after_pending = hops[hops.index("PENDING_AI_UNAVAILABLE") :]
    assert "OPEN" in after_pending, f"recovery skipped OPEN: {hops}"
    assert after_pending.index("OPEN") < after_pending.index("CONFIRMED"), hops

    still_pending = [
        row for row in rows.values() if row.status is ExceptionStatus.PENDING_AI_UNAVAILABLE
    ]
    assert store.run_day(24, DATA, config) is not None, "the store is not re-runnable"
    print(
        f"\n  {len(pending)} pending after the outage -> {len(recovered)} confirmed on day 24, "
        f"{len(still_pending)} still waiting; the store re-runs"
    )


# ===========================================================================
# 3. The two waiting states, which must never look or read alike
# ===========================================================================


@pytest.mark.charter_guard
def test_22_the_two_waiting_states_are_distinct_in_style_and_in_reason() -> None:
    """Requirement 22. Colour was fixed in 2a744b0; the REASON is fixed here.

    Both halves are needed. A shared reason string would put the entire
    distinction in a colour, which is where it lived when it was wrong, and a
    reader copying a row into a ticket would carry no distinction at all.
    """
    evidence = STATUS_STYLES[ExceptionStatus.PENDING_EVIDENCE]
    unavailable = STATUS_STYLES[ExceptionStatus.PENDING_AI_UNAVAILABLE]
    assert evidence != unavailable
    assert evidence.label != unavailable.label
    assert evidence.colour != unavailable.colour
    assert evidence.background != unavailable.background

    assert OUTAGE_REASON != PENDING_EVIDENCE_REASON
    assert not set(OUTAGE_REASON.lower().split()) >= set(PENDING_EVIDENCE_REASON.lower().split())
    assert "model" in OUTAGE_REASON and "model" not in PENDING_EVIDENCE_REASON
    print(
        f"\n  {evidence.label} {evidence.colour} vs {unavailable.label} "
        f"{unavailable.colour}; reasons differ in every word that matters"
    )


def test_both_waiting_states_render_side_by_side(config: AppConfig, tmp_path: Path) -> None:
    """The frame the screenshot uses, asserted rather than eyeballed.

    Days 1 and 12 rather than 1/12/24: by day 24 every PENDING_EVIDENCE row has
    had its credit arrive, so the two states cannot coexist. Stopping at 12
    leaves both alive, which is the only point in this run where a reader can
    see them together.
    """
    from settlesense.ui.queue import build_rows

    store = _store(config, tmp_path / "both.db", days=(1, 12))
    dataset = store.cumulative_dataset(12, DATA, config)
    run_ai_stage(store, dataset, config, OutageLLMClient(), 12)

    rows = build_rows(store)
    by_status = {
        status: [row for row in rows if row.status is status] for status in ExceptionStatus
    }
    waiting_evidence = by_status[ExceptionStatus.PENDING_EVIDENCE]
    waiting_model = by_status[ExceptionStatus.PENDING_AI_UNAVAILABLE]
    assert waiting_evidence, "no PENDING_EVIDENCE rows; the comparison is one-sided"
    assert waiting_model, "no PENDING_AI_UNAVAILABLE rows"
    assert waiting_evidence[0].style != waiting_model[0].style
    print(
        f"\n  one table: {len(waiting_model)} PENDING_AI_UNAVAILABLE and "
        f"{len(waiting_evidence)} PENDING_EVIDENCE, distinctly styled"
    )


# ===========================================================================
# 4. --simulate-outage refuses when the client was never working
# ===========================================================================


@pytest.mark.boundary_refusal
def test_25_and_26_simulate_outage_refuses_when_the_client_is_unreachable(
    config: AppConfig, tmp_path: Path, fresh_store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-zero exit, naming the difference. Fault-injected both ways.

    The failure this prevents: a demo pointed at an empty fixture directory
    produces a screen of PENDING_AI_UNAVAILABLE rows that is pixel-identical to
    a real outage. Without the probe there is nothing in the output that could
    tell the two apart.
    """
    store = fresh_store()
    dataset = store.cumulative_dataset(OUTAGE_DAY, DATA, config)

    # Reachable: the probe passes and says what it checked.
    detail = probe(store, dataset, config)
    assert "client OK" in detail

    # Unreachable: an empty fixture directory. The one misconfiguration whose
    # output is indistinguishable from the thing being demonstrated.
    empty = tmp_path / "no-fixtures"
    empty.mkdir()
    monkeypatch.setattr("settlesense.ui.build_state.FIXTURE_DIR", empty)
    with pytest.raises(ProbeFailed, match="no recordings"):
        probe(store, dataset, config)

    # And end to end, through the CLI, asserting the EXIT STATUS rather than
    # that a message appeared.
    code = build_state_main(
        [
            "--data",
            str(DATA),
            "--config",
            str(REPO / "config"),
            "--out",
            str(tmp_path / "cli.db"),
            "--days",
            "1,12",
            "--simulate-outage",
        ]
    )
    assert code == 3, f"--simulate-outage exited {code} with an unusable client"
    print("\n  probe passes when reachable; exits 3 when the fixture cache is empty")


def test_the_ai_stage_runs_only_under_the_flag(config: AppConfig, tmp_path: Path) -> None:
    """Without --simulate-outage the store is what it was before M10.

    Asserted because the alternative - running the AI stage by default - would
    have silently changed every number the M8 evidence queue publishes, and the
    committed screenshots with them.
    """
    plain = tmp_path / "plain.db"
    assert (
        build_state_main(
            [
                "--data",
                str(DATA),
                "--config",
                str(REPO / "config"),
                "--out",
                str(plain),
                "--days",
                "1,12",
            ]
        )
        == 0
    )
    store = ExceptionStore(plain)
    statuses = {row.status for row in store.get_queue(ALL_STATUSES)}
    assert ExceptionStatus.PENDING_AI_UNAVAILABLE not in statuses, (
        "the AI stage ran without the flag"
    )
    resolvers = {row.resolved_by for row in store.get_queue(ALL_STATUSES)}
    assert ResolutionSource.AI_VERIFIED not in resolvers
    print(f"\n  no flag: statuses {sorted(status.value for status in statuses)}, no AI resolver")


def test_run_loop_counts_three_outcomes_that_add_up(config: AppConfig, fresh_store: Any) -> None:
    """confirmed + abstained + unavailable == sent, and the rate excludes outages.

    `abstained` used to be `sent - confirmed`. That was correct with two
    outcomes and silently absorbs a third: every outage would have been
    reported as an abstention with no line of loop.py changing.
    """
    store = fresh_store()
    dataset = store.cumulative_dataset(OUTAGE_DAY, DATA, config)
    residual = tuple(store.get_queue(status_filter=RESIDUAL_STATES))

    down = run_loop(residual, dataset, config, OutageLLMClient())
    assert down.confirmed + down.abstained + down.unavailable == down.sent
    assert down.unavailable == down.sent and down.abstained == 0
    assert down.abstention_rate == "n/a", (
        f"a total outage reported an abstention rate of {down.abstention_rate}; the "
        "denominator includes cases nothing looked at"
    )

    up = run_loop(residual, dataset, config, ReplayLLMClient())
    assert up.unavailable == 0
    assert up.confirmed + up.abstained == up.sent
    print(
        f"\n  outage: {down.sent} sent = {down.unavailable} unavailable, rate n/a; "
        f"replay: {up.sent} sent = {up.abstained} abstained, rate {up.abstention_rate}"
    )


def test_the_engine_was_not_touched_by_either_module() -> None:
    """M9 and M10 must not have changed a rule, threshold, tolerance or category.

    Asserted structurally rather than by reading a diff: neither new module
    imports the matching engine's decision surface, and the taxonomy still has
    the members it had.
    """
    from settlesense.exceptions.taxonomy import VarianceCategory

    assert len(VarianceCategory) == 13, sorted(c.value for c in VarianceCategory)
    orchestrator = (REPO / "settlesense" / "ai" / "orchestrator.py").read_text(encoding="utf-8")
    exporter = (REPO / "settlesense" / "export" / "tally.py").read_text(encoding="utf-8")
    for name, source in (("orchestrator", orchestrator), ("exporter", exporter)):
        assert "from settlesense.matching.engine import run" not in source, (
            f"the {name} imports the engine's entry point; it must not re-decide"
        )
    print(f"\n  {len(VarianceCategory)} categories unchanged; neither module runs the engine")


def test_the_outage_client_records_what_it_was_asked(config: AppConfig, fresh_store: Any) -> None:
    """ "Nothing was sent" and "everything failed" must be distinguishable.

    A run reporting 53 unavailable could mean the model was down or that the
    stage never ran. The client's call log is what separates them.
    """
    store = fresh_store()
    dataset = store.cumulative_dataset(OUTAGE_DAY, DATA, config)
    client = OutageLLMClient()
    result = run_ai_stage(store, dataset, config, client, OUTAGE_DAY)
    assert len(client.calls) == result.report.sent, (len(client.calls), result.report.sent)
    assert len(set(client.calls)) == len(client.calls), "two exceptions produced one prompt"
    print(f"\n  {len(client.calls)} prompts attempted, all distinct, all failed")


def test_the_ai_stage_never_writes_closed(config: AppConfig, fresh_store: Any) -> None:
    """SDD 3: the exporter is the only writer of CLOSED. An outage is not an export."""
    store = fresh_store()
    dataset = store.cumulative_dataset(OUTAGE_DAY, DATA, config)
    run_ai_stage(store, dataset, config, OutageLLMClient(), OUTAGE_DAY)
    closed = [row for row in store.get_queue(ALL_STATUSES) if row.status is ExceptionStatus.CLOSED]
    assert not closed, f"the AI stage closed {len(closed)} exception(s)"
    assert all(row.closed_day is None for row in store.get_queue(ALL_STATUSES))
    print("\n  0 CLOSED after an outage run; no closed_day set")


def test_no_clock_is_read_by_the_outage_path() -> None:
    """D2. The stage takes `arrival_day`; it does not ask what time it is."""
    for module in ("ai/orchestrator.py", "ui/build_state.py"):
        source = (REPO / "settlesense" / module).read_text(encoding="utf-8")
        for banned in ("datetime.now(", "date.today(", "time.time("):
            assert banned not in source, f"{module} reads a clock: {banned}"
    # THE DERIVED DATE IS READ, NOT TYPED. My first version asserted
    # 2026-11-24 from memory; the window starts earlier and day 24 derives to
    # September. Asserting the PROPERTY - derived, in 2026 (D13), and different
    # for different days - is what the test is actually for. A literal here
    # would have to be re-typed every time the simulation window moved.
    config = load_config(REPO / "config")
    derived = as_of_for_arrival_day(OUTAGE_DAY, config)
    earlier = as_of_for_arrival_day(1, config)
    assert derived.year == 2026 and earlier.year == 2026, (derived, earlier)
    assert derived > earlier, "the arrival day does not move the date"
    assert (derived - earlier).days == OUTAGE_DAY - 1
    print(f"\n  no clock in the outage path; day 1 -> {earlier}, day {OUTAGE_DAY} -> {derived}")

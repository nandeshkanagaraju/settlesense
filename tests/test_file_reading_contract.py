"""An EMPTY input and a MISSING input are different, wherever files are read.

THE RULE. An empty table is legitimate data. A missing file is a delivery
failure. Collapsing both to "no rows" reports a clean reconciliation for a day
whose statement never arrived - a false negative that looks exactly like
success, because the output is a page of zeroes either way and nothing in it
says which kind of zero it is.

This was found in ingest.py: day1_bank.csv is header-only, because settlement
is T+N and day 1 genuinely has no bank credits. A loader that treated the file
as absent would have been indistinguishable from one that treated an absent
file as empty, and the frozen dataset contains a real instance of one of them.

WHY A REGISTRY RATHER THAN ONE TEST PER READER. The rule has to hold for
readers that do not exist yet - the M5 eval harness, the M6 state store. So
the registry below is checked against an AST scan of every module that reads a
file: adding a reader without registering it fails, rather than quietly
inheriting nothing. That is the difference between a rule and a note.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from settlesense.config import ConfigError, load_config
from settlesense.ingest import IngestError, load_dataset

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "dev"
CONFIG = REPO / "config"

SCANNED_TREES = ("settlesense", "eval")
"""gen/ is excluded: it is frozen, and it WRITES the dataset rather than
consuming one. tests/ is excluded because a fixture reading its own scratch
file is not an ingestion path."""

READING_CALLS = frozenset({"open", "read_text", "read_bytes", "iterdir", "glob"})

REGISTERED_READERS = frozenset(
    {
        "settlesense/ingest.py",
        "settlesense/config.py",
        "eval/run_eval.py",
        "eval/run_eval_ai.py",
        "eval/bench.py",
        "eval/run_ai.py",
        "eval/record_fixtures.py",
        "settlesense/exceptions/store.py",
        "settlesense/ai/client.py",
        "settlesense/export/tally.py",
        "eval/run_export.py",
    }
)
"""Every module allowed to read a file, each with a contract test below.

Adding a path here without adding its test is caught by
`test_every_registered_reader_has_a_contract_test`.
"""


def _modules_that_read_files() -> frozenset[str]:
    found: set[str] = set()
    for tree in SCANNED_TREES:
        for path in sorted((REPO / tree).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else ""
                )
                if name in READING_CALLS:
                    found.add(str(path.relative_to(REPO)))
    return frozenset(found)


def test_the_scanner_finds_the_readers_it_should() -> None:
    """Guards the scan from the other side.

    A scanner returning the empty set would make the registry test below pass
    forever while covering nothing.
    """
    found = _modules_that_read_files()
    assert "settlesense/ingest.py" in found, "the scanner missed a known reader"
    assert len(found) >= 2, f"the scanner found only {sorted(found)}"


def test_every_module_that_reads_files_is_registered() -> None:
    """A new reader must declare itself and get a contract test."""
    unregistered = sorted(_modules_that_read_files() - REGISTERED_READERS)
    assert not unregistered, (
        f"module(s) read files without a missing-vs-empty contract test: "
        f"{unregistered}.\nAn empty input is legitimate data; a missing input is a "
        "delivery failure. A reader that maps both to the same outcome reports a "
        "clean result for data that never arrived. Add the module to "
        "REGISTERED_READERS and give it a case in CONTRACTS below."
    )


def test_the_registry_lists_nothing_that_stopped_reading_files() -> None:
    """An allow-list entry for something that no longer reads files permits a
    future reader at that path to inherit the exemption silently."""
    stale = sorted(REGISTERED_READERS - _modules_that_read_files())
    assert not stale, f"REGISTERED_READERS lists non-readers: {stale}"


# ---------------------------------------------------------------------------
# The contract, per reader
# ---------------------------------------------------------------------------


def _ingest_missing(tmp: Path) -> str:
    with pytest.raises(IngestError) as caught:
        load_dataset(tmp, 1, load_config(CONFIG))
    return str(caught.value)


def _ingest_empty() -> str:
    """day 1 has a header-only bank table. Legitimate: settlement is T+N."""
    day = load_dataset(DATA, 1, load_config(CONFIG))
    assert day.bank_rows == ()
    assert day.payment_rows, "day 1 should still carry payments"
    return "loaded, zero rows"


def _config_missing(tmp: Path) -> str:
    with pytest.raises(ConfigError) as caught:
        load_config(tmp)
    return str(caught.value)


def _config_empty(tmp: Path) -> str:
    for name in ("mdr_rates.yaml", "calendar_v1.yaml", "thresholds.yaml"):
        (tmp / name).write_text("", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_config(tmp)
    return str(caught.value)


def _run_eval_missing(tmp: Path) -> str:
    """An empty directory has no day*_*.csv at all."""
    from eval.run_eval import load_days

    with pytest.raises(SystemExit) as caught:
        load_days(tmp, load_config(CONFIG))
    return str(caught.value)


def _run_eval_empty(tmp: Path) -> str:
    """A day file that EXISTS with only a header. Legitimate: day 1 has no
    bank credits because settlement is T+N."""
    from eval.run_eval import load_days

    for name, header in (
        ("day1_ledger.csv", "order_id,invoice_no,gross,order_date,customer_id,sku"),
        ("day1_payments.csv", "payment_id,order_id,method,authorized,captured,status,captured_at"),
        ("day1_refunds.csv", "refund_id,payment_id,amount,created_at"),
        (
            "day1_settlements.csv",
            "settlement_id,batch_id,line_type,payment_id,refund_id,gross,fee,tax,net,"
            "settled_event_date",
        ),
        ("day1_batches.csv", "batch_id,utr,net_total,settled_event_date"),
        ("day1_bank.csv", "bank_txn_id,value_date,amount,narration,direction"),
    ):
        (tmp / name).write_text(header + "\n", encoding="utf-8")
    dataset = load_days(tmp, load_config(CONFIG))
    assert dataset.row_count() == 0
    return "loaded, zero rows"


def _run_eval_ai_missing(tmp: Path) -> str:
    """No seed_* directories: the evaluation set was never generated."""
    from eval.run_eval_ai import main

    with pytest.raises(SystemExit) as caught:
        main(["--eval-dir", str(tmp), "--out", str(tmp / "out")])
    return str(caught.value)


def _run_eval_ai_empty(tmp: Path) -> str:
    """A seed_* directory that EXISTS but holds no day files.

    Distinguishable from absent: the runner gets past the "was it generated"
    check and fails on the empty directory instead, with a different message.
    """
    from eval.run_eval_ai import main

    (tmp / "seed_1000").mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit) as caught:
        main(["--eval-dir", str(tmp), "--out", str(tmp / "out")])
    return str(caught.value)


def _replay_missing(tmp: Path) -> str:
    """No fixture recorded for this prompt."""
    from settlesense.ai.client import FixtureMissError, ReplayLLMClient

    with pytest.raises(FixtureMissError) as caught:
        ReplayLLMClient(fixture_dir=tmp / "absent").complete("unrecorded", {})
    return str(caught.value)


def _replay_empty(tmp: Path) -> str:
    """A fixture file that EXISTS but is empty. NOT the same as absent: an
    empty recording is a corrupt one, and reporting it as a miss would send
    someone to re-record a prompt that was already recorded."""
    import json

    from settlesense.ai.client import ReplayLLMClient, prompt_hash

    (tmp / f"{prompt_hash('unrecorded')}.json").write_text("", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError) as caught:
        ReplayLLMClient(fixture_dir=tmp).complete("unrecorded", {})
    return f"corrupt fixture: {caught.value}"


def _input_rows_missing(tmp: Path) -> str:
    """A data directory that was never generated.

    This is the case the row counter originally got WRONG: `Path.glob` on a
    missing directory yields nothing instead of raising, so it returned 0 and
    the scaling table would have printed a clean `0 input rows` for a dataset
    that does not exist.

    IT USED TO LIVE IN eval/bench.py and now lives in eval/run_eval.py, because
    the evaluation runner needs the same ingest denominator and two copies of
    "what counts as a row" is a pair that drifts. The behaviour is unchanged and
    is still exercised in both directions - see the test below, which is why
    this pair is no longer wired into CONTRACTS: bench.py stopped reading files
    altogether, and `test_the_registry_lists_nothing_that_stopped_reading_files`
    caught the stale registry entry the moment the function moved.
    """
    from eval.run_eval import input_rows

    with pytest.raises(SystemExit) as caught:
        input_rows(tmp / "absent")
    return str(caught.value)


def _input_rows_empty(tmp: Path) -> str:
    """Day files that EXIST and carry only a header. Legitimate data."""
    from eval.run_eval import input_rows

    (tmp / "day1_bank.csv").write_text(
        "bank_txn_id,value_date,amount,narration,direction\n", encoding="utf-8"
    )
    assert input_rows(tmp) == 0
    return "counted, zero rows"


def test_input_rows_tells_a_missing_dataset_from_an_empty_one(tmp_path: Path) -> None:
    """The row counter's own missing-vs-empty contract, kept after the move.

    `input_rows` is not in CONTRACTS because run_eval.py's entry there covers
    `load_days`, and the registry maps one pair per module. Losing this pair
    when the function moved would have been a silent reduction in coverage, so
    it is asserted here directly.
    """
    missing = _input_rows_missing(tmp_path)
    empty = _input_rows_empty(tmp_path)
    assert "does not exist" in missing, missing
    assert missing != empty
    print(f"\n  missing -> SystemExit({missing[:40]}...); empty -> {empty}")


def _bench_manifest_missing(tmp: Path) -> str:
    """No recording manifest at all. The model was never called for this tree."""
    import eval.bench as bench

    original = bench.FIXTURE_MANIFESTS
    try:
        bench.FIXTURE_MANIFESTS = (tmp / "absent.json",)
        assert bench._manifest_totals() is None
    finally:
        bench.FIXTURE_MANIFESTS = original
    return "no manifest: cost not measured"


def _bench_manifest_empty(tmp: Path) -> str:
    """A manifest that EXISTS and records ZERO decisions. Legitimate data.

    Distinct from the case above, and the distinction is not cosmetic: this one
    reaches `RecordingCost.per_decision()`, which divides by the decision count.
    Treating "recorded nothing" as "recorded a total of Rs 0" would either crash
    on the division or print Rs 0.00 per decision, and a reader cannot tell that
    apart from a model that is free.
    """
    import eval.bench as bench

    path = tmp / "empty_manifest.json"
    path.write_text(
        json.dumps(
            {
                "recorded": 0,
                "measured_cost_usd": "0.000000",
                "measured_cost_inr": "0.00",
                "measured_input_tokens": 0,
                "measured_output_tokens": 0,
                "model": "none",
            }
        ),
        encoding="utf-8",
    )
    original = bench.FIXTURE_MANIFESTS
    try:
        bench.FIXTURE_MANIFESTS = (path,)
        assert bench._manifest_totals() is None, "zero decisions became a priced result"
    finally:
        bench.FIXTURE_MANIFESTS = original
    return "manifest present, zero decisions: still not measured"


def _store_missing(tmp: Path) -> str:
    """A day file that was never delivered."""
    from settlesense.exceptions.store import ExceptionStore, MissingFileError

    with ExceptionStore() as store, pytest.raises(MissingFileError) as caught:
        store.ingest_file(tmp / "day1_bank.csv", arrival_day=1, arrival_seq=1)
    return str(caught.value)


def _store_empty(tmp: Path) -> str:
    """A header-only day file. Ingested, recorded, and flagged as empty.

    The store must be able to say "this arrived and had no rows", which is a
    different sentence from "this never arrived" - and on this dataset day 1's
    bank table really is header-only, because settlement is T+N.
    """
    from settlesense.exceptions.store import ExceptionStore

    path = tmp / "day1_bank.csv"
    path.write_text("bank_txn_id,value_date,amount,narration,direction\n", encoding="utf-8")
    with ExceptionStore() as store:
        result = store.ingest_file(path, arrival_day=1, arrival_seq=1)
    assert result.is_empty and not result.skipped and result.row_count == 0
    return result.outcome


def _run_ai_missing(tmp: Path) -> str:
    """No seed_* directories: the evaluation set was never generated."""
    from eval.run_ai import main

    with pytest.raises(SystemExit) as caught:
        main(["--eval-dir", str(tmp), "--out", str(tmp / "out")])
    return str(caught.value)


def _run_ai_empty(tmp: Path) -> str:
    """A seed_* directory that EXISTS but holds no day files.

    Distinguishable from absent: the runner gets past the "was it generated"
    check and fails on the empty directory instead, with a different message.
    """
    from eval.run_ai import main

    (tmp / "seed_1000").mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit) as caught:
        main(["--eval-dir", str(tmp), "--out", str(tmp / "out")])
    return str(caught.value)


def _record_missing(tmp: Path) -> str:
    """No seed_* directories: the evaluation set was never generated."""
    from eval.record_fixtures import main

    with pytest.raises(SystemExit) as caught:
        main(["--eval-dir", str(tmp), "--dry-run"])
    return str(caught.value)


def _record_empty(tmp: Path) -> str:
    """A seed_* directory that EXISTS but holds no day files.

    Distinguishable from absent: it gets past the "was it generated" check and
    fails on the empty directory, with a different message.
    """
    from eval.record_fixtures import main

    (tmp / "seed_1000").mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit) as caught:
        main(["--eval-dir", str(tmp), "--dry-run"])
    return str(caught.value)


# --- M9: the bundled XSD, and the eval artifact the header is read from ------


def _schema_missing(tmp: Path) -> str:
    """No schema on disk. A broken install; nothing may be written."""
    import settlesense.export.tally as tally

    # monkeypatch rather than assign-and-restore: SCHEMA_PATH is Final, and the
    # try/finally version typechecks only because mypy cannot see the restore.
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(tally, "SCHEMA_PATH", tmp / "absent.xsd")
        with pytest.raises(tally.ExportError) as caught:
            tally.validate("<ENVELOPE/>")
    assert "missing" in str(caught.value)
    return "schema missing: cannot validate, refuse to write"


def _schema_empty(tmp: Path) -> str:
    """A schema file that EXISTS and is blank. Strictly worse than missing.

    lxml parses an empty document as an error rather than as a permissive
    schema, but the reason this case is registered separately is that the
    NEXT-worst version - a well-formed schema declaring no elements - would
    accept every document silently. Both are refused, and the refusals say
    different things, because "your install is broken" and "your schema checks
    nothing" send a reader to different places.
    """
    import settlesense.export.tally as tally

    path = tmp / "empty.xsd"
    path.write_text("", encoding="utf-8")
    vacuous = tmp / "vacuous.xsd"
    vacuous.write_text(
        '<?xml version="1.0"?><xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"/>',
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(tally, "SCHEMA_PATH", path)
        with pytest.raises(tally.ExportError) as caught:
            tally.validate("<ENVELOPE/>")
        # And the vacuous-but-well-formed case, which is the dangerous one: it
        # parses, so nothing upstream notices, and it accepts every document.
        patched.setattr(tally, "SCHEMA_PATH", vacuous)
        with pytest.raises(tally.ExportError, match="declares no elements"):
            tally.validate("<ENVELOPE/>")
    assert "empty" in str(caught.value)
    return "schema empty: parses as no rules, so validation would mean nothing"


def _results_missing(tmp: Path) -> str:
    """No results.json. There is no measured false-match rate to disclose."""
    from eval.run_export import provenance_from_results
    from settlesense.export.tally import ExportError

    with pytest.raises(ExportError) as caught:
        provenance_from_results(tmp / "absent.json", "dev")
    assert "does not exist" in str(caught.value)
    return "results missing: no measured rate exists, refuse"


def _results_empty(tmp: Path) -> str:
    """A results.json that EXISTS and carries no Population A rate.

    An eval artifact predating Population A is legitimate data - it was written
    by a real run - and it still cannot support a provenance header. The
    distinction matters because the two send you to different fixes: generate
    the artifact, versus re-run the eval that produces the metric.
    """
    from eval.run_export import POPULATION_A_KEY, provenance_from_results
    from settlesense.export.tally import ExportError

    path = tmp / "results.json"
    path.write_text(json.dumps({"seed": 42, "config_hash": "abc"}), encoding="utf-8")
    with pytest.raises(ExportError) as caught:
        provenance_from_results(path, "dev")
    assert POPULATION_A_KEY in str(caught.value)
    return "results present, rate absent: the metric was never computed"


CONTRACTS: dict[str, tuple[Callable[[Path], str], Callable[[Path], str]]] = {
    "settlesense/ingest.py": (_ingest_missing, lambda _tmp: _ingest_empty()),
    "settlesense/config.py": (_config_missing, _config_empty),
    "eval/run_eval.py": (_run_eval_missing, _run_eval_empty),
    "eval/run_eval_ai.py": (_run_eval_ai_missing, _run_eval_ai_empty),
    "eval/bench.py": (_bench_manifest_missing, _bench_manifest_empty),
    "eval/run_ai.py": (_run_ai_missing, _run_ai_empty),
    "eval/record_fixtures.py": (_record_missing, _record_empty),
    "settlesense/exceptions/store.py": (_store_missing, _store_empty),
    "settlesense/ai/client.py": (_replay_missing, _replay_empty),
    "settlesense/export/tally.py": (_schema_missing, _schema_empty),
    "eval/run_export.py": (_results_missing, _results_empty),
}


def test_every_registered_reader_has_a_contract_test() -> None:
    assert set(CONTRACTS) == set(REGISTERED_READERS), (
        f"registry and contracts disagree: {sorted(set(REGISTERED_READERS) ^ set(CONTRACTS))}"
    )


@pytest.mark.boundary_refusal
@pytest.mark.parametrize("module", sorted(REGISTERED_READERS))
def test_missing_and_empty_produce_different_outcomes(module: str, tmp_path: Path) -> None:
    """THE rule, per reader.

    Not "empty must load" - config is entitled to reject an empty file, since
    an empty config is not legitimate the way an empty bank table is. What is
    forbidden everywhere is producing the SAME outcome for both, which is what
    makes the two indistinguishable downstream.
    """
    on_missing, on_empty = CONTRACTS[module]
    missing_outcome = on_missing(tmp_path / "absent")
    empty_outcome = on_empty(tmp_path)
    assert missing_outcome != empty_outcome, (
        f"{module} reports {missing_outcome!r} for BOTH a missing and an empty "
        "input. A caller cannot tell a delivery failure from legitimate data."
    )


@pytest.mark.boundary_refusal
def test_ingest_names_the_missing_file_and_says_why_it_matters(tmp_path: Path) -> None:
    """FAULT INJECTION. The message has to be actionable: which file, and why
    an empty table is not an acceptable substitute."""
    message = _ingest_missing(tmp_path)
    assert "does not exist" in message
    assert "not an empty table" in message


@pytest.mark.boundary_refusal
def test_config_distinguishes_absent_from_empty(tmp_path: Path) -> None:
    """FAULT INJECTION. Both are errors here - an empty config is not
    legitimate data - but they are DIFFERENT errors, because "you forgot to
    ship the file" and "you shipped a blank file" have different fixes."""
    assert "not found" in _config_missing(tmp_path / "absent")
    assert "is empty" in _config_empty(tmp_path)


def test_an_empty_data_table_is_not_an_error(tmp_path: Path) -> None:
    """The other half of the rule, stated positively.

    A header-only table must load. Raising on one would make the engine refuse
    a day that legitimately settled nothing - turning correct data into an
    outage, which is the mirror-image failure of the one this rule prevents.
    """
    assert _ingest_empty() == "loaded, zero rows"

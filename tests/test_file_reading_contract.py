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
        "settlesense/ai/client.py",
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
    from settlesense.ai.client import ReplayLLMClient, ReplayMissError

    with pytest.raises(ReplayMissError) as caught:
        ReplayLLMClient(fixture_dir=tmp / "absent").complete("unrecorded")
    return str(caught.value)


def _replay_empty(tmp: Path) -> str:
    """A fixture file that EXISTS but is empty. NOT the same as absent: an
    empty recording is a corrupt one, and reporting it as a miss would send
    someone to re-record a prompt that was already recorded."""
    import json

    from settlesense.ai.client import ReplayLLMClient, prompt_hash

    (tmp / f"{prompt_hash('unrecorded')}.json").write_text("", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError) as caught:
        ReplayLLMClient(fixture_dir=tmp).complete("unrecorded")
    return f"corrupt fixture: {caught.value}"


def _bench_missing(tmp: Path) -> str:
    """A data directory that was never generated.

    This is the case bench.py originally got WRONG: `Path.glob` on a missing
    directory yields nothing instead of raising, so the row counter returned 0
    and the scaling table would have printed a clean `0 input rows` for a
    dataset that does not exist.
    """
    from eval.bench import _input_rows

    with pytest.raises(SystemExit) as caught:
        _input_rows(tmp / "absent")
    return str(caught.value)


def _bench_empty(tmp: Path) -> str:
    """Day files that EXIST and carry only a header. Legitimate data."""
    from eval.bench import _input_rows

    (tmp / "day1_bank.csv").write_text(
        "bank_txn_id,value_date,amount,narration,direction\n", encoding="utf-8"
    )
    assert _input_rows(tmp) == 0
    return "counted, zero rows"


CONTRACTS: dict[str, tuple[Callable[[Path], str], Callable[[Path], str]]] = {
    "settlesense/ingest.py": (_ingest_missing, lambda _tmp: _ingest_empty()),
    "settlesense/config.py": (_config_missing, _config_empty),
    "eval/run_eval.py": (_run_eval_missing, _run_eval_empty),
    "eval/run_eval_ai.py": (_run_eval_ai_missing, _run_eval_ai_empty),
    "eval/bench.py": (_bench_missing, _bench_empty),
    "settlesense/ai/client.py": (_replay_missing, _replay_empty),
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

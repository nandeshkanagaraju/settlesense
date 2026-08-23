"""M1 - tests for the adversarial generator.

Covers the three things a generator has to get right before anything downstream
can be believed: it reproduces exactly, its arithmetic closes to the cent, and
its ground truth describes the data it actually wrote.

Money is re-derived here from a rate table restated in this file. If these
tests imported gen's rates, a wrong rate would agree with itself and the check
would prove nothing.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pytest

from gen.generate import build_plan, main
from gen.lifecycle import (
    SettlementLineType,
    bank_txn_id_for,
    build_clean_dataset,
    load_working_calendar,
)
from gen.noise import NOISE_REGISTRY, NoiseRates, apply_noise
from gen.profiles import PROFILES
from gen.truth import (
    EdgeType,
    TruthEdge,
    TruthSelfCheckError,
    build_truth,
    check_cardinality,
    run_self_check,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CALENDAR = REPO_ROOT / "config" / "calendar_v1.yaml"

Q2 = Decimal("0.01")
ZERO = Decimal("0.00")

# Restated by hand. Importing gen.profiles here would let a wrong rate agree
# with itself; two independent statements of the same fact can disagree.
RATES: Mapping[str, Mapping[str, str]] = {
    "profile_a": {"card": "0.0200", "upi": "0.0000", "netbanking": "0.0175", "wallet": "0.0210"},
    "profile_b": {"card": "0.0235", "upi": "0.0000", "netbanking": "0.0190", "wallet": "0.0200"},
    "profile_c": {"card": "0.0180", "upi": "0.0000", "netbanking": "0.0160", "wallet": "0.0195"},
}
GST = Decimal("0.18")
MERCHANT_TO_PROFILE = {
    "AURORA RETAIL": "profile_a",
    "BLUEPEAK FOODS": "profile_b",
    "CARBON WORKS PVT LTD": "profile_c",
}

DAYS = 20  # enough settlement dates that batch-grain noise actually fires
RECORDS = 5_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _generate(out: Path, seed: int, *, withheld: bool = True, records: int = RECORDS) -> int:
    argv = [
        "--seed",
        str(seed),
        "--out",
        str(out),
        "--days",
        str(DAYS),
        "--records",
        str(records),
        "--calendar",
        str(CALENDAR),
    ]
    if withheld:
        argv.append("--include-withheld")
    return main(argv)


@pytest.fixture(scope="session")
def dev_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """seed 42, withheld noise ON. The workhorse dataset for most assertions."""
    out = tmp_path_factory.mktemp("dev")
    assert _generate(out, 42) == 0, "generator self-check failed on seed 42"
    return out


@pytest.fixture(scope="session")
def tuned_only_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """seed 42, withheld noise OFF - what engine development is allowed to see."""
    out = tmp_path_factory.mktemp("tuned")
    assert _generate(out, 42, withheld=False) == 0
    return out


@pytest.fixture(scope="session")
def holdout_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("holdout")
    assert _generate(out, 999) == 0
    return out


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_table(data: Path, table: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(data.glob(f"day*_{table}.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def read_truth(data: Path, seed: int) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((data / f"truth_{seed}.json").read_text("utf-8"))
    return payload


def q(text: str) -> Decimal:
    """Parse a possibly-noisy money cell the way the engine will have to."""
    cleaned = text.strip().replace(",", "").replace("₹", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    return Decimal(cleaned).quantize(Q2, rounding=ROUND_HALF_UP)


def file_hashes(data: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(data.glob("day*.csv"))
    }


def profile_of_batch(data: Path) -> dict[str, str]:
    """Recover each batch's profile from its bank narration, then its merchant."""
    batches = {b["batch_id"]: b for b in read_table(data, "batches")}
    mapping: dict[str, str] = {}
    for row in read_table(data, "bank"):
        for batch_id in batches:
            if bank_txn_id_for(batch_id) != row["bank_txn_id"]:
                continue
            for merchant, profile in MERCHANT_TO_PROFILE.items():
                squashed = merchant.replace(" ", "")
                if merchant in row["narration"] or squashed in row["narration"]:
                    mapping[batch_id] = profile
    return mapping


# ===========================================================================
# 1-3. Reproducibility
# ===========================================================================


def test_same_seed_produces_byte_identical_files(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    assert _generate(first, 42, records=900) == 0
    assert _generate(second, 42, records=900) == 0

    left, right = file_hashes(first), file_hashes(second)
    assert left.keys() == right.keys(), "different file sets for the same seed"
    assert left == right, sorted(k for k in left if left[k] != right[k])
    assert len(left) >= 6, "expected at least one full six-table day bundle"


def test_same_seed_produces_identical_truth(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    assert _generate(first, 42, records=900) == 0
    assert _generate(second, 42, records=900) == 0
    assert (first / "truth_42.json").read_bytes() == (second / "truth_42.json").read_bytes()


def test_different_seeds_produce_different_data(tmp_path: Path) -> None:
    first, second = tmp_path / "s42", tmp_path / "s999"
    assert _generate(first, 42, records=900) == 0
    assert _generate(second, 999, records=900) == 0

    left, right = file_hashes(first), file_hashes(second)
    shared = left.keys() & right.keys()
    assert shared, "the two runs share no filenames to compare"
    assert any(left[name] != right[name] for name in shared), "seed 999 reproduced seed 42"

    payments_42 = {r["payment_id"] for r in read_table(first, "payments")}
    payments_999 = {r["payment_id"] for r in read_table(second, "payments")}
    assert not payments_42 & payments_999, "payment IDs collide across seeds"


def test_regeneration_after_deletion_reproduces_exactly(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert _generate(out, 42, records=900) == 0
    before = file_hashes(out)
    truth_before = (out / "truth_42.json").read_bytes()

    for path in out.iterdir():
        path.unlink()
    assert not any(out.iterdir())

    assert _generate(out, 42, records=900) == 0
    assert file_hashes(out) == before
    assert (out / "truth_42.json").read_bytes() == truth_before


# ===========================================================================
# 4-7. Balance invariants
# ===========================================================================


def test_every_chain_balances_to_the_cent(dev_dir: Path) -> None:
    """gross - fee - tax - refunds == net, exact Decimal equality (SDD 3.1b)."""
    truth = read_truth(dev_dir, 42)
    lines = {line["settlement_id"]: line for line in read_table(dev_dir, "settlements")}
    refunds = {r["refund_id"]: q(r["amount"]) for r in read_table(dev_dir, "refunds")}

    checked = 0
    for case in truth["cases"]:
        fee = sum((q(lines[i]["fee"]) for i in case["payment_line_ids"]), ZERO)
        tax = sum((q(lines[i]["tax"]) for i in case["payment_line_ids"]), ZERO)
        refunded = sum((refunds[r] for r in case["refund_ids"]), ZERO)
        gross = q(case["expected_gross"])
        expected_net = q(case["expected_net"])
        assert expected_net == (gross - fee - tax - refunded).quantize(Q2), (
            f"case {case['case_id']}: {gross} - {fee} - {tax} - {refunded} != {expected_net}"
        )
        checked += 1
    assert checked >= RECORDS, f"only checked {checked} chains"


def test_batch_net_total_equals_signed_member_sum(dev_dir: Path) -> None:
    lines = read_table(dev_dir, "settlements")
    batches = read_table(dev_dir, "batches")
    summed: dict[str, Decimal] = {}
    for line in lines:
        summed[line["batch_id"]] = summed.get(line["batch_id"], ZERO) + q(line["net"])

    assert batches, "no batches were generated"
    for batch in batches:
        assert q(batch["net_total"]) == summed[batch["batch_id"]].quantize(Q2), (
            f"batch {batch['batch_id']}: net_total {batch['net_total']} != member sum"
        )


def test_a_batch_mixing_payment_and_refund_lines_still_balances(dev_dir: Path) -> None:
    """The regression case for signed-line arithmetic (SDD 3.1a).

    Without this, the previous test passes vacuously on data where refunds never
    enter a batch at all.
    """
    lines = read_table(dev_dir, "settlements")
    kinds: dict[str, set[str]] = {}
    for line in lines:
        kinds.setdefault(line["batch_id"], set()).add(line["line_type"])
    mixed = [b for b, k in kinds.items() if k == {"payment", "refund"}]
    assert mixed, "no batch contains both a PAYMENT and a REFUND line"

    batches = {b["batch_id"]: b for b in read_table(dev_dir, "batches")}
    for batch_id in mixed:
        members = [line for line in lines if line["batch_id"] == batch_id]
        credits = sum((q(x["net"]) for x in members if x["line_type"] == "payment"), ZERO)
        debits = sum((q(x["net"]) for x in members if x["line_type"] == "refund"), ZERO)
        assert q(batches[batch_id]["net_total"]) == (credits + debits).quantize(Q2)
        assert debits < ZERO, "refund lines must be negative"


def test_global_conservation_both_identities(dev_dir: Path) -> None:
    """Self-check assertion 10, recomputed from the written CSVs."""
    truth = read_truth(dev_dir, 42)
    lines = read_table(dev_dir, "settlements")
    batches = read_table(dev_dir, "batches")
    bank = read_table(dev_dir, "bank")

    # case side
    sum_gross = sum((q(c["expected_gross"]) for c in truth["cases"]), ZERO)
    sum_net = sum((q(c["expected_net"]) for c in truth["cases"]), ZERO)
    sum_fee = sum((q(line["fee"]) for line in lines), ZERO)
    sum_tax = sum((q(line["tax"]) for line in lines), ZERO)
    sum_refunds = sum((q(r["amount"]) for r in read_table(dev_dir, "refunds")), ZERO)
    assert sum_gross == (sum_net + sum_fee + sum_tax + sum_refunds).quantize(Q2), (
        f"case side: {sum_gross} != {sum_net} + {sum_fee} + {sum_tax} + {sum_refunds}"
    )

    # batch side, allowing exactly the deviation truth predicts
    sum_lines = sum((q(line["net"]) for line in lines), ZERO)
    sum_batches = sum((q(b["net_total"]) for b in batches), ZERO)
    sum_bank = sum((q(r["amount"]) for r in bank), ZERO)
    assert sum_batches.quantize(Q2) == sum_lines.quantize(Q2)

    expected_txn = {bank_txn_id_for(b["batch_id"]): b for b in batches}
    orphan = sum((q(r["amount"]) for r in bank if r["bank_txn_id"] not in expected_txn), ZERO)
    credited = {r["bank_txn_id"] for r in bank}
    missing = sum(
        (q(b["net_total"]) for b in batches if bank_txn_id_for(b["batch_id"]) not in credited),
        ZERO,
    )
    assert sum_bank.quantize(Q2) == (sum_batches + orphan - missing).quantize(Q2), (
        f"batch side: bank {sum_bank} != batches {sum_batches} "
        f"+ orphans {orphan} - missing {missing}"
    )
    assert sum_net.quantize(Q2) == sum_lines.quantize(Q2), "the two sides disagree"


def test_every_fee_recomputes_from_the_profile_rate(dev_dir: Path) -> None:
    payments = {p["payment_id"]: p for p in read_table(dev_dir, "payments")}
    owner = profile_of_batch(dev_dir)
    checked = 0
    upi_seen = 0
    for line in read_table(dev_dir, "settlements"):
        if line["line_type"] != "payment":
            continue
        profile = owner.get(line["batch_id"])
        if profile is None:  # batch whose credit never arrived: no narration to read
            continue
        method = payments[line["payment_id"]]["method"]
        gross = q(line["gross"])
        want_fee = (gross * Decimal(RATES[profile][method])).quantize(Q2, ROUND_HALF_UP)
        want_tax = (want_fee * GST).quantize(Q2, ROUND_HALF_UP)
        assert q(line["fee"]) == want_fee, (
            f"{line['settlement_id']}: fee {line['fee']} != {want_fee} ({profile}/{method})"
        )
        assert q(line["tax"]) == want_tax, f"{line['settlement_id']}: tax != {want_tax}"
        if method == "upi":
            upi_seen += 1
            assert q(line["fee"]) == ZERO, "UPI must be zero-rated"
        checked += 1
    assert checked > 1000, f"only recomputed {checked} fees"
    assert upi_seen > 0, "no UPI lines to prove the zero-rate path"


# ===========================================================================
# 8-11. Ground-truth integrity
# ===========================================================================


def test_every_truth_id_exists_in_the_dataset(dev_dir: Path) -> None:
    truth = read_truth(dev_dir, 42)
    known = {
        "order": {r["order_id"] for r in read_table(dev_dir, "ledger")},
        "payment": {r["payment_id"] for r in read_table(dev_dir, "payments")},
        "refund": {r["refund_id"] for r in read_table(dev_dir, "refunds")},
        "line": {r["settlement_id"] for r in read_table(dev_dir, "settlements")},
        "batch": {r["batch_id"] for r in read_table(dev_dir, "batches")},
        "bank": {r["bank_txn_id"] for r in read_table(dev_dir, "bank")},
    }
    endpoints = {
        "order_to_payment": ("order", ("payment",)),
        "payment_to_settlement": ("payment", ("line",)),
        "settlement_to_batch": ("line", ("batch",)),
        "batch_to_bank": ("batch", ("bank",)),
        "payment_to_refund": ("payment", ("refund", "line")),
    }
    for edge in truth["edges"]:
        src_kind, dst_kinds = endpoints[edge["edge_type"]]
        assert edge["src_id"] in known[src_kind], f"{edge} src unknown"
        assert any(edge["dst_id"] in known[k] for k in dst_kinds), f"{edge} dst unknown"

    for case in truth["cases"]:
        assert case["payment_id"] in known["payment"]
        assert case["order_id"] in known["order"]
        for line_id in case["settlement_line_ids"]:
            assert line_id in known["line"]
        for batch_id in case["batch_ids"]:
            assert batch_id in known["batch"]
        for bank_id in case["bank_txn_ids"]:
            assert bank_id in known["bank"]
    for link in truth["batch_links"]:
        assert link["batch_id"] in known["batch"]
        if link["bank_txn_id"] is not None:
            assert link["bank_txn_id"] in known["bank"]
    for row in truth["row_variances"]:
        pool = known["order"] if row["row_kind"] == "ledger_row" else known["bank"]
        assert row["row_id"] in pool, f"row variance {row['row_id']} not in the dataset"


def test_case_id_follows_the_sdd_formula(dev_dir: Path) -> None:
    for case in read_truth(dev_dir, 42)["cases"]:
        want = hashlib.sha256(b"case|" + case["payment_id"].encode()).hexdigest()[:16]
        assert case["case_id"] == want


# --- 9. typed edge cardinality --------------------------------------------


def _edges(data: Path, seed: int) -> list[TruthEdge]:
    return [
        TruthEdge(EdgeType(e["edge_type"]), e["src_id"], e["dst_id"])
        for e in read_truth(data, seed)["edges"]
    ]


def test_generated_edge_set_passes_cardinality(dev_dir: Path) -> None:
    assert check_cardinality(_edges(dev_dir, 42)) == []


def test_every_settlement_line_has_exactly_one_batch_edge(dev_dir: Path) -> None:
    counts = Counter(
        e.src_id for e in _edges(dev_dir, 42) if e.edge_type is EdgeType.SETTLEMENT_TO_BATCH
    )
    line_ids = {r["settlement_id"] for r in read_table(dev_dir, "settlements")}
    assert set(counts) == line_ids, "some line has no SETTLEMENT_TO_BATCH edge"
    assert set(counts.values()) == {1}, "a line belongs to more than one batch"


def test_a_batch_with_many_settlement_rows_passes(dev_dir: Path) -> None:
    """N:1 is NORMAL. The old 'no row partners twice' rule would fail here.

    This is the negative control for the whole cardinality design: the wrong
    invariant does not fail on bad data, it fails on correct data.
    """
    edges = _edges(dev_dir, 42)
    per_batch = Counter(e.dst_id for e in edges if e.edge_type is EdgeType.SETTLEMENT_TO_BATCH)
    assert max(per_batch.values()) > 50, "no batch is populated enough to prove the point"
    assert check_cardinality(edges) == [], "many-lines-per-batch was wrongly rejected"


def test_every_batch_has_at_most_one_bank_edge(dev_dir: Path) -> None:
    counts = Counter(e.src_id for e in _edges(dev_dir, 42) if e.edge_type is EdgeType.BATCH_TO_BANK)
    assert not [b for b, n in counts.items() if n > 1], "a batch was credited twice"


def test_order_to_payment_is_one_to_one_both_ways(dev_dir: Path) -> None:
    edges = [e for e in _edges(dev_dir, 42) if e.edge_type is EdgeType.ORDER_TO_PAYMENT]
    assert edges
    assert max(Counter(e.src_id for e in edges).values()) == 1
    assert max(Counter(e.dst_id for e in edges).values()) == 1


def test_payment_to_settlement_n_ge_2_is_legal_and_marks_a_split(dev_dir: Path) -> None:
    edges = [e for e in _edges(dev_dir, 42) if e.edge_type is EdgeType.PAYMENT_TO_SETTLEMENT]
    multi = {src for src, n in Counter(e.src_id for e in edges).items() if n >= 2}
    assert multi, "no split settlement in a run that includes withheld noise"
    assert check_cardinality(edges) == [], "1:N was wrongly rejected"

    by_payment = {c["payment_id"]: c for c in read_truth(dev_dir, 42)["cases"]}
    for payment_id in multi:
        case = by_payment[payment_id]
        assert case["true_category"] == "SPLIT_SETTLEMENT"
        assert len(case["payment_line_ids"]) >= 2
        # ONE case, never two (SDD 3.1).
        assert (
            sum(1 for c in read_truth(dev_dir, 42)["cases"] if c["payment_id"] == payment_id) == 1
        )


def test_payment_to_settlement_never_points_at_a_refund_line(dev_dir: Path) -> None:
    kind = {r["settlement_id"]: r["line_type"] for r in read_table(dev_dir, "settlements")}
    offenders = [
        e
        for e in _edges(dev_dir, 42)
        if e.edge_type is EdgeType.PAYMENT_TO_SETTLEMENT and kind[e.dst_id] != "payment"
    ]
    assert not offenders, f"{len(offenders)} edges reference a refund line"


def test_duplicate_identical_edges_are_rejected() -> None:
    edge = TruthEdge(EdgeType.SETTLEMENT_TO_BATCH, "SET_1", "BAT_1")
    assert check_cardinality([edge]) == []
    violations = check_cardinality([edge, edge])
    assert violations, "an exactly duplicated edge was accepted"
    assert "duplicate edge" in str(violations[0])


def test_payment_to_refund_has_no_upper_bound() -> None:
    """1:N with no cap - partial and multiple refunds are legal."""
    edges = [TruthEdge(EdgeType.PAYMENT_TO_REFUND, "PAY_1", f"RFD_{i}") for i in range(1, 8)]
    assert check_cardinality(edges) == []


# --- 10. noise is accounted for -------------------------------------------


def test_every_noise_instance_has_a_truth_annotation(dev_dir: Path) -> None:
    """Rebuild the run in-process and confirm the file records what was injected."""
    calendar = load_working_calendar(CALENDAR)
    profiles = {p.name: p for p in PROFILES}
    rng = random.Random(42)
    clean = build_clean_dataset(
        rng,
        build_plan(RECORDS, DAYS, PROFILES),
        calendar,
        calendar.next_working_day(calendar.window_start),
        42,
    )
    _, ledger = apply_noise(clean, rng, profiles, NoiseRates(), include_withheld=True)

    truth = read_truth(dev_dir, 42)
    recorded: set[tuple[str, str]] = set()
    for case in truth["cases"]:
        recorded |= {(n, case["payment_id"]) for n in case["noise_types"]}
    for link in truth["batch_links"]:
        recorded |= {(n, link["batch_id"]) for n in link["noise_types"]}
    for row in truth["row_variances"]:
        recorded.add((row["noise_type"], row["row_id"]))

    # Format and arrival noise are lossless presentation changes with no
    # category; they are counted, not attached to a target's truth.
    presentational = {"mixed_amount_formats", "out_of_order_arrival"}
    missing = [
        (a.noise_type, a.target_id)
        for a in ledger.annotations
        if a.noise_type not in presentational
        and a.category is not None
        and (a.noise_type, a.target_id) not in recorded
    ]
    assert not missing, f"{len(missing)} injected errors absent from truth, e.g. {missing[:3]}"

    counted = sum(ledger.counts.values())
    assert counted > 0
    assert truth["counts"]["cases"] > 0


# --- 10c. identifier namespaces -------------------------------------------


def test_settlement_and_batch_id_namespaces_are_disjoint(dev_dir: Path) -> None:
    line_ids = {r["settlement_id"] for r in read_table(dev_dir, "settlements")}
    batch_ids = {r["batch_id"] for r in read_table(dev_dir, "batches")}
    assert line_ids & batch_ids == set(), "settlement_id and batch_id share a namespace"
    assert line_ids and batch_ids


def test_every_line_batch_id_exists_in_the_batches_file(dev_dir: Path) -> None:
    batch_ids = {r["batch_id"] for r in read_table(dev_dir, "batches")}
    for line in read_table(dev_dir, "settlements"):
        assert line["batch_id"] in batch_ids, (
            f"{line['settlement_id']} references unknown batch {line['batch_id']}"
        )


# --- 11. the self-check must reject a corrupted dataset --------------------


def _rebuild(records: int = 600) -> tuple[Any, Any, Any, Any]:
    calendar = load_working_calendar(CALENDAR)
    profiles = {p.name: p for p in PROFILES}
    rng = random.Random(42)
    dataset = build_clean_dataset(
        rng,
        build_plan(records, 8, PROFILES),
        calendar,
        calendar.next_working_day(calendar.window_start),
        42,
    )
    return dataset, build_truth(dataset, calendar, profiles, 42), calendar, profiles


def test_self_check_accepts_the_untouched_dataset() -> None:
    dataset, truth, calendar, profiles = _rebuild()
    run_self_check(dataset, truth, calendar=calendar, profiles=profiles)


def test_self_check_raises_on_a_one_paisa_corruption() -> None:
    """The smallest possible lie must still fail. D1 exists for this margin."""
    dataset, truth, calendar, profiles = _rebuild()
    lines = list(dataset.settlement_lines)
    index = next(i for i, line in enumerate(lines) if line.line_type is SettlementLineType.PAYMENT)
    lines[index] = replace(lines[index], net=lines[index].net + Decimal("0.01"))
    corrupted = replace(dataset, settlement_lines=tuple(lines))

    with pytest.raises(TruthSelfCheckError) as excinfo:
        run_self_check(corrupted, truth, calendar=calendar, profiles=profiles)
    message = str(excinfo.value)
    assert lines[index].settlement_id in message or "conservation" in message


def test_self_check_raises_on_a_corrupted_batch_total() -> None:
    dataset, truth, calendar, profiles = _rebuild()
    batches = list(dataset.batches)
    batches[0] = replace(batches[0], net_total=batches[0].net_total + Decimal("0.01"))
    with pytest.raises(TruthSelfCheckError):
        run_self_check(
            replace(dataset, batches=tuple(batches)),
            truth,
            calendar=calendar,
            profiles=profiles,
        )


# ===========================================================================
# 12-17. Noise behaviour
# ===========================================================================


def _narration_for(data: Path, batch_id: str) -> str | None:
    want = bank_txn_id_for(batch_id)
    for row in read_table(data, "bank"):
        if row["bank_txn_id"] == want:
            return row["narration"]
    return None


def test_truncate_utr_leaves_a_genuine_prefix(dev_dir: Path) -> None:
    truth = read_truth(dev_dir, 42)
    batches = {b["batch_id"]: b["utr"] for b in read_table(dev_dir, "batches")}
    targets = [link for link in truth["batch_links"] if "truncate_utr" in link["noise_types"]]
    assert targets, "truncate_utr never fired"

    checked = 0
    for link in targets:
        if "drop_utr" in link["noise_types"] or "garbled_narration" in link["noise_types"]:
            continue  # a later injector overwrote the prefix
        narration = _narration_for(dev_dir, link["batch_id"])
        assert narration is not None
        utr = batches[link["batch_id"]]
        # NOT `t.isupper()`: an all-digit prefix such as "174092" has no cased
        # characters, so isupper() is False and the token would be skipped.
        tokens = [t for t in narration.split() if t.isalnum() and t == t.upper()]
        prefixes = [t for t in tokens if utr.startswith(t) and 6 <= len(t) <= 10]
        assert prefixes, f"batch {link['batch_id']}: no genuine prefix of {utr} in {narration!r}"
        assert utr not in narration, "the full UTR survived truncation"
        checked += 1
    assert checked, "every truncated batch was later overwritten; nothing proved"


def test_drop_utr_leaves_no_recoverable_utr_token(dev_dir: Path) -> None:
    """No token in the narration is the UTR or any usable prefix of it.

    Deliberately stated against the true UTR rather than as "no 10+ character
    alphanumeric token". An unspaced merchant variant ("AURORARETAIL") is a
    12-character alphanumeric token and is legitimate - the assertion that
    matters is that the UTR is unrecoverable, not that long tokens are absent.
    """
    truth = read_truth(dev_dir, 42)
    batches = {b["batch_id"]: b["utr"] for b in read_table(dev_dir, "batches")}
    targets = [link for link in truth["batch_links"] if "drop_utr" in link["noise_types"]]
    assert targets, "drop_utr never fired"

    for link in targets:
        narration = _narration_for(dev_dir, link["batch_id"])
        assert narration is not None
        utr = batches[link["batch_id"]]
        assert utr not in narration
        for token in narration.split():
            assert not (len(token) >= 6 and utr.startswith(token)), (
                f"batch {link['batch_id']}: {token!r} is still a usable prefix of {utr}"
            )


def test_duplicates_and_repeat_purchases_both_exist_with_different_categories(
    dev_dir: Path,
) -> None:
    truth = read_truth(dev_dir, 42)
    confirmed = [r for r in truth["row_variances"] if r["true_category"] == "DUPLICATE_CONFIRMED"]
    candidates = [c for c in truth["cases"] if c["true_category"] == "DUPLICATE_CANDIDATE"]
    assert confirmed, "no byte-identical duplicate ledger rows"
    assert candidates, "no genuine repeat purchases"
    assert {r["true_category"] for r in confirmed} != {c["true_category"] for c in candidates}, (
        "the two duplicate shapes share a category"
    )

    ledger = read_table(dev_dir, "ledger")
    counts = Counter(r["order_id"] for r in ledger)
    for row in confirmed:
        assert counts[row["row_id"]] >= 2, "DUPLICATE_CONFIRMED order appears once"

    # The whole point: amount alone cannot separate them.
    by_customer: dict[tuple[str, str], set[str]] = {}
    for row in ledger:
        by_customer.setdefault((row["customer_id"], row["gross"].strip()), set()).add(
            row["order_id"]
        )
    repeats = [k for k, orders in by_customer.items() if len(orders) > 1]
    assert repeats, "no same-customer same-amount pair with distinct order IDs"


def test_mixed_amount_formats_produces_at_least_three_distinct_shapes(dev_dir: Path) -> None:
    shapes: Counter[str] = Counter()
    tables = {
        "ledger": ["gross"],
        "payments": ["captured"],
        "refunds": ["amount"],
        "settlements": ["net"],
        "bank": ["amount"],
    }
    for table, columns in tables.items():
        for row in read_table(dev_dir, table):
            for column in columns:
                raw = row[column]
                if raw != raw.strip():
                    shapes["whitespace_padded"] += 1
                elif raw.startswith("(") and raw.endswith(")"):
                    shapes["parenthesised_debit"] += 1
                elif "," in raw:
                    shapes["thousands_separated"] += 1
                elif re.fullmatch(r"-?\d+", raw):
                    shapes["no_decimal_point"] += 1
                elif re.fullmatch(r"-?\d+\.\d", raw):
                    shapes["one_decimal_place"] += 1
                else:
                    shapes["plain_2dp"] += 1
    variants = {k: v for k, v in shapes.items() if k != "plain_2dp"}
    assert len(variants) >= 3, f"only {len(variants)} format variants: {dict(shapes)}"
    assert shapes["plain_2dp"] > 0


def test_format_variants_are_lossless(dev_dir: Path) -> None:
    """A variant renders the same number differently; it never changes it."""
    for row in read_table(dev_dir, "batches"):
        assert q(row["net_total"]) == Decimal(row["net_total"].strip().replace(",", ""))


def test_withheld_noise_is_absent_without_the_flag_and_present_with_it(
    dev_dir: Path, tuned_only_dir: Path
) -> None:
    withheld = {name for name, (_, gated) in NOISE_REGISTRY.items() if gated}
    assert withheld == {"garbled_narration", "split_settlement"}

    def applied(data: Path) -> set[str]:
        truth = read_truth(data, 42)
        names: set[str] = set()
        for case in truth["cases"]:
            names |= set(case["noise_types"])
        for link in truth["batch_links"]:
            names |= set(link["noise_types"])
        for row in truth["row_variances"]:
            names.add(row["noise_type"])
        return names

    tuned, full = applied(tuned_only_dir), applied(dev_dir)
    assert not (tuned & withheld), f"withheld noise leaked without the flag: {tuned & withheld}"
    assert withheld <= full, f"withheld noise missing with the flag: {withheld - full}"


def test_no_split_settlement_without_the_flag(tuned_only_dir: Path) -> None:
    """Engine development must never see a split. It is the withheld structure."""
    for case in read_truth(tuned_only_dir, 42)["cases"]:
        assert len(case["payment_line_ids"]) <= 1
        assert case["true_category"] != "SPLIT_SETTLEMENT"


def test_every_registered_noise_type_fires_at_least_once(dev_dir: Path, capsys: Any) -> None:
    calendar = load_working_calendar(CALENDAR)
    profiles = {p.name: p for p in PROFILES}
    rng = random.Random(42)
    clean = build_clean_dataset(
        rng,
        build_plan(RECORDS, DAYS, PROFILES),
        calendar,
        calendar.next_working_day(calendar.window_start),
        42,
    )
    _, ledger = apply_noise(clean, rng, profiles, NoiseRates(), include_withheld=True)
    silent = sorted(name for name in NOISE_REGISTRY if ledger.counts.get(name, 0) < 1)
    assert not silent, f"noise types that never fired in {RECORDS} records: {silent}"


# ===========================================================================
# 18-19. Determinism
# ===========================================================================


def test_money_cells_parse_to_two_decimal_places(dev_dir: Path) -> None:
    money_columns = {
        "ledger": ["gross"],
        "payments": ["authorized", "captured"],
        "refunds": ["amount"],
        "settlements": ["gross", "fee", "tax", "net"],
        "batches": ["net_total"],
        "bank": ["amount"],
    }
    checked = 0
    for table, columns in money_columns.items():
        for row in read_table(dev_dir, table):
            for column in columns:
                value = q(row[column])
                assert -value.as_tuple().exponent == 2, (
                    f"{table}.{column} = {row[column]!r} is not 2 dp"
                )
                assert isinstance(value, Decimal)
                checked += 1
    assert checked > 10_000, f"only checked {checked} money cells"


def test_dates_are_all_in_2026(dev_dir: Path) -> None:
    date_columns = {
        "ledger": ["order_date"],
        "payments": ["captured_at"],
        "refunds": ["created_at"],
        "settlements": ["settled_event_date"],
        "batches": ["settled_event_date"],
        "bank": ["value_date"],
    }
    for table, columns in date_columns.items():
        for row in read_table(dev_dir, table):
            for column in columns:
                assert date.fromisoformat(row[column]).year == 2026, (
                    f"D13: {table}.{column} = {row[column]}"
                )


def test_arrival_day_is_a_positive_int_never_a_date(dev_dir: Path) -> None:
    for case in read_truth(dev_dir, 42)["cases"]:
        value = case["arrival_day"]
        assert isinstance(value, int) and not isinstance(value, bool)
        assert value >= 1


def test_truth_file_declares_generator_commit_null(dev_dir: Path) -> None:
    """The freeze hash cannot exist before the commit that creates it (SDD 5.1)."""
    truth = read_truth(dev_dir, 42)
    assert truth["generator_commit"] is None
    assert "generator_commit" in (dev_dir / "truth_42.json").read_text("utf-8")


def _gen_python_files() -> Iterator[Path]:
    return iter(sorted((REPO_ROOT / "gen").glob("*.py")))


@pytest.mark.determinism
def test_no_float_in_generator_money_paths() -> None:
    """D1 - AST scan of gen/ for float literals and float() coercions.

    A float anywhere in gen/ is a defect: every money value here is Decimal, and
    every probability is resolved in integer basis points precisely so that no
    binary-rounding decision can reach the output.
    """
    offenders: list[str] = []
    for path in _gen_python_files():
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno} float literal {node.value!r}"
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "float"
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} float() call")
    assert not offenders, "D1 violated in gen/:\n  " + "\n  ".join(offenders)


@pytest.mark.determinism
def test_float_scan_detects_a_violation(tmp_path: Path) -> None:
    """Positive control: the scan above must be able to see a float."""
    sample = tmp_path / "money_sample.py"
    sample.write_text("RATE = 0.02\n\n\ndef fee(g):\n    return float(g) * RATE\n", "utf-8")
    tree = ast.parse(sample.read_text("utf-8"))
    literals = [
        n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)
    ]
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "float"
    ]
    assert len(literals) == 1 and len(calls) == 1


def test_holdout_seed_is_independently_valid(holdout_dir: Path) -> None:
    """seed 999 is the held-out set; it must stand on its own."""
    truth = read_truth(holdout_dir, 999)
    assert truth["seed"] == 999
    assert truth["generator_commit"] is None
    assert truth["counts"]["cases"] > 0
    assert check_cardinality(_edges(holdout_dir, 999)) == []

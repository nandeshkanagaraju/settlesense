"""The AI evaluation set: seeds 1000-1019, and the invariants that make it usable.

WHY THIS FILE RUNS WITHOUT THE DATA. The twenty datasets are ~146MB and
gitignored, so a fresh clone does not have them. A suite that skipped in that
case would report success for a set nobody checked - the exact failure this
project has a whole test file about. So the structure is:

  ALWAYS   verify EVAL_SET_MANIFEST.json: all twenty seeds recorded, the
           structural invariants hold in the recorded numbers, the spread is
           within what chance explains, no seed silently excluded.
  IF DATA  re-derive every figure from the datasets and assert it matches the
           manifest exactly, including the content hashes.

The manifest is therefore a CHECKABLE CLAIM rather than a note. The
always-branch is not a weaker version of the real test; it is the part that
catches a manifest edited to make a result look better, which is the failure
mode that actually matters here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from settlesense.config import AppConfig, load_config
from settlesense.ingest import DayDataset, load_dataset
from settlesense.matching.duplicates import find_candidate_duplicates, find_confirmed_duplicates
from settlesense.matching.engine import (
    fuzzy_verdicts_for,
    merge_days,
    residual_cases,
    run,
)
from settlesense.matching.fuzzy_utr import ScoringPath

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "data" / "eval"
MANIFEST_PATH = REPO / "EVAL_SET_MANIFEST.json"
AS_OF = date(2026, 11, 30)

FIRST_SEED, LAST_SEED = 1000, 1019
SEED_COUNT = LAST_SEED - FIRST_SEED + 1


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(REPO / "config")


def _data_present() -> bool:
    return EVAL_DIR.is_dir() and len(list(EVAL_DIR.glob("seed_*"))) == SEED_COUNT


needs_data = pytest.mark.skipif(
    not _data_present(),
    reason="data/eval/ absent (gitignored, ~146MB). Regenerate with `make eval-set`.",
)


# ---------------------------------------------------------------------------
# Always: the manifest is a checkable claim
# ---------------------------------------------------------------------------


def test_all_twenty_seeds_are_recorded_and_none_excluded(manifest: dict[str, Any]) -> None:
    """The pre-registration said all twenty, no exclusions. Asserted, not trusted.

    An exclusion is not forbidden - it is forbidden to be SILENT. If a seed is
    ever dropped it goes in `excluded` with a reason, and this test fails until
    README records it in the same commit.
    """
    declared = manifest["seed_range"]
    assert (declared["first"], declared["last"], declared["count"]) == (
        FIRST_SEED,
        LAST_SEED,
        SEED_COUNT,
    )
    assert declared["excluded"] == [], (
        f"seed(s) excluded: {declared['excluded']}. The README declaration commits "
        "to all twenty; an exclusion must be recorded there in the same commit."
    )
    assert sorted(int(s) for s in manifest["seeds"]) == list(range(FIRST_SEED, LAST_SEED + 1))


def test_the_manifest_names_the_commit_that_declared_the_range(
    manifest: dict[str, Any],
) -> None:
    """The declaration must PRECEDE the numbers, and be checkable in git log."""
    import subprocess

    declared_in = manifest["declared_in_commit"]
    assert declared_in, "no declaring commit recorded"
    result = subprocess.run(
        ["git", "log", "--format=%H", "-1", declared_in],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, f"declaring commit {declared_in} is not in this repository"
    message = subprocess.run(
        ["git", "log", "--format=%B", "-1", declared_in], capture_output=True, text=True, cwd=REPO
    ).stdout
    assert "PRE-REGISTER" in message, (
        f"commit {declared_in} does not read as the pre-registration: {message[:120]!r}"
    )


def test_the_structural_invariants_hold_in_every_recorded_seed(
    manifest: dict[str, Any],
) -> None:
    """Two exact relationships, and they are not coincidences.

      cases    == 5000 + pairs   each ambiguous pair adds exactly one repeat
                                 purchase, which is a new payment and so a new
                                 ReconciliationCase
      residual == 2 * pairs      the engine flags BOTH halves of a pair,
                                 because it cannot know which was injected

    If either breaks, something changed about what a duplicate pair IS, and the
    507 decisions are no longer 507 of the same thing.
    """
    for seed, row in sorted(manifest["seeds"].items()):
        pairs = row["duplicate_candidate_pairs"]
        assert row["cases"] == 5000 + pairs, (
            f"seed {seed}: {row['cases']} cases with {pairs} pairs; the "
            "one-repeat-per-pair relationship broke"
        )
        assert row["residual_cases"] == 2 * pairs, (
            f"seed {seed}: {row['residual_cases']} residual with {pairs} pairs; the "
            "engine stopped flagging both halves"
        )
        assert row["batches"] == 39, f"seed {seed} has {row['batches']} batches, expected 39"


def test_the_pair_count_is_stable_against_chance_not_against_a_guess(
    manifest: dict[str, Any],
) -> None:
    """THE stability requirement, asserted against the right yardstick.

    "~26 +/- a few" is not testable - a few is not a number. Duplicate pairs
    are rare independent events over ~5000 chains, so the count is Poisson-ish
    and its expected standard deviation is sqrt(mean). Measuring dispersion
    against THAT is what distinguishes ordinary sampling noise from a noise
    rate interacting with something seed-dependent.

    Observed: sd 4.25 against a Poisson 5.03, i.e. 0.84x - the spread is
    slightly NARROWER than chance alone would produce. The extremes (18 and 34)
    sit at -1.46 and +1.72 sigma. Nothing to investigate.
    """
    pairs = [row["duplicate_candidate_pairs"] for row in manifest["seeds"].values()]
    mean = statistics.mean(pairs)
    poisson_sd = math.sqrt(mean)
    observed_sd = statistics.stdev(pairs)
    dispersion = observed_sd / poisson_sd
    worst = max(abs(p - mean) / poisson_sd for p in pairs)
    print(
        f"\n  pairs: n={len(pairs)} total={sum(pairs)} mean={mean:.2f}"
        f"\n  dispersion vs Poisson: {dispersion:.2f}x   worst deviation: {worst:.2f} sigma"
        f"\n  range: {min(pairs)}..{max(pairs)}"
    )
    assert 0.4 <= dispersion <= 2.0, (
        f"pair-count dispersion is {dispersion:.2f}x the Poisson expectation. "
        "Over-dispersion means a noise rate is interacting with something "
        "seed-dependent; under-dispersion that far means the counts are not "
        "independent. Investigate before using this set."
    )
    assert worst <= 3.0, f"a seed deviates by {worst:.2f} sigma; investigate before using it"


def test_the_declared_decision_count_is_reachable(manifest: dict[str, Any]) -> None:
    """The pre-registration promised ~520 decisions. Realised 507.

    Counted as ONE decision per pair, which is what the question actually is:
    "which of these two orders is the duplicate". The engine flags both halves,
    so 1014 CASES are residual - reported separately so the two numbers are
    never confused for each other.
    """
    totals = manifest["totals"]
    pairs = sum(row["duplicate_candidate_pairs"] for row in manifest["seeds"].values())
    assert totals["duplicate_candidate_pairs"] == pairs == 507
    assert totals["cases_flagged_two_per_pair"] == 2 * pairs
    assert 400 <= pairs <= 640, (
        f"{pairs} decisions, against a pre-declared ~520. Outside this band the "
        "set is not the one that was declared."
    )


def test_no_seed_produced_a_natural_batch_amount_collision(manifest: dict[str, Any]) -> None:
    """The Path B generalisation evidence, and it CONFIRMS the limitation.

    Zero collisions across all twenty seeds - 780 batches in total. So Path B's
    abstention rule still has never fired on generated data, and its 6-of-8
    precision on seed 42 remains conditioned on low batch density exactly as
    LIMITATIONS.md says.

    This is a negative result and is recorded as one. Twenty seeds agreeing
    that collisions do not occur is not evidence that Path B handles them; it
    is evidence that this generator cannot produce the case that would test it.
    """
    collisions = {seed: row["batch_amount_collisions"] for seed, row in manifest["seeds"].items()}
    total_batches = sum(row["batches"] for row in manifest["seeds"].values())
    print(
        f"\n  batch amount collisions across {len(collisions)} seeds "
        f"({total_batches} batches): {sum(collisions.values())}"
    )
    assert sum(collisions.values()) == 0, (
        f"a natural collision appeared: {[s for s, c in collisions.items() if c]}. "
        "This is the case LIMITATIONS.md says has never occurred - update the note "
        "and use the seed as real evidence for Path B's abstention rule."
    )


def test_path_b_left_work_unresolved_on_every_seed(manifest: dict[str, Any]) -> None:
    """Path B abstaining or going ambiguous somewhere on every seed is what
    keeps zero-false-accepts from being a claim about a matcher that accepts
    everything."""
    unresolved = {s: row["p8_path_b_unresolved"] for s, row in manifest["seeds"].items()}
    print(
        f"\n  Path B unresolved per seed: "
        f"min={min(unresolved.values())} max={max(unresolved.values())}"
    )
    assert all(count > 0 for count in unresolved.values()), (
        f"Path B resolved everything on seed(s) {[s for s, c in unresolved.items() if not c]}; "
        "a matcher that never abstains has not been shown to be able to"
    )


# ---------------------------------------------------------------------------
# With the data present: the manifest is re-derived, not trusted
# ---------------------------------------------------------------------------


def _load_seed(seed: int, config: AppConfig) -> DayDataset:
    root = EVAL_DIR / f"seed_{seed}"
    days = sorted(
        {
            int(match.group(1))
            for path in root.glob("day*_*.csv")
            if (match := re.match(r"day(\d+)_", path.name))
        }
    )
    return merge_days([load_dataset(root, day, config) for day in days])


def _content_hash(seed: int) -> str:
    root = EVAL_DIR / f"seed_{seed}"
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "_summary.json"):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@needs_data
@pytest.mark.parametrize("seed", range(FIRST_SEED, LAST_SEED + 1))
def test_each_dataset_matches_its_recorded_content_hash(
    seed: int, manifest: dict[str, Any]
) -> None:
    """The datasets are not committed, so this is what makes them verifiable."""
    recorded = manifest["seeds"][str(seed)]
    assert _content_hash(seed) == recorded["content_sha256"], (
        f"seed {seed} does not match its recorded hash. Either the generator "
        "changed - which the freeze forbids - or the datasets were regenerated "
        "with different inputs."
    )


@needs_data
@pytest.mark.parametrize("seed", range(FIRST_SEED, LAST_SEED + 1))
def test_each_seed_reproduces_its_recorded_figures(
    seed: int, manifest: dict[str, Any], config: AppConfig
) -> None:
    """Every number in the manifest, re-derived from the data."""
    recorded = manifest["seeds"][str(seed)]
    dataset = _load_seed(seed, config)
    result = run(dataset, config, AS_OF)
    confirmed = find_confirmed_duplicates(dataset.ledger_rows)
    excluded = frozenset(i for v in confirmed for i in v.row_ids)
    pairs = find_candidate_duplicates(dataset.ledger_rows, excluded)
    verdicts = fuzzy_verdicts_for(dataset, config, AS_OF)

    assert len(result.cases) == recorded["cases"]
    assert len(pairs) == recorded["duplicate_candidate_pairs"]
    assert len(residual_cases(result)) == recorded["residual_cases"]
    assert len(result.batch_links) == recorded["batches"]
    assert sum(1 for b in result.batch_links if b.bank_row_id) == recorded["batches_linked"]
    assert len(verdicts) == recorded["p8_reached"]
    assert sum(1 for v in verdicts if v.path is ScoringPath.PREFIX) == recorded["p8_path_a"]


@needs_data
def test_the_engine_never_raises_on_any_evaluation_seed(config: AppConfig) -> None:
    """Twenty independent datasets through the full M2-M4 pipeline."""
    for seed in range(FIRST_SEED, LAST_SEED + 1):
        assert run(_load_seed(seed, config), config, AS_OF).cases


@needs_data
def test_the_data_presence_check_is_honest() -> None:
    """Guards the skipif from the other side.

    A presence check stuck at False would skip every data test forever while
    the suite reported green. This test only runs when data IS present, so
    reaching it proves the check can be True; the always-branch above covers
    the case where it is False.
    """
    assert _data_present()
    assert len(list(EVAL_DIR.glob("seed_*"))) == SEED_COUNT

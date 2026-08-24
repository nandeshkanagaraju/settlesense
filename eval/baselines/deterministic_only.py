"""M5 baseline - the deterministic engine, with no AI layer.

A THIN WRAPPER, deliberately. It calls settlesense.matching.engine.run and
returns what it returns. There is no reimplementation here, because a baseline
that re-derives the engine's logic is measuring a copy of the engine rather
than the engine, and the two drift the moment either is edited.

On this dataset this baseline IS the full system: the AI layer does not exist
until M7, so `settlesense` and `deterministic_only` return identical results.
That is stated rather than hidden, and `run_eval` prints both anyway - a table
showing them equal is the honest way to say the AI has not contributed yet,
and it becomes the before-and-after once M7 lands.
"""

from __future__ import annotations

from datetime import date

from settlesense.config import AppConfig
from settlesense.ingest import DayDataset
from settlesense.matching.engine import run
from settlesense.types import ReconciliationResult

__all__ = ["run_deterministic_only"]


def run_deterministic_only(
    dataset: DayDataset, config: AppConfig, as_of: date
) -> ReconciliationResult:
    """P1-P9 with no model calls. The headline throughput figure (PDD 8.4a)."""
    return run(dataset, config, as_of)

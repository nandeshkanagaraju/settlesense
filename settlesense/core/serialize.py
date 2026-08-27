"""Canonical serialization of the BUSINESS RESULT. Nothing else (SDD 8.1).

WHY THIS TAKES ONE ARGUMENT AND REFUSES A TUPLE. `run_with_telemetry` returns
`(ReconciliationResult, RunTelemetry)`. The single easiest way to reintroduce
the defect SDD 8.1 exists to prevent is for a caller to hand that whole tuple
to a serializer that helpfully reaches for element zero - because then the
telemetry is INSIDE the call, and the next person to touch it adds "just the
total seconds" to the output. So the tuple is a TypeError here, loudly, rather
than a best-effort read of its first element.

THERE IS NO STRIP STEP, and there is nothing to strip: ReconciliationResult
carries no float, no duration and no timestamp anywhere in its transitive type
graph, which `tests/test_timing.py` asserts by walking that graph.

DETERMINISTIC BY CONSTRUCTION. Keys sorted, no whitespace, Decimal rendered as
its own string rather than through float. Two runs over the same input produce
the same bytes, which is what makes a golden comparison meaningful and what
makes `result_hash` a stable identity rather than a per-process accident.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from decimal import Decimal
from enum import Enum
from typing import Any

from settlesense.types import ReconciliationResult

__all__ = ["SerializationError", "result_hash", "serialize_result"]


class SerializationError(TypeError):
    """Handed something that is not a ReconciliationResult."""


def _encode(value: object) -> str:
    """Decimal and Enum only. Anything else is a type that should not be here.

    Deliberately NOT a permissive `str(value)` fallback: that would silently
    serialize a datetime, a float wrapper, or a Path the moment one appeared
    in the result graph, turning a design violation into a formatting detail.
    """
    if isinstance(value, Decimal):
        # str(), never float(): Decimal("0.10") must not become 0.1.
        return str(value)
    if isinstance(value, Enum):
        return str(value)
    raise SerializationError(
        f"{type(value).__name__} has no canonical encoding and does not belong "
        f"in a ReconciliationResult: {value!r}"
    )


def serialize_result(result: ReconciliationResult) -> str:
    """Canonical JSON for hashing, golden comparison and persistence.

    Takes a ReconciliationResult. Handing it the (result, telemetry) tuple
    raises rather than quietly serializing element zero.
    """
    if isinstance(result, tuple):
        raise SerializationError(
            "serialize_result takes a ReconciliationResult, not the "
            "(result, telemetry) tuple returned by run_with_telemetry. Pass "
            "[0] explicitly at the call site, so that discarding telemetry is "
            "a visible decision rather than something this function did for "
            "you (SDD 8.1)."
        )
    if not isinstance(result, ReconciliationResult):
        raise SerializationError(
            f"serialize_result takes a ReconciliationResult, got {type(result).__name__}"
        )
    payload: dict[str, Any] = dataclasses.asdict(result)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_encode)


def result_hash(result: ReconciliationResult) -> str:
    """sha256 of the canonical bytes. The golden identity of a run."""
    return hashlib.sha256(serialize_result(result).encode("utf-8")).hexdigest()

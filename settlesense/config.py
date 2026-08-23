"""M0 - typed configuration loading.

This module is the enforcement point for two charter rules:

  D12  Scoring weights, thresholds and ratios are Decimal. The loader RAISES
       on any YAML float, naming the offending key path. Decimal-valued
       entries must be written as quoted strings in YAML.
  D13  Every date in this repo falls inside 2026. The loader raises on any
       date outside it.

Two further principles, both deliberate:

  No hidden defaults.  Every key is required. An absent key raises rather
  than falling back to a value buried in code, because a threshold that
  silently reverts to a default is a threshold nobody is measuring.

  No unknown keys.  An unrecognised key raises too. A typo'd key would
  otherwise sit in the file looking authoritative while the engine ignores
  it - which is exactly how a safety budget stops applying without anyone
  noticing.

Everything loads into frozen dataclasses. There are no magic numbers in
engine code; every value the engine compares against arrives from here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import yaml

__all__ = [
    "MONEY_QUANTUM",
    "REQUIRED_YEAR",
    "AppConfig",
    "CalendarConfig",
    "ConfidenceWeights",
    "ConfigError",
    "FuzzyUtrThresholds",
    "HypothesisLimits",
    "MdrRatesConfig",
    "ReportingAssumptions",
    "SafetyBudgets",
    "ThresholdsConfig",
    "ToleranceThresholds",
    "load_calendar",
    "load_config",
    "load_mdr_rates",
    "load_thresholds",
]

# D1: money is quantized to 2 dp with ROUND_HALF_UP, everywhere, always.
MONEY_QUANTUM: Final[Decimal] = Decimal("0.01")

# D13: the simulated timeline is 2026. Any other year is a defect.
REQUIRED_YEAR: Final[int] = 2026

_WEEKDAY_NAMES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
)

_ONE: Final[Decimal] = Decimal("1")
_ZERO: Final[Decimal] = Decimal("0")


class ConfigError(ValueError):
    """Raised on any malformed, missing, unknown or float-valued config entry.

    Always carries the dotted key path so the failure names the line to fix.
    """


# ---------------------------------------------------------------------------
# Primitive readers. Each names its key path in every error it raises.
# ---------------------------------------------------------------------------


def _fail(path: str, problem: str) -> ConfigError:
    return ConfigError(f"config error at {path!r}: {problem}")


def _reject_floats(node: Any, path: str) -> None:
    """D12 - walk the raw parsed tree and raise on the first float found.

    Runs before any coercion, so the error points at the YAML the author
    wrote rather than at a downstream symptom. A float is never acceptable
    in this repo's config: quote the value to make it a Decimal.
    """
    if isinstance(node, float):
        raise _fail(
            path,
            f"D12 violation - YAML float {node!r}. Decimal-valued config must be "
            f'a quoted string, e.g. "{node}". Floats are banned in scoring, '
            f"threshold and money paths.",
        )
    if isinstance(node, Mapping):
        for key, value in node.items():
            _reject_floats(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for index, value in enumerate(node):
            _reject_floats(value, f"{path}[{index}]")


def _mapping(node: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise _fail(path, f"expected a mapping, got {type(node).__name__}")
    for key in node:
        if not isinstance(key, str):
            raise _fail(path, f"expected string keys, got key {key!r}")
    return node


def _require(node: Mapping[str, Any], key: str, path: str) -> Any:
    """Read a required key. Absence is a loud failure, never a default."""
    if key not in node:
        known = ", ".join(sorted(node)) or "<none>"
        raise _fail(f"{path}.{key}", f"required key is missing. Keys present: {known}")
    return node[key]


def _reject_unknown(node: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    """Reject keys the loader does not read, so no config line is silently dead."""
    unknown = sorted(set(node) - allowed)
    if unknown:
        raise _fail(
            path,
            f"unknown key(s) {unknown}. Every key must be read by the loader; an "
            f"unread key is a threshold nobody is applying. Expected only: "
            f"{sorted(allowed)}",
        )


def _str(node: Mapping[str, Any], key: str, path: str) -> str:
    value = _require(node, key, path)
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{path}.{key}", f"expected a non-empty string, got {value!r}")
    return value


def _decimal(node: Mapping[str, Any], key: str, path: str) -> Decimal:
    """Coerce a QUOTED string to Decimal. Anything else raises.

    Ints are rejected as well as floats: a weight written as `1` rather than
    "1.00" reads as a count, and the distinction between a ratio and a count
    is one this project keeps visible.
    """
    where = f"{path}.{key}"
    value = _require(node, key, path)
    if isinstance(value, bool) or not isinstance(value, str):
        raise _fail(
            where,
            f"expected a quoted decimal string, got {type(value).__name__} {value!r}. "
            f'Write it as "{value}" so it loads as Decimal (D12).',
        )
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise _fail(where, f"{value!r} is not a valid decimal") from exc


def _money(node: Mapping[str, Any], key: str, path: str) -> Decimal:
    """A Decimal that is a rupee amount: quantized to 2 dp, ROUND_HALF_UP (D1)."""
    return _decimal(node, key, path).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _int(node: Mapping[str, Any], key: str, path: str) -> int:
    where = f"{path}.{key}"
    value = _require(node, key, path)
    # bool is a subclass of int; a YAML `yes` must not pass as a count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(where, f"expected an integer, got {type(value).__name__} {value!r}")
    return value


def _date_2026(value: Any, path: str) -> date:
    """Parse an ISO date string and enforce D13.

    Dates are written quoted so PyYAML hands back a string and the format is
    checked here rather than guessed by the parser.
    """
    if isinstance(value, date):
        raise _fail(
            path,
            f"date {value.isoformat()!r} is unquoted. Quote it so the format is "
            f"validated explicitly rather than inferred by the YAML parser.",
        )
    if not isinstance(value, str):
        raise _fail(path, f"expected a quoted ISO date string, got {value!r}")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _fail(path, f"{value!r} is not an ISO date (YYYY-MM-DD)") from exc
    if parsed.year != REQUIRED_YEAR:
        raise _fail(
            path,
            f"D13 violation - {value!r} is not in {REQUIRED_YEAR}. Every date in "
            f"this repo is in {REQUIRED_YEAR}.",
        )
    return parsed


def _sequence(node: Mapping[str, Any], key: str, path: str) -> Sequence[Any]:
    where = f"{path}.{key}"
    value = _require(node, key, path)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail(where, f"expected a list, got {type(value).__name__}")
    if not value:
        raise _fail(where, "list is empty; an empty list here is almost always a mistake")
    return value


def _assert_sums_to_one(weights: Mapping[str, Decimal], path: str) -> None:
    total = sum(weights.values(), start=_ZERO)
    if total != _ONE:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(weights.items()))
        raise _fail(path, f"weights must sum to exactly 1, got {total} ({detail})")


def _assert_unit_interval(value: Decimal, path: str) -> Decimal:
    if not (_ZERO <= value <= _ONE):
        raise _fail(path, f"expected a value in [0, 1], got {value}")
    return value


def _assert_positive(value: Decimal | int, path: str) -> None:
    if value <= 0:
        raise _fail(path, f"expected a positive value, got {value}")


def _read_yaml(path: Path) -> Mapping[str, Any]:
    """Read one YAML file, reject floats across the whole tree, return the mapping."""
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {path} is not valid YAML: {exc}") from exc
    if raw is None:
        raise ConfigError(f"config file {path} is empty")
    _reject_floats(raw, path.name)
    return _mapping(raw, path.name)


# ---------------------------------------------------------------------------
# config/mdr_rates.yaml
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MdrRatesConfig:
    """MDR fee rates per merchant profile, per payment method.

    fee = round(gross * rate_for(profile, method), 2)  ROUND_HALF_UP
    gst = round(fee * gst_rate, 2)                     ROUND_HALF_UP
    """

    version: str
    gst_rate: Decimal
    profiles: Mapping[str, Mapping[str, Decimal]]

    def profile_names(self) -> tuple[str, ...]:
        """Profile names, explicitly sorted (D4 - never dict iteration order)."""
        return tuple(sorted(self.profiles))

    def method_names(self, profile: str) -> tuple[str, ...]:
        """Method names for one profile, explicitly sorted (D4)."""
        return tuple(sorted(self._profile(profile)))

    def rate_for(self, profile: str, method: str) -> Decimal:
        """The MDR rate as a fraction of gross. Unknown profile or method raises."""
        methods = self._profile(profile)
        if method not in methods:
            raise ConfigError(
                f"unknown payment method {method!r} for profile {profile!r}. "
                f"Known methods: {sorted(methods)}"
            )
        return methods[method]

    def _profile(self, profile: str) -> Mapping[str, Decimal]:
        if profile not in self.profiles:
            raise ConfigError(
                f"unknown merchant profile {profile!r}. Known profiles: {self.profile_names()}"
            )
        return self.profiles[profile]


def load_mdr_rates(path: Path) -> MdrRatesConfig:
    """Load and validate config/mdr_rates.yaml."""
    raw = _read_yaml(path)
    root = path.name
    _reject_unknown(raw, frozenset({"version", "gst_rate", "profiles"}), root)

    gst_rate = _assert_unit_interval(_decimal(raw, "gst_rate", root), f"{root}.gst_rate")

    profiles_raw = _mapping(_require(raw, "profiles", root), f"{root}.profiles")
    if not profiles_raw:
        raise _fail(f"{root}.profiles", "at least one merchant profile is required")

    profiles: dict[str, Mapping[str, Decimal]] = {}
    for profile_name in sorted(profiles_raw):
        profile_path = f"{root}.profiles.{profile_name}"
        profile_node = _mapping(profiles_raw[profile_name], profile_path)
        _reject_unknown(profile_node, frozenset({"methods"}), profile_path)

        methods_path = f"{profile_path}.methods"
        methods_node = _mapping(_require(profile_node, "methods", profile_path), methods_path)
        if not methods_node:
            raise _fail(methods_path, "at least one payment method is required")

        methods: dict[str, Decimal] = {}
        for method_name in sorted(methods_node):
            rate = _decimal(methods_node, method_name, methods_path)
            methods[method_name] = _assert_unit_interval(rate, f"{methods_path}.{method_name}")
        profiles[profile_name] = MappingProxyType(methods)

    # Every profile must price every method. Deleting one method from one
    # profile would otherwise load clean and only surface at use time, months
    # later, as a wrong fee - which is precisely the silent gap this loader
    # exists to prevent. The expected set is the union across profiles, so the
    # payment-method taxonomy stays in the domain model rather than being
    # hard-coded here.
    expected_methods = frozenset().union(*(set(m) for m in profiles.values()))
    for profile_name in sorted(profiles):
        missing = sorted(expected_methods - set(profiles[profile_name]))
        if missing:
            raise _fail(
                f"{root}.profiles.{profile_name}.methods",
                f"missing rate(s) for method(s) {missing}. Every profile must price "
                f"every method configured anywhere in this file "
                f"({sorted(expected_methods)}). A profile silently missing a method "
                f"produces a wrong fee, not an error, at reconciliation time.",
            )

    return MdrRatesConfig(
        version=_str(raw, "version", root),
        gst_rate=gst_rate,
        profiles=MappingProxyType(profiles),
    )


# ---------------------------------------------------------------------------
# config/calendar_v1.yaml
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalendarConfig:
    """The versioned working-day calendar.

    `version` is recorded in every ReconciliationResult (SDD 8.1), so a result
    always states which calendar produced it.

    The holiday set is SYNTHETIC for the 2026 simulated timeline and is not
    the RBI schedule. See LIMITATIONS.md.
    """

    version: str
    timezone: str
    weekly_offs: frozenset[int]  # 0 = Monday .. 6 = Sunday
    holidays: frozenset[date]
    window_start: date
    window_end: date
    settlement_cycles: Mapping[str, int]  # profile -> T+N in working days

    def holiday_list(self) -> tuple[date, ...]:
        """Holidays, explicitly sorted (D4 - never set iteration order)."""
        return tuple(sorted(self.holidays))

    def weekly_off_list(self) -> tuple[int, ...]:
        """Weekly off weekday numbers, explicitly sorted (D4)."""
        return tuple(sorted(self.weekly_offs))

    def is_working_day(self, day: date) -> bool:
        """True if `day` is neither a weekly off nor a holiday.

        A pure lookup over calendar data. T+N roll-forward is matching logic
        and lives in settlesense/matching/timing.py, not here.
        """
        return day.weekday() not in self.weekly_offs and day not in self.holidays

    def settlement_cycle_for(self, profile: str) -> int:
        """T+N in working days for one merchant profile. Unknown profile raises."""
        if profile not in self.settlement_cycles:
            raise ConfigError(
                f"no settlement cycle configured for profile {profile!r}. "
                f"Known profiles: {tuple(sorted(self.settlement_cycles))}"
            )
        return self.settlement_cycles[profile]


def load_calendar(path: Path) -> CalendarConfig:
    """Load and validate config/calendar_v1.yaml."""
    raw = _read_yaml(path)
    root = path.name
    _reject_unknown(
        raw,
        frozenset(
            {
                "version",
                "timezone",
                "weekly_offs",
                "holidays",
                "simulation_window",
                "settlement_cycles",
            }
        ),
        root,
    )

    weekly_offs: set[int] = set()
    for index, name in enumerate(_sequence(raw, "weekly_offs", root)):
        where = f"{root}.weekly_offs[{index}]"
        if not isinstance(name, str) or name.lower() not in _WEEKDAY_NAMES:
            raise _fail(
                where,
                f"expected a weekday name from {sorted(_WEEKDAY_NAMES)}, got {name!r}",
            )
        weekly_offs.add(_WEEKDAY_NAMES[name.lower()])
    if len(weekly_offs) == 7:
        raise _fail(f"{root}.weekly_offs", "every day is a weekly off; no day would settle")

    window_path = f"{root}.simulation_window"
    window = _mapping(_require(raw, "simulation_window", root), window_path)
    _reject_unknown(window, frozenset({"start", "end"}), window_path)
    window_start = _date_2026(_require(window, "start", window_path), f"{window_path}.start")
    window_end = _date_2026(_require(window, "end", window_path), f"{window_path}.end")
    if window_start > window_end:
        raise _fail(window_path, f"start {window_start} is after end {window_end}")

    holidays: set[date] = set()
    for index, value in enumerate(_sequence(raw, "holidays", root)):
        where = f"{root}.holidays[{index}]"
        holiday = _date_2026(value, where)
        if not (window_start <= holiday <= window_end):
            raise _fail(
                where,
                f"holiday {holiday} falls outside the simulation window "
                f"{window_start}..{window_end}. A holiday the timeline never reaches "
                f"is dead config.",
            )
        if holiday in holidays:
            raise _fail(where, f"duplicate holiday {holiday}")
        holidays.add(holiday)

    cycles_path = f"{root}.settlement_cycles"
    cycles_node = _mapping(_require(raw, "settlement_cycles", root), cycles_path)
    if not cycles_node:
        raise _fail(cycles_path, "at least one profile settlement cycle is required")
    cycles: dict[str, int] = {}
    for profile_name in sorted(cycles_node):
        value = _int(cycles_node, profile_name, cycles_path)
        _assert_positive(value, f"{cycles_path}.{profile_name}")
        cycles[profile_name] = value

    return CalendarConfig(
        version=_str(raw, "version", root),
        timezone=_str(raw, "timezone", root),
        weekly_offs=frozenset(weekly_offs),
        holidays=frozenset(holidays),
        window_start=window_start,
        window_end=window_end,
        settlement_cycles=MappingProxyType(cycles),
    )


# ---------------------------------------------------------------------------
# config/thresholds.yaml
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuzzyUtrThresholds:
    """Fuzzy UTR scoring weights and acceptance gates (SDD 4.3).

    Accept ONLY if best >= accept_score AND (best - runner_up) >= min_separation.
    Failing either emits UTR_TRUNCATED_MAPPING into the residual set with every
    candidate attached, rather than picking the leader.
    """

    weight_prefix: Decimal
    weight_edit: Decimal
    weight_amount: Decimal
    score_quantum: Decimal
    accept_score: Decimal
    min_separation: Decimal


@dataclass(frozen=True)
class ToleranceThresholds:
    """Arithmetic tolerances, in rupees, quantized to 2 dp."""

    rounding_rupees: Decimal
    verifier_rupees: Decimal


@dataclass(frozen=True)
class ConfidenceWeights:
    """Verification-derived confidence weights (SDD 4.6).

    Confidence is computed by the verification layer and is never the model's
    self-report. Auto-confirm requires confidence >= auto_confirm AND
    verification_passed; confidence alone can never confirm.
    """

    weight_verification_passed: Decimal
    weight_residual_within_tolerance: Decimal
    weight_evidence_completeness: Decimal
    weight_candidate_separation: Decimal
    weight_freshness: Decimal
    auto_confirm: Decimal


@dataclass(frozen=True)
class HypothesisLimits:
    """Bounds on the hypothesis loop (SDD 4.4, 4.8). Counts, so ints."""

    max_per_exception: int
    llm_max_retries: int


@dataclass(frozen=True)
class SafetyBudgets:
    """Pre-declared per-category enablement budgets (PDD 7.3).

    Project safety thresholds for a synthetic evaluation. Not a claim about
    acceptable production loss.
    """

    max_residual_false_match_rate: Decimal
    max_gross_exposure_false_match_rupees: Decimal
    max_cost_per_1000_rows_rupees: Decimal


@dataclass(frozen=True)
class ReportingAssumptions:
    """Stated assumptions behind derived estimates (PDD 8.4)."""

    assumed_review_minutes_per_exception: int


@dataclass(frozen=True)
class ThresholdsConfig:
    version: str
    fuzzy_utr: FuzzyUtrThresholds
    tolerance: ToleranceThresholds
    confidence: ConfidenceWeights
    hypothesis: HypothesisLimits
    safety_budgets: SafetyBudgets
    reporting: ReportingAssumptions


def load_thresholds(path: Path) -> ThresholdsConfig:
    """Load and validate config/thresholds.yaml."""
    raw = _read_yaml(path)
    root = path.name
    _reject_unknown(
        raw,
        frozenset(
            {
                "version",
                "fuzzy_utr",
                "tolerance",
                "confidence",
                "hypothesis",
                "safety_budgets",
                "reporting",
            }
        ),
        root,
    )

    fuzzy_path = f"{root}.fuzzy_utr"
    fuzzy_node = _mapping(_require(raw, "fuzzy_utr", root), fuzzy_path)
    _reject_unknown(
        fuzzy_node,
        frozenset(
            {
                "weight_prefix",
                "weight_edit",
                "weight_amount",
                "score_quantum",
                "accept_score",
                "min_separation",
            }
        ),
        fuzzy_path,
    )
    fuzzy = FuzzyUtrThresholds(
        weight_prefix=_decimal(fuzzy_node, "weight_prefix", fuzzy_path),
        weight_edit=_decimal(fuzzy_node, "weight_edit", fuzzy_path),
        weight_amount=_decimal(fuzzy_node, "weight_amount", fuzzy_path),
        score_quantum=_decimal(fuzzy_node, "score_quantum", fuzzy_path),
        accept_score=_decimal(fuzzy_node, "accept_score", fuzzy_path),
        min_separation=_decimal(fuzzy_node, "min_separation", fuzzy_path),
    )
    _assert_sums_to_one(
        {
            "weight_prefix": fuzzy.weight_prefix,
            "weight_edit": fuzzy.weight_edit,
            "weight_amount": fuzzy.weight_amount,
        },
        fuzzy_path,
    )
    _assert_unit_interval(fuzzy.accept_score, f"{fuzzy_path}.accept_score")
    _assert_unit_interval(fuzzy.min_separation, f"{fuzzy_path}.min_separation")
    _assert_positive(fuzzy.score_quantum, f"{fuzzy_path}.score_quantum")

    tolerance_path = f"{root}.tolerance"
    tolerance_node = _mapping(_require(raw, "tolerance", root), tolerance_path)
    _reject_unknown(
        tolerance_node, frozenset({"rounding_rupees", "verifier_rupees"}), tolerance_path
    )
    tolerance = ToleranceThresholds(
        rounding_rupees=_money(tolerance_node, "rounding_rupees", tolerance_path),
        verifier_rupees=_money(tolerance_node, "verifier_rupees", tolerance_path),
    )
    _assert_positive(tolerance.rounding_rupees, f"{tolerance_path}.rounding_rupees")
    _assert_positive(tolerance.verifier_rupees, f"{tolerance_path}.verifier_rupees")

    confidence_path = f"{root}.confidence"
    confidence_node = _mapping(_require(raw, "confidence", root), confidence_path)
    _reject_unknown(
        confidence_node,
        frozenset(
            {
                "weight_verification_passed",
                "weight_residual_within_tolerance",
                "weight_evidence_completeness",
                "weight_candidate_separation",
                "weight_freshness",
                "auto_confirm",
            }
        ),
        confidence_path,
    )
    confidence = ConfidenceWeights(
        weight_verification_passed=_decimal(
            confidence_node, "weight_verification_passed", confidence_path
        ),
        weight_residual_within_tolerance=_decimal(
            confidence_node, "weight_residual_within_tolerance", confidence_path
        ),
        weight_evidence_completeness=_decimal(
            confidence_node, "weight_evidence_completeness", confidence_path
        ),
        weight_candidate_separation=_decimal(
            confidence_node, "weight_candidate_separation", confidence_path
        ),
        weight_freshness=_decimal(confidence_node, "weight_freshness", confidence_path),
        auto_confirm=_decimal(confidence_node, "auto_confirm", confidence_path),
    )
    _assert_sums_to_one(
        {
            "weight_verification_passed": confidence.weight_verification_passed,
            "weight_residual_within_tolerance": confidence.weight_residual_within_tolerance,
            "weight_evidence_completeness": confidence.weight_evidence_completeness,
            "weight_candidate_separation": confidence.weight_candidate_separation,
            "weight_freshness": confidence.weight_freshness,
        },
        confidence_path,
    )
    _assert_unit_interval(confidence.auto_confirm, f"{confidence_path}.auto_confirm")

    hypothesis_path = f"{root}.hypothesis"
    hypothesis_node = _mapping(_require(raw, "hypothesis", root), hypothesis_path)
    _reject_unknown(
        hypothesis_node, frozenset({"max_per_exception", "llm_max_retries"}), hypothesis_path
    )
    hypothesis = HypothesisLimits(
        max_per_exception=_int(hypothesis_node, "max_per_exception", hypothesis_path),
        llm_max_retries=_int(hypothesis_node, "llm_max_retries", hypothesis_path),
    )
    _assert_positive(hypothesis.max_per_exception, f"{hypothesis_path}.max_per_exception")
    if hypothesis.llm_max_retries < 0:
        raise _fail(f"{hypothesis_path}.llm_max_retries", "must not be negative")

    budgets_path = f"{root}.safety_budgets"
    budgets_node = _mapping(_require(raw, "safety_budgets", root), budgets_path)
    _reject_unknown(
        budgets_node,
        frozenset(
            {
                "max_residual_false_match_rate",
                "max_gross_exposure_false_match_rupees",
                "max_cost_per_1000_rows_rupees",
            }
        ),
        budgets_path,
    )
    budgets = SafetyBudgets(
        max_residual_false_match_rate=_decimal(
            budgets_node, "max_residual_false_match_rate", budgets_path
        ),
        max_gross_exposure_false_match_rupees=_money(
            budgets_node, "max_gross_exposure_false_match_rupees", budgets_path
        ),
        max_cost_per_1000_rows_rupees=_money(
            budgets_node, "max_cost_per_1000_rows_rupees", budgets_path
        ),
    )
    _assert_unit_interval(
        budgets.max_residual_false_match_rate,
        f"{budgets_path}.max_residual_false_match_rate",
    )

    reporting_path = f"{root}.reporting"
    reporting_node = _mapping(_require(raw, "reporting", root), reporting_path)
    _reject_unknown(
        reporting_node, frozenset({"assumed_review_minutes_per_exception"}), reporting_path
    )
    reporting = ReportingAssumptions(
        assumed_review_minutes_per_exception=_int(
            reporting_node, "assumed_review_minutes_per_exception", reporting_path
        )
    )
    _assert_positive(
        reporting.assumed_review_minutes_per_exception,
        f"{reporting_path}.assumed_review_minutes_per_exception",
    )

    return ThresholdsConfig(
        version=_str(raw, "version", root),
        fuzzy_utr=fuzzy,
        tolerance=tolerance,
        confidence=confidence,
        hypothesis=hypothesis,
        safety_budgets=budgets,
        reporting=reporting,
    )


# ---------------------------------------------------------------------------
# The whole configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """Every configured value the engine reads, plus a hash identifying them.

    `config_hash` is recorded in ReconciliationResult (SDD 8.1) so a result
    always states which configuration produced it. It is a sha256 over the
    canonical semantic content (D10), not over the YAML text: reflowing a
    comment leaves the hash unchanged, altering a threshold does not.
    """

    mdr: MdrRatesConfig
    calendar: CalendarConfig
    thresholds: ThresholdsConfig
    config_hash: str

    def calendar_version(self) -> str:
        """Convenience accessor - the string stamped into every result."""
        return self.calendar.version


def _canonical(config: AppConfig) -> str:
    """Canonical JSON for hashing: sorted keys, Decimals as strings, no floats."""
    payload = {
        "mdr": {
            "version": config.mdr.version,
            "gst_rate": str(config.mdr.gst_rate),
            "profiles": {
                profile: {
                    method: str(config.mdr.rate_for(profile, method))
                    for method in config.mdr.method_names(profile)
                }
                for profile in config.mdr.profile_names()
            },
        },
        "calendar": {
            "version": config.calendar.version,
            "timezone": config.calendar.timezone,
            "weekly_offs": list(config.calendar.weekly_off_list()),
            "holidays": [day.isoformat() for day in config.calendar.holiday_list()],
            "window_start": config.calendar.window_start.isoformat(),
            "window_end": config.calendar.window_end.isoformat(),
            "settlement_cycles": {
                profile: config.calendar.settlement_cycle_for(profile)
                for profile in sorted(config.calendar.settlement_cycles)
            },
        },
        "thresholds": {
            "version": config.thresholds.version,
            "fuzzy_utr": {
                "weight_prefix": str(config.thresholds.fuzzy_utr.weight_prefix),
                "weight_edit": str(config.thresholds.fuzzy_utr.weight_edit),
                "weight_amount": str(config.thresholds.fuzzy_utr.weight_amount),
                "score_quantum": str(config.thresholds.fuzzy_utr.score_quantum),
                "accept_score": str(config.thresholds.fuzzy_utr.accept_score),
                "min_separation": str(config.thresholds.fuzzy_utr.min_separation),
            },
            "tolerance": {
                "rounding_rupees": str(config.thresholds.tolerance.rounding_rupees),
                "verifier_rupees": str(config.thresholds.tolerance.verifier_rupees),
            },
            "confidence": {
                "weight_verification_passed": str(
                    config.thresholds.confidence.weight_verification_passed
                ),
                "weight_residual_within_tolerance": str(
                    config.thresholds.confidence.weight_residual_within_tolerance
                ),
                "weight_evidence_completeness": str(
                    config.thresholds.confidence.weight_evidence_completeness
                ),
                "weight_candidate_separation": str(
                    config.thresholds.confidence.weight_candidate_separation
                ),
                "weight_freshness": str(config.thresholds.confidence.weight_freshness),
                "auto_confirm": str(config.thresholds.confidence.auto_confirm),
            },
            "hypothesis": {
                "max_per_exception": config.thresholds.hypothesis.max_per_exception,
                "llm_max_retries": config.thresholds.hypothesis.llm_max_retries,
            },
            "safety_budgets": {
                "max_residual_false_match_rate": str(
                    config.thresholds.safety_budgets.max_residual_false_match_rate
                ),
                "max_gross_exposure_false_match_rupees": str(
                    config.thresholds.safety_budgets.max_gross_exposure_false_match_rupees
                ),
                "max_cost_per_1000_rows_rupees": str(
                    config.thresholds.safety_budgets.max_cost_per_1000_rows_rupees
                ),
            },
            "reporting": {
                "assumed_review_minutes_per_exception": (
                    config.thresholds.reporting.assumed_review_minutes_per_exception
                )
            },
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_config(config_dir: Path) -> AppConfig:
    """Load every config file from `config_dir` into one frozen AppConfig.

    Raises ConfigError, naming the dotted key path, on: a missing file, invalid
    YAML, a YAML float (D12), a date outside 2026 (D13), a missing key, an
    unknown key, a malformed value, or weights that do not sum to 1.
    """
    if not config_dir.is_dir():
        raise ConfigError(f"config directory not found: {config_dir}")

    mdr = load_mdr_rates(config_dir / "mdr_rates.yaml")
    calendar = load_calendar(config_dir / "calendar_v1.yaml")
    thresholds = load_thresholds(config_dir / "thresholds.yaml")

    # Cross-file check: every profile with an MDR rate table needs a settlement
    # cycle, and vice versa. A profile configured in one file and not the other
    # is a silent gap that only shows up as a wrong T+N months later.
    mdr_profiles = set(mdr.profile_names())
    calendar_profiles = set(calendar.settlement_cycles)
    if mdr_profiles != calendar_profiles:
        raise ConfigError(
            "merchant profiles disagree across config files: "
            f"mdr_rates.yaml has {sorted(mdr_profiles)}, "
            f"calendar_v1.yaml has {sorted(calendar_profiles)}. "
            f"Only in mdr_rates: {sorted(mdr_profiles - calendar_profiles)}. "
            f"Only in calendar: {sorted(calendar_profiles - mdr_profiles)}."
        )

    partial = AppConfig(mdr=mdr, calendar=calendar, thresholds=thresholds, config_hash="")
    digest = hashlib.sha256(_canonical(partial).encode("utf-8")).hexdigest()[:16]
    return AppConfig(mdr=mdr, calendar=calendar, thresholds=thresholds, config_hash=digest)

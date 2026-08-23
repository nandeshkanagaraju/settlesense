"""M0 - tests for the typed configuration loader.

Covers the loader's contract: every config file loads into a frozen dataclass,
every numeric arrives as Decimal rather than float (D12), and every way of
getting it wrong fails loudly with the offending key named.
"""

from __future__ import annotations

import dataclasses
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from settlesense.config import (
    AppConfig,
    CalendarConfig,
    ConfigError,
    MdrRatesConfig,
    ThresholdsConfig,
    load_calendar,
    load_config,
    load_mdr_rates,
    load_thresholds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"

MDR = "mdr_rates.yaml"
CALENDAR = "calendar_v1.yaml"
THRESHOLDS = "thresholds.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_config(tmp_path: Path) -> Path:
    """A private, writable copy of config/ that a test may corrupt freely."""
    target = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, target)
    return target


def _patch(config_dir: Path, filename: str, old: str, new: str) -> Path:
    """Replace the first occurrence of `old` in one config file.

    Asserts `old` is present, so a test cannot silently pass because the
    fixture text it meant to corrupt had been renamed.
    """
    path = config_dir / filename
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture text {old!r} not found in {filename}; update the test"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return config_dir


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    return _copy_config(tmp_path)


# ---------------------------------------------------------------------------
# 1. Each config loads into its frozen dataclass
# ---------------------------------------------------------------------------


def test_mdr_rates_loads() -> None:
    mdr = load_mdr_rates(CONFIG_DIR / MDR)
    assert isinstance(mdr, MdrRatesConfig)
    assert mdr.profile_names() == ("profile_a", "profile_b", "profile_c")
    assert mdr.rate_for("profile_a", "card") == Decimal("0.0200")
    assert mdr.rate_for("profile_b", "card") == Decimal("0.0235")
    assert mdr.rate_for("profile_c", "card") == Decimal("0.0180")
    # UPI is zero-rated on every profile.
    for profile in mdr.profile_names():
        assert mdr.rate_for(profile, "upi") == Decimal("0")


def test_calendar_loads() -> None:
    calendar = load_calendar(CONFIG_DIR / CALENDAR)
    assert isinstance(calendar, CalendarConfig)
    assert calendar.version == "calendar_v1"
    assert calendar.weekly_off_list() == (5, 6)  # Saturday, Sunday
    assert len(calendar.holiday_list()) == 8
    assert calendar.settlement_cycle_for("profile_a") == 2
    assert calendar.settlement_cycle_for("profile_b") == 1
    assert calendar.settlement_cycle_for("profile_c") == 3


def test_thresholds_load() -> None:
    thresholds = load_thresholds(CONFIG_DIR / THRESHOLDS)
    assert isinstance(thresholds, ThresholdsConfig)
    assert thresholds.fuzzy_utr.accept_score == Decimal("0.85")
    assert thresholds.fuzzy_utr.min_separation == Decimal("0.15")
    assert thresholds.tolerance.rounding_rupees == Decimal("1.00")
    assert thresholds.confidence.auto_confirm == Decimal("0.80")
    assert thresholds.hypothesis.max_per_exception == 3


def test_app_config_loads() -> None:
    config = load_config(CONFIG_DIR)
    assert isinstance(config, AppConfig)
    assert config.calendar_version() == "calendar_v1"
    assert len(config.config_hash) == 16


def test_all_dates_are_2026() -> None:
    """D13 - the loader is one of the places this is enforced."""
    calendar = load_calendar(CONFIG_DIR / CALENDAR)
    for holiday in calendar.holiday_list():
        assert holiday.year == 2026, f"D13 violation: {holiday}"
    assert calendar.window_start.year == 2026
    assert calendar.window_end.year == 2026


# ---------------------------------------------------------------------------
# 2. A deleted required key raises an error naming that key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "line", "key"),
    [
        (MDR, 'gst_rate: "0.1800"', "gst_rate"),
        (MDR, 'card: "0.0200"', "card"),
        (CALENDAR, "timezone: Asia/Kolkata", "timezone"),
        (THRESHOLDS, 'accept_score: "0.85"', "accept_score"),
        (THRESHOLDS, 'auto_confirm: "0.80"', "auto_confirm"),
        (THRESHOLDS, "max_per_exception: 3", "max_per_exception"),
        (THRESHOLDS, 'weight_freshness: "0.05"', "weight_freshness"),
    ],
)
def test_missing_required_key_raises_naming_the_key(
    config_dir: Path, filename: str, line: str, key: str
) -> None:
    _patch(config_dir, filename, line, "")
    with pytest.raises(ConfigError) as excinfo:
        load_config(config_dir)
    message = str(excinfo.value)
    assert key in message, f"error does not name the missing key {key!r}: {message}"
    assert filename in message, f"error does not name the file {filename!r}: {message}"


def test_missing_key_error_is_specific_not_generic(config_dir: Path) -> None:
    """The message must point at the line to fix, not merely say 'invalid config'."""
    _patch(config_dir, MDR, 'gst_rate: "0.1800"', "")
    with pytest.raises(ConfigError) as excinfo:
        load_config(config_dir)
    message = str(excinfo.value)
    assert "mdr_rates.yaml.gst_rate" in message
    assert "required key is missing" in message
    # It also lists what WAS present, so a typo is diagnosable at a glance.
    assert "Keys present:" in message


def test_profile_missing_a_method_raises(config_dir: Path) -> None:
    """A method priced on two profiles and absent on the third must not load.

    Regression test. The loader originally iterated whatever methods a profile
    happened to declare, so deleting one loaded clean and only surfaced as a
    wrong fee at reconciliation time.
    """
    _patch(config_dir, MDR, '      wallet: "0.0210"', "")  # profile_a only
    with pytest.raises(ConfigError) as excinfo:
        load_config(config_dir)
    message = str(excinfo.value)
    assert "profiles.profile_a.methods" in message, message
    assert "wallet" in message, message


def test_all_profiles_price_all_methods() -> None:
    mdr = load_mdr_rates(CONFIG_DIR / MDR)
    method_sets = {profile: mdr.method_names(profile) for profile in mdr.profile_names()}
    assert len(set(method_sets.values())) == 1, method_sets
    assert method_sets["profile_a"] == ("card", "netbanking", "upi", "wallet")


def test_missing_file_raises(config_dir: Path) -> None:
    (config_dir / THRESHOLDS).unlink()
    with pytest.raises(ConfigError, match=r"config file not found.*thresholds\.yaml"):
        load_config(config_dir)


# ---------------------------------------------------------------------------
# 3. A rate written as "2.00%" rather than a number raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    ['"2.00%"', '"2%"', '"₹2.00"', '"two percent"', '"0.02x"', '"1,0.02"'],
)
def test_non_numeric_rate_raises(config_dir: Path, bad_value: str) -> None:
    _patch(config_dir, MDR, 'card: "0.0200"', f"card: {bad_value}")
    with pytest.raises(ConfigError) as excinfo:
        load_config(config_dir)
    message = str(excinfo.value)
    assert "profile_a.methods.card" in message, message
    assert "not a valid decimal" in message, message


def test_percent_string_is_not_silently_coerced(config_dir: Path) -> None:
    """A '%' suffix must raise, never be stripped and read as 2.00 (a 200% rate)."""
    _patch(config_dir, MDR, 'card: "0.0200"', 'card: "2.00%"')
    with pytest.raises(ConfigError):
        load_config(config_dir)


def test_yaml_float_raises_d12(config_dir: Path) -> None:
    """D12 - an unquoted decimal parses as a YAML float and must be rejected."""
    _patch(config_dir, MDR, 'card: "0.0200"', "card: 0.0200")
    with pytest.raises(ConfigError) as excinfo:
        load_config(config_dir)
    message = str(excinfo.value)
    assert "D12" in message, message
    assert "profile_a.methods.card" in message, message


# ---------------------------------------------------------------------------
# 4. Every rate is a Decimal, never a float
# ---------------------------------------------------------------------------


def test_every_rate_is_decimal_not_float() -> None:
    mdr = load_mdr_rates(CONFIG_DIR / MDR)
    # Checked through `type(...)` rather than `isinstance(..., float)`. mypy
    # proves the isinstance branch unreachable from the ANNOTATION - which is a
    # claim about the loader, not a guarantee about what it returns. D12 exists
    # because a float can arrive from YAML at runtime, so the check must survive
    # the type checker being satisfied.
    assert type(mdr.gst_rate) is Decimal, f"gst_rate is {type(mdr.gst_rate).__name__}"
    for profile in mdr.profile_names():
        for method in mdr.method_names(profile):
            rate = mdr.rate_for(profile, method)
            assert type(rate) is Decimal, (
                f"{profile}.{method} is {type(rate).__name__}, not Decimal (D12)"
            )


def test_every_threshold_is_decimal_not_float() -> None:
    """Sweep every Decimal-annotated field on the thresholds tree (D12)."""
    thresholds = load_thresholds(CONFIG_DIR / THRESHOLDS)
    checked = 0
    for group_field in dataclasses.fields(thresholds):
        group = getattr(thresholds, group_field.name)
        if not dataclasses.is_dataclass(group):
            continue
        for field in dataclasses.fields(group):
            value = getattr(group, field.name)
            if isinstance(value, int) and not isinstance(value, bool):
                continue  # genuine counts: max_per_exception, retries, minutes
            # `type(...) is Decimal`, not isinstance: a float is not a Decimal
            # subclass, so mypy proves the isinstance-float branch unreachable
            # and refuses it. The runtime check must survive that - D12 is about
            # what YAML actually yields, not about what the annotation promises.
            assert type(value) is Decimal, (
                f"thresholds.{group_field.name}.{field.name} is "
                f"{type(value).__name__}, expected Decimal (D12)"
            )
            checked += 1
    assert checked > 0, "swept nothing; the thresholds tree changed shape"


# ---------------------------------------------------------------------------
# 5. GST is exactly 0.18 for all three profiles
# ---------------------------------------------------------------------------


def test_gst_rate_is_18_percent_for_every_profile() -> None:
    mdr = load_mdr_rates(CONFIG_DIR / MDR)
    assert mdr.profile_names() == ("profile_a", "profile_b", "profile_c")
    for profile in mdr.profile_names():
        # gst_rate is configured once and applies to every profile; asserting it
        # per profile keeps the test honest if it is ever made per-profile.
        assert mdr.gst_rate == Decimal("0.18"), f"{profile}: GST is {mdr.gst_rate}"
        assert isinstance(mdr.gst_rate, Decimal)


def test_gst_rate_equals_18_percent_exactly_not_approximately() -> None:
    """Decimal("0.1800") == Decimal("0.18") numerically, and neither is a float."""
    mdr = load_mdr_rates(CONFIG_DIR / MDR)
    assert mdr.gst_rate == Decimal("0.18")
    assert mdr.gst_rate * Decimal("100") == Decimal("18.0000")
    # The float trap this rule exists to avoid: binary 0.18 is not 0.18, so a
    # Decimal that genuinely holds 0.18 must compare UNEQUAL to the float.
    # If this ever passes, gst_rate has silently become a float.
    assert mdr.gst_rate != 0.18


# ---------------------------------------------------------------------------
# 6. Config objects are frozen
# ---------------------------------------------------------------------------


def test_app_config_is_frozen() -> None:
    config = load_config(CONFIG_DIR)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.config_hash = "tampered"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("attribute", "field", "value"),
    [
        ("mdr", "gst_rate", Decimal("0.99")),
        ("mdr", "version", "tampered"),
        ("calendar", "version", "tampered"),
        ("calendar", "timezone", "UTC"),
        ("thresholds", "version", "tampered"),
    ],
)
def test_nested_config_objects_are_frozen(attribute: str, field: str, value: object) -> None:
    config = load_config(CONFIG_DIR)
    target = getattr(config, attribute)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(target, field, value)


@pytest.mark.parametrize(
    ("group", "field"),
    [
        ("fuzzy_utr", "accept_score"),
        ("tolerance", "rounding_rupees"),
        ("confidence", "auto_confirm"),
        ("hypothesis", "max_per_exception"),
        ("safety_budgets", "max_residual_false_match_rate"),
        ("reporting", "assumed_review_minutes_per_exception"),
    ],
)
def test_threshold_groups_are_frozen(group: str, field: str) -> None:
    thresholds = load_thresholds(CONFIG_DIR / THRESHOLDS)
    target = getattr(thresholds, group)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(target, field, Decimal("0"))


def test_rate_mapping_is_not_writable() -> None:
    """Frozen is not enough if a nested mapping is still mutable."""
    mdr = load_mdr_rates(CONFIG_DIR / MDR)
    with pytest.raises(TypeError):
        mdr.profiles["profile_a"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        mdr.profiles["profile_a"]["card"] = Decimal("9")  # type: ignore[index]


# ---------------------------------------------------------------------------
# 7. Loading twice produces equal objects
# ---------------------------------------------------------------------------


def test_loading_twice_produces_equal_objects() -> None:
    first = load_config(CONFIG_DIR)
    second = load_config(CONFIG_DIR)
    assert first is not second
    assert first == second
    assert first.mdr == second.mdr
    assert first.calendar == second.calendar
    assert first.thresholds == second.thresholds


def test_config_hash_is_stable_across_loads() -> None:
    """Feeds ReconciliationResult.config_hash, so it must not drift (D6, D10)."""
    hashes = {load_config(CONFIG_DIR).config_hash for _ in range(5)}
    assert len(hashes) == 1, f"config_hash is unstable across loads: {sorted(hashes)}"


def test_config_hash_changes_when_a_threshold_changes(config_dir: Path) -> None:
    """The hash must actually identify the config, not just be constant."""
    baseline = load_config(CONFIG_DIR).config_hash
    _patch(config_dir, THRESHOLDS, 'accept_score: "0.85"', 'accept_score: "0.86"')
    assert load_config(config_dir).config_hash != baseline


def test_config_hash_ignores_comments_and_whitespace(config_dir: Path) -> None:
    """The hash covers semantic content, not YAML formatting."""
    baseline = load_config(CONFIG_DIR).config_hash
    path = config_dir / THRESHOLDS
    path.write_text(
        path.read_text(encoding="utf-8") + "\n# a trailing comment changes nothing\n",
        encoding="utf-8",
    )
    assert load_config(config_dir).config_hash == baseline

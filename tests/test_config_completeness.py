"""Every profile must price every method. A gap is an error, never a default.

Config that loads but is wrong is worse than config that fails to load. A
missing rate does not stay missing - it becomes a wrong fee at reconciliation
time, three modules downstream, presenting as an arithmetic variance the engine
will dutifully classify as a real exception. By then the loader is the last
place anyone looks.

The defect: `load_mdr_rates` validated each method it FOUND and never asked
which methods it should have found. Deleting `card` from profile_a produced a
config that parsed cleanly, typed cleanly, and was silently missing 25% of its
rate table.

The expected method set is the UNION across profiles, not a hard-coded list.
Hard-coding it here would put the payment-method taxonomy in the loader, so
adding a method would mean editing code as well as config - and the failure
mode of forgetting is exactly the silent gap this guards. The union means a
method configured anywhere becomes mandatory everywhere.
"""

from __future__ import annotations

import pathlib
import shutil
from typing import Any

import pytest
import yaml

from settlesense.config import ConfigError, load_mdr_rates

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
MDR_PATH = CONFIG_DIR / "mdr_rates.yaml"

# Restated by hand. Reading these from the file under test would make the
# parametrisation shrink in step with a config that lost entries.
PROFILE_NAMES = ("profile_a", "profile_b", "profile_c")
METHOD_NAMES = ("card", "upi", "netbanking", "wallet")


def _load_raw() -> dict[str, Any]:
    payload: dict[str, Any] = yaml.safe_load(MDR_PATH.read_text("utf-8"))
    return payload


def _write(tmp_path: pathlib.Path, raw: dict[str, Any]) -> pathlib.Path:
    target = tmp_path / "mdr_rates.yaml"
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return target


def _copy(raw: dict[str, Any]) -> dict[str, Any]:
    """A deep copy via the YAML round trip the tests write anyway."""
    duplicate: dict[str, Any] = yaml.safe_load(yaml.safe_dump(raw))
    return duplicate


def _without(raw: dict[str, Any], profile: str, method: str) -> dict[str, Any]:
    """A deep-enough copy with one (profile, method) rate removed."""
    mutated = _copy(raw)
    del mutated["profiles"][profile]["methods"][method]
    return mutated


# ===========================================================================
# 0. The fixture itself is honest
# ===========================================================================


def test_the_shipped_config_loads_and_is_complete() -> None:
    """Baseline. Every mutation below is measured against this loading cleanly."""
    config = load_mdr_rates(MDR_PATH)
    assert set(config.profiles) == set(PROFILE_NAMES)
    for profile in PROFILE_NAMES:
        assert set(config.profiles[profile]) == set(METHOD_NAMES), (
            f"{profile} prices {sorted(config.profiles[profile])}, expected {sorted(METHOD_NAMES)}"
        )


def test_a_faithful_roundtrip_still_loads(tmp_path: pathlib.Path) -> None:
    """Guards the guard: the rewrite must not itself be what breaks loading.

    Every deletion test below writes a re-serialised copy. If that roundtrip
    corrupted the file - dropping quotes and turning rates into YAML floats,
    say - the deletion tests would pass for the wrong reason entirely, since
    D12 makes a float raise too.
    """
    rewritten = _write(tmp_path, _load_raw())
    config = load_mdr_rates(rewritten)
    original = load_mdr_rates(MDR_PATH)
    assert config == original, "re-serialising the config changed its meaning"


# ===========================================================================
# 1. The named case from the brief
# ===========================================================================


@pytest.mark.config_refusal
def test_deleting_card_from_profile_a_raises_naming_both(tmp_path: pathlib.Path) -> None:
    """The message must name the profile AND the method, not just 'invalid config'.

    An error that says only "config invalid" costs the reader the same search
    the silent version did - it just moves it earlier. The whole value of failing
    at load time is that the message points at the line to fix.
    """
    broken = _write(tmp_path, _without(_load_raw(), "profile_a", "card"))

    with pytest.raises(ConfigError) as excinfo:
        load_mdr_rates(broken)

    message = str(excinfo.value)
    assert "profile_a" in message, f"error does not name the profile: {message}"
    assert "card" in message, f"error does not name the missing method: {message}"


# ===========================================================================
# 2 & 3. The full matrix - every profile x every method
# ===========================================================================


@pytest.mark.parametrize("profile", PROFILE_NAMES)
@pytest.mark.parametrize("method", METHOD_NAMES)
@pytest.mark.config_refusal
def test_any_missing_rate_raises(profile: str, method: str, tmp_path: pathlib.Path) -> None:
    """12 cases, generated. Hand-writing one would test one twelfth of the gap.

    The defect was per (profile, method): fixing `card` on `profile_a` alone
    would leave eleven identical holes. Parametrising is not tidiness here, it
    is the difference between testing the class of bug and testing one instance.
    """
    broken = _write(tmp_path, _without(_load_raw(), profile, method))

    with pytest.raises(ConfigError) as excinfo:
        load_mdr_rates(broken)

    message = str(excinfo.value)
    assert profile in message, f"({profile}, {method}) error omits the profile: {message}"
    assert method in message, f"({profile}, {method}) error omits the method: {message}"


@pytest.mark.parametrize("profile", PROFILE_NAMES)
@pytest.mark.config_refusal
def test_a_profile_missing_every_method_raises(profile: str, tmp_path: pathlib.Path) -> None:
    """The degenerate case: an empty method table is not "no methods priced".

    Worth pinning separately because an empty mapping is falsy, and a
    completeness check written as `if methods: compare(...)` would skip it.
    """
    raw = _copy(_load_raw())
    raw["profiles"][profile]["methods"] = {}
    broken = _write(tmp_path, raw)

    with pytest.raises(ConfigError) as excinfo:
        load_mdr_rates(broken)
    assert profile in str(excinfo.value)


# ===========================================================================
# 4. Present in one profile, absent in another - an ERROR, not a default
# ===========================================================================


@pytest.mark.config_refusal
def test_a_method_added_to_one_profile_becomes_mandatory_everywhere(
    tmp_path: pathlib.Path,
) -> None:
    """The union rule, stated as behaviour rather than as implementation.

    Adding a new method to one profile must not silently leave the other two
    unpriced. There is no sensible default: zero would be a free transaction and
    "copy another profile" would invent a commercial term nobody agreed.
    """
    raw = _copy(_load_raw())
    raw["profiles"]["profile_a"]["methods"]["emi"] = "0.0300"
    broken = _write(tmp_path, raw)

    with pytest.raises(ConfigError) as excinfo:
        load_mdr_rates(broken)

    message = str(excinfo.value)
    assert "emi" in message, f"the newly-added method is not named: {message}"
    assert "profile_b" in message or "profile_c" in message, (
        f"the error must name a profile that now lacks the method: {message}"
    )


def test_the_same_method_added_to_every_profile_loads(tmp_path: pathlib.Path) -> None:
    """The completing move works: it is a completeness rule, not a fixed list.

    Without this, the check above would also pass if the loader simply rejected
    any method outside a hard-coded set - which would make the taxonomy a code
    change, and is a different (worse) design.
    """
    raw = _copy(_load_raw())
    for profile in PROFILE_NAMES:
        raw["profiles"][profile]["methods"]["emi"] = "0.0300"
    extended = _write(tmp_path, raw)

    config = load_mdr_rates(extended)
    for profile in PROFILE_NAMES:
        assert "emi" in config.profiles[profile], f"{profile} lost the new method"


@pytest.mark.config_refusal
def test_a_missing_rate_is_never_defaulted_to_zero(tmp_path: pathlib.Path) -> None:
    """The specific silent behaviour: absent treated as free.

    `upi` is legitimately 0.0000 in every profile, so a loader that defaulted a
    missing rate to zero would be indistinguishable from a correct one on that
    method - and wrong by the full MDR on the other three. Deleting a NON-zero
    rate makes the difference observable.
    """
    broken = _write(tmp_path, _without(_load_raw(), "profile_b", "netbanking"))

    with pytest.raises(ConfigError):
        load_mdr_rates(broken)

    # And prove the rate really was non-zero, so a zero default would be wrong.
    original = load_mdr_rates(MDR_PATH)
    assert original.profiles["profile_b"]["netbanking"] > 0


def test_deleting_a_whole_profile_is_not_a_completeness_error(
    tmp_path: pathlib.Path,
) -> None:
    """Boundary: the union is over profiles PRESENT in the file.

    Removing a profile entirely is a different mistake with a different remedy,
    and cross-file consistency against the calendar is what catches it. This
    test exists so the completeness rule is not quietly widened into a
    profile-count assertion that would then reject a legitimate two-profile
    config.
    """
    raw = _copy(_load_raw())
    del raw["profiles"]["profile_c"]
    reduced = _write(tmp_path, raw)

    config = load_mdr_rates(reduced)
    assert set(config.profiles) == {"profile_a", "profile_b"}
    for profile in ("profile_a", "profile_b"):
        assert set(config.profiles[profile]) == set(METHOD_NAMES)


@pytest.mark.config_refusal
def test_an_empty_profiles_block_raises(tmp_path: pathlib.Path) -> None:
    """...but zero profiles is still an error: the union would be empty and the
    completeness check would pass over nothing at all."""
    raw = _copy(_load_raw())
    raw["profiles"] = {}
    with pytest.raises(ConfigError):
        load_mdr_rates(_write(tmp_path, raw))


# ===========================================================================
# The whole config directory still loads, so this rule did not break the others
# ===========================================================================


def test_the_full_config_directory_still_loads(tmp_path: pathlib.Path) -> None:
    from settlesense.config import load_config

    staged = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, staged)
    assert load_config(staged) == load_config(CONFIG_DIR)

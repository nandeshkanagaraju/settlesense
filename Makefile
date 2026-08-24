# SettleSense - build targets (SDD section 8).
#
# Targets whose implementation does not exist yet REFUSE and exit 1. They do
# not silently pass: a green target that ran nothing is worse than a red one,
# because it reads as evidence. Each refusal names the gate that implements it
# (SDD section 9). Implementing a module means deleting its `notimpl` line.
#
# `test` and `check` are real from M0 onward - they operate on the tree itself,
# not on a module that has yet to be written.

SHELL := /bin/bash

# Prefer the project virtualenv. A bare `python3` is whatever is on PATH, which
# here had neither ruff nor pytest installed - so `make check` failed at its
# first line and had never actually run a single check. A build target that
# cannot run is indistinguishable from one that passes, until you look.
VENV_PYTHON := $(wildcard .venv/bin/python)
PYTHON ?= $(if $(VENV_PYTHON),$(VENV_PYTHON),python3)
CONFIG_DIR ?= config

# $(call notimpl,<target>,<gate>,<module>)
define notimpl
	@echo "make $(1): not implemented."
	@echo "  Implemented at: $(2)  ->  $(3)"
	@echo "  See SDD section 9 for the gate map."
	@exit 1
endef

.PHONY: help gen gen-holdout eval-set test eval eval-ai ui check golden-accept bench config-check fault-report collection-baseline

help:
	@echo "SettleSense targets:"
	@echo "  gen           generate the dev dataset       (seed 42)"
	@echo "  gen-holdout   generate the held-out dataset  (seed 999, +withheld noise)"
	@echo "  eval-set      regenerate the AI evaluation set (seeds 1000-1019)"
	@echo "  test          pytest: no network, deterministic, under 60s"
	@echo "  check         ruff + mypy + determinism guard tests"
	@echo "  fault-report  guards proven able to fail, by category"
	@echo "  collection-baseline  re-record collected test counts (deliberately awkward)"
	@echo "  config-check  load every config file and print the config hash"
	@echo "  eval          held-out evaluation across all baselines"
	@echo "  eval-ai       real model run - the experiment, not a test"
	@echo "  bench         throughput scaling table"
	@echo "  ui            evidence queue"
	@echo "  golden-accept regenerate golden files (deliberately awkward)"

# --- data generation --------------------------------------------------------

gen:
	$(PYTHON) -m gen.generate --seed 42 --out data/ --days 20

gen-holdout:
	$(PYTHON) -m gen.generate --seed 999 --out data/holdout/ --days 20 --include-withheld

# The AI evaluation set: seeds 1000-1019, ALL TWENTY, declared in README before
# generation. ~146MB and gitignored - the datasets are defined by (frozen
# generator commit, seed) and regenerate byte-identically, so committing 2,880
# files would add size and no information. EVAL_SET_MANIFEST.json holds a
# content hash per seed; tests/test_eval_set.py verifies them when the data is
# present and checks the recorded invariants when it is not.
EVAL_SEEDS ?= $(shell seq 1000 1019)

eval-set:
	@for seed in $(EVAL_SEEDS); do \
	  echo "  seed $$seed"; \
	  $(PYTHON) -m gen.generate --seed $$seed --out data/eval/seed_$$seed/ --days 20 >/dev/null; \
	done
	@echo "generated $(words $(EVAL_SEEDS)) datasets into data/eval/"

# --- tests and static checks ------------------------------------------------

# Verbosity belongs to the CALLER. ARGS REPLACES the default flags rather than
# being appended to them, because pytest keeps the quiet reporter when -v
# follows -q:
#
#     pytest -q      -> no per-test names
#     pytest -q -v   -> no per-test names   <-- appending cannot work
#     pytest -v      -> per-test names
#
# -q therefore lives HERE and nowhere else. It is deliberately absent from
# pyproject `addopts`, which applies to every invocation and would silently
# override a flag typed on the command line.
#
#   make test                              quiet, per SDD section 8
#   make test ARGS="-v"                    per-test names
#   make test ARGS="-v tests/test_x.py"    one file, verbose
PYTEST_FLAGS ?= -q

test:
	$(PYTHON) -m pytest $(if $(strip $(ARGS)),$(ARGS),$(PYTEST_FLAGS))

# tests/ is type-checked too. It was excluded, and had accumulated six errors
# nobody could see - the same shape as `make check` running a python3 with no
# ruff installed. A check that is not running looks exactly like one that
# passes. The guards ARE the deliverable here; they get the same scrutiny as
# the code they guard.
check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy settlesense gen eval
	$(PYTHON) -m mypy --strict tests
	$(PYTHON) -m pytest -q -m determinism

# Loads all three config files through the strict loader and prints the hash
# recorded in every ReconciliationResult. Fails loudly on a YAML float (D12)
# or a date outside 2026 (D13).
config-check:
	@$(PYTHON) -c "from pathlib import Path; from settlesense.config import load_config; \
	c = load_config(Path('$(CONFIG_DIR)')); \
	print('calendar_version:', c.calendar.version); \
	print('config_hash:     ', c.config_hash); \
	print('profiles:        ', ', '.join(c.mdr.profile_names())); \
	print('holidays:        ', len(c.calendar.holiday_list()))"

# --- evaluation -------------------------------------------------------------

# Explicit contract - these exact paths are repeated in the README.
eval:
	$(call notimpl,eval,Gate 4 / M5,eval/run_eval.py)
	$(PYTHON) -m eval.run_eval \
	  --data data/holdout \
	  --truth data/holdout/truth_999.json \
	  --baselines all \
	  --out reports/eval

eval-ai:
	$(call notimpl,eval-ai,Gate 6 / M7,settlesense/ai/)

bench:
	$(call notimpl,bench,Gate 4 / M5a,eval/bench.py)
	$(PYTHON) -m eval.bench --sizes 500,5000,25000,100000 --out reports/bench.md

# --- interface --------------------------------------------------------------

ui:
	$(call notimpl,ui,Gate 7 / M8,settlesense/ui/app.py)
	$(PYTHON) -m streamlit run settlesense/ui/app.py

# --- goldens ----------------------------------------------------------------

# Golden files are IMMUTABLE by default. Regenerating them is the easiest way
# to make a real regression disappear, so it requires an explicit, awkward
# opt-in and is never reachable from test, check or eval.
golden-accept:
	@test "$$SETTLESENSE_ACCEPT_GOLDEN" = "1" || \
	  (echo "REFUSED. Regenerating goldens hides regressions."; \
	   echo "If a golden SHOULD change, state why in the commit message, then:"; \
	   echo "  SETTLESENSE_ACCEPT_GOLDEN=1 make golden-accept"; exit 1)
	$(PYTHON) -m pytest tests/golden --update-golden

# Every guard in this project is paired with a test that makes it FIRE. This
# counts those pairs by category. A check that has only ever passed may be
# inspecting an empty set or asserting something the code cannot violate; the
# difference is invisible in a green run, so it is counted separately.
fault-report:
	$(PYTHON) -m pytest -q -m "config_refusal or charter_guard or truth_injection or noise_accounting or hygiene or boundary_refusal"

# The count of tests that are COLLECTED, per file and in total. A drop means
# tests stopped being collected, which is invisible in a green run because
# everything that remains still passes.
#
# Deliberately NOT reachable from test or check, for the same reason
# golden-accept is not: a baseline that regenerates itself records whatever
# happened rather than what should happen, and running it is the easiest way to
# make a real regression disappear. Run it in the commit that changes the test
# set, and say so in the message.
collection-baseline:
	@test "$$SETTLESENSE_ACCEPT_BASELINE" = "1" || \
	  (echo "REFUSED. Rewriting the baseline hides tests that stopped being collected."; \
	   echo "If the test set SHOULD have changed, say why in the commit message, then:"; \
	   echo "  SETTLESENSE_ACCEPT_BASELINE=1 make collection-baseline"; exit 1)
	@$(PYTHON) -m pytest tests/ --collect-only -q 2>/dev/null | grep '::' | $(PYTHON) -c "$$BASELINE_SCRIPT"

define BASELINE_SCRIPT
import collections, json, sys
counts = collections.Counter(
    line.split("::", 1)[0] for line in sys.stdin.read().splitlines() if "::" in line
)
data = {
    "_comment": (
        "Collected-test counts, per file and in total. A DROP means tests stopped "
        "being COLLECTED, which is not the same as passing. "
        "tests/test_env_integrity.py compares against this. Regenerate DELIBERATELY "
        "with `SETTLESENSE_ACCEPT_BASELINE=1 make collection-baseline`, in the same "
        "commit that changes the test set - never automatically, and never to make "
        "a failure go away."
    ),
    "files": dict(sorted(counts.items())),
    "total": sum(counts.values()),
}
with open("tests/collection_baseline.json", "w") as handle:
    handle.write(json.dumps(data, indent=2) + "\n")
print(f"baseline updated: {data['total']} tests across {len(counts)} files")
endef
export BASELINE_SCRIPT

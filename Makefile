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
PYTHON ?= python3
CONFIG_DIR ?= config

# $(call notimpl,<target>,<gate>,<module>)
define notimpl
	@echo "make $(1): not implemented."
	@echo "  Implemented at: $(2)  ->  $(3)"
	@echo "  See SDD section 9 for the gate map."
	@exit 1
endef

.PHONY: help gen gen-holdout test eval eval-ai ui check golden-accept bench config-check

help:
	@echo "SettleSense targets:"
	@echo "  gen           generate the dev dataset       (seed 42)"
	@echo "  gen-holdout   generate the held-out dataset  (seed 999, +withheld noise)"
	@echo "  test          pytest: no network, deterministic, under 60s"
	@echo "  check         ruff + mypy + determinism guard tests"
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

# --- tests and static checks ------------------------------------------------

test:
	$(PYTHON) -m pytest -q

check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy settlesense gen eval
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

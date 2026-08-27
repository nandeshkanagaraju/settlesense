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

.PHONY: help gen gen-holdout eval-set test eval eval-ai eval-ai-loop record-fixtures demo-state ui ui-static check golden-accept bench config-check fault-report collection-baseline eval-holdout

help:
	@echo "SettleSense targets:"
	@echo "  gen           generate the dev dataset       (seed 42)"
	@echo "  gen-holdout   generate the held-out dataset  (seed 999, +withheld noise)"
	@echo "  eval-set      regenerate the AI evaluation set (seeds 1000-1019)"
	@echo "  test          pytest: no network, deterministic, under 120s"
	@echo "  check         ruff + mypy + determinism guard tests"
	@echo "  fault-report  guards proven able to fail, by category"
	@echo "  collection-baseline  re-record collected test counts (deliberately awkward)"
	@echo "  config-check  load every config file and print the config hash"
	@echo "  eval          evaluation on the DEV set (seed 42)"
	@echo "  eval-holdout  the HELD-OUT set (seed 999) - run ONCE, at the end"
	@echo "  eval-ai       real model run - the experiment, not a test"
	@echo "  eval-ai-loop  M7 verified hypothesis loop, oracle vs adversarial"
	@echo "  record-fixtures  record a 40-decision sample (SPENDS MONEY, needs a key)"
	@echo "  bench         throughput scaling table"
	@echo "  demo-state    build the state DB the queue reads (writes)"
	@echo "  ui            evidence queue, Streamlit (read-only)"
	@echo "  ui-static     evidence queue as one HTML file (read-only)"
	@echo "  golden-accept regenerate golden files (deliberately awkward)"

# --- data generation --------------------------------------------------------

gen:
	$(PYTHON) -m gen.generate --seed 42 --out data/dev/ --days 20

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

# `eval` RUNS THE DEV SET. This is the most important line in the Makefile.
#
# If the default target pointed at the holdout it would be run dozens of times
# during development and would stop being held out - not through any single bad
# decision, just through convenience. The holdout has its own target below,
# which refuses to be quiet about what it is.
eval:
	$(PYTHON) -m eval.run_eval \
	  --data data/dev \
	  --truth data/dev/truth_42.json \
	  --baselines all \
	  --out reports/eval

# THE HELD-OUT SET. Seed 999, plus the two withheld noise types. Run ONCE, at
# the end. Every run after the first is a run against data you have now seen.
eval-holdout:
	@echo "=============================================================="
	@echo "  This is the HELD-OUT set (seed 999)."
	@echo "  Record whatever it prints."
	@echo ""
	@echo "  Every run after the first is a run against data you have"
	@echo "  now seen. There is no way to un-see it and no way to tell"
	@echo "  from the output how many times this has been run."
	@echo "=============================================================="
	@echo ""
	$(PYTHON) -m eval.run_eval \
	  --data data/holdout \
	  --truth data/holdout/truth_999.json \
	  --baselines all \
	  --out reports/eval-holdout

# The 20-seed AI evaluation set (seeds 1000-1019), declared in README before
# generation. Runs the engine over all twenty and reports the residual surface
# M7 has to work with.
eval-ai:
	$(PYTHON) -m eval.run_eval_ai --eval-dir data/eval --out reports/eval-ai

# M7. The verified hypothesis loop, measured against the 20-seed evaluation set
# with THREE stand-in clients and no model at all:
#
#   oracle       always nominates the truth-correct row  -> the CEILING
#   adversarial  always nominates the wrong row          -> false confirms
#   silent       returns nothing schema-valid            -> must abstain
#
# The oracle's count is an upper bound no real model can exceed, so this
# answers "is it worth recording fixtures" before a rupee is spent.
eval-ai-loop:
	$(PYTHON) -m eval.run_ai --eval-dir data/eval --out reports/ai

# THE ONLY TARGET THAT SPENDS MONEY, and the only one that touches a network.
# Records a 40-decision stratified sample (20 the oracle confirms, 20 it
# rejects) against the pinned OpenAI snapshot. Needs OPENAI_API_KEY.
#
# --dry-run prints the exact sample and calls nothing, which is how to check the
# selection before paying for it.
record-fixtures:
	@test -n "$$OPENAI_API_KEY" || \
	  (echo "OPENAI_API_KEY is not set. Nothing else in this project needs it:"; \
	   echo "tests, eval and bench all replay from fixtures/llm/."; exit 1)
	$(PYTHON) -m eval.record_fixtures --eval-dir data/eval

# Throughput scaling (M5a). THE DEV SEED, never the holdout: a benchmark is
# re-run on every change, and a held-out set run dozens of times is no longer
# held out. Nothing about a throughput figure needs unseen data.
#
# Sizes are 500/5000/25000 rather than SDD 8's 500/5000/25000/100000. 100k is
# ATTEMPTED as a stretch when 25k finishes inside two minutes, and reports/bench.md
# states which branch was taken - a skipped 100k row is absent, never estimated
# from the smaller sizes.
BENCH_SIZES ?= 500,5000,25000

bench:
	$(PYTHON) -m eval.bench --sizes $(BENCH_SIZES) --out reports/bench.md

# --- interface --------------------------------------------------------------

# M8. The evidence queue. READ-ONLY over the state DB, no model calls.
#
# `demo-state` writes; `ui` and `ui-static` only read. Keeping the writer in a
# separate target is what makes "the UI is read-only" checkable rather than
# promised - the queue opens a database it did not create and refuses if there
# is none, because an empty queue and a missing DB look identical on screen.
demo-state:
	$(PYTHON) -m settlesense.ui.build_state --data data/dev --out reports/ui/state.db

ui: demo-state
	$(PYTHON) -m streamlit run settlesense/ui/app.py

# The static page is what gets screenshotted for the README and recorded for a
# video: legibility matters more than interactivity, and a file on disk is
# easier to capture than a server that has to be running.
ui-static: demo-state
	$(PYTHON) -m settlesense.ui.build_static --db reports/ui/state.db --out reports/ui/queue.html

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

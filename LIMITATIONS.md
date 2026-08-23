# Limitations

What this project does not establish, stated before anyone has to ask.

## Dataset

The data is synthetic, produced by an adversarial generator on a separate code
path that shares no normalization, identifier-parsing, date-handling or matching
utility with the engine. It is not merchant data and carries no claim to
reproduce a real settlement distribution.

### The generator was re-frozen once

The generator was re-frozen once, before any engine code existed, to correct six
dataset defects found by the M1 guard suite. No engine had been run against the
superseded dataset.

The original freeze (`m1f-generator-freeze`, commit `6ff7172`) is left in place;
the current one is `m1f-generator-freeze-2`, and `GENERATOR_MANIFEST.json`
records both through its `supersedes` field. The six corrections were: a
hardcoded `true_variance_amount` that made every money-weighted metric
unmeasurable; `UNEXPLAINED` and orphan credits absent from the dev dataset;
`ROUNDING_DIFFERENCE` producible by no injector, leaving SDD 4.2 pass P9
unreachable; a docstring describing a defect rather than the code; and a
predicate that silently skipped all-digit UTRs.

The freeze exists to stop the generator being tuned against engine results. That
guarantee is about ordering, not about dates: what matters is that the generator
was frozen **before the engine**, not that it was frozen early. At the time of
the re-freeze no matching code existed and nothing had been measured.

**From M3 onward there is no further re-freeze.** A generator defect found after
engine development begins is recorded here and lived with, because by then a
correction cannot be distinguished from tuning. `tests/test_generator_freeze.py`
enforces this: once `settlesense/matching/` stops being a stub, a third freeze
generation fails the build.

## Calendar

The working-day calendar (`config/calendar_v1.yaml`) uses a **synthetic holiday
set for the 2026 simulated timeline**. It is not the RBI bank-holiday schedule
and does not attempt to be. The evaluation measures whether the engine handles a
holiday correctly, not whether it knows India's actual holiday list. Inventing a
real-looking holiday list would be a false claim of accuracy. Swapping in real
dates is a config change requiring no code change.

## Baselines

## Thresholds

The safety budgets in `config/thresholds.yaml` — residual false-match rate,
gross-exposure false-match value, cost per 1,000 rows — are **project safety
thresholds for a synthetic evaluation**. They are not a claim about acceptable
production loss. Real deployment would require risk, compliance and
business-owner sign-off on every one of them.

## What this does not establish

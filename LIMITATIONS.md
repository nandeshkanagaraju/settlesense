# Limitations

What this project does not establish, stated before anyone has to ask.

## Dataset

The data is synthetic, produced by an adversarial generator on a separate code
path that shares no normalization, identifier-parsing, date-handling or matching
utility with the engine. It is not merchant data and carries no claim to
reproduce a real settlement distribution.

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

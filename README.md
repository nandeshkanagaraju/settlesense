## Results

TO BE MEASURED — see `make eval`

---

## Reproducing

```
make gen           # dev dataset,      seed 42
make gen-holdout   # held-out dataset, seed 999, includes withheld noise types
make eval          # all baselines against the held-out set
make bench         # throughput scaling table -> reports/bench.md
make test          # no network, deterministic, under 60 seconds
make check         # ruff + mypy + determinism guard tests
```

Targets that have no implementation yet refuse and exit non-zero. They never
pass silently.

## Modules built

Recorded here at submission: which Level 2 modules were built and which were
cut, without apology. Cutting a conditional module is a planned outcome, not
an incomplete submission.

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md).

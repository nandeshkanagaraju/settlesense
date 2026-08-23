# SettleSense — Claude Code Build Prompts (Module by Module)

**How to use this file**

1. Paste `SettleSense_PDD.md` and `SettleSense_SDD.md` into the repo root and tell Claude Code to read both before anything else.
2. Work one module at a time. Paste the **Build prompt**, let it finish, then paste the matching **Test prompt**.
3. After every module, run the **Verify** commands yourself. Do not move to the next module until they pass.
4. Commit after each completed module or meaningful verified milestone, with a real message. Do not create artificial commits solely to raise the count.

**Session opener — paste this first, once:**

```
Read SettleSense_PDD.md and SettleSense_SDD.md completely before writing any code.

Section 1 of the SDD is the Determinism Charter. Rules D1–D13 are hard constraints,
not suggestions. If any instruction I give you later appears to violate D1–D13, stop
and tell me instead of complying. In particular:
  D11 — every metric divides by ReconciliationCase (Population A). Batch-to-bank
        link metrics are Population B: separate table, separate denominator, never merged.
  D12 — scoring weights, thresholds and ratios are Decimal. The config loader RAISES
        on a YAML float.
  D13 — every date in this repo is in 2026. A 2025 date anywhere is a defect.

Standing rules for this whole project:
- Money is decimal.Decimal quantized to Decimal("0.01") with ROUND_HALF_UP. Never float.
- No datetime.now(), date.today(), or time.time() anywhere in settlesense/. Inject as_of.
- No module-level random. Pass a seeded random.Random explicitly.
- Sort every output list by an explicit key. Never rely on set or dict iteration order.
- gen/ must never import from settlesense/, and settlesense/ must never import from gen/.
- ReconciliationCase (SDD 3.1) is THE unit of measurement. Every rate divides by it.
- Every generated ID is a deterministic hash of a canonical tuple. Never uuid4().
- ALL dates in this project are in 2026. A 2025 date anywhere is a defect.
- Every metric is computed over ReconciliationCase (SDD 3.1). Batch-to-bank link
  metrics are a SEPARATE population, reported in a separate table, never merged.
- Truth is a typed edge set with per-type cardinality (SDD 3.2). Do NOT write a
  blanket "no row can partner twice" invariant — many settlement rows share one batch.
- ReconciliationResult contains NO wall-clock data. Timing is a separate RunTelemetry
  return value. If you find yourself stripping timings before a comparison, stop:
  the field is in the wrong object.
- Scoring weights and thresholds are Decimal. The config loader raises on a float.
- Type hints on every public function. Python 3.11+.

Confirm you have read both documents and restate the Determinism Charter in your own
words before we start Module 0.
```

---

## M0 — Skeleton, config, Makefile *(Gate 0, 22 Aug)*

### Build prompt

```
Build Module 0: the project skeleton.

Create the directory structure exactly as specified in SDD section 2. Empty __init__.py
files where needed.

Create pyproject.toml with: python >=3.11, and dependencies pytest, pytest-cov,
hypothesis, pyyaml, python-Levenshtein, streamlit, lxml, anthropic. Dev extras: ruff, mypy.

Create the three config files from SDD section 6 with real starting values:

config/mdr_rates.yaml — three merchant profiles (profile_a, profile_b, profile_c) with
per-method rates. Profile A: card 2.00%, upi 0.00%, netbanking 1.75%, wallet 2.10%.
Profile B: card 2.35%, upi 0.00%, netbanking 1.90%, wallet 2.00%. Profile C: card 1.80%,
upi 0.00%, netbanking 1.60%, wallet 1.95%. GST on fee 18% for all.

config/calendar_v1.yaml — version: calendar_v1, Mon-Fri working days, a list of 8
SYNTHETIC holidays (NOT the real RBI schedule; see LIMITATIONS.md) inside the
window 2026-09-01 to 2026-11-30 (ALL dates in this project are 2026), and settlement
cycles: profile_a T+2, profile_b T+1, profile_c T+3.

config/thresholds.yaml — every threshold in SDD section 6, including confidence weights.

Create settlesense/config.py: load each YAML into a frozen dataclass with typed fields.
Raise a clear error on a missing or malformed key. No defaults hidden in code — if a key
is absent, fail loudly.

Create the Makefile with the targets in SDD section 8. Targets that have no
implementation yet should echo "not implemented" and exit 1, not silently pass.

Create README.md as a stub whose FIRST section is "## Results" containing the literal
text "TO BE MEASURED — see make eval". Do not write a product introduction above it.

Create LIMITATIONS.md as a stub with headings: Dataset, Baselines, Thresholds,
What this does not establish.
```

### Test prompt

```
Write tests/test_m0_config.py and tests/test_determinism_guard.py.

test_m0_config.py:
1. Each of the three YAML configs loads into its frozen dataclass without error.
2. Loading a config with a deleted required key raises a clear error naming that key.
3. Loading a config with a rate as a string "2.00%" instead of a number raises.
4. All rate values load as Decimal, not float. Assert isinstance(rate, Decimal).
5. GST rate is exactly Decimal("0.18") for all three profiles.
6. Config objects are frozen — mutating a field raises FrozenInstanceError.
7. Loading the same config twice produces equal objects.

test_determinism_guard.py — these guard the charter and must exist from day one:
8. test_no_float_money: walk every dataclass in settlesense.types, assert no field
   annotated float. Fail with the offending class and field name.
9. test_no_wall_clock: AST-scan every .py under settlesense/, assert no call to
   datetime.now, date.today, time.time, or datetime.utcnow.
10. test_no_uuid4: AST-scan settlesense/, assert uuid4 is never called.
11. test_no_module_level_random: AST-scan gen/, assert the random module's top-level
    functions (random, randint, choice, shuffle) are never called directly.
12. test_gen_does_not_import_settlesense: AST-scan gen/, assert no import of settlesense.
13. test_settlesense_does_not_import_gen: the reverse.

Tests 8-13 must fail loudly with the exact file and line number of the violation.
Run them and show me the output.
```

**Verify:** `make check` passes. `pytest tests/test_determinism_guard.py -v` — all six guards pass. Commit: `M0: skeleton, typed config, determinism guards`.

---

## M1 — Adversarial generator *(Gate 1, 23 Aug)*

> Build clean chains and verify ground truth **before** touching noise. A generator that mislabels its own truth silently corrupts every metric you will report.

### Build prompt — part A, clean chains

```
Build Module 1 part A: gen/ clean lifecycle generation only. No noise yet.

gen/profiles.py — three merchant profiles matching config/mdr_rates.yaml, but define
the values INDEPENDENTLY inside gen/. Do not import from settlesense.config. The
generator must be a separate code path (SDD section 2). Duplication here is deliberate.

gen/lifecycle.py — generate_clean_chain(rng, profile, day) produces one complete,
internally consistent chain:
  LedgerRow -> PaymentRow -> (optional RefundRow)
            -> SettlementLine (payment | refund) -> SettlementBatch -> BankRow

Rules:
- All money Decimal, 2dp, ROUND_HALF_UP.
- fee = round(gross * rate[method], 2); tax = round(fee * Decimal("0.18"), 2)
- net = gross - fee - tax - sum(refunds)
- Batch net_total = SIGNED sum of member SettlementLine.net across both line types,
  computed by summation not
  by re-deriving from gross.
- Bank credit amount == batch net_total == sum of SIGNED nets across EVERY line in
  the batch (payments minus refunds), exactly;
  bank value_date = batch settled_event_date
  plus the profile's T+N, per calendar_v1.yaml.
- Narration format: "NEFT {UTR} {MERCHANT_NAME} SETTLEMENT"
- UPI has 0% MDR, so those chains must balance with fee == 0.

gen/generate.py — CLI: --seed, --out, --days (default 3), --records (default 5000),
--include-withheld (default False). Distribute records across the three profiles and
across the requested days. Write per-day CSVs into --out:
  day{N}_ledger.csv, day{N}_payments.csv, day{N}_refunds.csv,
  day{N}_settlements.csv, day{N}_batches.csv, day{N}_bank.csv

  SIX files. Chargebacks/disputes are OUT OF v1 SCOPE — do not generate a disputes
  file, a DISPUTE_DEBIT line type, a PAYMENT_TO_DISPUTE edge, or a CHARGEBACK_DEBIT
  category. If you find a reference to any of them anywhere, tell me; it is stale.

Everything derives from a single random.Random(seed) threaded through explicitly.
```

### Build prompt — part B, ground truth and self-check

```
Build Module 1 part B: gen/truth.py.

Read SDD 3.1 and 3.2 first. Truth is a TYPED EDGE SET, not a flat partner map.

Write truth_<seed>.json containing:
- edges: a list of TruthEdge {edge_type, src_id, dst_id} using the six EdgeType values
- cases: one entry per ReconciliationCase (one per captured payment) with its true
  variance category (closed taxonomy, PDD section 6), true variance amount, whether it
  is resolvable in principle, and which noise type if any was applied
- calendar_version, seed
- generator_commit: null   <-- literally null. The hash is unknowable pre-freeze and
  truth files are NEVER rewritten after the freeze. The real hash is published
  separately in GENERATOR_MANIFEST.json at M1F.

Do NOT write a "true partner" field that assumes one-to-one. Cardinality is per edge type.

CRITICAL — write run_self_check(dataset, truth) and call it before writing truth to disk.

Cardinality assertions — check EACH EDGE TYPE against its OWN declared cardinality
from the SDD 3.2 table. Do NOT write a blanket "no row partners twice" rule; that rule
is WRONG because many settlement rows legitimately share one batch.
1. ORDER_TO_PAYMENT is 1:1 in both directions
2. PAYMENT_TO_SETTLEMENT is 1:N; N>=2 marks the split-settlement case
3. SETTLEMENT_TO_BATCH is N:1 — many lines per batch is NORMAL, assert only that every
   settlement line has exactly one batch
4. BATCH_TO_BANK is 1:1 — a batch with two bank credits IS a generator bug, fail on it
5. PAYMENT_TO_REFUND is 1:N, no upper bound (partial and multiple refunds are legal)

Balance assertions:
6. Every clean chain balances to the cent: gross - fee - tax - refunds == net, exactly
7. Every SettlementBatch.net_total equals the SIGNED sum of its member SettlementLine.net
   across ALL line types. Test on a batch containing a payment, a refund and a
   refund line together — the regression test for signed-line arithmetic.
7a. Every bank credit equals its batch net_total exactly, including batches that
    contain a refund line.
7b. Per-type field invariants hold for every line (SDD 3.3 table). Assert both shapes.
7c. PAYMENT_TO_SETTLEMENT edges point ONLY at payment lines. Assert zero such edges
    reference a refund line — that ambiguity is the whole reason
    the entity is called a line and not a row.
8. Every truth entry references row IDs that exist in the dataset
9. Every fee equals the profile rate applied to gross, recomputed independently here
10. Global conservation, using the ONE canonical formula (SDD 3.1b):
      expected_net = gross - fee - tax - refunds
    Assert BOTH identities, exact Decimal equality:
      case-side : sum(gross) == sum(expected_net) + sum(fee) + sum(tax) + sum(refunds)
      batch-side: sum(bank credits) == sum(batch net_totals)
                                    == sum(SIGNED settlement line nets)

Date assertions:
11. Every event_date, file_content_date and derived DATE value is in 2026
12. arrival_day is a positive INTEGER day index (1, 2, 3...) — it is NOT a date and is
    NOT subject to the 2026 rule. Assert its type is int, never date or datetime.
    No wall-clock timestamp is written anywhere in the generator output.
13. Every settlement date respects the profile's T+N against calendar_v1.yaml

If any assertion fails, raise and write nothing. A truth file must never be written from
a dataset that does not balance.


Then run: python -m gen.generate --seed 42 --out data/ and show me the self-check output
and a summary table of what was generated.
```

### Build prompt — part C, noise

```
Build Module 1 part C: gen/noise.py.

Implement each noise injector from SDD section 5 as a separate pure function taking
(rows, rng, rate) and returning modified rows plus truth annotations.

Tuned-against noise:
- truncate_utr: keep a random 6-10 char prefix in the narration
- drop_utr: remove the UTR from the narration entirely
- merchant_name_variants: "ACME RETAIL" -> "ACME RTL", "ACME RETAIL PVT", "ACMERETAIL"
- mixed_amount_formats: emit "1,234.00" / "1234.0" / "1234" / " 1234.00 " in CSV
- duplicate_ledger_rows: exact duplicate AND a legitimate repeat purchase by the same
  customer for the same amount 3+ hours later. These must be distinguishable by the
  truth file but NOT trivially by amount alone.
- partial_captures: captured < authorized
- delayed_settlement: the settlement row appears on day N+1 or N+2, not day N
- out_of_order_arrival: emit files with a day label later than their content date
- unexplainable: a bank credit with no corresponding settlement anywhere, and a
  settlement with no bank credit that never arrives

WITHHELD (generate but gate behind --include-withheld):
- garbled_narration: transpose 2 adjacent characters in the UTR
- split_settlement: one payment's net split across two batches

Add a NOISE_REGISTRY dict mapping name -> (function, is_withheld). Log applied noise
counts to stderr.

Re-run the self-check after noise injection. It must still pass for every clean chain
and must correctly account for every injected error.
```

### Test prompt

```
Write tests/test_m1_generator.py.

Reproducibility:
1. Two runs with --seed 42 produce byte-identical CSV files. Compare file hashes.
2. --seed 42 and --seed 999 produce different data.
3. Regenerating with the same seed after deleting the output reproduces it exactly.

Balance invariants (run on seed 42 output):
4. Every clean chain: gross - fee - tax - refunds == net, exact Decimal equality.
5. Every batch net_total == sum of member nets.
6. Global money conservation matches self-check assertion 10 exactly — BOTH the
   case-side and batch-side identities, using expected_net = gross - fee - tax - refunds.
7. Every fee recomputed independently from the profile rate matches the generated fee.

Ground truth integrity:
8. Every truth entry's referenced row IDs exist in the dataset.
9. Validate typed edge cardinality via check_cardinality(edges). Assert:
   - every settlement line has exactly one SETTLEMENT_TO_BATCH edge;
   - a batch MAY have many settlement rows — assert a multi-row batch PASSES;
   - every batch has at most one BATCH_TO_BANK edge;
   - ORDER_TO_PAYMENT is 1:1 in both directions;
   - PAYMENT_TO_SETTLEMENT with N>=2 is legal and marks a split settlement;
   - duplicate identical edges are rejected.
   Do NOT assert that no row partners twice — that rule is wrong here.
10. Every injected noise instance has a corresponding truth annotation.
10c. ID namespaces are disjoint: set(settlement_id) & set(batch_id) == empty set.
     Assert every SettlementLine.batch_id exists in the batches file.
11. run_self_check raises when given a deliberately corrupted dataset — build one by
    mutating a single net amount by ₹0.01 and assert it raises.

Noise behaviour:
12. truncate_utr leaves a prefix that IS a genuine prefix of the true UTR.
13. drop_utr leaves no 10+ char alphanumeric token in the narration.
14. Duplicate rows and legitimate repeat purchases both exist, and their truth
    categories differ.
15. mixed_amount_formats produces at least 3 distinct string formats in the CSV.
16. Withheld noise types are ABSENT without --include-withheld and PRESENT with it.
17. Every noise type in NOISE_REGISTRY appears at least once in a 5000-record run.

Determinism:
18. Money values in output CSVs parse to Decimal with exactly 2 decimal places.
19. No float appears anywhere in the generator's money paths (AST scan of gen/).
```

**Verify:** `make gen && make gen-holdout`, self-check passes, `pytest tests/test_m1_generator.py -v` green. Then **freeze**.

---

## M1F — Freeze *(Gate 2, 24 Aug)*

Do this manually, not via Claude Code:

```bash
git add gen/ tests/test_m1_generator.py
git commit -m "M1: adversarial generator, frozen before engine development"
git rev-parse HEAD          # copy this hash
```

Paste the hash into `README.md` and `LIMITATIONS.md`:

After the freeze commit, run `git rev-parse HEAD` and write the ACTUAL output into
GENERATOR_MANIFEST.json. README and LIMITATIONS.md read the hash from that manifest.
Do NOT type a symbolic placeholder — no `GENERATOR_COMMIT`, no `<hash>`, no `TODO`.
`test_no_literal_placeholders` scans for all three and fails on a hit.

Freeze sequence — follow exactly, do not improvise:
1. Truth files already carry `generator_commit: null`. Leave them alone.
2. Commit `gen/`. This commit is the freeze point.
3. Write `GENERATOR_MANIFEST.json` at repo root:
   `{"generator_commit": "<real sha>", "frozen_at_arrival_day": 0, "seeds": {"dev": 42, "holdout": 999}, "calendar_version": "calendar_v1"}`
4. Substitute the real sha into README and LIMITATIONS.md.
5. NEVER regenerate or rewrite a truth file after step 2.
6. Eval output reports BOTH the truth-file seed and the manifest's generator_commit.

**Do not reopen any file under `gen/` after this point.** Your git history is the evidence that your unknown-unknowns test is real.

---

## M2 — Normalization *(Gate 3, 27 Aug)*

### Build prompt

```
Build Module 2: settlesense/normalize.py and settlesense/types.py.

types.py: every dataclass from SDD section 3, all frozen, all money as Decimal.
Add money(value) -> Decimal that quantizes to Decimal("0.01") ROUND_HALF_UP and
raises TypeError on float input. Every money field must go through it.

normalize.py — pure functions only, no I/O, no clock:
- normalize_utr(raw) -> str: uppercase, strip non-alphanumeric, collapse whitespace
- extract_utr_candidates(narration) -> tuple[str, ...]: return ALL plausible UTR tokens
  found, ranked longest-first, deterministically ordered
- parse_amount(raw) -> Decimal: handle "1,234.00", "1234.0", "1234", " 1234.00 ",
  "₹1,234.00", "(1234.00)" as negative. Raise ValueError with the offending input on
  anything unparseable. Never silently return 0.
- parse_date(raw, profile) -> date: allow-list of formats. DD/MM vs MM/DD resolved by
  profile config, never guessed. Ambiguous input with no profile rule raises.
- normalize_narration(raw) -> str: uppercase, collapse whitespace, strip punctuation
- normalize_merchant_name(raw) -> str: uppercase, remove PVT/LTD/PRIVATE/LIMITED,
  strip spaces

Add load_dataset(path, day, config) that reads the day's CSVs and returns typed,
normalized records. All row lists sorted by their primary ID.
```

### Test prompt

```
Write tests/test_m2_normalize.py. Every function gets hostile inputs, not just happy paths.

parse_amount — assert exact Decimal equality:
1. "1,234.00" -> Decimal("1234.00")
2. "1234.0" -> Decimal("1234.00")
3. "1234" -> Decimal("1234.00")
4. " ₹1,234.00 " -> Decimal("1234.00")
5. "(1234.00)" -> Decimal("-1234.00")
6. "" raises ValueError; "abc" raises ValueError; None raises
7. "1,23,456.00" (Indian lakh grouping) -> Decimal("123456.00")
8. parse_amount never returns a float — assert isinstance(result, Decimal)
9. "0.005" rounds to Decimal("0.01") under ROUND_HALF_UP, not banker's rounding
10. money(1.1) raises TypeError because 1.1 is a float

normalize_utr:
11. "utr-1234 5678" -> "UTR12345678"
12. "" -> ""
13. Idempotent: normalize_utr(normalize_utr(x)) == normalize_utr(x) for 20 inputs

extract_utr_candidates:
14. Narration with one UTR returns exactly that one
15. Narration with two 12-char tokens returns both, longest first
16. Narration with none returns empty tuple
17. Ordering is deterministic across 100 runs on the same input

parse_date:
18. "01/02/2026" with profile DD/MM -> date(2026,2,1)
19. Same string with profile MM/DD -> date(2026,1,2)
20. Ambiguous with no profile rule raises
21. "2026-13-01" raises

Property-based (hypothesis):
22. For any generated amount string in a valid format, parse_amount round-trips
23. normalize_merchant_name is idempotent

Integration:
24. load_dataset on day1 of seed-42 data returns the expected row counts per source
25. load_dataset twice returns equal objects (D6)
```

**Verify:** `pytest tests/test_m2_normalize.py -v`. Commit.

---

## M3 — Deterministic matching engine *(Gate 3, 27 Aug)*

### Build prompt

```
Build Module 3: settlesense/matching/ except fuzzy_utr.py.

Implement passes P1-P7 and P9 from SDD section 4.2 as separate modules. Each pass is a
pure function: (unmatched_pool, dataset, config, as_of) -> (matches, remaining_pool).

arithmetic.py:
- compute_fee(gross, method, profile, config) -> Decimal
- compute_gst(fee, config) -> Decimal
- explain_variance(expected, actual, components) -> VarianceBreakdown, attributing
  every rupee to a taxonomy category or leaving it in an explicit unexplained bucket
- The sum of attributed components plus unexplained must equal the total variance
  EXACTLY. Assert this inside the function.

timing.py:
- WorkingDayCalendar from config: add_working_days(d, n), is_working_day(d)
- settlement_due_date(captured_at, profile) using T+N per profile
- is_timing_explained(expected_date, actual_date, tolerance_days)

duplicates.py:
- Exact duplicate: same (order_id, gross, order_date)
- NOT a duplicate: same (customer_id, gross) but different order_id and >3h apart
- Return a DuplicateVerdict with the rule that fired, never a bare bool

engine.py — orchestrator:
- run(dataset, config, as_of) -> ReconciliationResult
- Passes execute in strict P1..P9 order; matched rows leave the pool
- ReconciliationResult holds: matches (sorted by match_id), residuals (sorted by
  exception_id), per-pass counts, Population A value totals, Population B value totals.
  Return CaseOutcome objects plus a SEPARATE batch-link result. Do NOT return generic
  row-level `matches` and `residuals` lists — that blurs the two populations and is
  exactly what D11 forbids.
- exception_id = sha256 of canonical tuple, first 16 hex chars (D10)
- Every list sorted explicitly before return (D4)

The engine takes as_of as a parameter. No clock access anywhere (D2).
```

### Test prompt

```
Write tests/test_m3_matching.py and tests/test_m3_engine.py.

Arithmetic — table-driven, exact Decimal equality:
1. ₹1000 card @ 2% -> fee ₹20.00, gst ₹3.60, net ₹976.40
2. ₹1000 upi @ 0% -> fee ₹0.00, gst ₹0.00, net ₹1000.00
3. ₹999.99 card @ 2.35% -> fee ₹23.50, gst ₹4.23 (verify ROUND_HALF_UP at the boundary)
4. ₹0.01 minimum -> no crash, no negative net
5. explain_variance: components + unexplained == total variance, exactly, for 50 cases
6. explain_variance with a deliberately unaccountable ₹7.13 puts exactly ₹7.13 in
   unexplained and does not silently absorb it into rounding

Timing:
7. T+2 from a Friday lands on Tuesday when Monday is a holiday
8. add_working_days(d, 0) == d when d is a working day
9. add_working_days on a Saturday rolls forward before counting
10. Each of the three profiles produces its correct T+N

Duplicates:
11. Exact duplicate rows are flagged, verdict rule == "exact_order_match"
12. Same customer, same amount, 4h apart, different order_id -> NOT duplicate
13. Same customer, same amount, 10 minutes apart, different order_id -> flagged for review,
    not auto-classified either way
14. Verdict always names the rule that fired

Engine:
15. Passes run in order — inject a row matchable by both P1 and P3, assert P1 claims it
16. A matched row never appears in the residual set
Population conservation (D11) — count and value must NEVER be conserved over raw
input rows. Source tables have different grains and one split-settlement payment
produces one case with several settlement rows, so a row-level identity is false
on valid data. Conserve over ReconciliationCase only:
17. resolved_case_count + residual_case_count == total ReconciliationCase count,
    for seed 42 day 1
18. Population A gross-exposure value: sum(expected_gross of resolved cases)
    + sum(expected_gross of residual cases) == sum(expected_gross of all cases),
    exact Decimal equality
19. Population B batch-net value is conserved SEPARATELY using batch.net_total: linked value
    + unlinked batch value == total batch net_total. Assert explicitly that
    Population A and Population B totals are NOT added together anywhere, and that
    the two denominators differ on this fixture
20. A split-settlement payment (>=2 PAYMENT lines) yields exactly ONE case with
    len(payment_line_ids) >= 2 — never two cases. A case with one payment line plus a
    refund line is NOT a split settlement: assert len(payment_line_ids) == 1 there. This is the regression test
    for the old row-level identity.
21. D6: run(dataset) twice returns byte-identical serialized ReconciliationResult
22. D10: case_ids and exception_ids are identical across two runs and across restarts
23. D4: shuffle the input row order, assert the sorted output is unchanged
24. as_of is honoured — the same dataset with two different as_of dates produces
    different timing classifications
25. Engine never raises on the full seed-42 dataset
26. Engine on an empty dataset returns an empty result, not a crash

Accuracy against ground truth (the real test):
27. Load truth_42.json. For every case the engine resolved, assert the outcome agrees
    with truth. Report deterministic precision and residual case count.
28. Assert zero false matches on clean chains — this must be exactly 0.
```

**Verify:** test 26 must be exactly zero false matches on clean chains. If not, stop and fix before proceeding — everything downstream depends on it.

---

## M4 — Fuzzy UTR *(Gate 3, 27 Aug)*

### Build prompt

```
Build Module 4: settlesense/matching/fuzzy_utr.py, then wire it in as pass P8
(after P7, before P9).

score_candidate(narration_utr, batch_utr, bank_amount, batch_net, date_gap) -> Decimal:
  0.5 * prefix_ratio + 0.3 * (1 - normalized_levenshtein) + 0.2 * amount_exact
  Hard gate: return Decimal("0") if date_gap exceeds the profile window.
  prefix_ratio = length of common prefix / length of true UTR.

resolve(bank_row, candidate_batches, config) -> FuzzyVerdict:
- Score all candidates, sort by (-score, batch_id) for a TOTAL ORDER (D5).
  Candidates are BATCHES, keyed by batch_id — never settlement_id.
- ACCEPT only if best >= 0.85 AND (best - runner_up) >= 0.15
- Otherwise return AMBIGUOUS with ALL candidates and their scores attached
- If two candidates tie exactly after the full tie-break chain, ABSTAIN. Never pick.

Ambiguous verdicts become residual exceptions with category UTR_TRUNCATED_MAPPING or
UTR_MISSING_MAPPING, carrying the full candidate list forward for the AI layer.

Use Decimal for scores, not float. Thresholds come from config, never hardcoded.
```

### Test prompt

```
Write tests/test_m4_fuzzy.py.

Scoring:
1. Identical UTR + exact amount + same date -> score 1.00
2. 6-char prefix of a 12-char UTR + exact amount -> score between 0.6 and 0.9
3. Completely different UTR -> score below 0.3
4. Date gap beyond window -> score exactly 0, even with a perfect UTR match
5. Scores are Decimal, never float
6. score_candidate is symmetric in its date handling and never raises on empty strings

Acceptance:
7. Clear winner (0.95 vs 0.40) -> ACCEPT
8. Two close candidates (0.90 vs 0.88) -> AMBIGUOUS, both attached
9. Best below 0.85 -> AMBIGUOUS even with no runner-up
10. Exact tie between two candidates -> ABSTAIN, and the verdict says so explicitly
11. Empty candidate list -> AMBIGUOUS with zero candidates, never a crash

Determinism (D5):
12. Shuffle candidate order 100 times; the verdict is identical every time
13. Two candidates with equal scores but different IDs sort by ID, deterministically

Against ground truth:
14. On truncate_utr cases in seed 42, report accept rate and accuracy of accepted matches
15. ZERO accepted matches may be wrong. Assert false-accept count == 0.
16. Report how many truncate_utr cases fuzzy matching resolved vs left ambiguous.
    Print this number — it directly determines how much surface the AI layer has left.

Regression guard:
17. After adding P8, rerun test_m3_engine.py assertion 26: still zero false matches.
```

> **Read test 16's output carefully.** If fuzzy matching resolves most truncated-UTR cases, your interpretive surface has shrunk and that is a finding to report, not a bug to work around.

---

## M5 — Eval harness and baselines *(Gate 4, 29 Aug)*

### Build prompt

```
Build Module 5: eval/ — metrics, three baselines, and the runner.

eval/metrics.py — implement every metric in PDD section 8.4. Each is a pure function
over (results, truth). Money-weighted metrics use Decimal throughout. Include
analyst_minutes_saved(results, minutes_per_review) and require the caller to pass the
assumption explicitly — it must appear in the output labelled "derived estimate,
assumes N min/review".

eval/baselines/naive.py — amount + date-window matching only. No identifiers.
eval/baselines/deterministic_only.py — a thin wrapper over settlesense.matching.engine.
eval/baselines/llm_only.py — STRONG baseline, built in good faith:
  - same normalized records from M2
  - candidate retrieval: for each bank row, retrieve the top 20 plausible batches
  - sensible chunking, structured JSON output, a carefully written prompt
  - This must NOT be a strawman. Write the prompt as if this baseline were the product.
  - Add a comment block explaining what you did to make it strong.

eval/run_eval.py:
  --data, --baselines all|naive|det|llm|settlesense, --truth, --out
Produces results.json and a markdown table. The markdown must include the residual-set
sentence from PDD section 8.3 with real numbers substituted.

Wire `make eval` to run against data/holdout with seed 999.
```

### Test prompt

```
Write tests/test_m5_eval.py. All LLM baselines use the replay client — no network.

Denominator discipline (SDD 3.1) — check these FIRST:
0a. Every Population A metric divides by len(cases). Assert the denominator equals the
    ReconciliationCase count, not row counts, not settlement rows, not bank rows.
0b. Batch-to-bank link metrics appear under a SEPARATE key in results.json and have
    their own denominator (batch count). Assert the two never appear in one table.
0c. A fixture where batch count and case count differ produces two clearly different
    denominators — assert they are not accidentally equal, which would hide a bug.
0d. One batch-to-bank failure containing 5 cases degrades Population A by exactly
    5 cases, each counted once — not 1, not 10.

Metrics correctness (hand-computed fixtures):
1. Match rate on a 10-CASE fixture with 7 confirmed cases == Decimal("0.70") exactly
2. Money-weighted match rate differs from row match rate when values are skewed —
   build a fixture where one ₹100,000 row is unmatched and 99 ₹10 rows are matched,
   assert row rate > 0.98 and money rate < 0.15
3. False-match rate counts only confirmed-but-wrong, not abstained
4. Abstentions never count as false matches
5. Cost per resolved exception with zero resolutions returns None, not a ZeroDivisionError
6. analyst_minutes_saved output string contains "derived estimate" and the assumption
7. Every money metric returns Decimal

Baselines — do NOT assert a ranking:
8. Naive baseline runs end to end and produces a complete metric set.
   DO NOT assert it is worse than deterministic-only. Naive may well match MORE cases
   by pairing on amount+date alone; it should match them less PRECISELY. Assert only
   that both produce valid metrics. Whatever the ordering turns out to be is a finding
   to report, not a test to satisfy. If a test needs the baseline to lose, delete the test.
9. Deterministic-only reproduces M3's numbers exactly
10. LLM-only baseline runs against the replay client with no network call
11. All baselines consume the identical normalized input — assert the same dataset
    object is passed to each

Runner:
12. run_eval produces results.json with every metric key present, no nulls except
    documented ones
13. Running eval twice on the same data produces identical results.json (D6)
14. The markdown table contains the residual-set sentence with numbers, not placeholders
15. run_eval on data with no residuals still produces a valid report

Guard:
16. Assert no network calls occur during the entire test module (monkeypatch socket)
```

**Verify:** `make eval` produces real numbers. **Paste them into README.md immediately** — this is the moment the project stops being a plan.

---

## M5a — Throughput benchmark *(Gate 4, 29 Aug — not cuttable)*

> The track bar reads *"Throughput plus measured accuracy plus an honest exception list."* Throughput is named first. This module makes it a measured deliverable rather than a claim.

### Build prompt

```
Read SDD section 8.1. Build the throughput instrumentation and benchmark harness.

settlesense/core/telemetry.py   <-- a SEPARATE module from types.py
1. StageTiming frozen dataclass: stage, seconds (float is fine HERE — telemetry is
   never hashed or compared), records_in, records_out, records_per_second property
   guarding division by zero.
2. MachineSpec: cpu, cores, ram_bytes, python_version.
3. RunTelemetry: timings tuple, peak_rss_bytes, machine.
4. StageTimer context manager using time.perf_counter(), appending to a collector
   passed at construction. No global state.
5. This module is the ONLY place in settlesense/ permitted to read a clock, and only
   perf_counter — never now(), today(), time(). Module docstring says so, cites D2.

Separation — this is the point of the module:
6. run_pipeline() returns a TUPLE: (ReconciliationResult, RunTelemetry).
7. ReconciliationResult contains NO timing field, no float, no timestamp, anywhere
   in its transitive type graph. Do NOT add one and strip it later.
8. settlesense/types.py must NOT import core/telemetry.py. Verify the import graph.
9. Golden serialization takes ReconciliationResult ONLY. The comparator has no strip
   step — there is nothing to strip. If you feel the need to write one, a field is in
   the wrong object; stop and tell me.
10. Telemetry writes to reports/. It never touches the state DB.

eval/bench.py
7. --sizes accepts a comma list, default 500,5000,25000,100000.
8. For each size: generate with a fixed seed, run the deterministic pipeline with the
   AI stage DISABLED, three repetitions, report the MEDIAN. Never the best run.
9. Measure peak memory with tracemalloc.
10. Capture machine spec automatically: platform.processor(), os.cpu_count(),
    total RAM, Python version. Write into the report header.
11. Benchmark the AI stage separately: seconds and rupees against RESIDUAL count,
    not row count. Use the replay cache so this costs nothing to re-run.
12. Emit reports/bench.md as a markdown table ready to paste into the README.

13. Makefile: bench target per SDD 8.
```

### Test prompt

```
Write tests/test_timing.py and tests/test_bench.py. No network, no real model.

Timing correctness:
1. StageTimer records a positive duration for a known sleep
2. records_per_second computes correctly for a known seconds/count pair
3. records_per_second returns 0.0, not ZeroDivisionError, when seconds is 0
4. Timings appear for EVERY stage the pipeline executed — assert the stage-name set
   equals the expected set exactly, so a newly added stage cannot go unmeasured

Separation — the important ones:
5. test_result_has_no_wallclock: walk ReconciliationResult's annotations RECURSIVELY;
   assert no float, no datetime, no timedelta, no field whose name contains
   time/duration/elapsed/seconds/rss
6. test_hash_ignores_telemetry: two runs with deliberately different timings (inject a
   sleep in one) produce IDENTICAL result hashes and IDENTICAL golden output
7. test_types_does_not_import_telemetry: inspect the import graph of settlesense/types.py
8. test_golden_serializer_signature: the serializer accepts ReconciliationResult and
   raises on being handed the (result, telemetry) tuple
9. test_no_wallclock_in_engine: grep settlesense/ for datetime.now/date.today/time.time;
   assert zero hits outside core/telemetry.py, and that telemetry.py uses only perf_counter

Bench harness:
10. bench runs at a small size and returns a table with one row per size
11. Median is reported, not min — feed three known durations via monkeypatched timer
    and assert the middle value is chosen
12. AI stage bench is reported separately from deterministic throughput
13. Machine spec fields are all populated and non-empty in the report header
14. bench executes zero model calls when the AI stage is disabled (assert on the
    replay client's call counter)

Guard:
15. Assert no network calls during the entire module (monkeypatch socket)
```

**Verify:** `make bench` writes `reports/bench.md`. Commit it. The deterministic rec/s figure is a headline number for the README and the video — if it is not in the thousands, profile before moving on.

---

## M6 — Incremental state *(Gate 5, 31 Aug — mandatory)*

### Build prompt

```
Build Module 6: settlesense/exceptions/store.py.

SQLite schema from SDD section 4.7: files, exceptions, audit, resolutions, watermarks.
resolutions.idempotency_key has a UNIQUE constraint.

ExceptionStore:
- ingest_file(path, day) -> IngestResult: hash content with sha256; if the hash exists,
  return IngestResult(skipped=True) and touch nothing
- reevaluate_open(dataset, config, as_of) -> list of newly closed exceptions. Runs the
  matching engine over ALL OPEN exceptions against the enlarged dataset.
- close_exception(exc, resolution_type, evidence_ids, day): builds
  idempotency_key = sha256(exception_id | resolution_type | sorted evidence_ids)
  and does INSERT OR IGNORE. Returns whether it was newly inserted.
- append_audit(exception_id, event, detail, day): append-only, never updated or deleted
- get_queue(status_filter) -> exceptions sorted by (-amount, exception_id)

Add a run_day(day, data_dir, config) driver that: ingests the day's files, re-evaluates
open exceptions, runs matching on new rows, and persists everything.

No clock access — day number and as_of are parameters.
```

### Test prompt

```
Write tests/test_m6_state.py. Use a temp SQLite file per test.

Idempotency (the core of this module):
1. Ingesting the same file twice: second call returns skipped=True, row count unchanged
2. Ingesting a byte-identical file under a different filename is still skipped
3. Ingesting a file with one byte changed is NOT skipped
4. close_exception called twice with identical arguments inserts exactly one resolution
5. close_exception with different evidence_ids produces a different idempotency_key
   and inserts a second row
6. Full Day 2 replay after a complete Day 1-2-3 run leaves the database identical —
   dump both to a canonical form and compare

Incremental resolution (the demo's spine):
7. Day 1 leaves exception E OPEN because its settlement file has not arrived
8. Day 2 file arrives; E transitions to CONFIRMED with confirmed_day == 2 and
   closed_day still None — export has not run, so it is explained but not actioned
9. E's audit trail contains both the Day 1 open event and the Day 2 confirm event, in order
10. An exception unresolvable on all three days remains OPEN with confirmed_day None
11. re-evaluation never reopens an already-CONFIRMED or CLOSED exception
12. re-evaluation never confirms an exception the enlarged data does not actually explain
13. Only CONFIRMED may transition to CLOSED. Assert OPEN->CLOSED, ABSTAINED->CLOSED
    and PENDING_*->CLOSED all raise. ABSTAINED reaches CLOSED only via
    ABSTAINED -> HUMAN_REVIEW -> CONFIRMED -> CLOSED; assert that path succeeds.

Audit integrity:
14. Audit entries are append-only — attempting UPDATE or DELETE raises
15. Audit ordering is stable and by (exception_id, arrival_day, sequence)

Determinism:
16. Running day1->day2->day3 twice from scratch produces identical database dumps (D6)
17. get_queue ordering is stable across runs
18. Running days out of order (3,1,2) produces the same final state as (1,2,3) —
    or, if you decide it must not, assert it raises a clear error. Pick one and test it.

End to end:
19. Full 3-day run on seed 42: report counts by lifecycle state — opened, confirmed,
    abstained, pending_ai_unavailable, closed_after_export. Assert every confirmed
    result agrees with truth_42.json. Use `closed_after_export` everywhere — in tests,
    README templates and demo notes. Use "incorrect confirmations" for a wrong
    CONFIRMED verdict. Never collapse explanation and accounting action into one
    state name — `confirmed` means the explanation verified, `closed_after_export`
    means the accounting entry was emitted.
    Assert every closure agrees with truth_42.json.
20. Zero incorrect confirmations — a case marked CONFIRMED whose explanation
    disagrees with truth_42.json. Assert exactly 0. Term is "incorrect confirmations",
    CLOSED is reserved for post-export state.
```

**Verify:** test 19 must be exactly zero. This is your Day 1→2→3 demo working.

---

## M7 — Hypothesis loop, verifier, confidence *(Gate 6, 1 Sep)*

### Build prompt

```
Build Module 7: settlesense/ai/.

client.py:
- LLMClient protocol: complete(prompt, schema) -> dict
- RealLLMClient: anthropic SDK, temperature=0, top_p=1, pinned model string.
  Its __init__ MUST raise RuntimeError if os.environ.get("PYTEST_CURRENT_TEST") is set (D7).
- ReplayLLMClient: reads fixtures/llm/<sha256(prompt)>.json. A cache miss raises
  FixtureMissError loudly, naming the missing hash. It must NEVER fall back to network.
- record_fixture(prompt, response) helper for building the fixture set

hypothesis.py:
- build_prompt(exception, dataset_slice, config) -> str. Deterministic: same exception
  produces a byte-identical prompt every time. Sort everything you interpolate.
- generate(exception, client) -> list[Hypothesis], max 3, ranked
- Validate output against the JSON schema in SDD section 4.4 BEFORE use. Invalid JSON
  after 2 retries -> treat as no hypothesis, do not crash.
- category must be in the closed enum, else reject the hypothesis
- Only exceptions whose category is in PDD section 6.2 may be sent. Assert this.

verifier.py:
- verify(hypothesis, dataset, config) -> VerificationResult
- Resolve every evidence_row_id; any missing -> immediate reject
- Evaluate the assertion using a SMALL ALLOW-LISTED EXPRESSION GRAMMAR.
  Never use eval() or exec(). Parse the assertion into a typed AST of permitted
  field references and comparison operators, then evaluate with Decimal.
- Recompute residual; check against tolerance from config
- Check the category's structural precondition (e.g. UTR_TRUNCATED_MAPPING requires the
  normalized narration UTR to be a genuine prefix of the candidate batch UTR)

confidence.py: the weighted formula from SDD section 4.6, weights from config.
Auto-confirm requires verification_passed AND confidence >= threshold. Confidence alone
can never confirm.

Wire it: residuals -> generate -> verify each in rank order -> first pass wins ->
all fail -> ABSTAINED.
```

### Test prompt

```
Write tests/test_m7_ai.py. Zero network calls — ReplayLLMClient only.

Client safety:
1. RealLLMClient.__init__ raises when PYTEST_CURRENT_TEST is set
2. ReplayLLMClient on a cache miss raises FixtureMissError naming the hash
3. ReplayLLMClient never opens a socket (monkeypatch socket and assert)

Prompt determinism:
4. build_prompt on the same exception produces a byte-identical string across 100 calls
5. Shuffling the input dataset row order does not change the prompt
6. Prompts for two different exceptions differ

Schema enforcement:
7. Valid JSON matching the schema is accepted
8. A category outside the closed enum is rejected
9. Missing required field -> rejected
10. Malformed JSON twice -> returns no hypothesis, does not raise
11. evidence_row_ids referencing a nonexistent row -> verifier rejects immediately

Verifier (this is the safety core — test it hardest):
12. A correct hypothesis with sound arithmetic -> passed=True
13. The same hypothesis with residual off by ₹0.01 -> passed=False
14. A hypothesis whose assertion is arithmetically true but whose category precondition
    fails -> passed=False
15. UTR_TRUNCATED_MAPPING where the narration UTR is NOT a prefix of the candidate
    -> passed=False even if amounts match exactly
16. The assertion parser rejects any expression outside the allow-list. Feed it
    "__import__('os').system('ls')" and assert it raises, does not execute
17. The parser rejects field references not in the permitted set
18. Verification uses Decimal — feed a float-producing path and assert it raises
19. A hypothesis that would confirm a match contradicting truth_42.json is caught by
    the verifier. Build at least 3 such adversarial hypotheses by hand.

Confidence:
20. verification_passed=False caps confidence such that auto-confirm is impossible
21. Two competing hypotheses both passing -> candidate_separation is low -> confidence
    below the auto-confirm threshold
22. Confidence is Decimal in [0, 1] for 50 generated inputs
23. Confidence never uses any model-reported value — assert the model's own confidence
    field, if present in a fixture, is ignored entirely

Loop:
24. First passing hypothesis wins; later ones are not evaluated
25. All three fail -> status ABSTAINED with all attempts recorded in the audit
26. An exception whose category is in PDD 6.1 (deterministic) is never sent to the model
27. End to end on seed 42 residuals with fixtures: report correctly explained,
    abstained, and false-matched. Assert false-match rate < 1% (the pre-declared gate).
```

---

## M8 — Evidence queue *(Gate 7, 2 Sep)*

### Build prompt

```
Build Module 8: settlesense/ui/app.py. Streamlit. Plain and legible, not styled.

One table, sorted by (-amount, exception_id):
  Exception ID | Category | Amount | Status | Confidence | Day opened | Day resolved

Row expansion shows: linked source rows (bank, settlement, ledger) with their key
fields; the AI hypothesis if any; the verification result and computed residual; the
abstention reason and competing candidates if abstained; the full audit trail.

A day selector (1/2/3) re-renders the queue as of that day so the Day1->Day2->Day3
narrowing is visible on screen.

Status colours: CONFIRMED green, CLOSED dark green with a check, ABSTAINED amber, OPEN grey,
PENDING_AI_UNAVAILABLE blue.

Read-only. Reads the SQLite store. No writes, no model calls.

If Streamlit proves fiddly, fall back to a static HTML table generated by a script —
legibility matters far more than interactivity.
```

### Test prompt

```
Write tests/test_m8_ui.py — test the data layer, not the rendering.

1. build_queue_view(store, day=1) returns rows sorted by (-amount, exception_id)
2. build_queue_view(day=2) shows the Day 1 exception as CONFIRMED when the Day 2
   evidence passes verification and no export has occurred
3. After a dry-run export, the same exception appears as CLOSED and its audit trail
   contains the export event
4. CONFIRMED and CLOSED render as visibly distinct labels — assert the label strings differ
5. Evidence expansion returns every linked row id present in the exception
6. An abstained exception's view includes all competing candidates with scores
7. The audit trail renders in chronological order
8. An empty store renders an empty table, not a crash
9. Money renders as "₹1,234.00" — exactly 2dp, thousands separators, no float artifacts
10. build_queue_view is deterministic across 50 calls
11. No model client is instantiated anywhere in the UI path
```

---

## M9 / M10 — Export and outage *(Gate 8, 3 Sep — cuttable)*

### Build prompt

```
Build Modules 9 and 10.

M9 settlesense/export/tally.py:
- build_batch(confirmed_exceptions, config) -> TallyBatch with
  idempotency_key = sha256(sorted exception_ids | batch_date)
- to_xml(batch) -> str, from a template, validated against the bundled XSD
- write_dry_run(batch, path): writes to disk, never transmits. Filename contains the
  idempotency key.
- Only CONFIRMED exceptions. Assert this — an ABSTAINED entry in the input raises.
- M9 is a Level 2 conditional module. If it is cut, the core submission is READ-ONLY
  and produces no external action. Say that plainly rather than implying a gap.
- On successful export, transition each exported exception CONFIRMED -> CLOSED, set
  closed_day, and append an export audit entry. The exporter is the ONLY writer of CLOSED.

M10 degradation, in settlesense/ai/client.py and the orchestrator:
- Timeout, HTTP error, or invalid JSON after 2 retries -> raise ModelUnavailable
- The orchestrator catches it, marks affected residuals PENDING_AI_UNAVAILABLE,
  leaves deterministic results untouched, confirms nothing, and returns normally
- The next run picks those exceptions back up
- Add a --simulate-outage flag to the runner for the demo
```

### Test prompt

```
Write tests/test_m9_export.py and tests/test_m10_degradation.py.

Export:
1. Generated XML validates against the XSD
2. Same confirmed set -> same idempotency_key across runs
3. Different confirmed set -> different key
4. An ABSTAINED exception in the input raises
5. An OPEN exception in the input raises
6. Exporting twice writes the same filename and does not duplicate content
7. Debit and credit totals in the batch balance exactly
8. Dry-run performs no network I/O (monkeypatch socket)
9. Money in XML has exactly 2 decimal places, no scientific notation

Degradation:
10. Client raising TimeoutError -> ModelUnavailable, orchestrator does not crash
11. Affected residuals are PENDING_AI_UNAVAILABLE, not ABSTAINED or CONFIRMED
12. Deterministic matches from the same run are unchanged and still CONFIRMED
13. Zero exceptions are confirmed during an outage. Assert exactly 0.
14. Re-running after recovery picks up PENDING_AI_UNAVAILABLE and processes them
15. Invalid JSON twice triggers the same safe path as a timeout
16. Partial outage — 3 of 10 calls fail — leaves 7 processed and 3 pending, nothing lost
17. Database state after an outage run is valid and re-runnable
```

---

## M11 — Cash position panel *(Gate 8, 3 Sep — OPTIONAL)*

> **M11 is optional and must not block submission. It may begin only after M5a, M6, M7 and M8 pass. If any protected gate slips, M11 is skipped.**

> The track headline is *"Run the books and the cash position."* The journal export covers the books. This is a read-only view over state the engine already holds, and it makes the submission answer the headline in full. **Do not start this if M5a, M6, M7 or M8 are incomplete.**

### Build prompt

```
OPTIONAL MODULE. Before starting, confirm M5a, M6, M7 and M8 are all done. If any is
incomplete, tell me to skip this and stop.

Read PDD section 8.4b. Build a READ-ONLY cash-position view. No new inputs, no new
matching logic, no forecasting model. It derives entirely from existing state.

CRITICAL — THE BASIS. The buckets reconcile against EXPECTED NET, never captured gross.
Bank credits are net of fees, GST and refunds; captured payments are
gross. Summing net buckets against a gross total is an accounting-basis error and the
invariant would be false by exactly the deduction total. Do not write it that way.

  cash_basis_total = sum(case.expected_net for all captured cases)
  where expected_net = gross - fee - tax - refunds (SDD 3.1b)
  settled_to_bank + in_flight + at_risk + unexplained == cash_basis_total

settlesense/cash/position.py
1. CashPosition frozen dataclass, all money as Decimal:
   basis: str                       # always "EXPECTED_NET" in v1
   cash_basis_total: Money
   settled_to_bank, in_flight, at_risk, unexplained: Money
   deductions_reference: Money      # gross - expected_net = fees + GST + refunds.
                                    # REPORTED not bucketed,
                                    # so a reader can tie back to gross
   expected_landings: tuple[tuple[date, Money], ...]   # sorted by date

2. compute_position(state, as_of: date, calendar) -> CashPosition
   - cash_basis_total: sum of expected_net over captured cases
   - settled_to_bank: expected_net of cases with a confirmed bank-credit link up to as_of
   - in_flight: expected_net of cases with no bank link, still inside expected T+N
   - at_risk: expected_net of cases past expected date and unlinked (MISSING_VS_LATE_CREDIT)
   - unexplained: expected_net of cases in any unresolved/abstained state
   - expected_landings: in_flight bucketed by working-day calendar

3. as_of is INJECTED per D2. No clock reads.
4. Invariant: the four buckets are mutually exclusive and exhaustive against
   cash_basis_total, to the cent. Raise if they do not reconcile.
5. Every case appears in exactly one bucket. Assert the case-id sets are disjoint and
   their union is the full captured-case set — reconciling on money alone can hide a
   double-count that happens to cancel.

5. Add one panel at the top of the evidence queue UI, WITH THE BASIS STATED:
   "Cash view — expected net settlement value after configured deductions"
   "₹X settled · ₹Y in flight, landing <range> · ₹Z at risk · ₹W unexplained"
   Each figure links to the underlying rows. No charts.

Do NOT add: forecasting, projections beyond known settlement windows, working-capital
advice, or anything requiring a new data source.
```

### Test prompt

```
Write tests/test_cash_position.py. No network, no real model.

Basis correctness — test these FIRST:
1. Known fixture produces exactly the expected four bucket values
2. Buckets are mutually exclusive by CASE ID, not just by money — assert the four
   case-id sets are pairwise disjoint
3. Buckets are exhaustive against cash_basis_total (expected NET), to the cent.
   Assert the invariant raises when a fixture is deliberately made to violate it.
3a. A fixture with non-zero fees, GST and refunds still reconciles. This is the
    regression test for the gross-vs-net basis error.
    This is the regression test for the gross-vs-net basis error: assert explicitly
    that the buckets do NOT sum to captured gross, and that
    captured_gross - cash_basis_total == deductions_reference
3b. CashPosition.basis == "EXPECTED_NET" and the UI label states the basis
4. expected_landings skips weekends and configured bank holidays
5. expected_landings is sorted ascending by date and contains no duplicate dates

Boundary behaviour:
6. A payment exactly ON its expected date is in_flight, not at_risk
7. A payment one working day past expected is at_risk
8. Empty state returns all zeros, not None and not an exception
9. A fully reconciled dataset returns in_flight == 0 and at_risk == 0

Determinism:
10. Same state and same as_of produce byte-identical CashPosition across two runs
11. Advancing as_of by one working day moves the correct payments from
    in_flight to at_risk and nothing else changes
12. No clock reads — grep the module for now/today/time

Guard:
13. Assert no network calls during the module (monkeypatch socket)
```

**Verify:**

```
settled_to_bank + in_flight + at_risk + unexplained == cash_basis_total
    where cash_basis_total = sum(ReconciliationCase.expected_net)
captured_gross - cash_basis_total == deductions_reference
```

Exact Decimal equality on both. The buckets reconcile to **expected net settlement value**, never to total captured gross — bank credits are net of fees, GST and refunds. If they do not reconcile, the bug is in your matching state rather than in this module, which makes the panel a useful cross-check on everything upstream.

---

## Final packaging prompt *(Gate 8–9, 3–4 Sep)*

```
Final packaging. No new features.

1. Rewrite README.md so the FIRST screen is, in this order:
   - Title, one line
   - The residual-set sentence with REAL measured numbers
   - The results table with real numbers from make eval
   - `make eval` reproduction commands including the holdout seed
   - Screenshot of the evidence queue
   - Architecture diagram
   - Day1->Day2->Day3 example
   - Link to LIMITATIONS.md
   No product introduction paragraph above the results.

2. Write LIMITATIONS.md covering: synthetic data only; the frozen generator commit hash;
   what the withheld noise types measured; threshold rationale and that ₹500 is a project
   safety threshold not a production loss claim; analyst-minutes-saved is a derived
   estimate with its assumption stated; the Tally export is schema-validated but not
   tested against a live Tally instance; and that these results do not establish
   production performance on real merchant data.

3. Generate the architecture diagram as a mermaid block from SDD section 4.

4. Run `make check`, `make test`, `make eval` one final time. Paste the real output
   into the README. Verify zero placeholders remain — grep for "TO BE MEASURED",
      "TODO", "XXX", "<hash>", "GENERATOR_COMMIT", "TO BE MEASURED" and fix every hit.
   IMPORTANT: `generator_commit: null` inside truth files is INTENTIONAL pre-freeze
   metadata, not a placeholder. Truth files are written before M1F, when no hash
   exists, and are never rewritten. The scan must NOT flag it. It flags only symbolic
   placeholders in FINAL artifacts: README, LIMITATIONS.md, reports/.
   After M1F, GENERATOR_MANIFEST.json holds the real SHA from `git rev-parse HEAD`,
   and README/LIMITATIONS/reports read it from there.

5. Print a final summary: total tests, pass rate, coverage, eval numbers, and any
   assertion currently failing.
```

---

## Test-count targets

| Module | Tests | Must be exactly zero |
|---|---|---|
| M0 | 13 | — |
| M1 | 19 | — |
| M2 | 25 | — |
| M3 | 28 | False matches on clean chains (t28) |
| M4 | 17 | False accepts (t15) |
| M5 | 16 | Network calls (t16) |
| M6 | 20 | Incorrect confirmations (t20) |
| M7 | 27 | — (false-match rate < 1%, t27) |
| M8 | 11 | — |
| M9/M10 | 17 | Confirmations during outage (t13) |
| M5a | 15 | Model calls during bench (t14) |
| M11 (optional) | 13 | Bucket overlap (t2) |
| **Total** | **~219** | |

`make test` target: under 60 seconds, zero network calls, byte-identical results on repeat runs.

---

## If you fall behind

| Date | If not done | Cut immediately |
|---|---|---|
| 27 Aug | M3 incomplete | Split-settlement subset-sum, M9, M10 |
| 29 Aug | No real eval numbers | All features. Fix correctness, run eval, report. |
| 31 Aug | M6 incomplete | M9, M8 polish. Keep Day 1→Day 2 and replay safety. |
| 1 Sep | M7 incomplete | Run the AI experiment on a 200-case sample and say so |

M8 compresses to a static HTML table in two hours if needed. M1–M6 do not compress.

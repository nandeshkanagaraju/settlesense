# SettleSense — Software Design Document (SDD)

**Version:** 1.0
**Companion to:** SettleSense_PDD.md
**Target runtime:** Python 3.11+

---

## 1. Determinism charter

Every rule below is testable, and each has a corresponding test in the suite. This section is the spine of the project — violations here invalidate the evaluation.

| # | Rule | Enforcement |
|---|---|---|
| D1 | All money is `decimal.Decimal`, quantized to 2 dp with `ROUND_HALF_UP`. **Floats are banned in money paths.** | `test_no_float_money` scans dataclass annotations and asserts no `float` on money fields |
| D2 | No `datetime.now()`, `date.today()`, or `time.time()` in engine code. An `as_of: date` is injected. | grep-based test over `settlesense/` |
| D3 | No module-level `random`. A single seeded `random.Random(seed)` is passed explicitly. | grep-based test over `gen/` |
| D4 | Output ordering never depends on `set` or `dict` iteration. All result lists are explicitly sorted by a stated key. | Golden-file tests run twice in-process and byte-compare |
| D5 | Candidate ranking uses a **total order** with an explicit tie-break chain. If the chain exhausts and candidates remain tied, the engine **abstains** — it never picks arbitrarily. | `test_tie_forces_abstention` |
| D6 | The engine is a pure function of (inputs, config, `as_of_date`). Same inputs → byte-identical `ReconciliationResult`. **Telemetry is a separate return value and is excluded by construction, not by stripping.** | `test_run_twice_identical`, `test_result_has_no_wallclock` |
| D7 | The LLM is **never called in tests.** Tests use a fixture-replay client. | `conftest.py` injects `ReplayLLMClient` by default; the real client raises if instantiated under pytest |
| D8 | Model calls, when made for real, use `temperature=0`, `top_p=1`, a fixed seed where supported, and a pinned model string. Non-determinism is still assumed and handled by the verifier. | Config assertion |
| D9 | File ingestion is idempotent by content hash. Re-ingesting a byte-identical file is a no-op. | `test_replay_no_duplicate` |
| D10 | Every ID the engine generates is derived deterministically (`sha256` of a canonical tuple), never `uuid4()`. | `test_ids_stable_across_runs` |
| D11 | All metrics are computed over `ReconciliationCase` (Population A). Batch↔bank link metrics (Population B) are reported separately and never merged into a Population A figure. | `test_metrics_single_denominator` |
| D12 | Scoring weights, thresholds and ratios are `Decimal`. The config loader **raises** on any YAML float. | `test_config_has_no_floats` |
| D13 | No date anywhere in the repo — fixtures, prompts, configs, generated data — falls outside 2026. | `test_no_stale_years` |

---

## 2. Repository layout

```
settlesense/
├── Makefile
├── README.md                    # results first
├── LIMITATIONS.md
├── pyproject.toml
├── config/
│   ├── mdr_rates.yaml           # method → rate, per merchant profile
│   ├── calendar_v1.yaml         # version: calendar_v1, working days, holidays, T+N
│   └── thresholds.yaml          # abstention + safety thresholds
├── gen/                         # ── INDEPENDENT PATH — FROZEN AT GATE 2 ──
│   ├── __init__.py
│   ├── generate.py              # CLI entrypoint
│   ├── profiles.py              # 3 merchant profiles
│   ├── lifecycle.py             # clean order→payment→settlement-line→batch→bank chains
│   ├── noise.py                 # noise injectors (incl. 2 withheld)
│   ├── truth.py                 # ground-truth writer + self-check
│   └── README.md                # "do not import from settlesense/"
├── settlesense/
│   ├── types.py                 # Money, domain records, Exception model
│   ├── config.py                # typed config loading
│   ├── ingest.py                # M2 — ALL file I/O lives here
│   ├── normalize.py             # M2 — pure functions only, zero I/O
│   ├── matching/
│   │   ├── exact.py             # M3
│   │   ├── arithmetic.py        # fee/GST/rounding
│   │   ├── timing.py            # calendar + T+N
│   │   ├── duplicates.py
│   │   ├── fuzzy_utr.py         # M4
│   │   └── engine.py            # orchestrator
│   ├── exceptions/
│   │   ├── taxonomy.py
│   │   └── store.py             # SQLite, watermarks, idempotency (M6)
│   ├── ai/
│   │   ├── client.py            # LLMClient protocol + Real + Replay
│   │   ├── hypothesis.py        # structured generation (M7)
│   │   ├── verifier.py          # deterministic verification
│   │   └── confidence.py
│   ├── export/
│   │   └── tally.py             # M9
│   └── ui/
│       └── app.py               # M8 evidence queue
├── eval/
│   ├── baselines/
│   │   ├── naive.py
│   │   ├── deterministic_only.py
│   │   └── llm_only.py
│   ├── metrics.py
│   └── run_eval.py              # M5
├── fixtures/
│   └── llm/                     # recorded model responses for tests
└── tests/
    ├── conftest.py
    ├── golden/
    └── test_*.py
```

**Hard rule:** `gen/` never imports from `settlesense/`, and `settlesense/` never imports from `gen/`. A test enforces this by AST-scanning imports in both directions.

---

## 3. Domain model

### 3.0 Source packages vs internal tables

The merchant supplies **three source packages**. The gateway package is a bundle, which is why three sources expand into **six** normalized internal tables. This distinction was previously blurred and is now normative.

| Source package | Artifacts inside | Internal tables produced |
|---|---|---|
| `gateway` | settlement report | `settlement_lines`, `settlement_batches`, `refund_rows` |
| `bank` | bank statement | `bank_rows` |
| `ledger` | order / invoice export | `ledger_rows`, `payment_rows` |

Six tables: `settlement_lines`, `settlement_batches`, `refund_rows`, `bank_rows`, `ledger_rows`, `payment_rows`. Generated as six `day{N}_*.csv` files.

**Chargebacks are out of scope for v1.** A chargeback is a batch-level debit whose accounting model needs a dispute artifact, its own truth edges, and its own conservation terms. Track 04 does not require it, and a category with no verifiable input is worse than no category. `CHARGEBACK_DEBIT`, `PAYMENT_TO_DISPUTE` and `dispute_rows` are therefore absent from v1 and recorded in `LIMITATIONS.md` as future scope. The `REFUND` line type already exercises the signed-line arithmetic that a dispute debit would use, so the design remains extensible without the scope.

### 3.1 The canonical reconciliation unit

Every rate, count, precision and value metric is computed over **one unit only**: the `ReconciliationCase`. Mixing ledger rows, payment rows, settlement rows, batches and bank credits in a single denominator makes "match rate" meaningless, so it is prohibited.

```python
# settlesense/types.py

Money = Decimal  # always quantized to Decimal("0.01"), ROUND_HALF_UP

@dataclass(frozen=True)
class ReconciliationCase:
    """THE canonical unit. One case per captured payment — ALWAYS one.

    A split settlement produces ONE case holding MULTIPLE payment_line_ids.
    It never produces two cases: that would double-count the denominator and
    make the match rate depend on how the gateway happened to batch the payout.

    Chosen because a payment is the finest grain with exactly one expected money
    outcome and one accounting treatment. All headline metrics use this
    denominator and no other.
    """
    case_id: str                            # sha256(b"case|" + payment_id)[:16], D10
    payment_id: str
    order_id: str
    merchant_profile: str
    expected_gross: Money
    expected_net: Money                     # gross - fee - tax - refunds; see §3.1b
    settlement_line_ids: tuple[str, ...]    # sorted; ALL lines touching this payment
    payment_line_ids: tuple[str, ...]       # sorted; PAYMENT lines only.
                                            # len > 1 => SPLIT_SETTLEMENT.
                                            # Case matching uses THIS field, never
                                            # settlement_line_ids — a refund line is
                                            # not a second settlement of the payment.

@dataclass(frozen=True)
class CaseOutcome:
    """What the engine concluded about one case. Goes into ReconciliationResult."""
    case_id: str
    status: ExceptionStatus
    observed_net: Money | None              # None when no bank credit was linked
    variance: Money | None                  # expected_net - observed_net
    category: str | None                    # from taxonomy.VARIANCE_CATEGORIES only.
                                            # Deduction categories (MDR_FEE,
                                            # GST_ON_FEE, REFUND_OFFSET) are
                                            # components of expected_net and are
                                            # NEVER emitted here. See PDD 6.1.
    batch_id: str | None
    bank_row_id: str | None
    resolved_by: str | None                 # DETERMINISTIC | AI_VERIFIED | HUMAN
    confidence: Decimal | None
```

**Two populations, reported separately and never combined:**

| Population | Unit | Denominator | Where reported |
|---|---|---|---|
| **A — payment cases** | `ReconciliationCase` | count of captured payments | **All headline metrics.** case match rate, residual count, residual precision, abstention rate, false-match rate, gross-exposure value, expected-net cash value |
| **B — batch↔bank links** | `TruthEdge(BATCH_TO_BANK)` | count of settlement batches | A separate, clearly labelled table. Its own link rate and its own false-link rate |

**Population C — row-grain variances (added at M1; reported via `ReconciliationResult.row_variances`).** Some variances belong to neither population because they are not payments and not batches: a `DUPLICATE_CONFIRMED` ledger row is not a payment, and an orphan bank credit has no batch. Without a home they are dropped from truth entirely and become unscoreable. They are recorded in `truth.row_variances` keyed by source row id, reported in their own small table with a row-count denominator, and **never merged into A or B**.

Rules, enforced by tests:

- Population A and Population B rates are **never averaged, summed, or presented in the same column**.
- A batch↔bank failure propagates into Population A only through the cases in that batch, each counted once.
- Money-weighted metrics for Population A use `expected_gross`; for Population B they use `batch.net_total`. The two money totals are reported separately and are not expected to be equal.

### 3.1a Batch composition — why the arithmetic closes

A batch is **not** a bag of payments. It is a set of signed line items, exactly as a real settlement report is:

```
batch.net_total = Σ (line.net for EVERY line in the batch, signed)
                = Σ PAYMENT line nets  −  Σ REFUND amounts

bank_credit.amount == batch.net_total          # exact, always
```

This is the piece that was previously missing. Deducting a refund from `expected_net` while leaving the batch and bank amount untouched would make the cash invariant unsatisfiable. With `REFUND` as a signed batch line, both sides move together and conservation closes. A future `DISPUTE_DEBIT` line type would slot into the same arithmetic unchanged.

Generator obligations that follow:

- Every refund emits **exactly one** `REFUND` line, in the batch settling on or after `refund.created_event_date`.
- `test_batch_total_equals_signed_line_sum` and `test_bank_credit_equals_batch_total` run on a fixture containing one PAYMENT line and one REFUND line in the same batch.

### 3.1b The expected-net formula — one definition, used everywhere

```
expected_net = gross - fee - tax - refunds
```

Refunds count once they exist as a `RefundRow` with a corresponding `REFUND` settlement line. There is no dispute term: chargebacks are out of v1 scope (§3.0).

| Term | Included in `expected_net`? |
|---|---|
| MDR fee | Yes |
| GST on fee | Yes |
| Refunds (any) | Yes |

Consequences, each of which must hold:

- `deductions_reference = expected_gross − expected_net` covers fees, GST and refunds.
- Global conservation: `Σ gross == Σ expected_net + Σ fees + Σ tax + Σ refunds`, and independently `Σ bank credits == Σ batch net_totals == Σ signed settlement line nets`. **Both must hold**; the first is case-side, the second is batch-side.
- **Two money bases exist and are labelled distinctly.** `expected_gross` measures *gross exposure* and is used for Population A row-value conservation. `expected_net` measures *cash* and is used for the cash panel and any metric explicitly about money landing in the bank. A metric never silently switches basis; every money metric names its basis in its label.

### 3.2 Typed truth relationships

The earlier invariant "no row may be the true partner of two rows" was **wrong**: one settlement batch legitimately contains many settlement rows. Truth is a typed edge set with declared cardinality, checked per type.

```python
class EdgeType(StrEnum):
    ORDER_TO_PAYMENT      = "order_to_payment"
    PAYMENT_TO_SETTLEMENT = "payment_to_settlement"
    SETTLEMENT_TO_BATCH   = "settlement_to_batch"
    BATCH_TO_BANK         = "batch_to_bank"
    PAYMENT_TO_REFUND     = "payment_to_refund"

@dataclass(frozen=True)
class TruthEdge:
    edge_type: EdgeType
    src_id: str
    dst_id: str
```

| Edge type | Cardinality | Invariant enforced |
|---|---|---|
| `ORDER_TO_PAYMENT` | 1 : 1 | Each order has at most one payment; each payment exactly one order |
| `PAYMENT_TO_SETTLEMENT` | 1 : N | **Links a payment ONLY to `PAYMENT` lines.** Refund lines are reached via `PAYMENT_TO_REFUND`, never this edge. N ≥ 2 (counting PAYMENT lines only) is the structural condition for the `SPLIT_SETTLEMENT` taxonomy value. This is an *edge type*, not a category — there is no `SPLIT_SETTLEMENT_GROUPING` value. |
| `SETTLEMENT_TO_BATCH` | N : 1 | `src_id` = `SettlementLine.settlement_id` (**every v1 line, both types**), `dst_id` = `batch_id`. Many lines per batch is the normal case, not an error |
| `BATCH_TO_BANK` | 1 : 1 | `src_id` = `batch_id`, `dst_id` = `bank_txn_id`. One batch credits once; two bank credits for one batch is a generator bug |
| `PAYMENT_TO_REFUND` | 1 : N | Links a payment to its `RefundRow`s, and separately to the corresponding `REFUND` lines |

`gen/truth.py` exposes `check_cardinality(edges) -> list[Violation]` which validates **each edge type against its own declared cardinality**. This runs before the generator is frozen and fails the build on any violation.

### 3.3 Row types

> **Identifier rule.** `settlement_id` is **payment-level only** — it names one `SettlementLine`. `batch_id` is **batch-level only** — it names one `SettlementBatch`. The two are never interchanged and never share a namespace. Reusing one name for both grains makes `SETTLEMENT_TO_BATCH` self-referential, corrupts evidence links, and silently breaks Population B joins. `test_id_namespaces_disjoint` asserts the two ID sets have empty intersection.

@dataclass(frozen=True)
class LedgerRow:
    order_id: str
    invoice_no: str
    gross: Money
    order_date: date
    customer_id: str
    sku: str

@dataclass(frozen=True)
class PaymentRow:
    payment_id: str
    order_id: str
    method: str            # card | upi | netbanking | wallet
    authorized: Money
    captured: Money
    status: str            # captured | refunded | partial | failed
    captured_at: date

@dataclass(frozen=True)
class RefundRow:
    refund_id: str
    payment_id: str
    amount: Money
    created_at: date

class SettlementLineType(StrEnum):
    PAYMENT = "payment"    # credit: a captured payment settling
    REFUND  = "refund"     # debit:  a refund deducted from the batch

@dataclass(frozen=True)
class SettlementLine:
    """A SIGNED LINE in a settlement batch. NOT 'one row per payment'.

    A batch holds lines of mixed type. Only PAYMENT lines represent a payment
    settling; a REFUND line is a deduction that references a payment without
    being a settlement of it. Conflating the two is what makes
    PAYMENT_TO_SETTLEMENT cardinality ambiguous.
    """
    settlement_id: str            # unique LINE id. Never a batch id.
    batch_id: str                 # FK to SettlementBatch.batch_id
    line_type: SettlementLineType
    payment_id: str               # the payment this line relates to
    refund_id: str | None         # set iff line_type == REFUND
    gross: Money
    fee: Money                    # 0 on REFUND lines
    tax: Money                    # 0 on REFUND lines
    net: Money                    # SIGNED: + for PAYMENT, − for REFUND
    settled_event_date: date
```

**Per-type field invariants**, each asserted by `test_line_invariants`:

| `line_type` | `payment_id` | `refund_id` | `gross`/`fee`/`tax` | `net` |
|---|---|---|---|---|
| `PAYMENT` | set | `None` | from the rate table | `+ (gross − fee − tax)` |
| `REFUND` | set | set | all `0` | `− refund.amount` |

```python

@dataclass(frozen=True)
class SettlementBatch:    # BATCH-level. The payout that hits the bank.
    batch_id: str         # THE batch identifier. NEVER named settlement_id.
    utr: str
    net_total: Money
    settled_event_date: date

@dataclass(frozen=True)
class BankRow:
    bank_txn_id: str
    value_date: date
    amount: Money
    narration: str        # free text; UTR may be truncated, garbled, or absent
    direction: str        # credit | debit
```

### Exception model

```python
@dataclass(frozen=True)
class Exception_:
    exception_id: str          # deterministic: sha256(canonical tuple)[:16]
    category: str              # from closed taxonomy
    amount: Money
    status: ExceptionStatus    # see lifecycle below
    confidence: Decimal        # verification-derived, 0.00–1.00
    evidence_row_ids: tuple[str, ...]   # sorted
    reason: str
    resolved_by: str           # DETERMINISTIC | AI_VERIFIED | HUMAN | None
    first_seen_day: int        # arrival_day, an int — never a timestamp
    confirmed_day: int | None
    closed_day: int | None
    audit: tuple[AuditEntry, ...]

@dataclass(frozen=True)
class AuditEntry:
    """Append-only. One per status change or evidence addition."""
    exception_id: str
    arrival_day: int              # integer day index, never a timestamp (D2)
    sequence: int                 # ordering within a day; caller-supplied
    from_status: ExceptionStatus | None   # None on the opening entry
    to_status: ExceptionStatus
    actor: str                    # DETERMINISTIC | AI_VERIFIED | HUMAN | EXPORTER
    note: str
    evidence_ids: tuple[str, ...] # sorted
```

### Exception lifecycle — CONFIRMED means explained, CLOSED means actioned

These were previously used interchangeably. They are now distinct states with a legal transition set.

```
                    ┌──────────────────────────┐
                    ▼                          │ (new evidence arrives)
  OPEN ──────▶ PENDING_EVIDENCE ───────────────┤
    │                                          │
    ├─────────▶ PENDING_AI_UNAVAILABLE ────────┘
    │
    ├─────────▶ CONFIRMED ────────▶ CLOSED
    │           (an explanation      (accounting action emitted;
    │            passed verification) the exporter is the ONLY writer)
    │                ▲
    │                │
    └─────────▶ ABSTAINED ──▶ HUMAN_REVIEW ─┘
```

**CLOSED has exactly one predecessor: `CONFIRMED`.** An abstained exception a human
resolves passes *through* `CONFIRMED` — the human's decision is an explanation, and
emitting the accounting entry is a separate act. `ABSTAINED → CLOSED` is illegal even
when a human approves both steps in one click; the store records both transitions.

| State | Meaning | Counts toward |
|---|---|---|
| `OPEN` | Detected, not yet explained | Residual set |
| `PENDING_EVIDENCE` | Expected file has not arrived | Residual set, flagged separately |
| `PENDING_AI_UNAVAILABLE` | Model down; deterministic result intact | Residual set, flagged separately |
| `CONFIRMED` | An explanation passed verification | **All accuracy metrics** |
| `ABSTAINED` | No hypothesis passed; needs a human | Abstention rate |
| `CLOSED` | Confirmed **and** the accounting action was emitted | Export metrics only |

`ExceptionStatus` has exactly these **six** members. `HUMAN_REVIEW` in the diagram above is **not a status** — it is the M8 review queue, i.e. where an `ABSTAINED` exception waits. Treating it as a seventh status would make the abstention-rate denominator ambiguous, since an exception would leave `ABSTAINED` merely by being looked at.

Enforced rules: **only `CONFIRMED` may transition to `CLOSED`** — `OPEN→CLOSED`, `ABSTAINED→CLOSED` and `PENDING_*→CLOSED` all raise; `CLOSED` is terminal; the exporter is the sole writer of `CLOSED`; accuracy metrics read `CONFIRMED`, never `CLOSED`; the exporter reads `CONFIRMED` and writes `CLOSED`. `test_illegal_transitions_rejected` asserts every non-listed transition raises.

---

## 4. Pipeline

```
ingest ─▶ normalize ─▶ deterministic match ─▶ residual set
                                                    │
                                                    ▼
                                          hypothesis generation (LLM)
                                                    │
                                                    ▼
                                          deterministic VERIFIER
                                                    │
                                      ┌─────────────┼─────────────┐
                                  CONFIRMED     next hypothesis  ABSTAIN
                                      │                            │
                                      ▼                            ▼
                              exception store ◀────────── human review queue
                                      │
                                      ▼
                          Tally XML export (dry-run, idempotent)
```

### 4.1 Normalization (M2)

- UTR: uppercase, strip non-alphanumerics, collapse whitespace. Retain both `raw` and `normalized`.
- Amounts: strip `₹`, thousands separators, handle `1,234.00` / `1234.0` / `1234` / `(1234.00)` for debits → `Decimal`.
- Dates: parse a fixed allow-list of ISO formats. A date with exactly one valid reading parses; a genuinely ambiguous one (`03/04/2026`) **raises** rather than being guessed. `parse_date(raw, rule)` takes a resolved rule, not a profile name — there is deliberately no config key, because every source in the frozen dataset emits ISO and a key with zero consumers is worse than none. Wire a rule through when a source actually emits an ambiguous date.
- **Month-name formats are refused outright.** `strptime`'s `%b` resolves through the process locale, and a locale-dependent parser inside a byte-compared system is precisely the hazard D1/D6 exist to prevent.
- See §4.1a — **four distinct date concepts** must never be conflated.
- Narrations: uppercase, collapse whitespace, extract UTR candidates by regex; keep all candidates, ranked.
- Every normalization is a pure function with a unit test containing at least one hostile input.

### 4.1a Four date concepts and the versioned calendar

Conflating these is how timing bugs become silent false matches. Each is a separate field with a separate type.

| Concept | Type | Meaning | Who sets it |
|---|---|---|---|
| `event_date` | `date` | When the business event occurred — capture, settlement, bank value date | Generator, from the simulated timeline |
| `file_content_date` | `date` | The period the file claims to cover, printed in its header | Generator; may legitimately disagree with `event_date` |
| `arrival_day` | `int` (1-indexed) | Which simulated delivery day the file was handed to the system | Generator; out-of-order arrival means `arrival_day` order ≠ `event_date` order |
| `as_of_date` | `date` | The engine's notion of "today" | **Injected** at every entry point, per D2 |

`arrival_day` is an integer sequence, never a timestamp. Replay safety keys on `(content_hash, arrival_day)`.

**Calendar.** `config/calendar_v1.yaml`, versioned and referenced by version string in every result:

```yaml
version: 1
timezone: Asia/Kolkata
weekly_offs: [saturday, sunday]      # per-profile override permitted
holidays:                            # SYNTHETIC for the evaluation — see note
  - 2026-01-15
  - 2026-03-04
  # ...
```

> **Note recorded in LIMITATIONS.md:** the v1 calendar is a **synthetic holiday set for the 2026 simulated timeline**, not the RBI holiday schedule. Real bank holidays are a config swap requiring no code change. The evaluation measures whether the engine handles a holiday correctly, not whether it knows India's actual holiday list — and inventing a real-looking holiday list would be a false claim of accuracy.

All simulated dates fall inside **2026**. Any 2025 date anywhere in the repo is a defect; `test_no_stale_years` greps fixtures, prompts and configs for `20(1\d|2[0-5])-` and fails on a hit.

### 4.2 Deterministic matching passes (M3)

Passes run in strict order. Once matched, a row is removed from the candidate pool.

| Pass | Rule |
|---|---|
| P1 exact payment | `settlement.payment_id == payment.payment_id` |
| P2 exact batch↔bank | Keyed on `batch_id`: normalized UTR equal, `net_total == bank.amount`, date within T+N window |
| P3 arithmetic | `gross − fee − tax − refunds == net`, fee/tax recomputed from rate table |
| P4 refund offset | Exact-amount refund matched by `refund_id` |
| P6 timing | Unmatched but explained by settlement date outside period per calendar |
| P7a duplicate **confirmed** | Byte-identical row content across all fields *and* a distinct source line number → `DUPLICATE_CONFIRMED`. This is an ingestion artefact and is deterministic. |
| P7b duplicate **candidate** | Same `(customer_id, gross)`, different `order_id` → **not** resolved here. Emitted as `DUPLICATE_CANDIDATE` into the residual set with both rows attached. Distinguishing a data duplicate from a genuine repeat purchase is interpretive and belongs to the AI layer. |
| P8 fuzzy UTR | See §4.3 |
| P9 rounding | Residual ≤ ₹1.00 after all above |

Everything surviving P9 is the **residual set**.

### 4.3 Fuzzy UTR (M4) — belongs in the deterministic layer

This is deliberately in Gate 3, not deferred. If fuzzy matching sits outside the deterministic baseline, the residual set is artificially inflated and the model looks better than it is.

Score is computed in `Decimal`, never float — float comparison near a threshold is a determinism hazard on a par with float money.

```python
WEIGHT_PREFIX = Decimal("0.50")   # loaded from config via a strict loader
WEIGHT_EDIT   = Decimal("0.30")   # that REJECTS any YAML float and requires
WEIGHT_AMOUNT = Decimal("0.20")   # quoted decimal strings

prefix_ratio = (Decimal(matched_chars) / Decimal(total_chars)).quantize(Q6)
edit_ratio   = (Decimal(distance)      / Decimal(max_len)).quantize(Q6)
score = (WEIGHT_PREFIX * prefix_ratio
       + WEIGHT_EDIT   * (Decimal(1) - edit_ratio)
       + WEIGHT_AMOUNT * amount_exact).quantize(Q6)   # Q6 = Decimal("0.000001")
```

`config/thresholds.yaml` numerics are quoted strings coerced to `Decimal` by `load_config()`, which **raises on encountering a Python float**. `test_config_has_no_floats` enforces it. Levenshtein distance is an integer; every ratio is an exact `Decimal` division quantized to 6 dp, so scoring is reproducible across platforms.

Accept only if `best ≥ Decimal("0.85")` **and** `best − runner_up ≥ Decimal("0.15")`. Otherwise emit `UTR_TRUNCATED_MAPPING` into the residual set with all candidates attached. Never accept on score alone when two candidates are close — that is exactly the ambiguity the AI layer exists to examine.

### 4.4 Hypothesis loop (M7)

Model output is a strict JSON schema, validated before use:

```json
{
  "category": "UTR_TRUNCATED_MAPPING",
  "candidate_id": "SET_1042",
  "assertion": {"lhs": "bank.amount", "op": "==", "rhs": "settlement.net_total"},
  "residual_amount": "4312.00",
  "evidence_row_ids": ["bank_87", "settlement_1042", "payment_9912"],
  "reason": "Narration holds a truncated UTR matching this batch prefix."
}
```

Constraints: `category` must be in the closed enum; `assertion` uses a small allow-listed expression grammar — **the verifier evaluates it, never `eval()`**; every `evidence_row_id` must exist in the loaded dataset. Up to 3 ranked hypotheses per exception; each verified in order; first pass wins; all fail → `ABSTAINED`.

### 4.5 Verifier

Independent of the model. For each hypothesis: resolve every evidence ID (missing → reject); recompute the assertion with `Decimal`; check the residual is zero or within tolerance; confirm the category's structural precondition (e.g. `UTR_TRUNCATED_MAPPING` requires the normalized narration UTR to be a prefix of the candidate). Returns `VerificationResult(passed, computed_residual, failure_reason)`.

### 4.6 Confidence (M7)

```
confidence = 0.40 × verification_passed
           + 0.20 × (1 if |residual| <= tolerance else 0)
           + 0.20 × evidence_completeness_ratio
           + 0.15 × candidate_separation          # (best − runner_up), clipped to [0,1]
           + 0.05 × freshness_ok
```

Weights live in `config/thresholds.yaml`. Auto-confirm requires `confidence ≥ 0.80` **and** `verification_passed`. Confidence alone can never confirm.

### 4.7 Incremental state (M6)

SQLite. Tables: `files` (`content_hash`, `arrival_day INTEGER`, `arrival_seq INTEGER`), `exceptions`, `audit`, `resolutions` (`idempotency_key UNIQUE`), `watermarks`.

**No wall-clock columns anywhere in the state DB.** `ingested_at` is removed; ordering uses `(arrival_day, arrival_seq)`, both integers supplied by the caller. A `TIMESTAMP`/`DATETIME`/`REAL` column in this schema is a D2 violation and `test_state_db_schema_has_no_timing_columns` fails on one.

Day-N ingestion: hash the file → if seen, no-op; else load, re-evaluate **all OPEN exceptions** against the enlarged dataset, close any that now verify with an audit entry, then run matching on new rows. `idempotency_key = sha256(exception_id | resolution_type | evidence_ids_sorted)`. An `INSERT OR IGNORE` on that key makes replay a no-op.

### 4.8 Degradation (M10)

Model client failure (timeout, HTTP error, invalid JSON after 2 retries) → residuals marked `PENDING_AI_UNAVAILABLE`. Deterministic results are untouched, nothing is confirmed, the process does not crash, and the next run picks the cases back up.

### 4.9 Tally export (M9)

Confirmed exceptions only. XML built from a schema template, validated against the XSD, wrapped with `idempotency_key`. Dry-run writes to disk and never transmits. Re-export produces the same batch reference. README labels it: *schema-validated Tally-compatible XML; not tested against a live Tally instance.*

---

## 5. Generator design (M1) — frozen at Gate 2

**Build order inside the generator:** clean chains first → verify ground truth is correct → then layer noise. A generator that mislabels its own truth corrupts every downstream metric silently, so ground-truth verification comes before any noise work.

Three merchant profiles with different MDR rates, settlement cycles (T+1 / T+2 / T+3), and refund rates.

| Noise type | Tuned against? |
|---|---|
| Truncated UTR in narration | Yes |
| Missing UTR entirely | Yes |
| Merchant-name variants in narration | Yes |
| Mixed amount formats | Yes |
| Duplicate ledger rows | Yes |
| Partial captures | Yes |
| Delayed settlement files | Yes |
| Out-of-order file arrival | Yes |
| Genuinely unexplainable rows | Yes |
| **Garbled narration with transposed characters** | **WITHHELD** |
| **Split settlement across two batches** | **WITHHELD** |

Withheld types are generated from day one but never used during engine development, and are reported separately as the unknown-unknowns result.

Ground truth is written to a **separate file** (`truth_<seed>.json`) that the engine never reads. `truth.py` runs a self-check before writing: every injected error is recoverable in principle, and every clean chain balances to the cent.

---

### 5.1 Generator freeze artifacts

`GENERATOR_MANIFEST.json` at repo root, written at M1F **after** the freeze commit:

```json
{"generator_commit": "<real sha>", "seeds": {"dev": 42, "holdout": 999},
 "calendar_version": "calendar_v1", "table_count": 6}
```

Truth files carry `generator_commit: null` — the hash cannot exist before the commit that freezes the generator, and truth files are never rewritten afterwards. Eval output reports the truth-file seed **and** the manifest's `generator_commit`. `test_no_literal_placeholders` fails on `GENERATOR_COMMIT`, `<hash>`, `TODO`, `XXX` or `TO BE MEASURED` anywhere outside this SDD and the build prompts (both of which use them only in prohibitions).

---

## 6. Configuration

`config/mdr_rates.yaml` — per profile, per method rate + GST rate.
`config/calendar_v1.yaml` — `version: calendar_v1`, working days, synthetic holiday list, T+N per method. The version string is recorded in `ReconciliationResult.calendar_version`.
`config/thresholds.yaml` — fuzzy accept 0.85, separation 0.15, rounding tolerance ₹1.00, confidence weights, auto-confirm 0.80, false-match budget 1%, gross-exposure false-match budget ₹500, cost budget ₹50/1k rows.

All loaded into frozen dataclasses. No magic numbers in code.

---

## 7. Test architecture

```
        /  Golden / snapshot  \      byte-compare full pipeline output
       /   Integration         \     multi-module, real SQLite, replay LLM
      /      Unit               \    pure functions, hostile inputs
```

| Layer | Count target | Runtime |
|---|---|---|
| Unit | ~120 | < 5s |
| Integration | ~25 | < 20s |
| Golden | ~6 | < 15s |
| Property-based | ~10 | < 10s |

`make test` must run in under 60 seconds with zero network calls.

**LLM in tests:** `fixtures/llm/<hash>.json` keyed by `sha256(prompt)`. `ReplayLLMClient` looks up the hash; a miss raises loudly rather than falling back to the network. `RealLLMClient.__init__` raises if `PYTEST_CURRENT_TEST` is set.

---

## 8. Makefile targets

```make
gen         # python -m gen.generate --seed 42 --out data/
gen-holdout # python -m gen.generate --seed 999 --out data/holdout/ --include-withheld
test        # pytest -q                       (no network, deterministic)
eval:                                # explicit contract — same paths in README
	python -m eval.run_eval \
	  --data data/holdout \
	  --truth data/holdout/truth_999.json \
	  --baselines all \
	  --out reports/eval
eval-ai     # real model; the experiment, not a test
ui          # streamlit run settlesense/ui/app.py
check       # ruff + mypy + determinism guard tests

# Golden files are IMMUTABLE by default. Regenerating them is the easiest way to
# make a real regression disappear, so it requires an explicit, awkward opt-in
# and is never reachable from test/check/eval.
golden-accept:                       # SETTLESENSE_ACCEPT_GOLDEN=1 make golden-accept
	@test "$$SETTLESENSE_ACCEPT_GOLDEN" = "1" || \
	  (echo "REFUSED. Regenerating goldens hides regressions."; \
	   echo "If a golden SHOULD change, state why in the commit message, then:"; \
	   echo "  SETTLESENSE_ACCEPT_GOLDEN=1 make golden-accept"; exit 1)
	pytest tests/golden --update-golden
bench       # python -m eval.bench --sizes 500,5000,25000,100000 --out reports/bench.md
```

### 8.1 Throughput instrumentation (M5a) — telemetry is NOT a result

The earlier design put timings inside the pipeline result and then stripped them before golden comparison. That was a patch over a design error. **Wall-clock data never enters the business result at all.** Two separate objects, two separate return values, two separate destinations.

```python
# settlesense/types.py
@dataclass(frozen=True)
class ReconciliationResult:
    """The business result. Serialized, hashed, compared, goldened.
    Contains NO wall-clock data of any kind — no durations, no timestamps,
    no memory figures. Adding a timing field here is a D6 violation."""
    cases: tuple[CaseOutcome, ...]             # Population A, sorted by case_id
    batch_links: tuple[BatchLinkOutcome, ...]  # Population B, sorted by batch_id
    row_variances: tuple[RowVarianceOutcome, ...]  # Population C, sorted by row_id
    exceptions: tuple[Exception_, ...]         # sorted by exception_id
    calendar_version: str
    config_hash: str

@dataclass(frozen=True)
class BatchLinkOutcome:
    """Population B. One per settlement batch. Batch-count denominator."""
    batch_id: str
    status: ExceptionStatus
    bank_row_id: str | None       # None when unlinked
    batch_net_total: Money
    linked_amount: Money | None   # bank credit amount when linked
    variance: Money | None        # batch_net_total - linked_amount
    resolved_by: str | None       # DETERMINISTIC | AI_VERIFIED | HUMAN
    confidence: Decimal | None

@dataclass(frozen=True)
class RowVarianceOutcome:
    """Population C. A variance whose subject is neither a payment nor a batch:
    a duplicate ledger row, an orphan bank credit. Row-count denominator."""
    row_id: str
    source_table: str                  # ledger_rows | bank_rows
    status: ExceptionStatus
    category: str | None
    amount: Money | None

# settlesense/core/telemetry.py  — a SEPARATE module, never imported by types.py
@dataclass(frozen=True)
class StageTiming:
    stage: str
    seconds: float          # float is fine HERE: telemetry is never compared or hashed
    records_in: int
    records_out: int

@dataclass(frozen=True)
class RunTelemetry:
    timings: tuple[StageTiming, ...]
    peak_rss_bytes: int
    machine: MachineSpec

def run_pipeline(...) -> tuple[ReconciliationResult, RunTelemetry]:
    """Two return values. Callers that persist or compare results take [0] only."""
```

**Separation rules, each with a test:**

| Rule | Test |
|---|---|
| `ReconciliationResult` has no float, duration, or timestamp field anywhere in its transitive type graph | `test_result_has_no_wallclock` walks annotations recursively |
| Result IDs, content hashes, `__eq__`, and serialization read **only** `ReconciliationResult` | `test_hash_ignores_telemetry` — two runs with wildly different timings hash identically |
| Golden files serialize `ReconciliationResult` only; the comparator has **no strip step** | `test_golden_serializer_signature` asserts the serializer accepts `ReconciliationResult` and rejects a tuple |
| `settlesense/types.py` never imports `core/telemetry.py` | import-graph test |
| Telemetry is written to `reports/`, never to the state DB | `test_state_db_schema_has_no_timing_columns` |

`eval/bench.py` runs the deterministic pipeline at each size with **no model calls**, three repetitions, reporting the **median** — never the best run. Peak memory via `tracemalloc`. Machine spec captured automatically into `reports/bench.md`. The AI stage is benchmarked separately, in seconds *and* rupees against residual count.

`reports/bench.md` is committed and linked from the README results section.

---

## 9. Module → gate map

| Module | Gate | Deadline |
|---|---|---|
| M0 skeleton, config, Makefile | 0 | 22 Aug |
| M1 generator | 1 | 23 Aug |
| M1F freeze + commit hash | 2 | 24 Aug |
| M2 normalize · M3 deterministic · M4 fuzzy UTR | 3 | 27 Aug |
| M5 eval harness + 3 baselines | 4 | 29 Aug |
| **M5a throughput bench (`make bench`)** | 4 | 29 Aug |
| M6 incremental state | 5 | 31 Aug |
| M7 hypothesis + verifier + confidence | 6 | 1 Sep |
| M8 evidence queue | 7 | 2 Sep |
| M9 export · M10 outage · packaging | 8 | 3 Sep |
| **M11 cash-position panel (optional)** | 8 | 3 Sep |
| Video + final eval run | 9 | 4 Sep |

Gates 1–5 are protected. Gates 7–9 absorb slippage: M8 compresses to a plain HTML table, M7 can run on a sampled residual set, M9, M10 and M11 are cuttable.

**M5a is not cuttable** despite sitting late — throughput is named first in the track bar, and the bench script is roughly ninety minutes of work once the deterministic pipeline exists. **M11 is cuttable without regret**; it is a read-only view over state the engine already holds.

# Abstain — Product Design Document (PDD)

**Version:** 1.0
**Date:** 22 August 2026
**Submission deadline:** 5 September 2026
**Track:** Razorpay AI Buildathon — Track 04, AI Finance Controller

---

## 1. One-line definition

Abstain is a verified settlement-reconciliation agent that resolves the computable deterministically, investigates only the genuinely ambiguous with a constrained language model, abstains when evidence is insufficient, and — if the optional export module is built — emits only the accounting entries it can prove.

---

## 2. Problem statement

A merchant's money passes through four systems that never agree on identifiers, timing, or amounts:

| System | Grain | Identifier | Amount |
|---|---|---|---|
| Order / invoice ledger | One row per order | `order_id`, `invoice_no` | Gross |
| Payment gateway | One row per payment | `payment_id` | Gross, captured |
| Settlement report | Signed lines — payment, refund (`settlement_id`) — grouped into batches (`batch_id`) | `settlement_id` (row), `batch_id` + `utr` (batch) | Net settlement amount, after fees, GST, refunds and supported deductions |
| Bank statement | One row per credit | Free-text narration containing a possibly-truncated UTR | Net |

A ₹1,000 card payment arrives in the bank as ₹976.40 after 2% MDR and 18% GST on that MDR. Multiply by hundreds of orders, add refunds that net off against unrelated batches, T+2 timing that splits a month-end, duplicate ledger rows, and bank narrations that truncate the UTR — and the finance team is left with an exception queue they work by hand.

The gap is specific and documented: the gateway tells you *what it settled*; it does not close your books. Order-level unpacking of a lumped net credit is left to the merchant, which is why an entire category of third-party reconciliation tooling exists.

---

## 3. Product thesis (non-negotiable)

> Abstain measures whether a constrained language model adds value **beyond** a strong deterministic reconciliation engine, and deploys AI only where the held-out evidence justifies it.

This is a measurement project as much as a product. The deterministic engine is built to its strongest reasonable form first. The AI layer must earn each category it is allowed to touch. **Zero AI uplift is a valid, publishable outcome and will be reported honestly.**

Two failure modes are explicitly forbidden:
1. Weakening the deterministic layer to create work for the model.
2. Inflating the "interpretive" taxonomy to make the model look necessary.

---

## 4. Users and jobs

| User | Job to be done | What Abstain gives them |
|---|---|---|
| Merchant finance associate | Close yesterday's settlement | A ranked exception queue with evidence, not a spreadsheet |
| Merchant accountant | Post journal entries correctly | Review verified reconciliation results and, **if M9 is included**, generate a dry-run, schema-validated Tally-compatible export |
| Finance lead | Know the unexplained exposure | Money-weighted unresolved value, not a row-count percentage |

Primary user for the demo is the finance associate. The evidence queue is the product surface.

---

## 5. Scope

Scope has **three levels**. This is the acceptance contract: cutting a conditional module does not make the submission incomplete, and must never be described as though it does.

### Level 1 — Core submission (required; never cut)

- Three-source ingestion: settlement report, bank statement, order/invoice ledger (six internal tables)
- Independent adversarial generator, frozen at a recorded commit
- Deterministic engine: normalization, exact matching, tolerance windows, fee arithmetic, timing calendar, duplicate detection, **fuzzy UTR resolution**
- Constrained AI hypothesis generation for residual interpretive cases only
- Deterministic verification of every hypothesis before confirmation
- Verification-derived confidence and explicit abstention
- Held-out evaluation with all baselines, measured on the residual set
- Throughput benchmark (`make bench`)
- Incremental state across Day 1 → Day 2 with replay safety
- Evidence queue showing the honest unresolved exception list

### Level 2 — Conditional enhancements (built only if every protected gate has passed)

| Module | Cut order |
|---|---|
| M9 dry-run Tally-compatible XML export | 2nd |
| M10 model-outage degradation demo | 3rd |
| M11 read-only cash-position panel | **1st — cut first** |
| Day 3 (beyond Day 1 → Day 2) | 4th |
| Split-settlement subset-sum | 5th |

Cutting any Level 2 item is a planned outcome, not a failure. The README states which were built and which were not, without apology.

### Level 3 — Explicitly out of scope

Chargeback and dispute handling, natural-language finance Q&A, live Tally/Zoho API integration, autonomous posting to a real ledger, tax advice, forecasting, payment routing, voice, multi-currency, and coverage of every conceivable settlement exception.

---

## 6. Variance taxonomy and the AI boundary

The taxonomy is **closed**. The model may only emit a category from this list.

### 6.1 Deterministically derivable — never sent to the model

| Category | Resolution rule |
|---|---|
| `MDR_FEE` | `fee = round(gross × rate[method], 2)` from a configured rate table |
| `GST_ON_FEE` | `gst = round(fee × 0.18, 2)` |
| `ROUNDING_DIFFERENCE` | Residual within ±₹1.00 after all other components |
| `DUPLICATE_CONFIRMED` | Byte-identical row content on a distinct source line — an ingestion artefact, decidable by rules |
| `T_PLUS_N_TIMING` | Settlement date falls outside the period per a configured working-day calendar |
| `REFUND_OFFSET` | Exact-amount refund matched to a known `refund_id` |
| `PARTIAL_CAPTURE` | Captured amount < authorised amount, both known |

> **Deduction categories are not variance categories.** `MDR_FEE`, `GST_ON_FEE` and `REFUND_OFFSET` describe *components of `expected_net`*, not unexplained differences. They are computed on every case and are never emitted as a variance. A coverage assertion of the form "every taxonomy category appears in truth" is therefore wrong. Coverage is asserted over **variance-producing categories only**, enumerated explicitly as `taxonomy.VARIANCE_CATEGORIES`.

### 6.2 Genuinely interpretive — eligible for the AI layer

| Category | Why rules may struggle |
|---|---|
| `UTR_TRUNCATED_MAPPING` | Narration holds a partial UTR matching more than one batch |
| `UTR_MISSING_MAPPING` | No UTR at all; must infer batch from amount, date, and merchant-name variant |
| `DUPLICATE_CANDIDATE` | Same amount and customer, different `order_id` — genuine repeat purchase or a double entry? Distinct from `DUPLICATE_CONFIRMED`, which rules decide. |
| `SPLIT_SETTLEMENT` | One payment spread across two batches |
| `MISSING_VS_LATE_CREDIT` | Absent today, or arriving on Day 3? |
| `UNEXPLAINED` | Terminal state — routes to human review |

**Honest note carried into the README:** several rows in 6.2 may fall to deterministic methods once fuzzy matching, candidate scoring, and subset-sum are built well. If that happens, the category moves to 6.1 and the AI's eligible surface shrinks. That migration is a result, not a setback, and it will be reported.

---

## 7. Safety model

### 7.1 The core invariant

> No exception is ever confirmed on the strength of model output alone. A hypothesis is confirmed only when a deterministic verifier independently recomputes its arithmetic assertion and validates every referenced evidence row.

### 7.2 Confidence

Confidence is **never** the model's self-report. It is computed by the verification layer from:

| Signal | Contribution |
|---|---|
| Verification outcome | Exactly one hypothesis passes → high; several pass → ambiguous |
| Arithmetic residual | Unexplained remainder is zero or within tolerance |
| Evidence completeness | All required source rows present and linked |
| Candidate separation | Best candidate clearly beats the runner-up |
| Data freshness | Expected files have arrived per watermark |

### 7.3 Pre-declared enablement rule

AI-assisted automatic confirmation is enabled **per category** only if, on the held-out set, it improves useful residual coverage or explanation precision **and**:

- residual false-match rate < **1%**
- gross-exposure false-match value (₹ expected gross) < **₹500 per evaluation run**
- cost < **₹50 per 1,000 processed rows**

Fail any condition → that category stays deterministic-only or human-review-only.

The ₹500 figure is a project safety threshold for a synthetic evaluation, not a claim about acceptable production loss. Real deployment would require risk, compliance, and business-owner sign-off on all thresholds.

### 7.4 Asymmetry of errors

A missed match costs an analyst minutes. A **false match silently closes a real financial loss and corrupts the books.** False-match rate is therefore the primary safety metric, reported before coverage, and abstention is always preferred to a guess.

### 7.5 Bounded external action

**If M9 is built (Level 2, conditional)**, one action leaves the system: a dry-run, schema-validated, Tally-compatible journal-entry batch built only from confirmed results, carrying an idempotency key. **If M9 is cut, the core submission is read-only and produces no external action** — a complete outcome, not a gap. Re-running produces the same batch reference and no second posting candidate. Import and approval remain with the merchant's accountant. Labelled precisely: *schema-validated Tally-compatible XML; not tested against a live Tally instance.*

---

## 8. Evaluation design

### 8.1 Independence

The data generator is a **separate code path** sharing no normalization, identifier-parsing, date-handling, or matching utilities with the engine. It is completed and committed **before** engine development begins, and the commit hash is published:

> The adversarial generator was frozen at the commit recorded in `GENERATOR_MANIFEST.json` before engine development began. Results reproduce with new seeds. Truth files carry `generator_commit: null`; the real hash lives only in the manifest, written after the freeze.

Two noise types are withheld from engine tuning and reported separately as an **unknown-unknowns** test.

### 8.2 Baselines

| Baseline | Definition |
|---|---|
| Naive | Amount + date-window matching only |
| Deterministic-only | Full normalization, rules, fuzzy UTR, fee arithmetic, duplicate detection — no AI |
| Strong LLM-only | Same normalized records, candidate retrieval, sensible chunking, carefully tuned structured prompt. Tuned in good faith; not a strawman. |
| Abstain | Deterministic engine + verified hypothesis loop + abstention |

### 8.3 The headline metric

Not the diluted whole-pipeline score. The residual-set sentence:

> Of the **N** exceptions the deterministic engine could not resolve, the verified hypothesis loop correctly explained **X**, abstained on **Y**, and false-matched **Z**.

### 8.4 Full metric set

**Every money metric names its basis in its own label — no exceptions.** Three bases exist and none is comparable to another: **expected gross** (gross exposure), **expected net** (cash), **batch net total** (Population B). A metric that says only "money-weighted" is ambiguous and is not permitted. The README results table repeats the basis in every row.

**All metrics below are Population A — computed over `ReconciliationCase`, one per captured payment.** Batch↔bank link metrics are Population B, reported in a separate table with their own denominator. The two are never averaged or combined. See SDD 3.1.

| Metric (Population A) | Formula / purpose |
|---|---|
| Case match rate (case count) | Confirmed cases ÷ total cases |
| Deterministic residual count | Cases unresolved after rules — the surface available to AI |
| Residual explanation precision | Correct AI explanations ÷ confirmed AI explanations |
| Residual abstention rate | Abstained ÷ total residuals |
| **Residual false-match rate** | Incorrect confirmations ÷ total confirmations — *primary safety metric* |
| Gross-exposure match rate (₹ expected gross) | Prevents small cases hiding large unresolved exposure |
| Gross-exposure false-match value (₹ expected gross) | ₹ incorrectly confirmed |
| Expected-net cash reconciled (₹ expected net) | Cash actually accounted for |
| Unresolved expected-net cash (₹ expected net) | Cash that cannot be explained |
| Cost per correctly resolved exception | Model cost ÷ correct AI explanations |
| **Throughput** | Records/second, reported per stage — *see 8.4a, first-class deliverable* |
| Analyst minutes saved | Verified automatic confirmations × assumed review minutes (**derived estimate**, assumption stated) |
| Evidence coverage | Decisions traceable to source rows |
| Unknown-unknown performance | Held-out noise types |
| Export idempotency rate **(M9 only)** | Repeat runs create no duplicate batches. Omitted from the core results table when M9 is cut. |

**Population B — batch↔bank links, reported separately:**

| Metric | Formula |
|---|---|
| Batch link rate (batch count) | Batches linked to a bank credit ÷ total batches |
| Batch false-link rate (batch count) | Incorrect links ÷ confirmed links |
| Batch-net false-link value (₹ batch net total) | ₹ incorrectly linked |
| Batch-net linked value (₹ batch net total) | Uses `batch.net_total` — never comparable to case `expected_gross` |

A batch↔bank failure propagates into Population A only through the cases inside that batch, each counted once.

### 8.4a Throughput — first-class deliverable

The track bar names three things and lists throughput **first**: *"Throughput plus measured accuracy plus an honest exception list."* It is therefore reported as a deliverable in its own right, not a footnote in the metrics table.

**Required output — a scaling table, run at four sizes:**

| Records | Deterministic (rec/s) | Residual count | AI stage (s) | Total wall-clock | Peak RSS |
|---|---|---|---|---|---|
| 500 | | | | | |
| 5,000 | | | | | |
| 25,000 | | | | | |
| 100,000 | | | | | |

**Rules:**

- Timing is per stage — ingest, normalize, each deterministic pass, AI residual loop, verification, persistence — so the reader can see where time goes and that the model is *not* on the hot path.
- The deterministic engine must run **without any model calls**, and that number is the headline throughput figure. It should be thousands of records per second.
- Report AI-stage cost separately in both seconds and rupees, because it scales with residual count, not row count. This is the point: a well-built deterministic layer makes the expensive stage small.
- Timings come from a committed benchmark script (`make bench`), not hand-timed runs, and the machine spec is stated.
- If 100,000 rows is infeasible in the time available, report up to 25,000 and say so plainly rather than extrapolating.

**Why this matters beyond compliance:** throughput is where the deterministic-first thesis pays off visibly. If the model sat in the main loop, the system would be orders of magnitude slower and costlier. The scaling table is the evidence for that claim.

### 8.4b Cash position — optional, cheap, directly on-brief

The track headline is *"Run the books and the cash position."* Abstain covers the books through verified journal entries. Cash position is a small addition derivable from data already in the pipeline, with no new inputs:

| Output | Derivation |
|---|---|
| Settled to bank | Expected-net value of cases with a confirmed bank credit |
| In flight | Captured payments with no bank credit yet, within the expected T+N window |
| Expected landing dates | In-flight amounts bucketed by working-day calendar |
| At risk | Past expected date, unmatched — the `MISSING_VS_LATE_CREDIT` set |
| Unexplained | Expected-net value of unresolved and abstained cases |

**Basis — an accounting requirement, not a presentation detail.** All four buckets are measured in **expected net settlement value**:

```
expected_net = gross − fee − tax − refunds
```

Chargebacks are out of v1 scope, so there is no dispute term (SDD §3.0). The buckets reconcile against `cash_basis_total = Σ expected_net`. They do **not** sum to captured gross: bank credits are net of fees, GST and refunds, so a four-bucket claim against gross would be false by exactly the deduction total. `deductions_reference = gross − expected_net` — fees, GST and refunds — is reported alongside so a reader can tie back to gross.

**Two money bases exist and are always labelled.** `expected_gross` measures gross exposure; `expected_net` measures cash. No metric switches basis silently.

One panel in the evidence queue, with the basis stated on screen:

> *Cash view — expected net settlement value after configured deductions*
> *₹4.2L settled · ₹1.8L in flight, landing 26–28 Aug · ₹12,400 at risk · ₹3,100 unexplained*

**Scope discipline:** this is a *read-only view over existing reconciliation state*, roughly two hours of work, and it must not become a forecasting model. **M11 is optional and must not block submission. It may begin only after M5a, M6, M7 and M8 pass. If any protected gate slips, M11 is skipped.** If it threatens any gate, it is cut without hesitation — the reconciliation loop is the submission, and cash position is a bonus that happens to echo the track's own headline.

### 8.4c Dates and calendar

All simulated dates are in **2026**. Four date concepts stay distinct throughout — `event_date`, `file_content_date`, `arrival_day` (integer), and the injected `as_of_date` (SDD 4.1a). The working-day calendar is versioned (`config/calendar_v1.yaml`) and its version string is recorded in every result.

The v1 holiday set is **synthetic for the simulated timeline**, not the RBI schedule. Stated plainly in `LIMITATIONS.md`: the evaluation measures whether the engine handles a holiday correctly, not whether it knows India's actual holiday list. Swapping in real dates is a config change requiring no code.

### 8.5 Dataset shape

Minimum **5,000 records** across **three distinct merchant profiles** with different MDR rates, settlement cycles, and refund rates — so results read as generalising rather than tuned to one distribution.

The track floor is a 50+ record batch. Exceeding it by two orders of magnitude is itself part of the argument — *"one cherry-picked match proves nothing"* is answered by volume, held-out seeds, and baselines together.

---

## 9. Demonstration script (5 minutes)

| Time | Beat |
|---|---|
| 0:00–0:45 | The problem, with one real number: a ₹1,000 order arriving as ₹976.40, times a thousand |
| 0:45–2:15 | Day 1 → Day 2 → Day 3 in the evidence queue. Day 2's file auto-closes a pending exception with an audit note. Replay the same file; show no duplicate resolution. |
| 2:15–3:15 | Architecture, spoken as a thesis: *deterministic resolves the computable, the model only proposes, a verifier decides* |
| 3:15–4:15 | Day 3 ambiguity → abstention with competing candidates shown. Then kill the model API mid-run: no crash, nothing wrongly resolved, residuals marked `PENDING_AI_UNAVAILABLE`, safe resume. |
| 4:15–5:00 | Results table, throughput scaling row, the residual-set sentence, the unresolved exception list on screen, and the honest limitation |

**Three things must be visibly on screen at some point, because they are the track's stated bar:** a throughput number, a measured accuracy number, and the actual list of exceptions the system could not resolve. Not described — shown.

---

## 10. Success criteria

**Must have:** frozen independent generator; deterministic engine; three baselines with real measured numbers on a held-out set; residual-set AI measurement; abstention with false-match measurement; Day 1 → Day 2 incremental state with replay safety; **throughput scaling table from `make bench`**; **a rendered unresolved-exception list**; evidence queue; results-first README with `make eval` and `LIMITATIONS.md`.

**Level 2 conditional (see §5):** Day 3, model-outage degradation, Tally export, unknown-unknown reporting.

**Optional enhancement (M11):** the read-only cash-position panel, derived from the same reconciled expected-net basis.

**Claim discipline.** The core submission is the verified settlement-close loop with Population A/B metrics, throughput, the unresolved exception list, and incremental state. If M11 ships it is presented as an extension. If M11 is cut, the README and video say the project closes settlement reconciliation and exposes unresolved cash exposure — and make **no** claim of a cash forecast. A cut optional module is not an incomplete submission, and must not be described as one.

**Explicitly not required:** any particular AI uplift number.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Generator ground truth is itself buggy — corrupts every metric silently | Ground-truth self-check suite before freezing; invariant assertions in the generator |
| Circular evaluation | Independent code path, frozen commit, withheld noise types |
| AI contributes ~0% and reads as "AI didn't do much" at an AI buildathon | Pre-written framing; the finding *is* the contribution |
| Over-scoping past the deadline | Gate calendar with hard cut-lines; Gates 1–5 protected, Gates 7–9 are the shock absorbers |
| Demo is visually flat | Evidence queue promoted ahead of export and outage work |

---

## 12. Non-goals restated

Abstain is not an AI CFO, not a chatbot, not a payment router, and not a tax adviser. It closes one loop — daily settlement close — and measures itself honestly while doing it.

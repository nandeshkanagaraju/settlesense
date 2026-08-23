"""M1 part C - noise injectors.

INDEPENDENT PATH. Nothing here imports from settlesense/.

Each injector is a pure function `(rows, rng, rate) -> NoiseResult`. It never
mutates its input: it returns a new dataset plus the annotations describing
exactly what it did, so truth is derived from what was done rather than guessed
from the result.

THREE KINDS OF NOISE, and the distinction is load-bearing:

  Presentation  truncate_utr, drop_utr, merchant_name_variants,
                mixed_amount_formats, out_of_order_arrival, garbled_narration
                Changes how a row is RENDERED or DELIVERED. The money is
                untouched, so every conservation identity must still hold
                EXACTLY after these run.

  Structural    partial_captures, duplicate_ledger_rows, delayed_settlement,
                split_settlement
                Changes rows or amounts. Batch totals and bank credits are
                re-SUMMED afterwards, never left stale - a batch whose total no
                longer equals its lines is a generator bug, not noise.

  Conservation- unexplainable
  breaking      The ONLY injector permitted to break conservation, and only by
                an amount the truth ledger predicts to the cent.

Two noise types are WITHHELD from engine development and reported separately as
the unknown-unknowns result. They are generated from day one but gated behind
--include-withheld.
"""

from __future__ import annotations

import hashlib
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Final, Protocol

from gen.lifecycle import (
    BankRow,
    CleanDataset,
    SettlementBatch,
    SettlementLine,
    SettlementLineType,
    bank_txn_id_for,
    money,
)
from gen.profiles import MerchantProfile
from gen.truth import VarianceCategory

__all__ = [
    "NOISE_ORDER",
    "NOISE_REGISTRY",
    "NoiseAnnotation",
    "NoiseLedger",
    "NoiseRates",
    "NoiseResult",
    "apply_noise",
    "delayed_settlement",
    "drop_utr",
    "duplicate_ledger_rows",
    "garbled_narration",
    "merchant_name_variants",
    "mixed_amount_formats",
    "out_of_order_arrival",
    "partial_captures",
    "split_settlement",
    "truncate_utr",
    "unexplainable",
]

ZERO: Final[Decimal] = Decimal("0.00")
_BASIS: Final[int] = 10_000


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseAnnotation:
    """One injected error, recorded at the grain it actually affects.

    `target_kind` keeps Population A and Population B apart (D11). A truncated
    UTR damages a BATCH link, not a case; it reaches cases only through the
    batch they sit in, and the eval joins case.batch_ids to find it. Recording
    it against a case here would merge the two populations at the source.
    """

    noise_type: str
    target_kind: str  # case | batch_link | ledger_row | bank_row
    target_id: str
    category: VarianceCategory | None
    resolvable: bool
    detail: str


@dataclass(frozen=True)
class NoiseResult:
    """What one injector produced. `rows` is a whole dataset, never mutated."""

    rows: CleanDataset
    annotations: tuple[NoiseAnnotation, ...] = ()
    # (table, row_id, column) -> literal text to write instead of the Decimal
    format_overrides: Mapping[tuple[str, str, str], str] | None = None
    # (table, row_id) -> the day bundle this row is DELIVERED in, if not its own
    day_overrides: Mapping[tuple[str, str], int] | None = None


@dataclass(frozen=True)
class NoiseRates:
    """Per-type injection rates. Decimal, never float (D12 in spirit).

    GRAIN MATTERS when reading these. Case-grain injectors sample ~5,000 rows;
    batch-grain injectors sample ~40. The same numeric rate therefore means very
    different absolute counts, and a batch-grain rate set as though it were
    case-grain silently produces a category that never occurs.
    """

    # batch-grain (~40 targets)
    truncate_utr: Decimal = Decimal("0.25")
    drop_utr: Decimal = Decimal("0.12")
    merchant_name_variants: Decimal = Decimal("0.30")
    unexplainable: Decimal = Decimal("0.06")
    garbled_narration: Decimal = Decimal("0.12")  # WITHHELD
    # case- and row-grain (~5,000 targets)
    mixed_amount_formats: Decimal = Decimal("0.08")
    duplicate_ledger_rows: Decimal = Decimal("0.010")
    partial_captures: Decimal = Decimal("0.020")
    delayed_settlement: Decimal = Decimal("0.015")
    out_of_order_arrival: Decimal = Decimal("0.04")
    split_settlement: Decimal = Decimal("0.010")  # WITHHELD

    def for_type(self, name: str) -> Decimal:
        if not hasattr(self, name):
            raise KeyError(f"no rate configured for noise type {name!r}")
        value: Decimal = getattr(self, name)
        return value


@dataclass(frozen=True)
class NoiseLedger:
    """Everything the noise pipeline did, and everything it therefore owes."""

    annotations: tuple[NoiseAnnotation, ...]
    counts: Mapping[str, int]
    format_overrides: Mapping[tuple[str, str, str], str]
    day_overrides: Mapping[tuple[str, str], int]
    withheld_applied: bool

    def of_type(self, noise_type: str) -> tuple[NoiseAnnotation, ...]:
        return tuple(a for a in self.annotations if a.noise_type == noise_type)

    def by_target(self, kind: str) -> Mapping[str, tuple[NoiseAnnotation, ...]]:
        out: dict[str, list[NoiseAnnotation]] = {}
        for annotation in self.annotations:
            if annotation.target_kind == kind:
                out.setdefault(annotation.target_id, []).append(annotation)
        return {key: tuple(value) for key, value in sorted(out.items())}

    def orphan_bank_txn_ids(self) -> frozenset[str]:
        """Bank credits injected with no batch behind them."""
        return frozenset(
            a.target_id
            for a in self.annotations
            if a.noise_type == "unexplainable" and a.detail.startswith("orphan_credit")
        )

    def unbanked_batch_ids(self) -> frozenset[str]:
        """Batches whose bank credit was withheld and never arrives."""
        return frozenset(
            a.target_id
            for a in self.annotations
            if a.noise_type == "unexplainable" and a.detail.startswith("missing_credit")
        )

    def duplicate_confirmed_order_ids(self) -> frozenset[str]:
        """Order IDs that legitimately appear twice: byte-identical ingest artefacts."""
        return frozenset(
            a.target_id
            for a in self.annotations
            if a.noise_type == "duplicate_ledger_rows"
            and a.category is VarianceCategory.DUPLICATE_CONFIRMED
        )

    def split_payment_ids(self) -> frozenset[str]:
        return frozenset(
            a.target_id for a in self.annotations if a.noise_type == "split_settlement"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noise_id(prefix: str, *parts: object) -> str:
    """Deterministic ID for a row noise invented (D10). Never uuid4."""
    canonical = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(f"noise|{prefix}|{canonical}".encode()).hexdigest()
    return f"{prefix}_{digest[:12].upper()}"


def _pick(rng: random.Random, population: Sequence[object], rate: Decimal) -> list[int]:
    """Choose indices at `rate`, resolved in integer basis points, sorted (D4).

    Comparing against rng.random() would put a float on a selection path and
    make the chosen set depend on binary rounding near the threshold.
    """
    if rate <= ZERO or not population:
        return []
    threshold = int((rate * _BASIS).to_integral_value(rounding=ROUND_HALF_UP))
    return [index for index in range(len(population)) if rng.randrange(_BASIS) < threshold]


def _utr_in(narration: str) -> str | None:
    for token in narration.split():
        if len(token) == 16 and token.isalnum() and token.isupper():
            return token
    return None


def _resum_batches(
    lines: Sequence[SettlementLine],
    batches: Sequence[SettlementBatch],
    bank_rows: Sequence[BankRow],
) -> tuple[tuple[SettlementBatch, ...], tuple[BankRow, ...]]:
    """Re-derive every batch total from its lines, and every credit from its batch.

    Structural noise MUST route through here. A batch whose net_total no longer
    equals the signed sum of its lines is not noise, it is a broken generator -
    and it would be indistinguishable from the arithmetic bug the engine exists
    to catch.
    """
    totals: dict[str, Decimal] = {}
    for line in lines:
        totals[line.batch_id] = totals.get(line.batch_id, ZERO) + line.net

    new_batches = tuple(
        replace(batch, net_total=money(totals.get(batch.batch_id, ZERO))) for batch in batches
    )
    # Keyed on the constructive link, not the narration: by the time structural
    # noise re-sums, a narration may already have lost its UTR, and a credit
    # that silently kept a stale amount would be a balance bug, not noise.
    total_by_txn = {bank_txn_id_for(b.batch_id): b.net_total for b in new_batches}
    new_bank = tuple(
        replace(row, amount=total_by_txn[row.bank_txn_id])
        if row.bank_txn_id in total_by_txn
        else row  # an orphan credit has no batch to follow
        for row in bank_rows
    )
    return new_batches, new_bank


def _sorted_dataset(dataset: CleanDataset) -> CleanDataset:
    """Re-sort every list by its stated key (D4)."""
    return CleanDataset(
        chains=tuple(sorted(dataset.chains, key=lambda c: c.payment.payment_id)),
        ledger_rows=tuple(sorted(dataset.ledger_rows, key=lambda r: (r.order_id, r.invoice_no))),
        payment_rows=tuple(sorted(dataset.payment_rows, key=lambda r: r.payment_id)),
        refund_rows=tuple(sorted(dataset.refund_rows, key=lambda r: r.refund_id)),
        settlement_lines=tuple(sorted(dataset.settlement_lines, key=lambda r: r.settlement_id)),
        batches=tuple(sorted(dataset.batches, key=lambda r: r.batch_id)),
        bank_rows=tuple(sorted(dataset.bank_rows, key=lambda r: r.bank_txn_id)),
    )


# ---------------------------------------------------------------------------
# PRESENTATION NOISE - money untouched, conservation must still hold exactly
# ---------------------------------------------------------------------------


def truncate_utr(
    rows: CleanDataset,
    rng: random.Random,
    rate: Decimal,
    *,
    profiles: Mapping[str, MerchantProfile] | None = None,
) -> NoiseResult:
    """Keep a random 6-10 character prefix of the UTR in the narration.

    Population B: this damages the batch-to-bank link, not any case.
    """
    bank = list(rows.bank_rows)
    utr_to_batch = {batch.utr: batch.batch_id for batch in rows.batches}
    annotations: list[NoiseAnnotation] = []

    for index in _pick(rng, bank, rate):
        row = bank[index]
        utr = _utr_in(row.narration)
        if utr is None or utr not in utr_to_batch:
            continue
        keep = rng.randint(6, 10)
        prefix = utr[:keep]
        bank[index] = replace(row, narration=row.narration.replace(utr, prefix, 1))
        # A prefix that still uniquely identifies one batch is resolvable by
        # fuzzy matching; one shared by two batches is genuinely ambiguous.
        shared = sum(1 for other in utr_to_batch if other.startswith(prefix))
        annotations.append(
            NoiseAnnotation(
                "truncate_utr",
                "batch_link",
                utr_to_batch[utr],
                VarianceCategory.UTR_TRUNCATED_MAPPING,
                resolvable=shared == 1,
                detail=(
                    f"narration keeps {keep} of 16 UTR chars; {shared} batch(es) share the prefix"
                ),
            )
        )
    return NoiseResult(replace(rows, bank_rows=tuple(bank)), tuple(annotations))


def drop_utr(
    rows: CleanDataset,
    rng: random.Random,
    rate: Decimal,
    *,
    profiles: Mapping[str, MerchantProfile] | None = None,
) -> NoiseResult:
    """Remove the UTR from the narration entirely."""
    bank = list(rows.bank_rows)
    utr_to_batch = {batch.utr: batch.batch_id for batch in rows.batches}
    amounts: dict[Decimal, int] = {}
    for batch in rows.batches:
        amounts[batch.net_total] = amounts.get(batch.net_total, 0) + 1
    annotations: list[NoiseAnnotation] = []

    for index in _pick(rng, bank, rate):
        row = bank[index]
        utr = _utr_in(row.narration)
        if utr is None or utr not in utr_to_batch:
            continue
        stripped = " ".join(token for token in row.narration.split() if token != utr)
        bank[index] = replace(row, narration=stripped)
        # With no UTR the batch must be inferred from amount, date and merchant.
        # A unique amount makes that decidable; a shared one does not.
        annotations.append(
            NoiseAnnotation(
                "drop_utr",
                "batch_link",
                utr_to_batch[utr],
                VarianceCategory.UTR_MISSING_MAPPING,
                resolvable=amounts.get(row.amount, 0) == 1,
                detail=(
                    f"no UTR in narration; {amounts.get(row.amount, 0)} batch(es) share the amount"
                ),
            )
        )
    return NoiseResult(replace(rows, bank_rows=tuple(bank)), tuple(annotations))


def merchant_name_variants(
    rows: CleanDataset,
    rng: random.Random,
    rate: Decimal,
    *,
    profiles: Mapping[str, MerchantProfile] | None = None,
) -> NoiseResult:
    """Vary the merchant name in the narration: abbreviated, suffixed, unspaced.

    On its own this changes nothing decidable - the UTR is the join key. It
    bites when it lands on a row that also lost its UTR, which is exactly the
    compounding the engine has to survive.
    """
    bank = list(rows.bank_rows)
    txn_to_batch = {bank_txn_id_for(b.batch_id): b.batch_id for b in rows.batches}
    if profiles is None:
        raise ValueError("merchant_name_variants needs the profile table")
    # Match the merchant name EXACTLY, from the known set. Defining it as
    # "every token that is not NEFT or the UTR" breaks the moment an earlier
    # injector truncates the UTR: the leftover prefix stops being recognised,
    # gets absorbed into the name, and the unspacing variant fuses the two into
    # one token - destroying a prefix that was supposed to stay recoverable.
    merchant_names = sorted(
        (profile.merchant_name for profile in profiles.values()), key=len, reverse=True
    )
    annotations: list[NoiseAnnotation] = []

    for index in _pick(rng, bank, rate):
        row = bank[index]
        batch_id = txn_to_batch.get(row.bank_txn_id)
        if batch_id is None:
            continue
        name = next((m for m in merchant_names if m in row.narration), None)
        if name is None:
            continue
        variant = _name_variant(name, rng)
        bank[index] = replace(row, narration=row.narration.replace(name, variant, 1))
        annotations.append(
            NoiseAnnotation(
                "merchant_name_variants",
                "batch_link",
                batch_id,
                # No category on its own: with the UTR intact the link is exact.
                category=None,
                resolvable=True,
                detail=f"merchant rendered as {variant!r} instead of {name!r}",
            )
        )
    return NoiseResult(replace(rows, bank_rows=tuple(bank)), tuple(annotations))


def _name_variant(name: str, rng: random.Random) -> str:
    words = name.split()
    style = rng.randrange(3)
    if style == 0 and len(words) > 1:  # "AURORA RETAIL" -> "AURORA RTL"
        last = words[-1]
        squeezed = last[0] + "".join(c for c in last[1:] if c not in "AEIOU")
        return " ".join([*words[:-1], squeezed[:4]])
    if style == 1:  # -> "AURORA RETAIL PVT"
        return f"{name} PVT" if not name.endswith("PVT LTD") else name.replace(" PVT LTD", "")
    return name.replace(" ", "")  # -> "AURORARETAIL"


def mixed_amount_formats(
    rows: CleanDataset,
    rng: random.Random,
    rate: Decimal,
    *,
    profiles: Mapping[str, MerchantProfile] | None = None,
) -> NoiseResult:
    """Render some amounts as "1,234.00" / "1234.0" / "1234" / " 1234.00 ".

    A rendering override, not a row change: the Decimal is untouched, only the
    CSV text differs. A format is only ever chosen if it is LOSSLESS for that
    value - emitting "1234" for 1234.56 would be corruption, not formatting.
    Negative nets also get the parenthesised debit form the normalizer must
    handle (SDD 4.1).
    """
    targets: list[tuple[str, str, Decimal]] = []
    targets += [("ledger", r.order_id, r.gross) for r in rows.ledger_rows]
    targets += [("payments", r.payment_id, r.captured) for r in rows.payment_rows]
    targets += [("refunds", r.refund_id, r.amount) for r in rows.refund_rows]
    targets += [("settlements", line.settlement_id, line.net) for line in rows.settlement_lines]
    targets += [("bank", r.bank_txn_id, r.amount) for r in rows.bank_rows]
    column = {
        "ledger": "gross",
        "payments": "captured",
        "refunds": "amount",
        "settlements": "net",
        "bank": "amount",
    }

    overrides: dict[tuple[str, str, str], str] = {}
    annotations: list[NoiseAnnotation] = []
    for index in _pick(rng, targets, rate):
        table, row_id, value = targets[index]
        text = _format_variant(value, rng)
        if text is None:
            continue
        overrides[(table, row_id, column[table])] = text
        annotations.append(
            NoiseAnnotation(
                "mixed_amount_formats",
                "ledger_row",
                row_id,
                category=None,  # lossless rendering: nothing to explain
                resolvable=True,
                detail=f"{table}.{column[table]} rendered as {text!r} for {value}",
            )
        )
    return NoiseResult(rows, tuple(annotations), format_overrides=overrides)


def _format_variant(value: Decimal, rng: random.Random) -> str | None:
    """A LOSSLESS alternative rendering, or None if none applies."""
    plain = f"{value:.2f}"
    choices: list[str] = [f"{value:,.2f}", f" {plain} "]
    if value < ZERO:
        choices.append(f"({-value:,.2f})")  # parenthesised debit
    cents = abs(int((value * 100).to_integral_value())) % 100
    if cents % 10 == 0:
        choices.append(f"{value:.1f}")  # 1234.50 -> "1234.5"
    if cents == 0:
        choices.append(f"{value:.0f}")  # 1234.00 -> "1234"
    return choices[rng.randrange(len(choices))]


def out_of_order_arrival(
    rows: CleanDataset,
    rng: random.Random,
    rate: Decimal,
    *,
    profiles: Mapping[str, MerchantProfile] | None = None,
) -> NoiseResult:
    """Deliver some rows in a file labelled LATER than their content date.

    A delivery-order fault, not a data fault: `arrival_day` stops agreeing with
    `event_date`, which is the whole reason SDD 4.1a keeps them as separate
    fields with separate types.
    """
    targets: list[tuple[str, str]] = []
    targets += [("settlements", line.settlement_id) for line in rows.settlement_lines]
    targets += [("bank", row.bank_txn_id) for row in rows.bank_rows]
    targets += [("refunds", row.refund_id) for row in rows.refund_rows]

    day_overrides: dict[tuple[str, str], int] = {}
    annotations: list[NoiseAnnotation] = []
    for index in _pick(rng, targets, rate):
        table, row_id = targets[index]
        lag = rng.randint(1, 2)
        day_overrides[(table, row_id)] = lag
        annotations.append(
            NoiseAnnotation(
                "out_of_order_arrival",
                "ledger_row",
                row_id,
                category=None,  # arrives late, but arrives intact
                resolvable=True,
                detail=f"{table} row delivered {lag} bundle(s) after its content date",
            )
        )
    return NoiseResult(rows, tuple(annotations), day_overrides=day_overrides)


def garbled_narration(
    rows: CleanDataset,
    rng: random.Random,
    rate: Decimal,
    *,
    profiles: Mapping[str, MerchantProfile] | None = None,
) -> NoiseResult:
    """WITHHELD. Transpose two adjacent characters inside the UTR.

    Categorised UTR_TRUNCATED_MAPPING because the taxonomy is closed and this
    presents as a UTR-mapping failure. `noise_type` is what the unknown-unknowns
    report slices on - the point is that the engine was never tuned for it.
    """
    bank = list(rows.bank_rows)
    utr_to_batch = {batch.utr: batch.batch_id for batch in rows.batches}
    annotations: list[NoiseAnnotation] = []

    for index in _pick(rng, bank, rate):
        row = bank[index]
        utr = _utr_in(row.narration)
        if utr is None or utr not in utr_to_batch or len(utr) < 2:
            continue
        at = rng.randrange(len(utr) - 1)
        if utr[at] == utr[at + 1]:
            continue  # transposing equal characters is a no-op, not noise
        garbled = utr[:at] + utr[at + 1] + utr[at] + utr[at + 2 :]
        bank[index] = replace(row, narration=row.narration.replace(utr, garbled, 1))
        annotations.append(
            NoiseAnnotation(
                "garbled_narration",
                "batch_link",
                utr_to_batch[utr],
                VarianceCategory.UTR_TRUNCATED_MAPPING,
                resolvable=True,  # edit distance 2 from the true UTR
                detail=f"UTR characters at {at},{at + 1} transposed",
            )
        )
    return NoiseResult(replace(rows, bank_rows=tuple(bank)), tuple(annotations))


# ---------------------------------------------------------------------------
# STRUCTURAL NOISE - rows change, so batches and credits are re-summed
# ---------------------------------------------------------------------------


def partial_captures(
    rows: CleanDataset,
    rng: random.Random,
    rate: Decimal,
    *,
    profiles: Mapping[str, MerchantProfile] | None = None,
) -> NoiseResult:
    """Capture less than was authorised, and settle the captured amount.

    Both figures stay known, which is what makes PARTIAL_CAPTURE deterministic
    (PDD 6.1) rather than interpretive. Fee and tax are recomputed on the
    captured amount, not scaled, so the arithmetic still closes to the cent.
    """
    if profiles is None:
        raise ValueError("partial_captures needs the profile rate table")
    chains = list(rows.chains)
    lines = {line.settlement_id: line for line in rows.settlement_lines}
    annotations: list[NoiseAnnotation] = []

    # Refunded chains are excluded: reducing the capture below a refund already
    # issued would make the refund exceed what was taken.
    eligible = [index for index, chain in enumerate(chains) if chain.refund is None]
    for pick in _pick(rng, eligible, rate):
        index = eligible[pick]
        chain = chains[index]
        line = lines.get(chain.payment_line.settlement_id)
        if line is None:
            continue
        authorized = chain.payment.authorized
        # 30%..90% of the authorised amount, in whole paise.
        paise = int((authorized * 100).to_integral_value())
        captured = money(Decimal(rng.randint((paise * 3) // 10, (paise * 9) // 10)) / 100)
        if captured >= authorized or captured <= ZERO:
            continue

        profile_rate = profiles[chain.profile_name].rate_for(chain.payment.method)
        fee = money(captured * profile_rate)
        tax = money(fee * _GST)
        net = money(captured - fee - tax)

        payment = replace(chain.payment, captured=captured, status="partial")
        pending = replace(chain.payment_line, gross=captured, fee=fee, tax=tax, net=net)
        chains[index] = replace(chain, payment=payment, payment_line=pending)
        lines[line.settlement_id] = replace(line, gross=captured, fee=fee, tax=tax, net=net)

        annotations.append(
            NoiseAnnotation(
                "partial_captures",
                "case",
                chain.payment.payment_id,
                VarianceCategory.PARTIAL_CAPTURE,
                resolvable=True,
                detail=f"captured {captured} of authorised {authorized}",
            )
        )

    payments = tuple(chain.payment for chain in chains)
    new_lines = tuple(lines.values())
    batches, bank = _resum_batches(new_lines, rows.batches, rows.bank_rows)
    return NoiseResult(
        _sorted_dataset(
            replace(
                rows,
                chains=tuple(chains),
                payment_rows=payments,
                settlement_lines=new_lines,
                batches=batches,
                bank_rows=bank,
            )
        ),
        tuple(annotations),
    )


def duplicate_ledger_rows(
    rows: CleanDataset,
    rng: random.Random,
    rate: Decimal,
    *,
    profiles: Mapping[str, MerchantProfile] | None = None,
) -> NoiseResult:
    """Two shapes that look identical on amount alone and are not the same thing.

    DUPLICATE_CONFIRMED - a byte-identical ledger row on a distinct source line.
    An ingestion artefact, decidable by rule (P7a): same order_id, no second
    payment behind it.

    DUPLICATE_CANDIDATE - a genuine repeat purchase. Same customer, same amount,
    DIFFERENT order_id, and a real payment that really settles (P7b). Whether
    this is a double entry or a real second sale is interpretive.

    Neither is separable by amount - that is the point. What separates them is
    order_id identity and whether money actually moved.

    Note on grain: SDD 3.3 gives LedgerRow a `order_date: date` with no time
    component, so "3+ hours later" is expressed as the same or a later DATE.
    A same-date repeat is deliberately included; it is the harder case.
    """
    ledger = list(rows.ledger_rows)
    chains = list(rows.chains)
    payments = list(rows.payment_rows)
    lines = list(rows.settlement_lines)
    annotations: list[NoiseAnnotation] = []

    line_by_payment = {
        line.payment_id: line
        for line in rows.settlement_lines
        if line.line_type is SettlementLineType.PAYMENT
    }
    sources = list(rows.chains)

    for pick in _pick(rng, sources, rate):
        chain = sources[pick]
        origin = chain.ledger
        if rng.randrange(2) == 0:
            # (a) byte-identical ingest duplicate. Same order_id, no new payment.
            ledger.append(origin)
            annotations.append(
                NoiseAnnotation(
                    "duplicate_ledger_rows",
                    "ledger_row",
                    origin.order_id,
                    VarianceCategory.DUPLICATE_CONFIRMED,
                    resolvable=True,
                    detail="byte-identical ledger row on a distinct source line",
                )
            )
            continue

        # (b) genuine repeat purchase: new order and payment, same customer and
        # amount, settling into the SAME batch as the original.
        template = line_by_payment.get(chain.payment.payment_id)
        if template is None:
            continue
        suffix = len([a for a in annotations if a.noise_type == "duplicate_ledger_rows"])
        new_order = _noise_id("ORD", origin.order_id, "repeat", suffix)
        new_payment_id = _noise_id("PAY", chain.payment.payment_id, "repeat", suffix)
        new_line_id = _noise_id("SET", template.settlement_id, "repeat", suffix)
        # SAME order_date as the original. LedgerRow has date granularity only
        # (SDD 3.3), so "3+ hours later" is the same date - and a same-date
        # repeat is the harder case anyway: date cannot separate it either.
        # It also keeps the repeat settling in the batch it belongs to, rather
        # than in one that closed before the purchase happened.
        repeat_date = origin.order_date

        repeat_ledger = replace(
            origin,
            order_id=new_order,
            invoice_no=f"{origin.invoice_no}-R{suffix:03d}",
            order_date=repeat_date,
        )
        repeat_payment = replace(
            chain.payment,
            payment_id=new_payment_id,
            order_id=new_order,
            captured_at=repeat_date,
            authorized=chain.payment.captured,  # a fresh FULL capture: cloning a
            status="captured",  # partially-captured payment would smuggle in an
        )  # unannotated PARTIAL_CAPTURE under a new id
        repeat_pending = replace(
            chain.payment_line, settlement_id=new_line_id, payment_id=new_payment_id
        )
        repeat_line = replace(
            template, settlement_id=new_line_id, payment_id=new_payment_id, refund_id=None
        )

        ledger.append(repeat_ledger)
        payments.append(repeat_payment)
        lines.append(repeat_line)
        chains.append(
            replace(
                chain,
                ledger=repeat_ledger,
                payment=repeat_payment,
                refund=None,
                payment_line=repeat_pending,
                refund_line=None,
            )
        )
        annotations.append(
            NoiseAnnotation(
                "duplicate_ledger_rows",
                "case",
                new_payment_id,
                VarianceCategory.DUPLICATE_CANDIDATE,
                resolvable=True,
                detail=(
                    f"genuine repeat purchase by {origin.customer_id} for {origin.gross} "
                    f"on {repeat_date.isoformat()}; original order {origin.order_id}"
                ),
            )
        )

    batches, bank = _resum_batches(lines, rows.batches, rows.bank_rows)
    return NoiseResult(
        _sorted_dataset(
            replace(
                rows,
                chains=tuple(chains),
                ledger_rows=tuple(ledger),
                payment_rows=tuple(payments),
                settlement_lines=tuple(lines),
                batches=batches,
                bank_rows=bank,
            )
        ),
        tuple(annotations),
    )


def delayed_settlement(
    rows: CleanDataset,
    rng: random.Random,
    rate: Decimal,
    *,
    profiles: Mapping[str, MerchantProfile] | None = None,
) -> NoiseResult:
    """Settle a payment one or two batches later than its own capture day.

    The line MOVES to the later batch rather than merely changing its date: a
    line whose settled date disagrees with its batch is a broken generator, and
    both batches are re-summed so the arithmetic still closes.
    """
    profile_of_payment = {c.payment.payment_id: c.profile_name for c in rows.chains}
    dates_by_profile: dict[str, list[date]] = {}
    batch_at: dict[tuple[str, date], SettlementBatch] = {}
    for batch in rows.batches:
        members = [line for line in rows.settlement_lines if line.batch_id == batch.batch_id]
        names = {profile_of_payment.get(line.payment_id) for line in members}
        names.discard(None)
        if len(names) != 1:
            continue
        name = str(next(iter(names)))
        dates_by_profile.setdefault(name, []).append(batch.settled_event_date)
        batch_at[(name, batch.settled_event_date)] = batch
    for name in dates_by_profile:
        dates_by_profile[name] = sorted(dates_by_profile[name])

    lines = list(rows.settlement_lines)
    annotations: list[NoiseAnnotation] = []
    movable = [
        index for index, line in enumerate(lines) if line.line_type is SettlementLineType.PAYMENT
    ]
    for pick in _pick(rng, movable, rate):
        index = movable[pick]
        line = lines[index]
        owner = profile_of_payment.get(line.payment_id)
        if owner is None:
            continue
        available = dates_by_profile.get(owner, [])
        if line.settled_event_date not in available:
            continue
        position = available.index(line.settled_event_date)
        lag = rng.randint(1, 2)
        if position + lag >= len(available):
            continue  # no later batch exists; a delay off the end is 'unexplainable'
        target_date = available[position + lag]
        target = batch_at[(owner, target_date)]
        lines[index] = replace(line, batch_id=target.batch_id, settled_event_date=target_date)
        annotations.append(
            NoiseAnnotation(
                "delayed_settlement",
                "case",
                line.payment_id,
                VarianceCategory.T_PLUS_N_TIMING,
                resolvable=True,
                detail=(
                    f"settled {target_date.isoformat()} in batch {target.batch_id}, "
                    f"{lag} batch(es) after {line.settled_event_date.isoformat()}"
                ),
            )
        )

    batches, bank = _resum_batches(lines, rows.batches, rows.bank_rows)
    return NoiseResult(
        _sorted_dataset(
            replace(rows, settlement_lines=tuple(lines), batches=batches, bank_rows=bank)
        ),
        tuple(annotations),
    )


def split_settlement(
    rows: CleanDataset,
    rng: random.Random,
    rate: Decimal,
    *,
    profiles: Mapping[str, MerchantProfile] | None = None,
) -> NoiseResult:
    """WITHHELD. Split one payment's settlement across two batches.

    Produces N>=2 PAYMENT lines for one payment - the structural condition for
    SPLIT_SETTLEMENT (SDD 3.2). It stays ONE ReconciliationCase: two cases would
    double-count the denominator and make the match rate depend on how the
    gateway happened to batch the payout.

    gross, fee and tax are split so each part is internally consistent AND the
    parts sum back exactly. Fee on the second part is the remainder rather than
    a second rounded product, so no paisa is created or lost.
    """
    if profiles is None:
        raise ValueError("split_settlement needs the profile rate table")
    method_of_payment = {p.payment_id: p.method for p in rows.payment_rows}
    profile_of_payment = {c.payment.payment_id: c.profile_name for c in rows.chains}
    dates_by_profile: dict[str, list[tuple[date, str]]] = {}
    for batch in rows.batches:
        members = [line for line in rows.settlement_lines if line.batch_id == batch.batch_id]
        names = {profile_of_payment.get(line.payment_id) for line in members}
        names.discard(None)
        if len(names) == 1:
            dates_by_profile.setdefault(str(next(iter(names))), []).append(
                (batch.settled_event_date, batch.batch_id)
            )
    for name in dates_by_profile:
        dates_by_profile[name] = sorted(dates_by_profile[name])

    lines = list(rows.settlement_lines)
    chains = list(rows.chains)
    chain_index = {chain.payment.payment_id: i for i, chain in enumerate(chains)}
    annotations: list[NoiseAnnotation] = []
    splittable = [
        index
        for index, line in enumerate(lines)
        if line.line_type is SettlementLineType.PAYMENT and line.gross >= Decimal("200.00")
    ]
    for pick in _pick(rng, splittable, rate):
        index = splittable[pick]
        line = lines[index]
        owner = profile_of_payment.get(line.payment_id)
        if owner is None:
            continue
        available = dates_by_profile.get(owner, [])
        position = next((i for i, (_, bid) in enumerate(available) if bid == line.batch_id), None)
        if position is None or position + 1 >= len(available):
            continue
        next_date, next_batch = available[position + 1]

        paise = int((line.gross * 100).to_integral_value())
        first_paise = rng.randint((paise * 3) // 10, (paise * 7) // 10)
        gross_a = money(Decimal(first_paise) / 100)
        gross_b = money(line.gross - gross_a)
        method = method_of_payment.get(line.payment_id)
        if method is None:
            continue
        # Each part is priced on ITS OWN gross, not given the remainder. A
        # remainder would make part B's fee disagree with the rate applied to
        # part B - the exact arithmetic the engine recomputes to verify a line.
        # The two parts may therefore total a paisa away from the unsplit fee,
        # which is correct: the gateway charged twice.
        profile_rate = profiles[owner].rate_for(method)
        fee_a = money(gross_a * profile_rate)
        tax_a = money(fee_a * _GST)
        fee_b = money(gross_b * profile_rate)
        tax_b = money(fee_b * _GST)
        net_a = money(gross_a - fee_a - tax_a)
        net_b = money(gross_b - fee_b - tax_b)
        if net_a <= ZERO or net_b <= ZERO:
            continue

        second_id = _noise_id("SET", line.settlement_id, "split")
        lines[index] = replace(line, gross=gross_a, fee=fee_a, tax=tax_a, net=net_a)
        lines.append(
            replace(
                line,
                settlement_id=second_id,
                batch_id=next_batch,
                settled_event_date=next_date,
                gross=gross_b,
                fee=fee_b,
                tax=tax_b,
                net=net_b,
            )
        )
        # The case must learn it now has two payment lines, or its expected_net,
        # fee and tax stay derived from part A alone and conservation breaks.
        position_in_chains = chain_index.get(line.payment_id)
        if position_in_chains is not None:
            chain = chains[position_in_chains]
            chains[position_in_chains] = replace(
                chain,
                payment_line=replace(
                    chain.payment_line, gross=gross_a, fee=fee_a, tax=tax_a, net=net_a
                ),
                extra_payment_lines=(
                    *chain.extra_payment_lines,
                    replace(
                        chain.payment_line,
                        settlement_id=second_id,
                        settled_event_date=next_date,
                        gross=gross_b,
                        fee=fee_b,
                        tax=tax_b,
                        net=net_b,
                    ),
                ),
            )
        annotations.append(
            NoiseAnnotation(
                "split_settlement",
                "case",
                line.payment_id,
                VarianceCategory.SPLIT_SETTLEMENT,
                resolvable=True,
                detail=(
                    f"net split {net_a} + {net_b} across batches {line.batch_id} and {next_batch}"
                ),
            )
        )

    batches, bank = _resum_batches(lines, rows.batches, rows.bank_rows)
    return NoiseResult(
        _sorted_dataset(
            replace(
                rows,
                chains=tuple(chains),
                settlement_lines=tuple(lines),
                batches=batches,
                bank_rows=bank,
            )
        ),
        tuple(annotations),
    )


# ---------------------------------------------------------------------------
# CONSERVATION-BREAKING NOISE - runs last, and owes the truth file an exact figure
# ---------------------------------------------------------------------------


def unexplainable(
    rows: CleanDataset,
    rng: random.Random,
    rate: Decimal,
    *,
    profiles: Mapping[str, MerchantProfile] | None = None,
) -> NoiseResult:
    """A credit with no settlement behind it, and a settlement whose credit never comes.

    The ONLY injector allowed to break conservation. Both shapes are terminal:
    resolvable_in_principle is False, and the honest outcome is the exception
    list, not an explanation.
    """
    bank = list(rows.bank_rows)
    annotations: list[NoiseAnnotation] = []

    # (a) an orphan bank credit: money arrives that no batch accounts for.
    for index in _pick(rng, rows.batches, rate):
        batch = rows.batches[index]
        txn_id = _noise_id("BNK", batch.batch_id, "orphan")
        amount = money(batch.net_total / Decimal(rng.randint(3, 11)))
        if amount <= ZERO:
            continue
        bank.append(
            BankRow(
                bank_txn_id=txn_id,
                value_date=batch.settled_event_date,
                amount=amount,
                narration=f"NEFT {_noise_id('UTR', txn_id)[4:]} UNIDENTIFIED CREDIT",
                direction="credit",
            )
        )
        annotations.append(
            NoiseAnnotation(
                "unexplainable",
                "bank_row",
                txn_id,
                VarianceCategory.UNEXPLAINED,
                resolvable=False,
                detail=f"orphan_credit {amount} with no settlement behind it",
            )
        )

    # (b) a batch whose bank credit never arrives.
    txn_to_batch = {bank_txn_id_for(b.batch_id): b.batch_id for b in rows.batches}
    survivors: list[BankRow] = []
    withheld = set(_pick(rng, bank, rate))
    for index, row in enumerate(bank):
        batch_id = txn_to_batch.get(row.bank_txn_id)
        if index in withheld and batch_id is not None:
            annotations.append(
                NoiseAnnotation(
                    "unexplainable",
                    "batch_link",
                    batch_id,
                    VarianceCategory.MISSING_VS_LATE_CREDIT,
                    resolvable=False,
                    detail=f"missing_credit {row.amount} never arrives for batch {batch_id}",
                )
            )
            continue
        survivors.append(row)

    return NoiseResult(
        _sorted_dataset(replace(rows, bank_rows=tuple(survivors))), tuple(annotations)
    )


# ---------------------------------------------------------------------------
# Registry and pipeline
# ---------------------------------------------------------------------------


class NoiseFn(Protocol):
    """Every injector is `(rows, rng, rate)` plus an optional profile table.

    The three positional parameters are the contract. `profiles` is keyword-only
    and defaulted so injectors that need the rate table can recompute fees
    honestly, without a module-level mutable holding it - which would make these
    functions impure and their output order-dependent.
    """

    def __call__(
        self,
        rows: CleanDataset,
        rng: random.Random,
        rate: Decimal,
        *,
        profiles: Mapping[str, MerchantProfile] | None = ...,
    ) -> NoiseResult: ...


# name -> (function, is_withheld)
NOISE_REGISTRY: Final[Mapping[str, tuple[NoiseFn, bool]]] = {
    "truncate_utr": (truncate_utr, False),
    "drop_utr": (drop_utr, False),
    "merchant_name_variants": (merchant_name_variants, False),
    "mixed_amount_formats": (mixed_amount_formats, False),
    "duplicate_ledger_rows": (duplicate_ledger_rows, False),
    "partial_captures": (partial_captures, False),
    "delayed_settlement": (delayed_settlement, False),
    "out_of_order_arrival": (out_of_order_arrival, False),
    "unexplainable": (unexplainable, False),
    "garbled_narration": (garbled_narration, True),
    "split_settlement": (split_settlement, True),
}

# Order is not cosmetic. Structural injectors run before presentation ones so
# re-summing sees final amounts, and `unexplainable` runs LAST because it is the
# only one permitted to leave conservation broken.
NOISE_ORDER: Final[tuple[str, ...]] = (
    "partial_captures",
    "duplicate_ledger_rows",
    "delayed_settlement",
    "split_settlement",
    "truncate_utr",
    "drop_utr",
    "merchant_name_variants",
    "garbled_narration",
    "mixed_amount_formats",
    "out_of_order_arrival",
    "unexplainable",
)

_GST: Final[Decimal] = Decimal("0.18")


def apply_noise(
    dataset: CleanDataset,
    rng: random.Random,
    profiles: Mapping[str, MerchantProfile],
    rates: NoiseRates,
    *,
    include_withheld: bool,
) -> tuple[CleanDataset, NoiseLedger]:
    """Run every enabled injector in NOISE_ORDER. Counts go to stderr."""
    current = dataset
    annotations: list[NoiseAnnotation] = []
    format_overrides: dict[tuple[str, str, str], str] = {}
    day_overrides: dict[tuple[str, str], int] = {}
    counts: dict[str, int] = {}

    for name in NOISE_ORDER:
        function, is_withheld = NOISE_REGISTRY[name]
        if is_withheld and not include_withheld:
            counts[name] = 0
            continue
        result = function(current, rng, rates.for_type(name), profiles=profiles)
        current = result.rows
        annotations.extend(result.annotations)
        if result.format_overrides:
            format_overrides.update(result.format_overrides)
        if result.day_overrides:
            day_overrides.update(result.day_overrides)
        counts[name] = len(result.annotations)

    # `unexplainable` runs last and can withhold a bank credit that an earlier
    # presentation injector already annotated. A delivery note about a row that
    # no longer exists is moot, so it is dropped. Only category-free annotations
    # are prunable: a dangling annotation that CARRIES truth is a real error and
    # must still fail the self-check.
    surviving = (
        {r.order_id for r in current.ledger_rows}
        | {r.payment_id for r in current.payment_rows}
        | {r.refund_id for r in current.refund_rows}
        | {line.settlement_id for line in current.settlement_lines}
        | {b.batch_id for b in current.batches}
        | {r.bank_txn_id for r in current.bank_rows}
    )
    kept = [
        a
        for a in annotations
        if a.category is not None or a.target_kind != "ledger_row" or a.target_id in surviving
    ]
    pruned = len(annotations) - len(kept)
    if pruned:
        counts["_pruned_targets_removed_later"] = pruned

    ledger = NoiseLedger(
        annotations=tuple(sorted(kept, key=lambda a: (a.noise_type, a.target_kind, a.target_id))),
        counts=dict(sorted(counts.items())),
        format_overrides=format_overrides,
        day_overrides=day_overrides,
        withheld_applied=include_withheld,
    )
    _log_counts(ledger)
    return _sorted_dataset(current), ledger


def _log_counts(ledger: NoiseLedger) -> None:
    print("noise injected:", file=sys.stderr)
    for name in NOISE_ORDER:
        _, is_withheld = NOISE_REGISTRY[name]
        count = ledger.counts.get(name, 0)
        tag = "  WITHHELD" if is_withheld else ""
        state = "" if count or not is_withheld else "  (gated: --include-withheld not set)"
        print(f"  {name:<24} {count:>6}{tag}{state}", file=sys.stderr)
    print(f"  {'-' * 24} {'-' * 6}", file=sys.stderr)
    print(f"  {'total annotations':<24} {len(ledger.annotations):>6}", file=sys.stderr)

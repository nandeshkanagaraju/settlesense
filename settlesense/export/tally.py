"""M9 - Dry-run, schema-validated, idempotent Tally-compatible XML export.

WHAT THIS PRODUCES. Schema-validated Tally-compatible XML; not tested against a
live Tally instance. The XSD in this directory is one written here and bundled
here, so a malformed document fails at `to_xml` rather than at somebody's import
prompt - but validating against a schema you wrote yourself catches your own
defects and says nothing about whether a third-party product accepts the result.
That claim needs a live instance and there has not been one.

DRY RUN MEANS NO TRANSMISSION, STRUCTURALLY. Nothing here opens a socket, and
there is no code path that could: the only sink is a file. `test_m9_export.py`
monkeypatches `socket.socket` to raise and exports a real batch through it.

THE EXPORTER CANNOT DETECT ITS OWN WORST INPUT. It runs on CONFIRMED
exceptions, and CONFIRMED means "an explanation passed verification", not "the
explanation is right". The held-out run (seed 999, commit 0c44419) confirmed 52
split settlements under a plausible WRONG category - T_PLUS_N_TIMING 48 times,
PARTIAL_CAPTURE 4 times - and every one of them is locally consistent with the
evidence attached to it. No assertion available inside this module distinguishes
them from correct confirmations, because there is nothing here to distinguish
them BY.

So the answer is not a check, it is a disclosure. Every batch carries a
provenance header naming the dataset, the seed, the config hash and the MEASURED
residual false-match rate for that dataset, and the rate is a required argument
with no default: a batch whose rate is unknown does not export, it raises. A
journal entry carrying a wrong category is worse than no entry, and an
accountant importing this has to be able to see what the numbers rest on.

THE PROVENANCE IS INJECTED, NEVER COMPUTED HERE. The false-match rate is
truth-derived - `eval/metrics.py`, Population A - and `settlesense/` may not
reach truth or read `reports/`. It arrives as an argument the same way `as_of`
does, supplied by `eval/run_export.py`, which is on the side of the fence
allowed to read an evaluation artifact.

AND IT IS INSIDE THE IDEMPOTENCY KEY. SDD 4.7's key was over the exception set
and the batch date alone. Two batches with the same confirmed set under
different configs would then write the same filename with different content,
which turns "exporting twice does not duplicate" from a guarantee into a
collision. Everything the header states is hashed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final

from lxml import etree

from settlesense.exceptions.store import ExceptionStore
from settlesense.exceptions.taxonomy import VarianceCategory
from settlesense.types import Exception_, ExceptionStatus, Money, money

__all__ = [
    "LEDGERS",
    "SCHEMA_PATH",
    "ExportError",
    "ExportProvenance",
    "JournalLine",
    "TallyBatch",
    "build_batch",
    "close_exported",
    "format_money",
    "to_xml",
    "validate",
    "write_dry_run",
]

SCHEMA_PATH: Final[Path] = Path(__file__).with_name("tally_voucher.xsd")

BASIS: Final[str] = (
    "schema-validated Tally-compatible XML; not tested against a live Tally instance"
)
"""The one sentence that must appear on the document itself.

In the header rather than a README only, because the file outlives the
repository it came from: a reviewer who receives one of these has no other way
to learn what it was and was not checked against.
"""

RECEIVABLE: Final[str] = "Razorpay Settlement Receivable"
"""The credit side of every entry here. Each explained variance says why the
gateway's settlement differed from the ledger's expectation, so the account
being relieved is always the same one; only the account absorbing the
difference changes."""


LEDGERS: Final[dict[VarianceCategory, str]] = {
    VarianceCategory.MDR_FEE: "Bank Charges - Payment Gateway",
    VarianceCategory.GST_ON_FEE: "Input GST - Payment Gateway",
    VarianceCategory.ROUNDING_DIFFERENCE: "Rounding Off",
    VarianceCategory.REFUND_OFFSET: "Refunds Payable",
    VarianceCategory.DUPLICATE_CONFIRMED: "Suspense - Duplicate Settlement",
    VarianceCategory.DUPLICATE_CANDIDATE: "Suspense - Duplicate Settlement",
    VarianceCategory.T_PLUS_N_TIMING: "Settlement In Transit",
    VarianceCategory.PARTIAL_CAPTURE: "Settlement In Transit",
    VarianceCategory.SPLIT_SETTLEMENT: "Settlement In Transit",
    VarianceCategory.UTR_TRUNCATED_MAPPING: "Settlement In Transit",
    VarianceCategory.UTR_MISSING_MAPPING: "Settlement In Transit",
    VarianceCategory.MISSING_VS_LATE_CREDIT: "Settlement In Transit",
}
"""Category -> the account that absorbs the difference. A CLOSED taxonomy.

UNEXPLAINED IS DELIBERATELY ABSENT, and its absence is the point rather than an
oversight. An unexplained variance has no journal entry by definition; posting
one to a suspense account would turn "we do not know what this is" into a
booked accounting fact. It raises, and a test asserts it raises rather than
falling through to a default.

Every other member of VARIANCE_CATEGORIES appears. `test_m9_export.py` asserts
the coverage by MEMBERSHIP - a category added to the taxonomy later must fail
here rather than inherit a neighbour's ledger, which is the defect shape that
already shipped once in the status style map (commit 2a744b0).
"""


class ExportError(RuntimeError):
    """Refusal to emit a document. Never a partial or best-effort export."""


def _decimal_or_raise(value: object, field: str) -> Decimal:
    """A runtime check the annotations cannot make on their own.

    TYPED `object` ON PURPOSE. Written against the declared type, mypy proves
    the branch unreachable and refuses it - correctly, for internal callers.
    But the provenance is built from an eval artifact read off disk, where a
    JSON number arrives as a float and nothing in the type system was involved.
    An unreachable guard is worse than no guard: it reads as protection and
    cannot fire. This one can, and `test_m9_export.py` fires it.
    """
    if isinstance(value, Decimal):
        return value
    raise ExportError(
        f"{field} is {type(value).__name__} {value!r}, not Decimal (D6). Binary "
        "floating point renders differently depending on how it was computed, and "
        "this figure goes on the face of an accounting document."
    )


@dataclass(frozen=True)
class ExportProvenance:
    """What the numbers in this batch rest on. Every field required.

    `residual_false_match_rate` is the MEASURED rate for the dataset the
    confirmations came from - the fraction of reconciliation cases the engine
    confirmed under a category truth disagrees with. On the dev seed that is
    0.000000; on the held-out seed it is 0.010456, over PDD 7.3's 1% budget.
    Both belong on the face of the document.
    """

    dataset: str
    seed: int
    config_hash: str
    residual_false_match_rate: Decimal

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ExportError("provenance.dataset is empty; name the dataset this came from")
        if not self.config_hash.strip():
            raise ExportError("provenance.config_hash is empty")
        if self.seed < 0:
            raise ExportError(f"provenance.seed is negative: {self.seed}")
        _decimal_or_raise(self.residual_false_match_rate, "residual_false_match_rate")
        if not Decimal("0") <= self.residual_false_match_rate <= Decimal("1"):
            raise ExportError(
                f"residual_false_match_rate {self.residual_false_match_rate} is not a "
                "rate in [0, 1]. A percentage passed as 1.05 would understate a "
                "1.0456% breach by two orders of magnitude."
            )

    @property
    def rate_text(self) -> str:
        """Six places, matching how the eval harness renders the same figure."""
        return f"{self.residual_false_match_rate:.6f}"


@dataclass(frozen=True)
class JournalLine:
    """One explained variance as a two-legged journal entry."""

    exception_id: str
    category: str
    debit_ledger: str
    credit_ledger: str
    amount: Money
    narration: str


@dataclass(frozen=True)
class TallyBatch:
    """A batch of journal entries and the provenance they were produced under."""

    batch_date: date
    provenance: ExportProvenance
    lines: tuple[JournalLine, ...]
    idempotency_key: str
    cleared: tuple[str, ...] = ()
    """CONFIRMED exceptions with NO journal entry, because nothing is owed.

    ON THE FACE OF THE DOCUMENT, not swallowed. A batch of 16 vouchers built
    from 283 confirmed exceptions invites exactly one question, and a reader who
    cannot answer it has to assume 267 entries went missing. They did not: the
    variance CLEARED when later files arrived, and a variance that no longer
    exists has nothing to post. Counted, stated in the header, and never merged
    with the vouchers.
    """

    @property
    def exception_ids(self) -> tuple[str, ...]:
        return tuple(line.exception_id for line in self.lines)

    @property
    def total_debits(self) -> Money:
        return money(sum((line.amount for line in self.lines), Decimal("0")))

    @property
    def total_credits(self) -> Money:
        return self.total_debits

    @property
    def filename(self) -> str:
        return f"tally-{self.batch_date.isoformat()}-{self.idempotency_key[:16]}.xml"


def format_money(amount: Money) -> str:
    """Exactly two places, never scientific notation, never a float.

    `Decimal("1E+3")` formats as "1E+3" under `str()` and as "1000.00" under
    `:.2f`, and the first would validate as text while meaning nothing to an
    importer. The format spec is the guarantee.
    """
    return f"{_decimal_or_raise(amount, 'amount'):.2f}"


def _canonical(batch_date: date, provenance: ExportProvenance, exception_ids: list[str]) -> str:
    """Everything the header states, plus the set the batch acts on.

    SDD 4.7 hashes exception ids and the batch date. The provenance fields are
    hashed too because they are RENDERED: two batches differing only in config
    hash produce different documents, and a key that could not tell them apart
    would name both with the same filename.
    """
    return "|".join(
        [
            "tally-batch-v1",
            batch_date.isoformat(),
            provenance.dataset,
            str(provenance.seed),
            provenance.config_hash,
            provenance.rate_text,
            *sorted(exception_ids),
        ]
    )


def build_batch(
    confirmed: tuple[Exception_, ...],
    resolved_categories: Mapping[str, str | None],
    provenance: ExportProvenance,
    batch_date: date,
) -> TallyBatch:
    """CONFIRMED exceptions -> one batch. Anything unpostable raises or clears.

    `provenance` has no default. The signature is where "a batch whose rate is
    unknown does not export" is actually enforced; a keyword with a fallback
    would move the decision to whoever forgot to pass it.

    THE RESOLVING CATEGORY IS AN ARGUMENT, AND IT IS NOT `exception.category`.
    An exception stores the category recorded AT DETECTION, and on this dataset
    274 of 283 confirmed rows were opened as UNEXPLAINED on day 1 or 12 and
    explained by a later day's files. Posting on the detected category would
    book 274 vouchers for a variance nobody ever claimed existed. The M8 queue
    already learned this the hard way and split the column in two; the exporter
    reads the same second column, computed by the engine over the full dataset
    and passed in by the caller.

    MISSING IS NOT CLEARED. A subject the engine no longer reports at all is an
    anomaly and raises; a subject it reports with category None has no variance
    left and is recorded as cleared. Collapsing the two would let a dropped
    subject look like a resolved one - the same missing-versus-empty rule the
    ingest layer follows.
    """
    if not confirmed:
        raise ExportError(
            "no confirmed exceptions to export. An empty batch is not a no-op - it "
            "would write a document asserting that nothing needed posting, which is "
            "a different claim from not having run."
        )

    wrong_status = sorted(
        f"{exception.exception_id} is {exception.status.value}"
        for exception in confirmed
        if exception.status is not ExceptionStatus.CONFIRMED
    )
    if wrong_status:
        raise ExportError(
            f"{len(wrong_status)} exception(s) are not CONFIRMED: {wrong_status[:5]}. "
            "Only an explanation that passed verification may be posted; an "
            "ABSTAINED or OPEN row is one nobody has explained yet."
        )

    missing = sorted(
        exception.exception_id
        for exception in confirmed
        if exception.exception_id not in resolved_categories
    )
    if missing:
        raise ExportError(
            f"{len(missing)} confirmed exception(s) have no resolving category: "
            f"{missing[:5]}. Absent is not the same as cleared - the engine no "
            "longer reports these subjects at all, and posting them would be "
            "guessing at what they resolved to."
        )

    lines: list[JournalLine] = []
    cleared: list[str] = []
    for exception in confirmed:
        resolved = resolved_categories[exception.exception_id]
        if resolved is None:
            # NOTHING IS OWED. The variance existed when the exception opened
            # and did not survive the arrival of the rest of the files. There
            # is no entry to make, and inventing a zero-value voucher would put
            # a posting in the books for a discrepancy that turned out not to
            # be one.
            cleared.append(exception.exception_id)
            continue
        try:
            category = VarianceCategory(resolved)
        except ValueError as error:
            raise ExportError(
                f"{exception.exception_id} resolved to {resolved!r}, which is not in the taxonomy"
            ) from error
        ledger = LEDGERS.get(category)
        if ledger is None:
            raise ExportError(
                f"{exception.exception_id} resolved to {category.value}, which has no "
                "ledger mapping. UNEXPLAINED has none deliberately: posting an "
                "unexplained variance to a suspense account books a fact nobody "
                "established."
            )
        if exception.amount <= Decimal("0"):
            raise ExportError(
                f"{exception.exception_id} has amount {exception.amount}. A voucher "
                "for zero or a negative would balance trivially and post nothing."
            )
        lines.append(
            JournalLine(
                exception_id=exception.exception_id,
                category=category.value,
                debit_ledger=ledger,
                credit_ledger=RECEIVABLE,
                amount=exception.amount,
                narration=(
                    f"SettleSense {category.value} on {exception.exception_id}; "
                    f"{exception.reason or 'no reason recorded'}"
                ),
            )
        )

    if not lines:
        raise ExportError(
            f"all {len(cleared)} confirmed exception(s) cleared; there is nothing to "
            "post. That is a legitimate outcome and it is not a document - writing an "
            "empty envelope would assert that a batch was produced."
        )

    # SORTED BY (-amount, exception_id), the queue's order. Two runs over the
    # same set must emit byte-identical documents, and dict or set ordering
    # would make that depend on insertion history (D5).
    lines.sort(key=lambda line: (-line.amount, line.exception_id))
    canonical = _canonical(batch_date, provenance, [line.exception_id for line in lines])
    return TallyBatch(
        batch_date=batch_date,
        provenance=provenance,
        lines=tuple(lines),
        idempotency_key=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        cleared=tuple(sorted(cleared)),
    )


def _leg(parent: etree._Element, ledger: str, amount: Money, *, debit: bool) -> None:
    """One side of an entry. Tally: debit is negative, ISDEEMEDPOSITIVE=Yes."""
    entry = etree.SubElement(parent, "ALLLEDGERENTRIES.LIST")
    etree.SubElement(entry, "LEDGERNAME").text = ledger
    etree.SubElement(entry, "ISDEEMEDPOSITIVE").text = "Yes" if debit else "No"
    etree.SubElement(entry, "AMOUNT").text = format_money(-amount if debit else amount)


def to_xml(batch: TallyBatch) -> str:
    """The document, validated before it is returned.

    Validation happens HERE rather than in the caller, so there is no path that
    produces an unvalidated string. A caller who forgot to validate would still
    have a plausible-looking file.
    """
    envelope = etree.Element("ENVELOPE")
    header = etree.SubElement(envelope, "HEADER")
    etree.SubElement(header, "TALLYREQUEST").text = "Import Data"
    etree.SubElement(
        header,
        "SETTLESENSE.PROVENANCE",
        dataset=batch.provenance.dataset,
        seed=str(batch.provenance.seed),
        configHash=batch.provenance.config_hash,
        residualFalseMatchRate=batch.provenance.rate_text,
        batchDate=batch.batch_date.isoformat(),
        idempotencyKey=batch.idempotency_key,
        voucherCount=str(len(batch.lines)),
        clearedCount=str(len(batch.cleared)),
        basis=BASIS,
    )

    body = etree.SubElement(envelope, "BODY")
    import_data = etree.SubElement(body, "IMPORTDATA")
    desc = etree.SubElement(import_data, "REQUESTDESC")
    etree.SubElement(desc, "REPORTNAME").text = "Vouchers"
    request = etree.SubElement(import_data, "REQUESTDATA")

    stamp = batch.batch_date.strftime("%Y%m%d")
    for index, line in enumerate(batch.lines, start=1):
        message = etree.SubElement(request, "TALLYMESSAGE")
        voucher = etree.SubElement(message, "VOUCHER", VCHTYPE="Journal", ACTION="Create")
        etree.SubElement(voucher, "DATE").text = stamp
        etree.SubElement(voucher, "VOUCHERNUMBER").text = f"SS-{batch.idempotency_key[:8]}-{index}"
        etree.SubElement(voucher, "NARRATION").text = line.narration
        etree.SubElement(voucher, "SETTLESENSE.EXCEPTIONID").text = line.exception_id
        etree.SubElement(voucher, "SETTLESENSE.CATEGORY").text = line.category
        _leg(voucher, line.debit_ledger, line.amount, debit=True)
        _leg(voucher, line.credit_ledger, line.amount, debit=False)

    rendered = etree.tostring(
        envelope, pretty_print=True, xml_declaration=True, encoding="UTF-8"
    ).decode("utf-8")
    validate(rendered)
    return rendered


def _schema() -> etree.XMLSchema:
    """The bundled XSD, or a refusal naming which way it failed.

    MISSING AND EMPTY ARE DIFFERENT, and here the difference is unusually
    sharp. A missing schema is a broken install. An EMPTY one - or a schema
    that parses and declares nothing - would let `validate` return successfully
    having checked nothing at all, and every document after it would be
    "schema-validated" against a schema with no rules in it. That is the one
    failure mode this whole layer exists to prevent, so it is checked rather
    than assumed.
    """
    if not SCHEMA_PATH.is_file():
        raise ExportError(
            f"{SCHEMA_PATH} is missing. The exporter cannot validate, and an "
            "unvalidated document must not be written - it would look exactly like "
            "a checked one."
        )
    if not SCHEMA_PATH.read_bytes().strip():
        raise ExportError(
            f"{SCHEMA_PATH} is empty. This is worse than missing: an empty schema "
            "parses as no rules, so every document would validate and the word "
            "'schema-validated' would mean nothing."
        )
    try:
        parsed = etree.parse(str(SCHEMA_PATH))
        schema = etree.XMLSchema(parsed)
    except etree.LxmlError as error:
        raise ExportError(f"{SCHEMA_PATH.name} is not a usable schema: {error}") from error
    if not parsed.getroot().findall("{http://www.w3.org/2001/XMLSchema}element"):
        raise ExportError(
            f"{SCHEMA_PATH.name} declares no elements, so it would accept any "
            "document. A schema that validates everything validates nothing."
        )
    return schema


def validate(xml: str) -> None:
    """Against the bundled XSD. Raises ExportError naming every failure."""
    schema = _schema()
    document = etree.fromstring(xml.encode("utf-8"))
    if not schema.validate(document):
        # EVERY failure, not the first. A document that broke the schema in
        # three places and reported one would send the next reader round the
        # loop three times.
        detail = str(schema.error_log) or "no detail"
        raise ExportError(
            f"the generated XML does not validate against {SCHEMA_PATH.name}: {detail}"
        )


def write_dry_run(batch: TallyBatch, directory: Path) -> Path:
    """Writes to disk and NEVER transmits. Filename carries the key.

    Rewriting the same batch overwrites byte-identical content rather than
    appending or minting a second file, which is what makes a re-run safe: the
    key is a function of the exception set, the date and everything the header
    states, so a different filename means a genuinely different document.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / batch.filename
    path.write_text(to_xml(batch), encoding="utf-8")
    return path


def close_exported(store: ExceptionStore, batch: TallyBatch, arrival_day: int) -> tuple[str, ...]:
    """CONFIRMED -> CLOSED for every exception in the batch. The only writer.

    Called AFTER the document is on disk. Closing first would leave an
    exception marked as actioned by a document that was never produced, and
    CLOSED is terminal - there is no path back.
    """
    closed: list[str] = []
    for line in batch.lines:
        store.close_exception(
            line.exception_id,
            arrival_day=arrival_day,
            note=f"exported in Tally batch {batch.idempotency_key[:16]} ({batch.filename})",
        )
        closed.append(line.exception_id)
    return tuple(sorted(closed))

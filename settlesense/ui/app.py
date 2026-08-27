"""M8 - the evidence queue as a Streamlit app. Read-only, no model calls.

    streamlit run settlesense/ui/app.py

EVERY NUMBER COMES FROM `ui/queue.py`, which the static HTML page also reads.
Two views, one data layer, so the page a reviewer screenshots and the app a
reviewer clicks cannot disagree.

THE DAY SELECTOR READS THE STORE. It is not 1/2/3: this store's days are 1, 12
and 24, checkpoints across a 24-day delivery window, and a hardcoded range
would be describing a demo rather than the data.

READ-ONLY. This module opens the store, queries it and closes it. It imports no
model client, and `test_ui.py` asserts that by AST rather than by reading this
sentence.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from eval.run_eval import load_days
from settlesense.config import load_config
from settlesense.exceptions.store import Population
from settlesense.ui.queue import (
    abstention_reason,
    arrival_days,
    as_display_dict,
    build_rows,
    evidence_index,
    money_trail,
    open_store,
    population_summaries,
    residual_sequence,
)

DB_PATH = Path("reports/ui/state.db")
DATA_PATH = Path("data/dev")
CONFIG_PATH = Path("config")
AS_OF = date(2026, 11, 30)


def main() -> None:  # pragma: no cover - requires a Streamlit runtime
    import streamlit as st

    st.set_page_config(page_title="SettleSense — evidence queue", layout="wide")
    st.title("SettleSense — evidence queue")

    config = load_config(CONFIG_PATH)
    dataset = load_days(DATA_PATH, config)
    with open_store(DB_PATH) as store:
        days = arrival_days(store)
        choice = st.selectbox(
            "Arrival day",
            ["all days", *[str(day) for day in days]],
            help="Read from the store, not hardcoded: this run's days are "
            + ", ".join(str(day) for day in days),
        )
        day = None if choice == "all days" else int(choice)

        st.subheader("Populations")
        columns = st.columns(3)
        for column, summary in zip(
            columns, population_summaries(store, dataset, config, AS_OF), strict=False
        ):
            column.metric(
                summary.label,
                f"{summary.residual} residual",
                f"of {summary.denominator:,} {summary.denominator_name}",
            )
        st.caption(
            "Three populations, three denominators. They are never averaged into one "
            "rate: a batch is not a payment."
        )

        sequence = residual_sequence(store, Population.B_BATCH_LINK)
        st.subheader("Residual over time — Population B")
        st.line_chart(
            {"residual": [count for _day, count in sequence]},
            x_label="checkpoint",
            y_label="open batches",
        )
        st.caption(
            f"{' → '.join(str(count) for _day, count in sequence)} across days "
            f"{', '.join(str(day) for day, _count in sequence)}. It rises before it "
            "falls, and that is correct: a residual is a QUEUE, not a burn-down — a "
            "later day delivers batches whose credit is still days away."
        )

        rows = build_rows(store, day)
        st.subheader(f"Queue — {len(rows)} tracked exceptions")
        verified = {
            value: sum(1 for row in rows if row.verified_by == value)
            for value in sorted({row.verified_by for row in rows})
        }
        st.caption(
            "**Verified by** carries the thesis: "
            + ", ".join(f"{count} {name}" for name, count in verified.items())
            + ". Nearly every resolution is a rule, not a model."
        )
        st.dataframe(
            [as_display_dict(row) for row in rows],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Evidence")
        selected = st.selectbox("Exception", [row.exception_id for row in rows], format_func=str)
        row = next(row for row in rows if row.exception_id == selected)
        # THE SHARED RESOLVER, not row.evidence_row_ids. Passing the stored
        # ids straight through showed "No source rows resolve" for rows the
        # static page renders a full trail for - two views disagreeing about
        # the evidence while agreeing about the numbers.
        cited = evidence_index(store, dataset, config)[row.exception_id]
        _render_evidence(st, row, dataset, cited)


def _render_evidence(
    st: Any, row: Any, dataset: Any, cited: tuple[str, ...]
) -> None:  # pragma: no cover
    """The five sections, in the order a reviewer needs them."""
    st.markdown("**1 · Money trail**")
    trail = money_trail(cited, dataset)
    if trail.steps:
        st.code(
            "\n".join(
                f"{stage:>11} | {row_id:<22} {fields}" for stage, row_id, fields in trail.steps
            )
        )
        if not trail.is_complete:
            st.caption("The chain stops before a bank credit — that is the finding.")
    else:
        st.caption("No source rows resolve for this exception.")

    st.markdown("**2 · AI hypothesis**")
    st.caption(
        "Recorded model responses live in fixtures/llm/. The static page "
        "(`make ui-static`) renders each ranked hypothesis with the check that "
        "rejected it."
    )

    st.markdown("**3 · Verification**")
    st.caption("Which check passed or failed, by name — see the static page.")

    st.markdown("**4 · Abstention**")
    reason = abstention_reason(row)
    st.write(reason or "Not abstained — this exception was resolved.")

    st.markdown("**5 · Audit trail**")
    if row.audit:
        st.code(
            "\n".join(
                f"day {entry.arrival_day:>3} seq {entry.sequence:>3} | "
                f"{entry.from_status or '—':<22} -> {entry.to_status:<18} "
                f"{entry.actor:<14} {entry.note}"
                for entry in row.audit
            )
        )
    else:
        st.caption("No audit entries.")


if __name__ == "__main__":  # pragma: no cover
    main()

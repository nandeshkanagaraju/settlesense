"""M8 - the evidence queue as a Streamlit app. Read-only, no model calls.

    make ui        # builds the state DB, then serves this

WHY THIS APP WAS KEPT RATHER THAN DELETED. Its two most important sections -
the ranked hypotheses and the verifier's verdict - used to say "see the static
page", which made the interactive view hollow in exactly the place the
architecture lives. The choice was to fill them in or to delete the app.

Filling them in, because the reason to delete was that two views can diverge -
and they did once, on evidence resolution, for 333 of 339 rows. The fix for
that is not fewer views; it is that neither view computes anything. Both now
call `queue.evidence_panel()` and only lay out what it returns. Deleting would
also contradict SDD 2 and SDD 8, which name this module and the `ui` target.

EVERY NUMBER AND EVERY VERDICT COMES FROM `ui/queue.py`. This module contains
no reconciliation logic, no verification, and no model client.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

from eval.run_eval import load_days
from settlesense.config import load_config
from settlesense.exceptions.store import Population
from settlesense.ui.queue import (
    CATEGORY_COLUMN_PIXELS,
    CATEGORY_COLUMNS,
    SEQUENCE_CAPTION,
    EvidencePanel,
    arrival_days,
    as_display_dict,
    build_rows,
    current_categories,
    evidence_index,
    evidence_panel,
    open_store,
    population_summaries,
    residual_sequence,
    scope_notice,
)

# Which store to render. Overridable because there are now TWO demo stores:
# `streamlit run` takes no arguments of its own, so the outage store built by
# `make ui-outage` had no way to reach this view - and the whole point of that
# store is that both waiting states are visible AT ONCE, which is a claim about
# what a reviewer sees rather than about what a test asserts. An environment
# variable is the only lever Streamlit leaves.
#
# READ-ONLY EITHER WAY. `open_store` refuses a database that does not exist
# rather than rendering an empty queue, so a typo fails loudly instead of
# producing a convincing screenshot of nothing.
#
# A COMMENT, NOT AN ATTRIBUTE DOCSTRING, and the difference is visible to a
# user. Streamlit's "magic" renders every bare expression at module level -
# including a string literal sitting under an assignment - so the docstring
# this started as was printed across the top of the app. Caught by looking at
# the screenshot, which is the only place it could have been caught.
DB_PATH = Path(os.environ.get("SETTLESENSE_DB", "reports/ui/state.db"))

DATA_PATH = Path("data/dev")
CONFIG_PATH = Path("config")
AS_OF = date(2026, 11, 30)


def main() -> None:  # pragma: no cover - requires a Streamlit runtime
    import streamlit as st

    from eval.run_ai import duplicate_exceptions
    from settlesense.ai.client import ReplayLLMClient
    from settlesense.exceptions.taxonomy import VarianceCategory

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
        for column, summary in zip(
            st.columns(3), population_summaries(store, dataset, config, AS_OF), strict=False
        ):
            column.metric(
                summary.label,
                f"{summary.residual} residual",
                f"of {summary.denominator:,} {summary.denominator_name}",
                delta_color="off",
            )
        st.caption(
            "Three populations, three denominators. They are never averaged into one "
            "rate: a batch is not a payment."
        )

        st.subheader("Residual over time — Population B")
        _render_sequence(st, residual_sequence(store, Population.B_BATCH_LINK))

        resolved = current_categories(dataset, config, AS_OF)
        rows = build_rows(store, day, resolved)
        st.subheader(f"Queue — {len(rows)} tracked exceptions")
        verified = {
            value: sum(1 for row in rows if row.verified_by == value)
            for value in sorted({row.verified_by for row in rows})
        }
        st.caption(
            "**Verified by** carries the thesis: "
            + ", ".join(f"{count} {name}" for name, count in verified.items())
            + ". Nearly every resolution is a rule, not a model. **Detected as** is the "
            "category at first sight; **Resolved as** is what closed it."
        )
        # EXPLICIT PIXEL WIDTHS on the category columns. The default sizing
        # clipped MISSING_VS_LATE_CREDIT to "MISSING_VS_LATE_CRED[", which
        # reads as a data error rather than a column width. `st.dataframe`
        # draws to a canvas, so this cannot be asserted from the DOM - the
        # width is asserted instead, and the render checked by eye.
        st.dataframe(
            [as_display_dict(row) for row in rows],
            hide_index=True,
            column_config={
                name: st.column_config.TextColumn(name, width=CATEGORY_COLUMN_PIXELS)
                for name in CATEGORY_COLUMNS
            },
        )
        st.caption(scope_notice(len(rows), len(rows)))

        st.subheader("Evidence")
        # DEFAULT TO A DUPLICATE PAIR. Both halves identical at every step is
        # the more legible finding, and it is what the 480-case abstention
        # reason rests on.
        duplicate = str(VarianceCategory.DUPLICATE_CANDIDATE)
        ordered = sorted(rows, key=lambda row: (row.category != duplicate, -row.amount))
        selected = st.selectbox("Exception", [row.exception_id for row in ordered])
        row = next(item for item in rows if item.exception_id == selected)

        cited = evidence_index(store, dataset, config)[row.exception_id]
        panel = evidence_panel(
            row,
            cited,
            dataset,
            config,
            ReplayLLMClient(),
            {pair.evidence_row_ids: pair for pair in duplicate_exceptions(dataset)},
        )
        _render_evidence(st, panel)


def _render_sequence(st: Any, sequence: list[tuple[int, int]]) -> None:  # pragma: no cover
    """Altair, not st.line_chart, for three reasons the default could not give.

    A y-axis pinned at ZERO - line_chart auto-scaled to -4 on a count of open
    exceptions, and a negative residual is impossible. REAL ARRIVAL DAYS on the
    x-axis instead of positional 0/1/2, which described the list rather than
    the data. And an axis title that is not clipped.
    """
    import altair as alt
    import pandas as pd

    frame = pd.DataFrame(
        {"day": [day for day, _count in sequence], "open": [c for _day, c in sequence]}
    )
    peak = max(frame["open"]) if len(frame) else 1
    chart = (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("day:O", title="arrival day", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "open:Q",
                title="open batch links",
                scale=alt.Scale(domain=[0, peak + 1]),
                axis=alt.Axis(tickMinStep=1),
            ),
            tooltip=["day", "open"],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"{' → '.join(str(count) for _day, count in sequence)} across days "
        f"{', '.join(str(day) for day, _count in sequence)}. {SEQUENCE_CAPTION}"
    )


def _render_evidence(st: Any, panel: EvidencePanel) -> None:  # pragma: no cover
    """The five sections, laid out. Nothing is computed here."""
    st.markdown("**1 · Money trail**")
    if panel.steps:
        st.code(
            "\n".join(
                f"{stage:>11} | {row_id:<22} {fields}" for stage, row_id, fields in panel.steps
            )
        )
        if not panel.trail_complete:
            st.caption("The chain stops before a bank credit — that is the finding.")
    else:
        st.caption("No source rows resolve for this exception.")

    st.markdown("**2 · AI hypothesis**")
    if not panel.eligible_for_model:
        st.caption("Not eligible for the model: decided by rules (PDD 6.1).")
    elif panel.no_recording or not panel.hypotheses:
        st.caption("No model response was recorded for this exception.")
    else:
        st.caption(
            f"The verifier took rank {panel.winning_rank}."
            if panel.winning_rank is not None
            else "No hypothesis survived verification."
        )
        for hypothesis in panel.hypotheses:
            verdict = "VERIFIED" if hypothesis.passed else "REJECTED"
            st.markdown(
                f"**rank {hypothesis.rank}** nominates `{hypothesis.candidate_id}` — **{verdict}**"
            )
            if not hypothesis.passed:
                st.caption(hypothesis.failure_reason)
            st.caption(f"_{hypothesis.reason}_")

    st.markdown("**3 · Verification**")
    if not panel.verification_ran:
        st.caption("No hypothesis-level check applies: settled by the deterministic passes.")
    else:
        st.markdown(f"**{'PASSED' if panel.verification_passed else 'FAILED'}**")
        st.code(", ".join(panel.checks_run) or "none")
        st.caption(
            "no residual applies to this category"
            if panel.computed_residual is None
            else f"computed residual {panel.computed_residual}"
        )
        if not panel.verification_passed:
            st.caption(panel.verification_failure)

    st.markdown("**4 · Abstention**")
    st.write(panel.abstention or "Not abstained — this exception was resolved.")
    if panel.competing:
        st.caption(
            "Competing candidates: "
            + " vs ".join(f"`{row_id}`" for row_id in panel.competing)
            + ". Neither can be eliminated on the evidence, so neither is chosen."
        )

    st.markdown("**5 · Audit trail**")
    if panel.audit:
        st.code(
            "\n".join(
                f"day {entry.arrival_day:>3} seq {entry.sequence:>3} | "
                f"{entry.from_status or '—':<22} -> {entry.to_status:<18} "
                f"{entry.actor:<14} {entry.note}"
                for entry in panel.audit
            )
        )
    else:
        st.caption("No audit entries.")


if __name__ == "__main__":  # pragma: no cover
    main()

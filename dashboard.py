import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import LOG_FILE


st.set_page_config(
    page_title="Local SDK Agent Dashboard",
    page_icon="📊",
    layout="wide",
)


def load_logs() -> pd.DataFrame:
    if not Path(LOG_FILE).exists():
        return pd.DataFrame()

    rows = []

    with Path(LOG_FILE).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def main() -> None:
    st.title("Local SDK Agent Observability Dashboard")
    st.caption("Tracks latency, retrieval quality, refusals, tool success, and approximate token usage.")

    df = load_logs()

    if df.empty:
        st.warning("No logs found yet. Ask a few questions in the main app first.")
        st.code("uv run streamlit run app.py", language="bash")
        return

    total_queries = len(df)
    avg_latency = round(df["latency_ms"].mean(), 2) if "latency_ms" in df else 0
    refusal_rate = round(df["refused"].mean() * 100, 2) if "refused" in df else 0
    tool_success_rate = round(df["tool_call_success"].mean() * 100, 2) if "tool_call_success" in df else 0

    if "retrieval_top_score" in df:
        avg_top_score = round(df["retrieval_top_score"].dropna().mean(), 4)
    else:
        avg_top_score = 0

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Queries", total_queries)
    col2.metric("Avg Latency", f"{avg_latency} ms")
    col3.metric("Refusal Rate", f"{refusal_rate}%")
    col4.metric("Tool Success", f"{tool_success_rate}%")
    col5.metric("Avg Top Score", avg_top_score)

    st.divider()

    st.subheader("Recent Queries")

    display_columns = [
        "timestamp",
        "question",
        "intent",
        "latency_ms",
        "retrieval_top_score",
        "refused",
        "failure_reason",
        "tool_call_success",
    ]

    existing_columns = [column for column in display_columns if column in df.columns]

    st.dataframe(
        df[existing_columns].tail(20).sort_index(ascending=False),
        use_container_width=True,
    )

    st.divider()

    st.subheader("Latency Over Time")

    if "latency_ms" in df.columns:
        st.line_chart(df["latency_ms"])

    st.subheader("Retrieval Top Score Over Time")

    if "retrieval_top_score" in df.columns:
        score_series = df["retrieval_top_score"].dropna()

        if not score_series.empty:
            st.line_chart(score_series)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Intent Distribution")

        if "intent" in df.columns:
            intent_counts = df["intent"].fillna("unknown").value_counts()
            st.bar_chart(intent_counts)

    with right:
        st.subheader("Failure Reasons")

        if "failure_reason" in df.columns:
            failure_counts = df["failure_reason"].fillna("none").value_counts()
            st.bar_chart(failure_counts)

    st.divider()

    st.subheader("Approx Token Usage")

    token_cols = []

    if "approx_input_tokens" in df.columns:
        token_cols.append("approx_input_tokens")

    if "approx_output_tokens" in df.columns:
        token_cols.append("approx_output_tokens")

    if token_cols:
        st.dataframe(df[token_cols].describe(), use_container_width=True)

    st.divider()

    with st.expander("Raw Logs"):
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
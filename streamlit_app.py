import argparse
import glob
import io
import os
import re
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_CSV_PATH = "out.csv"
RUNS_DIR = "runs"


@st.cache_data
def load_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


@st.cache_data
def load_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _log10_safe_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.log10(s.clip(lower=1e-12))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", default=None)
    return parser.parse_known_args()[0]


def _available_csv_paths() -> list[str]:
    paths: list[str] = []
    if os.path.exists(DEFAULT_CSV_PATH):
        paths.append(DEFAULT_CSV_PATH)

    if os.path.isdir(RUNS_DIR):
        run_paths = glob.glob(os.path.join(RUNS_DIR, "*.csv"))
        run_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        paths.extend(run_paths)

    # Dedup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        out.append(p)
    return out


def _label_for_path(path: str) -> str:
    base = os.path.basename(path)
    if path == DEFAULT_CSV_PATH:
        return f"{base} (latest/default)"
    if path.startswith(RUNS_DIR + os.sep):
        # Prefer a readable label-first display even if older runs were created
        # with timestamp-first filenames.
        #
        # New format: <label>_<timestamp>.csv
        # Old format: <timestamp>_<label>.csv
        m_new = re.match(r"^(?P<label>.+)_(?P<ts>\\d{8}T\\d{6}Z)\\.csv$", base)
        if m_new:
            return f"{m_new.group('label')} — {m_new.group('ts')} (saved run)"
        m_old = re.match(r"^(?P<ts>\\d{8}T\\d{6}Z)_(?P<label>.+)\\.csv$", base)
        if m_old:
            return f"{m_old.group('label')} — {m_old.group('ts')} (saved run)"
        return f"{base} (saved run)"
    return base


st.set_page_config(page_title="renc model viewer", layout="wide")

st.title("renc: model output viewer")
st.caption("Interactive viewer for model run CSVs (filters + charts).")

args = _parse_args()
uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])

df: Optional[pd.DataFrame]
loaded_from: str

if uploaded is not None:
    df = load_csv_bytes(uploaded.getvalue())
    loaded_from = f"upload: {uploaded.name}"
else:
    available_paths = _available_csv_paths()
    if not available_paths:
        st.info(
            "No CSV found yet. Generate one (recommended: `uv run renc run --label ...`) "
            "or upload a CSV in the sidebar."
        )
        st.stop()

    labels = [_label_for_path(p) for p in available_paths]
    default_idx = 0
    if args.csv:
        # Prefer an explicit CLI-provided file if it exists in the list.
        for i, p in enumerate(available_paths):
            if os.path.abspath(p) == os.path.abspath(args.csv):
                default_idx = i
                break

    selected_label = st.sidebar.selectbox(
        "Select a dataset",
        options=labels,
        index=default_idx,
        help="Saved runs are stored in `runs/` and can be generated via `uv run python -m src run --label ...`",
    )
    selected_path = available_paths[labels.index(selected_label)]
    df = load_csv_path(selected_path)
    loaded_from = selected_path

st.sidebar.success(f"Loaded: {loaded_from}")

if df is None:
    st.info("No CSV found yet. Generate one or upload a CSV in the sidebar.")
    st.stop()

st.subheader("Run summary")

run_label = (
    df["run_label"].dropna().iloc[0]
    if "run_label" in df.columns and df["run_label"].notna().any()
    else None
)
run_id = (
    df["run_id"].dropna().iloc[0]
    if "run_id" in df.columns and df["run_id"].notna().any()
    else None
)
model_variant = (
    df["model_variant"].dropna().iloc[0]
    if "model_variant" in df.columns and df["model_variant"].notna().any()
    else None
)
use_language_factor = (
    bool(df["use_language_factor"].dropna().iloc[0])
    if "use_language_factor" in df.columns and df["use_language_factor"].notna().any()
    else None
)
uk_missing_strategy = (
    df["uk_missing_strategy"].dropna().iloc[0]
    if "uk_missing_strategy" in df.columns and df["uk_missing_strategy"].notna().any()
    else None
)
uk_floor = (
    float(df["uk_floor"].dropna().iloc[0])
    if "uk_floor" in df.columns and df["uk_floor"].notna().any()
    else None
)

rs1, rs2, rs3, rs4, rs5, rs6 = st.columns(6)
with rs1:
    st.metric("Dataset", os.path.basename(str(loaded_from)))
with rs2:
    st.metric("run_label", "—" if run_label is None else str(run_label))
with rs3:
    st.metric("model_variant", "—" if model_variant is None else str(model_variant))
with rs4:
    st.metric(
        "use_language_factor",
        (
            "—"
            if use_language_factor is None
            else ("yes" if use_language_factor else "no")
        ),
    )
with rs5:
    st.metric(
        "uk_missing_strategy",
        "—" if uk_missing_strategy is None else str(uk_missing_strategy),
    )
with rs6:
    st.metric("uk_floor", "—" if uk_floor is None else f"{uk_floor:.3f}")

meta = {
    "loaded_from": loaded_from,
    "run_id": run_id,
}
with st.expander("Details", expanded=False):
    st.json(meta)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

country_col = "country_name" if "country_name" in df.columns else None

search = st.sidebar.text_input(
    "Search country", value="", placeholder="e.g. United Kingdom"
)

only_candidates = st.sidebar.checkbox(
    "Only single-listen candidates",
    value=False,
    help="Uses `is_single_listen_candidate`",
)

only_seen = st.sidebar.checkbox(
    "Only seen-in-listens", value=False, help="Uses `seen_in_listens` (if present)"
)

use_filters_for_comparison = st.sidebar.checkbox(
    "Use filters for comparison chart/metrics",
    value=False,
    help="If off, comparison uses the full dataset (recommended).",
)

rank_col = "model_rank" if "model_rank" in df.columns else None
if rank_col is not None:
    rank_min = int(pd.to_numeric(df[rank_col], errors="coerce").min())
    rank_max = int(pd.to_numeric(df[rank_col], errors="coerce").max())
    rank_range = st.sidebar.slider(
        "Rank range",
        min_value=rank_min,
        max_value=rank_max,
        value=(rank_min, min(rank_max, 200)),
    )
else:
    rank_range = None

filtered = df.copy()

if country_col is not None and search.strip():
    filtered = filtered[
        filtered[country_col]
        .astype(str)
        .str.contains(search.strip(), case=False, na=False)
    ]

if only_candidates and "is_single_listen_candidate" in filtered.columns:
    filtered = filtered[filtered["is_single_listen_candidate"] == True]  # noqa: E712

if only_seen and "seen_in_listens" in filtered.columns:
    filtered = filtered[filtered["seen_in_listens"] == True]  # noqa: E712

if rank_range is not None:
    filtered[rank_col] = pd.to_numeric(filtered[rank_col], errors="coerce")
    filtered = filtered[filtered[rank_col].between(rank_range[0], rank_range[1])]

st.sidebar.markdown("---")
st.sidebar.subheader("Sort")

sort_cols = [
    c for c in ["total_score", "model_rank", country_col] if c and c in filtered.columns
]
if not sort_cols:
    sort_cols = list(filtered.columns)

sort_by = st.sidebar.selectbox("Sort by", options=sort_cols)
sort_desc = st.sidebar.checkbox("Descending", value=True)

filtered_sorted = filtered.sort_values(
    by=sort_by, ascending=not sort_desc, ignore_index=True
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", f"{len(filtered_sorted):,}")
with col2:
    st.metric("Columns", f"{len(filtered_sorted.columns):,}")
with col3:
    if "alpha_3" in filtered_sorted.columns:
        dup_count = int(filtered_sorted.duplicated(subset=["alpha_3"]).sum())
        st.metric("Duplicate alpha_3 (filtered)", f"{dup_count:,}")

st.markdown("---")

st.subheader("Single-listen comparison")

comparison_df = filtered_sorted if use_filters_for_comparison else df
comparison_df = comparison_df.copy()

has_rank = rank_col is not None and rank_col in comparison_df.columns
has_score = "total_score" in comparison_df.columns
has_candidate = "is_single_listen_candidate" in comparison_df.columns
has_seen = "seen_in_listens" in comparison_df.columns

if has_rank:
    comparison_df[rank_col] = pd.to_numeric(comparison_df[rank_col], errors="coerce")

if has_score:
    comparison_df["total_score"] = pd.to_numeric(
        comparison_df["total_score"], errors="coerce"
    )
    # Allow switching between raw score and log variants (log can look odd for <1).
    y_mode = st.radio(
        "Y-axis",
        options=["score", "log10(score)", "log10(1+score)"],
        horizontal=True,
        index=2,
    )
    if y_mode == "score":
        comparison_df["y_score"] = comparison_df["total_score"]
        y_title = "total_score"
    elif y_mode == "log10(score)":
        comparison_df["y_score"] = _log10_safe_series(
            comparison_df["total_score"].fillna(0) + 1e-12
        )
        y_title = "log10(total_score)"
    else:
        comparison_df["y_score"] = _log10_safe_series(
            comparison_df["total_score"].clip(lower=0).fillna(0) + 1.0
        )
        y_title = "log10(1 + total_score)"

id_col = (
    "alpha_3"
    if "alpha_3" in comparison_df.columns
    else ("country_name" if "country_name" in comparison_df.columns else None)
)

pred_mask = (
    comparison_df["is_single_listen_candidate"] == True if has_candidate else False
)  # noqa: E712
seen_mask = (
    comparison_df["seen_in_listens"] == True if has_seen else False
)  # noqa: E712

predicted = (
    comparison_df[pred_mask].copy() if has_candidate else comparison_df.iloc[0:0].copy()
)
actual = comparison_df[seen_mask].copy() if has_seen else comparison_df.iloc[0:0].copy()

overlap_count = None
if id_col and not predicted.empty and not actual.empty:
    overlap_count = len(set(predicted[id_col].dropna()) & set(actual[id_col].dropna()))

mean_abs_rank_distance = None
rmse_rank_distance = None
mean_abs_rank_distance_norm = None
rmse_rank_distance_norm = None
if has_rank and not predicted.empty and not actual.empty:
    pred_ranks = (
        predicted[[rank_col]].dropna().sort_values(rank_col)[rank_col].to_numpy()
    )
    act_ranks = actual[[rank_col]].dropna().sort_values(rank_col)[rank_col].to_numpy()
    n = int(min(len(pred_ranks), len(act_ranks)))
    if n > 0:
        diffs = pred_ranks[:n] - act_ranks[:n]
        mean_abs_rank_distance = float(np.mean(np.abs(diffs)))
        rmse_rank_distance = float(np.sqrt(np.mean(diffs**2)))
        denom = float(pd.to_numeric(comparison_df[rank_col], errors="coerce").max())
        if denom > 0:
            mean_abs_rank_distance_norm = mean_abs_rank_distance / denom
            rmse_rank_distance_norm = rmse_rank_distance / denom

mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.metric(
        "Predicted candidates (count)",
        f"{int(pred_mask.sum()) if has_candidate else 0:,}",
    )
with mc2:
    st.metric(
        "Known single-listens (count)", f"{int(seen_mask.sum()) if has_seen else 0:,}"
    )
with mc3:
    st.metric(
        "Overlap (count)",
        "—" if overlap_count is None else f"{overlap_count:,}",
    )
with mc4:
    st.metric(
        "Mean abs rank distance",
        (
            "—"
            if mean_abs_rank_distance is None
            else (
                f"{mean_abs_rank_distance:.2f} ({mean_abs_rank_distance_norm:.3f} norm)"
                if mean_abs_rank_distance_norm is not None
                else f"{mean_abs_rank_distance:.2f}"
            )
        ),
    )

if rmse_rank_distance is not None:
    st.caption(
        f"RMSE rank distance: {rmse_rank_distance:.2f}"
        + (
            f" ({rmse_rank_distance_norm:.3f} norm)"
            if rmse_rank_distance_norm is not None
            else ""
        )
        + ". RMSE penalizes big misses more than mean absolute distance."
    )

if has_rank and has_score:
    plot = comparison_df[[rank_col, "y_score"]].copy()
    if "country_name" in comparison_df.columns:
        plot["country"] = comparison_df["country_name"]
    elif "alpha_3" in comparison_df.columns:
        plot["country"] = comparison_df["alpha_3"]
    else:
        plot["country"] = ""

    plot["predicted"] = pred_mask if has_candidate else False
    plot["actual"] = seen_mask if has_seen else False
    plot["group"] = np.select(
        [plot["predicted"] & plot["actual"], plot["predicted"], plot["actual"]],
        ["predicted & actual", "predicted", "actual"],
        default="other",
    )
    plot = plot.dropna(subset=[rank_col, "y_score"])

    color_scale = alt.Scale(
        domain=["predicted & actual", "predicted", "actual", "other"],
        range=["#7c3aed", "#f59e0b", "#10b981", "#94a3b8"],
    )

    chart = (
        alt.Chart(plot)
        .mark_circle()
        .encode(
            x=alt.X(f"{rank_col}:Q", title="model_rank"),
            y=alt.Y("y_score:Q", title=y_title),
            color=alt.Color("group:N", scale=color_scale, legend=alt.Legend(title="")),
            size=alt.Size(
                "group:N",
                scale=alt.Scale(
                    domain=["predicted & actual", "predicted", "actual", "other"],
                    range=[180, 140, 140, 30],
                ),
                legend=None,
            ),
            opacity=alt.Opacity(
                "group:N",
                scale=alt.Scale(
                    domain=["predicted & actual", "predicted", "actual", "other"],
                    range=[1.0, 0.95, 0.95, 0.35],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("country:N", title="country"),
                alt.Tooltip(f"{rank_col}:Q", title="rank"),
                alt.Tooltip("y_score:Q", title=y_title),
                alt.Tooltip("group:N", title="group"),
            ],
        )
        .properties(height=380)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")
else:
    st.info(
        "To show the comparison chart, ensure your CSV includes `model_rank` and `total_score` "
        "(and optionally `is_single_listen_candidate` + `seen_in_listens`)."
    )

st.markdown("---")

st.subheader("Table")

preferred_cols = [
    c
    for c in [
        "model_rank",
        "alpha_3",
        "country_name",
        "total_score",
        "pop_score",
        "net_score",
        "uk_score",
        "language_factor",
        "is_single_listen_candidate",
        "seen_in_listens",
        "predicted_single_listen_country",
        "prediction_date",
    ]
    if c in filtered_sorted.columns
]

if preferred_cols:
    table_df = filtered_sorted[preferred_cols]
else:
    table_df = filtered_sorted

st.dataframe(table_df, width="stretch", height=520)

csv_bytes = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered CSV",
    data=csv_bytes,
    file_name="out.filtered.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("Quick charts")

if "total_score" in filtered_sorted.columns:
    tmp = filtered_sorted.copy()
    tmp["total_score"] = pd.to_numeric(tmp["total_score"], errors="coerce")
    tmp["log10_total_score"] = _log10_safe_series(tmp["total_score"].fillna(0) + 1e-12)

    left, right = st.columns(2)
    with left:
        st.caption("Rank vs log10(total_score)")
        if rank_col is not None and rank_col in tmp.columns:
            plot_df = tmp[[rank_col, "log10_total_score"]].dropna()
            plot_df = plot_df.rename(columns={rank_col: "rank"})
            st.scatter_chart(plot_df, x="rank", y="log10_total_score", height=320)
        else:
            st.info("No `model_rank` column found to plot rank-vs-score.")

    with right:
        st.caption("Score distribution (binned log10(total_score))")
        hist_df = tmp[["log10_total_score"]].dropna().copy()
        if not hist_df.empty:
            bins = st.slider("Histogram bins", 10, 80, 30)
            hist_df["bin"] = pd.cut(hist_df["log10_total_score"], bins=bins)
            counts = (
                hist_df.groupby("bin", observed=True).size().reset_index(name="count")
            )
            counts["bin"] = counts["bin"].astype(str)
            st.bar_chart(counts, x="bin", y="count", height=320)
        else:
            st.info("No numeric `total_score` values to chart.")

if (
    "is_single_listen_candidate" in filtered_sorted.columns
    and "total_score" in filtered_sorted.columns
):
    st.caption("Top candidates by total_score")
    cand = filtered_sorted.copy()
    cand["total_score"] = pd.to_numeric(cand["total_score"], errors="coerce")
    cand = cand[cand["is_single_listen_candidate"] == True].sort_values(
        "total_score", ascending=False
    )
    top_n = st.slider("Top N", 5, 50, 20)
    cand = cand.head(top_n)
    label_col = "country_name" if "country_name" in cand.columns else "alpha_3"
    if label_col in cand.columns and not cand.empty:
        chart_df = cand[[label_col, "total_score"]].set_index(label_col)
        st.bar_chart(chart_df, height=360)

if (
    "seen_in_listens" in filtered_sorted.columns
    and "total_score" in filtered_sorted.columns
):
    st.caption("Seen vs not-seen (mean total_score)")
    tmp2 = filtered_sorted.copy()
    tmp2["total_score"] = pd.to_numeric(tmp2["total_score"], errors="coerce")
    summary = (
        tmp2.groupby("seen_in_listens", dropna=False)["total_score"]
        .mean()
        .reset_index()
        .rename(columns={"total_score": "mean_total_score"})
    )
    summary["seen_in_listens"] = summary["seen_in_listens"].astype(str)
    st.bar_chart(summary, x="seen_in_listens", y="mean_total_score", height=260)

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
RUNS_DIRS = ["data/runs", "runs"]


@st.cache_data
def load_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


@st.cache_data
def load_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", default=None)
    return parser.parse_known_args()[0]


def _available_csv_paths() -> list[str]:
    paths: list[str] = []
    if os.path.exists(DEFAULT_CSV_PATH):
        paths.append(DEFAULT_CSV_PATH)

    for runs_dir in RUNS_DIRS:
        if os.path.isdir(runs_dir):
            run_paths = glob.glob(os.path.join(runs_dir, "*.csv"))
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

    # Prefer: <label>_<timestamp>.csv
    m_new = re.match(r"^(?P<label>.+)_(?P<ts>\\d{8}T\\d{6}Z)\\.csv$", base)
    if m_new:
        return f"{m_new.group('label')} — {m_new.group('ts')}"
    return base


st.set_page_config(page_title="renc bayes viewer", layout="wide")
st.title("renc: bayesian run viewer")
st.caption(
    "Story-first viewer for Bayes-run CSVs: what data went in, what assumptions were made, "
    "how the model works, and why countries rank highly."
)

args = _parse_args()
uploaded = st.sidebar.file_uploader("Upload a Bayes CSV", type=["csv"])

df: Optional[pd.DataFrame]
loaded_from: str

if uploaded is not None:
    df = load_csv_bytes(uploaded.getvalue())
    loaded_from = f"upload: {uploaded.name}"
else:
    available_paths = _available_csv_paths()
    if not available_paths:
        st.info(
            "No CSV found yet. Generate one with "
            "`uv run python -m src bayes-run --label ...` or upload a CSV in the sidebar."
        )
        st.stop()

    labels = [_label_for_path(p) for p in available_paths]
    default_idx = 0
    if args.csv:
        for i, p in enumerate(available_paths):
            if os.path.abspath(p) == os.path.abspath(args.csv):
                default_idx = i
                break

    selected_label = st.sidebar.selectbox(
        "Select a dataset", options=labels, index=default_idx
    )
    selected_path = available_paths[labels.index(selected_label)]
    df = load_csv_path(selected_path)
    loaded_from = selected_path

st.sidebar.success(f"Loaded: {loaded_from}")

if df is None:
    st.stop()

if "bayes_p_one_mean" not in df.columns:
    st.error(
        "This CSV does not look like a bayes-run output (missing `bayes_p_one_mean`). "
        "Generate one via `uv run python -m src bayes-run --label ...`."
    )
    st.stop()

# Coerce key numeric columns.
for c in [
    "bayes_p_one_mean",
    "bayes_p_one_hdi_low",
    "bayes_p_one_hdi_high",
    "bayes_mu_mean",
    "bayes_mu_hdi_low",
    "bayes_mu_hdi_high",
    "bayes_lp_mean",
    "bayes_lp_hdi_low",
    "bayes_lp_hdi_high",
    "model_rank",
    "total_score",
]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

country_col = "country_name" if "country_name" in df.columns else "alpha_3"

st.subheader("What this model is doing (the story)")
st.markdown(
    """
We model *listen counts* per country \(Y_i\) using a Poisson regression:

- \(Y_i \\sim \\text{Poisson}(\\mu_i)\)
- \\(\\log \\mu_i = \\alpha + \\beta^T x_i\\)

Then we derive the probability you actually care about:

- \(P(Y_i=1 \\mid \\mu_i) = \\mu_i e^{-\\mu_i}\)

So countries with \\(\\mu \\approx 1\\) are the “sweet spot” for single listens.
"""
)

with st.expander(
    "Why the model can still feel surprising (aggregate constraints + incomplete counts)",
    expanded=False,
):
    st.markdown(
        """
We only supervise the model with:

- the per-country `COUNTRIES_LISTENS` counts you’ve verified (including explicit 0s)
- the aggregate `NUMBER_OF_COUNTRIES_WITH_LISTENS` (how many countries had any listens at all)

Because we don’t know *which* countries are in the remaining “has listens” set, we add a soft aggregate constraint:
\(\\sum_i P(Y_i > 0)\\) should match `NUMBER_OF_COUNTRIES_WITH_LISTENS` (after accounting for observed nonzero countries).

**Why it can feel unintuitive**: with few observed countries, many different coefficient settings can satisfy the aggregate
constraint while still fitting the observed counts.

**Mitigation**: keep adding verified 0/1/large counts. Each new observed country collapses a lot of uncertainty.
"""
    )

st.subheader("Run summary + diagnostics")
meta_cols = [
    "run_label",
    "run_id",
    "bayes_draws_cli",
    "bayes_tune_cli",
    "bayes_target_accept_cli",
    "bayes_seed_cli",
    "bayes_hdi_prob_cli",
    "bayes_rhat_max",
    "bayes_ess_bulk_min",
    "bayes_train_n",
    "bayes_train_n_one",
    "bayes_train_n_multi",
    "bayes_train_n_zero",
    "bayes_number_of_countries_with_listens",
    "bayes_aggregate_sigma",
    "bayes_number_of_countries_with_one_listen",
    "bayes_aggregate_one_sigma",
    "bayes_use_distance",
]
meta = {
    c: df[c].dropna().iloc[0]
    for c in meta_cols
    if c in df.columns and df[c].notna().any()
}
st.json({"loaded_from": loaded_from, **meta})

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
search = st.sidebar.text_input("Search country", value="", placeholder="e.g. Vietnam")
unknown_only = (
    st.sidebar.checkbox("Only bayes_label=unknown", value=True)
    if "bayes_label" in df.columns
    else False
)

filtered = df.copy()
if search.strip():
    filtered = filtered[
        filtered[country_col]
        .astype(str)
        .str.contains(search.strip(), case=False, na=False)
    ]
if unknown_only and "bayes_label" in filtered.columns:
    filtered = filtered[filtered["bayes_label"] == "unknown"]

st.subheader("Top candidates (Bayes)")
top_n = st.slider("Top N", 5, 100, 25)
top = filtered.sort_values("bayes_p_one_mean", ascending=False).head(top_n).copy()
cols = [
    c
    for c in [
        "bayes_rank",
        country_col,
        "alpha_3",
        "bayes_label",
        "bayes_p_one_mean",
        "bayes_p_one_hdi_low",
        "bayes_p_one_hdi_high",
        "bayes_mu_mean",
        "bayes_mu_hdi_low",
        "bayes_mu_hdi_high",
        "model_rank",
        "total_score",
    ]
    if c in top.columns
]
st.dataframe(top[cols], width="stretch", height=420)

st.subheader("How the training data (labels) looks")
if "bayes_label" in df.columns:
    label_counts = (
        df["bayes_label"]
        .fillna("unknown")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("bayes_label")
        .reset_index(name="rows")
    )
    st.dataframe(label_counts, width="stretch", height=220)

if "bayes_y_observed" in df.columns:
    train_rows = df[df["bayes_y_observed"].notna()].copy()
    if not train_rows.empty:
        train_rows["bayes_y_observed"] = pd.to_numeric(
            train_rows["bayes_y_observed"], errors="coerce"
        )
        st.caption("Training rows (what the model was actually fit on)")
        show = train_rows.sort_values(["bayes_y_observed"], ascending=[False]).copy()
        keep = [
            c
            for c in [country_col, "alpha_3", "bayes_label", "bayes_y_observed"]
            if c in show.columns
        ]
        st.dataframe(show[keep], width="stretch", height=260)

st.subheader("Input data explorer (features + missingness)")
input_cols = [
    c
    for c in [
        country_col,
        "alpha_3",
        "population",
        "internet_usage_pct",
        "internet_usage_record_year",
        "latitude",
        "longitude",
        "uk_distance_km",
        "uk_visits_number",
        "uk_visits_period",
        "languages",
    ]
    if c in df.columns
]
if input_cols:
    miss = (
        df[input_cols]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    left, right = st.columns(2)
    with left:
        st.caption("Missingness (fraction of rows missing)")
        st.dataframe(miss, width="stretch", height=320)
    with right:
        st.caption("Raw input rows (filtered)")
        st.dataframe(filtered[input_cols].head(200), width="stretch", height=320)
else:
    st.info("No raw input columns found in this CSV.")

st.subheader("What the model learned (coefficients)")
beta_rows = []
if "bayes_alpha_mean" in df.columns:
    beta_rows.append(
        {
            "param": "alpha (intercept)",
            "mean": float(df["bayes_alpha_mean"].dropna().iloc[0]),
            "hdi_low": (
                float(df["bayes_alpha_hdi_low"].dropna().iloc[0])
                if "bayes_alpha_hdi_low" in df.columns
                and df["bayes_alpha_hdi_low"].notna().any()
                else np.nan
            ),
            "hdi_high": (
                float(df["bayes_alpha_hdi_high"].dropna().iloc[0])
                if "bayes_alpha_hdi_high" in df.columns
                and df["bayes_alpha_hdi_high"].notna().any()
                else np.nan
            ),
        }
    )

for base in ["log_population", "internet_rate", "log1p_uk_distance_km"]:
    mean_col = f"bayes_beta_{base}_mean"
    if mean_col in df.columns and df[mean_col].notna().any():
        beta_rows.append(
            {
                "param": f"beta[{base}]",
                "mean": float(df[mean_col].dropna().iloc[0]),
                "hdi_low": (
                    float(df[f"bayes_beta_{base}_hdi_low"].dropna().iloc[0])
                    if f"bayes_beta_{base}_hdi_low" in df.columns
                    and df[f"bayes_beta_{base}_hdi_low"].notna().any()
                    else np.nan
                ),
                "hdi_high": (
                    float(df[f"bayes_beta_{base}_hdi_high"].dropna().iloc[0])
                    if f"bayes_beta_{base}_hdi_high" in df.columns
                    and df[f"bayes_beta_{base}_hdi_high"].notna().any()
                    else np.nan
                ),
            }
        )

if beta_rows:
    st.dataframe(pd.DataFrame(beta_rows), width="stretch", height=240)
    st.caption(
        "Interpretation: coefficients are on standardized features. Positive beta increases log(mu), "
        "making higher counts more likely; single-listen probability peaks near mu≈1."
    )
else:
    st.info("No coefficient summaries found in this CSV.")

st.subheader("Charts")

chart_df = filtered[[country_col, "bayes_p_one_mean"]].dropna().copy()
chart_df = chart_df.rename(columns={country_col: "country"})
chart_df["bayes_p_one_mean"] = pd.to_numeric(
    chart_df["bayes_p_one_mean"], errors="coerce"
)
chart_df = chart_df.dropna(subset=["bayes_p_one_mean"])

hist_bins = st.slider("Histogram bins", 10, 80, 30)
hist = chart_df.copy()
hist["bin"] = pd.cut(hist["bayes_p_one_mean"], bins=hist_bins)
counts = hist.groupby("bin", observed=True).size().reset_index(name="count")
counts["bin"] = counts["bin"].astype(str)
st.bar_chart(counts, x="bin", y="count", height=260)

if "model_rank" in filtered.columns:
    tmp = filtered[[country_col, "model_rank", "bayes_p_one_mean"]].dropna().copy()
    tmp = tmp.rename(columns={country_col: "country"})
    tmp["model_rank"] = pd.to_numeric(tmp["model_rank"], errors="coerce")
    tmp["bayes_p_one_mean"] = pd.to_numeric(tmp["bayes_p_one_mean"], errors="coerce")
    tmp = tmp.dropna(subset=["model_rank", "bayes_p_one_mean"])
    scat = (
        alt.Chart(tmp)
        .mark_circle()
        .encode(
            x=alt.X("model_rank:Q", title="heuristic model_rank"),
            y=alt.Y("bayes_p_one_mean:Q", title="Bayes P(listens=1) (mean)"),
            tooltip=[
                alt.Tooltip("country:N", title="country"),
                alt.Tooltip("model_rank:Q", title="heuristic_rank"),
                alt.Tooltip("bayes_p_one_mean:Q", title="p_one_mean"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(scat, width="stretch")

if "uk_distance_km" in filtered.columns:
    st.subheader("Distance diagnostics (if distance columns exist)")
    tmp = filtered[
        [country_col, "uk_distance_km", "bayes_p_one_mean", "bayes_mu_mean"]
    ].copy()
    tmp = tmp.rename(columns={country_col: "country"})
    tmp["uk_distance_km"] = pd.to_numeric(tmp["uk_distance_km"], errors="coerce")
    tmp["bayes_p_one_mean"] = pd.to_numeric(tmp["bayes_p_one_mean"], errors="coerce")
    tmp["bayes_mu_mean"] = pd.to_numeric(tmp["bayes_mu_mean"], errors="coerce")
    tmp = tmp.dropna(subset=["uk_distance_km", "bayes_p_one_mean", "bayes_mu_mean"])
    tmp["uk_distance_km_plus1"] = tmp["uk_distance_km"] + 1.0

    left, right = st.columns(2)
    with left:
        st.caption("P(Y=1) vs distance from UK (log scale on distance)")
        scat_d = (
            alt.Chart(tmp)
            .mark_circle()
            .encode(
                x=alt.X(
                    "uk_distance_km_plus1:Q",
                    title="distance from UK (km, +1; log scale)",
                    scale=alt.Scale(type="log"),
                ),
                y=alt.Y("bayes_p_one_mean:Q", title="Bayes P(listens=1) (mean)"),
                tooltip=[
                    alt.Tooltip("country:N", title="country"),
                    alt.Tooltip("uk_distance_km:Q", title="uk_distance_km"),
                    alt.Tooltip("bayes_p_one_mean:Q", title="p_one_mean"),
                    alt.Tooltip("bayes_mu_mean:Q", title="mu_mean"),
                ],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(scat_d, width="stretch")

    with right:
        st.caption("mu vs distance from UK (log scale on distance)")
        scat_m = (
            alt.Chart(tmp)
            .mark_circle()
            .encode(
                x=alt.X(
                    "uk_distance_km_plus1:Q",
                    title="distance from UK (km, +1; log scale)",
                    scale=alt.Scale(type="log"),
                ),
                y=alt.Y("bayes_mu_mean:Q", title="Bayes mu (mean)"),
                tooltip=[
                    alt.Tooltip("country:N", title="country"),
                    alt.Tooltip("uk_distance_km:Q", title="uk_distance_km"),
                    alt.Tooltip("bayes_mu_mean:Q", title="mu_mean"),
                    alt.Tooltip("bayes_p_one_mean:Q", title="p_one_mean"),
                ],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(scat_m, width="stretch")

st.markdown("---")
st.subheader("Debug: feature columns (if present)")

debug_cols = [
    c
    for c in [
        country_col,
        "uk_distance_km",
        "bayes_x_log_population",
        "bayes_x_internet_rate",
        "bayes_x_log1p_uk_distance_km",
        "bayes_z_log_population",
        "bayes_z_internet_rate",
        "bayes_z_log1p_uk_distance_km",
        "bayes_lp_mean",
        "bayes_lp_hdi_low",
        "bayes_lp_hdi_high",
        "bayes_mu_mean",
        "bayes_p_one_mean",
    ]
    if c in df.columns
]

if debug_cols:
    st.dataframe(
        filtered.sort_values("bayes_p_one_mean", ascending=False)[debug_cols].head(50),
        width="stretch",
        height=420,
    )
else:
    st.info("No debug feature columns present in this CSV yet.")

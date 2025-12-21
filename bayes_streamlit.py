import argparse
import glob
import io
import json
import os
import re
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

from src.schema import BAYES_RUN_SCHEMA, validate_frame

DEFAULT_CSV_PATH = "out.csv"
# Search both “new” and legacy run dirs, plus any notebook-produced artifacts.
RUNS_DIRS = ["data/runs", "runs", "notebooks/data/runs"]


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


def _meta_path_for_csv_path(csv_path: str) -> str:
    base, _ext = os.path.splitext(csv_path)
    return f"{base}.meta.json"


def _load_meta_for_loaded_from(loaded_from: str) -> dict | None:
    if not loaded_from or loaded_from.startswith("upload:"):
        return None
    if not os.path.exists(loaded_from):
        return None
    mp = _meta_path_for_csv_path(loaded_from)
    if not os.path.exists(mp):
        return None
    try:
        with open(mp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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

st.sidebar.markdown("---")
st.sidebar.subheader("Schema check")
schema_report = validate_frame(df, BAYES_RUN_SCHEMA)
if schema_report["ok"]:
    st.sidebar.success(f"OK: {schema_report['schema']}")
else:
    st.sidebar.error(f"Missing required columns: {schema_report['missing_required']}")

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

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
search = st.sidebar.text_input("Search country", value="", placeholder="e.g. France")
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

tabs = st.tabs(["Summary", "Data", "Bayes", "Heuristic", "Compare", "Map"])

with tabs[0]:
    st.subheader("Summary")
    st.markdown(
        """
We model per-country listen counts \(Y_i\) with a Poisson GLM, and derive:
\(P(Y_i=1)=\\mu_i e^{-\\mu_i}\\).

Use this viewer to:
- shortlist candidates
- explain why a country looks likely/unlikely
- compare runs (distance on/off, etc.)
"""
    )
    meta_cols = [
        "run_label",
        "run_id",
        "bayes_use_distance",
        "bayes_train_n",
        "bayes_train_n_one",
        "bayes_train_n_multi",
        "bayes_train_n_zero",
        "bayes_number_of_countries_with_listens",
        "bayes_number_of_countries_with_one_listen",
        "bayes_rhat_max",
        "bayes_ess_bulk_min",
    ]
    meta = {
        c: df[c].dropna().iloc[0]
        for c in meta_cols
        if c in df.columns and df[c].notna().any()
    }
    sidecar_meta = _load_meta_for_loaded_from(loaded_from)
    st.json({"loaded_from": loaded_from, **meta})
    if sidecar_meta is not None:
        with st.expander("Run metadata (sidecar JSON)", expanded=False):
            st.json(sidecar_meta)

with tabs[1]:
    st.subheader("Data explorer")
    # Keep this tab simple: show inputs, and include observed listens (if present)
    # as a column so you can sort/filter directly in the dataframe UI.
    if "bayes_y_observed" in df.columns:
        df["bayes_y_observed"] = pd.to_numeric(df["bayes_y_observed"], errors="coerce")

    # The sidebar filter defaults to bayes_label=unknown, which hides training rows.
    # For data inspection, default to showing the full dataset.
    respect_sidebar = st.checkbox("Respect sidebar filters on this tab", value=False)
    data_rows = filtered if respect_sidebar else df

    input_cols = [
        c
        for c in [
            country_col,
            "alpha_3",
            "bayes_y_observed",
            "population",
            "internet_usage_pct",
            "internet_usage_record_year",
            "latitude",
            "longitude",
            "uk_distance_km",
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
            st.caption("Missingness")
            st.dataframe(miss, width="stretch", height=320)
        with right:
            st.caption("Rows (filtered)")
            # Streamlit sorting can behave oddly with NaN vs 0. Pre-sort so any observed
            # values come first, then sort by the observed count (desc).
            view = data_rows[input_cols].copy()
            if "bayes_y_observed" in view.columns:
                y = pd.to_numeric(view["bayes_y_observed"], errors="coerce")
                view["_has_observed"] = y.notna()
                view["_y_sort"] = y.fillna(-1.0)
                view = view.sort_values(
                    by=["_has_observed", "_y_sort", country_col],
                    ascending=[False, False, True],
                    na_position="last",
                ).drop(columns=["_has_observed", "_y_sort"], errors="ignore")

            st.dataframe(view.head(300), width="stretch", height=320)
    else:
        st.info("No raw input columns found in this CSV.")

with tabs[2]:
    st.subheader("Bayes results")
    top_n = st.slider("Top N (Bayes)", 5, 200, 50)
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
            "uk_distance_km",
        ]
        if c in top.columns
    ]
    st.dataframe(top[cols], width="stretch", height=420)

    st.subheader("Explain a country")
    options = filtered[country_col].astype(str).fillna("").unique().tolist()
    if options:
        pick = st.selectbox("Country", options=options, index=0)
        row = filtered.loc[filtered[country_col].astype(str) == str(pick)]
        if not row.empty:
            r = row.iloc[0].to_dict()
            show = {
                k: r.get(k)
                for k in [
                    "country_name",
                    "alpha_3",
                    "population",
                    "internet_usage_pct",
                    "uk_distance_km",
                    "bayes_mu_mean",
                    "bayes_p_one_mean",
                    "bayes_lp_mean",
                    "bayes_label",
                    "bayes_y_observed",
                ]
                if k in r
            }
            st.json(show)
    else:
        st.info("No countries to select (filtered dataset empty).")

    st.subheader("Coefficients")
    beta_rows = []
    if "bayes_alpha_mean" in df.columns and df["bayes_alpha_mean"].notna().any():
        beta_rows.append(
            {
                "param": "alpha",
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
        st.dataframe(pd.DataFrame(beta_rows), width="stretch", height=220)
    else:
        st.info("No coefficient summaries found in this CSV.")

    st.subheader("Distribution")
    chart_df = filtered[[country_col, "bayes_p_one_mean"]].dropna().copy()
    chart_df = chart_df.rename(columns={country_col: "country"})
    chart_df["bayes_p_one_mean"] = pd.to_numeric(
        chart_df["bayes_p_one_mean"], errors="coerce"
    )
    chart_df = chart_df.dropna(subset=["bayes_p_one_mean"])
    bins = st.slider("Histogram bins", 10, 80, 30)
    chart_df["bin"] = pd.cut(chart_df["bayes_p_one_mean"], bins=bins)
    counts = chart_df.groupby("bin", observed=True).size().reset_index(name="count")
    counts["bin"] = counts["bin"].astype(str)
    st.bar_chart(counts, x="bin", y="count", height=260)

with tabs[3]:
    st.subheader("Heuristic results")
    if "model_rank" not in filtered.columns or "total_score" not in filtered.columns:
        st.info(
            "This run CSV does not include heuristic columns (`model_rank`, `total_score`)."
        )
    else:
        cols = [
            c
            for c in [
                country_col,
                "alpha_3",
                "model_rank",
                "total_score",
                "is_single_listen_candidate",
            ]
            if c in filtered.columns
        ]
        st.dataframe(
            filtered.sort_values("model_rank", ascending=True)[cols].head(300),
            width="stretch",
            height=520,
        )

with tabs[4]:
    st.subheader("Compare runs")
    st.caption("Pick two run CSVs and compare deltas (rank + P(Y=1)).")
    all_paths = _available_csv_paths()
    if len(all_paths) < 2:
        st.info("Need at least 2 run CSVs in run directories to compare.")
    else:
        labels_all = [_label_for_path(p) for p in all_paths]
        c1, c2 = st.columns(2)
        with c1:
            a_label = st.selectbox("Run A", options=labels_all, index=0)
        with c2:
            b_label = st.selectbox(
                "Run B", options=labels_all, index=1 if len(labels_all) > 1 else 0
            )
        path_a = all_paths[labels_all.index(a_label)]
        path_b = all_paths[labels_all.index(b_label)]
        df_a = load_csv_path(path_a)
        df_b = load_csv_path(path_b)
        rep_a = validate_frame(df_a, BAYES_RUN_SCHEMA)
        rep_b = validate_frame(df_b, BAYES_RUN_SCHEMA)
        if not rep_a["ok"]:
            st.warning(f"Run A missing required cols: {rep_a['missing_required']}")
        if not rep_b["ok"]:
            st.warning(f"Run B missing required cols: {rep_b['missing_required']}")
        if "alpha_3" not in df_a.columns or "alpha_3" not in df_b.columns:
            st.info("Cannot compare: both runs must include `alpha_3`.")
        else:
            a = df_a[
                [
                    c
                    for c in [
                        "alpha_3",
                        "country_name",
                        "bayes_rank",
                        "bayes_p_one_mean",
                        "bayes_mu_mean",
                    ]
                    if c in df_a.columns
                ]
            ].copy()
            b = df_b[
                [
                    c
                    for c in [
                        "alpha_3",
                        "country_name",
                        "bayes_rank",
                        "bayes_p_one_mean",
                        "bayes_mu_mean",
                    ]
                    if c in df_b.columns
                ]
            ].copy()
            a = a.rename(
                columns={
                    "bayes_rank": "rank_a",
                    "bayes_p_one_mean": "p1_a",
                    "bayes_mu_mean": "mu_a",
                    "country_name": "country_name_a",
                }
            )
            b = b.rename(
                columns={
                    "bayes_rank": "rank_b",
                    "bayes_p_one_mean": "p1_b",
                    "bayes_mu_mean": "mu_b",
                    "country_name": "country_name_b",
                }
            )
            cmp = pd.merge(a, b, on="alpha_3", how="inner")
            if "country_name_a" in cmp.columns:
                cmp["country_name"] = cmp["country_name_a"]
            elif "country_name_b" in cmp.columns:
                cmp["country_name"] = cmp["country_name_b"]
            for c in ["rank_a", "rank_b", "p1_a", "p1_b", "mu_a", "mu_b"]:
                if c in cmp.columns:
                    cmp[c] = pd.to_numeric(cmp[c], errors="coerce")
            if "rank_a" in cmp.columns and "rank_b" in cmp.columns:
                cmp["delta_rank"] = cmp["rank_b"] - cmp["rank_a"]
            if "p1_a" in cmp.columns and "p1_b" in cmp.columns:
                cmp["delta_p1"] = cmp["p1_b"] - cmp["p1_a"]
            sort_cols = [c for c in ["delta_p1", "delta_rank"] if c in cmp.columns]
            if not sort_cols:
                st.info(
                    "Nothing to sort by yet (need rank/probability columns in both runs)."
                )
                show = [
                    c
                    for c in [
                        "country_name",
                        "alpha_3",
                        "p1_a",
                        "p1_b",
                        "rank_a",
                        "rank_b",
                    ]
                    if c in cmp.columns
                ]
                st.dataframe(cmp[show].head(200), width="stretch", height=520)
            else:
                sort_by = st.selectbox("Sort by", options=sort_cols, index=0)
                st.dataframe(
                    cmp.sort_values(sort_by, ascending=False).head(200)[
                        [
                            c
                            for c in [
                                "country_name",
                                "alpha_3",
                                "p1_a",
                                "p1_b",
                                "delta_p1",
                                "rank_a",
                                "rank_b",
                                "delta_rank",
                            ]
                            if c in cmp.columns
                        ]
                    ],
                    width="stretch",
                    height=520,
                )

with tabs[5]:
    st.subheader("Map")
    if not all(c in df.columns for c in ["latitude", "longitude"]):
        st.info("This run CSV does not include `latitude`/`longitude` columns.")
    else:
        tmp = df.copy()
        tmp["latitude"] = pd.to_numeric(tmp["latitude"], errors="coerce")
        tmp["longitude"] = pd.to_numeric(tmp["longitude"], errors="coerce")
        tmp = tmp.dropna(subset=["latitude", "longitude"]).copy()

        if tmp.empty:
            st.info("No rows with valid lat/lon to plot.")
        else:
            st.caption(
                "Interactive map (PyDeck). Use toggles to overlay: training data, Bayes top-N, heuristic candidates, and your guessed list."
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                show_all = st.checkbox("Show all countries (gray)", value=True)
                show_training = st.checkbox(
                    "Show training rows (observed counts)", value=True
                )
            with c2:
                show_bayes_top = st.checkbox("Show Bayes top-N", value=True)
                bayes_top_n = st.slider("Bayes top N", 5, 200, 50)
            with c3:
                show_heur = st.checkbox("Show heuristic candidates", value=False)
                show_guesses = st.checkbox("Highlight guessed countries", value=True)

            show_data_points = st.checkbox(
                "Show countries present in COUNTRIES_LISTENS (incl unknown)",
                value=True,
            )

            guess_text = ""
            if show_guesses:
                guess_text = st.text_area(
                    "Guessed countries (comma or newline separated)",
                    value="",
                    placeholder="France\nPoland\nGermany\n...",
                    height=90,
                )

            guess_set = {
                g.strip().lower()
                for g in re.split(r"[,\n]+", guess_text or "")
                if g.strip()
            }

            tmp["lat"] = tmp["latitude"]
            tmp["lon"] = tmp["longitude"]
            tmp["country"] = tmp[country_col].astype(str)
            tmp["alpha_3_upper"] = (
                tmp["alpha_3"].astype(str).str.upper()
                if "alpha_3" in tmp.columns
                else ""
            )

            # Training labels from observed counts (if present).
            is_train = (
                tmp["bayes_y_observed"].notna()
                if "bayes_y_observed" in tmp.columns
                else pd.Series(False, index=tmp.index)
            )
            train_y = (
                pd.to_numeric(tmp["bayes_y_observed"], errors="coerce")
                if "bayes_y_observed" in tmp.columns
                else pd.Series(np.nan, index=tmp.index)
            )
            is_train_one = is_train & (train_y == 1)
            is_train_zero = is_train & (train_y == 0)
            is_train_multi = is_train & (train_y >= 2)

            is_bayes_top = pd.Series(False, index=tmp.index)
            if show_bayes_top and "bayes_rank" in tmp.columns:
                r = pd.to_numeric(tmp["bayes_rank"], errors="coerce")
                is_bayes_top = r.notna() & (r <= float(bayes_top_n))

            is_heur_cand = pd.Series(False, index=tmp.index)
            if show_heur and "is_single_listen_candidate" in tmp.columns:
                is_heur_cand = tmp["is_single_listen_candidate"] == True  # noqa: E712

            is_guess = pd.Series(False, index=tmp.index)
            if guess_set:
                is_guess = tmp["country"].astype(str).str.lower().isin(guess_set)

            # Colors (RGBA)
            COL_ALL = [140, 140, 140, 40]
            COL_TRAIN_ZERO = [239, 68, 68, 210]  # red
            COL_TRAIN_ONE = [16, 185, 129, 230]  # green
            COL_TRAIN_MULTI = [245, 158, 11, 230]  # amber
            COL_BAYES_TOP = [59, 130, 246, 210]  # blue
            COL_HEUR = [168, 85, 247, 210]  # purple
            COL_GUESS = [236, 72, 153, 235]  # pink
            COL_DATA = [34, 211, 238, 220]  # cyan

            radius_mode = st.radio(
                "Point sizing",
                options=["constant", "constant (big)", "by P(Y=1)", "by mu"],
                horizontal=True,
                index=1,
            )

            p1 = (
                pd.to_numeric(tmp.get("bayes_p_one_mean"), errors="coerce")
                if "bayes_p_one_mean" in tmp.columns
                else pd.Series(np.nan, index=tmp.index)
            )
            mu = (
                pd.to_numeric(tmp.get("bayes_mu_mean"), errors="coerce")
                if "bayes_mu_mean" in tmp.columns
                else pd.Series(np.nan, index=tmp.index)
            )

            if radius_mode == "by P(Y=1)":
                r = (p1.fillna(0.0).clip(lower=0.0, upper=0.37) / 0.37) ** 0.6
                tmp["radius_m"] = (8000 + 52000 * r).astype(float)
            elif radius_mode == "by mu":
                rr = np.log1p(mu.fillna(0.0).clip(lower=0.0))
                rr = (rr - float(rr.min())) / (float(rr.max()) - float(rr.min()) + 1e-9)
                tmp["radius_m"] = (8000 + 52000 * (rr**0.7)).astype(float)
            elif radius_mode == "constant (big)":
                tmp["radius_m"] = 52000.0
            else:
                tmp["radius_m"] = 18000.0

            size_scale = st.slider("Size scale", 0.5, 3.0, 1.5, 0.1)
            tmp["radius_m"] = tmp["radius_m"] * float(size_scale)

            tooltip_cols = [
                "country",
                "alpha_3_upper",
                "bayes_rank",
                "bayes_p_one_mean",
                "bayes_mu_mean",
                "uk_distance_km",
                "bayes_y_observed",
                "model_rank",
                "total_score",
            ]
            tooltip_cols = [
                c
                for c in tooltip_cols
                if c in tmp.columns or c in ["country", "alpha_3_upper"]
            ]
            tooltip = {
                "html": "<br/>".join([f"<b>{c}</b>: {{{c}}}" for c in tooltip_cols]),
                "style": {"backgroundColor": "rgba(20,20,20,0.85)", "color": "white"},
            }

            layers: list[pdk.Layer] = []

            if show_all:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=tmp,
                        get_position="[lon, lat]",
                        get_radius="radius_m",
                        get_fill_color=COL_ALL,
                        pickable=False,
                        stroked=False,
                    )
                )

            if show_training and is_train.any():
                for mask, color in [
                    (is_train_zero, COL_TRAIN_ZERO),
                    (is_train_one, COL_TRAIN_ONE),
                    (is_train_multi, COL_TRAIN_MULTI),
                ]:
                    sub = tmp.loc[mask].copy()
                    if sub.empty:
                        continue
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=sub,
                            get_position="[lon, lat]",
                            get_radius="radius_m",
                            get_fill_color=color,
                            pickable=True,
                            stroked=True,
                            get_line_color=[0, 0, 0, 180],
                            line_width_min_pixels=1,
                        )
                    )

            if show_data_points and "bayes_in_country_listens_map" in tmp.columns:
                mask = tmp["bayes_in_country_listens_map"] == True  # noqa: E712
                sub = tmp.loc[mask].copy()
                if not sub.empty:
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=sub,
                            get_position="[lon, lat]",
                            get_radius="radius_m",
                            get_fill_color=COL_DATA,
                            pickable=True,
                            stroked=True,
                            get_line_color=[0, 0, 0, 220],
                            line_width_min_pixels=1,
                        )
                    )

            if show_bayes_top and is_bayes_top.any():
                sub = tmp.loc[is_bayes_top].copy()
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=sub,
                        get_position="[lon, lat]",
                        get_radius="radius_m",
                        get_fill_color=COL_BAYES_TOP,
                        pickable=True,
                        stroked=True,
                        get_line_color=[0, 0, 0, 200],
                        line_width_min_pixels=1,
                    )
                )

            if show_heur and is_heur_cand.any():
                sub = tmp.loc[is_heur_cand].copy()
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=sub,
                        get_position="[lon, lat]",
                        get_radius="radius_m",
                        get_fill_color=COL_HEUR,
                        pickable=True,
                        stroked=True,
                        get_line_color=[0, 0, 0, 200],
                        line_width_min_pixels=1,
                    )
                )

            if show_guesses and is_guess.any():
                sub = tmp.loc[is_guess].copy()
                sub["radius_m"] = sub["radius_m"] * 1.35
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=sub,
                        get_position="[lon, lat]",
                        get_radius="radius_m",
                        get_fill_color=COL_GUESS,
                        pickable=True,
                        stroked=True,
                        get_line_color=[0, 0, 0, 240],
                        line_width_min_pixels=2,
                    )
                )

            view_state = pdk.ViewState(latitude=52.0, longitude=5.0, zoom=2.2, pitch=0)
            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_style=None,
                ),
                use_container_width=True,
            )

            with st.expander("Legend", expanded=False):
                st.markdown(
                    "- **Gray**: all countries\n"
                    "- **Green**: training (Y=1)\n"
                    "- **Red**: training (Y=0)\n"
                    "- **Amber**: training (Y>=2)\n"
                    "- **Cyan**: present in COUNTRIES_LISTENS (incl unknown)\n"
                    "- **Blue**: Bayes top-N\n"
                    "- **Purple**: heuristic candidate zone\n"
                    "- **Pink**: guessed countries (pasted list)\n"
                )

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FrameSchema:
    """
    Lightweight DataFrame schema:
    - required: must exist as columns
    - optional: may exist; used for diagnostics / viewers
    """

    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    @property
    def all_columns(self) -> tuple[str, ...]:
        # Preserve order, dedup.
        seen: set[str] = set()
        out: list[str] = []
        for c in (*self.required, *self.optional):
            if c in seen:
                continue
            seen.add(c)
            out.append(c)
        return tuple(out)


DATASET_SCHEMA = FrameSchema(
    name="dataset",
    required=("alpha_3",),
    optional=(
        "country_name",
        "population",
        "internet_usage_pct",
        "internet_usage_record_year",
        "english_speakers_pct",
        "languages",
        "latitude",
        "longitude",
        "uk_distance_km",
        # legacy/optional:
        "uk_visits_number",
        "uk_visits_period",
        "uk_spending_millions",
    ),
)


BAYES_RUN_SCHEMA = FrameSchema(
    name="bayes_run",
    required=("alpha_3", "bayes_p_one_mean"),
    optional=(
        "country_name",
        "bayes_label",
        "bayes_y_observed",
        "bayes_rank",
        "bayes_mu_mean",
        "bayes_mu_hdi_low",
        "bayes_mu_hdi_high",
        "bayes_p_one_hdi_low",
        "bayes_p_one_hdi_high",
        "bayes_lp_mean",
        "bayes_lp_hdi_low",
        "bayes_lp_hdi_high",
        # feature debug:
        "bayes_x_log_population",
        "bayes_x_internet_rate",
        "bayes_x_log1p_uk_distance_km",
        "bayes_z_log_population",
        "bayes_z_internet_rate",
        "bayes_z_log1p_uk_distance_km",
        # inputs (may or may not be included in run CSV):
        "population",
        "internet_usage_pct",
        "english_speakers_pct",
        "latitude",
        "longitude",
        "uk_distance_km",
        # metadata / toggles:
        "run_label",
        "run_id",
        "bayes_use_distance",
        "bayes_use_english",
        "bayes_draws_cli",
        "bayes_tune_cli",
        "bayes_target_accept_cli",
        "bayes_seed_cli",
        "bayes_hdi_prob_cli",
        "bayes_rhat_max",
        "bayes_ess_bulk_min",
        "bayes_train_n",
        "bayes_train_n_zero",
        "bayes_train_n_one",
        "bayes_train_n_multi",
        "bayes_number_of_countries_with_listens",
        "bayes_aggregate_sigma",
        "bayes_number_of_countries_with_one_listen",
        "bayes_aggregate_one_sigma",
    ),
)


def validate_frame(
    df: pd.DataFrame, schema: FrameSchema, *, allow_empty: bool = True
) -> dict:
    """
    Return a small validation report (JSON-serializable).
    Does not mutate the dataframe.
    """
    cols = list(df.columns)
    missing = [c for c in schema.required if c not in df.columns]
    present_optional = [c for c in schema.optional if c in df.columns]
    report = {
        "schema": schema.name,
        "rows": int(len(df)),
        "cols": int(len(cols)),
        "missing_required": missing,
        "present_optional": present_optional,
        "ok": (not missing) and (allow_empty or len(df) > 0),
    }
    return report


def require_schema(df: pd.DataFrame, schema: FrameSchema) -> None:
    missing = [c for c in schema.required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame does not satisfy schema '{schema.name}'. Missing columns: {missing}"
        )

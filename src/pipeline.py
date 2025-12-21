from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.bayes_model import BayesFitConfig, fit_poisson_glm_p_one
from src.config import (
    COUNTRIES_LISTENS,
    NUMBER_OF_COUNTRIES_WITH_LISTENS,
    TOTAL_NUMBER_OF_COUNTRIES_WITH_ONE_LISTEN,
)
from src.data import (
    add_internet_usage,
    add_languages_and_population,
    add_uk_distance,
    countries,
    load_dataset_csv,
)
from src.model import predict
from src.schema import BAYES_RUN_SCHEMA, DATASET_SCHEMA, require_schema


@dataclass(frozen=True)
class DatasetBuildConfig:
    dataset_csv: str | None = None
    use_distance: bool = True
    exclude_alpha3: tuple[str, ...] = ("GBR",)


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


META_VERSION = 1


def _slugify(text: str) -> str:
    import re

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "run"


async def build_dataset(cfg: DatasetBuildConfig) -> pd.DataFrame:
    """
    Build a country dataset either from a provided dataset CSV (preferred) or by
    fetching base country features and merging raw sources.
    """
    if cfg.dataset_csv:
        df = load_dataset_csv(cfg.dataset_csv)
    else:
        countries_array = await countries()
        df = pd.DataFrame({"alpha_3": countries_array})
        df = await add_languages_and_population(df)
        df = add_internet_usage(df)
        if bool(cfg.use_distance):
            df = add_uk_distance(df)

    # Exclusions (UK etc.)
    if cfg.exclude_alpha3:
        excl = {c.upper() for c in cfg.exclude_alpha3}
        df = df.loc[~df["alpha_3"].astype(str).str.upper().isin(excl)].copy()

    require_schema(df, DATASET_SCHEMA)
    return df


def run_heuristic(
    df: pd.DataFrame,
    *,
    use_language_factor: bool,
    language_english_factor: float,
    language_euro_latin_factor: float,
    language_other_factor: float,
) -> pd.DataFrame:
    require_schema(df, DATASET_SCHEMA)
    return predict(
        df,
        use_language_factor=use_language_factor,
        language_english_factor=language_english_factor,
        language_euro_latin_factor=language_euro_latin_factor,
        language_other_factor=language_other_factor,
    )


def run_bayes(
    df: pd.DataFrame,
    *,
    use_distance: bool,
    draws: int,
    tune: int,
    target_accept: float,
    seed: int | None,
    hdi_prob: float,
) -> pd.DataFrame:
    require_schema(df, DATASET_SCHEMA)
    return fit_poisson_glm_p_one(
        df,
        country_listens=COUNTRIES_LISTENS,
        number_of_countries_with_listens=int(NUMBER_OF_COUNTRIES_WITH_LISTENS),
        number_of_countries_with_one_listen=int(
            TOTAL_NUMBER_OF_COUNTRIES_WITH_ONE_LISTEN
        ),
        use_distance=bool(use_distance),
        config=BayesFitConfig(
            draws=int(draws),
            tune=int(tune),
            target_accept=float(target_accept),
            seed=seed,
            hdi_prob=float(hdi_prob),
        ),
    )


def write_run_artifacts(
    out_df: pd.DataFrame,
    *,
    runs_dir: str,
    label: str,
    meta: dict[str, Any],
) -> tuple[str, str]:
    """
    Write run CSV + a sidecar metadata JSON.
    Returns (csv_path, meta_path).
    """
    os.makedirs(runs_dir, exist_ok=True)
    ts = _utc_ts()
    slug = _slugify(label)
    run_id = f"{slug}_{ts}"

    csv_path = os.path.join(runs_dir, f"{run_id}.csv")
    meta_path = os.path.join(runs_dir, f"{run_id}.meta.json")

    # Persist lineage in the CSV itself as well (makes downstream viewers simpler).
    out = out_df.copy()
    out["run_label"] = label
    out["run_id"] = run_id

    # If this looks like a Bayes run, make sure we include a deterministic rank column.
    # (CLI adds this, but notebooks that call `run_bayes()` directly might not.)
    if "bayes_p_one_mean" in out.columns and "bayes_rank" not in out.columns:
        p1 = pd.to_numeric(out["bayes_p_one_mean"], errors="coerce")
        out["bayes_rank"] = p1.rank(ascending=False, method="min").astype("Int64")

    out.to_csv(csv_path, index=False)

    payload = {
        "meta_version": META_VERSION,
        "run_id": run_id,
        "run_label": label,
        "timestamp_utc": ts,
        **meta,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return csv_path, meta_path


def bayes_run_to_artifacts(
    pred_df: pd.DataFrame,
    bayes_df: pd.DataFrame,
    *,
    runs_dir: str,
    label: str,
    meta: dict[str, Any],
) -> tuple[str, str]:
    out = pd.merge(pred_df, bayes_df, on="alpha_3", how="left")
    require_schema(out, BAYES_RUN_SCHEMA)
    return write_run_artifacts(out, runs_dir=runs_dir, label=label, meta=meta)

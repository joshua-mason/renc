import numpy as np
import pandas as pd
import pycountry
import re
from typing import Iterable

GUARANTEED_MULTI_LISTEN_ZONE_RANK = 89
SINGLE_LISTEN_ZONE_END_RANK = 95
DATE_OF_RECORD = "25-05-2025"

EUROPEAN_AND_LATIN_LANGUAGES = {
    "spanish",
    "french",
    "italian",
    "portuguese",
    "romanian",
    "catalan",
    "galician",
    "basque",
    "german",
    "dutch",
    "flemish",
    "polish",
    "czech",
    "slovak",
    "slovenian",
    "croatian",
    "serbian",
    "bosnian",
    "montenegrin",
    "albanian",
    "maltese",
    "hungarian",
    "norwegian",
    "swedish",
    "danish",
    "finnish",
    "estonian",
    "latvian",
    "lithuanian",
    "icelandic",
    "irish",
    "scots gaelic",
    "welsh",
}


def _safe_log10(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return float(np.log10(value))


def _normalize_language_name(language: str) -> str | None:
    raw = str(language).strip()
    if not raw:
        return None
    lowered = raw.lower()

    # If this is an ISO code (e.g. "eng", "en"), try to resolve to the canonical name.
    if re.fullmatch(r"[a-z]{2,3}", lowered):
        try:
            lang = (
                pycountry.languages.get(alpha_2=lowered)
                or pycountry.languages.get(alpha_3=lowered)
                or pycountry.languages.get(bibliographic=lowered)
                or pycountry.languages.get(terminology=lowered)
            )
            if lang and getattr(lang, "name", None):
                return str(lang.name).strip().lower()
        except (KeyError, AttributeError):
            pass

    # Otherwise do a fuzzy lookup (handles abbreviations/variants).
    try:
        lang = pycountry.languages.lookup(raw)
        if getattr(lang, "name", None):
            return str(lang.name).strip().lower()
    except LookupError:
        pass

    # Fall back to the original token.
    return lowered


def _language_factor(
    languages: Iterable[str] | str | None,
    *,
    english_factor: float,
    euro_latin_factor: float,
    other_factor: float,
) -> float:
    if not languages:
        return float(other_factor)
    if isinstance(languages, str):
        languages_iterable = (languages,)
    else:
        languages_iterable = languages
    cleaned = {
        normalized
        for language in languages_iterable
        if language is not None
        for normalized in [_normalize_language_name(language)]
        if normalized
    }
    if "english" in cleaned:
        return float(english_factor)
    if cleaned & EUROPEAN_AND_LATIN_LANGUAGES:
        return float(euro_latin_factor)
    return float(other_factor)


# TODO: We may eventually want to cache this or pull the country name earlier.
def _resolve_country_name(alpha3: str | None) -> str | None:
    if not alpha3:
        return None
    try:
        country = pycountry.countries.get(alpha_3=alpha3)
    except (KeyError, AttributeError):
        return None
    return getattr(country, "name", None)


#     alpha_3, languages ,population,internet_usage_pct, internet_usage_record_year,languages, uk_visits_number, uk_spending_millions
def predict(
    df: pd.DataFrame,
    *,
    use_language_factor: bool = True,
    language_english_factor: float = 1.161,
    language_euro_latin_factor: float = 1.042,
    language_other_factor: float = 0.512,
    uk_missing_strategy: str = "p10",
    uk_floor: float = 0.05,
) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["population"] = pd.to_numeric(df_copy.get("population"), errors="coerce")
    df_copy["internet_usage_pct"] = pd.to_numeric(
        df_copy.get("internet_usage_pct"), errors="coerce"
    )
    df_copy["uk_visits_number"] = pd.to_numeric(
        df_copy.get("uk_visits_number"), errors="coerce"
    )

    df_copy["pop_score"] = df_copy["population"].apply(_safe_log10)
    df_copy["net_score"] = df_copy["internet_usage_pct"] / 100
    # UK visits are a strong proxy for cultural closeness. Missing is likely "unknown / not
    # tracked (often low)" rather than "average", so we impute conservatively.
    df_copy["uk_visits_missing"] = df_copy["uk_visits_number"].isna()
    df_copy["uk_score_raw"] = np.log10(df_copy["uk_visits_number"].clip(lower=0) + 1)

    strategy = str(uk_missing_strategy).lower().strip()
    if strategy not in {"p10", "p5", "median", "zero", "ignore"}:
        strategy = "p10"

    observed_raw = df_copy.loc[~df_copy["uk_visits_missing"], "uk_score_raw"].dropna()
    if observed_raw.empty:
        impute_value = 0.0
    elif strategy == "p5":
        impute_value = float(observed_raw.quantile(0.05))
    elif strategy == "p10":
        impute_value = float(observed_raw.quantile(0.10))
    elif strategy == "median":
        impute_value = float(observed_raw.median())
    elif strategy == "zero":
        impute_value = 0.0
    else:  # ignore
        impute_value = float("nan")

    df_copy["uk_missing_strategy"] = strategy
    df_copy["uk_floor"] = float(uk_floor)
    df_copy["uk_impute_value"] = impute_value

    # Fill missing with a conservative value, then map to (floor..1] so we never hard-zero
    # the full multiplicative score purely due to UK-visits.
    df_copy["uk_score_raw_imputed"] = df_copy["uk_score_raw"].copy()
    if strategy != "ignore":
        df_copy["uk_score_raw_imputed"] = df_copy["uk_score_raw_imputed"].fillna(
            impute_value
        )

    raw_min = float(df_copy["uk_score_raw_imputed"].min(skipna=True))
    raw_max = float(df_copy["uk_score_raw_imputed"].max(skipna=True))
    denom = raw_max - raw_min
    if not np.isfinite(denom) or denom <= 0:
        uk_scaled = pd.Series(1.0, index=df_copy.index)
    else:
        uk_scaled = (df_copy["uk_score_raw_imputed"] - raw_min) / denom
        uk_scaled = uk_scaled.clip(lower=0.0, upper=1.0)

    floor = float(uk_floor)
    if not np.isfinite(floor) or floor < 0:
        floor = 0.0
    if floor > 0.5:
        floor = 0.5

    df_copy["uk_score"] = floor + (1.0 - floor) * uk_scaled.fillna(floor)
    df_copy["language_english_factor"] = float(language_english_factor)
    df_copy["language_euro_latin_factor"] = float(language_euro_latin_factor)
    df_copy["language_other_factor"] = float(language_other_factor)
    df_copy["language_factor"] = df_copy["languages"].apply(
        lambda langs: _language_factor(
            langs,
            english_factor=float(language_english_factor),
            euro_latin_factor=float(language_euro_latin_factor),
            other_factor=float(language_other_factor),
        )
    )
    df_copy["use_language_factor"] = bool(use_language_factor)

    # Make the UK maximal UK-affinity. (UK self-row often has no "visits to UK".)
    if "alpha_3" in df_copy.columns:
        df_copy.loc[df_copy["alpha_3"] == "GBR", "uk_score"] = 1.0

    df_copy["total_score_base"] = (
        df_copy["pop_score"].fillna(0)
        * df_copy["net_score"].fillna(0)
        * df_copy["uk_score"].fillna(floor)
    )
    df_copy["total_score_with_language"] = df_copy["total_score_base"] * df_copy[
        "language_factor"
    ].fillna(0.5)
    df_copy["total_score_no_language"] = df_copy["total_score_base"]
    df_copy["total_score"] = (
        df_copy["total_score_with_language"]
        if use_language_factor
        else df_copy["total_score_no_language"]
    )

    df_sorted = df_copy.sort_values(
        by="total_score", ascending=False, ignore_index=True
    )
    df_sorted["model_rank"] = df_sorted.index + 1
    zone_start = GUARANTEED_MULTI_LISTEN_ZONE_RANK + 1
    df_sorted["is_single_listen_candidate"] = df_sorted["model_rank"].between(
        zone_start, SINGLE_LISTEN_ZONE_END_RANK
    )
    df_sorted["country_name"] = df_sorted["alpha_3"].apply(_resolve_country_name)
    df_sorted["predicted_single_listen_country"] = df_sorted.apply(
        lambda row: row["country_name"] if row["is_single_listen_candidate"] else None,
        axis=1,
    )
    df_sorted["prediction_date"] = DATE_OF_RECORD
    return df_sorted

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


def _language_factor(languages: Iterable[str] | str | None) -> float:
    if not languages:
        return 0.5
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
        return 1.5
    if cleaned & EUROPEAN_AND_LATIN_LANGUAGES:
        return 1.1
    return 0.5


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
def predict(df: pd.DataFrame, *, use_language_factor: bool = True) -> pd.DataFrame:
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
    df_copy["uk_score_raw"] = df_copy["uk_visits_number"].apply(_safe_log10)
    uk_score_median = df_copy["uk_score_raw"].median(skipna=True)
    if pd.isna(uk_score_median):
        uk_score_median = 1.0
    # Missing UK visits shouldn't zero-out a country; use a "normal" baseline.
    df_copy["uk_score"] = df_copy["uk_score_raw"].fillna(float(uk_score_median))
    df_copy["language_factor"] = df_copy["languages"].apply(_language_factor)
    df_copy["use_language_factor"] = bool(use_language_factor)

    # Make the UK maximal UK-affinity. (UK self-row often has no "visits to UK".)
    if "alpha_3" in df_copy.columns:
        current_max_uk_score = float(df_copy["uk_score"].max())
        df_copy.loc[df_copy["alpha_3"] == "GBR", "uk_score"] = current_max_uk_score

    df_copy["total_score_base"] = (
        df_copy["pop_score"].fillna(0)
        * df_copy["net_score"].fillna(0)
        * df_copy["uk_score"].fillna(float(uk_score_median))
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

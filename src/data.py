import pandas as pd
from pandas.core.base import ExtensionArray
import requests
from src.log import log
import pycountry
import os
import math
from pathlib import Path

COUNTRIES_LIST_URL: str = (
    "https://gist.githubusercontent.com/kalinchernev/486393efcca01623b18d/raw/daa24c9fea66afb7d68f8d69f0c4b8eeb9406e83/countries"
)
COUNTRIES_REST_API: str = "https://restcountries.com/v3.1/all"


def _resolve_data_path(relative_path: str) -> str:
    """
    Prefer the new `data/` directory convention, but remain backwards-compatible.

    Examples:
    - data/raw/internetusage.csv (preferred)
    - raw/internetusage.csv (fallback)
    """
    # Resolve relative to repo root so this works from any working directory
    # (e.g. notebooks run from `notebooks/`).
    repo_root = Path(__file__).resolve().parents[1]

    preferred = repo_root / "data" / relative_path
    if preferred.exists():
        return str(preferred)

    fallback = repo_root / relative_path
    return str(fallback)


async def countries() -> ExtensionArray:
    return pd.array([country.alpha_3 for country in pycountry.countries])


def load_dataset_csv(path: str) -> pd.DataFrame:
    """
    Load a pre-built country feature dataset (preferred for reproducibility).

    Required:
    - alpha_3 OR country_name (will be resolved to alpha-3 if possible)

    Optional (used by models/viewers):
    - population, internet_usage_pct, internet_usage_record_year
    - uk_visits_number, uk_visits_period, uk_spending_millions
    - languages
    - latitude, longitude, uk_distance_km
    """
    df = pd.read_csv(path)
    if "alpha_3" not in df.columns:
        if "country_name" not in df.columns:
            raise ValueError(
                "Dataset CSV must include either 'alpha_3' or 'country_name'."
            )

        def _to_alpha3(name: str | None) -> str | None:
            if not name or pd.isna(name):
                return None
            try:
                return pycountry.countries.lookup(str(name)).alpha_3
            except LookupError:
                try:
                    return pycountry.countries.search_fuzzy(str(name))[0].alpha_3
                except LookupError:
                    return None

        df["alpha_3"] = df["country_name"].apply(_to_alpha3)

    df["alpha_3"] = df["alpha_3"].astype(str).str.upper()
    df = df.dropna(subset=["alpha_3"]).drop_duplicates(subset=["alpha_3"], keep="first")
    return df


async def add_languages_and_population(df: pd.DataFrame) -> pd.DataFrame:
    languages_response = requests.api.get(
        COUNTRIES_REST_API + "?status=true&fields=languages,cca3,population,latlng",
        timeout=30,
    )
    body = languages_response.json()
    rows_list = []
    for country in body:
        country_code = country.get("cca3")
        if not country_code:
            continue
        # restcountries returns a dict like {"eng": "English", "fra": "French"}.
        # Use the values (human-readable names) so downstream logic can match.
        languages = (
            list(country.get("languages").values())
            if isinstance(country.get("languages"), dict)
            else []
        )
        latlng = (
            country.get("latlng") if isinstance(country.get("latlng"), list) else []
        )
        lat = latlng[0] if len(latlng) >= 2 else None
        lon = latlng[1] if len(latlng) >= 2 else None
        row = {
            "alpha_3": country_code,
            "languages": languages,
            "population": country.get("population"),
            "latitude": lat,
            "longitude": lon,
        }
        rows_list.append(row)
    languages_df = pd.DataFrame(rows_list)

    return pd.merge(df, languages_df, on="alpha_3", how="left")


def add_internet_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads raw/internetusage.csv and adds the internet usage data to the given df,
    joining on the alpha_3 column (which matches 'Country Code' from the csv).
    """
    usage_path = _resolve_data_path(os.path.join("raw", "internetusage.csv"))
    usage_df = pd.read_csv(usage_path, skipinitialspace=True, dtype=str)
    usage_df = usage_df.rename(columns={"Country Code": "alpha_3"})

    years = usage_df.columns[4:]

    def get_latest_usage(row):
        for year in reversed(years):
            value = row[year]
            if pd.notnull(value) and str(value).strip() != "":
                return float(value), year
        return None, None

    usage_df[["internet_usage_pct", "internet_usage_record_year"]] = usage_df.apply(
        lambda row: pd.Series(get_latest_usage(row)), axis=1
    )

    usage_add = usage_df[
        ["alpha_3", "internet_usage_pct", "internet_usage_record_year"]
    ].drop_duplicates(subset="alpha_3")

    return pd.merge(df, usage_add, on="alpha_3", how="left")


def add_uk_visits_abroad(df: pd.DataFrame) -> pd.DataFrame:
    visits_path = _resolve_data_path(os.path.join("raw", "ukvisitsabroad.csv"))
    visits_df = pd.read_csv(visits_path, sep="\t", dtype=str)

    def parse_number(value: str | None) -> float | None:
        if value is None or pd.isna(value):
            return None
        normalized = str(value).replace(",", "").replace("%", "")
        try:
            return float(normalized)
        except ValueError:
            return None

    def resolve_alpha3(name: str) -> str | None:
        if not name:
            return None
        try:
            return pycountry.countries.lookup(name).alpha_3
        except LookupError:
            try:
                return pycountry.countries.search_fuzzy(name)[0].alpha_3
            except LookupError:
                return None

    rows = []
    for _, row in visits_df.iterrows():
        country = row.get("Country [Note 3]")
        alpha3 = resolve_alpha3(country)
        if not alpha3:
            continue
        rows.append(
            {
                "alpha_3": alpha3,
                "uk_visits_period": row.get("Period"),
                "uk_visits_number": parse_number(row.get("Number of visits")),
                "uk_spending_millions": parse_number(
                    row.get("Spending in £millions[Note 4]")
                ),
            }
        )

    # The raw table contains multiple periods per country. If we merge as-is, we
    # create duplicate countries via a many-to-one join. For this V0 model we
    # keep a single "best available" row per country by taking the max visits.
    visits_add = pd.DataFrame(rows)
    visits_add = visits_add.sort_values(
        by=["uk_visits_number"], ascending=False, na_position="last"
    ).drop_duplicates(subset="alpha_3", keep="first")
    return pd.merge(df, visits_add, on="alpha_3", how="left")


def _haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float, /, *, r_km: float = 6371.0
) -> float:
    """
    Great-circle distance between two points on Earth (kilometers).
    Inputs are degrees.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return float(r_km * c)


def add_uk_distance(
    df: pd.DataFrame,
    *,
    uk_lat: float = 54.0,
    uk_lon: float = -2.0,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """
    Adds `uk_distance_km` using per-country `latitude`/`longitude`.

    - If lat/lon are missing for a country, uk_distance_km will be NaN.
    - UK reference is a fixed centroid-ish point (defaults near UK center).
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        df["uk_distance_km"] = float("nan")
        return df

    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)

    out = []
    for la, lo in zip(lat, lon, strict=False):
        if not (math.isfinite(float(la)) and math.isfinite(float(lo))):
            out.append(float("nan"))
            continue
        out.append(_haversine_km(float(uk_lat), float(uk_lon), float(la), float(lo)))

    df["uk_distance_km"] = out
    return df

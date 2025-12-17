import pandas as pd
from pandas.core.base import ExtensionArray
import requests
from src.log import log
import pycountry

COUNTRIES_LIST_URL: str = (
    "https://gist.githubusercontent.com/kalinchernev/486393efcca01623b18d/raw/daa24c9fea66afb7d68f8d69f0c4b8eeb9406e83/countries"
)
COUNTRIES_REST_API: str = "https://restcountries.com/v3.1/all"


async def countries() -> ExtensionArray:
    return pd.array([country.alpha_3 for country in pycountry.countries])


async def add_languages_and_population(df: pd.DataFrame) -> pd.DataFrame:
    languages_response = requests.api.get(
        COUNTRIES_REST_API + "?status=true&fields=languages,cca3,population"
    )
    body = languages_response.json()
    rows_list = []
    for country in body:
        country_code = country.get("cca3")
        if not country_code:
            continue
        languages = (
            list(country.get("languages").keys())
            if isinstance(country.get("languages"), dict)
            else []
        )
        row = {
            "alpha_3": country_code,
            "languages": languages,
            "population": country.get("population"),
        }
        rows_list.append(row)
    languages_df = pd.DataFrame(rows_list)

    return pd.merge(df, languages_df, on="alpha_3", how="left")


def add_internet_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads raw/internetusage.csv and adds the internet usage data to the given df,
    joining on the alpha_3 column (which matches 'Country Code' from the csv).
    """
    usage_df = pd.read_csv("raw/internetusage.csv", skipinitialspace=True, dtype=str)
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
    visits_df = pd.read_csv("raw/ukvisitsabroad.csv", sep="\t", dtype=str)

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

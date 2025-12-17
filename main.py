from src.data import (
    add_internet_usage,
    add_uk_visits_abroad,
    countries,
    add_languages_and_population,
)
import pandas as pd
import asyncio

from src.model import predict

# TODO check Gambon
CORRECT_COUNTRIES = ["San Marino", "Bhutan", "Fiji", "Faroe Islands", "Gabon"]


async def main():
    countries_array = await countries()
    print(f"Hello from renc! ${countries_array}")
    df = pd.DataFrame()
    df["alpha_3"] = countries_array
    countries_with_languages = await add_languages_and_population(df)
    print(countries_with_languages)
    with_internet = add_internet_usage(countries_with_languages)
    print(with_internet)
    with_uk_visits = add_uk_visits_abroad(with_internet)
    print(with_uk_visits)

    out = predict(with_uk_visits)
    out["seen_in_listens"] = out["country_name"].isin(CORRECT_COUNTRIES)
    out.to_csv("out.csv")
    print(out)


if __name__ == "__main__":
    asyncio.run(main())

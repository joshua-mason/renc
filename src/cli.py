import argparse
import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from src.config import CORRECT_COUNTRIES
from src.data import (
    add_internet_usage,
    add_languages_and_population,
    add_uk_visits_abroad,
    countries,
)
from src.model import predict


@dataclass(frozen=True)
class RunResult:
    label: str
    run_id: str
    csv_path: str


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "run"


async def _generate_run(
    label: str,
    runs_dir: str,
    *,
    use_language_factor: bool,
    uk_missing_strategy: str,
    uk_floor: float,
) -> RunResult:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(label)
    # Put the human-readable label first so run files are easier to scan.
    run_id = f"{slug}_{ts}"

    os.makedirs(runs_dir, exist_ok=True)
    csv_path = os.path.join(runs_dir, f"{run_id}.csv")

    countries_array = await countries()
    df = pd.DataFrame({"alpha_3": countries_array})

    df = await add_languages_and_population(df)
    df = add_internet_usage(df)
    df = add_uk_visits_abroad(df)

    out = predict(
        df,
        use_language_factor=use_language_factor,
        uk_missing_strategy=uk_missing_strategy,
        uk_floor=uk_floor,
    )
    if "country_name" in out.columns:
        out["seen_in_listens"] = out["country_name"].isin(CORRECT_COUNTRIES)
    out["run_label"] = label
    out["run_id"] = run_id
    out["model_variant"] = "with-language" if use_language_factor else "no-language"
    out["uk_missing_strategy_cli"] = uk_missing_strategy
    out["uk_floor_cli"] = float(uk_floor)

    out.to_csv(csv_path, index=False)
    return RunResult(label=label, run_id=run_id, csv_path=csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="renc")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Generate a labeled model run CSV")
    run.add_argument(
        "--label", required=True, help="Label for this run (e.g. 'v0', 'lang-tweak')"
    )
    run.add_argument(
        "--runs-dir", default="runs", help="Directory to store generated run CSVs"
    )
    run.add_argument(
        "--use-language",
        action="store_true",
        help="Include language_factor in the score (often hurts if listens are mostly from travellers)",
    )
    run.add_argument(
        "--uk-missing",
        default="p10",
        choices=["p10", "p5", "median", "zero", "ignore"],
        help="How to treat missing uk_visits_number (default: p10 conservative imputation).",
    )
    run.add_argument(
        "--uk-floor",
        type=float,
        default=0.05,
        help="Floor for uk_score after scaling (prevents multiplicative collapse).",
    )
    run.add_argument(
        "--launch",
        action="store_true",
        help="Launch Streamlit after generating, preloading this run",
    )

    args = parser.parse_args()

    if args.command == "run":
        result = asyncio.run(
            _generate_run(
                label=args.label,
                runs_dir=args.runs_dir,
                use_language_factor=bool(args.use_language),
                uk_missing_strategy=str(args.uk_missing),
                uk_floor=float(args.uk_floor),
            )
        )
        print(result.csv_path)

        if args.launch:
            # Run inside the same environment. If you invoke this via `uv run renc ...`,
            # Streamlit will be available.
            import subprocess

            subprocess.run(
                [
                    "streamlit",
                    "run",
                    "streamlit_app.py",
                    "--",
                    "--csv",
                    result.csv_path,
                ],
                check=True,
            )


if __name__ == "__main__":
    main()

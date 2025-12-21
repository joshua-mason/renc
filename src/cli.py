import argparse
import asyncio
import math
import os
import re
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, median

import pandas as pd

from src.pipeline import (
    DatasetBuildConfig,
    build_dataset,
    run_bayes,
    run_heuristic,
    write_run_artifacts,
)
from src.config import COUNTRIES_LISTENS
from src.schema import BAYES_RUN_SCHEMA, require_schema


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
    language_english_factor: float,
    language_euro_latin_factor: float,
    language_other_factor: float,
    uk_missing_strategy: str,
    uk_floor: float,
    dataset_csv: str | None = None,
    use_distance: bool = True,
    use_english: bool = True,
) -> RunResult:
    df = await build_dataset(
        DatasetBuildConfig(
            dataset_csv=dataset_csv,
            use_distance=bool(use_distance),
            use_english=bool(use_english),
        )
    )
    out = run_heuristic(
        df,
        use_language_factor=use_language_factor,
        language_english_factor=language_english_factor,
        language_euro_latin_factor=language_euro_latin_factor,
        language_other_factor=language_other_factor,
    )
    out["run_label"] = label
    out["run_id"] = run_id
    out["model_variant"] = "with-language" if use_language_factor else "no-language"
    out["language_english_factor_cli"] = float(language_english_factor)
    out["language_euro_latin_factor_cli"] = float(language_euro_latin_factor)
    out["language_other_factor_cli"] = float(language_other_factor)

    csv_path, _meta_path = write_run_artifacts(
        out,
        runs_dir=runs_dir,
        label=label,
        meta={
            "kind": "heuristic_run",
            "dataset_csv": dataset_csv,
            "use_distance": bool(use_distance),
            "use_english": bool(use_english),
            "exclude_alpha3": ["GBR"],
            "heuristic": {
                "use_language_factor": bool(use_language_factor),
                "language_english_factor": float(language_english_factor),
                "language_euro_latin_factor": float(language_euro_latin_factor),
                "language_other_factor": float(language_other_factor),
            },
        },
    )
    run_id = os.path.splitext(os.path.basename(csv_path))[0]
    return RunResult(label=label, run_id=run_id, csv_path=csv_path)


async def _generate_bayes_run(
    label: str,
    runs_dir: str,
    *,
    use_language_factor: bool,
    language_english_factor: float,
    language_euro_latin_factor: float,
    language_other_factor: float,
    uk_missing_strategy: str,
    uk_floor: float,
    draws: int,
    tune: int,
    target_accept: float,
    seed: int | None,
    hdi_prob: float,
    dataset_csv: str | None = None,
    use_distance: bool = True,
    use_english: bool = True,
) -> RunResult:
    df = await build_dataset(
        DatasetBuildConfig(
            dataset_csv=dataset_csv,
            use_distance=bool(use_distance),
            use_english=bool(use_english),
        )
    )
    out = run_heuristic(
        df,
        use_language_factor=use_language_factor,
        language_english_factor=language_english_factor,
        language_euro_latin_factor=language_euro_latin_factor,
        language_other_factor=language_other_factor,
    )

    # Fit Bayesian model + merge.
    bayes = run_bayes(
        df,
        use_distance=bool(use_distance),
        use_english=bool(use_english),
        draws=int(draws),
        tune=int(tune),
        target_accept=float(target_accept),
        seed=seed,
        hdi_prob=float(hdi_prob),
    )
    out = pd.merge(out, bayes, on="alpha_3", how="left")

    # Rank by Bayesian posterior mean P(Y=1).
    if "bayes_p_one_mean" in out.columns:
        out["bayes_rank"] = (
            pd.to_numeric(out["bayes_p_one_mean"], errors="coerce")
            .rank(ascending=False, method="min")
            .astype("Int64")
        )

    # No extra label sources beyond COUNTRIES_LISTENS.
    out["model_variant"] = "with-language" if use_language_factor else "no-language"
    out["language_english_factor_cli"] = float(language_english_factor)
    out["language_euro_latin_factor_cli"] = float(language_euro_latin_factor)
    out["language_other_factor_cli"] = float(language_other_factor)
    out["bayes_draws_cli"] = int(draws)
    out["bayes_tune_cli"] = int(tune)
    out["bayes_target_accept_cli"] = float(target_accept)
    out["bayes_seed_cli"] = seed if seed is None else int(seed)
    out["bayes_hdi_prob_cli"] = float(hdi_prob)

    csv_path, _meta_path = write_run_artifacts(
        out,
        runs_dir=runs_dir,
        label=label,
        meta={
            "kind": "bayes_run",
            "dataset_csv": dataset_csv,
            "use_distance": bool(use_distance),
            "use_english": bool(use_english),
            "exclude_alpha3": ["GBR"],
            "heuristic": {
                "use_language_factor": bool(use_language_factor),
                "language_english_factor": float(language_english_factor),
                "language_euro_latin_factor": float(language_euro_latin_factor),
                "language_other_factor": float(language_other_factor),
            },
            "bayes": {
                "draws": int(draws),
                "tune": int(tune),
                "target_accept": float(target_accept),
                "seed": seed,
                "hdi_prob": float(hdi_prob),
            },
        },
    )
    run_id = os.path.splitext(os.path.basename(csv_path))[0]
    # Ensure run artifact satisfies expected columns after we inject run_id/run_label during write.
    try:
        require_schema(pd.read_csv(csv_path), BAYES_RUN_SCHEMA)
    except Exception:
        # Keep CLI resilient; Streamlit sidebar schema check will surface details.
        pass
    return RunResult(label=label, run_id=run_id, csv_path=csv_path)


def _metric_wrong_at_rank(
    df_pred: pd.DataFrame, *, predicted_mask: pd.Series, incorrect_names: set[str]
) -> float:
    if "model_rank" not in df_pred.columns or "country_name" not in df_pred.columns:
        return float("nan")
    if predicted_mask.sum() == 0:
        return 0.0
    sub = df_pred.loc[predicted_mask, ["model_rank", "country_name"]].copy()
    sub = sub[sub["country_name"].isin(incorrect_names)]
    ranks = pd.to_numeric(sub["model_rank"], errors="coerce").dropna()
    if ranks.empty:
        return 0.0
    return float((1.0 / ranks).sum())


def _metric_overlap_count(
    df_pred: pd.DataFrame, *, predicted_mask: pd.Series, names: set[str]
) -> int:
    if "country_name" not in df_pred.columns:
        return 0
    if predicted_mask.sum() == 0:
        return 0
    predicted_names = set(df_pred.loc[predicted_mask, "country_name"].dropna())
    return int(len(predicted_names & names))


def _distance_to_zone(rank: float, *, zone_start: float, zone_end: float) -> float:
    """Distance in rank-space to the candidate zone [zone_start, zone_end]."""
    if zone_start <= rank <= zone_end:
        return 0.0
    return float(min(abs(rank - zone_start), abs(rank - zone_end)))


def _zone_closeness(rank: float, *, zone_start: float, zone_end: float) -> float:
    """Continuous closeness score to the zone. 1.0 in-zone, decays with distance."""
    d = _distance_to_zone(rank, zone_start=zone_start, zone_end=zone_end)
    return float(1.0 / (1.0 + d))


def _mean_zone_closeness(
    df_pred: pd.DataFrame, *, names: set[str], zone_start: float, zone_end: float
) -> float:
    """
    Mean closeness of named countries to the candidate zone.

    - For CORRECT: higher is better (want them near/in the zone)
    - For INCORRECT: lower is better (want them far away)
    """
    if not names:
        return float("nan")
    if "country_name" not in df_pred.columns or "model_rank" not in df_pred.columns:
        return float("nan")

    sub = df_pred.loc[df_pred["country_name"].isin(names), ["model_rank"]].copy()
    ranks = pd.to_numeric(sub["model_rank"], errors="coerce").dropna().to_list()
    if not ranks:
        return float("nan")

    values = [
        _zone_closeness(float(r), zone_start=zone_start, zone_end=zone_end)
        for r in ranks
    ]
    return float(mean(values))


async def _tune_language(
    *,
    label: str,
    out_dir: str,
    trials: int,
    seed: int,
    seeds: int,
    seed_start: int,
    uk_missing: str,
    uk_floor: float,
    eng_range: tuple[float, float],
    euro_range: tuple[float, float],
    other_range: tuple[float, float],
    holdout_frac_incorrect: float,
    holdout_frac_correct: float,
) -> str:
    rng = random.Random(seed)

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"tune-language_{label}_{ts}.csv")

    # Build features once.
    base = await build_dataset(DatasetBuildConfig(dataset_csv=None, use_distance=False))

    # With the simplified labeling policy, only use supervision from COUNTRIES_LISTENS:
    # - "correct" ≈ known one-listen countries
    # - "incorrect" ≈ known zero-listen countries
    incorrect_all = [k for k, v in COUNTRIES_LISTENS.items() if v == 0]
    correct_all = [k for k, v in COUNTRIES_LISTENS.items() if v == 1]

    rows: list[dict] = []
    best = None

    for i in range(trials):
        eng = rng.uniform(*eng_range)
        euro = rng.uniform(*euro_range)
        other = rng.uniform(*other_range)

        # Optional monotonic constraint: english >= euro >= other
        if not (eng >= euro >= other):
            # Project to a monotonic triple without adding extra complexity.
            triple = sorted([eng, euro, other], reverse=True)
            eng, euro, other = triple[0], triple[1], triple[2]

        pred = run_heuristic(
            base,
            use_language_factor=True,
            language_english_factor=eng,
            language_euro_latin_factor=euro,
            language_other_factor=other,
        )

        pred_mask = (
            pred["is_single_listen_candidate"] == True
            if "is_single_listen_candidate" in pred.columns
            else pd.Series(False, index=pred.index)
        )

        # Derive the candidate zone from the prediction mask.
        if "model_rank" in pred.columns and pred_mask.any():
            zone_start = float(pd.to_numeric(pred.loc[pred_mask, "model_rank"]).min())
            zone_end = float(pd.to_numeric(pred.loc[pred_mask, "model_rank"]).max())
        else:
            zone_start = 90.0
            zone_end = 95.0

        wrong_tests: list[float] = []
        wrong_trains: list[float] = []
        inc_close_tests: list[float] = []
        inc_close_trains: list[float] = []
        cor_close_tests: list[float] = []
        cor_close_trains: list[float] = []
        cor_overlap_tests: list[int] = []
        cor_overlap_trains: list[int] = []

        for split_seed in range(seed_start, seed_start + seeds):
            split_rng = random.Random(split_seed)
            incorrect = incorrect_all.copy()
            correct = correct_all.copy()
            split_rng.shuffle(incorrect)
            split_rng.shuffle(correct)

            n_inc_test = max(1, int(round(len(incorrect) * holdout_frac_incorrect)))
            n_cor_test = (
                max(1, int(round(len(correct) * holdout_frac_correct)))
                if correct
                else 0
            )

            incorrect_test = set(incorrect[:n_inc_test])
            incorrect_train = set(incorrect[n_inc_test:])
            correct_test = set(correct[:n_cor_test]) if n_cor_test else set()
            correct_train = set(correct[n_cor_test:]) if n_cor_test else set()

            wrong_tests.append(
                _metric_wrong_at_rank(
                    pred, predicted_mask=pred_mask, incorrect_names=incorrect_test
                )
            )
            wrong_trains.append(
                _metric_wrong_at_rank(
                    pred, predicted_mask=pred_mask, incorrect_names=incorrect_train
                )
            )

            inc_close_tests.append(
                _mean_zone_closeness(
                    pred, names=incorrect_test, zone_start=zone_start, zone_end=zone_end
                )
            )
            inc_close_trains.append(
                _mean_zone_closeness(
                    pred,
                    names=incorrect_train,
                    zone_start=zone_start,
                    zone_end=zone_end,
                )
            )
            cor_close_tests.append(
                _mean_zone_closeness(
                    pred, names=correct_test, zone_start=zone_start, zone_end=zone_end
                )
            )
            cor_close_trains.append(
                _mean_zone_closeness(
                    pred, names=correct_train, zone_start=zone_start, zone_end=zone_end
                )
            )

            cor_overlap_tests.append(
                _metric_overlap_count(
                    pred, predicted_mask=pred_mask, names=correct_test
                )
            )
            cor_overlap_trains.append(
                _metric_overlap_count(
                    pred, predicted_mask=pred_mask, names=correct_train
                )
            )

        def _safe_mean(xs: list[float]) -> float:
            vals = [x for x in xs if math.isfinite(x)]
            return float(mean(vals)) if vals else float("nan")

        def _safe_median(xs: list[float]) -> float:
            vals = [x for x in xs if math.isfinite(x)]
            return float(median(vals)) if vals else float("nan")

        wrong_test_mean = _safe_mean(wrong_tests)
        wrong_train_mean = _safe_mean(wrong_trains)
        inc_close_test_mean = _safe_mean(inc_close_tests)
        inc_close_train_mean = _safe_mean(inc_close_trains)
        cor_close_test_mean = _safe_mean(cor_close_tests)
        cor_close_train_mean = _safe_mean(cor_close_trains)

        wrong_test_median = _safe_median(wrong_tests)
        inc_close_test_median = _safe_median(inc_close_tests)
        cor_close_test_median = _safe_median(cor_close_tests)

        row = {
            "trial": i,
            "seed": seed,
            "seeds": seeds,
            "seed_start": seed_start,
            "eng": eng,
            "euro": euro,
            "other": other,
            "uk_missing": uk_missing,
            "uk_floor": uk_floor,
            "zone_start": zone_start,
            "zone_end": zone_end,
            "wrong_at_rank_test_mean": wrong_test_mean,
            "wrong_at_rank_test_median": wrong_test_median,
            "wrong_at_rank_train_mean": wrong_train_mean,
            "incorrect_zone_closeness_test_mean": inc_close_test_mean,
            "incorrect_zone_closeness_test_median": inc_close_test_median,
            "incorrect_zone_closeness_train_mean": inc_close_train_mean,
            "correct_zone_closeness_test_mean": cor_close_test_mean,
            "correct_zone_closeness_test_median": cor_close_test_median,
            "correct_zone_closeness_train_mean": cor_close_train_mean,
            "correct_overlap_test_mean": (
                float(mean(cor_overlap_tests)) if cor_overlap_tests else float("nan")
            ),
            "correct_overlap_train_mean": (
                float(mean(cor_overlap_trains)) if cor_overlap_trains else float("nan")
            ),
            "incorrect_count": len(incorrect_all),
            "correct_count": len(correct_all),
        }
        rows.append(row)

        # Prefer models that push INCORRECT away from the zone (low closeness),
        # and pull CORRECT toward it (high closeness).
        score_key = (
            inc_close_test_mean,
            -cor_close_test_mean,
            wrong_test_mean,
            inc_close_train_mean,
            -cor_close_train_mean,
        )
        if best is None or score_key < best[0]:
            best = (score_key, row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    if best is not None:
        print(
            "Best (by incorrect_zone_closeness_test_mean, then correct_zone_closeness_test_mean):"
        )
        print(best[1])
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(prog="renc")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Generate a labeled model run CSV")
    run.add_argument(
        "--label", required=True, help="Label for this run (e.g. 'v0', 'lang-tweak')"
    )
    run.add_argument(
        "--runs-dir", default="data/runs", help="Directory to store generated run CSVs"
    )
    run.add_argument(
        "--use-language",
        action="store_true",
        help="Include language_factor in the score (often hurts if listens are mostly from travellers)",
    )
    run.add_argument(
        "--dataset-csv",
        default=None,
        help="Optional pre-built dataset CSV (bypasses network fetch + raw merges). Must include alpha_3 or country_name.",
    )
    run.add_argument(
        "--no-distance",
        action="store_true",
        help="Disable distance-from-UK feature (no lat/lon fetch; heuristic score unaffected).",
    )
    run.add_argument(
        "--no-english",
        action="store_true",
        help="Disable English speakers % feature (from wikipedia_eng_lng_pop.csv).",
    )
    run.add_argument("--lang-eng", type=float, default=1.25, help="English multiplier")
    run.add_argument(
        "--lang-euro",
        type=float,
        default=1.0,
        help="European/Latin-language multiplier",
    )
    run.add_argument("--lang-other", type=float, default=0.75, help="Other multiplier")
    run.add_argument(
        "--uk-missing",
        default="p10",
        choices=["p10", "p5", "median", "zero", "ignore"],
        help="(Deprecated) No longer used; UK tourism factor removed from the model.",
    )
    run.add_argument(
        "--uk-floor",
        type=float,
        default=0.05,
        help="(Deprecated) No longer used; UK tourism factor removed from the model.",
    )
    run.add_argument(
        "--launch",
        action="store_true",
        help="Launch Streamlit after generating, preloading this run",
    )

    bayes_run = sub.add_parser(
        "bayes-run",
        help="Generate a labeled run CSV using a Bayesian Poisson GLM (outputs P(listens=1))",
    )
    bayes_run.add_argument(
        "--label", required=True, help="Label for this run (e.g. 'bayes-v1')"
    )
    bayes_run.add_argument(
        "--runs-dir", default="data/runs", help="Directory to store generated run CSVs"
    )
    bayes_run.add_argument(
        "--use-language",
        action="store_true",
        help="Include language_factor in the heuristic score (Bayes model currently ignores language).",
    )
    bayes_run.add_argument("--lang-eng", type=float, default=1.25)
    bayes_run.add_argument("--lang-euro", type=float, default=1.0)
    bayes_run.add_argument("--lang-other", type=float, default=0.75)
    bayes_run.add_argument(
        "--uk-missing",
        default="p10",
        choices=["p10", "p5", "median", "zero", "ignore"],
        help="(Deprecated) No longer used; UK tourism factor removed from the model.",
    )
    bayes_run.add_argument(
        "--uk-floor",
        type=float,
        default=0.05,
        help="(Deprecated) No longer used; UK tourism factor removed from the model.",
    )
    bayes_run.add_argument("--draws", type=int, default=1500)
    bayes_run.add_argument("--tune", type=int, default=1500)
    bayes_run.add_argument("--target-accept", type=float, default=0.9)
    bayes_run.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampler seed (set to -1 to disable deterministic seeding).",
    )
    bayes_run.add_argument("--hdi-prob", type=float, default=0.9)
    bayes_run.add_argument(
        "--dataset-csv",
        default=None,
        help="Optional pre-built dataset CSV (bypasses network fetch + raw merges). Must include alpha_3 or country_name.",
    )
    bayes_run.add_argument(
        "--no-distance",
        action="store_true",
        help="Disable distance-from-UK covariate in the Bayesian model.",
    )
    bayes_run.add_argument(
        "--no-english",
        action="store_true",
        help="Disable English speakers % covariate in the Bayesian model.",
    )
    bayes_run.add_argument(
        "--launch",
        action="store_true",
        help="Launch Streamlit after generating, preloading this run",
    )

    tune = sub.add_parser("tune-language", help="Random-search language multipliers")
    tune.add_argument("--label", required=True, help="Label for this tuning session")
    tune.add_argument("--out-dir", default="runs", help="Where to write results CSV")
    tune.add_argument("--trials", type=int, default=200, help="Number of trials")
    tune.add_argument("--seed", type=int, default=0, help="RNG seed")
    tune.add_argument(
        "--seeds",
        type=int,
        default=50,
        help="Number of different train/test split seeds to average over",
    )
    tune.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First split seed (will evaluate seed_start..seed_start+seeds-1)",
    )
    tune.add_argument(
        "--uk-missing",
        default="p10",
        choices=["p10", "p5", "median", "zero", "ignore"],
    )
    tune.add_argument("--uk-floor", type=float, default=0.05)
    tune.add_argument("--eng-min", type=float, default=1.0)
    tune.add_argument("--eng-max", type=float, default=1.5)
    tune.add_argument("--euro-min", type=float, default=0.9)
    tune.add_argument("--euro-max", type=float, default=1.2)
    tune.add_argument("--other-min", type=float, default=0.5)
    tune.add_argument("--other-max", type=float, default=1.0)
    tune.add_argument(
        "--holdout-incorrect",
        type=float,
        default=0.3,
        help="Fraction of incorrect list held out for evaluation",
    )
    tune.add_argument(
        "--holdout-correct",
        type=float,
        default=0.5,
        help="Fraction of correct list held out for evaluation",
    )

    args = parser.parse_args()

    if args.command == "run":
        result = asyncio.run(
            _generate_run(
                label=args.label,
                runs_dir=args.runs_dir,
                use_language_factor=bool(args.use_language),
                language_english_factor=float(args.lang_eng),
                language_euro_latin_factor=float(args.lang_euro),
                language_other_factor=float(args.lang_other),
                uk_missing_strategy=str(args.uk_missing),
                uk_floor=float(args.uk_floor),
                dataset_csv=(str(args.dataset_csv) if args.dataset_csv else None),
                use_distance=(not bool(args.no_distance)),
                use_english=(not bool(args.no_english)),
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
    elif args.command == "bayes-run":
        seed: int | None
        if int(args.seed) < 0:
            seed = None
        else:
            seed = int(args.seed)

        result = asyncio.run(
            _generate_bayes_run(
                label=args.label,
                runs_dir=args.runs_dir,
                use_language_factor=bool(args.use_language),
                language_english_factor=float(args.lang_eng),
                language_euro_latin_factor=float(args.lang_euro),
                language_other_factor=float(args.lang_other),
                uk_missing_strategy=str(args.uk_missing),
                uk_floor=float(args.uk_floor),
                draws=int(args.draws),
                tune=int(args.tune),
                target_accept=float(args.target_accept),
                seed=seed,
                hdi_prob=float(args.hdi_prob),
                dataset_csv=(str(args.dataset_csv) if args.dataset_csv else None),
                use_distance=(not bool(args.no_distance)),
                use_english=(not bool(args.no_english)),
            )
        )
        print(result.csv_path)

        if args.launch:
            import subprocess

            subprocess.run(
                [
                    "streamlit",
                    "run",
                    "bayes_streamlit.py",
                    "--",
                    "--csv",
                    result.csv_path,
                ],
                check=True,
            )
    elif args.command == "tune-language":
        out_path = asyncio.run(
            _tune_language(
                label=args.label,
                out_dir=args.out_dir,
                trials=int(args.trials),
                seed=int(args.seed),
                seeds=int(args.seeds),
                seed_start=int(args.seed_start),
                uk_missing=str(args.uk_missing),
                uk_floor=float(args.uk_floor),
                eng_range=(float(args.eng_min), float(args.eng_max)),
                euro_range=(float(args.euro_min), float(args.euro_max)),
                other_range=(float(args.other_min), float(args.other_max)),
                holdout_frac_incorrect=float(args.holdout_incorrect),
                holdout_frac_correct=float(args.holdout_correct),
            )
        )
        print(out_path)


if __name__ == "__main__":
    main()

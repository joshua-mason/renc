from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
import pycountry

import arviz as az
import pymc as pm


@dataclass(frozen=True)
class BayesFitConfig:
    draws: int = 1500
    tune: int = 1500
    target_accept: float = 0.9
    seed: int | None = 0
    hdi_prob: float = 0.9
    aggregate_sigma: float = 1.0
    aggregate_one_sigma: float = 1.0
    alpha_prior_sigma: float = 2.0
    beta_prior_sigma: float = 1.0
    # Negative Binomial over-dispersion prior (larger -> allow more dispersion).
    nb_alpha_sigma: float = 5.0
    mu_floor: float = 1e-9
    # Clipping log(mu) can introduce hard non-linearities that harm NUTS geometry.
    # Prefer leaving this high enough that we rarely/never hit it in practice.
    log_mu_clip: float = 20.0


def _resolve_country_to_alpha3(token: str | None) -> str | None:
    """
    Resolve a country token to ISO alpha-3.

    Accepts common inputs like:
    - alpha-3: "GBR"
    - names: "United Kingdom"
    - informal: "usa"
    """
    if token is None:
        return None
    raw = str(token).strip()
    if not raw:
        return None

    # If it's already alpha-3, trust it.
    if len(raw) == 3 and raw.isalpha():
        return raw.upper()

    lowered = raw.lower()
    aliases = {
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "usa": "United States",
        "u.s.a.": "United States",
        "us": "United States",
        "u.s.": "United States",
        # pycountry names sometimes differ from common usage:
        "turkey": "Türkiye",
        "turkiye": "Türkiye",
        "us virgin islands": "Virgin Islands, U.S.",
        "united states virgin islands": "Virgin Islands, U.S.",
        "northern marianas islands": "Northern Mariana Islands",
    }
    query = aliases.get(lowered, raw)

    try:
        return pycountry.countries.lookup(query).alpha_3
    except LookupError:
        try:
            return pycountry.countries.search_fuzzy(query)[0].alpha_3
        except LookupError:
            return None


def _standardize(
    x: np.ndarray, *, mean: float | None = None, std: float | None = None
) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=float)
    if mean is None:
        mean = float(np.nanmean(x))
    if std is None:
        std = float(np.nanstd(x))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return (x - mean) / std, float(mean), float(std)


def _feature_matrix(
    df: pd.DataFrame, *, use_distance: bool, use_english: bool
) -> tuple[np.ndarray, list[str]]:
    pop = pd.to_numeric(df.get("population"), errors="coerce").to_numpy(dtype=float)
    pop = np.log(np.clip(pop, 1.0, None))

    net = pd.to_numeric(df.get("internet_usage_pct"), errors="coerce").to_numpy(
        dtype=float
    )
    net = net / 100.0

    feats: list[np.ndarray] = [pop, net]
    names: list[str] = ["log_population", "internet_rate"]

    if bool(use_distance):
        dist_km = pd.to_numeric(df.get("uk_distance_km"), errors="coerce").to_numpy(
            dtype=float
        )
        dist_km = np.clip(dist_km, 0.0, None)
        dist = np.log1p(dist_km)
        feats.append(dist)
        names.append("log1p_uk_distance_km")

    if bool(use_english):
        eng_pct = pd.to_numeric(
            df.get("english_speakers_pct"), errors="coerce"
        ).to_numpy(dtype=float)
        eng = np.clip(eng_pct, 0.0, 100.0) / 100.0
        feats.append(eng)
        names.append("english_speakers_rate")

    X = np.column_stack(feats).astype(float)
    return X, names


def _counts_map_to_alpha3(
    country_listens: Mapping[str, int | None],
) -> tuple[dict[str, int], list[str]]:
    """
    Convert a {country_token -> listens} mapping into {alpha3 -> listens}.

    Returns:
    - alpha3_counts: only entries where listens is an int >= 0 and country resolves
    - unresolved_tokens: tokens we couldn't resolve (or invalid counts)
    """
    alpha3_counts: dict[str, int] = {}
    unresolved: list[str] = []

    for token, value in country_listens.items():
        a3 = _resolve_country_to_alpha3(token)
        if a3 is None:
            unresolved.append(str(token))
            continue
        if value is None:
            # Intentionally unknown.
            continue
        try:
            iv = int(value)
        except (TypeError, ValueError):
            unresolved.append(str(token))
            continue
        if iv < 0:
            unresolved.append(str(token))
            continue
        # If duplicates exist, keep the max (conservative: known that it reached at least this count).
        alpha3_counts[a3] = max(alpha3_counts.get(a3, 0), iv)

    unresolved.sort()
    return alpha3_counts, unresolved


def _lower_bounds_map_to_alpha3(
    lower_bounds: Mapping[str, int],
) -> tuple[dict[str, int], list[str]]:
    """
    Convert a {country_token -> lower_bound} mapping into {alpha3 -> lower_bound}.

    - lower_bound is interpreted as: Y >= lower_bound
    - keeps the max bound if duplicates exist
    """
    alpha3_bounds: dict[str, int] = {}
    unresolved: list[str] = []

    for token, value in lower_bounds.items():
        a3 = _resolve_country_to_alpha3(token)
        if a3 is None:
            unresolved.append(str(token))
            continue
        try:
            iv = int(value)
        except (TypeError, ValueError):
            unresolved.append(str(token))
            continue
        if iv < 0:
            unresolved.append(str(token))
            continue
        alpha3_bounds[a3] = max(alpha3_bounds.get(a3, 0), iv)

    unresolved.sort()
    return alpha3_bounds, unresolved


def fit_poisson_glm_p_one(
    df: pd.DataFrame,
    *,
    country_listens: Mapping[str, int | None],
    censored_country_lower_bounds: Mapping[str, int] | None = None,
    number_of_countries_with_listens: int | None = None,
    number_of_countries_with_one_listen: int | None = None,
    use_distance: bool = True,
    use_english: bool = True,
    likelihood: str = "poisson",
    config: BayesFitConfig = BayesFitConfig(),
) -> pd.DataFrame:
    """
    Fit a count GLM on observed per-country counts (COUNTRIES_LISTENS) and compute
    posterior P(Y=1) for every country.

    Supervision used (and only used):
    - country_listens: per-country counts (int >= 0) or None for unknown
    - number_of_countries_with_listens (optional): aggregate constraint on how many
      countries have Y>0 at the recording time.
    """
    like = str(likelihood).strip().lower()
    if like not in {"poisson", "negbin", "negative_binomial", "nb"}:
        raise ValueError(
            f"Unknown likelihood='{likelihood}'. Expected 'poisson' or 'negbin'."
        )
    like = "negbin" if like in {"negbin", "negative_binomial", "nb"} else "poisson"
    if "alpha_3" not in df.columns:
        raise ValueError("Expected df to include an 'alpha_3' column.")

    counts_alpha3, unresolved_tokens = _counts_map_to_alpha3(country_listens)
    censored_alpha3: dict[str, int] = {}
    censored_unresolved_tokens: list[str] = []
    if censored_country_lower_bounds:
        censored_alpha3, censored_unresolved_tokens = _lower_bounds_map_to_alpha3(
            censored_country_lower_bounds
        )

    # Track which countries appear in the COUNTRIES_LISTENS mapping at all (even if value is None).
    in_supervision_map: set[str] = set()
    for token in country_listens.keys():
        a3 = _resolve_country_to_alpha3(token)
        if a3 is not None:
            in_supervision_map.add(a3)
    for token in (censored_country_lower_bounds or {}).keys():
        a3 = _resolve_country_to_alpha3(token)
        if a3 is not None:
            in_supervision_map.add(a3)

    alpha3_series = df["alpha_3"].astype(str).str.upper()

    # Build observed y vector (NaN for unknown).
    y = pd.Series(np.nan, index=df.index, dtype=float)
    for a3, iv in counts_alpha3.items():
        y.loc[alpha3_series == a3] = float(iv)

    is_train = y.notna()

    # Build censored lower bounds (NaN if not censored). If a country is both
    # exactly observed and censored, prefer the exact observation.
    y_censored_lb = pd.Series(np.nan, index=df.index, dtype=float)
    for a3, lb in censored_alpha3.items():
        # Prefer exact observation if present.
        if bool((alpha3_series == a3).any()):
            y_censored_lb.loc[alpha3_series == a3] = float(lb)
    y_censored_lb.loc[is_train] = np.nan
    is_censored = y_censored_lb.notna()

    if int(is_train.sum()) < 5:
        raise ValueError(
            f"Not enough labeled training rows ({int(is_train.sum())}). "
            "Add more entries with numeric counts to COUNTRIES_LISTENS."
        )

    # Human-readable label for storytelling/debugging.
    label = pd.Series("unknown", index=df.index, dtype="string")
    label[y.notna() & (y == 0)] = "observed_zero"
    label[y.notna() & (y == 1)] = "observed_one"
    label[y.notna() & (y >= 2)] = "observed_multi"
    label[is_censored] = "censored"

    X_raw, feature_names = _feature_matrix(
        df, use_distance=bool(use_distance), use_english=bool(use_english)
    )

    # Fit-time imputation (train means) + standardization (train stats).
    X_train_raw = X_raw[is_train.to_numpy(), :]
    col_means = np.nanmean(X_train_raw, axis=0)
    col_stds = np.nanstd(X_train_raw, axis=0)

    # If a feature is entirely missing among training rows, nanmean/nanstd return NaN.
    # Fall back to safe defaults so we can still run and surface the missingness in debug.
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    col_stds = np.where(np.isfinite(col_stds) & (col_stds > 0), col_stds, 1.0)

    X_imputed = np.where(np.isfinite(X_raw), X_raw, col_means)
    X = (X_imputed - col_means) / col_stds
    X = np.where(np.isfinite(X), X, 0.0)
    X_train = X[is_train.to_numpy(), :]

    y_train = y[is_train].to_numpy(dtype=int)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=float(config.alpha_prior_sigma))
        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=float(config.beta_prior_sigma),
            shape=(X_train.shape[1],),
        )

        log_mu_train = alpha + pm.math.dot(X_train, beta)
        mu_train = pm.Deterministic(
            "mu_train",
            pm.math.exp(log_mu_train) + float(config.mu_floor),
        )
        if like == "poisson":
            pm.Poisson("y", mu=mu_train, observed=y_train)
        else:
            nb_alpha = pm.HalfNormal("nb_alpha", sigma=float(config.nb_alpha_sigma))
            pm.NegativeBinomial("y", mu=mu_train, alpha=nb_alpha, observed=y_train)

        # Predict for all rows.
        log_mu_all = pm.Deterministic("log_mu_all", alpha + pm.math.dot(X, beta))
        mu_all = pm.Deterministic(
            "mu_all",
            pm.math.exp(log_mu_all) + float(config.mu_floor),
        )
        if like == "poisson":
            p_one_all = pm.Deterministic("p_one_all", mu_all * pm.math.exp(-mu_all))
            p_zero_all = pm.math.exp(-mu_all)
        else:
            nb_alpha = model["nb_alpha"]
            # For NB(mu, alpha): p0 = (alpha/(alpha+mu))**alpha
            frac = nb_alpha / (nb_alpha + mu_all)
            p_zero_all = pm.Deterministic("p_zero_all", frac**nb_alpha)
            # p1 = alpha*(alpha/(alpha+mu))**alpha * (mu/(alpha+mu))
            p_one_all = pm.Deterministic(
                "p_one_all",
                nb_alpha * (frac**nb_alpha) * (mu_all / (nb_alpha + mu_all)),
            )

        # Censored supervision: for rows where we know Y >= k.
        #
        # Use PyMC's built-in `Censored` distribution for better sampler geometry.
        # Note: for a lower bound (Y >= k), we represent it as "right-censoring at k"
        # by setting upper=k and observing exactly k.
        if bool(is_censored.any()):
            cens_mask_all = is_censored.to_numpy()
            k_all = (
                pd.to_numeric(y_censored_lb.loc[is_censored], errors="coerce")
                .fillna(0)
                .astype(int)
                .to_numpy()
            )
            # Drop vacuous bounds (k<=0).
            keep = k_all > 0
            if bool(np.any(keep)):
                mu_cens = mu_all[cens_mask_all][keep]
                k = k_all[keep]
                # `pm.Censored` treats observations equal to `upper` as censored ABOVE that value,
                # i.e. it contributes log P(Y >= upper). That matches our lower-bound semantics.
                pm.Censored(
                    "y_censored_lower_bound",
                    (
                        pm.Poisson.dist(mu=mu_cens)
                        if like == "poisson"
                        else pm.NegativeBinomial.dist(
                            mu=mu_cens, alpha=model["nb_alpha"]
                        )
                    ),
                    lower=-np.inf,
                    upper=k,
                    observed=k,
                )

        # Aggregate constraint: total number of nonzero-listen countries.
        if number_of_countries_with_listens is not None:
            total = int(number_of_countries_with_listens)
            if total < 0:
                raise ValueError("number_of_countries_with_listens must be >= 0")

            censored_known_nonzero = int(
                (
                    pd.to_numeric(y_censored_lb.loc[is_censored], errors="coerce") >= 1
                ).sum()
            )
            observed_nonzero = int((y_train > 0).sum()) + censored_known_nonzero
            if total < observed_nonzero:
                raise ValueError(
                    f"number_of_countries_with_listens={total} is less than observed "
                    f"nonzero countries in COUNTRIES_LISTENS ({observed_nonzero})."
                )

            unknown_mask = (~(is_train | is_censored)).to_numpy()
            p_zero_unknown = p_zero_all[unknown_mask]
            p_nonzero_unknown = 1.0 - p_zero_unknown
            expected_total_nonzero = observed_nonzero + pm.math.sum(p_nonzero_unknown)

            pm.Potential(
                "nonzero_countries_constraint",
                pm.logp(
                    pm.Normal.dist(
                        mu=expected_total_nonzero, sigma=float(config.aggregate_sigma)
                    ),
                    total,
                ),
            )

        # Aggregate constraint: total number of exactly-one-listen countries.
        # This is particularly important because P(Y=1) is maximized at mu=1, so
        # with sparse labels the model can otherwise assign many countries mu≈1.
        if number_of_countries_with_one_listen is not None:
            total_one = int(number_of_countries_with_one_listen)
            if total_one < 0:
                raise ValueError("number_of_countries_with_one_listen must be >= 0")

            observed_one = int((y_train == 1).sum())
            if total_one < observed_one:
                raise ValueError(
                    f"number_of_countries_with_one_listen={total_one} is less than "
                    f"observed one-listen countries in COUNTRIES_LISTENS ({observed_one})."
                )

            unknown_mask = (~(is_train | is_censored)).to_numpy()
            p_one_unknown = p_one_all[unknown_mask]
            expected_total_one = observed_one + pm.math.sum(p_one_unknown)

            pm.Potential(
                "one_listen_countries_constraint",
                pm.logp(
                    pm.Normal.dist(
                        mu=expected_total_one,
                        sigma=float(config.aggregate_one_sigma),
                    ),
                    total_one,
                ),
            )

        trace = pm.sample(
            draws=int(config.draws),
            tune=int(config.tune),
            target_accept=float(config.target_accept),
            random_seed=None if config.seed is None else int(config.seed),
            progressbar=False,
        )

    # Summarize posterior with numpy quantiles to avoid extra deps and keep it transparent.
    def _summarize(var_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = trace.posterior[var_name].stack(sample=("chain", "draw")).values
        # arr shape: (n_rows, n_samples) or (n_samples, n_rows) depending on xarray order.
        if arr.ndim != 2:
            raise ValueError(f"Unexpected posterior shape for {var_name}: {arr.shape}")
        if arr.shape[0] != len(df) and arr.shape[1] == len(df):
            arr = arr.T

        mean = np.mean(arr, axis=1)
        hdi = float(config.hdi_prob)
        lo_q = (1.0 - hdi) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(arr, lo_q, axis=1)
        hi = np.quantile(arr, hi_q, axis=1)
        return mean, lo, hi

    mu_mean, mu_lo, mu_hi = _summarize("mu_all")
    p1_mean, p1_lo, p1_hi = _summarize("p_one_all")
    lp_mean, lp_lo, lp_hi = _summarize("log_mu_all")

    # Sampler diagnostics (handy for debugging).
    var_names = ["alpha", "beta"]
    if like == "negbin" and "nb_alpha" in trace.posterior:
        var_names.append("nb_alpha")
    summary = az.summary(trace, var_names=var_names, round_to=None)
    rhat_max = (
        float(summary["r_hat"].max()) if "r_hat" in summary.columns else float("nan")
    )
    ess_bulk_min = (
        float(summary["ess_bulk"].min())
        if "ess_bulk" in summary.columns
        else float("nan")
    )

    # Feature debug columns (raw + standardized).
    X_raw_df = pd.DataFrame(X_raw, columns=[f"bayes_x_{n}" for n in feature_names])
    X_z_df = pd.DataFrame(X, columns=[f"bayes_z_{n}" for n in feature_names])

    # Parameter summaries for storytelling/debugging.
    alpha_mean = float(trace.posterior["alpha"].mean().values)
    alpha_arr = trace.posterior["alpha"].stack(sample=("chain", "draw")).values
    hdi = float(config.hdi_prob)
    lo_q = (1.0 - hdi) / 2.0
    hi_q = 1.0 - lo_q
    alpha_hdi_low = float(np.quantile(alpha_arr, lo_q))
    alpha_hdi_high = float(np.quantile(alpha_arr, hi_q))

    beta_arr = trace.posterior["beta"].stack(sample=("chain", "draw")).values
    # beta_arr shape: (n_features, n_samples) or transpose
    if beta_arr.ndim != 2:
        raise ValueError(f"Unexpected posterior shape for beta: {beta_arr.shape}")
    if beta_arr.shape[0] != len(feature_names) and beta_arr.shape[1] == len(
        feature_names
    ):
        beta_arr = beta_arr.T
    beta_mean = beta_arr.mean(axis=1)
    beta_hdi_low = np.quantile(beta_arr, lo_q, axis=1)
    beta_hdi_high = np.quantile(beta_arr, hi_q, axis=1)

    nb_alpha_mean: float | None = None
    nb_alpha_hdi_low: float | None = None
    nb_alpha_hdi_high: float | None = None
    if like == "negbin" and "nb_alpha" in trace.posterior:
        nb_arr = trace.posterior["nb_alpha"].stack(sample=("chain", "draw")).values
        nb_alpha_mean = float(np.mean(nb_arr))
        nb_alpha_hdi_low = float(np.quantile(nb_arr, lo_q))
        nb_alpha_hdi_high = float(np.quantile(nb_arr, hi_q))

    # Calibration summaries (using posterior means; fast + good enough for UI/debugging).
    total_countries = int(len(df))
    target_nonzero = (
        int(number_of_countries_with_listens)
        if number_of_countries_with_listens is not None
        else None
    )
    target_one = (
        int(number_of_countries_with_one_listen)
        if number_of_countries_with_one_listen is not None
        else None
    )
    target_zero = (
        (total_countries - target_nonzero) if target_nonzero is not None else None
    )
    target_multi = (
        (target_nonzero - target_one)
        if (target_nonzero is not None and target_one is not None)
        else None
    )

    known_nonzero = int((y_train > 0).sum()) + int(
        (
            pd.to_numeric(y_censored_lb.loc[is_censored], errors="coerce").fillna(0)
            >= 1
        ).sum()
    )
    known_one = int((y_train == 1).sum())

    # Expected totals over unknown rows.
    unknown_mask = (~(is_train | is_censored)).to_numpy()
    if like == "poisson":
        p0_mean_all = np.exp(-np.clip(mu_mean, 0.0, None))
    else:
        # Use posterior mean of nb_alpha for a simple plug-in estimate.
        a = float(nb_alpha_mean) if nb_alpha_mean is not None else 1.0
        frac = a / (a + np.clip(mu_mean, 0.0, None))
        p0_mean_all = np.power(frac, a)
    expected_nonzero_unknown = float(np.nansum(1.0 - p0_mean_all[unknown_mask]))
    expected_total_nonzero_mean = float(known_nonzero + expected_nonzero_unknown)
    expected_total_one_mean = float(known_one + float(np.nansum(p1_mean[unknown_mask])))
    expected_total_zero_mean = float(total_countries - expected_total_nonzero_mean)
    expected_total_multi_mean = float(
        expected_total_nonzero_mean - expected_total_one_mean
    )

    out = pd.DataFrame(
        {
            "alpha_3": df["alpha_3"].astype(str).str.upper(),
            "bayes_in_country_listens_map": alpha3_series.isin(in_supervision_map),
            "bayes_label": label.astype(str),
            "bayes_y_observed": y,
            "bayes_y_censored_lower_bound": y_censored_lb,
            "bayes_is_censored": is_censored.to_numpy(),
            "bayes_lp_mean": lp_mean,
            "bayes_lp_hdi_low": lp_lo,
            "bayes_lp_hdi_high": lp_hi,
            "bayes_mu_mean": mu_mean,
            "bayes_mu_hdi_low": mu_lo,
            "bayes_mu_hdi_high": mu_hi,
            "bayes_p_one_mean": p1_mean,
            "bayes_p_one_hdi_low": p1_lo,
            "bayes_p_one_hdi_high": p1_hi,
            "bayes_hdi_prob": float(config.hdi_prob),
            "bayes_rhat_max": rhat_max,
            "bayes_ess_bulk_min": ess_bulk_min,
            "bayes_train_n": int(is_train.sum()),
            "bayes_train_n_zero": int((label == "observed_zero").sum()),
            "bayes_train_n_one": int((label == "observed_one").sum()),
            "bayes_train_n_multi": int((label == "observed_multi").sum()),
            "bayes_train_n_censored": int(is_censored.sum()),
            "bayes_alpha_mean": alpha_mean,
            "bayes_alpha_hdi_low": alpha_hdi_low,
            "bayes_alpha_hdi_high": alpha_hdi_high,
            "bayes_unresolved_label_tokens_count": int(len(unresolved_tokens)),
            "bayes_unresolved_censored_tokens_count": int(
                len(censored_unresolved_tokens)
            ),
            "bayes_likelihood": like,
            "bayes_total_countries": total_countries,
            "bayes_target_total_nonzero": (
                int(target_nonzero) if target_nonzero is not None else np.nan
            ),
            "bayes_target_total_zero": (
                int(target_zero) if target_zero is not None else np.nan
            ),
            "bayes_target_total_one": (
                int(target_one) if target_one is not None else np.nan
            ),
            "bayes_target_total_multi": (
                int(target_multi) if target_multi is not None else np.nan
            ),
            "bayes_expected_total_nonzero_mean": expected_total_nonzero_mean,
            "bayes_expected_total_zero_mean": expected_total_zero_mean,
            "bayes_expected_total_one_mean": expected_total_one_mean,
            "bayes_expected_total_multi_mean": expected_total_multi_mean,
            "bayes_number_of_countries_with_listens": (
                int(number_of_countries_with_listens)
                if number_of_countries_with_listens is not None
                else np.nan
            ),
            "bayes_aggregate_sigma": float(config.aggregate_sigma),
            "bayes_number_of_countries_with_one_listen": (
                int(number_of_countries_with_one_listen)
                if number_of_countries_with_one_listen is not None
                else np.nan
            ),
            "bayes_aggregate_one_sigma": float(config.aggregate_one_sigma),
            "bayes_use_distance": bool(use_distance),
            "bayes_use_english": bool(use_english),
            "bayes_use_censoring": bool(bool(censored_country_lower_bounds)),
            "bayes_alpha_prior_sigma": float(config.alpha_prior_sigma),
            "bayes_beta_prior_sigma": float(config.beta_prior_sigma),
            "bayes_nb_alpha_sigma": float(config.nb_alpha_sigma),
        }
    )
    if nb_alpha_mean is not None:
        out["bayes_nb_alpha_mean"] = float(nb_alpha_mean)
        out["bayes_nb_alpha_hdi_low"] = (
            float(nb_alpha_hdi_low) if nb_alpha_hdi_low is not None else np.nan
        )
        out["bayes_nb_alpha_hdi_high"] = (
            float(nb_alpha_hdi_high) if nb_alpha_hdi_high is not None else np.nan
        )
    for j, name in enumerate(feature_names):
        out[f"bayes_train_mean_{name}"] = float(col_means[j])
        out[f"bayes_train_std_{name}"] = float(col_stds[j])
        out[f"bayes_beta_{name}_mean"] = float(beta_mean[j])
        out[f"bayes_beta_{name}_hdi_low"] = float(beta_hdi_low[j])
        out[f"bayes_beta_{name}_hdi_high"] = float(beta_hdi_high[j])

    out = pd.concat([out, X_raw_df, X_z_df], axis=1)
    return out

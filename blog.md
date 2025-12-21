## renc notes (decision journal)

### Context (what this project is doing)
- **Problem**: Given a podcast’s per-country listen telemetry (limited/patchy), identify which countries are most likely to have had **exactly one** listen in a particular reporting window.
- **Why**: This started as a “sanity check” / heuristic ranking exercise (V0), and is evolving toward a probabilistic model (Bayesian/simulation) that can output `P(country has exactly 1 listen)`.
- **Data + features**:
  - `population` (from restcountries)
  - `internet_usage_pct` (from `raw/internetusage.csv`)
  - `uk_visits_number` (from `raw/ukvisitsabroad.csv`, used as a cultural proximity proxy)
  - `languages` (from restcountries; treated as optional and explicitly tested)
- **Labels we currently have**:
  - `CORRECT_COUNTRIES`: small set of true single-listen countries
  - `INCORRECT_COUNTRIES`: set that are not single-listen countries (negative evidence)
- **Evaluation/metrics (current)**:
  - **Overlap with correct**: how many “known correct” land in the candidate set
  - **Wrong@rank**: rank-weighted false-positive penalty on known incorrects (lower is better)
  - **Zone closeness**: continuous “distance-to-candidate-zone” metric used in tuning
- **Workflow**: generate labeled run CSVs into `runs/` and compare runs in Streamlit.

### Notes policy
- This file is intentionally **append-only**: new findings are added as new dated entries (we avoid rewriting history).


### 2025-12-19 — V0 heuristic scoring
- **Decision**: Use multiplicative score (filters) instead of additive points.
- **Formula**: `log10(population) * (internet_pct/100) * log10(uk_visits) * language_factor`.
- **Why**: Avoids the “China problem” (linear scaling) and enforces “no internet => no listens”.

### 2025-12-19 — UK visits data caused duplicates
- **Observation**: Output had duplicate countries.
- **Cause**: `ukvisitsabroad.csv` contains multiple periods per country; merging created many-to-one duplicates.
- **Decision**: Deduplicate to one row per country by taking the max `uk_visits_number` before merge.

### 2025-12-19 — UK score missing values + UK special-case
- **Decision**: If `uk_visits_number` missing, fill UK score with a “normal” baseline (median of log10(visits)).
- **Decision**: Set UK (`GBR`) to maximal UK-affinity (self-row otherwise often missing visits).

### 2025-12-19 — UK score: missingness rethink (revision)
- **Observation**: Median-imputing missing UK visits is likely optimistic.
- **Hypothesis**: Missing often means “too small / not tracked”, not “average”.
- **Decision**: Treat missing as **unknown but likely low** by imputing a low percentile (p10 by default) and applying a **non-zero floor** to the UK multiplier so the multiplicative model doesn’t collapse scores to 0.
- **Implementation**: `uk_raw = log10(uk_visits + 1)`, missing → p10/p5/median/zero/ignore (flagged), then scale to `(uk_floor..1]`.

### 2025-12-19 — Added run workflow + viewer
- **Decision**: Generate labeled run CSVs into `runs/` to track experiments over time.
- **Tooling**: `uv run python -m src run --label ...` and Streamlit dataset picker.

### 2025-12-19 — Language feature: bug fix → feature ablation
- **Bug**: We were originally using language codes (e.g. `eng`) rather than readable names.
- **Fix**: Use language names from restcountries and normalize/lookup language strings.
- **Finding**: Adding `language_factor` made results worse.
- **Interpretation**: Listens likely driven by travellers/cultural proximity; UK visits already captures this, while “official/local language” is the wrong mechanism.
- **Decision**: Keep language in code, but make it optional (`use_language_factor`) and log both `total_score_no_language` and `total_score_with_language` in every run for documentation.

### 2025-12-19 — Label correction: Bhutan
- **Correction**: Bhutan was incorrectly included as a \"known single-listen\" country earlier; it belongs in the incorrect list.
- **Impact**: This may have biased earlier comparisons (including the language-factor evaluation), so prior conclusions should be treated as provisional until re-run with corrected labels.

### 2025-12-19 — Negative evidence: incorrect countries list
- **Decision**: Maintain `INCORRECT_COUNTRIES` alongside `CORRECT_COUNTRIES` and surface them in plots.
- **Why**: This gives us a simple false-positive check (e.g., `predicted ∩ incorrect`) rather than optimizing only for overlap with a tiny positive set.

### 2025-12-19 — Potential new data: episode-level listens (patchy)
- **Observation**: We can sometimes recover per-episode country listens for countries in the incorrect list (patchy coverage).
- **Next step**: Use this as additional supervision (even if incomplete) to constrain models beyond binary “correct/incorrect” lists.

### 2025-12-19 — Future: use model ranking to score human guesses
- **Idea**: Use the model’s rank/score as a way to quantify how “bad” a guess was (e.g., penalize guesses that the model ranks extremely low).
- **Why**: Lets us compare different people/strategies consistently, even before we have full ground-truth per-country listen counts.

### 2025-12-19 — Language factors: now helpful (after tuning)
- **Observation**: With corrected labels and milder language multipliers (e.g. English ~1.25, Euro/Latin ~1.0, other ~0.75), language factors improve guesses in recent runs.
- **Note**: Earlier conclusions about language were confounded by the Bhutan label error and stronger multipliers.
- **Bug fix**: `Wrong@rank` now shows `0.0000` when there are zero incorrect overlaps (instead of “—”).
- **Open question**: How to set these multipliers systematically (candidate: Bayesian/simulation approach rather than manual tuning).

### 2025-12-19 — Option A: tune language factors via search + holdout
- **Decision**: Stop hand-tuning language multipliers and instead do a small random search over plausible ranges.
- **Method**: Evaluate each candidate set of multipliers across many random train/test splits (“many seeds”). Prefer candidates that:
  - push `INCORRECT_COUNTRIES` far from the candidate zone (low **zone closeness**)
  - pull `CORRECT_COUNTRIES` toward the zone (high **zone closeness**)
  - use `Wrong@rank` as an explicit false-positive tie-breaker
- **Output**: Save all trials to a CSV so results are reproducible and comparable across runs.

### 2025-12-19 — Blog name idea
- **Idea**: `twoinsix.com` as a short name for the write-up site.
- **Meaning**: captures the “pick 6 countries / maybe 2 hits” vibe of the project and keeps the framing memorable.

### 2025-12-19 — Language factor: keep vs remove (open question)
- **Concern**: Manually tuning language multipliers can be overfitting / “hacky”, especially with tiny labels.
- **Counterpoint**: It might still be a real signal (proxying cultural proximity / discovery), but we don’t yet know if it generalizes.
- **Decision (for now)**: Treat language as an optional feature with documented tuning runs; prefer simpler models unless language consistently improves metrics across many seeds and doesn’t increase false positives.

### 2025-12-20 — Bayesian V1: Poisson regression → P(listens=1)
- **Goal**: Move from a heuristic “rank score” to a probabilistic output per country: `P(listens == 1)`.
- **Model**: Bayesian Poisson GLM (count model):
  - \(Y_i \sim \text{Poisson}(\mu_i)\)
  - \(\log \mu_i = \alpha + \beta^T x_i\)
  - predictors \(x\): `log(population)`, `internet_usage_pct/100`, `log10(uk_visits_number + 1)` (standardized)
- **Single-listen probability**: derived from the posterior rate:
  - \(P(Y=1 \mid \mu) = \mu e^{-\mu}\)
  - We estimate this by computing `p_one_all = mu_all * exp(-mu_all)` for every posterior draw, then summarizing.

### 2025-12-20 — Bayesian V1 supervision: fixed-count multi-listen hack
- **Constraint**: We do not yet have exact listen counts for most countries, and we are still learning (so we want to avoid censored/latent class models for now).
- **Decision**: Use partial labels from `src/config.py`:
  - `CORRECT_COUNTRIES` → observed `Y=1`
  - `COUNTRIES_WITH_MORE_THAN_ONE_LISTEN` → observed `Y=K` (fixed), **K=100** by default
  - all other countries are prediction-only (not used in the likelihood)
- **Interpretation**: This is a deliberate training hack to give the model *some* “high-count” contrast without needing full data.
- **Limitation**: Outputs should be treated as a **ranking tool** (relative probabilities) until we replace the fixed K with a principled `Y >= 2` likelihood (censoring / ordinal / mixture).

### 2025-12-20 — Tooling: bayes-run CSV + Streamlit view
- **Implementation**: Add `bayes-run` CLI command to generate standard run CSVs with extra Bayesian columns:
  - `bayes_mu_mean`, `bayes_mu_hdi_low`, `bayes_mu_hdi_high`
  - `bayes_p_one_mean`, `bayes_p_one_hdi_low`, `bayes_p_one_hdi_high`
  - `bayes_rank` (descending `bayes_p_one_mean`)
- **Workflow**: Keep the same `runs/` + Streamlit dataset picker; Bayesian runs are just another saved CSV for comparison.

### 2025-12-20 — Pivot: supervision only from `COUNTRIES_LISTENS` + aggregate listens count
- **Change**: Stop using any auxiliary label lists (`CORRECT_COUNTRIES`, `INCORRECT_COUNTRIES`, `COUNTRIES_WITH_MORE_THAN_ONE_LISTEN`) for Bayesian training.
- **New supervision sources** (and only these):
  - `COUNTRIES_LISTENS`: mapping of `country -> listens` where listens is:
    - an integer count (0, 1, 2, …) when we’ve verified it from the episode
    - `None` when unknown/uncertain
  - `NUMBER_OF_COUNTRIES_WITH_LISTENS`: a single aggregate number for how many countries had **any** listens at the time of recording.
- **Why**: This makes the model easier to reason about and avoids “secret” priors/data sneaking in via hand-picked lists.

### 2025-12-20 — Aggregate constraint model (missing membership)
- **Problem**: We know an aggregate count (“95 countries had any listens”), but we don’t know the full set membership.
- **Modeling trick**: add a soft constraint using expected nonzero probability:
  - For a Poisson country model \(Y_i \\sim \\text{Poisson}(\\mu_i)\),
    \(P(Y_i > 0) = 1 - e^{-\\mu_i}\).
  - For countries with unknown counts, we sum these probabilities and constrain the total:
    \[
      \\text{observed\_nonzero} + \\sum_{i \\in \\text{unknown}} (1 - e^{-\\mu_i})
      \\approx \\texttt{NUMBER\_OF\_COUNTRIES\_WITH\_LISTENS}
    \]
  - Implemented as a `pm.Potential` with a Normal error term (tunable `aggregate_sigma`).
- **Why it helps**: The aggregate constraint prevents the posterior from collapsing into “almost everything is 0”, even with sparse per-country counts.

### 2025-12-20 — Storytelling UI + debugging outputs
- **Decision**: Build a separate Streamlit app focused on explaining the Bayes model and letting us inspect inputs and labels.
- **Implementation**:
  - `bayes_streamlit.py`: story-first viewer for Bayes-run CSVs:
    - explains the model and the derived \(P(Y=1)\)
    - shows training rows (`bayes_y_observed`) and label counts (`bayes_label`)
    - provides an input data explorer (raw feature columns + missingness)
    - shows learned coefficients (`bayes_alpha_*`, `bayes_beta_*`) and intervals
  - Add rich debug columns in Bayes CSV output:
    - raw features: `bayes_x_*`
    - standardized features: `bayes_z_*`
    - linear predictor: `bayes_lp_*`
    - sampler diagnostics: `bayes_rhat_max`, `bayes_ess_bulk_min`

### 2025-12-20 — Data directory convention + reproducible datasets
- **Decision**: Use a `data/` directory structure:
  - `data/raw/…` preferred for raw input files (fallback to legacy `raw/…`)
  - `data/runs/` as the default output directory for generated CSV runs
- **New capability**: `--dataset-csv` (for both `run` and `bayes-run`) to load a prebuilt country-feature dataset directly (bypassing network fetch + raw merges). This is intended for “better datasets” you curate over time.

### 2025-12-20 — Numerical robustness fixes (so Bayes runs don’t crash)
- **Issue**: When observed countries have missing feature columns, training feature standardization can produce NaNs, and Poisson rates can underflow to 0 causing `-inf` likelihood.
- **Fixes**:
  - If a training feature column is entirely missing, fall back to safe standardization defaults (`mean=0`, `std=1`) and fill remaining standardized NaNs with 0.
  - Add a small \(\mu\) floor and clip \(\log \mu\) before exponentiating to prevent rate underflow.
- **Note**: Sampling diagnostics like `rhat` and `ess` should be monitored (we surface them in the run CSV and in `bayes_streamlit.py`).

### 2025-12-20 — New covariate: distance from the UK (lat/lon → haversine)
- **Motivation**: Add an objective “traveller friction” signal to help explain listens without hand-curating special cases like UK/Spain/USA (which we know are high, but don’t know exact counts for).
- **Data decision**: Pull country `latitude` and `longitude` from RestCountries `latlng` in the same fetch used for `languages` and `population`.
- **Feature engineering**:
  - compute `uk_distance_km` using haversine from a fixed UK reference point (approx UK centroid: `lat=54.0`, `lon=-2.0`)
  - use `log1p(uk_distance_km)` as the Bayesian GLM covariate (better scaling and less sensitivity to very large distances)
- **Implementation**:
  - `src/data.py`: `add_languages_and_population()` now also stores `latitude`/`longitude`; new `add_uk_distance()` adds `uk_distance_km`
  - `src/cli.py`: both `run` and `bayes-run` pipelines call `add_uk_distance()` when building the dataset via network fetch
  - `src/bayes_model.py`: Bayes feature matrix now includes `log1p_uk_distance_km` alongside `log_population` and `internet_rate`
- **Important caveat**: If you run with `--dataset-csv`, we currently trust the CSV “as-is” and do not backfill lat/lon or distance. So your dataset CSV must already include `latitude`/`longitude` (or `uk_distance_km`) to get this term.

### 2025-12-20 — New calibration constraint: total number of one-listen countries
- **Observation**: With sparse labels, a Poisson GLM can assign many countries \(\mu \\approx 1\), because that maximizes \(P(Y{=}1)=\\mu e^{-\\mu}\) (peak \(\u2248 0.367\)). This can yield implausibly high `P(exactly 1)` for obvious high-listen countries (e.g. France).
- **Decision**: Use the known aggregate `TOTAL_NUMBER_OF_COUNTRIES_WITH_ONE_LISTEN` (currently `6`) as an additional soft constraint:
  - \[
    \\text{observed\_one} + \\sum_{i \\in \\text{unknown}} P(Y_i = 1) \\approx \\texttt{TOTAL\_NUMBER\_OF\_COUNTRIES\_WITH\_ONE\_LISTEN}
    \]
- **Implementation**: Add a second `pm.Potential` constraint (`one_listen_countries_constraint`) with tunable `aggregate_one_sigma`.


* would total number of listens at the time be a useful stat to introduce too? to know N? maybe minus UK listens if we are not using uk listens... but might be hard to get
* given we only have data for say 30 countries, is it helpful to remove countries from the data that we have no data for, and we know won't be viable options? e.g. uk, france, spain, etc.. ? could that actually help narrow the problem? Or like countriew we know there are active listeners, that aren't just people on holiday - as actually the dataset could be split into these two categories of countries, and we are only interested in the ones that have listens from people visiting? 
* I want to start plotting the data on a map. how easy is this? or how easy is it to do in a web app? if option in streamlit this si fine for now butif I publish this I miht try a react app style situation
* might it be possible to estimate total listens if we know they were 38th uk podcast in the charts on may 22nd?
* we can get a list of countries it definitely has been listened in here but not sure how useful itis https://www.podchaser.com/podcasts/what-did-you-do-yesterday-with-5823474/insights
* reflection: we are basically trying to create a model for uk tourism (maybe a big mistake that we are ignoring ireland and australian tourism)
* estimates of most popular countries from rephonic: United Kingdom (68%)
Ireland (11%)
Australia (10%)
United States (7%)
Canada (2%)
New Zealand (2%)
* I kinda wanna refactor the code, maybe start from scratch cause I thihnk it is a bit of a mess at the moment tbh, we probably should consider the way we want the "data pipeline" to work and how we want to view it more holistically than me just tagging on ideasz as we go, now we have something working
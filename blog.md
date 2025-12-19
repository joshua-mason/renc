## renc notes (decision journal)

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

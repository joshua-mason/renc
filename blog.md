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

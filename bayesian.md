## Bayesian / simulation roadmap for renc

Goal: move from a deterministic ranking (`total_score`) to **probabilistic predictions** like:
- `P(country has >=1 listen)`
- `P(country has exactly 1 listen)`
- expected listens `E[listens]`

Where `listens` can be interpreted as a count outcome for a fixed time window (your `DATE_OF_RECORD`).

---

### 0) What we have now (baseline)
We currently compute a heuristic score and rank countries. This is useful, but it is not a generative model and has no uncertainty.

The Bayesian shift is: define a **data-generating process** for listens and then infer its parameters.

---

## Path A (best first Bayesian model): Negative Binomial regression + posterior predictive

### Model
Let `Y_i` be listens from country `i` in the window.

A common Bayesian count model is Negative Binomial:

- Likelihood:
  - `Y_i ~ NegBin(mu_i, alpha)`
- Link:
  - `log(mu_i) = beta0 + beta_pop*log_pop_i + beta_net*net_i + beta_uk*log_uk_i + beta_lang*lang_i + ...`

Why **NegBin**: counts are usually over-dispersed (variance > mean) and Poisson is often too tight.

### Priors (simple, “professional” defaults)
- `beta0 ~ Normal(0, 2)` (or broader if needed)
- `beta_* ~ Normal(0, 1)` (regularizing)
- `alpha ~ Exponential(1)` (overdispersion)

### Output probabilities
Once you sample from the posterior, you can estimate per-country probabilities via posterior predictive simulation:
- `P(Y_i = 1) ≈ mean( y_i_draw == 1 )`
- `P(Y_i >= 1) ≈ mean( y_i_draw >= 1 )`

### What data do we need?
Best: actual per-country listens.
But you currently only have partial labels (a few correct/incorrect for “exactly one”).
So Path A becomes easiest once you add more observed data (see “episode-level patchy” below).

---

## Path B (fits your situation better): Hurdle / zero-inflated model (two-stage)

If most countries are zero, use a two-part model:

### Hurdle model
- Stage 1 (any listens):
  - `Z_i ~ Bernoulli(p_i)` where `Z_i = 1[Y_i > 0]`
  - `logit(p_i) = gamma0 + gamma^T x_i`
- Stage 2 (positive counts):
  - `Y_i | Z_i=1 ~ TruncatedNegBin(mu_i, alpha)`

### Why it helps
It separates:
- “can this country show up at all?” (access/discovery)
- “how many listens, given it shows up?”

This better matches real podcast dynamics (many zeros + heavy tail).

### Output probabilities
You get probabilities directly:
- `P(Y_i = 0) = 1 - p_i`
- `P(Y_i = 1)` comes from combining the hurdle probability with the positive-count distribution.

---

## Path C (very applicable with sparse/aggregate info): ABC / simulation-based Bayes on summary statistics

If you *cannot* write a clean likelihood because you only know summaries like:
- number of countries with any listens
- number with exactly one

…then do simulation-based inference:

### ABC idea
1) Sample parameters from priors
2) Simulate per-country listens from a generative model (like Path B)
3) Compute summary stats from the simulation (e.g. #countries with `Y=1`)
4) Keep/reweight parameter samples that match your observed summaries

This is very “Bayesian + simulation” and matches your current data situation.

---

## How to incorporate your current labels (correct vs incorrect)
Right now you have:
- a small set you believe are **exactly-1** (positives)
- a set you believe are **not exactly-1** (negatives)

Two practical ways to use this:

### 1) Treat them as noisy labels on `Y_i=1`
Define `S_i` = observed label, which can be wrong (label noise).
Model:
- `P(S_i=positive | Y_i=1) = sensitivity`
- `P(S_i=positive | Y_i!=1) = 1 - specificity`

This prevents the model from overfitting to a tiny, imperfect label set.

### 2) Use them as evaluation only (recommended early)
Fit the model using whatever quantitative listen data you can scrape (even patchy), then evaluate with:
- false positives on known-incorrect
- probability mass on known-correct

---

## Where “language factor” fits in Bayesian modeling
Instead of hard-coded multipliers, represent language as a feature with a prior:
- `beta_lang ~ Normal(0, 0.5)` (shrunk toward 0)

This naturally prevents “hacky” overemphasis unless the data supports it.

---

## Episode-level listens (patchy) — why it’s valuable
Even partial per-country listen counts for some episodes/time windows will unlock Path A/B.
With that, you can:
- train on episode `t` outcomes
- include episode-level random effects (some episodes are more global)

Hierarchical extension:
- `log(mu_{i,t}) = beta^T x_i + u_t` where `u_t ~ Normal(0, sigma_episode)`

---

## A practical learning plan (how to get started)

### Step 1: Learn the building blocks
- Bayesian regression basics, priors, posterior predictive checks.
- Count models: Poisson vs Negative Binomial.
- Zero inflation / hurdle models.

### Step 2: Implement a tiny prototype
- Use synthetic data to ensure you can recover parameters.
- Then apply to your country feature table.

### Step 3: Add uncertainty-first outputs
- For each country: `P(Y=1)`, `P(Y>=1)`, credible intervals.

### Step 4: Evaluate against your labels
- Use `Wrong@rank`-style penalties and zone-closeness as diagnostics.

---

## Suggested podcasts (good for learning while walking/shopping)
- **Learning Bayesian Statistics**: long-form Bayes learning, lots of foundational episodes.
- **Quantitude**: stats thinking, causal reasoning, and modeling intuition.
- **Data Skeptic**: accessible ML/statistics topics; useful for intuition and tradeoffs.

(When you’re back at a laptop, I can point you to a few PyMC/Stan tutorial series that map directly to Path A/B.)



## renc

Heuristic model to rank countries by likely podcast listens (population × internet usage × UK affinity × language factor), plus a Streamlit viewer to explore outputs.

### Quickstart

- **Generate a labeled run** (writes `runs/<label>_<timestamp>.csv`):

```bash
uv run python -m src run --label "v0"
```

- **Generate + launch Streamlit**:

```bash
uv run python -m src run --label "v0" --launch
```

- **Launch Streamlit (and pick a saved run in the sidebar)**:

```bash
uv run streamlit run streamlit_app.py
```

### Notes

- Known single-listen countries for comparison live in `src/config.py` (`CORRECT_COUNTRIES`).
- Raw inputs are in `raw/` and feature building is in `src/data.py`.

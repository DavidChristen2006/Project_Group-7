# Equity Hedge Explorer — HSG / LSEG research project

A Streamlit interface for LSEG-backed stock and ETF hedge research. Displays
candidate option strategies, mechanical payoff curves, the stock–ETF joint
scenario heatmap, and available ML model comparisons in one guided dashboard.
**No brokerage trades are executed.** Data can be stale; options have
expiration and contract identity risks. This is not investment advice.

## Quick start (Windows / macOS / Linux)

1. Clone or download the GitHub repository and open a terminal **inside it**.
2. Use **Python 3.11+** and install: `python -m pip install -r requirements.txt`.
3. Copy `lseg-data.config.example.json` to `lseg-data.config.json`, filling in
   your **own** LSEG Platform Session credentials. **Never commit** the latter.
   Open LSEG Workspace and ensure your account has the relevant data access.
   The application uses `ld.open_session()` with the configured default
   Platform Session (no explicit session-name override).
4. Start with `python -m streamlit run hedging_app.py`, or `./run.ps1`
   (Windows PowerShell). You do **not** need to edit Python path constants.

The repository automatically resolves bundled code and the standard data
directory using `Path(__file__).resolve()`. Historical data and trained
model artifacts are intentionally *not* in this GitHub-ready package due
to size/licensing. If you have existing datasets, keep their original
directory tree under `data/processed/`. Without those files, saved ML
research panels will explain what is missing; live data requires LSEG.

## Main workflow

1. Choose **Hedge Calculator** (the default) for a short position form with
   stock, shares, NYSE hedge duration, optional ETF, and risk appetite.
2. Press **Calculate hedge once**. The app retrieves the stock and eligible
   direct options, attempts the selected ETF analysis, and (where 252-session
   beta/R-squared are available) automatically fetches the historical paired
   closes for risk comparisons. The stock result still works if ETF retrieval
   or historical calibration fails. Changing charts does **not** rerun LSEG.
3. The calculator displays the ML recommendation, direct-option expiration
   payoff curve, ETF payoff and stock/ETF return heatmap. The heatmap defaults
   to a 50% ETF hedge; its strategy chooser updates locally without buttons.
4. **Advanced Hedge Research** reuses the same snapshots and opens contract
   details, quote audits, independent/linked scenarios, broad-market forecasts,
   8-year historical calibration, and automatic paired-block bootstrap.
   Heavy simulations are cached by their inputs. Advanced parameters are
   only applied when the next hedge calculation is submitted.
5. **Historical ML Research** starts with a concise read-only **Overview**.
   The five detailed pages remain available: Stock Risk Forecasts, Model
   Comparison, Experiment Results, Hedging Performance and Decision Quality
   Audit. Missing saved datasets are reported rather than fabricated.

### Cross-currency ETF research

**This does not add exchange-rate conversion.** Stock and ETF quotes may be
in different currencies; combined P&L, premium arithmetic and hedge quantities
therefore may **not** be economically meaningful across currencies. No FX gain,
loss or financing is modeled, and the current currency pairing is not verified.
Do not treat these values as reliable cross-currency hedging instructions.

### Automatic calculations and controls

Only the main **Calculate hedge** action fetches the live position and selected
ETF. Saved results are shared between the calculator and advanced pages.
Chart-only selectors and scenario cursors rerender immediately; they never
need a new live-data retrieval. Research-specific configuration changes require
another Calculate action. Data-dependent historical calibration and 3,000-path
bootstrap are cached (a failed retrieval is reported; they do not invent data).

## Repository layout

| Location | Purpose |
| --- | --- |
| `hedging_app.py` | Streamlit entry point / research orchestration |
| `dashboard_visuals.py` | Presentation-only, reusable Plotly charts |
| `project_paths.py` | Central repository-relative path helpers |
| `data_pipeline/` | Active source, payoff, dataset and evaluation modules |
| `ml/`, `pipeline/`, `lseg_api/` | Reusable ML, historical and market-data logic |
| `risk_aware_hedge.py`, `six_horizon_ml.py` | Saved-model inference and research |
| `research_archive/` | Optional older diagnostic/experimental scripts in ZIP |
| `tests/` | Reproducible offline structural and plotting tests |
| `docs/` | Architecture, validation limitations and GenAI disclosures |

### Scientific interpretation

- Historical validation and test datasets must not be mixed for model
  selection; some saved exploratory tests are **not independent holdouts**.
- Model training covers six predefined horizons: 21, 42, 63, 84, 126, 252
  NYSE sessions. Intermediate/extrapolated horizons are not separately fitted.
- Contract multipliers, strike and option-expiry mismatches, historical root
  identity, quote staleness, basis risk, CVaR sample size, transaction costs,
  and short-ETF funding constraints require user scrutiny.
- No raw LSEG data are redistributed; users need authorization to collect
  their own data. Academic research results are not trade instructions.

## Verify the refactor

`py -m pytest -q`


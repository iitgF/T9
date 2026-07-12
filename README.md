# Volatility Forecasting: Classical vs ML vs Deep Learning

Term project (B.Sc. Hons Data Science & AI, IIT Guwahati, Trimester 9) comparing
one-day-ahead equity-volatility forecasts from classical econometric baselines,
gradient-boosted trees, and an LSTM, under a strict walk-forward protocol with no
look-ahead bias.

**Headline result:** on 18.5 years of S&P 500 daily data (2008–2026), a monthly
re-estimated **EGARCH(1,1)** delivers the best out-of-sample QLIKE (1.504), statistically
tied with GJR-GARCH (Diebold–Mariano p = 0.47). Both leverage models significantly beat
plain GARCH(1,1) (p < 0.0001) on a test window containing the 2022 bear market and the
April-2025 tariff shock. The LSTM ranks third, ahead of plain GARCH and statistically
indistinguishable from the winner (p = 0.19), and the learned models take both
point-accuracy metrics (LSTM the RMSE, XGBoost the MAE). Yet no learned model beats the
leverage GARCH family on the risk-relevant QLIKE criterion.

## Leaderboard (out-of-sample, 1,395 test days, 2020-12-07 to 2026-06-29)

| Model | RMSE | MAE | QLIKE |
|---|---|---|---|
| EGARCH(1,1) | 0.00728 | 0.00556 | **1.5042** |
| GJR-GARCH(1,1) | 0.00731 | 0.00554 | 1.5091 |
| LSTM | **0.00686** | 0.00499 | 1.5287 |
| GARCH(1,1) | 0.00751 | 0.00576 | 1.5569 |
| EWMA(0.94) | 0.00750 | 0.00573 | 1.5814 |
| XGBoost | 0.00691 | **0.00496** | 1.5911 |
| Rolling(21) | 0.00762 | 0.00571 | 1.6344 |
| Naive | 0.00951 | 0.00672 | 5440.50 |

## Repository layout

```
volforecast/                 modular pipeline (importable package)
  config.py                  run parameters and the shared walk-forward split
  data.py                    Yahoo Finance loader + synthetic GARCH generator
  metrics.py                 RMSE, MAE, QLIKE, Diebold–Mariano test
  features.py                engineered features and the learning target
  baselines.py               naive, rolling, EWMA, GARCH/GJR-GARCH/EGARCH
  ml_xgboost.py              walk-forward gradient boosting
  ml_lstm.py                 LSTM with early stopping and anti-leakage guards
  evaluate.py                leaderboard + significance helpers
  pipeline.py                process-parallel orchestration (~1 min end-to-end)

volatility_forecasting_consolidated.ipynb   full annotated study, every module inline
volatility_forecasting_starter.ipynb        development notebook, imports from volforecast
report.tex                                   term-project report (compile on Overleaf)
T9.pdf                                       compiled report
Project_Video_Presentation.pptx              video presentation deck
fig_*.png                                    figures used by the report
```

## Reproducing

```bash
pip install numpy pandas matplotlib scipy statsmodels arch xgboost torch yfinance
python -c "from volforecast.pipeline import run_all_forecasts; \
          from volforecast.config import CFG; from volforecast.data import load_returns; \
          from volforecast.evaluate import build_leaderboard; \
          r = load_returns(CFG); fc = run_all_forecasts(r, CFG); \
          print(build_leaderboard(fc, r, CFG).round(6))"
```

Or run `volatility_forecasting_consolidated.ipynb` top to bottom.

All seeds are fixed (42) and every model shares a single train/test split, so results are
reproducible. Tested with Python 3.13, numpy 2.4, pandas 3.0, arch 8.0, xgboost 3.3,
torch 2.12, statsmodels 0.14.

## Note on AI assistance

AI-assisted tooling (Claude Code) was used for code refactoring, optimization, and drafting
support. All modelling decisions, verification of results, and the final text are the
author's responsibility, and all reported numbers were reproduced end-to-end from the
committed code.

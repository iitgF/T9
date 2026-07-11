# Volatility Forecasting: Classical vs ML vs Deep Learning

Term project (B.Sc. Hons Data Science & AI, IIT Guwahati, Trimester 9) comparing
one-day-ahead equity-volatility forecasts from classical econometric baselines,
gradient-boosted trees, and an LSTM, under a strict walk-forward protocol with no
look-ahead bias.

**Headline result:** on 17 years of S&P 500 daily data (2008–2024), a monthly
re-estimated **GARCH(1,1)** delivers the best out-of-sample QLIKE (1.488). EGARCH and
GJR-GARCH are statistically indistinguishable behind it (Diebold–Mariano p = 0.87).
XGBoost attains the lowest RMSE and MAE, yet no learned model beats the GARCH family on
the risk-relevant QLIKE criterion. The classic finding that GARCH is hard to beat on
daily data reproduces cleanly.

## Leaderboard (out-of-sample, 1,284 test days)

| Model | RMSE | MAE | QLIKE |
|---|---|---|---|
| GARCH(1,1) | 0.00887 | 0.00633 | **1.4879** |
| EGARCH(1,1) | 0.00852 | 0.00603 | 1.4943 |
| GJR-GARCH(1,1) | 0.00881 | 0.00611 | 1.4974 |
| EWMA(0.94) | 0.00934 | 0.00657 | 1.5161 |
| XGBoost | **0.00816** | **0.00547** | 1.5567 |
| Rolling(21) | 0.00948 | 0.00646 | 1.5798 |
| LSTM | 0.00965 | 0.00570 | 1.5915 |
| Naive | 0.01114 | 0.00747 | 1192.69 |

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

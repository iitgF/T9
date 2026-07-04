# Orchestration: the three GARCH variants, XGBoost and the LSTM are mutually independent
# CPU-bound pipelines (repeated MLE / boosting / SGD fits), so they run in separate
# processes. Threads would not help here (GIL); processes cut wall-clock time to roughly
# the slowest single pipeline. Each pipeline is internally unchanged — same walk-forward
# semantics, same numbers as running the cells sequentially.
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from .baselines import bl_naive, bl_rolling, bl_ewma, bl_garch_family
from .ml_lstm import bl_lstm
from .ml_xgboost import bl_xgboost


def _fc_only(fn, *args, **kwargs):
    # bl_xgboost / bl_lstm return (forecast, fitted_model); the model objects stay in the
    # worker — only the forecast array crosses the process boundary.
    out = fn(*args, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def run_all_forecasts(r, cfg, parallel=True):
    # cheap closed-form baselines: run inline
    forecasts = {
        "Naive":       bl_naive(r),
        "Rolling(21)": bl_rolling(r, cfg["roll_window"]),
        "EWMA(0.94)":  bl_ewma(r, cfg["ewma_lambda"]),
    }
    jobs = {
        "GARCH(1,1)":     partial(bl_garch_family, r, cfg, vol="GARCH", o=0),
        "GJR-GARCH(1,1)": partial(bl_garch_family, r, cfg, vol="GARCH", o=1),
        "EGARCH(1,1)":    partial(bl_garch_family, r, cfg, vol="EGARCH", o=1),
        "XGBoost":        partial(_fc_only, bl_xgboost, r, cfg),
        "LSTM":           partial(_fc_only, bl_lstm, r, cfg),
    }
    t0 = time.perf_counter()
    if parallel:
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            futs = {name: ex.submit(job) for name, job in jobs.items()}
            for name, fut in futs.items():
                forecasts[name] = fut.result()
                print(f"  {name:<15s} done  [{time.perf_counter() - t0:5.0f}s]")
    else:                                     # serial fallback (debugging / profiling)
        for name, job in jobs.items():
            forecasts[name] = job()
            print(f"  {name:<15s} done  [{time.perf_counter() - t0:5.0f}s]")
    return forecasts

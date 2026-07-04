import numpy as np
import pandas as pd

from .config import EPS, test_start_index
from .metrics import rmse, mae, qlike, qlike_series, dm_test

BASELINE_NAMES = ["Naive", "Rolling(21)", "EWMA(0.94)",
                  "GARCH(1,1)", "GJR-GARCH(1,1)", "EGARCH(1,1)"]


def build_leaderboard(forecasts, r, cfg):
    sl = slice(test_start_index(r, cfg), len(r))
    proxy_test = (r.values ** 2)[sl]          # realized-variance proxy = squared return
    vol_true = np.sqrt(proxy_test)

    rows = []
    for name, fc in forecasts.items():
        vp = fc[sl]
        mask = np.isfinite(vp)
        vp_m, px_m, vt_m = vp[mask], proxy_test[mask], vol_true[mask]
        vf = np.sqrt(np.clip(vp_m, EPS, None))
        rows.append({"Model": name, "N": mask.sum(),
                     "RMSE": rmse(vf, vt_m), "MAE": mae(vf, vt_m), "QLIKE": qlike(vp_m, px_m)})
    return pd.DataFrame(rows).set_index("Model").sort_values("QLIKE")


# DM test on QLIKE: best model overall vs the best *baseline* (skipping the best model itself).
def dm_vs_best_baseline(forecasts, leaderboard, r, cfg, baselines=BASELINE_NAMES):
    sl = slice(test_start_index(r, cfg), len(r))
    proxy_test = (r.values ** 2)[sl]

    best = leaderboard.index[0]
    ref = leaderboard.loc[[m for m in baselines if m != best], "QLIKE"].idxmin()
    m = np.isfinite(forecasts[best][sl]) & np.isfinite(forecasts[ref][sl])
    stat, p = dm_test(qlike_series(forecasts[best][sl][m], proxy_test[m]),
                      qlike_series(forecasts[ref][sl][m],  proxy_test[m]))
    return best, ref, stat, p

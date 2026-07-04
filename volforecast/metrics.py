# Evaluation utilities — implemented and tested, do not change the logic.
import numpy as np
from scipy.stats import t as tdist

from .config import EPS


def rmse(vol_f, vol_true):
    vol_f, vol_true = np.asarray(vol_f), np.asarray(vol_true)
    return float(np.sqrt(np.mean((vol_f - vol_true) ** 2)))


def mae(vol_f, vol_true):
    return float(np.mean(np.abs(np.asarray(vol_f) - np.asarray(vol_true))))


def qlike_series(var_f, proxy_var):
    v = np.clip(np.asarray(var_f), EPS, None)
    p = np.asarray(proxy_var) + EPS
    return p / v - np.log(p / v) - 1.0        # per-observation QLIKE loss


def qlike(var_f, proxy_var):
    return float(np.mean(qlike_series(var_f, proxy_var)))


# Diebold-Mariano. loss1, loss2 = per-obs loss arrays (from qlike_series or squared error).
# Returns (stat, p_value). Negative stat => model 1 has lower loss (better).
def dm_test(loss1, loss2, h=1):
    d = np.asarray(loss1) - np.asarray(loss2)
    T = len(d); dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    s = gamma0
    for k in range(1, h):                      # autocovariances up to h-1 (0 for 1-step)
        s += 2 * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    stat = dbar / np.sqrt(s / T)
    stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)   # HLN correction
    p = 2 * tdist.cdf(-abs(stat), df=T - 1)
    return float(stat), float(p)

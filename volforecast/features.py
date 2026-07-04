import numpy as np
import pandas as pd

from .config import EPS


def make_features(r, max_lag=10):
    X = pd.DataFrame(index=r.index)
    for k in range(1, max_lag + 1):
        X[f"r_lag{k}"]  = r.shift(k)
        X[f"r2_lag{k}"] = (r ** 2).shift(k)
    X["rv5"]  = (r ** 2).shift(1).rolling(5).mean()
    X["rv21"] = (r ** 2).shift(1).rolling(21).mean()
    X["absmean5"] = r.abs().shift(1).rolling(5).mean()
    # TODO: add features you can justify — realized skew/kurt, day-of-week, VIX, etc.
    return X


# target: log of a smoothed realized variance (avoids the noisy-proxy blow-up).
# rv_target looks 5 days FORWARD (today + next 4), so it is only known once those returns
# have occurred — a `buffer` of stale days is left out of every training refit to respect that.
def make_target(r):
    rv_target = (r ** 2).rolling(5).mean().shift(-4)
    return np.log(rv_target + EPS)

# Walk-forward XGBoost on engineered features, predicting log smoothed realized variance.
import numpy as np

from .config import test_start_index
from .features import make_features, make_target


def bl_xgboost(r, cfg, refit_every=None, buffer=5):
    import xgboost as xgb
    refit_every = cfg["garch_refit_every"] if refit_every is None else refit_every

    X = make_features(r)
    y = make_target(r)
    full = X.join(y.rename("y"))

    T = len(r)
    fc = np.full(T, np.nan)
    Xv = X.values                               # hoisted once — no per-day .iloc in the loop
    bad = X.isna().any(axis=1).values
    start = test_start_index(r, cfg)
    model = None
    # NOTE: warm-starting each refit from the previous booster (xgb_model=) was considered
    # and rejected — it either grows the ensemble every refit or changes which trees the
    # model contains, so predictions drift from the reference run; a from-scratch 400-tree
    # depth-3 fit on ~3-4k rows takes only a couple of seconds anyway.
    for t0 in range(start, T, refit_every):
        # only rows whose 5-day-forward target is fully realized before day t0 may be used
        # for training — this is what keeps the periodic refit walk-forward-honest.
        train = full.iloc[:t0 - buffer].dropna()
        Xtr, ytr = train.drop(columns="y"), train["y"]
        model = xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8)
        model.fit(Xtr, ytr)
        # the same model serves every day until the next refit, and each day's features use
        # past data only — so predicting the whole block at once is identical to per-day calls
        t1 = min(t0 + refit_every, T)
        ok = ~bad[t0:t1]
        if ok.any():
            fc[np.arange(t0, t1)[ok]] = np.exp(model.predict(Xv[t0:t1][ok]))
    return fc, model

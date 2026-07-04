# Classical baselines. Each returns a full-length array of one-day-ahead variance
# forecasts where the forecast for day t uses information up to day t-1 only.
import numpy as np

from .config import test_start_index


def bl_naive(r):
    # tomorrow's variance = today's squared return
    return (r.shift(1) ** 2).values


def bl_rolling(r, k):
    # variance of the previous k returns
    return r.shift(1).rolling(k).var().values


def bl_ewma(r, lam):
    v = np.empty(len(r)); v[:] = np.nan
    v[0] = r.iloc[:22].var()
    rv = r.values
    for t in range(1, len(r)):
        v[t] = lam * v[t-1] + (1 - lam) * rv[t-1] ** 2   # uses r_{t-1}: no look-ahead
    return v


# GARCH-family baseline: refit params every `garch_refit_every` days, roll the variance
# recursion forward daily with the observed returns in between (fast AND valid).
# One function drives plain GARCH(1,1), GJR-GARCH (o=1, leverage via a threshold term) and
# EGARCH (o=1, leverage via the standardized-residual term) — recursions taken directly from
# the `arch` package's own garch_recursion / egarch_recursion so they match the fitted params.
def bl_garch_family(r, cfg, vol="GARCH", o=0, dist="normal"):
    from arch import arch_model
    rr = r.values * 100.0                       # scale up for numerical stability
    T = len(r); fc = np.full(T, np.nan)
    om = al = be = ga = None                     # params hoisted to floats at each refit
    sig2 = None; log_sig2 = None; z = None
    c = np.sqrt(2 / np.pi)                       # E|z| for a standard normal z (EGARCH)
    start = test_start_index(r, cfg)
    for t in range(start, T):
        if (t - start) % cfg["garch_refit_every"] == 0:
            lo = 0 if cfg["garch_window"] is None else max(0, t - cfg["garch_window"])
            res = arch_model(rr[lo:t], mean="Zero", vol=vol, p=1, o=o, q=1,
                             dist=dist).fit(disp="off")
            p = res.params
            om, al, be = p["omega"], p["alpha[1]"], p["beta[1]"]
            ga = p["gamma[1]"] if o == 1 else 0.0
            last_sigma = res.conditional_volatility[-1]
            sig2 = last_sigma ** 2
            log_sig2 = np.log(sig2)
            z = rr[t-1] / last_sigma                     # standardized residual, for EGARCH
        elif vol == "EGARCH":
            log_sig2 = om + be * log_sig2 + al * (abs(z) - c) + ga * z
            sig2 = np.exp(log_sig2)
        elif o == 1:                                       # GJR-GARCH (threshold ARCH)
            lev = ga * rr[t-1] ** 2 if rr[t-1] < 0 else 0.0
            sig2 = om + al * rr[t-1] ** 2 + lev + be * sig2
        else:                                               # plain GARCH(1,1)
            sig2 = om + al * rr[t-1] ** 2 + be * sig2
        fc[t] = sig2 / (100.0 ** 2)             # back to original scale
        if vol == "EGARCH":
            z = rr[t] / np.sqrt(sig2)            # today's standardized residual, feeds tomorrow
    return fc

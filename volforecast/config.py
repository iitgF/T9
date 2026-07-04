EPS = 1e-12  # numerical floor so we never divide by / log zero

CFG = {
    "ticker":       "^GSPC",     # S&P 500. Try "^NSEI" (Nifty 50), "BTC-USD", "GC=F" (gold)
    "start":        "2008-01-01",
    "end":          "2024-12-31",
    "test_frac":    0.30,        # last 30% of the sample is the out-of-sample test window
    "roll_window":  21,          # window for the rolling-std baseline (~1 trading month)
    "ewma_lambda":  0.94,        # RiskMetrics decay
    "garch_refit_every": 22,     # refit GARCH params every N days (roll recursion in between)
    "garch_window": None,        # None = expanding window; or an int for a rolling estimation window
}


def test_start_index(r, cfg):
    # first index of the out-of-sample walk-forward test window
    return int(len(r) * (1 - cfg["test_frac"]))

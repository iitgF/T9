import numpy as np
import pandas as pd


def load_returns(cfg):
    import yfinance as yf
    px = yf.download(cfg["ticker"], start=cfg["start"], end=cfg["end"],
                     auto_adjust=True, progress=False)
    close = px["Close"].dropna()
    if isinstance(close, pd.DataFrame):     # yfinance sometimes returns a 1-col frame
        close = close.iloc[:, 0]
    r = np.log(close).diff().dropna()
    r.name = "ret"
    return r


def synthetic_returns(n=4000, omega=2e-6, alpha=0.08, beta=0.90, seed=42):
    rng = np.random.default_rng(seed)
    var = np.empty(n); ret = np.empty(n); var[0] = omega / (1 - alpha - beta); ret[0] = 0.0
    for t in range(1, n):
        var[t] = omega + alpha * ret[t-1]**2 + beta * var[t-1]
        ret[t] = rng.normal(0, np.sqrt(var[t]))
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(ret, index=idx, name="ret")

"""
Fetch free historical data for 0DTE options backtesting.

Data sources:
- yfinance: SPY/QQQ intraday price data (free, 1-min bars up to 30 days, daily bars years back)
- CBOE: VIX data for implied volatility context
- Synthetic options pricing via Black-Scholes when real options data unavailable

For proper backtesting with real options tick data, upgrade to:
- ThetaData ($40-80/mo) — best value
- Databento (usage-based) — OPRA tick data
- CBOE DataShop — official source
"""

import os
import datetime
import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy.stats import norm

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_underlying_intraday(symbol: str, period: str = "30d", interval: str = "1m") -> pd.DataFrame:
    """
    Fetch intraday price data from yfinance.
    
    Free tier limitations:
    - 1m data: max 30 days history
    - 5m data: max 60 days
    - 15m data: max 60 days
    - 1h data: max 730 days
    """
    print(f"[Data] Fetching {symbol} intraday ({interval}, {period})...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    
    # Clean up
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    
    # Save
    fname = f"{symbol}_intraday_{interval}_{period}.parquet"
    df.to_parquet(DATA_DIR / fname)
    print(f"[Data] Saved {len(df)} bars to {fname}")
    
    return df


def fetch_underlying_daily(symbol: str, years: int = 5) -> pd.DataFrame:
    """Fetch daily OHLCV data going back several years."""
    print(f"[Data] Fetching {symbol} daily ({years}y)...")
    end = datetime.date.today()
    start = end - datetime.timedelta(days=years * 365)
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d")
    
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    
    fname = f"{symbol}_daily_{years}y.parquet"
    df.to_parquet(DATA_DIR / fname)
    print(f"[Data] Saved {len(df)} daily bars to {fname}")
    
    return df


def fetch_vix(period: str = "5y") -> pd.DataFrame:
    """Fetch VIX index for implied volatility context."""
    print(f"[Data] Fetching VIX ({period})...")
    vix = yf.Ticker("^VIX")
    df = vix.history(period=period)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close"]]
    df.columns = ["open", "high", "low", "close"]
    
    df.to_parquet(DATA_DIR / f"VIX_{period}.parquet")
    print(f"[Data] Saved {len(df)} VIX bars")
    
    return df


# ==================== Black-Scholes Synthetic Options ====================

def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes option price."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes delta."""
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes theta (per day)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    
    if option_type == "call":
        theta = common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    return theta / 365  # per day


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (per 1% vol change)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def generate_option_chain(
    S: float,
    T: float,  # time to expiry in years (e.g., 6.5/252 for 1 trading day)
    r: float = 0.05,
    sigma: float = 0.20,
    strike_range_pct: float = 0.05,
    strike_interval: float = 1.0,
) -> pd.DataFrame:
    """
    Generate a synthetic option chain using Black-Scholes.
    
    For 0DTE: T should be fraction of a day (e.g., 4h remaining = 4/(252*6.5))
    
    Args:
        S: Current underlying price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Implied volatility (annualized)
        strike_range_pct: How far OTM to generate strikes (e.g., 0.05 = ±5%)
        strike_interval: Strike price interval (1 for SPX/XSP, 1 for SPY)
    
    Returns:
        DataFrame with columns: strike, type, price, delta, gamma, theta, vega
    """
    strikes = np.arange(
        S * (1 - strike_range_pct),
        S * (1 + strike_range_pct),
        strike_interval
    )
    
    rows = []
    for K in strikes:
        for opt_type in ["call", "put"]:
            price = bs_price(S, K, T, r, sigma, opt_type)
            delta = bs_delta(S, K, T, r, sigma, opt_type)
            gamma = bs_gamma(S, K, T, r, sigma)
            theta = bs_theta(S, K, T, r, sigma, opt_type)
            vega = bs_vega(S, K, T, r, sigma)
            
            rows.append({
                "strike": K,
                "type": opt_type,
                "price": round(price, 2),
                "bid": round(price * 0.95, 2),  # synthetic bid/ask spread
                "ask": round(price * 1.05, 2),
                "delta": round(delta, 4),
                "gamma": round(gamma, 6),
                "theta": round(theta, 4),
                "vega": round(vega, 4),
            })
    
    return pd.DataFrame(rows)


def estimate_intraday_iv(vix_close: float, time_of_day_hours: float = 4.0) -> float:
    """
    Estimate intraday IV from VIX.
    
    VIX represents 30-day SPX IV. For 0DTE, actual IV is typically
    higher due to gamma risk. This is a rough approximation.
    
    Args:
        vix_close: Previous day's VIX close
        time_of_day_hours: Hours remaining in trading day
    """
    base_iv = vix_close / 100
    # 0DTE IV tends to be 1.2-1.5x the 30-day IV
    # Later in the day, IV can spike or collapse
    intraday_multiplier = 1.3 + (0.2 * (6.5 - time_of_day_hours) / 6.5)
    return base_iv * intraday_multiplier


if __name__ == "__main__":
    # Fetch data for all target instruments
    for sym in ["SPY", "QQQ"]:
        try:
            fetch_underlying_daily(sym, years=5)
            fetch_underlying_intraday(sym, period="30d", interval="1m")
        except Exception as e:
            print(f"[Error] {sym}: {e}")
    
    # VIX for IV estimation
    try:
        fetch_vix("5y")
    except Exception as e:
        print(f"[Error] VIX: {e}")
    
    # Demo: Generate synthetic option chain
    print("\n--- Synthetic SPX Option Chain (S=6000, 4h to expiry, VIX=15) ---")
    chain = generate_option_chain(
        S=6000,
        T=4 / (252 * 6.5),  # 4 hours remaining
        sigma=0.15 * 1.3,  # VIX 15 * intraday multiplier
        strike_range_pct=0.02,
        strike_interval=5.0  # SPX strikes every $5
    )
    print(chain[chain["type"] == "put"].head(20).to_string(index=False))

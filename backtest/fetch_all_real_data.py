"""
Fetch real historical options data from ALL available sources.

Source 1: yfinance — current option chains (free, no auth)
Source 2: Tradier Sandbox — historical daily OHLCV + Greeks (free, needs token)
Source 3: ThetaData — tick-level historical (paid $40/mo, needs token)

Run: python fetch_all_real_data.py
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ==================== Source 1: yfinance (free, no auth) ====================

def fetch_yfinance_chains(symbol: str = "SPY", max_expirations: int = 10) -> pd.DataFrame:
    """
    Fetch all available option chains from yfinance.
    Gets current bid/ask/IV for all strikes and expirations.
    No historical data but gives us real market pricing to validate BS model.
    """
    print("\n" + "=" * 70)
    print(f"📊 SOURCE 1: yfinance — Current Options Chains for {symbol}")
    print("=" * 70)
    
    ticker = yf.Ticker(symbol)
    expirations = ticker.options
    
    if not expirations:
        print("  No options data available")
        return pd.DataFrame()
    
    print(f"  {len(expirations)} expirations available")
    
    # Get underlying price
    hist = ticker.history(period="1d")
    underlying_price = hist["Close"].iloc[-1] if not hist.empty else 0
    print(f"  {symbol} price: ${underlying_price:.2f}")
    
    all_chains = []
    for i, exp in enumerate(expirations[:max_expirations]):
        try:
            chain = ticker.option_chain(exp)
            
            for side, df in [("call", chain.calls), ("put", chain.puts)]:
                df = df.copy()
                df["type"] = side
                df["expiration"] = exp
                df["underlying_price"] = underlying_price
                
                # Calculate days to expiry
                exp_date = pd.to_datetime(exp)
                today = pd.to_datetime(datetime.now().date())
                df["dte"] = (exp_date - today).days
                
                all_chains.append(df)
            
            n_calls = len(chain.calls)
            n_puts = len(chain.puts)
            print(f"  [{i+1}/{min(len(expirations), max_expirations)}] {exp} "
                  f"(DTE={df['dte'].iloc[0]}): {n_calls}C + {n_puts}P")
            
        except Exception as e:
            print(f"  [{i+1}] {exp}: error — {str(e)[:60]}")
        
        time.sleep(0.3)
    
    if all_chains:
        combined = pd.concat(all_chains, ignore_index=True)
        fname = DATA_DIR / f"{symbol}_options_all_chains.csv"
        combined.to_csv(fname, index=False)
        print(f"\n  ✅ Saved {len(combined)} contracts to {fname}")
        
        # Analysis: bid/ask spread by DTE
        print(f"\n  📊 Bid/Ask Spread Analysis:")
        for dte in sorted(combined["dte"].unique())[:5]:
            subset = combined[(combined["dte"] == dte) & (combined["bid"] > 0)]
            if subset.empty:
                continue
            subset = subset.copy()
            subset["spread"] = subset["ask"] - subset["bid"]
            subset["spread_pct"] = subset["spread"] / ((subset["bid"] + subset["ask"]) / 2) * 100
            
            puts_otm = subset[(subset["type"] == "put") & 
                             (subset["strike"] < underlying_price * 0.98)]
            
            if not puts_otm.empty:
                avg_spread = puts_otm["spread_pct"].mean()
                med_spread = puts_otm["spread_pct"].median()
                print(f"    DTE={dte}: OTM puts avg spread {avg_spread:.1f}%, "
                      f"median {med_spread:.1f}% ({len(puts_otm)} contracts)")
        
        # Real IV surface
        print(f"\n  📊 Implied Volatility Surface (OTM puts):")
        for dte in sorted(combined["dte"].unique())[:5]:
            subset = combined[(combined["dte"] == dte) & 
                             (combined["type"] == "put") & 
                             (combined["impliedVolatility"] > 0) &
                             (combined["strike"] < underlying_price)]
            if subset.empty:
                continue
            
            # Group by moneyness
            subset = subset.copy()
            subset["moneyness"] = (subset["strike"] / underlying_price * 100).round(0)
            
            iv_by_money = subset.groupby("moneyness")["impliedVolatility"].mean()
            sample = iv_by_money.tail(8)
            ivs = " | ".join(f"{m:.0f}%={v:.1%}" for m, v in sample.items())
            print(f"    DTE={dte}: {ivs}")
        
        return combined
    
    return pd.DataFrame()


# ==================== Source 2: Tradier Sandbox (free, needs token) ====================

def fetch_tradier_data(symbol: str = "SPY") -> pd.DataFrame:
    """Fetch from Tradier Sandbox API."""
    
    token = os.environ.get("TRADIER_TOKEN", "")
    if not token:
        token_file = Path(__file__).parent.parent / ".tradier_token"
        if token_file.exists():
            token = token_file.read_text().strip()
    
    if not token:
        print("\n" + "=" * 70)
        print("📊 SOURCE 2: Tradier Sandbox — SKIPPED (no token)")
        print("=" * 70)
        print("""
  To enable: 
  1. Sign up free at https://tradier.com
  2. Go to Settings → API Access → Get Sandbox Token
  3. Save: echo "YOUR_TOKEN" > /root/clawd/0dte-options-bot/.tradier_token
""")
        return pd.DataFrame()
    
    print("\n" + "=" * 70)
    print(f"📊 SOURCE 2: Tradier Sandbox — Historical Options for {symbol}")
    print("=" * 70)
    
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    base = "https://sandbox.tradier.com/v1"
    
    # Get expirations
    r = requests.get(f"{base}/markets/options/expirations",
                     params={"symbol": symbol}, headers=headers)
    data = r.json()
    exps = data.get("expirations", {}).get("date", [])
    if isinstance(exps, str): exps = [exps]
    
    print(f"  {len(exps)} expirations | Next 5: {exps[:5]}")
    
    # Get chains with greeks for first 5 expirations
    all_chains = []
    for exp in exps[:5]:
        r = requests.get(f"{base}/markets/options/chains",
                        params={"symbol": symbol, "expiration": exp, "greeks": "true"},
                        headers=headers)
        data = r.json()
        
        options = data.get("options", {})
        if options is None:
            continue
        opts = options.get("option", [])
        if isinstance(opts, dict): opts = [opts]
        
        for opt in opts:
            greeks = opt.get("greeks", {}) or {}
            all_chains.append({
                "symbol": opt.get("symbol", ""),
                "type": opt.get("option_type", ""),
                "strike": opt.get("strike", 0),
                "expiration": exp,
                "bid": opt.get("bid", 0) or 0,
                "ask": opt.get("ask", 0) or 0,
                "last": opt.get("last", 0) or 0,
                "volume": opt.get("volume", 0) or 0,
                "open_interest": opt.get("open_interest", 0) or 0,
                "delta": greeks.get("delta", 0) or 0,
                "gamma": greeks.get("gamma", 0) or 0,
                "theta": greeks.get("theta", 0) or 0,
                "vega": greeks.get("vega", 0) or 0,
                "iv": greeks.get("mid_iv", 0) or 0,
                "rho": greeks.get("rho", 0) or 0,
            })
        
        print(f"  {exp}: {len(opts)} contracts")
        time.sleep(0.5)
    
    if all_chains:
        df = pd.DataFrame(all_chains)
        df.to_csv(DATA_DIR / f"{symbol}_tradier_chains.csv", index=False)
        print(f"\n  ✅ Saved {len(df)} contracts with real Greeks")
        
        # Show real delta vs strike
        print(f"\n  📊 Real Greeks (nearest expiry puts, delta 5-20):")
        nearest = df[(df["expiration"] == exps[0]) & 
                     (df["type"] == "put") & 
                     (abs(df["delta"]) >= 0.05) & 
                     (abs(df["delta"]) <= 0.20)]
        nearest = nearest.sort_values("strike", ascending=False)
        
        for _, row in nearest.head(15).iterrows():
            mid = (row["bid"] + row["ask"]) / 2
            spread_pct = (row["ask"] - row["bid"]) / mid * 100 if mid > 0 else 0
            print(f"    K={row['strike']:.0f} δ={row['delta']:+.3f} IV={row['iv']:.1%} "
                  f"bid={row['bid']:.2f} ask={row['ask']:.2f} spread={spread_pct:.1f}% "
                  f"θ={row['theta']:.3f} OI={row['open_interest']}")
        
        return df
    
    return pd.DataFrame()


# ==================== Source 3: ThetaData ($40/mo) ====================

def fetch_thetadata(symbol: str = "SPY") -> pd.DataFrame:
    """Fetch from ThetaData API (requires subscription)."""
    
    token = os.environ.get("THETADATA_TOKEN", "")
    if not token:
        token_file = Path(__file__).parent.parent / ".thetadata_token"
        if token_file.exists():
            token = token_file.read_text().strip()
    
    if not token:
        print("\n" + "=" * 70)
        print("📊 SOURCE 3: ThetaData — SKIPPED (no token)")
        print("=" * 70)
        print("""
  To enable ($40/mo — best historical options data):
  1. Sign up at https://www.thetadata.net
  2. Subscribe to Value plan ($40/mo) or higher
  3. Get API key from dashboard
  4. Save: echo "YOUR_KEY" > /root/clawd/0dte-options-bot/.thetadata_token
  
  Gets you: 4+ years of 1-minute options data with full Greeks!
""")
        return pd.DataFrame()
    
    print("\n" + "=" * 70)
    print(f"📊 SOURCE 3: ThetaData — Historical Options for {symbol}")
    print("=" * 70)
    
    # ThetaData REST API
    base = "https://api.thetadata.net"
    headers = {"Accept": "application/json"}
    
    # Get historical end-of-day option quotes
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y%m%d")
    
    print(f"  Fetching {symbol} options EOD data: {start_date} → {end_date}")
    
    # List expirations
    r = requests.get(f"{base}/v2/list/expirations",
                     params={"root": symbol},
                     headers={**headers, "Authorization": f"Bearer {token}"})
    
    if r.status_code != 200:
        print(f"  Error: {r.status_code} — {r.text[:100]}")
        return pd.DataFrame()
    
    data = r.json()
    exps = data.get("response", [])
    print(f"  {len(exps)} expirations available")
    
    # For each recent expiration, get option chain snapshot
    all_data = []
    today_str = datetime.now().strftime("%Y%m%d")
    
    for exp in exps[-10:]:  # Last 10 expirations
        exp_str = str(exp)
        
        # Get EOD quotes for this expiration
        r = requests.get(f"{base}/v2/bulk_snapshot/option/quote",
                        params={
                            "root": symbol,
                            "exp": exp_str,
                            "use_csv": "true",
                        },
                        headers={**headers, "Authorization": f"Bearer {token}"})
        
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            if len(lines) > 1:
                print(f"  {exp_str}: {len(lines)-1} contracts")
                # Parse CSV response
                import io
                df = pd.read_csv(io.StringIO(r.text))
                df["expiration"] = exp_str
                all_data.append(df)
        
        time.sleep(0.5)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(DATA_DIR / f"{symbol}_thetadata_historical.csv", index=False)
        print(f"\n  ✅ Saved {len(combined)} records")
        return combined
    
    return pd.DataFrame()


# ==================== Pricing Comparison ====================

def compare_pricing(yf_chains: pd.DataFrame):
    """Compare real pricing vs our synthetic Black-Scholes model."""
    from data_fetcher import bs_price as synthetic_bs, estimate_intraday_iv
    
    if yf_chains.empty:
        return
    
    print("\n" + "=" * 70)
    print("📊 REAL vs SYNTHETIC PRICING COMPARISON")
    print("=" * 70)
    
    # Get underlying price
    underlying = yf_chains["underlying_price"].iloc[0]
    
    # Focus on 1DTE puts (our target)
    one_dte = yf_chains[(yf_chains["dte"] == 1) & (yf_chains["type"] == "put")]
    if one_dte.empty:
        # Try closest DTE
        min_dte = yf_chains["dte"].min()
        one_dte = yf_chains[(yf_chains["dte"] == min_dte) & (yf_chains["type"] == "put")]
        if one_dte.empty:
            print("  No 1DTE data available")
            return
        print(f"  Using DTE={min_dte} (closest available)")
    
    one_dte = one_dte[(one_dte["bid"] > 0) & (one_dte["strike"] < underlying)].copy()
    one_dte = one_dte.sort_values("strike", ascending=False)
    
    T = 1.0 / 252  # 1 trading day
    
    print(f"\n  Underlying: ${underlying:.2f}")
    print(f"\n  {'Strike':>8} {'Real Mid':>9} {'Synth':>8} {'Diff':>7} {'Diff%':>7} "
          f"{'Real IV':>8} {'Bid/Ask':>12} {'Spread%':>8}")
    print("  " + "-" * 80)
    
    diffs = []
    
    for _, row in one_dte.iterrows():
        K = row["strike"]
        real_bid = row["bid"]
        real_ask = row["ask"]
        real_mid = (real_bid + real_ask) / 2
        real_iv = row.get("impliedVolatility", 0.20)
        
        if real_mid <= 0 or real_iv <= 0:
            continue
        
        # Moneyness filter: only show 90-100% of spot
        if K / underlying < 0.90 or K / underlying > 1.0:
            continue
        
        # Synthetic with REAL IV
        synth_real_iv = synthetic_bs(underlying, K, T, 0.05, real_iv, "put")
        
        # Synthetic with our estimated IV (VIX-based)
        vix_est = 17.0  # approximate current VIX
        est_iv = estimate_intraday_iv(vix_est, 6.5)
        synth_est_iv = synthetic_bs(underlying, K, T, 0.05, est_iv, "put")
        
        diff = synth_real_iv - real_mid
        diff_pct = diff / real_mid * 100 if real_mid > 0 else 0
        spread_pct = (real_ask - real_bid) / real_mid * 100 if real_mid > 0 else 0
        
        diffs.append(diff_pct)
        
        print(f"  {K:>8.0f} {real_mid:>9.2f} {synth_real_iv:>8.2f} {diff:>+7.2f} "
              f"{diff_pct:>+6.1f}% {real_iv:>7.1%} {real_bid:>5.2f}/{real_ask:<5.2f} "
              f"{spread_pct:>7.1f}%")
    
    if diffs:
        print(f"\n  📊 Summary:")
        print(f"    Mean BS error: {np.mean(diffs):+.1f}%")
        print(f"    Median BS error: {np.median(diffs):+.1f}%")
        print(f"    Std: {np.std(diffs):.1f}%")
        print(f"    Our synthetic pricing {'overestimates' if np.mean(diffs) > 0 else 'underestimates'} "
              f"by ~{abs(np.mean(diffs)):.1f}% on average")
        
        avg_spread = one_dte["ask"].mean() - one_dte["bid"].mean()
        avg_mid = ((one_dte["bid"] + one_dte["ask"]) / 2).mean()
        print(f"    Average real bid/ask spread: ${avg_spread:.2f} ({avg_spread/avg_mid*100:.1f}%)")


# ==================== Main ====================

def main():
    print("=" * 70)
    print("🚀 REAL OPTIONS DATA — ALL SOURCES")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # Source 1: yfinance (always works)
    yf_data = fetch_yfinance_chains("SPY", max_expirations=10)
    
    # Source 2: Tradier (if token available)
    tradier_data = fetch_tradier_data("SPY")
    
    # Source 3: ThetaData (if token available)
    theta_data = fetch_thetadata("SPY")
    
    # Compare real vs synthetic
    if not yf_data.empty:
        compare_pricing(yf_data)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 DATA SOURCES SUMMARY")
    print("=" * 70)
    print(f"  yfinance:  {'✅' if not yf_data.empty else '❌'} "
          f"({len(yf_data)} contracts)" if not yf_data.empty else "  yfinance: ❌")
    print(f"  Tradier:   {'✅' if not tradier_data.empty else '⏳ needs token'} "
          f"({len(tradier_data)} contracts)" if not tradier_data.empty else "  Tradier:  ⏳ needs token")
    print(f"  ThetaData: {'✅' if not theta_data.empty else '⏳ needs token'} "
          f"({len(theta_data)} records)" if not theta_data.empty else "  ThetaData: ⏳ needs token")


if __name__ == "__main__":
    main()

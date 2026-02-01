"""
Fetch REAL historical options data from free APIs.

Sources (in order of preference):
1. Tradier Sandbox API — free, daily OHLC for options contracts, no CC required
2. CBOE delayed quotes — end-of-day options chain snapshots
3. yfinance — limited options chain (current only, no history)

For Tradier: Sign up at https://tradier.com → get sandbox API token
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path
from itertools import product

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ==================== Tradier Sandbox API ====================

TRADIER_SANDBOX_URL = "https://sandbox.tradier.com/v1"


def tradier_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def get_option_expirations(symbol: str, token: str) -> list:
    """Get available option expiration dates for a symbol."""
    r = requests.get(
        f"{TRADIER_SANDBOX_URL}/markets/options/expirations",
        params={"symbol": symbol, "includeAllRoots": "true"},
        headers=tradier_headers(token),
    )
    data = r.json()
    
    if "expirations" in data and data["expirations"]:
        dates = data["expirations"].get("date", [])
        if isinstance(dates, str):
            dates = [dates]
        return dates
    return []


def get_option_chain(symbol: str, expiration: str, token: str) -> pd.DataFrame:
    """
    Get full option chain for a symbol at a specific expiration.
    Returns DataFrame with strikes, bid, ask, last, greeks, etc.
    """
    r = requests.get(
        f"{TRADIER_SANDBOX_URL}/markets/options/chains",
        params={
            "symbol": symbol,
            "expiration": expiration,
            "greeks": "true",
        },
        headers=tradier_headers(token),
    )
    data = r.json()
    
    if "options" not in data or data["options"] is None:
        return pd.DataFrame()
    
    options = data["options"].get("option", [])
    if isinstance(options, dict):
        options = [options]
    
    if not options:
        return pd.DataFrame()
    
    rows = []
    for opt in options:
        greeks = opt.get("greeks", {}) or {}
        rows.append({
            "symbol": opt.get("symbol", ""),
            "underlying": opt.get("underlying", symbol),
            "type": opt.get("option_type", ""),
            "strike": opt.get("strike", 0),
            "expiration": opt.get("expiration_date", expiration),
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
        })
    
    return pd.DataFrame(rows)


def get_option_history(option_symbol: str, token: str,
                       start: str = None, end: str = None) -> pd.DataFrame:
    """
    Get historical daily OHLCV for a specific option contract.
    
    option_symbol: OCC format, e.g., "SPY260130C00690000"
    """
    params = {"symbol": option_symbol, "interval": "daily"}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    
    r = requests.get(
        f"{TRADIER_SANDBOX_URL}/markets/history",
        params=params,
        headers=tradier_headers(token),
    )
    data = r.json()
    
    if "history" not in data or data["history"] is None:
        return pd.DataFrame()
    
    days = data["history"].get("day", [])
    if isinstance(days, dict):
        days = [days]
    
    if not days:
        return pd.DataFrame()
    
    df = pd.DataFrame(days)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    
    return df


def fetch_spy_option_chains(token: str, num_expirations: int = 5):
    """
    Fetch option chains for the next N SPY expirations.
    Saves each chain as a CSV file.
    """
    print("[Tradier] Fetching SPY option expirations...")
    expirations = get_option_expirations("SPY", token)
    
    if not expirations:
        print("  No expirations found!")
        return
    
    print(f"  Found {len(expirations)} expirations")
    print(f"  Next 5: {expirations[:5]}")
    
    all_chains = []
    for exp in expirations[:num_expirations]:
        print(f"\n  Fetching chain for {exp}...")
        chain = get_option_chain("SPY", exp, token)
        
        if chain.empty:
            print(f"    Empty chain")
            continue
        
        print(f"    {len(chain)} contracts | "
              f"Strikes: {chain['strike'].min():.0f}-{chain['strike'].max():.0f} | "
              f"Puts: {len(chain[chain['type']=='put'])}, Calls: {len(chain[chain['type']=='call'])}")
        
        # Show sample with real bid/ask
        puts = chain[(chain["type"] == "put") & (chain["bid"] > 0)].sort_values("strike", ascending=False)
        if not puts.empty:
            sample = puts.head(5)
            print(f"    Sample puts (bid/ask/delta/IV):")
            for _, row in sample.iterrows():
                print(f"      K={row['strike']:.0f} bid={row['bid']:.2f} ask={row['ask']:.2f} "
                      f"δ={row['delta']:.3f} IV={row['iv']:.1%} OI={row['open_interest']}")
        
        all_chains.append(chain)
        time.sleep(0.5)  # Rate limit
    
    if all_chains:
        combined = pd.concat(all_chains, ignore_index=True)
        fname = DATA_DIR / f"SPY_options_chains_real.csv"
        combined.to_csv(fname, index=False)
        print(f"\n  Saved {len(combined)} contracts to {fname}")
        return combined
    
    return pd.DataFrame()


def fetch_historical_option_prices(token: str, symbol: str = "SPY",
                                    strikes_around_atm: int = 10,
                                    days_back: int = 60):
    """
    Fetch historical daily prices for options near ATM.
    
    This gets actual traded prices (OHLCV) for individual contracts
    over time — real market data, not synthetic.
    """
    print(f"\n[Tradier] Fetching historical option prices for {symbol}...")
    
    # Get current price
    r = requests.get(
        f"{TRADIER_SANDBOX_URL}/markets/quotes",
        params={"symbols": symbol},
        headers=tradier_headers(token),
    )
    quote_data = r.json()
    
    current_price = None
    if "quotes" in quote_data:
        q = quote_data["quotes"].get("quote", {})
        current_price = q.get("last", q.get("close", None))
    
    if current_price is None:
        print("  Could not get current price")
        return
    
    print(f"  {symbol} current price: ${current_price:.2f}")
    
    # Get nearest expiration(s)
    expirations = get_option_expirations(symbol, token)
    if not expirations:
        print("  No expirations")
        return
    
    # Use first 3 expirations
    target_exps = expirations[:3]
    
    all_history = []
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    for exp in target_exps:
        print(f"\n  Expiration: {exp}")
        chain = get_option_chain(symbol, exp, token)
        
        if chain.empty:
            continue
        
        # Pick strikes around ATM
        strikes = sorted(chain["strike"].unique())
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - current_price))
        selected = strikes[max(0, atm_idx - strikes_around_atm):
                          min(len(strikes), atm_idx + strikes_around_atm + 1)]
        
        print(f"    Selected {len(selected)} strikes: {selected[0]:.0f} - {selected[-1]:.0f}")
        
        for strike in selected:
            for opt_type in ["call", "put"]:
                contracts = chain[(chain["strike"] == strike) & (chain["type"] == opt_type)]
                if contracts.empty:
                    continue
                
                opt_symbol = contracts.iloc[0]["symbol"]
                
                hist = get_option_history(opt_symbol, token, start=start_date)
                
                if not hist.empty:
                    hist["option_symbol"] = opt_symbol
                    hist["strike"] = strike
                    hist["type"] = opt_type
                    hist["expiration"] = exp
                    all_history.append(hist)
                
                time.sleep(0.3)  # Rate limit
        
        print(f"    Fetched history for {len(all_history)} contracts so far")
    
    if all_history:
        combined = pd.concat(all_history)
        fname = DATA_DIR / f"{symbol}_options_history_real.csv"
        combined.to_csv(fname)
        print(f"\n  Saved {len(combined)} data points to {fname}")
        return combined
    
    return pd.DataFrame()


# ==================== Compare Real vs Synthetic ====================

def compare_real_vs_synthetic(chain_df: pd.DataFrame, underlying_price: float):
    """
    Compare real market bid/ask with our synthetic BS prices.
    Shows how much our synthetic pricing over/underestimates.
    """
    from data_fetcher import bs_price, bs_delta
    
    print("\n" + "=" * 80)
    print("📊 REAL vs SYNTHETIC OPTIONS PRICING COMPARISON")
    print("=" * 80)
    
    if chain_df.empty:
        print("No data to compare")
        return
    
    # Filter to puts with bid > 0 (the ones we'd trade for iron condors)
    puts = chain_df[(chain_df["type"] == "put") & (chain_df["bid"] > 0)].copy()
    calls = chain_df[(chain_df["type"] == "call") & (chain_df["bid"] > 0)].copy()
    
    for label, opts in [("PUTS", puts), ("CALLS", calls)]:
        if opts.empty:
            continue
        
        print(f"\n{label} (S=${underlying_price:.2f}):")
        print(f"{'Strike':>8} {'Real Bid':>9} {'Real Ask':>9} {'Real Mid':>9} "
              f"{'Synth':>8} {'Diff%':>7} {'Real IV':>8} {'Real δ':>7}")
        print("-" * 75)
        
        is_call = label == "CALLS"
        
        for _, row in opts.iterrows():
            K = row["strike"]
            real_bid = row["bid"]
            real_ask = row["ask"]
            real_mid = (real_bid + real_ask) / 2
            real_iv = row["iv"]
            real_delta = row["delta"]
            
            # Our synthetic price (using real IV if available, else estimate)
            T = 1.0 / 252  # ~1DTE
            sigma = real_iv if real_iv > 0 else 0.20
            
            synth = bs_price(underlying_price, K, T, 0.05, sigma, "call" if is_call else "put")
            
            diff_pct = ((synth - real_mid) / real_mid * 100) if real_mid > 0 else 0
            
            if abs(real_delta) < 0.01 or abs(real_delta) > 0.50:
                continue  # Skip deep ITM/OTM
            
            print(f"{K:>8.0f} {real_bid:>9.2f} {real_ask:>9.2f} {real_mid:>9.2f} "
                  f"{synth:>8.2f} {diff_pct:>+6.1f}% {real_iv:>7.1%} {real_delta:>+7.3f}")


# ==================== Main ====================

def main():
    print("=" * 70)
    print("📊 REAL OPTIONS DATA FETCHER")
    print("=" * 70)
    
    token = os.environ.get("TRADIER_TOKEN", "")
    
    if not token:
        # Check for token file
        token_file = Path(__file__).parent.parent / ".tradier_token"
        if token_file.exists():
            token = token_file.read_text().strip()
    
    if not token:
        print("""
⚠️  No Tradier API token found!

To get free real options data:
1. Sign up at https://tradier.com (free, no CC required)
2. Go to Settings → API Access → Sandbox Token
3. Either:
   a) Set environment variable: export TRADIER_TOKEN=your_token
   b) Save to file: echo "your_token" > .tradier_token

Then re-run this script.
""")
        
        # Fall back to yfinance current chain (limited but free, no auth)
        print("Falling back to yfinance for current options chain...")
        import yfinance as yf
        
        spy = yf.Ticker("SPY")
        exps = spy.options
        print(f"Available expirations: {exps[:5]}...")
        
        if exps:
            # Get first expiration chain
            chain = spy.option_chain(exps[0])
            calls = chain.calls
            puts = chain.puts
            
            print(f"\nExpiration: {exps[0]}")
            print(f"  Calls: {len(calls)}, Puts: {len(puts)}")
            
            # Save
            calls["type"] = "call"
            puts["type"] = "put"
            combined = pd.concat([calls, puts], ignore_index=True)
            combined.to_csv(DATA_DIR / "SPY_current_chain_yfinance.csv", index=False)
            
            # Show sample
            print(f"\n  Sample puts (near ATM):")
            last_price = spy.history(period="1d")["Close"].iloc[-1]
            near_atm = puts[abs(puts["strike"] - last_price) < 20].sort_values("strike", ascending=False)
            
            for _, row in near_atm.head(10).iterrows():
                print(f"    K={row['strike']:.0f} bid={row['bid']:.2f} ask={row['ask']:.2f} "
                      f"vol={row['volume']} OI={row['openInterest']} IV={row.get('impliedVolatility', 0):.1%}")
            
            print(f"\n  Saved to {DATA_DIR}/SPY_current_chain_yfinance.csv")
            
            # Compare real vs synthetic
            compare_real_vs_synthetic(
                pd.DataFrame({
                    "type": combined["type"],
                    "strike": combined["strike"],
                    "bid": combined["bid"],
                    "ask": combined["ask"],
                    "iv": combined.get("impliedVolatility", 0),
                    "delta": 0,  # yfinance doesn't give greeks
                }),
                last_price
            )
        
        return
    
    # With Tradier token
    print(f"\n✅ Tradier token found")
    
    # 1. Fetch current chains with real greeks
    chains = fetch_spy_option_chains(token, num_expirations=3)
    
    # 2. Fetch historical prices for near-ATM options
    history = fetch_historical_option_prices(token, "SPY", 
                                              strikes_around_atm=10, days_back=30)
    
    # 3. Compare real vs synthetic
    if chains is not None and not chains.empty:
        # Get underlying price
        r = requests.get(
            f"{TRADIER_SANDBOX_URL}/markets/quotes",
            params={"symbols": "SPY"},
            headers=tradier_headers(token),
        )
        price = r.json()["quotes"]["quote"]["last"]
        
        # Use nearest expiration chain
        nearest_exp = sorted(chains["expiration"].unique())[0]
        nearest_chain = chains[chains["expiration"] == nearest_exp]
        
        compare_real_vs_synthetic(nearest_chain, price)


if __name__ == "__main__":
    main()

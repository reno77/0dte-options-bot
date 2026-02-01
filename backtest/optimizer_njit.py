"""
Numba @njit Accelerated 1DTE Options Backtest Optimizer

Compiles all Black-Scholes math and backtest loops to native machine code.
Grid searches over strategy parameters using multiprocessing (16 cores).

Same approach that hit 499 backtests/sec on the crypto bot.
"""

import time
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
from numba import njit, prange
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from data_fetcher import fetch_underlying_intraday, fetch_vix, DATA_DIR

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ==================== Numba-compiled Black-Scholes ====================

@njit(cache=True)
def norm_cdf(x):
    """Fast approximation of standard normal CDF (Abramowitz & Stegun)."""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = 1.0
    if x < 0:
        sign = -1.0
    x = abs(x) / np.sqrt(2.0)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return 0.5 * (1.0 + sign * y)


@njit(cache=True)
def norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@njit(cache=True)
def bs_price(S, K, T, r, sigma, is_call):
    """Black-Scholes option price. is_call=True for call, False for put."""
    if T <= 1e-10:
        if is_call:
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    if is_call:
        return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    else:
        return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


@njit(cache=True)
def bs_delta(S, K, T, r, sigma, is_call):
    """Black-Scholes delta."""
    if T <= 1e-10:
        if is_call:
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    
    if is_call:
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1.0


@njit(cache=True)
def estimate_iv(vix_close, hours_remaining):
    """Estimate intraday IV from VIX."""
    base_iv = vix_close / 100.0
    multiplier = 1.3 + (0.2 * (6.5 - min(hours_remaining, 6.5)) / 6.5)
    return base_iv * multiplier


# ==================== Core Backtest Engine (njit) ====================

@njit(cache=True)
def find_strike_by_delta(S, T, r, sigma, target_delta, strike_interval, is_call, search_range_pct=0.05):
    """
    Find the strike price closest to target delta.
    Returns (strike, premium, delta) or (-1, 0, 0) if not found.
    """
    if is_call:
        # Search strikes above S
        start = S + strike_interval
        end = S * (1.0 + search_range_pct)
    else:
        # Search strikes below S (start just below S, go down)
        start = S - strike_interval
        end = S * (1.0 - search_range_pct)
    
    best_strike = -1.0
    best_diff = 999.0
    best_premium = 0.0
    best_delta = 0.0
    
    strike = start
    while (strike <= end) if is_call else (strike >= end):
        d = bs_delta(S, strike, T, r, sigma, is_call)
        diff = abs(abs(d) - target_delta)
        
        if diff < best_diff:
            best_diff = diff
            best_strike = strike
            best_premium = bs_price(S, strike, T, r, sigma, is_call)
            best_delta = d
        
        if is_call:
            strike += strike_interval
        else:
            strike -= strike_interval
    
    return best_strike, best_premium, best_delta


@njit(cache=True)
def run_single_ic_backtest(
    # Price data arrays
    prices,           # close prices, shape (N,)
    highs,            # high prices
    lows,             # low prices
    day_indices,      # int array: day index for each bar (0, 0, 0, ..., 1, 1, 1, ...)
    bar_in_day,       # int array: bar index within day (0, 1, 2, ...)
    bars_per_day,     # int array: total bars per day
    vix_per_day,      # float array: VIX close for each day
    
    # Strategy parameters
    delta_min,        # min delta for short strikes (decimal, e.g., 0.08)
    delta_max,        # max delta for short strikes
    spread_width,     # spread width in dollars
    stop_loss_mult,   # stop loss multiplier on premium
    entry_bar_start,  # bar index to start entries (e.g., 60 for 10:30)
    entry_bar_end,    # bar index to stop entries
    entry_interval,   # min bars between entries
    max_entries,      # max entries per day
    strike_interval,  # strike price interval
    
    # Constants
    r = 0.05,
    total_bars_per_day = 390.0,  # 6.5h * 60min
):
    """
    Run a single 1DTE iron condor backtest. All logic compiled to native code.
    
    Returns: (total_pnl, num_trades, num_wins, num_losses, max_dd, 
              avg_premium, total_return_pct, win_rate)
    """
    N = len(prices)
    num_days = int(day_indices[-1]) + 1
    
    equity = 100000.0
    peak_equity = equity
    max_dd = 0.0
    
    total_pnl = 0.0
    num_trades = 0
    num_wins = 0
    num_losses = 0
    total_premium = 0.0
    
    # Position tracking (max 10 open positions at once)
    MAX_POS = 10
    pos_active = np.zeros(MAX_POS, dtype=np.int32)  # 0=empty, 1=active
    pos_short_call = np.zeros(MAX_POS)
    pos_long_call = np.zeros(MAX_POS)
    pos_short_put = np.zeros(MAX_POS)
    pos_long_put = np.zeros(MAX_POS)
    pos_premium = np.zeros(MAX_POS)
    pos_stop = np.zeros(MAX_POS)
    pos_expiry_day = np.zeros(MAX_POS, dtype=np.int32)
    
    target_delta = (delta_min + delta_max) / 2.0
    
    for i in range(N):
        day_idx = int(day_indices[i])
        bar_idx = int(bar_in_day[i])
        n_bars = int(bars_per_day[day_idx]) if day_idx < len(bars_per_day) else 390
        
        S = prices[i]
        mins_remaining = max(float(n_bars - bar_idx), 0.0)
        hours_remaining = mins_remaining / 60.0
        
        vix = vix_per_day[day_idx] if day_idx < len(vix_per_day) else 20.0
        sigma = estimate_iv(vix, hours_remaining)
        
        # ---- Check expiring positions ----
        for p in range(MAX_POS):
            if pos_active[p] == 0:
                continue
            if pos_expiry_day[p] != day_idx:
                continue
            
            T_rem = mins_remaining / (252.0 * 6.5 * 60.0)
            
            call_val = (bs_price(S, pos_short_call[p], T_rem, r, sigma, True) -
                       bs_price(S, pos_long_call[p], T_rem, r, sigma, True))
            put_val = (bs_price(S, pos_short_put[p], T_rem, r, sigma, False) -
                      bs_price(S, pos_long_put[p], T_rem, r, sigma, False))
            
            total_val = call_val + put_val
            prem = pos_premium[p]
            
            # Stop loss
            if total_val > prem + pos_stop[p]:
                pnl = -pos_stop[p] * 100.0  # ×100 multiplier
                equity += pnl
                total_pnl += pnl
                num_trades += 1
                num_losses += 1
                pos_active[p] = 0
            elif mins_remaining <= 1.0:
                # Expiration
                pnl = (prem - max(total_val, 0.0)) * 100.0
                equity += pnl
                total_pnl += pnl
                num_trades += 1
                if pnl > 0:
                    num_wins += 1
                else:
                    num_losses += 1
                pos_active[p] = 0
            
            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd = (equity - peak_equity) / peak_equity
            if dd < max_dd:
                max_dd = dd
        
        # ---- Entry logic ----
        if bar_idx < entry_bar_start or bar_idx > entry_bar_end:
            continue
        
        # Count active positions entered today (expiring tomorrow)
        entries_today = 0
        for p in range(MAX_POS):
            if pos_active[p] == 1 and pos_expiry_day[p] == day_idx + 1:
                entries_today += 1
        
        if entries_today >= max_entries:
            continue
        
        # Check entry interval (use entry_bar tracking per day)
        # Simple approach: only enter at specific bar intervals
        if (bar_idx - entry_bar_start) % entry_interval != 0:
            continue
        
        # Find empty slot
        slot = -1
        for p in range(MAX_POS):
            if pos_active[p] == 0:
                slot = p
                break
        if slot == -1:
            continue
        
        # Next trading day
        next_day = day_idx + 1
        if next_day >= num_days:
            continue
        
        # T for 1DTE
        T_1dte = (mins_remaining + 6.5 * 60.0) / (252.0 * 6.5 * 60.0)
        sigma_entry = estimate_iv(vix, hours_remaining + 6.5)
        
        # Find short call
        sc_strike, sc_prem, sc_delta = find_strike_by_delta(
            S, T_1dte, r, sigma_entry, target_delta, strike_interval, True)
        
        if sc_strike < 0 or abs(sc_delta) < delta_min or abs(sc_delta) > delta_max:
            continue
        
        # Find short put
        sp_strike, sp_prem, sp_delta = find_strike_by_delta(
            S, T_1dte, r, sigma_entry, target_delta, strike_interval, False)
        
        if sp_strike < 0 or abs(sp_delta) < delta_min or abs(sp_delta) > delta_max:
            continue
        
        # Long legs
        lc_strike = sc_strike + spread_width
        lp_strike = sp_strike - spread_width
        
        lc_prem = bs_price(S, lc_strike, T_1dte, r, sigma_entry, True)
        lp_prem = bs_price(S, lp_strike, T_1dte, r, sigma_entry, False)
        
        call_credit = sc_prem - lc_prem
        put_credit = sp_prem - lp_prem
        
        if call_credit <= 0.05 or put_credit <= 0.05:
            continue
        
        ic_premium = call_credit + put_credit
        
        # Open position
        pos_active[slot] = 1
        pos_short_call[slot] = sc_strike
        pos_long_call[slot] = lc_strike
        pos_short_put[slot] = sp_strike
        pos_long_put[slot] = lp_strike
        pos_premium[slot] = ic_premium
        pos_stop[slot] = ic_premium * stop_loss_mult
        pos_expiry_day[slot] = next_day
        total_premium += ic_premium
    
    # Force-close remaining
    for p in range(MAX_POS):
        if pos_active[p] == 1:
            pnl = pos_premium[p] * 0.5 * 100.0
            equity += pnl
            total_pnl += pnl
            num_trades += 1
            num_wins += 1
            pos_active[p] = 0
    
    if equity > peak_equity:
        peak_equity = equity
    dd = (equity - peak_equity) / peak_equity
    if dd < max_dd:
        max_dd = dd
    
    total_return_pct = (equity - 100000.0) / 100000.0 * 100.0
    win_rate = (num_wins / num_trades * 100.0) if num_trades > 0 else 0.0
    avg_prem = (total_premium / num_trades) if num_trades > 0 else 0.0
    
    return (total_pnl, num_trades, num_wins, num_losses, max_dd * 100.0,
            avg_prem, total_return_pct, win_rate, equity)


# ==================== Data Preparation ====================

def prepare_data(intraday_df, vix_df):
    """Convert DataFrames to numpy arrays for njit functions."""
    intraday_df = intraday_df.copy()
    intraday_df.index = pd.to_datetime(intraday_df.index, utc=True)
    intraday_df["date"] = intraday_df.index.date
    
    dates = sorted(intraday_df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    prices = intraday_df["close"].values.astype(np.float64)
    highs = intraday_df["high"].values.astype(np.float64)
    lows = intraday_df["low"].values.astype(np.float64)
    
    day_indices = np.array([date_to_idx[d] for d in intraday_df["date"]], dtype=np.int32)
    
    # Bar index within each day
    bar_in_day = np.zeros(len(intraday_df), dtype=np.int32)
    bars_per_day_list = []
    for d in dates:
        mask = intraday_df["date"] == d
        n = mask.sum()
        bar_in_day[mask.values] = np.arange(n, dtype=np.int32)
        bars_per_day_list.append(n)
    
    bars_per_day = np.array(bars_per_day_list, dtype=np.int32)
    
    # VIX per trading day
    vix_per_day = np.full(len(dates), 20.0, dtype=np.float64)
    if vix_df is not None and not vix_df.empty:
        vix_df.index = pd.to_datetime(vix_df.index, utc=True)
        for i, d in enumerate(dates):
            prev = vix_df[vix_df.index.date <= d]
            if not prev.empty:
                vix_per_day[i] = prev["close"].iloc[-1]
    
    return prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day, dates


# ==================== Grid Search ====================

def run_single_config(args):
    """Worker function for multiprocessing."""
    (prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
     delta_min, delta_max, spread_width, stop_loss_mult,
     entry_bar_start, entry_bar_end, entry_interval, max_entries, strike_interval) = args
    
    result = run_single_ic_backtest(
        prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
        delta_min, delta_max, spread_width, stop_loss_mult,
        entry_bar_start, entry_bar_end, entry_interval, max_entries, strike_interval
    )
    
    return {
        "delta_min": delta_min,
        "delta_max": delta_max,
        "spread_width": spread_width,
        "stop_loss_mult": stop_loss_mult,
        "entry_bar_start": entry_bar_start,
        "entry_bar_end": entry_bar_end,
        "entry_interval": entry_interval,
        "max_entries": max_entries,
        "strike_interval": strike_interval,
        "total_pnl": round(result[0], 2),
        "num_trades": int(result[1]),
        "num_wins": int(result[2]),
        "num_losses": int(result[3]),
        "max_dd_pct": round(result[4], 2),
        "avg_premium": round(result[5], 2),
        "total_return_pct": round(result[6], 2),
        "win_rate": round(result[7], 1),
        "final_equity": round(result[8], 2),
    }


def generate_param_grid():
    """Generate all parameter combinations to test."""
    grid = {
        "delta_min":       [0.05, 0.08, 0.10, 0.12],
        "delta_max":       [0.15, 0.20, 0.25],
        "spread_width":    [5.0, 10.0, 15.0],
        "stop_loss_mult":  [1.0, 1.5, 2.0, 2.5],
        "entry_bar_start": [30, 60, 90],          # 10:00, 10:30, 11:00
        "entry_bar_end":   [240, 300, 330],        # 13:30, 14:30, 15:00
        "entry_interval":  [30, 45, 60],           # minutes between entries
        "max_entries":     [2, 3, 4, 5],
        "strike_interval": [1.0],                  # SPY = $1 strikes
    }
    
    from itertools import product
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    
    return keys, combos


def main():
    print("=" * 70)
    print("🚀 NUMBA @njit 1DTE OPTIONS OPTIMIZER — Alienware RTX 3090")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   CPU cores: {cpu_count()}")
    print("=" * 70)
    
    # Load or fetch data
    print("\n📥 Loading data...")
    intraday_file = DATA_DIR / "SPY_intraday_1m_1mo.csv"
    vix_file = DATA_DIR / "VIX_3y.csv"
    
    if intraday_file.exists():
        print("  Using cached SPY intraday data")
        intraday = pd.read_csv(intraday_file, index_col=0, parse_dates=True)
    else:
        intraday = fetch_underlying_intraday("SPY", period="1mo", interval="1m")
    
    if vix_file.exists():
        print("  Using cached VIX data")
        vix = pd.read_csv(vix_file, index_col=0, parse_dates=True)
    else:
        vix = fetch_vix("3y")
    
    print(f"  SPY: {len(intraday)} bars, {intraday.index[0]} → {intraday.index[-1]}")
    
    # Prepare numpy arrays
    print("\n⚙️  Preparing data arrays...")
    prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day, dates = \
        prepare_data(intraday, vix)
    
    print(f"  {len(dates)} trading days, {len(prices)} total bars")
    
    # Warm up JIT
    print("\n🔥 Warming up Numba JIT compiler...")
    t0 = time.time()
    _ = run_single_ic_backtest(
        prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
        0.08, 0.20, 10.0, 1.5, 60, 300, 45, 3, 1.0
    )
    jit_time = time.time() - t0
    print(f"  JIT compile + first run: {jit_time:.2f}s")
    
    # Speed test
    t0 = time.time()
    for _ in range(100):
        _ = run_single_ic_backtest(
            prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
            0.08, 0.20, 10.0, 1.5, 60, 300, 45, 3, 1.0
        )
    speed = 100 / (time.time() - t0)
    print(f"  Speed: {speed:.0f} backtests/sec (single core)")
    
    # Generate grid
    keys, combos = generate_param_grid()
    total = len(combos)
    print(f"\n📊 Grid search: {total} parameter combinations")
    print(f"   Estimated time: {total / speed / cpu_count():.1f}s ({cpu_count()} cores)")
    
    # Build arguments
    args_list = []
    for combo in combos:
        params = dict(zip(keys, combo))
        # Skip invalid combos
        if params["delta_min"] >= params["delta_max"]:
            continue
        if params["entry_bar_start"] >= params["entry_bar_end"]:
            continue
        
        args_list.append((
            prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
            params["delta_min"], params["delta_max"], params["spread_width"],
            params["stop_loss_mult"], params["entry_bar_start"], params["entry_bar_end"],
            params["entry_interval"], params["max_entries"], params["strike_interval"]
        ))
    
    valid_combos = len(args_list)
    print(f"   Valid combos (after filtering): {valid_combos}")
    
    # Run grid search
    print(f"\n🏃 Running {valid_combos} backtests on {cpu_count()} cores...")
    t0 = time.time()
    
    with Pool(cpu_count()) as pool:
        results = pool.map(run_single_config, args_list)
    
    elapsed = time.time() - t0
    actual_speed = valid_combos / elapsed
    print(f"\n✅ Done! {valid_combos} configs in {elapsed:.1f}s ({actual_speed:.0f}/sec)")
    
    # Filter and sort results
    results = [r for r in results if r["num_trades"] >= 1]  # Min trades
    results.sort(key=lambda x: x["total_return_pct"], reverse=True)
    
    # Print top 20
    print("\n" + "=" * 100)
    print("🏆 TOP 20 CONFIGS (by total return)")
    print("=" * 100)
    print(f"{'#':>3} {'Return%':>8} {'WR%':>6} {'Trades':>7} {'MaxDD%':>7} {'AvgPrem':>8} {'Final$':>12} {'Δmin':>5} {'Δmax':>5} {'Width':>6} {'SL×':>4} {'Entries':>8} {'Int':>4}")
    print("-" * 100)
    
    for i, r in enumerate(results[:20]):
        print(f"{i+1:>3} {r['total_return_pct']:>8.2f} {r['win_rate']:>6.1f} {r['num_trades']:>7} "
              f"{r['max_dd_pct']:>7.2f} {r['avg_premium']:>8.2f} {r['final_equity']:>12.2f} "
              f"{r['delta_min']:>5.2f} {r['delta_max']:>5.2f} {r['spread_width']:>6.1f} "
              f"{r['stop_loss_mult']:>4.1f} {r['max_entries']:>8} {r['entry_interval']:>4}")
    
    # Count configs meeting targets
    targets = [
        ("Return > 1%", lambda r: r["total_return_pct"] > 1),
        ("Return > 2%", lambda r: r["total_return_pct"] > 2),
        ("Return > 5%", lambda r: r["total_return_pct"] > 5),
        ("WR > 80%", lambda r: r["win_rate"] > 80),
        ("WR > 90%", lambda r: r["win_rate"] > 90),
        ("DD < 1%", lambda r: abs(r["max_dd_pct"]) < 1),
        ("DD < 2%", lambda r: abs(r["max_dd_pct"]) < 2),
    ]
    
    print(f"\n📊 Target hits out of {len(results)} valid configs:")
    for name, fn in targets:
        hits = sum(1 for r in results if fn(r))
        pct = (hits/len(results)*100) if len(results) > 0 else 0
        print(f"  {name}: {hits} ({pct:.1f}%)")
    
    # Best config details
    if results:
        best = results[0]
        print(f"\n🏆 BEST CONFIG:")
        print(json.dumps(best, indent=2))
    
    # Save all results
    with open(RESULTS_DIR / "1dte_optimizer_results.json", "w") as f:
        json.dump({
            "meta": {
                "total_combos": total,
                "valid_combos": valid_combos,
                "elapsed_sec": round(elapsed, 1),
                "speed_per_sec": round(actual_speed, 0),
                "trading_days": len(dates),
                "date_range": f"{dates[0]} → {dates[-1]}",
            },
            "top_20": results[:20],
            "all_results": results,
        }, f, indent=2, default=str)
    
    # Plot top config equity curve (re-run with daily tracking)
    if results:
        best = results[0]
        print(f"\n📈 Generating equity curve for best config...")
        
        # Re-run best config to get daily P&L
        res = run_single_ic_backtest(
            prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
            best["delta_min"], best["delta_max"], best["spread_width"],
            best["stop_loss_mult"], best["entry_bar_start"], best["entry_bar_end"],
            best["entry_interval"], best["max_entries"], best["strike_interval"]
        )
        
        # Distribution of returns
        returns = [r["total_return_pct"] for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Return distribution
        axes[0, 0].hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title(f"Return Distribution ({len(results)} configs)")
        axes[0, 0].set_xlabel("Total Return %")
        axes[0, 0].set_ylabel("Count")
        
        # Win rate vs Return
        wr = [r["win_rate"] for r in results]
        axes[0, 1].scatter(wr, returns, alpha=0.3, s=10, c='steelblue')
        axes[0, 1].set_title("Win Rate vs Return")
        axes[0, 1].set_xlabel("Win Rate %")
        axes[0, 1].set_ylabel("Total Return %")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max DD vs Return
        dds = [abs(r["max_dd_pct"]) for r in results]
        axes[1, 0].scatter(dds, returns, alpha=0.3, s=10, c='steelblue')
        axes[1, 0].set_title("Max Drawdown vs Return")
        axes[1, 0].set_xlabel("Max Drawdown %")
        axes[1, 0].set_ylabel("Total Return %")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Spread width vs Return (grouped)
        for sw in [5.0, 10.0, 15.0]:
            sw_returns = [r["total_return_pct"] for r in results if r["spread_width"] == sw]
            if sw_returns:
                axes[1, 1].hist(sw_returns, bins=30, alpha=0.5, label=f"${sw:.0f} wide")
        axes[1, 1].set_title("Return by Spread Width")
        axes[1, 1].set_xlabel("Total Return %")
        axes[1, 1].legend()
        
        plt.suptitle(f"1DTE Iron Condor Optimization — {valid_combos} configs, {len(dates)} days", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "1dte_optimizer_charts.png", dpi=150)
        plt.close()
        
        print(f"  Charts saved to {RESULTS_DIR}/1dte_optimizer_charts.png")
    
    print(f"\n✅ All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

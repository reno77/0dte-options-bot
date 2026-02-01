"""
Multi-Year 1DTE Options Optimizer using 1h bars (730 days from yfinance)
+ 15m bars for recent 60 days.

Uses Numba @njit for speed. Searches parameter grid on Alienware 16 cores.
"""

import time
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
from numba import njit, prange
from multiprocessing import Pool, cpu_count
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from data_fetcher import DATA_DIR

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ==================== Data Fetching ====================

def fetch_spy_hourly():
    """Fetch SPY 1h bars — up to 730 days from yfinance."""
    import yfinance as yf
    print("[Data] Fetching SPY 1h (730d)...")
    df = yf.Ticker("SPY").history(period="730d", interval="1h")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    df.to_csv(DATA_DIR / "SPY_1h_730d.csv")
    print(f"  {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")
    return df


def fetch_spy_15m():
    """Fetch SPY 15m bars — up to 60 days from yfinance."""
    import yfinance as yf
    print("[Data] Fetching SPY 15m (60d)...")
    df = yf.Ticker("SPY").history(period="60d", interval="15m")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    df.to_csv(DATA_DIR / "SPY_15m_60d.csv")
    print(f"  {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")
    return df


def fetch_vix_data():
    """Fetch VIX daily."""
    import yfinance as yf
    print("[Data] Fetching VIX (5y)...")
    df = yf.Ticker("^VIX").history(period="5y")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close"]]
    df.columns = ["open", "high", "low", "close"]
    df.to_csv(DATA_DIR / "VIX_5y.csv")
    print(f"  {len(df)} bars")
    return df


# ==================== Numba Compiled Core ====================

@njit(cache=True)
def norm_cdf(x):
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


@njit(cache=True)
def bs_price(S, K, T, r, sigma, is_call):
    if T <= 1e-10:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if is_call:
        return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    else:
        return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


@njit(cache=True)
def bs_delta(S, K, T, r, sigma, is_call):
    if T <= 1e-10:
        if is_call: return 1.0 if S > K else 0.0
        else: return -1.0 if S < K else 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return norm_cdf(d1) if is_call else norm_cdf(d1) - 1.0


@njit(cache=True)
def estimate_iv(vix, hours_rem):
    base = vix / 100.0
    mult = 1.3 + (0.2 * (6.5 - min(hours_rem, 6.5)) / 6.5)
    return base * mult


@njit(cache=True)
def find_strike(S, T, r, sigma, target_delta, interval, is_call):
    """Find strike closest to target delta. Returns (strike, premium, delta)."""
    if is_call:
        start = S + interval
        end = S * 1.08
    else:
        start = S - interval
        end = S * 0.92
    
    best_k, best_diff, best_p, best_d = -1.0, 999.0, 0.0, 0.0
    k = start
    
    if is_call:
        while k <= end:
            d = bs_delta(S, k, T, r, sigma, True)
            diff = abs(abs(d) - target_delta)
            if diff < best_diff:
                best_diff, best_k = diff, k
                best_p = bs_price(S, k, T, r, sigma, True)
                best_d = d
            k += interval
    else:
        while k >= end:
            d = bs_delta(S, k, T, r, sigma, False)
            diff = abs(abs(d) - target_delta)
            if diff < best_diff:
                best_diff, best_k = diff, k
                best_p = bs_price(S, k, T, r, sigma, False)
                best_d = d
            k -= interval
    
    return best_k, best_p, best_d


@njit(cache=True)
def run_backtest(
    prices, day_idx_arr, bar_in_day_arr, bars_per_day_arr, vix_arr,
    delta_min, delta_max, spread_width, sl_mult,
    entry_bar_start, entry_bar_end, entry_interval, max_entries,
    strike_interval, bar_minutes,
):
    """
    Core 1DTE IC backtest — works with any bar size (15m, 1h).
    
    bar_minutes: 15 for 15m bars, 60 for 1h bars
    """
    N = len(prices)
    num_days = int(day_idx_arr[-1]) + 1
    
    equity = 100000.0
    peak = equity
    max_dd = 0.0
    total_pnl = 0.0
    n_trades = 0
    n_wins = 0
    n_losses = 0
    sum_premium = 0.0
    
    # Daily equity for Sharpe
    daily_equity = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    
    MAX_POS = 15
    p_active = np.zeros(MAX_POS, dtype=np.int32)
    p_sc = np.zeros(MAX_POS)  # short call
    p_lc = np.zeros(MAX_POS)  # long call
    p_sp = np.zeros(MAX_POS)  # short put
    p_lp = np.zeros(MAX_POS)  # long put
    p_prem = np.zeros(MAX_POS)
    p_stop = np.zeros(MAX_POS)
    p_expiry = np.zeros(MAX_POS, dtype=np.int32)
    
    target_delta = (delta_min + delta_max) / 2.0
    mins_per_bar = bar_minutes
    total_mins_day = 390.0  # 6.5h
    
    prev_day = -1
    
    for i in range(N):
        d = int(day_idx_arr[i])
        b = int(bar_in_day_arr[i])
        nb = int(bars_per_day_arr[d]) if d < len(bars_per_day_arr) else 26
        S = prices[i]
        
        mins_remaining = max(float(nb - b) * mins_per_bar, 0.0)
        hrs_remaining = mins_remaining / 60.0
        vix = vix_arr[d] if d < len(vix_arr) else 20.0
        sigma = estimate_iv(vix, hrs_remaining)
        
        # Track daily boundary
        if d != prev_day and prev_day >= 0 and prev_day < num_days:
            daily_equity[prev_day] = equity
        prev_day = d
        
        # --- Evaluate expiring positions ---
        for p in range(MAX_POS):
            if p_active[p] == 0 or p_expiry[p] != d:
                continue
            
            T_rem = mins_remaining / (252.0 * total_mins_day)
            
            cv = bs_price(S, p_sc[p], T_rem, 0.05, sigma, True) - bs_price(S, p_lc[p], T_rem, 0.05, sigma, True)
            pv = bs_price(S, p_sp[p], T_rem, 0.05, sigma, False) - bs_price(S, p_lp[p], T_rem, 0.05, sigma, False)
            
            total_v = cv + pv
            prem = p_prem[p]
            
            if total_v > prem + p_stop[p]:
                pnl = -p_stop[p] * 100.0
                equity += pnl
                total_pnl += pnl
                daily_pnl[d] += pnl
                n_trades += 1
                n_losses += 1
                p_active[p] = 0
            elif mins_remaining <= mins_per_bar:
                pnl = (prem - max(total_v, 0.0)) * 100.0
                equity += pnl
                total_pnl += pnl
                daily_pnl[d] += pnl
                n_trades += 1
                if pnl > 0:
                    n_wins += 1
                else:
                    n_losses += 1
                p_active[p] = 0
            
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        # --- Entry logic ---
        if b < entry_bar_start or b > entry_bar_end:
            continue
        if (b - entry_bar_start) % entry_interval != 0:
            continue
        
        # Count today's entries
        entries = 0
        for p in range(MAX_POS):
            if p_active[p] == 1 and p_expiry[p] == d + 1:
                entries += 1
        if entries >= max_entries:
            continue
        
        # Find empty slot
        slot = -1
        for p in range(MAX_POS):
            if p_active[p] == 0:
                slot = p
                break
        if slot == -1:
            continue
        
        if d + 1 >= num_days:
            continue
        
        # 1DTE: remaining today + full next day
        T_1dte = (mins_remaining + total_mins_day) / (252.0 * total_mins_day)
        sigma_e = estimate_iv(vix, hrs_remaining + 6.5)
        
        sc_k, sc_p_val, sc_d = find_strike(S, T_1dte, 0.05, sigma_e, target_delta, strike_interval, True)
        sp_k, sp_p_val, sp_d = find_strike(S, T_1dte, 0.05, sigma_e, target_delta, strike_interval, False)
        
        if sc_k < 0 or sp_k < 0:
            continue
        if abs(sc_d) < delta_min or abs(sc_d) > delta_max:
            continue
        if abs(sp_d) < delta_min or abs(sp_d) > delta_max:
            continue
        
        lc_k = sc_k + spread_width
        lp_k = sp_k - spread_width
        lc_p = bs_price(S, lc_k, T_1dte, 0.05, sigma_e, True)
        lp_p = bs_price(S, lp_k, T_1dte, 0.05, sigma_e, False)
        
        cc = sc_p_val - lc_p
        pc = sp_p_val - lp_p
        
        if cc <= 0.05 or pc <= 0.05:
            continue
        
        ic_prem = cc + pc
        
        p_active[slot] = 1
        p_sc[slot] = sc_k
        p_lc[slot] = lc_k
        p_sp[slot] = sp_k
        p_lp[slot] = lp_k
        p_prem[slot] = ic_prem
        p_stop[slot] = ic_prem * sl_mult
        p_expiry[slot] = d + 1
        sum_premium += ic_prem
    
    # Force-close remaining
    for p in range(MAX_POS):
        if p_active[p] == 1:
            pnl = p_prem[p] * 0.5 * 100.0
            equity += pnl
            total_pnl += pnl
            n_trades += 1
            n_wins += 1
    
    if d < num_days:
        daily_equity[d] = equity
    
    if equity > peak:
        peak = equity
    dd = (equity - peak) / peak
    if dd < max_dd:
        max_dd = dd
    
    ret_pct = (equity - 100000.0) / 1000.0  # as %
    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    avg_p = (sum_premium / n_trades) if n_trades > 0 else 0.0
    
    # Sharpe (annualized from daily P&L)
    valid_daily = daily_pnl[:num_days]
    mean_d = 0.0
    std_d = 0.0
    count_d = 0
    for dd_i in range(num_days):
        if daily_equity[dd_i] > 0:
            mean_d += daily_pnl[dd_i]
            count_d += 1
    if count_d > 0:
        mean_d /= count_d
        for dd_i in range(num_days):
            if daily_equity[dd_i] > 0:
                std_d += (daily_pnl[dd_i] - mean_d) ** 2
        std_d = np.sqrt(std_d / count_d)
    
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0
    
    return (total_pnl, n_trades, n_wins, n_losses, max_dd * 100.0,
            avg_p, ret_pct, wr, equity, sharpe)


# ==================== Grid Search ====================

def worker(args):
    (prices, day_idx, bar_day, bars_pd, vix,
     dmin, dmax, sw, slm, ebs, ebe, ei, me, si, bm) = args
    
    r = run_backtest(prices, day_idx, bar_day, bars_pd, vix,
                     dmin, dmax, sw, slm, ebs, ebe, ei, me, si, bm)
    
    return {
        "delta_min": dmin, "delta_max": dmax, "spread_width": sw,
        "stop_loss_mult": slm, "entry_bar_start": ebs, "entry_bar_end": ebe,
        "entry_interval": ei, "max_entries": me,
        "total_pnl": round(r[0], 2), "num_trades": int(r[1]),
        "num_wins": int(r[2]), "num_losses": int(r[3]),
        "max_dd_pct": round(r[4], 2), "avg_premium": round(r[5], 2),
        "total_return_pct": round(r[6], 2), "win_rate": round(r[7], 1),
        "final_equity": round(r[8], 2), "sharpe": round(r[9], 2),
    }


def prepare_arrays(df, vix_df, bar_minutes):
    """Convert DataFrame to numpy arrays."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df["date"] = df.index.date
    dates = sorted(df["date"].unique())
    d2i = {d: i for i, d in enumerate(dates)}
    
    prices = df["close"].values.astype(np.float64)
    day_idx = np.array([d2i[d] for d in df["date"]], dtype=np.int32)
    
    bar_day = np.zeros(len(df), dtype=np.int32)
    bpd = []
    for d in dates:
        mask = df["date"] == d
        n = mask.sum()
        bar_day[mask.values] = np.arange(n, dtype=np.int32)
        bpd.append(n)
    bars_pd = np.array(bpd, dtype=np.int32)
    
    vix_arr = np.full(len(dates), 20.0, dtype=np.float64)
    if vix_df is not None:
        vix_df = vix_df.copy()
        vix_df.index = pd.to_datetime(vix_df.index, utc=True)
        for i, d in enumerate(dates):
            prev = vix_df[vix_df.index.date <= d]
            if not prev.empty:
                vix_arr[i] = prev["close"].iloc[-1]
    
    return prices, day_idx, bar_day, bars_pd, vix_arr, dates


def main():
    print("=" * 70)
    print("🚀 MULTI-YEAR 1DTE OPTIMIZER — Alienware (16 cores)")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # ---- Fetch Data ----
    print("\n📥 Fetching data...")
    
    h_file = DATA_DIR / "SPY_1h_730d.csv"
    m_file = DATA_DIR / "SPY_15m_60d.csv"
    v_file = DATA_DIR / "VIX_5y.csv"
    
    if h_file.exists():
        hourly = pd.read_csv(h_file, index_col=0, parse_dates=True)
        print(f"  [cached] SPY 1h: {len(hourly)} bars")
    else:
        hourly = fetch_spy_hourly()
    
    if m_file.exists():
        m15 = pd.read_csv(m_file, index_col=0, parse_dates=True)
        print(f"  [cached] SPY 15m: {len(m15)} bars")
    else:
        m15 = fetch_spy_15m()
    
    if v_file.exists():
        vix = pd.read_csv(v_file, index_col=0, parse_dates=True)
        print(f"  [cached] VIX: {len(vix)} bars")
    else:
        vix = fetch_vix_data()
    
    # ---- Run on 1h data (730 days) ----
    print("\n" + "=" * 70)
    print("📊 PHASE 1: 1h bars — 730 trading days (~3 years)")
    print("=" * 70)
    
    prices_h, didx_h, bday_h, bpd_h, vix_h, dates_h = prepare_arrays(hourly, vix, 60)
    print(f"  {len(dates_h)} trading days, {len(prices_h)} bars")
    print(f"  {dates_h[0]} → {dates_h[-1]}")
    print(f"  Price range: ${prices_h.min():.2f} — ${prices_h.max():.2f}")
    print(f"  VIX range: {vix_h.min():.1f} — {vix_h.max():.1f}")
    
    # Warmup
    print("\n🔥 JIT warmup...")
    t0 = time.time()
    _ = run_backtest(prices_h, didx_h, bday_h, bpd_h, vix_h,
                     0.10, 0.20, 10.0, 1.5, 1, 5, 1, 3, 1.0, 60.0)
    print(f"  Compile: {time.time()-t0:.2f}s")
    
    # Speed test
    t0 = time.time()
    for _ in range(100):
        _ = run_backtest(prices_h, didx_h, bday_h, bpd_h, vix_h,
                         0.10, 0.20, 10.0, 1.5, 1, 5, 1, 3, 1.0, 60.0)
    spd = 100 / (time.time() - t0)
    print(f"  Speed: {spd:.0f} backtests/sec (single core)")
    
    # Grid — for 1h bars, bar indices are different (7 bars/day vs 26 for 15m)
    # entry_bar_start: 1=10:30, 2=11:30, etc for 1h
    grid = []
    for dmin in [0.05, 0.08, 0.10, 0.12, 0.15]:
        for dmax in [0.15, 0.20, 0.25, 0.30]:
            if dmin >= dmax: continue
            for sw in [5.0, 10.0, 15.0, 20.0]:
                for slm in [1.0, 1.5, 2.0, 2.5, 3.0]:
                    for ebs in [0, 1, 2]:  # 9:30, 10:30, 11:30
                        for ebe in [4, 5, 6]:  # 13:30, 14:30, 15:30
                            if ebs >= ebe: continue
                            for ei in [1, 2]:  # every bar or every 2 bars
                                for me in [2, 3, 4, 5]:
                                    grid.append((
                                        prices_h, didx_h, bday_h, bpd_h, vix_h,
                                        dmin, dmax, sw, slm, ebs, ebe, ei, me, 1.0, 60.0
                                    ))
    
    total = len(grid)
    print(f"\n📊 Grid: {total} configs")
    print(f"   Est time: {total/spd/cpu_count():.1f}s")
    
    t0 = time.time()
    with Pool(cpu_count()) as pool:
        results_h = pool.map(worker, grid)
    elapsed = time.time() - t0
    print(f"\n✅ {total} configs in {elapsed:.1f}s ({total/elapsed:.0f}/sec)")
    
    # Filter & sort
    results_h = [r for r in results_h if r["num_trades"] >= 10]
    results_h.sort(key=lambda x: x["sharpe"], reverse=True)
    
    print(f"\n  Valid configs (≥10 trades): {len(results_h)}")
    
    # Print top 20
    print("\n" + "=" * 120)
    print("🏆 TOP 20 CONFIGS — SORTED BY SHARPE (1h, ~3 years)")
    print("=" * 120)
    print(f"{'#':>3} {'Sharpe':>7} {'Return%':>8} {'WR%':>6} {'Trades':>7} {'DD%':>7} {'AvgPrem':>8} {'Final$':>12} {'Δmin':>5} {'Δmax':>5} {'Width':>6} {'SL×':>4} {'MaxE':>5}")
    print("-" * 120)
    
    for i, r in enumerate(results_h[:20]):
        print(f"{i+1:>3} {r['sharpe']:>7.2f} {r['total_return_pct']:>8.2f} {r['win_rate']:>6.1f} "
              f"{r['num_trades']:>7} {r['max_dd_pct']:>7.2f} {r['avg_premium']:>8.2f} "
              f"{r['final_equity']:>12.2f} {r['delta_min']:>5.2f} {r['delta_max']:>5.2f} "
              f"{r['spread_width']:>6.1f} {r['stop_loss_mult']:>4.1f} {r['max_entries']:>5}")
    
    # Stats
    if results_h:
        profitable = sum(1 for r in results_h if r["total_return_pct"] > 0)
        print(f"\n📊 Stats:")
        print(f"  Profitable: {profitable}/{len(results_h)} ({profitable/len(results_h)*100:.1f}%)")
        print(f"  Sharpe > 1: {sum(1 for r in results_h if r['sharpe'] > 1)}")
        print(f"  Sharpe > 2: {sum(1 for r in results_h if r['sharpe'] > 2)}")
        print(f"  Return > 50%: {sum(1 for r in results_h if r['total_return_pct'] > 50)}")
        print(f"  Return > 100%: {sum(1 for r in results_h if r['total_return_pct'] > 100)}")
        print(f"  DD < 5%: {sum(1 for r in results_h if abs(r['max_dd_pct']) < 5)}")
        print(f"  DD < 10%: {sum(1 for r in results_h if abs(r['max_dd_pct']) < 10)}")
        
        best = results_h[0]
        print(f"\n🏆 BEST CONFIG (by Sharpe):")
        print(json.dumps(best, indent=2))
    
    # Save
    with open(RESULTS_DIR / "multiyear_1h_results.json", "w") as f:
        json.dump({
            "meta": {
                "data": "SPY 1h bars",
                "trading_days": len(dates_h),
                "date_range": f"{dates_h[0]} → {dates_h[-1]}",
                "total_configs": total,
                "valid_configs": len(results_h),
                "elapsed_sec": round(elapsed, 1),
                "speed_per_sec": round(total/elapsed, 0),
            },
            "top_50": results_h[:50],
        }, f, indent=2, default=str)
    
    # ---- Charts ----
    if results_h:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        rets = [r["total_return_pct"] for r in results_h]
        sharpes = [r["sharpe"] for r in results_h]
        wrs = [r["win_rate"] for r in results_h]
        dds = [abs(r["max_dd_pct"]) for r in results_h]
        
        axes[0,0].hist(rets, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0,0].axvline(0, color='red', ls='--')
        axes[0,0].set_title(f"Return Distribution ({len(results_h)} configs, {len(dates_h)} days)")
        axes[0,0].set_xlabel("Total Return %")
        
        axes[0,1].scatter(sharpes, rets, alpha=0.3, s=8, c='steelblue')
        axes[0,1].set_title("Sharpe vs Return")
        axes[0,1].set_xlabel("Sharpe Ratio"); axes[0,1].set_ylabel("Return %")
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].scatter(dds, rets, alpha=0.3, s=8, c='steelblue')
        axes[1,0].set_title("Max DD vs Return")
        axes[1,0].set_xlabel("Max Drawdown %"); axes[1,0].set_ylabel("Return %")
        axes[1,0].grid(True, alpha=0.3)
        
        for sw in [5, 10, 15, 20]:
            sw_r = [r["total_return_pct"] for r in results_h if r["spread_width"] == sw]
            if sw_r: axes[1,1].hist(sw_r, bins=30, alpha=0.4, label=f"${sw}")
        axes[1,1].set_title("Return by Spread Width")
        axes[1,1].set_xlabel("Return %"); axes[1,1].legend()
        
        plt.suptitle(f"1DTE Iron Condor — Multi-Year Optimization ({len(dates_h)} days, {total} configs)", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "multiyear_optimizer.png", dpi=150)
        plt.close()
        print(f"\n📈 Charts saved")
    
    print(f"\n✅ All saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

"""
IV Skew-Adjusted Synthetic Pricing Model

Uses the real IV surface captured from yfinance option chains to build
a realistic pricing model that accounts for volatility smile/skew.

Key improvements over flat-IV Black-Scholes:
1. IV varies by moneyness (OTM puts have higher IV due to skew)
2. IV varies by DTE (term structure)
3. Bid/ask spread modeled by moneyness
4. More accurate premium estimates for iron condor backtests

Runs on Alienware for speed (Numba @njit compiled).
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from numba import njit, prange
from multiprocessing import Pool, cpu_count
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from data_fetcher import DATA_DIR

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ==================== IV Surface Fitting ====================

def load_real_iv_surface(chains_file: str = None) -> pd.DataFrame:
    """
    Load real options chain data and extract IV surface.
    Returns DataFrame with: moneyness, dte, iv, bid_ask_spread_pct, type
    """
    if chains_file is None:
        chains_file = DATA_DIR / "SPY_options_all_chains.csv"
    
    df = pd.read_csv(chains_file)
    
    # Clean data
    df = df[df["impliedVolatility"] > 0].copy()
    df = df[df["bid"] > 0].copy()
    df = df[df["ask"] > 0].copy()
    
    # Calculate moneyness = strike / underlying_price
    df["moneyness"] = df["strike"] / df["underlying_price"]
    
    # Bid/ask spread
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"] * 100
    
    print(f"Loaded {len(df)} options with valid IV")
    print(f"  DTE range: {df['dte'].min()} - {df['dte'].max()}")
    print(f"  Moneyness range: {df['moneyness'].min():.3f} - {df['moneyness'].max():.3f}")
    print(f"  IV range: {df['impliedVolatility'].min():.3f} - {df['impliedVolatility'].max():.3f}")
    
    return df


def fit_iv_skew(df: pd.DataFrame) -> dict:
    """
    Fit parametric IV skew model to real data.
    
    Model: IV(m, T) = a + b*(1-m) + c*(1-m)^2 + d/sqrt(T)
    
    Where:
    - m = moneyness (strike/spot)
    - T = DTE (days to expiry)
    - a = ATM IV level
    - b = skew slope (linear)
    - c = smile curvature (quadratic)
    - d = term structure adjustment
    
    Fits separately for puts and calls.
    """
    params = {}
    
    for opt_type in ["put", "call"]:
        subset = df[df["type"] == opt_type].copy()
        
        if opt_type == "put":
            # OTM puts: moneyness < 1
            subset = subset[subset["moneyness"] < 1.0]
        else:
            # OTM calls: moneyness > 1
            subset = subset[subset["moneyness"] > 1.0]
        
        if len(subset) < 10:
            print(f"  Not enough {opt_type} data ({len(subset)} points)")
            continue
        
        # Features
        m = subset["moneyness"].values
        T = np.maximum(subset["dte"].values, 0.5)  # Avoid div by zero
        iv = subset["impliedVolatility"].values
        
        # Fit quadratic skew: IV = a + b*(1-m) + c*(1-m)^2 + d/sqrt(T)
        def skew_model(X, a, b, c, d):
            m, T = X
            return a + b * (1 - m) + c * (1 - m)**2 + d / np.sqrt(T)
        
        try:
            popt, pcov = curve_fit(
                skew_model, (m, T), iv,
                p0=[0.15, 0.5, 2.0, 0.01],
                bounds=([0, -2, -10, -0.5], [1, 5, 50, 0.5]),
                maxfev=10000
            )
            
            residuals = iv - skew_model((m, T), *popt)
            rmse = np.sqrt(np.mean(residuals**2))
            r2 = 1 - np.sum(residuals**2) / np.sum((iv - iv.mean())**2)
            
            params[opt_type] = {
                "a": float(popt[0]),  # ATM IV base
                "b": float(popt[1]),  # Skew slope
                "c": float(popt[2]),  # Smile curvature
                "d": float(popt[3]),  # Term structure
                "rmse": float(rmse),
                "r2": float(r2),
                "n_points": len(subset),
            }
            
            print(f"\n  {opt_type.upper()} IV Skew Model:")
            print(f"    IV(m,T) = {popt[0]:.4f} + {popt[1]:.4f}*(1-m) + "
                  f"{popt[2]:.4f}*(1-m)² + {popt[3]:.4f}/√T")
            print(f"    R² = {r2:.4f}, RMSE = {rmse:.4f}")
            print(f"    Points: {len(subset)}")
            
        except Exception as e:
            print(f"  Failed to fit {opt_type}: {e}")
            # Fallback: simple linear skew
            params[opt_type] = {
                "a": float(np.median(iv)),
                "b": 0.5,
                "c": 2.0,
                "d": 0.01,
                "rmse": 0,
                "r2": 0,
                "n_points": len(subset),
                "fallback": True,
            }
    
    # Also fit bid/ask spread model
    for opt_type in ["put", "call"]:
        subset = df[(df["type"] == opt_type) & (df["spread_pct"] > 0) & 
                    (df["spread_pct"] < 200)].copy()
        
        if len(subset) < 5:
            params[f"{opt_type}_spread"] = {"a": 5.0, "b": 50.0}
            continue
        
        m = subset["moneyness"].values
        spreads = subset["spread_pct"].values
        
        # Spread model: spread_pct = a + b * |1-m|
        try:
            def spread_model(m, a, b):
                return a + b * np.abs(1 - m)
            
            popt, _ = curve_fit(spread_model, m, spreads, p0=[3.0, 100.0],
                               bounds=([0, 0], [50, 1000]))
            
            params[f"{opt_type}_spread"] = {
                "a": float(popt[0]),
                "b": float(popt[1]),
            }
            print(f"  {opt_type.upper()} spread: {popt[0]:.1f}% + {popt[1]:.1f}% * |1-m|")
        except:
            params[f"{opt_type}_spread"] = {"a": 5.0, "b": 50.0}
    
    return params


def plot_iv_surface(df: pd.DataFrame, params: dict, save_path: str = None):
    """Visualize the fitted IV surface vs real data."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Put IV skew (real vs fitted)
    puts = df[(df["type"] == "put") & (df["moneyness"] < 1.0) & (df["moneyness"] > 0.9)]
    if not puts.empty and "put" in params:
        for dte in sorted(puts["dte"].unique())[:5]:
            subset = puts[puts["dte"] == dte].sort_values("moneyness")
            axes[0, 0].scatter(subset["moneyness"], subset["impliedVolatility"], 
                              alpha=0.5, s=20, label=f"DTE={dte}")
        
        # Fitted curve
        m_range = np.linspace(0.9, 1.0, 100)
        p = params["put"]
        for dte in [1, 5, 20]:
            iv_fitted = p["a"] + p["b"] * (1 - m_range) + p["c"] * (1 - m_range)**2 + p["d"] / np.sqrt(dte)
            axes[0, 0].plot(m_range, iv_fitted, '--', linewidth=2, label=f"Fit DTE={dte}")
        
        axes[0, 0].set_title("Put IV Skew (Real vs Fitted)")
        axes[0, 0].set_xlabel("Moneyness (K/S)")
        axes[0, 0].set_ylabel("Implied Volatility")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Call IV skew
    calls = df[(df["type"] == "call") & (df["moneyness"] > 1.0) & (df["moneyness"] < 1.1)]
    if not calls.empty and "call" in params:
        for dte in sorted(calls["dte"].unique())[:5]:
            subset = calls[calls["dte"] == dte].sort_values("moneyness")
            axes[0, 1].scatter(subset["moneyness"], subset["impliedVolatility"],
                              alpha=0.5, s=20, label=f"DTE={dte}")
        
        m_range = np.linspace(1.0, 1.1, 100)
        p = params["call"]
        for dte in [1, 5, 20]:
            iv_fitted = p["a"] + p["b"] * (1 - m_range) + p["c"] * (1 - m_range)**2 + p["d"] / np.sqrt(dte)
            axes[0, 1].plot(m_range, iv_fitted, '--', linewidth=2, label=f"Fit DTE={dte}")
        
        axes[0, 1].set_title("Call IV Skew (Real vs Fitted)")
        axes[0, 1].set_xlabel("Moneyness (K/S)")
        axes[0, 1].set_ylabel("Implied Volatility")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Bid/Ask spread by moneyness
    for opt_type, color in [("put", "blue"), ("call", "red")]:
        subset = df[(df["type"] == opt_type) & (df["spread_pct"] > 0) & (df["spread_pct"] < 100)]
        if not subset.empty:
            axes[1, 0].scatter(subset["moneyness"], subset["spread_pct"],
                              alpha=0.3, s=10, color=color, label=f"{opt_type}s")
    
    axes[1, 0].set_title("Bid/Ask Spread by Moneyness")
    axes[1, 0].set_xlabel("Moneyness (K/S)")
    axes[1, 0].set_ylabel("Spread %")
    axes[1, 0].set_ylim(0, 80)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. IV term structure (ATM)
    atm = df[(df["moneyness"] > 0.97) & (df["moneyness"] < 1.03)]
    if not atm.empty:
        atm_grouped = atm.groupby("dte")["impliedVolatility"].agg(["mean", "std"]).reset_index()
        axes[1, 1].bar(atm_grouped["dte"], atm_grouped["mean"], 
                       yerr=atm_grouped["std"], alpha=0.7, color="steelblue")
        axes[1, 1].set_title("IV Term Structure (Near-ATM)")
        axes[1, 1].set_xlabel("Days to Expiry")
        axes[1, 1].set_ylabel("Implied Volatility")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("SPY Options — IV Surface Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    save_to = save_path or RESULTS_DIR / "iv_surface_analysis.png"
    plt.savefig(save_to, dpi=150)
    plt.close()
    print(f"\n📊 IV surface chart saved to {save_to}")


# ==================== Numba-compiled Skew-Adjusted BS ====================

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
    if is_call: return norm_cdf(d1)
    else: return norm_cdf(d1) - 1.0


@njit(cache=True)
def skew_iv(moneyness, dte_days, a, b, c, d, vix_base):
    """
    Get IV adjusted for skew.
    
    IV = vix_base * (a + b*(1-m) + c*(1-m)^2 + d/sqrt(T))
    
    vix_base scales the entire surface up/down with VIX level.
    """
    m = moneyness
    T = max(dte_days, 0.5)
    
    # Base IV from skew model
    iv = a + b * (1.0 - m) + c * (1.0 - m)**2 + d / np.sqrt(T)
    
    # Scale by VIX level (our params were fit at a specific VIX, scale proportionally)
    # If VIX was ~15 when we fit, and current VIX is different, scale
    iv = iv * (vix_base / 15.0)
    
    # Clamp to reasonable range
    return max(min(iv, 3.0), 0.05)


@njit(cache=True)
def bid_ask_spread_pct(moneyness, spread_a, spread_b):
    """Estimate bid/ask spread as percentage of mid price."""
    return spread_a + spread_b * abs(1.0 - moneyness)


@njit(cache=True)
def find_strike_by_delta_skew(
    S, T_years, r, vix,
    target_delta, strike_interval, is_call,
    # Skew params
    skew_a, skew_b, skew_c, skew_d,
    spread_a, spread_b,
    dte_days,
    search_range_pct=0.08
):
    """
    Find strike by delta using skew-adjusted IV.
    Returns (strike, premium_mid, premium_bid, delta, iv_used).
    """
    if is_call:
        start = S + strike_interval
        end = S * (1.0 + search_range_pct)
    else:
        start = S - strike_interval
        end = S * (1.0 - search_range_pct)
    
    best_strike = -1.0
    best_diff = 999.0
    best_prem_mid = 0.0
    best_prem_bid = 0.0
    best_delta = 0.0
    best_iv = 0.0
    
    strike = start
    while (strike <= end) if is_call else (strike >= end):
        m = strike / S
        sigma = skew_iv(m, dte_days, skew_a, skew_b, skew_c, skew_d, vix)
        
        d = bs_delta(S, strike, T_years, r, sigma, is_call)
        diff = abs(abs(d) - target_delta)
        
        if diff < best_diff:
            best_diff = diff
            best_strike = strike
            best_delta = d
            best_iv = sigma
            
            # Premium at mid
            prem_mid = bs_price(S, strike, T_years, r, sigma, is_call)
            best_prem_mid = prem_mid
            
            # Adjust for bid/ask spread (seller gets bid, which is lower)
            spread = bid_ask_spread_pct(m, spread_a, spread_b) / 100.0
            best_prem_bid = prem_mid * (1.0 - spread / 2.0)  # Bid ~ mid - half spread
        
        if is_call:
            strike += strike_interval
        else:
            strike -= strike_interval
    
    return best_strike, best_prem_mid, best_prem_bid, best_delta, best_iv


@njit(cache=True)
def run_ic_backtest_skew(
    prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
    # Strategy params
    delta_min, delta_max, spread_width, stop_loss_mult,
    entry_bar_start, entry_bar_end, entry_interval, max_entries, strike_interval,
    # Skew model params (puts)
    put_skew_a, put_skew_b, put_skew_c, put_skew_d,
    put_spread_a, put_spread_b,
    # Skew model params (calls)
    call_skew_a, call_skew_b, call_skew_c, call_skew_d,
    call_spread_a, call_spread_b,
    # Use bid price for selling (realistic)
    use_bid_price=True,
    # Constants
    r=0.05,
    total_bars_per_day=390.0,
):
    """
    Run 1DTE iron condor backtest with IV skew-adjusted pricing.
    Same logic as flat-IV version but uses skew model for pricing.
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
    total_premium_flat = 0.0  # Track what flat IV would have given
    
    MAX_POS = 10
    pos_active = np.zeros(MAX_POS, dtype=np.int32)
    pos_short_call = np.zeros(MAX_POS)
    pos_long_call = np.zeros(MAX_POS)
    pos_short_put = np.zeros(MAX_POS)
    pos_long_put = np.zeros(MAX_POS)
    pos_premium = np.zeros(MAX_POS)
    pos_stop = np.zeros(MAX_POS)
    pos_expiry_day = np.zeros(MAX_POS, dtype=np.int32)
    pos_premium_flat = np.zeros(MAX_POS)  # For comparison
    
    target_delta = (delta_min + delta_max) / 2.0
    
    for i in range(N):
        day_idx = int(day_indices[i])
        bar_idx = int(bar_in_day[i])
        n_bars = int(bars_per_day[day_idx]) if day_idx < len(bars_per_day) else 390
        
        S = prices[i]
        mins_remaining = max(float(n_bars - bar_idx), 0.0)
        hours_remaining = mins_remaining / 60.0
        
        vix = vix_per_day[day_idx] if day_idx < len(vix_per_day) else 15.0
        
        # ---- Check expiring positions ----
        for p in range(MAX_POS):
            if pos_active[p] == 0:
                continue
            if pos_expiry_day[p] != day_idx:
                continue
            
            T_rem = mins_remaining / (252.0 * 6.5 * 60.0)
            dte_rem = mins_remaining / (6.5 * 60.0)  # DTE in days
            
            # Use skew IV for mark-to-market
            sc_m = pos_short_call[p] / S
            sp_m = pos_short_put[p] / S
            lc_m = pos_long_call[p] / S
            lp_m = pos_long_put[p] / S
            
            sc_iv = skew_iv(sc_m, dte_rem, call_skew_a, call_skew_b, call_skew_c, call_skew_d, vix)
            lc_iv = skew_iv(lc_m, dte_rem, call_skew_a, call_skew_b, call_skew_c, call_skew_d, vix)
            sp_iv = skew_iv(sp_m, dte_rem, put_skew_a, put_skew_b, put_skew_c, put_skew_d, vix)
            lp_iv = skew_iv(lp_m, dte_rem, put_skew_a, put_skew_b, put_skew_c, put_skew_d, vix)
            
            call_val = (bs_price(S, pos_short_call[p], T_rem, r, sc_iv, True) -
                       bs_price(S, pos_long_call[p], T_rem, r, lc_iv, True))
            put_val = (bs_price(S, pos_short_put[p], T_rem, r, sp_iv, False) -
                      bs_price(S, pos_long_put[p], T_rem, r, lp_iv, False))
            
            total_val = call_val + put_val
            prem = pos_premium[p]
            
            # Stop loss
            if total_val > prem + pos_stop[p]:
                pnl = -pos_stop[p] * 100.0
                equity += pnl
                total_pnl += pnl
                num_trades += 1
                num_losses += 1
                pos_active[p] = 0
            elif mins_remaining <= 1.0:
                pnl = (prem - max(total_val, 0.0)) * 100.0
                equity += pnl
                total_pnl += pnl
                num_trades += 1
                if pnl > 0:
                    num_wins += 1
                else:
                    num_losses += 1
                pos_active[p] = 0
            
            if equity > peak_equity:
                peak_equity = equity
            dd = (equity - peak_equity) / peak_equity
            if dd < max_dd:
                max_dd = dd
        
        # ---- Entry logic ----
        if bar_idx < entry_bar_start or bar_idx > entry_bar_end:
            continue
        
        entries_today = 0
        for p in range(MAX_POS):
            if pos_active[p] == 1 and pos_expiry_day[p] == day_idx + 1:
                entries_today += 1
        
        if entries_today >= max_entries:
            continue
        
        if (bar_idx - entry_bar_start) % entry_interval != 0:
            continue
        
        slot = -1
        for p in range(MAX_POS):
            if pos_active[p] == 0:
                slot = p
                break
        if slot == -1:
            continue
        
        next_day = day_idx + 1
        if next_day >= num_days:
            continue
        
        # T for 1DTE
        T_1dte = (mins_remaining + 6.5 * 60.0) / (252.0 * 6.5 * 60.0)
        dte_1dte = (mins_remaining + 6.5 * 60.0) / (6.5 * 60.0)
        
        # Find short call with skew-adjusted IV
        sc_strike, sc_prem_mid, sc_prem_bid, sc_delta, sc_iv = find_strike_by_delta_skew(
            S, T_1dte, r, vix, target_delta, strike_interval, True,
            call_skew_a, call_skew_b, call_skew_c, call_skew_d,
            call_spread_a, call_spread_b, dte_1dte
        )
        
        if sc_strike < 0 or abs(sc_delta) < delta_min or abs(sc_delta) > delta_max:
            continue
        
        # Find short put with skew-adjusted IV
        sp_strike, sp_prem_mid, sp_prem_bid, sp_delta, sp_iv = find_strike_by_delta_skew(
            S, T_1dte, r, vix, target_delta, strike_interval, False,
            put_skew_a, put_skew_b, put_skew_c, put_skew_d,
            put_spread_a, put_spread_b, dte_1dte
        )
        
        if sp_strike < 0 or abs(sp_delta) < delta_min or abs(sp_delta) > delta_max:
            continue
        
        # Long legs
        lc_strike = sc_strike + spread_width
        lp_strike = sp_strike - spread_width
        
        # Long leg pricing with skew
        lc_m = lc_strike / S
        lp_m = lp_strike / S
        lc_iv = skew_iv(lc_m, dte_1dte, call_skew_a, call_skew_b, call_skew_c, call_skew_d, vix)
        lp_iv = skew_iv(lp_m, dte_1dte, put_skew_a, put_skew_b, put_skew_c, put_skew_d, vix)
        
        lc_prem = bs_price(S, lc_strike, T_1dte, r, lc_iv, True)
        lp_prem = bs_price(S, lp_strike, T_1dte, r, lp_iv, False)
        
        # Use bid price for shorts (what seller receives) and ask for longs (what buyer pays)
        if use_bid_price:
            # Short legs: we get bid
            sc_prem = sc_prem_bid
            sp_prem = sp_prem_bid
            # Long legs: we pay ask (mid + half spread)
            lc_spread = bid_ask_spread_pct(lc_m, call_spread_a, call_spread_b) / 100.0
            lp_spread = bid_ask_spread_pct(lp_m, put_spread_a, put_spread_b) / 100.0
            lc_prem_ask = lc_prem * (1.0 + lc_spread / 2.0)
            lp_prem_ask = lp_prem * (1.0 + lp_spread / 2.0)
        else:
            sc_prem = sc_prem_mid
            sp_prem = sp_prem_mid
            lc_prem_ask = lc_prem
            lp_prem_ask = lp_prem
        
        call_credit = sc_prem - lc_prem_ask
        put_credit = sp_prem - lp_prem_ask
        
        if call_credit <= 0.05 or put_credit <= 0.05:
            continue
        
        ic_premium = call_credit + put_credit
        
        # Also calculate flat IV premium for comparison
        flat_iv = vix / 100.0 * 1.3
        sc_prem_flat = bs_price(S, sc_strike, T_1dte, r, flat_iv, True)
        lc_prem_flat = bs_price(S, lc_strike, T_1dte, r, flat_iv, True)
        sp_prem_flat = bs_price(S, sp_strike, T_1dte, r, flat_iv, False)
        lp_prem_flat = bs_price(S, lp_strike, T_1dte, r, flat_iv, False)
        ic_premium_flat = (sc_prem_flat - lc_prem_flat) + (sp_prem_flat - lp_prem_flat)
        
        # Open position
        pos_active[slot] = 1
        pos_short_call[slot] = sc_strike
        pos_long_call[slot] = lc_strike
        pos_short_put[slot] = sp_strike
        pos_long_put[slot] = lp_strike
        pos_premium[slot] = ic_premium
        pos_stop[slot] = ic_premium * stop_loss_mult
        pos_expiry_day[slot] = next_day
        pos_premium_flat[slot] = ic_premium_flat
        total_premium += ic_premium
        total_premium_flat += ic_premium_flat
    
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
    avg_prem_flat = (total_premium_flat / num_trades) if num_trades > 0 else 0.0
    
    return (total_pnl, num_trades, num_wins, num_losses, max_dd * 100.0,
            avg_prem, total_return_pct, win_rate, equity, avg_prem_flat)


# ==================== Optimizer ====================

def run_single_config_skew(args):
    """Worker function for multiprocessing."""
    (prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
     delta_min, delta_max, spread_width, stop_loss_mult,
     entry_bar_start, entry_bar_end, entry_interval, max_entries, strike_interval,
     put_skew_a, put_skew_b, put_skew_c, put_skew_d,
     put_spread_a, put_spread_b,
     call_skew_a, call_skew_b, call_skew_c, call_skew_d,
     call_spread_a, call_spread_b,
     use_bid_price) = args
    
    result = run_ic_backtest_skew(
        prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
        delta_min, delta_max, spread_width, stop_loss_mult,
        entry_bar_start, entry_bar_end, entry_interval, max_entries, strike_interval,
        put_skew_a, put_skew_b, put_skew_c, put_skew_d,
        put_spread_a, put_spread_b,
        call_skew_a, call_skew_b, call_skew_c, call_skew_d,
        call_spread_a, call_spread_b,
        use_bid_price,
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
        "total_pnl": round(result[0], 2),
        "num_trades": int(result[1]),
        "num_wins": int(result[2]),
        "num_losses": int(result[3]),
        "max_dd_pct": round(result[4], 2),
        "avg_premium_skew": round(result[5], 4),
        "total_return_pct": round(result[6], 2),
        "win_rate": round(result[7], 1),
        "final_equity": round(result[8], 2),
        "avg_premium_flat": round(result[9], 4),
        "premium_uplift_pct": round((result[5] / result[9] - 1) * 100, 1) if result[9] > 0 else 0,
    }


def prepare_data(df, vix_df):
    """Same as optimizer_njit — convert to numpy arrays."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df["date"] = df.index.date
    
    dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    prices = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    
    day_indices = np.array([date_to_idx[d] for d in df["date"]], dtype=np.int32)
    
    bar_in_day = np.zeros(len(df), dtype=np.int32)
    bars_per_day_list = []
    for d in dates:
        mask = df["date"] == d
        n = mask.sum()
        bar_in_day[mask.values] = np.arange(n, dtype=np.int32)
        bars_per_day_list.append(n)
    
    bars_per_day = np.array(bars_per_day_list, dtype=np.int32)
    
    vix_per_day = np.full(len(dates), 15.0, dtype=np.float64)
    if vix_df is not None and not vix_df.empty:
        vix_df.index = pd.to_datetime(vix_df.index, utc=True)
        for i, d in enumerate(dates):
            prev = vix_df[vix_df.index.date <= d]
            if not prev.empty:
                vix_per_day[i] = prev["close"].iloc[-1]
    
    return prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day, dates


def main():
    from itertools import product
    
    print("=" * 70)
    print("🚀 IV SKEW-ADJUSTED 1DTE OPTIMIZER")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   CPU cores: {cpu_count()}")
    print("=" * 70)
    
    # Step 1: Load and fit IV surface
    print("\n📊 Step 1: Fitting IV skew model from real data...")
    real_data = load_real_iv_surface()
    skew_params = fit_iv_skew(real_data)
    
    # Save params
    with open(RESULTS_DIR / "iv_skew_params.json", "w") as f:
        json.dump(skew_params, f, indent=2)
    print(f"\n  Saved skew params to {RESULTS_DIR}/iv_skew_params.json")
    
    # Plot IV surface
    plot_iv_surface(real_data, skew_params)
    
    # Extract params for numba
    put_p = skew_params.get("put", {"a": 0.15, "b": 0.5, "c": 2.0, "d": 0.01})
    call_p = skew_params.get("call", {"a": 0.15, "b": -0.3, "c": 1.0, "d": 0.01})
    put_sp = skew_params.get("put_spread", {"a": 5.0, "b": 50.0})
    call_sp = skew_params.get("call_spread", {"a": 5.0, "b": 50.0})
    
    # Step 2: Load price data
    print("\n📥 Step 2: Loading price data...")
    
    # Use 1h data for multi-year backtest
    hourly_file = DATA_DIR / "SPY_1h_730d.csv"
    vix_file = DATA_DIR / "VIX_5y.csv"
    
    if hourly_file.exists():
        print("  Using cached SPY 1h data")
        intraday = pd.read_csv(hourly_file, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        print("  Fetching SPY 1h data...")
        intraday = yf.Ticker("SPY").history(period="730d", interval="1h")
        intraday.columns = [c.lower() for c in intraday.columns]
        intraday.to_csv(hourly_file)
    
    if vix_file.exists():
        print("  Using cached VIX data")
        vix = pd.read_csv(vix_file, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        print("  Fetching VIX data...")
        vix = yf.Ticker("^VIX").history(period="5y")
        vix.columns = [c.lower() for c in vix.columns]
        vix.to_csv(vix_file)
    
    print(f"  SPY: {len(intraday)} bars")
    
    # Prepare data
    prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day, dates = \
        prepare_data(intraday, vix)
    print(f"  {len(dates)} trading days, {len(prices)} total bars")
    print(f"  VIX range: {vix_per_day.min():.1f} - {vix_per_day.max():.1f}")
    
    # Step 3: JIT warmup
    print("\n🔥 Step 3: JIT warmup...")
    t0 = time.time()
    _ = run_ic_backtest_skew(
        prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
        0.05, 0.15, 5.0, 1.0, 30, 240, 30, 2, 1.0,
        put_p["a"], put_p["b"], put_p["c"], put_p["d"],
        put_sp["a"], put_sp["b"],
        call_p["a"], call_p["b"], call_p["c"], call_p["d"],
        call_sp["a"], call_sp["b"],
        True,
    )
    print(f"  JIT compile: {time.time()-t0:.2f}s")
    
    # Speed test
    t0 = time.time()
    for _ in range(50):
        _ = run_ic_backtest_skew(
            prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
            0.05, 0.15, 5.0, 1.0, 30, 240, 30, 2, 1.0,
            put_p["a"], put_p["b"], put_p["c"], put_p["d"],
            put_sp["a"], put_sp["b"],
            call_p["a"], call_p["b"], call_p["c"], call_p["d"],
            call_sp["a"], call_sp["b"],
            True,
        )
    speed = 50 / (time.time() - t0)
    print(f"  Speed: {speed:.0f} backtests/sec (single core)")
    
    # Step 4: Grid search
    print("\n📊 Step 4: Grid search...")
    
    grid = {
        "delta_min":       [0.05, 0.08, 0.10, 0.12],
        "delta_max":       [0.15, 0.20, 0.25],
        "spread_width":    [5.0, 10.0, 15.0],
        "stop_loss_mult":  [1.0, 1.5, 2.0, 2.5],
        "entry_bar_start": [1, 2, 3],              # 1h bars: bar 1~3 = 10:30, 11:30, 12:30
        "entry_bar_end":   [4, 5, 6],              # bar 4~6 = 13:30, 14:30, 15:30
        "entry_interval":  [1, 2],                  # 1h bars apart
        "max_entries":     [2, 3, 4, 5],
    }
    
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    
    args_list = []
    for combo in combos:
        params = dict(zip(keys, combo))
        if params["delta_min"] >= params["delta_max"]:
            continue
        if params["entry_bar_start"] >= params["entry_bar_end"]:
            continue
        
        args_list.append((
            prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
            params["delta_min"], params["delta_max"], params["spread_width"],
            params["stop_loss_mult"], params["entry_bar_start"], params["entry_bar_end"],
            params["entry_interval"], params["max_entries"], 1.0,
            put_p["a"], put_p["b"], put_p["c"], put_p["d"],
            put_sp["a"], put_sp["b"],
            call_p["a"], call_p["b"], call_p["c"], call_p["d"],
            call_sp["a"], call_sp["b"],
            True,  # use_bid_price
        ))
    
    total = len(args_list)
    print(f"  {total} valid combos (from {len(combos)} total)")
    print(f"  Estimated time: {total / speed / cpu_count():.1f}s ({cpu_count()} cores)")
    
    # Run
    print(f"\n🏃 Running {total} backtests on {cpu_count()} cores...")
    t0 = time.time()
    
    with Pool(cpu_count()) as pool:
        results = pool.map(run_single_config_skew, args_list)
    
    elapsed = time.time() - t0
    actual_speed = total / elapsed
    print(f"\n✅ Done! {total} configs in {elapsed:.1f}s ({actual_speed:.0f}/sec)")
    
    # Filter and sort
    results = [r for r in results if r["num_trades"] >= 5]
    results.sort(key=lambda x: x["total_return_pct"], reverse=True)
    
    # Print results
    print(f"\n{'='*120}")
    print(f"🏆 TOP 20 CONFIGS — IV SKEW-ADJUSTED (by total return)")
    print(f"{'='*120}")
    print(f"{'#':>3} {'Return%':>8} {'WR%':>6} {'Trades':>7} {'MaxDD%':>7} "
          f"{'AvgPrem':>8} {'FlatPrem':>8} {'Uplift%':>8} {'Final$':>12} "
          f"{'Δmin':>5} {'Δmax':>5} {'Width':>6} {'SL×':>4} {'Entries':>8}")
    print("-" * 120)
    
    for i, r in enumerate(results[:20]):
        print(f"{i+1:>3} {r['total_return_pct']:>8.2f} {r['win_rate']:>6.1f} "
              f"{r['num_trades']:>7} {r['max_dd_pct']:>7.2f} "
              f"{r['avg_premium_skew']:>8.4f} {r['avg_premium_flat']:>8.4f} "
              f"{r['premium_uplift_pct']:>8.1f} {r['final_equity']:>12.2f} "
              f"{r['delta_min']:>5.2f} {r['delta_max']:>5.2f} "
              f"{r['spread_width']:>6.1f} {r['stop_loss_mult']:>4.1f} "
              f"{r['max_entries']:>8}")
    
    # Stats
    if results:
        returns = [r["total_return_pct"] for r in results]
        uplifts = [r["premium_uplift_pct"] for r in results]
        
        print(f"\n📊 Summary ({len(results)} valid configs):")
        print(f"  Avg return: {np.mean(returns):.2f}%")
        print(f"  Median return: {np.median(returns):.2f}%")
        print(f"  % profitable: {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}%")
        print(f"  Avg premium uplift (skew vs flat): {np.mean(uplifts):.1f}%")
        print(f"  Median premium uplift: {np.median(uplifts):.1f}%")
    
    # Save results
    output = {
        "meta": {
            "model": "iv_skew_adjusted",
            "total_combos": total,
            "valid_results": len(results),
            "elapsed_sec": round(elapsed, 1),
            "speed_per_sec": round(actual_speed, 0),
            "trading_days": len(dates),
            "date_range": f"{dates[0]} → {dates[-1]}",
            "skew_params": skew_params,
        },
        "top_20": results[:20],
        "all_results": results,
    }
    
    with open(RESULTS_DIR / "skew_optimizer_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    # Comparison chart: skew vs flat
    if results:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Return distribution: skew vs flat
        skew_returns = [r["total_return_pct"] for r in results]
        axes[0, 0].hist(skew_returns, bins=50, alpha=0.7, color="steelblue", 
                        edgecolor="black", label="Skew-Adjusted")
        axes[0, 0].set_title(f"Return Distribution — Skew Model ({len(results)} configs)")
        axes[0, 0].set_xlabel("Total Return %")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Premium uplift
        axes[0, 1].hist(uplifts, bins=50, alpha=0.7, color="orange", edgecolor="black")
        axes[0, 1].axvline(x=0, color="red", linestyle="--", alpha=0.5)
        axes[0, 1].set_title("Premium Uplift: Skew vs Flat IV")
        axes[0, 1].set_xlabel("Premium Change %")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Win rate vs return
        wr = [r["win_rate"] for r in results]
        axes[1, 0].scatter(wr, skew_returns, alpha=0.3, s=10, c="steelblue")
        axes[1, 0].set_title("Win Rate vs Return (Skew Model)")
        axes[1, 0].set_xlabel("Win Rate %")
        axes[1, 0].set_ylabel("Total Return %")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Drawdown vs return
        dds = [abs(r["max_dd_pct"]) for r in results]
        axes[1, 1].scatter(dds, skew_returns, alpha=0.3, s=10, c="steelblue")
        axes[1, 1].set_title("Max Drawdown vs Return (Skew Model)")
        axes[1, 1].set_xlabel("Max Drawdown %")
        axes[1, 1].set_ylabel("Total Return %")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f"IV Skew-Adjusted 1DTE Optimizer — {len(dates)} days, {total} configs",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "skew_optimizer_charts.png", dpi=150)
        plt.close()
        print(f"\n📊 Charts saved to {RESULTS_DIR}/skew_optimizer_charts.png")
    
    print(f"\n✅ All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

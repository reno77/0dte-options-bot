"""
Risk Analysis & Hardening for 1DTE Iron Condor Backtest

Addresses Grok's valid criticisms:
1. Slippage modeling — 0.5-1 tick per leg, sensitivity analysis
2. Rolling risk metrics — rolling Sharpe, Calmar, underwater equity curve
3. Tail analysis — worst days, max consecutive losses, CVaR, fat tail stats

Runs on Windows NUC (192.168.0.143) or any machine with Python 3.10+.
Reads existing backtest data (SPY 1h bars) and IV skew params from results/.

Usage:
    python risk_analysis.py
"""

import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit, prange
from scipy import stats as sp_stats

# Try plotly for interactive charts, fall back to matplotlib
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR.mkdir(exist_ok=True)


# ==================== Numba BS Primitives ====================

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
def skew_iv(moneyness, dte_days, a, b, c, d, vix_base):
    m = moneyness
    T = max(dte_days, 0.5)
    iv = a + b * (1.0 - m) + c * (1.0 - m) ** 2 + d / np.sqrt(T)
    iv = iv * (vix_base / 15.0)
    return max(min(iv, 3.0), 0.05)


@njit(cache=True)
def bid_ask_spread_pct(moneyness, spread_a, spread_b):
    return spread_a + spread_b * abs(1.0 - moneyness)


@njit(cache=True)
def find_strike_by_delta_skew(
    S, T_years, r, vix,
    target_delta, strike_interval, is_call,
    skew_a, skew_b, skew_c, skew_d,
    spread_a, spread_b,
    dte_days,
    search_range_pct=0.08
):
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
            prem_mid = bs_price(S, strike, T_years, r, sigma, is_call)
            best_prem_mid = prem_mid
            spread = bid_ask_spread_pct(m, spread_a, spread_b) / 100.0
            best_prem_bid = prem_mid * (1.0 - spread / 2.0)
        if is_call:
            strike += strike_interval
        else:
            strike -= strike_interval
    return best_strike, best_prem_mid, best_prem_bid, best_delta, best_iv


# ==================== Enhanced Backtest with Daily P&L ====================

@njit(cache=True)
def run_ic_backtest_daily_pnl(
    prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
    delta_min, delta_max, spread_width, stop_loss_mult,
    entry_bar_start, entry_bar_end, entry_interval, max_entries, strike_interval,
    put_skew_a, put_skew_b, put_skew_c, put_skew_d,
    put_spread_a, put_spread_b,
    call_skew_a, call_skew_b, call_skew_c, call_skew_d,
    call_spread_a, call_spread_b,
    slippage_per_leg=0.0,  # Additional slippage in $ per leg
    r=0.05,
):
    """
    Enhanced backtest that returns daily P&L array + per-trade details.
    slippage_per_leg: extra $ slippage per option leg (4 legs per IC).
    """
    N = len(prices)
    num_days = int(day_indices[-1]) + 1

    equity = 100000.0
    peak_equity = equity
    max_dd = 0.0

    # Daily tracking
    daily_pnl = np.zeros(num_days)
    daily_equity = np.zeros(num_days)
    daily_num_trades = np.zeros(num_days, dtype=np.int32)
    daily_num_wins = np.zeros(num_days, dtype=np.int32)
    daily_num_losses = np.zeros(num_days, dtype=np.int32)
    daily_vix = np.zeros(num_days)

    # Per-trade tracking (max 10K trades)
    MAX_TRADES = 10000
    trade_pnl = np.zeros(MAX_TRADES)
    trade_day = np.zeros(MAX_TRADES, dtype=np.int32)
    trade_premium = np.zeros(MAX_TRADES)
    trade_is_win = np.zeros(MAX_TRADES, dtype=np.int32)
    trade_idx = 0

    total_pnl = 0.0
    num_trades = 0
    num_wins = 0
    num_losses = 0
    total_premium = 0.0

    MAX_POS = 10
    pos_active = np.zeros(MAX_POS, dtype=np.int32)
    pos_short_call = np.zeros(MAX_POS)
    pos_long_call = np.zeros(MAX_POS)
    pos_short_put = np.zeros(MAX_POS)
    pos_long_put = np.zeros(MAX_POS)
    pos_premium = np.zeros(MAX_POS)
    pos_stop = np.zeros(MAX_POS)
    pos_expiry_day = np.zeros(MAX_POS, dtype=np.int32)

    target_delta = (delta_min + delta_max) / 2.0

    prev_day = -1

    for i in range(N):
        day_idx = int(day_indices[i])
        bar_idx = int(bar_in_day[i])
        n_bars = int(bars_per_day[day_idx]) if day_idx < len(bars_per_day) else 390

        S = prices[i]
        mins_remaining = max(float(n_bars - bar_idx), 0.0)

        vix = vix_per_day[day_idx] if day_idx < len(vix_per_day) else 15.0

        # Track daily VIX
        if day_idx < num_days:
            daily_vix[day_idx] = vix

        # Record equity at start of new day
        if day_idx != prev_day:
            if prev_day >= 0 and prev_day < num_days:
                daily_equity[prev_day] = equity
            prev_day = day_idx

        # ---- Check expiring positions ----
        for p in range(MAX_POS):
            if pos_active[p] == 0:
                continue
            if pos_expiry_day[p] != day_idx:
                continue

            T_rem = mins_remaining / (252.0 * 6.5 * 60.0)
            dte_rem = mins_remaining / (6.5 * 60.0)

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

            # Stop loss check
            if total_val > prem + pos_stop[p]:
                pnl = (-pos_stop[p] - slippage_per_leg * 4) * 100.0
                equity += pnl
                total_pnl += pnl
                num_trades += 1
                num_losses += 1
                if day_idx < num_days:
                    daily_pnl[day_idx] += pnl
                    daily_num_trades[day_idx] += 1
                    daily_num_losses[day_idx] += 1
                if trade_idx < MAX_TRADES:
                    trade_pnl[trade_idx] = pnl
                    trade_day[trade_idx] = day_idx
                    trade_premium[trade_idx] = prem * 100.0
                    trade_is_win[trade_idx] = 0
                    trade_idx += 1
                pos_active[p] = 0

            elif mins_remaining <= 1.0:
                pnl = (prem - max(total_val, 0.0) - slippage_per_leg * 4) * 100.0
                equity += pnl
                total_pnl += pnl
                num_trades += 1
                if pnl > 0:
                    num_wins += 1
                    if day_idx < num_days:
                        daily_num_wins[day_idx] += 1
                else:
                    num_losses += 1
                    if day_idx < num_days:
                        daily_num_losses[day_idx] += 1
                if day_idx < num_days:
                    daily_pnl[day_idx] += pnl
                    daily_num_trades[day_idx] += 1
                if trade_idx < MAX_TRADES:
                    trade_pnl[trade_idx] = pnl
                    trade_day[trade_idx] = day_idx
                    trade_premium[trade_idx] = prem * 100.0
                    trade_is_win[trade_idx] = 1 if pnl > 0 else 0
                    trade_idx += 1
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

        T_1dte = (mins_remaining + 6.5 * 60.0) / (252.0 * 6.5 * 60.0)
        dte_1dte = (mins_remaining + 6.5 * 60.0) / (6.5 * 60.0)

        sc_strike, sc_prem_mid, sc_prem_bid, sc_delta, sc_iv = find_strike_by_delta_skew(
            S, T_1dte, r, vix, target_delta, strike_interval, True,
            call_skew_a, call_skew_b, call_skew_c, call_skew_d,
            call_spread_a, call_spread_b, dte_1dte
        )
        if sc_strike < 0 or abs(sc_delta) < delta_min or abs(sc_delta) > delta_max:
            continue

        sp_strike, sp_prem_mid, sp_prem_bid, sp_delta, sp_iv = find_strike_by_delta_skew(
            S, T_1dte, r, vix, target_delta, strike_interval, False,
            put_skew_a, put_skew_b, put_skew_c, put_skew_d,
            put_spread_a, put_spread_b, dte_1dte
        )
        if sp_strike < 0 or abs(sp_delta) < delta_min or abs(sp_delta) > delta_max:
            continue

        lc_strike = sc_strike + spread_width
        lp_strike = sp_strike - spread_width
        lc_m = lc_strike / S
        lp_m = lp_strike / S
        lc_iv = skew_iv(lc_m, dte_1dte, call_skew_a, call_skew_b, call_skew_c, call_skew_d, vix)
        lp_iv = skew_iv(lp_m, dte_1dte, put_skew_a, put_skew_b, put_skew_c, put_skew_d, vix)

        lc_prem = bs_price(S, lc_strike, T_1dte, r, lc_iv, True)
        lp_prem = bs_price(S, lp_strike, T_1dte, r, lp_iv, False)

        # Bid for shorts, ask for longs
        sc_prem = sc_prem_bid
        sp_prem = sp_prem_bid
        lc_spread = bid_ask_spread_pct(lc_m, call_spread_a, call_spread_b) / 100.0
        lp_spread = bid_ask_spread_pct(lp_m, put_spread_a, put_spread_b) / 100.0
        lc_prem_ask = lc_prem * (1.0 + lc_spread / 2.0)
        lp_prem_ask = lp_prem * (1.0 + lp_spread / 2.0)

        call_credit = sc_prem - lc_prem_ask - slippage_per_leg * 2
        put_credit = sp_prem - lp_prem_ask - slippage_per_leg * 2

        if call_credit <= 0.05 or put_credit <= 0.05:
            continue

        ic_premium = call_credit + put_credit

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
            pnl = (pos_premium[p] * 0.5 - slippage_per_leg * 4) * 100.0
            equity += pnl
            total_pnl += pnl
            num_trades += 1
            num_wins += 1
            pos_active[p] = 0

    # Final day equity
    if prev_day >= 0 and prev_day < num_days:
        daily_equity[prev_day] = equity

    # Forward-fill equity for non-trading days
    for d in range(1, num_days):
        if daily_equity[d] == 0.0:
            daily_equity[d] = daily_equity[d - 1]
    if daily_equity[0] == 0.0:
        daily_equity[0] = 100000.0

    return (
        daily_pnl, daily_equity, daily_num_trades, daily_num_wins, daily_num_losses,
        daily_vix,
        trade_pnl[:trade_idx], trade_day[:trade_idx], trade_premium[:trade_idx],
        trade_is_win[:trade_idx],
        total_pnl, num_trades, num_wins, num_losses, max_dd * 100.0, equity
    )


# ==================== Data Loading ====================

def load_data():
    """Load SPY 1h bars and VIX data, prepare arrays for backtest."""
    # Find data files
    spy_files = sorted(DATA_DIR.glob("SPY*1h*.csv"))
    if not spy_files:
        # Try to load from the multiyear optimizer path
        spy_files = sorted(DATA_DIR.glob("SPY*.csv"))

    if not spy_files:
        print("ERROR: No SPY data files found in", DATA_DIR)
        print("  Run data_fetcher.py first, or ensure SPY_intraday_1h_730d.csv exists")
        sys.exit(1)

    spy_file = spy_files[0]
    print(f"Loading: {spy_file}")
    df = pd.read_csv(spy_file, index_col=0, parse_dates=True)

    # Ensure columns exist
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            # Try title case
            if col.title() in df.columns:
                df.rename(columns={col.title(): col}, inplace=True)

    # Handle timezone-aware index
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_convert('America/New_York')
    
    # Filter to market hours (if datetime index)
    if hasattr(df.index, 'hour'):
        df = df[(df.index.hour >= 9) & (df.index.hour < 16)]

    # Assign day indices based on calendar date
    if hasattr(df.index, 'date'):
        dates = pd.Series(df.index.date, index=df.index)
    else:
        # Fallback: treat every N bars as a day (7 bars per day for 1h data)
        n_bars_approx = 7  # ~7 hours of trading per day in 1h bars
        dates = pd.Series([i // n_bars_approx for i in range(len(df))])

    unique_dates = dates.unique()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    day_indices = dates.map(date_to_idx).values.astype(np.float64)
    num_days = len(unique_dates)

    # Bar within day
    bar_in_day = np.zeros(len(df), dtype=np.float64)
    for d_idx in range(num_days):
        mask = day_indices == d_idx
        n = mask.sum()
        bar_in_day[mask] = np.arange(n, dtype=np.float64)

    bars_per_day = np.array([
        (day_indices == d).sum() for d in range(num_days)
    ], dtype=np.float64)

    prices = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)

    # Load VIX
    vix_files = sorted(DATA_DIR.glob("VIX*.csv"))
    if vix_files:
        vix_df = pd.read_csv(vix_files[0], index_col=0, parse_dates=True)
        # Normalize VIX index to tz-naive dates
        if hasattr(vix_df.index, 'tz') and vix_df.index.tz is not None:
            vix_df.index = vix_df.index.tz_localize(None)
        vix_col = 'close' if 'close' in vix_df.columns else 'Close'
        # Build a dict of date -> vix close
        vix_dict = {}
        for idx_val, row in vix_df.iterrows():
            vix_dict[idx_val.date() if hasattr(idx_val, 'date') else idx_val] = row[vix_col]
        
        vix_per_day = np.full(num_days, 15.0)
        last_vix = 15.0
        for i, d in enumerate(unique_dates):
            if d in vix_dict:
                vix_per_day[i] = vix_dict[d]
                last_vix = vix_dict[d]
            else:
                vix_per_day[i] = last_vix
    else:
        print("WARNING: No VIX data, using default 15.0")
        vix_per_day = np.full(num_days, 15.0)

    print(f"Loaded {len(df)} bars, {num_days} trading days")
    print(f"SPY range: ${prices.min():.0f} - ${prices.max():.0f}")
    print(f"VIX range: {vix_per_day.min():.1f} - {vix_per_day.max():.1f}")

    return (prices, highs, lows, day_indices, bar_in_day, bars_per_day,
            vix_per_day, num_days, unique_dates)


def load_skew_params():
    """Load IV skew params from results."""
    params_file = RESULTS_DIR / "iv_skew_params.json"
    if not params_file.exists():
        print(f"WARNING: {params_file} not found, using defaults")
        return {
            "put": {"a": 0.083, "b": 1.606, "c": -0.758, "d": 0.161},
            "call": {"a": 0.138, "b": 0.153, "c": 1.972, "d": -0.008},
            "put_spread": {"a": 0.921, "b": 55.264},
            "call_spread": {"a": 6.213, "b": 6.615},
        }
    with open(params_file) as f:
        return json.load(f)


# ==================== Analysis Functions ====================

def compute_rolling_metrics(daily_pnl, daily_equity, window=30):
    """Compute rolling Sharpe, Calmar, and drawdown."""
    n = len(daily_pnl)

    # Daily returns (percentage)
    daily_returns = np.zeros(n)
    for i in range(1, n):
        if daily_equity[i - 1] > 0:
            daily_returns[i] = daily_pnl[i] / daily_equity[i - 1]

    # Rolling Sharpe (annualized)
    rolling_sharpe = np.full(n, np.nan)
    for i in range(window, n):
        chunk = daily_returns[i - window:i]
        mu = np.mean(chunk)
        sigma = np.std(chunk)
        if sigma > 1e-10:
            rolling_sharpe[i] = mu / sigma * np.sqrt(252)

    # Running drawdown
    running_dd = np.zeros(n)
    peak = daily_equity[0]
    for i in range(n):
        if daily_equity[i] > peak:
            peak = daily_equity[i]
        if peak > 0:
            running_dd[i] = (daily_equity[i] - peak) / peak * 100

    # Rolling Calmar (annualized return / max DD in window)
    rolling_calmar = np.full(n, np.nan)
    for i in range(window, n):
        chunk_ret = daily_returns[i - window:i]
        ann_ret = np.mean(chunk_ret) * 252
        max_dd_in_window = abs(np.min(running_dd[i - window:i]))
        if max_dd_in_window > 0.001:
            rolling_calmar[i] = ann_ret / (max_dd_in_window / 100)

    return daily_returns, rolling_sharpe, running_dd, rolling_calmar


def compute_tail_stats(trade_pnl, daily_pnl, daily_equity):
    """Compute tail risk statistics."""
    results = {}

    # Filter to trading days (non-zero P&L)
    active_days = daily_pnl[daily_pnl != 0]
    daily_returns = []
    for i in range(1, len(daily_equity)):
        if daily_equity[i - 1] > 0 and daily_pnl[i] != 0:
            daily_returns.append(daily_pnl[i] / daily_equity[i - 1])
    daily_returns = np.array(daily_returns) if daily_returns else np.array([0.0])

    # Worst N days
    sorted_daily = np.sort(active_days)
    results['worst_5_days'] = sorted_daily[:5].tolist()
    results['worst_10_days'] = sorted_daily[:10].tolist()
    results['best_5_days'] = sorted_daily[-5:][::-1].tolist()

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for pnl in trade_pnl:
        if pnl < 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0
    results['max_consecutive_losses'] = int(max_consec)

    # Max consecutive wins
    max_consec_w = 0
    current_consec_w = 0
    for pnl in trade_pnl:
        if pnl > 0:
            current_consec_w += 1
            max_consec_w = max(max_consec_w, current_consec_w)
        else:
            current_consec_w = 0
    results['max_consecutive_wins'] = int(max_consec_w)

    # VaR and CVaR (Expected Shortfall)
    if len(daily_returns) > 10:
        sorted_ret = np.sort(daily_returns)
        n = len(sorted_ret)
        idx_5 = max(int(n * 0.05), 1)
        idx_1 = max(int(n * 0.01), 1)

        results['var_95'] = float(sorted_ret[idx_5]) * 100  # as %
        results['var_99'] = float(sorted_ret[idx_1]) * 100
        results['cvar_95'] = float(np.mean(sorted_ret[:idx_5])) * 100  # Expected shortfall
        results['cvar_99'] = float(np.mean(sorted_ret[:idx_1])) * 100
    else:
        results['var_95'] = results['var_99'] = 0.0
        results['cvar_95'] = results['cvar_99'] = 0.0

    # Skewness and Kurtosis of returns
    if len(daily_returns) > 20:
        results['skewness'] = float(sp_stats.skew(daily_returns))
        results['kurtosis'] = float(sp_stats.kurtosis(daily_returns))  # excess kurtosis
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = sp_stats.jarque_bera(daily_returns)
        results['jarque_bera_stat'] = float(jb_stat)
        results['jarque_bera_pvalue'] = float(jb_pvalue)
        results['is_normal'] = jb_pvalue > 0.05
    else:
        results['skewness'] = results['kurtosis'] = 0.0
        results['jarque_bera_stat'] = results['jarque_bera_pvalue'] = 0.0
        results['is_normal'] = False

    # Profit factor
    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]
    results['total_win_amount'] = float(np.sum(wins))
    results['total_loss_amount'] = float(np.sum(losses))
    results['profit_factor'] = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 else float('inf')
    results['avg_win'] = float(np.mean(wins)) if len(wins) > 0 else 0.0
    results['avg_loss'] = float(np.mean(losses)) if len(losses) > 0 else 0.0
    results['win_loss_ratio'] = abs(results['avg_win'] / results['avg_loss']) if results['avg_loss'] != 0 else float('inf')

    # Expectancy
    if len(trade_pnl) > 0:
        results['expectancy_per_trade'] = float(np.mean(trade_pnl))
    else:
        results['expectancy_per_trade'] = 0.0

    return results


def run_slippage_sensitivity(data_tuple, skew_params, config, slippage_levels=None):
    """Run backtest at multiple slippage levels to test sensitivity."""
    if slippage_levels is None:
        slippage_levels = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day, num_days, _ = data_tuple
    pp = skew_params['put']
    cp = skew_params['call']
    ps = skew_params['put_spread']
    cs = skew_params['call_spread']

    results = []
    for slip in slippage_levels:
        t0 = time.time()
        out = run_ic_backtest_daily_pnl(
            prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
            config['delta_min'], config['delta_max'],
            config['spread_width'], config['stop_loss_mult'],
            config['entry_bar_start'], config['entry_bar_end'],
            config['entry_interval'], config['max_entries'],
            config.get('strike_interval', 1.0),
            pp['a'], pp['b'], pp['c'], pp['d'], ps['a'], ps['b'],
            cp['a'], cp['b'], cp['c'], cp['d'], cs['a'], cs['b'],
            slippage_per_leg=slip,
        )
        elapsed = time.time() - t0
        total_pnl, n_trades, n_wins, n_losses, max_dd, final_eq = out[10], out[11], out[12], out[13], out[14], out[15]
        wr = n_wins / n_trades * 100 if n_trades > 0 else 0
        ret_pct = (final_eq - 100000) / 100000 * 100

        results.append({
            'slippage_per_leg': slip,
            'total_slippage_per_ic': slip * 4,
            'total_return_pct': round(ret_pct, 2),
            'num_trades': int(n_trades),
            'win_rate': round(wr, 1),
            'max_dd_pct': round(max_dd, 2),
            'final_equity': round(final_eq, 2),
            'total_pnl': round(total_pnl, 2),
            'elapsed_sec': round(elapsed, 2),
        })
        print(f"  Slippage ${slip:.2f}/leg (${slip*4:.2f}/IC): "
              f"Return={ret_pct:.1f}%, WR={wr:.1f}%, DD={max_dd:.2f}%, "
              f"Trades={n_trades}")

    return results


# ==================== Plotting ====================

def plot_comprehensive_analysis(daily_pnl, daily_equity, daily_returns,
                                 rolling_sharpe, running_dd, rolling_calmar,
                                 trade_pnl, trade_is_win, daily_vix,
                                 tail_stats, slippage_results,
                                 unique_dates, config, save_dir=None):
    """Generate comprehensive risk analysis charts."""
    if save_dir is None:
        save_dir = RESULTS_DIR

    n_days = len(daily_pnl)
    x_days = np.arange(n_days)

    # ========== CHART 1: Equity + Drawdown + Rolling Sharpe ==========
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 2, hspace=0.35, wspace=0.3)

    # 1a. Equity curve with underwater plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x_days, daily_equity, 'b-', linewidth=1.2, label='Equity')
    ax1.fill_between(x_days, 100000, daily_equity,
                     where=daily_equity >= 100000, alpha=0.15, color='green')
    ax1.fill_between(x_days, 100000, daily_equity,
                     where=daily_equity < 100000, alpha=0.15, color='red')
    # Highlight worst 5 days
    worst_day_indices = np.argsort(daily_pnl)[:5]
    for wi in worst_day_indices:
        if daily_pnl[wi] < 0:
            ax1.axvline(wi, color='red', alpha=0.3, linewidth=1.5)
    ax1.set_title('Equity Curve (worst 5 days highlighted in red)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # 1b. Underwater (drawdown) curve
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(x_days, running_dd, 0, alpha=0.5, color='red')
    ax2.plot(x_days, running_dd, 'r-', linewidth=0.8)
    ax2.set_title('Underwater Equity Curve (Drawdown %)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # 1c. Rolling Sharpe
    ax3 = fig.add_subplot(gs[2, 0])
    valid_sharpe = ~np.isnan(rolling_sharpe)
    ax3.plot(x_days[valid_sharpe], rolling_sharpe[valid_sharpe], 'b-', linewidth=1)
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(2, color='green', linestyle='--', alpha=0.5, label='Sharpe=2')
    ax3.set_title('Rolling 30-Day Sharpe Ratio (Annualized)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Sharpe')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 1d. Rolling Calmar
    ax4 = fig.add_subplot(gs[2, 1])
    valid_calmar = ~np.isnan(rolling_calmar)
    calmar_capped = np.clip(rolling_calmar[valid_calmar], -50, 200)
    ax4.plot(x_days[valid_calmar], calmar_capped, 'g-', linewidth=1)
    ax4.axhline(1, color='orange', linestyle='--', alpha=0.5, label='Calmar=1')
    ax4.set_title('Rolling 30-Day Calmar Ratio', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Calmar')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 1e. Daily P&L distribution with VaR/CVaR
    ax5 = fig.add_subplot(gs[3, 0])
    active_pnl = daily_pnl[daily_pnl != 0]
    if len(active_pnl) > 0:
        bins = min(80, max(20, len(active_pnl) // 5))
        ax5.hist(active_pnl, bins=bins, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)

        # Mark VaR/CVaR
        active_ret = daily_returns[daily_returns != 0]
        if len(active_ret) > 10:
            var95_val = np.percentile(active_pnl, 5)
            var99_val = np.percentile(active_pnl, 1)
            ax5.axvline(var95_val, color='orange', linewidth=2, linestyle='--', label=f'VaR 95%: ${var95_val:.0f}')
            ax5.axvline(var99_val, color='red', linewidth=2, linestyle='--', label=f'VaR 99%: ${var99_val:.0f}')

        # Overlay normal distribution fit
        mu, sigma = np.mean(active_pnl), np.std(active_pnl)
        if sigma > 0:
            x_norm = np.linspace(active_pnl.min(), active_pnl.max(), 200)
            y_norm = sp_stats.norm.pdf(x_norm, mu, sigma) * len(active_pnl) * (active_pnl.max() - active_pnl.min()) / bins
            ax5.plot(x_norm, y_norm, 'r--', linewidth=1.5, label='Normal fit')

    ax5.set_title('Daily P&L Distribution + VaR', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Daily P&L ($)')
    ax5.set_ylabel('Frequency')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 1f. Per-trade P&L distribution
    ax6 = fig.add_subplot(gs[3, 1])
    if len(trade_pnl) > 0:
        wins = trade_pnl[trade_is_win == 1]
        losses = trade_pnl[trade_is_win == 0]
        all_bins = np.linspace(trade_pnl.min(), trade_pnl.max(), 60)
        if len(wins) > 0:
            ax6.hist(wins, bins=all_bins, alpha=0.6, color='green', label=f'Wins ({len(wins)})', edgecolor='black', linewidth=0.3)
        if len(losses) > 0:
            ax6.hist(losses, bins=all_bins, alpha=0.6, color='red', label=f'Losses ({len(losses)})', edgecolor='black', linewidth=0.3)
    ax6.set_title('Per-Trade P&L Distribution', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Trade P&L ($)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 1g. Slippage sensitivity
    ax7 = fig.add_subplot(gs[4, 0])
    if slippage_results:
        slips = [r['slippage_per_leg'] for r in slippage_results]
        rets = [r['total_return_pct'] for r in slippage_results]
        wrs = [r['win_rate'] for r in slippage_results]

        color1 = 'tab:blue'
        ax7.plot(slips, rets, 'o-', color=color1, linewidth=2, markersize=6)
        ax7.set_xlabel('Slippage per Leg ($)')
        ax7.set_ylabel('Total Return (%)', color=color1)
        ax7.tick_params(axis='y', labelcolor=color1)

        ax7b = ax7.twinx()
        color2 = 'tab:orange'
        ax7b.plot(slips, wrs, 's--', color=color2, linewidth=1.5, markersize=5)
        ax7b.set_ylabel('Win Rate (%)', color=color2)
        ax7b.tick_params(axis='y', labelcolor=color2)

        # Mark breakeven
        for i, ret in enumerate(rets):
            if ret <= 0:
                ax7.axvline(slips[i], color='red', linestyle=':', alpha=0.5)
                ax7.annotate(f'Breakeven ~${slips[i]:.2f}/leg',
                            xy=(slips[i], ret), fontsize=9, color='red')
                break

    ax7.set_title('Slippage Sensitivity Analysis', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # 1h. P&L vs VIX
    ax8 = fig.add_subplot(gs[4, 1])
    active_mask = daily_pnl != 0
    if active_mask.sum() > 10:
        ax8.scatter(daily_vix[active_mask], daily_pnl[active_mask],
                   alpha=0.4, s=15, c=np.where(daily_pnl[active_mask] > 0, 'green', 'red'))
        # Trend line
        z = np.polyfit(daily_vix[active_mask], daily_pnl[active_mask], 1)
        p = np.poly1d(z)
        vix_range = np.linspace(daily_vix[active_mask].min(), daily_vix[active_mask].max(), 50)
        ax8.plot(vix_range, p(vix_range), 'b--', linewidth=1.5, label=f'Trend: {z[0]:.1f}$/VIX pt')
    ax8.set_title('Daily P&L vs VIX Level', fontsize=11, fontweight='bold')
    ax8.set_xlabel('VIX')
    ax8.set_ylabel('Daily P&L ($)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 1i. Monthly returns heatmap (text-based)
    ax9 = fig.add_subplot(gs[5, :])
    # Compute monthly returns
    if unique_dates is not None and len(unique_dates) > 0:
        month_labels = []
        month_returns = []
        current_month = None
        month_pnl = 0.0
        month_start_eq = 100000.0

        for d in range(len(daily_pnl)):
            if d < len(unique_dates):
                m = unique_dates[d].month if hasattr(unique_dates[d], 'month') else (d // 21) + 1
                y = unique_dates[d].year if hasattr(unique_dates[d], 'year') else 2024
                ym = (y, m)
            else:
                ym = (0, 0)

            if current_month is None:
                current_month = ym
                month_start_eq = daily_equity[max(d - 1, 0)]

            if ym != current_month:
                ret = month_pnl / month_start_eq * 100 if month_start_eq > 0 else 0
                month_labels.append(f"{current_month[0]}-{current_month[1]:02d}")
                month_returns.append(ret)
                current_month = ym
                month_pnl = 0.0
                month_start_eq = daily_equity[max(d - 1, 0)]

            month_pnl += daily_pnl[d]

        # Last month
        if month_pnl != 0:
            ret = month_pnl / month_start_eq * 100 if month_start_eq > 0 else 0
            if current_month:
                month_labels.append(f"{current_month[0]}-{current_month[1]:02d}")
                month_returns.append(ret)

        if month_labels:
            colors = ['green' if r > 0 else 'red' for r in month_returns]
            bars = ax9.bar(range(len(month_labels)), month_returns, color=colors, alpha=0.7,
                          edgecolor='black', linewidth=0.5)
            ax9.set_xticks(range(len(month_labels)))
            ax9.set_xticklabels(month_labels, rotation=45, fontsize=8)

            # Annotate bars
            for bar, ret in zip(bars, month_returns):
                ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{ret:.1f}%', ha='center', va='bottom', fontsize=7)

    ax9.set_title('Monthly Returns (%)', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Return (%)')
    ax9.axhline(0, color='black', linewidth=0.5)
    ax9.grid(True, alpha=0.3, axis='y')

    # Overall title
    cfg_str = (f"delta={config['delta_min']}-{config['delta_max']}, "
               f"W=${config['spread_width']}, SL={config['stop_loss_mult']}x, "
               f"Entries={config['max_entries']}/day")
    fig.suptitle(f'1DTE Iron Condor — Comprehensive Risk Analysis\n{cfg_str}',
                 fontsize=14, fontweight='bold', y=0.995)

    save_path = save_dir / "risk_analysis_comprehensive.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Comprehensive chart saved to {save_path}")

    # ========== CHART 2: Tail Statistics Summary ==========
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

    # 2a. QQ plot
    ax = axes2[0, 0]
    active_ret = daily_returns[daily_returns != 0]
    if len(active_ret) > 20:
        (osm, osr), (slope, intercept, r_val) = sp_stats.probplot(active_ret, dist="norm")
        ax.scatter(osm, osr, alpha=0.5, s=15, color='steelblue')
        ax.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2, label=f'R²={r_val**2:.3f}')
    ax.set_title('Q-Q Plot (Returns vs Normal)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2b. Cumulative daily P&L (sorted) — risk curve
    ax = axes2[0, 1]
    active_pnl = daily_pnl[daily_pnl != 0]
    if len(active_pnl) > 0:
        sorted_pnl = np.sort(active_pnl)
        cumulative = np.cumsum(sorted_pnl)
        pct = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl) * 100
        ax.plot(pct, cumulative, 'b-', linewidth=1.5)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        # Mark 5% and 1% tails
        ax.axvline(5, color='orange', linestyle=':', alpha=0.7, label='5th percentile')
        ax.axvline(1, color='red', linestyle=':', alpha=0.7, label='1st percentile')
    ax.set_title('Cumulative P&L from Worst to Best Days', fontsize=11, fontweight='bold')
    ax.set_xlabel('Percentile (%)')
    ax.set_ylabel('Cumulative P&L ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2c. Consecutive losses distribution
    ax = axes2[1, 0]
    if len(trade_pnl) > 0:
        consec_losses = []
        current = 0
        for pnl in trade_pnl:
            if pnl < 0:
                current += 1
            else:
                if current > 0:
                    consec_losses.append(current)
                current = 0
        if current > 0:
            consec_losses.append(current)

        if consec_losses:
            max_cl = max(consec_losses)
            bins = range(1, max_cl + 2)
            ax.hist(consec_losses, bins=bins, alpha=0.7, color='salmon', edgecolor='black', align='left')
            ax.axvline(np.mean(consec_losses), color='blue', linestyle='--',
                      label=f'Mean: {np.mean(consec_losses):.1f}')
    ax.set_title('Consecutive Loss Streaks Distribution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Consecutive Losses')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2d. Tail stats summary table
    ax = axes2[1, 1]
    ax.axis('off')
    stats_text = [
        ['Metric', 'Value'],
        ['-' * 20, '-' * 15],
        ['Skewness', f"{tail_stats.get('skewness', 0):.3f}"],
        ['Excess Kurtosis', f"{tail_stats.get('kurtosis', 0):.3f}"],
        ['Jarque-Bera p-value', f"{tail_stats.get('jarque_bera_pvalue', 0):.4f}"],
        ['Normal Distribution?', 'Yes' if tail_stats.get('is_normal', False) else 'No (fat tails)'],
        ['', ''],
        ['VaR 95% (daily)', f"{tail_stats.get('var_95', 0):.3f}%"],
        ['VaR 99% (daily)', f"{tail_stats.get('var_99', 0):.3f}%"],
        ['CVaR 95% (Exp. Shortfall)', f"{tail_stats.get('cvar_95', 0):.3f}%"],
        ['CVaR 99% (Exp. Shortfall)', f"{tail_stats.get('cvar_99', 0):.3f}%"],
        ['', ''],
        ['Max Consecutive Losses', f"{tail_stats.get('max_consecutive_losses', 0)}"],
        ['Max Consecutive Wins', f"{tail_stats.get('max_consecutive_wins', 0)}"],
        ['Profit Factor', f"{tail_stats.get('profit_factor', 0):.2f}"],
        ['Win/Loss Ratio', f"{tail_stats.get('win_loss_ratio', 0):.2f}"],
        ['Avg Win', f"${tail_stats.get('avg_win', 0):.2f}"],
        ['Avg Loss', f"${tail_stats.get('avg_loss', 0):.2f}"],
        ['Expectancy/Trade', f"${tail_stats.get('expectancy_per_trade', 0):.2f}"],
    ]
    for i, row in enumerate(stats_text):
        y_pos = 0.95 - i * 0.05
        weight = 'bold' if i == 0 else 'normal'
        ax.text(0.1, y_pos, row[0], fontsize=10, fontweight=weight,
                fontfamily='monospace', transform=ax.transAxes)
        ax.text(0.7, y_pos, row[1], fontsize=10, fontweight=weight,
                fontfamily='monospace', transform=ax.transAxes)

    fig2.suptitle('Tail Risk Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path2 = save_dir / "risk_analysis_tails.png"
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Tail analysis chart saved to {save_path2}")

    return save_path, save_path2


def generate_plotly_interactive(daily_pnl, daily_equity, rolling_sharpe,
                                 running_dd, daily_vix, unique_dates,
                                 config, save_dir=None):
    """Generate interactive Plotly HTML dashboard."""
    if not HAS_PLOTLY:
        print("⚠️  Plotly not available, skipping interactive charts")
        return None

    if save_dir is None:
        save_dir = RESULTS_DIR

    n = len(daily_equity)
    # Use date labels if available
    if unique_dates is not None and len(unique_dates) >= n:
        x_labels = [str(d) for d in unique_dates[:n]]
    else:
        x_labels = list(range(n))

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'Equity Curve',
            'Drawdown (%)',
            'Rolling 30-Day Sharpe',
            'Daily P&L ($)'
        ),
        row_heights=[0.35, 0.2, 0.2, 0.25]
    )

    # Equity
    fig.add_trace(go.Scatter(
        x=x_labels, y=daily_equity,
        mode='lines', name='Equity',
        line=dict(color='blue', width=1.5),
        hovertemplate='$%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=x_labels, y=running_dd,
        mode='lines', name='Drawdown',
        fill='tozeroy',
        line=dict(color='red', width=1),
        hovertemplate='%{y:.2f}%<extra></extra>'
    ), row=2, col=1)

    # Rolling Sharpe
    fig.add_trace(go.Scatter(
        x=x_labels, y=rolling_sharpe,
        mode='lines', name='Sharpe',
        line=dict(color='green', width=1),
        hovertemplate='%{y:.2f}<extra></extra>'
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color="gray", row=3, col=1)

    # Daily P&L
    colors = ['green' if p > 0 else 'red' for p in daily_pnl]
    fig.add_trace(go.Bar(
        x=x_labels, y=daily_pnl,
        name='Daily P&L',
        marker_color=colors,
        hovertemplate='$%{y:,.0f}<extra></extra>'
    ), row=4, col=1)

    cfg_str = (f"delta={config['delta_min']}-{config['delta_max']}, "
               f"W=${config['spread_width']}, SL={config['stop_loss_mult']}x, "
               f"Entries={config['max_entries']}/day")

    fig.update_layout(
        title=f'1DTE Iron Condor — Interactive Risk Dashboard<br><sub>{cfg_str}</sub>',
        height=1200,
        showlegend=False,
        template='plotly_white',
        hovermode='x unified'
    )

    save_path = save_dir / "risk_dashboard_interactive.html"
    fig.write_html(str(save_path))
    print(f"📊 Interactive dashboard saved to {save_path}")
    return save_path


# ==================== Main ====================

def main():
    print("=" * 70)
    print("1DTE Iron Condor — Risk Analysis & Hardening")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    data_tuple = load_data()
    prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day, num_days, unique_dates = data_tuple

    # Load skew params
    skew_params = load_skew_params()
    print(f"  IV skew params loaded (put R²={skew_params['put'].get('r2', 'N/A')})")

    # Best config from skew optimizer
    config = {
        'delta_min': 0.12,
        'delta_max': 0.25,
        'spread_width': 15.0,
        'stop_loss_mult': 1.0,
        'entry_bar_start': 2,
        'entry_bar_end': 6,
        'entry_interval': 1,
        'max_entries': 5,
        'strike_interval': 1.0,
    }
    # Also test conservative config
    config_conservative = {
        'delta_min': 0.05,
        'delta_max': 0.15,
        'spread_width': 5.0,
        'stop_loss_mult': 1.0,
        'entry_bar_start': 0,
        'entry_bar_end': 4,
        'entry_interval': 1,
        'max_entries': 2,
        'strike_interval': 1.0,
    }

    pp = skew_params['put']
    cp = skew_params['call']
    ps = skew_params['put_spread']
    cs = skew_params['call_spread']

    for label, cfg in [("AGGRESSIVE", config), ("CONSERVATIVE", config_conservative)]:
        print(f"\n{'=' * 60}")
        print(f"  Config: {label}")
        print(f"  delta={cfg['delta_min']}-{cfg['delta_max']}, W=${cfg['spread_width']}, "
              f"SL={cfg['stop_loss_mult']}x, Entries={cfg['max_entries']}/day")
        print(f"{'=' * 60}")

        # [2/5] Run base backtest with daily P&L
        print(f"\n[2/5] Running base backtest (no extra slippage)...")
        t0 = time.time()
        out = run_ic_backtest_daily_pnl(
            prices, highs, lows, day_indices, bar_in_day, bars_per_day, vix_per_day,
            cfg['delta_min'], cfg['delta_max'],
            cfg['spread_width'], cfg['stop_loss_mult'],
            cfg['entry_bar_start'], cfg['entry_bar_end'],
            cfg['entry_interval'], cfg['max_entries'],
            cfg['strike_interval'],
            pp['a'], pp['b'], pp['c'], pp['d'], ps['a'], ps['b'],
            cp['a'], cp['b'], cp['c'], cp['d'], cs['a'], cs['b'],
            slippage_per_leg=0.0,
        )
        elapsed = time.time() - t0
        print(f"  Backtest completed in {elapsed:.1f}s")

        daily_pnl, daily_equity, daily_num_trades, daily_num_wins, daily_num_losses, \
            daily_vix_arr, trade_pnl, trade_day, trade_premium, trade_is_win, \
            total_pnl, n_trades, n_wins, n_losses, max_dd, final_eq = out

        print(f"\n  Summary:")
        print(f"    Total Return: {(final_eq - 100000) / 1000:.1f}% (${final_eq:,.0f})")
        print(f"    Trades: {n_trades} ({n_wins}W/{n_losses}L, {n_wins/n_trades*100:.1f}% WR)")
        print(f"    Max DD: {max_dd:.2f}%")

        # [3/5] Rolling metrics
        print(f"\n[3/5] Computing rolling risk metrics...")
        daily_returns, rolling_sharpe, running_dd, rolling_calmar = \
            compute_rolling_metrics(daily_pnl, daily_equity, window=30)

        valid_sharpe = rolling_sharpe[~np.isnan(rolling_sharpe)]
        if len(valid_sharpe) > 0:
            print(f"    Sharpe — Mean: {np.mean(valid_sharpe):.1f}, "
                  f"Min: {np.min(valid_sharpe):.1f}, Max: {np.max(valid_sharpe):.1f}")

        # [4/5] Tail analysis
        print(f"\n[4/5] Computing tail risk statistics...")
        tail_stats = compute_tail_stats(trade_pnl, daily_pnl, daily_equity)
        print(f"    Skewness: {tail_stats['skewness']:.3f}")
        print(f"    Excess Kurtosis: {tail_stats['kurtosis']:.3f}")
        print(f"    Normal? {'Yes' if tail_stats['is_normal'] else 'No (fat tails)'}")
        print(f"    VaR 95%: {tail_stats['var_95']:.3f}%  |  CVaR 95%: {tail_stats['cvar_95']:.3f}%")
        print(f"    VaR 99%: {tail_stats['var_99']:.3f}%  |  CVaR 99%: {tail_stats['cvar_99']:.3f}%")
        print(f"    Max consecutive losses: {tail_stats['max_consecutive_losses']}")
        print(f"    Profit factor: {tail_stats['profit_factor']:.2f}")
        print(f"    Avg win: ${tail_stats['avg_win']:.2f}  |  Avg loss: ${tail_stats['avg_loss']:.2f}")
        print(f"    Worst 5 days: {[f'${x:.0f}' for x in tail_stats['worst_5_days']]}")

        # [5/5] Slippage sensitivity
        print(f"\n[5/5] Running slippage sensitivity analysis...")
        slippage_results = run_slippage_sensitivity(data_tuple, skew_params, cfg)

        # Find breakeven slippage
        for sr in slippage_results:
            if sr['total_return_pct'] <= 0:
                print(f"\n  ⚠️  Strategy breaks even at ~${sr['slippage_per_leg']:.2f}/leg slippage")
                break
        else:
            print(f"\n  ✅ Strategy profitable even at ${slippage_results[-1]['slippage_per_leg']:.2f}/leg slippage!")

        # Generate charts
        print(f"\n  Generating charts...")
        suffix = label.lower()
        save_dir = RESULTS_DIR

        # Override save names per config
        orig_names = {}
        chart1, chart2 = plot_comprehensive_analysis(
            daily_pnl, daily_equity, daily_returns,
            rolling_sharpe, running_dd, rolling_calmar,
            trade_pnl, trade_is_win, daily_vix_arr,
            tail_stats, slippage_results,
            unique_dates, cfg, save_dir=save_dir
        )

        # Rename files with suffix
        for src_name, dst_name in [
            ("risk_analysis_comprehensive.png", f"risk_analysis_comprehensive_{suffix}.png"),
            ("risk_analysis_tails.png", f"risk_analysis_tails_{suffix}.png"),
        ]:
            src = save_dir / src_name
            dst = save_dir / dst_name
            if src.exists():
                src.replace(dst)  # .replace() works on Windows (overwrites)
                print(f"  -> {dst.name}")

        # Interactive HTML
        html_path = generate_plotly_interactive(
            daily_pnl, daily_equity, rolling_sharpe,
            running_dd, daily_vix_arr, unique_dates,
            cfg, save_dir=save_dir
        )
        if html_path and html_path.exists():
            dst = save_dir / f"risk_dashboard_{suffix}.html"
            html_path.replace(dst)
            print(f"  -> {dst.name}")

        # Save tail stats JSON (convert numpy types)
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        stats_path = save_dir / f"tail_stats_{suffix}.json"
        with open(stats_path, 'w') as f:
            json.dump(make_serializable({
                'config': cfg,
                'tail_stats': tail_stats,
                'slippage_sensitivity': slippage_results,
                'summary': {
                    'total_return_pct': round((final_eq - 100000) / 1000, 2),
                    'num_trades': int(n_trades),
                    'win_rate': round(n_wins / n_trades * 100, 1) if n_trades > 0 else 0,
                    'max_dd_pct': round(max_dd, 2),
                    'final_equity': round(final_eq, 2),
                }
            }), f, indent=2)
        print(f"  → {stats_path.name}")

    print("\n" + "=" * 70)
    print("✅ Risk analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

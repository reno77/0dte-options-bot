"""
1DTE Options Backtest Runner

Adapted from 0DTE strategies for 1-day-to-expiration options.
Key differences from 0DTE:
- Options have ~1 full trading day of life (bought day before expiry)
- More theta premium to collect
- Wider stop losses needed (more time for adverse moves)
- Entry is typically at market open or specific time, held overnight to next day close

Strategies:
1. MEIC Iron Condor — sell iron condors with 1DTE, manage throughout the day
2. ORB Credit Spread — use opening range to pick direction, sell 1DTE spreads
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Ensure we can import from backtest directory
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data_fetcher import (
    fetch_underlying_intraday, fetch_underlying_daily, fetch_vix,
    bs_price, bs_delta, generate_option_chain, estimate_intraday_iv,
    DATA_DIR
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ==================== 1DTE Iron Condor ====================

class OneDTEIronCondorBacktest:
    """
    1DTE Iron Condor Strategy
    
    - Enter iron condor positions at a specific time each day
    - Options expire the NEXT trading day at close
    - Hold overnight (key difference from 0DTE)
    - More premium but more risk from overnight gaps
    
    Parameters tuned for 1DTE:
    - Wider delta range (8-20 vs 5-15 for 0DTE) — more premium needed for overnight risk
    - Wider stop losses (1.5x premium vs 1x for 0DTE)
    - Spread width $10 (vs $5 for 0DTE) — more room for 1-day moves
    """
    
    def __init__(
        self,
        delta_min: float = 8,
        delta_max: float = 20,
        spread_width: float = 10.0,
        stop_loss_multiplier: float = 1.5,
        entry_hour: float = 10.5,  # 10:30 AM ET
        max_entries_per_day: int = 3,
        entry_interval_min: int = 45,
        strike_interval: float = 5.0,
    ):
        self.delta_min = delta_min / 100
        self.delta_max = delta_max / 100
        self.spread_width = spread_width
        self.stop_loss_mult = stop_loss_multiplier
        self.entry_hour = entry_hour
        self.max_entries = max_entries_per_day
        self.entry_interval = entry_interval_min
        self.strike_interval = strike_interval
        self.trades = []
        self.daily_results = []
    
    def run(self, intraday_data: pd.DataFrame, vix_data: pd.DataFrame, 
            account_size: float = 100000) -> dict:
        """
        Run 1DTE iron condor backtest.
        
        For each trading day:
        1. At entry time, sell iron condor expiring NEXT day
        2. Monitor throughout current day and next day
        3. Apply stop losses, take profits, or let expire
        """
        self.trades = []
        self.daily_results = []
        equity = account_size
        
        intraday_data = intraday_data.copy()
        intraday_data["date"] = intraday_data.index.date
        trading_days = sorted(intraday_data["date"].unique())
        
        open_positions = []  # Carry overnight
        
        for day_idx, day in enumerate(trading_days):
            day_data = intraday_data[intraday_data["date"] == day].sort_index()
            if len(day_data) < 60:
                continue
            
            # Get VIX
            vix_close = 20.0
            if vix_data is not None and not vix_data.empty:
                prev = [d for d in vix_data.index.date if d <= day]
                if prev:
                    vix_close = vix_data.loc[vix_data.index.date == prev[-1], "close"].iloc[-1]
            
            market_open = day_data.index[0]
            market_close = day_data.index[-1]
            total_min = (market_close - market_open).total_seconds() / 60
            
            day_pnl = 0
            entries_today = 0
            last_entry_time = None
            
            for timestamp, bar in day_data.iterrows():
                minutes_elapsed = (timestamp - market_open).total_seconds() / 60
                minutes_remaining = max(total_min - minutes_elapsed, 0)
                current_price = bar["close"]
                
                # For EXISTING positions (entered yesterday, expiring today):
                # T_remaining = minutes left today / total minutes in a year
                hours_remaining = minutes_remaining / 60
                
                closed_positions = []
                for pos in open_positions:
                    if pos["expiry_day"] == day:
                        # This position expires today
                        T_rem = minutes_remaining / (252 * 6.5 * 60)
                        sigma = estimate_intraday_iv(vix_close, hours_remaining)
                        
                        # Evaluate each side
                        call_val = (
                            bs_price(current_price, pos["short_call"], T_rem, 0.05, sigma, "call") -
                            bs_price(current_price, pos["long_call"], T_rem, 0.05, sigma, "call")
                        )
                        put_val = (
                            bs_price(current_price, pos["short_put"], T_rem, 0.05, sigma, "put") -
                            bs_price(current_price, pos["long_put"], T_rem, 0.05, sigma, "put")
                        )
                        
                        total_current_value = call_val + put_val
                        total_premium = pos["total_premium"]
                        stop_loss = total_premium * self.stop_loss_mult
                        
                        # Check stop loss (total position)
                        if total_current_value > total_premium + stop_loss:
                            pnl = -stop_loss
                            pos["status"] = "stopped"
                            pos["pnl"] = pnl
                            pos["exit_time"] = str(timestamp)
                            day_pnl += pnl
                            self.trades.append(pos)
                            closed_positions.append(pos)
                        elif minutes_remaining <= 1:
                            # Expiration
                            pnl = total_premium - max(total_current_value, 0)
                            pos["status"] = "expired"
                            pos["pnl"] = pnl
                            pos["exit_time"] = str(timestamp)
                            day_pnl += pnl
                            self.trades.append(pos)
                            closed_positions.append(pos)
                    
                    elif pos["expiry_day"] < day:
                        # Shouldn't happen but clean up
                        closed_positions.append(pos)
                
                for cp in closed_positions:
                    if cp in open_positions:
                        open_positions.remove(cp)
                
                # NEW ENTRIES: enter positions expiring next trading day
                current_hour = 9.5 + minutes_elapsed / 60
                if current_hour < self.entry_hour:
                    continue
                if entries_today >= self.max_entries:
                    continue
                if last_entry_time and (timestamp - last_entry_time).total_seconds() / 60 < self.entry_interval:
                    continue
                
                # Check we have a next trading day
                if day_idx + 1 >= len(trading_days):
                    continue
                next_day = trading_days[day_idx + 1]
                
                # T for 1DTE: approximately 1 trading day + remaining today
                T_1dte = (minutes_remaining + 6.5 * 60) / (252 * 6.5 * 60)
                sigma = estimate_intraday_iv(vix_close, hours_remaining + 6.5)
                
                # Generate chain and find strikes
                chain = generate_option_chain(
                    S=current_price, T=T_1dte, r=0.05, sigma=sigma,
                    strike_range_pct=0.05, strike_interval=self.strike_interval
                )
                
                # Find short call
                calls = chain[chain["type"] == "call"].copy()
                calls["abs_delta"] = calls["delta"].abs()
                valid_calls = calls[
                    (calls["abs_delta"] >= self.delta_min) &
                    (calls["abs_delta"] <= self.delta_max) &
                    (calls["strike"] > current_price)
                ]
                
                # Find short put
                puts = chain[chain["type"] == "put"].copy()
                puts["abs_delta"] = puts["delta"].abs()
                valid_puts = puts[
                    (puts["abs_delta"] >= self.delta_min) &
                    (puts["abs_delta"] <= self.delta_max) &
                    (puts["strike"] < current_price)
                ]
                
                if valid_calls.empty or valid_puts.empty:
                    continue
                
                target = (self.delta_min + self.delta_max) / 2
                short_call_row = valid_calls.iloc[(valid_calls["abs_delta"] - target).abs().argmin()]
                short_put_row = valid_puts.iloc[(valid_puts["abs_delta"] - target).abs().argmin()]
                
                sc = short_call_row["strike"]
                lc = sc + self.spread_width
                sp = short_put_row["strike"]
                lp = sp - self.spread_width
                
                # Price long legs
                lc_price = bs_price(current_price, lc, T_1dte, 0.05, sigma, "call")
                lp_price = bs_price(current_price, lp, T_1dte, 0.05, sigma, "put")
                
                call_credit = short_call_row["price"] - lc_price
                put_credit = short_put_row["price"] - lp_price
                
                if call_credit <= 0.10 or put_credit <= 0.10:
                    continue
                
                total_premium = call_credit + put_credit
                
                position = {
                    "entry_time": str(timestamp),
                    "entry_price": current_price,
                    "short_call": sc,
                    "long_call": lc,
                    "short_put": sp,
                    "long_put": lp,
                    "call_credit": round(call_credit, 2),
                    "put_credit": round(put_credit, 2),
                    "total_premium": round(total_premium, 2),
                    "short_call_delta": round(float(short_call_row["delta"]), 4),
                    "short_put_delta": round(float(short_put_row["delta"]), 4),
                    "expiry_day": next_day,
                    "vix": vix_close,
                    "status": "open",
                    "pnl": 0,
                    "exit_time": None,
                }
                
                open_positions.append(position)
                entries_today += 1
                last_entry_time = timestamp
            
            equity += day_pnl
            self.daily_results.append({
                "date": day,
                "pnl": round(day_pnl, 2),
                "equity": round(equity, 2),
                "open_positions": len(open_positions),
                "vix": vix_close,
            })
        
        # Force-close any remaining open positions
        for pos in open_positions:
            pos["status"] = "force_closed"
            pos["pnl"] = pos["total_premium"] * 0.5  # Assume partial win
            self.trades.append(pos)
        
        return self._stats(account_size)
    
    def _stats(self, initial: float) -> dict:
        if not self.trades:
            return {"error": "No trades"}
        
        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        daily_df = pd.DataFrame(self.daily_results)
        total_ret = (daily_df["equity"].iloc[-1] - initial) / initial
        peak = daily_df["equity"].expanding().max()
        dd = ((daily_df["equity"] - peak) / peak).min()
        
        statuses = {}
        for t in self.trades:
            s = t["status"]
            statuses[s] = statuses.get(s, 0) + 1
        
        return {
            "strategy": "1DTE Iron Condor",
            "total_trades": len(self.trades),
            "total_return_pct": round(total_ret * 100, 2),
            "win_rate_pct": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "win_loss_ratio": round(abs(np.mean(wins) / np.mean(losses)), 2) if losses and wins else 0,
            "max_drawdown_pct": round(dd * 100, 2),
            "profitable_days_pct": round((daily_df["pnl"] > 0).mean() * 100, 1),
            "total_pnl": round(sum(pnls), 2),
            "final_equity": round(daily_df["equity"].iloc[-1], 2),
            "status_breakdown": statuses,
            "avg_premium_collected": round(np.mean([t["total_premium"] for t in self.trades]), 2),
            "avg_vix": round(np.mean([t.get("vix", 20) for t in self.trades]), 1),
        }


# ==================== 1DTE ORB Credit Spread ====================

class OneDTEORBBacktest:
    """
    1DTE Opening Range Breakout Credit Spread
    
    - Wait for opening range (first 60 min)
    - On breakout: sell credit spread on OPPOSITE side, expiring next day
    - More premium than 0DTE but overnight gap risk
    """
    
    def __init__(
        self,
        opening_range_min: int = 60,
        delta_target: float = 12,
        spread_width: float = 10.0,
        stop_loss_multiplier: float = 2.0,
        max_trades: int = 1,
        strike_interval: float = 5.0,
    ):
        self.or_minutes = opening_range_min
        self.delta_target = delta_target / 100
        self.spread_width = spread_width
        self.stop_loss_mult = stop_loss_multiplier
        self.max_trades = max_trades
        self.strike_interval = strike_interval
        self.trades = []
        self.daily_results = []
    
    def run(self, intraday_data: pd.DataFrame, vix_data: pd.DataFrame,
            account_size: float = 100000) -> dict:
        self.trades = []
        self.daily_results = []
        equity = account_size
        
        intraday_data = intraday_data.copy()
        intraday_data["date"] = intraday_data.index.date
        trading_days = sorted(intraday_data["date"].unique())
        
        open_positions = []
        
        for day_idx, day in enumerate(trading_days):
            day_data = intraday_data[intraday_data["date"] == day].sort_index()
            if len(day_data) < self.or_minutes + 30:
                continue
            
            vix_close = 20.0
            if vix_data is not None and not vix_data.empty:
                prev = [d for d in vix_data.index.date if d <= day]
                if prev:
                    vix_close = vix_data.loc[vix_data.index.date == prev[-1], "close"].iloc[-1]
            
            market_open = day_data.index[0]
            market_close = day_data.index[-1]
            total_min = (market_close - market_open).total_seconds() / 60
            
            # Opening range
            or_end = market_open + pd.Timedelta(minutes=self.or_minutes)
            or_data = day_data[day_data.index <= or_end]
            if or_data.empty:
                continue
            range_high = or_data["high"].max()
            range_low = or_data["low"].min()
            
            day_pnl = 0
            entries_today = 0
            breakout_found = False
            
            for timestamp, bar in day_data.iterrows():
                minutes_elapsed = (timestamp - market_open).total_seconds() / 60
                minutes_remaining = max(total_min - minutes_elapsed, 0)
                current_price = bar["close"]
                hours_remaining = minutes_remaining / 60
                
                # Check expiring positions
                closed = []
                for pos in open_positions:
                    if pos["expiry_day"] == day:
                        T_rem = minutes_remaining / (252 * 6.5 * 60)
                        sigma = estimate_intraday_iv(vix_close, hours_remaining)
                        
                        if pos["spread_type"] == "put_spread":
                            val = (
                                bs_price(current_price, pos["short_strike"], T_rem, 0.05, sigma, "put") -
                                bs_price(current_price, pos["long_strike"], T_rem, 0.05, sigma, "put")
                            )
                        else:
                            val = (
                                bs_price(current_price, pos["short_strike"], T_rem, 0.05, sigma, "call") -
                                bs_price(current_price, pos["long_strike"], T_rem, 0.05, sigma, "call")
                            )
                        
                        stop = pos["credit"] * self.stop_loss_mult
                        
                        if val > pos["credit"] + stop:
                            pos["pnl"] = -stop
                            pos["status"] = "stopped"
                            pos["exit_time"] = str(timestamp)
                            day_pnl += pos["pnl"]
                            self.trades.append(pos)
                            closed.append(pos)
                        elif minutes_remaining <= 1:
                            pos["pnl"] = pos["credit"] - max(val, 0)
                            pos["status"] = "expired"
                            pos["exit_time"] = str(timestamp)
                            day_pnl += pos["pnl"]
                            self.trades.append(pos)
                            closed.append(pos)
                
                for c in closed:
                    if c in open_positions:
                        open_positions.remove(c)
                
                # New entries after opening range
                if minutes_elapsed < self.or_minutes + 5:
                    continue
                if entries_today >= self.max_trades or breakout_found:
                    continue
                if day_idx + 1 >= len(trading_days):
                    continue
                
                # Detect breakout
                direction = None
                if bar["close"] > range_high:
                    direction = "bullish"
                elif bar["close"] < range_low:
                    direction = "bearish"
                
                if direction is None:
                    continue
                
                breakout_found = True
                next_day = trading_days[day_idx + 1]
                T_1dte = (minutes_remaining + 6.5 * 60) / (252 * 6.5 * 60)
                sigma = estimate_intraday_iv(vix_close, hours_remaining + 6.5)
                
                chain = generate_option_chain(
                    S=current_price, T=T_1dte, r=0.05, sigma=sigma,
                    strike_range_pct=0.05, strike_interval=self.strike_interval
                )
                
                if direction == "bullish":
                    # Sell put spread (bearish side)
                    opts = chain[chain["type"] == "put"].copy()
                    opts["abs_delta"] = opts["delta"].abs()
                    valid = opts[(opts["abs_delta"] >= self.delta_target * 0.5) &
                                (opts["abs_delta"] <= self.delta_target * 2) &
                                (opts["strike"] < current_price)]
                    spread_type = "put_spread"
                else:
                    opts = chain[chain["type"] == "call"].copy()
                    opts["abs_delta"] = opts["delta"].abs()
                    valid = opts[(opts["abs_delta"] >= self.delta_target * 0.5) &
                                (opts["abs_delta"] <= self.delta_target * 2) &
                                (opts["strike"] > current_price)]
                    spread_type = "call_spread"
                
                if valid.empty:
                    continue
                
                short_row = valid.iloc[(valid["abs_delta"] - self.delta_target).abs().argmin()]
                
                if spread_type == "put_spread":
                    long_strike = short_row["strike"] - self.spread_width
                    long_price = bs_price(current_price, long_strike, T_1dte, 0.05, sigma, "put")
                else:
                    long_strike = short_row["strike"] + self.spread_width
                    long_price = bs_price(current_price, long_strike, T_1dte, 0.05, sigma, "call")
                
                credit = short_row["price"] - long_price
                if credit < 0.30:
                    continue
                
                position = {
                    "entry_time": str(timestamp),
                    "entry_price": current_price,
                    "direction": direction,
                    "spread_type": spread_type,
                    "short_strike": float(short_row["strike"]),
                    "long_strike": float(long_strike),
                    "credit": round(float(credit), 2),
                    "short_delta": round(float(short_row["delta"]), 4),
                    "expiry_day": next_day,
                    "range_high": range_high,
                    "range_low": range_low,
                    "status": "open",
                    "pnl": 0,
                    "exit_time": None,
                }
                
                open_positions.append(position)
                entries_today += 1
            
            equity += day_pnl
            self.daily_results.append({
                "date": day,
                "pnl": round(day_pnl, 2),
                "equity": round(equity, 2),
                "range_width": round(range_high - range_low, 2),
            })
        
        # Force-close remaining
        for pos in open_positions:
            pos["status"] = "force_closed"
            pos["pnl"] = pos["credit"] * 0.5
            self.trades.append(pos)
        
        return self._stats(account_size)
    
    def _stats(self, initial: float) -> dict:
        if not self.trades:
            return {"error": "No trades"}
        
        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        daily_df = pd.DataFrame(self.daily_results)
        total_ret = (daily_df["equity"].iloc[-1] - initial) / initial
        peak = daily_df["equity"].expanding().max()
        dd = ((daily_df["equity"] - peak) / peak).min()
        
        bullish = sum(1 for t in self.trades if t.get("direction") == "bullish")
        bearish = sum(1 for t in self.trades if t.get("direction") == "bearish")
        
        statuses = {}
        for t in self.trades:
            s = t["status"]
            statuses[s] = statuses.get(s, 0) + 1
        
        return {
            "strategy": "1DTE ORB Credit Spread",
            "total_trades": len(self.trades),
            "total_return_pct": round(total_ret * 100, 2),
            "win_rate_pct": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "max_drawdown_pct": round(dd * 100, 2),
            "profitable_days_pct": round((daily_df["pnl"] > 0).mean() * 100, 1),
            "total_pnl": round(sum(pnls), 2),
            "final_equity": round(daily_df["equity"].iloc[-1], 2),
            "bullish_trades": bullish,
            "bearish_trades": bearish,
            "status_breakdown": statuses,
            "avg_credit": round(np.mean([t["credit"] for t in self.trades]), 2),
        }


# ==================== Main Runner ====================

def main():
    print("=" * 70)
    print("🚀 1DTE OPTIONS BACKTEST — Alienware RTX 3090")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # Fetch data
    print("\n📥 Fetching data...")
    
    spy_intraday = fetch_underlying_intraday("SPY", period="1mo", interval="1m")
    spy_daily = fetch_underlying_daily("SPY", years=3)
    vix = fetch_vix("3y")
    
    print(f"\n📊 SPY intraday: {len(spy_intraday)} bars")
    print(f"   Range: {spy_intraday.index[0]} → {spy_intraday.index[-1]}")
    print(f"   Price: ${spy_intraday['close'].iloc[-1]:.2f}")
    print(f"📊 VIX last: {vix['close'].iloc[-1]:.1f}")
    
    # ---- 1DTE Iron Condor ----
    print("\n" + "=" * 70)
    print("📊 Strategy 1: 1DTE Iron Condor (MEIC-style)")
    print("=" * 70)
    
    ic_bt = OneDTEIronCondorBacktest(
        delta_min=8,
        delta_max=20,
        spread_width=10,
        stop_loss_multiplier=1.5,
        entry_hour=10.5,
        max_entries_per_day=3,
        entry_interval_min=45,
        strike_interval=1.0,  # SPY has $1 strikes
    )
    
    ic_stats = ic_bt.run(spy_intraday, vix)
    print(json.dumps(ic_stats, indent=2))
    
    # Save IC results
    if ic_bt.daily_results:
        ic_df = pd.DataFrame(ic_bt.daily_results)
        ic_df.to_csv(RESULTS_DIR / "1dte_ic_daily.csv", index=False)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes[0].plot(range(len(ic_df)), ic_df["equity"], 'b-', linewidth=1.5)
        axes[0].set_title("1DTE Iron Condor — Equity Curve")
        axes[0].set_ylabel("Equity ($)")
        axes[0].grid(True, alpha=0.3)
        
        colors = ['green' if x > 0 else 'red' for x in ic_df["pnl"]]
        axes[1].bar(range(len(ic_df)), ic_df["pnl"], color=colors, alpha=0.7)
        axes[1].set_title("1DTE Iron Condor — Daily P&L")
        axes[1].set_ylabel("P&L ($)")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "1dte_ic_results.png", dpi=150)
        plt.close()
    
    # ---- 1DTE ORB ----
    print("\n" + "=" * 70)
    print("📊 Strategy 2: 1DTE ORB Credit Spread")
    print("=" * 70)
    
    orb_bt = OneDTEORBBacktest(
        opening_range_min=60,
        delta_target=12,
        spread_width=10,
        stop_loss_multiplier=2.0,
        max_trades=1,
        strike_interval=1.0,
    )
    
    orb_stats = orb_bt.run(spy_intraday, vix)
    print(json.dumps(orb_stats, indent=2))
    
    # Save ORB results
    if orb_bt.daily_results:
        orb_df = pd.DataFrame(orb_bt.daily_results)
        orb_df.to_csv(RESULTS_DIR / "1dte_orb_daily.csv", index=False)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes[0].plot(range(len(orb_df)), orb_df["equity"], 'b-', linewidth=1.5)
        axes[0].set_title("1DTE ORB Credit Spread — Equity Curve")
        axes[0].set_ylabel("Equity ($)")
        axes[0].grid(True, alpha=0.3)
        
        colors = ['green' if x > 0 else 'red' for x in orb_df["pnl"]]
        axes[1].bar(range(len(orb_df)), orb_df["pnl"], color=colors, alpha=0.7)
        axes[1].set_title("1DTE ORB — Daily P&L")
        axes[1].set_ylabel("P&L ($)")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "1dte_orb_results.png", dpi=150)
        plt.close()
    
    # ---- Comparison ----
    print("\n" + "=" * 70)
    print("📊 STRATEGY COMPARISON")
    print("=" * 70)
    
    metrics = ["total_trades", "total_return_pct", "win_rate_pct", 
               "avg_win", "avg_loss", "max_drawdown_pct",
               "profitable_days_pct", "total_pnl", "final_equity"]
    
    print(f"{'Metric':<25}{'1DTE IC':>20}{'1DTE ORB':>20}")
    print("-" * 65)
    for m in metrics:
        v1 = ic_stats.get(m, "N/A")
        v2 = orb_stats.get(m, "N/A")
        if isinstance(v1, float): v1 = f"{v1:.2f}"
        if isinstance(v2, float): v2 = f"{v2:.2f}"
        print(f"{m:<25}{str(v1):>20}{str(v2):>20}")
    
    # Save comparison
    with open(RESULTS_DIR / "1dte_comparison.json", "w") as f:
        json.dump({"iron_condor": ic_stats, "orb": orb_stats}, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {RESULTS_DIR}/")
    print("\n⚠️  Note: Uses synthetic Black-Scholes pricing (no real bid/ask spreads).")
    print("   Real results will differ. This gives directional insight on strategy viability.")


if __name__ == "__main__":
    main()

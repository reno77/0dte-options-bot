"""
Main backtesting engine — runs all strategies and compares results.

Usage:
    python backtest_engine.py              # Fetch data + run all strategies
    python backtest_engine.py --no-fetch   # Use cached data only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from data_fetcher import fetch_underlying_intraday, fetch_underlying_daily, fetch_vix, DATA_DIR
from strategies.iron_condor_meic import MEICBacktester, MEICConfig
from strategies.orb_credit_spread import ORBBacktester, ORBConfig

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_or_fetch_data(symbol: str, fetch: bool = True) -> tuple:
    """Load cached data or fetch fresh."""
    intraday_files = list(DATA_DIR.glob(f"{symbol}_intraday_*.parquet"))
    daily_files = list(DATA_DIR.glob(f"{symbol}_daily_*.parquet"))
    vix_files = list(DATA_DIR.glob("VIX_*.parquet"))
    
    if fetch or not intraday_files:
        intraday = fetch_underlying_intraday(symbol, period="30d", interval="1m")
    else:
        intraday = pd.read_parquet(intraday_files[-1])
    
    if fetch or not daily_files:
        daily = fetch_underlying_daily(symbol, years=3)
    else:
        daily = pd.read_parquet(daily_files[-1])
    
    if fetch or not vix_files:
        vix = fetch_vix("3y")
    else:
        vix = pd.read_parquet(vix_files[-1])
    
    return intraday, daily, vix


def run_meic_backtest(intraday: pd.DataFrame, vix: pd.DataFrame, account_size: float = 100000) -> dict:
    """Run MEIC Iron Condor backtest."""
    print("\n" + "=" * 60)
    print("📊 MEIC Breakeven Iron Condor Backtest")
    print("=" * 60)
    
    config = MEICConfig(
        delta_min=5,
        delta_max=15,
        spread_width=5,
        min_entry_interval_min=30,
        max_positions=4,
        entry_start_time="10:30",
        entry_end_time="14:30",
    )
    
    bt = MEICBacktester(config)
    stats = bt.run_backtest(intraday, vix, account_size=account_size, strike_interval=5.0)
    
    print(json.dumps(stats, indent=2))
    
    # Save equity curve
    if bt.daily_pnl:
        daily_df = pd.DataFrame(bt.daily_pnl)
        daily_df.to_csv(RESULTS_DIR / "meic_daily.csv", index=False)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curve
        axes[0].plot(daily_df["date"].apply(str), daily_df["equity"], 'b-', linewidth=1.5)
        axes[0].set_title("MEIC Iron Condor — Equity Curve")
        axes[0].set_ylabel("Equity ($)")
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Daily P&L
        colors = ['green' if x > 0 else 'red' for x in daily_df["pnl"]]
        axes[1].bar(range(len(daily_df)), daily_df["pnl"], color=colors, alpha=0.7)
        axes[1].set_title("MEIC — Daily P&L")
        axes[1].set_ylabel("P&L ($)")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "meic_results.png", dpi=150)
        plt.close()
        print(f"\n📈 Charts saved to {RESULTS_DIR}/meic_results.png")
    
    return stats


def run_orb_backtest(intraday: pd.DataFrame, vix: pd.DataFrame, account_size: float = 100000) -> dict:
    """Run ORB Credit Spread backtest."""
    print("\n" + "=" * 60)
    print("📊 ORB Credit Spread Backtest")
    print("=" * 60)
    
    config = ORBConfig(
        opening_range_minutes=60,
        delta_target=10,
        spread_width=5,
        stop_loss_multiplier=1.5,
        max_trades_per_day=2,
    )
    
    bt = ORBBacktester(config)
    stats = bt.run_backtest(intraday, vix, account_size=account_size)
    
    print(json.dumps(stats, indent=2))
    
    # Save equity curve
    if bt.daily_pnl:
        daily_df = pd.DataFrame(bt.daily_pnl)
        daily_df.to_csv(RESULTS_DIR / "orb_daily.csv", index=False)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curve
        axes[0].plot(daily_df["date"].apply(str), daily_df["equity"], 'b-', linewidth=1.5)
        axes[0].set_title("ORB Credit Spread — Equity Curve")
        axes[0].set_ylabel("Equity ($)")
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Daily P&L with direction coloring
        colors = []
        for _, row in daily_df.iterrows():
            if row["pnl"] > 0:
                colors.append("green")
            elif row["pnl"] < 0:
                colors.append("red")
            else:
                colors.append("gray")
        
        axes[1].bar(range(len(daily_df)), daily_df["pnl"], color=colors, alpha=0.7)
        axes[1].set_title("ORB — Daily P&L")
        axes[1].set_ylabel("P&L ($)")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "orb_results.png", dpi=150)
        plt.close()
        print(f"\n📈 Charts saved to {RESULTS_DIR}/orb_results.png")
    
    return stats


def compare_strategies(results: dict):
    """Print side-by-side comparison of all strategies."""
    print("\n" + "=" * 80)
    print("📊 STRATEGY COMPARISON")
    print("=" * 80)
    
    metrics = [
        "total_trades", "total_return_pct", "win_rate_pct",
        "avg_win", "avg_loss", "max_drawdown_pct",
        "profitable_days_pct", "total_pnl", "final_equity"
    ]
    
    header = f"{'Metric':<25}" + "".join(f"{name:>20}" for name in results.keys())
    print(header)
    print("-" * len(header))
    
    for metric in metrics:
        row = f"{metric:<25}"
        for name, stats in results.items():
            val = stats.get(metric, "N/A")
            if isinstance(val, float):
                row += f"{val:>20.2f}"
            else:
                row += f"{str(val):>20}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="0DTE Options Backtester")
    parser.add_argument("--no-fetch", action="store_true", help="Use cached data only")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol (default: SPY)")
    parser.add_argument("--account-size", type=float, default=100000, help="Starting account size")
    args = parser.parse_args()
    
    print(f"🚀 0DTE Options Backtester")
    print(f"   Symbol: {args.symbol}")
    print(f"   Account: ${args.account_size:,.0f}")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Load data
    intraday, daily, vix = load_or_fetch_data(args.symbol, fetch=not args.no_fetch)
    print(f"\n📅 Intraday data: {len(intraday)} bars ({intraday.index[0]} → {intraday.index[-1]})")
    print(f"📅 Daily data: {len(daily)} bars")
    print(f"📅 VIX data: {len(vix)} bars")
    
    # Run all strategies
    results = {}
    
    results["MEIC"] = run_meic_backtest(intraday, vix, args.account_size)
    results["ORB"] = run_orb_backtest(intraday, vix, args.account_size)
    
    # Compare
    compare_strategies(results)
    
    # Save combined results
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ All results saved to {RESULTS_DIR}/")
    print(f"\n⚠️  Note: These results use synthetic Black-Scholes pricing.")
    print(f"   Real options data (ThetaData, Databento) will give more accurate results.")
    print(f"   Synthetic pricing tends to underestimate bid/ask spreads and IV skew.")


if __name__ == "__main__":
    main()

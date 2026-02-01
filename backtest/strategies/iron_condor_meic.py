"""
Breakeven Iron Condor (MEIC) Strategy

Based on research from thetaprofits.com (5,600+ trades, 34/37 months profitable):
- Sell iron condors at 5-15 delta on both sides
- Equal premium on both sides
- Stop loss per side = total premium collected
- Multiple entries per day, minimum 30 min apart
- Take profit at $0.05 per short leg

Key insight: 39% win rate, but avg win = 2.2x avg loss = positive EV
Biggest risk: double stop-loss (both sides hit in same trade)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

import sys
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from data_fetcher import bs_price, bs_delta, generate_option_chain, estimate_intraday_iv


class TradeStatus(Enum):
    OPEN = "open"
    WIN = "win"  # Both sides expired worthless or hit TP
    LOSS_CALL = "loss_call"  # Call side stopped out, put side wins
    LOSS_PUT = "loss_put"  # Put side stopped out, call side wins
    DOUBLE_STOP = "double_stop"  # Both sides stopped out (worst case)


@dataclass
class IronCondorTrade:
    """Represents a single iron condor position."""
    entry_time: pd.Timestamp
    underlying_price: float
    
    # Short call spread
    short_call_strike: float
    long_call_strike: float
    short_call_premium: float
    long_call_premium: float
    short_call_delta: float
    
    # Short put spread
    short_put_strike: float
    long_put_strike: float
    short_put_premium: float
    long_put_premium: float
    short_put_delta: float
    
    # Combined
    total_premium: float = 0.0
    call_side_premium: float = 0.0
    put_side_premium: float = 0.0
    stop_loss_per_side: float = 0.0
    
    # Result
    status: TradeStatus = TradeStatus.OPEN
    exit_time: Optional[pd.Timestamp] = None
    pnl: float = 0.0
    max_risk: float = 0.0
    
    def __post_init__(self):
        self.call_side_premium = self.short_call_premium - self.long_call_premium
        self.put_side_premium = self.short_put_premium - self.long_put_premium
        self.total_premium = self.call_side_premium + self.put_side_premium
        self.stop_loss_per_side = self.total_premium  # Breakeven IC rule
        # Max risk = spread width - premium collected (per side)
        spread_width = self.long_call_strike - self.short_call_strike
        self.max_risk = spread_width - self.total_premium


@dataclass
class MEICConfig:
    """MEIC strategy configuration."""
    delta_min: float = 5.0  # Minimum delta for short strikes (in delta units, e.g., 5 = 0.05)
    delta_max: float = 15.0  # Maximum delta for short strikes
    spread_width: float = 5.0  # Width of each vertical spread ($5 for SPX)
    min_entry_interval_min: int = 30  # Minimum minutes between entries
    take_profit_cents: float = 0.05  # Close short legs at this price
    max_positions: int = 4  # Max concurrent positions
    entry_start_time: str = "10:30"  # Earliest entry (ET)
    entry_end_time: str = "14:30"  # Latest entry (ET)
    risk_per_trade_pct: float = 0.5  # % of account per trade
    max_daily_risk_pct: float = 2.0  # Max daily risk


class MEICBacktester:
    """
    Backtests the Breakeven Iron Condor (MEIC) strategy.
    
    Uses synthetic options pricing (Black-Scholes) with VIX-derived IV
    when real options data is unavailable.
    """
    
    def __init__(self, config: MEICConfig = None):
        self.config = config or MEICConfig()
        self.trades: List[IronCondorTrade] = []
        self.daily_pnl: List[dict] = []
    
    def select_strikes(
        self,
        S: float,
        T: float,
        sigma: float,
        r: float = 0.05,
        strike_interval: float = 5.0,
    ) -> Optional[Tuple[dict, dict]]:
        """
        Select iron condor strikes based on delta targets.
        
        Returns tuple of (call_spread, put_spread) dicts or None if no valid strikes.
        """
        chain = generate_option_chain(
            S=S, T=T, r=r, sigma=sigma,
            strike_range_pct=0.05,
            strike_interval=strike_interval
        )
        
        delta_min = self.config.delta_min / 100  # Convert to decimal
        delta_max = self.config.delta_max / 100
        width = self.config.spread_width
        
        # Find short call: delta between target range (positive delta for calls)
        calls = chain[chain["type"] == "call"].copy()
        calls["abs_delta"] = calls["delta"].abs()
        valid_short_calls = calls[
            (calls["abs_delta"] >= delta_min) & 
            (calls["abs_delta"] <= delta_max) &
            (calls["strike"] > S)
        ].sort_values("abs_delta")
        
        # Find short put: delta between target range (negative delta for puts)
        puts = chain[chain["type"] == "put"].copy()
        puts["abs_delta"] = puts["delta"].abs()
        valid_short_puts = puts[
            (puts["abs_delta"] >= delta_min) & 
            (puts["abs_delta"] <= delta_max) &
            (puts["strike"] < S)
        ].sort_values("abs_delta")
        
        if valid_short_calls.empty or valid_short_puts.empty:
            return None
        
        # Pick strikes closest to middle of delta range
        target_delta = (delta_min + delta_max) / 2
        
        short_call = valid_short_calls.iloc[
            (valid_short_calls["abs_delta"] - target_delta).abs().argmin()
        ]
        short_put = valid_short_puts.iloc[
            (valid_short_puts["abs_delta"] - target_delta).abs().argmin()
        ]
        
        # Long legs = short ± width
        long_call_strike = short_call["strike"] + width
        long_put_strike = short_put["strike"] - width
        
        # Price the long legs
        long_call_price = bs_price(S, long_call_strike, T, r, sigma, "call")
        long_put_price = bs_price(S, long_put_strike, T, r, sigma, "put")
        
        # Check premium is roughly equal on both sides (within 30%)
        call_credit = short_call["price"] - long_call_price
        put_credit = short_put["price"] - long_put_price
        
        if call_credit <= 0 or put_credit <= 0:
            return None
        
        premium_ratio = min(call_credit, put_credit) / max(call_credit, put_credit)
        if premium_ratio < 0.5:  # Too asymmetric
            return None
        
        call_spread = {
            "short_strike": short_call["strike"],
            "long_strike": long_call_strike,
            "short_premium": short_call["price"],
            "long_premium": long_call_price,
            "short_delta": short_call["delta"],
            "credit": call_credit,
        }
        
        put_spread = {
            "short_strike": short_put["strike"],
            "long_strike": long_put_strike,
            "short_premium": short_put["price"],
            "long_premium": long_put_price,
            "short_delta": short_put["delta"],
            "credit": put_credit,
        }
        
        return (call_spread, put_spread)
    
    def evaluate_position(
        self,
        trade: IronCondorTrade,
        current_price: float,
        current_time: pd.Timestamp,
        T_remaining: float,
        sigma: float,
        r: float = 0.05,
    ) -> IronCondorTrade:
        """
        Evaluate an open iron condor position against current prices.
        Check stop losses, take profit, and expiration.
        """
        if trade.status != TradeStatus.OPEN:
            return trade
        
        # Current value of each spread
        call_spread_value = (
            bs_price(current_price, trade.short_call_strike, T_remaining, r, sigma, "call") -
            bs_price(current_price, trade.long_call_strike, T_remaining, r, sigma, "call")
        )
        put_spread_value = (
            bs_price(current_price, trade.short_put_strike, T_remaining, r, sigma, "put") -
            bs_price(current_price, trade.long_put_strike, T_remaining, r, sigma, "put")
        )
        
        call_pnl = trade.call_side_premium - call_spread_value
        put_pnl = trade.put_side_premium - put_spread_value
        
        # Check stop losses (per side = total premium)
        call_stopped = call_spread_value >= (trade.call_side_premium + trade.stop_loss_per_side)
        put_stopped = put_spread_value >= (trade.put_side_premium + trade.stop_loss_per_side)
        
        if call_stopped and put_stopped:
            trade.status = TradeStatus.DOUBLE_STOP
            trade.pnl = call_pnl + put_pnl
            trade.exit_time = current_time
        elif call_stopped:
            trade.status = TradeStatus.LOSS_CALL
            # Call side stopped, put side continues (approximate: assume put expires worthless)
            trade.pnl = -trade.stop_loss_per_side + trade.put_side_premium
            trade.exit_time = current_time
        elif put_stopped:
            trade.status = TradeStatus.LOSS_PUT
            trade.pnl = trade.call_side_premium + (-trade.stop_loss_per_side)
            trade.exit_time = current_time
        elif T_remaining <= 0:
            # Expiration
            trade.status = TradeStatus.WIN
            trade.pnl = trade.total_premium  # Both sides expire worthless
            trade.exit_time = current_time
        
        return trade
    
    def run_backtest(
        self,
        price_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        account_size: float = 100000,
        strike_interval: float = 5.0,
    ) -> dict:
        """
        Run MEIC backtest over intraday price data.
        
        Args:
            price_data: DataFrame with 1-min OHLCV bars (index = datetime)
            vix_data: DataFrame with daily VIX close
            account_size: Starting account size
            strike_interval: Strike price interval (5 for SPX, 1 for SPY)
        
        Returns:
            Dict with backtest results
        """
        self.trades = []
        self.daily_pnl = []
        equity = account_size
        
        # Group by trading day
        price_data = price_data.copy()
        price_data["date"] = price_data.index.date
        
        trading_days = sorted(price_data["date"].unique())
        
        for day in trading_days:
            day_data = price_data[price_data["date"] == day].sort_index()
            
            if len(day_data) < 60:  # Need at least 1 hour of data
                continue
            
            # Get VIX for IV estimation
            vix_close = 20.0  # default
            if vix_data is not None and not vix_data.empty:
                vix_dates = vix_data.index.date
                prev_days = [d for d in vix_dates if d <= day]
                if prev_days:
                    vix_close = vix_data.loc[vix_data.index.date == prev_days[-1], "close"].iloc[-1]
            
            day_trades = []
            last_entry_time = None
            day_pnl = 0
            
            # Market hours: 9:30 - 16:00 ET
            market_open = day_data.index[0]
            market_close = day_data.index[-1]
            total_minutes = (market_close - market_open).total_seconds() / 60
            
            for i, (timestamp, bar) in enumerate(day_data.iterrows()):
                current_price = bar["close"]
                minutes_elapsed = (timestamp - market_open).total_seconds() / 60
                minutes_remaining = max(total_minutes - minutes_elapsed, 0)
                T_remaining = minutes_remaining / (252 * 6.5 * 60)  # Convert to years
                
                # Estimate IV
                hours_remaining = minutes_remaining / 60
                sigma = estimate_intraday_iv(vix_close, hours_remaining)
                
                # --- Evaluate open positions ---
                for trade in day_trades:
                    if trade.status == TradeStatus.OPEN:
                        self.evaluate_position(
                            trade, current_price, timestamp,
                            T_remaining, sigma
                        )
                        if trade.status != TradeStatus.OPEN:
                            day_pnl += trade.pnl
                
                # --- Entry logic ---
                # Check time window
                hour = minutes_elapsed / 60 + 9.5  # Approximate ET hour
                entry_start = float(self.config.entry_start_time.replace(":", "."))
                entry_end = float(self.config.entry_end_time.replace(":", "."))
                
                if hour < entry_start or hour > entry_end:
                    continue
                
                # Check interval since last entry
                if last_entry_time is not None:
                    mins_since_last = (timestamp - last_entry_time).total_seconds() / 60
                    if mins_since_last < self.config.min_entry_interval_min:
                        continue
                
                # Check max positions
                open_count = sum(1 for t in day_trades if t.status == TradeStatus.OPEN)
                if open_count >= self.config.max_positions:
                    continue
                
                # Try to enter a trade
                result = self.select_strikes(
                    S=current_price,
                    T=T_remaining,
                    sigma=sigma,
                    strike_interval=strike_interval,
                )
                
                if result is None:
                    continue
                
                call_spread, put_spread = result
                
                trade = IronCondorTrade(
                    entry_time=timestamp,
                    underlying_price=current_price,
                    short_call_strike=call_spread["short_strike"],
                    long_call_strike=call_spread["long_strike"],
                    short_call_premium=call_spread["short_premium"],
                    long_call_premium=call_spread["long_premium"],
                    short_call_delta=call_spread["short_delta"],
                    short_put_strike=put_spread["short_strike"],
                    long_put_strike=put_spread["long_strike"],
                    short_put_premium=put_spread["short_premium"],
                    long_put_premium=put_spread["long_premium"],
                    short_put_delta=put_spread["short_delta"],
                )
                
                day_trades.append(trade)
                last_entry_time = timestamp
            
            # Close any remaining open positions at EOD
            for trade in day_trades:
                if trade.status == TradeStatus.OPEN:
                    trade.status = TradeStatus.WIN
                    trade.pnl = trade.total_premium
                    trade.exit_time = day_data.index[-1]
                    day_pnl += trade.pnl
            
            self.trades.extend(day_trades)
            equity += day_pnl
            
            self.daily_pnl.append({
                "date": day,
                "trades": len(day_trades),
                "pnl": day_pnl,
                "equity": equity,
            })
        
        return self._compute_stats(account_size)
    
    def _compute_stats(self, initial_capital: float) -> dict:
        """Compute backtest statistics."""
        if not self.trades:
            return {"error": "No trades"}
        
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        # Status breakdown
        status_counts = {}
        for t in self.trades:
            s = t.status.value
            status_counts[s] = status_counts.get(s, 0) + 1
        
        daily_df = pd.DataFrame(self.daily_pnl)
        
        total_return = (daily_df["equity"].iloc[-1] - initial_capital) / initial_capital
        
        # Max drawdown
        peak = daily_df["equity"].expanding().max()
        drawdown = (daily_df["equity"] - peak) / peak
        max_dd = drawdown.min()
        
        stats = {
            "total_trades": len(self.trades),
            "total_return_pct": round(total_return * 100, 2),
            "win_rate_pct": round(len(wins) / len(pnls) * 100, 1),
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "win_loss_ratio": round(abs(np.mean(wins) / np.mean(losses)), 2) if losses and wins else 0,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_daily_pnl": round(daily_df["pnl"].mean(), 2),
            "profitable_days_pct": round((daily_df["pnl"] > 0).mean() * 100, 1),
            "avg_trades_per_day": round(daily_df["trades"].mean(), 1),
            "status_breakdown": status_counts,
            "total_pnl": round(sum(pnls), 2),
            "final_equity": round(daily_df["equity"].iloc[-1], 2),
        }
        
        return stats


if __name__ == "__main__":
    # Quick test with synthetic data
    config = MEICConfig(
        delta_min=5,
        delta_max=15,
        spread_width=5,
        min_entry_interval_min=30,
        max_positions=4,
    )
    
    bt = MEICBacktester(config)
    
    # Demo strike selection
    result = bt.select_strikes(
        S=6000.0,
        T=4 / (252 * 6.5),  # 4 hours left
        sigma=0.20,
        strike_interval=5.0,
    )
    
    if result:
        call_spread, put_spread = result
        print("=== MEIC Strike Selection Demo ===")
        print(f"Underlying: $6,000")
        print(f"\nCall spread: sell {call_spread['short_strike']} / buy {call_spread['long_strike']}")
        print(f"  Credit: ${call_spread['credit']:.2f} | Delta: {call_spread['short_delta']:.4f}")
        print(f"\nPut spread: sell {put_spread['short_strike']} / buy {put_spread['long_strike']}")
        print(f"  Credit: ${put_spread['credit']:.2f} | Delta: {put_spread['short_delta']:.4f}")
        print(f"\nTotal premium: ${call_spread['credit'] + put_spread['credit']:.2f}")
        print(f"Stop loss per side: ${call_spread['credit'] + put_spread['credit']:.2f}")
    else:
        print("No valid strikes found")

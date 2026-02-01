"""
Opening Range Breakout (ORB) Credit Spread Strategy

Based on research from r/algotrading and OptionAlpha:
- Define opening range during first hour (9:30-10:30 AM ET)
- When price breaks out above range → sell put credit spread (bullish)
- When price breaks out below range → sell call credit spread (bearish)
- Logic: breakout establishes directional bias, sell premium on the opposite side

Performance: Outperformed SPY from May 2022 to July 2025 (~640 trades)
Note: Some performance decay observed in 2025 (crowded trade)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

import sys
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from data_fetcher import bs_price, bs_delta, generate_option_chain, estimate_intraday_iv


class ORBDirection(Enum):
    BULLISH = "bullish"  # Breakout above → sell put spreads
    BEARISH = "bearish"  # Breakout below → sell call spreads
    NONE = "none"


class ORBTradeStatus(Enum):
    OPEN = "open"
    WIN = "win"
    STOPPED = "stopped"
    EXPIRED_ITM = "expired_itm"


@dataclass
class ORBTrade:
    entry_time: pd.Timestamp
    direction: ORBDirection
    underlying_price: float
    short_strike: float
    long_strike: float
    spread_type: str  # "put_spread" or "call_spread"
    credit: float
    short_delta: float
    stop_loss: float  # Usually 1x-2x credit
    
    status: ORBTradeStatus = ORBTradeStatus.OPEN
    exit_time: Optional[pd.Timestamp] = None
    pnl: float = 0.0
    max_risk: float = 0.0
    
    def __post_init__(self):
        spread_width = abs(self.long_strike - self.short_strike)
        self.max_risk = spread_width - self.credit


@dataclass
class ORBConfig:
    opening_range_minutes: int = 60  # First 60 min (9:30-10:30)
    breakout_buffer_pct: float = 0.0  # Extra % above/below range to confirm breakout
    delta_target: float = 10.0  # Target delta for short strike (in delta units)
    spread_width: float = 5.0  # Spread width in dollars
    stop_loss_multiplier: float = 1.5  # Stop loss = multiplier × credit
    max_trades_per_day: int = 2  # Max trades per day
    risk_per_trade_pct: float = 1.0  # % of account per trade
    strike_interval: float = 5.0  # Strike interval ($5 for SPX)
    min_credit: float = 0.50  # Minimum credit to accept


class ORBBacktester:
    """
    Backtests the Opening Range Breakout credit spread strategy.
    """
    
    def __init__(self, config: ORBConfig = None):
        self.config = config or ORBConfig()
        self.trades: List[ORBTrade] = []
        self.daily_pnl: List[dict] = []
    
    def detect_opening_range(self, day_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Compute the opening range (high/low) from the first N minutes.
        
        Returns (range_high, range_low)
        """
        start = day_data.index[0]
        end_time = start + pd.Timedelta(minutes=self.config.opening_range_minutes)
        
        or_data = day_data[day_data.index <= end_time]
        
        if or_data.empty:
            return None, None
        
        range_high = or_data["high"].max()
        range_low = or_data["low"].min()
        
        return range_high, range_low
    
    def detect_breakout(
        self,
        current_bar: pd.Series,
        range_high: float,
        range_low: float,
    ) -> ORBDirection:
        """Check if current bar breaks out of opening range."""
        buffer = (range_high - range_low) * self.config.breakout_buffer_pct
        
        if current_bar["close"] > range_high + buffer:
            return ORBDirection.BULLISH
        elif current_bar["close"] < range_low - buffer:
            return ORBDirection.BEARISH
        
        return ORBDirection.NONE
    
    def select_spread(
        self,
        direction: ORBDirection,
        S: float,
        T: float,
        sigma: float,
        r: float = 0.05,
    ) -> Optional[dict]:
        """
        Select credit spread based on breakout direction.
        
        Bullish breakout → sell OTM put credit spread
        Bearish breakout → sell OTM call credit spread
        """
        target_delta = self.config.delta_target / 100
        width = self.config.spread_width
        
        chain = generate_option_chain(
            S=S, T=T, r=r, sigma=sigma,
            strike_range_pct=0.05,
            strike_interval=self.config.strike_interval,
        )
        
        if direction == ORBDirection.BULLISH:
            # Sell put spread below the market
            puts = chain[chain["type"] == "put"].copy()
            puts["abs_delta"] = puts["delta"].abs()
            valid = puts[
                (puts["abs_delta"] >= target_delta * 0.5) &
                (puts["abs_delta"] <= target_delta * 2.0) &
                (puts["strike"] < S)
            ]
            
            if valid.empty:
                return None
            
            short_put = valid.iloc[(valid["abs_delta"] - target_delta).abs().argmin()]
            long_put_strike = short_put["strike"] - width
            long_put_price = bs_price(S, long_put_strike, T, r, sigma, "put")
            
            credit = short_put["price"] - long_put_price
            if credit < self.config.min_credit:
                return None
            
            return {
                "spread_type": "put_spread",
                "short_strike": short_put["strike"],
                "long_strike": long_put_strike,
                "credit": credit,
                "short_delta": short_put["delta"],
            }
        
        else:  # BEARISH
            # Sell call spread above the market
            calls = chain[chain["type"] == "call"].copy()
            calls["abs_delta"] = calls["delta"].abs()
            valid = calls[
                (calls["abs_delta"] >= target_delta * 0.5) &
                (calls["abs_delta"] <= target_delta * 2.0) &
                (calls["strike"] > S)
            ]
            
            if valid.empty:
                return None
            
            short_call = valid.iloc[(valid["abs_delta"] - target_delta).abs().argmin()]
            long_call_strike = short_call["strike"] + width
            long_call_price = bs_price(S, long_call_strike, T, r, sigma, "call")
            
            credit = short_call["price"] - long_call_price
            if credit < self.config.min_credit:
                return None
            
            return {
                "spread_type": "call_spread",
                "short_strike": short_call["strike"],
                "long_strike": long_call_strike,
                "credit": credit,
                "short_delta": short_call["delta"],
            }
    
    def evaluate_position(
        self,
        trade: ORBTrade,
        current_price: float,
        current_time: pd.Timestamp,
        T_remaining: float,
        sigma: float,
        r: float = 0.05,
    ) -> ORBTrade:
        """Evaluate open position against stop loss and expiration."""
        if trade.status != ORBTradeStatus.OPEN:
            return trade
        
        if trade.spread_type == "put_spread":
            spread_value = (
                bs_price(current_price, trade.short_strike, T_remaining, r, sigma, "put") -
                bs_price(current_price, trade.long_strike, T_remaining, r, sigma, "put")
            )
        else:
            spread_value = (
                bs_price(current_price, trade.short_strike, T_remaining, r, sigma, "call") -
                bs_price(current_price, trade.long_strike, T_remaining, r, sigma, "call")
            )
        
        current_loss = spread_value - trade.credit
        
        # Stop loss check
        if current_loss >= trade.stop_loss:
            trade.status = ORBTradeStatus.STOPPED
            trade.pnl = -trade.stop_loss
            trade.exit_time = current_time
        elif T_remaining <= 0:
            # Expiration
            if trade.spread_type == "put_spread":
                itm = current_price < trade.short_strike
            else:
                itm = current_price > trade.short_strike
            
            if itm:
                intrinsic = abs(current_price - trade.short_strike)
                spread_width = abs(trade.long_strike - trade.short_strike)
                loss = min(intrinsic, spread_width) - trade.credit
                trade.pnl = -loss
                trade.status = ORBTradeStatus.EXPIRED_ITM
            else:
                trade.pnl = trade.credit
                trade.status = ORBTradeStatus.WIN
            
            trade.exit_time = current_time
        
        return trade
    
    def run_backtest(
        self,
        price_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        account_size: float = 100000,
    ) -> dict:
        """Run ORB backtest over intraday price data."""
        self.trades = []
        self.daily_pnl = []
        equity = account_size
        
        price_data = price_data.copy()
        price_data["date"] = price_data.index.date
        trading_days = sorted(price_data["date"].unique())
        
        for day in trading_days:
            day_data = price_data[price_data["date"] == day].sort_index()
            
            if len(day_data) < self.config.opening_range_minutes + 30:
                continue
            
            # Get VIX
            vix_close = 20.0
            if vix_data is not None and not vix_data.empty:
                vix_dates = vix_data.index.date
                prev_days = [d for d in vix_dates if d <= day]
                if prev_days:
                    vix_close = vix_data.loc[vix_data.index.date == prev_days[-1], "close"].iloc[-1]
            
            # Detect opening range
            range_high, range_low = self.detect_opening_range(day_data)
            if range_high is None:
                continue
            
            market_open = day_data.index[0]
            total_minutes = (day_data.index[-1] - market_open).total_seconds() / 60
            
            day_trades = []
            breakout_detected = False
            
            for timestamp, bar in day_data.iterrows():
                minutes_elapsed = (timestamp - market_open).total_seconds() / 60
                minutes_remaining = max(total_minutes - minutes_elapsed, 0)
                T_remaining = minutes_remaining / (252 * 6.5 * 60)
                
                hours_remaining = minutes_remaining / 60
                sigma = estimate_intraday_iv(vix_close, hours_remaining)
                
                # Evaluate open positions
                for trade in day_trades:
                    if trade.status == ORBTradeStatus.OPEN:
                        self.evaluate_position(
                            trade, bar["close"], timestamp,
                            T_remaining, sigma
                        )
                
                # Only look for entries after opening range + some buffer
                if minutes_elapsed < self.config.opening_range_minutes + 5:
                    continue
                
                if len(day_trades) >= self.config.max_trades_per_day:
                    continue
                
                if not breakout_detected:
                    direction = self.detect_breakout(bar, range_high, range_low)
                    
                    if direction == ORBDirection.NONE:
                        continue
                    
                    breakout_detected = True
                    
                    spread = self.select_spread(
                        direction=direction,
                        S=bar["close"],
                        T=T_remaining,
                        sigma=sigma,
                    )
                    
                    if spread is None:
                        continue
                    
                    trade = ORBTrade(
                        entry_time=timestamp,
                        direction=direction,
                        underlying_price=bar["close"],
                        short_strike=spread["short_strike"],
                        long_strike=spread["long_strike"],
                        spread_type=spread["spread_type"],
                        credit=spread["credit"],
                        short_delta=spread["short_delta"],
                        stop_loss=spread["credit"] * self.config.stop_loss_multiplier,
                    )
                    
                    day_trades.append(trade)
            
            # Close remaining at EOD
            day_pnl = 0
            for trade in day_trades:
                if trade.status == ORBTradeStatus.OPEN:
                    trade.status = ORBTradeStatus.WIN
                    trade.pnl = trade.credit
                    trade.exit_time = day_data.index[-1]
                day_pnl += trade.pnl
            
            self.trades.extend(day_trades)
            equity += day_pnl
            
            self.daily_pnl.append({
                "date": day,
                "trades": len(day_trades),
                "direction": day_trades[0].direction.value if day_trades else "none",
                "pnl": day_pnl,
                "equity": equity,
                "range_high": range_high,
                "range_low": range_low,
                "range_width": range_high - range_low,
            })
        
        return self._compute_stats(account_size)
    
    def _compute_stats(self, initial_capital: float) -> dict:
        """Compute backtest statistics."""
        if not self.trades:
            return {"error": "No trades"}
        
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        daily_df = pd.DataFrame(self.daily_pnl)
        total_return = (daily_df["equity"].iloc[-1] - initial_capital) / initial_capital
        
        peak = daily_df["equity"].expanding().max()
        drawdown = (daily_df["equity"] - peak) / peak
        max_dd = drawdown.min()
        
        # Direction breakdown
        bullish_trades = [t for t in self.trades if t.direction == ORBDirection.BULLISH]
        bearish_trades = [t for t in self.trades if t.direction == ORBDirection.BEARISH]
        
        return {
            "total_trades": len(self.trades),
            "total_return_pct": round(total_return * 100, 2),
            "win_rate_pct": round(len(wins) / len(pnls) * 100, 1),
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "profitable_days_pct": round((daily_df["pnl"] > 0).mean() * 100, 1),
            "bullish_trades": len(bullish_trades),
            "bearish_trades": len(bearish_trades),
            "avg_range_width": round(daily_df["range_width"].mean(), 2),
            "total_pnl": round(sum(pnls), 2),
            "final_equity": round(daily_df["equity"].iloc[-1], 2),
        }


if __name__ == "__main__":
    config = ORBConfig(
        opening_range_minutes=60,
        delta_target=10,
        spread_width=5,
        stop_loss_multiplier=1.5,
    )
    
    bt = ORBBacktester(config)
    
    # Demo
    spread = bt.select_spread(
        direction=ORBDirection.BULLISH,
        S=6000.0,
        T=4 / (252 * 6.5),
        sigma=0.20,
    )
    
    if spread:
        print("=== ORB Spread Selection Demo (Bullish breakout) ===")
        print(f"Underlying: $6,000")
        print(f"Type: {spread['spread_type']}")
        print(f"Sell {spread['short_strike']} / Buy {spread['long_strike']}")
        print(f"Credit: ${spread['credit']:.2f}")
        print(f"Delta: {spread['short_delta']:.4f}")

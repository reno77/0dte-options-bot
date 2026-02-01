"""
Iron Condor Execution Engine for Moomoo OpenD

Handles the full lifecycle of 0DTE/1DTE iron condor positions:
1. Strike selection (by delta from option chain)
2. Entry (4-leg order: sell call spread + sell put spread)
3. Position monitoring (mark-to-market, stop loss)
4. Exit (close all legs, or let expire)

Designed for paper trading first — validates everything before execution.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from enum import Enum

import numpy as np
import pandas as pd

from moomoo import (
    OpenSecTradeContext, TrdSide, TrdEnv, OrderType, 
    TrdMarket, RET_OK, RET_ERROR, SecurityFirm
)

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    PENDING = "pending"       # Order submitted, not filled
    OPEN = "open"            # Filled, actively managed
    STOPPED = "stopped"      # Closed by stop loss
    EXPIRED = "expired"      # Expired (held to expiry)
    CLOSED = "closed"        # Manually closed
    ERROR = "error"          # Order failed


@dataclass
class IronCondorLeg:
    """Single option leg."""
    code: str = ""          # Moomoo option code (e.g. "US.SPY260202P00685000")
    strike: float = 0.0
    option_type: str = ""   # "CALL" or "PUT"
    side: str = ""          # "BUY" or "SELL"
    premium: float = 0.0    # Fill price
    delta: float = 0.0
    order_id: str = ""
    fill_price: float = 0.0
    quantity: int = 1


@dataclass
class IronCondorPosition:
    """Full iron condor position (4 legs)."""
    id: str = ""
    timestamp: str = ""
    underlying: str = "SPY"
    expiration: str = ""
    
    # Short legs (sold)
    short_call: Optional[IronCondorLeg] = None
    long_call: Optional[IronCondorLeg] = None
    short_put: Optional[IronCondorLeg] = None
    long_put: Optional[IronCondorLeg] = None
    
    # Position metrics
    net_credit: float = 0.0       # Total premium collected
    max_risk: float = 0.0         # Max loss per contract
    spread_width: float = 0.0
    quantity: int = 1             # Number of contracts
    
    # Risk management
    stop_loss_mult: float = 1.0   # Stop at N× premium
    stop_loss_price: float = 0.0  # Absolute stop level
    
    # Status
    status: PositionStatus = PositionStatus.PENDING
    pnl: float = 0.0
    close_reason: str = ""
    close_time: str = ""
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class IronCondorEngine:
    """
    Manages iron condor positions through moomoo OpenD.
    
    Strategy parameters:
    - delta_min/max: Target delta range for short strikes
    - spread_width: Distance between short and long strikes
    - stop_loss_mult: Close position when unrealized loss > N× premium
    - max_positions: Maximum concurrent positions
    - entry_interval_min: Minimum minutes between entries
    """
    
    def __init__(
        self,
        trade_ctx: OpenSecTradeContext,
        trd_env: TrdEnv = TrdEnv.SIMULATE,
        # Strategy params (from optimizer best config)
        delta_min: float = 0.05,
        delta_max: float = 0.15,
        spread_width: float = 5.0,
        stop_loss_mult: float = 1.0,
        max_positions: int = 5,
        entry_interval_min: int = 30,
        # Execution params
        underlying: str = "SPY",
        order_type: OrderType = OrderType.NORMAL,
        quantity: int = 1,
    ):
        self.ctx = trade_ctx
        self.trd_env = trd_env
        
        # Strategy
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.spread_width = spread_width
        self.stop_loss_mult = stop_loss_mult
        self.max_positions = max_positions
        self.entry_interval_min = entry_interval_min
        
        # Execution
        self.underlying = underlying
        self.order_type = order_type
        self.quantity = quantity
        
        # State
        self.positions: List[IronCondorPosition] = []
        self.last_entry_time: Optional[datetime] = None
        self.trade_log: List[Dict] = []
        
        logger.info(f"IronCondorEngine initialized: δ={delta_min}-{delta_max}, "
                    f"width=${spread_width}, SL={stop_loss_mult}×, "
                    f"max_pos={max_positions}")
    
    # ==================== Strike Selection ====================
    
    def find_strikes(
        self,
        chain: pd.DataFrame,
        underlying_price: float,
        expiration: str,
    ) -> Optional[Dict]:
        """
        Find optimal strikes for iron condor from live option chain.
        
        Returns dict with short_call, long_call, short_put, long_put
        or None if no valid strikes found.
        """
        if chain.empty:
            logger.warning("Empty option chain")
            return None
        
        # Filter to target expiration
        exp_chain = chain[chain["expiration"] == expiration].copy()
        if exp_chain.empty:
            logger.warning(f"No options for expiration {expiration}")
            return None
        
        # Find short call (above underlying, target delta)
        calls = exp_chain[
            (exp_chain["type"].str.upper().isin(["CALL", "C"])) &
            (exp_chain["strike"] > underlying_price) &
            (exp_chain["bid"] > 0.05)  # Must have a bid
        ].copy()
        
        if calls.empty:
            logger.warning("No valid call strikes found")
            return None
        
        # If we have delta from the chain, use it
        if "delta" in calls.columns and calls["delta"].notna().any():
            calls["abs_delta"] = calls["delta"].abs()
            target_delta = (self.delta_min + self.delta_max) / 2
            valid_calls = calls[
                (calls["abs_delta"] >= self.delta_min) &
                (calls["abs_delta"] <= self.delta_max)
            ]
            if not valid_calls.empty:
                # Pick closest to target delta
                short_call = valid_calls.iloc[
                    (valid_calls["abs_delta"] - target_delta).abs().argsort()[:1]
                ].iloc[0]
            else:
                # Fallback: closest to target delta in full chain
                short_call = calls.iloc[
                    (calls["abs_delta"] - target_delta).abs().argsort()[:1]
                ].iloc[0]
        else:
            # No delta available — estimate by distance from underlying
            # Roughly: delta 0.10 ≈ 2-3% OTM for 0-1DTE
            target_pct = 0.025  # 2.5% OTM as proxy for ~10 delta
            target_strike = underlying_price * (1 + target_pct)
            calls["dist"] = (calls["strike"] - target_strike).abs()
            short_call = calls.nsmallest(1, "dist").iloc[0]
        
        sc_strike = float(short_call["strike"])
        lc_strike = sc_strike + self.spread_width
        
        # Find short put (below underlying, target delta)
        puts = exp_chain[
            (exp_chain["type"].str.upper().isin(["PUT", "P"])) &
            (exp_chain["strike"] < underlying_price) &
            (exp_chain["bid"] > 0.05)
        ].copy()
        
        if puts.empty:
            logger.warning("No valid put strikes found")
            return None
        
        if "delta" in puts.columns and puts["delta"].notna().any():
            puts["abs_delta"] = puts["delta"].abs()
            valid_puts = puts[
                (puts["abs_delta"] >= self.delta_min) &
                (puts["abs_delta"] <= self.delta_max)
            ]
            if not valid_puts.empty:
                short_put = valid_puts.iloc[
                    (valid_puts["abs_delta"] - target_delta).abs().argsort()[:1]
                ].iloc[0]
            else:
                short_put = puts.iloc[
                    (puts["abs_delta"] - target_delta).abs().argsort()[:1]
                ].iloc[0]
        else:
            target_strike = underlying_price * (1 - target_pct)
            puts["dist"] = (puts["strike"] - target_strike).abs()
            short_put = puts.nsmallest(1, "dist").iloc[0]
        
        sp_strike = float(short_put["strike"])
        lp_strike = sp_strike - self.spread_width
        
        # Validate spread
        if sc_strike <= underlying_price or sp_strike >= underlying_price:
            logger.warning(f"Invalid strikes: SC={sc_strike}, SP={sp_strike}, S={underlying_price}")
            return None
        
        # Get premiums (use mid price for estimation)
        sc_mid = (float(short_call.get("bid", 0)) + float(short_call.get("ask", 0))) / 2
        sp_mid = (float(short_put.get("bid", 0)) + float(short_put.get("ask", 0))) / 2
        
        # Find long legs in chain
        lc_row = exp_chain[
            (exp_chain["type"].str.upper().isin(["CALL", "C"])) &
            (exp_chain["strike"] == lc_strike)
        ]
        lp_row = exp_chain[
            (exp_chain["type"].str.upper().isin(["PUT", "P"])) &
            (exp_chain["strike"] == lp_strike)
        ]
        
        lc_mid = 0.0
        lp_mid = 0.0
        if not lc_row.empty:
            lc_mid = (float(lc_row.iloc[0].get("bid", 0)) + 
                     float(lc_row.iloc[0].get("ask", 0))) / 2
        if not lp_row.empty:
            lp_mid = (float(lp_row.iloc[0].get("bid", 0)) + 
                     float(lp_row.iloc[0].get("ask", 0))) / 2
        
        call_credit = sc_mid - lc_mid
        put_credit = sp_mid - lp_mid
        net_credit = call_credit + put_credit
        max_risk = self.spread_width - net_credit
        
        if net_credit <= 0.10:
            logger.warning(f"Net credit too low: ${net_credit:.2f}")
            return None
        
        result = {
            "underlying_price": underlying_price,
            "expiration": expiration,
            "short_call_strike": sc_strike,
            "long_call_strike": lc_strike,
            "short_put_strike": sp_strike,
            "long_put_strike": lp_strike,
            "call_credit": round(call_credit, 2),
            "put_credit": round(put_credit, 2),
            "net_credit": round(net_credit, 2),
            "max_risk": round(max_risk, 2),
            "stop_loss": round(net_credit * self.stop_loss_mult, 2),
            "short_call_delta": float(short_call.get("delta", 0)) if "delta" in short_call.index else 0,
            "short_put_delta": float(short_put.get("delta", 0)) if "delta" in short_put.index else 0,
        }
        
        logger.info(f"Found IC: SC={sc_strike} SP={sp_strike} "
                    f"credit=${net_credit:.2f} risk=${max_risk:.2f}")
        return result
    
    # ==================== Order Execution ====================
    
    def _build_option_code(self, strike: float, option_type: str, expiration: str) -> str:
        """
        Build moomoo option code.
        Format: US.SPY{YYMMDD}{C/P}{strike*1000:08d}
        Example: US.SPY260202P00685000
        """
        exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
        exp_str = exp_dt.strftime("%y%m%d")
        cp = "C" if option_type.upper() in ("CALL", "C") else "P"
        strike_int = int(strike * 1000)
        return f"US.{self.underlying}{exp_str}{cp}{strike_int:08d}"
    
    def place_iron_condor(
        self,
        strikes: Dict,
        dry_run: bool = True,
    ) -> Optional[IronCondorPosition]:
        """
        Place a full iron condor order (4 legs).
        
        In moomoo, we submit each leg as individual orders:
        1. SELL short call
        2. BUY long call  
        3. SELL short put
        4. BUY long put
        
        Args:
            strikes: Output from find_strikes()
            dry_run: If True, simulate only (log but don't execute)
        
        Returns:
            IronCondorPosition or None on failure
        """
        exp = strikes["expiration"]
        
        # Build option codes
        sc_code = self._build_option_code(strikes["short_call_strike"], "CALL", exp)
        lc_code = self._build_option_code(strikes["long_call_strike"], "CALL", exp)
        sp_code = self._build_option_code(strikes["short_put_strike"], "PUT", exp)
        lp_code = self._build_option_code(strikes["long_put_strike"], "PUT", exp)
        
        position = IronCondorPosition(
            id=f"IC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            underlying=self.underlying,
            expiration=exp,
            short_call=IronCondorLeg(
                code=sc_code, strike=strikes["short_call_strike"],
                option_type="CALL", side="SELL",
                premium=strikes["call_credit"],
                delta=strikes.get("short_call_delta", 0),
            ),
            long_call=IronCondorLeg(
                code=lc_code, strike=strikes["long_call_strike"],
                option_type="CALL", side="BUY",
            ),
            short_put=IronCondorLeg(
                code=sp_code, strike=strikes["short_put_strike"],
                option_type="PUT", side="SELL",
                premium=strikes["put_credit"],
                delta=strikes.get("short_put_delta", 0),
            ),
            long_put=IronCondorLeg(
                code=lp_code, strike=strikes["long_put_strike"],
                option_type="PUT", side="BUY",
            ),
            net_credit=strikes["net_credit"],
            max_risk=strikes["max_risk"],
            spread_width=self.spread_width,
            quantity=self.quantity,
            stop_loss_mult=self.stop_loss_mult,
            stop_loss_price=strikes["stop_loss"],
            status=PositionStatus.PENDING,
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{'DRY RUN - ' if dry_run else ''}IRON CONDOR ORDER")
        logger.info(f"{'='*60}")
        logger.info(f"  Underlying: {self.underlying} @ ${strikes['underlying_price']:.2f}")
        logger.info(f"  Expiration: {exp}")
        logger.info(f"  SELL {sc_code} (call) @ ~${strikes['call_credit']:.2f}")
        logger.info(f"  BUY  {lc_code} (call)")
        logger.info(f"  SELL {sp_code} (put)  @ ~${strikes['put_credit']:.2f}")
        logger.info(f"  BUY  {lp_code} (put)")
        logger.info(f"  Net Credit: ${strikes['net_credit']:.2f} × {self.quantity} = "
                    f"${strikes['net_credit'] * self.quantity * 100:.2f}")
        logger.info(f"  Max Risk:   ${strikes['max_risk']:.2f} × {self.quantity} = "
                    f"${strikes['max_risk'] * self.quantity * 100:.2f}")
        logger.info(f"  Stop Loss:  ${strikes['stop_loss']:.2f}")
        logger.info(f"{'='*60}")
        
        if dry_run:
            position.status = PositionStatus.OPEN
            self.positions.append(position)
            self.last_entry_time = datetime.now()
            self._log_trade("OPEN_DRY", position)
            logger.info("  ✅ DRY RUN — position logged (no real orders placed)")
            return position
        
        # Real execution — place 4 individual orders
        legs = [
            (sc_code, TrdSide.SELL_SHORT, "short_call"),
            (lc_code, TrdSide.BUY, "long_call"),
            (sp_code, TrdSide.SELL_SHORT, "short_put"),
            (lp_code, TrdSide.BUY, "long_put"),
        ]
        
        order_ids = []
        for code, side, leg_name in legs:
            ret, data = self.ctx.place_order(
                price=0.0,  # Market order
                qty=self.quantity,
                code=code,
                trd_side=side,
                order_type=OrderType.MARKET,
                trd_env=self.trd_env,
            )
            
            if ret != RET_OK:
                logger.error(f"  ❌ Failed to place {leg_name}: {data}")
                # Cancel already-placed legs
                for oid in order_ids:
                    self.ctx.modify_order(
                        modify_order_op=2,  # Cancel
                        order_id=oid,
                        qty=0, price=0,
                        trd_env=self.trd_env,
                    )
                position.status = PositionStatus.ERROR
                position.close_reason = f"Failed on {leg_name}: {data}"
                return None
            
            order_id = str(data.iloc[0].get("order_id", ""))
            order_ids.append(order_id)
            getattr(position, leg_name).order_id = order_id
            logger.info(f"  ✅ {leg_name} placed: {code} → order_id={order_id}")
            
            time.sleep(0.2)  # Brief pause between legs
        
        position.status = PositionStatus.OPEN
        self.positions.append(position)
        self.last_entry_time = datetime.now()
        self._log_trade("OPEN", position)
        
        logger.info(f"  ✅ All 4 legs placed successfully!")
        return position
    
    # ==================== Position Monitoring ====================
    
    def check_positions(self, underlying_price: float) -> List[Dict]:
        """
        Check all open positions for stop loss or expiry.
        Returns list of actions taken.
        """
        actions = []
        now = datetime.now()
        
        for pos in self.positions:
            if pos.status != PositionStatus.OPEN:
                continue
            
            # Check expiry (close at 3:45 PM ET on expiry day)
            exp_date = datetime.strptime(pos.expiration, "%Y-%m-%d")
            if now.date() >= exp_date.date() and now.hour >= 15 and now.minute >= 45:
                pos.status = PositionStatus.EXPIRED
                pos.close_time = now.isoformat()
                pos.close_reason = "Expiration"
                
                # Estimate P&L at expiry
                sc = pos.short_call.strike if pos.short_call else 999999
                sp = pos.short_put.strike if pos.short_put else 0
                
                if sp <= underlying_price <= sc:
                    # Expired worthless — full profit
                    pos.pnl = pos.net_credit * pos.quantity * 100
                else:
                    # Breached — calculate intrinsic value
                    if underlying_price > sc:
                        loss = min(underlying_price - sc, pos.spread_width)
                    else:
                        loss = min(sp - underlying_price, pos.spread_width)
                    pos.pnl = (pos.net_credit - loss) * pos.quantity * 100
                
                self._log_trade("EXPIRE", pos)
                actions.append({"action": "expired", "position": pos.id, "pnl": pos.pnl})
                logger.info(f"  📋 {pos.id} EXPIRED: P&L=${pos.pnl:.2f}")
                continue
            
            # Check stop loss (mark-to-market estimation)
            # Estimate current IC value based on underlying price vs strikes
            sc = pos.short_call.strike if pos.short_call else 999999
            sp = pos.short_put.strike if pos.short_put else 0
            
            if underlying_price >= sc:
                # Call side breached
                call_loss = min(underlying_price - sc, pos.spread_width)
                unrealized_loss = call_loss - pos.net_credit
            elif underlying_price <= sp:
                # Put side breached
                put_loss = min(sp - underlying_price, pos.spread_width)
                unrealized_loss = put_loss - pos.net_credit
            else:
                # Within range — still profitable (simplified)
                unrealized_loss = -pos.net_credit * 0.3  # Estimate ~30% theta gain
            
            if unrealized_loss > pos.stop_loss_price:
                pos.status = PositionStatus.STOPPED
                pos.close_time = now.isoformat()
                pos.close_reason = f"Stop loss hit (loss=${unrealized_loss:.2f})"
                pos.pnl = -unrealized_loss * pos.quantity * 100
                
                self._log_trade("STOP", pos)
                actions.append({"action": "stopped", "position": pos.id, "pnl": pos.pnl})
                logger.info(f"  🛑 {pos.id} STOPPED: P&L=${pos.pnl:.2f}")
        
        return actions
    
    def close_position(
        self,
        position: IronCondorPosition,
        reason: str = "manual",
        dry_run: bool = True,
    ) -> bool:
        """Close all legs of a position."""
        if position.status != PositionStatus.OPEN:
            logger.warning(f"Position {position.id} not open (status={position.status})")
            return False
        
        logger.info(f"Closing position {position.id}: {reason}")
        
        if not dry_run:
            # Close each leg (reverse the original side)
            legs = [
                (position.short_call, TrdSide.BUY),     # Buy back short call
                (position.long_call, TrdSide.SELL),      # Sell long call
                (position.short_put, TrdSide.BUY),       # Buy back short put
                (position.long_put, TrdSide.SELL),        # Sell long put
            ]
            
            for leg, side in legs:
                if leg and leg.code:
                    ret, data = self.ctx.place_order(
                        price=0.0,
                        qty=self.quantity,
                        code=leg.code,
                        trd_side=side,
                        order_type=OrderType.MARKET,
                        trd_env=self.trd_env,
                    )
                    if ret != RET_OK:
                        logger.error(f"Failed to close {leg.code}: {data}")
                    time.sleep(0.2)
        
        position.status = PositionStatus.CLOSED
        position.close_time = datetime.now().isoformat()
        position.close_reason = reason
        self._log_trade("CLOSE", position)
        
        return True
    
    # ==================== Entry Logic ====================
    
    def should_enter(self) -> bool:
        """Check if we should enter a new position."""
        # Max positions check
        open_count = sum(1 for p in self.positions if p.status == PositionStatus.OPEN)
        if open_count >= self.max_positions:
            return False
        
        # Entry interval check
        if self.last_entry_time:
            elapsed = (datetime.now() - self.last_entry_time).total_seconds() / 60
            if elapsed < self.entry_interval_min:
                return False
        
        # Time window check (10:00 AM - 1:30 PM ET)
        # This needs timezone handling — for now use a simple check
        now = datetime.now()
        # Assume we're running in ET or can convert
        hour = now.hour
        if hour < 10 or hour > 13:
            return False
        if hour == 13 and now.minute > 30:
            return False
        
        return True
    
    # ==================== Logging ====================
    
    def _log_trade(self, action: str, position: IronCondorPosition):
        """Log trade to memory."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "position_id": position.id,
            "underlying": position.underlying,
            "expiration": position.expiration,
            "net_credit": position.net_credit,
            "pnl": position.pnl,
            "status": position.status.value,
            "short_call": position.short_call.strike if position.short_call else 0,
            "short_put": position.short_put.strike if position.short_put else 0,
        }
        self.trade_log.append(entry)
    
    def get_summary(self) -> Dict:
        """Get summary of all positions."""
        total_pnl = sum(p.pnl for p in self.positions)
        open_pos = [p for p in self.positions if p.status == PositionStatus.OPEN]
        closed_pos = [p for p in self.positions if p.status != PositionStatus.OPEN 
                      and p.status != PositionStatus.PENDING]
        
        wins = sum(1 for p in closed_pos if p.pnl > 0)
        losses = sum(1 for p in closed_pos if p.pnl <= 0)
        
        return {
            "total_positions": len(self.positions),
            "open": len(open_pos),
            "closed": len(closed_pos),
            "wins": wins,
            "losses": losses,
            "win_rate": f"{wins/(wins+losses)*100:.1f}%" if (wins+losses) > 0 else "N/A",
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(closed_pos), 2) if closed_pos else 0,
        }
    
    def save_state(self, filepath: str):
        """Save engine state to JSON."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "delta_min": self.delta_min,
                "delta_max": self.delta_max,
                "spread_width": self.spread_width,
                "stop_loss_mult": self.stop_loss_mult,
                "max_positions": self.max_positions,
                "entry_interval_min": self.entry_interval_min,
                "underlying": self.underlying,
                "quantity": self.quantity,
            },
            "positions": [p.to_dict() for p in self.positions],
            "trade_log": self.trade_log,
            "summary": self.get_summary(),
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load engine state from JSON."""
        with open(filepath) as f:
            state = json.load(f)
        
        self.trade_log = state.get("trade_log", [])
        # Reconstruct positions from saved state
        for p_data in state.get("positions", []):
            pos = IronCondorPosition(
                id=p_data["id"],
                timestamp=p_data["timestamp"],
                underlying=p_data["underlying"],
                expiration=p_data["expiration"],
                net_credit=p_data["net_credit"],
                max_risk=p_data["max_risk"],
                spread_width=p_data["spread_width"],
                quantity=p_data["quantity"],
                stop_loss_mult=p_data["stop_loss_mult"],
                stop_loss_price=p_data["stop_loss_price"],
                status=PositionStatus(p_data["status"]),
                pnl=p_data["pnl"],
            )
            # Reconstruct legs
            for leg_name in ["short_call", "long_call", "short_put", "long_put"]:
                if p_data.get(leg_name):
                    leg_data = p_data[leg_name]
                    setattr(pos, leg_name, IronCondorLeg(**leg_data))
            
            self.positions.append(pos)
        
        logger.info(f"State loaded: {len(self.positions)} positions, "
                    f"{len(self.trade_log)} trade log entries")

"""
Moomoo OpenD Gateway Manager

Handles connection to moomoo OpenD (TCP gateway on port 11111).
Provides:
- Connection management with auto-reconnect
- Account info + balance queries
- US options chain with Greeks
- Paper/live trading context switching
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

try:
    from moomoo import (
        OpenSecTradeContext, OpenQuoteContext,
        TrdSide, TrdEnv, OrderType, TrdMarket,
        SecurityFirm, SubType, KLType, KL_FIELD,
        RET_OK, RET_ERROR
    )
except ImportError:
    print("ERROR: moomoo SDK not installed. Run: pip install moomoo")
    sys.exit(1)

logger = logging.getLogger(__name__)


class MoomooGateway:
    """
    Manages connection to moomoo OpenD gateway.
    
    OpenD must be running on the target host (default localhost:11111).
    For paper trading, the moomoo account must have paper trading enabled.
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        security_firm: SecurityFirm = SecurityFirm.FUTUINC,
        paper_trading: bool = True,
    ):
        self.host = host
        self.port = port
        self.security_firm = security_firm
        self.paper_trading = paper_trading
        self.trd_env = TrdEnv.SIMULATE if paper_trading else TrdEnv.REAL
        
        self._trade_ctx: Optional[OpenSecTradeContext] = None
        self._quote_ctx: Optional[OpenQuoteContext] = None
        
        logger.info(f"MoomooGateway initialized: {host}:{port} "
                    f"({'PAPER' if paper_trading else 'LIVE'})")
    
    # ==================== Connection Management ====================
    
    def connect(self) -> bool:
        """Connect to OpenD gateway. Returns True if successful."""
        try:
            # Trade context for US market
            self._trade_ctx = OpenSecTradeContext(
                host=self.host,
                port=self.port,
                filter_trdmarket=TrdMarket.US,
                security_firm=self.security_firm,
            )
            
            # Quote context for market data
            self._quote_ctx = OpenQuoteContext(
                host=self.host,
                port=self.port,
            )
            
            # Verify connection by checking account list
            ret, data = self._trade_ctx.get_acc_list()
            if ret != RET_OK:
                logger.error(f"Failed to get account list: {data}")
                return False
            
            logger.info(f"Connected to OpenD. Accounts: {len(data)} found")
            logger.info(f"Account data:\n{data}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Clean disconnect from OpenD."""
        if self._trade_ctx:
            self._trade_ctx.close()
            self._trade_ctx = None
        if self._quote_ctx:
            self._quote_ctx.close()
            self._quote_ctx = None
        logger.info("Disconnected from OpenD")
    
    @property
    def is_connected(self) -> bool:
        return self._trade_ctx is not None and self._quote_ctx is not None
    
    def _ensure_connected(self):
        if not self.is_connected:
            raise ConnectionError("Not connected to OpenD. Call connect() first.")
    
    # ==================== Account Info ====================
    
    def get_accounts(self) -> pd.DataFrame:
        """Get list of trading accounts."""
        self._ensure_connected()
        ret, data = self._trade_ctx.get_acc_list()
        if ret != RET_OK:
            raise RuntimeError(f"get_acc_list failed: {data}")
        return data
    
    def get_balance(self) -> Dict:
        """Get account balance and buying power."""
        self._ensure_connected()
        ret, data = self._trade_ctx.accinfo_query(trd_env=self.trd_env)
        if ret != RET_OK:
            raise RuntimeError(f"accinfo_query failed: {data}")
        
        row = data.iloc[0]
        return {
            "cash": float(row.get("cash", 0)),
            "total_assets": float(row.get("total_assets", 0)),
            "market_val": float(row.get("market_val", 0)),
            "available_funds": float(row.get("available_funds", 0)),
            "maintenance_margin": float(row.get("maintenance_margin", 0)),
            "unrealized_pnl": float(row.get("unrealized_pl", 0)),
            "realized_pnl": float(row.get("realized_pl", 0)),
        }
    
    def get_positions(self) -> pd.DataFrame:
        """Get current open positions."""
        self._ensure_connected()
        ret, data = self._trade_ctx.position_list_query(trd_env=self.trd_env)
        if ret != RET_OK:
            raise RuntimeError(f"position_list_query failed: {data}")
        return data
    
    def get_orders(self, status_filter: str = "") -> pd.DataFrame:
        """Get order history. status_filter: '' for all, or specific status."""
        self._ensure_connected()
        ret, data = self._trade_ctx.order_list_query(trd_env=self.trd_env)
        if ret != RET_OK:
            raise RuntimeError(f"order_list_query failed: {data}")
        return data
    
    # ==================== Market Data ====================
    
    def get_option_chain(
        self,
        symbol: str = "SPY",
        start_date: str = "",
        end_date: str = "",
        option_type: str = "ALL",
        strike_min: float = 0,
        strike_max: float = 0,
    ) -> pd.DataFrame:
        """
        Get option chain with Greeks from moomoo.
        
        Args:
            symbol: Underlying ticker (SPY, QQQ, SPX)
            start_date: Expiration start date (YYYY-MM-DD)
            end_date: Expiration end date (YYYY-MM-DD)
            option_type: "CALL", "PUT", or "ALL"
            strike_min/max: Filter by strike range (0 = no filter)
        
        Returns:
            DataFrame with columns: strike, type, bid, ask, last, volume, 
            open_interest, iv, delta, gamma, theta, vega, expiration
        """
        self._ensure_connected()
        
        # Map symbol to moomoo format
        code = f"US.{symbol}"
        
        # Get option expirations
        ret, data = self._quote_ctx.get_option_expiration_date(code=code)
        if ret != RET_OK:
            raise RuntimeError(f"get_option_expiration_date failed: {data}")
        
        expirations = data["strike_time"].tolist() if "strike_time" in data.columns else []
        logger.info(f"{symbol} has {len(expirations)} option expirations")
        
        # Filter expirations by date range
        if start_date or end_date:
            filtered = []
            for exp in expirations:
                exp_str = str(exp)[:10]
                if start_date and exp_str < start_date:
                    continue
                if end_date and exp_str > end_date:
                    continue
                filtered.append(exp)
            expirations = filtered
        
        if not expirations:
            logger.warning(f"No expirations found for {symbol} in range")
            return pd.DataFrame()
        
        # Get option chain for each expiration
        all_options = []
        for exp in expirations[:5]:  # Limit to avoid rate limits
            ret, data = self._quote_ctx.get_option_chain(
                code=code,
                start=str(exp)[:10],
                end=str(exp)[:10],
            )
            if ret != RET_OK:
                logger.warning(f"get_option_chain failed for {exp}: {data}")
                continue
            
            for _, row in data.iterrows():
                opt = {
                    "code": row.get("code", ""),
                    "name": row.get("name", ""),
                    "strike": float(row.get("option_strike", 0)),
                    "type": row.get("option_type", ""),
                    "expiration": str(exp)[:10],
                    "last": float(row.get("last_price", 0)),
                    "bid": float(row.get("bid_price", 0)),
                    "ask": float(row.get("ask_price", 0)),
                    "volume": int(row.get("volume", 0)),
                    "open_interest": int(row.get("open_interest", 0)),
                }
                all_options.append(opt)
            
            time.sleep(0.3)  # Rate limit
        
        df = pd.DataFrame(all_options)
        
        # Apply strike filter
        if strike_min > 0:
            df = df[df["strike"] >= strike_min]
        if strike_max > 0:
            df = df[df["strike"] <= strike_max]
        
        logger.info(f"Got {len(df)} options for {symbol}")
        return df
    
    def get_quote(self, code: str) -> Dict:
        """Get real-time quote for a single security."""
        self._ensure_connected()
        ret, data = self._quote_ctx.get_market_snapshot([code])
        if ret != RET_OK:
            raise RuntimeError(f"get_market_snapshot failed: {data}")
        if data.empty:
            return {}
        row = data.iloc[0]
        return {
            "code": code,
            "last": float(row.get("last_price", 0)),
            "bid": float(row.get("bid_price", 0)),
            "ask": float(row.get("ask_price", 0)),
            "volume": int(row.get("volume", 0)),
            "high": float(row.get("high_price", 0)),
            "low": float(row.get("low_price", 0)),
            "open": float(row.get("open_price", 0)),
            "prev_close": float(row.get("prev_close_price", 0)),
        }
    
    def get_underlying_price(self, symbol: str = "SPY") -> float:
        """Get current price of underlying."""
        q = self.get_quote(f"US.{symbol}")
        return q.get("last", 0.0)


# Avoid circular import — import pandas at module level
import pandas as pd

"""
0DTE/1DTE Paper Trading Bot — Main Loop

Connects to moomoo OpenD, runs iron condor strategy on paper account.
Monitors positions, manages risk, logs everything.

Usage:
  python paper_bot.py                    # Dry run (no orders, just logging)
  python paper_bot.py --live-paper       # Paper trading via moomoo OpenD
  python paper_bot.py --config config.json  # Custom config
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.moomoo_gateway import MoomooGateway
from execution.iron_condor import IronCondorEngine, PositionStatus

# ==================== Config ====================

DEFAULT_CONFIG = {
    # Connection
    "opend_host": "127.0.0.1",
    "opend_port": 11111,
    "paper_trading": True,
    
    # Strategy (from optimizer best config — conservative/Sharpe)
    "underlying": "SPY",
    "delta_min": 0.05,
    "delta_max": 0.15,
    "spread_width": 5.0,
    "stop_loss_mult": 1.0,
    "max_positions": 2,
    "entry_interval_min": 30,
    "quantity": 1,
    
    # Schedule (ET times)
    "entry_start_hour": 10,
    "entry_start_min": 0,
    "entry_end_hour": 13,
    "entry_end_min": 30,
    
    # Execution
    "poll_interval_sec": 60,     # Check positions every 60s
    "dry_run": True,             # Don't actually place orders
    
    # Paths
    "state_file": "state/paper_bot_state.json",
    "log_file": "logs/paper_bot.log",
}


def setup_logging(log_file: str, level: int = logging.INFO):
    """Configure logging to file and console."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(ch)


logger = logging.getLogger(__name__)


class PaperBot:
    """
    Main paper trading bot. Orchestrates:
    1. Connect to moomoo OpenD
    2. Fetch option chain
    3. Find strikes → place iron condor
    4. Monitor positions → check stop loss / expiry
    5. Loop
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.running = False
        
        # State directory
        state_dir = Path(config["state_file"]).parent
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Gateway connection
        self.gateway = MoomooGateway(
            host=config["opend_host"],
            port=config["opend_port"],
            paper_trading=config["paper_trading"],
        )
        
        # Iron condor engine (initialized after connection)
        self.engine: IronCondorEngine = None
        
        # Stats
        self.cycle_count = 0
        self.last_chain_fetch = None
        self.cached_chain = None
    
    def start(self):
        """Start the paper trading bot."""
        logger.info("=" * 70)
        logger.info("🚀 0DTE/1DTE PAPER TRADING BOT")
        logger.info(f"   Mode: {'DRY RUN' if self.config['dry_run'] else 'PAPER TRADING'}")
        logger.info(f"   Underlying: {self.config['underlying']}")
        logger.info(f"   Delta: {self.config['delta_min']}-{self.config['delta_max']}")
        logger.info(f"   Spread: ${self.config['spread_width']}")
        logger.info(f"   Stop Loss: {self.config['stop_loss_mult']}×")
        logger.info(f"   Max Positions: {self.config['max_positions']}")
        logger.info("=" * 70)
        
        if not self.config["dry_run"]:
            # Connect to moomoo OpenD
            logger.info("Connecting to moomoo OpenD...")
            if not self.gateway.connect():
                logger.error("Failed to connect to OpenD. Is it running?")
                logger.info("Start OpenD first: ./FutuOpenD/FutuOpenD &")
                return False
            
            # Show account info
            try:
                balance = self.gateway.get_balance()
                logger.info(f"Account balance: ${balance['total_assets']:,.2f}")
                logger.info(f"Available funds: ${balance['available_funds']:,.2f}")
            except Exception as e:
                logger.warning(f"Could not get balance: {e}")
            
            # Initialize engine with trade context
            from moomoo import TrdEnv
            self.engine = IronCondorEngine(
                trade_ctx=self.gateway._trade_ctx,
                trd_env=TrdEnv.SIMULATE if self.config["paper_trading"] else TrdEnv.REAL,
                delta_min=self.config["delta_min"],
                delta_max=self.config["delta_max"],
                spread_width=self.config["spread_width"],
                stop_loss_mult=self.config["stop_loss_mult"],
                max_positions=self.config["max_positions"],
                entry_interval_min=self.config["entry_interval_min"],
                underlying=self.config["underlying"],
                quantity=self.config["quantity"],
            )
        else:
            # Dry run — create engine without real connection
            self.engine = IronCondorEngine(
                trade_ctx=None,  # No real context
                delta_min=self.config["delta_min"],
                delta_max=self.config["delta_max"],
                spread_width=self.config["spread_width"],
                stop_loss_mult=self.config["stop_loss_mult"],
                max_positions=self.config["max_positions"],
                entry_interval_min=self.config["entry_interval_min"],
                underlying=self.config["underlying"],
                quantity=self.config["quantity"],
            )
        
        # Load previous state if exists
        state_file = self.config["state_file"]
        if Path(state_file).exists():
            self.engine.load_state(state_file)
            logger.info(f"Loaded state from {state_file}")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        logger.info("Bot started. Press Ctrl+C to stop.")
        
        return True
    
    def run_loop(self):
        """Main trading loop."""
        while self.running:
            try:
                self._run_cycle()
                self.cycle_count += 1
                
                # Save state periodically
                if self.cycle_count % 5 == 0:
                    self.engine.save_state(self.config["state_file"])
                
                time.sleep(self.config["poll_interval_sec"])
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                time.sleep(10)
        
        self._shutdown()
    
    def _run_cycle(self):
        """Single trading cycle."""
        now = datetime.now()
        
        # Check market hours (simplified — assumes ET)
        hour = now.hour
        
        # Before market: show status
        if hour < 9 or (hour == 9 and now.minute < 30):
            if self.cycle_count == 0:
                logger.info("⏳ Waiting for market open (9:30 AM ET)")
            return
        
        # After market: wrap up
        if hour >= 16:
            if self.cycle_count % 10 == 0:
                summary = self.engine.get_summary()
                logger.info(f"📊 End of day: {json.dumps(summary)}")
            return
        
        # During market hours
        logger.debug(f"Cycle #{self.cycle_count}")
        
        # 1. Get underlying price
        if self.config["dry_run"]:
            # In dry run, use yfinance for price
            try:
                import yfinance as yf
                ticker = yf.Ticker(self.config["underlying"])
                hist = ticker.history(period="1d", interval="1m")
                underlying_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0
            except:
                underlying_price = 692.0  # Fallback
        else:
            underlying_price = self.gateway.get_underlying_price(self.config["underlying"])
        
        if underlying_price <= 0:
            logger.warning("Could not get underlying price")
            return
        
        # 2. Check existing positions
        if self.engine.positions:
            actions = self.engine.check_positions(underlying_price)
            for act in actions:
                logger.info(f"Position action: {act}")
        
        # 3. Consider new entries
        if self.engine.should_enter():
            chain = self._get_option_chain()
            if chain is not None and not chain.empty:
                # Find next expiration (0DTE or 1DTE)
                today = now.strftime("%Y-%m-%d")
                tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
                
                # Try 0DTE first, then 1DTE
                for exp in [today, tomorrow]:
                    strikes = self.engine.find_strikes(chain, underlying_price, exp)
                    if strikes:
                        position = self.engine.place_iron_condor(
                            strikes, dry_run=self.config["dry_run"]
                        )
                        if position:
                            logger.info(f"✅ New IC position: {position.id}")
                        break
        
        # 4. Log status every 10 cycles
        if self.cycle_count % 10 == 0:
            summary = self.engine.get_summary()
            open_pos = sum(1 for p in self.engine.positions 
                         if p.status == PositionStatus.OPEN)
            logger.info(f"📊 Status: {self.config['underlying']}=${underlying_price:.2f} | "
                       f"Open:{open_pos} | PnL:${summary['total_pnl']:.2f} | "
                       f"WR:{summary['win_rate']} | Cycle:{self.cycle_count}")
    
    def _get_option_chain(self) -> pd.DataFrame:
        """Get option chain (cached for 5 minutes)."""
        now = datetime.now()
        
        if (self.cached_chain is not None and 
            self.last_chain_fetch and
            (now - self.last_chain_fetch).total_seconds() < 300):
            return self.cached_chain
        
        try:
            if self.config["dry_run"]:
                # Use yfinance for option chain in dry run
                import yfinance as yf
                ticker = yf.Ticker(self.config["underlying"])
                expirations = ticker.options
                
                if not expirations:
                    return pd.DataFrame()
                
                # Get nearest 2 expirations
                all_chains = []
                for exp in expirations[:2]:
                    chain = ticker.option_chain(exp)
                    for side, df in [("call", chain.calls), ("put", chain.puts)]:
                        df = df.copy()
                        df["type"] = side
                        df["expiration"] = exp
                        # Rename columns to match our format
                        if "impliedVolatility" in df.columns:
                            df["iv"] = df["impliedVolatility"]
                        all_chains.append(df)
                    time.sleep(0.3)
                
                self.cached_chain = pd.concat(all_chains, ignore_index=True)
            else:
                # Use moomoo for real chain
                self.cached_chain = self.gateway.get_option_chain(
                    symbol=self.config["underlying"]
                )
            
            self.last_chain_fetch = now
            return self.cached_chain
            
        except Exception as e:
            logger.error(f"Failed to get option chain: {e}")
            return pd.DataFrame()
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Signal {signum} received, shutting down...")
        self.running = False
    
    def _shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        
        # Save final state
        self.engine.save_state(self.config["state_file"])
        
        # Print final summary
        summary = self.engine.get_summary()
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 FINAL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(json.dumps(summary, indent=2))
        
        # Disconnect
        self.gateway.disconnect()
        logger.info("Bot stopped.")


def main():
    parser = argparse.ArgumentParser(description="0DTE/1DTE Paper Trading Bot")
    parser.add_argument("--config", type=str, help="Config JSON file")
    parser.add_argument("--live-paper", action="store_true", 
                       help="Connect to moomoo OpenD for paper trading")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Dry run mode (default)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="OpenD host")
    parser.add_argument("--port", type=int, default=11111,
                       help="OpenD port")
    parser.add_argument("--once", action="store_true",
                       help="Run single cycle then exit")
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config.update(json.load(f))
    
    # CLI overrides
    if args.live_paper:
        config["dry_run"] = False
    config["opend_host"] = args.host
    config["opend_port"] = args.port
    
    # Setup logging
    setup_logging(config["log_file"])
    
    # Start bot
    bot = PaperBot(config)
    if bot.start():
        if args.once:
            bot._run_cycle()
            bot._shutdown()
        else:
            bot.run_loop()


if __name__ == "__main__":
    main()

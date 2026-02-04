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

# Market sentiment integration
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "trading" / "ml" / "sentiment"))
    from market_sentiment import get_market_stress, should_skip_0dte_entry, get_0dte_config_overrides
    HAS_SENTIMENT = True
except ImportError:
    HAS_SENTIMENT = False

# ==================== Config ====================

DEFAULT_CONFIG = {
    # Connection
    "opend_host": "127.0.0.1",
    "opend_port": 11111,
    "paper_trading": True,
    
    # Strategy (from optimizer best config — conservative/Sharpe)
    "underlying": "SPY",
    "delta_min": 0.08,
    "delta_max": 0.20,
    "spread_width": 10.0,
    "stop_loss_mult": 1.5,
    "max_positions": 3,
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
        self._last_underlying_price = 0
    
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
            # Use OPTIONS paper account (4190846), not STOCK (4190847)
            options_acc_id = self.config.get("options_acc_id", 4190846)
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
                acc_id=options_acc_id,
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
    
    def _get_et_now(self):
        """Get current time in US/Eastern."""
        try:
            import pytz
            et = pytz.timezone('US/Eastern')
            return datetime.now(et)
        except ImportError:
            # Fallback: assume SGT (GMT+8), ET = SGT - 13h (EST) or SGT - 12h (EDT)
            # Conservative: use -13h (EST, Nov-Mar)
            return datetime.now() - timedelta(hours=13)

    def _run_cycle(self):
        """Single trading cycle."""
        now_et = self._get_et_now()
        now = datetime.now()  # Local time for logging
        
        # Check market hours in ET
        hour = now_et.hour
        minute = now_et.minute
        weekday = now_et.weekday()  # 0=Mon, 6=Sun
        
        # No trading on weekends
        if weekday >= 5:
            if self.cycle_count == 0:
                logger.info(f"Weekend - no trading (ET: {now_et.strftime('%A %H:%M')})")
            return
        
        # Before market: show status
        if hour < 9 or (hour == 9 and minute < 30):
            if self.cycle_count == 0:
                logger.info(f"Waiting for market open (ET: {now_et.strftime('%H:%M')}, opens 9:30)")
            return
        
        # After market: wrap up
        if hour >= 16:
            if self.cycle_count % 10 == 0:
                summary = self.engine.get_summary()
                logger.info(f"End of day (ET: {now_et.strftime('%H:%M')}): {json.dumps(summary)}")
            return
        
        # During market hours
        logger.debug(f"Cycle #{self.cycle_count} (ET: {now_et.strftime('%H:%M')})")
        
        # 1. Get underlying price
        underlying_price = 0
        logger.info(f"Cycle {self.cycle_count}: getting price...")
        
        # Try moomoo first (may not have US ETF quote rights)
        if not self.config["dry_run"]:
            try:
                underlying_price = self.gateway.get_underlying_price(self.config["underlying"])
            except Exception as e:
                logger.debug(f"Moomoo quote unavailable: {e}")
        
        # Fallback: yfinance (always works, free)
        if underlying_price <= 0:
            try:
                import yfinance as yf
                ticker = yf.Ticker(self.config["underlying"])
                underlying_price = float(ticker.fast_info.last_price)
                logger.info(f"  Price from yfinance: ${underlying_price:.2f}")
            except Exception as e:
                logger.warning(f"yfinance fallback failed: {e}")
        
        if underlying_price <= 0:
            logger.warning("Could not get underlying price")
            return
        
        self._last_underlying_price = underlying_price
        
        # 2. Check existing positions
        if self.engine.positions:
            actions = self.engine.check_positions(underlying_price)
            for act in actions:
                logger.info(f"Position action: {act}")
        
        # 3. Check market sentiment before entries
        sentiment_skip = False
        if HAS_SENTIMENT:
            try:
                skip, reason = should_skip_0dte_entry()
                if skip:
                    logger.warning(f"🛑 SENTIMENT BLOCK: {reason}")
                    sentiment_skip = True
                else:
                    stress = get_market_stress()
                    sig = stress.get('signal', 'unknown')
                    score = stress.get('stress_score', 0)
                    stale = stress.get('stale', True)
                    if not stale:
                        emoji = {"calm": "🟢", "neutral": "🟡", "elevated": "🟠", "high_stress": "🔴"}
                        logger.info(f"  {emoji.get(sig, '⚪')} Sentiment: {sig} (stress={score:.2f})")
                        
                        # Apply config overrides for elevated stress
                        if sig == 'elevated':
                            overrides = get_0dte_config_overrides({
                                'delta_min': self.config['delta_min'],
                                'delta_max': self.config['delta_max'],
                                'spread_width': self.config['spread_width'],
                                'max_positions': self.config['max_positions'],
                                'stop_loss_mult': self.config['stop_loss_mult'],
                            })
                            logger.info(f"  Sentiment adjustments: {overrides.get('reason', '')}")
                            # Temporarily adjust engine params
                            self.engine.delta_min = overrides.get('delta_min', self.engine.delta_min)
                            self.engine.delta_max = overrides.get('delta_max', self.engine.delta_max)
                            self.engine.spread_width = overrides.get('spread_width', self.engine.spread_width)
                            self.engine.max_positions = overrides.get('max_positions', self.engine.max_positions)
                    else:
                        logger.debug("  Sentiment data stale — using defaults")
            except Exception as e:
                logger.debug(f"  Sentiment check error: {e}")

        # 3b. Consider new entries
        logger.info(f"  Checking entry conditions...")
        should = self.engine.should_enter()
        logger.info(f"  Should enter: {should}")
        if should and not sentiment_skip:
            chain = self._get_option_chain()
            if chain is not None and not chain.empty:
                # Find next valid expiration
                # If after 4 PM ET, today's options are expired — use tomorrow
                today = now_et.strftime("%Y-%m-%d")
                tomorrow = (now_et + timedelta(days=1)).strftime("%Y-%m-%d")
                if now_et.hour >= 16:
                    today = tomorrow  # Today's options expired, skip to tomorrow
                
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
            # Include sentiment in status
            sent_str = ""
            if HAS_SENTIMENT:
                try:
                    stress = get_market_stress()
                    if not stress.get('stale'):
                        sent_str = f" | Mkt:{stress.get('signal','?')}({stress.get('stress_score',0):.2f})"
                except Exception:
                    pass
            logger.info(f"📊 Status: {self.config['underlying']}=${underlying_price:.2f} | "
                       f"Open:{open_pos} | PnL:${summary['total_pnl']:.2f} | "
                       f"WR:{summary['win_rate']} | Cycle:{self.cycle_count}{sent_str}")
    
    def _get_option_chain(self) -> pd.DataFrame:
        """Get option chain (cached for 5 minutes)."""
        now = datetime.now()
        
        if (self.cached_chain is not None and 
            self.last_chain_fetch and
            (now - self.last_chain_fetch).total_seconds() < 300):
            return self.cached_chain
        
        try:
            if not self.config["dry_run"]:
                # Use moomoo — filter to near-ATM strikes to limit API calls
                underlying_price = self._last_underlying_price or 695
                strike_range = underlying_price * 0.06  # ±6% from ATM
                now_et = self._get_et_now()
                # If after 4 PM ET, start from tomorrow
                if now_et.hour >= 16:
                    start_dt = now_et + timedelta(days=1)
                else:
                    start_dt = now_et
                self.cached_chain = self.gateway.get_option_chain(
                    symbol=self.config["underlying"],
                    start_date=start_dt.strftime("%Y-%m-%d"),
                    end_date=(start_dt + timedelta(days=2)).strftime("%Y-%m-%d"),
                    strike_min=underlying_price - strike_range,
                    strike_max=underlying_price + strike_range,
                )
                self.last_chain_fetch = now
                return self.cached_chain
            elif self.config["dry_run"]:
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

# 0DTE Options Trading Bot

Automated 0-day-to-expiration (0DTE) options trading bot for US markets, powered by [moomoo OpenD](https://www.moomoo.com/OpenAPI).

## Target Instruments

| Ticker | Type | Settlement | Style | Notes |
|--------|------|-----------|-------|-------|
| **SPX** | S&P 500 Index Option | Cash | European | Primary target — no assignment risk, 60/40 tax |
| **XSP** | Mini-SPX (1/10th) | Cash | European | Smaller account friendly |
| **SPY** | S&P 500 ETF Option | Shares | American | High liquidity |
| **QQQ** | Nasdaq 100 ETF Option | Shares | American | Nasdaq exposure |
| **NDX** | Nasdaq 100 Index Option | Cash | European | Large notional |

## Strategies

### 1. Breakeven Iron Condor (MEIC)
Sell iron condors at 5-15 delta with tight stop losses equal to total premium collected per side. Multiple entries throughout the day, 30 min apart.

### 2. Opening Range Breakout (ORB)
Define opening range (9:30-10:30 AM ET), sell credit spreads on the opposite side of the breakout.

### 3. Afternoon Iron Condors
Enter 1:00-2:00 PM ET for accelerated theta decay. 0.2-0.3% OTM, $5 wide spreads.

### 4. Directional Credit Spreads
Delta-based bull put / bear call spreads with technical signal confirmation.

## Architecture

```
┌─────────────────────────────────────────┐
│              Jetson (Live Bot)           │
│                                         │
│  bot.py ──► strategy.py ──► trader.py   │
│                 │               │       │
│           market_data.py   moomoo SDK   │
│                                 │       │
│                           OpenD Gateway │
│                                 │       │
└─────────────────────────────────┼───────┘
                                  │
                          moomoo Servers
                                  │
                            US Exchanges
```

## Project Structure

```
0dte-options-bot/
├── README.md
├── requirements.txt
├── config.json              # Strategy parameters
├── backtest/                # Backtesting framework
│   ├── data/                # Historical options data
│   ├── backtest_engine.py   # Core backtester
│   ├── strategies/          # Strategy implementations
│   └── results/             # Backtest output
├── src/                     # Live trading modules
│   ├── bot.py               # Main loop
│   ├── strategy.py          # Strategy engine
│   ├── trader.py            # moomoo order execution
│   ├── market_data.py       # Options chain & quotes
│   ├── greeks.py            # Options Greeks calculator
│   └── risk.py              # Position sizing & risk mgmt
└── tests/                   # Unit tests
```

## Setup

### Prerequisites
- Python 3.10+
- moomoo account with options trading enabled
- moomoo OpenD installed and running

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
1. Install and start [moomoo OpenD](https://www.moomoo.com/download/OpenAPI)
2. Copy `config.example.json` to `config.json`
3. Set your trading parameters

## Development Phases

- [x] Phase 0: Research & repo setup
- [ ] Phase 1: Backtesting framework (Alienware)
- [ ] Phase 2: Paper trading via moomoo OpenD (Jetson)
- [ ] Phase 3: Live trading

## Disclaimer

This software is for educational and research purposes only. Options trading involves substantial risk of loss. Past backtest performance does not guarantee future results. Use at your own risk.

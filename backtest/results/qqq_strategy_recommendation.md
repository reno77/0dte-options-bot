# QQQ vs SPY: 1DTE Iron Condor Analysis

*Date: 2026-02-02 | Source: Moomoo OpenD real-time option chains*

## Market Data

| Factor | SPY | QQQ |
|--------|-----|-----|
| Price | ~$600 | ~$620 |
| Daily Volatility | 2.41% | 3.65% (1.5× SPY) |
| 1DTE ATM IV | ~24% | ~24% |
| OTM Put Skew | Higher skew | Similar pattern |
| ATM Bid/Ask | $0.03-0.05 | $0.03-0.06 |
| OTM Bid/Ask | Tighter | Slightly wider |
| Daily Volume (ATM) | 5K-10K+ | 5K-7K |
| Index | S&P 500 | Nasdaq 100 |
| Settlement | American | American |
| Expiry Time | 4:15 PM ET | 4:15 PM ET |
| Sector Concentration | Diversified | Tech-heavy (50%+) |
| Correlation | — | ~0.90 to SPY |

## QQQ 1DTE Chain Snapshot (Feb 3 expiry)

**ATM:** $620 (call δ=0.55, put δ=-0.45)

### OTM Puts
| Strike | Mid Price | Delta |
|--------|-----------|-------|
| P590 | $0.16 | -0.025 |
| P595 | $0.26 | -0.041 |
| P600 | $0.44 | -0.068 |
| P605 | $0.80 | -0.116 |
| P610 | $1.40 | -0.191 |
| P615 | $2.37 | -0.300 |

### OTM Calls
| Strike | Mid Price | Delta |
|--------|-----------|-------|
| C630 | $0.78 | 0.166 |
| C631 | $0.59 | 0.135 |
| C633 | $0.31 | 0.082 |
| C635 | $0.15 | 0.046 |
| C637 | $0.07 | 0.024 |

## Example 1DTE Iron Condor (10-delta wings)

- **Sell P605** ($0.80) / **Buy P600** ($0.44) = **$0.36 credit** (put side)
- **Sell C633** ($0.31) / **Buy C638** ($0.05) = **$0.26 credit** (call side)
- **Total credit:** ~$0.62 on $5 wide = **12.4% of max risk**
- **Breakevens:** ~$604.38 / ~$633.62 (4.7% range)

## QQQ Advantages for 1DTE Iron Condors

1. **Higher volatility** = higher premiums (3.65% daily vs SPY 2.41%)
2. **More premium per contract** at equivalent delta levels
3. **Tech concentration** = more predictable around tech earnings
4. **Good liquidity** — 5K+ daily volume ATM
5. **$1 strike spacing near ATM** — precise strike selection
6. **2DTE premiums ~30-50% higher** than 1DTE (more flexibility)

## QQQ Disadvantages / Risks

1. **Tech concentration** = single-sector risk (AAPL, MSFT, NVDA dominate)
2. **Wider daily moves** — 1.5× SPY average daily range → more breaches
3. **Slightly wider bid/ask OTM** — more slippage on wing fills
4. **No cash settlement** — American style, assignment risk (unlike SPX/NDX)
5. **~0.90 correlation to SPY** — limited diversification benefit

## Strategy Recommendation: Combined SPY + QQQ

- **Run iron condors on BOTH** — partial diversification
- **QQQ**: Higher premium, higher risk → slightly wider wings or fewer entries
- **SPY**: Tighter spreads, more liquid → more entries per day
- **Allocation**: 60% SPY / 40% QQQ
- **Avoid**: Running both during tech earnings weeks (QQQ gap risk)

### Key Pricing Observations

- QQQ collects ~15-20% more premium than SPY at equivalent delta
- But QQQ has ~50% more daily movement → net risk-adjusted is similar
- OTM put skew is similar between QQQ and SPY
- $5 wide spreads on QQQ capture ~12% credit (comparable to SPY)
- 10-delta strikes are ~2% OTM for QQQ (vs ~1.5% for SPY)

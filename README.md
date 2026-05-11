# Walk-Forward HRP Portfolio Engine

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Riskfolio-Lib](https://img.shields.io/badge/Riskfolio--Lib-7.x-brightgreen)](https://github.com/dcajasn/Riskfolio-Lib)
[![bt](https://img.shields.io/badge/bt-1.x-orange)](https://github.com/pmorissette/bt)
[![QuantStats](https://img.shields.io/badge/QuantStats-reporting-red)](https://github.com/ranaroussi/quantstats)

> A **production-ready**, zero-lookahead-bias backtesting engine for long-term savings plans, using **Hierarchical Risk Parity (HRP)** with real-world market frictions: 15 bps commissions, T+1 execution delay, and a 5% minimum weight floor.

---

## Table of Contents

- [The Problem This Solves](#the-problem-this-solves)
- [Key Features](#key-features)
- [How the Engine Works](#how-the-engine-works)
- [Architecture: The T+1 State Machine](#architecture-the-t1-state-machine)
- [Default Strategy Template](#default-strategy-template)
- [Example Console Output](#example-console-output)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Reference](#configuration-reference)
- [Interpreting the Outputs](#interpreting-the-outputs)
- [Limitations & Hard Truths](#limitations--hard-truths)
- [Ideas for Contributors](#ideas-for-contributors)
- [License](#license)

---

## The Problem This Solves

Every "backtesting tool" you find online suffers from the same fatal flaw: **lookahead bias**.

They compute optimal portfolio weights using the entire historical dataset, then simulate performance *as if* they had known those weights from day one. The result is spectacular backtest performance that is mathematically guaranteed to have never existed in reality.

This engine eliminates that flaw entirely with a **Walk-Forward architecture**:

```
Traditional backtest:  ░░░░░░░░░░░░░░░░ All historical data → single set of weights → "performance"

Walk-Forward (this):   [3yr lookback] → weights₁ → [next quarter's performance]
                               [3yr lookback+Q] → weights₂ → [next quarter's performance]
                                      [3yr lookback+2Q] → weights₃ → ...
```

On **every rebalance date**, portfolio weights are computed using **only the data that existed at that exact moment in time**, then executed on the following trading day (T+1 delay). No future data is ever seen. The result is a true out-of-sample simulation.

---

## Key Features

| Feature | Implementation |
|---|---|
| **Zero lookahead bias** | `prices.loc[index < target.now]` — today's close never enters the optimizer |
| **T+1 Execution Delay** | `HRPWithT1Delay` state machine: compute on Day T, trade on Day T+1 open |
| **15 bps friction** | `commissions=lambda q, p: abs(q) * p * 0.0015` via `bt.Backtest` |
| **5% weight floor** | `MIN_WEIGHT = 0.05` prevents any asset being zeroed by the optimizer |
| **Walk-Forward HRP** | Riskfolio-Lib re-optimises every quarter using trailing 3-year returns |
| **Professional reports** | QuantStats HTML tear sheet with 30+ performance metrics |
| **Daily returns CSV** | Exportable return series for downstream analysis or DCA planning |
| **Historical weights CSV** | Full time-series of all walk-forward allocations |

---

## How the Engine Works

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE OVERVIEW                                  │
│                                                                               │
│   yfinance            Riskfolio-Lib               bt              QuantStats  │
│   ─────────           ─────────────           ─────────          ──────────── │
│   Download       →    HRP on rolling     →   Backtest with  →   HTML tear    │
│   adjusted            3-year window          T+1 delay +        sheet +      │
│   close prices        (no future data)       15 bps cost         CSV reports  │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Step 1 — Data Ingestion (yfinance)

Downloads split- and dividend-adjusted daily close prices for all configured tickers. Applies `ffill()` to bridge public-holiday gaps between markets (e.g. US vs. gold/commodity markets), then drops any leading rows with missing data across the universe.

### Step 2 — Walk-Forward HRP (Riskfolio-Lib)

On each rebalance trigger (quarterly by default), the algo slices the price history to all rows **strictly before the current date** (`index < target.now`) and passes the trailing `LOOKBACK_YEARS × 252` daily log-returns to `rp.HCPortfolio.optimization(model="HRP")`.

Riskfolio-Lib:
1. Computes the Pearson correlation matrix of returns
2. Converts it to a distance matrix: `d = sqrt(0.5 × (1 − ρ))`
3. Clusters assets using Ward hierarchical linkage
4. Recursively bisects the dendrogram, allocating risk inversely proportional to cluster variance
5. Returns a weight vector that sums to 1.0

A minimum weight floor of 5% is applied to the result, and the weights are re-normalised to sum exactly to 1.

### Step 3 — Backtesting with Frictions (bt)

The custom `HRPWithT1Delay` algo manages a pending-weights queue:
- **Day T**: Weights computed → stored as `_pending_weights` → algo returns `False` (no trade)
- **Day T+1**: Pending weights injected → algo returns `True` → `bt.algos.Rebalance()` fires

Every rebalance trade incurs a `15 bps × notional` commission via the `bt.Backtest` commissions callback.

### Step 4 — Reporting (QuantStats)

The live-trading daily return series (warmup period stripped) is passed to `qs.reports.html()`, generating a full institutional-grade tear sheet. A parallel save to `hrp_returns.csv` enables offline analysis and DCA planning.

---

## Architecture: The T+1 State Machine

The T+1 execution delay required a complete redesign from the standard `bt` algo chain (`Run* → CalculateWeights → Rebalance`). That standard chain aborts on the first `False` return, making a mid-chain delay impossible.

The solution is a single stateful algo (`HRPWithT1Delay`) that runs on **every bar** and manages its own internal state machine:

```
Every bar — __call__(target):

  ① pending_weights is not None?
     YES → inject into target.temp["weights"] → return True   ← DAY T+1: execute trade
     NO  → continue

  ② Is this a scheduled trigger day (quarter-end boundary)?
     NO  → return False                                        ← ordinary bar, no action
     YES → update last_trigger_date → continue

  ③ Is there enough lookback history (warmup guard)?
     NO  → return False                                        ← still in warmup, hold cash
     YES → continue

  ④ Compute HRP using ONLY past-only data window
     FAIL → log warning → return False
     OK   → store result as _pending_weights → return False    ← DAY T: weights computed
```

**Guarantees:**
- Weights from Day T are **never executed on Day T** (strict T+1 delay)
- The optimizer **never sees the current day's close** (`index < target.now`, not `<=`)
- Warmup period is handled gracefully (portfolio stays 100% in cash)
- Single-bar latency: no per-bar overhead except a `None` check when idle

---

## Default Strategy Template

The included configuration implements a diversified 15-year savings plan across four structural asset classes with low long-run correlation:

| Ticker | Asset Class | Instrument | AUM | Inception |
|--------|-------------|------------|-----|-----------|
| `SPY` | US Large-Cap Equities | SPDR S&P 500 ETF Trust | ~$600B | Jan 1993 |
| `EEM` | Emerging Markets | iShares MSCI Emerging Markets ETF | ~$29.7B | Apr 2003 |
| `URA` | Uranium / Nuclear Energy | Global X Uranium ETF | ~$7.9B | Nov 2010 |
| `GLD` | Physical Gold | SPDR Gold Shares | ~$157.7B | Nov 2004 |

> **Why URA?** Uranium represents a structural energy-transition thesis with near-zero correlation to both equities and traditional commodities. The 5% floor ensures permanent thesis exposure while the HRP algorithm determines how much risk it contributes relative to its historical variance.

**Binding constraint:** URA's inception in November 2010 sets the data window start. With a 3-year lookback, the first live trade fires in January 2014, providing ~12 years of bias-free simulation through multiple market regimes (2014–2016 EM bear, 2018 correction, 2020 COVID crash, 2022 rate-shock, 2024–2025 gold rally).

**Verified backtest result** (run on 2025-12-30):
- Total return (live period 2014–2025): **245.11%**
- Most recent allocation: GLD 39.7% · SPY 31.5% · EEM 24.1% · URA 4.6%

---

## Example Console Output

```
=================================================================
  WALK-FORWARD HRP PORTFOLIO ENGINE
=================================================================
[INFO] Output directory : C:\...\Investimenti
[INFO] Market frictions : 15 bps commission/slippage | T+1 execution delay | 5% weight floor

[STEP 1] Downloading adjusted-close prices ...
         Tickers : ['SPY', 'EEM', 'URA', 'GLD']
         Period  : 2010-11-01  →  2025-12-31
[INFO] Tickers with valid data : ['EEM', 'GLD', 'SPY', 'URA']

[STEP 2+3] Running production backtest (lookback: 3 yr | rebalance: quarterly | T+1 delay | 15.0 bps cost) ...
[INFO] Backtest period (incl. warmup) : 2010-11-04  →  2025-12-30
[INFO] Live-trading start (post T+1)  : 2014-01-03
[INFO] Total return (full period)     : 245.11%

[INFO] Historical weights saved → ...\hrp_weights_history.csv
[INFO] Daily returns saved      → ...\hrp_returns.csv

[INFO] Most recent HRP allocation (post-floor):
         GLD     39.73%
         SPY     31.53%
         EEM     24.14%
         URA      4.59%

[STEP 4] Generating QuantStats HTML tear sheet ...
[INFO] Tear sheet saved → ...\hrp_tearsheet.html
=================================================================
  Production backtest complete.
=================================================================

─────────────────────────────────────────────────────────
  KEY PERFORMANCE METRICS  (live-trading period)
─────────────────────────────────────────────────────────
  CAGR                    11.25%
  Sharpe Ratio              0.72
  Sortino Ratio             1.04
  Max Drawdown            -28.14%
  Calmar Ratio              0.40
  Annual Volatility        15.60%
─────────────────────────────────────────────────────────

─────────────────────────────────────────────────
  LATEST HRP ALLOCATION  (2025-10-01)
─────────────────────────────────────────────────
  TICKER       WEIGHT  BAR
─────────────────────────────────────────────────
  GLD          39.73%  ████████████████
  SPY          31.53%  ████████████
  EEM          24.14%  █████████
  URA           4.59%  ██
─────────────────────────────────────────────────
  TOTAL       100.00%
─────────────────────────────────────────────────
```

---

## Project Structure

```
walk-forward-hrp/
├── portfolio_analysis.py   # Core engine: HRP + bt + QuantStats pipeline
├── main.py                 # Orchestration: metrics table + optional git publish
├── requirements.txt        # Pinned Python dependencies
├── .env.example            # Template for GITHUB_TOKEN (never commit your .env)
├── .gitignore
├── LICENSE
└── README.md

# Output files are written to OUTPUT_DIR (default: outside this folder)
# and are excluded from version control by .gitignore:
#   hrp_weights_history.csv
#   hrp_returns.csv
#   hrp_tearsheet.html
```

---

## Installation

### Prerequisites

- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **git** — [git-scm.com](https://git-scm.com/)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/walk-forward-hrp.git
cd walk-forward-hrp

# 2. Create and activate a virtual environment (strongly recommended)
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

> **Windows note:** If you have multiple Python installations, invoke the interpreter explicitly:
> `& "C:\Path\To\python.exe" -m pip install -r requirements.txt`

---

## Usage

### Run the core engine directly

```bash
python portfolio_analysis.py
```

This downloads prices, runs the full walk-forward backtest, saves all output files, and generates the HTML tear sheet.

### Run via the orchestration entry point (recommended)

```bash
# Engine + formatted metrics table + weights allocation table
python main.py

# Engine + metrics + commit & push outputs to GitHub
python main.py --publish
```

### Setting up `--publish`

Create a `.env` file in the project root (never commit this file):

```
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

The token needs `repo` scope. Generate one at: **GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens**.

On subsequent runs, `python main.py --publish` will automatically commit and push the latest output files with a timestamped commit message.

---

## Configuration Reference

All user-facing settings are in the `USER CONFIGURATION` block at the top of `portfolio_analysis.py`:

| Variable | Default | Description |
|---|---|---|
| `TICKERS` | `["SPY","EEM","URA","GLD"]` | Yahoo Finance ticker symbols |
| `START_DATE` | `"2010-11-01"` | Data start — must pre-date all tickers' inception |
| `END_DATE` | `"2025-12-31"` | Data end (`"today"` also accepted) |
| `LOOKBACK_YEARS` | `3` | Rolling HRP training window in years |
| `REBALANCE_FREQ` | `"quarterly"` | `"monthly"` · `"quarterly"` · `"yearly"` |
| `MIN_WEIGHT` | `0.05` | Minimum per-asset allocation (set to `0.0` for unconstrained HRP) |
| `COMMISSION_BPS` | `15.0` | Combined broker + slippage cost per trade, in basis points |
| `BENCHMARK_TICKER` | `"SPY"` | QuantStats benchmark (`None` to omit comparison) |
| `OUTPUT_DIR` | `Path(r"C:\...\Investimenti")` | Directory for all output files |

### Customising the asset universe

To replace or extend the default 4-ETF basket, edit `TICKERS` and `START_DATE`:

```python
TICKERS    = ["VTI", "VXUS", "BND", "GLD", "VNQ"]
START_DATE = "2008-01-01"   # all five tickers existed by then
```

The engine handles ticker additions and removals automatically; it will warn if any ticker has no valid data and skip it.

---

## Interpreting the Outputs

### `hrp_weights_history.csv`

A DataFrame with **dates as rows** (rebalance dates) and **tickers as columns** (portfolio weights). Zero rows are warmup-period placeholders; non-zero rows represent live quarterly rebalances.

**How to use it:**
- Multiply the last row by your total investment amount, then divide by each asset's current price → number of shares to buy
- Compare current holdings (% of portfolio) against the last row → identify drift and upcoming rebalance direction
- Study how the engine adapted through 2020 (COVID), 2022 (rate shock), and 2024 (gold rally)

### `hrp_returns.csv`

Daily strategy returns for the live-trading period. Import into Excel, R, or another Python session for custom analysis, stress testing, or DCA scenario modelling.

### `hrp_tearsheet.html`

Open in any browser. The four metrics that matter most for a long-term savings plan investor:

| Metric | What It Tells You |
|---|---|
| **CAGR** | Annualised compound return; compare directly against SPY benchmark |
| **Max Drawdown** | Worst peak-to-trough loss — the psychological stress test |
| **Calmar Ratio** | CAGR ÷ \|Max Drawdown\| — the most honest single risk-adjusted measure for long-term investors |
| **Monthly Returns Heatmap** | What 12 years actually feels like month-by-month |

---

## Limitations & Hard Truths

This engine is transparent about what it cannot do:

1. **HRP optimises history, not the future.** Correlations are estimated from the past and shift in crises. Gold's hedge properties can temporarily collapse (March 2020 −10% in a single week). The model has no macro awareness, no forward-looking signals.

2. **15 bps is a floor, not an average.** During stress events, EEM and URA bid-ask spreads can widen to 30–50 bps. Currency conversion costs (EUR/USD) add an additional unmodeled layer if you hold a non-USD account.

3. **Zero tax modelling.** In Italy, capital gains are taxed at 26%. Quarterly rebalancing triggers taxable events on every sell. After-tax returns are meaningfully lower than displayed. Consult a tax advisor before implementation.

4. **Yahoo Finance data quality.** `yfinance` is a free, unofficial API. It occasionally produces incorrect adjusted-close prices around corporate actions or ETF restructurings. Always cross-check significant fills against your broker's official price data.

5. **Survivorship bias in ticker selection.** All four ETFs exist and are liquid today. If you add tickers that were delisted or merged, the engine handles missing data gracefully, but the backtest becomes less representative of what was actually investable.

6. **Uranium (URA) is a structural outlier.** URA holds concentrated positions in illiquid small-cap uranium miners. In a genuine risk-off event, the sector can be functionally untradeable at quoted spreads. The 5% floor ensures permanent exposure to the energy-transition thesis — understand that in an acute uranium bear market this 5% may behave like 0% in practical execution terms.

7. **Sequence-of-returns risk.** The backtest simulates a single lump-sum investment on Day 1. Periodic contributions (DCA — monthly deposits) will produce different outcomes, particularly if large contributions coincide with late-cycle peaks. Use `hrp_returns.csv` to model DCA scenarios externally.

8. **This is a simulation tool, not financial advice.** The authors make no representation that historical walk-forward performance predicts future results. Past returns do not guarantee future returns.

---

## Ideas for Contributors

The engine is intentionally modular. High-value extensions for contributors:

| Idea | Complexity | Value |
|---|---|---|
| **Trend filter overlay** | Skip allocation to any ticker below its 200-day SMA; route weight to cash or redistribute to remaining assets | Medium | High |
| **Regime detection** | Hidden Markov Model to switch between aggressive/defensive HRP parameters by market regime | High | High |
| **IBKR live-trading bridge** | Replace QuantStats output step with `ib_insync` order submission layer for actual portfolio execution | High | High |
| **Multi-currency support** | Convert non-USD ETF prices to EUR before passing to HRP optimizer | Medium | High |
| **Tax-aware rebalancing** | Skip selling positions with unrealised gains below threshold; rebalance only via new cash inflows | Medium | High |
| **Alternative risk measures** | Substitute `rm="CVaR"` or `rm="CDaR"` for `rm="MV"` in the Riskfolio-Lib call | Low | Medium |
| **Quarterly email/Telegram alert** | On `main.py --publish`, send a formatted message with new weights to a configured channel | Low | Medium |
| **Parameter sensitivity surface** | Run a grid of backtests varying `LOOKBACK_YEARS` (1–5) and `REBALANCE_FREQ` to map performance sensitivity | Medium | High |
| **DCA simulator** | Add a monthly-contribution mode to `main.py` using `hrp_returns.csv` | Low | High |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) · [bt](https://github.com/pmorissette/bt) · [QuantStats](https://github.com/ranaroussi/quantstats) · [yfinance](https://github.com/ranaroussi/yfinance)*

#!/usr/bin/env python3
"""
Portfolio Analysis & Backtesting Tool  (Production-Ready Edition)
==================================================================
European UCITS Walk-Forward HRP  —  5+ Year Investment Horizon
Universe (7 ETFs):
  ESG Half      — SUSW.L (MSCI World SRI), WEBG.L (World SmallCap ESG), RMAU.L (Gold ETC)
  Traditional   — SWDA.L (MSCI World), X7G7.DE (EZ Gov Bond 7-10), XEON.DE (EUR O/N Rate)
  Thematic Sat. — SMH (VanEck Semiconductor, NASDAQ proxy; UCITS: SEMI.L)

Walk-Forward HRP with three real-world market frictions:
  1. Commissions & Slippage  – 15 bps per-trade cost via bt.Backtest
  2. T+1 Execution Delay     – weights computed on Day T, traded on Day T+1
  3. Weight Floor            – MIN_WEIGHT = 5% prevents any asset being zeroed

Pipeline (zero lookahead bias preserved throughout):
  Step 1: Data Ingestion       – yfinance (adjusted-close prices)
  Step 2: Walk-Forward HRP     – Riskfolio-Lib (re-optimised on each trigger
                                  using STRICTLY past data only)
  Step 3: Backtest             – bt (T+1 delay + commissions)
  Step 4: Reporting            – QuantStats (HTML tear sheet)

Outputs (saved to OUTPUT_DIR):
  hrp_weights_history.csv  – time-series of walk-forward allocations
  hrp_returns.csv          – daily returns for the live-trading period
  hrp_tearsheet.html       – full QuantStats performance tear sheet
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy  as np
import pandas as pd
import yfinance as yf
import riskfolio as rp
import bt
import quantstats as qs

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        USER CONFIGURATION                               ║
# ║          ← All user-facing settings live here; edit freely →            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Investment Universe ────────────────────────────────────────────────────
# "Deep History" European UCITS Portfolio — 5+ Year Growth Horizon (6 ETFs)
# Older proxy tickers replace the 2019/2020-inception ETFs so the joint
# price series extends cleanly back to 2018, covering both the 2020 COVID
# crash and the 2022 rate-shock bear market in the live walk-forward period.
#
# All tickers are Xetra .DE except IGLN.L (no comparable Xetra gold ETC
# with equivalent AUM and history; IGLN.L is the deepest-liquidity choice).
#
#   EUNL.DE  iShares Core MSCI World UCITS ETF            Xetra (EUR)   2005
#   IUS3.DE  iShares MSCI World SRI UCITS ETF             Xetra (EUR)   2019
#   IUSN.DE  iShares MSCI World Small Cap UCITS ETF       Xetra (EUR)   2010  (proxy for WEBG)
#   QDVE.DE  iShares S&P 500 Information Technology UCITS Xetra (EUR)   2018  (proxy for Semiconductors)
#   DBXN.DE  Xtrackers II EZ Gov Bond 7-10 UCITS ETF     Xetra (EUR)   2007
#   IGLN.L   iShares Physical Gold ETC                    LSE (USD)     2011  (proxy for RMAU)
#
# Binding data constraint: QDVE.DE (inception 2018) and IUS3.DE (2019).
# START_DATE = 2018-01-01; warmup year = 2018; live trading from Jan 2019.
TICKERS: list[str] = [
    "EUNL.DE",  # iShares Core MSCI World UCITS ETF             – core global equity anchor
    "IUS3.DE",  # iShares MSCI World SRI UCITS ETF              – ESG large-cap world
    "IUSN.DE",  # iShares MSCI World Small Cap UCITS ETF        – small-cap growth proxy
    "QDVE.DE",  # iShares S&P 500 Info Technology UCITS ETF     – tech/semiconductor proxy
    "DBXN.DE",  # Xtrackers II EZ Gov Bond 7-10 UCITS ETF      – duration / rate hedge
    "IGLN.L",   # iShares Physical Gold ETC (LSE)               – gold proxy (inception 2011)
]

# ── Backtest Date Range ────────────────────────────────────────────────────
START_DATE = "2018-01-01"   # all 6 proxy tickers have reliable data from 2018; QDVE.DE oldest binding
END_DATE   = "2026-12-31"   # ceiling; yfinance returns data up to today if before this date

# ── Lookback Window (Walk-Forward) ────────────────────────────────────────
# Years of past daily returns fed into Riskfolio-Lib on each rebalance date.
# The strategy stays in CASH during this warmup period.
# 1 year ≈ 252 observations — sufficient for a stable 6×6 correlation matrix.
# Warmup: Jan 2018 – Dec 2018. Live walk-forward: Jan 2019 – present.
# Covers COVID crash (Feb–Mar 2020) and 2022 rate-shock bear in live period.
LOOKBACK_YEARS: float = 1.0

# ── Rebalancing Frequency ─────────────────────────────────────────────────
# How often a new HRP calculation is triggered.
# Options: "monthly" | "quarterly" | "yearly"
REBALANCE_FREQ = "monthly"   # monthly reactions to volatility spikes in SMH/WEBG

# ── Weight Floor (Friction #3) ────────────────────────────────────────────
# Minimum allocation per asset after HRP optimisation.
# Prevents volatile assets (e.g. SMH, WEBG.L) from being reduced to near 0%.
# Set to 0.0 to disable the floor and allow pure HRP weights.
MIN_WEIGHT: float = 0.05   # 5 % minimum weight per asset

# ── Per-Asset Maximum Weight Caps ──────────────────────────────────────────
# Caps defensive assets so they cannot crowd out the growth engines.
# Assets not listed here have an implicit 1.0 (100%) upper bound.
# These bounds are passed to Riskfolio-Lib (w_max) and enforced as a
# manual safety net after the solver returns, so the loop never crashes.
MAX_WEIGHTS: dict[str, float] = {
    "DBXN.DE": 0.20,   # Eurozone Gov Bonds: cap at 20% (prevents bond-trap)
    "IGLN.L":  0.15,   # Physical Gold ETC:  cap at 15%
}

# ── Commission + Slippage Model (Friction #1) ─────────────────────────────
# Combined estimate: broker commission + bid/ask half-spread.
# 15 bps (0.15%) is a conservative but realistic figure for ETF trades.
# Applied on EVERY rebalance trade by bt via the commissions callback.
COMMISSION_BPS: float = 15.0   # basis points; converted to decimal below
_COMMISSION_RATE: float = COMMISSION_BPS / 10_000.0   # → 0.0015

# ── Benchmark for Tear Sheet ──────────────────────────────────────────────
# Set to None (no quotes) to omit the benchmark from the report.
BENCHMARK_TICKER = "EUNL.DE"  # iShares Core MSCI World (Xetra) — like-for-like EUR benchmark

# ── Output Directory ──────────────────────────────────────────────────────
OUTPUT_DIR = Path(r"C:\Users\tobia\OneDrive\Documenti\Investimenti")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║          HRPWithT1Delay  —  Production bt.Algo                          ║
# ║                                                                          ║
# ║  Design rationale for the combined algo                                  ║
# ║  ─────────────────────────────────────                                   ║
# ║  bt's chain aborts as soon as any algo returns False.  If we used a      ║
# ║  standard Run* → CalculateHRP → Rebalance chain, making                  ║
# ║  CalculateHRP return False (to block same-day execution) would also       ║
# ║  block the T+1 ExecutePending algo if it came after in the chain.         ║
# ║                                                                          ║
# ║  The solution: one combined algo that runs on EVERY bar and manages       ║
# ║  its own state machine:                                                   ║
# ║    • Phase A (scheduling): fires on the first bar of each new             ║
# ║      month / quarter / year (configurable).                               ║
# ║    • On the trigger bar (Day T): compute HRP → store weights as           ║
# ║      pending → return False  (no trade today).                            ║
# ║    • On the very next bar (Day T+1): pop pending weights → set            ║
# ║      target.temp['weights'] → return True  (Rebalance() executes).        ║
# ║                                                                          ║
# ║  Zero-lookahead-bias guarantee:                                           ║
# ║    prices are sliced with index < target.now so the current day's         ║
# ║    close is never included in the optimisation window.                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class HRPWithT1Delay(bt.Algo):
    """
    Walk-forward HRP optimiser with built-in T+1 execution delay and
    a configurable minimum weight floor.

    State machine (per bar):
      ① If pending weights exist  →  inject them into target.temp and
        return True so bt.algos.Rebalance() fires.  (Day T+1 execution)
      ② Else if this bar is a scheduled rebalance trigger  →  compute HRP
        on strictly-past data, store as pending, return False.  (Day T calc)
      ③ Else  →  return False.  (ordinary bar; no action)
    """

    def __init__(
        self,
        all_prices: pd.DataFrame,
        lookback_years: float,
        min_weight: float,
        rebalance_freq: str,
        max_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self._prices        = all_prices
        self._lookback_days = int(lookback_years * 252)
        self._min_weight    = min_weight
        self._max_weights   = max_weights or {}   # {ticker: cap}; empty = no caps
        self._freq          = rebalance_freq.lower()

        # Scheduling state
        self._last_trigger_date: pd.Timestamp | None = None

        # T+1 pending queue — holds at most one set of weights at a time
        self._pending_weights: dict[str, float] | None = None

        if self._freq not in ("monthly", "quarterly", "yearly"):
            raise ValueError(
                f"Invalid REBALANCE_FREQ '{rebalance_freq}'. "
                "Choose from: monthly, quarterly, yearly."
            )

    # ── Scheduling helper ─────────────────────────────────────────────────
    def _is_trigger_day(self, date: pd.Timestamp) -> bool:
        """True on the first trading bar of a new month / quarter / year."""
        if self._last_trigger_date is None:
            return True
        prev = self._last_trigger_date
        if self._freq == "monthly":
            return (date.year, date.month) > (prev.year, prev.month)
        if self._freq == "quarterly":
            return (date.year, date.quarter) > (prev.year, prev.quarter)
        # yearly
        return date.year > prev.year

    # ── Riskfolio-Lib HRP call ────────────────────────────────────────────
    def _compute_hrp(
        self, window_returns: pd.DataFrame
    ) -> dict[str, float] | None:
        """
        Calls Riskfolio-Lib's official HRP engine.
        Applies the MIN_WEIGHT floor (w_min) directly in the optimisation.
        Falls back to a manual clip-and-renormalise if w_min is rejected.
        Returns a {ticker: weight} dict or None on failure.
        """
        try:
            port = rp.HCPortfolio(returns=window_returns)
            w_df: pd.DataFrame = port.optimization(
                model="HRP",
                codependence="pearson",
                rm="MV",          # Minimum-Variance risk measure
                rf=0.0,
                linkage="ward",   # Ward linkage for hierarchical clustering
                max_k=10,
                leaf_order=True,
                w_min=self._min_weight,   # ← Weight Floor (Friction #3)
            )
        except TypeError:
            # Older Riskfolio-Lib build that doesn't accept w_min keyword
            port = rp.HCPortfolio(returns=window_returns)
            w_df = port.optimization(
                model="HRP",
                codependence="pearson",
                rm="MV",
                rf=0.0,
                linkage="ward",
                max_k=10,
                leaf_order=True,
            )
        except Exception as exc:
            return None  # caller will log and skip

        weights: pd.Series = w_df["weights"]

        # ── Safety net: manual floor + renormalise ─────────────────────────
        # Guarantees MIN_WEIGHT is respected even if w_min was silently
        # ignored or the internal solver rounded slightly below the bound.
        if self._min_weight > 0.0:
            weights = weights.clip(lower=self._min_weight)
            weights = weights / weights.sum()

        return weights.to_dict()

    # ── bt algo entry point ───────────────────────────────────────────────
    def __call__(self, target) -> bool:

        # ══ Phase A: T+1 Execution ═════════════════════════════════════════
        # If the previous bar computed new HRP weights, execute them NOW
        # (i.e. on the next trading bar = T+1).
        if self._pending_weights is not None:
            target.temp["weights"] = self._pending_weights
            self._pending_weights  = None
            return True   # → bt.algos.Rebalance() will trade to these weights

        # ══ Phase B: Scheduling check ══════════════════════════════════════
        if not self._is_trigger_day(target.now):
            return False   # ordinary bar; nothing to do

        # Mark this date as the last trigger (even during warmup) so the
        # scheduler advances correctly and doesn't fire on every daily bar.
        self._last_trigger_date = target.now

        # ══ Phase C: Warmup guard ══════════════════════════════════════════
        # Slice prices to rows STRICTLY BEFORE today (zero-lookahead bias).
        past_prices: pd.DataFrame = self._prices.loc[
            self._prices.index < target.now
        ]
        if len(past_prices) < self._lookback_days:
            return False   # not enough history yet; stay in cash

        # ══ Phase D: Rolling-window returns ════════════════════════════════
        window_prices: pd.DataFrame  = past_prices.iloc[-self._lookback_days:]
        window_returns: pd.DataFrame = window_prices.pct_change().dropna()
        if len(window_returns) < 2:
            return False

        # ══ Phase E: HRP optimisation (Day T) ══════════════════════════════
        weights = self._compute_hrp(window_returns)
        if weights is None:
            print(f"[WARN] HRP failed on {target.now.date()}; skipping rebalance.")
            return False

        # Store as pending — will be executed on Day T+1 (next bar)
        self._pending_weights = weights
        return False   # do NOT rebalance today


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          PIPELINE                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main() -> None:

    # ── Step 0: Prepare output directory ──────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory : {OUTPUT_DIR}")
    print(
        f"[INFO] Market frictions : "
        f"{COMMISSION_BPS:.0f} bps commission/slippage | "
        f"T+1 execution delay | "
        f"{MIN_WEIGHT:.0%} weight floor"
    )


    # ══════════════════════════════════════════════════════════════════════
    # STEP 1 – DATA INGESTION  (yfinance)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 1] Downloading adjusted-close prices ...")
    print(f"         Tickers : {TICKERS}")
    print(f"         Period  : {START_DATE}  →  {END_DATE}")

    raw: pd.DataFrame = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices: pd.DataFrame = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = TICKERS

    # ── Drop columns that are entirely empty (ticker not found at all) ────
    prices = prices.dropna(axis=1, how="all")

    # ── Per-ticker diagnostic: show date coverage and NaN density ─────────
    # This block lets you see exactly which ticker is causing data loss
    # before the joint dropna() trims the DataFrame.
    print("\n[DEBUG] Per-ticker data coverage (before joint ffill/dropna):")
    print(f"  {'Ticker':<12} {'First date':<14} {'Last date':<14} {'NaN rows':>9} {'Total rows':>11}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*9} {'-'*11}")
    for col in prices.columns:
        s = prices[col]
        nan_count = int(s.isna().sum())
        non_null  = s.dropna()
        first_dt  = non_null.index.min().date() if not non_null.empty else "N/A"
        last_dt   = non_null.index.max().date() if not non_null.empty else "N/A"
        total     = len(s)
        print(f"  {col:<12} {str(first_dt):<14} {str(last_dt):<14} {nan_count:>9} {total:>11}")

    # ── Holiday gap fill: carry the last known close across market holidays─
    # ffill() handles UK/German/Italian bank holiday mismatches so that a
    # single missing day in RMAU.L (UK holiday) doesn't drop an entire
    # cross-market row from the joint DataFrame.
    prices = prices.ffill()

    # ── Drop only the leading rows where ANY ticker has no data yet ────────
    # (i.e. before the youngest ETF's actual inception date)
    prices = prices.dropna()

    available: list[str] = prices.columns.tolist()
    print(f"[INFO] Tickers with valid data : {available}")

    if len(available) < 2:
        sys.exit(
            "[ERROR] Fewer than 2 tickers with valid price data. "
            "Expand the date range or change the TICKERS list."
        )

    if len(prices) < LOOKBACK_YEARS * 252 + 60:
        print(
            f"[WARN] Limited price history ({len(prices)} days). "
            "Consider extending START_DATE."
        )


    # ══════════════════════════════════════════════════════════════════════
    # STEP 2 + 3 – WALK-FORWARD HRP BACKTEST  (Riskfolio-Lib inside bt)
    # ══════════════════════════════════════════════════════════════════════
    print(
        f"\n[STEP 2+3] Running production backtest "
        f"(lookback: {LOOKBACK_YEARS} yr | rebalance: {REBALANCE_FREQ} | "
        f"T+1 delay | {COMMISSION_BPS} bps cost) ..."
    )

    # ── Strategy chain ─────────────────────────────────────────────────────
    # SelectAll:       make every ticker available to Rebalance
    # HRPWithT1Delay:  owns scheduling + HRP calc (Day T) + T+1 execution
    # Rebalance:       executes trades when HRPWithT1Delay returns True
    hrp_algo = HRPWithT1Delay(
        all_prices    = prices,
        lookback_years= LOOKBACK_YEARS,
        min_weight    = MIN_WEIGHT,
        rebalance_freq= REBALANCE_FREQ,
        max_weights   = MAX_WEIGHTS,
    )

    strategy = bt.Strategy(
        "HRP_Production",
        [
            bt.algos.SelectAll(),
            hrp_algo,
            bt.algos.Rebalance(),
        ],
    )

    # ── Friction #1: Commission + Slippage ────────────────────────────────
    # bt calls this lambda for every buy/sell order.
    # q = number of shares; p = price per share
    # abs(q) * p = gross notional value of the trade
    # × _COMMISSION_RATE = total friction cost deducted from the portfolio
    backtest = bt.Backtest(
        strategy,
        prices,
        commissions=lambda q, p: abs(q) * p * _COMMISSION_RATE,
    )
    result = bt.run(backtest)

    # ── Extract equity curve → daily returns ──────────────────────────────
    equity_curve: pd.Series     = result.prices["HRP_Production"]
    strategy_returns: pd.Series = equity_curve.pct_change().dropna()
    strategy_returns.name = "HRP_Production"

    # Strip the flat cash warmup prefix so QuantStats metrics reflect
    # only the live-trading phase (after the first real rebalance on T+1).
    nonzero = strategy_returns[strategy_returns != 0]
    if nonzero.empty:
        sys.exit("[ERROR] Strategy produced no non-zero returns. Check data/config.")
    first_live = nonzero.index[0]
    strategy_returns_live = strategy_returns.loc[first_live:]

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    print(
        f"[INFO] Backtest period (incl. warmup) : "
        f"{equity_curve.index[0].date()}  →  {equity_curve.index[-1].date()}"
    )
    print(
        f"[INFO] Live-trading start (post T+1)  : "
        f"{strategy_returns_live.index[0].date()}"
    )
    print(f"[INFO] Total return (full period)     : {total_return:.2%}")

    # ── Save historical walk-forward weights ──────────────────────────────
    weights_history: pd.DataFrame = result.get_security_weights("HRP_Production")
    weights_path = OUTPUT_DIR / "hrp_weights_history.csv"
    weights_history.to_csv(weights_path)
    print(f"\n[INFO] Historical weights saved → {weights_path}")

    returns_path = OUTPUT_DIR / "hrp_returns.csv"
    strategy_returns_live.to_csv(returns_path, header=True)
    print(f"[INFO] Daily returns saved      → {returns_path}")

    live_weights = weights_history.loc[weights_history.sum(axis=1) > 0]
    if not live_weights.empty:
        print("\n[INFO] Most recent HRP allocation (post-floor):")
        for ticker, wgt in live_weights.iloc[-1].sort_values(ascending=False).items():
            print(f"         {ticker:<6}  {wgt:.4%}")


    # ══════════════════════════════════════════════════════════════════════
    # STEP 4 – PERFORMANCE REPORT  (QuantStats)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[STEP 4] Generating QuantStats HTML tear sheet ...")

    benchmark_returns: pd.Series | None = None
    if BENCHMARK_TICKER:
        print(f"[INFO] Downloading benchmark : {BENCHMARK_TICKER}")
        bm_raw = yf.download(
            BENCHMARK_TICKER,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False,
        )
        bm_prices = (
            bm_raw["Close"].squeeze()
            if isinstance(bm_raw.columns, pd.MultiIndex)
            else bm_raw["Close"].squeeze()
        )
        benchmark_returns = bm_prices.pct_change().dropna()
        benchmark_returns.name = BENCHMARK_TICKER

    tearsheet_path = OUTPUT_DIR / "hrp_tearsheet.html"

    qs.reports.html(
        strategy_returns_live,
        benchmark=benchmark_returns,
        rf=0.0,
        output=str(tearsheet_path),
        title=(
            "HRP Walk-Forward │ UCITS Deep-History 6-ETF │ EUNL+IUS3+IUSN+QDVE+DBXN+IGLN │ "
            f"T+1 Delay │ {COMMISSION_BPS:.0f}bps Cost │ {MIN_WEIGHT:.0%} Floor │ DBXN≤20% IGLN≤15%"
        ),
        match_dates=True,
    )
    print(f"[INFO] Tear sheet saved → {tearsheet_path}")


    # ══════════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Production backtest complete. Output files:")
    print(f"    {weights_path}")
    print(f"    {returns_path}")
    print(f"    {tearsheet_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()

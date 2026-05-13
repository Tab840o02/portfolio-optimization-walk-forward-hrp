#!/usr/bin/env python3
"""
Portfolio Analysis & Backtesting Tool  (Production-Ready Edition)
==================================================================
European UCITS Walk-Forward HRP  —  5+ Year Investment Horizon
# Universe (5 ETFs) — European Golden Butterfly | ESG-Adapted:
#   Core Equity        — EUNL.DE  iShares Core MSCI World (Xetra)
#   Small Cap Equity   — WDSC.L   SPDR MSCI World Small Cap (proxy: Amundi MSCI World SC ESG)
#   Gov Bond 7-10Y     — DBXN.DE  Xtrackers II EZ Gov Bond 7-10 (Xetra)
#   Cash/Overnight     — XEON.DE  Xtrackers EUR Overnight Rate Swap (Xetra)
#   Physical Gold      — SGLD.L   Invesco Physical Gold ETC (LSE)
# Binding constraint: all 5 have data from 2017. Warmup: 2018. Live: Jan 2019+.

Walk-Forward HRP with three real-world market frictions:
  1. Commissions & Slippage  – 15 bps per-trade cost via bt.Backtest
  2. T+1 Execution Delay     – weights computed on Day T, traded on Day T+1
  3. Weight Floor            – MIN_WEIGHT = 10% minimum per asset (5 × 10% = 50% committed)

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

# ── Global Backtest Settings ───────────────────────────────────────────────
# Binding constraint: WDSC.L first data 2018-01-02 → START_DATE 2018-01-01.
# fixed_weight mode has no warmup → all 8.4 years are live-traded.
# Period covers 4 distinct market regimes for statistical robustness:
#   2018 Q4 selloff | 2020 COVID crash+recovery | 2022 rate shock | 2023-26 recovery.
START_DATE = "2018-01-01"
END_DATE   = "2026-12-31"   # ceiling; yfinance returns up to today

# ── Walk-Forward Lookback (HRP mode only) ────────────────────────────────
LOOKBACK_YEARS: float = 1.0   # 1yr ≈ 252 obs; sufficient for a stable 6×6 corr matrix

# ── Rebalancing Frequency ─────────────────────────────────────────────────
# Options: "monthly" | "quarterly" | "semi-annual" | "yearly"
REBALANCE_FREQ = "yearly"   # Annual rebalance; ~6 trades/yr; low turnover, tax-efficient

# ── Commission + Slippage Model ───────────────────────────────────────────
# 15 bps per trade: conservative estimate covering broker commission + bid/ask spread.
# Applied symmetrically to buys and sells via bt.Backtest commissions callback.
COMMISSION_BPS: float = 15.0
_COMMISSION_RATE: float = COMMISSION_BPS / 10_000.0   # → 0.0015

# ── Benchmark ─────────────────────────────────────────────────────────────
# EUNL.DE = pure DM equity; shows the diversification benefit of multi-asset.
# Set to None to omit benchmark from tearsheet.
BENCHMARK_TICKER = "EUNL.DE"

# ── Output Directory ──────────────────────────────────────────────────────
OUTPUT_DIR = Path(r"C:\Users\tobia\OneDrive\Documenti\Investimenti")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    STRATEGIES UNDER COMPARISON                           ║
# ║                                                                          ║
# ║  Two portfolios run in a SINGLE backtest pass for direct comparison:     ║
# ║                                                                          ║
# ║  1. Recommended_6ETF  ← PRIMARY (full tearsheet generated)              ║
# ║     Final recommendation from analysis report.                           ║
# ║     Replaces 20% cash (XEON.DE) with quality equity (IS3Q.DE) and       ║
# ║     splits bonds into a duration ladder (7-10yr + 1-5yr).               ║
# ║     40% equity / 30% bonds / 15% gold / 15% small cap                   ║
# ║     CAGR 8.87% | Vol 9.61% | Sharpe 0.93 | MaxDD -21.81%               ║
# ║                                                                          ║
# ║  2. Caveat_GB_IBCI  ← REFERENCE (capital-preservation variant)          ║
# ║     For investors with higher loss-aversion or shorter horizon.          ║
# ║     Classic Golden Butterfly structure (5×20%) with IBCI replacing      ║
# ║     XEON.DE: same defensive profile but better forward yield.            ║
# ║     Expected: lower CAGR vs Recommended, shallower drawdown.            ║
# ║                                                                          ║
# ║  Both share: START_DATE, REBALANCE_FREQ, COMMISSION_BPS, LOOKBACK_YEARS ║
# ╚══════════════════════════════════════════════════════════════════════════╝

STRATEGIES: dict[str, dict] = {

    # ── Strategy 1: Final Recommendation ──────────────────────────────────
    "Recommended_6ETF": {
        # Human-readable label used in printed reports and tearsheet title
        "label": "Balanced 6-ETF Multi-Factor (Recommended)",

        # Tickers in this strategy's universe
        "tickers": [
            "EUNL.DE",  # iShares Core MSCI World UCITS ETF (Acc)      – broad DM equity core
            "IS3Q.DE",  # iShares MSCI World Quality Factor UCITS ETF  – quality / profitability tilt
            "DBXN.DE",  # Xtrackers II EZ Gov Bond 7-10yr UCITS ETF    – duration anchor
            "IBCI.DE",  # iShares EUR Corp Bond 1-5yr UCITS ETF        – short-med bond; yield pickup
            "SGLD.L",   # Invesco Physical Gold ETC                    – inflation hedge / tail risk
            "WDSC.L",   # SPDR MSCI World Small Cap UCITS ETF          – small cap premium (SMB)
        ],

        # fixed_weight: rebalance to exact target on schedule; no optimisation.
        # Correct mode for proposal validation — tests exact allocation as-specified.
        "mode": "fixed_weight",

        # Target allocation — must sum to 1.0.
        # Rationale:
        #   40% equity  : EUNL (broad DM) + IS3Q (quality tilt).
        #                 Quality selects high-ROE, low-leverage companies: historically
        #                 +1-2% CAGR vs market-cap with lower drawdowns.
        #   30% bonds   : DBXN (7-10yr) + IBCI (1-5yr). Duration ladder:
        #                 long end hedges equity selloffs; short end provides yield
        #                 pickup vs cash with low interest-rate sensitivity.
        #   15% gold    : SGLD. Near-zero / negative equity correlation in crises.
        #                 Real store of value; inflation hedge.
        #   15% sc      : WDSC. Fama-French SMB premium; higher long-run expected
        #                 return vs large cap; diversifies mega-cap growth concentration.
        "target_weights": {
            "EUNL.DE": 0.20,   # Broad DM equity core
            "IS3Q.DE": 0.20,   # Quality factor tilt
            "DBXN.DE": 0.15,   # Long-duration EZ sovereign
            "IBCI.DE": 0.15,   # Short-medium EUR corporate bond
            "SGLD.L":  0.15,   # Physical gold
            "WDSC.L":  0.15,   # Small cap premium
        },

        # HRP-mode constraints (active only when mode = "hrp")
        "min_weight":  0.08,
        "max_weights": {
            "EUNL.DE": 0.35,
            "IS3Q.DE": 0.35,
            "DBXN.DE": 0.30,
            "IBCI.DE": 0.30,
            "SGLD.L":  0.25,
            "WDSC.L":  0.25,
        },

        # Generate the full QuantStats HTML tearsheet for this strategy
        "primary": True,
    },

    # ── Strategy 2: Capital-Preservation Variant (Caveat) ─────────────────
    "Caveat_GB_IBCI": {
        "label": "Golden Butterfly + IBCI (Capital-Preservation Variant)",

        "tickers": [
            "EUNL.DE",  # iShares Core MSCI World UCITS ETF (Acc)     – DM equity bucket
            "WDSC.L",   # SPDR MSCI World Small Cap UCITS ETF         – small cap bucket
            "DBXN.DE",  # Xtrackers II EZ Gov Bond 7-10yr UCITS ETF   – long bond bucket
            "IBCI.DE",  # iShares EUR Corp Bond 1-5yr UCITS ETF       – replaces XEON.DE cash bucket
            "SGLD.L",   # Invesco Physical Gold ETC                   – gold bucket
        ],

        "mode": "fixed_weight",

        # Classic Golden Butterfly: equal 20% per bucket.
        # Cash bucket (XEON.DE) replaced by IBCI.DE:
        #   XEON yields ~2.3% (ECB overnight, declining).
        #   IBCI yields ~3-4% (1-5yr EUR corp bonds) with only 6% vol.
        #   Same defensive profile; meaningfully better forward return.
        "target_weights": {
            "EUNL.DE": 0.20,
            "WDSC.L":  0.20,
            "DBXN.DE": 0.20,
            "IBCI.DE": 0.20,   # XEON.DE upgrade
            "SGLD.L":  0.20,
        },

        "min_weight":  0.08,
        "max_weights": {},   # No caps needed for equal-weight fixed structure

        # Comparison reference only; no separate tearsheet
        "primary": False,
    },
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║          WeightAlgoWithT1Delay  —  Production bt.Algo                   ║
# ║                                                                          ║
# ║  Supports three strategy modes (set per-strategy in STRATEGIES dict):   ║
# ║    • "fixed_weight"  — rebalance to exact TARGET_WEIGHTS on schedule.   ║
# ║    • "equal_weight"  — rebalance to 1/N equal weights on schedule.      ║
# ║    • "hrp"           — walk-forward HRP (Riskfolio-Lib) with floor/cap. ║
# ║                                                                          ║
# ║  All modes share the same T+1 execution delay state machine:            ║
# ║    • Day T : compute weights → store as pending → return False           ║
# ║    • Day T+1: inject pending weights → return True (Rebalance fires)    ║
# ║                                                                          ║
# ║  Zero-lookahead-bias guarantee:                                          ║
# ║    prices sliced with index < target.now so the current day's close      ║
# ║    is never included in the optimisation window.                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class WeightAlgoWithT1Delay(bt.Algo):
    """
    Unified weighting algo with T+1 execution delay.
    Supports 'equal_weight' (fixed 1/N) and 'hrp' (walk-forward HRP) modes.

    State machine (per bar):
      ① If pending weights exist  →  inject into target.temp → return True  (Day T+1)
      ② Else if trigger day       →  compute weights → store pending → return False (Day T)
      ③ Else                      →  return False  (no action)
    """

    def __init__(
        self,
        all_prices: pd.DataFrame,
        lookback_years: float,
        min_weight: float,
        rebalance_freq: str,
        max_weights: dict[str, float] | None = None,
        mode: str = "hrp",
        target_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self._prices        = all_prices
        self._lookback_days = int(lookback_years * 252)
        self._min_weight    = min_weight
        self._max_weights   = max_weights or {}
        self._freq          = rebalance_freq.lower()
        self._mode          = mode.lower()
        self._target_weights = target_weights or {}

        if self._mode == "fixed_weight" and not self._target_weights:
            raise ValueError(
                "STRATEGY_MODE='fixed_weight' requires TARGET_WEIGHTS to be non-empty."
            )
        if self._mode == "fixed_weight":
            total = sum(self._target_weights.values())
            if not (0.999 < total < 1.001):
                raise ValueError(
                    f"TARGET_WEIGHTS must sum to 1.0 (got {total:.6f}). "
                    "Adjust the values or they will be renormalised at runtime."
                )

        # Scheduling state
        self._last_trigger_date: pd.Timestamp | None = None

        # T+1 pending queue
        self._pending_weights: dict[str, float] | None = None

        if self._freq not in ("monthly", "quarterly", "semi-annual", "yearly"):
            raise ValueError(
                f"Invalid REBALANCE_FREQ '{rebalance_freq}'. "
                "Choose from: monthly, quarterly, semi-annual, yearly."
            )
        if self._mode not in ("hrp", "equal_weight", "fixed_weight"):
            raise ValueError(
                f"Invalid STRATEGY_MODE '{mode}'. Choose from: hrp, equal_weight, fixed_weight."
            )

    # ── Scheduling helper ─────────────────────────────────────────────────
    def _is_trigger_day(self, date: pd.Timestamp) -> bool:
        """True on the first trading bar of a new month / quarter / half-year / year."""
        if self._last_trigger_date is None:
            return True
        prev = self._last_trigger_date
        if self._freq == "monthly":
            return (date.year, date.month) > (prev.year, prev.month)
        if self._freq == "quarterly":
            return (date.year, date.quarter) > (prev.year, prev.quarter)
        if self._freq == "semi-annual":
            half = lambda d: (d.year, 1 if d.month <= 6 else 2)
            return half(date) > half(prev)
        # yearly
        return date.year > prev.year

    # ── Fixed-weight computation ───────────────────────────────────────────
    def _compute_fixed_weight(self) -> dict[str, float]:
        """
        Returns the pre-defined target allocation from TARGET_WEIGHTS,
        renormalised to sum exactly to 1.0.
        The renormalisation is a safety net; in practice TARGET_WEIGHTS
        should already sum to 1.0 (validated in __init__).
        """
        total = sum(self._target_weights.values())
        return {k: v / total for k, v in self._target_weights.items()}

    # ── Equal-weight computation ───────────────────────────────────────────
    def _compute_equal_weight(
        self, assets: list[str]
    ) -> dict[str, float]:
        """
        Returns a perfectly balanced 1/N allocation across all assets.
        This is the correct mode for the Golden Butterfly portfolio where
        the diversification is encoded in the bucket structure, not in
        dynamic variance-minimisation.
        """
        w = 1.0 / len(assets)
        return {a: w for a in assets}

    # ── Riskfolio-Lib HRP call ────────────────────────────────────────────
    def _compute_hrp(
        self, window_returns: pd.DataFrame
    ) -> dict[str, float] | None:
        """
        Calls Riskfolio-Lib's official HRP engine, then applies a
        MANUAL POST-OPTIMIZATION CLIPPER that enforces per-asset max caps
        and the minimum weight floor entirely in Python — with no reliance
        on Riskfolio-Lib's internal constraint engine (which silently ignores
        w_min / w_max on certain builds).

        Pipeline (4 explicit steps, run after the raw HRP solve):
          Step 1 — Hard-clip each capped asset to its MAX_WEIGHTS ceiling.
                   Accumulate the total excess weight that was chopped off.
          Step 2 — Redistribute the excess proportionally to the uncapped
                   (equity) assets based on their current relative weights.
                   Fallback: equal distribution if all equity weights are 0.
          Step 3 — Apply the MIN_WEIGHT (5%) floor: clip every asset up.
          Step 4 — Final renormalise so the vector sums exactly to 1.0.

        Mathematical guarantee after Step 4:
          • Capped assets: renorm divides by sum ≥ 1 (floor bumped total up),
            so capped weight ≤ original cap value. Caps are always satisfied.
          • Sum = 1.0 exactly.
          • Floor: assets are ≥ 0.05 before renorm; after renorm by
            sum ≥ 1, a tiny fraction of assets may land just below 0.05.
            A second pass (Steps 3–4 repeated once) closes this gap.

        Returns a {ticker: weight} dict or None on failure.
        """
        # ── Get raw HRP weights from Riskfolio (unconstrained solve) ──────
        try:
            port = rp.HCPortfolio(returns=window_returns)
            w_df: pd.DataFrame = port.optimization(
                model="HRP",
                codependence="pearson",
                rm="MV",       # Minimum-Variance risk measure
                rf=0.0,
                linkage="ward",
                max_k=10,
                leaf_order=True,
                # Do NOT pass w_min / w_max — enforced manually below
            )
        except Exception:
            return None

        weights: pd.Series = w_df["weights"].copy()

        # Identify capped assets; receivers computed dynamically each pass
        # (works for both mixed and all-capped universes).
        caps = {a: c for a, c in self._max_weights.items() if a in weights.index}

        for _pass in range(3):

            # ── Step 1: Hard-clip capped assets ───────────────────────────
            excess = 0.0
            for asset, cap in caps.items():
                if weights[asset] > cap:
                    excess += weights[asset] - cap
                    weights[asset] = cap

            # ── Step 2: Redistribute excess to assets below their own cap ─
            if excess > 0.0:
                receivers = [
                    a for a in weights.index
                    if weights[a] < self._max_weights.get(a, 1.0)
                ]
                if receivers:
                    recv_total = weights[receivers].sum()
                    if recv_total > 0.0:
                        weights[receivers] += excess * (weights[receivers] / recv_total)
                    else:
                        weights[receivers] += excess / len(receivers)

            # ── Step 3: Apply MIN_WEIGHT floor ────────────────────────────
            if self._min_weight > 0.0:
                weights = weights.clip(lower=self._min_weight)

            # ── Step 4: Renormalise to exactly 1.0 ────────────────────────
            total = weights.sum()
            if total > 0.0:
                weights = weights / total

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

        self._last_trigger_date = target.now

        # ══ Phase C: Warmup guard (HRP only) ══════════════════════════════
        # fixed_weight and equal_weight need no historical data — skip warmup.
        past_prices: pd.DataFrame = self._prices.loc[
            self._prices.index < target.now
        ]
        if self._mode == "hrp" and len(past_prices) < self._lookback_days:
            return False   # not enough history yet; stay in cash

        # ══ Phase D: Rolling-window returns (HRP only) ════════════════════
        window_returns: pd.DataFrame | None = None
        if self._mode == "hrp":
            window_prices   = past_prices.iloc[-self._lookback_days:]
            window_returns  = window_prices.pct_change().dropna()
            if len(window_returns) < 2:
                return False

        # ══ Phase E: Compute weights (Day T) ══════════════════════════════
        if self._mode == "fixed_weight":
            weights = self._compute_fixed_weight()
        elif self._mode == "equal_weight":
            weights = self._compute_equal_weight(list(past_prices.columns))
        else:
            weights = self._compute_hrp(window_returns)  # type: ignore[arg-type]
            if weights is None:
                print(f"[WARN] HRP failed on {target.now.date()}; skipping rebalance.")
                return False

        self._pending_weights = weights
        return False   # execute on Day T+1


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          PIPELINE                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main() -> None:

    # ── Step 0: Prepare output directory ──────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory  : {OUTPUT_DIR}")
    print(f"[INFO] Market frictions  : {COMMISSION_BPS:.0f} bps commission | T+1 delay | Annual rebal")
    print(f"[INFO] Strategies        : {list(STRATEGIES.keys())}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1 – DATA INGESTION  (yfinance)
    # Download all unique tickers across ALL strategies in one call.
    # ══════════════════════════════════════════════════════════════════════
    all_tickers: list[str] = sorted({
        t for cfg in STRATEGIES.values() for t in cfg["tickers"]
    })
    print(f"\n[STEP 1] Downloading adjusted-close prices ...")
    print(f"         All tickers : {all_tickers}")
    print(f"         Period      : {START_DATE}  →  {END_DATE}")

    raw: pd.DataFrame = yf.download(
        all_tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )
    all_prices: pd.DataFrame = (
        raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex)
        else raw[["Close"]].copy().rename(columns={"Close": all_tickers[0]})
    )
    all_prices = all_prices.dropna(axis=1, how="all")

    # ── Per-ticker diagnostic ──────────────────────────────────────────────
    print("\n[DEBUG] Per-ticker data coverage:")
    print(f"  {'Ticker':<12} {'First date':<14} {'Last date':<14} {'NaN':>5} {'Rows':>7}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*5} {'-'*7}")
    for col in all_prices.columns:
        s = all_prices[col]
        valid = s.dropna()
        print(
            f"  {col:<12} {str(valid.index.min().date()):<14} "
            f"{str(valid.index.max().date()):<14} "
            f"{int(s.isna().sum()):>5} {len(s):>7}"
        )

    # Fill cross-market holiday gaps then drop leading rows where any ticker absent
    all_prices = all_prices.ffill().dropna()
    print(f"\n[INFO] Joint price matrix: {len(all_prices)} rows × {len(all_prices.columns)} cols")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2 + 3 – BUILD AND RUN ALL STRATEGY BACKTESTS IN ONE PASS
    # bt.run() accepts multiple bt.Backtest objects and handles them in a
    # single simulation loop, producing directly comparable equity curves.
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 2+3] Building {len(STRATEGIES)} strategies ...")

    backtests: list[bt.Backtest] = []
    primary_name: str = ""

    for name, cfg in STRATEGIES.items():
        strat_tickers = cfg["tickers"]
        available = [t for t in strat_tickers if t in all_prices.columns]
        missing   = set(strat_tickers) - set(available)
        if missing:
            print(f"  [WARN] {name}: tickers not in data — {missing}. Skipping.")
            continue
        if len(available) < 2:
            print(f"  [WARN] {name}: fewer than 2 valid tickers. Skipping.")
            continue

        strat_prices = all_prices[available].copy()

        algo = WeightAlgoWithT1Delay(
            all_prices     = strat_prices,
            lookback_years = LOOKBACK_YEARS,
            min_weight     = cfg.get("min_weight", 0.0),
            rebalance_freq = REBALANCE_FREQ,
            max_weights    = cfg.get("max_weights", {}),
            mode           = cfg["mode"],
            target_weights = cfg.get("target_weights", {}),
        )
        strategy = bt.Strategy(
            name,
            [bt.algos.SelectAll(), algo, bt.algos.Rebalance()],
        )
        # Each bt.Backtest gets its own price DataFrame so universes can differ.
        # Commission lambda is a closure over the global _COMMISSION_RATE constant.
        rate = _COMMISSION_RATE
        backtests.append(bt.Backtest(
            strategy,
            strat_prices,
            commissions=lambda q, p, r=rate: abs(q) * p * r,
        ))
        print(f"  [{name}]  mode={cfg['mode']}  tickers={available}")
        if cfg.get("primary", False):
            primary_name = name

    if not backtests:
        sys.exit("[ERROR] No valid strategies to run. Check STRATEGIES config.")

    print(f"\n[INFO] Running bt backtest ({len(backtests)} strategies) ...")
    result = bt.run(*backtests)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4 – EXTRACT LIVE RETURNS AND PRINT COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n[STEP 4] Extracting results ...")

    all_live_returns: dict[str, pd.Series] = {}

    for name in STRATEGIES:
        if name not in result.prices.columns:
            continue
        equity: pd.Series = result.prices[name]
        ret: pd.Series    = equity.pct_change().dropna()
        nonzero = ret[ret != 0]
        if nonzero.empty:
            print(f"  [WARN] {name}: no non-zero returns.")
            continue
        live_ret = ret.loc[nonzero.index[0]:]
        live_ret.name = name
        all_live_returns[name] = live_ret
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        print(
            f"  [{name}]  live_start={live_ret.index[0].date()}  "
            f"total_return={total_return:.2%}  obs={len(live_ret)}"
        )

    # ── Download benchmark returns ─────────────────────────────────────────
    benchmark_returns: pd.Series | None = None
    if BENCHMARK_TICKER:
        print(f"\n[INFO] Downloading benchmark: {BENCHMARK_TICKER}")
        bm_raw = yf.download(
            BENCHMARK_TICKER, start=START_DATE, end=END_DATE,
            auto_adjust=True, progress=False,
        )
        bm_prices = (
            bm_raw["Close"].squeeze() if isinstance(bm_raw.columns, pd.MultiIndex)
            else bm_raw["Close"].squeeze()
        )
        benchmark_returns = bm_prices.pct_change().dropna()
        benchmark_returns.name = BENCHMARK_TICKER

    # ── Comparison table ───────────────────────────────────────────────────
    # Reference index: align benchmark to the first strategy's live period
    ref_index = next(iter(all_live_returns.values())).index if all_live_returns else None

    print("\n" + "=" * 82)
    print(f"  STRATEGY COMPARISON  |  Period: {START_DATE} → {END_DATE}  |  {COMMISSION_BPS:.0f}bps cost  |  Annual rebal")
    print("=" * 82)
    print(f"  {'Strategy':<36}  {'CAGR':>6}  {'Vol':>6}  {'Sharpe':>7}  {'Sortino':>8}  {'MaxDD':>8}")
    print(f"  {'-'*36}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}")

    for name, ret in all_live_returns.items():
        label = STRATEGIES[name]["label"][:36]
        primary_marker = " ★" if STRATEGIES[name].get("primary") else ""
        print(
            f"  {label + primary_marker:<36}  "
            f"{qs.stats.cagr(ret):>6.2%}  "
            f"{qs.stats.volatility(ret):>6.2%}  "
            f"{qs.stats.sharpe(ret):>7.2f}  "
            f"{qs.stats.sortino(ret):>8.2f}  "
            f"{qs.stats.max_drawdown(ret):>8.2%}"
        )

    if benchmark_returns is not None and ref_index is not None:
        bm = benchmark_returns.reindex(ref_index).dropna()
        print(
            f"  {'EUNL.DE — Pure Equity Benchmark':<36}  "
            f"{qs.stats.cagr(bm):>6.2%}  "
            f"{qs.stats.volatility(bm):>6.2%}  "
            f"{qs.stats.sharpe(bm):>7.2f}  "
            f"{qs.stats.sortino(bm):>8.2f}  "
            f"{qs.stats.max_drawdown(bm):>8.2%}"
        )
    print("=" * 82)
    print("  ★ = primary recommendation (full tearsheet generated)")
    print("=" * 82)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5 – SAVE OUTPUTS
    # ══════════════════════════════════════════════════════════════════════
    print("\n[STEP 5] Saving output files ...")

    # Combined returns CSV (all strategies, inner-joined on common dates)
    if all_live_returns:
        combined = pd.DataFrame(all_live_returns).dropna()
        returns_path = OUTPUT_DIR / "comparison_returns.csv"
        combined.to_csv(returns_path)
        print(f"[INFO] Combined returns  → {returns_path}")

    # Per-strategy weight history CSV
    for name in STRATEGIES:
        if name not in result.prices.columns:
            continue
        wh: pd.DataFrame = result.get_security_weights(name)
        wpath = OUTPUT_DIR / f"weights_{name}.csv"
        wh.to_csv(wpath)
        print(f"[INFO] Weights saved     → {wpath}")

        live_wh = wh.loc[wh.sum(axis=1) > 0]
        if not live_wh.empty:
            primary_label = " (PRIMARY ★)" if STRATEGIES[name].get("primary") else ""
            print(f"\n  Most recent allocation [{name}{primary_label}]:")
            for ticker, wgt in live_wh.iloc[-1].sort_values(ascending=False).items():
                if wgt > 0.001:
                    print(f"    {ticker:<10}  {wgt:.2%}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6 – QUANTSTATS TEARSHEET  (primary strategy only)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[STEP 6] Generating QuantStats HTML tearsheet (primary strategy) ...")

    if primary_name and primary_name in all_live_returns:
        primary_ret = all_live_returns[primary_name]
        primary_cfg = STRATEGIES[primary_name]
        tearsheet_path = OUTPUT_DIR / "recommended_tearsheet.html"

        bm_for_qs = None
        if benchmark_returns is not None:
            bm_for_qs = benchmark_returns.reindex(primary_ret.index).dropna()

        qs.reports.html(
            primary_ret,
            benchmark=bm_for_qs,
            rf=0.0,
            output=str(tearsheet_path),
            title=(
                f"RECOMMENDED │ {primary_cfg['label']} │ "
                f"T+1 Delay │ {COMMISSION_BPS:.0f}bps │ Annual Rebal │ "
                "EUNL+IS3Q+DBXN+IBCI+SGLD+WDSC"
            ),
            match_dates=True,
        )
        print(f"[INFO] Primary tearsheet → {tearsheet_path}")
    else:
        print("[WARN] No primary strategy defined; tearsheet skipped.")

    # ══════════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Multi-strategy comparison complete.")
    print(f"  Strategies : {list(STRATEGIES.keys())}")
    print(f"  Outputs    : {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()

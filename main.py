#!/usr/bin/env python3
"""
main.py
=======
Orchestration entry point for the Walk-Forward HRP Portfolio Engine.

Responsibilities:
  1. Runs portfolio_analysis.py (preserves it as a standalone, importable module).
  2. Reads hrp_returns.csv and prints a formatted key-metrics table.
  3. Reads hrp_weights_history.csv and prints a visual allocation bar chart.
  4. Optionally (--publish): commits all project files and the latest output
     files to the configured GitHub remote.

Usage:
  python main.py                # run engine + metrics table + weights bar chart
  python main.py --publish      # all of the above + git commit & push

Prerequisites for --publish:
  Create a .env file in this folder (never commit it; it is in .gitignore):
    GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
  Required GitHub token scope: "Contents" (read & write) on the target repo.
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
ENGINE_FILE = SCRIPT_DIR / "portfolio_analysis.py"
PYTHON      = sys.executable   # same interpreter that launched main.py

# Output files are written to portfolio_analysis.OUTPUT_DIR.
# We resolve it at import time so main.py stays in sync automatically.
try:
    import portfolio_analysis as _engine
    OUTPUT_DIR  = _engine.OUTPUT_DIR
except Exception:
    # Fallback: place outputs next to main.py (cross-platform default)
    OUTPUT_DIR = SCRIPT_DIR / "output"

WEIGHTS_CSV = OUTPUT_DIR / "hrp_weights_history.csv"
RETURNS_CSV = OUTPUT_DIR / "hrp_returns.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Run the engine
# ─────────────────────────────────────────────────────────────────────────────

def run_engine() -> int:
    """Invoke portfolio_analysis.py as a child process. Returns exit code."""
    proc = subprocess.run([PYTHON, str(ENGINE_FILE)], check=False)
    return proc.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Standardised metrics table
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics() -> None:
    """
    Read hrp_returns.csv and display key performance metrics computed via
    QuantStats — the same library used for the HTML tear sheet, ensuring
    consistent numbers between console output and the full report.
    """
    if not RETURNS_CSV.exists():
        print(f"[WARN] Returns file not found: {RETURNS_CSV}")
        return

    try:
        import quantstats as qs
    except ImportError:
        print("[WARN] quantstats not installed; skipping metrics table.")
        return

    returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True).squeeze()
    returns.name = "HRP_Production"

    # Silence QuantStats' internal deprecation warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cagr      = qs.stats.cagr(returns)
        sharpe    = qs.stats.sharpe(returns)
        sortino   = qs.stats.sortino(returns)
        max_dd    = qs.stats.max_drawdown(returns)
        calmar    = qs.stats.calmar(returns)
        vol       = qs.stats.volatility(returns)

    start = returns.index[0].date()
    end   = returns.index[-1].date()

    print("\n" + "─" * 57)
    print(f"  KEY PERFORMANCE METRICS  ({start} → {end})")
    print("─" * 57)
    print(f"  {'CAGR':<24} {cagr:>10.2%}")
    print(f"  {'Sharpe Ratio':<24} {sharpe:>10.2f}")
    print(f"  {'Sortino Ratio':<24} {sortino:>10.2f}")
    print(f"  {'Max Drawdown':<24} {max_dd:>10.2%}")
    print(f"  {'Calmar Ratio':<24} {calmar:>10.2f}")
    print(f"  {'Annual Volatility':<24} {vol:>10.2%}")
    print("─" * 57)
    print("  (full tear sheet → hrp_tearsheet.html)")
    print("─" * 57)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Allocation bar chart
# ─────────────────────────────────────────────────────────────────────────────

def print_weights() -> None:
    """
    Read hrp_weights_history.csv and display the most recent quarterly
    allocation as a horizontal ASCII bar chart (40-column scale).
    """
    if not WEIGHTS_CSV.exists():
        print(f"[WARN] Weights file not found: {WEIGHTS_CSV}")
        return

    df = pd.read_csv(WEIGHTS_CSV, index_col=0, parse_dates=True)
    live = df.loc[df.sum(axis=1) > 0]
    if live.empty:
        print("[WARN] No live-trading rows found in weights history.")
        return

    last_date = live.index[-1]
    alloc = live.iloc[-1].sort_values(ascending=False)

    print("\n" + "─" * 55)
    print(f"  LATEST HRP ALLOCATION  ({last_date.date()})")
    print("─" * 55)
    print(f"  {'TICKER':<8} {'WEIGHT':>8}  CHART (40 cols = 100%)")
    print("─" * 55)
    for ticker, w in alloc.items():
        bar = "█" * max(1, int(w * 40))
        print(f"  {ticker:<8} {w:>7.2%}  {bar}")
    print("─" * 55)
    print(f"  {'TOTAL':<8} {alloc.sum():>7.2%}")
    print("─" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Git commit + push (--publish flag only)
# ─────────────────────────────────────────────────────────────────────────────

def git_publish() -> None:
    """
    Stage all tracked project files, commit with a timestamped message,
    and push to the configured GitHub remote.

    Authentication: injects GITHUB_TOKEN into the HTTPS remote URL temporarily,
    then restores the clean URL immediately after the push.  The token is read
    from the GITHUB_TOKEN environment variable or from a .env file — it is
    NEVER written to disk or printed to stdout.
    """
    # Load token from .env if present
    env_file = SCRIPT_DIR / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            # Manual .env parser (fallback if python-dotenv not installed)
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        sys.exit(
            "\n[ERROR] GITHUB_TOKEN is not set.\n"
            "Create a .env file with:\n"
            "  GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx\n"
            "Or set it as a system environment variable.\n"
            "See .env.example for details."
        )

    commit_msg = (
        f"chore: quarterly HRP rebalance — {datetime.date.today().isoformat()}"
    )

    # ── Inject token into remote URL (HTTPS only) ──────────────────────────
    url_result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True, text=True, cwd=SCRIPT_DIR,
    )
    raw_url = url_result.stdout.strip()

    if not raw_url:
        sys.exit(
            "[ERROR] No git remote named 'origin' found.\n"
            "Set one up first:\n"
            "  git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git"
        )

    authed_url = raw_url
    uses_https = raw_url.startswith("https://")
    if uses_https:
        authed_url = raw_url.replace("https://", f"https://{token}@", 1)
        subprocess.run(
            ["git", "remote", "set-url", "origin", authed_url],
            cwd=SCRIPT_DIR, check=True,
        )

    try:
        for cmd in [
            ["git", "add", "-A"],
            ["git", "commit", "-m", commit_msg],
            ["git", "push", "origin", "main"],
        ]:
            result = subprocess.run(cmd, cwd=SCRIPT_DIR)
            # "nothing to commit" is exit code 1 on some git builds — not fatal
            if result.returncode != 0:
                if "nothing to commit" in (result.stdout or "") + (result.stderr or ""):
                    print("[INFO] Nothing new to commit.")
                    break
                print(f"[WARN] Command returned non-zero: {' '.join(cmd)}")
    finally:
        # Always restore clean URL — even if push fails — so the token
        # is never left embedded in .git/config on disk
        if uses_https:
            subprocess.run(
                ["git", "remote", "set-url", "origin", raw_url],
                cwd=SCRIPT_DIR, check=True,
            )

    print(f"[INFO] Pushed: \"{commit_msg}\"")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-Forward HRP Portfolio Engine — orchestration entry point.\n"
            "Runs the backtest, prints a metrics table, and optionally pushes "
            "output files to GitHub."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "After the engine finishes, commit all files and push to the "
            "GitHub remote. Requires GITHUB_TOKEN in .env or environment."
        ),
    )
    args = parser.parse_args()

    # 1. Run the backtest engine
    rc = run_engine()
    if rc != 0:
        sys.exit(f"[ERROR] portfolio_analysis.py exited with code {rc}.")

    # 2. Print standardised metrics
    print_metrics()

    # 3. Print allocation bar chart
    print_weights()

    # 4. Optionally commit + push
    if args.publish:
        git_publish()


if __name__ == "__main__":
    main()

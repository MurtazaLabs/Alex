"""Start the human-confirmed reference strategy through the REAL pipeline.

Why this exists: the E4 window trades a strategy VERSION that must pass the
deterministic backtest gate and then accrue paper probation (the gate proved
it on the past; probation observes it on the present). The LLM loop can
produce candidates, but the confirmed PRIMARY strategy - Mean Reversion,
Bollinger Bands + RSI on H1, human-confirmed 2026-10-16 - is authored
deterministically in ``strategy/templates.py``. This script pushes those
reference specs through the SAME gate, registry and probation
(``autonomy.proposal.propose_spec``) against REAL bars - no LLM, no
speculative veto, no special path to money.

Nothing here can trade: probation is paper-only, and strategy approval
remains a human button (the research routes / CLI).

Run:
  .venv311/Scripts/python.exe scripts/start_reference_strategy.py --dry-run
  .venv311/Scripts/python.exe scripts/start_reference_strategy.py

  --dry-run evaluates the gate only (no version, no probation, no DB rows).
  Without it, passing specs are versioned and paper probation STARTS - the
  accrual clock the E4 window needs.

Output is ASCII-only (cp1252 console safe); each run writes one evidence
JSON into logs/reference_strategy/. Data must come from a REAL source - if
the feed chain serves mock bars the script refuses to run.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from forex_ai_agent.autonomy.proposal import propose_spec
from forex_ai_agent.autonomy.validator import BacktestValidator
from forex_ai_agent.data.providers.factory import get_feed_chain
from forex_ai_agent.strategy.templates import mean_reversion_specs

SYMBOL = "EUR/USD"
TIMEFRAME = "H1"
HISTORY_DAYS = 365
LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "reference_strategy"


def _fetch_df(bars) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [float(getattr(b, "volume", 0) or 0) for b in bars],
        },
        index=[b.timestamp for b in bars],
    )
    df.index = pd.to_datetime(df.index)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _fmt_verdict(report) -> str:
    full = report.full_backtest or {}
    wf = report.walk_forward or {}
    parts = [
        f"trades={full.get('total_trades', 0)}",
        f"win={float(full.get('win_rate', 0.0)):.0%}",
        f"pf={float(full.get('profit_factor', 0.0)):.2f}",
        f"sharpe={float(full.get('sharpe_ratio', 0.0)):.2f}",
        f"dd={float(full.get('max_drawdown_pct', 0.0)):.1%}",
    ]
    if wf:
        parts.append(f"oos_trades={wf.get('oos_trades', 0)}")
        parts.append(f"oos_sharpe={float(wf.get('oos_sharpe', 0.0)):.2f}")
        parts.append(f"overfit={float(wf.get('overfitting_score', 0.0)):.2f}")
    return "  ".join(parts)


def _trim_wf(wf) -> dict:
    if not wf:
        return {}
    analysis = wf.get("overfit_analysis") or {}
    return {
        "oos_trades": wf.get("oos_trades"),
        "oos_sharpe": wf.get("oos_sharpe"),
        "oos_max_drawdown": wf.get("oos_max_drawdown"),
        "overfitting_score": wf.get("overfitting_score"),
        "overfit_composite": analysis.get("score"),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate the gate only; no version, no probation, no DB rows",
    )
    parser.add_argument("--pair", default=SYMBOL)
    parser.add_argument("--days", type=int, default=HISTORY_DAYS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    print("\n=== REFERENCE STRATEGY: Mean Reversion (BB + RSI) -> real pipeline ===")
    mode = "DRY RUN (gate only)" if args.dry_run else "LIVE PIPELINE (version + probation)"
    print(f"  mode: {mode}\n")

    # ------------------------------------------------------------------
    # 1. REAL bars
    # ------------------------------------------------------------------
    print(f"[1] Fetching real {TIMEFRAME} bars for {args.pair} ({args.days} days)...")
    chain = get_feed_chain()
    print(f"  feed chain active source: {chain.active_source}")
    end = datetime.utcnow()
    start = end - timedelta(days=args.days)
    limit = min(args.days * 24 + 50, 100_000)
    try:
        bars = await chain.fetch_candles(args.pair, TIMEFRAME, start, end=end, limit=limit)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] feed error: {type(exc).__name__}: {exc}")
        return 1
    if chain.active_source == "mock":
        print("  [FAIL] feed degraded to mock - refusing to gate on synthetic data")
        print("RESULT: FAIL - no real data")
        return 1
    if not bars or len(bars) < 250:
        print(f"  [FAIL] not enough bars (got {0 if not bars else len(bars)}, need >= 250)")
        print("RESULT: FAIL - insufficient data")
        return 1
    df = _fetch_df(bars)
    print(
        f"  [OK] {len(df)} bars  {df.index.min()} -> {df.index.max()}"
    )

    # ------------------------------------------------------------------
    # 2. Reference specs through the SAME gate
    # ------------------------------------------------------------------
    specs = mean_reversion_specs(args.pair)
    print(f"\n[2] Gating {len(specs)} reference spec(s) (deterministic, no LLM)...")
    evidence = {
        "ran_at": datetime.utcnow().isoformat(),
        "pair": args.pair,
        "timeframe": TIMEFRAME,
        "days": args.days,
        "bars": int(len(df)),
        "data_range": {"start": str(df.index.min()), "end": str(df.index.max())},
        "feed_source": chain.active_source,
        "dry_run": bool(args.dry_run),
        "specs": [],
    }
    registered = []
    for spec in specs:
        print(f"\n  --- {spec.name} (direction={spec.direction}) ---")
        entry = {"name": spec.name, "direction": spec.direction}
        if args.dry_run:
            report = await BacktestValidator().validate_spec(
                spec, df, symbol=args.pair
            )
            res = None
        else:
            res = await propose_spec(args.pair, spec, df, log_to_db=True)
            report = res.validation
        entry["passed"] = bool(report and report.passed)
        entry["reasons"] = list(report.reasons) if report else ["no report"]
        entry["full_backtest"] = dict(report.full_backtest or {}) if report else {}
        entry["walk_forward"] = _trim_wf(report.walk_forward if report else {})
        verdict = "PASS" if entry["passed"] else "REJECT"
        print(f"  gate: {verdict}")
        print(f"  {_fmt_verdict(report)}")
        if not entry["passed"]:
            for reason in entry["reasons"]:
                print(f"    - {reason}")
        if res is not None:
            entry["status"] = res.status
            entry["version_id"] = str(res.version.id) if res.version else None
            entry["messages"] = list(res.messages)
            for msg in res.messages:
                print(f"  pipeline: {msg}")
            m = re.search(r"probation active: (\d+) trades / (\d+) days", " ".join(res.messages))
            if m:
                entry["probation"] = {
                    "required_trades": int(m.group(1)),
                    "required_days": int(m.group(2)),
                }
            if res.version is not None:
                registered.append(entry)
        evidence["specs"].append(entry)

    # ------------------------------------------------------------------
    # 3. Evidence + summary
    # ------------------------------------------------------------------
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = LOG_DIR / f"reference_strategy_{stamp}.json"
    out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    passed = sum(1 for s in evidence["specs"] if s["passed"])
    print(f"\n[3] Summary: {passed}/{len(evidence['specs'])} specs passed the gate")
    if args.dry_run:
        print("  dry run: no version registered, no probation started")
    else:
        for s in registered:
            prob = s.get("probation") or {}
            print(
                f"  registered {s['version_id']} ({s['name']}): "
                f"probation {prob.get('required_trades', '?')} trades / "
                f"{prob.get('required_days', '?')} days"
            )
    print(f"  evidence: {out}")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

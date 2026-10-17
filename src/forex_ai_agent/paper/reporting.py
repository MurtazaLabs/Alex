"""Paper Trading Reporting - Daily/weekly/monthly reports."""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def generate_daily_report(engine, tracker) -> str:
    """Generate daily paper trading report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Daily Paper Trading Report - {datetime.utcnow().strftime('%Y-%m-%d')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; }}
        .header {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
        .metric {{ display: inline-block; background: #f8f9fa; padding: 15px; margin: 5px; border-radius: 5px; min-width: 150px; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Daily Paper Trading Report</h1>
        <p>Date: {datetime.utcnow().strftime('%Y-%m-%d')} | Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>

    {_generate_account_section(engine)}
    {_generate_daily_pnl_section(engine)}
    {_generate_trades_section(engine)}
    {_generate_open_positions_section(engine)}
    {_generate_risk_section(engine)}

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666;">
        <p>Paper Trading Report - Forex AI Agent | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </footer>
</body>
</html>"""
    return html


def _generate_account_section(engine) -> str:
    acc = engine.account
    return f"""
    <h2>Account Summary</h2>
    <div class="metric">
        <div class="metric-label">Balance</div>
        <div class="metric-value">${acc.balance:,.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Equity</div>
        <div class="metric-value">${acc.equity:,.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Used Margin</div>
        <div class="metric-value">${acc.used_margin:,.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Free Margin</div>
        <div class="metric-value">${acc.free_margin:,.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Leverage</div>
        <div class="metric-value">{acc.leverage}:1</div>
    </div>
    """


def _generate_daily_pnl_section(engine) -> str:
    acc = engine.account
    pnl_class = "positive" if acc.daily_pnl >= 0 else "negative"
    pnl_pct = (acc.daily_pnl / acc.daily_start_equity * 100) if acc.daily_start_equity else 0

    return f"""
    <h2>Daily P&L</h2>
    <div class="metric">
        <div class="metric-label">Day Start Equity</div>
        <div class="metric-value">${acc.daily_start_equity:,.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Current P&L</div>
        <div class="metric-value {pnl_class}">${acc.daily_pnl:,.2f} ({pnl_pct:+.2f}%)</div>
    </div>
    <div class="metric">
        <div class="metric-label">Trades Today</div>
        <div class="metric-value">{acc.daily_trades}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Daily Loss Limit</div>
        <div class="metric-value">${acc.daily_start_equity * 0.02:,.2f} (2%)</div>
    </div>
    """


def _generate_trades_section(engine) -> str:
    if not engine.closed_trades:
        return "<h2>Closed Trades</h2><p>No trades closed today.</p>"

    today = datetime.utcnow().date()
    today_trades = [t for t in engine.closed_trades if t.close_time.date() == today]

    if not today_trades:
        return "<h2>Closed Trades</h2><p>No trades closed today.</p>"

    rows = ""
    for t in today_trades:
        pnl_class = "positive" if t.profit >= 0 else "negative"
        side_class = "positive" if t.side.value == "BUY" else "negative"
        rows += f"""
        <tr>
            <td>{t.id}</td>
            <td>{t.symbol}</td>
            <td><span class="{side_class}">{t.side.value}</span></td>
            <td>{t.volume:.2f}</td>
            <td>{t.entry_price:.5f}</td>
            <td>{t.exit_price:.5f}</td>
            <td class="{pnl_class}">${t.profit:,.2f}</td>
            <td>{t.r_multiple:.2f}R</td>
            <td>{t.hold_time_seconds/3600:.1f}h</td>
            <td>{t.reason}</td>
        </tr>
        """

    return f"""
    <h2>Closed Trades ({len(today_trades)})</h2>
    <table>
        <thead>
            <tr>
                <th>ID</th><th>Symbol</th><th>Side</th><th>Vol</th>
                <th>Entry</th><th>Exit</th><th>P&L</th><th>R</th><th>Hold</th><th>Reason</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def _generate_open_positions_section(engine) -> str:
    if not engine.positions:
        return "<h2>Open Positions</h2><p>No open positions.</p>"

    rows = ""
    for pos_id, pos in engine.positions.items():
        pnl_class = "positive" if pos.unrealized_pnl >= 0 else "negative"
        side_class = "positive" if pos.side.value == "BUY" else "negative"
        hold_hours = (datetime.utcnow() - pos.open_time).total_seconds() / 3600

        rows += f"""
        <tr>
            <td>{pos_id}</td>
            <td>{pos.symbol}</td>
            <td><span class="{side_class}">{pos.side.value}</span></td>
            <td>{pos.volume:.2f}</td>
            <td>{pos.entry_price:.5f}</td>
            <td>{pos.current_price:.5f}</td>
            <td class="{pnl_class}">${pos.unrealized_pnl:,.2f}</td>
            <td>{pos.sl:.5f if pos.sl else '-'}</td>
            <td>{pos.tp:.5f if pos.tp else '-'}</td>
            <td>{hold_hours:.1f}h</td>
        </tr>
        """

    return f"""
    <h2>Open Positions ({len(engine.positions)})</h2>
    <table>
        <thead>
            <tr>
                <th>ID</th><th>Symbol</th><th>Side</th><th>Vol</th>
                <th>Entry</th><th>Current</th><th>Unrealized</th><th>SL</th><th>TP</th><th>Hold</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def _generate_risk_section(engine) -> str:
    acc = engine.account
    total_unrealized = sum(p.unrealized_pnl for p in engine.positions.values())
    unrealized_class = "positive" if total_unrealized >= 0 else "negative"

    return f"""
    <h2>Risk Metrics</h2>
    <div class="metric">
        <div class="metric-label">Max Drawdown</div>
        <div class="metric-value negative">{acc.max_drawdown_pct:.2f}%</div>
    </div>
    <div class="metric">
        <div class="metric-label">Peak Equity</div>
        <div class="metric-value">${acc.peak_equity:,.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Current Drawdown</div>
        <div class="metric-value negative">{acc.max_drawdown_pct:.2f}%</div>
    </div>
    <div class="metric">
        <div class="metric-label">Total Unrealized</div>
        <div class="metric-value {unrealized_class}">${total_unrealized:,.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Open Positions</div>
        <div class="metric-value">{len(engine.positions)}</div>
    </div>
    """


def generate_weekly_report(engine, tracker) -> str:
    """Generate weekly performance report."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)

    week_trades = [t for t in engine.closed_trades if start_date <= t.close_time <= end_date]

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Weekly Paper Trading Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; }}
        .metric {{ display: inline-block; background: #f8f9fa; padding: 15px; margin: 5px; border-radius: 5px; min-width: 150px; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <h1>Weekly Paper Trading Report</h1>
    <p>Week: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</p>

    <h2>Weekly Summary</h2>
    <div class="metric">
        <div class="metric-label">Trades</div>
        <div class="metric-value">{len([t for t in engine.closed_trades if start_date <= t.close_time <= end_date])}</div>
    </div>

    <p>Full weekly analytics available in performance tracker.</p>
</body>
</html>"""


def generate_monthly_report(engine, tracker) -> str:
    """Generate monthly performance report."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Monthly Paper Trading Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; }}
        .metric {{ display: inline-block; background: #f8f9fa; padding: 15px; margin: 5px; border-radius: 5px; min-width: 150px; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
    </style>
</head>
<body>
    <h1>Monthly Paper Trading Report</h1>
    <p>Month: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}</p>

    <p>Full monthly analytics available in performance tracker.</p>
</body>
</html>"""


def save_report(html: str, path: str) -> bool:
    """Save HTML report to file."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        return True
    except Exception as e:
        print(f"Report save failed: {e}")
        return False
"""Backtest Reporting - HTML/PDF generation with charts."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def generate_html_report(
    result: Any,  # BacktestResult or WalkForwardResult
    title: str = "Backtest Report",
) -> str:
    """Generate HTML report with embedded charts."""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4a90d9; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        h3 {{ color: #34495e; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #4a90d9; }}
        .metric-label {{ font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #1a1a2e; margin-top: 5px; }}
        .metric-value.positive {{ color: #27ae60; }}
        .metric-value.negative {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #2c3e50; }}
        tr:hover {{ background: #f8f9fa; }}
        .chart-container {{ margin: 20px 0; text-align: center; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .section {{ margin: 30px 0; padding: 20px; background: #fafafa; border-radius: 8px; }}
        .trade-list {{ max-height: 400px; overflow-y: auto; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
        .badge-green {{ background: #d4edda; color: #155724; }}
        .badge-red {{ background: #f8d7da; color: #721c24; }}
        .badge-blue {{ background: #cce5ff; color: #004085; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>

        {_generate_summary_section(result)}
        {_generate_equity_chart(result)}
        {_generate_drawdown_chart(result)}
        {_generate_trades_section(result)}
        {_generate_breakdown_section(result)}
        {_generate_monthly_returns_section(result)}

        <div class="footer">
            <p>This report was generated automatically by the Forex AI Agent Backtesting Engine.</p>
            <p>Past performance does not guarantee future results. Trading involves substantial risk.</p>
        </div>
    </div>
</body>
</html>"""

    return html


def _generate_summary_section(result: Any) -> str:
    """Generate summary metrics section."""
    summary = result.metrics.get("summary", {}) if hasattr(result, 'metrics') else {}

    metrics = [
        ("Total Return", f"{summary.get('total_return_pct', 0):.2f}%", "positive" if summary.get('total_return_pct', 0) > 0 else "negative"),
        ("Annualized Return", f"{summary.get('annualized_return', 0):.2f}%", "positive" if summary.get('annualized_return', 0) > 0 else "negative"),
        ("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}", "positive" if summary.get('sharpe_ratio', 0) > 1 else ""),
        ("Sortino Ratio", f"{summary.get('sortino_ratio', 0):.2f}", "positive" if summary.get('sortino_ratio', 0) > 1 else ""),
        ("Calmar Ratio", f"{summary.get('calmar_ratio', 0):.2f}", "positive" if summary.get('calmar_ratio', 0) > 1 else ""),
        ("Max Drawdown", f"{summary.get('max_drawdown_pct', 0):.2f}%", "negative"),
        ("Profit Factor", f"{summary.get('profit_factor', 0):.2f}", "positive" if summary.get('profit_factor', 0) > 1 else "negative"),
        ("Win Rate", f"{summary.get('win_rate', 0)*100:.1f}%", "positive" if summary.get('win_rate', 0) > 0.5 else "negative"),
        ("Expectancy", f"${summary.get('expectancy', 0):.2f}", "positive" if summary.get('expectancy', 0) > 0 else "negative"),
        ("Total Trades", str(summary.get('total_trades', 0)), ""),
    ]

    cards = ""
    for label, value, cls in metrics:
        cards += f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{value}</div>
        </div>
        """

    return f"""
    <div class="section">
        <h2>Performance Summary</h2>
        <div class="metrics-grid">{cards}</div>
    </div>
    """


def _generate_equity_chart(result: Any) -> str:
    """Generate equity curve chart as base64 PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return "<p><em>Matplotlib not available for chart generation</em></p>"

    equity_curve = []
    if hasattr(result, 'equity_curve'):
        equity_curve = result.equity_curve
    elif hasattr(result, 'combined_equity_curve'):
        equity_curve = result.combined_equity_curve

    if not equity_curve:
        return ""

    times = [t for t, _ in equity_curve]
    values = [v for _, v in equity_curve]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, values, color='#4a90d9', linewidth=1.5)
    ax.fill_between(times, values, alpha=0.1, color='#4a90d9')

    ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax.set_ylabel('Equity ($)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add initial balance line
    if hasattr(result, 'config') and hasattr(result.config, 'initial_balance'):
        ax.axhline(y=result.config.initial_balance, color='gray', linestyle='--', alpha=0.5, label='Initial Balance')
        ax.legend()

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f"""
    <div class="section">
        <h2>Equity Curve</h2>
        <div class="chart-container">
            <img src="data:image/png;base64,{img_base64}" alt="Equity Curve">
        </div>
    </div>
    """


def _generate_drawdown_chart(result: Any) -> str:
    """Generate drawdown chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return ""

    equity_curve = []
    if hasattr(result, 'equity_curve'):
        equity_curve = result.equity_curve
    elif hasattr(result, 'combined_equity_curve'):
        equity_curve = result.combined_equity_curve

    if not equity_curve:
        return ""

    times = [t for t, _ in equity_curve]
    values = np.array([v for _, v in equity_curve])

    # Calculate drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1], sharex=True)

    # Equity
    ax1.plot(times, values, color='#4a90d9', linewidth=1)
    ax1.set_ylabel('Equity ($)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(times, 0, drawdown, color='#e74c3c', alpha=0.3)
    ax2.plot(times, drawdown, color='#e74c3c', linewidth=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('Equity & Drawdown', fontsize=14, fontweight='bold')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f"""
    <div class="section">
        <h2>Drawdown Analysis</h2>
        <div class="chart-container">
            <img src="data:image/png;base64,{img_base64}" alt="Drawdown Chart">
        </div>
    </div>
    """


def _generate_trades_section(result: Any) -> str:
    """Generate trades table."""
    trades = []
    if hasattr(result, 'trades'):
        trades = result.trades
    elif hasattr(result, 'combined_trades'):
        trades = result.combined_trades
    else:
        # Collect from windows
        for w in getattr(result, 'windows', []):
            if w.test_result and 'trades' in w.test_result:
                trades.extend(w.test_result['trades'])

    if not trades:
        return ""

    # Limit to last 50 trades
    trades = trades[-50:]

    rows = ""
    for t in trades:
        profit = t.get('profit', 0)
        profit_class = "positive" if profit >= 0 else "negative"
        side_badge = "badge-green" if t.get('side') == 'BUY' else "badge-red"
        reason = t.get('reason', 'MANUAL')

        rows += f"""
        <tr>
            <td>{t.get('id', '')[:8]}</td>
            <td>{t.get('symbol', '')}</td>
            <td><span class="badge {side_badge}">{t.get('side', '')}</span></td>
            <td>{t.get('volume', 0):.2f}</td>
            <td>{t.get('open_price', 0):.5f}</td>
            <td>{t.get('close_price', 0):.5f}</td>
            <td class="{profit_class}">${profit:.2f}</td>
            <td>{t.get('hold_time_hours', 0):.1f}h</td>
            <td>{t.get('reason', '')}</td>
            <td>{pd.to_datetime(t.get('open_time', '')).strftime('%Y-%m-%d %H:%M') if t.get('open_time') else ''}</td>
        </tr>
        """

    return f"""
    <div class="section">
        <h2>Recent Trades (Last 50)</h2>
        <div class="trade-list">
            <table>
                <thead>
                    <tr>
                        <th>ID</th><th>Symbol</th><th>Side</th><th>Vol</th>
                        <th>Entry</th><th>Exit</th><th>P&L</th>
                        <th>Hold</th><th>Reason</th><th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    </div>
    """


def _generate_breakdown_section(result: Any) -> str:
    """Generate performance breakdowns."""
    breakdowns = result.metrics.get("breakdowns", {}) if hasattr(result, 'metrics') else {}

    if not breakdowns:
        return ""

    sections = ""
    for name, data in breakdowns.items():
        if not data:
            continue

        rows = ""
        for key, vals in data.items():
            if isinstance(vals, dict):
                wr = vals.get('win_rate', 0) * 100
                profit = vals.get('total_profit', 0)
                trades = vals.get('trades', 0)
                rows += f"<tr><td>{key}</td><td>{trades}</td><td>{wr:.1f}%</td><td>${profit:.2f}</td></tr>"

        if rows:
            sections += f"""
            <div class="section">
                <h3>{name.replace('_', ' ').title()}</h3>
                <table>
                    <thead><tr><th>{name.title()}</th><th>Trades</th><th>Win Rate</th><th>Total P&L</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

    if not sections:
        return ""

    return f"""
    <div class="section">
        <h2>Performance Breakdowns</h2>
        {sections}
    </div>
    """


def _generate_monthly_returns_section(result: Any) -> str:
    """Generate monthly returns heatmap."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    equity_curve = []
    if hasattr(result, 'equity_curve'):
        equity_curve = result.equity_curve
    elif hasattr(result, 'combined_equity_curve'):
        equity_curve = result.combined_equity_curve

    if not equity_curve:
        return ""

    times = [t for t, _ in equity_curve]
    values = [v for _, v in equity_curve]

    df = pd.DataFrame({"value": values}, index=pd.DatetimeIndex(times))
    monthly = df.resample("M").last().ffill()
    monthly_returns = monthly["value"].pct_change().dropna() * 100

    if len(monthly_returns) < 2:
        return ""

    # Create year-month matrix
    monthly_returns.index = monthly_returns.index.to_period("M")
    matrix = monthly_returns.unstack().T

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix.values, aspect='auto', cmap='RdYlGn', vmin=-5, vmax=5)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45, ha='right')
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(i) for i in matrix.index])
    ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Return %')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f"""
    <div class="section">
        <h2>Monthly Returns Heatmap</h2>
        <div class="chart-container">
            <img src="data:image/png;base64,{img_base64}" alt="Monthly Returns">
        </div>
    </div>
    """


def generate_pdf_report(result: Any, output_path: str, title: str = "Backtest Report") -> bool:
    """Generate PDF report (requires weasyprint)."""
    try:
        from weasyprint import HTML
        html = generate_html_report(result, title)
        HTML(string=html).write_pdf(output_path)
        return True
    except ImportError:
        print("weasyprint not installed. Install with: pip install weasyprint")
        return False
    except Exception as e:
        print(f"PDF generation failed: {e}")
        return False


def save_html_report(result: Any, output_path: str, title: str = "Backtest Report") -> bool:
    """Save HTML report to file."""
    try:
        html = generate_html_report(result, title)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return True
    except Exception as e:
        print(f"HTML report generation failed: {e}")
        return False
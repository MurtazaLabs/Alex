"""Discord notification service — trade alerts, errors, daily reports.

Sends rich embed messages to a Discord channel via webhook. Used for:
- Trade fills (entry, exit, P&L)
- Kill switch trips
- Error alerts
- Daily performance reports
- Strategy approvals/rejections

Configuration: set DISCORD_WEBHOOK_URL in .env or environment.
All notifications are fire-and-forget; failures never block trading.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)

__all__ = ["DiscordNotifier", "get_notifier"]

# Discord embed color constants
COLOR_GREEN = 0x00FF00
COLOR_RED = 0xFF0000
COLOR_YELLOW = 0xFFFF00
COLOR_BLUE = 0x0099FF
COLOR_ORANGE = 0xFF8800
COLOR_PURPLE = 0x9900FF


class DiscordNotifier:
    """Fire-and-forget Discord webhook notifications.

    Every method is async and never raises — a failed notification is logged
    but never blocks or crashes the trading pipeline.
    """

    def __init__(self, webhook_url: Optional[str] = None) -> None:
        settings = get_settings()
        self._url = webhook_url or settings.discord_webhook_url
        self._enabled = settings.discord_notify_enabled and bool(self._url)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def _send(
        self,
        *,
        content: str = "",
        embeds: Optional[List[Dict[str, Any]]] = None,
        username: str = "Alex Trading",
    ) -> bool:
        """Send a message to Discord. Returns True on success, never raises."""
        if not self._enabled or not self._url:
            return False
        payload: Dict[str, Any] = {"username": username}
        if content:
            payload["content"] = content
        if embeds:
            payload["embeds"] = embeds[:10]  # Discord limit
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(self._url, json=payload)
                if res.status_code == 204 or res.status_code == 200:
                    return True
                logger.warning("discord_notify_failed status=%s body=%s", res.status_code, res.text[:200])
                return False
        except Exception as exc:
            logger.warning("discord_notify_error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Trade notifications
    # ------------------------------------------------------------------

    async def trade_filled(
        self, *, symbol: str, side: str, volume: float,
        entry: float, sl: float, tp: float, risk_usd: float,
        strategy_id: str = "", intent_key: str = "",
    ) -> bool:
        """Notify: order filled."""
        color = COLOR_GREEN if side == "BUY" else COLOR_RED
        embed = {
            "title": f"✅ Order Filled: {side} {symbol}",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Volume", "value": f"{volume} lots", "inline": True},
                {"name": "Entry", "value": str(entry), "inline": True},
                {"name": "Stop Loss", "value": str(sl), "inline": True},
                {"name": "Take Profit", "value": str(tp), "inline": True},
                {"name": "Risk", "value": f"${risk_usd:.2f}", "inline": True},
                {"name": "Strategy", "value": strategy_id or "manual", "inline": True},
            ],
        }
        return await self._send(embeds=[embed])

    async def trade_closed(
        self, *, symbol: str, side: str, volume: float,
        entry: float, exit_price: float, pnl_usd: float,
        pnl_pips: float, strategy_id: str = "",
    ) -> bool:
        """Notify: position closed with P&L."""
        profit = pnl_usd >= 0
        color = COLOR_GREEN if profit else COLOR_RED
        emoji = "💰" if profit else "📉"
        embed = {
            "title": f"{emoji} Position Closed: {symbol}",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Side", "value": side, "inline": True},
                {"name": "Volume", "value": f"{volume} lots", "inline": True},
                {"name": "Entry", "value": str(entry), "inline": True},
                {"name": "Exit", "value": str(exit_price), "inline": True},
                {"name": "P&L", "value": f"${pnl_usd:.2f} ({pnl_pips:+.1f} pips)", "inline": True},
                {"name": "Strategy", "value": strategy_id or "manual", "inline": True},
            ],
        }
        return await self._send(embeds=[embed])

    # ------------------------------------------------------------------
    # Intent / approval notifications
    # ------------------------------------------------------------------

    async def intent_created(
        self, *, symbol: str, side: str, volume: float,
        entry: float, sl: float, tp: float, risk_usd: float,
        rr_ratio: float, intent_key: str,
    ) -> bool:
        """Notify: new order intent created (needs approval)."""
        embed = {
            "title": f"🔔 New Intent: {side} {symbol}",
            "color": COLOR_BLUE,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Volume", "value": f"{volume} lots", "inline": True},
                {"name": "Entry", "value": str(entry), "inline": True},
                {"name": "SL / TP", "value": f"{sl} / {tp}", "inline": True},
                {"name": "Risk", "value": f"${risk_usd:.2f}", "inline": True},
                {"name": "R:R", "value": f"{rr_ratio:.1f}:1", "inline": True},
                {"name": "Intent", "value": f"`{intent_key[:12]}`", "inline": True},
            ],
            "footer": {"text": "Review in Trading page to approve"},
        }
        return await self._send(embeds=[embed])

    async def token_granted(
        self, *, token_id: str, intent_key: str,
        symbol: str, side: str, volume: float, granted_by: str,
    ) -> bool:
        """Notify: approval token granted."""
        embed = {
            "title": f"✅ Token Granted: {side} {symbol}",
            "color": COLOR_GREEN,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Token", "value": f"`{token_id[:12]}`", "inline": True},
                {"name": "Volume", "value": f"{volume} lots", "inline": True},
                {"name": "Granted by", "value": granted_by, "inline": True},
            ],
        }
        return await self._send(embeds=[embed])

    # ------------------------------------------------------------------
    # Kill switch / error notifications
    # ------------------------------------------------------------------

    async def kill_switch_tripped(self, *, reason: str) -> bool:
        """Notify: kill switch tripped."""
        embed = {
            "title": "🚨 KILL SWITCH TRIPPED",
            "color": COLOR_RED,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Reason", "value": reason, "inline": False},
            ],
            "footer": {"text": "All execution halted. Re-arm required."},
        }
        return await self._send(content="@everyone", embeds=[embed])

    async def error_alert(self, *, message: str, context: str = "") -> bool:
        """Notify: system error."""
        embed = {
            "title": "⚠️ Error Alert",
            "color": COLOR_ORANGE,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Message", "value": message[:1024], "inline": False},
                {"name": "Context", "value": context[:1024], "inline": False} if context else None,
            ],
        }
        embed["fields"] = [f for f in embed["fields"] if f]
        return await self._send(embeds=[embed])

    # ------------------------------------------------------------------
    # Daily report
    # ------------------------------------------------------------------

    async def daily_report(
        self, *, equity: float, day_pnl: float, trades: int,
        wins: int, losses: int, open_positions: int,
        drawdown_pct: float, win_rate: float,
    ) -> bool:
        """Notify: daily performance report."""
        profit = day_pnl >= 0
        color = COLOR_GREEN if profit else COLOR_RED
        emoji = "📈" if profit else "📉"
        embed = {
            "title": f"{emoji} Daily Report — {datetime.utcnow().strftime('%Y-%m-%d')}",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Equity", "value": f"${equity:.2f}", "inline": True},
                {"name": "Day P&L", "value": f"${day_pnl:+.2f}", "inline": True},
                {"name": "Win Rate", "value": f"{win_rate:.0f}% ({wins}W/{losses}L)", "inline": True},
                {"name": "Trades", "value": str(trades), "inline": True},
                {"name": "Open Positions", "value": str(open_positions), "inline": True},
                {"name": "Drawdown", "value": f"{drawdown_pct:.1f}%", "inline": True},
            ],
        }
        return await self._send(embeds=[embed])

    # ------------------------------------------------------------------
    # Strategy notifications
    # ------------------------------------------------------------------

    async def strategy_approved(self, *, strategy_id: str, name: str) -> bool:
        embed = {
            "title": f"🧠 Strategy Approved: {name}",
            "color": COLOR_PURPLE,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [{"name": "ID", "value": f"`{strategy_id}`", "inline": False}],
        }
        return await self._send(embeds=[embed])

    async def strategy_retired(self, *, strategy_id: str, name: str, reason: str) -> bool:
        embed = {
            "title": f"🗑️ Strategy Retired: {name}",
            "color": COLOR_YELLOW,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "ID", "value": f"`{strategy_id}`", "inline": False},
                {"name": "Reason", "value": reason, "inline": False},
            ],
        }
        return await self._send(embeds=[embed])

    # ------------------------------------------------------------------
    # Generic
    # ------------------------------------------------------------------

    async def info(self, *, title: str, message: str, color: int = COLOR_BLUE) -> bool:
        """Generic info notification."""
        embed = {
            "title": title,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "description": message[:2048],
        }
        return await self._send(embeds=[embed])


# Module-level singleton (lazy)
_notifier: Optional[DiscordNotifier] = None


def get_notifier() -> DiscordNotifier:
    """Get the shared DiscordNotifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = DiscordNotifier()
    return _notifier

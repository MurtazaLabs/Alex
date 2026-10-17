"""WebSocket support for real-time Forex AI Agent updates.

Provides real-time streaming of:
- Market data updates
- Signal status changes
- Paper trade P&L updates
- Research progress
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Set, Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from forex_ai_agent.config.settings import get_settings
from forex_ai_agent.core.event_bus import InMemoryEventBus

logger = logging.getLogger(__name__)


class WSConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.user_subscriptions: Dict[WebSocket, List[str]] = (
            {}
        )  # connection -> subscribed topics
        self.event_bus: InMemoryEventBus = InMemoryEventBus()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.user_subscriptions[websocket] = []
        logger.info("WebSocket connected", total_connections=len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        self.user_subscriptions.pop(websocket, None)
        logger.info(
            "WebSocket disconnected",
            total_connections=len(self.active_connections),
        )

    def subscribe(self, websocket: WebSocket, topic: str) -> None:
        """Subscribe a connection to a topic."""
        if websocket in self.active_connections:
            if topic not in self.user_subscriptions.get(websocket, []):
                self.user_subscriptions.setdefault(websocket, []).append(topic)
                logger.debug(f"Subscribed to {topic}")

    def unsubscribe(self, websocket: WebSocket, topic: str) -> None:
        """Unsubscribe a connection from a topic."""
        subs = self.user_subscriptions.get(websocket, [])
        if topic in subs:
            self.user_subscriptions[websocket] = [t for t in subs if t != topic]
            logger.debug(f"Unsubscribed from {topic}")

    async def broadcast(self, message: Dict[str, any], topics: Optional[List[str]] = None) -> None:
        """Broadcast a message to active connections."""
        # Filter by topics if specified
        targets = self.active_connections
        if topics:
            targets = {
                ws
                for ws in self.active_connections
                if any(t in self.user_subscriptions.get(ws, []) for t in topics)
            }

        if not targets:
            return

        disconnected: Set[WebSocket] = set()
        for ws in targets:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                disconnected.add(ws)

        # Clean up dead connections
        for ws in disconnected:
            self.disconnect(ws)

    async def emit_market_data(self, payload: Dict[str, any]) -> None:
        """Emit market data update to all connections."""
        await self.broadcast(
            {"type": "market_data", "payload": payload}, topics=["market"]
        )

    async def emit_signal_update(self, payload: Dict[str, any]) -> None:
        """Emit signal status update to all connections."""
        await self.broadcast(
            {"type": "signal_update", "payload": payload}, topics=["signals"]
        )

    async def emit_paper_trade(self, payload: Dict[str, any]) -> None:
        """Emit paper trade update to all connections."""
        await self.broadcast(
            {"type": "paper_trade", "payload": payload}, topics=["paper_trades"]
        )

    async def emit_research_progress(self, payload: Dict[str, any]) -> None:
        """Emit research progress update to all connections."""
        await self.broadcast(
            {"type": "research_progress", "payload": payload}, topics=["research"]
        )


# Global manager instance
ws_manager = WSConnectionManager()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates."""
    await ws_manager.connect(websocket)

    try:
        # Send welcome message
        await websocket.send_text(
            json.dumps(
                {
                    "type": "welcome",
                    "message": "Connected to Forex AI Agent real-time updates",
                    "subscribed_topics": [],
                }
            )
        )

        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                await handle_websocket_message(websocket, msg)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON"})
                )
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                await websocket.send_text(
                    json.dumps({"type": "error", "message": str(e)})
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        ws_manager.disconnect(websocket)


async def handle_websocket_message(websocket: WebSocket, msg: Dict[str, any]) -> None:
    """Handle a WebSocket client message."""
    msg_type = msg.get("type", "")

    if msg_type == "subscribe":
        topic = msg.get("topic", "")
        if topic:
            ws_manager.subscribe(websocket, topic)
            await websocket.send_text(
                json.dumps({"type": "subscribed", "topic": topic})
            )
        else:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Topic required"})
            )

    elif msg_type == "unsubscribe":
        topic = msg.get("topic", "")
        if topic:
            ws_manager.unsubscribe(websocket, topic)
            await websocket.send_text(
                json.dumps({"type": "unsubscribed", "topic": topic})
            )
        else:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Topic required"})
            )

    elif msg_type == "get_status":
        # Return current status
        # Note: would need app instance in production
        await websocket.send_text(
            json.dumps({"type": "status", "payload": {"status": "ok"}})
        )

    else:
        await websocket.send_text(
            json.dumps({"type": "error", "message": f"Unknown message type: {msg_type}"})
        )
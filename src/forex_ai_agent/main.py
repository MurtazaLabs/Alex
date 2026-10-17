"""Forex AI Agent entry point.

Run with:
    uvicorn forex_ai_agent.main:app --reload
"""

from .interfaces.api.app import app

__all__ = ["app"]

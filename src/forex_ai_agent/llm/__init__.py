"""LLM client package for Alex.

The LLM narrates over deterministic evidence; it never decides trades.
All public symbols are re-exported here for convenience.
"""
from .client import LLMClient, LLMResponse, LLMUnavailable

__all__ = ["LLMClient", "LLMResponse", "LLMUnavailable"]

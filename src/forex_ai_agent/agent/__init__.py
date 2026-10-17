"""Alex agent layer — conversational control plane over the deterministic core.

The AgentCore is the brain behind both chat surfaces (dashboard Chat page and
CLI ``chat``). It routes natural-language intent to deterministic tools,
collects their observations as evidence, and has a strict-JSON LLM compose the
final answer. The LLM never produces numbers: every figure in an answer traces
to a tool observation (or the answer is "insufficient evidence").
"""

from .core import AgentCore, AgentTurnResult
from .memory import ConversationMemory
from .tools import get_tool_registry, ToolRegistry

__all__ = ["AgentCore", "AgentTurnResult", "ConversationMemory", "get_tool_registry", "ToolRegistry"]

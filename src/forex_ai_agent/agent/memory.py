"""Session conversation memory for the agent (Phase-1 scope: in-process).

Holds the rolling transcript of the current chat session (both surfaces,
separate sessions). The LLM sees the last few turns so follow-ups like
"what about GBP/JPY?" resolve the pair from context. SQLite persistence of
threads arrives in Phase 4.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional


class ConversationMemory:
    """Bounded per-session transcript with follow-up context extraction."""

    def __init__(self, max_turns: int = 12) -> None:
        self._turns: Deque[Dict[str, Any]] = deque(maxlen=max_turns)

    def add_user(self, text: str) -> None:
        self._turns.append({"role": "user", "text": text, "ts": time.time()})

    def add_assistant(self, text: str, tools_used: List[str]) -> None:
        self._turns.append(
            {"role": "assistant", "text": text, "tools": list(tools_used), "ts": time.time()}
        )

    @property
    def turns(self) -> List[Dict[str, Any]]:
        return list(self._turns)

    def last_entity(self) -> Optional[str]:
        """Most recent FOREX pair mentioned in this session (for follow-ups)."""
        for turn in reversed(self._turns):
            found = extract_pair(turn.get("text", ""))
            if found:
                return found
        return None

    def clear(self) -> None:
        self._turns.clear()


_PAIR_RE = None  # compiled lazily to keep import light


def extract_pair(text: str) -> Optional[str]:
    """Extract the first 'XXX/YYY' forex pair from free text."""
    import re

    global _PAIR_RE
    if _PAIR_RE is None:
        _PAIR_RE = re.compile(r"\b([A-Z]{3})/([A-Z]{3})\b")
    m = _PAIR_RE.search(text.upper())
    return f"{m.group(1)}/{m.group(2)}" if m else None

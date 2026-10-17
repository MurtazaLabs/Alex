"""End-to-end smoke test of the Phase 1 orchestration pipeline."""
import asyncio

from forex_ai_agent.agents import build_default_specialists
from forex_ai_agent.core.event_bus import InMemoryEventBus
from forex_ai_agent.core.orchestrator import Orchestrator
from forex_ai_agent.core.permission_manager import PermissionLevel, PermissionManager


async def main():
    bus = InMemoryEventBus()
    orch = Orchestrator(
        PermissionManager(current_level=PermissionLevel.SIGNAL_GENERATION),
        event_bus=bus,
        specialists=build_default_specialists(bus),
    )
    queries = [
        "Analyze EUR/USD trend",
        "open trade EUR/USD now",
        "Research liquidity sweeps",
    ]
    for query in queries:
        result = await orch.process_request(query)
        state = result.get("final_state", result.get("status"))
        print(f"{query!r:44} -> {result['status']:>12} | {state}")
    print(f"\nevts published (in-memory audit trail): {len(bus.published)}")


if __name__ == "__main__":
    asyncio.run(main())
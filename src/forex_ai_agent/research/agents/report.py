"""Report Agent - Generates research reports."""

from __future__ import annotations

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.research.agents.base import BaseResearchAgent
from forex_ai_agent.research.models import Hypothesis, ResearchPhase
from forex_ai_agent.paper.reporting import generate_daily_report as generate_html_report, save_report


class ReportAgent(BaseResearchAgent):
    """
    Report Agent - Generates research reports.

    Capabilities:
    - HTML report generation
    - PDF report generation (with weasyprint)
    - Executive summary creation
    """

    def __init__(
        self,
        event_bus=None,
    ):
        super().__init__("report", event_bus)

    def _register_capabilities(self):
        self.capabilities = [
            "html_report_generation",
            "pdf_report_generation",
            "executive_summary",
        ]

    async def execute(
        self,
        hypothesis: any,
        input_data: dict,
    ) -> dict:
        phase = input_data.get("phase", "REPORT_GENERATION")

        if phase == "REPORT_GENERATION":
            return await self.generate_report(hypothesis, input_data)
        else:
            return {"error": f"Unknown phase: {phase}"}

    async def generate_report(
        self,
        hypothesis: any,
        input_data: dict,
    ) -> dict:
        """Generate research report."""
        bt_output = input_data.get("backtesting_output", {})
        best_variant = bt_output.get("best_variant", {})

        # Create mock engine with account and trades
        class MockAccount:
            def __init__(self):
                self.balance = 100000.0
                self.equity = 100000.0
                self.used_margin = 0.0
                self.free_margin = 100000.0
                self.leverage = 100
                self.daily_pnl = 0.0
                self.daily_start_equity = 100000.0
                self.max_drawdown = 0.0
                self.max_drawdown_pct = 0.0
                self.daily_trades = 0
                self.peak_equity = 100000.0

        class MockEngine:
            def __init__(self):
                self.account = MockAccount()
                self.metrics = {
                    "summary": best_variant.get("metrics", {}).get("summary", {}),
                    "breakdowns": best_variant.get("metrics", {}).get("breakdowns", {}),
                }
                self.trades = best_variant.get("metrics", {}).get("trades", [])
                self.equity_curve = best_variant.get("equity_curve", [])
                self.combined_equity_curve = best_variant.get("equity_curve", [])
                self.window_equity_curves = []
                self.config = None
                self.closed_trades = best_variant.get("metrics", {}).get("trades", [])
                self.positions = {}

        engine = MockEngine()

        from forex_ai_agent.paper.reporting import generate_daily_report as generate_html_report, save_report

        html_report = generate_html_report(engine, None)

        # Save report
        report_path = f"research_report_{hypothesis.id}.html"
        save_report(html_report, report_path)

        return {
            "report_path": report_path,
            "report_generated": True,
            "report_length": len(html_report),
        }


async def execute(
    self,
    hypothesis: any,
    input_data: dict,
) -> dict:
    phase = input_data.get("phase", "REPORT_GENERATION")

    if phase == "REPORT_GENERATION":
        return await self.generate_report(hypothesis, input_data)
    else:
        return {"error": f"Unknown phase: {phase}"}


def create_report_agent(
    event_bus=None,
) -> "ReportAgent":
    """Create a report agent."""
    return ReportAgent(event_bus=event_bus)
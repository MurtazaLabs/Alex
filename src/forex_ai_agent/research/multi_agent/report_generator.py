"""Research Report Generator - LaTeX/PDF report generation with executive summaries."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from forex_ai_agent.research.models import Hypothesis, ResearchSession


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    include_executive_summary: bool = True
    include_methodology: bool = True
    include_detailed_results: bool = True
    include_charts: bool = True
    include_appendix: bool = True
    format: str = "latex"  # latex, html, markdown, pdf
    template: str = "standard"  # standard, academic, executive, technical
    language: str = "en"
    logo_path: Optional[str] = None
    author: str = "Forex AI Agent"
    institution: str = "Forex AI Research"


class LatexGenerator:
    """Generates LaTeX code for research reports."""

    @staticmethod
    def generate_preamble(config: ReportConfig) -> str:
        """Generate LaTeX preamble."""
        return r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{float}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{listings}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{siunitx}
\usepackage{pgfplots}
\usepackage{pgfplotstable}

\geometry{margin=1in}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small """ + config.institution + r""" - """ + config.author + r"""}
\fancyhead[R]{\small \thepage}
\fancyfoot[C]{\small Confidential - """ + config.institution + r"""}

\definecolor{primary}{RGB}{0, 51, 102}
\definecolor{secondary}{RGB}{0, 102, 153}
\definecolor{accent}{RGB}{204, 0, 0}

\hypersetup{
    colorlinks=true,
    linkcolor=primary,
    citecolor=secondary,
    urlcolor=secondary
}

\title{""" + config.title + r"""}
\author{""" + config.author + r""" \\ """ + config.institution + r"""}
\date{""" + datetime.utcnow().strftime("%B %d, %Y") + r"""}

\begin{document}
\maketitle
\thispagestyle{empty}
\newpage
\tableofcontents
\newpage
"""

    @staticmethod
    def generate_executive_summary(content: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        return f"""
\\section{{Executive Summary}}
{content.get('summary', 'Executive summary not available.')}

\\subsection{{Key Findings}}
\\begin{{itemize}}
""" + "\n".join([f"\\item {finding}" for finding in content.get('key_findings', [])]) + """
\\end{{itemize}}

\\subsection{{Recommendations}}
\\begin{{itemize}}
""" + "\n".join([f"\\item {rec}" for rec in content.get('recommendations', [])]) + """
\\end{{itemize}}
"""

    @staticmethod
    def generate_methodology(content: Dict[str, Any]) -> str:
        """Generate methodology section."""
        return f"""
\\section{{Methodology}}
{content.get('overview', 'Methodology overview not available.')}

\\subsection{{Data Sources}}
{content.get('data_sources', 'Data sources not specified.')}

\\subsection{{Signal Generation}}
{content.get('signal_generation', 'Signal generation methodology not specified.')}

\\subsection{{Backtesting Framework}}
{content.get('backtesting', 'Backtesting framework not specified.')}

\\subsection{{Statistical Validation}}
{content.get('statistical_validation', 'Statistical validation not specified.')}

\\subsection{{Risk Management}}
{content.get('risk_management', 'Risk management not specified.')}
"""

    @staticmethod
    def generate_results_section(results: Dict[str, Any]) -> str:
        """Generate results section with tables."""
        latex = r"\section{Results}"
        
        # Summary metrics table
        if 'summary' in results:
            latex += """
\\subsection{Performance Summary}
\\begin{table}[htbp]
\\centering
\\caption{Key Performance Metrics}
\\begin{tabular}{lc}
\\toprule
Metric & Value \\\\
\\midrule
"""
            for metric, value in results['summary'].items():
                latex += f"{metric} & {value} \\\\\n"
            latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

        # Trade breakdown table
        if 'by_session' in results:
            latex += """
\\subsection{Performance by Session}
\\begin{table}[htbp]
\\centering
\\caption{Performance by Trading Session}
\\begin{tabular}{lcccc}
\\toprule
Session & Trades & Win Rate & Avg R-Multiple & Profit \\\\
\\midrule
"""
            for session, metrics in results['by_session'].items():
                latex += f"{session} & {metrics.get('trades', 0)} & {metrics.get('win_rate', 0):.1%} & {metrics.get('avg_r', 0):.2f} & \\${metrics.get('profit', 0):,.2f} \\\\\n"
            latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

        # Strategy comparison
        if 'by_strategy' in results:
            latex += """
\\subsection{Strategy Comparison}
\\begin{table}[htbp]
\\centering
\\caption{Strategy Version Comparison}
\\begin{tabular}{lccccc}
\\toprule
Version & Trades & Win Rate & Sharpe & Profit Factor & Status \\\\
\\midrule
"""
            for version, metrics in results['by_strategy'].items():
                latex += f"{version} & {metrics.get('trades', 0)} & {metrics.get('win_rate', 0):.1%} & {metrics.get('sharpe', 0):.2f} & {metrics.get('pf', 0):.2f} & {metrics.get('status', 'N/A')} \\\\\n"
            latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

        return latex

    @staticmethod
    def generate_critique_section(critique: Dict[str, Any]) -> str:
        """Generate critique section."""
        latex = r"\section{Critique and Validation}"
        
        latex += f"""
\\subsection{{Overall Assessment}}
Overall Score: {critique.get('overall_score', 0):.2f}/1.00 \\\\
Verdict: \textbf{{{critique.get('verdict', 'UNKNOWN')}}}
"""
        
        if critique.get('critical_issues'):
            latex += r"""
\subsection{Critical Issues}
\begin{itemize}
""" + "\n".join([f"\\item {issue}" for issue in critique['critical_issues']]) + r"""
\end{itemize}
"""
        
        if critique.get('warnings'):
            latex += r"""
\subsection{Warnings}
\begin{itemize}
""" + "\n".join([f"\\item {warning}" for warning in critique['warnings']]) + """
\end{itemize}
"""
        
        if critique.get('suggestions'):
            latex += r"""
\subsection{Suggestions for Improvement}
\begin{itemize}
""" + "\n".join([f"\\item {suggestion}" for suggestion in critique['suggestions']]) + """
\end{itemize}
"""
        
        return latex

    @staticmethod
    def generate_appendix(appendix_content: Dict[str, List[Any]]) -> str:
        """Generate appendix section."""
        latex = r"\appendix"
        
        if 'all_trades' in appendix_content:
            latex += r"""
\section{Complete Trade Log}
\begin{longtable}{lcccccccc}
\caption{Complete Trade Log} \\
\toprule
ID & Symbol & Side & Volume & Entry & Exit & P&L & R-Multiple & Reason \\
\midrule
\endfirsthead
\toprule
ID & Symbol & Side & Volume & Entry & Exit & P&L & R-Multiple & Reason \\
\midrule
\endhead
\bottomrule
\endfoot
"""
            for trade in appendix_content['all_trades'][-100:]:  # Last 100 trades
                latex += f"{trade.get('id', '')} & {trade.get('symbol', '')} & {trade.get('side', '')} & {trade.get('volume', 0):.2f} & {trade.get('entry_price', 0):.5f} & {trade.get('exit_price', 0):.5f} & {trade.get('profit', 0):+.2f} & {trade.get('r_multiple', 0):.2f} & {trade.get('reason', '')} \\\\\n"
            latex += r"""\bottomrule
\end{longtable}
"""
        
        if 'pattern_details' in appendix_content:
            latex += r"""
\section{Discovered Patterns}
"""
            for pattern in appendix_content.get('pattern_details', []):
                latex += f"""
\\subsection{{{pattern.get('name', 'Unknown')}}}
\textbf{{Type:}} {pattern.get('pattern_type', 'Unknown')} \\\\
\textbf{{Confidence:}} {pattern.get('confidence', 0):.2f} \\\\
\textbf{{Win Rate:}} {pattern.get('win_rate', 0):.1%} \\\\
\textbf{{Avg Return:}} {pattern.get('avg_return', 0):.4f} \\\\
\textbf{{Occurrences:}} {pattern.get('occurrences', 0)} \\\\
\textbf{{Description:}} {pattern.get('description', '')} \\\\
"""
        
        return latex


class ExecutiveSummaryGenerator:
    """Generates executive summaries for research reports."""

    @staticmethod
    def generate(hypothesis, pipeline_results: Dict[str, Any]) -> str:
        """Generate executive summary."""
        summary = []
        
        summary.append(f"# Executive Summary: {hypothesis.title}")
        summary.append("")
        summary.append(f"**Research Question:** {hypothesis.research_question}")
        summary.append("")
        
        # Get best result
        bt_output = pipeline_results.get('backtesting_output', {})
        best = bt_output.get('best_variant', {})
        
        if best.get('valid'):
            metrics = best.get('metrics', {}).get('summary', {})
            summary.append("## Key Results")
            summary.append(f"- **Total Return:** {best.get('metrics', {}).get('summary', {}).get('total_return_pct', 0):.2f}%")
            summary.append(f"- **Sharpe Ratio:** {metrics.get('sharpe_ratio', 0):.2f}")
            summary.append(f"- **Max Drawdown:** {metrics.get('max_drawdown_pct', 0):.2f}%")
            summary.append(f"- **Win Rate:** {metrics.get('win_rate', 0):.1%}")
            summary.append(f"- **Profit Factor:** {metrics.get('profit_factor', 0):.2f}")
            summary.append(f"- **Total Trades:** {metrics.get('total_trades', 0)}")
            summary.append("")
        
        # Critique summary
        critique = pipeline_results.get('critique_output', {}).get('critique', {})
        if critique:
            summary.append("## Validation")
            summary.append(f"- **Overall Score:** {critique.get('overall_score', 0):.2f}/1.00")
            summary.append(f"- **Verdict:** {critique.get('verdict', 'UNKNOWN')}")
            if critique.get('critical_issues'):
                summary.append(f"- **Critical Issues:** {len(critique['critical_issues'])} found")
            summary.append("")
        
        # Knowledge integration
        ki = pipeline_results.get('knowledge_integration_output', {})
        if ki:
            summary.append("## Knowledge Integration")
            summary.append(f"- **Concepts Added:** {ki.get('concepts_added', 0)}")
            summary.append(f"- **Relationships Inferred:** {ki.get('relationships_inferred', 0)}")
            summary.append("")
        
        return "\n".join(summary)


class ResearchReportGenerator:
    """
    Generates comprehensive research reports in multiple formats.
    """

    def __init__(
        self,
        config: ReportConfig = None,
        knowledge_agent: Any = None,
        event_bus = None,
    ):
        self.config = config or ReportConfig()
        self.knowledge_agent = knowledge_agent
        self.latex_generator = LatexGenerator()
        self.executive_generator = ExecutiveSummaryGenerator()
        self.event_bus = event_bus

    async def generate_report(
        self,
        hypothesis: Hypothesis,
        pipeline_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a complete research report."""
        
        # Generate all sections
        executive_summary = self.executive_generator.generate(
            self.config, pipeline_results
        )
        
        latex = self.latex_generator.generate_preamble(self.config)
        
        # Executive summary
        latex += self.latex_generator.generate_executive_summary({
            'summary': self.executive_generator.generate(self.config, {'backtesting_output': pipeline_results.get('backtesting_output', {})}),
            'key_findings': self._extract_key_findings(pipeline_results),
            'recommendations': self._generate_recommendations(pipeline_results),
        })
        
        # Methodology
        latex += self.latex_generator.generate_methodology({
            'overview': 'Multi-agent research pipeline with literature review, quant formalization, backtesting, critique, statistical validation, and knowledge integration.',
            'data_sources': 'Academic papers, practitioner blogs, internal knowledge base',
            'signal_generation': 'Automated signal generation via quant agent with parameter space exploration',
            'backtesting': 'Event-driven backtest engine with realistic execution models',
            'statistical_validation': 'Walk-forward analysis, Monte Carlo simulation, bootstrap confidence intervals',
            'risk_management': 'Fixed fractional position sizing, correlation limits, daily loss limits',
        })
        
        # Results
        bt_output = pipeline_results.get('backtesting_output', {})
        best = bt_output.get('best_variant', {})
        
        if best.get('valid'):
            latex += self.latex_generator.generate_results_section({
                'summary': best.get('metrics', {}).get('summary', {}),
                'by_session': best.get('metrics', {}).get('breakdowns', {}).get('by_session', {}),
                'by_strategy': pipeline_results.get('knowledge_integration_output', {}).get('strategies', {}),
            })
        
        # Critique
        critique = pipeline_results.get('critique_output', {}).get('critique', {})
        if critique:
            latex += self.latex_generator.generate_critique_section(critique)
        
        # Appendix
        latex += self.latex_generator.generate_appendix({
            'all_trades': pipeline_results.get('backtesting_output', {}).get('best_variant', {}).get('metrics', {}).get('trades', []),
            'pattern_details': [],  # Would be populated from knowledge agent
        })
        
        # Close document
        latex += r"\end{document}"
        
        # Save to file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{hypothesis.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.tex"
        
        # Also generate HTML version
        html_report = self._generate_html_report(hypothesis, pipeline_results)
        
        return {
            "latex": latex,
            "html": html_report,
            "filename": filename,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _extract_key_findings(self, pipeline_results: Dict) -> List[str]:
        findings = []
        
        bt = pipeline_results.get('backtesting_output', {})
        best = bt.get('best_variant', {})
        
        if best.get('valid'):
            metrics = best.get('metrics', {}).get('summary', {})
            if metrics.get('total_return_pct', 0) > 0:
                findings.append(f"Positive total return of {metrics.get('total_return_pct', 0):.2f}% achieved")
            if metrics.get('sharpe_ratio', 0) > 1:
                findings.append(f"Sharpe ratio of {metrics.get('sharpe_ratio', 0):.2f} exceeds 1.0 threshold")
            if metrics.get('profit_factor', 0) > 1.5:
                findings.append(f"Profit factor of {metrics.get('profit_factor', 0):.2f} indicates positive expectancy")
        
        critique = pipeline_results.get('critique_output', {}).get('critique', {})
        if critique.get('verdict') == 'PASS':
            findings.append("All critical validation checks passed")
        
        return findings or ["No significant findings"]

    def _generate_recommendations(self, pipeline_results: Dict) -> List[str]:
        recommendations = []
        
        critique = pipeline_results.get('critique_output', {}).get('critique', {})
        if critique.get('suggestions'):
            recommendations.extend(critique['suggestions'][:3])
        
        bt = pipeline_results.get('backtesting_output', {})
        best = bt.get('best_variant', {})
        if best.get('valid'):
            metrics = best.get('metrics', {}).get('summary', {})
            if metrics.get('max_drawdown_pct', 0) > 10:
                recommendations.append("Consider reducing position sizes to limit maximum drawdown")
            if metrics.get('win_rate', 0) < 0.4:
                recommendations.append("Review entry criteria to improve win rate")
        
        recommendations.append("Conduct walk-forward analysis on additional out-of-sample data")
        recommendations.append("Test strategy robustness across different market regimes")
        recommendations.append("Implement live paper trading before capital allocation")
        
        return recommendations[:5]

    def _generate_html_report(self, hypothesis: Hypothesis, pipeline_results: Dict) -> str:
        """Generate HTML version of the report."""
        # Simplified HTML generation
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Research Report: {hypothesis.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #1a1a2e; }}
        .metric {{ display: inline-block; background: #f8f9fa; padding: 15px; margin: 10px; border-radius: 8px; min-width: 200px; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #1a1a2e; }}
        .positive {{ color: #27ae60; }} .negative {{ color: #e74c3c; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
        th {{ background: #3498db; color: white; }}
    </style>
</head>
<body>
    <h1>{hypothesis.title}</h1>
    <p><strong>Research Question:</strong> {hypothesis.research_question}</p>
    <p><strong>Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    
    <h2>Executive Summary</h2>
    <p>{self._extract_key_findings({'backtesting_output': pipeline_results.get('backtesting_output', {})})}</p>
    
    <h2>Key Metrics</h2>
    <div class="metrics-container">
        {self._generate_html_metrics(pipeline_results)}
    </div>
</body>
</html>
"""

    def _generate_html_metrics(self, pipeline_results: Dict) -> str:
        bt = pipeline_results.get('backtesting_output', {})
        best = bt.get('best_variant', {})
        
        if not best.get('valid'):
            return "<p>No valid backtest results</p>"
        
        metrics = best.get('metrics', {}).get('summary', {})
        html = ""
        metrics_to_show = [
            ('Total Return', f"{metrics.get('total_return_pct', 0):.2f}%", 'positive' if metrics.get('total_return_pct', 0) > 0 else 'negative'),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}", 'positive' if metrics.get('sharpe_ratio', 0) > 1 else 'negative'),
            ('Max Drawdown', f"{metrics.get('max_drawdown_pct', 0):.2f}%", 'negative'),
            ('Win Rate', f"{metrics.get('win_rate', 0):.1%}", 'positive' if metrics.get('win_rate', 0) > 0.5 else 'negative'),
            ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}", 'positive' if metrics.get('profit_factor', 0) > 1.5 else 'negative'),
            ('Total Trades', str(metrics.get('total_trades', 0)), ''),
        ]
        
        for label, value, cls in metrics_to_show:
            cls_attr = f' class="{cls}"' if cls else ''
            html += f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value {cls}">{value}</div></div>'
        
        return html


# Factory function
def create_report_generator(
    config: ReportConfig = None,
    knowledge_agent = None,
    event_bus = None,
) -> ResearchReportGenerator:
    """Create a research report generator."""
    return ResearchReportGenerator(
        config=config,
        knowledge_agent=knowledge_agent,
    )
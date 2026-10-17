"""Autonomy loop: propose → validate → version → probation → human approval.

Modules
-------
``validator``
    Deterministic gate: compiles a spec, backtests it, walk-forwards it,
    and only passes strategies that survive out-of-sample + overfitting
    checks. Every rejection carries explicit machine-readable reasons.
``proposal``
    The LLM step: research brief in, strategy-spec JSON out (fallback spec
    when no LLM). Then runs the validator and versions survivors.
``probation``
    Paper-trading probation arena: a candidate must earn N trades / M days
    of paper performance before it can reach the human approval queue.

The activation decision itself is ALWAYS the human's (Level 3→4 wall).
"""

from .probation import ProbationStore, ProbationStatus
from .proposal import ProposalResult, propose_for_pair
from .validator import BacktestValidator, SpecSignalGenerator, ValidationReport

__all__ = [
    "ProbationStore",
    "ProbationStatus",
    "ProposalResult",
    "propose_for_pair",
    "BacktestValidator",
    "SpecSignalGenerator",
    "ValidationReport",
]

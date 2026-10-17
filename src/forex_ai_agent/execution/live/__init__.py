"""Stage E4: extremely limited live deployment package.

The pieces that turn an approved E3 intent into a transmitted order and
back into auditable numbers:

* :mod:`caps`         — the E4 hard caps (frozen constants)
* :mod:`retcodes`     — BROKER_FAULT_TABLE: every retcode owns a verdict
* :mod:`sessions`     — London/New York UTC session gate
* :mod:`news`         — +/-30 min blackout around high-impact events
* :mod:`fills`        — expected-vs-actual fill reconciliation in pips
* :mod:`tracker`      — broker positions + realized P&L ledger + sync
* :mod:`signoff`      — recorded, time-boxed human sign-off (append-only)
* :mod:`expiry`       — placed-order TTL tracker (cancel after 300s)
* :mod:`transmitter`  — the one guarded path to the wire
* :mod:`daily_report` — deterministic daily/weekly accounting
* :mod:`evaluation`   — 30-day window scoring vs declared criteria

Constitution reminder: nothing here runs at ``MAX_PERMISSION_LEVEL=0``.
The transmitter's gate chain includes the broker adapter's own
permission + shadow-mode refusal, so even a fully wired transmitter
cannot send before the human sign-off grants AUTONOMOUS(4) and shadow
mode is explicitly switched off.
"""

from .caps import (
    ALEX_LIVE_MAGIC,
    DEFAULT_LIVE_CAPS,
    E4_PAIR,
    LiveCaps,
    london_window,
    new_york_window,
)
from .evaluation import (
    CRITERION_AUDIT,
    CRITERION_PNL_BAND,
    CRITERION_RECONCILIATION,
    CRITERION_SAFETY,
    EVAL_ELIGIBLE,
    EVAL_EXTEND_REVIEW,
    EVAL_RETIRE_RECOMMENDED,
    EvaluationCriterion,
    EvaluationReport,
    evaluate_window,
)
from .expiry import ExpiredOrder, OrderExpiry, TrackedOrder
from .daily_report import (
    DailyReport,
    TradeRecord,
    build_daily_report,
    format_daily_report_markdown,
)
from .fills import (
    FILL_DRIFT,
    FILL_EXACT,
    FILL_VOLUME_MISMATCH,
    FILL_WITHIN_TOLERANCE,
    FillReconciliation,
    reconcile_fill,
    signed_slippage_pips,
    summarize_reconciliations,
)
from .news import BlackoutCheck, NewsBlackout, NewsEvent
from .retcodes import (
    BROKER_FAULT_TABLE,
    VERDICT_CRITICAL,
    VERDICT_FILLED,
    VERDICT_PARTIAL,
    VERDICT_PLACED,
    VERDICT_REJECTED,
    VERDICT_RETRY,
    FaultAction,
    classify_retcode,
)
from .sessions import (
    SESSION_LONDON,
    SESSION_LONDON_NY,
    SESSION_NEW_YORK,
    SessionCheck,
    SessionFilter,
)
from .tracker import EquitySync, PositionBook, PositionTracker, sync_account_into
from .signoff import SCOPE_DRILL, SCOPE_WINDOW, LiveSignoff, SignoffStore
from .transmitter import (
    TRANSMIT_CRITICAL,
    TRANSMIT_FILLED,
    TRANSMIT_PARTIAL,
    TRANSMIT_PLACED,
    TRANSMIT_REFUSED_BROKER_GATE,
    TRANSMIT_REFUSED_EXPIRED,
    TRANSMIT_REFUSED_GOVERNOR,
    TRANSMIT_REFUSED_INTENT,
    TRANSMIT_REFUSED_NEWS,
    TRANSMIT_REFUSED_PAIR,
    TRANSMIT_REFUSED_SESSION,
    TRANSMIT_REFUSED_TOKEN,
    TRANSMIT_REFUSED_VOLUME,
    TRANSMIT_REJECTED,
    TRANSMIT_RETRY,
    LiveTransmitter,
    TransmissionResult,
)

__all__ = [
    # caps
    "E4_PAIR",
    "ALEX_LIVE_MAGIC",
    "LiveCaps",
    "DEFAULT_LIVE_CAPS",
    "london_window",
    "new_york_window",
    # retcodes
    "VERDICT_FILLED",
    "VERDICT_PLACED",
    "VERDICT_PARTIAL",
    "VERDICT_RETRY",
    "VERDICT_REJECTED",
    "VERDICT_CRITICAL",
    "FaultAction",
    "BROKER_FAULT_TABLE",
    "classify_retcode",
    # sessions
    "SESSION_LONDON",
    "SESSION_NEW_YORK",
    "SESSION_LONDON_NY",
    "SessionCheck",
    "SessionFilter",
    # news
    "NewsEvent",
    "BlackoutCheck",
    "NewsBlackout",
    # fills
    "FILL_EXACT",
    "FILL_WITHIN_TOLERANCE",
    "FILL_DRIFT",
    "FILL_VOLUME_MISMATCH",
    "FillReconciliation",
    "signed_slippage_pips",
    "reconcile_fill",
    "summarize_reconciliations",
    # tracker
    "PositionBook",
    "EquitySync",
    "PositionTracker",
    "sync_account_into",
    # signoff
    "SCOPE_DRILL",
    "SCOPE_WINDOW",
    "LiveSignoff",
    "SignoffStore",
    # transmitter
    "TRANSMIT_FILLED",
    "TRANSMIT_PLACED",
    "TRANSMIT_PARTIAL",
    "TRANSMIT_RETRY",
    "TRANSMIT_REJECTED",
    "TRANSMIT_CRITICAL",
    "TRANSMIT_REFUSED_INTENT",
    "TRANSMIT_REFUSED_EXPIRED",
    "TRANSMIT_REFUSED_PAIR",
    "TRANSMIT_REFUSED_VOLUME",
    "TRANSMIT_REFUSED_SESSION",
    "TRANSMIT_REFUSED_NEWS",
    "TRANSMIT_REFUSED_GOVERNOR",
    "TRANSMIT_REFUSED_BROKER_GATE",
    "TRANSMIT_REFUSED_TOKEN",
    "LiveTransmitter",
    "TransmissionResult",
    # report
    "TradeRecord",
    "DailyReport",
    "build_daily_report",
    "format_daily_report_markdown",
    # expiry
    "TrackedOrder",
    "ExpiredOrder",
    "OrderExpiry",
    # evaluation
    "CRITERION_SAFETY",
    "CRITERION_PNL_BAND",
    "CRITERION_RECONCILIATION",
    "CRITERION_AUDIT",
    "EVAL_ELIGIBLE",
    "EVAL_EXTEND_REVIEW",
    "EVAL_RETIRE_RECOMMENDED",
    "EvaluationCriterion",
    "EvaluationReport",
    "evaluate_window",
]

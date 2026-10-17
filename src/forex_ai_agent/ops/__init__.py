"""Operational tooling: scheduled re-verification of the Stage E gates.

The Stage E ladder ships each rung as *test → live verify → doc update →
suite count*. A one-off live verify proves the rung on the day it lands;
this package keeps the proof honest afterwards by re-running the same
live scripts on a schedule and recording only what they actually said.

Nothing here trades, reads a broker or holds a permission: it launches a
verify script as a subprocess, parses the ``RESULT:`` line that script
printed, and appends one greppable line to a log. The zero-orders property
is re-checked by the verify script itself; this layer's job is to make that
re-check continuous and impossible to fake (a green line requires the
literal ``RESULT: PASS`` *and* exit code 0 *and* no order evidence).
"""

from .daily_verify import (
    EXIT_CODES,
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    Outcome,
    format_log_line,
    parse_verify_output,
    run_daily_check,
)
from .daily_schedule import (
    DEFAULT_TASK_NAME,
    DEFAULT_TIME,
    build_create_argv,
    build_delete_argv,
    build_query_argv,
    build_run_argv,
    build_task_command,
    install_task,
    is_windows,
    parse_task_detail,
    run_task_now,
    status_task,
    uninstall_task,
)

__all__ = [
    "EXIT_CODES",
    "STATUS_BLOCKED",
    "STATUS_ERROR",
    "STATUS_FAIL",
    "STATUS_PASS",
    "STATUS_SKIP",
    "Outcome",
    "format_log_line",
    "parse_verify_output",
    "run_daily_check",
    "DEFAULT_TASK_NAME",
    "DEFAULT_TIME",
    "build_create_argv",
    "build_delete_argv",
    "build_query_argv",
    "build_run_argv",
    "build_task_command",
    "install_task",
    "is_windows",
    "parse_task_detail",
    "run_task_now",
    "status_task",
    "uninstall_task",
]

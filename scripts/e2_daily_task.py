"""Manage the daily E2 zero-orders re-check in Windows Task Scheduler.

    python scripts/e2_daily_task.py install [--time 07:15] [--task-name ...]
    python scripts/e2_daily_task.py status
    python scripts/e2_daily_task.py run-now        # fire it without waiting
    python scripts/e2_daily_task.py uninstall

The task runs the project's virtualenv Python on
``scripts/e2_daily_check.py`` once a day, which re-runs the E2 live verify
and logs its RESULT line to ``logs/daily_verify/e2_live_verify.log``. The
task runs as the current user and only while that user is logged on, which
is what the MT5 terminal needs for its IPC connection.

Nothing here reads the broker: it registers (or removes) a schedule and
reports what Windows stored.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_ai_agent.ops.daily_schedule import (  # noqa: E402
    DEFAULT_TASK_NAME,
    DEFAULT_TIME,
    install_task,
    is_windows,
    run_task_now,
    status_task,
    uninstall_task,
)

DEFAULT_CHECK_SCRIPT = ROOT / "scripts" / "e2_daily_check.py"
DEFAULT_LOG = ROOT / "logs" / "daily_verify" / "e2_live_verify.log"


def _default_python() -> Path:
    candidate = ROOT / ".venv311" / "Scripts" / "python.exe"
    return candidate if candidate.is_file() else Path(sys.executable)


def _print_result(result) -> None:
    print(f"  command: {' '.join(result.argv) if result.argv else '-'}")
    if result.output:
        print(f"  output:  {result.output}")
    if result.task:
        print(
            "  task:    "
            f"{result.field('TaskName')} | next={result.field('Next Run Time')} "
            f"| last={result.field('Last Run Time')} "
            f"| last_result={result.field('Last Result')} "
            f"| runs={result.field('Task To Run')}"
        )


def _safe_console() -> None:
    """schtasks reports are localised; never crash printing them."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> int:
    _safe_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("install", "status", "run-now", "uninstall"))
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--time", dest="time_hhmm", default=DEFAULT_TIME,
                        help="daily start time, HH:MM local (default 07:15)")
    parser.add_argument("--python", default=str(_default_python()),
                        help="interpreter the task should use")
    parser.add_argument("--check-script", default=str(DEFAULT_CHECK_SCRIPT))
    args = parser.parse_args(argv)

    if not is_windows():
        print("Task Scheduler is Windows-only; this action needs a POSIX equivalent.")

    try:
        if args.action == "install":
            result = install_task(
                Path(args.check_script),
                task_name=args.task_name,
                time_hhmm=args.time_hhmm,
                python_exe=Path(args.python),
            )
        elif args.action == "status":
            result = status_task(task_name=args.task_name)
        elif args.action == "run-now":
            result = run_task_now(task_name=args.task_name)
        else:
            result = uninstall_task(task_name=args.task_name)
    except ValueError as exc:
        print(f"[e2_daily_task] {exc}")
        return 2

    print(f"[e2_daily_task] {args.action}: {'OK' if result.ok else 'FAILED'}")
    _print_result(result)
    if args.action == "install" and result.ok:
        print(f"  daily at {args.time_hhmm} local; log: {DEFAULT_LOG}")
        print(f"  prove it now:  {sys.executable} {Path(args.check_script).name}"
              f"   (or: python {Path(__file__).name} run-now)")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Daily scheduled run of the E2 live verify, logging its RESULT line.

This is what the Windows scheduled task runs (see
``scripts/e2_daily_task.py``). It re-runs the read-only + shadow live
verification against the real MT5 terminal and appends one greppable line
per run to ``logs/daily_verify/e2_live_verify.log`` — the literal
``RESULT:`` line plus the zero-orders evidence — so the proof that nothing
was sent is re-checked continuously instead of once.

Every path is resolved from this file, so it works from any working
directory (Task Scheduler sets none).

Run:  .venv311/Scripts/python.exe scripts/e2_daily_check.py
Exit: 0 PASS · 1 FAIL · 2 SKIP (not attempted) · 3 BLOCKED (no terminal)
      · 4 ERROR (crash / timeout / no verdict)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_ai_agent.ops.daily_verify import (  # noqa: E402
    DEFAULT_KEEP_LINES,
    DEFAULT_TIMEOUT_SECONDS,
    run_daily_check,
)

DEFAULT_VERIFY = ROOT / "scripts" / "e2_live_verify.py"
DEFAULT_LOG_DIR = ROOT / "logs" / "daily_verify"


def _safe_console() -> None:
    """Never crash on the verify script's box-drawing/arrow characters.

    Task Scheduler captures stdout, but a manual run in a cp1252 console
    would otherwise die on a UnicodeEncodeError while echoing the report.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> int:
    _safe_console()
    parser = argparse.ArgumentParser(
        description="Run the E2 live verify and log its RESULT line.",
    )
    parser.add_argument("--script", default=str(DEFAULT_VERIFY),
                        help="verify script to run (default: scripts/e2_live_verify.py)")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR),
                        help="where the run log and JSON sidecar live")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
                        help="hard limit for one run, in seconds")
    parser.add_argument("--keep-lines", type=int, default=DEFAULT_KEEP_LINES,
                        help="log lines retained (oldest are trimmed)")
    parser.add_argument("--show-output", action="store_true",
                        help="echo the verify script's full output too")
    args = parser.parse_args(argv)

    outcome = run_daily_check(
        Path(args.script),
        log_dir=Path(args.log_dir),
        timeout=args.timeout,
        keep_lines=args.keep_lines,
        root=ROOT,
    )

    if args.show_output and outcome.output:
        print(outcome.output)
    log_file = Path(args.log_dir) / f"{Path(args.script).stem}.log"
    print(
        f"[e2_daily_check] status={outcome.status} rc={outcome.returncode} "
        f"result={outcome.result_line or '-'} "
        f"ev={outcome.evidence_str() or '-'} "
        f"log={log_file}"
    )
    print(f"[e2_daily_check] logged: {_last_line(log_file)}")
    return outcome.exit_code


def _last_line(path: Path) -> str:
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return "(log unavailable)"
    return lines[-1] if lines else "(empty)"


if __name__ == "__main__":
    sys.exit(main())

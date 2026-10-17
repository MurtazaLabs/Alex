"""Run a live-verify script and log the ``RESULT:`` line it printed.

Continuous re-verification for the Stage E gates. The runner:

* launches the verify script in a *subprocess* with the same interpreter,
  from the repository root, so it works identically from a shell, from
  Task Scheduler (which sets no working directory) and from CI;
* parses the script's own verdict — the ``RESULT: PASS`` / ``RESULT: FAIL``
  line — and never invents one;
* appends one greppable line per run to ``<log_dir>/<script stem>.log``,
  carrying the verbatim RESULT line plus the zero-orders evidence it could
  read (``sent_calls``, ``sent_writes``, magic-tagged deals, per-check
  flags), so the log says *what* was re-checked, not merely that a process
  exited;
* mirrors the latest verdict into a JSON sidecar for dashboards;
* reports a distinct status for "not attempted" (SKIP) and "no terminal"
  (BLOCKED) so a weekly "terminal was closed" night does not look like a
  failed proof — while never turning an unverified night green.

Statuses and exit codes::

    PASS    0   the script printed RESULT: PASS and exited 0
    FAIL    1   the script printed RESULT: FAIL, or order evidence was found
    SKIP    2   not attempted — the script printed its own ``SKIP:`` line
    BLOCKED 3   could not reach the terminal — nothing was verified today
    ERROR   4   crash, timeout, or no parseable verdict at all

The rule that matters: a PASS requires the literal ``RESULT: PASS`` line
**and** exit code 0 **and** zero order evidence. Any mismatch is a FAIL.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"
STATUS_BLOCKED = "BLOCKED"
STATUS_ERROR = "ERROR"

EXIT_CODES: Dict[str, int] = {
    STATUS_PASS: 0,
    STATUS_FAIL: 1,
    STATUS_SKIP: 2,
    STATUS_BLOCKED: 3,
    STATUS_ERROR: 4,
}

DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_KEEP_LINES = 400
DEFAULT_HISTORY = 30
_LOCK_STALE_SECONDS = 3600.0

_RESULT_RE = re.compile(r"^\s*RESULT:\s*(PASS|FAIL)\s*$", re.MULTILINE)
_SKIP_RE = re.compile(r"^\s*SKIP:", re.MULTILINE)
_CHECK_RE = re.compile(r"^\s*Check (\d+)\b[^:]*:\s*(True|False)\s*$", re.MULTILINE)
_FAIL_RE = re.compile(r"^\s*(?:FAIL|Traceback|ERROR)\b.*$", re.MULTILINE)
_LINE_RE = re.compile(
    r"^\S+Z status=(?P<status>\S+) rc=(?P<rc>-?\d+) dur=(?P<dur>\S+) "
    r"run=(?P<run>\d+) streak=(?P<streak>[+-]?\d+) prev=(?P<prev>\S+)"
)
_STATUS_RE = re.compile(r"\bstatus=(\S+)")
_STREAK_RE = re.compile(r"\bstreak=([+-]?\d+)")

# Markers that mean "the terminal was not there to read". Both are printed
# by the E scripts on a connect failure; neither is a shadow failure.
_BLOCKED_MARKERS: Tuple[str, ...] = (
    "cannot connect to the MT5 terminal",
    "MetaTrader5 package is not installed",
    "MetaTrader5 not installed",
    "initialize() failed",
    "terminal not found",
)

# Zero-orders evidence lifted verbatim out of the verify script's output.
#: (field, pattern, group) — group defaults to 1 when the third item is omitted.
#: ``magic_deals`` reads the COUNT in "<n> with magic <magic>", never the
#: magic number itself: the count is what must be 0.
_EVIDENCE_PATTERNS: Tuple[Tuple[str, re.Pattern, int], ...] = (
    ("sent_calls", re.compile(r"sent_calls=(\d+)"), 1),
    ("sent_writes", re.compile(r"sent_writes=(\d+)"), 1),
    ("magic_deals", re.compile(r"(\d+)\s+with magic\s+(\d+)"), 1),
)


@dataclass(frozen=True)
class Outcome:
    """What one verification run actually said."""

    status: str
    returncode: Optional[int]
    result_line: Optional[str]
    detail: str = ""
    evidence: Mapping[str, int] = field(default_factory=dict)
    checks: str = ""
    output: str = ""
    skipped: bool = False  # True when nothing was attempted at all

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.status, EXIT_CODES[STATUS_ERROR])

    @property
    def passed(self) -> bool:
        return self.status == STATUS_PASS

    def evidence_str(self) -> str:
        parts = [f"{k}={v}" for k, v in sorted(self.evidence.items())]
        if self.checks:
            parts.append(f"checks={self.checks}")
        return ",".join(parts)


def _one_line(text: str, limit: int = 200) -> str:
    """Collapse arbitrary text into a single log-safe, bounded field."""
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        flat = flat[: limit - 3] + "..."
    return flat


def _last_meaningful_line(output: str) -> str:
    """The most useful failure line for a human reading the log."""
    matches = [m.group(0).strip() for m in _FAIL_RE.finditer(output or "")]
    if matches:
        return matches[-1]
    lines = [ln.strip() for ln in (output or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _check_digest(output: str) -> str:
    """``1111`` — each numbered ``Check N`` flag in order (True→1/False→0)."""
    coded = {int(n): 1 if flag == "True" else 0 for n, flag in _CHECK_RE.findall(output or "")}
    return "".join(str(coded[i]) for i in sorted(coded))


def parse_verify_output(output: str, returncode: int) -> Outcome:
    """Turn a verify script's (stdout, exit code) into a typed :class:`Outcome`.

    Never raises: an unparseable run is an ERROR, which is not green.
    """
    output = output or ""
    result_match = _RESULT_RE.search(output)
    result_line = f"RESULT: {result_match.group(1)}" if result_match else None

    evidence: Dict[str, int] = {}
    for name, pattern, group in _EVIDENCE_PATTERNS:
        match = pattern.search(output)
        if match:
            evidence[name] = int(match.group(group))

    checks = _check_digest(output)

    escaped = False
    if evidence.get("magic_deals"):  # an order carrying the reserved magic exists
        escaped = True
    if evidence.get("sent_calls") or evidence.get("sent_writes"):
        escaped = True

    if escaped:
        return Outcome(
            status=STATUS_FAIL,
            returncode=returncode,
            result_line=result_line,
            detail=(
                "ORDER EVIDENCE: "
                + ", ".join(f"{k}={v}" for k, v in sorted(evidence.items()) if v)
                + " — a write escaped the shadow"
            ),
            evidence=evidence,
            checks=checks,
            output=output,
        )

    if result_match:
        verdict = result_match.group(1)
        if verdict == "FAIL":
            return Outcome(
                status=STATUS_FAIL,
                returncode=returncode,
                result_line=result_line,
                detail=_last_meaningful_line(output) or "RESULT: FAIL",
                evidence=evidence,
                checks=checks,
                output=output,
            )
        if returncode != 0:
            # The script claimed PASS but the process failed: not green.
            return Outcome(
                status=STATUS_FAIL,
                returncode=returncode,
                result_line=result_line,
                detail=f"exit code {returncode} with {result_line}",
                evidence=evidence,
                checks=checks,
                output=output,
            )
        return Outcome(
            status=STATUS_PASS,
            returncode=returncode,
            result_line=result_line,
            evidence=evidence,
            checks=checks,
            output=output,
        )

    if _SKIP_RE.search(output):
        # A printed SKIP: line is required — a bare exit code 2 is not a
        # verdict (Python also exits 2 for a bad script path), and calling
        # that "not attempted" would hide a broken schedule.
        reason = _last_meaningful_line(output) or "verification not attempted"
        return Outcome(
            status=STATUS_SKIP,
            returncode=returncode,
            result_line=None,
            detail=reason,
            evidence=evidence,
            checks=checks,
            output=output,
            skipped=True,
        )

    if any(marker in output for marker in _BLOCKED_MARKERS):
        return Outcome(
            status=STATUS_BLOCKED,
            returncode=returncode,
            result_line=None,
            detail=_last_meaningful_line(output) or "terminal unavailable",
            evidence=evidence,
            checks=checks,
            output=output,
        )

    return Outcome(
        status=STATUS_ERROR,
        returncode=returncode,
        result_line=None,
        detail=_last_meaningful_line(output) or "no RESULT line and no recognisable failure",
        evidence=evidence,
        checks=checks,
        output=output,
    )


def _read_last_status(log_path: Path) -> Tuple[Optional[str], int]:
    """(status, streak) of the newest logged run, if any."""
    try:
        lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except (OSError, UnicodeDecodeError):
        return None, 0
    for line in reversed(lines):
        status_match = _STATUS_RE.search(line)
        streak_match = _STREAK_RE.search(line)
        if status_match:
            return status_match.group(1), int(streak_match.group(1)) if streak_match else 0
    return None, 0


def _count_lines(log_path: Path) -> int:
    try:
        return sum(1 for _ in log_path.open("r", encoding="utf-8", errors="replace"))
    except OSError:
        return 0


_RUN_RE = re.compile(r"\brun=(\d+)")


def _last_run_index(log_path: Path) -> int:
    """Highest run number still in the log (0 when there is none)."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    numbers = [int(m) for m in _RUN_RE.findall(text)]
    return max(numbers) if numbers else 0


def format_log_line(
    *,
    status: str,
    returncode: Optional[int],
    duration_s: float,
    run_index: int,
    streak: int,
    previous_status: Optional[str],
    result_line: Optional[str],
    detail: str = "",
    evidence: str = "",
) -> str:
    """One greppable log line carrying the verbatim verdict and its evidence."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    marker = ""
    if previous_status == STATUS_PASS and status != STATUS_PASS:
        marker = "!! REGRESSION "
    elif previous_status is not None and previous_status != STATUS_PASS and status == STATUS_PASS:
        marker = "** RECOVERED "
    parts = [
        f"{marker}{ts}",
        f"status={status}",
        f"rc={'none' if returncode is None else returncode}",
        f"dur={duration_s:.1f}s",
        f"run={run_index}",
        f"streak={streak:+d}",
        f"prev={previous_status or '-'}",
    ]
    if result_line:
        parts.append(f'result="{_one_line(result_line)}"')
    if evidence:
        parts.append(f"ev={evidence}")
    if detail and status != STATUS_PASS:
        parts.append(f'detail="{_one_line(detail)}"')
    return " ".join(parts)


def _trim_log(log_path: Path, keep_lines: int) -> None:
    if keep_lines <= 0:
        return
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    if len(lines) <= keep_lines:
        return
    tail = lines[-keep_lines:]
    tmp = log_path.with_suffix(log_path.suffix + ".tmp")
    tmp.write_text("\n".join(tail) + "\n", encoding="utf-8")
    os.replace(tmp, log_path)


def _write_sidecar(json_path: Path, payload: Mapping) -> None:
    history = list(payload.get("history", []))
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload, history=history), indent=2), encoding="utf-8")
    os.replace(tmp, json_path)


def _read_sidecar_history(json_path: Path, limit: int) -> List[Dict]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    history = data.get("history") or []
    return list(history)[-limit:]


class _CheckLock:
    """Cross-process guard so a slow run cannot overlap the next one."""

    def __init__(self, path: Path, stale_seconds: float = _LOCK_STALE_SECONDS) -> None:
        self.path = path
        self.stale_seconds = stale_seconds
        self.acquired = False
        self.holder_age: Optional[float] = None

    def __enter__(self) -> "_CheckLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age <= self.stale_seconds:
                self.holder_age = age
                return self
            # Stale (a killed run left it behind): reclaim the file, then
            # take it exclusively so a concurrent starter still loses.
            try:
                self.path.unlink()
            except OSError:
                pass
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            self.holder_age = time.time() - self.path.stat().st_mtime
            return self
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
        self.acquired = True
        return self

    def __exit__(self, *exc_info) -> bool:
        if self.acquired:
            try:
                self.path.unlink()
            except OSError:
                pass
        return False


def run_daily_check(
    script: Path,
    *,
    log_dir: Optional[Path] = None,
    python_exe: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    keep_lines: int = DEFAULT_KEEP_LINES,
    history_limit: int = DEFAULT_HISTORY,
    root: Optional[Path] = None,
    extra_args: Sequence[str] = (),
    env: Optional[Mapping[str, str]] = None,
    log: bool = True,
) -> Outcome:
    """Run ``script`` once, log its RESULT line, and return the outcome.

    ``root`` defaults to the verify script's grandparent directory (the
    repository root for the ``scripts/`` layout), which is what makes the
    command working-directory independent — Task Scheduler provides none.
    """
    script = Path(script).resolve()
    root = Path(root).resolve() if root else script.parent.parent
    log_dir = Path(log_dir).resolve() if log_dir else root / "logs" / "daily_verify"
    log_path = log_dir / f"{script.stem}.log"
    json_path = log_dir / f"{script.stem}.json"
    lock_path = log_dir / f"{script.stem}.lock"

    log_dir.mkdir(parents=True, exist_ok=True)
    previous_status, previous_streak = _read_last_status(log_path)
    previous_runs = max(_last_run_index(log_path), _read_sidecar_runs(json_path))

    with _CheckLock(lock_path, stale_seconds=max(_LOCK_STALE_SECONDS, timeout + 300.0)) as lock:
        if not lock.acquired:
            age = lock.holder_age or 0.0
            detail = f"another check is running (lock held for {age:.0f}s) — nothing verified this run"
            outcome = Outcome(
                status=STATUS_SKIP,
                returncode=None,
                result_line=None,
                detail=detail,
                skipped=True,
            )
            if log:
                _append_run(
                    log_path, json_path, script, outcome, 0.0,
                    previous_status, previous_streak, previous_runs, keep_lines,
                    history_limit, root,
                )
            return outcome

        started = time.monotonic()
        child_env = dict(os.environ)
        child_env["PYTHONIOENCODING"] = "utf-8"
        if env:
            child_env.update({str(k): str(v) for k, v in env.items()})
        argv = [python_exe or sys.executable, str(script), *extra_args]
        try:
            completed = subprocess.run(
                argv,
                cwd=str(root),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            partial = (exc.output or b"").decode("utf-8", "replace") if isinstance(exc.output, bytes) else (exc.output or "")
            outcome = Outcome(
                status=STATUS_ERROR,
                returncode=None,
                result_line=None,
                detail=f"timeout after {timeout:.0f}s — no verdict",
                output=partial,
            )
        except OSError as exc:
            duration = time.monotonic() - started
            outcome = Outcome(
                status=STATUS_ERROR,
                returncode=None,
                result_line=None,
                detail=f"could not launch {script.name}: {exc}",
            )
        else:
            duration = time.monotonic() - started
            output = (completed.stdout or b"").decode("utf-8", "replace")
            outcome = parse_verify_output(output, completed.returncode)

    if log:
        _append_run(
            log_path, json_path, script, outcome, duration,
            previous_status, previous_streak, previous_runs, keep_lines,
            history_limit, root,
        )
    return outcome


def _read_sidecar_runs(json_path: Path) -> int:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    try:
        return int(data.get("runs", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _append_run(
    log_path: Path,
    json_path: Path,
    script: Path,
    outcome: Outcome,
    duration: float,
    previous_status: Optional[str],
    previous_streak: int,
    previous_runs: int,
    keep_lines: int,
    history_limit: int,
    root: Path,
) -> str:
    if previous_status == outcome.status:
        streak = (previous_streak + 1) if previous_streak >= 0 else 1
    else:
        streak = -1 if outcome.status != STATUS_PASS else 1

    run_index = previous_runs + 1
    line = format_log_line(
        status=outcome.status,
        returncode=outcome.returncode,
        duration_s=duration,
        run_index=run_index,
        streak=streak,
        previous_status=previous_status,
        result_line=outcome.result_line,
        detail=outcome.detail,
        evidence=outcome.evidence_str(),
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    _trim_log(log_path, keep_lines)

    history = _read_sidecar_history(json_path, history_limit)
    history.append(
        {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": outcome.status,
            "rc": outcome.returncode,
            "duration_s": round(duration, 1),
            "result_line": outcome.result_line,
            "evidence": dict(outcome.evidence),
            "checks": outcome.checks,
            "detail": _one_line(outcome.detail),
        }
    )
    payload = {
        "script": str(script),
        "root": str(root),
        "log": str(log_path),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": run_index,
        "log_lines": _count_lines(log_path),
        "last": history[-1],
        "history": history[-history_limit:],
    }
    try:
        _write_sidecar(json_path, payload)
    except OSError:
        pass  # the log line is the deliverable; a missing sidecar is not a verdict
    return line

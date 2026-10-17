"""Register the daily re-verification as a Windows scheduled task.

Thin, testable wrapper around ``schtasks``: every argv list is built by a
pure function so the exact command can be asserted in unit tests without
touching the machine's task store, and the subprocess runner is injectable.

The registered task runs the check script with the project's virtualenv
interpreter. ``schtasks`` supplies no working directory, which is why the
check script resolves every path from its own ``__file__``.

On POSIX, :func:`crontab_line` prints the equivalent crontab entry instead
of pretending to register anything.
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

DEFAULT_TASK_NAME = "Alex-E2LiveVerify"
DEFAULT_TIME = "07:15"

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{1,2})$")
_TASK_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]*$")

_QUERY_LABELS = (
    "TaskName",
    "Status",
    "Next Run Time",
    "Last Run Time",
    "Last Result",
    "Task To Run",
    "Start In",
    "Schedule Type",
    "Start Time",
    "Run As User",
)


@dataclass(frozen=True)
class TaskResult:
    """Result of one ``schtasks`` interaction."""

    ok: bool
    argv: List[str] = field(default_factory=list)
    rc: Optional[int] = None
    output: str = ""
    task: Mapping[str, str] = field(default_factory=dict)

    def field(self, name: str, default: str = "-") -> str:
        return str(self.task.get(name, default))


def is_windows() -> bool:
    return platform.system().lower().startswith("win")


def normalize_time(value: str) -> str:
    """``7:5``/``07:05`` → ``07:05``; anything else is a ValueError."""
    match = _TIME_RE.match(str(value).strip())
    if not match:
        raise ValueError(f"start time must be HH:MM (24h), got {value!r}")
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"start time out of range: {value!r}")
    return f"{hour:02d}:{minute:02d}"


def validate_task_name(name: str) -> str:
    name = str(name).strip()
    if not _TASK_NAME_RE.match(name):
        raise ValueError(
            f"invalid task name {name!r}: letters, digits, space, dot, dash and "
            "underscore only"
        )
    if len(name) > 200:
        raise ValueError(f"task name too long ({len(name)} chars): {name[:40]!r}...")
    return name


def build_task_command(
    python_exe: Path,
    check_script: Path,
    extra_args: Sequence[str] = (),
) -> str:
    """The command string ``schtasks /TR`` will store (paths quoted: spaces)."""
    parts = [f'"{Path(python_exe)}"', f'"{Path(check_script)}"']
    parts.extend(str(a) for a in extra_args)
    return " ".join(parts)


def build_create_argv(
    task_name: str,
    python_exe: Path,
    check_script: Path,
    time_hhmm: str = DEFAULT_TIME,
    extra_args: Sequence[str] = (),
    force: bool = True,
    schtasks: str = "schtasks",
) -> List[str]:
    """``schtasks /Create`` for a daily run of ``check_script``."""
    name = validate_task_name(task_name)
    moment = normalize_time(time_hhmm)
    command = build_task_command(python_exe, check_script, extra_args)
    argv = [
        schtasks, "/Create",
        "/TN", name,
        "/SC", "DAILY",
        "/ST", moment,
        "/TR", command,
    ]
    if force:
        argv.append("/F")
    return argv


def build_query_argv(task_name: str, schtasks: str = "schtasks") -> List[str]:
    return [schtasks, "/Query", "/TN", validate_task_name(task_name), "/FO", "LIST", "/V"]


def build_run_argv(task_name: str, schtasks: str = "schtasks") -> List[str]:
    return [schtasks, "/Run", "/TN", validate_task_name(task_name)]


def build_delete_argv(task_name: str, schtasks: str = "schtasks") -> List[str]:
    return [schtasks, "/Delete", "/TN", validate_task_name(task_name), "/F"]


def parse_task_detail(text: str) -> Dict[str, str]:
    """Pick the useful fields out of ``schtasks /Query /FO LIST /V`` output.

    Field labels are localised by Windows, so unrecognised output degrades to
    a ``raw`` entry rather than raising.
    """
    found: Dict[str, str] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        label, value = label.strip(), value.strip()
        for known in _QUERY_LABELS:
            if label.lower() == known.lower():
                found[known] = value
                break
    if not found:
        found["raw"] = " ".join((text or "").split())[:500]
    return found


def crontab_line(python_exe: Path, check_script: Path, time_hhmm: str = DEFAULT_TIME) -> str:
    """The equivalent crontab entry (POSIX hosts)."""
    moment = normalize_time(time_hhmm)
    hour, minute = moment.split(":")
    return (
        f"{int(hour)} {int(minute)} * * * cd {Path(check_script).parent.parent} && "
        f"{Path(python_exe)} {Path(check_script)} >> logs/daily_verify/cron.log 2>&1"
    )


def _default_runner(argv: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def install_task(
    check_script: Path,
    *,
    task_name: str = DEFAULT_TASK_NAME,
    time_hhmm: str = DEFAULT_TIME,
    python_exe: Optional[Path] = None,
    extra_args: Sequence[str] = (),
    runner: Optional[Runner] = None,
    windows: Optional[bool] = None,
) -> TaskResult:
    """Create (or overwrite) the daily task and report what Windows stored."""
    check_script = Path(check_script)
    if not check_script.is_file():
        raise ValueError(f"check script not found: {check_script}")
    python_exe = Path(python_exe) if python_exe else Path(_default_python_exe())
    if not Path(python_exe).is_file():
        raise ValueError(f"interpreter not found: {python_exe}")
    windows = is_windows() if windows is None else windows
    argv = build_create_argv(task_name, python_exe, check_script, time_hhmm, extra_args)

    if not windows:
        return TaskResult(
            ok=False,
            argv=argv,
            rc=None,
            output=(
                "Task Scheduler is Windows-only; add this crontab entry instead: "
                + crontab_line(python_exe, check_script, time_hhmm)
            ),
        )

    run = runner or _default_runner
    result = run(argv)
    output = (result.stdout or "") + (result.stderr or "")
    ok = result.returncode == 0
    task: Dict[str, str] = {}
    if ok:
        task = parse_task_detail(run(build_query_argv(task_name)).stdout or "")
    return TaskResult(ok=ok, argv=argv, rc=result.returncode, output=output.strip(), task=task)


def uninstall_task(
    *,
    task_name: str = DEFAULT_TASK_NAME,
    runner: Optional[Runner] = None,
    windows: Optional[bool] = None,
) -> TaskResult:
    windows = is_windows() if windows is None else windows
    argv = build_delete_argv(task_name)
    if not windows:
        return TaskResult(ok=False, argv=argv, rc=None, output="Task Scheduler is Windows-only")
    result = (runner or _default_runner)(argv)
    return TaskResult(
        ok=result.returncode == 0,
        argv=argv,
        rc=result.returncode,
        output=((result.stdout or "") + (result.stderr or "")).strip(),
    )


def status_task(
    *,
    task_name: str = DEFAULT_TASK_NAME,
    runner: Optional[Runner] = None,
    windows: Optional[bool] = None,
) -> TaskResult:
    windows = is_windows() if windows is None else windows
    argv = build_query_argv(task_name)
    if not windows:
        return TaskResult(ok=False, argv=argv, rc=None, output="Task Scheduler is Windows-only")
    result = (runner or _default_runner)(argv)
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    return TaskResult(
        ok=result.returncode == 0,
        argv=argv,
        rc=result.returncode,
        output=output,
        task=parse_task_detail(output) if result.returncode == 0 else {},
    )


def run_task_now(
    *,
    task_name: str = DEFAULT_TASK_NAME,
    runner: Optional[Runner] = None,
    windows: Optional[bool] = None,
) -> TaskResult:
    """Ask Task Scheduler to fire the task immediately (no waiting for the clock)."""
    windows = is_windows() if windows is None else windows
    argv = build_run_argv(task_name)
    if not windows:
        return TaskResult(ok=False, argv=argv, rc=None, output="Task Scheduler is Windows-only")
    result = (runner or _default_runner)(argv)
    return TaskResult(
        ok=result.returncode == 0,
        argv=argv,
        rc=result.returncode,
        output=((result.stdout or "") + (result.stderr or "")).strip(),
    )


def _default_python_exe() -> str:
    import sys

    return sys.executable

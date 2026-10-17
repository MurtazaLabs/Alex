"""Tests for `forex_ai_agent.ops.daily_schedule`.

The scheduler is only allowed to be a thin, inspectable wrapper: these tests
assert the exact argv Windows would receive and drive every action through
an injected runner, so nothing here creates, runs or deletes a real
scheduled task.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forex_ai_agent.ops.daily_schedule import (
    DEFAULT_TASK_NAME,
    build_create_argv,
    build_delete_argv,
    build_query_argv,
    build_run_argv,
    build_task_command,
    crontab_line,
    install_task,
    normalize_time,
    parse_task_detail,
    run_task_now,
    status_task,
    uninstall_task,
    validate_task_name,
)

PYTHON = Path(r"E:\Forex trading bot\.venv311\Scripts\python.exe")
CHECK = Path(r"E:\Forex trading bot\scripts\e2_daily_check.py")

QUERY_OUTPUT = """\
Folder: \\
HostName:                             DESKTOP-K0GMQ5I
TaskName:                             \\Alex-E2LiveVerify
Next Run Time:                        2026-10-12 07:15:00
Status:                               Ready
Logon Mode:                           Interactive/Background
Last Run Time:                        2026-10-11 07:15:02
Last Result:                          0
Author:                               DESKTOP-K0GMQ5I\\Administrator
Task To Run:                          "E:\\Forex trading bot\\.venv311\\Scripts\\python.exe" "E:\\Forex trading bot\\scripts\\e2_daily_check.py"
Start In:                             N/A
Schedule Type:                        Daily
Start Time:                           07:15:00
"""


class _FakeRunner:
    """Records the argv it was handed and replays scripted results."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        out = self.outputs.pop(0) if self.outputs else ("", 0)
        if isinstance(out, tuple):
            stdout, rc = out
        else:
            stdout, rc = out, 0
        return subprocess.CompletedProcess(args=list(argv), returncode=rc, stdout=stdout, stderr="")


# ── input validation ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [("07:15", "07:15"), ("7:5", "07:05"), ("0:00", "00:00"), ("23:59", "23:59")],
)
def test_normalize_time(raw: str, expected: str):
    assert normalize_time(raw) == expected


@pytest.mark.parametrize("raw", ["", "7", "24:00", "07:60", "7am", "07:15:00", "-1:00"])
def test_normalize_time_rejects_nonsense(raw: str):
    with pytest.raises(ValueError):
        normalize_time(raw)


def test_task_name_validation():
    assert validate_task_name(" Alex-E2LiveVerify ") == "Alex-E2LiveVerify"
    assert validate_task_name("Alex.Stage-E2") == "Alex.Stage-E2"
    for bad in ("", "  ", "Alex/E2", "-leading-dash", "a" * 300):
        with pytest.raises(ValueError):
            validate_task_name(bad)


# ── argv builders ─────────────────────────────────────────────────────────


def test_task_command_quotes_both_paths():
    assert build_task_command(PYTHON, CHECK) == (
        f'"{PYTHON}" "{CHECK}"'
    )


def test_create_argv_is_a_daily_schedule_of_the_check_script():
    argv = build_create_argv(DEFAULT_TASK_NAME, PYTHON, CHECK, "7:15")
    assert argv[:2] == ["schtasks", "/Create"]
    assert argv[argv.index("/TN") + 1] == DEFAULT_TASK_NAME
    assert argv[argv.index("/SC") + 1] == "DAILY"
    assert argv[argv.index("/ST") + 1] == "07:15"
    assert argv[argv.index("/TR") + 1] == build_task_command(PYTHON, CHECK)
    assert "/F" in argv  # overwrite, so re-running install is idempotent


def test_create_argv_can_skip_the_force_flag():
    assert "/F" not in build_create_argv(DEFAULT_TASK_NAME, PYTHON, CHECK, force=False)


def test_query_run_and_delete_argv():
    assert build_query_argv(DEFAULT_TASK_NAME) == [
        "schtasks", "/Query", "/TN", DEFAULT_TASK_NAME, "/FO", "LIST", "/V",
    ]
    assert build_run_argv(DEFAULT_TASK_NAME) == [
        "schtasks", "/Run", "/TN", DEFAULT_TASK_NAME,
    ]
    assert build_delete_argv(DEFAULT_TASK_NAME) == [
        "schtasks", "/Delete", "/TN", DEFAULT_TASK_NAME, "/F",
    ]


def test_crontab_line_is_the_posix_equivalent():
    line = crontab_line(PYTHON, CHECK, "07:15")
    assert line.startswith("7 15 * * *")
    assert str(CHECK.parent.parent) in line
    assert str(CHECK) in line
    assert line.endswith("cron.log 2>&1")


# ── schtasks output parsing ───────────────────────────────────────────────


def test_parse_task_detail_picks_the_fields_we_care_about():
    fields = parse_task_detail(QUERY_OUTPUT)
    assert fields["Next Run Time"] == "2026-10-12 07:15:00"
    assert fields["Last Result"] == "0"
    assert fields["Schedule Type"] == "Daily"
    assert fields["Task To Run"].endswith('e2_daily_check.py"')


def test_parse_task_detail_degrades_instead_of_raising():
    fields = parse_task_detail("FEHLER: Der Task existiert nicht.")
    assert "raw" in fields


# ── actions (injected runner: no real task is touched) ────────────────────


def test_install_creates_then_reports_what_windows_stored():
    runner = _FakeRunner([("SUCCESS: The scheduled task was created.", 0), (QUERY_OUTPUT, 0)])
    result = install_task(CHECK, time_hhmm="7:15", python_exe=PYTHON, runner=runner, windows=True)

    assert result.ok is True
    assert "/Create" in result.argv
    assert runner.calls[0][runner.calls[0].index("/ST") + 1] == "07:15"
    assert runner.calls[1] == build_query_argv(DEFAULT_TASK_NAME)
    assert result.field("Next Run Time") == "2026-10-12 07:15:00"
    assert result.field("Status") == "Ready"


def test_install_reports_a_failure_without_claiming_success():
    runner = _FakeRunner([("ERROR: Access is denied.", 1)])
    result = install_task(CHECK, python_exe=PYTHON, runner=runner, windows=True)
    assert result.ok is False
    assert "Access is denied" in result.output
    assert len(runner.calls) == 1  # no pointless query after a failure


def test_install_off_windows_hands_back_the_crontab_line_instead():
    runner = _FakeRunner([])
    result = install_task(CHECK, python_exe=PYTHON, runner=runner, windows=False)
    assert result.ok is False
    assert runner.calls == []
    assert "crontab" in result.output


def test_install_validates_its_inputs():
    with pytest.raises(ValueError):
        install_task(Path("does/not/exist.py"), python_exe=PYTHON, windows=True)
    with pytest.raises(ValueError):
        install_task(CHECK, python_exe=Path("no/such/python.exe"), windows=True)
    with pytest.raises(ValueError):
        install_task(CHECK, python_exe=PYTHON, time_hhmm="99:99", windows=True)


def test_uninstall_status_and_run_now_use_the_right_subcommands():
    ok_query = _FakeRunner([(QUERY_OUTPUT, 0)])
    status = status_task(runner=ok_query, windows=True)
    assert status.ok and status.field("Last Result") == "0"

    delete = _FakeRunner([("SUCCESS: The scheduled task was deleted.", 0)])
    assert uninstall_task(runner=delete, windows=True).ok is True
    assert delete.calls[0] == build_delete_argv(DEFAULT_TASK_NAME)

    run = _FakeRunner([("SUCCESS: Attempted to run the scheduled task.", 0)])
    assert run_task_now(runner=run, windows=True).ok is True
    assert run.calls[0] == build_run_argv(DEFAULT_TASK_NAME)


def test_actions_off_windows_do_not_shell_out():
    runner = _FakeRunner([])
    for action in (uninstall_task, status_task, run_task_now):
        result = action(runner=runner, windows=False)
        assert result.ok is False
        assert "Windows-only" in result.output
    assert runner.calls == []

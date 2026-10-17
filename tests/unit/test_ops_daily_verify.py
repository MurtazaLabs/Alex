"""Tests for `forex_ai_agent.ops.daily_verify` — the continuous re-check.

Two properties are pinned here, above everything else:

1. a PASS in the log requires the literal ``RESULT: PASS`` line *and* exit
   code 0 *and* zero order evidence — the zero-orders proof can never be
   logged green from a broken or lying run;
2. every other outcome is distinguishable (FAIL / SKIP / BLOCKED / ERROR),
   so "the terminal was closed" never masquerades as either a pass or a
   proof failure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from forex_ai_agent.ops.daily_verify import (
    EXIT_CODES,
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    format_log_line,
    parse_verify_output,
    run_daily_check,
)

PASS_OUTPUT = """feed chain active source: mt5
fetched 520 real H1 bars, 22 trading days

Check 1 — read-only snapshot vs independent terminal read: True
  permission=0 shadow=True writes=0 audit=['CONTROL', 'READ']
Check 2 — reconciliation drill vs the real account (read-only): True
Check 3 — shadow run over 22 trading days: True
Check 4 — zero real orders (audit log + broker statement): True
  sent_calls=0 sent_writes=0
  broker history in window: 12 deal(s) total, 0 with magic 314159 (must be 0)
  position book unchanged: [] → []

RESULT: PASS
"""


def _stub(tmp_path: Path, name: str, body: str) -> Path:
    """A throwaway stand-in for the verify script (runs for real)."""
    script = tmp_path / f"{name}.py"
    script.write_text(body, encoding="utf-8")
    return script


# ── verdict parsing ───────────────────────────────────────────────────────


def test_pass_requires_the_result_line_and_a_zero_exit():
    outcome = parse_verify_output(PASS_OUTPUT, 0)
    assert outcome.status == STATUS_PASS
    assert outcome.result_line == "RESULT: PASS"
    assert outcome.exit_code == 0
    assert outcome.passed


def test_pass_result_with_a_nonzero_exit_is_never_green():
    outcome = parse_verify_output(PASS_OUTPUT, 1)
    assert outcome.status == STATUS_FAIL
    assert "exit code 1" in outcome.detail
    assert outcome.exit_code == 1


def test_fail_result_line_carries_the_reason():
    outcome = parse_verify_output(
        "FAIL: only 3 trading days, need >=10\nRESULT: FAIL\n", 1
    )
    assert outcome.status == STATUS_FAIL
    assert outcome.result_line == "RESULT: FAIL"
    assert "only 3 trading days" in outcome.detail


def test_skip_needs_the_script_to_say_skip():
    outcome = parse_verify_output(
        "SKIP: feed degraded to mock — no real data to verify against", 2
    )
    assert outcome.status == STATUS_SKIP
    assert outcome.skipped is True


def test_a_bare_exit_code_two_is_error_not_skip():
    # Python exits 2 when the script file does not exist: that is a broken
    # schedule, not "not attempted".
    outcome = parse_verify_output(
        "python.exe: can't open file 'scripts/nope.py': [Errno 2] No such file", 2
    )
    assert outcome.status == STATUS_ERROR
    assert outcome.exit_code == 4


def test_terminal_unavailable_is_blocked_not_failed():
    outcome = parse_verify_output(
        "FAIL: cannot connect to the MT5 terminal: (-10005,)\n", 1
    )
    assert outcome.status == STATUS_BLOCKED
    assert outcome.exit_code == 3
    assert outcome.passed is False


def test_unparseable_output_is_error():
    outcome = parse_verify_output("boom", 1)
    assert outcome.status == STATUS_ERROR
    assert outcome.exit_code == 4


@pytest.mark.parametrize(
    "output",
    [
        PASS_OUTPUT.replace("sent_calls=0 sent_writes=0", "sent_calls=1 sent_writes=0"),
        PASS_OUTPUT.replace("sent_calls=0 sent_writes=0", "sent_calls=0 sent_writes=1"),
        PASS_OUTPUT.replace("0 with magic 314159", "1 with magic 314159"),
    ],
)
def test_any_order_evidence_overrides_a_printed_pass(output: str):
    outcome = parse_verify_output(output, 0)
    assert outcome.status == STATUS_FAIL
    assert "ORDER EVIDENCE" in outcome.detail
    assert any(v > 0 for v in outcome.evidence.values())
    assert outcome.passed is False


def test_magic_count_reads_the_count_not_the_magic_number():
    outcome = parse_verify_output(PASS_OUTPUT, 0)
    assert outcome.evidence["magic_deals"] == 0
    assert outcome.evidence["sent_calls"] == 0
    assert outcome.evidence["sent_writes"] == 0


def test_check_digest_summarises_every_numbered_check():
    outcome = parse_verify_output(PASS_OUTPUT, 0)
    assert outcome.checks == "1111"  # checks 1..4, all True


def test_evidence_str_is_log_ready():
    outcome = parse_verify_output(PASS_OUTPUT, 0)
    assert outcome.evidence_str() == (
        "magic_deals=0,sent_calls=0,sent_writes=0,checks=1111"
    )


# ── log line ──────────────────────────────────────────────────────────────


def test_log_line_carries_the_verbatim_result_line_and_evidence():
    line = format_log_line(
        status=STATUS_PASS, returncode=0, duration_s=71.42, run_index=3,
        streak=3, previous_status=STATUS_PASS, result_line="RESULT: PASS",
        evidence="sent_calls=0,sent_writes=0,magic_deals=0,checks=1111",
    )
    assert 'status=PASS rc=0 dur=71.4s run=3 streak=+3 prev=PASS' in line
    assert 'result="RESULT: PASS"' in line
    assert "ev=sent_calls=0,sent_writes=0,magic_deals=0,checks=1111" in line
    assert "\n" not in line


def test_log_line_marks_a_regression_and_a_recovery():
    after_pass = format_log_line(
        status=STATUS_FAIL, returncode=1, duration_s=5.0, run_index=2, streak=-1,
        previous_status=STATUS_PASS, result_line="RESULT: FAIL", detail="x",
    )
    assert after_pass.startswith("!! REGRESSION ")
    recovered = format_log_line(
        status=STATUS_PASS, returncode=0, duration_s=5.0, run_index=3, streak=1,
        previous_status=STATUS_FAIL, result_line="RESULT: PASS",
    )
    assert recovered.startswith("** RECOVERED ")


def test_log_line_detail_is_flattened_and_bounded():
    line = format_log_line(
        status=STATUS_FAIL, returncode=1, duration_s=1.0, run_index=1, streak=-1,
        previous_status=None, result_line="RESULT: FAIL",
        detail="line one\nline two\t" + "x" * 500,
    )
    assert "line one line two" in line
    assert "\n" not in line and "\t" not in line
    assert len(line) < 500


def test_pass_line_omits_the_detail_field():
    line = format_log_line(
        status=STATUS_PASS, returncode=0, duration_s=1.0, run_index=1, streak=1,
        previous_status=None, result_line="RESULT: PASS", detail="noise",
    )
    assert "detail=" not in line


def test_exit_codes_are_distinct_per_status():
    assert EXIT_CODES == {
        STATUS_PASS: 0, STATUS_FAIL: 1, STATUS_SKIP: 2, STATUS_BLOCKED: 3,
        STATUS_ERROR: 4,
    }


# ── end-to-end runs against real subprocesses ─────────────────────────────


def _read_lines(log_path: Path):
    return [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_run_logs_the_result_line_and_writes_a_sidecar(tmp_path):
    script = _stub(tmp_path, "verify", f'print({PASS_OUTPUT!r})\n')
    outcome = run_daily_check(script, log_dir=tmp_path / "logs", python_exe=None)

    assert outcome.status == STATUS_PASS
    lines = _read_lines(tmp_path / "logs" / "verify.log")
    assert len(lines) == 1
    assert 'status=PASS' in lines[0] and 'result="RESULT: PASS"' in lines[0]
    assert "ev=magic_deals=0,sent_calls=0,sent_writes=0,checks=1111" in lines[0]

    sidecar = json.loads((tmp_path / "logs" / "verify.json").read_text(encoding="utf-8"))
    assert sidecar["last"]["status"] == STATUS_PASS
    assert sidecar["last"]["result_line"] == "RESULT: PASS"
    assert sidecar["runs"] == 1


def test_run_records_a_regression_then_a_recovery(tmp_path):
    script = _stub(tmp_path, "verify", f'print({PASS_OUTPUT!r})\n')
    log_dir = tmp_path / "logs"

    assert run_daily_check(script, log_dir=log_dir).status == STATUS_PASS
    script.write_text(
        "print('FAIL: no bars')\nprint('RESULT: FAIL')\nimport sys; sys.exit(1)\n",
        encoding="utf-8",
    )
    assert run_daily_check(script, log_dir=log_dir).status == STATUS_FAIL
    script.write_text(f'print({PASS_OUTPUT!r})\n', encoding="utf-8")
    assert run_daily_check(script, log_dir=log_dir).status == STATUS_PASS

    lines = _read_lines(log_dir / "verify.log")
    assert len(lines) == 3
    assert "!! REGRESSION" in lines[1] and "streak=-1" in lines[1]
    assert "** RECOVERED" in lines[2] and "streak=+1" in lines[2]
    assert "run=3" in lines[2]


def test_run_exit_code_matches_the_status(tmp_path):
    failing = _stub(tmp_path, "verify", "print('RESULT: FAIL')\nimport sys; sys.exit(1)\n")
    outcome = run_daily_check(failing, log_dir=tmp_path / "logs")
    assert outcome.status == STATUS_FAIL
    assert outcome.exit_code == 1

    noisy = _stub(tmp_path, "verify", "print('nothing useful here')\nimport sys; sys.exit(3)\n")
    assert run_daily_check(noisy, log_dir=tmp_path / "logs").exit_code == 4


def test_run_does_not_care_about_the_working_directory(tmp_path):
    # Task Scheduler sets no working directory and the path has spaces.
    spaced = tmp_path / "Forex trading bot" / "scripts"
    spaced.mkdir(parents=True)
    script = spaced / "verify.py"
    script.write_text(f"print({PASS_OUTPUT!r})\n", encoding="utf-8")
    outcome = run_daily_check(script, log_dir=tmp_path / "logs", root=spaced.parent)
    assert outcome.status == STATUS_PASS
    assert (tmp_path / "logs" / "verify.log").is_file()


def test_run_times_out_as_error_not_green(tmp_path):
    sleepy = _stub(tmp_path, "verify", "import time; time.sleep(30)\nprint('RESULT: PASS')\n")
    outcome = run_daily_check(sleepy, log_dir=tmp_path / "logs", timeout=1.0)
    assert outcome.status == STATUS_ERROR
    assert "timeout" in outcome.detail
    assert "result=" not in _read_lines(tmp_path / "logs" / "verify.log")[0]


def test_run_of_a_missing_script_is_error_and_still_logged(tmp_path):
    outcome = run_daily_check(tmp_path / "nope.py", log_dir=tmp_path / "logs")
    assert outcome.status == STATUS_ERROR
    assert "nope.py" in outcome.detail or "can't open" in outcome.detail
    assert "status=ERROR" in _read_lines(tmp_path / "logs" / "nope.log")[0]


def test_second_run_while_the_first_holds_the_lock_is_skipped(tmp_path):
    script = _stub(tmp_path, "verify", "print('RESULT: PASS')\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    lock = log_dir / "verify.lock"
    lock.write_text("4242 in-flight\n", encoding="utf-8")

    outcome = run_daily_check(script, log_dir=log_dir)
    assert outcome.status == STATUS_SKIP
    assert outcome.skipped is True
    assert "another check is running" in outcome.detail

    lines = _read_lines(log_dir / "verify.log")
    assert len(lines) == 1 and "status=SKIP" in lines[0]
    assert "result=" not in lines[0]


def test_a_stale_lock_is_reclaimed(tmp_path):
    script = _stub(tmp_path, "verify", "print('RESULT: PASS')\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    lock = log_dir / "verify.lock"
    lock.write_text("1 dead\n", encoding="utf-8")
    stale = time.time() - 7200
    import os

    os.utime(lock, (stale, stale))

    assert run_daily_check(script, log_dir=log_dir).status == STATUS_PASS


def test_the_lock_is_released_after_a_run(tmp_path):
    script = _stub(tmp_path, "verify", "print('RESULT: PASS')\n")
    log_dir = tmp_path / "logs"
    run_daily_check(script, log_dir=log_dir)
    assert not (log_dir / "verify.lock").exists()


def test_log_retention_keeps_the_newest_lines(tmp_path):
    script = _stub(tmp_path, "verify", "print('RESULT: PASS')\n")
    log_dir = tmp_path / "logs"
    for _ in range(5):
        run_daily_check(script, log_dir=log_dir, keep_lines=3)

    lines = _read_lines(log_dir / "verify.log")
    assert len(lines) == 3
    assert "run=5" in lines[-1]  # run numbering survives trimming


def test_the_log_directory_is_created_on_demand(tmp_path):
    script = _stub(tmp_path, "verify", "print('RESULT: PASS')\n")
    nested = tmp_path / "deep" / "daily_verify"
    assert not nested.exists()
    run_daily_check(script, log_dir=nested)
    assert (nested / "verify.log").is_file()

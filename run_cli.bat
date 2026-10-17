@echo off
rem Launch the Alex CLI on the Python 3.11 venv (.venv311).
setlocal
set ROOT=%~dp0
if exist "%ROOT%.venv311\Scripts\python.exe" (
    "%ROOT%.venv311\Scripts\python.exe" -m forex_ai_agent.cli %*
) else (
    echo [run_cli] .venv311 not found - falling back to PATH python
    python -m forex_ai_agent.cli %*
)
endlocal

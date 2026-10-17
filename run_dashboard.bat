@echo off
rem Launch the Alex dashboard on the Python 3.11 venv (.venv311).
rem The old .venv (3.10) lacks yfinance, so its feed chain degrades to mock-only.
setlocal
set ROOT=%~dp0
if exist "%ROOT%.venv311\Scripts\python.exe" (
    "%ROOT%.venv311\Scripts\python.exe" -m streamlit run "%ROOT%dashboard\app.py" %*
) else (
    echo [run_dashboard] .venv311 not found - falling back to PATH streamlit
    streamlit run "%ROOT%dashboard\app.py" %*
)
endlocal

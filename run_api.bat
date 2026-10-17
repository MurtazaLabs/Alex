@echo off
rem Launch the Alex REST API (serves alex-ui + JSON API) on the Python 3.11 venv.
setlocal
set ROOT=%~dp0

rem Fail visibly if port 8000 is already taken (another Alex instance running?)
netstat -ano | findstr :8000 | findstr LISTENING
if not errorlevel 1 (
    echo [run_api] Port 8000 is ALREADY IN USE - Alex may already be running.
    echo [run_api] Try opening http://localhost:8000 in your browser first.
    echo [run_api] To stop the other instance, close its window or run:
    echo [run_api]     taskkill /PID ^<pid-from-above^> /F
    pause
    exit /b 1
)

if exist "%ROOT%.venv311\Scripts\python.exe" (
    "%ROOT%.venv311\Scripts\python.exe" -m uvicorn forex_ai_agent.interfaces.api.app:app --host 0.0.0.0 --port 8000 %*
) else (
    echo [run_api] .venv311 not found - falling back to PATH python
    python -m uvicorn forex_ai_agent.interfaces.api.app:app --host 0.0.0.0 --port 8000 %*
)

rem Keep the window open if the server crashed, so the error can be read
if errorlevel 1 (
    echo.
    echo [run_api] The server exited with an error. See message above.
    pause
)
endlocal

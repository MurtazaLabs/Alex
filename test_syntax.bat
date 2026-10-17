@echo off
cd /d "E:\Forex trading bot"
"E:\Forex trading bot\.venv311\Scripts\python.exe" -c "import os; os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///:memory:'; os.environ['ENVIRONMENT'] = 'development'; import sys; sys.path.insert(0, r'E:\Forex trading bot'); import dashboard.app; print('Import OK - Syntax check passed')"
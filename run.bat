@echo off
rem Start PostgreSQL + the web experiment platform. Paths come from .env in the project root.
cd /d "%~dp0"

rem Load KEY=VALUE lines from .env (lines starting with # are skipped)
if exist ".env" for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do set "%%a=%%b"

rem Start PostgreSQL if PGBIN is set and it is not already running
if not defined PGBIN (
    echo PGBIN is not set in .env, skip starting PostgreSQL
) else (
    "%PGBIN%\pg_ctl.exe" -D "%PGDATA%" status >nul 2>&1
    if errorlevel 1 (
        "%PGBIN%\pg_ctl.exe" -D "%PGDATA%" -l "%PGDATA%\log\startup.log" -w start
    ) else (
        echo PostgreSQL is already running
    )
)

if not defined PYTHON_EXE set "PYTHON_EXE=python"
"%PYTHON_EXE%" app.py
pause

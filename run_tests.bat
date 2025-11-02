@echo off
REM Run tests for Snail Trails GPU simulation

echo =========================================
echo   Snail Trails - Test Suite
echo =========================================
echo.

REM Check if pytest is installed
python -c "import pytest" 2>nul
if errorlevel 1 (
    echo Installing test dependencies...
    pip install pytest pytest-cov -q
)

echo Running tests...
echo.

REM Run tests
python -m pytest tests/ -v --tb=short

echo.
echo =========================================
echo Test suite complete!
echo =========================================

pause

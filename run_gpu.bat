@echo off
echo ========================================
echo  Snail Trails GPU - 10M Agents Edition
echo  Optimized for NVIDIA RTX 4090
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Python detected!
echo.

REM Check if dependencies are installed
echo Checking dependencies...
pip show moderngl >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Dependencies OK!
echo.
echo Starting simulation...
echo Press Ctrl+C or close window to stop
echo.

REM Run the simulation
python snail_trails_gpu.py

pause

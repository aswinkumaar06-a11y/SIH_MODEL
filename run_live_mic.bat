@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo =====================================================================
echo    SIH 26172: EDGE VOICE ACTIVATOR - LIVE LAPTOP MICROPHONE
echo    Hardware Target: ESP32-S3 (<100KB Flash, <256KB SRAM)
echo =====================================================================
echo.
echo [1] Immediate Start with Pre-Enrolled Keyword 'ZORA'
echo [2] Record Your OWN Voice / Custom Keyword via Laptop Mic (3 Shots)
echo [3] Custom Keyword ^& Threshold
echo.
set /p choice="Enter option (1, 2, or 3) [Default: 1]: "

if "%choice%"=="" set choice=1
if "%choice%"=="1" (
    echo.
    echo Starting live microphone listener for 'ZORA'...
    echo Speak into your laptop microphone and say 'ZORA'!
    echo Press Ctrl+C to exit.
    echo.
    .venv\Scripts\python.exe scripts\live_mic_activator.py --keyword ZORA
) else if "%choice%"=="2" (
    echo.
    set /p kw="Enter keyword to enroll (e.g., HELIOS, JARVIS, your name) [Default: ZORA]: "
    if "!kw!"=="" set kw=ZORA
    echo.
    echo Starting interactive 3-shot microphone enrollment for '!kw!'...
    .venv\Scripts\python.exe scripts\live_mic_activator.py --keyword !kw! --record-user
) else if "%choice%"=="3" (
    echo.
    set /p kw="Enter keyword [Default: ZORA]: "
    if "!kw!"=="" set kw=ZORA
    set /p th="Enter sensitivity threshold (0.80 - 0.95) [Default: 0.87]: "
    if "!th!"=="" set th=0.87
    echo.
    echo Starting with '!kw!' (threshold: !th!)...
    .venv\Scripts\python.exe scripts\live_mic_activator.py --keyword !kw! --threshold !th!
)

pause

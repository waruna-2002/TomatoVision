@echo off
title TomatoVision AI - Master Production Suite
color 0A

echo ======================================================================
echo          ?? TOMATOVISION AI: AGRO-VISION QUALITY SUITE ??
echo ======================================================================
echo.
echo [*] Starting Python AI Backend Server on Port 8000...
start /min "TomatoVision AI Server" cmd /c "cd /d "%~dp0tomatovision-ml" && .venv\Scripts\python.exe api_server.py"

timeout /t 3 /nobreak > nul

echo [*] Starting TomatoVision App on Chrome...
start /min "TomatoVision App" cmd /c "cd /d "%~dp0tomatovision_app" && flutter run -d chrome --web-port 50005"

echo.
echo ======================================================================
echo  [+] AI Backend Server:  http://localhost:8000
echo  [+] TomatoVision App:   http://localhost:50005
echo ======================================================================
echo  System is fully active and ready for Demonstration / Presentation!
echo ======================================================================
echo.

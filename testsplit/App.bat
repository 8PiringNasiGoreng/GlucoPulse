@echo off
echo ================================================
echo   Diabetes AI Prediction System
echo   Memulai aplikasi, harap tunggu...
echo ================================================
echo.
 
cd /d "%~dp0"
 
echo Menjalankan aplikasi...
start "" http://localhost:5000
py app.py
 
pause

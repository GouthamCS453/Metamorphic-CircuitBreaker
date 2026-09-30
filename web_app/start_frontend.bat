@echo off
REM start_frontend.bat — Windows launcher for React dev server
cd /d "%~dp0\frontend"
echo Starting Aegis MetroHealth React frontend on http://localhost:5173 ...
npm.cmd run dev

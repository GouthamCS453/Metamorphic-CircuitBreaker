@echo off
REM start_backend.bat — Windows launcher for FastAPI server
REM Run from the main_implementation\ root directory

cd /d "%~dp0\.."
echo Starting Aegis MetroHealth FastAPI backend...
echo Loading ConvNeXt-Base model (~1 GB — first start takes ~30s)...
.venv\Scripts\python.exe web_app\run_server.py

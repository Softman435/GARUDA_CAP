@echo off
REM Starts GARUDA CAPTIONS AI (backend + web interface on one address) and opens your browser.
setlocal
cd /d "%~dp0\..\backend"
if not exist .venv\Scripts\activate.bat (echo Run scripts\install.bat first. & exit /b 1)
if not exist ..\frontend\dist\index.html (echo The web interface is not built. Run scripts\install.bat first. & exit /b 1)
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
where ffmpeg >nul 2>&1 || echo WARNING: FFmpeg not found on PATH - videos cannot be read. See docs\WINDOWS_SETUP.md step 3.
echo.
echo  GARUDA CAPTIONS AI is starting.
echo  Open:  http://127.0.0.1:8000     (your browser opens automatically)
echo  Log:   data\logs\garuda.log      (full technical details if something goes wrong)
echo  Keep this window open while you use the app. Close it (or press Ctrl+C) to stop.
echo.
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
endlocal

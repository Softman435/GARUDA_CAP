@echo off
REM Measures caption accuracy on your own audio + correct text and compares models.
REM   scripts\evaluate.bat --data "C:\my-dataset" --models base small kn-medium --language kn --limit 20
setlocal
cd /d "%~dp0\..\backend"
if not exist .venv\Scripts\activate.bat (echo Run scripts\install.bat first. & exit /b 1)
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
python -m app.evaluate %*
endlocal & exit /b %ERRORLEVEL%

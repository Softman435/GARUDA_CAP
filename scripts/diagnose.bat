@echo off
REM Checks Python, packages, FFmpeg, settings and the speech model. Nothing is changed.
REM   scripts\diagnose.bat                      quick check (downloads nothing)
REM   scripts\diagnose.bat --deep               also loads the model and runs real speech recognition
REM   scripts\diagnose.bat --deep --model base --audio "C:\path\my video.mp4" --language kn-en
REM A full Python traceback is printed for any step that fails - send that text when asking for help.
setlocal
cd /d "%~dp0\..\backend"
if not exist .venv\Scripts\activate.bat (echo Run scripts\install.bat first. & exit /b 1)
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
python -m app.diagnostics %*
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" (echo Diagnostics finished: no problems found.) else (echo Diagnostics finished: problems were found - read the "Problems found" list above.)
endlocal & exit /b %CODE%

@echo off
REM One-time setup of the IndicConformer engine inside WSL2 (Ubuntu 22.04) on Windows 11.
REM Before this: put the .nemo file into  data\models\indic\   (see docs\INDIC_CONFORMER.md)
setlocal
cd /d "%~dp0\.."
set "DISTRO=Ubuntu-22.04"
where wsl >nul 2>&1 || (
  echo WSL is not installed. Open PowerShell as Administrator, run:   wsl --install -d %DISTRO%
  echo Restart the computer when asked, open "Ubuntu 22.04" once to create a username and password, then run this script again.
  exit /b 1
)
wsl -d %DISTRO% -- echo ready >nul 2>&1 || (
  echo The Linux system "%DISTRO%" is not installed yet. Open PowerShell as Administrator, run:   wsl --install -d %DISTRO%
  echo Restart if asked, open "Ubuntu 22.04" once to create a username and password, then run this script again.
  exit /b 1
)
set "SH="
for /f "delims=" %%i in ('wsl -d %DISTRO% -- wslpath -a "%~dp0setup_indic.sh"') do set "SH=%%i"
if not defined SH (echo Could not find setup_indic.sh from inside Ubuntu. & exit /b 1)
echo Installing inside Ubuntu. This takes 20-60 minutes and shows a lot of text. Do not close this window.
wsl -d %DISTRO% -- bash "%SH%"
if errorlevel 1 (
  echo.
  echo FAILED. Scroll up to the first line that says ERROR and send me the 20 lines around it.
  exit /b 1
)
echo.
echo Next:   scripts\diagnose.bat --indic     (must say READY)
endlocal

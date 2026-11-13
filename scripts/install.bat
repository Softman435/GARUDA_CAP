@echo off
REM GARUDA CAPTIONS AI - one-time install for Windows 10/11.  Double-click, or run from a terminal.
REM Safe to run again at any time: it re-uses the existing environment and only changes files inside this folder.
setlocal
cd /d "%~dp0\.."
set "FAILED="
echo =====================================================
echo  GARUDA CAPTIONS AI - install
echo =====================================================

echo.
echo [1/6] Looking for Python 3.10 - 3.13 (3.12 is best)...
set "PY="
for %%V in (3.12 3.11 3.10 3.13) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo ERROR: No supported Python was found. Install Python 3.12 from https://www.python.org/downloads/windows/
  echo        and tick "Add python.exe to PATH" on the first screen. See docs\WINDOWS_SETUP.md, step 1.
  exit /b 1
)
%PY% --version
%PY% -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,13) else 1)" >nul 2>&1 && echo NOTE: Python 3.13 works but is less tested than 3.12. If "pip install" fails below, install Python 3.12 and run this script again.

echo.
echo [2/6] Creating or re-using the project environment in backend\.venv ...
if not exist backend\.venv\Scripts\python.exe (
  %PY% -m venv backend\.venv || (echo ERROR: could not create the environment. & exit /b 1)
)
call backend\.venv\Scripts\activate.bat || (echo ERROR: could not activate the environment. & exit /b 1)

echo.
echo [3/6] Installing Python packages (this can take 5-10 minutes; faster-whisper is large)...
python -m pip install --upgrade pip || (echo ERROR: could not update pip. Check your internet connection. & exit /b 1)
python -m pip install -r backend\requirements.txt || (echo ERROR: package installation failed. Read the message above. & exit /b 1)

echo.
echo [4/6] Checking FFmpeg and FFprobe...
where ffmpeg >nul 2>&1 || set "FAILED=1"
where ffprobe >nul 2>&1 || set "FAILED=1"
if defined FAILED (
  echo WARNING: FFmpeg / FFprobe were NOT found on PATH. The app cannot read videos without them.
  echo          Install with:   winget install Gyan.FFmpeg
  echo          then close this window, open a NEW one and check:   ffmpeg -version   and   ffprobe -version
  echo          Details: docs\WINDOWS_SETUP.md, step 3.
) else (
  echo FFmpeg and FFprobe found.
)

echo.
echo [5/6] Building the web interface (needs Node.js)...
if exist frontend\dist\index.html (
  echo Already built. Delete the folder frontend\dist to rebuild it.
) else (
  where node >nul 2>&1 || (echo ERROR: Node.js is not installed. Install the LTS version from https://nodejs.org - see docs\WINDOWS_SETUP.md step 2. & exit /b 1)
  pushd frontend
  call npm install || (popd & echo ERROR: npm install failed. & exit /b 1)
  call npm run build || (popd & echo ERROR: the web interface build failed. & exit /b 1)
  popd
)

echo.
echo [6/6] Settings and fonts...
if not exist .env copy .env.example .env >nul
if exist "%WINDIR%\Fonts\Nirmala.ttc" copy /Y "%WINDIR%\Fonts\Nirmala.ttc" backend\fonts\ >nul
if exist "%WINDIR%\Fonts\Nirmala.ttf"  copy /Y "%WINDIR%\Fonts\Nirmala.ttf"  backend\fonts\ >nul
if exist "%WINDIR%\Fonts\NirmalaB.ttf" copy /Y "%WINDIR%\Fonts\NirmalaB.ttf" backend\fonts\ >nul

echo.
echo =====================================================
echo  Install finished.
echo.
echo  SPEECH MODEL: it is NOT in this ZIP. It is downloaded automatically the first time you click
echo  "Generate captions" (internet needed once). Approximate size: tiny 75 MB, base 145 MB, small 480 MB.
echo  It is stored in the folder  data\models  inside this project, so you can see it and delete it.
echo.
echo  Next:   scripts\diagnose.bat      (checks everything, does not download anything)
echo          scripts\start.bat         (starts the app)
echo =====================================================
endlocal

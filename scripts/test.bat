@echo off
REM Runs the automated tests.
REM   scripts\test.bat          normal tests (no model download)
REM   scripts\test.bat e2e      ALSO runs the real speech-recognition test (downloads the tiny model once, ~75 MB)
REM   For real speech:  set GARUDA_E2E_AUDIO=C:\path\speech.wav   and optionally  set GARUDA_E2E_LANG=kn-en
setlocal
cd /d "%~dp0\.."
if not exist backend\.venv\Scripts\activate.bat (echo Run scripts\install.bat first. & exit /b 1)
call backend\.venv\Scripts\activate.bat
set PYTHONUTF8=1
python -m pip install -r backend\requirements-dev.txt || exit /b 1
if /i "%1"=="e2e" set GARUDA_E2E=1
echo === 1. Media, captions, export (real FFmpeg) ===
python -m unittest discover -s backend\tests -p "test_core.py" -v || exit /b 1
python -m unittest discover -s backend\tests -p "test_fonts.py" -v || exit /b 1
echo === 2. Speech-recognition logic (mocked library; plus REAL library when "e2e") ===
python -m unittest discover -s backend\tests -p "test_transcribe.py" -v || exit /b 1
python -m unittest discover -s backend\tests -p "test_kannada_models.py" -v || exit /b 1
python -m unittest discover -s backend\tests -p "test_indic.py" -v || exit /b 1
python -m unittest discover -s backend\tests -p "test_evaluate.py" -v || exit /b 1
python -m unittest discover -s backend\tests -p "test_convert_model.py" -v || exit /b 1
python -m unittest discover -s backend\tests -p "test_training_kit.py" -v || exit /b 1
echo === 3. API tests (pytest) ===
python -m pytest backend\tests\test_api.py -v || exit /b 1
echo === 4. Web interface tests ===
if exist frontend\node_modules (
  pushd frontend
  call npm test
  popd
) else (
  echo Skipped: frontend\node_modules not found. Run scripts\install.bat first.
)
endlocal

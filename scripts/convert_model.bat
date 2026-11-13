@echo off
REM One-time conversion of a Hugging Face Whisper model to the faster-whisper format (needs internet).
REM   scripts\convert_model.bat vasista22/whisper-kannada-small kn-small-ct2 small
REM   scripts\convert_model.bat my-account/my-kannada-whisper my-model-ct2 small
REM SOURCE can also be a folder on this computer. PyTorch is installed in a SEPARATE environment (backend\.venv-convert,
REM about 1 GB) so the app itself stays light. Delete that folder afterwards if you like.
setlocal
cd /d "%~dp0\.."
if "%~2"=="" (echo Usage: scripts\convert_model.bat SOURCE NAME [SIZE] & echo Example: scripts\convert_model.bat vasista22/whisper-kannada-small kn-small-ct2 small & exit /b 1)
if not exist backend\.venv\Scripts\python.exe (echo Run scripts\install.bat first. & exit /b 1)
if not exist backend\.venv-convert\Scripts\python.exe (
  echo Creating the conversion environment ...
  backend\.venv\Scripts\python.exe -m venv backend\.venv-convert || exit /b 1
  backend\.venv-convert\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
  backend\.venv-convert\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu || (echo ERROR: PyTorch could not be installed. If you use Python 3.13, install Python 3.12 and run scripts\install.bat again. & exit /b 1)
  backend\.venv-convert\Scripts\python.exe -m pip install "transformers>=4.36" "ctranslate2>=4.5,<5" huggingface_hub sentencepiece || exit /b 1
)
set PYTHONUTF8=1
backend\.venv-convert\Scripts\python.exe backend\tools\convert_model.py %*
endlocal & exit /b %ERRORLEVEL%

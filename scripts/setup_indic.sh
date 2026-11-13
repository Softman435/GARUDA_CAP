#!/usr/bin/env bash
# Runs INSIDE WSL2 (Ubuntu 22.04). Started for you by scripts\setup_indic.bat.
# Installs what AI4Bharat documents for IndicConformer: PyTorch + their fork of NeMo (branch nemo-v2).
# It takes 20-60 minutes and downloads several GB. Everything stays in your Ubuntu home folder.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
VENV="$HOME/indic-venv"
NEMO_DIR="$HOME/ai4bharat-NeMo"

echo "== 1/5  system packages (Ubuntu may ask for the password you chose for Ubuntu) =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-pip build-essential git ffmpeg libsndfile1 sox

echo "== 2/5  Python environment in $VENV =="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "== 3/5  PyTorch (CPU version) =="
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "== 4/5  AI4Bharat NeMo fork (the longest step) =="
pip install packaging "huggingface_hub==0.23.2"
if [ ! -d "$NEMO_DIR/.git" ]; then
  git clone https://github.com/AI4Bharat/NeMo.git "$NEMO_DIR"
fi
cd "$NEMO_DIR"
git checkout nemo-v2
bash reinstall.sh

echo "== 5/5  telling the app how to start this environment =="
mkdir -p "$PROJECT/data"
echo "wsl -d ${WSL_DISTRO_NAME:-Ubuntu-22.04} -- $VENV/bin/python" > "$PROJECT/data/indic_runner.txt"
echo "Saved: $PROJECT/data/indic_runner.txt"
"$VENV/bin/python" "$PROJECT/backend/tools/indic_worker.py" --check || true
echo
echo "Finished. Back in Windows run:   scripts\\diagnose.bat --indic"

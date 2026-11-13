#!/usr/bin/env bash
# macOS / Linux equivalent of start.bat (after: python3 -m venv backend/.venv && pip install -r backend/requirements.txt)
cd "$(dirname "$0")/../backend" && . .venv/bin/activate && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

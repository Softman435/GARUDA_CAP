@echo off
REM Developer mode: Python server with auto-reload + Vite dev server. Open http://localhost:5173
cd /d "%~dp0\.."
start "GARUDA backend" cmd /k "cd backend && call .venv\Scripts\activate.bat && python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
start "GARUDA frontend" cmd /k "cd frontend && npm run dev"

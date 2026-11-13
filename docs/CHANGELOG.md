# Version 4 - IndicConformer (.nemo)
New engine `indic` (`app/indic.py` + `tools/indic_worker.py`): pause-based chunking, worker bridge for AI4Bharat's NeMo fork via WSL2, progress/cancel/timeout, honest phrase/word timing; `scripts/setup_indic.bat|.sh`; `diagnose.bat --indic`; `docs/INDIC_CONFORMER.md`; 29 tests (stand-in for NeMo).

# Version 3 - better Kannada
New model choices `kn-medium` (ready, Kannada-tuned), `kn-small` (after conversion), `custom` (your fine-tune); `scripts\\evaluate.bat` accuracy tool; `scripts\\convert_model.bat`; `training/` Colab kit; optional `data\\vocabulary.txt`; Kannada warnings for tiny/base; 28 new tests.

# Fix 2 - PyAV incompatibility (found on a real PC)
`TypeError: open() got an unexpected keyword argument 'metadata_errors'` = faster-whisper 1.2.1 + PyAV 19.0.1. `transcribe.py` now passes the audio as a numpy array (new `load_audio_array`), so PyAV is never used; `diagnostics.py` does the same; `requirements.txt` adds numpy; 4 regression tests added.

# Changes in this repair (speech recognition, Kannada + English, Windows scripts)

| File | Change | Why |
|---|---|---|
| backend/app/transcribe.py | Rewritten: staged errors, full traceback logging, argument check against the installed library, segment validation, phrase-timing fallback, no empty "success", model cache + idle release, mixed Kannada+English mode (`kn-en`), CPU/compute-type checks | Original swallowed the traceback behind one vague message |
| backend/app/logging_setup.py (new) | Rotating log file `data/logs/garuda.log`, secrets masked | Diagnostics must survive |
| backend/app/diagnostics.py (new) | Health report + deep self-test (`python -m app.diagnostics`, `/api/diagnostics`) | Find the exact failing step |
| backend/app/media.py | Audio level normalisation with safe retry, real FFmpeg errors logged, `wav_duration` | Better recognition input, diagnosable failures |
| backend/app/config.py | CPU-first defaults (base, int8, beam 3), models in `data/models`, new settings | 16 GB CPU laptop |
| backend/app/jobs.py | Uses the new API, logs real causes, always cleans the work folder | Truthful errors |
| backend/app/main.py, schemas.py | Logging setup, `/api/diagnostics`, default model `base` | |
| backend/requirements.txt | Pinned compatible ranges for faster-whisper and ctranslate2 | Unpinned "latest" can change behaviour |
| backend/tests/test_transcribe.py, fake_faster_whisper.py (new) | 39 tests (37 run, 2 real-library skipped unless `GARUDA_E2E=1`) | Test the real function, not only a mocked frontend |
| backend/tests/test_api.py | 5 new API tests (not runnable in the repair sandbox) | |
| scripts/install.bat (new), setup.bat, start.bat, diagnose.bat (new), test.bat | Python 3.10-3.13 detection, FFmpeg check, model-location notes | Reliable Windows use |
| frontend/src/components/Editor.tsx | Hint for mixed-language mode | |
| .env.example, README.md, docs/WINDOWS_SETUP.md, docs/FONTS.md | Updated to match | |

# 🦅 GARUDA CAPTIONS AI

Upload a video of your own voice. The app listens to the real audio **on your computer**, writes word-by-word
(or phrase-by-phrase) captions with timestamps, lets you correct them, and exports a **new** MP4 with the captions
burned in, plus SRT and VTT files. Your original video is never changed. No paid API, no account, nothing is uploaded.

Built for **Kannada, English and mixed Kannada + English** speech on an ordinary laptop **CPU** (no NVIDIA card needed).

## System requirements
* Windows 10/11 64-bit, 8 GB RAM or more (16 GB is comfortable), about 2 GB free disk for the app and a small model
* Python 3.12 recommended (3.10 - 3.13 supported), Node.js LTS (once, to build the page), FFmpeg
* Internet once for installing and once per model download. Afterwards everything works offline.

## Install and start (Windows)
1. Install Python, Node.js and FFmpeg: follow `docs\WINDOWS_SETUP.md` (about 15 minutes).
2. Double-click **`scripts\install.bat`** (once).
3. Double-click **`scripts\diagnose.bat`** to confirm there are no problems.
4. Double-click **`scripts\start.bat`**. Open http://127.0.0.1:8000.

| Script | What it does |
|---|---|
| `scripts\install.bat` | creates `backend\.venv`, installs pinned packages, checks FFmpeg, builds the web page (`setup.bat` just calls it) |
| `scripts\start.bat` | starts the app and opens the browser |
| `scripts\diagnose.bat` | health report; add `--deep` to run real speech recognition and show the full error trace of any failing step |
| `scripts\test.bat` | automated tests; `scripts\test.bat e2e` also tests the real speech library |
| `scripts\setup_indic.bat` | one-time WSL2 setup for the IndicConformer `.nemo` model (`diagnose.bat --indic` checks it) |
| `scripts\evaluate.bat` | measures caption accuracy (WER/CER) on your own audio + correct text and compares models |
| `scripts\convert_model.bat` | one-time conversion of a Hugging Face Whisper model (for example the small Kannada fine-tune) |
| `scripts\dev.bat` | developer mode with auto-reload |

## How to use
1. **New Project**, drop a video. 2. **Transcript** tab: pick the language and model, **Generate captions from audio**.
3. Fix words/timings. 4. **Style** tab: word-by-word or phrase, font, size, colours, outline, shadow, position.
5. **Export** tab: video with burned-in captions, SRT, VTT.

### Languages
The default language is **Auto-detect** and the default model is **Auto (higher accuracy)**: it uses **large-v3** for
Auto-detect and all non-Kannada/Kanglish selections (including English, Telugu, and Kannada + English), and recommends
**kn-medium** for explicitly selected Kannada or Kanglish. These are existing pre-trained models; the app does not train a new model. `large-v3` needs about 10 GB RAM
and can be very slow on a CPU. Changing models can change the words and their spelling; no speech model guarantees
exact text, so proof-read or compare with a verified transcript.
The Kannada-tuned models are intended for Kannada; using them for other languages can produce inaccurate captions.
Use **large-v3** for mixed or non-Kannada speech.

| Choice | Meaning |
|---|---|
| **Kannada + English (mixed speech)** | Whisper detects the language for every ~30-second window and gets a mixed Kannada/English example prompt, so Kannada stays in Kannada script and English words stay in English. Nothing is translated. |
| **Kanglish (Kannada in Latin letters)** | Recognizes Kannada, then writes the Kannada-script output in uppercase Latin letters (for example, `ನನ್ನ ಹೆಸರು` → `NANNA HESARU`). This is transliteration, not translation or extra model training; review names and spelling. |
| Kannada | Forces Kannada. English words inside may come out re-spelled in Kannada letters. |
| English | Forces English. |
| Auto-detect | Whisper picks one language per window. |
| Hindi, Telugu, Tamil, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Urdu, Assamese, Sanskrit, Nepali | Supported by Whisper; quality varies. Odia is not supported. |

**Known limits (read these):** Whisper decides the language per ~30-second window, **not per word**, so a single English
word in the middle of a Kannada sentence can still be re-spelled. Kannada accuracy is lower than English and rises with
bigger models. Always proof-read: the captions are editable for that reason. The app never translates speech.
Kanglish changes the script after recognition; it cannot correct words the speech model misheard.
Word timing comes from faster-whisper's own alignment; where it fails the caption keeps honest **phrase-level** timing
(marked "phrase") and is never presented as word-level. WhisperX (a stronger aligner) is not used because it needs
PyTorch, which is too heavy for this CPU-only design.
Voice-activity detection pads speech boundaries by one second to help retain quiet words at clip edges, and
symbol-only output segments and invalid Unicode replacement characters are discarded (with a warning for the latter).
Implausibly long word alignments fall back to phrase-level timing. These safeguards cannot recover speech that is
missing or unclear in the audio; review the transcript before export.

### Models (CPU laptop, 16 GB RAM)
| Model | RAM | Speed on a CPU | Notes |
|---|---|---|---|
| tiny | ~1 GB | fastest | weak for Kannada |
| base | ~1 GB | fast | lower accuracy than large-v3; weak for Kannada |
| small | ~2 GB | a few times slower than real time | general multilingual model |
| medium | ~5 GB | very slow | general multilingual model |
| **large-v3** | ~10 GB | very slow | highest-accuracy general multilingual model available |
| **kn-medium** | ~3 GB | slow (test a 1-minute clip first) | **Kannada-tuned** (IIT Madras fine-tune); downloads ~1.5 GB once. Best Kannada Whisper model available here |
| **kn-small** | ~1.5 GB | faster | Kannada-tuned; appears after a one-time `convert_model.bat` |
| **custom** | depends | depends | your own fine-tuned model via `CUSTOM_MODEL` in `.env` |
| **indic** | ~2 GB (in WSL) | fast for its size | AI4Bharat IndicConformer `.nemo`; shown only after its WSL runner is configured |

## IndicConformer (your `.nemo` file)
The app can also use AI4Bharat's **IndicConformer** model (the `.nemo` file) for Kannada and 12 other Indian languages.
It runs through a worker in WSL2 Ubuntu because it needs AI4Bharat's Linux-only NeMo fork. Setup is in
`docs\INDIC_CONFORMER.md` (copy the file to `data\models\indic`, run `scripts\setup_indic.bat`, check with
`scripts\diagnose.bat --indic`, then choose model **indic**). The file is **already trained**; nothing is trained here.
The bridge is tested with a stand-in for NeMo; the real model has NOT been run by me.

## Better Kannada (read `docs\KANNADA_ACCURACY.md`)
English is good because Whisper saw a lot of English. For Kannada: choose model **kn-medium**; measure with
`scripts\evaluate.bat`; add names to `data\vocabulary.txt`; and, if you have your own recordings + transcripts, train
a model with the Colab kit in `training\README.md` and use it as **custom**. Training needs a GPU (free Google Colab),
not your laptop. **No data repository was supplied when this version was built, so nothing was trained here.**

Settings that matter are in `.env`: `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE=int8`, `WHISPER_CPU_THREADS`,
`WHISPER_BEAM_SIZE` (1 = fastest). The model is loaded once, reused, and released after 5 idle minutes.
A GPU is never required; if `WHISPER_DEVICE=cuda` is set without a usable card the app falls back to the CPU.

### Supported media
MP4, MOV, WebM, MKV, AVI (checked by FFprobe on the real content, not by file name), up to 2 GB / 3 hours by default
(`MAX_UPLOAD_MB`, `MAX_DURATION_SECONDS`). The video must contain an audio track.

## What was wrong with "Speech recognition failed (TypeError)" and what changed
**Root cause (confirmed from a real run on a Windows 11 PC):** the log said
`TypeError: open() got an unexpected keyword argument 'metadata_errors'`. When it is given a file path, faster-whisper
reads the audio with the **PyAV** package (`av.open(path, metadata_errors="ignore")`). The unpinned install pulled in
PyAV 19.0.1, whose `open()` no longer accepts that argument, so every transcription failed at that step. The old
code then replaced the message with "Speech recognition failed (TypeError)" and threw the traceback away, so the cause
stayed hidden. (The model-size advice in the old message could never have helped.)
**The fix:** the app already converts your video to a clean 16 kHz mono WAV with FFmpeg, so it now hands faster-whisper
that audio **as an array**. faster-whisper then never calls PyAV, and the bug cannot happen whatever PyAV version is
installed. A regression test reproduces this exact error.
**What the new code does about it:**
* stages are separated, so every failure has its own specific message and code;
* the full original traceback, library versions and settings go to `data\logs\garuda.log`;
* arguments are checked against the installed library (`inspect.signature`), so "unexpected keyword argument" is impossible;
* every segment is validated, so a `None` / NaN / negative timestamp cannot raise a `TypeError`;
* if the library itself fails with word timing on, the app retries with phrase-level timing, then without the voice filter, and tells you in a warning what happened;
* a result with no valid caption is a **failure** ("No speech was detected"), never a success;
* `scripts\diagnose.bat --deep` runs each setting separately on your computer and prints the exact failing line.

## Troubleshooting
| Symptom | What to do |
|---|---|
| `TypeError: open() got an unexpected keyword argument 'metadata_errors'` | Fixed in this version (audio is passed as an array, so PyAV is not used). If you see it, you are running an old `backend\\app\\transcribe.py`. |
| `Speech recognition failed (TypeError: ...)` | Open `data\logs\garuda.log` (the traceback is there). Run `scripts\diagnose.bat --deep` and keep the output. Often fixed by `scripts\install.bat` again (installs the pinned versions). |
| Message says word timing failed, captions are "phrase" | The library's word aligner failed; captions still have real phrase timing. Details are in the log. |
| `model has to be downloaded ... download failed` | Connect to the internet once and retry. Models are saved in `data\models`. |
| `not on this computer and downloads are turned off` | Set `ALLOW_MODEL_DOWNLOAD=true` in `.env` once, or copy the model into `data\models`. |
| `This video has no audio track` | Use a video with sound. |
| `No speech was detected` | Check the volume and that someone speaks; try another model or language. |
| `FFmpeg not found` | `winget install Gyan.FFmpeg`, open a NEW window, check `ffmpeg -version` / `ffprobe -version`. |
| `ModuleNotFoundError` / packages missing | Run `scripts\install.bat` again (the app must be started with `scripts\start.bat`, which uses `backend\.venv`). |
| `pip install` fails | Use Python 3.12; check the internet connection; run `install.bat` again. |
| Out of memory | Use `tiny`/`base`, close other programs. |
| Kannada shows as boxes in export | Read `docs\FONTS.md`. |

**Reading the log:** each line is `time LEVEL [part] message`. Search for `ERROR`; a block starting with
`Traceback (most recent call last)` ends with the real error. The log never stores passwords or tokens.

## Where things are stored
| What | Folder |
|---|---|
| Speech models | `data\models` |
| Projects, uploads, exports, database | `data\projects`, `data\garuda.db` |
| Temporary work files | `data\projects\<id>\work` (deleted after every job) |
| Log | `data\logs\garuda.log` |

## Tests: what exists and how to run it
`scripts\test.bat` (details in the table above). Test groups: media/captions/export with **real FFmpeg**
(`test_core.py`); the real `transcribe()` function against a **mocked** speech library (`test_transcribe.py`);
API tests (`test_api.py`); web-interface logic (`frontend`); and an optional **real-library** test (`scripts\test.bat e2e`).

### Honest verification status of this version
| Item | Status |
|---|---|
| Core media, caption, export tests (42, real FFmpeg) | **Run and passed** in the repair sandbox |
| Transcription logic tests (41 + 2 skipped real-library; includes the PyAV `metadata_errors` regression) | **Run and passed** |
| Kannada model registry / custom model / vocabulary (12), accuracy tool (10), converter logic (4), training-kit helper (2) | **Run and passed** (mocked speech library) |
| IndicConformer bridge + worker (29 tests, stand-in for NeMo) | **Run and passed** |
| IndicConformer in real WSL/NeMo, and its real accuracy | **NOT run by me** (`scripts\\diagnose.bat --indic`) |
| Real Kannada accuracy of `kn-medium` / `kn-small` on your videos | **NOT measured by me** (no internet/audio/model). Use `scripts\\evaluate.bat` |
| `convert_model` (needs PyTorch) and `finetune_whisper.py` (needs a GPU) | **NOT run** (written from the standard recipe; see `training\\README.md`) |
| Real faster-whisper + real model, Kannada/mixed audio | **NOT run by me** (no library, model, internet or sample audio in the repair sandbox). The PyAV error above was found by a real run on the user's PC; confirm the fix with `scripts\\diagnose.bat --deep` and `test.bat e2e` |
| API tests (`test_api.py`, incl. new ones), full upload-to-export flow in a browser | **NOT run** (FastAPI/SQLAlchemy could not be installed there) |
| Windows `.bat` scripts | **Reviewed, not executed** (no Windows in the sandbox) |
| React components, frontend vitest | Syntax-checked only |
| Kannada rendering in exported video | Needs a Kannada font; check a short test export (see `docs\FONTS.md`) |

## Updating safely
Close the app, copy the folder as a backup (your `data` folder holds projects and models), unzip the new version
next to it, copy `.env` and `data` into the new folder, run `scripts\install.bat` then `scripts\diagnose.bat`.

## Project layout
```
backend/app/    config, logging_setup, media, transcribe, indic, diagnostics, evaluate, captions, render, fonts, db, jobs, schemas, main
backend/tests/  test_core.py, test_transcribe.py (+ fake_faster_whisper.py), test_api.py
frontend/src/   React + TypeScript editor
scripts/        install.bat, start.bat, diagnose.bat, test.bat, dev.bat (setup.bat forwards to install.bat)
docs/           WINDOWS_SETUP.md, KANNADA_ACCURACY.md, INDIC_CONFORMER.md, FONTS.md, DEPLOYMENT.md
training/       Colab fine-tuning kit (make_metadata.py, finetune_whisper.py, README.md)
backend/tools/  convert_model.py (used by scripts\\convert_model.bat)
```
API docs appear at http://127.0.0.1:8000/docs while the app runs. Health: `/api/health`, `/api/ready`, `/api/diagnostics`.

## Not built
Login/accounts (single-user, listens on 127.0.0.1 only), database migrations, B-roll, cloud transcription,
drag-to-resize on the timeline. No optional service in this project costs money.

## Security notes
Real media inspection (extension never trusted), size/duration/resolution limits, random internal filenames,
path-traversal-safe IDs, FFmpeg started with argument lists (no shell), strict validation, rate limit, CORS allow-list,
generic error messages (details only in the local log), secrets masked in logs, `.env` is git-ignored.

"""AI4Bharat IndicConformer (a NeMo .nemo file) as a second speech engine.

Why a "worker"?  This model was trained with AI4Bharat's own fork of NVIDIA NeMo, which targets Linux and cannot be
installed on plain Windows. So the app does everything that is portable and a small worker script
(backend/tools/indic_worker.py) does the actual recognition in an environment that HAS that NeMo fork (the supported
setup is WSL2 Ubuntu on Windows 11: scripts\\setup_indic.bat).

The app side (this file):
  1. finds the pauses in the audio and cuts it into chunks of at most ~25 seconds (the model is trained on short audio)
  2. writes a manifest, starts the worker, shows progress, supports cancel and a time limit
  3. validates what comes back and builds captions with REAL times (chunk start + word time when the model gives word
     times; otherwise honest phrase-level times for the whole chunk)

The recognition itself (accuracy) comes from the model and cannot be checked without running it.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

from .config import IndicConfig
from .transcribe import INDIC, LANGUAGES, MIXED, Cancelled, TranscribeError, _num

log = logging.getLogger("garuda.indic")

# Languages of this model that the app can ask for (the model covers the 22 official languages; English is not one).
SUPPORTED = {"kn", "hi", "te", "ta", "ml", "bn", "mr", "gu", "pa", "ur", "as", "sa", "ne"}
LOG_HINT = "The technical details were saved in data\\logs\\garuda.log."
SETUP_HELP = "Run scripts\\setup_indic.bat (see docs\\INDIC_CONFORMER.md)."
MIN_SPEECH = 0.3          # a sound shorter than this is treated as noise, not speech


# ---------------------------------------------------------------------------------------------- paths + command
def parse_runner(text: str) -> list[str]:
    """'wsl -d Ubuntu-22.04 -- /home/me/indic-venv/bin/python' -> list of arguments."""
    import shlex
    return [part.strip('"') for part in shlex.split(text or "", posix=(os.name != "nt"))]


def is_wsl(runner: list[str]) -> bool:
    return bool(runner) and Path(runner[0]).name.lower() in ("wsl", "wsl.exe")


def to_runtime_path(path: Path | str, runner: list[str]) -> str:
    """The path as the worker sees it. Inside WSL, C:\\a\\b becomes /mnt/c/a/b."""
    text = str(path)
    if is_wsl(runner):
        posix = text.replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", posix):
            return f"/mnt/{posix[0].lower()}{posix[2:]}"
        return posix
    return text


# ---------------------------------------------------------------------------------------------- audio chunking
def detect_silences(wav: Path, *, noise_db: float = -35.0, min_silence: float = 0.4,
                    total: float | None = None) -> list[tuple[float, float]]:
    """(start, end) of every pause, found with FFmpeg's silencedetect filter."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav), "-af",
           f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TranscribeError("Could not analyse the audio for pauses (is FFmpeg installed?).", "audio_analysis_failed") from exc
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        found_start = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        found_end = re.search(r"silence_end:\s*([\d.]+)", line)
        if found_start:
            start = max(0.0, float(found_start.group(1)))
        elif found_end and start is not None:
            silences.append((start, float(found_end.group(1))))
            start = None
    if start is not None:                                # the pause lasts until the end of the file
        silences.append((start, total if total is not None else start))
    return silences


def plan_chunks(silences: list[tuple[float, float]], total: float, *, max_chunk: float = 25.0,
                pad: float = 0.1) -> list[tuple[float, float]]:
    """Cut the audio into chunks of at most `max_chunk` seconds, only at pauses where possible.

    Speech = everything between pauses. Neighbouring pieces of speech are joined while the chunk stays short enough; a
    single piece longer than `max_chunk` is split into equal parts (the only case where a cut can fall inside speech).
    Each chunk gets a small `pad` of surrounding silence so the first and last sound are not clipped.
    """
    if total <= 0:
        return []
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for begin, end in sorted(silences):
        begin, end = max(0.0, begin), min(total, end)
        if begin > cursor:
            spans.append((cursor, begin))
        cursor = max(cursor, end)
    if cursor < total:
        spans.append((cursor, total))
    spans = [(a, b) for a, b in spans if b - a >= MIN_SPEECH]

    pieces: list[tuple[float, float]] = []
    for a, b in spans:
        if b - a > max_chunk:
            parts = math.ceil((b - a) / max_chunk)
            step = (b - a) / parts
            pieces.extend((a + i * step, a + (i + 1) * step) for i in range(parts))
        else:
            pieces.append((a, b))

    chunks: list[list[float]] = []
    for a, b in pieces:
        if chunks and b - chunks[-1][0] <= max_chunk:
            chunks[-1][1] = b
        else:
            chunks.append([a, b])

    padded: list[tuple[float, float]] = []
    for index, (a, b) in enumerate(chunks):
        low = padded[-1][1] if padded else 0.0
        high = chunks[index + 1][0] if index + 1 < len(chunks) else total
        padded.append((round(max(low, a - pad), 3), round(min(high, b + pad, total), 3)))
    return padded


def slice_wav(src: Path, chunks: list[tuple[float, float]], out_dir: Path) -> list[Path]:
    """Write each chunk as its own 16 kHz mono WAV file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with wave.open(str(src), "rb") as fh:
        rate, width, channels = fh.getframerate(), fh.getsampwidth(), fh.getnchannels()
        frames = fh.readframes(fh.getnframes())
    frame_bytes = width * channels
    for index, (a, b) in enumerate(chunks):
        first, last = int(round(a * rate)) * frame_bytes, int(round(b * rate)) * frame_bytes
        target = out_dir / f"chunk{index:04d}.wav"
        with wave.open(str(target), "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(rate)
            out.writeframes(frames[first:last])
        paths.append(target)
    return paths


# ---------------------------------------------------------------------------------------------- running the worker
def _worker_command(cfg: IndicConfig, *args: str) -> list[str]:
    return [*cfg.runner, to_runtime_path(cfg.worker_path, cfg.runner), *args]


def run_worker(cfg: IndicConfig, manifest: Path, output: Path, total_chunks: int,
               on_progress: Callable[[float], None], cancel: threading.Event) -> str:
    """Start the worker, follow its progress, honour cancel and the time limit. Returns the worker's error text."""
    command = _worker_command(cfg, "--manifest", to_runtime_path(manifest, cfg.runner),
                              "--output", to_runtime_path(output, cfg.runner))
    log.info("Starting IndicConformer worker: %s", " ".join(command))
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise TranscribeError(f"The IndicConformer runner could not be started ('{cfg.runner[0] if cfg.runner else ''}' "
                              f"was not found). {SETUP_HELP}", "indic_runner_missing") from exc
    except OSError as exc:
        log.exception("Could not start the IndicConformer worker")
        raise TranscribeError(f"The IndicConformer worker could not be started ({type(exc).__name__}). {LOG_HINT}",
                              "indic_runner_missing") from exc

    errors: list[str] = []

    def read_stdout() -> None:
        for line in proc.stdout or []:
            match = re.match(r"PROGRESS\s+(\d+)\s+(\d+)", line)
            if match and total_chunks:
                try:
                    on_progress(min(0.99, int(match.group(1)) / total_chunks))
                except Exception:  # noqa: BLE001 - progress problems must not stop recognition
                    log.debug("progress callback failed", exc_info=True)

    def read_stderr() -> None:
        for line in proc.stderr or []:
            errors.append(line.rstrip())
            del errors[:-300]

    threads = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
    for thread in threads:
        thread.start()
    started = time.time()
    try:
        while proc.poll() is None:
            if cancel.is_set():
                proc.terminate()
                raise Cancelled()
            if time.time() - started > cfg.timeout_s:
                proc.terminate()
                raise TranscribeError(f"IndicConformer took longer than the time limit ({cfg.timeout_s // 60} minutes) "
                                      "and was stopped. Try a shorter video.", "indic_timeout")
            time.sleep(0.1)
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        for thread in threads:
            thread.join(timeout=2)
    tail = "\n".join(errors[-40:])
    if proc.returncode not in (0, None):
        log.error("IndicConformer worker exited with code %s:\n%s", proc.returncode, tail)
    return tail


def check_runtime(cfg: IndicConfig, *, load_model: bool = False) -> dict[str, Any]:
    """Ask the worker to describe its environment (and optionally load the model). Never raises."""
    if not cfg.runner:
        return {"ok": False, "problems": [f"No runner is configured. {SETUP_HELP}"]}
    args = ["--check"] + (["--model", to_runtime_path(cfg.model_path, cfg.runner)] if load_model else [])
    try:
        proc = subprocess.run(_worker_command(cfg, *args), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=900, check=False)
    except FileNotFoundError:
        return {"ok": False, "problems": [f"The runner '{cfg.runner[0]}' was not found. {SETUP_HELP}"]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "problems": [f"The runner could not be started: {type(exc).__name__}: {exc}"]}
    try:
        report = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "problems": ["The worker did not answer. Its error output was:",
                                          (proc.stderr or proc.stdout or "(nothing)").strip()[-800:]]}
    return report


# ---------------------------------------------------------------------------------------------- main entry point
def _clean_words(raw: Any, offset: float, length: float) -> list[dict]:
    """Validated word timings of one chunk, converted to times in the whole recording."""
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        text = " ".join(text.split()) if isinstance(text, str) else ""
        start, end = _num(item.get("start")), _num(item.get("end"))
        if text and start is not None and end is not None and 0 <= start < end <= length + 0.5:
            out.append({"text": text, "start": round(offset + start, 3), "end": round(offset + end, 3),
                        "kind": "word", "confidence": None})
    return out


def transcribe_indic(audio: Path, *, language: str, cfg: IndicConfig | None, on_progress: Callable[[float], None],
                     cancel: threading.Event, include_detail: bool = True) -> dict:
    """Same result shape as transcribe.transcribe(): {"language","words","warnings","engine"}."""
    if language not in LANGUAGES:
        raise TranscribeError(f"Unsupported language: {language}", "bad_language")
    warnings: list[str] = []
    code = language
    if language in ("auto", MIXED):
        code = "kn"
        warnings.append("IndicConformer needs a language, so Kannada was used. It writes Kannada in Kannada letters: English "
                        "words spoken inside a sentence will be written in Kannada letters or may be missed. For English-heavy "
                        "videos use the Whisper models.")
    elif code not in SUPPORTED:
        raise TranscribeError(f"IndicConformer does not support '{LANGUAGES.get(language, language)}'. "
                              "Use a Whisper model for English.", "bad_language")
    if cfg is None or not cfg.model_path.is_file():
        raise TranscribeError("The IndicConformer model file was not found. Put the .nemo file in data\\models\\indic.",
                              "indic_model_missing")
    if not cfg.runner:
        raise TranscribeError(f"IndicConformer is not set up yet. {SETUP_HELP}", "indic_runner_missing")

    audio = Path(audio)
    try:
        with wave.open(str(audio), "rb") as fh:
            if (fh.getnchannels(), fh.getsampwidth(), fh.getframerate()) != (1, 2, 16000):
                raise TranscribeError("The audio must be 16 kHz mono 16-bit WAV (the app creates this from your video).", "audio_format")
            total = fh.getnframes() / fh.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        raise TranscribeError("The extracted audio file could not be read.", "audio_unreadable") from exc

    chunks = plan_chunks(detect_silences(audio, total=total), total, max_chunk=cfg.chunk_seconds)
    if not chunks:
        raise TranscribeError("No speech was detected in this video's audio. Check that people are speaking and the "
                              "volume is not too low.", "no_speech")
    log.info("IndicConformer: %.1f s of audio cut into %d chunk(s)", total, len(chunks))

    work = Path(tempfile.mkdtemp(prefix="garuda-indic-", dir=str(audio.parent)))
    try:
        paths = slice_wav(audio, chunks, work / "chunks")
        manifest, output = work / "manifest.json", work / "result.json"
        manifest.write_text(json.dumps({
            "model": to_runtime_path(cfg.model_path, cfg.runner), "language": code, "decoder": cfg.decoder,
            "chunks": [{"id": i, "path": to_runtime_path(p, cfg.runner), "offset": chunks[i][0],
                        "length": round(chunks[i][1] - chunks[i][0], 3)} for i, p in enumerate(paths)],
        }, ensure_ascii=False), encoding="utf-8")
        on_progress(0.02)
        stderr_tail = run_worker(cfg, manifest, output, len(paths), lambda p: on_progress(0.02 + p * 0.96), cancel)

        try:
            result = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.error("IndicConformer produced no usable result. Worker messages:\n%s", stderr_tail)
            raise TranscribeError(f"IndicConformer did not produce a result. {LOG_HINT} Run scripts\\diagnose.bat --indic.",
                                  "indic_worker_failed") from exc
        if isinstance(result.get("error"), dict):
            err = result["error"]
            log.error("IndicConformer worker error %s: %s\n%s", err.get("type"), err.get("message"), err.get("traceback", ""))
            hint = ""
            if "multilingual" in str(err.get("message", "")) or "multisoftmax" in str(err.get("message", "")):
                hint = " This looks like stock NeMo instead of AI4Bharat's fork: run scripts\\setup_indic.bat again."
            detail = f"{err.get('type', 'Error')}: {str(err.get('message', ''))[:140]}" if include_detail else str(err.get("type", "Error"))
            raise TranscribeError(f"IndicConformer failed ({detail}).{hint} {LOG_HINT}", "indic_worker_failed")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    words: list[dict] = []
    phrase_chunks = 0
    for item in result.get("chunks", []):
        try:
            index = int(item["id"])
            begin, end = chunks[index]
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        text = item.get("text")
        text = " ".join(text.split()) if isinstance(text, str) else ""
        timed = _clean_words(item.get("words"), begin, end - begin)
        if timed:
            words.extend(timed)
        elif text and end > begin:
            phrase_chunks += 1
            words.append({"text": text, "start": round(begin, 3), "end": round(end, 3), "kind": "phrase", "confidence": None})
    words = [w for w in sorted(words, key=lambda w: (w["start"], w["end"])) if w["end"] > w["start"]]
    if not words:
        raise TranscribeError("No speech was detected in this video's audio. Check that people are speaking and the "
                              "volume is not too low.", "no_speech")
    if phrase_chunks:
        warnings.append(f"{phrase_chunks} caption(s) have phrase-level timing only (the model gave no per-word times for them). "
                        "Their timing is real (the start and end of the spoken chunk) but not per word.")
    warnings.append("Accuracy depends on the model and the recording; please proof-read the Kannada captions.")
    return {"language": code, "words": words, "warnings": warnings,
            "engine": {"model": INDIC, "device": "cpu", "decoder": cfg.decoder, "chunks": len(chunks),
                       "timing": "word" if any(w["kind"] == "word" for w in words) else "phrase", "segments": len(words)}}

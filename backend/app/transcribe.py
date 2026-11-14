"""Local speech recognition with faster-whisper (runs on this computer; the internet is only needed once, to
download a model).

How a transcription runs (each stage reports its OWN, specific error instead of one vague message):

  1. check the request (language, model, audio file)
  2. import the speech library and describe the installed versions
  3. choose a runtime that really works here (CPU first; compute type checked against what is installed)
  4. load the model once and reuse it (freed again after an idle period)
  5. decode, trying word-level timing first and falling back to phrase-level timing if the library fails
  6. validate every segment (no None / NaN / negative / reversed timestamps) and refuse to report
     "success" when no valid caption came out

The FULL technical traceback of every failure is written to the log file (data/logs/garuda.log). The web
page only receives a short, safe message that points to that log.

This module imports only the Python standard library at the top, so it can be tested without the
speech library. The speech library is imported lazily inside functions.
"""
from __future__ import annotations

import gc
import inspect
import logging
import math
import re
import threading
import wave
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("garuda.transcribe")

# ---------------------------------------------------------------------------------------------- options
MIXED = "kn-en"
KANGLISH = "kanglish"

# Language codes Whisper supports (and English). Quality differs a lot between languages: proof-read everything.
LANGUAGES = {
    "auto": "Auto-detect (one language per 30-second window)",
    MIXED: "Kannada + English (mixed speech)",
    KANGLISH: "Kanglish (Kannada in Latin letters)",
    "kn": "Kannada",
    "en": "English",
    "hi": "Hindi", "te": "Telugu", "ta": "Tamil", "ml": "Malayalam",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "as": "Assamese",
    "sa": "Sanskrit", "ne": "Nepali",
}
# Whisper has no Odia model support; keep it out rather than pretend.
UNSUPPORTED_NOTE = "Odia is not supported by Whisper models. Use another language or edit captions by hand."

MODELS = {
    "tiny": "Fastest (~1 GB RAM). Lowest accuracy; weak for Kannada.",
    "base": "Fast (~1 GB RAM). Good first test on a CPU laptop; Kannada accuracy is limited.",
    "small": "Balanced (~2 GB RAM, several times slower than real time on a CPU). Better for Kannada.",
    "medium": "Better for Indian languages (~5 GB RAM, slow on CPU).",
    "large-v3": "Best accuracy (~10 GB RAM, very slow on CPU). Only if your laptop can handle it.",
}
WEAK_LANGS = {"kn", MIXED, KANGLISH, "te", "ta", "ml", "bn", "mr", "gu", "pa", "as", "sa", "ur", "ne"}

# --- Kannada-tuned models -------------------------------------------------------------------------------
# General Whisper models were trained on very little Kannada. Models that were FINE-TUNED on Kannada speech are
# far better at it. They are in CTranslate2 format, which is what faster-whisper loads.
CUSTOM = "custom"
INDIC = "indic"           # AI4Bharat IndicConformer (a .nemo file run by a worker in WSL2): see indic.py
INDIC_LABEL = ("IndicConformer (AI4Bharat): Kannada specialist, much faster than Whisper on a CPU. Needs the one-time "
               "WSL setup (scripts\\setup_indic.bat). Writes Kannada in Kannada letters only.")
KANNADA_TUNED = {
    # Ready to use: a CTranslate2 conversion (by a third party) of vasista22/whisper-kannada-medium
    # (SPRING Lab, IIT Madras; Apache-2.0). Downloaded automatically on first use (~1.5 GB).
    "kn-medium": {"source": "itsskofficial/livewhisper-kn-medium",
                  "label": "Kannada-tuned MEDIUM (IIT Madras fine-tune): much better Kannada. ~1.5 GB download once, "
                           "~3 GB RAM, slow on a CPU - try a 1-minute clip first."},
    # Needs a one-time conversion from vasista22/whisper-kannada-small: scripts\\convert_model.bat
    "kn-small": {"folder": "kn-small-ct2",
                 "label": "Kannada-tuned SMALL (converted): better Kannada, faster than kn-medium."},
}
KN_SMALL_HELP = ("The 'kn-small' model has not been converted yet. Run:  scripts\\convert_model.bat "
                 "vasista22/whisper-kannada-small kn-small-ct2 small")


def model_choices(model_dir: Path | None, custom_model: str | None = None, indic: bool = False) -> dict[str, str]:
    """Every model name the app currently offers, with a short description (used by the API and validation)."""
    choices = dict(MODELS)
    choices["kn-medium"] = KANNADA_TUNED["kn-medium"]["label"]
    if model_dir and (Path(model_dir) / KANNADA_TUNED["kn-small"]["folder"]).is_dir():
        choices["kn-small"] = KANNADA_TUNED["kn-small"]["label"]
    if custom_model:
        choices[CUSTOM] = f"Your own model ({custom_model})"
    if indic:
        choices[INDIC] = INDIC_LABEL
    return choices


_REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def resolve_model_source(name: str, model_dir: Path | None, custom_model: str | None) -> tuple[str, bool]:
    """Turn a model name into what WhisperModel() should load. Returns (source, is_local_folder)."""
    if name in MODELS:
        return name, False
    if name == "kn-medium":
        return KANNADA_TUNED["kn-medium"]["source"], False
    if name == "kn-small":
        folder = Path(model_dir or ".") / KANNADA_TUNED["kn-small"]["folder"]
        if not folder.is_dir():
            raise TranscribeError(KN_SMALL_HELP, "model_missing")
        return str(folder), True
    if name == CUSTOM:
        value = (custom_model or "").strip()
        if not value:
            raise TranscribeError("No custom model is set. Put CUSTOM_MODEL=... in the .env file.", "bad_model")
        if Path(value).is_dir():
            return value, True
        if _REPO_ID.match(value):
            return value, False                       # a Hugging Face repo id (downloaded once)
        raise TranscribeError("The folder in CUSTOM_MODEL was not found. Check the path in your .env file.", "model_missing")
    raise TranscribeError(f"Unknown model: {name}", "bad_model")


def vocab_prompt(words: list[str] | None, max_chars: int = 150) -> str:
    """Names and special terms for Whisper to expect (from data/vocabulary.txt). Short on purpose: prompts are limited."""
    out: list[str] = []
    used = 0
    for word in words or []:
        word = " ".join(str(word).split())
        if not word:
            continue
        if used + len(word) + 2 > max_chars:
            break
        out.append(word)
        used += len(word) + 2
    return ", ".join(out)

# A short example of how this speaker talks. Whisper continues in the style of its prompt, so a sentence
# that mixes Kannada script with English words in Latin letters nudges it to KEEP English words in English
# instead of re-spelling them in Kannada letters. It is a hint, not a guarantee.
MIXED_PROMPT = ("ನಮಸ್ಕಾರ, ಇವತ್ತು ನಾವು ಈ video ಬಗ್ಗೆ ಮಾತಾಡೋಣ. "
                "This is a mixed Kannada and English talk. ಕೆಲವು English words ಇರುತ್ತವೆ.")

_KANNADA_CONSONANTS = {
    "ಕ": "K", "ಖ": "KH", "ಗ": "G", "ಘ": "GH", "ಙ": "NG",
    "ಚ": "CH", "ಛ": "CHH", "ಜ": "J", "ಝ": "JH", "ಞ": "NY",
    "ಟ": "T", "ಠ": "TH", "ಡ": "D", "ಢ": "DH", "ಣ": "N",
    "ತ": "T", "ಥ": "TH", "ದ": "D", "ಧ": "DH", "ನ": "N",
    "ಪ": "P", "ಫ": "PH", "ಬ": "B", "ಭ": "BH", "ಮ": "M",
    "ಯ": "Y", "ರ": "R", "ಱ": "R", "ಲ": "L", "ವ": "V",
    "ಶ": "SH", "ಷ": "SH", "ಸ": "S", "ಹ": "H", "ಳ": "L",
}
_KANNADA_VOWELS = {
    "ಅ": "A", "ಆ": "A", "ಇ": "I", "ಈ": "I", "ಉ": "U", "ಊ": "U",
    "ಋ": "RU", "ೠ": "RU", "ಎ": "E", "ಏ": "E", "ಐ": "AI",
    "ಒ": "O", "ಓ": "O", "ಔ": "AU",
}
_KANNADA_VOWEL_SIGNS = {
    "ಾ": "A", "ಿ": "I", "ೀ": "I", "ು": "U", "ೂ": "U",
    "ೃ": "RU", "ೄ": "RU", "ೆ": "E", "ೇ": "E", "ೈ": "AI",
    "ೊ": "O", "ೋ": "O", "ೌ": "AU",
}
_KANNADA_VIRAMA = "್"


def kannada_to_kanglish(text: str) -> str:
    """Romanize Kannada orthography into readable, ASCII-only Kanglish and uppercase the result."""
    out: list[str] = []
    for index, char in enumerate(text):
        if char in _KANNADA_CONSONANTS:
            out.append(_KANNADA_CONSONANTS[char])
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if next_char != _KANNADA_VIRAMA and next_char not in _KANNADA_VOWEL_SIGNS:
                out.append("A")
        elif char in _KANNADA_VOWELS:
            out.append(_KANNADA_VOWELS[char])
        elif char in _KANNADA_VOWEL_SIGNS:
            out.append(_KANNADA_VOWEL_SIGNS[char])
        elif char == "ಂ":
            out.append("M")
        elif char == "ಃ":
            out.append("H")
        elif char in (_KANNADA_VIRAMA, "\u200c", "\u200d"):
            continue
        else:
            out.append(char.upper())
    return "".join(out)


def _romanize_kanglish_result(result: dict) -> dict:
    for word in result.get("words", []):
        word["text"] = kannada_to_kanglish(word["text"])
    result["language"] = KANGLISH
    result.setdefault("warnings", []).append(
        "Kanglish converts Kannada script to uppercase Latin letters. This is a pronunciation-style "
        "transliteration, not translation; review names and spelling."
    )
    return result

LOG_HINT = "The technical details were saved in data\\logs\\garuda.log."
MAX_WORD_DURATION_S = 5.0


class TranscribeError(Exception):
    """A problem that is safe to show to the user. `code` is stable and machine-readable."""

    def __init__(self, message: str, code: str = "transcribe_failed"):
        super().__init__(message)
        self.message = message
        self.code = code


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------------------------------------- helpers
_PATH_RE = re.compile(r"([A-Za-z]:\\[^\s'\"<>|]+|/(?:home|Users|root|tmp|var|mnt)/[^\s'\"<>|]+)")


def _detail(exc: BaseException, *, include_message: bool) -> str:
    """'TypeError: short message' with file paths removed, or just the class name in production."""
    name = type(exc).__name__
    if not include_message:
        return name
    message = _PATH_RE.sub("<path>", " ".join(str(exc).split()))
    return f"{name}: {message[:160]}" if message else name


def _num(value: Any) -> float | None:
    """A finite float, or None for None / NaN / inf / non-numbers (bool is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def installed_versions() -> dict[str, str]:
    """Versions of the packages that decide whether speech recognition works."""
    from importlib import metadata

    out: dict[str, str] = {}
    for package in ("faster-whisper", "ctranslate2", "onnxruntime", "av", "tokenizers", "huggingface-hub", "numpy"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = "not installed"
        except Exception:  # noqa: BLE001 - a broken package record must not break diagnostics
            out[package] = "unknown"
    return out


def supported_kwargs(func: Callable, wanted: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep only the keyword arguments that `func` really accepts. Returns (accepted, dropped_names).

    This prevents "got an unexpected keyword argument" TypeErrors when the installed library is older or
    newer than the one this code was written for. Dropped arguments are logged and reported.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return dict(wanted), []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(wanted), []
    accepted = {k: v for k, v in wanted.items() if k in params}
    return accepted, [k for k in wanted if k not in params]


def plan_options(
    language: str, *, beam_size: int, word_timestamps: bool, vad_filter: bool, accepts: set[str] | None = None,
    kannada_only_model: bool = False, vocabulary: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """The arguments for WhisperModel.transcribe() and any warnings about them. Pure and easy to test.

    * the task is ALWAYS "transcribe" (never translate: Kannada stays Kannada, English stays English)
    * "auto" = let Whisper detect the language;  "kn-en" = per-window detection + a mixed-language prompt
    * `accepts` (names the installed library supports) lets callers and tests check compatibility
    """
    warnings: list[str] = []
    options: dict[str, Any] = {
        "task": "transcribe",
        "beam_size": int(beam_size),
        "word_timestamps": bool(word_timestamps),
        "vad_filter": bool(vad_filter),
        # Reading the previous text back in is the main cause of endless repetition on hard audio.
        "condition_on_previous_text": False,
    }
    if vad_filter:
        # Preserve quiet word beginnings/endings at the edge of detected speech regions.
        options["vad_parameters"] = {"speech_pad_ms": 1000}
    vocab = vocab_prompt(vocabulary)
    if kannada_only_model and language in ("auto", MIXED):
        # A model fine-tuned only on Kannada cannot detect languages reliably: Kannada is forced.
        options["language"] = "kn"
        if language == MIXED:
            options["initial_prompt"] = MIXED_PROMPT
        warnings.append("This Kannada-tuned model was trained on Kannada speech, so Kannada is forced. English words inside "
                        "a sentence may be written in Kannada letters. For mixed or other-language speech, use the general "
                        "'large-v3' model, or edit the captions.")
    elif language == KANGLISH:
        options["language"] = "kn"
    elif kannada_only_model and language != "kn":
        options["language"] = language
        warnings.append(f"This Kannada-tuned model was trained for Kannada, but the selected language is "
                        f"{LANGUAGES.get(language, language)}. Recognition may be inaccurate; use the general "
                        "'large-v3' model for other languages.")
    elif language == "auto":
        options["language"] = None
    elif language == MIXED:
        options["initial_prompt"] = MIXED_PROMPT
        if accepts is None or "multilingual" in accepts:
            options["language"] = None
            options["multilingual"] = True
        else:
            options["language"] = "kn"
            warnings.append("This version of the speech library cannot re-detect the language for every window, so "
                            "Kannada was forced. English words may appear in Kannada letters. Update faster-whisper to fix this.")
    else:
        options["language"] = language
    if vocab:
        options["initial_prompt"] = f"{options['initial_prompt']} {vocab}" if options.get("initial_prompt") else vocab
    if accepts is not None:
        options = {k: v for k, v in options.items() if k in accepts}
    return options, warnings


# ---------------------------------------------------------------------------------------------- runtime + model cache
def resolve_runtime(device: str, compute_type: str) -> tuple[str, str, list[str]]:
    """Pick a device and compute type that this computer can really use (CPU is always the fallback)."""
    warnings: list[str] = []
    dev = device if device in ("cpu", "cuda") else "cpu"
    if dev != device:
        warnings.append(f"Unknown device '{device}'; the CPU is used.")
    supported: set[str] | None = None
    try:
        import ctranslate2

        if dev == "cuda" and ctranslate2.get_cuda_device_count() == 0:
            warnings.append("No usable NVIDIA graphics card was found, so the CPU is used instead.")
            dev = "cpu"
        supported = set(ctranslate2.get_supported_compute_types(dev))
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        log.warning("Could not query ctranslate2 for supported compute types", exc_info=True)
        if dev == "cuda":
            warnings.append("The graphics card could not be used, so the CPU is used instead.")
            dev, supported = "cpu", None
    chosen = compute_type
    if supported and compute_type not in supported:
        chosen = next((c for c in ("int8", "int8_float32", "float32") if c in supported), "default")
        warnings.append(f"Compute type '{compute_type}' is not supported on the {dev}; '{chosen}' is used.")
    return dev, chosen, warnings


_cache_lock = threading.RLock()
_models: dict[tuple, Any] = {}
_idle_timer: list[threading.Timer | None] = [None]


def _import_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscribeError(
            "The speech library (faster-whisper) is not installed. Run scripts\\install.bat again.", "whisper_missing") from exc
    except Exception as exc:  # noqa: BLE001 - e.g. a Windows DLL that cannot be loaded
        log.exception("faster-whisper is installed but could not be imported")
        raise TranscribeError(
            f"The speech library is installed but could not start ({type(exc).__name__}). {LOG_HINT} "
            "Run scripts\\diagnose.bat for a full report.", "whisper_import_failed") from exc
    return WhisperModel


def _classify_load_error(exc: BaseException, model_size: str, allow_download: bool) -> TranscribeError:
    names = " ".join(c.__name__ for c in type(exc).__mro__)
    text = f"{names} {exc}".lower()
    offline_words = ("localentrynotfound", "connection", "timeout", "offline", "max retries", "getaddrinfo",
                     "name resolution", "ssl", "couldn't connect", "cannot find an appropriate cached snapshot",
                     "not found in local cache")
    if isinstance(exc, MemoryError):
        return TranscribeError("Not enough memory to load this model. Choose a smaller model.", "out_of_memory")
    if any(word in text for word in offline_words):
        if not allow_download:
            return TranscribeError(
                f"The '{model_size}' model is not on this computer and downloads are turned off "
                "(ALLOW_MODEL_DOWNLOAD=false). Turn downloads on once, or copy the model into data\\models.",
                "model_missing")
        return TranscribeError(
            f"The '{model_size}' model has to be downloaded once (internet needed), but the download failed. "
            f"Check the internet connection and try again. {LOG_HINT}", "model_download_failed")
    return TranscribeError(f"Could not load the '{model_size}' model ({type(exc).__name__}). {LOG_HINT}", "model_load_failed")


def get_model(model_size: str, *, device: str, compute_type: str, model_dir: Path | None,
              allow_download: bool, cpu_threads: int = 0, label: str | None = None) -> tuple[Any, bool]:
    """Return (model, was_cached). The model is loaded once and then reused.
    `model_size` is what WhisperModel loads (a size name, a Hugging Face repo id or a folder); `label` is the
    friendly name used in messages."""
    label = label or model_size
    key = (model_size, device, compute_type, str(model_dir), cpu_threads)
    with _cache_lock:
        cached = _models.get(key)
        if cached is not None:
            return cached, True
        WhisperModel = _import_whisper_model()
        wanted = {"device": device, "compute_type": compute_type, "download_root": str(model_dir) if model_dir else None,
                  "local_files_only": not allow_download, "cpu_threads": cpu_threads}
        kwargs, dropped = supported_kwargs(WhisperModel, wanted)
        if dropped:
            log.warning("WhisperModel does not accept %s in this version; skipped.", dropped)
        if model_dir:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
        log.info("Loading model '%s' = %s (device=%s, compute_type=%s, threads=%s, downloads allowed=%s)",
                 label, model_size, device, compute_type, cpu_threads or "default", allow_download)
        try:
            model = WhisperModel(model_size, **kwargs)
        except Exception as exc:  # noqa: BLE001
            log.exception("Model load failed (model=%s source=%s %s)", label, model_size, installed_versions())
            raise _classify_load_error(exc, label, allow_download) from exc
        _models.clear()          # keep at most one model in memory (16 GB laptops)
        _models[key] = model
        return model, False


def release_models() -> int:
    """Free every loaded model (and its memory). Returns how many were released."""
    with _cache_lock:
        count = len(_models)
        _models.clear()
    if count:
        gc.collect()
        log.info("Released %d speech model(s) from memory.", count)
    return count


def loaded_models() -> int:
    with _cache_lock:
        return len(_models)


def _schedule_idle_release(seconds: int) -> None:
    with _cache_lock:
        if _idle_timer[0] is not None:
            _idle_timer[0].cancel()
            _idle_timer[0] = None
        if seconds > 0:
            timer = threading.Timer(seconds, release_models)
            timer.daemon = True
            _idle_timer[0] = timer
            timer.start()


# ---------------------------------------------------------------------------------------------- decoding
def audio_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as fh:
            rate = fh.getframerate()
            return fh.getnframes() / rate if rate else None
    except (wave.Error, EOFError, OSError):
        return None


def load_audio_array(path: Path):
    """Read the extracted 16 kHz mono 16-bit WAV into a float32 numpy array (values -1..1).

    The array is handed to faster-whisper DIRECTLY. Given a file path, faster-whisper decodes it with the PyAV
    package, and some PyAV releases are incompatible with some faster-whisper releases (this is exactly what caused
    "open() got an unexpected keyword argument 'metadata_errors'"). The app has already produced a clean WAV with
    FFmpeg, so PyAV is simply not needed here.
    """
    import numpy as np

    try:
        with wave.open(str(path), "rb") as fh:
            channels, width, rate = fh.getnchannels(), fh.getsampwidth(), fh.getframerate()
            frames = fh.readframes(fh.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        log.exception("Could not read the extracted audio %s", Path(path).name)
        raise TranscribeError("The extracted audio file could not be read.", "audio_unreadable") from exc
    if (channels, width, rate) != (1, 2, 16000):
        raise TranscribeError("The audio must be 16 kHz mono 16-bit WAV (the app creates this automatically from "
                              "your video).", "audio_format")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    del frames
    samples /= 32768.0
    return samples


def consume_segments(segments, *, total: float, want_words: bool, cancel: threading.Event,
                     progress: Callable[[float], None]) -> tuple[list[dict], dict[str, int]]:
    """Turn the library's segments into validated caption entries.

    Every value is checked, so a missing/NaN/negative timestamp can never cause a TypeError or a bogus
    caption. A segment whose words cannot all be trusted keeps honest PHRASE-level timing instead.
    """
    out: list[dict] = []
    stats = {"phrase_segments": 0, "dropped": 0, "filtered": 0, "unicode_replacements": 0}
    for seg in segments:  # the library does the real decoding lazily, inside this loop
        if cancel.is_set():
            raise Cancelled()
        text = getattr(seg, "text", None)
        text = text if isinstance(text, str) else ""
        text_replacements = text.count("\ufffd")
        text = " ".join(text.replace("\ufffd", "").split())
        symbol_only_text = bool(text) and not any(char.isalnum() for char in text)
        if symbol_only_text:
            text = ""
        start, end = _num(getattr(seg, "start", None)), _num(getattr(seg, "end", None))

        good, spoken, word_replacements = [], 0, 0
        for w in ((getattr(seg, "words", None) or []) if want_words else []):
            token = getattr(w, "word", None)
            token = token.strip() if isinstance(token, str) else ""
            word_replacements += token.count("\ufffd")
            token = token.replace("\ufffd", "").strip()
            if not token or not any(char.isalnum() for char in token):
                continue
            spoken += 1
            ws, we = _num(getattr(w, "start", None)), _num(getattr(w, "end", None))
            if (ws is not None and we is not None and ws >= 0 and we > ws
                    and we - ws <= MAX_WORD_DURATION_S):
                prob = _num(getattr(w, "probability", None))
                good.append({"text": token, "start": round(ws, 3), "end": round(we, 3), "kind": "word",
                             "confidence": None if prob is None else round(min(max(prob, 0.0), 1.0), 3)})
        if good and len(good) == spoken:
            stats["unicode_replacements"] += word_replacements
            out.extend(good)
        elif text and start is not None and end is not None and start >= 0 and end > start:
            stats["unicode_replacements"] += text_replacements
            stats["phrase_segments"] += 1  # alignment missing or incomplete: keep real phrase timing only
            out.append({"text": text, "start": round(start, 3), "end": round(end, 3), "kind": "phrase", "confidence": None})
        elif text:
            stats["unicode_replacements"] += text_replacements
            stats["dropped"] += 1
        elif symbol_only_text and not good:
            stats["unicode_replacements"] += text_replacements
            stats["filtered"] += 1
        else:
            stats["unicode_replacements"] += max(text_replacements, word_replacements)
        if end is not None:
            try:
                progress(min(0.99, end / total))
            except Exception:  # noqa: BLE001 - a progress-reporting problem must never fail the transcription
                log.debug("progress callback failed", exc_info=True)

    # Final validation: inside the audio, in order, never reversed.
    valid: list[dict] = []
    for item in sorted(out, key=lambda x: (x["start"], x["end"])):
        if item["start"] >= total + 0.5:
            stats["dropped"] += 1
            continue
        item["end"] = round(min(item["end"], total + 0.5), 3)
        if item["end"] > item["start"]:
            valid.append(item)
        else:
            stats["dropped"] += 1
    return valid, stats


def _script_summary(words: list[dict]) -> str:
    kannada = latin = 0
    for item in words:
        for ch in item["text"]:
            if "\u0c80" <= ch <= "\u0cff":
                kannada += 1
            elif ch.isascii() and ch.isalpha():
                latin += 1
    return f"{kannada} Kannada letters and {latin} English letters"


def transcribe(
    audio: Path, *, language: str, model_size: str, device: str, compute_type: str,
    model_dir: Path | None, allow_download: bool,
    on_progress: Callable[[float], None], cancel: threading.Event,
    cpu_threads: int = 0, beam_size: int = 3, idle_release_s: int = 0, include_detail: bool = True,
    custom_model: str | None = None, vocabulary: list[str] | None = None, indic: Any = None,
) -> dict:
    """Returns {"language", "words": [{text,start,end,kind,confidence}], "warnings": [...], "engine": {...}}.

    Raises TranscribeError (safe message + stable code) or Cancelled. Never returns an empty success.
    """
    # 1 --- request ---------------------------------------------------------------------------------------
    if language not in LANGUAGES:
        raise TranscribeError(f"Unsupported language: {language}. {UNSUPPORTED_NOTE if language == 'or' else ''}".strip(),
                              "bad_language")
    if model_size == INDIC:                     # a different engine: the IndicConformer worker (see indic.py)
        from . import indic as indic_engine
        indic_language = "kn" if language == KANGLISH else language
        result = indic_engine.transcribe_indic(Path(audio), language=indic_language, cfg=indic, on_progress=on_progress,
                                               cancel=cancel, include_detail=include_detail)
        return _romanize_kanglish_result(result) if language == KANGLISH else result
    if model_size not in model_choices(model_dir, custom_model) and model_size != "kn-small":
        raise TranscribeError(f"Unknown model: {model_size}", "bad_model")
    model_source, _is_local = resolve_model_source(model_size, model_dir, custom_model)
    audio = Path(audio)
    if not audio.is_file() or audio.stat().st_size < 1000:
        raise TranscribeError("The extracted audio is missing or empty, so there is nothing to recognise.", "audio_missing")
    duration = audio_seconds(audio)
    if duration is not None and duration < 0.2:
        raise TranscribeError("The audio is too short to recognise (under a fifth of a second).", "audio_too_short")

    samples = load_audio_array(audio)        # fail fast on unreadable / wrongly formatted audio
    duration = len(samples) / 16000.0

    # 2-4 --- library, runtime, model ----------------------------------------------------------------------
    warnings: list[str] = []
    device, compute_type, runtime_warnings = resolve_runtime(device, compute_type)
    warnings.extend(runtime_warnings)
    try:
        model, was_cached = get_model(model_source, device=device, compute_type=compute_type, model_dir=model_dir,
                                      allow_download=allow_download, cpu_threads=cpu_threads, label=model_size)
    except BaseException:
        _schedule_idle_release(idle_release_s)
        raise
    try:
        accepts = set(inspect.signature(model.transcribe).parameters)
    except (TypeError, ValueError):
        accepts = None

    total = max(float(duration or 0.0), 0.001)
    first_error: BaseException | None = None
    last_error: BaseException | None = None
    words: list[dict] = []
    info: Any = None
    used = ""
    stats = {"phrase_segments": 0, "dropped": 0}
    option_warnings: list[str] = []
    queue = [(True, True), (False, True), (False, False)]   # (word timing, voice-activity filter)
    seen: set[tuple[bool, bool]] = set()
    retried_without_vad = False

    try:
        # 5 --- decode (with fallbacks) ---------------------------------------------------------------------
        while queue and not words:
            word_timing, vad = queue.pop(0)
            if (word_timing, vad) in seen:
                continue
            seen.add((word_timing, vad))
            options, option_warnings = plan_options(language, beam_size=beam_size, word_timestamps=word_timing,
                                                    vad_filter=vad, accepts=accepts,
                                                    kannada_only_model=model_size in KANNADA_TUNED, vocabulary=vocabulary)
            label = f"{'word' if word_timing else 'phrase'}-level timing, voice filter {'on' if vad else 'off'}"
            log.info("Decoding %s (%.1f s of audio, %s) with options %s", audio.name, duration, label,
                     {k: v for k, v in options.items() if k != "initial_prompt"})
            try:
                segments, info = model.transcribe(samples, **options)  # an array: faster-whisper skips PyAV
                length = _num(getattr(info, "duration", None))
                if length and length > 0:
                    total = max(length, 0.001)
                words, stats = consume_segments(segments, total=total, want_words=word_timing, cancel=cancel,
                                                progress=on_progress)
            except Cancelled:
                raise
            except MemoryError as exc:
                log.exception("Out of memory while decoding")
                raise TranscribeError("Ran out of memory while transcribing. Choose a smaller model.", "out_of_memory") from exc
            except Exception as exc:  # noqa: BLE001 - recorded in full, then the next fallback is tried
                last_error = exc
                first_error = first_error or exc
                words = []
                log.exception("Decoding failed with %s (model=%s, device=%s, compute=%s, versions=%s)",
                              label, model_size, device, compute_type, installed_versions())
                continue
            if words:
                used = label
                break
            # Decoding WORKED but found nothing. That is not a library failure, so the exception fallbacks are
            # cancelled; the only extra try is without the voice filter, which can drop soft speech.
            queue.clear()
            if vad and not retried_without_vad:
                retried_without_vad = True
                queue.append((word_timing, False))
                log.info("No speech found with the voice-activity filter; retrying without it.")
    finally:
        _schedule_idle_release(idle_release_s)

    # 6 --- never report success without real captions ----------------------------------------------------
    if not words:
        if first_error is not None:
            raise TranscribeError(
                f"Speech recognition failed ({_detail(first_error, include_message=include_detail)}). {LOG_HINT}",
                "decode_failed") from last_error
        raise TranscribeError("No speech was detected in this video's audio. Check that people are speaking and "
                              "the volume is not too low.", "no_speech")

    if language == KANGLISH:
        for word in words:
            word["text"] = kannada_to_kanglish(word["text"])

    warnings.extend(option_warnings)
    if first_error is not None:
        warnings.append(f"The speech library failed with its first settings ({_detail(first_error, include_message=include_detail)}); "
                        f"the captions were made with {used} instead. {LOG_HINT}")
    if retried_without_vad and "voice filter off" in used:
        warnings.append("Voice-activity detection found no speech, so recognition was repeated without it. "
                        "Silent parts can then produce made-up words: check the captions carefully.")
    detected = getattr(info, "language", None) or (language if language not in ("auto", MIXED) else "und")
    if stats["phrase_segments"]:
        warnings.append(f"{stats['phrase_segments']} caption(s) only have phrase-level timing (word timing was "
                        "unavailable for them). Their timing is real but not per word.")
    if stats["dropped"]:
        warnings.append(f"{stats['dropped']} piece(s) of text were left out because their timestamps were invalid.")
    if stats["filtered"]:
        warnings.append(f"{stats['filtered']} caption segment(s) containing only symbols were ignored.")
    if stats["unicode_replacements"]:
        warnings.append(f"Removed {stats['unicode_replacements']} invalid Unicode replacement character(s) from "
                        "recognized captions. Please review the affected text.")
    if language == KANGLISH:
        warnings.append("Kanglish converts Kannada script to uppercase Latin letters. This is a pronunciation-style "
                        "transliteration, not translation; review names and spelling.")
    elif language == MIXED:
        warnings.append("Mixed Kannada + English: Whisper chooses the language for each ~30-second window, not for each "
                        f"word, so English words inside a Kannada sentence may be re-spelled. The result has "
                        f"{_script_summary(words)}. Please proof-read and edit.")
    elif detected in WEAK_LANGS:
        warnings.append(f"Accuracy for {LANGUAGES.get(detected, detected)} is lower than for English. Please proof-read every caption.")
    if language in ("kn", MIXED, KANGLISH) and model_size in ("tiny", "base"):
        warnings.append(f"The general '{model_size}' model is very weak for Kannada. For much better Kannada choose the "
                        "'kn-medium' model (Kannada-tuned), or at least 'small'.")
    if language == "auto":
        warnings.append(f"Language was auto-detected as '{detected}'. If that is wrong, pick the language yourself and run again.")
    word_level = any(w["kind"] == "word" for w in words)
    return {"language": KANGLISH if language == KANGLISH else detected, "words": words, "warnings": warnings,
            "engine": {"model": model_size, "device": device, "compute_type": compute_type, "model_was_cached": was_cached,
                       "timing": "word" if word_level else "phrase", "decode_mode": used, "segments": len(words)}}

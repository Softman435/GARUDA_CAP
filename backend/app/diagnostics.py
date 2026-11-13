"""Health check and diagnostic report.

Quick report (no model is loaded):      python -m app.diagnostics
Deep self-test (loads the model and
really runs speech recognition):        python -m app.diagnostics --deep [--model base] [--audio my.wav] [--language kn-en]

The deep test runs the three recognition settings one by one and prints the COMPLETE Python traceback of any
that fails, so the exact cause of an error such as "Speech recognition failed (TypeError)" becomes visible.
It is also available (quick report only) in the running app at http://127.0.0.1:8000/api/diagnostics.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path

from . import indic, media, transcribe
from .config import settings

SUPPORTED_PYTHON = ((3, 10), (3, 13))


def _tool_version(name: str) -> str | None:
    path = media.tool_paths().get(name)
    if not path:
        return None
    try:
        out = subprocess.run([name, "-version"], capture_output=True, text=True, timeout=15, check=False).stdout
        return out.splitlines()[0] if out else "found"
    except (OSError, subprocess.SubprocessError):
        return "found (version unreadable)"


def _model_on_disk(model: str) -> bool:
    root = settings.whisper_model_dir
    if not root or not Path(root).is_dir():
        return False
    if model == transcribe.INDIC:
        return settings.indic_available
    try:
        source, is_local = transcribe.resolve_model_source(model, root, settings.custom_model)
    except transcribe.TranscribeError:
        return False
    if is_local:
        return (Path(source) / "model.bin").is_file()
    repo_id = f"Systran/faster-whisper-{source}" if source in transcribe.MODELS else source
    snapshots = Path(root) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any((snapshot / "model.bin").is_file() for snapshot in snapshots.iterdir())


def collect(deep: bool = False, *, model: str | None = None, audio: Path | None = None, language: str = "kn-en") -> dict:
    """Structured report. `problems` lists anything that will stop captions from being generated."""
    problems: list[str] = []
    versions = transcribe.installed_versions()
    py = sys.version_info[:2]
    if not (SUPPORTED_PYTHON[0] <= py <= SUPPORTED_PYTHON[1]):
        problems.append(f"Python {py[0]}.{py[1]} is outside the supported range 3.10-3.13.")
    if versions["faster-whisper"] == "not installed":
        problems.append("faster-whisper is not installed (run scripts\\install.bat).")
    if versions["ctranslate2"] == "not installed":
        problems.append("ctranslate2 is not installed (run scripts\\install.bat).")
    tools = {name: _tool_version(name) for name in ("ffmpeg", "ffprobe")}
    for name, version in tools.items():
        if version is None:
            problems.append(f"{name} was not found on PATH (see docs\\WINDOWS_SETUP.md, step 3).")

    runtime: dict = {}
    try:
        device, compute, notes = transcribe.resolve_runtime(settings.whisper_device, settings.whisper_compute_type)
        runtime = {"device": device, "compute_type": compute, "notes": notes}
    except Exception as exc:  # noqa: BLE001
        runtime = {"error": f"{type(exc).__name__}: {exc}"}

    chosen = model or settings.whisper_default_model
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "virtual_environment": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "packages": versions, "tools": tools, "runtime": runtime,
        "settings": {"model": chosen, "device": settings.whisper_device, "compute_type": settings.whisper_compute_type,
                     "cpu_threads": settings.whisper_cpu_threads or "default", "beam_size": settings.whisper_beam_size,
                     "allow_model_download": settings.allow_model_download, "audio_normalize": settings.audio_normalize,
                     "model_dir": str(settings.whisper_model_dir), "data_dir": str(settings.data_dir),
                     "log_dir": str(settings.log_dir)},
        "model_on_disk": _model_on_disk(chosen), "model_loaded_in_memory": transcribe.loaded_models(),
        "languages": list(transcribe.LANGUAGES), "problems": problems,
        "indic": {"model_file": bool(settings.indic_available), "runner_set": bool(settings.indic_runner)},
    }
    if deep:
        report["deep_test"] = deep_test(chosen, audio=audio, language=language)
        if not report["deep_test"]["ok"]:
            problems.append("The deep recognition test failed (see 'deep_test').")
    return report


def deep_test(model_size: str, *, audio: Path | None, language: str) -> dict:
    """Load the model and run each recognition setting separately. Never raises."""
    steps: list[dict] = []
    result = {"ok": False, "model": model_size, "steps": steps}
    work = Path(tempfile.mkdtemp(prefix="garuda-diag-"))
    try:
        wav = audio
        if wav is None:  # a 4-second tone: exercises the whole engine; it contains no speech by design
            wav = work / "tone.wav"
            media.require_tools()
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True, timeout=60)
            result["note"] = "No audio file was given, so a test tone was used. 'No speech detected' is the CORRECT result for it."
        t0 = __import__("time").time()
        try:
            device, compute, _ = transcribe.resolve_runtime(settings.whisper_device, settings.whisper_compute_type)
            model_source, _ = transcribe.resolve_model_source(model_size, settings.whisper_model_dir, settings.custom_model)
            model, cached = transcribe.get_model(
                model_source, device=device, compute_type=compute, model_dir=settings.whisper_model_dir,
                allow_download=settings.allow_model_download, cpu_threads=settings.whisper_cpu_threads, label=model_size,
            )
            steps.append({"step": "load model", "ok": True, "seconds": round(__import__("time").time() - t0, 1), "cached": cached})
        except Exception as exc:  # noqa: BLE001
            steps.append({"step": "load model", "ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            return result
        for word_timing, vad in ((False, False), (False, True), (True, True)):
            name = f"decode: word timing={'on' if word_timing else 'off'}, voice filter={'on' if vad else 'off'}"
            try:
                options, _ = transcribe.plan_options(language, beam_size=settings.whisper_beam_size, word_timestamps=word_timing,
                                                     vad_filter=vad, accepts=set(__import__("inspect").signature(model.transcribe).parameters),
                                                     kannada_only_model=model_size in transcribe.KANNADA_TUNED)
                segments, info = model.transcribe(transcribe.load_audio_array(wav), **options)
                words, stats = transcribe.consume_segments(segments, total=max(float(getattr(info, "duration", 0) or 0), 0.001),
                                                           want_words=word_timing, cancel=threading.Event(), progress=lambda p: None)
                steps.append({"step": name, "ok": True, "captions": len(words), "language": getattr(info, "language", None),
                              "sample": " ".join(w["text"] for w in words[:12])})
            except Exception as exc:  # noqa: BLE001
                steps.append({"step": name, "ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        result["ok"] = all(s["ok"] for s in steps)
    except Exception as exc:  # noqa: BLE001
        steps.append({"step": "prepare", "ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    return result


def _print(report: dict) -> None:
    def line(label: str, value) -> None:
        print(f"  {label:<26} {value}")

    print("\n=== GARUDA CAPTIONS AI - diagnostic report ===")
    line("Python", f"{report['python']}  (virtual environment: {'yes' if report['virtual_environment'] else 'NO'})")
    line("System", report["platform"])
    print("\nSpeech packages:")
    for name, version in report["packages"].items():
        line(name, version)
    print("\nMedia tools:")
    for name, version in report["tools"].items():
        line(name, version or "NOT FOUND")
    print("\nConfiguration (from .env / defaults):")
    for key, value in report["settings"].items():
        line(key, value)
    line("runtime in use", report["runtime"])
    line("model already downloaded", "yes" if report["model_on_disk"] else "no (it is downloaded on first use)")
    if "deep_test" in report:
        deep = report["deep_test"]
        print(f"\nDeep recognition test (model '{deep['model']}'):")
        if deep.get("note"):
            print("  NOTE:", deep["note"])
        for step in deep["steps"]:
            print(f"  [{'OK' if step['ok'] else 'FAILED'}] {step['step']}", {k: v for k, v in step.items() if k not in ("step", "ok", "traceback", "error")} or "")
            if not step["ok"]:
                print("      error:", step["error"])
                print("      " + step["traceback"].replace("\n", "\n      "))
    print("\nProblems found:" if report["problems"] else "\nNo problems found.")
    for problem in report["problems"]:
        print("  -", problem)
    print("\nFull technical log:", Path(report["settings"]["log_dir"]) / "garuda.log")


def _indic_report() -> int:
    """Check the IndicConformer setup step by step and print what is missing."""
    print("\n=== IndicConformer (.nemo) check ===")
    cfg = settings.indic_config()
    if cfg is None:
        print("  model file: NOT FOUND. Copy the .nemo file into data\\models\\indic (or set INDIC_MODEL in .env).")
        return 1
    print(f"  model file : {cfg.model_path.name} ({cfg.model_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  runner     : {' '.join(cfg.runner) or 'NOT SET - run scripts\\setup_indic.bat'}")
    print(f"  decoder    : {cfg.decoder}")
    report = indic.check_runtime(cfg, load_model=bool(cfg.runner))
    for key in ("python", "torch", "nemo", "ai4bharat_fork", "model_loaded", "model_class"):
        if key in report:
            print(f"  {key:<14} {report[key]}")
    for problem in report.get("problems", []):
        print("  PROBLEM:", problem)
    print("\nResult:", "READY" if report.get("ok") else "NOT READY")
    return 0 if report.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GARUDA CAPTIONS AI diagnostics")
    parser.add_argument("--deep", action="store_true", help="load the model and run real speech recognition")
    parser.add_argument("--model", default=None, choices=sorted({*transcribe.model_choices(settings.whisper_model_dir, settings.custom_model), "kn-small"}))
    parser.add_argument("--audio", type=Path, default=None, help="a WAV/MP3/MP4 file to transcribe in the deep test")
    parser.add_argument("--language", default="kn-en", choices=sorted(transcribe.LANGUAGES))
    parser.add_argument("--indic", action="store_true", help="check the IndicConformer (.nemo) setup and try to load the model")
    args = parser.parse_args(argv)
    if args.indic:
        return _indic_report()
    wav = args.audio
    if wav is not None:
        if not wav.is_file():
            print(f"\nERROR: the file was not found:\n  {wav}\n"
                  "Check the path. In File Explorer, right-click your video and choose 'Copy as path', then paste it "
                  "after --audio (keep the quotes).")
            return 2
        if True:      # always convert to the exact WAV format the engine expects (16 kHz mono)
            converted = Path(tempfile.mkdtemp(prefix="garuda-diag-in-")) / "input.wav"
            try:
                media.extract_audio(wav, converted, normalize=settings.audio_normalize)
            except media.MediaError as exc:
                print(f"\nERROR: {exc.message}\nThe file may be damaged, not a video, or have no audio track. "
                      "FFmpeg's own message is in data\\logs\\garuda.log.")
                return 2
            wav = converted
    report = collect(deep=args.deep, model=args.model, audio=wav, language=args.language)
    _print(report)
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())

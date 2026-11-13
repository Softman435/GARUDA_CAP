"""Runs AI4Bharat IndicConformer (.nemo) speech recognition. Started by the app; not meant to be run by hand.

It must run in a Python that has AI4Bharat's NeMo fork installed (scripts/setup_indic.sh inside WSL2 Ubuntu does that).
It uses only the calls that AI4Bharat documents for these models:

    model = ASRModel.restore_from(path)
    model.cur_decoder = "rnnt"            # or "ctc"
    model.transcribe([wav], batch_size=1, language_id="kn")

Modes
  --manifest M --output O   transcribe every chunk listed in M and write the result JSON to O
  --check [--model F]       print one JSON line describing this environment (and try to load model F)

Protocol: progress lines "PROGRESS done total" on stdout; the result (or an {"error": ...} object) goes to the output file.
This file imports nothing from the app, so it can live in any Python environment.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import traceback
from typing import Any


def _unwrap(output: Any) -> Any:
    """NeMo versions return a list of strings, a list of Hypothesis objects, or a tuple (best, all). Take the first item."""
    while isinstance(output, (list, tuple)) and output:
        output = output[0]
    return output


def normalise(output: Any) -> tuple[str, list[dict] | None]:
    """(text, word timings or None) from whatever transcribe() returned."""
    item = _unwrap(output)
    if isinstance(item, str):
        return " ".join(item.split()), None
    text = getattr(item, "text", None)
    text = " ".join(text.split()) if isinstance(text, str) else ""
    words = None
    stamp = getattr(item, "timestamp", None) or getattr(item, "timestep", None)
    if isinstance(stamp, dict) and isinstance(stamp.get("word"), list):
        words = [{"text": w.get("word", w.get("char", "")), "start": w.get("start"), "end": w.get("end")}
                 for w in stamp["word"] if isinstance(w, dict)]
    return text, words


def check(model_path: str | None) -> dict:
    """Describe this environment. The most important fact: is this AI4Bharat's NeMo fork (needed) or stock NeMo?"""
    report: dict[str, Any] = {"ok": False, "python": sys.version.split()[0], "problems": []}
    try:
        import torch
        report["torch"] = torch.__version__
    except Exception as exc:  # noqa: BLE001
        report["problems"].append(f"PyTorch is not usable: {type(exc).__name__}: {exc}")
        return report
    try:
        import nemo
        from nemo.collections.asr.modules import RNNTJoint
        report["nemo"] = getattr(nemo, "__version__", "unknown")
        parameters = inspect.signature(RNNTJoint.__init__).parameters
        report["ai4bharat_fork"] = "language_keys" in parameters and "multilingual" in parameters
        if not report["ai4bharat_fork"]:
            report["problems"].append("This is stock NVIDIA NeMo. IndicConformer needs AI4Bharat's NeMo fork "
                                      "(branch nemo-v2): run scripts\\setup_indic.bat again.")
    except Exception as exc:  # noqa: BLE001
        report["problems"].append(f"NeMo is not usable: {type(exc).__name__}: {exc}")
        return report
    if model_path and report["ai4bharat_fork"]:
        try:
            from nemo.collections.asr.models import ASRModel
            model = ASRModel.restore_from(restore_path=model_path, map_location="cpu")
            report["model_loaded"] = True
            report["model_class"] = type(model).__name__
        except Exception as exc:  # noqa: BLE001
            report["model_loaded"] = False
            report["problems"].append(f"The model file could not be loaded: {type(exc).__name__}: {str(exc)[:300]}")
    report["ok"] = not report["problems"]
    return report


def transcribe_chunks(manifest: dict, progress=lambda done, total: None) -> dict:
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.restore_from(restore_path=manifest["model"], map_location="cpu")
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    model.cur_decoder = manifest.get("decoder", "rnnt")
    language = manifest["language"]
    use_timestamps = manifest.get("word_timestamps", True)

    chunks = manifest["chunks"]
    results = []
    for done, chunk in enumerate(chunks, 1):
        kwargs: dict[str, Any] = {"batch_size": 1, "language_id": language}
        if model.cur_decoder == "ctc":
            kwargs["logprobs"] = False
        output = None
        if use_timestamps:
            try:
                output = model.transcribe([chunk["path"]], timestamps=True, **kwargs)   # newer NeMo only
            except TypeError:
                use_timestamps = False                      # this NeMo has no word timestamps: use plain text
        if output is None:
            output = model.transcribe([chunk["path"]], **kwargs)
        text, words = normalise(output)
        results.append({"id": chunk["id"], "text": text, "words": words})
        progress(done, len(chunks))
    return {"chunks": results, "engine": {"decoder": model.cur_decoder, "word_timestamps": use_timestamps,
                                          "model_class": type(model).__name__}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args(argv)

    if args.check:
        print(json.dumps(check(args.model), ensure_ascii=False))
        return 0
    if not args.manifest or not args.output:
        parser.error("--manifest and --output are required")
    try:
        with open(args.manifest, encoding="utf-8") as fh:
            manifest = json.load(fh)
        result = transcribe_chunks(manifest, progress=lambda d, t: print(f"PROGRESS {d} {t}", flush=True))
        code = 0
    except BaseException as exc:  # noqa: BLE001 - everything must reach the app as a structured error
        traceback.print_exc()
        result = {"error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
        code = 3
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)
    return code


if __name__ == "__main__":
    sys.exit(main())

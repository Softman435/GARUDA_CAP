"""Convert a Hugging Face Whisper model (for example a Kannada fine-tune) into the CTranslate2 format that
faster-whisper loads, and save it in data/models/<name>.

Normally you run it through  scripts\\convert_model.bat  (that creates a separate environment with PyTorch,
which this converter needs but the app itself does not).

  scripts\\convert_model.bat vasista22/whisper-kannada-small kn-small-ct2 small

Arguments: SOURCE (Hugging Face id or a local folder), NAME (folder to create in data/models), SIZE (the OpenAI model
this was fine-tuned from: tiny/base/small/medium/large-v3; used only to fetch a missing tokenizer file).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
NEEDED_FILES = ("tokenizer.json", "preprocessor_config.json")


def fetch_missing_files(output: Path, source: str, size: str, *, downloader=None, log=print) -> list[str]:
    """faster-whisper needs tokenizer.json and preprocessor_config.json next to the weights. Fine-tune repositories often
    leave them out. Take each from the source if present, else from the original OpenAI model it was tuned from
    (the tokenizer is identical across a Whisper size)."""
    if downloader is None:
        from huggingface_hub import hf_hub_download as downloader   # imported lazily: only the converter needs it
    copied: list[str] = []
    for name in NEEDED_FILES:
        if (output / name).is_file():
            continue
        local = Path(source) / name
        if Path(source).is_dir() and local.is_file():
            shutil.copy2(local, output / name)
            copied.append(f"{name} (from the source folder)")
            continue
        for repo in ([source] if not Path(source).is_dir() else []) + [f"openai/whisper-{size}"]:
            try:
                path = downloader(repo_id=repo, filename=name)
            except Exception as exc:  # noqa: BLE001 - try the next place
                log(f"  {name} is not in {repo} ({type(exc).__name__}); trying the next source")
                continue
            shutil.copy2(path, output / name)
            copied.append(f"{name} (from {repo})")
            break
        else:
            raise RuntimeError(f"Could not find {name} in the source or in openai/whisper-{size}.")
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("name")
    parser.add_argument("size", nargs="?", default="small", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--quantization", default="int8", choices=["int8", "int8_float16", "float16", "float32"],
                        help="int8 = smallest and best for a CPU")
    parser.add_argument("--force", action="store_true", help="replace an existing folder")
    args = parser.parse_args(argv)

    output = REPO_DIR / "data" / "models" / args.name
    if output.exists() and not args.force:
        print(f"The folder already exists: {output}\nAdd --force to replace it.")
        return 2
    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        print("ctranslate2 / transformers / torch are missing. Run this through scripts\\convert_model.bat, "
              "which installs them in a separate environment.")
        return 1

    print(f"Converting {args.source} -> {output}  (quantization {args.quantization}). This can take several minutes.")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        TransformersConverter(args.source).convert(str(output), quantization=args.quantization, force=args.force)
        for line in fetch_missing_files(output, args.source, args.size):
            print("  added", line)
    except Exception as exc:  # noqa: BLE001
        print(f"\nConversion FAILED: {type(exc).__name__}: {exc}")
        shutil.rmtree(output, ignore_errors=True)
        return 1
    print(f"\nDone. Model saved in: {output}")
    if args.name == "kn-small-ct2":
        print("Restart the app; 'kn-small' now appears in the model list.")
    else:
        print(f"To use it, add this line to the .env file:   CUSTOM_MODEL=data\\models\\{args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

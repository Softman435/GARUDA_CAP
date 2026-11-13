"""Measure how accurate the captions are on YOUR data, and compare models.

Put audio/video files and the correct text for each in one folder, in either layout:
  A) pairs:      clip1.wav + clip1.txt, clip2.mp4 + clip2.txt, ...   (the .txt holds exactly what is said)
  B) metadata:   a metadata.csv with a file column (file_name / audio / path / file) and a text column
                 (text / transcription / sentence / transcript / label)  - the usual Hugging Face "audiofolder" layout

Run (from the project folder):
  scripts\\evaluate.bat --data "C:\\my-dataset" --models base small kn-medium --language kn --limit 20

It runs the same recognition code as the app and prints, for every model:
  WER = word error rate   (share of words that are wrong; 0% perfect, lower is better)
  CER = character error rate (more forgiving for Kannada, whose long joined words make WER harsh)
  speed = processing time divided by audio length
Results are also saved in data\\evaluation\\ as a CSV file.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path

from . import media, transcribe
from .config import settings

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".mov", ".mkv", ".webm", ".avi"}
FILE_COLUMNS = ("file_name", "audio", "path", "file", "filename")
TEXT_COLUMNS = ("text", "transcription", "sentence", "transcript", "label", "normalized_text")


# ----------------------------------------------------------------------------------------------------- metrics
def normalize(text: str) -> str:
    """Compare only the words: Unicode-normalised, case-folded, punctuation (including the Kannada/Hindi danda) removed."""
    text = unicodedata.normalize("NFC", str(text)).casefold()
    kept = []
    for ch in text:
        category = unicodedata.category(ch)
        kept.append(" " if category.startswith(("P", "S", "Z", "C")) else ch)   # punctuation, symbols, spaces, controls
    return " ".join("".join(kept).split())


def edit_distance(a: list, b: list) -> int:
    """Smallest number of insertions, deletions and substitutions that turns `a` into `b`."""
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> tuple[int, int]:
    """(errors, reference word count)."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    return edit_distance(ref, hyp), len(ref)


def cer(reference: str, hypothesis: str) -> tuple[int, int]:
    """(errors, reference character count), ignoring spaces."""
    ref, hyp = list(normalize(reference).replace(" ", "")), list(normalize(hypothesis).replace(" ", ""))
    return edit_distance(ref, hyp), len(ref)


# ----------------------------------------------------------------------------------------------------- dataset
def find_pairs(folder: Path) -> list[tuple[Path, str]]:
    """List of (audio file, correct text). Raises ValueError with a helpful message if nothing usable is found."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"The dataset folder was not found: {folder}")
    pairs: list[tuple[Path, str]] = []
    metadata = next(iter(sorted(folder.rglob("metadata.csv"))), None)
    if metadata is not None:
        with open(metadata, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        columns = {name.strip().lower(): name for name in (rows[0].keys() if rows else [])}
        file_col = next((columns[c] for c in FILE_COLUMNS if c in columns), None)
        text_col = next((columns[c] for c in TEXT_COLUMNS if c in columns), None)
        if not file_col or not text_col:
            raise ValueError(f"{metadata.name} needs a file column {FILE_COLUMNS} and a text column {TEXT_COLUMNS}; "
                             f"found: {list(columns)}")
        for row in rows:
            audio = (metadata.parent / row[file_col].strip())
            if audio.is_file() and (row[text_col] or "").strip():
                pairs.append((audio, row[text_col].strip()))
    else:
        for audio in sorted(folder.rglob("*")):
            if audio.suffix.lower() in AUDIO_EXTENSIONS and audio.with_suffix(".txt").is_file():
                text = audio.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
                if text:
                    pairs.append((audio, text))
    if not pairs:
        raise ValueError("No audio+text pairs were found. Use clip.wav + clip.txt pairs, or a metadata.csv file "
                         "(see the top of backend/app/evaluate.py).")
    return pairs


# ----------------------------------------------------------------------------------------------------- run
def evaluate_model(pairs: list[tuple[Path, str]], model: str, language: str, *, work: Path, log=print) -> dict:
    """Transcribe every pair with one model and score it."""
    wer_err = wer_len = cer_err = cer_len = failed = 0
    audio_seconds = elapsed = 0.0
    rows = []
    for index, (audio, reference) in enumerate(pairs, 1):
        wav = work / f"{model}-{index}.wav"
        try:
            media.extract_audio(audio, wav, normalize=settings.audio_normalize)
        except media.MediaError as exc:
            log(f"  [{index}/{len(pairs)}] SKIPPED {audio.name}: {exc.message}")
            failed += 1
            continue
        seconds = transcribe.audio_seconds(wav) or 0.0
        started = time.time()
        try:
            result = transcribe.transcribe(
                wav, language=language, model_size=model, device=settings.whisper_device,
                compute_type=settings.whisper_compute_type, model_dir=settings.whisper_model_dir,
                allow_download=settings.allow_model_download, on_progress=lambda p: None, cancel=threading.Event(),
                cpu_threads=settings.whisper_cpu_threads, beam_size=settings.whisper_beam_size,
                custom_model=settings.custom_model or None, vocabulary=settings.load_vocabulary(),
                indic=settings.indic_config())
            hypothesis = " ".join(w["text"] for w in result["words"])
            note = ""
        except transcribe.TranscribeError as exc:
            if exc.code in ("model_missing", "model_download_failed", "model_load_failed", "whisper_missing",
                            "whisper_import_failed", "bad_model", "bad_language", "indic_model_missing",
                            "indic_runner_missing", "indic_worker_failed", "indic_timeout"):
                raise                                   # the model itself is unusable: stop, do not score
            hypothesis, note = "", exc.code             # e.g. no speech found: counts as every word missing
        elapsed += time.time() - started
        audio_seconds += seconds
        we, wn = wer(reference, hypothesis)
        ce, cn = cer(reference, hypothesis)
        wer_err, wer_len, cer_err, cer_len = wer_err + we, wer_len + wn, cer_err + ce, cer_len + cn
        rows.append({"model": model, "file": audio.name, "wer": round(we / wn, 4) if wn else None,
                     "cer": round(ce / cn, 4) if cn else None, "reference": reference, "hypothesis": hypothesis, "note": note})
        log(f"  [{index}/{len(pairs)}] {audio.name}: WER {we / wn:.0%}  CER {ce / cn:.0%}" if wn and cn else f"  [{index}] {audio.name}")
    return {"model": model, "files": len(rows), "skipped": failed,
            "wer": wer_err / wer_len if wer_len else None, "cer": cer_err / cer_len if cer_len else None,
            "speed": elapsed / audio_seconds if audio_seconds else None, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure caption accuracy (WER/CER) on your own audio + text.")
    parser.add_argument("--data", required=True, type=Path, help="folder with audio files and their correct text")
    parser.add_argument("--models", nargs="+", default=[settings.whisper_default_model], help="e.g. base small kn-medium")
    parser.add_argument("--language", default="kn", choices=sorted(transcribe.LANGUAGES))
    parser.add_argument("--limit", type=int, default=0, help="use only the first N files (0 = all)")
    args = parser.parse_args(argv)
    try:
        pairs = find_pairs(args.data)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        return 2
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"\nEvaluating {len(pairs)} file(s), language '{args.language}', models: {', '.join(args.models)}")
    results, all_rows = [], []
    work = Path(tempfile.mkdtemp(prefix="garuda-eval-"))
    try:
        for model in args.models:
            print(f"\n=== model: {model} ===")
            try:
                result = evaluate_model(pairs, model, args.language, work=work)
            except transcribe.TranscribeError as exc:
                print(f"  Cannot use this model: {exc.message}")
                continue
            results.append(result)
            all_rows.extend(result["rows"])
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    if not results:
        return 1
    print("\n================ SUMMARY (lower is better) ================")
    print(f"{'model':<14}{'files':>6}{'WER':>9}{'CER':>9}{'time/audio':>13}")
    for r in results:
        print(f"{r['model']:<14}{r['files']:>6}{r['wer']:>9.1%}{r['cer']:>9.1%}"
              f"{(str(round(r['speed'], 1)) + 'x') if r['speed'] is not None else '-':>13}")
    out_dir = settings.data_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"evaluation-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "file", "wer", "cer", "note", "reference", "hypothesis"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPer-file results (open in Excel): {out}")
    print("Tip: WER is harsh for Kannada because words are long. Compare models with the SAME files and watch the trend.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

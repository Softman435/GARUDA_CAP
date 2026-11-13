"""Turn a folder of audio files + transcripts into the standard "audiofolder" training format (metadata.csv).

Layouts understood:
  1) clip.wav + clip.txt  (same name, the .txt contains exactly what is said)
  2) a CSV/TSV file given with --table, with a path column and a text column

  python make_metadata.py --data my_dataset
  python make_metadata.py --data my_dataset --table transcripts.csv --path-col audio --text-col sentence

Result: my_dataset/metadata.csv with the columns  file_name,text  (UTF-8). Clips longer than 30 seconds are reported:
Whisper can only learn from clips up to 30 seconds, so cut longer ones.
"""
from __future__ import annotations

import argparse
import csv
import sys
import wave
from pathlib import Path

AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


def wav_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as fh:
            return fh.getnframes() / fh.getframerate()
    except (wave.Error, EOFError, OSError, ZeroDivisionError):
        return None


def collect(data: Path, table: Path | None = None, path_col: str = "path", text_col: str = "text") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if table:
        with open(table, encoding="utf-8-sig", newline="") as fh:
            delimiter = "\t" if table.suffix.lower() == ".tsv" else ","
            for record in csv.DictReader(fh, delimiter=delimiter):
                audio, text = (record.get(path_col) or "").strip(), (record.get(text_col) or "").strip()
                if audio and text and (data / audio).is_file():
                    rows.append((Path(audio).as_posix(), text))
    else:
        for audio in sorted(data.rglob("*")):
            if audio.suffix.lower() in AUDIO and audio.with_suffix(".txt").is_file():
                text = audio.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
                if text:
                    rows.append((audio.relative_to(data).as_posix(), text))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--text-col", default="text")
    args = parser.parse_args(argv)
    if not args.data.is_dir():
        print(f"Folder not found: {args.data}")
        return 2
    rows = collect(args.data, args.table, args.path_col, args.text_col)
    if not rows:
        print("No usable audio + text pairs were found.")
        return 1
    with open(args.data / "metadata.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_name", "text"])
        writer.writerows(rows)
    long_clips = [name for name, _ in rows if (s := wav_seconds(args.data / name)) is not None and s > 30]
    print(f"Wrote {len(rows)} rows to {args.data / 'metadata.csv'}")
    if long_clips:
        print(f"WARNING: {len(long_clips)} clip(s) are longer than 30 seconds (Whisper cannot train on them). First: {long_clips[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

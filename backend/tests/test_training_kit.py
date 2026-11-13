"""The metadata helper of the Colab training kit (training/make_metadata.py)."""
import csv
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
import make_metadata as mm  # noqa: E402


def wav(path, seconds):
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1); fh.setsampwidth(2); fh.setframerate(16000)
        fh.writeframes(b"\x00\x00" * int(16000 * seconds))


class MakeMetadata(unittest.TestCase):
    def test_pairs_are_written_as_utf8_metadata_csv_and_long_clips_are_reported(self):
        d = Path(tempfile.mkdtemp())
        wav(d / "a.wav", 1); (d / "a.txt").write_text("ನಮಸ್ಕಾರ friends", encoding="utf-8")
        wav(d / "long.wav", 31); (d / "long.txt").write_text("x", encoding="utf-8")
        wav(d / "no_text.wav", 1)
        self.assertEqual(mm.main(["--data", str(d)]), 0)
        rows = list(csv.reader(open(d / "metadata.csv", encoding="utf-8")))
        self.assertEqual(rows[0], ["file_name", "text"])
        self.assertIn(["a.wav", "ನಮಸ್ಕಾರ friends"], rows)
        self.assertEqual(len(rows), 3, "the clip without text is left out")
        self.assertEqual(mm.wav_seconds(d / "long.wav"), 31.0)

    def test_table_layout_and_errors(self):
        d = Path(tempfile.mkdtemp())
        wav(d / "x.wav", 1)
        (d / "t.csv").write_text("audio,sentence\nx.wav,ಇದು ಪರೀಕ್ಷೆ\nmissing.wav,abc\n", encoding="utf-8")
        self.assertEqual(mm.collect(d, d / "t.csv", "audio", "sentence"), [("x.wav", "ಇದು ಪರೀಕ್ಷೆ")])
        self.assertEqual(mm.main(["--data", str(d / "nope")]), 2)
        self.assertEqual(mm.main(["--data", str(Path(tempfile.mkdtemp()))]), 1)


if __name__ == "__main__":
    unittest.main()

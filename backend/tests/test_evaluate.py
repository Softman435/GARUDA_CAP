"""Tests for the accuracy tool (app/evaluate.py). Metric expectations are calculated by hand."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_faster_whisper as fw  # noqa: E402
from app import evaluate as ev  # noqa: E402
from app import transcribe as tr  # noqa: E402
from app.config import settings  # noqa: E402
from test_transcribe import make_wav  # noqa: E402


class Metrics(unittest.TestCase):
    def test_wer_by_hand(self):
        self.assertEqual(ev.wer("the cat sat", "the cat sat"), (0, 3))
        self.assertEqual(ev.wer("the cat sat", "the dog sat"), (1, 3))            # 1 substitution
        self.assertEqual(ev.wer("the cat sat", "the sat"), (1, 3))                # 1 deletion
        self.assertEqual(ev.wer("the cat", "the big cat"), (1, 2))                # 1 insertion
        self.assertEqual(ev.wer("a b c d", ""), (4, 4))                           # nothing recognised
        self.assertEqual(ev.wer("", "abc"), (1, 0))                         # one inserted word

    def test_cer_by_hand(self):
        self.assertEqual(ev.cer("abcd", "abxd"), (1, 4))
        self.assertEqual(ev.cer("ab cd", "abcd"), (0, 4), "spaces are ignored")

    def test_punctuation_case_and_unicode_are_normalised(self):
        self.assertEqual(ev.normalize("Hello,  WORLD!"), "hello world")
        self.assertEqual(ev.normalize("ನಮಸ್ಕಾರ।  ಇವತ್ತು, friends?"), "ನಮಸ್ಕಾರ ಇವತ್ತು friends")
        self.assertEqual(ev.wer("ನಮಸ್ಕಾರ friends.", "ನಮಸ್ಕಾರ Friends"), (0, 2))
        composed, decomposed = "\u00e9", "e\u0301"
        self.assertEqual(ev.normalize(composed), ev.normalize(decomposed))

    def test_kannada_word_differences_are_counted(self):
        self.assertEqual(ev.wer("ನಮಸ್ಕಾರ ಇವತ್ತು ನಾವು", "ನಮಸ್ಕಾರ ಇವತ್ತು ನೀವು"), (1, 3))
        errors, total = ev.cer("ಇವತ್ತು", "ಇವತು")
        self.assertGreater(errors, 0)
        self.assertLess(errors / total, 1)


class Dataset(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_pairs_layout(self):
        make_wav(self.dir / "a.wav")
        (self.dir / "a.txt").write_text("ನಮಸ್ಕಾರ friends", encoding="utf-8")
        make_wav(self.dir / "orphan.wav")                        # no text: ignored
        (self.dir / "empty.txt").write_text("", encoding="utf-8")
        pairs = ev.find_pairs(self.dir)
        self.assertEqual([(p.name, t) for p, t in pairs], [("a.wav", "ನಮಸ್ಕಾರ friends")])

    def test_metadata_csv_layout_with_common_column_names(self):
        (self.dir / "clips").mkdir()
        make_wav(self.dir / "clips" / "x.wav")
        make_wav(self.dir / "clips" / "y.wav")
        with open(self.dir / "metadata.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file_name", "sentence"])
            writer.writerow(["clips/x.wav", "ಇದು ಪರೀಕ್ಷೆ"])
            writer.writerow(["clips/y.wav", ""])               # empty text: skipped
            writer.writerow(["clips/missing.wav", "text"])     # missing file: skipped
        pairs = ev.find_pairs(self.dir)
        self.assertEqual([(p.name, t) for p, t in pairs], [("x.wav", "ಇದು ಪರೀಕ್ಷೆ")])

    def test_clear_errors(self):
        with self.assertRaises(ValueError):
            ev.find_pairs(self.dir / "nope")
        with self.assertRaises(ValueError) as ctx:
            ev.find_pairs(self.dir)
        self.assertIn("No audio+text pairs", str(ctx.exception))
        (self.dir / "metadata.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            ev.find_pairs(self.dir)
        self.assertIn("needs a file column", str(ctx.exception))


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.out = Path(tempfile.mkdtemp())
        self._old = (settings.data_dir, settings.whisper_model_dir)
        settings.data_dir = self.out
        settings.whisper_model_dir = self.out / "models"
        tr.release_models()

    def tearDown(self):
        settings.data_dir, settings.whisper_model_dir = self._old
        tr.release_models()

    def test_whole_tool_scores_a_model_and_writes_a_csv(self):
        make_wav(self.dir / "a.wav", 2.0)
        (self.dir / "a.txt").write_text("ನಮಸ್ಕಾರ friends ಇವತ್ತು ಬನ್ನಿ", encoding="utf-8")   # the model will miss "ಬನ್ನಿ"

        def behaviour(options):
            return [fw.seg("x", 0.1, 1.4, [fw.Word(0.1, 0.5, " ನಮಸ್ಕಾರ", 0.9), fw.Word(0.6, 0.9, " friends", 0.9),
                                            fw.Word(1.0, 1.4, " ಇವತ್ತು", 0.9)])], fw.info("kn", 2.0)
        with fw.install(behaviour):
            code = ev.main(["--data", str(self.dir), "--models", "small", "--language", "kn-en"])
        self.assertEqual(code, 0)
        files = list((self.out / "evaluation").glob("*.csv"))
        self.assertEqual(len(files), 1)
        rows = list(csv.DictReader(open(files[0], encoding="utf-8-sig")))
        self.assertEqual(rows[0]["hypothesis"], "ನಮಸ್ಕಾರ friends ಇವತ್ತು")
        self.assertAlmostEqual(float(rows[0]["wer"]), 0.25, places=4)           # 1 of 4 words missing

    def test_no_speech_counts_as_all_words_missing(self):
        make_wav(self.dir / "a.wav", 2.0)
        (self.dir / "a.txt").write_text("one two three", encoding="utf-8")
        with fw.install(lambda o: ([], fw.info("en", 2.0))):
            result = ev.evaluate_model(ev.find_pairs(self.dir), "base", "en", work=self.out, log=lambda *_: None)
        self.assertEqual(result["wer"], 1.0)
        self.assertEqual(result["rows"][0]["note"], "no_speech")

    def test_an_unusable_model_stops_instead_of_giving_fake_scores(self):
        make_wav(self.dir / "a.wav", 2.0)
        (self.dir / "a.txt").write_text("one two", encoding="utf-8")
        with fw.install(init_error=ValueError("bad weights")):
            with self.assertLogs("garuda.transcribe", level="ERROR"):
                code = ev.main(["--data", str(self.dir), "--models", "base", "--language", "en"])
        self.assertEqual(code, 1)
        self.assertEqual(list((self.out / "evaluation").glob("*.csv")) if (self.out / "evaluation").exists() else [], [])


if __name__ == "__main__":
    unittest.main()

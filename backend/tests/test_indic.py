"""IndicConformer bridge (app/indic.py + tools/indic_worker.py).

The REAL NeMo model cannot run in the test environment. These tests run the real bridge and the real worker script
against a small STAND-IN for the `nemo` package (made in a temp folder), so the chunking, protocol, progress, errors,
timeout and cancel handling are tested. Recognition quality of the real model is NOT tested here.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import indic_worker as worker  # noqa: E402
from app import indic  # noqa: E402
from app import transcribe as tr  # noqa: E402
from app.config import IndicConfig  # noqa: E402

WORKER = Path(__file__).resolve().parents[1] / "tools" / "indic_worker.py"


def make_wav(path: Path, spans, total: float) -> Path:
    """16 kHz mono WAV: a 300 Hz tone inside each (start, end) span, silence elsewhere."""
    import math
    rate = 16000
    frames = bytearray()
    for n in range(int(total * rate)):
        t = n / rate
        loud = any(a <= t < b for a, b in spans)
        value = int(12000 * math.sin(2 * math.pi * 300 * t)) if loud else 0
        frames += value.to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(bytes(frames))
    return path


# =====================================================================================================
class ChunkPlanning(unittest.TestCase):
    def test_no_pauses_short_audio_is_one_chunk(self):
        self.assertEqual(indic.plan_chunks([], 10.0, pad=0), [(0.0, 10.0)])

    def test_pieces_are_joined_until_the_chunk_limit(self):
        silences = [(3.0, 3.5), (8.0, 8.6)]           # speech: 0-3, 3.5-8, 8.6-12
        self.assertEqual(indic.plan_chunks(silences, 12.0, max_chunk=25, pad=0), [(0.0, 12.0)])
        self.assertEqual(indic.plan_chunks(silences, 12.0, max_chunk=6, pad=0), [(0.0, 3.0), (3.5, 8.0), (8.6, 12.0)])

    def test_padding_stays_inside_the_pauses_and_inside_the_recording(self):
        silences = [(3.0, 3.5), (8.0, 8.6)]
        chunks = indic.plan_chunks(silences, 12.0, max_chunk=6, pad=0.1)
        self.assertEqual(chunks, [(0.0, 3.1), (3.4, 8.1), (8.5, 12.0)])
        for (a, b), (c, d) in zip(chunks, chunks[1:]):
            self.assertLessEqual(b, c, "chunks must not overlap")

    def test_one_long_speech_stretch_is_split_into_equal_parts(self):
        chunks = indic.plan_chunks([], 50.0, max_chunk=20, pad=0)
        self.assertEqual(len(chunks), 3)
        self.assertAlmostEqual(chunks[0][1], 50 / 3, places=2)
        self.assertEqual((chunks[0][0], chunks[-1][1]), (0.0, 50.0))
        self.assertTrue(all(b - a <= 20 for a, b in chunks))

    def test_tiny_noises_are_not_speech_and_silence_only_gives_nothing(self):
        self.assertEqual(indic.plan_chunks([(0.0, 4.8)], 5.0, pad=0), [], "0.2 s blip is dropped")
        self.assertEqual(indic.plan_chunks([(0.0, 5.0)], 5.0, pad=0), [])
        self.assertEqual(indic.plan_chunks([], 0.0), [])

    def test_leading_and_trailing_pauses_are_left_out(self):
        self.assertEqual(indic.plan_chunks([(0.0, 2.0), (6.0, 8.0)], 8.0, pad=0), [(2.0, 6.0)])


class AudioHandling(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_silence_detection_with_real_ffmpeg(self):
        wav = make_wav(self.dir / "a.wav", [(0.0, 1.5), (3.5, 6.0)], 6.0)
        silences = indic.detect_silences(wav, total=6.0)
        self.assertEqual(len(silences), 1)
        self.assertAlmostEqual(silences[0][0], 1.5, delta=0.15)
        self.assertAlmostEqual(silences[0][1], 3.5, delta=0.15)
        chunks = indic.plan_chunks(silences, 6.0, max_chunk=3.0, pad=0)
        self.assertEqual(len(chunks), 2)

    def test_trailing_silence_runs_to_the_end(self):
        wav = make_wav(self.dir / "t.wav", [(0.0, 2.0)], 5.0)
        silences = indic.detect_silences(wav, total=5.0)
        self.assertAlmostEqual(silences[-1][1], 5.0, delta=0.05)

    def test_slicing_writes_correct_lengths(self):
        wav = make_wav(self.dir / "s.wav", [(0, 4)], 4.0)
        paths = indic.slice_wav(wav, [(0.0, 1.0), (1.5, 4.0)], self.dir / "out")
        lengths = []
        for p in paths:
            with wave.open(str(p), "rb") as fh:
                lengths.append(fh.getnframes() / fh.getframerate())
                self.assertEqual((fh.getnchannels(), fh.getsampwidth(), fh.getframerate()), (1, 2, 16000))
        self.assertEqual(lengths, [1.0, 2.5])


class Commands(unittest.TestCase):
    def test_wsl_paths(self):
        wsl = ["wsl", "-d", "Ubuntu-22.04", "--", "/home/me/venv/bin/python"]
        self.assertEqual(indic.to_runtime_path(r"C:\Users\Dell\x y\a.wav", wsl), "/mnt/c/Users/Dell/x y/a.wav")
        self.assertEqual(indic.to_runtime_path(r"D:\m.nemo", ["wsl.exe"]), "/mnt/d/m.nemo")
        self.assertEqual(indic.to_runtime_path("/already/linux", wsl), "/already/linux")
        self.assertEqual(indic.to_runtime_path(r"C:\a\b", ["python"]), r"C:\a\b", "a normal runner keeps native paths")
        self.assertTrue(indic.is_wsl(wsl) and not indic.is_wsl(["python"]) and not indic.is_wsl([]))

    def test_runner_parsing(self):
        self.assertEqual(indic.parse_runner("wsl -d Ubuntu-22.04 -- /home/me/indic-venv/bin/python"),
                         ["wsl", "-d", "Ubuntu-22.04", "--", "/home/me/indic-venv/bin/python"])
        self.assertEqual(indic.parse_runner(""), [])


# =====================================================================================================
class WorkerFunctions(unittest.TestCase):
    class Hyp:
        def __init__(self, text, words=None):
            self.text = text
            self.timestamp = {"word": words} if words is not None else {}

    def test_normalise_every_shape_nemo_versions_return(self):
        self.assertEqual(worker.normalise(["ನಮಸ್ಕಾರ   ಸ್ನೇಹಿತರೇ"]), ("ನಮಸ್ಕಾರ ಸ್ನೇಹಿತರೇ", None))
        self.assertEqual(worker.normalise((["ಒಂದು"], ["ಎರಡು"])), ("ಒಂದು", None))
        self.assertEqual(worker.normalise([self.Hyp("ಹಲೋ")]), ("ಹಲೋ", None))
        self.assertEqual(worker.normalise([[self.Hyp("ಹಲೋ")]]), ("ಹಲೋ", None))
        text, words = worker.normalise([self.Hyp("ಹಲೋ ಜಗತ್ತು", [{"word": "ಹಲೋ", "start": 0.1, "end": 0.5},
                                                              {"word": "ಜಗತ್ತು", "start": 0.6, "end": 1.0}])])
        self.assertEqual(words, [{"text": "ಹಲೋ", "start": 0.1, "end": 0.5}, {"text": "ಜಗತ್ತು", "start": 0.6, "end": 1.0}])
        self.assertEqual(worker.normalise([]), ("", None))
        self.assertEqual(worker.normalise(None), ("", None))


# =====================================================================================================
FAKE_NEMO = '''
import time, sys
class _Hyp:
    def __init__(self, text, words=None):
        self.text = text
        self.timestamp = {"word": words} if words else {}
class ASRModel:
    cur_decoder = "rnnt"
    @classmethod
    def restore_from(cls, restore_path, map_location=None):
        mode = open(restore_path, encoding="utf-8").read().strip()
        if mode == "badmodel":
            raise ValueError("RNNTJoint.__init__() got an unexpected keyword argument 'multilingual'")
        m = cls(); m.mode = mode; return m
    def eval(self): return self
    def freeze(self): pass
    def transcribe(self, paths, batch_size=1, language_id=None, timestamps=None, logprobs=None):
        if self.mode == "slow":
            time.sleep(30)
        if timestamps and self.mode == "notimestamps":
            raise TypeError("transcribe() got an unexpected keyword argument 'timestamps'")
        if self.mode == "empty":
            return [""]
        name = paths[0].replace("\\\\", "/").split("/")[-1]
        text = "ನಮಸ್ಕಾರ " + name + " " + str(language_id) + " " + self.cur_decoder
        if timestamps and self.mode == "words":
            return [_Hyp(text, [{"word": "ನಮಸ್ಕಾರ", "start": 0.2, "end": 0.7}, {"word": "ಜಗತ್ತು", "start": 0.8, "end": 99.0}])]
        return [text]
'''


class WorkerEndToEnd(unittest.TestCase):
    """Real worker script in a subprocess, with a stand-in `nemo` package."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        pkg = self.root / "fake" / "nemo" / "collections" / "asr" / "models"
        pkg.mkdir(parents=True)
        for folder in (pkg.parents[3], pkg.parents[2], pkg.parents[1], pkg.parents[0], pkg):
            (folder / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "__init__.py").write_text(FAKE_NEMO, encoding="utf-8")
        self._old = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(self.root / "fake")
        self.audio = make_wav(self.root / "audio.wav", [(0.0, 2.0), (4.0, 6.0)], 6.0)
        self.log = []

    def tearDown(self):
        if self._old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = self._old

    def cfg(self, mode, **kw):
        model = self.root / f"{mode}.nemo"
        model.write_text(mode, encoding="utf-8")
        return IndicConfig(model_path=model, runner=[sys.executable], worker_path=WORKER, chunk_seconds=3.0, **kw)

    def run_it(self, mode, language="kn", cancel=None, **kw):
        return indic.transcribe_indic(self.audio, language=language, cfg=self.cfg(mode, **kw),
                                      on_progress=self.log.append, cancel=cancel or threading.Event())

    def test_plain_text_gives_phrase_captions_with_real_chunk_times(self):
        result = self.run_it("plain")
        words = result["words"]
        self.assertEqual(len(words), 2, "two spoken stretches => two chunks")
        self.assertTrue(all(w["kind"] == "phrase" for w in words))
        self.assertIn("kn rnnt", words[0]["text"])
        self.assertTrue(words[0]["text"].startswith("ನಮಸ್ಕಾರ chunk0000.wav"))
        self.assertAlmostEqual(words[0]["start"], 0.0, delta=0.2)
        self.assertAlmostEqual(words[0]["end"], 2.1, delta=0.3)
        self.assertAlmostEqual(words[1]["start"], 3.9, delta=0.3)
        self.assertTrue(all(0 <= w["start"] < w["end"] <= 6.0 + 0.01 for w in words))
        self.assertEqual(result["engine"]["timing"], "phrase")
        self.assertTrue(any("phrase-level" in w for w in result["warnings"]))
        self.assertEqual(result["language"], "kn")
        self.assertTrue(self.log and max(self.log) <= 1.0 and self.log == sorted(self.log))

    def test_word_timestamps_are_offset_and_invalid_ones_dropped(self):
        result = self.run_it("words")
        words = result["words"]
        self.assertTrue(all(w["kind"] == "word" for w in words))
        # second word of each chunk ends at 99 s: impossible, so it is dropped; first word keeps chunk offset
        self.assertEqual([w["text"] for w in words], ["ನಮಸ್ಕಾರ", "ನಮಸ್ಕಾರ"])
        self.assertAlmostEqual(words[1]["start"], 3.9 + 0.2, delta=0.3)
        self.assertEqual(result["engine"]["timing"], "word")

    def test_a_nemo_without_timestamp_support_falls_back_to_text(self):
        result = self.run_it("notimestamps")
        self.assertEqual(len(result["words"]), 2)
        self.assertEqual(result["words"][0]["kind"], "phrase")

    def test_ctc_decoder_choice_reaches_the_model(self):
        result = self.run_it("plain", decoder="ctc")
        self.assertIn("ctc", result["words"][0]["text"])
        self.assertEqual(result["engine"]["decoder"], "ctc")

    def test_mixed_and_auto_force_kannada_with_a_warning(self):
        for language in ("kn-en", "auto"):
            result = self.run_it("plain", language=language)
            self.assertTrue(any("English" in w and "Kannada letters" in w for w in result["warnings"]))
            self.assertIn("kn rnnt", result["words"][0]["text"])

    def test_other_indian_language_code_is_passed_through(self):
        result = self.run_it("plain", language="hi")
        self.assertIn("hi rnnt", result["words"][0]["text"])

    def test_english_is_refused_with_advice(self):
        with self.assertRaises(tr.TranscribeError) as ctx:
            self.run_it("plain", language="en")
        self.assertEqual(ctx.exception.code, "bad_language")
        self.assertIn("Whisper", ctx.exception.message)

    def test_worker_error_is_reported_safely_with_a_hint_about_the_fork(self):
        with self.assertLogs("garuda.indic", level="ERROR") as logs:
            with self.assertRaises(tr.TranscribeError) as ctx:
                self.run_it("badmodel")
        self.assertEqual(ctx.exception.code, "indic_worker_failed")
        self.assertIn("ValueError", ctx.exception.message)
        self.assertIn("AI4Bharat's fork", ctx.exception.message)
        self.assertNotIn("Traceback", ctx.exception.message)
        self.assertIn("Traceback", "\n".join(logs.output), "the full traceback goes to the log")

    def test_empty_text_everywhere_is_no_speech_not_success(self):
        with self.assertRaises(tr.TranscribeError) as ctx:
            self.run_it("empty")
        self.assertEqual(ctx.exception.code, "no_speech")

    def test_silent_audio_is_no_speech_and_the_worker_is_never_started(self):
        self.audio = make_wav(self.root / "silent.wav", [], 3.0)
        with self.assertRaises(tr.TranscribeError) as ctx:
            self.run_it("plain")
        self.assertEqual(ctx.exception.code, "no_speech")

    def test_time_limit_stops_the_worker(self):
        started = time.time()
        with self.assertRaises(tr.TranscribeError) as ctx:
            indic.transcribe_indic(self.audio, language="kn", cfg=IndicConfig(
                model_path=self.cfg("slow").model_path, runner=[sys.executable], worker_path=WORKER, timeout_s=1),
                on_progress=lambda p: None, cancel=threading.Event())
        self.assertEqual(ctx.exception.code, "indic_timeout")
        self.assertLess(time.time() - started, 20)

    def test_cancel_stops_the_worker(self):
        flag = threading.Event()
        threading.Timer(1.0, flag.set).start()
        started = time.time()
        with self.assertRaises(tr.Cancelled):
            self.run_it("slow", cancel=flag)
        self.assertLess(time.time() - started, 20)

    def test_temporary_files_are_removed(self):
        before = set(self.root.glob("garuda-indic-*"))
        self.run_it("plain")
        self.assertEqual(set(self.root.glob("garuda-indic-*")), before)

    def test_missing_runner_model_and_bad_runner(self):
        cfg = self.cfg("plain")
        with self.assertRaises(tr.TranscribeError) as ctx:
            indic.transcribe_indic(self.audio, language="kn", cfg=IndicConfig(cfg.model_path, [], worker_path=WORKER),
                                   on_progress=lambda p: None, cancel=threading.Event())
        self.assertEqual(ctx.exception.code, "indic_runner_missing")
        self.assertIn("setup_indic.bat", ctx.exception.message)
        with self.assertRaises(tr.TranscribeError) as ctx:
            indic.transcribe_indic(self.audio, language="kn", cfg=None, on_progress=lambda p: None, cancel=threading.Event())
        self.assertEqual(ctx.exception.code, "indic_model_missing")
        with self.assertRaises(tr.TranscribeError) as ctx:
            indic.transcribe_indic(self.audio, language="kn", cfg=IndicConfig(cfg.model_path, ["no-such-program-xyz"], worker_path=WORKER),
                                   on_progress=lambda p: None, cancel=threading.Event())
        self.assertEqual(ctx.exception.code, "indic_runner_missing")

    def test_the_main_transcribe_function_routes_the_indic_model(self):
        result = tr.transcribe(self.audio, language="kn", model_size="indic", device="cpu", compute_type="int8",
                               model_dir=None, allow_download=False, on_progress=lambda p: None, cancel=threading.Event(),
                               indic=self.cfg("plain"))
        self.assertEqual(result["engine"]["model"], "indic")
        self.assertIn("indic", tr.model_choices(None, None, True))
        self.assertNotIn("indic", tr.model_choices(None, None, False))

    def test_check_mode_reports_stock_nemo_versus_the_ai4bharat_fork(self):
        stock = self.root / "fake" / "nemo" / "collections" / "asr" / "modules.py"
        stock.write_text("import inspect\nclass RNNTJoint:\n    def __init__(self, jointnet): pass\n", encoding="utf-8")
        (self.root / "fake" / "torch.py").write_text("__version__ = '2.0'\n", encoding="utf-8")
        (self.root / "fake" / "nemo" / "__init__.py").write_text("__version__ = '1.23'\n", encoding="utf-8")
        report = indic.check_runtime(self.cfg("plain"), load_model=False)
        self.assertFalse(report["ok"])
        self.assertFalse(report["ai4bharat_fork"])
        self.assertIn("stock NVIDIA NeMo", " ".join(report["problems"]))
        stock.write_text("class RNNTJoint:\n    def __init__(self, jointnet, language_keys=None, multilingual=False): pass\n", encoding="utf-8")
        report = indic.check_runtime(self.cfg("plain"), load_model=True)
        self.assertTrue(report["ai4bharat_fork"] and report["ok"], report)
        self.assertTrue(report["model_loaded"])
        bad = indic.check_runtime(self.cfg("badmodel"), load_model=True)
        self.assertFalse(bad["ok"])
        self.assertIn("could not be loaded", " ".join(bad["problems"]))

    def test_check_without_a_runner_is_a_clear_problem(self):
        report = indic.check_runtime(IndicConfig(self.cfg("plain").model_path, [], worker_path=WORKER))
        self.assertFalse(report["ok"])
        self.assertIn("setup_indic.bat", report["problems"][0])


if __name__ == "__main__":
    unittest.main()

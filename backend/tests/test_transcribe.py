"""Tests for the REAL `app.transcribe.transcribe()` function.

Two groups, clearly separated:
  * MOCKED-LIBRARY tests (class names ending ...Mocked): the speech library is replaced by
    tests/fake_faster_whisper.py. They prove the app's own logic: arguments, fallbacks, validation,
    error codes, logging, caching. They need no model download and run in seconds.
  * REAL-LIBRARY test (class RealLibraryEndToEnd): runs only when faster-whisper is installed AND the
    environment variable GARUDA_E2E=1 is set. It loads a real model (download on first use).

Run:   python -m unittest discover -s backend/tests -p "test_transcribe.py" -v
"""
import math
import os
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_faster_whisper as fw  # noqa: E402
from app import captions as cap  # noqa: E402
from app import transcribe as tr  # noqa: E402

NOT_CANCELLED = threading.Event()


def make_wav(path: Path, seconds: float = 2.0) -> Path:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="garuda-tr-"))
        self.wav = make_wav(self.tmp / "audio.wav")
        self.progress: list[float] = []
        tr.release_models()

    def tearDown(self):
        tr.release_models()

    def run_tr(self, **overrides):
        args = dict(language="kn", model_size="base", device="cpu", compute_type="int8", model_dir=self.tmp / "models",
                    allow_download=True, on_progress=self.progress.append, cancel=NOT_CANCELLED)
        args.update(overrides)
        audio = args.pop("audio", self.wav)
        return tr.transcribe(audio, **args)


def good_segments(options):
    words = [fw.Word(0.10, 0.55, " ನಮಸ್ಕಾರ", 0.93), fw.Word(0.60, 0.90, " friends", 0.88), fw.Word(1.00, 1.40, " ಇವತ್ತು", 0.90)]
    return [fw.seg("ನಮಸ್ಕಾರ friends ಇವತ್ತು", 0.1, 1.4, words if options["word_timestamps"] else None)], fw.info("kn", 2.0)


# =====================================================================================================
class PlanOptionsMocked(unittest.TestCase):
    """Correct arguments for the transcription call, per language setting."""

    def test_kannada_english_and_auto_settings(self):
        kn, _ = tr.plan_options("kn", beam_size=3, word_timestamps=True, vad_filter=True)
        en, _ = tr.plan_options("en", beam_size=3, word_timestamps=True, vad_filter=True)
        auto, _ = tr.plan_options("auto", beam_size=3, word_timestamps=True, vad_filter=True)
        self.assertEqual((kn["language"], en["language"], auto["language"]), ("kn", "en", None))
        for options in (kn, en, auto):
            self.assertEqual(options["task"], "transcribe", "speech must never be translated")
            self.assertEqual(options["beam_size"], 3)
            self.assertIs(options["word_timestamps"], True)
            self.assertNotIn("initial_prompt", options)

    def test_mixed_mode_detects_per_window_and_uses_a_mixed_prompt(self):
        options, warnings = tr.plan_options("kn-en", beam_size=3, word_timestamps=True, vad_filter=True)
        self.assertIsNone(options["language"])
        self.assertTrue(options["multilingual"])
        self.assertEqual(options["task"], "transcribe")
        self.assertIn("ಕನ್ನಡ".replace("ಕನ್ನಡ", "ಕೆಲವು"), options["initial_prompt"])
        self.assertIn("English", options["initial_prompt"])
        self.assertEqual(warnings, [])

    def test_mixed_mode_falls_back_to_kannada_with_a_warning_on_old_libraries(self):
        accepts = {"language", "task", "beam_size", "word_timestamps", "vad_filter", "initial_prompt",
                   "condition_on_previous_text"}
        options, warnings = tr.plan_options("kn-en", beam_size=3, word_timestamps=True, vad_filter=True, accepts=accepts)
        self.assertEqual(options["language"], "kn")
        self.assertNotIn("multilingual", options)
        self.assertTrue(warnings and "Update faster-whisper" in warnings[0])

    def test_unsupported_arguments_are_removed_instead_of_causing_a_TypeError(self):
        def old_api(audio, language=None, beam_size=5):
            return None
        accepted, dropped = tr.supported_kwargs(old_api, {"language": "kn", "word_timestamps": True, "beam_size": 2})
        self.assertEqual(accepted, {"language": "kn", "beam_size": 2})
        self.assertEqual(dropped, ["word_timestamps"])
        accepted, dropped = tr.supported_kwargs(lambda **kw: None, {"anything": 1})
        self.assertEqual((accepted, dropped), ({"anything": 1}, []))

    def test_language_and_model_lists(self):
        for code in ("auto", "kn", "en", "kn-en", "kanglish"):
            self.assertIn(code, tr.LANGUAGES)
        self.assertEqual(set(tr.MODELS), {"tiny", "base", "small", "medium", "large-v3"})
        self.assertIn("kn-en", tr.WEAK_LANGS)

    def test_kanglish_forces_kannada_recognition_and_romanizes_the_example(self):
        options, warnings = tr.plan_options("kanglish", beam_size=3, word_timestamps=True, vad_filter=True)
        self.assertEqual(options["language"], "kn")
        self.assertEqual(options["task"], "transcribe")
        self.assertEqual(warnings, [])
        self.assertEqual(
            tr.kannada_to_kanglish("ನನ್ನ ಹೆಸರು ಗರುಡ ನಾನು ಒಬ್ಬ ವಿದ್ಯಾರ್ಥಿ"),
            "NANNA HESARU GARUDA NANU OBBA VIDYARTHI",
        )


# =====================================================================================================
class SegmentValidationMocked(unittest.TestCase):
    def consume(self, segments, total=10.0, want_words=True):
        return tr.consume_segments(iter(segments), total=total, want_words=want_words, cancel=NOT_CANCELLED,
                                   progress=lambda p: None)

    def test_none_nan_negative_and_reversed_values_never_crash_and_never_become_captions(self):
        nan = float("nan")
        segments = [
            fw.seg("a", None, None, [fw.Word(None, None, "a", None)]),               # all None: the old code hit a TypeError here
            fw.seg("b", 1.0, 2.0, [fw.Word(1.0, nan, "b", 0.5)]),                     # NaN end
            fw.seg("c", 3.0, 3.0, [fw.Word(3.0, 3.0, "c", 0.5)]),                     # zero length
            fw.seg("d", 5.0, 4.0, [fw.Word(5.0, 4.0, "d", 0.5)]),                     # reversed
            fw.seg("e", -2.0, -1.0, [fw.Word(-2.0, -1.0, "e", 0.5)]),                 # negative
            fw.seg("ok", 6.0, 7.0, [fw.Word(6.0, 7.0, "ok", None)]),                  # valid; probability missing
        ]
        words, stats = self.consume(segments)
        # "b" has a bad WORD time but a valid SEGMENT time, so it is kept as honest phrase-level timing.
        self.assertEqual([(w["text"], w["kind"]) for w in words], [("b", "phrase"), ("ok", "word")])
        self.assertEqual((words[0]["start"], words[0]["end"]), (1.0, 2.0))
        self.assertIsNone(words[1]["confidence"])
        self.assertEqual(stats["dropped"], 4, "None/None, zero-length, reversed and negative segments are dropped")
        self.assertTrue(all(math.isfinite(w["start"]) and math.isfinite(w["end"]) and 0 <= w["start"] < w["end"] for w in words))

    def test_a_segment_with_untrustworthy_words_keeps_honest_phrase_timing(self):
        words, stats = self.consume([fw.seg("hello big world", 1.0, 3.0,
                                            [fw.Word(1.0, 1.5, " hello", 0.9), fw.Word(None, 2.0, " big", 0.9), fw.Word(2.0, 3.0, " world", 0.9)])])
        self.assertEqual(len(words), 1)
        self.assertEqual((words[0]["kind"], words[0]["text"], words[0]["start"], words[0]["end"]), ("phrase", "hello big world", 1.0, 3.0))
        self.assertEqual(stats["phrase_segments"], 1)

    def test_word_level_is_never_invented_when_the_library_returns_no_words(self):
        words, _ = self.consume([fw.seg("only phrase", 0.5, 2.5, None)])
        self.assertEqual([(w["kind"], w["start"], w["end"]) for w in words], [("phrase", 0.5, 2.5)])

    def test_timestamps_are_clamped_inside_the_audio_and_sorted(self):
        words, stats = self.consume([fw.seg("late", 50.0, 51.0, None), fw.seg("b", 2.0, 3.0, None), fw.seg("a", 1.0, 1.5, None),
                                     fw.seg("edge", 9.9, 12.0, None)], total=10.0)
        self.assertEqual([w["text"] for w in words], ["a", "b", "edge"])
        self.assertLessEqual(words[-1]["end"], 10.5)
        self.assertEqual(stats["dropped"], 1)
        self.assertTrue(all(w["start"] < w["end"] for w in words))

    def test_cancellation_is_honoured_while_the_library_is_decoding(self):
        flag = threading.Event()
        flag.set()
        with self.assertRaises(tr.Cancelled):
            tr.consume_segments(iter([fw.seg("x", 0, 1)]), total=5, want_words=True, cancel=flag, progress=lambda p: None)

    def test_a_failing_progress_callback_does_not_fail_the_transcription(self):
        def boom(_):
            raise RuntimeError("database is locked")
        words, _ = tr.consume_segments(iter([fw.seg("x", 0.0, 1.0)]), total=5, want_words=False, cancel=NOT_CANCELLED, progress=boom)
        self.assertEqual(len(words), 1)


# =====================================================================================================
class TranscribeFunctionMocked(Base):
    """The real transcribe() with the library replaced by the fake."""

    def test_success_with_kannada_and_english_words_in_the_same_sentence(self):
        with fw.install(good_segments) as model:
            result = self.run_tr(language="kn-en")
        texts = [w["text"] for w in result["words"]]
        self.assertEqual(texts, ["ನಮಸ್ಕಾರ", "friends", "ಇವತ್ತು"], "both scripts must be preserved unchanged")
        self.assertTrue(all(w["kind"] == "word" for w in result["words"]))
        self.assertTrue(all(0 <= w["start"] < w["end"] <= 2.5 for w in result["words"]))
        self.assertEqual(result["engine"]["timing"], "word")
        self.assertEqual(result["engine"]["device"], "cpu")
        self.assertEqual(result["engine"]["compute_type"], "int8")
        call = model.transcribe_calls[0]
        self.assertIsNone(call["language"])
        self.assertTrue(call["multilingual"])
        self.assertEqual(call["task"], "transcribe")
        self.assertTrue(call["word_timestamps"] and call["vad_filter"])
        import numpy as np
        self.assertIsInstance(call["audio"], np.ndarray, "audio is passed as an array so PyAV is never used")
        self.assertEqual(call["audio"].dtype, np.float32)
        self.assertEqual(len(call["audio"]), 32000, "2 seconds at 16 kHz")
        self.assertLessEqual(float(np.abs(call["audio"]).max()), 1.0)
        self.assertTrue(any("Kannada letters" in w and "English letters" in w for w in result["warnings"]))
        self.assertTrue(self.progress and max(self.progress) <= 0.99)

    def test_model_and_runtime_arguments_reach_the_library(self):
        with fw.install(good_segments) as model:
            self.run_tr(language="en", model_size="small", cpu_threads=4, allow_download=False)
        init = model.init_calls[0]
        self.assertEqual((init["model"], init["device"], init["compute_type"], init["cpu_threads"]), ("small", "cpu", "int8", 4))
        self.assertTrue(init["local_files_only"], "downloads off => local files only")
        self.assertTrue(str(init["download_root"]).endswith("models"))
        self.assertEqual(model.transcribe_calls[0]["language"], "en")

    def test_the_model_is_loaded_once_and_reused_then_released(self):
        with fw.install(good_segments) as model:
            first = self.run_tr()
            second = self.run_tr()
            self.assertEqual(len(model.init_calls), 1, "load once, reuse")
            self.assertFalse(first["engine"]["model_was_cached"])
            self.assertTrue(second["engine"]["model_was_cached"])
            self.assertEqual(tr.loaded_models(), 1)
            self.assertEqual(tr.release_models(), 1)
            self.assertEqual(tr.loaded_models(), 0)
            self.run_tr()
            self.assertEqual(len(model.init_calls), 2, "reloaded after release")

    def test_idle_timer_releases_memory(self):
        with fw.install(good_segments):
            self.run_tr(idle_release_s=1)
            self.assertEqual(tr.loaded_models(), 1)
            for _ in range(40):
                if tr.loaded_models() == 0:
                    break
                threading.Event().wait(0.1)
            self.assertEqual(tr.loaded_models(), 0)

    def test_libraries_without_multilingual_get_forced_kannada_and_a_warning(self):
        with fw.install(good_segments, drop_multilingual=True) as model:
            result = self.run_tr(language="kn-en")
        self.assertEqual(model.transcribe_calls[0]["language"], "kn")
        self.assertTrue(any("Update faster-whisper" in w for w in result["warnings"]))

    # ---- the failure the user reported -------------------------------------------------------------
    def test_TypeError_inside_the_library_triggers_phrase_fallback_and_is_logged_with_traceback(self):
        def behaviour(options):
            if options["word_timestamps"]:
                raise TypeError("unsupported operand type(s) for -: 'NoneType' and 'float'")
            return good_segments(options)
        with fw.install(behaviour):
            with self.assertLogs("garuda.transcribe", level="ERROR") as logs:
                result = self.run_tr()
        self.assertEqual(result["engine"]["timing"], "phrase")
        self.assertTrue(all(w["kind"] == "phrase" for w in result["words"]), "word timing must not be faked")
        joined = " ".join(result["warnings"])
        self.assertIn("TypeError", joined)
        self.assertIn("garuda.log", joined)
        full_log = "\n".join(logs.output)
        self.assertIn("Traceback (most recent call last)", full_log, "the ORIGINAL traceback must be in the log")
        self.assertIn("unsupported operand type(s)", full_log)

    def test_TypeError_while_consuming_the_lazy_segment_stream_is_handled_too(self):
        def exploding_stream():
            yield fw.seg("first", 0.0, 1.0)
            raise TypeError("'NoneType' object is not iterable")
        calls = {"n": 0}

        def behaviour(options):
            calls["n"] += 1
            return (exploding_stream() if calls["n"] == 1 else good_segments(options)[0]), fw.info("kn", 2.0)
        with fw.install(behaviour):
            with self.assertLogs("garuda.transcribe", level="ERROR"):
                result = self.run_tr()
        self.assertGreaterEqual(calls["n"], 2)
        self.assertEqual(result["words"][0]["text"], "ನಮಸ್ಕಾರ friends ಇವತ್ತು")

    def test_when_every_setting_fails_the_error_is_safe_specific_and_chained(self):
        def behaviour(options):
            raise TypeError(r"bad value at C:\Users\Dell\Secret\file.py")
        with fw.install(behaviour) as model:
            with self.assertLogs("garuda.transcribe", level="ERROR"):
                with self.assertRaises(tr.TranscribeError) as ctx:
                    self.run_tr()
        error = ctx.exception
        self.assertEqual(error.code, "decode_failed")
        self.assertIn("TypeError", error.message)
        self.assertIn("garuda.log", error.message)
        self.assertNotIn("Dell", error.message, "file paths must not reach the web page")
        self.assertNotIn("Traceback", error.message)
        self.assertIsInstance(error.__cause__, TypeError, "the original exception stays attached for the log")
        self.assertEqual(len(model.transcribe_calls), 3, "three different settings were tried")

    def test_production_mode_hides_the_exception_message_but_keeps_the_class_name(self):
        with fw.install(lambda o: (_ for _ in ()).throw(TypeError("secret detail"))):
            with self.assertLogs("garuda.transcribe", level="ERROR"):
                with self.assertRaises(tr.TranscribeError) as ctx:
                    self.run_tr(include_detail=False)
        self.assertIn("(TypeError)", ctx.exception.message)
        self.assertNotIn("secret detail", ctx.exception.message)

    # ---- never claim success without captions ------------------------------------------------------
    def test_no_speech_is_an_error_not_a_success(self):
        with fw.install(lambda o: ([], fw.info("en", 2.0))) as model:
            with self.assertRaises(tr.TranscribeError) as ctx:
                self.run_tr()
        self.assertEqual(ctx.exception.code, "no_speech")
        self.assertEqual([c["vad_filter"] for c in model.transcribe_calls], [True, False], "retried once without the voice filter")

    def test_only_invalid_segments_is_also_no_speech(self):
        with fw.install(lambda o: ([fw.seg("x", None, None, None), fw.seg("y", 3.0, 2.0, None)], fw.info("en", 2.0))):
            with self.assertRaises(tr.TranscribeError) as ctx:
                self.run_tr()
        self.assertEqual(ctx.exception.code, "no_speech")

    def test_made_up_text_warning_when_the_voice_filter_had_to_be_switched_off(self):
        def behaviour(options):
            return ([], fw.info("en", 2.0)) if options["vad_filter"] else good_segments(options)
        with fw.install(behaviour):
            result = self.run_tr()
        self.assertTrue(any("made-up words" in w for w in result["warnings"]))

    # ---- request / media problems ------------------------------------------------------------------
    def test_bad_language_and_model_are_rejected_before_any_work(self):
        with fw.install(good_segments) as model:
            for kwargs, code in (({"language": "xx"}, "bad_language"), ({"language": "or"}, "bad_language"), ({"model_size": "huge"}, "bad_model")):
                with self.assertRaises(tr.TranscribeError) as ctx:
                    self.run_tr(**kwargs)
                self.assertEqual(ctx.exception.code, code)
            self.assertEqual(model.init_calls, [], "no model may be loaded for an invalid request")

    def test_missing_empty_and_too_short_audio(self):
        with fw.install(good_segments):
            for audio, code in ((self.tmp / "nope.wav", "audio_missing"), (self.tmp / "empty.wav", "audio_missing"),
                                (make_wav(self.tmp / "tiny.wav", 0.05), "audio_too_short")):
                if audio.name == "empty.wav":
                    audio.write_bytes(b"")
                with self.assertRaises(tr.TranscribeError) as ctx:
                    self.run_tr(audio=audio)
                self.assertEqual(ctx.exception.code, code, audio.name)

    # ---- model loading problems --------------------------------------------------------------------
    def test_speech_library_missing(self):
        saved = sys.modules.pop("faster_whisper", None)
        sys.modules["faster_whisper"] = None          # makes `import faster_whisper` raise ImportError
        try:
            with self.assertRaises(tr.TranscribeError) as ctx:
                self.run_tr()
        finally:
            sys.modules.pop("faster_whisper", None)
            if saved is not None:
                sys.modules["faster_whisper"] = saved
        self.assertEqual(ctx.exception.code, "whisper_missing")
        self.assertIn("install.bat", ctx.exception.message)

    def test_speech_library_installed_but_broken(self):
        class BrokenImport(dict):
            pass
        module = type(sys)("faster_whisper")
        module.__getattr__ = lambda name: (_ for _ in ()).throw(OSError("DLL load failed"))
        saved = sys.modules.get("faster_whisper")
        sys.modules["faster_whisper"] = module
        try:
            with self.assertLogs("garuda.transcribe", level="ERROR"):
                with self.assertRaises(tr.TranscribeError) as ctx:
                    self.run_tr()
        finally:
            sys.modules.pop("faster_whisper", None)
            if saved is not None:
                sys.modules["faster_whisper"] = saved
        self.assertEqual(ctx.exception.code, "whisper_import_failed")

    def test_model_files_missing_when_downloads_are_off(self):
        error = type("LocalEntryNotFoundError", (OSError,), {})("Cannot find an appropriate cached snapshot folder")
        with fw.install(init_error=error):
            with self.assertLogs("garuda.transcribe", level="ERROR"):
                with self.assertRaises(tr.TranscribeError) as ctx:
                    self.run_tr(allow_download=False)
        self.assertEqual(ctx.exception.code, "model_missing")
        self.assertIn("ALLOW_MODEL_DOWNLOAD", ctx.exception.message)

    def test_model_download_failure_when_downloads_are_allowed(self):
        error = type("ConnectionError", (OSError,), {})("Max retries exceeded: getaddrinfo failed")
        with fw.install(init_error=error):
            with self.assertLogs("garuda.transcribe", level="ERROR"):
                with self.assertRaises(tr.TranscribeError) as ctx:
                    self.run_tr()
        self.assertEqual(ctx.exception.code, "model_download_failed")
        self.assertIn("internet", ctx.exception.message)

    def test_other_model_load_failures_and_out_of_memory(self):
        for error, code in ((ValueError("Invalid model data"), "model_load_failed"), (MemoryError(), "out_of_memory")):
            with fw.install(init_error=error):
                with self.assertLogs("garuda.transcribe", level="ERROR"):
                    with self.assertRaises(tr.TranscribeError) as ctx:
                        self.run_tr()
            self.assertEqual(ctx.exception.code, code)

    def test_out_of_memory_while_decoding(self):
        with fw.install(lambda o: (_ for _ in ()).throw(MemoryError())):
            with self.assertLogs("garuda.transcribe", level="ERROR"):
                with self.assertRaises(tr.TranscribeError) as ctx:
                    self.run_tr()
        self.assertEqual(ctx.exception.code, "out_of_memory")

    def test_cancel_during_decoding(self):
        flag = threading.Event()
        flag.set()
        with fw.install(good_segments):
            with self.assertRaises(tr.Cancelled):
                self.run_tr(cancel=flag)

    def test_gpu_request_falls_back_to_cpu_without_crashing(self):
        with fw.install(good_segments) as model:
            result = self.run_tr(device="cuda")
        self.assertIn(model.init_calls[0]["device"], ("cpu", "cuda"))
        if model.init_calls[0]["device"] == "cpu":       # no NVIDIA card (or no ctranslate2) in this environment
            self.assertEqual(result["engine"]["device"], "cpu")

    def test_unknown_device_uses_cpu(self):
        device, compute, notes = tr.resolve_runtime("tpu", "int8")
        self.assertEqual(device, "cpu")
        self.assertTrue(notes)


class PyAVIncompatibilityRegression(Base):
    """The REAL failure seen on a user's PC: faster-whisper 1.2.1 + PyAV 19 =>
    "TypeError: open() got an unexpected keyword argument 'metadata_errors'" whenever the library decodes a FILE."""

    PYAV_ERROR = "open() got an unexpected keyword argument 'metadata_errors'"

    def behaviour(self, options):
        if isinstance(options["audio"], (str, bytes)):         # the library would call av.open(path, metadata_errors=...)
            raise TypeError(self.PYAV_ERROR)
        return good_segments(options)

    def test_audio_is_given_as_an_array_so_the_broken_file_decoder_is_never_reached(self):
        with fw.install(self.behaviour):
            result = self.run_tr(language="kn-en")
        self.assertEqual([w["text"] for w in result["words"]], ["ನಮಸ್ಕಾರ", "friends", "ಇವತ್ತು"])
        self.assertEqual(result["engine"]["timing"], "word")
        self.assertFalse(any("TypeError" in w for w in result["warnings"]), "no fallback should have been needed")

    def test_the_old_behaviour_would_have_failed(self):
        """Documents the bug: passing the file path to this library version raises the reported TypeError."""
        with fw.install(self.behaviour):
            module = sys.modules["faster_whisper"]
            with self.assertRaises(TypeError) as ctx:
                module.WhisperModel("base").transcribe(str(self.wav))
        self.assertIn("metadata_errors", str(ctx.exception))

    def test_unreadable_and_wrongly_formatted_audio_are_clear_errors(self):
        bad = self.tmp / "garbage.wav"
        bad.write_bytes(b"not a wav file" * 200)
        stereo = self.tmp / "stereo44k.wav"
        with wave.open(str(stereo), "wb") as fh:
            fh.setnchannels(2)
            fh.setsampwidth(2)
            fh.setframerate(44100)
            fh.writeframes(b"\x00\x00" * 2 * 44100)
        with fw.install(good_segments):
            for audio, code in ((bad, "audio_unreadable"), (stereo, "audio_format")):
                with self.assertLogs("garuda.transcribe", level="ERROR") if code == "audio_unreadable" else self.subTest(code):
                    with self.assertRaises(tr.TranscribeError) as ctx:
                        self.run_tr(audio=audio)
                self.assertEqual(ctx.exception.code, code)

    def test_real_ffmpeg_output_is_accepted_end_to_end(self):
        """Audio made by the app's own FFmpeg step (normalised) goes through transcribe() with the fake library."""
        import subprocess
        from app import media
        video = self.tmp / "v.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
                        "-f", "lavfi", "-i", "sine=frequency=300:duration=2", "-shortest", "-pix_fmt", "yuv420p", str(video)], check=True)
        wav = self.tmp / "extracted.wav"
        media.extract_audio(video, wav, normalize=True)
        with fw.install(self.behaviour) as model:
            result = self.run_tr(audio=wav)
        self.assertTrue(result["words"])
        self.assertAlmostEqual(len(model.transcribe_calls[0]["audio"]) / 16000, 2.0, delta=0.2)


class ErrorTextSafety(unittest.TestCase):
    def test_paths_are_removed_from_error_details(self):
        text = tr._detail(OSError(r"cannot open C:\Users\Dell\Videos\a.mp4 and /home/dell/b.wav"), include_message=True)
        self.assertNotIn("Dell", text)
        self.assertNotIn("/home/dell", text)
        self.assertIn("<path>", text)
        self.assertEqual(tr._detail(OSError("x"), include_message=False), "OSError")


# =====================================================================================================
class SubtitleUnicode(unittest.TestCase):
    """Kannada + English text must survive grouping and export unchanged."""

    ITEMS = [{"text": "ನಮಸ್ಕಾರ", "start": 0.1, "end": 0.5, "kind": "word"}, {"text": "friends", "start": 0.6, "end": 1.0, "kind": "word"},
             {"text": "ಇವತ್ತು", "start": 1.1, "end": 1.5, "kind": "word"}, {"text": "meeting", "start": 1.6, "end": 2.0, "kind": "word"}]

    def test_srt_and_vtt_keep_both_scripts_and_valid_timestamps(self):
        words = cap.to_words(self.ITEMS)
        srt, vtt = cap.to_srt(words, cap.DEFAULT_STYLE), cap.to_vtt(words, cap.DEFAULT_STYLE)
        for text in (srt, vtt):
            data = text.encode("utf-8").decode("utf-8")
            self.assertIn("ನಮಸ್ಕಾರ", data)
            self.assertIn("friends", data)
            self.assertIn("ಇವತ್ತು", data)
            self.assertIn("meeting", data)
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertRegex(srt, r"\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}")
        # every cue ends after it starts
        import re
        for start, end in re.findall(r"(\d\d:\d\d:\d\d,\d{3}) --> (\d\d:\d\d:\d\d,\d{3})", srt):
            self.assertLess(start, end)

    def test_files_are_written_as_utf_8(self):
        path = Path(tempfile.mkdtemp()) / "c.srt"
        path.write_text(cap.to_srt(cap.to_words(self.ITEMS), cap.DEFAULT_STYLE), encoding="utf-8")
        self.assertIn("ಇವತ್ತು", path.read_text(encoding="utf-8"))


# =====================================================================================================
def _real_library_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(os.environ.get("GARUDA_E2E") == "1" and _real_library_available(),
                     "real-library test: needs faster-whisper installed and GARUDA_E2E=1 (downloads a model on first use)")
class RealLibraryEndToEnd(Base):
    """Uses the GENUINE faster-whisper and a REAL model. Set GARUDA_E2E_AUDIO=path\\to\\speech.wav for real speech."""

    def test_real_model_on_a_silent_file_reports_no_speech_cleanly(self):
        with self.assertRaises(tr.TranscribeError) as ctx:
            self.run_tr(model_size=os.environ.get("GARUDA_E2E_MODEL", "tiny"), language="auto",
                        model_dir=Path(os.environ.get("WHISPER_MODEL_DIR", self.tmp / "models")))
        self.assertEqual(ctx.exception.code, "no_speech")

    @unittest.skipUnless(os.environ.get("GARUDA_E2E_AUDIO"), "set GARUDA_E2E_AUDIO to a speech WAV file")
    def test_real_model_on_real_speech(self):
        result = self.run_tr(audio=Path(os.environ["GARUDA_E2E_AUDIO"]), model_size=os.environ.get("GARUDA_E2E_MODEL", "base"),
                             language=os.environ.get("GARUDA_E2E_LANG", "kn-en"),
                             model_dir=Path(os.environ.get("WHISPER_MODEL_DIR", self.tmp / "models")))
        self.assertTrue(result["words"])
        for w in result["words"]:
            self.assertTrue(math.isfinite(w["start"]) and math.isfinite(w["end"]) and w["end"] > w["start"] >= 0)


if __name__ == "__main__":
    unittest.main()

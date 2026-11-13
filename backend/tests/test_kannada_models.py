"""Kannada-tuned models, custom models and the vocabulary hint (mocked speech library)."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_faster_whisper as fw  # noqa: E402
from app import transcribe as tr  # noqa: E402
from test_transcribe import Base, good_segments  # noqa: E402


class Registry(unittest.TestCase):
    def setUp(self):
        self.models = Path(tempfile.mkdtemp())

    def test_general_models_and_the_ready_made_kannada_model_are_always_offered(self):
        choices = tr.model_choices(self.models)
        for name in ("tiny", "base", "small", "medium", "large-v3", "kn-medium"):
            self.assertIn(name, choices)
        self.assertNotIn("kn-small", choices, "only listed after it has been converted")
        self.assertNotIn("custom", choices)

    def test_converted_small_and_custom_models_appear_when_present(self):
        (self.models / "kn-small-ct2").mkdir()
        choices = tr.model_choices(self.models, "my-model-folder")
        self.assertIn("kn-small", choices)
        self.assertIn("custom", choices)

    def test_sources(self):
        self.assertEqual(tr.resolve_model_source("base", self.models, None), ("base", False))
        self.assertEqual(tr.resolve_model_source("kn-medium", self.models, None),
                         ("itsskofficial/livewhisper-kn-medium", False))
        (self.models / "kn-small-ct2").mkdir()
        source, local = tr.resolve_model_source("kn-small", self.models, None)
        self.assertTrue(local and source.endswith("kn-small-ct2"))
        folder = self.models / "mine"
        folder.mkdir()
        self.assertEqual(tr.resolve_model_source("custom", self.models, str(folder)), (str(folder), True))
        self.assertEqual(tr.resolve_model_source("custom", self.models, "someone/their-ct2-model"),
                         ("someone/their-ct2-model", False))

    def test_helpful_errors_for_missing_models(self):
        with self.assertRaises(tr.TranscribeError) as ctx:
            tr.resolve_model_source("kn-small", self.models, None)
        self.assertEqual(ctx.exception.code, "model_missing")
        self.assertIn("convert_model.bat", ctx.exception.message)
        for value, code in ((None, "bad_model"), (str(self.models / "nope"), "model_missing")):
            with self.assertRaises(tr.TranscribeError) as ctx:
                tr.resolve_model_source("custom", self.models, value)
            self.assertEqual(ctx.exception.code, code)

    def test_vocabulary_prompt_is_short_and_clean(self):
        self.assertEqual(tr.vocab_prompt(["  Garuda ", "", "ಬೆಂಗಳೂರು"]), "Garuda, ಬೆಂಗಳೂರು")
        self.assertEqual(tr.vocab_prompt(None), "")
        long = tr.vocab_prompt(["word%03d" % i for i in range(100)])
        self.assertLessEqual(len(long), 150)


class KannadaModelBehaviour(Base):
    def test_kanglish_generation_keeps_word_timings_and_romanizes_kannada(self):
        with fw.install(good_segments) as model:
            result = self.run_tr(language="kanglish", model_size="kn-medium")
        self.assertEqual(model.transcribe_calls[0]["language"], "kn")
        self.assertEqual(result["language"], "kanglish")
        self.assertEqual([word["text"] for word in result["words"]], ["NAMASKARA", "FRIENDS", "IVATTU"])
        self.assertTrue(all(word["kind"] == "word" for word in result["words"]))
        self.assertTrue(any("not translation" in warning for warning in result["warnings"]))

    def test_kannada_tuned_model_forces_kannada_keeps_the_mixed_prompt_and_warns(self):
        with fw.install(good_segments) as model:
            result = self.run_tr(language="kn-en", model_size="kn-medium")
        call = model.transcribe_calls[0]
        self.assertEqual(call["language"], "kn")
        self.assertFalse(call["multilingual"])
        self.assertIn("English", call["initial_prompt"])
        self.assertEqual(call["task"], "transcribe")
        self.assertEqual(model.init_calls[0]["model"], "itsskofficial/livewhisper-kn-medium")
        self.assertTrue(any("English words inside a sentence may be written in Kannada letters" in w for w in result["warnings"]))
        self.assertEqual(result["engine"]["model"], "kn-medium")

    def test_auto_with_a_kannada_tuned_model_also_means_kannada(self):
        with fw.install(good_segments) as model:
            self.run_tr(language="auto", model_size="kn-medium")
        self.assertEqual(model.transcribe_calls[0]["language"], "kn")
        self.assertIsNone(model.transcribe_calls[0]["initial_prompt"], "auto has no mixed prompt, only the forced language")

    def test_kannada_tuned_model_warns_when_another_language_is_selected(self):
        with fw.install(good_segments) as model:
            result = self.run_tr(language="en", model_size="kn-medium")
        self.assertEqual(model.transcribe_calls[0]["language"], "en")
        self.assertTrue(any("selected language is English" in warning and "large-v3" in warning
                            for warning in result["warnings"]))

    def test_general_models_keep_per_window_language_detection(self):
        with fw.install(good_segments) as model:
            self.run_tr(language="kn-en", model_size="small")
        self.assertIsNone(model.transcribe_calls[0]["language"])
        self.assertTrue(model.transcribe_calls[0]["multilingual"])

    def test_small_general_models_are_flagged_as_weak_for_kannada(self):
        with fw.install(good_segments):
            result = self.run_tr(language="kn", model_size="base")
        self.assertTrue(any("kn-medium" in w for w in result["warnings"]))
        with fw.install(good_segments):
            ok = self.run_tr(language="en", model_size="base")
        self.assertFalse(any("kn-medium" in w for w in ok["warnings"]))

    def test_custom_local_model_is_loaded_from_its_folder(self):
        folder = self.tmp / "my-finetune-ct2"
        folder.mkdir()
        with fw.install(good_segments) as model:
            result = self.run_tr(language="kn", model_size="custom", custom_model=str(folder))
        self.assertEqual(model.init_calls[0]["model"], str(folder))
        self.assertEqual(result["engine"]["model"], "custom")

    def test_missing_converted_small_model_gives_the_conversion_command(self):
        with fw.install(good_segments) as model:
            with self.assertRaises(tr.TranscribeError) as ctx:
                self.run_tr(model_size="kn-small")
        self.assertEqual(ctx.exception.code, "model_missing")
        self.assertEqual(model.init_calls, [])

    def test_vocabulary_reaches_the_prompt(self):
        with fw.install(good_segments) as model:
            self.run_tr(language="kn", model_size="small", vocabulary=["Garuda", "ಬೆಂಗಳೂರು"])
        self.assertEqual(model.transcribe_calls[0]["initial_prompt"], "Garuda, ಬೆಂಗಳೂರು")
        with fw.install(good_segments) as model:
            self.run_tr(language="kn-en", model_size="small", vocabulary=["Garuda"])
        self.assertTrue(model.transcribe_calls[0]["initial_prompt"].endswith("Garuda"))
        self.assertIn("Kannada and English", model.transcribe_calls[0]["initial_prompt"])


if __name__ == "__main__":
    unittest.main()

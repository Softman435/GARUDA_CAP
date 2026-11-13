"""Tests for the parts of the converter that do not need PyTorch (tokenizer fallback)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import convert_model as cm  # noqa: E402


class FetchMissingFiles(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.cache = Path(tempfile.mkdtemp())
        self.asked = []

    def downloader(self, repo_id, filename):
        self.asked.append((repo_id, filename))
        if repo_id == "someone/kn-model" and filename == "tokenizer.json":
            raise FileNotFoundError("not in this repository")          # typical for fine-tune repositories
        path = self.cache / f"{repo_id.replace('/', '_')}-{filename}"
        path.write_text(f"{repo_id}:{filename}", encoding="utf-8")
        return str(path)

    def test_missing_tokenizer_comes_from_the_original_openai_model(self):
        copied = cm.fetch_missing_files(self.out, "someone/kn-model", "small", downloader=self.downloader, log=lambda *_: None)
        self.assertEqual((self.out / "tokenizer.json").read_text(encoding="utf-8"), "openai/whisper-small:tokenizer.json")
        self.assertEqual((self.out / "preprocessor_config.json").read_text(encoding="utf-8"),
                         "someone/kn-model:preprocessor_config.json", "taken from the source when it has it")
        self.assertEqual(len(copied), 2)

    def test_files_that_already_exist_are_left_alone(self):
        (self.out / "tokenizer.json").write_text("mine", encoding="utf-8")
        (self.out / "preprocessor_config.json").write_text("mine", encoding="utf-8")
        self.assertEqual(cm.fetch_missing_files(self.out, "x/y", "base", downloader=self.downloader), [])
        self.assertEqual(self.asked, [])

    def test_local_source_folder_is_used_first(self):
        source = Path(tempfile.mkdtemp())
        (source / "tokenizer.json").write_text("local-tok", encoding="utf-8")
        (source / "preprocessor_config.json").write_text("local-pre", encoding="utf-8")
        cm.fetch_missing_files(self.out, str(source), "small", downloader=self.downloader)
        self.assertEqual((self.out / "tokenizer.json").read_text(encoding="utf-8"), "local-tok")
        self.assertEqual(self.asked, [])

    def test_error_when_no_source_has_the_file(self):
        def never(repo_id, filename):
            raise FileNotFoundError(filename)
        with self.assertRaises(RuntimeError):
            cm.fetch_missing_files(self.out, "a/b", "small", downloader=never, log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()

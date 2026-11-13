"""TrueType Collection font discovery and coverage tests."""
import sys
import tempfile
import unittest
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import fonts  # noqa: E402


class FontCollectionTests(unittest.TestCase):
    def test_collection_faces_are_listed_and_checked_for_glyph_coverage(self):
        project_fonts = Path(__file__).resolve().parents[1] / "fonts"
        with tempfile.TemporaryDirectory() as temp_dir:
            collection_path = Path(temp_dir) / "dejavu.ttc"
            collection = TTCollection()
            collection.fonts = [
                TTFont(str(project_fonts / "DejaVuSans.ttf")),
                TTFont(str(project_fonts / "DejaVuSans-Bold.ttf")),
            ]
            try:
                collection.save(str(collection_path))
            finally:
                collection.close()

            listed = fonts.list_fonts(Path(temp_dir))
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["family"], "DejaVu Sans")
            self.assertEqual(listed[0]["file"], "dejavu.ttc")
            self.assertEqual(fonts.missing_glyphs("Hello", "DejaVu Sans", Path(temp_dir)), [])
            self.assertEqual(fonts.missing_glyphs("ಕ", "DejaVu Sans", Path(temp_dir)), ["ಕ"])


if __name__ == "__main__":
    unittest.main()

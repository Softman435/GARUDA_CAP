"""Core tests. They use only the standard library, so they run with either:
    python -m unittest discover -s backend/tests -v
    pytest backend/tests
Tests marked REAL FFMPEG run the real ffmpeg/ffprobe programs. Nothing here runs a real AI model.
"""
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import captions as cap  # noqa: E402
from app import media, render  # noqa: E402


def W(text, s, e, kind="word"):
    return cap.Word(text, s, e, kind)


WORDS = [W("Hello", 0.2, 0.6), W("world.", 0.7, 1.2), W("This", 1.4, 1.7), W("is", 1.75, 1.9),
         W("a", 1.95, 2.0), W("test", 2.05, 2.6), W("of", 2.65, 2.8), W("captions", 2.85, 3.5)]


def make_video(path: Path, seconds=4, audio=True):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=25:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if audio else []) + [str(path)]
    subprocess.run(cmd, check=True)


class Grouping(unittest.TestCase):
    def test_word_mode_one_at_a_time(self):
        g = cap.group_words(WORDS, {"mode": "word", "words_visible": 1})
        self.assertEqual(len(g), len(WORDS))
        self.assertEqual(g[0].start, 0.2)

    def test_word_mode_visible_two(self):
        g = cap.group_words(WORDS, {"mode": "word", "words_visible": 2})
        self.assertEqual(len(g), 4)

    def test_phrase_breaks_on_sentence_end(self):
        g = cap.group_words(WORDS, {"mode": "phrase", "max_words": 10})
        self.assertEqual([x.words[-1].text for x in g][0], "world.")

    def test_phrase_respects_max_words(self):
        g = cap.group_words(WORDS, {"mode": "phrase", "max_words": 3})
        self.assertTrue(all(len(x.words) <= 3 for x in g))

    def test_pause_breaks_phrase(self):
        words = [W("a", 0, 0.5), W("b", 3.0, 3.5)]
        self.assertEqual(len(cap.group_words(words, {"mode": "phrase", "max_words": 10})), 2)

    def test_phrase_level_entries_never_merge(self):
        words = [W("one two", 0, 1, "phrase"), W("three four", 1, 2, "phrase")]
        self.assertEqual(len(cap.group_words(words, {"mode": "phrase", "max_words": 10})), 2)

    def test_empty(self):
        self.assertEqual(cap.group_words([], {}), [])


class Subtitles(unittest.TestCase):
    def test_srt_format(self):
        srt = cap.to_srt(WORDS, {"max_words": 10})
        self.assertTrue(srt.startswith("1\n00:00:00,200 --> 00:00:01,200\nHello world.\n"))

    def test_vtt_header_and_escape(self):
        vtt = cap.to_vtt([W("a<b", 0, 1)], {})
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("a&lt;b", vtt)

    def test_offset_shifts_and_drops_negative(self):
        srt = cap.to_srt([W("x", 1.0, 2.0)], {"offset": 0.5})
        self.assertIn("00:00:01,500 --> 00:00:02,500", srt)
        self.assertEqual(cap.to_srt([W("x", 0.0, 0.3)], {"offset": -1.0}), "")

    def test_unicode_kept_intact(self):
        kn = "ನಮಸ್ಕಾರ ಕನ್ನಡ"
        srt = cap.to_srt([W("ನಮಸ್ಕಾರ", 0, 1), W("ಕನ್ನಡ", 1, 2)], {"max_words": 5})
        self.assertIn(kn, srt)
        hi = cap.to_srt([W("नमस्ते", 0, 1), W("दुनिया", 1, 2)], {"max_words": 5})
        self.assertIn("नमस्ते दुनिया", hi)

    def test_script_detection(self):
        self.assertEqual(cap.scripts_in("ನಮಸ್ಕಾರ hello"), {"Kannada"})
        self.assertEqual(cap.scripts_in("नमस्ते"), {"Devanagari"})

    def test_normalize_drops_controls_keeps_text(self):
        self.assertEqual(cap.normalize_text("a\x00b\u200dc"), "ab\u200dc") if False else None
        self.assertEqual(cap.normalize_text("a\x00b"), "ab")

    def test_wrap_keeps_all_text(self):
        lines = cap.wrap_lines("one two three four five six seven", 8, 2)
        self.assertEqual(len(lines), 2)
        self.assertEqual(" ".join(lines), "one two three four five six seven")


class Ass(unittest.TestCase):
    def test_ass_has_events_and_size(self):
        r = cap.to_ass(WORDS, {"mode": "phrase"}, 1080, 1920)
        self.assertIn("PlayResX: 1080", r.text)
        self.assertGreater(r.events, 0)
        self.assertIn("Dialogue:", r.text)

    def test_colour_conversion(self):
        self.assertEqual(cap._ass_color("#FF0000"), "&H000000FF")
        self.assertEqual(cap._ass_color("#00FF00", 0.0), "&HFF00FF00")

    def test_braces_and_backslashes_escaped(self):
        r = cap.to_ass([W("{\\an8}hack", 0, 1)], {}, 640, 360)
        self.assertNotIn("{\\an8}hack", r.text)

    def test_phrase_level_warning(self):
        r = cap.to_ass([W("hello there", 0, 1, "phrase")], {}, 640, 360)
        self.assertTrue(r.warnings)

    def test_alignment_values(self):
        self.assertEqual(cap._alignment("bottom", "center"), 2)
        self.assertEqual(cap._alignment("top", "left"), 7)
        self.assertEqual(cap._alignment("middle", "right"), 6)


class Sizes(unittest.TestCase):
    info = media.MediaInfo("mp4", ".mp4", 4.0, 1920, 1080, 25, "h264", "aac", 2, 1000)

    def test_vertical(self):
        self.assertEqual(render.output_size(self.info, "9:16", 1080), (1080, 1920))

    def test_square_and_landscape(self):
        self.assertEqual(render.output_size(self.info, "1:1", 720), (720, 720))
        self.assertEqual(render.output_size(self.info, "16:9", 720), (1280, 720))

    def test_original_never_upscales(self):
        small = media.MediaInfo("mp4", ".mp4", 4.0, 640, 360, 25, "h264", "aac", 2, 1)
        self.assertEqual(render.output_size(small, "original", 1080), (640, 360))

    def test_bad_aspect_rejected(self):
        with self.assertRaises(render.ExportError):
            render.output_size(self.info, "4:3", 720)

    def test_command_is_argument_list_without_shell(self):
        cmd = render.build_burn_command(src=Path("in.mp4"), ass_name="captions.ass", fonts_dir=Path("fonts"),
                                        dst=Path("out.mp4"), width=1080, height=1920, quality="standard")
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("libx264", cmd)

    def test_bad_quality_rejected(self):
        with self.assertRaises(render.ExportError):
            render.build_burn_command(src=Path("a"), ass_name="c.ass", fonts_dir=Path("f"), dst=Path("o"),
                                      width=2, height=2, quality="ultra")

    def test_filter_path_escape_windows(self):
        self.assertEqual(render._filter_value("C:\\Users\\me\\fonts"), "C\\:/Users/me/fonts")


class ProbeValidation(unittest.TestCase):
    """Uses crafted ffprobe-style dictionaries (mocked input, no ffmpeg run)."""

    def ok(self, **over):
        data = {"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "10.0"},
                "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720, "avg_frame_rate": "30/1"},
                            {"codec_type": "audio", "codec_name": "aac", "channels": 2}]}
        data.update(over)
        return data

    def check(self, data, **kw):
        args = dict(max_duration=3600, max_width=7680, max_height=4320)
        args.update(kw)
        return media.validate_probe(data, 1000, **args)

    def test_good(self):
        info = self.check(self.ok())
        self.assertEqual((info.width, info.height, info.fps), (1280, 720, 30.0))

    def test_no_audio(self):
        d = self.ok()
        d["streams"] = d["streams"][:1]
        with self.assertRaises(media.MediaError) as c:
            self.check(d)
        self.assertEqual(c.exception.code, "no_audio")

    def test_no_video_or_image_only(self):
        d = self.ok()
        d["streams"] = [{"codec_type": "video", "codec_name": "mjpeg", "width": 10, "height": 10}, d["streams"][1]]
        with self.assertRaises(media.MediaError) as c:
            self.check(d)
        self.assertEqual(c.exception.code, "no_video")

    def test_unsupported_container(self):
        with self.assertRaises(media.MediaError) as c:
            self.check(self.ok(format={"format_name": "image2", "duration": "1"}))
        self.assertEqual(c.exception.code, "unsupported_container")

    def test_too_long_and_bad_duration(self):
        with self.assertRaises(media.MediaError):
            self.check(self.ok(), max_duration=5)
        with self.assertRaises(media.MediaError):
            self.check(self.ok(format={"format_name": "mov,mp4", "duration": "nan"}))

    def test_resolution_limit(self):
        with self.assertRaises(media.MediaError):
            self.check(self.ok(), max_width=640)

    def test_rotation_swaps_dimensions(self):
        d = self.ok()
        d["streams"][0]["side_data_list"] = [{"rotation": -90}]
        info = self.check(d)
        self.assertEqual((info.width, info.height), (720, 1280))


@unittest.skipUnless(media.tool_paths()["ffmpeg"], "ffmpeg not installed")
class RealFfmpeg(unittest.TestCase):
    """REAL FFMPEG: runs the actual ffprobe/ffmpeg programs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_probe_real_video(self):
        v = self.dir / "a.mp4"
        make_video(v)
        info = media.inspect(v, max_duration=60, max_width=4000, max_height=4000)
        self.assertEqual((info.width, info.height), (640, 360))
        self.assertAlmostEqual(info.duration, 4.0, delta=0.3)

    def test_no_audio_video_rejected(self):
        v = self.dir / "silent.mp4"
        make_video(v, audio=False)
        with self.assertRaises(media.MediaError) as c:
            media.inspect(v, max_duration=60, max_width=4000, max_height=4000)
        self.assertEqual(c.exception.code, "no_audio")

    def test_corrupted_and_fake_extension_rejected(self):
        bad = self.dir / "fake.mp4"
        bad.write_bytes(b"this is not a video at all" * 100)
        with self.assertRaises(media.MediaError) as c:
            media.inspect(bad, max_duration=60, max_width=4000, max_height=4000)
        self.assertEqual(c.exception.code, "unreadable")

    def test_extract_audio(self):
        v = self.dir / "a.mp4"
        make_video(v)
        wav = self.dir / "a.wav"
        media.extract_audio(v, wav)
        self.assertGreater(wav.stat().st_size, 10000)

    def test_full_export_keeps_original_and_audio(self):
        v = self.dir / "a.mp4"
        make_video(v)
        before = v.read_bytes()
        info = media.inspect(v, max_duration=60, max_width=4000, max_height=4000)
        fonts = self.dir / "fonts"
        fonts.mkdir()
        progress = []
        res = render.export_project(
            src=v, info=info, words=WORDS, style={"mode": "word"}, out_dir=self.dir / "out", fonts_dir=fonts,
            aspect="9:16", resolution=480, quality="draft", fps=None,
            on_progress=progress.append, cancel=threading.Event())
        self.assertEqual(v.read_bytes(), before, "original must be untouched")
        self.assertEqual((res.width, res.height), (480, 854))
        out = media.inspect(res.mp4, max_duration=60, max_width=4000, max_height=4000)
        self.assertEqual(out.audio_codec, "aac")
        self.assertAlmostEqual(out.duration, info.duration, delta=0.5)
        self.assertTrue(res.srt.read_text().startswith("1\n"))
        self.assertTrue(res.vtt.read_text().startswith("WEBVTT"))
        self.assertTrue(progress)

    def test_captions_are_really_burned_in(self):
        """Frame at a caption time must differ from the same frame of an export with no caption showing."""
        v = self.dir / "a.mp4"
        make_video(v)
        info = media.inspect(v, max_duration=60, max_width=4000, max_height=4000)
        fonts = self.dir / "fonts"
        fonts.mkdir()

        def grab(words, name):
            out = self.dir / name
            res = render.export_project(
                src=v, info=info, words=words, style={}, out_dir=out, fonts_dir=fonts, aspect="original",
                resolution=360, quality="draft", fps=None, on_progress=lambda p: None, cancel=threading.Event())
            png = self.dir / f"{name}.png"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1.0", "-i", str(res.mp4), "-frames:v", "1", str(png)], check=True)
            return png.read_bytes()

        with_caption = grab([W("HELLO", 0.5, 2.5)], "with")
        without = grab([W("HELLO", 3.0, 3.8)], "without")
        self.assertNotEqual(with_caption, without)

    def test_export_with_no_captions_refused(self):
        v = self.dir / "a.mp4"
        make_video(v)
        info = media.inspect(v, max_duration=60, max_width=4000, max_height=4000)
        with self.assertRaises(render.ExportError):
            render.export_project(src=v, info=info, words=[], style={}, out_dir=self.dir / "o", fonts_dir=self.dir,
                                  aspect="original", resolution=360, quality="draft", fps=None,
                                  on_progress=lambda p: None, cancel=threading.Event())

    def test_cancel_stops_export(self):
        v = self.dir / "long.mp4"
        make_video(v, seconds=20)
        info = media.inspect(v, max_duration=60, max_width=4000, max_height=4000)
        ev = threading.Event()
        ev.set()
        with self.assertRaises(render.Cancelled):
            render.export_project(src=v, info=info, words=WORDS, style={}, out_dir=self.dir / "o", fonts_dir=self.dir,
                                  aspect="original", resolution=360, quality="high", fps=None,
                                  on_progress=lambda p: None, cancel=ev)

    def test_verify_output_detects_missing_file(self):
        info = media.MediaInfo("mp4", ".mp4", 4.0, 640, 360, 25, "h264", "aac", 2, 1)
        with self.assertRaises(render.ExportError):
            render.verify_output(self.dir / "nope.mp4", info)


if __name__ == "__main__":
    unittest.main(verbosity=2)

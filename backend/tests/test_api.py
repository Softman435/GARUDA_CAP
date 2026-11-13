"""API tests (pytest + FastAPI TestClient). Run with:  pytest backend/tests/test_api.py

NOTE FROM THE AUTHOR: these tests were written but could NOT be executed in the environment where this
project was generated (FastAPI/SQLAlchemy could not be installed there). Run them on your computer with
scripts\\test.bat and fix anything that fails. The speech model is never loaded: transcription is replaced
by a fake so the tests are fast and need no download (that part is MOCKED; uploads and exports use real FFmpeg).
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

_tmp = tempfile.mkdtemp(prefix="garuda-test-")
os.environ["GARUDA_DATA_DIR"] = _tmp
os.environ["APP_ENV"] = "development"
os.environ["WHISPER_MODEL"] = "base"
os.environ["WHISPER_DEVICE"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app import diagnostics, jobs, main, transcribe  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    main.settings.rate_limit_per_minute = 10000
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("v") / "sample.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(path)], check=True)
    return path


@pytest.fixture()
def fake_transcribe(monkeypatch):
    def fake(audio, **kw):
        kw["on_progress"](0.5)
        return {"language": "en", "warnings": [], "words": [
            {"text": "Hello", "start": 0.2, "end": 0.6, "kind": "word", "confidence": 0.9},
            {"text": "world.", "start": 0.7, "end": 1.2, "kind": "word", "confidence": 0.9}]}
    monkeypatch.setattr(transcribe, "transcribe", fake)


def upload(client, path: Path, name="sample.mp4"):
    with open(path, "rb") as fh:
        return client.post("/api/projects", files={"file": (name, fh, "video/mp4")})


def wait(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_health_and_ready(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/ready").status_code in (200, 503)


def test_upload_rejects_fake_video(client, tmp_path):
    bad = tmp_path / "fake.mp4"
    bad.write_bytes(b"not a video" * 1000)
    r = upload(client, bad)
    assert r.status_code in (415, 422)
    assert "error" in r.json()


def test_upload_rejects_empty(client, tmp_path):
    empty = tmp_path / "e.mp4"
    empty.write_bytes(b"")
    assert upload(client, empty).status_code == 400


def test_path_traversal_name_is_sanitised(client, video):
    r = upload(client, video, name="../../etc/passwd.mp4")
    assert r.status_code == 201
    assert "/" not in r.json()["name"] and ".." not in r.json()["name"]


def test_unknown_and_malformed_ids_are_404(client):
    assert client.get("/api/projects/" + "0" * 32).status_code == 404
    assert client.get("/api/projects/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/exports/" + "0" * 32 + "/video").status_code == 404


def test_caption_validation(client, video):
    pid = upload(client, video).json()["id"]
    url = f"/api/projects/{pid}/captions"
    assert client.put(url, json={"words": [{"text": "a", "start": 2, "end": 1}]}).status_code == 422
    assert client.put(url, json={"words": [{"text": "a", "start": -1, "end": 1}]}).status_code == 422
    assert client.put(url, json={"words": [{"text": "", "start": 0, "end": 1}]}).status_code == 422
    assert client.put(url, json={"words": [{"text": "a", "start": 0, "end": 1}], "style": {"color": "red"}}).status_code == 422
    assert client.put(url, json={"words": [{"text": "a", "start": 99, "end": 100}]}).status_code == 422  # after video end
    ok = client.put(url, json={"words": [{"text": "ನಮಸ್ಕಾರ", "start": 0, "end": 1}]})
    assert ok.status_code == 200
    assert client.get(f"/api/projects/{pid}").json()["words"][0]["text"] == "ನಮಸ್ಕಾರ"


def test_export_without_captions_refused(client, video):
    pid = upload(client, video).json()["id"]
    assert client.post(f"/api/projects/{pid}/export", json={}).status_code == 409


def test_export_option_validation(client, video):
    pid = upload(client, video).json()["id"]
    assert client.post(f"/api/projects/{pid}/export", json={"aspect": "4:3"}).status_code == 422
    assert client.post(f"/api/projects/{pid}/export", json={"resolution": 99999}).status_code == 422


def test_full_workflow_with_mocked_transcription(client, video, fake_transcribe):
    original_bytes = video.read_bytes()
    pid = upload(client, video).json()["id"]

    job = client.post(f"/api/projects/{pid}/transcribe", json={"language": "en", "model": "tiny"}).json()
    assert wait(client, job["id"])["status"] == "completed"
    proj = client.get(f"/api/projects/{pid}").json()
    assert [w["text"] for w in proj["words"]] == ["Hello", "world."]

    words = proj["words"]
    words[0]["text"] = "Howdy"  # the user edits a word
    assert client.put(f"/api/projects/{pid}/captions", json={"words": words, "style": proj["style"]}).status_code == 200
    assert "Howdy" in client.get(f"/api/projects/{pid}/subtitles.srt").text

    job = client.post(f"/api/projects/{pid}/export", json={"aspect": "9:16", "resolution": 480, "quality": "draft"}).json()
    done = wait(client, job["id"])
    assert done["status"] == "completed", done
    eid = done["result"]["export_id"]
    mp4 = client.get(f"/api/exports/{eid}/video")
    assert mp4.status_code == 200 and mp4.content[:4] != b""
    assert client.get(f"/api/exports/{eid}/srt").status_code == 200
    assert client.get(f"/api/exports/{eid}/vtt").text.startswith("WEBVTT")
    # the original upload is still byte-identical on disk
    media = client.get(f"/api/projects/{pid}/media")
    assert media.content == original_bytes


def test_duplicate_job_rejected(client, video, monkeypatch):
    import threading
    gate = threading.Event()

    def slow(audio, **kw):
        gate.wait(10)
        return {"language": "en", "warnings": [], "words": []}
    monkeypatch.setattr(transcribe, "transcribe", slow)
    pid = upload(client, video).json()["id"]
    first = client.post(f"/api/projects/{pid}/transcribe", json={})
    assert first.status_code == 202
    second = client.post(f"/api/projects/{pid}/transcribe", json={})
    assert second.status_code == 409
    assert client.post(f"/api/jobs/{first.json()['id']}/cancel").status_code == 200
    gate.set()


def test_delete_project_removes_files(client, video):
    pid = upload(client, video).json()["id"]
    folder = jobs.project_dir(pid)
    assert folder.exists()
    assert client.delete(f"/api/projects/{pid}").status_code == 204
    assert not folder.exists()
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_invalid_language_rejected(client, video):
    pid = upload(client, video).json()["id"]
    assert client.post(f"/api/projects/{pid}/transcribe", json={"language": "xx"}).status_code == 422


# ---------------------------------------------------------------------------------------------------------
# Added in the speech-recognition repair. NOTE: like the rest of this file these need FastAPI installed and
# could not be run in the sandbox where they were written; run scripts\\test.bat on your computer.
# ---------------------------------------------------------------------------------------------------------
def test_diagnostics_endpoint_reports_versions_and_settings(client):
    body = client.get("/api/diagnostics").json()
    assert body["settings"]["model"] == "base" and body["settings"]["device"] == "cpu"
    assert "faster-whisper" in body["packages"] and "ffmpeg" in body["tools"]
    assert "Traceback" not in str(body)


def test_ttc_font_can_be_listed_and_served(client, tmp_path, monkeypatch):
    from fontTools.ttLib import TTCollection, TTFont

    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    collection = TTCollection()
    collection.fonts = [
        TTFont(str(main.settings.fonts_dir / "DejaVuSans.ttf")),
        TTFont(str(main.settings.fonts_dir / "DejaVuSans-Bold.ttf")),
    ]
    try:
        collection.save(str(fonts_dir / "dejavu.ttc"))
    finally:
        collection.close()
    monkeypatch.setattr(main.settings, "fonts_dir", fonts_dir)

    listed = client.get("/api/fonts").json()["fonts"]
    assert any(font["family"] == "DejaVu Sans" and font["file"] == "dejavu.ttc" for font in listed)
    served = client.get("/api/fonts/file/dejavu.ttc")
    assert served.status_code == 200 and served.content == (fonts_dir / "dejavu.ttc").read_bytes()


def test_deep_diagnostics_resolves_kannada_model_and_options(monkeypatch):
    seen = {"model": None, "options": []}

    class FakeModel:
        def transcribe(self, audio, language=None, task=None, beam_size=None, word_timestamps=False, vad_filter=False,
                       initial_prompt=None, condition_on_previous_text=False, multilingual=False):
            options = {"language": language, "task": task, "beam_size": beam_size, "word_timestamps": word_timestamps,
                       "vad_filter": vad_filter, "initial_prompt": initial_prompt,
                       "condition_on_previous_text": condition_on_previous_text, "multilingual": multilingual}
            seen["options"].append(options)
            return iter(()), type("Info", (), {"duration": 1.0, "language": "kn"})()

    def fake_get_model(source, **kwargs):
        seen["model"] = source
        return FakeModel(), False

    monkeypatch.setattr(diagnostics.transcribe, "resolve_runtime", lambda *_: ("cpu", "int8", []))
    monkeypatch.setattr(diagnostics.transcribe, "get_model", fake_get_model)
    monkeypatch.setattr(diagnostics.transcribe, "load_audio_array", lambda _: [0.0])
    monkeypatch.setattr(diagnostics.transcribe, "consume_segments", lambda *_args, **_kwargs: ([], {}))

    result = diagnostics.deep_test("kn-medium", audio=Path("unused.wav"), language="kn-en")

    assert result["ok"]
    assert seen["model"] == "itsskofficial/livewhisper-kn-medium"
    assert all(options["language"] == "kn" for options in seen["options"])
    assert all("English" in options["initial_prompt"] for options in seen["options"])


def test_model_on_disk_requires_complete_model_weights(monkeypatch, tmp_path):
    root = tmp_path / "models"
    snapshot = root / "models--itsskofficial--livewhisper-kn-medium" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (root / "models--itsskofficial--livewhisper-kn-medium" / "blobs").mkdir()
    (root / "models--itsskofficial--livewhisper-kn-medium" / "blobs" / "partial.incomplete").write_bytes(b"partial")
    monkeypatch.setattr(diagnostics.settings, "whisper_model_dir", root)

    assert not diagnostics._model_on_disk("kn-medium")
    (snapshot / "model.bin").write_bytes(b"complete")
    assert diagnostics._model_on_disk("kn-medium")


def test_options_offer_kannada_english_and_mixed(client):
    body = client.get("/api/options").json()
    for code in ("auto", "kn", "en", "kn-en", "kanglish"):
        assert code in body["languages"]
    assert body["default_model"] == "base"

def test_options_hide_indic_until_model_and_runner_are_configured(client, tmp_path, monkeypatch):
    model = tmp_path / "indic.nemo"
    model.write_bytes(b"model")
    monkeypatch.setattr(main.settings, "indic_model", str(model))
    monkeypatch.setattr(main.settings, "indic_runner", "")
    assert "indic" not in client.get("/api/options").json()["models"]
    monkeypatch.setattr(main.settings, "indic_runner", sys.executable)
    assert "indic" in client.get("/api/options").json()["models"]


def test_mixed_language_is_accepted_and_reaches_the_transcriber(client, video, monkeypatch):
    seen = {}

    def fake(audio, **kw):
        seen.update(kw)
        kw["on_progress"](0.5)
        return {"language": "kn", "warnings": [], "engine": {"timing": "word"},
                "words": [{"text": "ನಮಸ್ಕಾರ", "start": 0.2, "end": 0.6, "kind": "word", "confidence": 0.9}]}
    monkeypatch.setattr(transcribe, "transcribe", fake)
    pid = upload(client, video).json()["id"]
    job = client.post(f"/api/projects/{pid}/transcribe", json={"language": "kn-en", "model": "tiny"}).json()
    assert wait(client, job["id"])["status"] == "completed"
    assert seen["language"] == "kn-en" and seen["model_size"] == "tiny" and seen["device"] == "cpu"
    assert client.get(f"/api/projects/{pid}").json()["words"][0]["text"] == "ನಮಸ್ಕಾರ"


def test_transcription_failure_shows_a_safe_message_and_never_a_traceback(client, video, monkeypatch):
    def broken(audio, **kw):
        raise transcribe.TranscribeError("Speech recognition failed (TypeError). The technical details were saved in data\\logs\\garuda.log.",
                                         "decode_failed")
    monkeypatch.setattr(transcribe, "transcribe", broken)
    pid = upload(client, video).json()["id"]
    job = client.post(f"/api/projects/{pid}/transcribe", json={"language": "kn", "model": "base"}).json()
    done = wait(client, job["id"])
    assert done["status"] == "failed"
    assert "TypeError" in done["error"] and "Traceback" not in done["error"] and "File \"" not in done["error"]
    assert client.get(f"/api/projects/{pid}").json()["caption_count"] == 0, "a failed job must not leave fake captions"


def test_no_speech_is_reported_as_failure_not_success(client, video, monkeypatch):
    def silent(audio, **kw):
        raise transcribe.TranscribeError("No speech was detected in this video's audio.", "no_speech")
    monkeypatch.setattr(transcribe, "transcribe", silent)
    pid = upload(client, video).json()["id"]
    job = client.post(f"/api/projects/{pid}/transcribe", json={"language": "en", "model": "base"}).json()
    assert wait(client, job["id"])["status"] == "failed"


def test_work_folder_is_removed_after_a_failed_job(client, video, monkeypatch):
    def broken(audio, **kw):
        raise transcribe.TranscribeError("boom", "decode_failed")
    monkeypatch.setattr(transcribe, "transcribe", broken)
    pid = upload(client, video).json()["id"]
    job = client.post(f"/api/projects/{pid}/transcribe", json={"language": "en", "model": "base"}).json()
    wait(client, job["id"])
    assert not (jobs.project_dir(pid) / "work").exists()
    assert any((jobs.project_dir(pid) / "original").iterdir()), "the original upload must be untouched"

"""FFmpeg / FFprobe helpers. Standard library only.

Every external command is built as an argument list and run WITHOUT a shell,
so a hostile filename or caption can never become a shell command.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path


log = logging.getLogger("garuda.media")


class MediaError(Exception):
    """A problem with the media that is safe to show to the user."""

    def __init__(self, message: str, code: str = "invalid_media"):
        super().__init__(message)
        self.message = message
        self.code = code


# Containers we accept (matched against ffprobe's comma-separated format_name).
ALLOWED_CONTAINERS = {"mov", "mp4", "matroska", "webm", "avi", "mpegts", "flv", "3gp", "m4a"}
# Video codecs that are really a single still picture, not video.
STILL_IMAGE_CODECS = {"mjpeg", "png", "bmp", "gif", "tiff", "webp"}


def tool_paths() -> dict[str, str | None]:
    return {"ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}


def require_tools() -> None:
    missing = [name for name, path in tool_paths().items() if not path]
    if missing:
        raise MediaError(
            f"{' and '.join(missing)} not found. Install FFmpeg and make sure it is on your PATH "
            "(see docs/WINDOWS_SETUP.md).",
            code="ffmpeg_missing",
        )


@dataclass
class MediaInfo:
    container: str
    extension: str
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    audio_channels: int
    size_bytes: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _fps(rate: str | None) -> float:
    try:
        num, _, den = (rate or "0/1").partition("/")
        den_f = float(den or 1)
        return round(float(num) / den_f, 3) if den_f else 0.0
    except ValueError:
        return 0.0


def probe(path: Path, timeout: int = 60) -> dict:
    """Run ffprobe and return its JSON. Raises MediaError if the file is unreadable."""
    require_tools()
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MediaError("Reading the video took too long. The file may be damaged.", "probe_timeout") from exc
    if proc.returncode != 0:
        raise MediaError("This file is not a readable video (it may be corrupted or not a video).", "unreadable")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError("Could not understand the video information.", "unreadable") from exc


def validate_probe(
    data: dict, size_bytes: int, *, max_duration: int, max_width: int, max_height: int
) -> MediaInfo:
    """Turn ffprobe output into MediaInfo, or raise MediaError with a helpful message."""
    fmt = data.get("format") or {}
    names = {n.strip() for n in str(fmt.get("format_name", "")).split(",") if n.strip()}
    if not names & ALLOWED_CONTAINERS:
        raise MediaError(
            f"Unsupported file type ({fmt.get('format_name', 'unknown')}). Please use MP4, MOV, WebM, MKV or AVI.",
            "unsupported_container",
        )

    streams = data.get("streams") or []
    video = next(
        (s for s in streams if s.get("codec_type") == "video"
         and s.get("codec_name") not in STILL_IMAGE_CODECS
         and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise MediaError("No video stream found in this file.", "no_video")
    if audio is None:
        raise MediaError(
            "This video has no audio track, so there is no speech to caption. Please upload a video with sound.",
            "no_audio",
        )

    try:
        duration = float(fmt.get("duration") or video.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        raise MediaError("Could not determine the video length. The file may be damaged.", "bad_duration")
    if duration > max_duration:
        raise MediaError(
            f"Video is too long ({duration / 60:.1f} min). The limit is {max_duration / 60:.0f} minutes.",
            "too_long",
        )

    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise MediaError("Could not read the video size.", "bad_dimensions")
    if width > max_width or height > max_height:
        raise MediaError(f"Video resolution {width}x{height} is above the limit {max_width}x{max_height}.", "too_large_res")

    # Rotation metadata (phone videos): swap sizes so the app sees what the viewer sees.
    rotation = 0
    for side in video.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(abs(float(side["rotation"]))) % 360
    rotation = int((video.get("tags") or {}).get("rotate", rotation) or rotation) % 360
    if rotation in (90, 270):
        width, height = height, width

    if "webm" in names:
        ext = ".webm"
    elif "matroska" in names:
        ext = ".mkv"
    elif "avi" in names:
        ext = ".avi"
    elif "mov" in names or "mp4" in names:
        ext = ".mp4"
    else:
        ext = ".mp4"

    return MediaInfo(
        container=",".join(sorted(names)),
        extension=ext,
        duration=round(duration, 3),
        width=width,
        height=height,
        fps=_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        video_codec=str(video.get("codec_name", "")),
        audio_codec=str(audio.get("codec_name", "")),
        audio_channels=int(audio.get("channels") or 0),
        size_bytes=size_bytes,
    )


def inspect(path: Path, *, max_duration: int, max_width: int, max_height: int, timeout: int = 60) -> MediaInfo:
    return validate_probe(
        probe(path, timeout), path.stat().st_size,
        max_duration=max_duration, max_width=max_width, max_height=max_height,
    )


# Speech-friendly volume levelling: remove rumble below 70 Hz, then bring the loudness to a standard level.
NORMALIZE_FILTER = "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11"


def extract_audio(src: Path, dst: Path, timeout: int = 3600, normalize: bool = True) -> None:
    """Extract mono 16 kHz WAV (what speech models expect). Does not touch `src`.

    With normalize=True the volume is levelled first; if that filter ever fails, extraction is retried
    without it, so normalisation can never be the reason a video cannot be captioned.
    """
    require_tools()
    base = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000"]
    attempts: list[str | None] = ([NORMALIZE_FILTER] if normalize else []) + [None]
    for audio_filter in attempts:
        cmd = base + (["-af", audio_filter] if audio_filter else []) + ["-c:a", "pcm_s16le", str(dst)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            dst.unlink(missing_ok=True)
            raise MediaError("Extracting the audio took too long.", "audio_timeout") from exc
        if proc.returncode == 0 and dst.is_file() and dst.stat().st_size >= 1000:
            if audio_filter is None and normalize:
                log.warning("Audio was extracted WITHOUT volume normalisation (the filter failed first).")
            return
        # The real FFmpeg message goes to the log (never to the web page).
        log.warning("FFmpeg audio extraction failed (filter=%s, exit=%s): %s", audio_filter, proc.returncode,
                    (proc.stderr or "").strip()[-600:])
        dst.unlink(missing_ok=True)
    raise MediaError("Could not extract audio from this video.", "audio_extract_failed")


def wav_duration(path: Path) -> float | None:
    """Length in seconds of a PCM WAV file, or None if it cannot be read."""
    try:
        with wave.open(str(path), "rb") as fh:
            rate = fh.getframerate()
            return fh.getnframes() / rate if rate else None
    except (wave.Error, EOFError, OSError):
        return None

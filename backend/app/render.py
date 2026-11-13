"""Export: build FFmpeg commands, run them with live progress, and verify the result.

Standard library only. FFmpeg is always started with an argument list (never a shell string).
The original upload is only ever READ; every output goes to a separate directory.
"""
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import captions as cap
from .media import MediaError, MediaInfo, probe, require_tools

ASPECTS = ("original", "9:16", "16:9", "1:1")
QUALITY = {  # name -> (x264 crf, x264 preset)
    "draft": (28, "veryfast"),
    "standard": (23, "fast"),
    "high": (18, "medium"),
}
RESOLUTIONS = (360, 480, 720, 1080, 1440)  # size of the SHORT side in pixels


class ExportError(Exception):
    def __init__(self, message: str, code: str = "export_failed"):
        super().__init__(message)
        self.message = message
        self.code = code


class Cancelled(Exception):
    pass


def _even(n: float) -> int:
    return max(2, int(round(n / 2.0)) * 2)


def output_size(info: MediaInfo, aspect: str, resolution: int) -> tuple[int, int]:
    """Pick the output frame size. `resolution` is the short side; never upscales 'original'."""
    if aspect not in ASPECTS:
        raise ExportError(f"Unsupported aspect ratio: {aspect}", "bad_options")
    if aspect == "original":
        short = min(info.width, info.height)
        scale = min(1.0, resolution / short)
        return _even(info.width * scale), _even(info.height * scale)
    ratios = {"9:16": (9, 16), "16:9": (16, 9), "1:1": (1, 1)}
    rw, rh = ratios[aspect]
    unit = resolution / min(rw, rh)
    return _even(rw * unit), _even(rh * unit)


def _filter_value(path: str) -> str:
    """Escape a path for use as an FFmpeg filter option value (works for Windows C:\\ paths too)."""
    p = path.replace("\\", "/").replace("'", r"\'")
    return p.replace(":", r"\:")


def build_burn_command(
    *, src: Path, ass_name: str, fonts_dir: Path, dst: Path, width: int, height: int,
    quality: str, fps: float | None = None,
) -> list[str]:
    """FFmpeg argument list that scales/pads to width x height and burns in ass_name.

    `ass_name` must be a plain file name that exists in the command's working directory.
    """
    if quality not in QUALITY:
        raise ExportError(f"Unsupported quality preset: {quality}", "bad_options")
    crf, preset = QUALITY[quality]
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        f"ass=filename='{_filter_value(ass_name)}':fontsdir='{_filter_value(str(fonts_dir))}'"
    )
    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostdin", "-v", "error", "-progress", "pipe:1", "-nostats",
           "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf]
    if fps:
        cmd += ["-r", f"{fps:g}"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)]
    return cmd


@dataclass
class ExportResult:
    mp4: Path
    srt: Path
    vtt: Path
    width: int
    height: int
    duration: float
    warnings: list[str]


def run_ffmpeg(
    cmd: list[str], *, cwd: Path, total_s: float, on_progress: Callable[[float], None],
    cancel: threading.Event, set_proc: Callable[[subprocess.Popen | None], None] | None = None,
    timeout: int = 6 * 3600,
) -> None:
    require_tools()
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if set_proc:
        set_proc(proc)
    timer = threading.Timer(timeout, proc.kill)
    timer.start()
    stderr_chunks: list[str] = []
    err_thread = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read() if proc.stderr else ""), daemon=True)
    err_thread.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel.is_set():
                proc.terminate()
                raise Cancelled()
            key, _, value = line.strip().partition("=")
            if key in ("out_time_us", "out_time_ms") and value.lstrip("-").isdigit() and total_s > 0:
                on_progress(min(0.99, max(0.0, int(value) / 1_000_000 / total_s)))
        code = proc.wait()
        err_thread.join(timeout=2)
        if cancel.is_set():
            raise Cancelled()
        if code != 0:
            tail = (stderr_chunks[0] if stderr_chunks else "").strip().splitlines()[-3:]
            raise ExportError("FFmpeg could not render the video. " + " ".join(tail)[:300], "ffmpeg_failed")
    finally:
        timer.cancel()
        timer.join()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        err_thread.join()
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
        if set_proc:
            set_proc(None)


def verify_output(path: Path, source: MediaInfo, *, tolerance_s: float = 1.0) -> MediaInfo:
    """Check the exported file really exists, has video AND audio, and matches the source length."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ExportError("The export finished but the output file is missing.", "output_missing")
    data = probe(path)
    kinds = {s.get("codec_type") for s in data.get("streams", [])}
    if "video" not in kinds:
        raise ExportError("The exported file has no video stream.", "output_invalid")
    if "audio" not in kinds:
        raise ExportError("The exported file lost its audio.", "output_invalid")
    try:
        duration = float(data["format"]["duration"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ExportError("The exported file has no valid duration.", "output_invalid") from exc
    if abs(duration - source.duration) > tolerance_s:
        raise ExportError(
            f"Exported length ({duration:.2f}s) does not match the original ({source.duration:.2f}s).",
            "output_invalid",
        )
    v = next(s for s in data["streams"] if s.get("codec_type") == "video")
    return MediaInfo(
        container="mp4", extension=".mp4", duration=duration, width=int(v["width"]), height=int(v["height"]),
        fps=0.0, video_codec=v.get("codec_name", ""),
        audio_codec=next(s for s in data["streams"] if s.get("codec_type") == "audio").get("codec_name", ""),
        audio_channels=0, size_bytes=path.stat().st_size,
    )


def export_project(
    *, src: Path, info: MediaInfo, words: list[cap.Word], style: dict, out_dir: Path, fonts_dir: Path,
    aspect: str, resolution: int, quality: str, fps: float | None,
    on_progress: Callable[[float], None], cancel: threading.Event,
    set_proc: Callable[[subprocess.Popen | None], None] | None = None,
) -> ExportResult:
    """Render captions.mp4 + captions.srt + captions.vtt into out_dir (a new folder)."""
    if not words:
        raise ExportError("There are no captions to export. Transcribe the video or add captions first.", "no_captions")
    if resolution not in RESOLUTIONS:
        raise ExportError(f"Unsupported resolution: {resolution}", "bad_options")
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = output_size(info, aspect, resolution)
    st = cap.merged_style(style)

    ass = cap.to_ass(words, st, width, height)
    (out_dir / "captions.ass").write_text(ass.text, encoding="utf-8")
    (out_dir / "captions.srt").write_text(cap.to_srt(words, st), encoding="utf-8")
    (out_dir / "captions.vtt").write_text(cap.to_vtt(words, st), encoding="utf-8")

    dst = out_dir / "captioned.mp4"
    cmd = build_burn_command(
        src=src, ass_name="captions.ass", fonts_dir=fonts_dir, dst=dst, width=width, height=height,
        quality=quality, fps=fps,
    )
    run_ffmpeg(cmd, cwd=out_dir, total_s=info.duration, on_progress=on_progress, cancel=cancel, set_proc=set_proc)
    out_info = verify_output(dst, info)
    return ExportResult(
        mp4=dst, srt=out_dir / "captions.srt", vtt=out_dir / "captions.vtt",
        width=out_info.width, height=out_info.height, duration=out_info.duration, warnings=ass.warnings,
    )

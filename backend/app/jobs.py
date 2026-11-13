"""Background jobs: a small local worker pool with status, progress, cancel and crash recovery."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select

from . import captions as cap
from . import media, render, transcribe
from .config import settings
from .db import Export, Job, Project, SessionLocal, new_id

log = logging.getLogger("garuda.jobs")
_pool = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs, thread_name_prefix="garuda-job")
_cancel: dict[str, threading.Event] = {}
_procs: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()

ACTIVE = ("queued", "running")


def project_dir(project_id: str):
    return settings.projects_dir / project_id


def recover_interrupted() -> None:
    """On startup any job still queued/running was killed by a restart: mark it failed (truthfully)."""
    with SessionLocal() as s:
        for job in s.scalars(select(Job).where(Job.status.in_(ACTIVE))):
            job.status, job.error = "failed", "The app was restarted while this job was running. Please start it again."
        s.commit()


def _update(job_id: str, **fields) -> None:
    with SessionLocal() as s:
        job = s.get(Job, job_id)
        if job and (job.status != "cancelled" or fields.get("status") == "cancelled"):
            for k, v in fields.items():
                setattr(job, k, v)
            s.commit()


def create_job(project_id: str, kind: str, params: dict) -> Job | None:
    """Create a queued job. Returns None if the project already has an active job of that kind."""
    with SessionLocal() as s:
        busy = s.scalars(select(Job).where(Job.project_id == project_id, Job.kind == kind, Job.status.in_(ACTIVE))).first()
        if busy:
            return None
        job = Job(id=new_id(), project_id=project_id, kind=kind, params=json.dumps(params), message="Waiting to start")
        s.add(job)
        s.commit()
        s.refresh(job)
    with _lock:
        _cancel[job.id] = threading.Event()
    runner = _run_transcribe if kind == "transcribe" else _run_export
    _pool.submit(runner, job.id)
    return job


def cancel_job(job_id: str) -> bool:
    with _lock:
        ev = _cancel.get(job_id)
        proc = _procs.get(job_id)
    with SessionLocal() as s:
        job = s.get(Job, job_id)
        if not job or job.status not in ACTIVE:
            return False
        job.status, job.message = "cancelled", "Cancelled"
        s.commit()
    if ev:
        ev.set()
    if proc and proc.poll() is None:
        proc.terminate()
    return True


def _run_transcribe(job_id: str) -> None:
    ev = _cancel[job_id]
    if ev.is_set():
        return
    tmp = None
    try:
        with SessionLocal() as s:
            job = s.get(Job, job_id)
            project = s.get(Project, job.project_id)
            params = json.loads(job.params)
            src = project_dir(project.id) / "original" / project.media_file
            pid = project.id
        _update(job_id, status="running", message="Extracting audio", progress=0.02)
        tmp = project_dir(pid) / "work"
        tmp.mkdir(parents=True, exist_ok=True)
        wav = tmp / "audio.wav"
        media.extract_audio(src, wav, normalize=settings.audio_normalize)
        _update(job_id, message="Loading speech model (first run may download it)", progress=0.05)
        result = transcribe.transcribe(
            wav, language=params["language"], model_size=params["model"], device=settings.whisper_device,
            compute_type=settings.whisper_compute_type, model_dir=settings.whisper_model_dir,
            allow_download=settings.allow_model_download, cpu_threads=settings.whisper_cpu_threads,
            beam_size=settings.whisper_beam_size, idle_release_s=settings.whisper_idle_release_s,
            include_detail=settings.show_error_details, custom_model=settings.custom_model or None,
            vocabulary=settings.load_vocabulary(), indic=settings.indic_config(),
            on_progress=lambda p: _update(job_id, progress=0.05 + p * 0.93, message="Recognising speech"), cancel=ev)
        log.info("Transcription finished for project %s: %s", pid, result.get("engine"))
        with SessionLocal() as s:
            project = s.get(Project, pid)
            project.words = json.dumps(result["words"], ensure_ascii=False)
            project.language = result["language"]
            project.warnings = json.dumps(result["warnings"], ensure_ascii=False)
            s.commit()
        _update(job_id, status="completed", progress=1.0, message="Done", result=json.dumps(
            {"words": len(result["words"]), "language": result["language"], "warnings": result["warnings"],
             "engine": result.get("engine", {})}))
    except transcribe.Cancelled:
        _update(job_id, status="cancelled", message="Cancelled")
    except (transcribe.TranscribeError, media.MediaError) as exc:
        # Keep the real cause (including the original exception chain) in the log; the page gets the safe message.
        log.error("Transcription job %s failed [%s]: %s", job_id, getattr(exc, "code", "?"), exc.message, exc_info=exc)
        _update(job_id, status="failed", error=exc.message, message="Failed")
    except Exception as exc:  # never leak internals to the user
        log.exception("Unexpected error in transcription job %s", job_id)
        _update(job_id, status="failed", error=f"Unexpected error ({type(exc).__name__}). {transcribe.LOG_HINT}", message="Failed")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        with _lock:
            _cancel.pop(job_id, None)


def _run_export(job_id: str) -> None:
    ev = _cancel[job_id]
    if ev.is_set():
        return
    out_dir = None
    try:
        with SessionLocal() as s:
            job = s.get(Job, job_id)
            project = s.get(Project, job.project_id)
            params = json.loads(job.params)
            src = project_dir(project.id) / "original" / project.media_file
            info = media.MediaInfo(**project.get_info())
            words = cap.to_words(project.get_words())
            style = project.get_style()
            pid = project.id
        _update(job_id, status="running", message="Rendering video", progress=0.01)
        export_id = new_id()
        out_dir = project_dir(pid) / "exports" / export_id

        def set_proc(p):
            with _lock:
                if p is None:
                    _procs.pop(job_id, None)
                else:
                    _procs[job_id] = p

        res = render.export_project(
            src=src, info=info, words=words, style=style, out_dir=out_dir, fonts_dir=settings.fonts_dir,
            aspect=params["aspect"], resolution=params["resolution"], quality=params["quality"], fps=params.get("fps"),
            on_progress=lambda p: _update(job_id, progress=p, message="Rendering video"), cancel=ev, set_proc=set_proc)
        with SessionLocal() as s:
            s.add(Export(id=export_id, project_id=pid, job_id=job_id, folder=export_id, width=res.width,
                         height=res.height, duration=res.duration, options=json.dumps(params)))
            s.commit()
        _update(job_id, status="completed", progress=1.0, message="Export ready", result=json.dumps(
            {"export_id": export_id, "width": res.width, "height": res.height, "warnings": res.warnings}))
    except render.Cancelled:
        _update(job_id, status="cancelled", message="Cancelled")
        if out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)
    except (render.ExportError, media.MediaError) as exc:
        log.error("Export job %s failed: %s", job_id, exc.message, exc_info=exc)
        _update(job_id, status="failed", error=exc.message, message="Failed")
        if out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)
    except Exception as exc:
        log.exception("Unexpected error in export job %s", job_id)
        _update(job_id, status="failed", error=f"Unexpected error ({type(exc).__name__}). {transcribe.LOG_HINT}", message="Failed")
        if out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)
    finally:
        with _lock:
            _cancel.pop(job_id, None)
            _procs.pop(job_id, None)

"""GARUDA CAPTIONS AI - FastAPI application.

Local mode: everything (speech recognition, rendering) runs on this computer. Nothing is uploaded anywhere.
There is no login in local mode: project ids are random 128-bit values and the server listens on
127.0.0.1 only. Add authentication before exposing this to the internet (see docs/DEPLOYMENT.md).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import captions as cap
from . import fonts as fontlib
from . import diagnostics, jobs, media, transcribe
from .config import settings
from .logging_setup import setup_logging
from .db import Export, Job, Project, get_session, init_db, new_id
from .schemas import CaptionsIn, ExportIn, RenameIn, StyleIn, TranscribeIn

log = logging.getLogger("garuda")
LOG_FILE = setup_logging(settings.log_dir, settings.log_level)

ID_RE = re.compile(r"^[0-9a-f]{32}$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    problems = settings.validate_for_production()
    if problems:
        raise RuntimeError("Unsafe production configuration: " + " ".join(problems))
    init_db()
    jobs.recover_interrupted()
    log.info("GARUDA CAPTIONS AI starting. Log file: %s", LOG_FILE)
    if not all(media.tool_paths().values()):
        log.warning("FFmpeg/FFprobe not found on PATH. Uploads and exports will fail until it is installed.")
    yield


app = FastAPI(title="GARUDA CAPTIONS AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type"],
)

# ------------------------------------------------------------------ protection middleware

_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api"):
        # simple per-client rate limit
        ip = request.client.host if request.client else "?"
        q = _hits[ip]
        t = time.monotonic()
        while q and t - q[0] > 60:
            q.popleft()
        if len(q) >= settings.rate_limit_per_minute:
            return JSONResponse({"error": {"code": "rate_limited", "message": "Too many requests. Please wait a moment."}}, 429)
        q.append(t)
        # JSON bodies are small; only the upload route may be large
        is_upload = request.method == "POST" and path == "/api/projects"
        length = request.headers.get("content-length")
        limit = settings.max_upload_bytes + 1024 * 1024 if is_upload else settings.max_json_kb * 1024
        if length and length.isdigit() and int(length) > limit:
            return JSONResponse({"error": {"code": "too_large", "message": "The request is too large."}}, 413)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


def err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@app.exception_handler(HTTPException)
async def http_exc(_: Request, exc: HTTPException):
    d = exc.detail if isinstance(exc.detail, dict) else {"code": "error", "message": str(exc.detail)}
    return JSONResponse({"error": d}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exc(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
    msg = f"{loc}: {first.get('msg', 'invalid value')}" if loc else first.get("msg", "invalid request")
    return JSONResponse({"error": {"code": "invalid_request", "message": msg}}, status_code=422)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse({"error": {"code": "server_error", "message": "Something went wrong on the server."}}, status_code=500)


# ------------------------------------------------------------------ helpers

def get_project(project_id: str, db: Session) -> Project:
    if not ID_RE.match(project_id):
        raise err(404, "not_found", "Project not found.")
    project = db.get(Project, project_id)
    if not project:
        raise err(404, "not_found", "Project not found.")
    return project


def project_out(p: Project, full: bool = False) -> dict:
    out = {
        "id": p.id, "name": p.name, "language": p.language, "created_at": p.created_at.isoformat(),
        "updated_at": p.updated_at.isoformat(), "media": p.get_info(), "caption_count": len(p.get_words()),
    }
    if full:
        out.update({"words": p.get_words(), "style": cap.merged_style(p.get_style()), "warnings": json.loads(p.warnings or "[]"),
                    "exports": [{"id": e.id, "width": e.width, "height": e.height, "created_at": e.created_at.isoformat()}
                                for e in sorted(p.exports, key=lambda e: e.created_at, reverse=True)]})
    return out


def job_out(j: Job) -> dict:
    return {"id": j.id, "project_id": j.project_id, "kind": j.kind, "status": j.status, "progress": round(j.progress, 3),
            "message": j.message, "error": j.error, "where": j.where, "result": json.loads(j.result or "{}")}


def safe_title(filename: str) -> str:
    stem = Path(filename or "video").name.rsplit(".", 1)[0]
    stem = re.sub(r"[^\w .\-()]", "", stem, flags=re.UNICODE).strip()[:120]
    return stem or "Untitled video"


# ------------------------------------------------------------------ system

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/ready")
def ready():
    tools = media.tool_paths()
    try:
        import faster_whisper  # noqa: F401
        whisper = True
    except ImportError:
        whisper = False
    ok = all(tools.values())
    body = {"ready": ok, "ffmpeg": bool(tools["ffmpeg"]), "ffprobe": bool(tools["ffprobe"]), "faster_whisper": whisper,
            "processing": "local", "max_upload_mb": settings.max_upload_mb}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/api/diagnostics")
def api_diagnostics():
    """Quick local health report (no model is loaded). Run scripts\\diagnose.bat for the deep test."""
    return diagnostics.collect(deep=False)


@app.get("/api/options")
def options():
    return {"languages": transcribe.LANGUAGES, "models": transcribe.model_choices(settings.whisper_model_dir, settings.custom_model, settings.indic_configured), "default_model": settings.whisper_default_model,
            "weak_languages": sorted(transcribe.WEAK_LANGS), "processing": "local",
            "notes": ["Everything runs on this computer. Videos are never uploaded.", transcribe.UNSUPPORTED_NOTE]}


@app.get("/api/fonts")
def list_fonts():
    return {"fonts": fontlib.list_fonts(settings.fonts_dir)}


@app.get("/api/fonts/file/{name}")
def font_file(name: str):
    path = (settings.fonts_dir / Path(name).name)
    if path.suffix.lower() not in (".ttf", ".otf", ".ttc") or not path.is_file():
        raise err(404, "not_found", "Font not found.")
    return FileResponse(path)


@app.post("/api/glyph-check")
def glyph_check(payload: dict):
    text = str(payload.get("text", ""))[:200000]
    family = str(payload.get("font_family", "Noto Sans"))[:60]
    missing = fontlib.missing_glyphs(text, family, settings.fonts_dir)
    return {"font_found": missing is not None, "missing": missing or [], "scripts": sorted(cap.scripts_in(text))}


# ------------------------------------------------------------------ projects

@app.post("/api/projects", status_code=201)
async def create_project(file: UploadFile = File(...), name: str = Form(""), db: Session = Depends(get_session)):
    try:
        media.require_tools()
    except media.MediaError as exc:
        raise err(503, exc.code, exc.message)
    project_id = new_id()
    root = jobs.project_dir(project_id)
    original = root / "original"
    original.mkdir(parents=True, exist_ok=True)
    tmp_path = original / f"upload-{uuid.uuid4().hex}.part"
    try:
        size = 0
        with open(tmp_path, "wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise err(413, "too_large", f"File is larger than the {settings.max_upload_mb} MB limit.")
                fh.write(chunk)
        if size == 0:
            raise err(400, "empty_file", "The uploaded file is empty.")
        try:  # inspect the REAL content; the file name/extension is never trusted
            info = media.inspect(tmp_path, max_duration=settings.max_duration_s, max_width=settings.max_width,
                                 max_height=settings.max_height, timeout=settings.ffprobe_timeout_s)
        except media.MediaError as exc:
            raise err(415 if exc.code in ("unsupported_container", "unreadable", "no_video") else 422, exc.code, exc.message)
        safe_name = f"{uuid.uuid4().hex}{info.extension}"
        tmp_path.rename(original / safe_name)
        title = (name.strip() or safe_title(file.filename or ""))[:200]
        project = Project(id=project_id, name=title, original_name=safe_title(file.filename or ""), media_file=safe_name,
                          media_info=json.dumps(info.as_dict()), style=json.dumps(cap.DEFAULT_STYLE))
        db.add(project)
        db.commit()
        return project_out(project, full=True)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)  # clean up incomplete uploads
        raise
    finally:
        await file.close()


@app.get("/api/projects")
def list_projects(db: Session = Depends(get_session)):
    rows = db.scalars(select(Project).order_by(Project.updated_at.desc())).all()
    return {"projects": [project_out(p) for p in rows]}


@app.get("/api/projects/{project_id}")
def read_project(project_id: str, db: Session = Depends(get_session)):
    return project_out(get_project(project_id, db), full=True)


@app.put("/api/projects/{project_id}/name")
def rename_project(project_id: str, body: RenameIn, db: Session = Depends(get_session)):
    p = get_project(project_id, db)
    p.name = body.name.strip()
    db.commit()
    return project_out(p)


@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: str, db: Session = Depends(get_session)):
    p = get_project(project_id, db)
    busy = db.scalars(select(Job).where(Job.project_id == p.id, Job.status.in_(jobs.ACTIVE))).first()
    if busy:
        raise err(409, "job_active", "Cancel the running job before deleting this project.")
    db.delete(p)
    db.commit()
    shutil.rmtree(jobs.project_dir(project_id), ignore_errors=True)


@app.get("/api/projects/{project_id}/media")
def project_media(project_id: str, db: Session = Depends(get_session)):
    p = get_project(project_id, db)
    path = jobs.project_dir(p.id) / "original" / p.media_file
    if not path.is_file():
        raise err(404, "not_found", "The video file is missing.")
    return FileResponse(path, media_type="video/webm" if path.suffix == ".webm" else "video/mp4" if path.suffix == ".mp4" else None)


# ------------------------------------------------------------------ captions

@app.put("/api/projects/{project_id}/captions")
def save_captions(project_id: str, body: CaptionsIn, db: Session = Depends(get_session)):
    p = get_project(project_id, db)
    duration = float(p.get_info().get("duration", 0))
    for w in body.words:
        if duration and w.start > duration + 1:
            raise err(422, "bad_timestamp", f"Caption '{w.text[:20]}' starts after the end of the video.")
    p.words = json.dumps([w.model_dump() for w in body.words], ensure_ascii=False)
    if body.style is not None:
        p.style = json.dumps(body.style.model_dump())
    db.commit()
    return {"saved": True, "caption_count": len(body.words), "updated_at": p.updated_at.isoformat()}


def _subtitle_response(p: Project, kind: str) -> PlainTextResponse:
    words = cap.to_words(p.get_words())
    if not words:
        raise err(409, "no_captions", "There are no captions yet.")
    style = p.get_style()
    text = cap.to_srt(words, style) if kind == "srt" else cap.to_vtt(words, style)
    fname = re.sub(r"[^\w\-]", "_", p.name)[:60] or "captions"
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{fname}.{kind}"'})


@app.get("/api/projects/{project_id}/subtitles.srt")
def srt(project_id: str, db: Session = Depends(get_session)):
    return _subtitle_response(get_project(project_id, db), "srt")


@app.get("/api/projects/{project_id}/subtitles.vtt")
def vtt(project_id: str, db: Session = Depends(get_session)):
    return _subtitle_response(get_project(project_id, db), "vtt")


# ------------------------------------------------------------------ jobs

def _start(p: Project, kind: str, params: dict):
    job = jobs.create_job(p.id, kind, params)
    if job is None:
        raise err(409, "job_active", f"A {kind} job is already running for this project.")
    return job_out(job)


@app.post("/api/projects/{project_id}/transcribe", status_code=202)
def start_transcribe(project_id: str, body: TranscribeIn, db: Session = Depends(get_session)):
    p = get_project(project_id, db)
    return _start(p, "transcribe", {"language": body.language, "model": body.model})


@app.post("/api/projects/{project_id}/export", status_code=202)
def start_export(project_id: str, body: ExportIn, db: Session = Depends(get_session)):
    p = get_project(project_id, db)
    if not p.get_words():
        raise err(409, "no_captions", "There are no captions to export yet.")
    return _start(p, "export", body.model_dump())


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str, db: Session = Depends(get_session)):
    j = db.get(Job, job_id) if ID_RE.match(job_id) else None
    if not j:
        raise err(404, "not_found", "Job not found.")
    return job_out(j)


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str, db: Session = Depends(get_session)):
    j = db.get(Job, job_id) if ID_RE.match(job_id) else None
    if not j:
        raise err(404, "not_found", "Job not found.")
    if not jobs.cancel_job(job_id):
        raise err(409, "not_cancellable", "This job already finished.")
    return {"cancelled": True}


@app.get("/api/exports/{export_id}/{kind}")
def download(export_id: str, kind: str, db: Session = Depends(get_session)):
    files = {"video": ("captioned.mp4", "video/mp4"), "srt": ("captions.srt", "text/plain"), "vtt": ("captions.vtt", "text/vtt")}
    if not ID_RE.match(export_id) or kind not in files:
        raise err(404, "not_found", "File not found.")
    e = db.get(Export, export_id)
    if not e:
        raise err(404, "not_found", "File not found.")
    fname, mime = files[kind]
    path = jobs.project_dir(e.project_id) / "exports" / e.folder / fname
    if not path.is_file():  # only files that the database knows about are ever served
        raise err(404, "not_found", "File not found.")
    p = db.get(Project, e.project_id)
    nice = re.sub(r"[^\w\-]", "_", p.name)[:60] or "video"
    return FileResponse(path, media_type=mime, filename=f"{nice}-captioned{path.suffix}")


# ------------------------------------------------------------------ frontend (production build)

if settings.frontend_dist.is_dir():
    app.mount("/assets", StaticFiles(directory=settings.frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api"):
            raise err(404, "not_found", "Not found.")
        candidate = (settings.frontend_dist / full_path).resolve()
        if full_path and settings.frontend_dist.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(settings.frontend_dist / "index.html")

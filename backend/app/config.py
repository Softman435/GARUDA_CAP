"""Application settings, read from environment variables (and an optional .env file).

Only standard-library code lives here so it can be imported anywhere.
No secret has a default value that would be safe to use in production.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent


def _load_dotenv() -> None:
    """Tiny .env loader (KEY=VALUE lines). Real environment variables win."""
    # The Xet CAS transfer path can stall on some Windows networks; normal Hub HTTPS downloads work there.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    for candidate in (REPO_DIR / ".env", BACKEND_DIR / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            if value[:1] in ("'", '"'):  # quoted value: keep everything inside the quotes
                end = value.find(value[0], 1)
                value = value[1:end] if end > 0 else value[1:]
            else:                         # unquoted: drop a trailing "  # comment"
                value = value.split(" #", 1)[0].strip()
            os.environ.setdefault(key.strip(), value)


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class IndicConfig:
    """How to run the AI4Bharat IndicConformer (.nemo) model. It cannot run on plain Windows: a worker script is started
    with `runner` (for example a Python inside WSL2 Ubuntu that has AI4Bharat's NeMo fork)."""

    model_path: Path
    runner: list[str]
    decoder: str = "rnnt"                # "rnnt" (recommended) or "ctc"
    timeout_s: int = 3 * 3600
    worker_path: Path = field(default_factory=lambda: BACKEND_DIR / "tools" / "indic_worker.py")
    chunk_seconds: float = 25.0


@dataclass
class Settings:
    env: str = "development"
    data_dir: Path = field(default_factory=lambda: REPO_DIR / "data")
    fonts_dir: Path = field(default_factory=lambda: BACKEND_DIR / "fonts")
    frontend_dist: Path = field(default_factory=lambda: REPO_DIR / "frontend" / "dist")
    database_url: str = ""
    allowed_origins: list[str] = field(default_factory=list)

    max_upload_mb: int = 2048
    max_duration_s: int = 3 * 60 * 60
    max_width: int = 7680
    max_height: int = 4320
    max_json_kb: int = 4096
    max_concurrent_jobs: int = 1
    rate_limit_per_minute: int = 120

    ffmpeg_timeout_s: int = 6 * 60 * 60
    ffprobe_timeout_s: int = 60

    # CPU-first defaults for a laptop with 16 GB RAM and no NVIDIA graphics card.
    whisper_default_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_model_dir: Path | None = None   # default: <data folder>/models (set in load_settings)
    whisper_cpu_threads: int = 0            # 0 = let the library choose (4 threads)
    whisper_beam_size: int = 5              # 1 = fastest, 5 = most accurate (slower on CPU)
    whisper_idle_release_s: int = 300       # free the model's memory after this many idle seconds
    audio_normalize: bool = True            # level the volume of the extracted audio before recognition
    custom_model: str = ""                  # your own fine-tuned CTranslate2 model: a folder or a Hugging Face repo id
    indic_model: str = ""                   # path of the IndicConformer .nemo file (default: first *.nemo in data/models/indic)
    indic_runner: str = ""                  # command that starts a Python with AI4Bharat NeMo (set by scripts\\setup_indic.bat)
    indic_decoder: str = "rnnt"
    indic_timeout_minutes: int = 180
    log_level: str = "INFO"
    # Models are only downloaded when this is true (default true so first run works;
    # set to false to forbid any network access by the app).
    allow_model_download: bool = True

    app_secret_key: str = ""

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def vocabulary_file(self) -> Path:
        return self.data_dir / "vocabulary.txt"

    def load_vocabulary(self) -> list[str]:
        """Optional names/terms (one per line) that Whisper should expect, from data/vocabulary.txt."""
        try:
            lines = self.vocabulary_file.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            return []
        return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")][:60]

    @property
    def indic_available(self) -> bool:
        """True when an IndicConformer .nemo model file is present."""
        return bool(self.indic_model) and Path(self.indic_model).is_file()

    @property
    def indic_configured(self) -> bool:
        """True when both the IndicConformer model and its worker runner are configured."""
        return self.indic_available and bool(self.indic_runner.strip())

    def indic_config(self) -> IndicConfig | None:
        """Everything needed to run IndicConformer, or None when the .nemo file is missing."""
        if not self.indic_available:
            return None
        import shlex
        text = self.indic_runner.strip()
        runner = [part.strip('"') for part in shlex.split(text, posix=(os.name != "nt"))] if text else []
        return IndicConfig(model_path=Path(self.indic_model), runner=runner, decoder=self.indic_decoder,
                           timeout_s=max(60, self.indic_timeout_minutes * 60))

    @property
    def show_error_details(self) -> bool:
        """Short technical hints (error class + message) are shown in the web page only in development."""
        return self.env != "production"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.fonts_dir.mkdir(parents=True, exist_ok=True)

    def validate_for_production(self) -> list[str]:
        problems: list[str] = []
        if self.env == "production":
            if len(self.app_secret_key) < 32:
                problems.append("APP_SECRET_KEY must be set to a random value of at least 32 characters.")
            if not self.allowed_origins or "*" in self.allowed_origins:
                problems.append("ALLOWED_ORIGINS must list your real frontend origin(s), not '*'.")
        return problems


def load_settings() -> Settings:
    _load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # harmless Windows notice from the model downloader
    s = Settings()
    s.env = os.environ.get("APP_ENV", "development").lower()
    if os.environ.get("GARUDA_DATA_DIR"):
        s.data_dir = Path(os.environ["GARUDA_DATA_DIR"]).resolve()
    if os.environ.get("GARUDA_FONTS_DIR"):
        s.fonts_dir = Path(os.environ["GARUDA_FONTS_DIR"]).resolve()
    s.database_url = os.environ.get("DATABASE_URL") or f"sqlite:///{(s.data_dir / 'garuda.db').as_posix()}"
    origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000")
    s.allowed_origins = [o.strip() for o in origins.split(",") if o.strip()]
    s.max_upload_mb = _int("MAX_UPLOAD_MB", s.max_upload_mb)
    s.max_duration_s = _int("MAX_DURATION_SECONDS", s.max_duration_s)
    s.max_concurrent_jobs = max(1, _int("MAX_CONCURRENT_JOBS", s.max_concurrent_jobs))
    s.rate_limit_per_minute = _int("RATE_LIMIT_PER_MINUTE", s.rate_limit_per_minute)
    s.whisper_default_model = os.environ.get("WHISPER_MODEL", s.whisper_default_model)
    s.whisper_device = os.environ.get("WHISPER_DEVICE", s.whisper_device)
    s.whisper_compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", s.whisper_compute_type)
    s.whisper_cpu_threads = max(0, _int("WHISPER_CPU_THREADS", s.whisper_cpu_threads))
    s.whisper_beam_size = min(10, max(1, _int("WHISPER_BEAM_SIZE", s.whisper_beam_size)))
    s.whisper_idle_release_s = max(0, _int("WHISPER_IDLE_RELEASE_SECONDS", s.whisper_idle_release_s))
    s.audio_normalize = os.environ.get("AUDIO_NORMALIZE", "true").lower() != "false"
    s.log_level = os.environ.get("LOG_LEVEL", s.log_level).upper()
    custom = os.environ.get("CUSTOM_MODEL", "").strip().strip('"')
    if custom and not Path(custom).is_absolute() and (REPO_DIR / custom).exists():
        custom = str(REPO_DIR / custom)          # allow paths relative to the project folder
    s.custom_model = custom

    # --- IndicConformer (AI4Bharat) ------------------------------------------------------------------------
    indic_dir = s.data_dir / "models" / "indic"
    model = os.environ.get("INDIC_MODEL", "").strip().strip('"')
    if model and not Path(model).is_absolute():
        model = str(REPO_DIR / model)
    if not model and indic_dir.is_dir():
        found = sorted(indic_dir.glob("*.nemo"))
        model = str(found[0]) if found else ""
    s.indic_model = model
    runner = os.environ.get("INDIC_RUNNER", "").strip()
    runner_file = s.data_dir / "indic_runner.txt"            # written by scripts\\setup_indic.bat / setup_indic.sh
    if not runner and runner_file.is_file():
        runner = (runner_file.read_text(encoding="utf-8-sig").splitlines() or [""])[0].strip()
    s.indic_runner = runner
    decoder = os.environ.get("INDIC_DECODER", "rnnt").strip().lower()
    s.indic_decoder = decoder if decoder in ("rnnt", "ctc") else "rnnt"
    s.indic_timeout_minutes = max(1, _int("INDIC_TIMEOUT_MINUTES", s.indic_timeout_minutes))
    # Models live inside the project's data folder by default, so they are easy to find and delete.
    s.whisper_model_dir = Path(os.environ["WHISPER_MODEL_DIR"]).resolve() if os.environ.get("WHISPER_MODEL_DIR") \
        else s.data_dir / "models"
    s.allow_model_download = os.environ.get("ALLOW_MODEL_DOWNLOAD", "true").lower() != "false"
    s.app_secret_key = os.environ.get("APP_SECRET_KEY", "")
    s.ensure_dirs()
    return s


settings = load_settings()

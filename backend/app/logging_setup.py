"""Central logging: console + a rotating log file (data/logs/garuda.log).

The log file keeps the FULL technical details (including Python tracebacks) so a problem can be diagnosed,
while the web page only ever shows a short, safe message. Obvious secrets are masked before anything is written.
"""
from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

_REDACT = re.compile(r"(?i)\b(password|passwd|token|secret|api[_-]?key|authorization)\b(\s*[=:]\s*)\S+")
_state: dict = {"done": False, "path": None}


class _RedactFilter(logging.Filter):
    """Masks values that follow words like 'password=' or 'token:' in log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a broken format string must never stop logging
            return True
        masked = _REDACT.sub(lambda m: f"{m.group(1)}{m.group(2)}<hidden>", message)
        if masked != message:
            record.msg, record.args = masked, ()
        return True


def setup_logging(log_dir: Path, level: str = "INFO") -> Path | None:
    """Configure logging once. Returns the log file path (None if the file could not be created)."""
    if _state["done"]:
        return _state["path"]
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    redact = _RedactFilter()

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.addFilter(redact)
        root.addHandler(console)

    path: Path | None = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        candidate = log_dir / "garuda.log"
        handler = logging.handlers.RotatingFileHandler(candidate, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(formatter)
        handler.addFilter(redact)
        root.addHandler(handler)
        path = candidate
    except OSError:
        root.warning("Could not create the log file in %s; logging to the console only.", log_dir)
    _state["done"], _state["path"] = True, path
    return path

"""Font lookup and glyph-coverage checks, so broken Kannada/Hindi text never passes silently.

Fonts placed in backend/fonts/ (.ttf/.otf) are used by the exporter (libass `fontsdir`).
Coverage is checked with fontTools when the font file is found there. If the chosen font is
only a system font we cannot inspect, we return `None` (= "unverified") and the caller
shows a warning instead of pretending everything is fine.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # fontTools is a normal dependency, but keep the import soft for tiny setups
    from fontTools.ttLib import TTCollection, TTFont
except Exception:  # pragma: no cover
    TTFont = None  # type: ignore[assignment]
    TTCollection = None  # type: ignore[assignment]


def _font_paths(fonts_dir: Path) -> list[Path]:
    return sorted(path for suffix in ("*.ttf", "*.otf", "*.ttc") for path in fonts_dir.glob(suffix))


@contextmanager
def _font_faces(path: Path) -> Iterator[list]:
    if path.suffix.lower() == ".ttc":
        collection = TTCollection(str(path), lazy=True)
        try:
            yield collection.fonts
        finally:
            collection.close()
    else:
        font = TTFont(str(path), lazy=True, fontNumber=0)
        try:
            yield [font]
        finally:
            font.close()


def list_fonts(fonts_dir: Path) -> list[dict]:
    out = []
    if not fonts_dir.is_dir() or TTFont is None or TTCollection is None:
        return out
    seen = set()
    for path in _font_paths(fonts_dir):
        try:
            with _font_faces(path) as faces:
                for font in faces:
                    names = font["name"]
                    family = names.getDebugName(1) or path.stem
                    key = (family.casefold(), path.name.casefold())
                    if key not in seen:
                        seen.add(key)
                        out.append({"family": family, "file": path.name, "glyphs": len(font.getGlyphOrder())})
        except Exception:
            continue
    return out


def _cmap_for(fonts_dir: Path, family: str) -> set[int] | None:
    if TTFont is None or TTCollection is None or not fonts_dir.is_dir():
        return None
    wanted = family.strip().lower()
    cmap: set[int] = set()
    found = False
    for path in _font_paths(fonts_dir):
        try:
            with _font_faces(path) as faces:
                for font in faces:
                    names = {(font["name"].getDebugName(i) or "").lower() for i in (1, 16)}
                    if wanted in names:
                        found = True
                        cmap |= set(font.getBestCmap().keys())
        except Exception:
            continue
    return cmap if found else None


def missing_glyphs(text: str, family: str, fonts_dir: Path) -> list[str] | None:
    """Characters in `text` that the font cannot draw.

    Returns None if the font file is not in fonts_dir (cannot verify), [] if fully covered.
    """
    cmap = _cmap_for(fonts_dir, family)
    if cmap is None:
        return None
    missing = []
    for ch in dict.fromkeys(text):  # unique, ordered
        if ch.isspace() or ord(ch) < 32:
            continue
        # Zero-width joiners are used inside Indic conjuncts and need no glyph of their own.
        if ch in "‌‍":
            continue
        if ord(ch) not in cmap:
            missing.append(ch)
    return missing

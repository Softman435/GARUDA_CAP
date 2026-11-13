"""Request validation (Pydantic v2). The backend is the authority; the frontend checks are only for convenience."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from . import captions as cap
from .render import ASPECTS, QUALITY, RESOLUTIONS
from .config import settings
from .transcribe import LANGUAGES, model_choices

Color = Field(pattern=r"^#[0-9a-fA-F]{6}$")


class StyleIn(BaseModel):
    mode: Literal["word", "phrase"] = "phrase"
    font_family: str = Field("DejaVu Sans", min_length=1, max_length=60, pattern=r"^[\w .\-]+$")
    font_size: int = Field(64, ge=12, le=300)
    bold: bool = True
    color: str = Field("#FFFFFF", pattern=r"^#[0-9a-fA-F]{6}$")
    highlight_color: str = Field("#FFD400", pattern=r"^#[0-9a-fA-F]{6}$")
    highlight: bool = True
    outline_color: str = Field("#000000", pattern=r"^#[0-9a-fA-F]{6}$")
    outline_width: float = Field(4, ge=0, le=20)
    shadow: float = Field(2, ge=0, le=20)
    box: bool = False
    box_color: str = Field("#000000", pattern=r"^#[0-9a-fA-F]{6}$")
    box_opacity: float = Field(0.6, ge=0, le=1)
    position: Literal["top", "middle", "bottom"] = "bottom"
    align: Literal["left", "center", "right"] = "center"
    margin_v: int = Field(120, ge=0, le=800)
    margin_h: int = Field(60, ge=0, le=800)
    line_spacing: float = Field(0, ge=-20, le=100)
    offset: float = Field(0.0, ge=-30, le=30)
    max_words: int = Field(4, ge=1, le=20)
    max_chars: int = Field(28, ge=5, le=80)
    max_lines: int = Field(2, ge=1, le=4)
    words_visible: int = Field(1, ge=1, le=8)
    animation: Literal["none", "fade", "pop"] = "none"
    uppercase: bool = False

    @field_validator("offset", "outline_width", "shadow", "box_opacity", "line_spacing")
    @classmethod
    def finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be a finite number")
        return v


class WordIn(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    start: float = Field(ge=0, le=60 * 60 * 24)
    end: float = Field(ge=0, le=60 * 60 * 24)
    kind: Literal["word", "phrase"] = "word"
    confidence: float | None = Field(None, ge=0, le=1)

    @field_validator("text")
    @classmethod
    def clean(cls, v: str) -> str:
        v = cap.normalize_text(v).strip()
        if not v:
            raise ValueError("caption text cannot be empty")
        return v

    @model_validator(mode="after")
    def order(self):
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise ValueError("timestamps must be finite numbers")
        if self.end <= self.start:
            raise ValueError("end time must be greater than start time")
        return self


class CaptionsIn(BaseModel):
    words: list[WordIn] = Field(max_length=30000)
    style: StyleIn | None = None


class TranscribeIn(BaseModel):
    language: str = "auto"
    model: str = "base"

    @field_validator("language")
    @classmethod
    def lang(cls, v: str) -> str:
        if v not in LANGUAGES:
            raise ValueError(f"unsupported language '{v}'")
        return v

    @field_validator("model")
    @classmethod
    def mdl(cls, v: str) -> str:
        if v not in model_choices(settings.whisper_model_dir, settings.custom_model, settings.indic_available):
            raise ValueError(f"unknown model '{v}'")
        return v


class ExportIn(BaseModel):
    aspect: str = "original"
    resolution: int = 1080
    quality: str = "standard"
    fps: float | None = Field(None, ge=10, le=60)

    @field_validator("aspect")
    @classmethod
    def a(cls, v: str) -> str:
        if v not in ASPECTS:
            raise ValueError(f"aspect must be one of {ASPECTS}")
        return v

    @field_validator("resolution")
    @classmethod
    def r(cls, v: int) -> int:
        if v not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {RESOLUTIONS}")
        return v

    @field_validator("quality")
    @classmethod
    def q(cls, v: str) -> str:
        if v not in QUALITY:
            raise ValueError(f"quality must be one of {tuple(QUALITY)}")
        return v


class RenameIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)

"""Caption logic: grouping words into display groups and writing SRT / VTT / ASS.

Standard library only (so it is easy to test and has no hidden dependencies).
The browser preview (frontend/src/lib/captions.ts) uses the SAME grouping rules,
so what you see in the editor is what the export contains.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")

DEFAULT_STYLE: dict = {
    "mode": "phrase",            # "word" or "phrase"
    "font_family": "DejaVu Sans",
    "font_size": 64,             # px at a 1080-pixel short side; scales with output size
    "bold": True,
    "color": "#FFFFFF",
    "highlight_color": "#FFD400",
    "highlight": True,           # colour the word being spoken
    "outline_color": "#000000",
    "outline_width": 4,
    "shadow": 2,
    "box": False,
    "box_color": "#000000",
    "box_opacity": 0.6,
    "position": "bottom",        # top | middle | bottom
    "align": "center",           # left | center | right
    "margin_v": 120,             # px at 1080 short side
    "margin_h": 60,
    "line_spacing": 0,
    "offset": 0.0,               # seconds, shifts every caption
    "max_words": 4,              # phrase mode: words per caption
    "max_chars": 28,             # phrase mode: characters per line
    "max_lines": 2,
    "words_visible": 1,          # word mode: how many words are shown at once
    "animation": "none",         # none | fade | pop
    "uppercase": False,
}


@dataclass
class Word:
    text: str
    start: float
    end: float
    kind: str = "word"           # "word" (real word timing) or "phrase" (phrase-level only)
    confidence: float | None = None


@dataclass
class Group:
    words: list[Word]
    start: float
    end: float


def merged_style(style: dict | None) -> dict:
    out = dict(DEFAULT_STYLE)
    if style:
        out.update({k: v for k, v in style.items() if k in DEFAULT_STYLE})
    return out


def to_words(items: list[dict]) -> list[Word]:
    words = [
        Word(
            text=str(i["text"]).strip(),
            start=float(i["start"]),
            end=float(i["end"]),
            kind=i.get("kind", "word"),
            confidence=i.get("confidence"),
        )
        for i in items
        if str(i.get("text", "")).strip()
    ]
    words.sort(key=lambda w: (w.start, w.end))
    return words


# --------------------------------------------------------------------------- grouping

SENTENCE_END = re.compile(r"[.!?…।॥۔]+[\"')\]]*$")
CLAUSE_END = re.compile(r"[,;:،]+[\"')\]]*$")
PAUSE_BREAK_S = 0.7


def group_words(words: list[Word], style: dict) -> list[Group]:
    """Split words into display groups according to the style.

    word mode   -> chunks of `words_visible` words (default 1 = one word at a time)
    phrase mode -> short phrases that break on sentence ends, clause marks and pauses,
                   and never exceed max_words or max_chars * max_lines characters.
    """
    st = merged_style(style)
    groups: list[Group] = []
    current: list[Word] = []

    def flush() -> None:
        chunk = list(current)
        if chunk:
            groups.append(Group(chunk, chunk[0].start, chunk[-1].end))
        current.clear()

    if st["mode"] == "word":
        n = max(1, int(st["words_visible"]))
        for w in words:
            current.append(w)
            if len(current) >= n:
                flush()
        flush()
        return groups

    max_words = max(1, int(st["max_words"]))
    max_chars = max(5, int(st["max_chars"])) * max(1, int(st["max_lines"]))
    for w in words:
        if w.kind == "phrase":  # phrase-level entries are already a phrase; never merge them
            flush()
            current.append(w)
            flush()
            continue
        if current:
            gap = w.start - current[-1].end
            text_len = len(" ".join(x.text for x in current + [w]))
            if gap >= PAUSE_BREAK_S or len(current) >= max_words or text_len > max_chars:
                flush()
        current.append(w)
        if SENTENCE_END.search(w.text) or CLAUSE_END.search(w.text):
            flush()
    flush()
    return groups


def _display(text: str, style: dict) -> str:
    return text.upper() if style.get("uppercase") else text


def group_text(group: Group, style: dict) -> str:
    return _display(" ".join(w.text for w in group.words), style)


def wrap_lines(text: str, max_chars: int, max_lines: int) -> list[str]:
    """Greedy word wrap that is safe for scripts without much spacing (counts code points)."""
    lines: list[str] = []
    line = ""
    for token in text.split():
        candidate = f"{line} {token}".strip()
        if len(candidate) <= max_chars or not line:
            line = candidate
        else:
            lines.append(line)
            line = token
    if line:
        lines.append(line)
    if len(lines) > max_lines:  # keep text intact: fold the overflow into the last line
        lines = lines[: max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return lines


def apply_offset(words: list[Word], offset: float) -> list[Word]:
    out = []
    for w in words:
        s, e = w.start + offset, w.end + offset
        if e <= 0:
            continue
        out.append(Word(w.text, max(0.0, s), e, w.kind, w.confidence))
    return out


# --------------------------------------------------------------------------- SRT / VTT

def _ts(seconds: float, sep: str) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _subtitle_groups(words: list[Word], style: dict) -> list[tuple[float, float, str]]:
    st = merged_style(style)
    # Subtitle files always use readable phrases, even when the video uses word-by-word.
    phrase_style = dict(st, mode="phrase")
    shifted = apply_offset(words, st["offset"])
    out = []
    for g in group_words(shifted, phrase_style):
        lines = wrap_lines(group_text(g, st), int(st["max_chars"]), int(st["max_lines"]))
        out.append((g.start, g.end, "\n".join(lines)))
    return out


def to_srt(words: list[Word], style: dict | None = None) -> str:
    rows = _subtitle_groups(words, style or {})
    return "".join(
        f"{i}\n{_ts(s, ',')} --> {_ts(e, ',')}\n{text}\n\n" for i, (s, e, text) in enumerate(rows, 1)
    )


def to_vtt(words: list[Word], style: dict | None = None) -> str:
    rows = _subtitle_groups(words, style or {})
    body = "".join(f"{_ts(s, '.')} --> {_ts(e, '.')}\n{text.replace('&', '&amp;').replace('<', '&lt;')}\n\n" for s, e, text in rows)
    return "WEBVTT\n\n" + body


# --------------------------------------------------------------------------- ASS (burned-in)

def _ass_color(hex_color: str, alpha: float = 1.0) -> str:
    """#RRGGBB + opacity -> ASS &HAABBGGRR (ASS alpha: 00 = opaque, FF = transparent)."""
    r, g, b = hex_color[1:3], hex_color[3:5], hex_color[5:7]
    a = int(round((1.0 - max(0.0, min(1.0, alpha))) * 255))
    return f"&H{a:02X}{b}{g}{r}".upper()


def _ass_escape(text: str) -> str:
    # Backslash, braces start ASS override blocks; newlines are written as \N.
    text = text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
    return text.replace("\r", "").replace("\n", "\\N")


def _ass_time(seconds: float) -> str:
    cs_total = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs_total, 360_000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _alignment(position: str, align: str) -> int:
    row_base = {"bottom": 0, "middle": 3, "top": 6}[position]
    col = {"left": 1, "center": 2, "right": 3}[align]
    return row_base + col


@dataclass
class AssResult:
    text: str
    events: int = 0
    warnings: list[str] = field(default_factory=list)


def to_ass(words: list[Word], style: dict | None, width: int, height: int) -> AssResult:
    """Build an ASS subtitle script sized for a width x height output frame."""
    st = merged_style(style)
    scale = min(width, height) / 1080.0
    size = max(8, round(st["font_size"] * scale))
    outline = round(st["outline_width"] * scale, 2)
    shadow = round(st["shadow"] * scale, 2)
    margin_v = round(st["margin_v"] * scale)
    margin_h = round(st["margin_h"] * scale)
    spacing = round(st["line_spacing"] * scale, 2)

    primary = _ass_color(st["color"])
    outline_col = _ass_color(st["outline_color"])
    back = _ass_color(st["box_color"], st["box_opacity"] if st["box"] else 0.0)
    if st["box"]:
        border_style, outline_col = 3, _ass_color(st["box_color"], st["box_opacity"])
        outline = max(outline, round(10 * scale, 2))  # padding inside the box
    else:
        border_style = 1

    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{st['font_family'].replace(',', ' ')},{size},{primary},{primary},{outline_col},{back},"
        f"{-1 if st['bold'] else 0},0,0,0,100,100,{spacing},0,{border_style},{outline},{shadow},"
        f"{_alignment(st['position'], st['align'])},{margin_h},{margin_h},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    shifted = apply_offset(words, st["offset"])
    groups = group_words(shifted, st)
    max_chars = int(st["max_chars"])
    max_lines = int(st["max_lines"])
    highlight = _ass_color(st["highlight_color"])
    fx = {"none": "", "fade": r"{\fad(120,80)}", "pop": r"{\fscx80\fscy80\t(0,120,\fscx100\fscy100)}"}[st["animation"]]
    lines_out: list[str] = []
    warnings: list[str] = []

    def line_for(group: Group, active: int | None) -> str:
        parts: list[str] = []
        for idx, w in enumerate(group.words):
            t = _ass_escape(_display(w.text, st))
            if active is not None and idx == active:
                t = r"{\1c" + highlight + r"}" + t + r"{\1c" + primary + r"}"
            parts.append(t)
        # Wrap by plain text length, then re-insert line breaks between the words.
        plain = " ".join(_display(w.text, st) for w in group.words)
        wrapped = wrap_lines(plain, max_chars, max_lines)
        if len(wrapped) > 1:
            counts, used = [], 0
            for ln in wrapped:
                n = len(ln.split())
                counts.append((used, used + n))
                used += n
            joined = []
            for a, b in counts:
                joined.append(" ".join(parts[a:b]))
            return r"\N".join(joined)
        return " ".join(parts)

    for g in groups:
        per_word = st["highlight"] and len(g.words) > 1 and all(w.kind == "word" for w in g.words)
        if not per_word:
            lines_out.append(f"Dialogue: 0,{_ass_time(g.start)},{_ass_time(g.end)},Default,,0,0,0,,{fx}{line_for(g, None)}")
            continue
        for i, w in enumerate(g.words):
            start = w.start if i > 0 else g.start
            nxt = g.words[i + 1].start if i + 1 < len(g.words) else g.end
            end = max(nxt, w.end) if i + 1 < len(g.words) else g.end
            if end <= start:
                continue
            lines_out.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{fx if i == 0 else ''}{line_for(g, i)}"
            )

    if any(w.kind == "phrase" for w in words):
        warnings.append("Some captions only have phrase-level timing (word timing was not available).")
    return AssResult(header + "\n".join(lines_out) + "\n", events=len(lines_out), warnings=warnings)


# --------------------------------------------------------------------------- scripts / glyphs

SCRIPT_RANGES = {
    "Kannada": (0x0C80, 0x0CFF),
    "Devanagari": (0x0900, 0x097F),
    "Telugu": (0x0C00, 0x0C7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Bengali": (0x0980, 0x09FF),   # also Assamese
    "Gujarati": (0x0A80, 0x0AFF),
    "Gurmukhi": (0x0A00, 0x0A7F),  # Punjabi
    "Odia": (0x0B00, 0x0B7F),
    "Arabic": (0x0600, 0x06FF),    # Urdu
}


def scripts_in(text: str) -> set[str]:
    found: set[str] = set()
    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                found.add(name)
    return found


def normalize_text(text: str) -> str:
    """NFC-normalise (keeps Indic conjuncts intact) and drop control characters."""
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if ch in "\n\t " or unicodedata.category(ch)[0] != "C")

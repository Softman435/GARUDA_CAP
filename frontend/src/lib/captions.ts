// Caption grouping. This is a port of backend/app/captions.py so the preview matches the export.

export type Word = { text: string; start: number; end: number; kind: "word" | "phrase"; confidence?: number | null };

export type Style = {
  mode: "word" | "phrase";
  font_family: string; font_size: number; bold: boolean;
  color: string; highlight_color: string; highlight: boolean;
  outline_color: string; outline_width: number; shadow: number;
  box: boolean; box_color: string; box_opacity: number;
  position: "top" | "middle" | "bottom"; align: "left" | "center" | "right";
  margin_v: number; margin_h: number; line_spacing: number; offset: number;
  max_words: number; max_chars: number; max_lines: number; words_visible: number;
  animation: "none" | "fade" | "pop"; uppercase: boolean;
};

export type Group = { words: Word[]; start: number; end: number };

const SENTENCE_END = /[.!?…।॥۔]+["')\]]*$/u;
const CLAUSE_END = /[,;:،]+["')\]]*$/u;
const PAUSE_BREAK_S = 0.7;

export function groupWords(words: Word[], st: Style): Group[] {
  const groups: Group[] = [];
  let cur: Word[] = [];
  const flush = () => {
    if (cur.length) groups.push({ words: cur, start: cur[0].start, end: cur[cur.length - 1].end });
    cur = [];
  };
  const sorted = [...words].sort((a, b) => a.start - b.start || a.end - b.end);
  if (st.mode === "word") {
    const n = Math.max(1, st.words_visible);
    for (const w of sorted) {
      cur.push(w);
      if (cur.length >= n) flush();
    }
    flush();
    return groups;
  }
  const maxWords = Math.max(1, st.max_words);
  const maxChars = Math.max(5, st.max_chars) * Math.max(1, st.max_lines);
  for (const w of sorted) {
    if (w.kind === "phrase") { flush(); cur.push(w); flush(); continue; }
    if (cur.length) {
      const gap = w.start - cur[cur.length - 1].end;
      const len = [...cur, w].map((x) => x.text).join(" ").length;
      if (gap >= PAUSE_BREAK_S || cur.length >= maxWords || len > maxChars) flush();
    }
    cur.push(w);
    if (SENTENCE_END.test(w.text) || CLAUSE_END.test(w.text)) flush();
  }
  flush();
  return groups;
}

export function wrapLines(text: string, maxChars: number, maxLines: number): string[] {
  const lines: string[] = [];
  let line = "";
  for (const token of text.split(/\s+/).filter(Boolean)) {
    const candidate = `${line} ${token}`.trim();
    if (candidate.length <= maxChars || !line) line = candidate;
    else { lines.push(line); line = token; }
  }
  if (line) lines.push(line);
  if (lines.length > maxLines) return [...lines.slice(0, maxLines - 1), lines.slice(maxLines - 1).join(" ")];
  return lines;
}

/** The group showing at time t (after applying the timing offset), and which word is being spoken. */
export function activeAt(groups: Group[], t: number, offset: number): { group: Group; index: number } | null {
  const time = t - offset;
  for (const g of groups) {
    if (time >= g.start && time < g.end) {
      let index = -1;
      g.words.forEach((w, i) => { if (time >= w.start) index = i; });
      return { group: g, index };
    }
  }
  return null;
}

export function validateWord(w: Word, duration: number): string | null {
  if (!w.text.trim()) return "Text is empty";
  if (!Number.isFinite(w.start) || !Number.isFinite(w.end)) return "Time must be a number";
  if (w.start < 0) return "Start cannot be negative";
  if (w.end <= w.start) return "End must be after start";
  if (duration && w.start > duration + 1) return "Starts after the video ends";
  return null;
}

export function fmtTime(s: number): string {
  if (!Number.isFinite(s)) return "0:00.0";
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

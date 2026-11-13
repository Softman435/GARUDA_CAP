import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, pollJob, ApiError, type Job, type Options, type Project } from "../api";
import { activeAt, fmtTime, groupWords, validateWord, wrapLines, type Group, type Style, type Word } from "../lib/captions";
import { recommendedModel } from "../lib/modelRecommendations";

type Snap = { words: Word[]; style: Style };
type Tab = "transcript" | "style" | "export";

const hexToRgba = (hex: string, a: number) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};

export default function Editor({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const [project, setProject] = useState<Project | null>(null);
  const [loadError, setLoadError] = useState("");
  const [words, setWords] = useState<Word[]>([]);
  const [style, setStyle] = useState<Style | null>(null);
  const [past, setPast] = useState<Snap[]>([]);
  const [future, setFuture] = useState<Snap[]>([]);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState("");
  const [tab, setTab] = useState<Tab>("transcript");
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [options, setOptions] = useState<Options | null>(null);
  const [fonts, setFonts] = useState<{ family: string; file: string }[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [glyphWarn, setGlyphWarn] = useState("");

  const [lang, setLang] = useState("auto");
  const [modelChoice, setModelChoice] = useState("auto");
  const [job, setJob] = useState<Job | null>(null);
  const [aspect, setAspect] = useState("original");
  const [resolution, setResolution] = useState(1080);
  const [quality, setQuality] = useState("standard");
  const [exportRows, setExportRows] = useState<Project["exports"]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const [avail, setAvail] = useState({ w: 640, h: 360 });
  const lastKey = useRef({ key: "", at: 0 });
  const stop = useRef({ stopped: false });
  const model = modelChoice === "auto"
    ? recommendedModel(lang, options?.models ?? {}, options?.default_model ?? "base")
    : modelChoice;

  // ------------------------------------------------------------------ load
  useEffect(() => {
    api.project(projectId).then((p) => {
      setProject(p); setWords(p.words ?? []); setStyle(p.style ?? null); setWarnings(p.warnings ?? []);
      setExportRows(p.exports ?? []);
    }).catch((e) => setLoadError(e.message));
    api.options().then(setOptions).catch(() => {});
    api.fonts().then((f) => setFonts(f.fonts)).catch(() => {});
    return () => { stop.current.stopped = true; };
  }, [projectId]);

  useEffect(() => { // make the server's fonts available to the preview
    if (!fonts.length) return;
    const css = fonts.map((f) => `@font-face{font-family:"${f.family}";src:url("/api/fonts/file/${encodeURIComponent(f.file)}")}`).join("\n");
    const el = document.createElement("style");
    el.textContent = css;
    document.head.appendChild(el);
    return () => el.remove();
  }, [fonts]);

  useEffect(() => {
    const el = stage.current;
    if (!el) return;
    const measure = () => setAvail({ w: el.clientWidth, h: Math.max(220, Math.min(window.innerHeight * 0.55, 620)) });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [project]);

  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => { if (dirty) e.preventDefault(); };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  // ------------------------------------------------------------------ history
  const commit = useCallback((nextWords: Word[], nextStyle: Style, key = "") => {
    if (!style) return;
    const now = Date.now();
    const coalesce = key && lastKey.current.key === key && now - lastKey.current.at < 900;
    lastKey.current = { key, at: now };
    if (!coalesce) { setPast((p) => [...p.slice(-99), { words, style }]); setFuture([]); }
    setWords(nextWords); setStyle(nextStyle); setDirty(true);
  }, [words, style]);

  const undo = useCallback(() => {
    if (!past.length || !style) return;
    const prev = past[past.length - 1];
    setPast(past.slice(0, -1));
    setFuture([...future, { words, style }]);
    setWords(prev.words); setStyle(prev.style); setDirty(true);
    lastKey.current = { key: "", at: 0 };
  }, [past, future, words, style]);

  const redo = useCallback(() => {
    if (!future.length || !style) return;
    const next = future[future.length - 1];
    setFuture(future.slice(0, -1));
    setPast([...past, { words, style }]);
    setWords(next.words); setStyle(next.style); setDirty(true);
    lastKey.current = { key: "", at: 0 };
  }, [past, future, words, style]);

  // ------------------------------------------------------------------ playback
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let raf = 0;
    const tick = () => { setT(v.currentTime); raf = requestAnimationFrame(tick); };
    const onPlay = () => { setPlaying(true); raf = requestAnimationFrame(tick); };
    const onPause = () => { setPlaying(false); cancelAnimationFrame(raf); setT(v.currentTime); };
    const onSeek = () => setT(v.currentTime);
    v.addEventListener("play", onPlay); v.addEventListener("pause", onPause); v.addEventListener("seeked", onSeek);
    return () => { cancelAnimationFrame(raf); v.removeEventListener("play", onPlay); v.removeEventListener("pause", onPause); v.removeEventListener("seeked", onSeek); };
  }, [project]);

  const seek = (s: number) => {
    const v = videoRef.current;
    if (!v || !project) return;
    v.currentTime = Math.max(0, Math.min(project.media.duration, s));
    setT(v.currentTime);
  };
  const toggle = () => { const v = videoRef.current; if (v) (v.paused ? v.play() : v.pause()); };

  // ------------------------------------------------------------------ save
  const invalid = useMemo(() => {
    if (!project) return -1;
    return words.findIndex((w) => validateWord(w, project.media.duration));
  }, [words, project]);

  const save = useCallback(async (): Promise<boolean> => {
    if (!style || !project) return false;
    if (invalid >= 0) { setNotice(`Caption ${invalid + 1}: ${validateWord(words[invalid], project.media.duration)}`); return false; }
    setSaving(true);
    try {
      await api.saveCaptions(project.id, words, style);
      setDirty(false); setNotice("Saved.");
      return true;
    } catch (e: any) { setNotice(`Save failed: ${e.message}`); return false; }
    finally { setSaving(false); }
  }, [project, words, style, invalid]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName);
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { e.preventDefault(); save(); return; }
      if (typing) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); redo(); }
      else if (e.key === " ") { e.preventDefault(); toggle(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); seek(t - (e.shiftKey ? 5 : 1)); }
      else if (e.key === "ArrowRight") { e.preventDefault(); seek(t + (e.shiftKey ? 5 : 1)); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // ------------------------------------------------------------------ glyph check
  useEffect(() => {
    if (!style || !words.length) { setGlyphWarn(""); return; }
    const id = setTimeout(() => {
      api.glyphCheck(words.map((w) => w.text).join(" "), style.font_family).then((r) => {
        const nonLatin = r.scripts.length > 0;
        if (r.missing.length) setGlyphWarn(`The font "${style.font_family}" cannot draw these characters: ${r.missing.slice(0, 12).join(" ")}. Choose a font that supports ${r.scripts.join(", ") || "this text"}.`);
        else if (!r.font_found && nonLatin) setGlyphWarn(`Your text uses ${r.scripts.join(", ")} script, but "${style.font_family}" is not in the app's font folder, so it cannot be checked. Pick a font from the list (see docs/FONTS.md) and check the preview before exporting.`);
        else setGlyphWarn("");
      }).catch(() => {});
    }, 600);
    return () => clearTimeout(id);
  }, [words, style?.font_family]);

  // ------------------------------------------------------------------ jobs
  const runJob = async (start: () => Promise<Job>, after?: (j: Job) => void) => {
    setNotice("");
    try {
      const first = await start();
      setJob(first);
      stop.current.stopped = false;
      const done = await pollJob(first.id, setJob, stop.current);
      if (done.status === "completed") after?.(done);
    } catch (e: any) { setJob(null); setNotice(e instanceof ApiError ? e.message : String(e)); }
  };

  const transcribe = async () => {
    if (words.length && !confirm("This replaces the current captions (including your edits). Continue?")) return;
    await runJob(() => api.transcribe(projectId, lang, model), async () => {
      const p = await api.project(projectId);
      setProject(p); setWords(p.words ?? []); setWarnings(p.warnings ?? []); setPast([]); setFuture([]); setDirty(false);
      setTab("transcript");
    });
  };

  const doExport = async () => {
    if (!(await save())) return;
    await runJob(() => api.exportVideo(projectId, { aspect, resolution, quality }), async () => {
      const p = await api.project(projectId);
      setExportRows(p.exports ?? []);
    });
  };

  const busy = job && (job.status === "queued" || job.status === "running");

  // ------------------------------------------------------------------ word editing
  const setWord = (i: number, patch: Partial<Word>, key: string) => {
    if (!style) return;
    commit(words.map((w, k) => (k === i ? { ...w, ...patch } : w)), style, `${key}${i}`);
  };
  const del = (i: number) => { if (style) { commit(words.filter((_, k) => k !== i), style); setSelected(null); } };
  const split = (i: number) => {
    if (!style) return;
    const w = words[i];
    const parts = w.text.split(/\s+/).filter(Boolean);
    if (parts.length < 2) { setNotice("Only captions with two or more words can be split."); return; }
    const half = Math.ceil(parts.length / 2);
    const a = parts.slice(0, half).join(" "), b = parts.slice(half).join(" ");
    const mid = w.start + ((w.end - w.start) * a.length) / (a.length + b.length);
    commit([...words.slice(0, i), { ...w, text: a, end: mid }, { ...w, text: b, start: mid }, ...words.slice(i + 1)], style);
  };
  const merge = (i: number) => {
    if (!style || i >= words.length - 1) return;
    const a = words[i], b = words[i + 1];
    commit([...words.slice(0, i), { text: `${a.text} ${b.text}`, start: a.start, end: b.end, kind: a.kind === "phrase" || b.kind === "phrase" ? "phrase" : "word", confidence: null }, ...words.slice(i + 2)], style);
  };
  const add = () => {
    if (!style || !project) return;
    const start = Math.min(t, Math.max(0, project.media.duration - 1));
    const next = [...words, { text: "New caption", start, end: start + 1, kind: "word" as const }].sort((a, b) => a.start - b.start);
    commit(next, style);
    setSelected(next.findIndex((w) => w.start === start && w.text === "New caption"));
  };
  const patchStyle = <K extends keyof Style>(k: K, v: Style[K]) => { if (style) commit(words, { ...style, [k]: v }, `style-${String(k)}`); };

  // ------------------------------------------------------------------ preview
  const groups: Group[] = useMemo(() => (style ? groupWords(words, style) : []), [words, style]);
  const active = style ? activeAt(groups, t, style.offset) : null;

  if (loadError) return <div className="p-6"><p role="alert" className="rounded bg-red-900/60 p-3">{loadError}</p><button className="btn mt-3" onClick={onClose}>Back</button></div>;
  if (!project || !style) return <p className="p-6 text-slate-400">Loading project…</p>;

  const m = project.media;
  const ratio = aspect === "original" ? m.width / m.height : aspect === "9:16" ? 9 / 16 : aspect === "16:9" ? 16 / 9 : 1;
  let pw = Math.min(avail.w, avail.h * ratio), ph = pw / ratio;
  if (ph > avail.h) { ph = avail.h; pw = ph * ratio; }
  const scale = Math.min(pw, ph) / 1080;

  const renderCaption = () => {
    if (!active) return null;
    const g = active.group;
    const plain = g.words.map((w) => (style.uppercase ? w.text.toUpperCase() : w.text));
    const lines = wrapLines(plain.join(" "), style.max_chars, style.max_lines);
    let used = 0;
    const perWord = style.highlight && g.words.length > 1 && g.words.every((w) => w.kind === "word");
    const rows = lines.map((ln) => {
      const n = ln.split(/\s+/).length;
      const part = plain.slice(used, used + n).map((txt, k) => (
        <span key={used + k} style={perWord && used + k === active.index ? { color: style.highlight_color } : undefined}>{txt}{k < n - 1 ? " " : ""}</span>
      ));
      used += n;
      return <div key={used}>{part}</div>;
    });
    const pos: React.CSSProperties = style.position === "middle" ? { top: "50%", transform: "translateY(-50%)" }
      : style.position === "top" ? { top: style.margin_v * scale } : { bottom: style.margin_v * scale };
    return (
      <div key={`${g.start}`} className="pointer-events-none absolute" style={{
        left: style.margin_h * scale, right: style.margin_h * scale, textAlign: style.align, ...pos,
        animation: style.animation === "fade" ? "garuda-fade .15s" : style.animation === "pop" ? "garuda-pop .15s" : undefined,
      }}>
        <span style={{
          display: "inline-block", fontFamily: `"${style.font_family}", "Nirmala UI", "Noto Sans", sans-serif`, fontWeight: style.bold ? 700 : 400,
          fontSize: style.font_size * scale, color: style.color, lineHeight: 1.25 + style.line_spacing / 100,
          WebkitTextStroke: `${style.outline_width * scale * 2}px ${style.outline_color}`, paintOrder: "stroke fill",
          textShadow: style.shadow ? `${style.shadow * scale}px ${style.shadow * scale}px ${style.shadow * scale}px rgba(0,0,0,.8)` : undefined,
          background: style.box ? hexToRgba(style.box_color, style.box_opacity) : undefined, padding: style.box ? `${8 * scale}px ${16 * scale}px` : undefined,
        }}>{rows}</span>
      </div>
    );
  };

  const latest = exportRows?.[0];
  const tabBtn = (id: Tab, label: string) => (
    <button role="tab" aria-selected={tab === id} className={`editor-tab flex-1 border-b-2 ${tab === id ? "border-gold text-gold" : "border-transparent text-slate-400"}`} onClick={() => setTab(id)}>{label}</button>
  );

  return (
    <div className="editor-shell flex min-h-full flex-col lg:h-full lg:flex-row">
      {/* ---------------- left: video + timeline */}
      <section className="editor-preview flex min-w-0 flex-1 flex-col" aria-label="Video preview">
        <div className="editor-toolbar flex flex-wrap items-center gap-2">
          <button className="btn" onClick={() => { if (!dirty || confirm("You have unsaved changes. Leave anyway?")) onClose(); }}>← Projects</button>
          <h1 className="min-w-0 flex-1 truncate font-semibold">{project.name}</h1>
          <button className="btn" onClick={undo} disabled={!past.length} title="Undo (Ctrl+Z)">Undo</button>
          <button className="btn" onClick={redo} disabled={!future.length} title="Redo (Ctrl+Y)">Redo</button>
          <button className="btn-primary" onClick={save} disabled={saving || !dirty} title="Save (Ctrl+S)">{saving ? "Saving…" : dirty ? "Save" : "Saved"}</button>
        </div>

        <div ref={stage} className="editor-stage flex w-full justify-center">
          <div className="relative overflow-hidden bg-black" style={{ width: pw, height: ph }}>
            <video ref={videoRef} src={api.mediaUrl(project.id)} className="h-full w-full object-contain" playsInline preload="metadata" />
            {renderCaption()}
          </div>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button className="btn" onClick={toggle}>{playing ? "Pause" : "Play"}</button>
          <span className="font-mono text-xs text-slate-300">{fmtTime(t)} / {fmtTime(m.duration)}</span>
          <input aria-label="Seek" type="range" className="min-w-[120px] flex-1" min={0} max={m.duration} step={0.05} value={t} onChange={(e) => seek(Number(e.target.value))} />
          <label className="flex items-center gap-1 text-xs">Volume
            <input aria-label="Volume" type="range" min={0} max={1} step={0.05} defaultValue={1} onChange={(e) => { if (videoRef.current) videoRef.current.volume = Number(e.target.value); }} />
          </label>
        </div>
        <p className="mt-1 text-xs text-slate-500">{m.width}×{m.height} · {m.fps} fps · {m.video_codec}/{m.audio_codec} · Space = play/pause, ←/→ = seek, Ctrl+Z/Y = undo/redo, Ctrl+S = save</p>

        <div className="mt-3">
          <div className="mb-1 flex items-center gap-2 text-xs">
            <span>Timeline zoom</span>
            <input aria-label="Timeline zoom" type="range" min={1} max={20} step={0.5} value={zoom} onChange={(e) => setZoom(Number(e.target.value))} />
            <span>{zoom}×</span>
          </div>
          <div className="timeline-shell overflow-x-auto rounded border border-line bg-panel" data-testid="timeline">
            <div className="relative h-16 cursor-pointer" style={{ width: `${zoom * 100}%` }}
              onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); seek(((e.clientX - r.left) / r.width) * m.duration); }}>
              {words.map((w, i) => (
                <button key={i} title={w.text} onClick={(e) => { e.stopPropagation(); setSelected(i); seek(w.start); setTab("transcript"); }}
                  className={`absolute top-3 h-10 overflow-hidden rounded border px-1 text-left text-[11px] leading-10 ${selected === i ? "border-gold bg-gold/40" : "border-saffron/50 bg-saffron/20"}`}
                  style={{ left: `${(w.start / m.duration) * 100}%`, width: `max(3px, ${((w.end - w.start) / m.duration) * 100}%)` }}>
                  {w.text}
                </button>
              ))}
              <div className="pointer-events-none absolute top-0 h-full w-px bg-white" style={{ left: `${(t / m.duration) * 100}%` }} />
            </div>
          </div>
        </div>
      </section>

      {/* ---------------- right: panels */}
      <aside className="editor-sidebar flex w-full flex-col border-t border-line bg-panel lg:h-full lg:w-[420px] lg:border-l lg:border-t-0" aria-label="Editor panels">
        <div role="tablist" className="flex border-b border-line">{tabBtn("transcript", "Transcript")}{tabBtn("style", "Style")}{tabBtn("export", "Export")}</div>
        {notice && <p role="status" className="bg-ink px-3 py-2 text-sm text-amber-300">{notice}</p>}
        {glyphWarn && <p role="alert" className="bg-amber-900/50 px-3 py-2 text-xs">{glyphWarn}</p>}
        {job && (
          <div className="border-b border-line p-3 text-sm" role="status" aria-live="polite">
            <div className="flex items-center justify-between"><span>{job.kind === "export" ? "Export" : "Transcription"}: {job.status}{job.message ? ` — ${job.message}` : ""}</span><span className="text-xs text-slate-400">{job.where === "local" ? "running locally" : "cloud"}</span></div>
            <progress className="mt-1 w-full" value={job.progress} max={1} />
            {job.error && <p className="mt-1 text-red-300">{job.error}</p>}
            {busy && <button className="btn mt-2" onClick={() => api.cancel(job.id).catch((e) => setNotice(e.message))}>Cancel</button>}
          </div>
        )}

        <div className="editor-panel min-h-0 flex-1 overflow-auto">
          {tab === "transcript" && (
            <div>
              <fieldset className="mb-3 rounded border border-line p-2" disabled={!!busy}>
                <legend className="px-1 text-xs text-slate-400">Speech recognition (runs on this computer)</legend>
                <div className="grid grid-cols-2 gap-2">
                  <div><label className="label" htmlFor="lang">Language</label>
                    <select id="lang" className="input" value={lang} onChange={(e) => setLang(e.target.value)}>
                      {Object.entries(options?.languages ?? { auto: "Auto-detect" }).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                    </select></div>
                  <div><label className="label" htmlFor="model">Model</label>
                    <select id="model" className="input" value={modelChoice} onChange={(e) => setModelChoice(e.target.value)}>
                      <option value="auto">Auto (higher accuracy)</option>
                      {Object.keys(options?.models ?? { small: "" }).map((k) => <option key={k} value={k}>{k}</option>)}
                    </select></div>
                </div>
                <p className="mt-1 text-xs text-slate-400">{options?.models[model]}</p>
                <p className="mt-1 text-xs text-slate-500">Auto chooses {model}: large-v3 for Auto-detect and all non-Kannada/Kanglish choices, including English, Telugu, and mixed Kannada-English; kn-medium for explicitly selected Kannada or Kanglish. These are existing models, not newly trained ones. large-v3 is slow and needs about 10 GB RAM; review spelling before export.</p>
                {(model === "kn-medium" || model === "kn-small") && lang !== "kn" && lang !== "kanglish" && <p role="alert" className="mt-1 text-xs text-amber-300">This model is tuned for Kannada. Auto and mixed mode will force Kannada, and other-language speech may be mistranscribed. Use large-v3 for mixed or other-language audio.</p>}
                {options?.weak_languages.includes(lang) && <p className="mt-1 text-xs text-amber-300">Accuracy for this language is lower than English. For Kannada choose the model "kn-medium" (Kannada-tuned); the small general models are weak. Proof-read every caption.</p>}
                {lang === "kn-en" && <p className="mt-1 text-xs text-sky-300">Mixed mode keeps Kannada and English as spoken (nothing is translated). Whisper picks the language for each ~30-second window, so an English word inside a Kannada sentence may be re-spelled: check and edit the captions.</p>}
                {lang === "kanglish" && <p className="mt-1 text-xs text-sky-300">Writes Kannada pronunciation in uppercase Latin letters (for example, NANNA HESARU). This transliterates the Kannada model output; it does not improve recognition accuracy. Review names and spelling.</p>}
                <button className="btn-primary mt-2 w-full" onClick={transcribe}>{words.length ? "Re-transcribe" : "Generate captions from audio"}</button>
                <p className="mt-1 text-xs text-slate-500">First use of a model downloads it once (internet needed). After that it works offline.</p>
              </fieldset>
              {warnings.map((w, i) => <p key={i} className="mb-1 rounded bg-amber-900/40 p-2 text-xs">{w}</p>)}

              <div className="mb-2 flex items-center justify-between">
                <h2 className="text-sm font-semibold">{words.length} captions</h2>
                <button className="btn" onClick={add}>+ Add at playhead</button>
              </div>
              {words.length === 0 && <p className="text-sm text-slate-400">No captions yet. Generate them from the audio above, or add one by hand.</p>}
              <ul className="space-y-1">
                {words.map((w, i) => {
                  const err = validateWord(w, m.duration);
                  const now = style && t - style.offset >= w.start && t - style.offset < w.end;
                  return (
                    <li key={i} className={`rounded border p-1.5 ${selected === i ? "border-gold" : now ? "border-saffron" : "border-line"}`} onClick={() => setSelected(i)}>
                      <div className="flex items-center gap-1">
                        <button className="btn !px-2" aria-label={`Play from caption ${i + 1}`} onClick={() => seek(w.start)}>▶</button>
                        <input aria-label={`Caption ${i + 1} text`} className="input" value={w.text} onChange={(e) => setWord(i, { text: e.target.value }, "text")} />
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-1 text-xs">
                        <input aria-label={`Caption ${i + 1} start`} type="number" step="0.05" min="0" className="input !w-20" value={w.start} onChange={(e) => setWord(i, { start: Number(e.target.value) }, "start")} />
                        <span>→</span>
                        <input aria-label={`Caption ${i + 1} end`} type="number" step="0.05" min="0" className="input !w-20" value={w.end} onChange={(e) => setWord(i, { end: Number(e.target.value) }, "end")} />
                        <button className="btn !px-2" onClick={() => split(i)}>Split</button>
                        <button className="btn !px-2" onClick={() => merge(i)} disabled={i === words.length - 1}>Merge ↓</button>
                        <button className="btn !px-2" onClick={() => del(i)}>Delete</button>
                        {w.kind === "phrase" && <span className="text-amber-300" title="Word alignment failed; only phrase timing">phrase timing</span>}
                        {w.confidence != null && w.confidence < 0.5 && <span className="text-amber-300" title="The model was unsure about this word">low confidence</span>}
                      </div>
                      {err && <p role="alert" className="mt-1 text-xs text-red-300">{err}</p>}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          {tab === "style" && <StylePanel style={style} fonts={fonts} patch={patchStyle} />}

          {tab === "export" && (
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-2">
                <div><label className="label" htmlFor="aspect">Shape</label>
                  <select id="aspect" className="input" value={aspect} onChange={(e) => setAspect(e.target.value)}>
                    <option value="original">Original</option><option value="9:16">9:16 vertical</option><option value="16:9">16:9 wide</option><option value="1:1">1:1 square</option>
                  </select></div>
                <div><label className="label" htmlFor="res">Size</label>
                  <select id="res" className="input" value={resolution} onChange={(e) => setResolution(Number(e.target.value))}>
                    {[360, 480, 720, 1080, 1440].map((r) => <option key={r} value={r}>{r}p</option>)}
                  </select></div>
                <div><label className="label" htmlFor="q">Quality</label>
                  <select id="q" className="input" value={quality} onChange={(e) => setQuality(e.target.value)}>
                    <option value="draft">Draft (fast)</option><option value="standard">Standard</option><option value="high">High (slow)</option>
                  </select></div>
              </div>
              <p className="text-xs text-slate-400">The preview on the left shows this shape. Your original video and voice are not changed. A new MP4 is created.</p>
              <button className="btn-primary w-full" onClick={doExport} disabled={!!busy || words.length === 0}>Save & export MP4 with captions</button>
              {words.length === 0 && <p className="text-xs text-slate-400">Add captions first.</p>}
              {latest && (
                <div className="rounded border border-line p-2 text-sm">
                  <p className="mb-2 font-medium">Latest export ({latest.width}×{latest.height})</p>
                  <div className="flex flex-wrap gap-2">
                    <a className="btn" href={api.downloadUrl(latest.id, "video")} download>Download MP4</a>
                    <a className="btn" href={api.downloadUrl(latest.id, "srt")} download>SRT</a>
                    <a className="btn" href={api.downloadUrl(latest.id, "vtt")} download>VTT</a>
                  </div>
                </div>
              )}
              <div className="rounded border border-line p-2 text-sm">
                <p className="mb-2 font-medium">Subtitle files only (from saved captions)</p>
                <div className="flex gap-2">
                  <a className="btn" href={api.srtUrl(project.id)} download>Download SRT</a>
                  <a className="btn" href={api.vttUrl(project.id)} download>Download VTT</a>
                </div>
              </div>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

function StylePanel({ style, fonts, patch }: { style: Style; fonts: { family: string }[]; patch: <K extends keyof Style>(k: K, v: Style[K]) => void }) {
  const families = Array.from(new Set([...fonts.map((f) => f.family), style.font_family]));
  const num = (k: keyof Style, label: string, min: number, max: number, step = 1) => (
    <div><label className="label" htmlFor={`s-${k}`}>{label}</label>
      <input id={`s-${k}`} type="number" className="input" min={min} max={max} step={step} value={style[k] as number}
        onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) patch(k, Math.min(max, Math.max(min, v)) as never); }} /></div>
  );
  const color = (k: keyof Style, label: string) => (
    <div><label className="label" htmlFor={`s-${k}`}>{label}</label>
      <input id={`s-${k}`} type="color" className="h-9 w-full rounded border border-line bg-ink" value={style[k] as string} onChange={(e) => patch(k, e.target.value as never)} /></div>
  );
  const check = (k: keyof Style, label: string) => (
    <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={style[k] as boolean} onChange={(e) => patch(k, e.target.checked as never)} />{label}</label>
  );
  const select = (k: keyof Style, label: string, opts: [string, string][]) => (
    <div><label className="label" htmlFor={`s-${k}`}>{label}</label>
      <select id={`s-${k}`} className="input" value={style[k] as string} onChange={(e) => patch(k, e.target.value as never)}>{opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></div>
  );
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        {select("mode", "Caption mode", [["phrase", "Phrase by phrase"], ["word", "Word by word"]])}
        <div><label className="label" htmlFor="s-font">Font</label>
          <select id="s-font" className="input" value={style.font_family} onChange={(e) => patch("font_family", e.target.value)}>{families.map((f) => <option key={f}>{f}</option>)}</select></div>
        {num("font_size", "Font size", 12, 300)}
        {check("bold", "Bold")}
        {color("color", "Text colour")}
        {color("highlight_color", "Highlight colour")}
        {check("highlight", "Highlight spoken word")}
        {style.mode === "word" ? num("words_visible", "Words visible at once", 1, 8) : num("max_words", "Max words per caption", 1, 20)}
        {style.mode === "phrase" && num("max_chars", "Max characters per line", 5, 80)}
        {num("max_lines", "Max lines", 1, 4)}
        {color("outline_color", "Outline colour")}
        {num("outline_width", "Outline thickness", 0, 20, 0.5)}
        {num("shadow", "Shadow", 0, 20, 0.5)}
        {check("box", "Background box")}
        {style.box && color("box_color", "Box colour")}
        {style.box && num("box_opacity", "Box opacity (0–1)", 0, 1, 0.05)}
        {select("position", "Position", [["bottom", "Bottom"], ["middle", "Middle"], ["top", "Top"]])}
        {select("align", "Alignment", [["center", "Centre"], ["left", "Left"], ["right", "Right"]])}
        {num("margin_v", "Safe margin (vertical)", 0, 800)}
        {num("margin_h", "Safe margin (sides)", 0, 800)}
        {num("line_spacing", "Line spacing", -20, 100)}
        {num("offset", "Timing offset (seconds)", -30, 30, 0.05)}
        {select("animation", "Animation", [["none", "None"], ["fade", "Fade in"], ["pop", "Pop"]])}
        {check("uppercase", "UPPERCASE")}
      </div>
    </div>
  );
}

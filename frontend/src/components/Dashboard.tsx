import { useCallback, useEffect, useRef, useState } from "react";
import { api, uploadVideo, type Project } from "../api";

const ACCEPT = "video/mp4,video/quicktime,video/webm,video/x-matroska,.mp4,.mov,.webm,.mkv,.avi";

const fmtSize = (b: number) => (b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`);
const fmtDur = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function Dashboard({ onOpen }: { onOpen: (id: string) => void }) {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const picker = useRef<HTMLInputElement>(null);
  const canceller = useRef<{ cancel?: () => void }>({});

  const load = useCallback(() => {
    api.projects().then((r) => setProjects(r.projects)).catch((e) => { setError(e.message); setProjects([]); });
  }, []);
  useEffect(load, [load]);

  const upload = async (file: File | undefined) => {
    if (!file) return;
    setError("");
    if (file.size === 0) return setError("That file is empty.");
    if (!file.type.startsWith("video/") && !/\.(mp4|mov|webm|mkv|avi)$/i.test(file.name)) {
      return setError("Please choose a video file (MP4, MOV, WebM, MKV or AVI).");
    }
    setProgress(0);
    try {
      const p = await uploadVideo(file, setProgress, canceller.current);
      onOpen(p.id);
    } catch (e: any) {
      if (e.code !== "cancelled") setError(e.message);
    } finally {
      setProgress(null);
      if (picker.current) picker.current.value = "";
    }
  };

  const remove = async (p: Project) => {
    if (!confirm(`Delete "${p.name}" and all its files? This cannot be undone.`)) return;
    try { await api.remove(p.id); load(); } catch (e: any) { setError(e.message); }
  };

  return (
    <div className="dashboard-page mx-auto max-w-6xl px-4 py-7 sm:px-8 sm:py-10">
      <section className="dashboard-hero">
        <div className="hero-copy">
          <span className="eyebrow"><span className="eyebrow-line" /> YOUR LOCAL CAPTION STUDIO</span>
          <h1>Every story<br /><span>deserves a voice.</span></h1>
          <p>Turn your video’s real audio into beautifully timed captions. Edit, style, and export — all on your computer.</p>
          <div className="hero-trust"><span className="trust-icon">✦</span> Private by design <span className="trust-divider">·</span> Works offline after setup</div>
          <div className="repeat-motif" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none"><path d="M20 7v5h-5M4 17v-5h5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/><path d="M5.5 9A7 7 0 0 1 17 6l3 2M4 16l3 2a7 7 0 0 0 11.5-3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
            <span>CREATE <i>·</i> REFINE <i>·</i> REPEAT</span>
          </div>
        </div>

        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false); }}
          onDrop={(e) => { e.preventDefault(); setDragging(false); upload(e.dataTransfer.files[0]); }}
          className={`upload-card ${dragging ? "upload-card-active" : ""}`}
        >
          {progress === null ? (
            <>
              <div className="upload-icon" aria-hidden>
                <svg viewBox="0 0 24 24" fill="none"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </div>
              <h2>Start with a video</h2>
              <p>Drop your file here, or browse to choose one</p>
              <input ref={picker} id="file" type="file" accept={ACCEPT} className="sr-only" onChange={(e) => upload(e.target.files?.[0])} />
              <label htmlFor="file" className="btn-primary upload-button">Choose a video <span aria-hidden>↗</span></label>
              <div className="upload-footnote"><span>MP4</span><span>MOV</span><span>WEBM</span><span>MKV</span><span>AVI</span><i>Video must include audio</i></div>
            </>
          ) : (
            <div role="status" aria-live="polite" className="upload-progress">
              <div className="upload-icon" aria-hidden>↥</div>
              <h2>{progress < 1 ? "Uploading your video…" : "Checking the video…"}</h2>
              <p>{progress < 1 ? `${Math.round(progress * 100)}% complete` : "This usually takes just a moment."}</p>
              <progress className="mt-4 w-full" value={progress} max={1} />
              <button className="btn mt-4" onClick={() => canceller.current.cancel?.()}>Cancel upload</button>
            </div>
          )}
        </div>
      </section>

      {error && <p role="alert" className="mt-5 rounded-xl border border-red-400/20 bg-red-950/60 p-4 text-sm text-red-200">{error}</p>}

      <section className="projects-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">YOUR WORKSPACE</span>
            <h2>Recent projects <span className="project-count">{projects?.length ?? 0}</span></h2>
          </div>
          <p>Pick up right where you left off.</p>
        </div>
        {projects === null && <p className="mt-5 text-sm text-slate-400">Loading your workspace…</p>}
        {projects && projects.length === 0 && (
          <div className="empty-projects"><span aria-hidden>▧</span><p>No projects yet</p><small>Choose a video above to create your first caption project.</small></div>
        )}
        <ul className="project-grid">
          {projects?.map((p) => (
            <li key={p.id} className="project-card">
              <button className="project-card-main" onClick={() => onOpen(p.id)}>
                <span className={`project-art ${p.media.width < p.media.height ? "project-art-portrait" : ""}`} aria-hidden>
                  <span className="art-play">▶</span>
                  <span className="art-caption">Aa</span>
                  <span className="art-duration">{fmtDur(p.media.duration)}</span>
                </span>
                <span className="project-info">
                  <span className="project-name">{p.name}</span>
                  <span className="project-meta">{p.media.width} × {p.media.height} <i>·</i> {fmtSize(p.media.size_bytes)}</span>
                  <span className="caption-pill"><span className="caption-pill-dot" />{p.caption_count} captions</span>
                </span>
                <span className="project-arrow" aria-hidden>↗</span>
              </button>
              <div className="project-actions">
                <button className="project-open" onClick={() => onOpen(p.id)}>Open project <span aria-hidden>→</span></button>
                <button className="project-delete" onClick={() => remove(p)} aria-label={`Delete ${p.name}`}>Delete</button>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

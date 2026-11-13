import type { Style, Word } from "./lib/captions";

export type MediaInfo = { duration: number; width: number; height: number; fps: number; video_codec: string; audio_codec: string; size_bytes: number; container: string };
export type ExportRow = { id: string; width: number; height: number; created_at: string };
export type Project = {
  id: string; name: string; language: string; created_at: string; updated_at: string; media: MediaInfo; caption_count: number;
  words?: Word[]; style?: Style; warnings?: string[]; exports?: ExportRow[];
};
export type Job = { id: string; project_id: string; kind: string; status: "queued" | "running" | "completed" | "failed" | "cancelled"; progress: number; message: string; error: string; where: string; result: Record<string, any> };
export type Options = { languages: Record<string, string>; models: Record<string, string>; default_model: string; weak_languages: string[]; notes: string[] };
export type Ready = { ready: boolean; ffmpeg: boolean; ffprobe: boolean; faster_whisper: boolean };

export class ApiError extends Error {
  code: string;
  constructor(message: string, code = "error") { super(message); this.code = code; }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try { res = await fetch(url, init); } catch { throw new ApiError("Cannot reach the GARUDA server. Is it running?", "network"); }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let data: any = null;
  try { data = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) throw new ApiError(data?.error?.message ?? `Request failed (${res.status})`, data?.error?.code);
  return data as T;
}

const json = (method: string, body: unknown): RequestInit => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  ready: () => fetch("/api/ready").then((r) => r.json() as Promise<Ready>).catch(() => null),
  options: () => request<Options>("/api/options"),
  fonts: () => request<{ fonts: { family: string; file: string }[] }>("/api/fonts"),
  projects: () => request<{ projects: Project[] }>("/api/projects"),
  project: (id: string) => request<Project>(`/api/projects/${id}`),
  rename: (id: string, name: string) => request<Project>(`/api/projects/${id}/name`, json("PUT", { name })),
  remove: (id: string) => request<void>(`/api/projects/${id}`, { method: "DELETE" }),
  saveCaptions: (id: string, words: Word[], style: Style) => request<{ saved: boolean }>(`/api/projects/${id}/captions`, json("PUT", { words, style })),
  transcribe: (id: string, language: string, model: string) => request<Job>(`/api/projects/${id}/transcribe`, json("POST", { language, model })),
  exportVideo: (id: string, o: { aspect: string; resolution: number; quality: string }) => request<Job>(`/api/projects/${id}/export`, json("POST", o)),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  cancel: (id: string) => request<{ cancelled: boolean }>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  glyphCheck: (text: string, font_family: string) => request<{ font_found: boolean; missing: string[]; scripts: string[] }>("/api/glyph-check", json("POST", { text, font_family })),
  mediaUrl: (id: string) => `/api/projects/${id}/media`,
  srtUrl: (id: string) => `/api/projects/${id}/subtitles.srt`,
  vttUrl: (id: string) => `/api/projects/${id}/subtitles.vtt`,
  downloadUrl: (exportId: string, kind: "video" | "srt" | "vtt") => `/api/exports/${exportId}/${kind}`,
};

/** Upload with progress. Returns the created project, or rejects with an ApiError. */
export function uploadVideo(file: File, onProgress: (p: number) => void, signal: { cancel?: () => void }): Promise<Project> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);
    xhr.open("POST", "/api/projects");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      let data: any = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new ApiError(data?.error?.message ?? `Upload failed (${xhr.status})`, data?.error?.code));
    };
    xhr.onerror = () => reject(new ApiError("Cannot reach the GARUDA server. Is it running?", "network"));
    xhr.onabort = () => reject(new ApiError("Upload cancelled.", "cancelled"));
    signal.cancel = () => xhr.abort();
    xhr.send(form);
  });
}

export async function pollJob(id: string, onUpdate: (j: Job) => void, stop: { stopped: boolean }): Promise<Job> {
  for (;;) {
    const j = await api.job(id);
    onUpdate(j);
    if (["completed", "failed", "cancelled"].includes(j.status) || stop.stopped) return j;
    await new Promise((r) => setTimeout(r, 800));
  }
}

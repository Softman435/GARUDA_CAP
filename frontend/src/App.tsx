import { useEffect, useState } from "react";
import { api, type Ready } from "./api";
import Dashboard from "./components/Dashboard";
import Editor from "./components/Editor";

export default function App() {
  const [projectId, setProjectId] = useState<string | null>(() => new URLSearchParams(location.search).get("p"));
  const [ready, setReady] = useState<Ready | null | undefined>(undefined);

  useEffect(() => { api.ready().then(setReady); }, []);

  const open = (id: string | null) => {
    setProjectId(id);
    history.replaceState(null, "", id ? `?p=${id}` : location.pathname);
  };

  return (
    <div className="app-shell flex h-full flex-col">
      <header className="app-header flex items-center justify-between px-5 py-3 sm:px-8">
        <button className="brand-lockup flex items-center gap-3 text-left" onClick={() => open(null)} aria-label="Go to dashboard">
          <span className="brand-mark" aria-hidden>🦅</span>
          <span>
            <span className="block font-bold tracking-[0.12em]">GARUDA <span className="text-saffron">STUDIO</span></span>
            <span className="brand-caption block text-[10px] uppercase tracking-[0.2em]">AI caption workspace</span>
          </span>
        </button>
        <span className="local-badge" title="Videos never leave this computer">
          <span className="status-dot" /> Local processing
        </span>
      </header>
      {ready === null && (
        <p role="alert" className="bg-red-900/60 px-4 py-2 text-sm">
          Cannot reach the GARUDA server. Start it with <code>scripts\start.bat</code> and refresh this page.
        </p>
      )}
      {ready && !ready.ready && (
        <p role="alert" className="bg-red-900/60 px-4 py-2 text-sm">
          FFmpeg was not found on this computer. Install it and restart (see docs/WINDOWS_SETUP.md).
        </p>
      )}
      {ready && ready.ready && !ready.faster_whisper && (
        <p role="alert" className="bg-amber-900/60 px-4 py-2 text-sm">
          Speech recognition is not installed. In the backend environment run: <code>pip install faster-whisper</code>
        </p>
      )}
      <main className="min-h-0 flex-1 overflow-auto">
        {projectId ? <Editor key={projectId} projectId={projectId} onClose={() => open(null)} /> : <Dashboard onOpen={open} />}
      </main>
    </div>
  );
}

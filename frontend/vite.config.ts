import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the browser talks to Vite (port 5173) and Vite forwards /api to the Python server (port 8000).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { environment: "node" },
});

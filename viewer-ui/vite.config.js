import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../aisuite/tracing/static/viewer",
    emptyOutDir: true,
  },
});

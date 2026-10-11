import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev only: the Vite server proxies /api to the Node backend (npm run dev runs both).
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: { "/api": "http://localhost:3000" },
  },
});

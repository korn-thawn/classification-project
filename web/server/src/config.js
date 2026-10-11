// Settings from environment variables; defaults suit running on the inspection laptop.
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
export const PROJECT_ROOT = resolve(here, "..", "..", "..");

export const PORT = Number(process.env.PORT ?? 3000);
export const HOST = process.env.HOST ?? "0.0.0.0"; // reachable from phones on the LAN
export const PY_API = (process.env.PY_API ?? "http://127.0.0.1:8000").replace(/\/$/, "");
export const RAW_DIR = resolve(PROJECT_ROOT, process.env.RAW_DIR ?? "data/raw");
export const CLIENT_DIST = resolve(here, "..", "..", "client", "dist");
export const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB ?? 10);
export const LABELS = ["OK", "NG"];

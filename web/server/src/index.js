// Express backend: serves the React build, forwards inspections to the Python
// FastAPI server (localhost only) and saves training samples into data/raw.
import express from "express";
import multer from "multer";
import { existsSync } from "node:fs";
import { mkdir, readdir, writeFile } from "node:fs/promises";
import { networkInterfaces } from "node:os";
import { join } from "node:path";
import {
  CLIENT_DIST, HOST, LABELS, MAX_UPLOAD_MB, PORT, PY_API, RAW_DIR,
} from "./config.js";

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: MAX_UPLOAD_MB * 1024 * 1024 } });
const SESSION_RE = /^[\w-]{1,64}$/;

/** Wrap multer so oversize files return 413 JSON instead of a generic 500. */
function singleFile(req, res, next) {
  upload.single("file")(req, res, (err) => {
    if (err?.code === "LIMIT_FILE_SIZE") return res.status(413).json({ detail: `file larger than ${MAX_UPLOAD_MB} MB` });
    if (err) return next(err);
    if (!req.file) return res.status(400).json({ detail: "missing field 'file'" });
    next();
  });
}

/** Call the Python API; throws a 502-tagged error if it is unreachable. */
async function callPython(path, init) {
  try {
    return await fetch(PY_API + path, init);
  } catch (e) {
    const err = new Error(`Python inference server ไม่ได้รัน (${PY_API}) - รัน python server.py ก่อน`);
    err.status = 502;
    err.cause = e;
    throw err;
  }
}

app.get("/api/health", async (_req, res) => {
  let python;
  try {
    const r = await callPython("/v1/health", { signal: AbortSignal.timeout(3000) });
    python = r.ok ? await r.json() : { status: "error", detail: (await r.json()).detail };
  } catch (e) {
    python = { status: "down", detail: e.message };
  }
  res.json({ node: "ok", python });
});

app.post("/api/inspect", singleFile, async (req, res, next) => {
  try {
    const form = new FormData();
    form.append("file", new Blob([req.file.buffer], { type: req.file.mimetype }), req.file.originalname || "image.jpg");
    const r = await callPython("/v1/inspect", { method: "POST", body: form });
    res.status(r.status).json(await r.json());
  } catch (e) {
    next(e);
  }
});

function timestamp() {
  const d = new Date();
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}-${p(d.getMilliseconds(), 3)}000`;
}

async function countImages(dir) {
  return existsSync(dir) ? (await readdir(dir)).filter((f) => /\.(png|jpe?g|bmp)$/i.test(f)).length : 0;
}

app.post("/api/samples", singleFile, async (req, res, next) => {
  try {
    const { label, session } = req.body;
    if (!LABELS.includes(label)) return res.status(400).json({ detail: `label must be one of ${LABELS.join(", ")}` });
    if (!SESSION_RE.test(session ?? "")) return res.status(400).json({ detail: "session must match [A-Za-z0-9_-]{1,64}" });
    const dir = join(RAW_DIR, session, label);
    await mkdir(dir, { recursive: true });
    const file = join(dir, `${timestamp()}.jpg`);
    await writeFile(file, req.file.buffer, { flag: "wx" });
    const counts = Object.fromEntries(await Promise.all(LABELS.map(async (l) => [l, await countImages(join(RAW_DIR, session, l))])));
    res.json({ saved: file, session, counts });
  } catch (e) {
    next(e);
  }
});

app.use("/api", (_req, res) => res.status(404).json({ detail: "not found" }));

if (existsSync(CLIENT_DIST)) {
  app.use(express.static(CLIENT_DIST));
  app.use((req, res, next) => (req.method === "GET" ? res.sendFile(join(CLIENT_DIST, "index.html")) : next()));
} else {
  app.get("/", (_req, res) => res.type("text").send("ยังไม่ได้ build client: รัน npm run build ก่อน (หรือใช้ npm run dev)"));
}

app.use((err, _req, res, _next) => {
  if (!err.status) console.error(err);
  res.status(err.status ?? 500).json({ detail: err.message });
});

app.listen(PORT, HOST, () => {
  console.log(`OK/NG web on port ${PORT} -> Python ${PY_API}, samples -> ${RAW_DIR}`);
  console.log(`  http://localhost:${PORT}`);
  for (const addrs of Object.values(networkInterfaces())) {
    for (const a of addrs ?? []) if (a.family === "IPv4" && !a.internal) console.log(`  http://${a.address}:${PORT}   (มือถือใน Wi-Fi เดียวกัน)`);
  }
});

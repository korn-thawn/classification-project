"""FastAPI inspection server.

POST /v1/inspect  multipart image -> {result, score, threshold, model_version, latency_ms}
GET  /v1/health   service + model status
GET  /            simple upload page
Run: python server.py   (config path can be overridden with env OKNG_CONFIG)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from vision.config import load_config
from vision.inference import Predictor, resolve_checkpoint

CONFIG_PATH = os.environ.get("OKNG_CONFIG", "config.yaml")
cfg = load_config(CONFIG_PATH)
MAX_BYTES = int(cfg["server"]["max_upload_mb"] * 1024 * 1024)
state: dict[str, Predictor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = resolve_checkpoint(cfg)
    pred = Predictor(ckpt, device=cfg["train"]["device"], default_threshold=cfg["inference"]["threshold"])
    pred.warmup()  # so the first request is not slow
    state["pred"] = pred
    print(f"loaded {ckpt} ({pred.model_version}) on {pred.device}, threshold {pred.threshold:.4f}")
    yield
    state.clear()


app = FastAPI(title="OK/NG inspection", version="0.1.0", lifespan=lifespan)


@app.get("/v1/health")
def health() -> dict:
    pred = state.get("pred")
    if pred is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ok", "model_version": pred.model_version, "device": str(pred.device),
            "threshold": pred.threshold, "classes": pred.classes}


@app.post("/v1/inspect")
def inspect(file: UploadFile = File(...)) -> dict:
    pred = state["pred"]
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {cfg['server']['max_upload_mb']} MB")
    t0 = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "cannot decode image")
    try:
        score = pred.score(img)
    except ValueError as e:  # e.g. ROI outside the uploaded image
        raise HTTPException(422, str(e)) from e
    return {
        "result": pred.decide(score),
        "score": round(score, 6),
        "threshold": round(pred.threshold, 6),
        "model_version": pred.model_version,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


INDEX_HTML = """<!doctype html>
<html lang="th"><head><meta charset="utf-8"><title>OK/NG inspection</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;color:#222}
 #drop{border:2px dashed #999;border-radius:8px;padding:24px;text-align:center}
 img{max-width:100%;max-height:360px;margin-top:16px;border-radius:6px}
 #res{font-size:48px;font-weight:700;margin-top:16px} .OK{color:#1a7f37} .NG{color:#cf222e}
 pre{background:#f6f8fa;padding:12px;border-radius:6px}
</style></head><body>
<h2>OK/NG inspection</h2>
<div id="drop"><input type="file" id="f" accept="image/*"><p>เลือกรูปเพื่อส่งตรวจ</p></div>
<img id="preview" hidden><div id="res"></div><pre id="json" hidden></pre>
<script>
const f = document.getElementById('f');
f.onchange = async () => {
  const file = f.files[0]; if (!file) return;
  const img = document.getElementById('preview');
  img.src = URL.createObjectURL(file); img.hidden = false;
  const res = document.getElementById('res'), pre = document.getElementById('json');
  res.textContent = '...'; res.className = '';
  const fd = new FormData(); fd.append('file', file);
  const r = await fetch('/v1/inspect', {method: 'POST', body: fd});
  const j = await r.json();
  pre.textContent = JSON.stringify(j, null, 2); pre.hidden = false;
  if (r.ok) { res.textContent = j.result; res.className = j.result; }
  else { res.textContent = 'error'; }
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


if __name__ == "__main__":
    uvicorn.run(app, host=cfg["server"]["host"], port=int(cfg["server"]["port"]))

// Thin wrappers around the Node backend; errors carry the server's `detail` text.
async function request(path, init) {
  const r = await fetch(path, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${r.status}`);
  return body;
}

export const getHealth = () => request("/api/health");

export function inspect(blob) {
  const fd = new FormData();
  fd.append("file", blob, "capture.jpg");
  return request("/api/inspect", { method: "POST", body: fd });
}

export function saveSample(blob, label, session) {
  const fd = new FormData();
  fd.append("label", label);
  fd.append("session", session);
  fd.append("file", blob, "sample.jpg");
  return request("/api/samples", { method: "POST", body: fd });
}

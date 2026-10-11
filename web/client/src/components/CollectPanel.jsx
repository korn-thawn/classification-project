import { useEffect, useState } from "react";
import { saveSample } from "../api.js";
import CameraCapture from "./CameraCapture.jsx";

// Session name format matches capture.py (YYYYMMDD-HHMMSS); one session = one
// day/lighting/lot, because train.py splits train/val/test by session.
function newSessionName() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

export default function CollectPanel() {
  const [session, setSession] = useState(() => localStorage.getItem("okng.session") || newSessionName());
  const [shot, setShot] = useState(null); // {blob, url, width, height}
  const [counts, setCounts] = useState(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  useEffect(() => localStorage.setItem("okng.session", session), [session]);
  useEffect(() => () => shot && URL.revokeObjectURL(shot.url), [shot]);

  async function save(label) {
    setBusy(true);
    try {
      const r = await saveSample(shot.blob, label, session);
      setCounts(r.counts);
      setMsg(`บันทึกเป็น ${label} แล้ว`);
      setShot(null);
    } catch (e) {
      setMsg(`บันทึกไม่ได้: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <label className="session">
        session
        <input value={session} onChange={(e) => setSession(e.target.value)} />
        <button onClick={() => { setSession(newSessionName()); setCounts(null); }}>ใหม่</button>
      </label>
      {counts && <p className="muted">session นี้: OK {counts.OK} · NG {counts.NG}</p>}
      <CameraCapture
        onCapture={(img) => { setMsg(""); setShot({ ...img, url: URL.createObjectURL(img.blob) }); }}
        disabled={busy}
      />
      {shot && (
        <>
          <img className="preview" src={shot.url} alt="ภาพที่จะบันทึก" />
          <p className="muted">{shot.width}×{shot.height}</p>
          <div className="row">
            <button className="label OK" disabled={busy} onClick={() => save("OK")}>OK</button>
            <button className="label NG" disabled={busy} onClick={() => save("NG")}>NG</button>
          </div>
        </>
      )}
      {msg && <p className="muted">{msg}</p>}
    </section>
  );
}

import { useEffect, useState } from "react";
import { inspect } from "../api.js";
import CameraCapture from "./CameraCapture.jsx";
import ResultCard from "./ResultCard.jsx";

export default function InspectPanel() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  async function onCapture({ blob }) {
    setPreview(URL.createObjectURL(blob));
    setResult(null);
    setError("");
    setBusy(true);
    try {
      setResult(await inspect(blob));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <CameraCapture onCapture={onCapture} disabled={busy} />
      {preview && <img className="preview" src={preview} alt="ภาพที่ส่งตรวจ" />}
      {busy && <p className="muted">กำลังตรวจ...</p>}
      {error && <p className="error">{error}</p>}
      {result && <ResultCard result={result} />}
    </section>
  );
}

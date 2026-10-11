import { useEffect, useRef, useState } from "react";
import { grabFrame, normalizeImage } from "../imageUtils.js";

// Phones on plain-HTTP LAN cannot use getUserMedia, so the main path is the
// file input with capture="environment" (opens the native camera app).
// Live preview is offered only in a secure context (e.g. http://localhost).
const LIVE_SUPPORTED = window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;

export default function CameraCapture({ onCapture, disabled }) {
  const inputRef = useRef(null);
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (videoRef.current) videoRef.current.srcObject = stream;
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, [stream]);

  async function handle(blob) {
    setError("");
    try {
      onCapture(await normalizeImage(blob));
    } catch (e) {
      setError(`อ่านภาพไม่ได้: ${e.message}`);
    }
  }

  async function toggleLive() {
    if (stream) return setStream(null);
    try {
      setStream(await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      }));
    } catch (e) {
      setError(`เปิดกล้องไม่ได้: ${e.message}`);
    }
  }

  return (
    <div className="camera">
      <video ref={videoRef} autoPlay playsInline muted hidden={!stream} />
      <div className="row">
        {stream ? (
          <button className="primary" disabled={disabled} onClick={async () => handle(await grabFrame(videoRef.current))}>
            ถ่าย
          </button>
        ) : (
          <button className="primary" disabled={disabled} onClick={() => inputRef.current.click()}>
            ถ่ายรูป / เลือกรูป
          </button>
        )}
        {LIVE_SUPPORTED && <button onClick={toggleLive}>{stream ? "ปิดกล้องสด" : "กล้องสด"}</button>}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        capture="environment"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = ""; // allow picking the same file again
          if (f) handle(f);
        }}
      />
      {error && <p className="error">{error}</p>}
    </div>
  );
}

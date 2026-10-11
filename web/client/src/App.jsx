import { useEffect, useState } from "react";
import { getHealth } from "./api.js";
import InspectPanel from "./components/InspectPanel.jsx";
import CollectPanel from "./components/CollectPanel.jsx";

export default function App() {
  const [tab, setTab] = useState("inspect");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const load = () => getHealth().then(setHealth).catch(() => setHealth({ node: "down" }));
    load();
    const id = setInterval(load, 10000);
    return () => clearInterval(id);
  }, []);

  const py = health?.python;
  return (
    <main>
      <header>
        <h1>OK/NG inspection</h1>
        <div className="status">
          <span className={health?.node === "ok" ? "dot ok" : "dot bad"} /> Node
          <span className={py?.status === "ok" ? "dot ok" : "dot bad"} /> Python
          {py?.model_version && <small> · {py.model_version} ({py.device})</small>}
        </div>
        {py && py.status !== "ok" && <p className="error">{py.detail}</p>}
      </header>
      <nav className="tabs">
        <button className={tab === "inspect" ? "active" : ""} onClick={() => setTab("inspect")}>ตรวจ</button>
        <button className={tab === "collect" ? "active" : ""} onClick={() => setTab("collect")}>เก็บข้อมูล</button>
      </nav>
      {tab === "inspect" ? <InspectPanel /> : <CollectPanel />}
    </main>
  );
}

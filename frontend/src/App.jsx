import { useEffect, useMemo, useState } from "react";

const API = "http://127.0.0.1:8000";
const WS = "ws://127.0.0.1:8000/ws/fallback";

const empty = {
  prediction: null,
  circuit_breaker: null,
  fallback: {
    action: "NORMAL",
    state: "CLOSED",
    driver_present: true,
    acknowledged: false,
    countdown_s: 0,
    reason: "Upload a traffic-sign image to begin live analysis.",
    critical: false,
  },
};

const familyNames = {
  geometric: "Geometric",
  photometric: "Photometric",
  sensor_noise: "Sensor / Noise",
};

function toneFor(state) {
  return state === "OPEN" ? "open" : state === "HALF_OPEN" ? "half" : "closed";
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(empty);
  const [driverPresent, setDriverPresent] = useState(true);
  const [busy, setBusy] = useState(false);
  const [online, setOnline] = useState(false);
  const [tab, setTab] = useState("live");

  useEffect(() => {
    const ws = new WebSocket(WS);
    ws.onopen = () => setOnline(true);
    ws.onclose = () => setOnline(false);
    ws.onerror = () => setOnline(false);
    ws.onmessage = (event) => {
      try { setResult(JSON.parse(event.data)); } catch {}
    };
    return () => ws.close();
  }, []);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function evaluate() {
    if (!file) return;
    setBusy(true);
    try {
      const form = new FormData();
      form.append("image", file);
      const response = await fetch(
        `${API}/api/evaluate?driver_present=${driverPresent}`,
        { method: "POST", body: form }
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Evaluation failed.");
      setResult(data);
      setTab("live");
    } catch (error) {
      setResult((current) => ({
        ...current,
        fallback: {
          ...current.fallback,
          action: "SAFE_PULL_OVER",
          critical: true,
          reason: error.message,
        },
      }));
    } finally {
      setBusy(false);
    }
  }

  async function acknowledge() {
    const response = await fetch(`${API}/api/takeover/ack`, { method: "POST" });
    if (!response.ok) return;
    setResult((current) => ({
      ...current,
      fallback: { ...current.fallback, acknowledged: true },
    }));
  }

  const cb = result.circuit_breaker;
  const prediction = result.prediction;
  const fallback = result.fallback || empty.fallback;
  const state = cb?.state || fallback.state || "CLOSED";
  const tone = toneFor(state);
  const tests = cb?.test_results || [];
  const flips = tests.filter((x) => x.flipped);
  const family = cb?.family_instability || {};
  const confidence = prediction ? Number(prediction.confidence) * 100 : 0;
  const cbi = cb ? Number(cb.cbi) : null;
  const config = cb?.details?.config || {};
  const thresholds = {
    warn: Number(config.theta_warn ?? 0.25),
    trip: Number(config.theta_trip ?? 0.55),
  };

  const connectionText = useMemo(
    () => (online ? "LIVE INFERENCE ONLINE" : "BACKEND OFFLINE"),
    [online]
  );

  function chooseFile(e) {
    const next = e.target.files?.[0];
    if (next) setFile(next);
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <div className="live-badge"><i /> {connectionText}</div>
          <h1>Metamorphic Circuit Breaker Studio</h1>
          <p>Runtime transform-brittleness detection and safety fallback for vision models.</p>
        </div>
        <div className="header-meta">
          <span>GTSRB</span>
          <span>MobileNetV3-Small</span>
          <span>43 classes</span>
        </div>
      </header>

      <nav className="tabs">
        <button className={tab === "live" ? "active" : ""} onClick={() => setTab("live")}>Live Circuit Breaker</button>
        <button className={tab === "diagnostics" ? "active" : ""} onClick={() => setTab("diagnostics")}>Diagnostics</button>
        <button className={tab === "fallback" ? "active" : ""} onClick={() => setTab("fallback")}>Safety Fallback</button>
      </nav>

      {tab === "live" && (
        <>
          <section className={`state-banner ${tone}`}>
            <strong>
              {state === "CLOSED" && "CLOSED | AUTO-APPROVED"}
              {state === "HALF_OPEN" && "HALF-OPEN | MONITOR / WARNING"}
              {state === "OPEN" && "OPEN | SAFETY INTERCEPT | PREDICTION BLOCKED"}
            </strong>
            <span>CBI = {cbi == null ? "—" : cbi.toFixed(4)} &nbsp;|&nbsp; {(cb?.action || "WAITING").replaceAll("_", " ")}</span>
          </section>

          <section className="metrics">
            <div className="metric-card">
              <label>BASELINE PREDICTION</label>
              <strong>{prediction?.label || "—"}</strong>
              <small>{prediction ? `Class ${prediction.index}` : "Awaiting image"}</small>
            </div>
            <div className="metric-card">
              <label>CONFIDENCE (c₀)</label>
              <strong>{prediction ? `${confidence.toFixed(1)}%` : "—"}</strong>
              <div className="bar"><b style={{width: `${confidence}%`}} /></div>
            </div>
            <div className="metric-card">
              <label>PREDICTION FLIPS</label>
              <strong>{tests.length ? `${flips.length} / ${tests.length}` : "—"}</strong>
              <small>{flips.length ? "Instability detected" : "No tests executed"}</small>
            </div>
            <div className="metric-card">
              <label>CBI SCORE</label>
              <strong>{cbi == null ? "—" : cbi.toFixed(4)}</strong>
              <small>Warn {thresholds.warn.toFixed(2)} · Trip {thresholds.trip.toFixed(2)}</small>
            </div>
          </section>

          <section className="source-grid">
            <div className="card">
              <div className="card-title">
                <div><label>INPUT IMAGE</label><h2>Traffic Sign</h2></div>
                <span className="muted">JPG / PNG</span>
              </div>
              <label className={`uploader ${preview ? "filled" : ""}`}>
                {preview ? <img src={preview} alt="Selected traffic sign" /> : (
                  <div>
                    <b>Choose a traffic-sign image</b>
                    <span>MobileNetV3-Small will evaluate the baseline and metamorphic variants.</span>
                  </div>
                )}
                <input type="file" accept="image/*" onChange={chooseFile} />
              </label>
              <div className="driver-row">
                <label>Driver status</label>
                <select value={driverPresent ? "present" : "absent"} onChange={(e) => setDriverPresent(e.target.value === "present")}>
                  <option value="present">Driver present</option>
                  <option value="absent">Driver absent / autonomous</option>
                </select>
              </div>
              <button className="primary" disabled={!file || busy} onClick={evaluate}>
                {busy ? "Running 22-test analysis…" : "Run Live Metamorphic Circuit Breaker"}
              </button>
            </div>

            <div className="card">
              <div className="card-title">
                <div><label>DIAGNOSTIC SUMMARY</label><h2>Runtime Analysis</h2></div>
                <span className={`state-pill ${tone}`}>{state}</span>
              </div>
              <div className="summary-list">
                <div><span>Peak unstable family</span><b>{familyNames[cb?.peak_family] || cb?.peak_family || "—"}</b></div>
                <div><span>Peak family IRₖ</span><b>{cb ? Number(cb.peak_family_score).toFixed(4) : "—"}</b></div>
                <div><span>Cross-family spread</span><b>{cb ? Number(cb.cross_family_spread).toFixed(4) : "—"}</b></div>
                <div><span>Compromised families</span><b>{cb?.compromised_families?.length ?? "—"}</b></div>
              </div>
              <div className="decision-box">
                <label>CURRENT ACTION</label>
                <strong>{(cb?.action || "WAITING FOR ANALYSIS").replaceAll("_", " ")}</strong>
                <span>{fallback.reason}</span>
              </div>
            </div>
          </section>
        </>
      )}

      {tab === "diagnostics" && (
        <section className="diagnostics">
          <div className="card">
            <div className="card-title">
              <div><label>CBI ATTRIBUTION</label><h2>Composite Brittleness Index</h2></div>
              <span className={`state-pill ${tone}`}>{state}</span>
            </div>
            <div className="gauge">
              <div className="gauge-track">
                <span className="zone green" />
                <span className="zone yellow" />
                <span className="zone red" />
                {cbi != null && <i style={{left: `${Math.min(cbi,1)*100}%`}} />}
              </div>
              <div className="gauge-labels"><span>0</span><span>theta_warn {thresholds.warn}</span><span>theta_trip {thresholds.trip}</span><span>1.0</span></div>
            </div>
            <div className="formula">
              CBI(x) = α · max IRₖ(x) + β · CFS(x) + γ · (1 − c₀)
              <br />
              {cbi == null ? "Run an evaluation to populate the attribution." :
                `Observed CBI = ${cbi.toFixed(4)}`}
            </div>
          </div>

          <div className="two-col">
            <div className="card">
              <div className="card-title"><div><label>LEVEL 2</label><h2>Family Instability IRₖ</h2></div></div>
              {Object.entries(family).length ? Object.entries(family).map(([key, value]) => (
                <div className="family-row" key={key}>
                  <span>{familyNames[key] || key}</span>
                  <div className="mini-bar"><b style={{width: `${Number(value)*100}%`}} /></div>
                  <strong>{Number(value).toFixed(3)}</strong>
                </div>
              )) : <p className="empty">No family results yet.</p>}
            </div>
            <div className="card">
              <div className="card-title"><div><label>LEVEL 1</label><h2>Transformation Scores</h2></div></div>
              {Object.entries(cb?.type_scores || {}).length ? Object.entries(cb.type_scores).map(([key, value]) => (
                <div className="family-row" key={key}>
                  <span>{key}</span>
                  <div className="mini-bar"><b style={{width: `${Number(value)*100}%`}} /></div>
                  <strong>{Number(value).toFixed(3)}</strong>
                </div>
              )) : <p className="empty">No transformation scores yet.</p>}
            </div>
          </div>

          <div className="card">
            <div className="card-title"><div><label>FULL MATRIX</label><h2>Metamorphic Test Results</h2></div><span>{tests.length} tests</span></div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>ID</th><th>Transform</th><th>Family</th><th>Severity</th><th>Weight</th><th>Prediction</th><th>Conf.</th><th>Result</th></tr></thead>
                <tbody>
                  {tests.map((t) => (
                    <tr key={t.id}>
                      <td>{t.id}</td><td>{t.transform}</td><td>{familyNames[t.family] || t.family}</td>
                      <td>{t.severity}</td><td>{Number(t.weight).toFixed(1)}</td><td>{t.prediction}</td>
                      <td>{(Number(t.confidence)*100).toFixed(1)}%</td>
                      <td><span className={t.flipped ? "flip" : "stable"}>{t.flipped ? "FLIP" : "STABLE"}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      {tab === "fallback" && (
        <section className="fallback-page">
          <div className={`fallback-hero ${tone}`}>
            <label>FALLBACK DECISION</label>
            <strong>{fallback.action.replaceAll("_", " ")}</strong>
            <p>{fallback.reason}</p>
          </div>
          <div className="two-col">
            <div className="card">
              <label>DRIVER / OCCUPANCY</label>
              <h2>{fallback.driver_present ? "Driver Present" : "Driver Absent"}</h2>
              <p className="muted">Occupancy is supplied by the caller; it is not inferred by the ML model.</p>
              <div className="occupancy"><span className={fallback.driver_present ? "on" : ""}>DRIVER PRESENT</span><span className={!fallback.driver_present ? "on" : ""}>DRIVER ABSENT</span></div>
            </div>
            <div className="card">
              <label>TAKEOVER STATUS</label>
              <h2>{fallback.acknowledged ? "Acknowledged" : fallback.action === "TAKEOVER_REQUESTED" ? "Awaiting acknowledgement" : "Not requested"}</h2>
              {fallback.action === "TAKEOVER_REQUESTED" && !fallback.acknowledged && (
                <button className="takeover" onClick={acknowledge}>Acknowledge Driver Takeover</button>
              )}
              {fallback.action === "TAKEOVER_REQUESTED" && !fallback.acknowledged && <div className="countdown">{Number(fallback.countdown_s).toFixed(1)}<small>s</small></div>}
              <p className="notice">SAFE_PULL_OVER is a simulated safety decision in this prototype; no physical vehicle actuator is controlled.</p>
            </div>
          </div>
        </section>
      )}

      <footer>
        <span>METAMORPHIC CIRCUIT BREAKER · LIVE PYTORCH INFERENCE</span>
        <span>22 metamorphic tests · 3 semantic families · external safety layer</span>
      </footer>
    </div>
  );
}

export default App;

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
  const [countdown, setCountdown] = useState(0);

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

  useEffect(() => {
    if (fallback.action !== "TAKEOVER_REQUESTED" || fallback.acknowledged) {
      setCountdown(Number(fallback.countdown_s || 0));
      return;
    }
    setCountdown(Number(fallback.countdown_s || 0));
    const timer = setInterval(() => {
      setCountdown((value) => {
        const next = Math.max(0, value - 0.1);
        return Number(next.toFixed(1));
      });
    }, 100);
    return () => clearInterval(timer);
  }, [fallback.action, fallback.acknowledged, fallback.countdown_s]);

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
          <h1>Metamorphic Circuit Breaker</h1>
          <p>Runtime robustness monitoring for vision-based perception.</p>
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
        <button className={tab === "fallback" ? "active" : ""} onClick={() => setTab("fallback")}>Safety Response</button>
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
          <div className="card gradcam-section">
            <div className="card-title">
              <div><label>MODEL EXPLAINABILITY</label><h2>Grad-CAM Attention & Comparison</h2></div>
              <span className="muted">{result.gradcam?.comparisons?.length || 0} flip comparison(s)</span>
            </div>
            {result.gradcam ? (
              <>
                <p className="diagnostic-copy">Grad-CAM highlights image regions contributing to the selected prediction. Comparisons use the actual transformed images that triggered prediction flips.</p>
                <div className="gradcam-grid">
                  <div className="cam-card baseline">
                    <div className="cam-label"><span>BASELINE</span><b>{result.gradcam.baseline.label}</b></div>
                    <div className="cam-images"><img src={result.gradcam.baseline.image} alt="Baseline input" /><img src={result.gradcam.baseline.heatmap} alt="Baseline Grad-CAM heatmap" /></div>
                    <div className="cam-caption">Original image → model attention</div>
                    <strong>{(Number(result.gradcam.baseline.confidence) * 100).toFixed(1)}% confidence</strong>
                  </div>
                  {result.gradcam.comparisons.map((item) => (
                    <div className={`cam-card ${item.prediction_changed ? "changed" : ""}`} key={item.test_id}>
                      <div className="cam-label"><span>{item.test_id} · {item.test_name}</span><b>{item.label}</b></div>
                      <div className="cam-images"><img src={item.image} alt={`${item.test_name} transformed image`} /><img src={item.heatmap} alt={`${item.test_name} Grad-CAM heatmap`} /></div>
                      <div className="cam-caption">{item.prediction_changed ? "ATTENTION / PREDICTION SHIFT" : "ATTENTION COMPARISON"}</div>
                      <strong>{(Number(item.confidence) * 100).toFixed(1)}% confidence</strong>
                    </div>
                  ))}
                </div>
                {result.gradcam.comparisons.length === 0 && <div className="cam-empty">No prediction flips required a comparison. The baseline Grad-CAM above shows where the model focused.</div>}
              </>
            ) : <div className="cam-empty">Run a live evaluation to generate the baseline Grad-CAM and comparison maps.</div>}
          </div>
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
              <label>DRIVER STATUS</label>
              <h2>{fallback.driver_present ? "Driver Present" : "Driver Absent"}</h2>
              <p className="muted">Driver availability is provided by the test operator and is not inferred by the perception model. If no driver is available, the system selects a safe pull-over immediately.</p>
              <div className="occupancy"><span className={fallback.driver_present ? "on" : ""}>DRIVER PRESENT</span><span className={!fallback.driver_present ? "on" : ""}>DRIVER ABSENT</span></div>
            </div>
            <div className="card">
              <label>DRIVER TAKEOVER</label>
              <h2>{fallback.acknowledged ? "Takeover acknowledged" : fallback.action === "TAKEOVER_REQUESTED" ? "Awaiting driver response" : fallback.action === "SAFE_PULL_OVER" ? "Safe pull-over selected" : "No takeover requested"}</h2>
              {fallback.action === "TAKEOVER_REQUESTED" && !fallback.acknowledged && (
                <button className="takeover" onClick={acknowledge}>Confirm Driver Takeover</button>
              )}
              {fallback.action === "TAKEOVER_REQUESTED" && !fallback.acknowledged && <div className="countdown">{countdown.toFixed(1)}<small>s remaining</small></div>}
              <p className="notice">Prototype only · Safe pull-over is a simulated safety response. No physical vehicle controls are connected.</p>
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

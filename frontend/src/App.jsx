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

  // Calibration Studio State
  const [calibData, setCalibData] = useState(null);
  const [alpha, setAlpha] = useState(0.50);
  const [beta, setBeta] = useState(0.35);
  const [gamma, setGamma] = useState(0.15);
  const [tauFam, setTauFam] = useState(0.05);
  const [thetaTrip, setThetaTrip] = useState(0.30);
  const [thetaWarn, setThetaWarn] = useState(0.150);
  const [autoDeriveWarn, setAutoDeriveWarn] = useState(true);
  const [activePresetId, setActivePresetId] = useState("op_j");
  const [activeChart, setActiveChart] = useState("calibration_heatmap");
  const [calibMsg, setCalibMsg] = useState("");
  const [calibSaving, setCalibSaving] = useState(false);
  const [simIr, setSimIr] = useState(0.40);
  const [simCfs, setSimCfs] = useState(0.33);
  const [simUnc, setSimUnc] = useState(0.20);

  async function fetchCalibration() {
    try {
      const res = await fetch(`${API}/api/calibration`);
      if (res.ok) {
        const data = await res.json();
        setCalibData(data);
        if (data.active_config) {
          setAlpha(Number(data.active_config.alpha));
          setBeta(Number(data.active_config.beta));
          setGamma(Number(data.active_config.gamma));
          setTauFam(Number(data.active_config.tau_fam));
          setThetaTrip(Number(data.active_config.theta_trip));
          setThetaWarn(Number(data.active_config.theta_warn));
        }
      }
    } catch {}
  }

  useEffect(() => {
    fetchCalibration();
  }, []);

  useEffect(() => {
    const ws = new WebSocket(WS);
    ws.onopen = () => setOnline(true);
    ws.onclose = () => setOnline(false);
    ws.onerror = () => setOnline(false);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "config_updated") {
          fetchCalibration();
        } else {
          setResult(data);
        }
      } catch {}
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

  function selectPreset(preset) {
    setAlpha(preset.alpha);
    setBeta(preset.beta);
    setGamma(preset.gamma);
    setTauFam(preset.tau_fam);
    setThetaTrip(preset.theta_trip);
    setThetaWarn(preset.theta_warn);
    setActivePresetId(preset.id);
    setCalibMsg(`Loaded preset: "${preset.name}". Click "Apply to Live Circuit Breaker" to activate.`);
  }

  function handleThetaTripChange(val) {
    const next = Number(val);
    setThetaTrip(next);
    if (autoDeriveWarn) {
      setThetaWarn(Number((0.5 * next).toFixed(3)));
    }
  }

  function autoNormalizeWeights() {
    const sum = alpha + beta + gamma;
    if (sum <= 0) return;
    const a = Number((alpha / sum).toFixed(2));
    const b = Number((beta / sum).toFixed(2));
    const g = Number((1.0 - a - b).toFixed(2));
    setAlpha(a);
    setBeta(b);
    setGamma(g);
    setCalibMsg("Weights normalized to sum = 1.00");
  }

  async function applyCalibration() {
    setCalibSaving(true);
    try {
      const res = await fetch(`${API}/api/calibration/update`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          alpha,
          beta,
          gamma,
          tau_fam: tauFam,
          theta_trip: thetaTrip,
          theta_warn: thetaWarn,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Update failed");
      setCalibMsg("Active Circuit Breaker successfully updated! Subsequent evaluations will use these parameters.");
      fetchCalibration();
    } catch (err) {
      setCalibMsg(`Update failed: ${err.message}`);
    } finally {
      setCalibSaving(false);
    }
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
    const timer = setInterval(async () => {
      setCountdown((value) => {
        const next = Math.max(0, value - 0.1);
        if (next <= 0) {
          fetch(`${API}/api/takeover/status`)
            .then((response) => response.ok ? response.json() : null)
            .then((data) => {
              if (data) setResult((current) => ({ ...current, fallback: data }));
            })
            .catch(() => {});
        }
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
        <button className={tab === "calibration" ? "active" : ""} onClick={() => setTab("calibration")}>Parameter Calibration Studio</button>
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

      {tab === "calibration" && (
        <section className="calib-studio">
          <div className="studio-header">
            <div>
              <label>RUNTIME CALIBRATION & BENCHMARKING</label>
              <h2>Parameter Calibration Studio</h2>
              <p>Tune composite brittleness index (CBI) weights and three-state decision boundaries with live scenario simulation and empirical evaluation metrics.</p>
            </div>
            <div className="active-config-pill">
              ACTIVE: α={calibData?.active_config?.alpha ?? alpha} · β={calibData?.active_config?.beta ?? beta} · γ={calibData?.active_config?.gamma ?? gamma} | τ={calibData?.active_config?.tau_fam ?? tauFam} | θ_trip={calibData?.active_config?.theta_trip ?? thetaTrip} | θ_warn={calibData?.active_config?.theta_warn ?? thetaWarn}
            </div>
          </div>

          <div className="card">
            <div className="card-title">
              <div>
                <label>EMPIRICAL PRESETS</label>
                <h2>Calibrated Operating Points</h2>
              </div>
              <span className="muted">Rigorous GTSRB v4 Calibration</span>
            </div>
            <div className="presets-grid">
              {calibData?.presets && Object.values(calibData.presets).map((p) => (
                <div className={`preset-card ${activePresetId === p.id ? "active" : ""}`} key={p.id}>
                  <div className="preset-top">
                    <div>
                      <h3>{p.name}</h3>
                      <span className="preset-badge">{p.badge}</span>
                    </div>
                  </div>
                  <p className="preset-desc">{p.description}</p>
                  <div className="preset-params">
                    <span>τ: <b>{p.tau_fam}</b></span>
                    <span>θ_trip: <b>{p.theta_trip}</b></span>
                    <span>θ_warn: <b>{p.theta_warn}</b></span>
                    <span>α/β/γ: <b>{p.alpha}/{p.beta}/{p.gamma}</b></span>
                  </div>
                  <div className="preset-metrics">
                    <div><span>Recall:</span><strong>{(p.recall * 100).toFixed(1)}%</strong></div>
                    <div><span>FAR:</span><strong>{(p.far * 100).toFixed(1)}%</strong></div>
                    <div><span>Youden J:</span><strong>{p.youden_j.toFixed(4)}</strong></div>
                  </div>
                  <button className="preset-btn" onClick={() => selectPreset(p)}>
                    {activePresetId === p.id ? "✓ Currently Selected" : "Load Preset"}
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="studio-row">
            <div className="card">
              <div className="card-title">
                <div>
                  <label>INTERACTIVE TUNER</label>
                  <h2>Formula Weights & Decision Thresholds</h2>
                </div>
              </div>

              <div className="slider-block">
                <div className="slider-label-row">
                  <span>α — Peak Family Instability max_k(IR_k)</span>
                  <span className="slider-val">{alpha.toFixed(2)}</span>
                </div>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={alpha} onChange={(e) => { setAlpha(Number(e.target.value)); setActivePresetId("custom"); }}
                  className="calib-slider"
                />
                <span className="slider-hint">Weight attributed to the single most degraded metamorphic family.</span>
              </div>

              <div className="slider-block">
                <div className="slider-label-row">
                  <span>β — Cross-Family Spread Score (CFS)</span>
                  <span className="slider-val">{beta.toFixed(2)}</span>
                </div>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={beta} onChange={(e) => { setBeta(Number(e.target.value)); setActivePresetId("custom"); }}
                  className="calib-slider"
                />
                <span className="slider-hint">Fraction of semantic families exceeding the τ_fam threshold.</span>
              </div>

              <div className="slider-block">
                <div className="slider-label-row">
                  <span>γ — Baseline Uncertainty (1 − c₀)</span>
                  <span className="slider-val">{gamma.toFixed(2)}</span>
                </div>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={gamma} onChange={(e) => { setGamma(Number(e.target.value)); setActivePresetId("custom"); }}
                  className="calib-slider"
                />
                <span className="slider-hint">Inverted softmax confidence margin of the original unperturbed prediction.</span>
              </div>

              {(() => {
                const wSum = Number((alpha + beta + gamma).toFixed(2));
                const isOk = Math.abs(wSum - 1.0) <= 0.015;
                return (
                  <div className={`weight-sum-alert ${isOk ? "ok" : "warn"}`}>
                    <span>Total Weight Sum: <b>{wSum.toFixed(2)}</b> / 1.00 {isOk ? "(Valid constraint)" : "(Must sum to 1.00)"}</span>
                    {!isOk && <button className="small-btn" onClick={autoNormalizeWeights}>Auto-Normalize</button>}
                  </div>
                );
              })()}

              <hr style={{ borderColor: "#29253a", margin: "16px 0" }} />

              <div className="slider-block">
                <div className="slider-label-row">
                  <span>τ_fam — Family Compromise Threshold</span>
                  <span className="slider-val">{tauFam.toFixed(2)}</span>
                </div>
                <input
                  type="range" min="0.05" max="0.50" step="0.05"
                  value={tauFam} onChange={(e) => { setTauFam(Number(e.target.value)); setActivePresetId("custom"); }}
                  className="calib-slider"
                />
                <span className="slider-hint">Individual family IR_k must exceed τ_fam to count toward cross-family spread.</span>
              </div>

              <div className="slider-block">
                <div className="slider-label-row">
                  <span>θ_trip — Tripped / OPEN Threshold</span>
                  <span className="slider-val">{thetaTrip.toFixed(2)}</span>
                </div>
                <input
                  type="range" min="0.10" max="0.90" step="0.05"
                  value={thetaTrip} onChange={(e) => { handleThetaTripChange(e.target.value); setActivePresetId("custom"); }}
                  className="calib-slider"
                />
                <span className="slider-hint">CBI ≥ θ_trip trips the breaker, intercepts perception, and requests safety response.</span>
              </div>

              <div className="slider-block">
                <div className="slider-label-row">
                  <span>θ_warn — Monitor / HALF-OPEN Threshold</span>
                  <span className="slider-val">{thetaWarn.toFixed(3)}</span>
                </div>
                <input
                  type="range" min="0.05" max="0.50" step="0.025"
                  value={thetaWarn}
                  disabled={autoDeriveWarn}
                  onChange={(e) => { setThetaWarn(Number(e.target.value)); setAutoDeriveWarn(false); setActivePresetId("custom"); }}
                  className="calib-slider"
                />
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4 }}>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, cursor: "pointer", color: "#a5a0b7" }}>
                    <input
                      type="checkbox" checked={autoDeriveWarn}
                      onChange={(e) => {
                        setAutoDeriveWarn(e.target.checked);
                        if (e.target.checked) setThetaWarn(Number((0.5 * thetaTrip).toFixed(3)));
                      }}
                    />
                    Auto-derive θ_warn = 0.5 × θ_trip (Section 3.4 rule)
                  </label>
                  {thetaWarn >= thetaTrip && <span style={{ color: "#ff7188", fontSize: 11 }}>θ_warn must be &lt; θ_trip!</span>}
                </div>
              </div>

              <div className="actions-bar">
                <button
                  className="primary"
                  disabled={calibSaving || Math.abs((alpha + beta + gamma) - 1.0) > 0.015 || thetaWarn >= thetaTrip}
                  onClick={applyCalibration}
                >
                  {calibSaving ? "Applying parameters..." : "Apply to Live Circuit Breaker"}
                </button>
              </div>
              {calibMsg && <div className="calib-msg">{calibMsg}</div>}
            </div>

            <div className="card">
              <div className="card-title">
                <div>
                  <label>STATE MACHINE & SIMULATOR</label>
                  <h2>Live Formula & Scenario Testing</h2>
                </div>
              </div>

              <div className="formula" style={{ marginBottom: 16 }}>
                <b>Active CBI Formula:</b>
                <br />
                CBI(x) = {alpha.toFixed(2)} · max_k IR_k  +  {beta.toFixed(2)} · CFS  +  {gamma.toFixed(2)} · (1 − c₀)
                <br /><br />
                <b>Decision Boundaries:</b>
                <br />
                CLOSED    if CBI &lt; {thetaWarn.toFixed(3)} (Auto-Approved)
                <br />
                HALF-OPEN if {thetaWarn.toFixed(3)} ≤ CBI &lt; {thetaTrip.toFixed(2)} (Warning Monitor)
                <br />
                OPEN      if CBI ≥ {thetaTrip.toFixed(2)} (Trip Intercept)
              </div>

              <div style={{ padding: "16px", background: "#100e18", border: "1px solid #29253a", borderRadius: "10px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <label style={{ margin: 0 }}>HYPOTHETICAL INPUT SIMULATOR</label>
                  <span className="muted">Test state machine transitions</span>
                </div>

                <div className="slider-block" style={{ marginBottom: 10 }}>
                  <div className="slider-label-row">
                    <span style={{ fontSize: 12 }}>Simulated max_k IR_k</span>
                    <span className="slider-val">{simIr.toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min="0" max="1" step="0.05"
                    value={simIr} onChange={(e) => setSimIr(Number(e.target.value))}
                    className="calib-slider"
                  />
                </div>

                <div className="slider-block" style={{ marginBottom: 10 }}>
                  <div className="slider-label-row">
                    <span style={{ fontSize: 12 }}>Simulated CFS</span>
                    <span className="slider-val">{simCfs.toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min="0" max="1" step="0.3333"
                    value={simCfs} onChange={(e) => setSimCfs(Number(e.target.value))}
                    className="calib-slider"
                  />
                </div>

                <div className="slider-block" style={{ marginBottom: 10 }}>
                  <div className="slider-label-row">
                    <span style={{ fontSize: 12 }}>Simulated Uncertainty (1 − c₀)</span>
                    <span className="slider-val">{simUnc.toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min="0" max="1" step="0.05"
                    value={simUnc} onChange={(e) => setSimUnc(Number(e.target.value))}
                    className="calib-slider"
                  />
                </div>

                {(() => {
                  const sCbi = Number((alpha * simIr + beta * simCfs + gamma * simUnc).toFixed(4));
                  const sTone = sCbi < thetaWarn ? "closed" : sCbi < thetaTrip ? "half" : "open";
                  const sState = sCbi < thetaWarn ? "CLOSED (Normal Operation)" : sCbi < thetaTrip ? "HALF-OPEN (Warning Flag)" : "OPEN (Safety Intercept)";
                  return (
                    <div>
                      <div className="gauge">
                        <div className="gauge-track">
                          <span className="zone green" style={{ width: `${Math.min(100, thetaWarn * 100)}%` }} />
                          <span className="zone yellow" style={{ width: `${Math.min(100, (thetaTrip - thetaWarn) * 100)}%` }} />
                          <span className="zone red" style={{ width: `${Math.max(0, (1 - thetaTrip) * 100)}%` }} />
                          <i style={{ left: `${Math.min(100, Math.max(0, sCbi * 100))}%` }} />
                        </div>
                        <div className="gauge-labels">
                          <span>0</span>
                          <span>θ_warn {thetaWarn.toFixed(2)}</span>
                          <span>θ_trip {thetaTrip.toFixed(2)}</span>
                          <span>1.0</span>
                        </div>
                      </div>
                      <div className={`sim-result state-banner ${sTone}`} style={{ margin: "10px 0 0" }}>
                        <div>Simulated CBI = <strong>{sCbi.toFixed(4)}</strong> → <b>{sState}</b></div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title">
              <div>
                <label>EXPERIMENTAL AUDIT & RESEARCH VISUALS</label>
                <h2>Standardized v4 Validation Results</h2>
              </div>
              <span className="muted">Device: Intel CPU · Held-Out Pool N=11,763</span>
            </div>

            <div className="chart-pills">
              {calibData?.charts && calibData.charts.map((c) => (
                <button
                  key={c.id}
                  className={`chart-pill ${activeChart === c.id ? "active" : ""}`}
                  onClick={() => setActiveChart(c.id)}
                >
                  {c.title}
                </button>
              ))}
            </div>

            {calibData?.charts && (() => {
              const cur = calibData.charts.find((c) => c.id === activeChart) || calibData.charts[0];
              return (
                <div className="chart-display">
                  <img
                    src={`${API}/api/calibration/chart/${cur.filename}`}
                    alt={cur.title}
                    className="chart-img"
                  />
                  <div className="chart-desc">
                    <strong>{cur.title}:</strong> {cur.desc}
                  </div>
                </div>
              );
            })()}
          </div>

          {calibData?.grid_points && calibData.grid_points.length > 0 && (
            <div className="card">
              <div className="card-title">
                <div>
                  <label>PARTITION A GRID SEARCH</label>
                  <h2>Top 25 Calibrated Operating Points (Sorted by Youden's J)</h2>
                </div>
                <span className="muted">Click "Load" to test any point</span>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Action</th>
                      <th>τ_fam</th>
                      <th>θ_trip</th>
                      <th>θ_warn</th>
                      <th>Recall</th>
                      <th>FAR</th>
                      <th>Youden's J</th>
                      <th>Precision</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calibData.grid_points.map((pt, idx) => (
                      <tr key={idx}>
                        <td>
                          <button
                            className="load-btn"
                            onClick={() => {
                              setTauFam(pt.tau_fam);
                              setThetaTrip(pt.theta_trip);
                              setThetaWarn(pt.theta_warn);
                              setActivePresetId("custom");
                              setCalibMsg(`Loaded grid point: τ=${pt.tau_fam}, θ=${pt.theta_trip}`);
                            }}
                          >
                            Load
                          </button>
                        </td>
                        <td>{pt.tau_fam.toFixed(2)}</td>
                        <td>{pt.theta_trip.toFixed(2)}</td>
                        <td>{pt.theta_warn.toFixed(3)}</td>
                        <td>{(pt.recall * 100).toFixed(2)}%</td>
                        <td>{(pt.far * 100).toFixed(2)}%</td>
                        <td><b style={{ color: "#7ee7bf" }}>{pt.youden_j.toFixed(4)}</b></td>
                        <td>{(pt.precision * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
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

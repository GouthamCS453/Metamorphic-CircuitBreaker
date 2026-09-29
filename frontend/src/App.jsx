import { useEffect, useMemo, useState } from "react";

const API = "http://127.0.0.1:8000";
const WS = "ws://127.0.0.1:8000/ws/fallback";

const initialResult = {
  prediction: null,
  circuit_breaker: null,
  fallback: {
    action: "NORMAL",
    state: "CLOSED",
    driver_present: true,
    acknowledged: false,
    countdown_s: null,
    reason: "Waiting for an image.",
    critical: false,
  },
};

const stateCopy = {
  CLOSED: {
    title: "SYSTEM STABLE",
    detail: "Autonomous operation permitted",
    tone: "stable",
  },
  HALF_OPEN: {
    title: "TAKEOVER READY",
    detail: "Monitoring degraded prediction stability",
    tone: "warning",
  },
  OPEN: {
    title: "AUTOMATED CONTROL BLOCKED",
    detail: "Safety fallback active",
    tone: "critical",
  },
};

function clampPercent(value) {
  return Math.min(100, Math.max(0, Number(value) * 100));
}

function StatusDot({ tone = "neutral" }) {
  return <span className={`status-dot ${tone}`} aria-hidden="true" />;
}

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [driverPresent, setDriverPresent] = useState(true);
  const [result, setResult] = useState(initialResult);
  const [busy, setBusy] = useState(false);
  const [connection, setConnection] = useState("connecting");

  useEffect(() => {
    const ws = new WebSocket(WS);

    ws.onopen = () => setConnection("connected");
    ws.onclose = () => setConnection("disconnected");
    ws.onerror = () => setConnection("error");

    ws.onmessage = (event) => {
      try {
        setResult(JSON.parse(event.data));
      } catch {
        // Ignore non-JSON WebSocket messages.
      }
    };

    return () => ws.close();
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
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
        {
          method: "POST",
          body: form,
        }
      );

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Evaluation failed.");
      }

      setResult(data);
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
    try {
      const response = await fetch(`${API}/api/takeover/ack`, {
        method: "POST",
      });
      if (!response.ok) throw new Error("Acknowledgement failed.");

      setResult((current) => ({
        ...current,
        fallback: {
          ...current.fallback,
          acknowledged: true,
          action: "TAKEOVER_REQUESTED",
        },
      }));
    } catch (error) {
      setResult((current) => ({
        ...current,
        fallback: {
          ...current.fallback,
          critical: true,
          reason: error.message,
        },
      }));
    }
  }

  function handleFileChange(event) {
    setFile(event.target.files?.[0] || null);
  }

  function handleDrop(event) {
    event.preventDefault();
    const dropped = event.dataTransfer.files?.[0];
    if (dropped?.type.startsWith("image/")) setFile(dropped);
  }

  const fallback = result.fallback || initialResult.fallback;
  const cb = result.circuit_breaker;
  const prediction = result.prediction;
  const state = cb?.state || fallback.state || "CLOSED";
  const stateInfo = stateCopy[state] || stateCopy.CLOSED;
  const action = fallback.action || "NORMAL";

  const confidence = prediction ? clampPercent(prediction.confidence) : 0;
  const cbi = cb ? Number(cb.cbi) : null;
  const cbiPercent = cbi == null ? 0 : Math.min(100, Math.max(0, cbi * 100));

  const modeLabel = fallback.driver_present
    ? "DRIVER ASSIST / READY"
    : "AUTONOMOUS / DRIVERLESS";

  const fallbackClass = action.toLowerCase();

  const connectionLabel = useMemo(() => {
    if (connection === "connected") return "SYSTEM ONLINE";
    if (connection === "connecting") return "CONNECTING";
    if (connection === "error") return "API ERROR";
    return "SYSTEM OFFLINE";
  }, [connection]);

  return (
    <main className="app-shell">
      <div className="dashboard">
        <header className="topbar">
          <div className="brand">
            <div className="brand-mark" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <div>
              <p className="eyebrow">VEHICLE SAFETY SYSTEM</p>
              <h1>Metamorphic Circuit Breaker</h1>
              <p className="subtitle">
                Autonomous perception &amp; safety monitoring console
              </p>
            </div>
          </div>

          <div className="topbar-meta">
            <div className={`online-pill ${connection}`}>
              <StatusDot tone={connection === "connected" ? "online" : "neutral"} />
              <span>{connectionLabel}</span>
            </div>
            <span className="demo-badge">RESEARCH PROTOTYPE</span>
          </div>
        </header>

        <section className={`alert-banner ${stateInfo.tone} ${fallbackClass}`}>
          <div className="alert-icon" aria-hidden="true">
            {state === "OPEN" ? "!" : state === "HALF_OPEN" ? "!" : "✓"}
          </div>
          <div className="alert-copy">
            <span className="micro-label">SAFETY STATE</span>
            <strong>{stateInfo.title}</strong>
            <span>{stateInfo.detail}</span>
          </div>
          <div className="alert-state">
            <span>{state}</span>
            <small>{cb ? `CBI ${cbi.toFixed(4)}` : "CBI —"}</small>
          </div>
        </section>

        <section className="overview-grid">
          <article className="panel vehicle-panel">
            <div className="panel-heading">
              <div>
                <span className="micro-label">VEHICLE STATUS</span>
                <h2>Perception Monitor</h2>
              </div>
              <StatusDot tone={stateInfo.tone} />
            </div>

            <div className="vehicle-visual" aria-hidden="true">
              <div className="vehicle-ring ring-one" />
              <div className="vehicle-ring ring-two" />
              <div className="vehicle-outline">
                <span className="wheel wheel-left" />
                <span className="wheel wheel-right" />
                <span className="vehicle-window" />
              </div>
            </div>

            <div className="vehicle-details">
              <div>
                <span className="micro-label">OPERATING MODE</span>
                <strong>{modeLabel}</strong>
              </div>
              <div>
                <span className="micro-label">DRIVER</span>
                <strong className={fallback.driver_present ? "ok-text" : "warn-text"}>
                  {fallback.driver_present ? "PRESENT" : "ABSENT"}
                </strong>
              </div>
            </div>
          </article>

          <article className="panel safety-panel">
            <div className="panel-heading">
              <div>
                <span className="micro-label">CIRCUIT BREAKER</span>
                <h2>Safety State</h2>
              </div>
              <span className={`state-chip ${stateInfo.tone}`}>{state}</span>
            </div>

            <div className="safety-state">
              <strong>{stateInfo.title}</strong>
              <span>{cb?.action || "NORMAL OPERATION"}</span>
            </div>

            <div className="metric">
              <div className="metric-label">
                <span>CIRCUIT BREAKER INDEX</span>
                <strong>{cbi == null ? "—" : cbi.toFixed(4)}</strong>
              </div>
              <div className="meter cbi-meter">
                <span style={{ width: `${cbiPercent}%` }} />
              </div>
              <div className="meter-scale">
                <span>LOW</span><span>HIGH</span>
              </div>
            </div>
          </article>
        </section>

        <section className="section-heading">
          <div>
            <span className="micro-label">PERCEPTION</span>
            <h2>Camera &amp; Model Analysis</h2>
          </div>
          <span className="section-note">GTSRB / MobileNetV3-Small</span>
        </section>

        <section className="perception-grid">
          <article className="panel camera-panel">
            <div className="panel-heading">
              <div>
                <span className="micro-label">CAM-01</span>
                <h2>Camera Input</h2>
              </div>
              <span className="frame-status">
                <StatusDot tone={previewUrl ? "online" : "neutral"} />
                {previewUrl ? "FRAME READY" : "WAITING"}
              </span>
            </div>

            <label
              className={`dropzone ${previewUrl ? "has-image" : ""}`}
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
            >
              {previewUrl ? (
                <>
                  <img src={previewUrl} alt="Selected traffic sign" />
                  <div className="camera-overlay top-left">CAM-01</div>
                  <div className="camera-overlay top-right">FRAME ANALYSIS</div>
                  <div className="camera-overlay bottom-left">
                    {file?.name}
                  </div>
                  <div className="scan-line" />
                </>
              ) : (
                <div className="drop-content">
                  <div className="camera-icon" aria-hidden="true">
                    ◫
                  </div>
                  <strong>DROP CAMERA IMAGE</strong>
                  <span>or select a traffic-sign image</span>
                  <em>JPG / PNG · manual evaluation</em>
                </div>
              )}
              <input type="file" accept="image/*" onChange={handleFileChange} />
            </label>
          </article>

          <article className="panel model-panel">
            <div className="panel-heading">
              <div>
                <span className="micro-label">MODEL OUTPUT</span>
                <h2>Perception Result</h2>
              </div>
              <span className="model-badge">43 CLASSES</span>
            </div>

            <div className="prediction">
              <span className="micro-label">PREDICTED LABEL</span>
              <strong>{prediction?.label ?? "Awaiting frame"}</strong>
              <span className="prediction-class">
                {prediction ? `Class ${prediction.index}` : "No evaluation yet"}
              </span>
            </div>

            <div className="metric confidence-metric">
              <div className="metric-label">
                <span>CONFIDENCE</span>
                <strong>{prediction ? `${confidence.toFixed(1)}%` : "—"}</strong>
              </div>
              <div className="meter confidence-meter">
                <span style={{ width: `${confidence}%` }} />
              </div>
            </div>

            <div className="analysis-row">
              <span>Peak family</span>
              <strong>{cb?.peak_family || "—"}</strong>
            </div>
            <div className="analysis-row">
              <span>Cross-family spread</span>
              <strong>
                {cb?.cross_family_spread == null
                  ? "—"
                  : Number(cb.cross_family_spread).toFixed(4)}
              </strong>
            </div>
          </article>
        </section>

        <section className="panel metamorphic-panel">
          <div className="panel-heading">
            <div>
              <span className="micro-label">METAMORPHIC SAFETY ANALYSIS</span>
              <h2>Consistency Monitor</h2>
            </div>
            <span className="analysis-badge">
              <StatusDot tone="online" /> ANALYSIS COMPLETED
            </span>
          </div>

          <div className="metamorphic-grid">
            <div className="mr-item"><span>MR1</span><strong>Rotation</strong><em>Framework result available</em></div>
            <div className="mr-item"><span>MR2</span><strong>Horizontal Flip</strong><em>Framework result available</em></div>
            <div className="mr-item"><span>MR3</span><strong>Zoom</strong><em>Framework result available</em></div>
            <div className="mr-item"><span>MR4</span><strong>Brightness / Contrast</strong><em>Framework result available</em></div>
            <div className="mr-item"><span>MR5</span><strong>Gaussian Noise</strong><em>Framework result available</em></div>
            <div className="mr-item"><span>MR6</span><strong>Gaussian Blur</strong><em>Framework result available</em></div>
            <div className="mr-item"><span>MR7</span><strong>Sharpening</strong><em>Framework result available</em></div>
            <div className="mr-item"><span>MR8</span><strong>Saturation</strong><em>Framework result available</em></div>
          </div>
          <p className="data-note">
            Individual MR scores are not exposed by the current API, so this panel does not invent per-relation results.
          </p>
        </section>

        <section className={`fallback-panel ${fallbackClass}`}>
          <div className="fallback-header">
            <div>
              <span className="micro-label">FALLBACK CONTROL</span>
              <h2>
                {action === "SAFE_PULL_OVER"
                  ? "Safety Fallback Active"
                  : action === "TAKEOVER_REQUESTED"
                    ? "Takeover Requested"
                    : "Autonomous Operation"}
              </h2>
            </div>
            <span className="action-chip">{action.replaceAll("_", " ")}</span>
          </div>

          <div className="fallback-body">
            <div className="fallback-copy">
              <div className="fallback-status-line">
                <StatusDot tone={fallback.critical ? "critical" : action === "TAKEOVER_REQUESTED" ? "warning" : "online"} />
                <strong>{fallback.reason}</strong>
              </div>

              {fallback.countdown_s != null && !fallback.acknowledged && (
                <div className="takeover-countdown">
                  <span>TAKEOVER WINDOW</span>
                  <strong>{Number(fallback.countdown_s).toFixed(1)}<small>s</small></strong>
                </div>
              )}

              {fallback.acknowledged && (
                <div className="acknowledged">
                  <StatusDot tone="online" />
                  Driver takeover acknowledged
                </div>
              )}
            </div>

            <div className="fallback-actions">
              <div className="driver-toggle">
                <span className={fallback.driver_present ? "selected" : ""}>
                  <StatusDot tone={fallback.driver_present ? "online" : "neutral"} />
                  DRIVER PRESENT
                </span>
                <span className={!fallback.driver_present ? "selected" : ""}>
                  <StatusDot tone={!fallback.driver_present ? "warning" : "neutral"} />
                  DRIVER ABSENT
                </span>
              </div>

              {action === "TAKEOVER_REQUESTED" && !fallback.acknowledged && (
                <button className="takeover-button" onClick={acknowledge}>
                  ACKNOWLEDGE TAKEOVER
                </button>
              )}

              {action === "SAFE_PULL_OVER" && (
                <div className="simulation-notice">
                  <span>SIMULATION EVENT</span>
                  Safe pull-over is represented as a safety decision only.
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="evaluation-bar">
          <div>
            <span className="micro-label">MANUAL EVALUATION</span>
            <strong>{file ? file.name : "Select a camera image to begin"}</strong>
          </div>
          <button onClick={evaluate} disabled={!file || busy}>
            <span>{busy ? "ANALYZING FRAME…" : "RUN SAFETY EVALUATION"}</span>
            <b>→</b>
          </button>
        </section>

        <footer className="footer">
          <span>METAMORPHIC CIRCUIT BREAKER · VEHICLE SAFETY DEMONSTRATION</span>
          <span>SAFE_PULL_OVER is simulated · No physical vehicle actuator is controlled</span>
        </footer>
      </div>
    </main>
  );
}

export default App;

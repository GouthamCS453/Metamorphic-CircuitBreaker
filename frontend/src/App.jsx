import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8000";

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

function App() {
  const [file, setFile] = useState(null);
  const [driverPresent, setDriverPresent] = useState(true);
  const [result, setResult] = useState(initialResult);
  const [busy, setBusy] = useState(false);
  const [connection, setConnection] = useState("connecting");

  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws/fallback");

    ws.onopen = () => setConnection("connected");
    ws.onclose = () => setConnection("disconnected");
    ws.onerror = () => setConnection("error");

    ws.onmessage = (event) => {
      try {
        setResult(JSON.parse(event.data));
      } catch {
        // Ignore non-JSON messages such as the WebSocket health response.
      }
    };

    return () => ws.close();
  }, []);

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
    await fetch(`${API}/api/takeover/ack`, { method: "POST" });
    setResult((current) => ({
      ...current,
      fallback: {
        ...current.fallback,
        acknowledged: true,
        action: "TAKEOVER_REQUESTED",
      },
    }));
  }

  const fallback = result.fallback || initialResult.fallback;
  const cb = result.circuit_breaker;
  const prediction = result.prediction;

  return (
    <main className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">SAFETY MONITOR</p>
          <h1>Metamorphic Circuit Breaker</h1>
          <p className="subtitle">
            Vehicle fallback demonstration using the existing circuit breaker.
          </p>
        </div>
        <span className={`connection ${connection}`}>
          API: {connection}
        </span>
      </header>

      <section className={`status-card ${fallback.action.toLowerCase()}`}>
        <span className="status-label">FALLBACK ACTION</span>
        <strong>{fallback.action}</strong>
        <p>{fallback.reason}</p>
        {fallback.countdown_s != null && (
          <div className="countdown">
            Takeover window: {fallback.countdown_s.toFixed(1)} s
          </div>
        )}
      </section>

      <section className="controls card">
        <h2>Evaluation</h2>
        <label className="upload">
          <span>Select traffic-sign image</span>
          <input
            type="file"
            accept="image/*"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />
        </label>

        <label className="checkbox">
          <input
            type="checkbox"
            checked={driverPresent}
            onChange={(event) => setDriverPresent(event.target.checked)}
          />
          Driver present
        </label>

        <div className="actions">
          <button onClick={evaluate} disabled={!file || busy}>
            {busy ? "Evaluating..." : "Evaluate Image"}
          </button>

          {fallback.action === "TAKEOVER_REQUESTED" &&
            !fallback.acknowledged && (
              <button className="secondary" onClick={acknowledge}>
                Acknowledge Takeover
              </button>
            )}
        </div>
      </section>

      <section className="grid">
        <article className="card">
          <h2>Perception</h2>
          <dl>
            <div>
              <dt>Label</dt>
              <dd>{prediction?.label ?? "—"}</dd>
            </div>
            <div>
              <dt>Class</dt>
              <dd>{prediction?.index ?? "—"}</dd>
            </div>
            <div>
              <dt>Confidence</dt>
              <dd>
                {prediction ? `${(prediction.confidence * 100).toFixed(2)}%` : "—"}
              </dd>
            </div>
          </dl>
        </article>

        <article className="card">
          <h2>Circuit Breaker</h2>
          <dl>
            <div>
              <dt>State</dt>
              <dd>{cb?.state ?? fallback.state}</dd>
            </div>
            <div>
              <dt>Action</dt>
              <dd>{cb?.action ?? "—"}</dd>
            </div>
            <div>
              <dt>CBI</dt>
              <dd>{cb ? cb.cbi.toFixed(4) : "—"}</dd>
            </div>
          </dl>
        </article>

        <article className="card">
          <h2>Fallback Status</h2>
          <dl>
            <div>
              <dt>Driver</dt>
              <dd>{fallback.driver_present ? "Present" : "Absent"}</dd>
            </div>
            <div>
              <dt>Acknowledged</dt>
              <dd>{fallback.acknowledged ? "Yes" : "No"}</dd>
            </div>
            <div>
              <dt>Critical</dt>
              <dd>{fallback.critical ? "Yes" : "No"}</dd>
            </div>
          </dl>
        </article>
      </section>

      <footer>
        SAFE_PULL_OVER is a simulated safety event. No physical vehicle actuator
        is controlled by this demonstration.
      </footer>
    </main>
  );
}

export default App;

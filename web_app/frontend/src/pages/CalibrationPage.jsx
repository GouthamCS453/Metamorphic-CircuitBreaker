// src/pages/CalibrationPage.jsx
import { useState, useEffect } from "react";
import {
  Settings, Loader2, CheckCircle, AlertTriangle, XCircle,
  Zap, RotateCcw, SlidersHorizontal
} from "lucide-react";
import {
  fetchConfig, updateConfig, fetchProfiles, applyProfile, simulate
} from "../api/client";
import CbiGauge from "../components/CbiGauge";

function Slider({ label, sub, value, min, max, step, onChange, warn }) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1.5">
        <div>
          <span className="text-sm font-semibold text-slate-800 dark:text-slate-200">{label}</span>
          {sub && <span className="text-xs text-slate-500 ml-2">{sub}</span>}
        </div>
        <span className="text-sm font-bold text-baltic_blue dark:text-baltic_blue-700 font-mono">
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className={`w-full h-2 rounded-full appearance-none cursor-pointer bg-slate-200 dark:bg-[#01151d] ${
          warn ? "accent-red-500" : "accent-baltic_blue"
        }`}
      />
      <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

export default function CalibrationPage() {
  const [config, setConfig] = useState({
    alpha: 0.5,
    beta: 0.35,
    gamma: 0.15,
    tau_fam: 0.35,
    theta_warn: 0.25,
    theta_trip: 0.55
  });
  const [profiles, setProfiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState(null);
  const [simVals, setSimVals] = useState({
    hypothetical_ir_k: 0.40,
    hypothetical_cfs: 0.33,
    hypothetical_uncertainty: 0.20
  });
  const [simResult, setSimResult] = useState(null);
  const [simLoading, setSimLoading] = useState(false);

  useEffect(() => {
    Promise.all([fetchConfig(), fetchProfiles()])
      .then(([c, p]) => {
        setConfig(c.data);
        setProfiles(p.data);
      })
      .finally(() => setLoading(false));
  }, []);

  const total = parseFloat((config.alpha + config.beta + config.gamma).toFixed(4));
  const sumOk = Math.abs(total - 1.0) < 1e-3;
  const threshOk = config.theta_warn < config.theta_trip;

  const handleSave = async () => {
    if (!sumOk || !threshOk) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const r = await updateConfig(config);
      setConfig(r.data);
      setSaveMsg({ ok: true, text: "Configuration saved and applied to all future inferences." });
    } catch (e) {
      setSaveMsg({ ok: false, text: e.response?.data?.detail || "Save failed" });
    }
    setSaving(false);
  };

  const handleApplyProfile = async (id) => {
    try {
      const r = await applyProfile(id);
      setConfig(r.data);
      setSaveMsg({ ok: true, text: "Clinical profile applied successfully." });
    } catch {}
  };

  const handleSimulate = async () => {
    if (!sumOk) return;
    setSimLoading(true);
    try {
      const r = await simulate({ ...config, ...simVals });
      setSimResult(r.data);
    } catch {}
    setSimLoading(false);
  };

  const setAlpha = (v) => setConfig(c => ({ ...c, alpha: v }));
  const setBeta  = (v) => setConfig(c => ({ ...c, beta: v }));
  const setGamma = (v) => setConfig(c => ({ ...c, gamma: v }));

  if (loading) {
    return (
      <div className="flex justify-center py-20">
        <Loader2 className="w-8 h-8 spinner text-baltic_blue" />
      </div>
    );
  }

  const simStateColor = {
    CLOSED: "text-verdigris",
    HALF_OPEN: "text-amber-500",
    OPEN: "text-red-500"
  };
  const simStateIcon = { CLOSED: CheckCircle, HALF_OPEN: AlertTriangle, OPEN: XCircle };
  const SimIcon = simResult ? simStateIcon[simResult.state] : null;

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-6">
        <div className="section-label mb-1">Safety Calibration System</div>
        <h1 className="text-3xl font-extrabold text-baltic_blue dark:text-white flex items-center gap-2.5">
          <SlidersHorizontal className="w-8 h-8 text-teal" /> Parameter Calibration Studio
        </h1>
        <p className="text-slate-600 dark:text-slate-400 text-sm mt-1">
          Tune Composite Brittleness Index weights and state boundary thresholds in real time. Changes take effect on the running server immediately.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Parameter Sliders */}
        <div className="space-y-5">
          {/* Weights Card */}
          <div className="card p-6 space-y-4">
            <div className="text-xs font-bold uppercase tracking-wider text-baltic_blue dark:text-baltic_blue-700">
              CBI Component Weights (Sum = 1.0)
            </div>
            <div
              className={`text-xs px-3.5 py-2 rounded-lg font-mono flex items-center justify-between ${
                sumOk
                  ? "bg-verdigris-900/30 text-verdigris border border-verdigris/30"
                  : "bg-red-50 text-red-600 border border-red-200"
              }`}
            >
              <span>alpha + beta + gamma = {total.toFixed(4)}</span>
              <span>{sumOk ? "(Valid Sum)" : "(Must Equal 1.000)"}</span>
            </div>

            <div className="space-y-4 pt-1">
              <Slider
                label="alpha — Peak Family Instability"
                sub="max_k IR_k(x)"
                value={config.alpha}
                min={0}
                max={1}
                step={0.05}
                onChange={setAlpha}
              />
              <Slider
                label="beta — Cross-Family Spread"
                sub="CFS(x)"
                value={config.beta}
                min={0}
                max={1}
                step={0.05}
                onChange={setBeta}
              />
              <Slider
                label="gamma — Model Baseline Uncertainty"
                sub="1 - c0"
                value={config.gamma}
                min={0}
                max={1}
                step={0.05}
                onChange={setGamma}
              />
            </div>
          </div>

          {/* Decision Thresholds Card */}
          <div className="card p-6 space-y-4">
            <div className="text-xs font-bold uppercase tracking-wider text-baltic_blue dark:text-baltic_blue-700">
              Decision Boundary Thresholds
            </div>
            {!threshOk && (
              <div className="text-xs text-red-600 bg-red-50 dark:bg-red-950/40 px-3.5 py-2 rounded-lg border border-red-200">
                Warning: theta_warn must be strictly less than theta_trip
              </div>
            )}
            <div className="space-y-4">
              <Slider
                label="tau_fam — Family Activation Threshold"
                sub="fraction of flips to compromise family"
                value={config.tau_fam}
                min={0.05}
                max={0.80}
                step={0.05}
                onChange={v => setConfig(c => ({ ...c, tau_fam: v }))}
              />
              <Slider
                label="theta_warn — Warning / HALF-OPEN Threshold"
                value={config.theta_warn}
                min={0.05}
                max={0.90}
                step={0.05}
                onChange={v => setConfig(c => ({ ...c, theta_warn: v }))}
                warn={!threshOk}
              />
              <Slider
                label="theta_trip — Trip / OPEN Intercept Threshold"
                value={config.theta_trip}
                min={0.05}
                max={0.99}
                step={0.05}
                onChange={v => setConfig(c => ({ ...c, theta_trip: v }))}
                warn={!threshOk}
              />
            </div>
          </div>

          {/* Apply Button */}
          <button
            onClick={handleSave}
            disabled={saving || !sumOk || !threshOk}
            className="btn-primary w-full py-3.5 flex items-center justify-center gap-2 font-bold shadow-md"
          >
            {saving ? (
              <>
                <Loader2 className="w-4 h-4 spinner" /> Applying Parameters to Server...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 text-cream-DEFAULT" /> Save & Activate Live Configuration
              </>
            )}
          </button>

          {saveMsg && (
            <div
              className={`p-3.5 rounded-xl text-sm ${
                saveMsg.ok
                  ? "bg-verdigris-900/30 text-verdigris border border-verdigris/40"
                  : "bg-red-50 text-red-600 border border-red-200"
              }`}
            >
              {saveMsg.text}
            </div>
          )}
        </div>

        {/* Right: Live Preview, Simulator & Profiles */}
        <div className="space-y-5">
          {/* Live Mathematical Formula Preview */}
          <div className="card p-6">
            <div className="text-xs font-bold uppercase tracking-wider text-baltic_blue dark:text-baltic_blue-700 mb-3">
              Active Decision Formula
            </div>
            <div className="bg-slate-900 text-slate-200 rounded-xl p-4 font-mono text-xs leading-relaxed space-y-1.5">
              <div className="text-slate-400">CBI(x) = alpha * max_k(IR_k) + beta * CFS + gamma * (1 - c0)</div>
              <div>
                = <span className="text-mint_leaf font-bold">{config.alpha.toFixed(2)}</span> * max_k(IR_k) + <span className="text-mint_leaf font-bold">{config.beta.toFixed(2)}</span> * CFS + <span className="text-mint_leaf font-bold">{config.gamma.toFixed(2)}</span> * (1 - c0)
              </div>
              <div className="pt-2 text-slate-400">
                CLOSED &nbsp;&nbsp;&nbsp; if CBI &lt; {config.theta_warn.toFixed(2)}<br />
                HALF-OPEN if {config.theta_warn.toFixed(2)} &le; CBI &lt; {config.theta_trip.toFixed(2)}<br />
                OPEN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if CBI &ge; {config.theta_trip.toFixed(2)}
              </div>
            </div>
          </div>

          {/* Hypothetical Scenario Simulator */}
          <div className="card p-6">
            <div className="text-xs font-bold uppercase tracking-wider text-baltic_blue dark:text-baltic_blue-700 mb-3">
              Hypothetical What-If Simulator
            </div>
            <div className="space-y-3.5">
              <Slider
                label="Simulated Peak Family IR_k"
                value={simVals.hypothetical_ir_k}
                min={0}
                max={1}
                step={0.05}
                onChange={v => setSimVals(s => ({ ...s, hypothetical_ir_k: v }))}
              />
              <Slider
                label="Simulated Cross-Family Spread (CFS)"
                value={simVals.hypothetical_cfs}
                min={0}
                max={1}
                step={0.05}
                onChange={v => setSimVals(s => ({ ...s, hypothetical_cfs: v }))}
              />
              <Slider
                label="Simulated Baseline Uncertainty (1 - c0)"
                value={simVals.hypothetical_uncertainty}
                min={0}
                max={1}
                step={0.05}
                onChange={v => setSimVals(s => ({ ...s, hypothetical_uncertainty: v }))}
              />
            </div>

            <button
              onClick={handleSimulate}
              disabled={simLoading || !sumOk}
              className="btn-secondary w-full mt-4 font-semibold shadow-sm"
            >
              {simLoading ? "Calculating..." : "Compute Simulated State Transition"}
            </button>

            {simResult && SimIcon && (
              <div className="mt-4 pt-3 border-t border-slate-100 dark:border-[#033e56] space-y-3">
                <CbiGauge cbi={simResult.cbi} thetaWarn={config.theta_warn} thetaTrip={config.theta_trip} />
                <div className={`flex items-center gap-2 font-bold text-sm ${simStateColor[simResult.state]}`}>
                  <SimIcon className="w-5 h-5" />
                  Simulated CBI = {simResult.cbi.toFixed(4)} &rarr; {simResult.state.replace("_", "-")}
                </div>
                <div className="text-[11px] text-slate-500 font-mono flex gap-4">
                  <span>alpha: {simResult.components.alpha_part.toFixed(4)}</span>
                  <span>beta: {simResult.components.beta_part.toFixed(4)}</span>
                  <span>gamma: {simResult.components.gamma_part.toFixed(4)}</span>
                </div>
              </div>
            )}
          </div>

          {/* Clinical Risk Profiles */}
          <div className="card p-6">
            <div className="text-xs font-bold uppercase tracking-wider text-baltic_blue dark:text-baltic_blue-700 mb-3">
              One-Click Clinical Risk Profiles
            </div>
            <div className="space-y-2.5">
              {profiles.map((p) => (
                <div
                  key={p.id}
                  className="flex items-start justify-between gap-3 p-3.5 rounded-xl border border-slate-200 dark:border-[#033e56] bg-slate-50 dark:bg-[#01151d]/40"
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-bold text-slate-900 dark:text-white">{p.label}</div>
                    <div className="text-xs text-slate-600 dark:text-slate-400 mt-0.5 leading-relaxed">{p.description}</div>
                    <div className="text-[11px] font-mono text-baltic_blue dark:text-teal font-semibold mt-1">
                      alpha={p.config.alpha} beta={p.config.beta} gamma={p.config.gamma} theta_trip={p.config.theta_trip}
                    </div>
                  </div>
                  <button
                    onClick={() => { handleApplyProfile(p.id); setConfig(p.config); }}
                    className="btn-secondary text-xs px-3 py-1.5 flex-shrink-0 flex items-center gap-1 shadow-xs"
                  >
                    <RotateCcw className="w-3 h-3 text-baltic_blue" /> Apply
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

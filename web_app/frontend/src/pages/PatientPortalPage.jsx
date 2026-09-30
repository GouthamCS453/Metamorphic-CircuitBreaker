// src/pages/PatientPortalPage.jsx
// Patient-only view: Upload / preset → run CB → see clean results with original image
import { useState, useRef, useCallback, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  Upload, Loader2, CheckCircle, AlertTriangle, XCircle,
  ChevronDown, ChevronUp, FileText, ArrowRight, FlaskConical,
  Check, ImageIcon, Activity
} from "lucide-react";
import { predictFromFile, predictFromPreset, fetchPresets, BASE_URL } from "../api/client";
import StateBadge from "../components/StateBadge";
import CbiGauge from "../components/CbiGauge";
import { useRole, ROLES } from "../context/RoleContext";

const CLASS_FULL = {
  MEL: "Melanoma",
  NV: "Melanocytic Nevus",
  BCC: "Basal Cell Carcinoma",
  AK: "Actinic Keratosis",
  BKL: "Benign Keratosis-like Lesion",
  DF: "Dermatofibroma",
  VASC: "Vascular Lesion",
  SCC: "Squamous Cell Carcinoma",
};

const SITES = [
  "Back / Torso", "Forearm", "Lower Leg", "Face / Malar",
  "Scalp", "Neck", "Chest", "Hand / Foot",
];

export default function PatientPortalPage() {
  const { role } = useRole();
  const isDoctor = role === ROLES.DOCTOR || role === ROLES.ADMIN;

  const [presets, setPresets] = useState([]);
  const [mode, setMode] = useState("preset");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selectedPreset, setSelectedPreset] = useState(null);
  const [patientCtx, setPatientCtx] = useState({
    patient_name: "", patient_age: "", patient_sex: "", lesion_site: "", symptoms: ""
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showTechDetails, setShowTechDetails] = useState(false);
  const [showTests, setShowTests] = useState(false);
  const fileRef = useRef();

  useEffect(() => {
    fetchPresets()
      .then(r => {
        setPresets(r.data);
        if (r.data?.length > 0) setSelectedPreset(r.data[0]);
      })
      .catch(() => {});
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0] || e.target?.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null); setError(null);
  }, []);

  const handleSubmit = async () => {
    setLoading(true); setResult(null); setError(null);
    try {
      let res;
      if (mode === "upload" && file) {
        const fd = new FormData();
        fd.append("file", file);
        Object.entries(patientCtx).forEach(([k, v]) => v && fd.append(k, v));
        res = await predictFromFile(fd);
      } else if (mode === "preset" && selectedPreset) {
        res = await predictFromPreset(selectedPreset.preset_id, patientCtx);
      }
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || "Server error during analysis.");
    } finally {
      setLoading(false);
    }
  };

  const report  = result?.report;
  const gradcam = result?.gradcam;
  const cfg     = report?.config_used;

  // ── State-specific visual config ─────────────────────────────────────────────
  const STATE_CFG = {
    CLOSED: {
      Icon: CheckCircle,
      color: "text-verdigris dark:text-mint_leaf",
      bg: "bg-verdigris-900/20 dark:bg-verdigris-100/30 border-verdigris/40",
      ring: "ring-verdigris/30",
      // Patient-friendly explanation:
      patient_title: "Your Screening is Complete",
      patient_msg:   "The AI analysis is stable and consistent. Our system has automatically verified this result — you will receive it below.",
      // What doctor sees differs; handled in isDoctor check
    },
    HALF_OPEN: {
      Icon: AlertTriangle,
      color: "text-amber-600 dark:text-amber-400",
      bg: "bg-amber-50 dark:bg-amber-950/30 border-amber-300 dark:border-amber-700/50",
      ring: "ring-amber-400/30",
      patient_title: "Your Case is Under Specialist Review",
      patient_msg:   "The AI detected borderline sensitivity during safety testing. Your case has been forwarded to Dr. Sarah Jenkins, MD for a secondary verification. You will be contacted within 2 hours.",
    },
    OPEN: {
      Icon: XCircle,
      color: "text-red-600 dark:text-red-400",
      bg: "bg-red-50 dark:bg-red-950/30 border-red-300 dark:border-red-700/50",
      ring: "ring-red-500/30",
      patient_title: "Specialist Review Required",
      patient_msg:   "Our AI safety system detected model instability during testing. For your safety, the automated AI prediction has been withheld. A Consultant Dermatologist will review your case directly.",
    },
  };

  const stCfg = report ? STATE_CFG[report.state] : null;

  return (
    <div className="max-w-5xl mx-auto px-4 py-10 space-y-8">

      {/* ── Page header ─────────────────────────────────────────────────────── */}
      <div>
        <div className="section-label mb-1">
          {isDoctor ? "Physician Screening Workspace" : "Patient AI Screening Service"}
        </div>
        <h1 className="text-3xl font-extrabold text-baltic_blue dark:text-white">
          {isDoctor ? "Run Circuit Breaker — Physician Mode" : "Skin Lesion AI Screening"}
        </h1>
        <p className="text-slate-600 dark:text-slate-400 mt-1 text-sm max-w-2xl">
          {isDoctor
            ? "Select a preset or upload to run the 22-test metamorphic suite. As Physician you will see full technical output including GradCAM attention maps and test decomposition."
            : "Upload a dermatoscopic image or select a demo preset. The system runs 22 safety tests before returning a result."
          }
        </p>
      </div>

      {/* ── Input form (only shown before result) ────────────────────────────── */}
      {!result && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* Source / Mode selector */}
          <div className="lg:col-span-2 card p-6 space-y-6">
            {/* Mode tabs */}
            <div className="flex rounded-xl border border-slate-200 dark:border-[#033e56] overflow-hidden text-sm font-semibold bg-slate-100 dark:bg-[#01151d] p-1.5">
              {[["preset", "Demo Presets"], ["upload", "Upload Image"]].map(([m, l]) => (
                <button
                  key={m}
                  onClick={() => { setMode(m); setResult(null); setError(null); }}
                  className={`flex-1 py-2 rounded-lg transition-all ${
                    mode === m
                      ? "bg-baltic_blue text-white shadow"
                      : "text-slate-600 dark:text-slate-400 hover:text-baltic_blue"
                  }`}
                >
                  {l}
                </button>
              ))}
            </div>

            {mode === "upload" ? (
              <div
                onDragOver={e => e.preventDefault()}
                onDrop={onDrop}
                onClick={() => fileRef.current.click()}
                className="border-2 border-dashed border-slate-300 dark:border-[#033e56] hover:border-baltic_blue rounded-2xl p-12 text-center cursor-pointer transition-colors group bg-slate-50 dark:bg-[#01151d]/50"
              >
                <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={onDrop} />
                {preview ? (
                  <div className="space-y-3">
                    <img src={preview} alt="preview" className="mx-auto max-h-56 rounded-xl object-contain shadow border border-slate-200" />
                    <p className="text-xs text-slate-500">Click or drag to replace</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Upload className="w-12 h-12 text-baltic_blue/40 group-hover:text-baltic_blue mx-auto transition-colors" />
                    <p className="text-slate-700 dark:text-slate-300 font-semibold">
                      Drag &amp; drop or <span className="text-baltic_blue font-bold">browse</span>
                    </p>
                    <p className="text-slate-400 text-xs">JPG / PNG · Dermatoscopic or clinical lesion images</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Select a verified ISIC test preset:
                </p>

                {presets.length === 0 ? (
                  <div className="py-8 text-center text-slate-400 text-sm">Loading presets...</div>
                ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {presets.map(p => {
                      const sel = selectedPreset?.preset_id === p.preset_id;
                      const stColors = {
                        CLOSED:    "border-verdigris/50 bg-verdigris-900/20 text-verdigris",
                        HALF_OPEN: "border-amber-400/50 bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-400",
                        OPEN:      "border-red-400/50 bg-red-50 dark:bg-red-950/30 text-red-700 dark:text-red-400",
                      }[p.expected_state] || "border-slate-300";
                      return (
                        <button
                          key={p.preset_id}
                          onClick={() => { setSelectedPreset(p); setError(null); }}
                          className={`text-left rounded-xl p-4 border-2 transition-all ${
                            sel
                              ? "border-baltic_blue ring-2 ring-baltic_blue/20 bg-baltic_blue-900/20 dark:bg-baltic_blue-200/30 shadow"
                              : "border-slate-200 dark:border-[#033e56] hover:border-baltic_blue/40"
                          }`}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className={`text-[10px] font-extrabold uppercase px-2 py-0.5 rounded-full border ${stColors}`}>
                              {p.expected_state.replace("_", "-")}
                            </span>
                            {sel && <Check className="w-4 h-4 text-baltic_blue dark:text-mint_leaf" />}
                          </div>
                          <div className="font-bold text-sm text-slate-900 dark:text-white leading-snug">{p.name}</div>
                          <div className="flex items-center gap-3 mt-2 text-[11px] text-slate-500 font-mono">
                            <span>CBI ref: {p.cbi_ref}</span>
                            <span>Flips: {p.flips_ref}</span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}

                {/* Preset image thumbnail */}
                {selectedPreset && (
                  <div className="mt-4 flex items-center gap-4 p-4 rounded-xl bg-slate-50 dark:bg-[#022a39]/40 border border-slate-200 dark:border-[#033e56]">
                    <img
                      src={`${BASE_URL}/api/presets/${selectedPreset.preset_id}/image`}
                      alt={selectedPreset.name}
                      className="w-24 h-24 rounded-xl object-cover border border-slate-200 dark:border-slate-700 shadow flex-shrink-0"
                    />
                    <div className="text-xs space-y-1.5">
                      <div className="font-bold text-slate-800 dark:text-slate-100 text-sm">{selectedPreset.name}</div>
                      <div className="text-slate-500 leading-relaxed">{selectedPreset.description?.slice(0, 140)}</div>
                      <div className="text-baltic_blue dark:text-mint_leaf font-semibold">Ready to run 22 metamorphic tests</div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Patient context + Run button */}
          <div className="space-y-5">
            <div className="card p-5 space-y-4">
              <p className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Patient Info (Optional)
              </p>
              <div>
                <label className="text-xs font-semibold text-slate-600 dark:text-slate-400 block mb-1">Full Name</label>
                <input
                  value={patientCtx.patient_name}
                  onChange={e => setPatientCtx(p => ({ ...p, patient_name: e.target.value }))}
                  placeholder="e.g. John Doe"
                  className="w-full bg-slate-50 dark:bg-[#01151d] border border-slate-200 dark:border-[#033e56] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-baltic_blue"
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs font-semibold text-slate-600 dark:text-slate-400 block mb-1">Age</label>
                  <input
                    value={patientCtx.patient_age}
                    onChange={e => setPatientCtx(p => ({ ...p, patient_age: e.target.value }))}
                    placeholder="e.g. 48"
                    className="w-full bg-slate-50 dark:bg-[#01151d] border border-slate-200 dark:border-[#033e56] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-baltic_blue"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-600 dark:text-slate-400 block mb-1">Sex</label>
                  <select
                    value={patientCtx.patient_sex}
                    onChange={e => setPatientCtx(p => ({ ...p, patient_sex: e.target.value }))}
                    className="w-full bg-slate-50 dark:bg-[#01151d] border border-slate-200 dark:border-[#033e56] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-baltic_blue"
                  >
                    <option value="">Any</option>
                    <option>Male</option><option>Female</option><option>Other</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-600 dark:text-slate-400 block mb-1">Lesion Site</label>
                <select
                  value={patientCtx.lesion_site}
                  onChange={e => setPatientCtx(p => ({ ...p, lesion_site: e.target.value }))}
                  className="w-full bg-slate-50 dark:bg-[#01151d] border border-slate-200 dark:border-[#033e56] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-baltic_blue"
                >
                  <option value="">Unspecified</option>
                  {SITES.map(s => <option key={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-600 dark:text-slate-400 block mb-1">Clinical Notes</label>
                <textarea
                  value={patientCtx.symptoms}
                  onChange={e => setPatientCtx(p => ({ ...p, symptoms: e.target.value }))}
                  rows={2}
                  placeholder="e.g. Itching for 3 months, border changed..."
                  className="w-full bg-slate-50 dark:bg-[#01151d] border border-slate-200 dark:border-[#033e56] rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:border-baltic_blue"
                />
              </div>
            </div>

            <button
              onClick={handleSubmit}
              disabled={loading || (mode === "upload" ? !file : !selectedPreset)}
              className="btn-primary w-full py-4 text-base font-bold shadow-lg flex items-center justify-center gap-2"
            >
              {loading ? (
                <><Loader2 className="w-5 h-5 spinner" /> Running 22 Tests...</>
              ) : (
                <><FlaskConical className="w-5 h-5" /> Run AI Screening</>
              )}
            </button>

            {error && (
              <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 text-red-700 dark:text-red-300 text-sm">
                {error}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── RESULTS ──────────────────────────────────────────────────────────── */}
      {result && report && stCfg && (
        <div className="space-y-6">

          {/* ── 1. Primary state banner ─────────────────────────────────────── */}
          <div className={`rounded-2xl border-2 p-8 ${stCfg.bg} ring-4 ${stCfg.ring} transition-all`}>
            <div className="flex items-start gap-5">
              <stCfg.Icon className={`w-12 h-12 ${stCfg.color} flex-shrink-0 mt-0.5`} strokeWidth={2} />
              <div className="flex-1">
                <div className={`text-2xl font-black ${stCfg.color} mb-1`}>
                  {report.state.replace("_", "-")} — {stCfg.patient_title}
                </div>
                <p className="text-slate-700 dark:text-slate-300 leading-relaxed max-w-2xl">
                  {stCfg.patient_msg}
                </p>

                {/* ── THE DIAGNOSIS BLOCK — Shown prominently for CLOSED ── */}
                {report.state === "CLOSED" && (
                  <div className="mt-6 p-6 rounded-2xl bg-white dark:bg-[#022a39] border border-verdigris/30 shadow-sm">
                    <div className="text-xs font-bold uppercase tracking-widest text-verdigris mb-3">
                      AI Diagnosis — Auto-Verified
                    </div>
                    <div className="flex items-end gap-6 flex-wrap">
                      <div>
                        <div className="text-5xl font-black text-baltic_blue dark:text-white tracking-tight">
                          {report.baseline_label}
                        </div>
                        <div className="text-xl font-semibold text-slate-600 dark:text-slate-300 mt-1">
                          {CLASS_FULL[report.baseline_label] || report.baseline_label}
                        </div>
                      </div>
                      <div className="pb-1">
                        <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-0.5">Confidence</div>
                        <div className="text-4xl font-black text-teal dark:text-mint_leaf">
                          {(report.baseline_confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="pb-1">
                        <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-0.5">Safety Tests</div>
                        <div className="text-3xl font-black text-slate-700 dark:text-slate-200">
                          {report.n_tests - report.n_flips}<span className="text-sm font-semibold text-slate-500">/{report.n_tests} passed</span>
                        </div>
                      </div>
                    </div>
                    <div className="mt-3 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                      This diagnosis has been verified stable across all 22 metamorphic perturbation tests
                      (CBI = {report.cbi.toFixed(4)}, below warning threshold of {cfg?.theta_warn}).
                    </div>
                  </div>
                )}

                {/* ── HALF-OPEN: show preliminary result with amber caveat ── */}
                {report.state === "HALF_OPEN" && (
                  <div className="mt-6 p-6 rounded-2xl bg-white dark:bg-[#022a39] border border-amber-300 dark:border-amber-700/50 shadow-sm">
                    <div className="text-xs font-bold uppercase tracking-widest text-amber-700 dark:text-amber-400 mb-2">
                      Preliminary AI Assessment (Pending Verification)
                    </div>
                    <div className="flex items-end gap-6 flex-wrap mb-3">
                      <div>
                        <div className="text-4xl font-black text-slate-700 dark:text-slate-200">{report.baseline_label}</div>
                        <div className="text-base font-semibold text-slate-500 mt-0.5">{CLASS_FULL[report.baseline_label]}</div>
                      </div>
                      <div className="pb-1">
                        <div className="text-xs text-slate-500 mb-0.5">Confidence</div>
                        <div className="text-3xl font-bold text-slate-600 dark:text-slate-300">
                          {(report.baseline_confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    <p className="text-amber-700 dark:text-amber-300 text-sm font-medium border-l-4 border-amber-400 pl-3">
                      This result is indicative only. A dermatologist will validate it within 2 hours.
                    </p>
                  </div>
                )}

                {/* ── OPEN: No prediction shown to patient ── */}
                {report.state === "OPEN" && (
                  <div className="mt-6 p-5 rounded-xl bg-red-100 dark:bg-red-950/60 border border-red-300 dark:border-red-800">
                    <div className="font-bold text-red-800 dark:text-red-200 mb-1">Automated Diagnosis Withheld</div>
                    <p className="text-sm text-red-700 dark:text-red-300">
                      The AI safety system intercepted this prediction due to multi-family model instability.
                      A Consultant Dermatologist will perform a direct evaluation.
                    </p>
                    <p className="mt-3 font-mono text-xs text-red-600 dark:text-red-400">
                      Reference ID: <strong>{result.case_id}</strong>
                    </p>
                  </div>
                )}

                {/* Action row */}
                <div className="mt-5 flex items-center gap-3 flex-wrap">
                  {report.state !== "CLOSED" && (
                    <Link to="/doctor" className={`${report.state === "OPEN" ? "btn-danger" : "btn-primary"} flex items-center gap-1.5 text-sm shadow-sm`}>
                      {report.state === "OPEN" ? "Specialist Override Panel" : "View Physician Review"} <ArrowRight className="w-4 h-4" />
                    </Link>
                  )}
                  <span className="text-xs text-slate-500 font-mono">Case: {result.case_id}</span>
                </div>
              </div>
            </div>
          </div>

          {/* ── 2. Original image display (always) + GradCAM (doctor or flips) ─ */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Original image — always show */}
            <div className="card p-6 space-y-3">
              <div className="flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-baltic_blue dark:text-baltic_blue-700" />
                <div className="text-sm font-bold text-slate-800 dark:text-slate-200">
                  Lesion Image
                </div>
              </div>
              {gradcam?.original_image_b64 ? (
                <img
                  src={`data:image/png;base64,${gradcam.original_image_b64}`}
                  alt="original lesion"
                  className="w-full max-w-sm mx-auto rounded-xl border border-slate-200 dark:border-[#033e56] shadow-sm object-cover"
                />
              ) : (
                <div className="w-full aspect-square max-w-sm mx-auto rounded-xl bg-slate-100 dark:bg-[#022a39] flex items-center justify-center text-slate-400">
                  No image
                </div>
              )}
              {report.state === "CLOSED" && (
                <p className="text-xs text-center text-slate-500">
                  Input lesion — AI verified stable
                </p>
              )}
            </div>

            {/* GradCAM baseline — shown always for doctor, only for non-CLOSED for patients */}
            {(isDoctor || report.state !== "CLOSED") && gradcam?.baseline_cam_b64 && (
              <div className="card p-6 space-y-3">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-teal dark:text-mint_leaf" />
                  <div className="text-sm font-bold text-slate-800 dark:text-slate-200">
                    Baseline Grad-CAM Attention
                  </div>
                </div>
                <img
                  src={`data:image/png;base64,${gradcam.baseline_cam_b64}`}
                  alt="baseline gradcam"
                  className="w-full max-w-sm mx-auto rounded-xl border border-slate-200 dark:border-[#033e56] shadow-sm object-cover"
                />
                <p className="text-xs text-center text-slate-500">
                  ConvNeXt focus area for {report.baseline_label} ({(report.baseline_confidence * 100).toFixed(1)}%)
                </p>
              </div>
            )}
          </div>

          {/* ── 3. Flip GradCAMs (shown to doctor on HALF_OPEN/OPEN, or for anyone with flips) ── */}
          {gradcam?.flip_cams?.length > 0 && (isDoctor || report.state !== "CLOSED") && (
            <div className="card p-6">
              <div className="text-sm font-bold text-slate-800 dark:text-slate-200 mb-4">
                Attention Drift Under Metamorphic Transforms ({gradcam.flip_cams.length} flip{gradcam.flip_cams.length !== 1 ? "s" : ""})
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                {gradcam.flip_cams.map((fc, i) => (
                  <div key={i} className="space-y-2">
                    <img
                      src={`data:image/png;base64,${fc.cam_overlay_b64}`}
                      alt={fc.test_name}
                      className="w-full rounded-xl border border-red-200 dark:border-red-900 shadow-sm"
                    />
                    <div className="px-1">
                      <div className="font-semibold text-xs text-red-700 dark:text-red-300">
                        {report.baseline_label} &rarr; {fc.flipped_label}
                      </div>
                      <div className="text-[11px] text-slate-500 mt-0.5">
                        {fc.test_name} &bull; {fc.severity} &bull; w={fc.weight}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── 4. Quick metrics strip ── */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              { label: "Circuit State", value: report.state.replace("_", "-"), sub: "" },
              { label: "CBI Risk Score", value: report.cbi.toFixed(4), sub: "Composite brittleness" },
              { label: "Prediction Flips", value: `${report.n_flips} / ${report.n_tests}`, sub: "Metamorphic tests" },
              { label: "Peak Family IR", value: `${report.peak_family_score.toFixed(3)}`, sub: report.peak_family || "" },
            ].map(({ label, value, sub }) => (
              <div key={label} className="stat-card text-center">
                <div className="section-label">{label}</div>
                <div className="text-2xl font-black text-baltic_blue dark:text-white mt-1">{value}</div>
                {sub && <div className="text-[11px] text-slate-500 mt-0.5">{sub}</div>}
              </div>
            ))}
          </div>

          {/* ── 5. CBI Gauge ── */}
          {cfg && (
            <div className="card p-6">
              <div className="text-sm font-bold text-slate-800 dark:text-slate-200 mb-4">
                Composite Brittleness Index Gauge
              </div>
              <CbiGauge cbi={report.cbi} thetaWarn={cfg.theta_warn} thetaTrip={cfg.theta_trip} />
            </div>
          )}

          {/* ── 6. Technical details (accordion, doctor sees by default) ─────── */}
          <div className="card overflow-hidden">
            <button
              onClick={() => setShowTechDetails(v => !v)}
              className="w-full px-6 py-4 flex items-center justify-between text-sm font-bold text-slate-800 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-[#033e56]/30 transition-colors"
            >
              <span className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-baltic_blue dark:text-baltic_blue-700" />
                {isDoctor ? "Mathematical Attribution & Full Test Matrix" : "Technical Safety Report"}
              </span>
              {showTechDetails ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
            </button>
            {showTechDetails && cfg && (
              <div className="px-6 pb-6 pt-2 border-t border-slate-100 dark:border-[#033e56] space-y-6">
                {/* Formula */}
                <div className="bg-slate-900 text-slate-200 rounded-xl p-5 font-mono text-xs leading-relaxed space-y-1">
                  <div className="text-slate-400">CBI(x) = alpha * max_k(IR_k) + beta * CFS + gamma * (1 - c0)</div>
                  <div>= {cfg.alpha} * {report.peak_family_score.toFixed(4)} + {cfg.beta} * {report.cross_family_spread.toFixed(4)} + {cfg.gamma} * {(1 - report.baseline_confidence).toFixed(4)}</div>
                  <div>= {(cfg.alpha * report.peak_family_score).toFixed(4)} + {(cfg.beta * report.cross_family_spread).toFixed(4)} + {(cfg.gamma * (1 - report.baseline_confidence)).toFixed(4)}</div>
                  <div className="text-mint_leaf font-bold text-sm pt-1">= {report.cbi.toFixed(4)}</div>
                </div>

                {/* Family breakdown */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {Object.entries(report.family_instability).map(([fam, ir]) => {
                    const compromised = ir >= cfg.tau_fam;
                    return (
                      <div key={fam} className="stat-card">
                        <div className="section-label">{fam}</div>
                        <div className={`text-2xl font-black mt-1 ${compromised ? "text-red-500" : "text-verdigris"}`}>
                          {ir.toFixed(4)}
                        </div>
                        <div className="text-xs text-slate-500">
                          {compromised ? `Compromised (>= ${cfg.tau_fam})` : `Stable (< ${cfg.tau_fam})`}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Test table toggle */}
                <button
                  onClick={() => setShowTests(v => !v)}
                  className="text-xs font-bold text-baltic_blue dark:text-baltic_blue-700 hover:underline flex items-center gap-1"
                >
                  {showTests ? "Collapse" : "Expand"} 22-test evaluation matrix
                  {showTests ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                </button>
                {showTests && (
                  <div className="overflow-x-auto max-h-80 border border-slate-200 dark:border-[#033e56] rounded-xl">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="bg-slate-100 dark:bg-[#01151d] text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-[#033e56]">
                          {["ID", "Transform", "Family", "Severity", "Weight", "Prediction", "Confidence", "Result"].map(h => (
                            <th key={h} className="py-2 px-3 text-left font-bold">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 dark:divide-[#033e56]">
                        {report.test_results.map(r => (
                          <tr key={r.test_id} className={r.flipped ? "bg-red-50 dark:bg-red-950/20" : ""}>
                            <td className="py-2 px-3 font-mono font-bold">{r.test_id}</td>
                            <td className="py-2 px-3">{r.test_name}</td>
                            <td className="py-2 px-3">{r.family}</td>
                            <td className="py-2 px-3">{r.severity}</td>
                            <td className="py-2 px-3 font-mono">{r.weight}</td>
                            <td className="py-2 px-3 font-bold">{r.pred_label}</td>
                            <td className="py-2 px-3 font-mono">{(r.confidence * 100).toFixed(1)}%</td>
                            <td className={`py-2 px-3 font-bold ${r.flipped ? "text-red-600" : "text-verdigris"}`}>
                              {r.flipped ? "FLIP" : "Stable"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ── Reset ── */}
          <button
            onClick={() => { setResult(null); setFile(null); setPreview(null); }}
            className="btn-secondary w-full py-3 font-semibold"
          >
            Screen Another Image
          </button>
        </div>
      )}
    </div>
  );
}

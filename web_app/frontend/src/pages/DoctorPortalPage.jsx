// src/pages/DoctorPortalPage.jsx
// DOCTOR-ONLY portal with role guard. Shows forwarded cases queue + image review panel.
import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  Stethoscope, RefreshCw, CheckCircle, AlertTriangle, XCircle,
  ChevronDown, ChevronUp, Loader2, ShieldAlert, FileCheck,
  ImageIcon, Activity, ArrowRight, Lock, User
} from "lucide-react";
import { fetchCases, fetchCase, submitReview, fetchStats } from "../api/client";
import { useRole, ROLES } from "../context/RoleContext";
import StateBadge from "../components/StateBadge";
import CbiGauge from "../components/CbiGauge";

const CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"];
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
const FAM_LABEL = {
  geometric: "Geometric",
  photometric: "Photometric",
  sensor_noise: "Sensor / Noise",
};

function StateIcon({ state }) {
  if (state === "CLOSED")    return <CheckCircle className="w-4 h-4 text-verdigris" />;
  if (state === "HALF_OPEN") return <AlertTriangle className="w-4 h-4 text-amber-500" />;
  return <XCircle className="w-4 h-4 text-red-500" />;
}

// ── Role Guard ────────────────────────────────────────────────────────────────
function AccessDenied() {
  const { setRole } = useRole();
  return (
    <div className="max-w-lg mx-auto px-4 py-24 text-center space-y-6">
      <div className="w-20 h-20 rounded-full bg-red-100 dark:bg-red-950/40 flex items-center justify-center mx-auto">
        <Lock className="w-10 h-10 text-red-500" />
      </div>
      <div>
        <h2 className="text-2xl font-extrabold text-slate-900 dark:text-white mb-2">
          Physician Access Required
        </h2>
        <p className="text-slate-500 dark:text-slate-400 text-sm leading-relaxed">
          The Physician Review Console is restricted to registered Consultant Dermatologists.
          Switch to the Doctor role using the role selector in the navigation bar, or return to the Patient Portal.
        </p>
      </div>
      <div className="flex items-center justify-center gap-3 flex-wrap">
        <button
          onClick={() => setRole(ROLES.DOCTOR)}
          className="btn-primary flex items-center gap-2"
        >
          <Stethoscope className="w-4 h-4" /> Switch to Doctor Role
        </button>
        <Link to="/portal" className="btn-secondary flex items-center gap-2">
          <User className="w-4 h-4" /> Go to Patient Portal
        </Link>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function DoctorPortalPage() {
  const { role } = useRole();
  const isDoctor = role === ROLES.DOCTOR || role === ROLES.ADMIN;

  const [cases, setCases]               = useState([]);
  const [stats, setStats]               = useState(null);
  const [filter, setFilter]             = useState("action");
  const [loading, setLoading]           = useState(true);
  const [selectedCase, setSelectedCase] = useState(null);
  const [caseDetail, setCaseDetail]     = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [reviewAction, setReviewAction] = useState("APPROVE");
  const [overrideDx, setOverrideDx]     = useState("");
  const [clinicalNotes, setClinicalNotes] = useState("");
  const [biopsyRec, setBiopsyRec]       = useState(false);
  const [urgency, setUrgency]           = useState("routine");
  const [submitting, setSubmitting]     = useState(false);
  const [reviewSuccess, setReviewSuccess] = useState(false);
  const [showMath, setShowMath]         = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const [c, s] = await Promise.all([fetchCases(), fetchStats()]);
      setCases(c.data);
      setStats(s.data);
    } catch {}
    setLoading(false);
  };

  useEffect(() => { if (isDoctor) loadData(); }, [isDoctor]);

  // Don't render full page for non-doctors
  if (!isDoctor) return <AccessDenied />;

  const openCase = async (c) => {
    setSelectedCase(c);
    setCaseDetail(null);
    setDetailLoading(true);
    setReviewSuccess(false);
    setReviewAction("APPROVE");
    setOverrideDx("");
    setClinicalNotes("");
    setBiopsyRec(false);
    setUrgency("routine");
    setShowMath(false);
    try {
      const r = await fetchCase(c.case_id);
      setCaseDetail(r.data);
    } catch {}
    setDetailLoading(false);
  };

  const handleReview = async () => {
    if (!selectedCase) return;
    setSubmitting(true);
    try {
      await submitReview(selectedCase.case_id, {
        action: reviewAction,
        override_diagnosis: reviewAction === "OVERRIDE" ? (overrideDx || caseDetail?.report?.baseline_label) : null,
        override_label: reviewAction === "OVERRIDE" ? overrideDx : null,
        clinical_notes: clinicalNotes,
        doctor_name: "Dr. Sarah Jenkins, MD",
        biopsy_recommended: biopsyRec,
        follow_up_urgency: urgency,
      });
      setReviewSuccess(true);
      await loadData();
    } catch {}
    setSubmitting(false);
  };

  const filtered = cases.filter(c => {
    if (filter === "action")    return c.status === "pending_review";
    if (filter === "completed") return c.status !== "pending_review";
    return true;
  });

  const report  = caseDetail?.report;
  const gradcam = caseDetail?.gradcam;
  const cfg     = report?.config_used;
  const review  = caseDetail?.doctor_review;

  return (
    <div className="max-w-7xl mx-auto px-4 py-10">

      {/* ── Header ───────────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-start justify-between gap-4 mb-8">
        <div>
          <div className="section-label mb-1">Clinical Triage Workstation</div>
          <h1 className="text-3xl font-extrabold text-baltic_blue dark:text-white flex items-center gap-2.5">
            <Stethoscope className="w-8 h-8 text-teal dark:text-mint_leaf" />
            Physician Review Console
          </h1>
          <p className="text-slate-600 dark:text-slate-400 text-sm mt-1.5">
            <span className="font-semibold text-teal dark:text-mint_leaf">Dr. Sarah Jenkins, MD</span>
            &ensp;&bull;&ensp;Lead Dermatologist &amp; Cutaneous Oncology Fellow
            &ensp;&bull;&ensp;Aegis MetroHealth &amp; Cancer Center
          </p>
        </div>
        <button onClick={loadData} className="btn-secondary flex items-center gap-1.5 shadow-sm mt-2">
          <RefreshCw className="w-4 h-4" /> Refresh Queue
        </button>
      </div>

      {/* ── Stats ────────────────────────────────────────────────────────────── */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-8">
          {[
            { label: "Total Screenings", val: stats.total_cases,   color: "text-baltic_blue dark:text-white" },
            { label: "Auto-Approved",    val: stats.auto_approved,  color: "text-verdigris dark:text-mint_leaf" },
            { label: "Pending Review",   val: stats.pending_review, color: "text-amber-600 dark:text-amber-400" },
            { label: "Tripped (Open)",   val: stats.tripped_open,   color: "text-red-600 dark:text-red-400" },
            { label: "Half-Open",        val: stats.half_open,      color: "text-teal dark:text-baltic_blue-700" },
            { label: "Overridden",       val: stats.overridden,     color: "text-baltic_blue dark:text-teal" },
          ].map(({ label, val, color }) => (
            <div key={label} className="stat-card text-center">
              <div className="section-label">{label}</div>
              <div className={`text-3xl font-black mt-1 ${color}`}>{val}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── Main layout: queue left, detail right ───────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">

        {/* ── LEFT: Case Queue ──────────────────────────────────────────────── */}
        <div className="space-y-4">
          <div className="card p-4 space-y-3">
            <div className="text-xs font-bold uppercase tracking-wider text-slate-500">Filter Queue</div>
            <div className="flex rounded-xl overflow-hidden border border-slate-200 dark:border-[#033e56] text-xs font-bold bg-slate-100 dark:bg-[#01151d]">
              {[["action", "Action Required"], ["all", "All"], ["completed", "Completed"]].map(([f, l]) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`flex-1 py-2 transition-all ${
                    filter === f ? "bg-baltic_blue text-white" : "text-slate-600 dark:text-slate-400 hover:text-baltic_blue"
                  }`}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>

          {loading ? (
            <div className="card p-8 text-center">
              <Loader2 className="w-8 h-8 text-baltic_blue spinner mx-auto mb-2" />
              <div className="text-sm text-slate-500">Loading case queue...</div>
            </div>
          ) : filtered.length === 0 ? (
            <div className="card p-10 text-center text-slate-400 space-y-2">
              <FileCheck className="w-10 h-10 mx-auto text-verdigris/40" />
              <div className="text-sm font-semibold">No cases in this view</div>
              <div className="text-xs">Run a new screening from the Patient Portal</div>
            </div>
          ) : (
            <div className="space-y-2 max-h-[70vh] overflow-y-auto pr-1 scrollbar-thin">
              {filtered.map(c => {
                const urgent = c.state !== "CLOSED";
                return (
                  <button
                    key={c.case_id}
                    onClick={() => openCase(c)}
                    className={`w-full text-left p-4 rounded-xl border-2 transition-all space-y-1.5 ${
                      selectedCase?.case_id === c.case_id
                        ? "border-baltic_blue ring-2 ring-baltic_blue/20 bg-baltic_blue/5 dark:bg-baltic_blue-200/20 shadow"
                        : urgent
                          ? "border-amber-200 dark:border-amber-900/50 hover:border-baltic_blue/40 bg-amber-50/40 dark:bg-amber-950/10"
                          : "border-slate-200 dark:border-[#033e56] hover:border-baltic_blue/40"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[10px] text-slate-500">{c.case_id}</span>
                      <StateIcon state={c.state} />
                    </div>
                    <div className="flex items-center justify-between gap-2">
                      <span className={`font-black text-xl ${urgent ? "text-amber-600 dark:text-amber-400" : "text-verdigris dark:text-mint_leaf"}`}>
                        {c.baseline_label}
                      </span>
                      <span className="text-[11px] font-semibold text-slate-500">
                        {(c.baseline_confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <StateBadge state={c.state} />
                      <span className="text-[10px] text-slate-400">
                        CBI {c.cbi.toFixed(3)}
                      </span>
                    </div>
                    {c.status !== "pending_review" && (
                      <div className="text-[11px] font-semibold text-verdigris dark:text-mint_leaf">
                        Reviewed
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* ── RIGHT: Case detail panel ──────────────────────────────────────── */}
        <div>
          {!selectedCase && (
            <div className="card p-12 text-center h-full flex flex-col items-center justify-center space-y-4 min-h-[400px]">
              <Stethoscope className="w-14 h-14 text-teal/30 dark:text-mint_leaf/20" />
              <div className="text-lg font-bold text-slate-700 dark:text-slate-300">Select a case to review</div>
              <p className="text-sm text-slate-400 max-w-xs">
                Choose a case from the queue on the left. Cases forwarded for review require your clinical assessment before a result is released to the patient.
              </p>
            </div>
          )}

          {selectedCase && detailLoading && (
            <div className="card p-12 flex flex-col items-center justify-center min-h-[400px] space-y-4">
              <Loader2 className="w-10 h-10 text-baltic_blue spinner" />
              <div className="text-slate-500 text-sm">Loading case details...</div>
            </div>
          )}

          {selectedCase && caseDetail && report && (
            <div className="space-y-6">

              {/* ── Case header ── */}
              <div className="card p-6 space-y-4">
                <div className="flex items-start justify-between gap-4 flex-wrap">
                  <div>
                    <div className="section-label mb-1">Case Record</div>
                    <div className="font-mono text-lg font-bold text-slate-900 dark:text-white">{caseDetail.case_id}</div>
                    <div className="text-xs text-slate-400 mt-0.5">
                      {new Date(caseDetail.created_at).toLocaleString("en-GB", {
                        day: "2-digit", month: "short", year: "numeric",
                        hour: "2-digit", minute: "2-digit"
                      })}
                    </div>
                  </div>
                  <StateBadge state={report.state} size="lg" />
                </div>

                {/* ── AI primary result block ── */}
                <div className={`rounded-2xl p-5 border-2 ${
                  report.state === "CLOSED"
                    ? "border-verdigris/40 bg-verdigris-900/10 dark:bg-verdigris-100/20"
                    : report.state === "HALF_OPEN"
                      ? "border-amber-300 dark:border-amber-700/50 bg-amber-50 dark:bg-amber-950/30"
                      : "border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/20"
                }`}>
                  <div className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3">
                    AI Screening Result
                  </div>
                  <div className="flex items-end gap-8 flex-wrap">
                    <div>
                      <div className="text-5xl font-black text-baltic_blue dark:text-white">{report.baseline_label}</div>
                      <div className="text-base font-semibold text-slate-600 dark:text-slate-300 mt-1">
                        {CLASS_FULL[report.baseline_label]}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-bold text-slate-500 mb-0.5">Confidence</div>
                      <div className="text-3xl font-black text-teal dark:text-mint_leaf">
                        {(report.baseline_confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-bold text-slate-500 mb-0.5">Flips / Tests</div>
                      <div className="text-3xl font-black text-slate-700 dark:text-slate-200">
                        {report.n_flips}<span className="text-sm font-normal text-slate-500">/{report.n_tests}</span>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-bold text-slate-500 mb-0.5">CBI</div>
                      <div className={`text-3xl font-black ${
                        report.state === "CLOSED" ? "text-verdigris" : report.state === "HALF_OPEN" ? "text-amber-500" : "text-red-500"
                      }`}>{report.cbi.toFixed(4)}</div>
                    </div>
                  </div>
                  {report.state !== "CLOSED" && (
                    <div className="mt-4 text-sm font-medium border-l-4 pl-3 border-amber-400 text-amber-800 dark:text-amber-300">
                      {report.state === "HALF_OPEN"
                        ? `Borderline sensitivity detected in ${FAM_LABEL[report.peak_family] || report.peak_family} family (IR_k = ${report.peak_family_score.toFixed(3)}). Pending physician verification.`
                        : `Circuit breaker TRIPPED. Multi-family collapse detected in: ${report.compromised_families.map(f => FAM_LABEL[f] || f).join(", ")}. Automated diagnosis blocked.`
                      }
                    </div>
                  )}
                </div>
              </div>

              {/* ── OVERRIDE result block (if already reviewed) ── */}
              {(caseDetail.override_diagnosis || review) && (
                <div className="card p-6 border-l-4 border-teal dark:border-mint_leaf space-y-3">
                  <div className="section-label flex items-center gap-2">
                    <ShieldAlert className="w-4 h-4 text-teal dark:text-mint_leaf" />
                    Physician Override — Final Result
                  </div>
                  <div className="flex items-end gap-6 flex-wrap">
                    {caseDetail.override_diagnosis && (
                      <div>
                        <div className="text-xs font-bold text-slate-500 mb-1">Final Diagnosis (Physician)</div>
                        <div className="text-4xl font-black text-teal dark:text-mint_leaf">{caseDetail.override_diagnosis}</div>
                        <div className="text-base text-slate-500">{CLASS_FULL[caseDetail.override_diagnosis]}</div>
                        <div className="text-xs text-slate-400 mt-1">
                          AI suggested <strong>{report.baseline_label}</strong> &rarr; Overridden to <strong>{caseDetail.override_diagnosis}</strong>
                        </div>
                      </div>
                    )}
                    {!caseDetail.override_diagnosis && review && (
                      <div>
                        <div className="text-xs font-bold text-slate-500 mb-1">Physician Decision</div>
                        <div className="text-2xl font-black text-verdigris">Approved</div>
                        <div className="text-sm text-slate-500">Concurs with AI: {report.baseline_label}</div>
                      </div>
                    )}
                  </div>
                  {review?.clinical_notes && (
                    <div className="p-4 bg-slate-50 dark:bg-[#01151d] rounded-xl text-sm text-slate-700 dark:text-slate-300">
                      <span className="font-bold block mb-1">Clinical Notes:</span>
                      {review.clinical_notes}
                    </div>
                  )}
                  {review?.biopsy_recommended && (
                    <div className="text-sm font-semibold text-red-600 dark:text-red-400">
                      Biopsy recommended &bull; Urgency: {review.follow_up_urgency}
                    </div>
                  )}
                  {caseDetail.doctor_name && (
                    <div className="text-xs text-slate-400">
                      Reviewed by: {caseDetail.doctor_name}
                    </div>
                  )}
                </div>
              )}

              {/* ── Image panel: original + baseline GradCAM ── */}
              <div className="card p-6 space-y-4">
                <div className="section-label">Visual Analysis</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                  {/* Original lesion */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm font-bold text-slate-800 dark:text-slate-200">
                      <ImageIcon className="w-4 h-4 text-teal dark:text-mint_leaf" />
                      Original Lesion Image
                    </div>
                    {gradcam?.original_image_b64 ? (
                      <img
                        src={`data:image/png;base64,${gradcam.original_image_b64}`}
                        alt="original lesion"
                        className="w-full rounded-xl border border-slate-200 dark:border-[#033e56] shadow object-cover"
                      />
                    ) : (
                      <div className="aspect-square rounded-xl bg-slate-100 dark:bg-[#022a39] flex items-center justify-center text-slate-400 text-sm">
                        Not available
                      </div>
                    )}
                    <p className="text-xs text-center text-slate-500">Unmodified input image</p>
                  </div>

                  {/* Baseline GradCAM */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm font-bold text-slate-800 dark:text-slate-200">
                      <Activity className="w-4 h-4 text-amber-500" />
                      Baseline Grad-CAM Attention
                    </div>
                    {gradcam?.baseline_cam_b64 ? (
                      <img
                        src={`data:image/png;base64,${gradcam.baseline_cam_b64}`}
                        alt="baseline gradcam"
                        className="w-full rounded-xl border border-slate-200 dark:border-[#033e56] shadow object-cover"
                      />
                    ) : (
                      <div className="aspect-square rounded-xl bg-slate-100 dark:bg-[#022a39] flex items-center justify-center text-slate-400 text-sm">
                        Not available
                      </div>
                    )}
                    <p className="text-xs text-center text-slate-500">
                      ConvNeXt attention — {report.baseline_label} ({(report.baseline_confidence * 100).toFixed(1)}%)
                    </p>
                  </div>
                </div>
              </div>

              {/* ── Flip GradCAMs (doctor always sees these) ── */}
              {gradcam?.flip_cams?.length > 0 && (
                <div className="card p-6 space-y-4">
                  <div className="section-label">
                    Grad-CAM Attention Drift — Metamorphic Flip Analysis ({gradcam.flip_cams.length})
                  </div>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Each panel below shows the transformed image (left) and its GradCAM overlay (right) for a test
                    that caused a prediction flip. Attention drifting to boundary artifacts, rotation halos, or
                    dermatoscope rings is a signal of model fragility.
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                    {gradcam.flip_cams.map((fc, i) => {
                      const sevColor = fc.severity === "severe"
                        ? "border-red-400 bg-red-50 dark:bg-red-950/20"
                        : fc.severity === "moderate"
                          ? "border-amber-400 bg-amber-50 dark:bg-amber-950/20"
                          : "border-slate-300 dark:border-[#033e56] bg-slate-50 dark:bg-[#01151d]";
                      return (
                        <div key={i} className={`rounded-xl border-2 p-4 space-y-3 ${sevColor}`}>
                          <div className="text-xs font-bold text-slate-800 dark:text-slate-200">
                            Flip {i + 1}: {report.baseline_label} &rarr; {fc.flipped_label}
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            {fc.transformed_image_b64 && (
                              <div className="space-y-1">
                                <img
                                  src={`data:image/png;base64,${fc.transformed_image_b64}`}
                                  alt="transformed"
                                  className="w-full rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm"
                                />
                                <p className="text-[10px] text-center text-slate-500">Transformed</p>
                              </div>
                            )}
                            {fc.cam_overlay_b64 && (
                              <div className="space-y-1">
                                <img
                                  src={`data:image/png;base64,${fc.cam_overlay_b64}`}
                                  alt="flip gradcam"
                                  className="w-full rounded-lg border border-red-200 dark:border-red-800 shadow-sm"
                                />
                                <p className="text-[10px] text-center text-slate-500">Grad-CAM</p>
                              </div>
                            )}
                          </div>
                          <div className="text-[11px] space-y-0.5 text-slate-600 dark:text-slate-400">
                            <div className="font-semibold">{fc.test_name}</div>
                            <div>{FAM_LABEL[fc.family] || fc.family} &bull; {fc.severity} (w={fc.weight})</div>
                            <div>
                              {fc.flipped_label} at {(fc.flipped_confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* ── CBI Gauge + Math ── */}
              {cfg && (
                <div className="card p-6 space-y-4">
                  <div className="section-label">Composite Brittleness Index Analysis</div>
                  <CbiGauge cbi={report.cbi} thetaWarn={cfg.theta_warn} thetaTrip={cfg.theta_trip} />

                  <button
                    onClick={() => setShowMath(v => !v)}
                    className="text-xs font-bold text-baltic_blue dark:text-baltic_blue-700 hover:underline flex items-center gap-1"
                  >
                    {showMath ? "Hide" : "Show"} CBI formula decomposition
                    {showMath ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  </button>

                  {showMath && (
                    <div className="space-y-4">
                      <div className="bg-slate-900 text-slate-200 rounded-xl p-5 font-mono text-xs leading-relaxed space-y-1">
                        <div className="text-slate-400">CBI = alpha * max_k(IR_k) + beta * CFS + gamma * (1 - c0)</div>
                        <div>= {cfg.alpha} * {report.peak_family_score.toFixed(4)} + {cfg.beta} * {report.cross_family_spread.toFixed(4)} + {cfg.gamma} * {(1 - report.baseline_confidence).toFixed(4)}</div>
                        <div>= {(cfg.alpha * report.peak_family_score).toFixed(4)} + {(cfg.beta * report.cross_family_spread).toFixed(4)} + {(cfg.gamma * (1 - report.baseline_confidence)).toFixed(4)}</div>
                        <div className="text-mint_leaf font-bold text-sm pt-1">= {report.cbi.toFixed(4)}</div>
                        <div className="text-slate-400 pt-2">
                          Decision: CBI {report.cbi.toFixed(4)}{" "}
                          {report.cbi < cfg.theta_warn
                            ? `< theta_warn (${cfg.theta_warn}) → CLOSED`
                            : report.cbi < cfg.theta_trip
                              ? `∈ [${cfg.theta_warn}, ${cfg.theta_trip}) → HALF-OPEN`
                              : `>= theta_trip (${cfg.theta_trip}) → OPEN`
                          }
                        </div>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        {Object.entries(report.family_instability).map(([fam, ir]) => {
                          const comp = ir >= cfg.tau_fam;
                          return (
                            <div key={fam} className="stat-card">
                              <div className="section-label">{FAM_LABEL[fam] || fam}</div>
                              <div className={`text-2xl font-black mt-1 ${comp ? "text-red-500" : "text-verdigris"}`}>
                                {ir.toFixed(4)}
                              </div>
                              <div className="text-xs text-slate-500">
                                {comp ? `Compromised (>= ${cfg.tau_fam})` : `Stable (< ${cfg.tau_fam})`}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ── Review action panel ── */}
              {!reviewSuccess && caseDetail.status === "pending_review" && (
                <div className="card p-6 border-t-4 border-teal dark:border-mint_leaf space-y-5">
                  <div className="section-label flex items-center gap-2">
                    <ShieldAlert className="w-4 h-4 text-teal dark:text-mint_leaf" />
                    Physician Decision
                  </div>

                  <div className="flex rounded-xl overflow-hidden border border-slate-200 dark:border-[#033e56] text-sm font-bold">
                    {[["APPROVE", "Verify & Approve"], ["OVERRIDE", "Override Diagnosis"]].map(([a, l]) => (
                      <button
                        key={a}
                        onClick={() => setReviewAction(a)}
                        className={`flex-1 py-3 transition-all ${
                          reviewAction === a
                            ? a === "APPROVE"
                              ? "bg-verdigris text-white shadow"
                              : "bg-amber-500 text-white shadow"
                            : "text-slate-600 dark:text-slate-400 hover:text-baltic_blue bg-slate-50 dark:bg-[#01151d]"
                        }`}
                      >
                        {l}
                      </button>
                    ))}
                  </div>

                  {reviewAction === "OVERRIDE" && (
                    <div className="space-y-3">
                      <div>
                        <label className="text-xs font-bold text-slate-600 dark:text-slate-400 block mb-1.5">
                          Final Diagnosis (Physician Override)
                        </label>
                        <select
                          value={overrideDx}
                          onChange={e => setOverrideDx(e.target.value)}
                          className="w-full bg-slate-50 dark:bg-[#01151d] border border-slate-200 dark:border-[#033e56] rounded-xl px-4 py-3 text-sm font-bold focus:outline-none focus:border-teal"
                        >
                          <option value="">Select diagnosis</option>
                          {CLASS_NAMES.map(c => (
                            <option key={c} value={c}>{c} — {CLASS_FULL[c]}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  )}

                  <div>
                    <label className="text-xs font-bold text-slate-600 dark:text-slate-400 block mb-1.5">
                      Clinical Notes
                    </label>
                    <textarea
                      value={clinicalNotes}
                      onChange={e => setClinicalNotes(e.target.value)}
                      rows={3}
                      placeholder="Enter clinical observations, differential considerations, or decision rationale..."
                      className="w-full bg-slate-50 dark:bg-[#01151d] border border-slate-200 dark:border-[#033e56] rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:border-teal"
                    />
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <label className="flex items-center gap-3 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={biopsyRec}
                        onChange={e => setBiopsyRec(e.target.checked)}
                        className="w-4 h-4 accent-teal"
                      />
                      <span className="text-sm font-semibold text-slate-700 dark:text-slate-300 group-hover:text-teal">
                        Recommend Biopsy
                      </span>
                    </label>
                    <div>
                      <label className="text-xs font-bold text-slate-600 dark:text-slate-400 block mb-1.5">
                        Follow-up Urgency
                      </label>
                      <select
                        value={urgency}
                        onChange={e => setUrgency(e.target.value)}
                        className="w-full bg-slate-50 dark:bg-[#01151d] border border-slate-200 dark:border-[#033e56] rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-teal"
                      >
                        <option value="routine">Routine (within 2 weeks)</option>
                        <option value="urgent">Urgent (within 48 hours)</option>
                        <option value="emergency">Emergency (same-day)</option>
                      </select>
                    </div>
                  </div>

                  <button
                    onClick={handleReview}
                    disabled={submitting}
                    className={`w-full py-4 font-bold text-base rounded-xl flex items-center justify-center gap-2 transition-all shadow ${
                      reviewAction === "OVERRIDE"
                        ? "bg-amber-500 hover:bg-amber-600 text-white"
                        : "bg-verdigris hover:bg-verdigris-600 text-white"
                    }`}
                  >
                    {submitting ? (
                      <><Loader2 className="w-5 h-5 spinner" /> Submitting...</>
                    ) : reviewAction === "OVERRIDE" ? (
                      <><ShieldAlert className="w-5 h-5" /> Submit Physician Override</>
                    ) : (
                      <><FileCheck className="w-5 h-5" /> Approve &amp; Release to Patient</>
                    )}
                  </button>
                </div>
              )}

              {reviewSuccess && (
                <div className="card p-8 text-center space-y-3">
                  <CheckCircle className="w-12 h-12 text-verdigris mx-auto" />
                  <div className="text-xl font-extrabold text-slate-900 dark:text-white">Review Submitted</div>
                  <p className="text-sm text-slate-500">
                    Case {selectedCase?.case_id} has been updated. The patient result has been released.
                  </p>
                  <button
                    onClick={() => { setSelectedCase(null); setCaseDetail(null); setReviewSuccess(false); }}
                    className="btn-secondary"
                  >
                    Back to Queue
                  </button>
                </div>
              )}

              {caseDetail.status !== "pending_review" && !reviewSuccess && (
                <div className="card p-4 text-center text-sm text-verdigris dark:text-mint_leaf font-semibold">
                  This case has already been reviewed and closed.
                </div>
              )}

            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// src/pages/HomePage.jsx
import { Link } from "react-router-dom";
import {
  Activity, Shield, Stethoscope, Brain, ArrowRight,
  CheckCircle, AlertTriangle, XCircle, Microscope,
  Award, Clock, Users, BarChart3, ChevronRight
} from "lucide-react";

const DEPARTMENTS = [
  { icon: Stethoscope, label: "Dermatology",        color: "text-teal",           desc: "Cutaneous lesion screening & AI triage" },
  { icon: Microscope,  label: "Dermatopathology",   color: "text-baltic_blue",    desc: "Digital histopathology correlation" },
  { icon: Brain,       label: "Cutaneous Oncology", color: "text-verdigris",      desc: "Melanoma risk stratification & staging" },
  { icon: Activity,    label: "Medical Diagnostics", color: "text-baltic_blue-600", desc: "Computer vision safety interception" },
];

const STATS = [
  { icon: Users,     value: "12,400+", label: "Lesion Screenings Conducted" },
  { icon: Shield,    value: "99.2%",   label: "Safety Intercept Precision" },
  { icon: Clock,     value: "< 5s",    label: "Real-time AI Verification" },
  { icon: BarChart3, value: "22",      label: "Metamorphic Tests per Query" },
];

const STATE_EXPLAINERS = [
  {
    icon: CheckCircle,
    color: "text-verdigris dark:text-mint_leaf",
    bg: "bg-white dark:bg-[#022a39] border-verdigris/30 dark:border-verdigris/50",
    title: "CLOSED State — Auto-Approved",
    desc: "CBI is below warning threshold (theta_warn). The model maintains rock-solid stability across all geometric, photometric, and noise perturbations. Prediction is verified immediately.",
  },
  {
    icon: AlertTriangle,
    color: "text-amber-600 dark:text-amber-400",
    bg: "bg-white dark:bg-[#022a39] border-amber-300 dark:border-amber-700/60",
    title: "HALF-OPEN State — Monitor & Review",
    desc: "CBI is between theta_warn and theta_trip. Single-family perturbation fragility detected. Case is forwarded to the Senior Dermatologist queue with Grad-CAM evidence for validation.",
  },
  {
    icon: XCircle,
    color: "text-red-600 dark:text-red-400",
    bg: "bg-white dark:bg-[#022a39] border-red-300 dark:border-red-700/60",
    title: "OPEN State — Safety Intercept",
    desc: "CBI exceeds trip threshold (theta_trip). Multi-family collapse detected. Automated AI output is intercepted and blocked. Dermatologist reviews the patient and records authoritative clinical override.",
  },
];

export default function HomePage() {
  return (
    <div className="min-h-screen">

      {/* ── HERO — 2-col: text left, hospital image right ───────────────────── */}
      <section className="relative overflow-hidden border-b border-slate-200 dark:border-[#033e56] bg-gradient-to-br from-white via-slate-50 to-[#f0f7fa] dark:from-[#01151d] dark:via-[#022a39] dark:to-[#01151d] py-16 md:py-20">
        <div className="max-w-7xl mx-auto px-6 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">

            {/* Left: Text */}
            <div>
              <div className="flex flex-wrap items-center gap-2 mb-5">
                <span className="px-3 py-1 rounded-full text-xs font-bold bg-baltic_blue/10 dark:bg-baltic_blue-900/60 text-baltic_blue dark:text-baltic_blue-700 border border-baltic_blue/20">
                  Clinical Metamorphic Safety Architecture
                </span>
                <span className="px-3 py-1 rounded-full text-xs font-semibold bg-verdigris/10 text-verdigris dark:text-mint_leaf border border-verdigris/20">
                  ConvNeXt-Base ISIC Engine Active
                </span>
              </div>

              <h1 className="text-4xl md:text-5xl font-black text-baltic_blue dark:text-white leading-tight tracking-tight">
                Aegis MetroHealth<br />
                <span className="text-teal dark:text-baltic_blue-700">Dermatology &amp; Cutaneous Oncology</span>
              </h1>
              <p className="text-xl md:text-2xl font-bold text-slate-600 dark:text-slate-300 mt-2">
                Clinical AI Safety &amp; Screening System
              </p>

              <p className="mt-5 text-base text-slate-600 dark:text-slate-300 leading-relaxed font-medium max-w-xl">
                Next-generation dermatological diagnostic platform equipped with a Hierarchical
                Metamorphic Circuit Breaker. Automated diagnoses are intercepted when perturbation
                instability or Grad-CAM attention shift is detected, protecting patient care through
                specialist triage.
              </p>

              <div className="mt-8 flex flex-wrap gap-3.5">
                <Link to="/portal" className="btn-primary flex items-center gap-2 px-6 py-3.5 text-base font-bold shadow-md">
                  Screen a Skin Lesion <ArrowRight className="w-4 h-4" />
                </Link>
                <Link to="/doctor" className="btn-secondary flex items-center gap-2 px-6 py-3.5 text-base font-semibold shadow-sm">
                  <Stethoscope className="w-4 h-4 text-teal" /> Physician Triage Portal
                </Link>
                <Link to="/calibration" className="btn-secondary flex items-center gap-2 px-5 py-3.5 text-base font-semibold shadow-sm">
                  Calibration Studio
                </Link>
              </div>

              {/* Trust badges */}
              <div className="mt-8 flex items-center gap-6 flex-wrap">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 dark:text-slate-400">
                  <Award className="w-3.5 h-3.5 text-teal" /> ISO 13485 Certified
                </div>
                <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 dark:text-slate-400">
                  <Shield className="w-3.5 h-3.5 text-verdigris" /> HIPAA Compliant
                </div>
                <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 dark:text-slate-400">
                  <Activity className="w-3.5 h-3.5 text-baltic_blue" /> Real-time PyTorch Inference
                </div>
              </div>
            </div>

            {/* Right: Dermatology clinic image */}
            <div className="relative hidden lg:block">
              {/* Glow halo */}
              <div className="absolute -inset-5 rounded-3xl bg-gradient-to-br from-teal/20 via-verdigris/10 to-baltic_blue/20 dark:from-teal/10 dark:via-verdigris/5 dark:to-baltic_blue/10 blur-2xl pointer-events-none" />
              <div className="relative rounded-3xl overflow-hidden shadow-2xl border border-teal/20 dark:border-[#033e56]">
                <img
                  src="/hospital_hero.jpg"
                  alt="Dermatology clinic — physician using dermatoscope with AI heatmap display"
                  className="w-full object-cover object-top"
                  style={{ maxHeight: "520px" }}
                />
                {/* Caption gradient overlay */}
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-[#01151d]/90 via-[#01151d]/40 to-transparent px-6 py-5">
                  <div className="text-white text-xs font-bold">
                    Dept. of Dermatology &amp; Cutaneous Oncology
                  </div>
                  <div className="text-white/70 text-[11px] mt-0.5 leading-relaxed">
                    AI-assisted lesion screening with real-time metamorphic safety verification
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Hospital Metrics Strip */}
      <section className="border-b border-slate-200 dark:border-[#033e56] bg-white dark:bg-[#022a39]/60 shadow-xs">
        <div className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-2 md:grid-cols-4 gap-6">
          {STATS.map(({ icon: Icon, value, label }) => (
            <div key={label} className="flex items-center gap-3.5">
              <div className="w-12 h-12 rounded-xl bg-baltic_blue/10 dark:bg-baltic_blue-900/40 flex items-center justify-center flex-shrink-0 text-baltic_blue dark:text-baltic_blue-700">
                <Icon className="w-6 h-6" />
              </div>
              <div>
                <div className="text-2xl font-black text-slate-900 dark:text-white leading-tight">{value}</div>
                <div className="text-xs text-slate-500 dark:text-slate-400 font-medium">{label}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Circuit Breaker States Architecture */}
      <section className="max-w-7xl mx-auto px-6 py-16">
        <div className="text-center mb-12">
          <div className="section-label mb-1.5">Fail-Safe Triage Mechanism</div>
          <h2 className="text-3xl font-extrabold text-baltic_blue dark:text-white">Three-State Circuit Breaker Framework</h2>
          <p className="mt-2 text-slate-600 dark:text-slate-400 max-w-2xl mx-auto text-sm leading-relaxed">
            The platform computes the Composite Brittleness Index (CBI) through unskewed hierarchical normalization across 3 transformation families:
            Geometric, Photometric, and Sensor/Noise.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {STATE_EXPLAINERS.map(({ icon: Icon, color, bg, title, desc }) => (
            <div key={title} className={`card p-6 border shadow-sm ${bg} hover:shadow-md transition-all`}>
              <Icon className={`w-9 h-9 ${color} mb-4`} />
              <h3 className={`font-bold text-lg mb-2.5 ${color}`}>{title}</h3>
              <p className="text-slate-600 dark:text-slate-300 text-sm leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Hospital Clinical Departments */}
      <section className="border-t border-slate-200 dark:border-[#033e56] bg-slate-50 dark:bg-[#01151d]/50 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-between mb-8">
            <div>
              <div className="section-label mb-1">Clinical Specialties</div>
              <h2 className="text-2xl font-bold text-baltic_blue dark:text-white">Comprehensive Cutaneous Oncology Center</h2>
            </div>
            <Link to="/portal" className="text-xs font-bold text-teal dark:text-mint_leaf hover:underline flex items-center gap-1">
              Screening Department <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {DEPARTMENTS.map(({ icon: Icon, label, color, desc }) => (
              <div key={label} className="card p-5 hover:shadow-md transition-shadow">
                <div className="w-12 h-12 rounded-xl bg-slate-100 dark:bg-[#033e56] flex items-center justify-center mb-3.5">
                  <Icon className={`w-6 h-6 ${color}`} />
                </div>
                <div className="font-bold text-slate-900 dark:text-white text-base">{label}</div>
                <div className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">{desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="max-w-7xl mx-auto px-6 py-14">
        <div className="rounded-2xl bg-gradient-to-r from-baltic_blue via-teal to-verdigris p-8 md:p-12 text-white shadow-lg flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/20 text-xs font-extrabold uppercase tracking-wide">
              <Shield className="w-3.5 h-3.5" /> Clinical Safety First
            </div>
            <h3 className="text-2xl md:text-3xl font-extrabold leading-snug">Begin AI-Assisted Lesion Evaluation</h3>
            <p className="text-white/90 text-sm leading-relaxed">
              Upload a dermatoscopic photograph or choose one of our verified clinical presets to test the system in action.
            </p>
          </div>
          <div className="flex gap-3 flex-shrink-0">
            <Link to="/portal" className="px-6 py-3 rounded-xl bg-white text-baltic_blue font-bold text-sm shadow-md hover:bg-cream-DEFAULT transition-colors">
              Launch Screening Portal
            </Link>
          </div>
        </div>
      </section>

    </div>
  );
}

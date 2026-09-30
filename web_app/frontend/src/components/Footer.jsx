// src/components/Footer.jsx
import { Activity, Phone, Mail, MapPin } from "lucide-react";
import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-slate-200 dark:border-[#033e56] bg-white dark:bg-[#01151d] transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-6 py-10 grid grid-cols-1 md:grid-cols-3 gap-8">
        <div>
          <div className="flex items-center gap-2 mb-3">
            <div className="w-8 h-8 rounded-lg bg-baltic_blue flex items-center justify-center text-white">
              <Activity className="w-4 h-4" />
            </div>
            <span className="font-extrabold text-baltic_blue dark:text-white text-base">Aegis MetroHealth Hospital</span>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
            Centre for Advanced Dermatology & Cutaneous Oncology. Clinical AI-assisted
            lesion evaluation protected by the Metamorphic Circuit Breaker safety interception framework.
          </p>
        </div>

        <div>
          <div className="text-xs font-bold uppercase tracking-wider text-baltic_blue dark:text-baltic_blue-700 mb-3">
            Navigation & Portals
          </div>
          <div className="flex flex-col gap-2">
            {[
              { to: "/", label: "Hospital Home" },
              { to: "/portal", label: "Patient AI Screening" },
              { to: "/doctor", label: "Physician Review Portal" },
              { to: "/calibration", label: "Safety Calibration Studio" },
              { to: "/cases", label: "Audit Case Registry" },
            ].map(({ to, label }) => (
              <Link
                key={to}
                to={to}
                className="text-sm text-slate-600 dark:text-slate-400 hover:text-baltic_blue dark:hover:text-mint_leaf transition-colors font-medium"
              >
                {label}
              </Link>
            ))}
          </div>
        </div>

        <div>
          <div className="text-xs font-bold uppercase tracking-wider text-baltic_blue dark:text-baltic_blue-700 mb-3">
            Clinical Contact & Triage
          </div>
          <div className="flex flex-col gap-2.5 text-sm text-slate-600 dark:text-slate-400">
            <div className="flex items-center gap-2">
              <Phone className="w-4 h-4 text-teal" /> 1-800-AEGIS-911 (Triage Hotline)
            </div>
            <div className="flex items-center gap-2">
              <Mail className="w-4 h-4 text-teal" /> dermatology-ai@aegishealth.org
            </div>
            <div className="flex items-center gap-2">
              <MapPin className="w-4 h-4 text-teal" /> 1200 Medical Centre Parkway, Pavilion 4
            </div>
          </div>
        </div>
      </div>

      <div className="border-t border-slate-100 dark:border-[#033e56] py-4 text-center text-xs text-slate-500">
        &copy; 2026 Aegis MetroHealth Hospital · Metamorphic Circuit Breaker Framework · ConvNeXt-Base on ISIC 2019
      </div>
    </footer>
  );
}

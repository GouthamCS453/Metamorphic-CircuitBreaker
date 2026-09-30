// src/components/Navbar.jsx
import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  Activity, Shield, Stethoscope, UserCog, ChevronDown,
  Heart, Phone, Clock, Menu, X, Sun, Moon
} from "lucide-react";
import { useRole, ROLES } from "../context/RoleContext";
import { useTheme } from "../context/ThemeContext";

const ROLE_LABELS = {
  [ROLES.PATIENT]: { label: "Patient View", icon: Heart, color: "text-teal-500 dark:text-teal-400" },
  [ROLES.DOCTOR]:  { label: "Dr. Sarah Jenkins, MD", icon: Stethoscope, color: "text-baltic_blue dark:text-baltic_blue-700" },
  [ROLES.ADMIN]:   { label: "Safety Admin", icon: UserCog, color: "text-amber-600 dark:text-amber-400" },
};

const NAV_LINKS = [
  { to: "/", label: "Hospital Home" },
  { to: "/portal", label: "Patient Screening" },
  { to: "/doctor", label: "Physician Portal" },
  { to: "/calibration", label: "Calibration Studio" },
  { to: "/cases", label: "Case Registry" },
];

export default function Navbar() {
  const { role, setRole } = useRole();
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [roleOpen, setRoleOpen] = useState(false);
  const current = ROLE_LABELS[role];
  const Icon = current.icon;

  const handleRoleSwitch = (newRole) => {
    setRole(newRole);
    setRoleOpen(false);
    // If switching to patient while on the doctor-only page, redirect to portal
    if (newRole === ROLES.PATIENT && location.pathname === "/doctor") {
      navigate("/portal");
    }
    // If switching to doctor, go to the doctor portal
    if (newRole === ROLES.DOCTOR && location.pathname !== "/doctor") {
      navigate("/doctor");
    }
  };

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200 dark:border-[#033e56] bg-white/95 dark:bg-[#01151d]/95 backdrop-blur-md shadow-sm transition-colors duration-200">
      {/* Top Clinical Announcement Bar */}
      <div className="bg-baltic_blue dark:bg-[#022a39] text-white px-4 py-1.5 text-center text-xs flex items-center justify-center gap-6">
        <span className="flex items-center gap-1.5"><Phone className="w-3.5 h-3.5 text-mint_leaf" /> Emergency Triage: 1-800-AEGIS-911</span>
        <span className="hidden sm:flex items-center gap-1.5"><Clock className="w-3.5 h-3.5 text-mint_leaf" /> Dermatology & Cutaneous Oncology Unit · Open 24/7</span>
        <span className="hidden md:inline text-cream-DEFAULT">Aegis MetroHealth Hospital · Clinical AI Metamorphic Circuit Breaker System</span>
      </div>

      {/* Main navigation */}
      <div className="max-w-7xl mx-auto flex items-center justify-between px-4 md:px-6 h-16">
        {/* Hospital Branding */}
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-baltic_blue flex items-center justify-center shadow-md group-hover:bg-teal transition-colors">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="text-base font-extrabold text-baltic_blue dark:text-white leading-none">Aegis MetroHealth</div>
            <div className="text-[11px] text-teal dark:text-slate-400 font-medium leading-none mt-1">Dermatology AI Safety Platform</div>
          </div>
        </Link>

        {/* Desktop links */}
        <nav className="hidden lg:flex items-center gap-1.5">
          {NAV_LINKS.map(({ to, label }) => {
            const active = location.pathname === to;
            return (
              <Link
                key={to}
                to={to}
                className={`px-3.5 py-2 rounded-lg text-sm font-semibold transition-all ${
                  active
                    ? "bg-baltic_blue-900 text-baltic_blue dark:bg-[#022a39] dark:text-baltic_blue-700 shadow-sm"
                    : "text-slate-600 hover:text-baltic_blue hover:bg-slate-100 dark:text-slate-300 dark:hover:text-white dark:hover:bg-[#022a39]"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Right side controls */}
        <div className="flex items-center gap-2.5">
          {/* Light / Dark Mode Toggle */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className="p-2 rounded-lg border border-slate-200 dark:border-[#033e56] bg-slate-100 dark:bg-[#022a39] text-slate-600 dark:text-slate-300 hover:text-baltic_blue dark:hover:text-white transition-colors"
            title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {theme === "dark" ? <Sun className="w-4 h-4 text-cream-DEFAULT" /> : <Moon className="w-4 h-4 text-baltic_blue" />}
          </button>

          {/* Role selector for demo switching */}
          <div className="relative">
            <button
              onClick={() => setRoleOpen(!roleOpen)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-[#033e56] bg-white dark:bg-[#022a39] hover:bg-slate-50 dark:hover:bg-[#033e56] text-sm font-semibold shadow-sm transition-all"
            >
              <Icon className={`w-4 h-4 ${current.color}`} />
              <span className={`hidden sm:block ${current.color}`}>{current.label}</span>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
            </button>
            {roleOpen && (
              <div className="absolute right-0 mt-2 w-64 rounded-xl border border-slate-200 dark:border-[#033e56] bg-white dark:bg-[#022a39] shadow-xl overflow-hidden z-50">
                <div className="px-3.5 py-2.5 border-b border-slate-100 dark:border-[#033e56] text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Switch Active Role
                </div>
                {Object.entries(ROLE_LABELS).map(([r, { label, icon: RIcon, color }]) => (
                  <button
                    key={r}
                    onClick={() => handleRoleSwitch(r)}
                    className={`w-full flex items-center gap-3 px-4 py-3 text-sm hover:bg-slate-50 dark:hover:bg-[#033e56] transition-colors ${
                      r === role ? "bg-slate-100 dark:bg-[#033e56]/80 font-bold" : "font-medium"
                    }`}
                  >
                    <RIcon className={`w-4 h-4 ${color}`} />
                    <span className={color}>{label}</span>
                    {r === role && <span className="ml-auto text-xs text-baltic_blue dark:text-mint_leaf font-semibold">Active</span>}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Circuit Breaker Status Indicator */}
          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-verdigris-900/30 dark:bg-[#022a39] border border-verdigris/30 dark:border-verdigris/40 text-xs font-semibold text-verdigris dark:text-mint_leaf">
            <Shield className="w-3.5 h-3.5 text-verdigris" />
            <span className="hidden sm:block">CB Online</span>
          </div>

          {/* Mobile menu toggle */}
          <button
            className="lg:hidden p-2 rounded-lg text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile nav drawer */}
      {mobileOpen && (
        <div className="lg:hidden border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-[#01151d] px-4 py-3 flex flex-col gap-1.5 shadow-lg">
          {NAV_LINKS.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              onClick={() => setMobileOpen(false)}
              className={`px-3 py-2.5 rounded-lg text-sm font-semibold transition-colors ${
                location.pathname === to
                  ? "bg-baltic_blue text-white"
                  : "text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800"
              }`}
            >
              {label}
            </Link>
          ))}
        </div>
      )}
    </header>
  );
}

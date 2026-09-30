// src/pages/CaseRegistryPage.jsx
import { useState, useEffect } from "react";
import { fetchCases } from "../api/client";
import { Loader2, Search, Database } from "lucide-react";
import StateBadge from "../components/StateBadge";

export default function CaseRegistryPage() {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");

  useEffect(() => {
    fetchCases()
      .then(r => setCases(r.data))
      .finally(() => setLoading(false));
  }, []);

  const filtered = cases.filter(c =>
    c.case_id.includes(query.toUpperCase()) ||
    c.baseline_label?.includes(query.toUpperCase()) ||
    c.state?.includes(query.toUpperCase())
  );

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="section-label mb-1">Clinical Audit & Quality Assurance</div>
          <h1 className="text-3xl font-extrabold text-baltic_blue dark:text-white flex items-center gap-2.5">
            <Database className="w-8 h-8 text-teal" /> Screening Case Registry
          </h1>
          <p className="text-slate-600 dark:text-slate-400 text-sm mt-1">
            Complete audit trail of all AI skin lesion evaluations, circuit breaker transitions, and physician overrides.
          </p>
        </div>

        <div className="relative">
          <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search Case ID, State, Label..."
            className="pl-9 pr-4 py-2 bg-white dark:bg-[#022a39] border border-slate-200 dark:border-[#033e56] rounded-xl text-sm text-slate-800 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:border-baltic_blue w-72 shadow-xs"
          />
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-20">
          <Loader2 className="w-8 h-8 spinner text-baltic_blue" />
        </div>
      ) : filtered.length === 0 ? (
        <div className="card p-12 text-center text-slate-500">
          No cases match your query. Try screening an image in the Patient Portal.
        </div>
      ) : (
        <div className="card overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead>
                <tr className="border-b border-slate-200 dark:border-[#033e56] bg-slate-50 dark:bg-[#01151d] text-slate-600 dark:text-slate-400">
                  {["Case ID", "Timestamp", "Circuit State", "AI Prediction", "Confidence", "CBI Score", "Flips", "Clinical Outcome"].map(h => (
                    <th key={h} className="py-3 px-4 text-xs font-bold uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-[#033e56]">
                {filtered.map((c, i) => (
                  <tr
                    key={c.case_id}
                    className={`hover:bg-slate-50/80 dark:hover:bg-[#022a39]/60 transition-colors ${
                      i % 2 === 0 ? "bg-white dark:bg-[#022a39]/30" : "bg-slate-50/40 dark:bg-transparent"
                    }`}
                  >
                    <td className="py-3.5 px-4 font-mono text-xs font-bold text-slate-800 dark:text-slate-200">{c.case_id}</td>
                    <td className="py-3.5 px-4 text-xs text-slate-500 dark:text-slate-400">
                      {new Date(c.created_at).toLocaleDateString()} {new Date(c.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </td>
                    <td className="py-3.5 px-4"><StateBadge state={c.state} status={c.status} /></td>
                    <td className="py-3.5 px-4 font-extrabold text-baltic_blue dark:text-white">{c.baseline_label}</td>
                    <td className="py-3.5 px-4 font-mono text-xs text-slate-700 dark:text-slate-300">{(c.baseline_confidence * 100).toFixed(1)}%</td>
                    <td className="py-3.5 px-4 font-mono text-xs font-semibold text-slate-700 dark:text-slate-300">{c.cbi.toFixed(4)}</td>
                    <td className="py-3.5 px-4 text-xs text-slate-600 dark:text-slate-400 font-semibold">{c.n_flips}</td>
                    <td className="py-3.5 px-4 text-xs">
                      {c.override_diagnosis ? (
                        <span className="font-bold text-baltic_blue dark:text-teal">
                          Override: {c.override_diagnosis}
                        </span>
                      ) : c.doctor_name ? (
                        <span className="text-verdigris font-semibold">
                          Verified by {c.doctor_name.split(",")[0]}
                        </span>
                      ) : (
                        <span className="text-slate-500 capitalize">{c.status?.replace("_", " ")}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

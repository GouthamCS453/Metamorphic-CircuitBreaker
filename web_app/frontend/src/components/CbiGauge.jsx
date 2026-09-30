// src/components/CbiGauge.jsx
// Visual CBI gauge bar with threshold markers
export default function CbiGauge({ cbi, thetaWarn, thetaTrip }) {
  const pct = Math.round(cbi * 100);
  const warnPct = Math.round(thetaWarn * 100);
  const tripPct = Math.round(thetaTrip * 100);

  const color =
    cbi < thetaWarn ? "#00a896" : cbi < thetaTrip ? "#d97706" : "#dc2626";

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center text-xs text-slate-600 dark:text-slate-400">
        <span>
          CBI = <span className="font-extrabold text-sm" style={{ color }}>{cbi.toFixed(4)}</span>
        </span>
        <span className="font-mono text-xs text-slate-500">
          theta_warn={thetaWarn.toFixed(2)} | theta_trip={thetaTrip.toFixed(2)}
        </span>
      </div>

      <div className="relative h-6 rounded-full overflow-hidden bg-slate-200 dark:bg-[#01151d] border border-slate-300 dark:border-[#033e56] shadow-inner">
        {/* Zone backgrounds */}
        <div
          className="absolute inset-y-0 left-0 bg-verdigris/25 dark:bg-verdigris-900/40"
          style={{ width: `${warnPct}%` }}
        />
        <div
          className="absolute inset-y-0 bg-amber-200/50 dark:bg-amber-900/40"
          style={{ left: `${warnPct}%`, width: `${tripPct - warnPct}%` }}
        />
        <div
          className="absolute inset-y-0 right-0 bg-red-200/50 dark:bg-red-900/40"
          style={{ left: `${tripPct}%` }}
        />

        {/* Threshold lines */}
        <div
          className="absolute inset-y-0 w-0.5 bg-amber-600 z-10"
          style={{ left: `${warnPct}%` }}
          title={`theta_warn = ${thetaWarn}`}
        />
        <div
          className="absolute inset-y-0 w-0.5 bg-red-600 z-10"
          style={{ left: `${tripPct}%` }}
          title={`theta_trip = ${thetaTrip}`}
        />

        {/* Value marker */}
        <div
          className="absolute inset-y-0 w-2 rounded-full shadow-md transition-all duration-500 z-20"
          style={{
            left: `calc(${Math.min(pct, 98)}% - 4px)`,
            backgroundColor: color
          }}
        />
      </div>

      <div className="flex justify-between text-[11px] font-semibold text-slate-500 dark:text-slate-400">
        <span className="text-verdigris">0.0 (CLOSED - Safe)</span>
        <span className="text-amber-600 dark:text-amber-400">HALF-OPEN (Warning)</span>
        <span className="text-red-600 dark:text-red-400">OPEN (Tripped) 1.0</span>
      </div>
    </div>
  );
}

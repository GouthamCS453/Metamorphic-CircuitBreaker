// src/components/StateBadge.jsx
export default function StateBadge({ state, status }) {
  if (status === "overridden") return <span className="badge-override">Doctor Overridden</span>;
  if (status === "approved")   return <span className="badge-approved">Doctor Verified</span>;
  if (state === "CLOSED")      return <span className="badge-closed">Auto-Approved</span>;
  if (state === "HALF_OPEN")   return <span className="badge-halfopen">Under Review</span>;
  if (state === "OPEN")        return <span className="badge-open">Tripped — Blocked</span>;
  return <span className="badge-closed">{state}</span>;
}

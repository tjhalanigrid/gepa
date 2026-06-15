import { useEffect, useMemo, useState } from "react";
import { ClipboardCheck, Plus, ArrowLeft, Trash2, AlertTriangle, CheckCircle, Clock, Search, TrendingUp, Zap } from "lucide-react";
import { AnnotatedImage } from "../inspections/AnnotatedImage";
import { formatCostRange } from "../../lib/api";
import {
  getClaims,
  deleteClaim,
  subscribeClaims,
  claimSeverity,
  claimConfidence,
  relativeTime,
  type Claim,
  type ClaimStatus,
  type SeverityLevel,
} from "../../lib/claimsStore";
import { getVehicles, subscribeVehicles } from "../../lib/vehiclesStore";
import type { VehicleRegistration } from "../../types/vehicle";

interface ClaimsPageProps {
  onNavigate: (page: string) => void;
  activeVehicle: VehicleRegistration | null;
}

const STATUS_META: Record<ClaimStatus, { label: string; bg: string; fg: string; icon: React.ReactNode }> = {
  "auto-approved": { label: "Auto-Approved", bg: "#d1fae5", fg: "#065f46", icon: <CheckCircle size={12} /> },
  "pending-review": { label: "Pending Review", bg: "#fef9c3", fg: "#854d0e", icon: <Clock size={12} /> },
  "no-damage": { label: "No Damage", bg: "#f0f0eb", fg: "#666660", icon: <AlertTriangle size={12} /> },
};

const SEVERITY_META: Record<SeverityLevel, { bg: string; fg: string }> = {
  Low: { bg: "#d1fae5", fg: "#047857" },
  Med: { bg: "#fff7ed", fg: "#c2410c" },
  High: { bg: "#fee2e2", fg: "#b91c1c" },
};

function StatusBadge({ status }: { status: ClaimStatus }) {
  const m = STATUS_META[status];
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", background: m.bg, color: m.fg, fontSize: "10px", fontWeight: 700, padding: "3px 8px", borderRadius: "20px", whiteSpace: "nowrap" }}>
      {m.icon} {m.label}
    </span>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const color = value >= 90 ? "#10b981" : value >= 75 ? "#6366f1" : "#f97316";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <span style={{ fontSize: "12px", color: "#0a0a0a", fontWeight: 600, minWidth: "34px" }}>{value > 0 ? `${value}%` : "--"}</span>
      <div style={{ width: "70px", height: "5px", background: "#ececea", borderRadius: "3px", overflow: "hidden" }}>
        <div style={{ width: `${value}%`, height: "100%", background: value > 0 ? color : "#d4d4d0", borderRadius: "3px" }} />
      </div>
    </div>
  );
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function fmtAvg(seconds: number): string {
  if (!seconds) return "--";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  return `${(seconds / 60).toFixed(1)} min`;
}

export function ClaimsPage({ onNavigate, activeVehicle }: ClaimsPageProps) {
  const [claims, setClaims] = useState<Claim[]>([]);
  const [vehicles, setVehicles] = useState<VehicleRegistration[]>(getVehicles());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<ClaimStatus | "all">("all");
  const [vehicleFilter, setVehicleFilter] = useState<string>(activeVehicle?.id ?? "all");

  useEffect(() => {
    const load = () => setClaims(getClaims());
    load();
    return subscribeClaims(load);
  }, []);

  useEffect(() => {
    const load = () => setVehicles(getVehicles());
    load();
    return subscribeVehicles(load);
  }, []);

  // When the user switches the active vehicle, scope the dashboard to it.
  useEffect(() => {
    if (activeVehicle) setVehicleFilter(activeVehicle.id);
  }, [activeVehicle?.id]);

  const selected = claims.find((c) => c.id === selectedId) ?? null;

  // Vehicle scope drives both the stat cards and the table.
  const scoped = useMemo(
    () => (vehicleFilter === "all" ? claims : claims.filter((c) => c.vehicleId === vehicleFilter)),
    [claims, vehicleFilter],
  );

  const stats = useMemo(() => {
    const total = scoped.length;
    const auto = scoped.filter((c) => c.status === "auto-approved").length;
    const pending = scoped.filter((c) => c.status === "pending-review").length;
    const timed = scoped.filter((c) => c.inferenceS);
    const avg = timed.length ? timed.reduce((s, c) => s + (c.inferenceS ?? 0), 0) / timed.length : 0;
    return { total, auto, pending, rate: total ? Math.round((auto / total) * 100) : 0, avg };
  }, [scoped]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return scoped.filter((c) => {
      if (statusFilter !== "all" && c.status !== statusFilter) return false;
      if (!q) return true;
      return (
        c.id.toLowerCase().includes(q) ||
        c.vehicle.toLowerCase().includes(q) ||
        c.findings.some((f) => f.damage.toLowerCase().includes(q) || f.area.toLowerCase().includes(q))
      );
    });
  }, [scoped, query, statusFilter]);

  // ── Detail view ──────────────────────────────────────────────────────────
  if (selected) {
    return (
      <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "18px" }}>
        <button
          onClick={() => setSelectedId(null)}
          style={{ alignSelf: "flex-start", display: "flex", alignItems: "center", gap: "6px", background: "#ffffff", border: "1px solid rgba(0,0,0,0.08)", borderRadius: "8px", padding: "8px 14px", fontSize: "12px", fontWeight: 600, color: "#444440", cursor: "pointer" }}
        >
          <ArrowLeft size={14} /> All claims
        </button>

        <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "28px" }}>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "20px", flexWrap: "wrap", gap: "12px" }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px" }}>
                <h2 style={{ fontSize: "20px", fontWeight: 800, color: "#0a0a0a" }}>{selected.id}</h2>
                <StatusBadge status={selected.status} />
              </div>
              <p style={{ fontSize: "12px", color: "#888882" }}>{selected.vehicle} · {fmtDate(selected.createdAt)}</p>
            </div>
            <button
              onClick={() => { deleteClaim(selected.id); setSelectedId(null); }}
              style={{ display: "flex", alignItems: "center", gap: "6px", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: "8px", padding: "8px 14px", fontSize: "12px", fontWeight: 600, color: "#b91c1c", cursor: "pointer" }}
            >
              <Trash2 size={13} /> Delete
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: selected.detections.length ? "1.6fr 1fr" : "1fr", gap: "20px", alignItems: "start" }}>
            <AnnotatedImage src={selected.thumbnail} detections={selected.detections} naturalSize={{ w: selected.imgW, h: selected.imgH }} />

            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              <div style={{ fontSize: "11px", fontWeight: 700, color: "#0a0a0a", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Findings ({selected.findings.length})
              </div>
              {selected.findings.length === 0 ? (
                <p style={{ fontSize: "12px", color: "#888882" }}>No damage was detected.</p>
              ) : (
                selected.findings.map((f, i) => (
                  <div key={i} style={{ padding: "10px 12px", background: "#fafafa", borderRadius: "10px", border: "1px solid rgba(0,0,0,0.05)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "2px" }}>
                      <span style={{ fontSize: "12px", fontWeight: 600, color: "#0a0a0a" }}>{f.area}</span>
                      <span style={{ fontSize: "9px", fontWeight: 700, color: f.severity === "Severe" ? "#b91c1c" : f.severity === "Moderate" ? "#c2410c" : "#854d0e", background: f.severity === "Severe" ? "#fee2e2" : f.severity === "Moderate" ? "#fff7ed" : "#fef9c3", padding: "1px 6px", borderRadius: "4px" }}>
                        {f.severity.toUpperCase()}
                      </span>
                    </div>
                    <div style={{ fontSize: "11px", color: "#666660" }}>{f.damage} — {formatCostRange(f.costMin, f.costMax)}</div>
                  </div>
                ))
              )}

              <div style={{ marginTop: "8px", padding: "12px 16px", background: "#fffbeb", border: "1px solid var(--accent)", borderRadius: "8px" }}>
                <div style={{ fontSize: "11px", fontWeight: 600, color: "#0a0a0a", marginBottom: "2px" }}>Estimated Total Repair Cost</div>
                <div style={{ fontSize: "20px", fontWeight: 800, color: "#0a0a0a" }}>{formatCostRange(selected.totalMin, selected.totalMax)}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Dashboard (list) view ─────────────────────────────────────────────────
  const statCards = [
    { label: "TOTAL CLAIMS", value: String(stats.total), sub: "from inspections", subColor: "#888882", icon: <ClipboardCheck size={16} /> },
    { label: "AUTO-APPROVED", value: String(stats.auto), sub: `${stats.rate}% rate`, subColor: "#10b981", icon: <CheckCircle size={16} /> },
    { label: "AVG ASSESSMENT", value: fmtAvg(stats.avg), sub: "per claim", subColor: "#888882", icon: <Zap size={16} /> },
    { label: "PENDING REVIEW", value: String(stats.pending), sub: `${stats.pending} escalated`, subColor: stats.pending ? "#ef4444" : "#888882", icon: <Clock size={16} /> },
  ];

  return (
    <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <h1 style={{ fontSize: "26px", fontWeight: 800, color: "#0a0a0a", letterSpacing: "-0.02em" }}>Claims Dashboard</h1>
          <p style={{ fontSize: "13px", color: "#888882", marginTop: "4px" }}>
            {(() => {
              if (vehicleFilter === "all") return "Real-time overview of all claims across every vehicle.";
              const v = vehicles.find((x) => x.id === vehicleFilter);
              return v ? `Showing claims for ${v.year} ${v.make} ${v.model}.` : "Real-time overview of claims.";
            })()}
          </p>
        </div>
        <button
          onClick={() => onNavigate("inspections")}
          style={{ display: "flex", alignItems: "center", gap: "6px", background: "var(--accent)", color: "#0a0a0a", border: "none", borderRadius: "8px", padding: "10px 18px", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}
        >
          <Plus size={15} /> New Inspection
        </button>
      </div>

      {/* Stat cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "16px" }}>
        {statCards.map((s) => (
          <div key={s.label} style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "20px", boxShadow: "0 1px 4px rgba(0,0,0,0.04)" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
              <span style={{ fontSize: "11px", fontWeight: 700, color: "#888882", letterSpacing: "0.06em" }}>{s.label}</span>
              <div style={{ width: "30px", height: "30px", background: "#f5f5f0", borderRadius: "8px", display: "flex", alignItems: "center", justifyContent: "center", color: "#0a0a0a" }}>{s.icon}</div>
            </div>
            <div style={{ fontSize: "30px", fontWeight: 800, color: "#0a0a0a", lineHeight: 1 }}>{s.value}</div>
            <div style={{ fontSize: "12px", color: s.subColor, fontWeight: 600, marginTop: "8px", display: "flex", alignItems: "center", gap: "4px" }}>
              {s.subColor === "#10b981" && <TrendingUp size={12} />}{s.sub}
            </div>
          </div>
        ))}
      </div>

      {/* Search + filter */}
      {claims.length > 0 && (
        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
          <div style={{ position: "relative", flex: 1, minWidth: "240px", maxWidth: "520px" }}>
            <Search size={16} color="#aaaaaa" style={{ position: "absolute", left: "14px", top: "50%", transform: "translateY(-50%)" }} />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search claims by ID, vehicle, or damage…"
              style={{ width: "100%", padding: "12px 14px 12px 38px", fontSize: "13px", border: "1px solid rgba(0,0,0,0.1)", borderRadius: "10px", outline: "none", background: "#ffffff", color: "#0a0a0a", boxSizing: "border-box" }}
            />
          </div>
          <select
            value={vehicleFilter}
            onChange={(e) => setVehicleFilter(e.target.value)}
            style={{ padding: "12px 14px", fontSize: "13px", border: "1px solid rgba(0,0,0,0.1)", borderRadius: "10px", background: "#ffffff", color: "#444440", cursor: "pointer", outline: "none", maxWidth: "220px" }}
          >
            <option value="all">All vehicles</option>
            {vehicles.map((v) => (
              <option key={v.id} value={v.id}>{v.year} {v.make} {v.model}</option>
            ))}
          </select>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as ClaimStatus | "all")}
            style={{ padding: "12px 14px", fontSize: "13px", border: "1px solid rgba(0,0,0,0.1)", borderRadius: "10px", background: "#ffffff", color: "#444440", cursor: "pointer", outline: "none" }}
          >
            <option value="all">All statuses</option>
            <option value="auto-approved">Auto-Approved</option>
            <option value="pending-review">Pending Review</option>
            <option value="no-damage">No Damage</option>
          </select>
        </div>
      )}

      {/* Empty state */}
      {claims.length === 0 ? (
        <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px dashed rgba(0,0,0,0.12)", padding: "56px 24px", textAlign: "center" }}>
          <div style={{ width: "56px", height: "56px", background: "var(--accent)", borderRadius: "16px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
            <ClipboardCheck size={26} color="#0a0a0a" />
          </div>
          <h3 style={{ fontSize: "15px", fontWeight: 700, color: "#0a0a0a", marginBottom: "6px" }}>No claims yet</h3>
          <p style={{ fontSize: "12px", color: "#888882", marginBottom: "20px", maxWidth: "320px", margin: "0 auto 20px" }}>
            Run an AI inspection on a vehicle photo and the result will appear here as a claim.
          </p>
          <button
            onClick={() => onNavigate("inspections")}
            style={{ display: "inline-flex", alignItems: "center", gap: "6px", background: "#0a0a0a", color: "var(--accent)", border: "none", borderRadius: "8px", padding: "10px 20px", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}
          >
            <Plus size={15} /> Start an inspection
          </button>
        </div>
      ) : (
        <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", overflow: "hidden" }}>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "820px" }}>
              <thead>
                <tr style={{ background: "#fafafa" }}>
                  {["CLAIM ID", "VEHICLE", "STATUS", "SEVERITY", "ESTIMATE", "CONFIDENCE", "SUBMITTED"].map((h) => (
                    <th key={h} style={{ padding: "12px 20px", textAlign: "left", fontSize: "10px", fontWeight: 700, color: "#888882", letterSpacing: "0.06em", borderBottom: "1px solid rgba(0,0,0,0.06)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((c, i) => {
                  const sev = claimSeverity(c);
                  const conf = claimConfidence(c);
                  return (
                    <tr
                      key={c.id}
                      onClick={() => setSelectedId(c.id)}
                      style={{ borderBottom: i < filtered.length - 1 ? "1px solid rgba(0,0,0,0.04)" : "none", cursor: "pointer" }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = "#fafafa")}
                      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                    >
                      <td style={{ padding: "14px 20px" }}>
                        <span style={{ fontSize: "13px", fontWeight: 700, color: "#2563eb" }}>{c.id}</span>
                      </td>
                      <td style={{ padding: "14px 20px", fontSize: "13px", color: "#0a0a0a", whiteSpace: "nowrap" }}>{c.vehicle}</td>
                      <td style={{ padding: "14px 20px" }}><StatusBadge status={c.status} /></td>
                      <td style={{ padding: "14px 20px" }}>
                        {sev ? (
                          <span style={{ fontSize: "11px", fontWeight: 700, color: SEVERITY_META[sev].fg, background: SEVERITY_META[sev].bg, padding: "3px 10px", borderRadius: "20px" }}>{sev}</span>
                        ) : (
                          <span style={{ fontSize: "12px", color: "#aaaaaa" }}>--</span>
                        )}
                      </td>
                      <td style={{ padding: "14px 20px", fontSize: "13px", fontWeight: 600, color: "#0a0a0a", whiteSpace: "nowrap" }}>
                        {c.totalMin || c.totalMax ? formatCostRange(c.totalMin, c.totalMax) : "--"}
                      </td>
                      <td style={{ padding: "14px 20px" }}><ConfidenceBar value={conf} /></td>
                      <td style={{ padding: "14px 20px", fontSize: "12px", color: "#888882", whiteSpace: "nowrap" }}>{relativeTime(c.createdAt)}</td>
                    </tr>
                  );
                })}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={7} style={{ padding: "32px", textAlign: "center", fontSize: "12px", color: "#888882" }}>
                      {vehicleFilter !== "all" ? "No claims for this vehicle yet." : "No claims match your search."}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

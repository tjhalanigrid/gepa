import { useEffect, useState } from "react";
import { Car, Clock, CheckCircle, AlertTriangle, ArrowRight } from "lucide-react";
import type { VehicleRegistration, VehicleStatus } from "../../types/vehicle";
import { formatCostRange } from "../../lib/api";
import {
  getClaims,
  subscribeClaims,
  claimSeverity,
  relativeTime,
  type Claim,
  type ClaimStatus,
} from "../../lib/claimsStore";

const statusColors: Record<VehicleStatus, { bg: string; text: string; dot: string }> = {
  active: { bg: "#d1fae5", text: "#065f46", dot: "#10b981" },
  inactive: { bg: "#f3f4f6", text: "#6b7280", dot: "#9ca3af" },
  maintenance: { bg: "#fff7ed", text: "#c2410c", dot: "#f97316" },
  pending: { bg: "#fef9c3", text: "#854d0e", dot: "var(--accent)" },
};

const claimStatusStyle: Record<ClaimStatus, { label: string; bg: string; text: string; dot: string }> = {
  "auto-approved": { label: "Auto-approved", bg: "#d1fae5", text: "#065f46", dot: "#10b981" },
  "pending-review": { label: "Pending review", bg: "#fef9c3", text: "#854d0e", dot: "var(--accent)" },
  "no-damage": { label: "No damage", bg: "#f0f0eb", text: "#666660", dot: "#9ca3af" },
};

const severityColor: Record<string, string> = { Low: "#10b981", Med: "#f97316", High: "#ef4444" };

interface StatCardProps {
  label: string;
  value: number | string;
  icon: React.ReactNode;
  sub?: string;
}

function StatCard({ label, value, icon, sub }: StatCardProps) {
  return (
    <div
      style={{
        background: "#ffffff", borderRadius: "12px", padding: "20px", border: "1px solid rgba(0,0,0,0.06)",
        display: "flex", flexDirection: "column", gap: "10px", boxShadow: "0 1px 4px rgba(0,0,0,0.04)",
        transition: "border-color 0.18s ease", cursor: "default",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
      onMouseLeave={(e) => (e.currentTarget.style.borderColor = "rgba(0,0,0,0.06)")}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ color: "#888882", fontSize: "11px", fontWeight: 500 }}>{label}</span>
        <div style={{ width: "34px", height: "34px", background: "#f5f5f0", borderRadius: "8px", display: "flex", alignItems: "center", justifyContent: "center", color: "#0a0a0a" }}>
          {icon}
        </div>
      </div>
      <div style={{ color: "#0a0a0a", fontSize: "24px", fontWeight: 700, lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: "10px", color: "#888882" }}>{sub}</div>}
    </div>
  );
}

interface DashboardPageProps {
  onNavigate: (page: string) => void;
  activeVehicle: VehicleRegistration | null;
}

export function DashboardPage({ onNavigate, activeVehicle }: DashboardPageProps) {
  const [claims, setClaims] = useState<Claim[]>([]);

  useEffect(() => {
    const load = () => setClaims(getClaims());
    load();
    return subscribeClaims(load);
  }, []);

  if (!activeVehicle) {
    return (
      <div style={{ padding: "24px", display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ width: "64px", height: "64px", background: "#f5f5f0", borderRadius: "16px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
            <Car size={28} color="#888882" />
          </div>
          <h2 style={{ fontSize: "16px", fontWeight: 600, color: "#0a0a0a", marginBottom: "6px" }}>No vehicle selected</h2>
          <p style={{ fontSize: "13px", color: "#888882", marginBottom: "20px" }}>
            Click on the vehicle selector at the bottom-left of the sidebar to choose a vehicle.
          </p>
          <button
            onClick={() => onNavigate("register")}
            style={{ padding: "10px 20px", background: "var(--accent)", color: "#0a0a0a", border: "none", borderRadius: "8px", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}
          >
            + Register a Vehicle
          </button>
        </div>
      </div>
    );
  }

  const sc = statusColors[activeVehicle.status];
  // Scope every metric on this page to the currently selected vehicle.
  const vehicleClaims = claims.filter((c) => c.vehicleId === activeVehicle.id);
  const recent = vehicleClaims.slice(0, 5);
  const sumMin = vehicleClaims.reduce((s, c) => s + c.totalMin, 0);
  const sumMax = vehicleClaims.reduce((s, c) => s + c.totalMax, 0);
  const autoApproved = vehicleClaims.filter((c) => c.status === "auto-approved").length;
  const pending = vehicleClaims.filter((c) => c.status === "pending-review").length;

  return (
    <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Vehicle banner */}
      <div style={{ background: "linear-gradient(135deg, rgb(10, 10, 10) 0%, rgb(26, 26, 26) 100%)", borderRadius: "16px", padding: "42px 48px", display: "flex", alignItems: "center", justifyContent: "space-between", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", right: "32px", top: "-20px", width: "120px", height: "120px", background: "rgb(245, 197, 24)", borderRadius: "50%", opacity: 0.06 }} />
        <div style={{ position: "absolute", right: "80px", bottom: "-30px", width: "80px", height: "80px", background: "rgb(245, 197, 24)", borderRadius: "50%", opacity: 0.08 }} />
        <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
          <div style={{ width: "56px", height: "56px", background: "var(--accent)", borderRadius: "14px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "16px", fontWeight: 800, color: "#0a0a0a" }}>
            {activeVehicle.make.slice(0, 2).toUpperCase()}
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "6px" }}>
              <h2 style={{ color: "#ffffff", fontSize: "18px", fontWeight: 700 }}>
                {activeVehicle.year} {activeVehicle.make} {activeVehicle.model}
              </h2>
              <span style={{ background: sc.bg, color: sc.text, fontSize: "10px", fontWeight: 600, padding: "2px 8px", borderRadius: "8px", display: "inline-flex", alignItems: "center", gap: "4px", textTransform: "capitalize" }}>
                <span style={{ width: "5px", height: "5px", borderRadius: "50%", background: sc.dot }} />
                {activeVehicle.status}
              </span>
            </div>
            <div style={{ display: "flex", gap: "20px" }}>
              {[
                { label: "Plate", val: activeVehicle.licenseplate },
                { label: "VIN", val: activeVehicle.vin.slice(-8) },
                { label: "Owner", val: activeVehicle.ownerName },
              ].map((i) => (
                <span key={i.label} style={{ fontSize: "12px", color: "#888882" }}>
                  <span style={{ color: "#555550" }}>{i.label}: </span>{i.val}
                </span>
              ))}
            </div>
          </div>
        </div>
        <button
          onClick={() => onNavigate("inspections")}
          style={{ padding: "12px 24px", background: "var(--accent)", color: "#0a0a0a", border: "none", borderRadius: "8px", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}
        >
          Start Inspection
        </button>
      </div>

      {/* Stats grid (real) */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "14px" }}>
        <StatCard label="Total Claims" value={vehicleClaims.length} icon={<Car size={16} />} sub={`${activeVehicle.make} ${activeVehicle.model}`} />
        <StatCard label="Auto-Approved" value={autoApproved} icon={<CheckCircle size={16} />} sub={`${vehicleClaims.length ? Math.round((autoApproved / vehicleClaims.length) * 100) : 0}% rate`} />
        <StatCard label="Pending Review" value={pending} icon={<Clock size={16} />} sub="Awaiting decision" />
        <StatCard label="Est. Repair Cost" value={vehicleClaims.length ? formatCostRange(sumMin, sumMax) : "₹0"} icon={<AlertTriangle size={16} />} sub="This vehicle" />
      </div>

      {/* Recent Claims (real) */}
      <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", overflow: "hidden" }}>
        <div style={{ padding: "14px 20px", borderBottom: "1px solid rgba(0,0,0,0.06)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <h3 style={{ fontSize: "13px", fontWeight: 600, color: "#0a0a0a" }}>Recent Claims — {activeVehicle.make} {activeVehicle.model}</h3>
          {vehicleClaims.length > 0 && (
            <button
              onClick={() => onNavigate("claims")}
              style={{ display: "flex", alignItems: "center", gap: "4px", background: "transparent", border: "none", cursor: "pointer", fontSize: "12px", fontWeight: 600, color: "#2563eb" }}
            >
              View all <ArrowRight size={13} />
            </button>
          )}
        </div>

        {recent.length === 0 ? (
          <div style={{ padding: "40px 24px", textAlign: "center" }}>
            <p style={{ fontSize: "13px", color: "#888882", marginBottom: "14px" }}>No claims yet. Run an inspection to create one.</p>
            <button
              onClick={() => onNavigate("inspections")}
              style={{ padding: "9px 18px", background: "var(--accent)", color: "#0a0a0a", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 700, cursor: "pointer" }}
            >
              Start Inspection
            </button>
          </div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "#fafafa" }}>
                {["Claim ID", "Vehicle", "Damage", "Severity", "Est. Cost", "Status", "Submitted"].map((h) => (
                  <th key={h} style={{ padding: "9px 20px", textAlign: "left", fontSize: "10px", fontWeight: 700, color: "#888882", letterSpacing: "0.05em", borderBottom: "1px solid rgba(0,0,0,0.05)" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {recent.map((c, i) => {
                const cs = claimStatusStyle[c.status];
                const sev = claimSeverity(c);
                return (
                  <tr
                    key={c.id}
                    style={{ borderBottom: i < recent.length - 1 ? "1px solid rgba(0,0,0,0.04)" : "none", cursor: "pointer" }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "#fafafa")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                    onClick={() => onNavigate("claims")}
                  >
                    <td style={{ padding: "12px 20px" }}>
                      <span style={{ fontSize: "12px", color: "#2563eb", fontWeight: 700 }}>{c.id}</span>
                    </td>
                    <td style={{ padding: "12px 20px", fontSize: "12px", color: "#0a0a0a" }}>{c.vehicle}</td>
                    <td style={{ padding: "12px 20px", fontSize: "12px", color: "#666660", textTransform: "capitalize" }}>
                      {c.findings[0] ? c.findings[0].damage : "—"}
                    </td>
                    <td style={{ padding: "12px 20px" }}>
                      {sev ? <span style={{ fontSize: "11px", fontWeight: 600, color: severityColor[sev] }}>● {sev}</span> : <span style={{ color: "#aaaaaa", fontSize: "12px" }}>—</span>}
                    </td>
                    <td style={{ padding: "12px 20px", fontSize: "12px", color: "#0a0a0a", fontWeight: 600, whiteSpace: "nowrap" }}>
                      {c.totalMin || c.totalMax ? formatCostRange(c.totalMin, c.totalMax) : "—"}
                    </td>
                    <td style={{ padding: "12px 20px" }}>
                      <span style={{ background: cs.bg, color: cs.text, fontSize: "10px", fontWeight: 600, padding: "3px 8px", borderRadius: "8px", display: "inline-flex", alignItems: "center", gap: "4px" }}>
                        <span style={{ width: "5px", height: "5px", borderRadius: "50%", background: cs.dot }} />
                        {cs.label}
                      </span>
                    </td>
                    <td style={{ padding: "12px 20px", fontSize: "12px", color: "#888882", whiteSpace: "nowrap" }}>{relativeTime(c.createdAt)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

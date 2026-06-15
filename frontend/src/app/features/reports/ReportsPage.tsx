import { useEffect, useMemo, useState } from "react";
import { FileText, CheckCircle, AlertTriangle, Clock, FileDown, FileType } from "lucide-react";
import type { VehicleRegistration } from "../../types/vehicle";
import { AnnotatedImage } from "../inspections/AnnotatedImage";
import { formatCostRange } from "../../lib/api";
import { getClaims, subscribeClaims, type Claim, type ClaimStatus } from "../../lib/claimsStore";
import { getVehicles, subscribeVehicles } from "../../lib/vehiclesStore";
import { exportReportPdf, exportReportWord } from "../../lib/reportExport";

const SECTIONS = ["Executive Summary", "Damage Photo", "Analysis Results", "Cost Breakdown"] as const;
type Section = (typeof SECTIONS)[number];

const STATUS_META: Record<ClaimStatus, { label: string; bg: string; fg: string; icon: React.ReactNode }> = {
  "auto-approved": { label: "Auto-approved", bg: "#d1fae5", fg: "#065f46", icon: <CheckCircle size={11} /> },
  "pending-review": { label: "Pending review", bg: "#fef9c3", fg: "#854d0e", icon: <Clock size={11} /> },
  "no-damage": { label: "No damage", bg: "#f0f0eb", fg: "#666660", icon: <AlertTriangle size={11} /> },
};

function avgConfidence(c: Claim): number {
  const scored = c.findings.filter((f) => f.confidence > 0);
  if (!scored.length) return 0;
  return Math.round(scored.reduce((s, f) => s + f.confidence, 0) / scored.length);
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
}

interface ReportsPageProps {
  activeVehicle: VehicleRegistration | null;
  onNavigate: (page: string) => void;
}

export function ReportsPage({ activeVehicle, onNavigate }: ReportsPageProps) {
  const [claims, setClaims] = useState<Claim[]>([]);
  const [vehicles, setVehicles] = useState<VehicleRegistration[]>(getVehicles());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [section, setSection] = useState<Section>("Executive Summary");
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

  // Scope reports to the active vehicle; re-sync when it changes.
  useEffect(() => {
    if (activeVehicle) setVehicleFilter(activeVehicle.id);
  }, [activeVehicle?.id]);

  const scoped = useMemo(
    () => (vehicleFilter === "all" ? claims : claims.filter((c) => c.vehicleId === vehicleFilter)),
    [claims, vehicleFilter],
  );

  const selected = useMemo(
    () => scoped.find((c) => c.id === selectedId) ?? scoped[0] ?? null,
    [scoped, selectedId],
  );

  return (
    <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "16px" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <h2 style={{ fontSize: "15px", fontWeight: 700, color: "#0a0a0a" }}>Damage Reports</h2>
          <p style={{ fontSize: "11px", color: "#888882", marginTop: "2px" }}>
            {(() => {
              const v = vehicles.find((x) => x.id === vehicleFilter);
              const label = vehicleFilter === "all" || !v ? "All vehicles" : `${v.year} ${v.make} ${v.model}`;
              return `${label} · ${scoped.length} ${scoped.length === 1 ? "report" : "reports"}`;
            })()}
          </p>
        </div>
        <div style={{ display: "flex", gap: "10px" }}>
          <select
            value={vehicleFilter}
            onChange={(e) => { setVehicleFilter(e.target.value); setSelectedId(null); }}
            style={{ padding: "9px 12px", fontSize: "12px", border: "1px solid rgba(0,0,0,0.1)", borderRadius: "8px", background: "#ffffff", color: "#444440", cursor: "pointer", outline: "none", maxWidth: "220px" }}
          >
            <option value="all">All vehicles</option>
            {vehicles.map((v) => (
              <option key={v.id} value={v.id}>{v.year} {v.make} {v.model}</option>
            ))}
          </select>
          <button
            onClick={() => onNavigate("inspections")}
            style={{ padding: "9px 16px", background: "var(--accent)", color: "#0a0a0a", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 700, cursor: "pointer" }}
          >
            + New Inspection
          </button>
        </div>
      </div>

      {scoped.length === 0 || !selected ? (
        <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px dashed rgba(0,0,0,0.12)", padding: "56px 24px", textAlign: "center" }}>
          <div style={{ width: "56px", height: "56px", background: "var(--accent)", borderRadius: "16px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
            <FileText size={26} color="#0a0a0a" />
          </div>
          <h3 style={{ fontSize: "15px", fontWeight: 700, color: "#0a0a0a", marginBottom: "6px" }}>No reports yet</h3>
          <p style={{ fontSize: "12px", color: "#888882", marginBottom: "20px" }}>Run an AI inspection and its report will appear here.</p>
          <button
            onClick={() => onNavigate("inspections")}
            style={{ display: "inline-flex", alignItems: "center", gap: "6px", background: "#0a0a0a", color: "var(--accent)", border: "none", borderRadius: "8px", padding: "10px 20px", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}
          >
            Start an inspection
          </button>
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: "16px", minHeight: "calc(100vh - 200px)" }}>
          {/* Left: report list (real claims) */}
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {scoped.map((r) => {
              const active = selected.id === r.id;
              const m = STATUS_META[r.status];
              return (
                <button
                  key={r.id}
                  onClick={() => setSelectedId(r.id)}
                  style={{
                    display: "flex", flexDirection: "column", gap: "6px", padding: "14px 16px",
                    background: active ? "#fffbeb" : "#ffffff",
                    border: `1px solid ${active ? "var(--accent)" : "rgba(0,0,0,0.06)"}`,
                    borderRadius: "10px", cursor: "pointer", textAlign: "left", width: "100%",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <code style={{ fontSize: "11px", fontWeight: 700, color: active ? "var(--accent)" : "#0a0a0a", background: active ? "rgba(245,197,24,0.12)" : "#f5f5f0", padding: "2px 7px", borderRadius: "4px" }}>
                      {r.id}
                    </code>
                    <span style={{ fontSize: "9px", color: "#888882" }}>
                      {new Date(r.createdAt).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                    </span>
                  </div>
                  <div style={{ fontSize: "12px", fontWeight: 500, color: "#0a0a0a", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.vehicle}</div>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: "3px", fontSize: "9px", fontWeight: 700, color: m.fg, background: m.bg, padding: "2px 6px", borderRadius: "20px" }}>
                      {m.icon} {m.label}
                    </span>
                    <span style={{ fontSize: "11px", fontWeight: 700, color: "#0a0a0a" }}>{formatCostRange(r.totalMin, r.totalMax)}</span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Right: report viewer */}
          <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", overflow: "hidden", display: "flex", flexDirection: "column" }}>
            <div style={{ padding: "14px 20px", borderBottom: "1px solid rgba(0,0,0,0.06)", display: "flex", alignItems: "center", gap: "10px" }}>
              <span style={{ fontSize: "12px", fontWeight: 600, color: "#0a0a0a", flex: 1 }}>
                Report — {selected.id}
              </span>
              <button
                onClick={() => exportReportPdf(selected)}
                title="Export the full report as PDF"
                style={{ display: "inline-flex", alignItems: "center", gap: "6px", padding: "7px 14px", background: "#0a0a0a", color: "var(--accent)", border: "none", borderRadius: "8px", fontSize: "11px", fontWeight: 700, cursor: "pointer" }}
              >
                <FileDown size={13} /> Export PDF
              </button>
              <button
                onClick={() => exportReportWord(selected)}
                title="Export the full report as a Word document"
                style={{ display: "inline-flex", alignItems: "center", gap: "6px", padding: "7px 14px", background: "#f5f5f0", color: "#0a0a0a", border: "1px solid rgba(0,0,0,0.1)", borderRadius: "8px", fontSize: "11px", fontWeight: 700, cursor: "pointer" }}
              >
                <FileType size={13} /> Export Word
              </button>
            </div>

            <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
              {/* Section nav */}
              <div style={{ width: "180px", borderRight: "1px solid rgba(0,0,0,0.06)", padding: "16px 0", flexShrink: 0 }}>
                <div style={{ fontSize: "9px", fontWeight: 700, color: "#888882", letterSpacing: "0.08em", padding: "0 16px", marginBottom: "8px" }}>REPORT SECTIONS</div>
                {SECTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => setSection(s)}
                    style={{
                      width: "100%", display: "flex", alignItems: "center", gap: "8px",
                      padding: "8px 16px", background: section === s ? "#fffbeb" : "transparent",
                      border: "none", cursor: "pointer", fontSize: "11px",
                      color: section === s ? "#0a0a0a" : "#888882",
                      fontWeight: section === s ? 600 : 400,
                      borderLeft: `2px solid ${section === s ? "var(--accent)" : "transparent"}`,
                      textAlign: "left",
                    }}
                  >
                    {section === s ? <CheckCircle size={11} color="var(--accent)" /> : <span style={{ width: 11 }} />}
                    {s}
                  </button>
                ))}
              </div>

              {/* Content */}
              <div style={{ flex: 1, padding: "24px 28px", overflowY: "auto" }}>
                <div style={{ textAlign: "center", marginBottom: "24px", paddingBottom: "20px", borderBottom: "2px solid rgba(0,0,0,0.06)" }}>
                  <div style={{ fontSize: "10px", fontWeight: 700, color: "var(--accent)", letterSpacing: "0.1em", marginBottom: "8px" }}>VEHICLE DAMAGE ASSESSMENT REPORT</div>
                  <h2 style={{ fontSize: "20px", fontWeight: 800, color: "#0a0a0a", marginBottom: "4px" }}>
                    {selected.id} | {selected.vehicle}
                  </h2>
                  <p style={{ fontSize: "11px", color: "#888882" }}>
                    Generated {fmtDate(selected.createdAt)}
                    {avgConfidence(selected) > 0 ? ` | AI Confidence: ${avgConfidence(selected)}%` : ""}
                  </p>
                </div>

                {section === "Executive Summary" && (
                  <div>
                    <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#0a0a0a", marginBottom: "12px" }}>Executive Summary</h3>
                    <p style={{ fontSize: "13px", color: "#444440", lineHeight: 1.8, marginBottom: "16px" }}>
                      This report presents an AI-assisted damage assessment for a {selected.vehicle}.{" "}
                      {selected.findings.length} damaged component{selected.findings.length === 1 ? " was" : "s were"} identified, with an estimated total
                      repair cost of <strong>{formatCostRange(selected.totalMin, selected.totalMax)}</strong>. The claim was{" "}
                      {selected.status === "pending-review" ? "flagged for human review" : selected.status === "no-damage" ? "assessed with no visible damage" : "auto-approved"} by the assessment pipeline.
                    </p>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", marginTop: "20px" }}>
                      {[
                        { label: "Total Repair Cost", value: formatCostRange(selected.totalMin, selected.totalMax) },
                        { label: "AI Confidence", value: avgConfidence(selected) > 0 ? `${avgConfidence(selected)}%` : "—" },
                        { label: "Damage Regions", value: String(selected.detections.length) },
                      ].map((stat) => (
                        <div key={stat.label} style={{ background: "#f5f5f0", borderRadius: "10px", padding: "14px", textAlign: "center" }}>
                          <div style={{ fontSize: "16px", fontWeight: 800, color: "#0a0a0a" }}>{stat.value}</div>
                          <div style={{ fontSize: "10px", color: "#888882", marginTop: "2px" }}>{stat.label}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {section === "Damage Photo" && (
                  <div>
                    <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#0a0a0a", marginBottom: "14px" }}>Damage Photo</h3>
                    <AnnotatedImage src={selected.thumbnail} detections={selected.detections} naturalSize={{ w: selected.imgW, h: selected.imgH }} />
                    {selected.maskThumbnail && (
                      <div style={{ marginTop: "20px" }}>
                        <div style={{ fontSize: "11px", fontWeight: 700, color: "#0a0a0a", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "8px" }}>
                          SAM2 Damage Mask
                        </div>
                        <img src={selected.maskThumbnail} alt="SAM2 damage mask" style={{ width: "100%", borderRadius: "10px", display: "block", border: "1px solid rgba(0,0,0,0.06)" }} />
                      </div>
                    )}
                    {selected.mergedThumbnail && (
                      <div style={{ marginTop: "20px" }}>
                        <div style={{ fontSize: "11px", fontWeight: 700, color: "#0a0a0a", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "8px" }}>
                          Merged (VLM ∪ SAM2)
                        </div>
                        <img src={selected.mergedThumbnail} alt="Merged VLM and SAM2 overlay" style={{ width: "100%", borderRadius: "10px", display: "block", border: "1px solid rgba(0,0,0,0.06)" }} />
                      </div>
                    )}
                  </div>
                )}

                {section === "Analysis Results" && (
                  <div>
                    <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#0a0a0a", marginBottom: "14px" }}>Damage Analysis Results</h3>
                    {selected.findings.length === 0 ? (
                      <p style={{ fontSize: "12px", color: "#888882" }}>No damage was detected.</p>
                    ) : (
                      selected.findings.map((r, i) => (
                        <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: "12px", padding: "12px 14px", background: "#fafafa", borderRadius: "8px", marginBottom: "8px", border: "1px solid rgba(0,0,0,0.04)" }}>
                          <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: r.severity === "Severe" ? "#ef4444" : r.severity === "Moderate" ? "#f97316" : "var(--accent)", flexShrink: 0, marginTop: "4px" }} />
                          <div style={{ flex: 1 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "2px" }}>
                              <span style={{ fontSize: "12px", fontWeight: 600, color: "#0a0a0a" }}>{r.area}</span>
                              <span style={{ fontSize: "9px", background: r.severity === "Severe" ? "#fee2e2" : r.severity === "Moderate" ? "#fff7ed" : "#fef9c3", color: r.severity === "Severe" ? "#b91c1c" : r.severity === "Moderate" ? "#c2410c" : "#854d0e", padding: "1px 6px", borderRadius: "4px", fontWeight: 700 }}>
                                {r.severity.toUpperCase()}
                              </span>
                            </div>
                            <div style={{ fontSize: "11px", color: "#666660" }}>{r.damage}</div>
                          </div>
                          <div style={{ textAlign: "right", flexShrink: 0 }}>
                            <div style={{ fontSize: "12px", fontWeight: 700, color: "#0a0a0a" }}>{formatCostRange(r.costMin, r.costMax)}</div>
                            {r.confidence > 0 && <div style={{ fontSize: "10px", color: "#888882" }}>{r.confidence}% conf.</div>}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}

                {section === "Cost Breakdown" && (
                  <div>
                    <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#0a0a0a", marginBottom: "14px" }}>Cost Breakdown</h3>
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                      <thead>
                        <tr style={{ background: "#f5f5f0" }}>
                          {["Component", "Damage", "Severity", "Estimated Cost"].map((h) => (
                            <th key={h} style={{ padding: "8px 12px", textAlign: "left", fontSize: "10px", fontWeight: 700, color: "#888882", letterSpacing: "0.05em" }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {selected.findings.map((row, i) => (
                          <tr key={i} style={{ borderBottom: "1px solid rgba(0,0,0,0.04)" }}>
                            <td style={{ padding: "10px 12px", fontSize: "12px", color: "#444440" }}>{row.area}</td>
                            <td style={{ padding: "10px 12px", fontSize: "12px", color: "#444440" }}>{row.damage}</td>
                            <td style={{ padding: "10px 12px", fontSize: "12px", color: "#444440" }}>{row.severity}</td>
                            <td style={{ padding: "10px 12px", fontSize: "12px", color: "#0a0a0a", fontWeight: 700 }}>{formatCostRange(row.costMin, row.costMax)}</td>
                          </tr>
                        ))}
                        <tr style={{ background: "#fffbeb", borderTop: "2px solid var(--accent)" }}>
                          <td colSpan={3} style={{ padding: "10px 12px", fontSize: "12px", fontWeight: 700, color: "#0a0a0a" }}>TOTAL ESTIMATE</td>
                          <td style={{ padding: "10px 12px", fontSize: "14px", fontWeight: 800, color: "#0a0a0a" }}>{formatCostRange(selected.totalMin, selected.totalMax)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import { Shield, Plus, FileText, Trash2, ArrowLeft, CheckCircle, Clock, AlertCircle, Pencil } from "lucide-react";
import type { VehicleRegistration } from "../../types/vehicle";
import type { AuthUser } from "../../lib/authStore";
import { getSettings } from "../../lib/settingsStore";
import {
  getInsuranceClaims,
  saveInsuranceClaim,
  deleteInsuranceClaim,
  subscribeInsurance,
  emptyInsuranceClaim,
  type InsuranceClaim,
} from "../../lib/insuranceStore";

interface InsurancePageProps {
  activeVehicle: VehicleRegistration | null;
  user?: AuthUser | null;
}

// ── Small styled form primitives ──────────────────────────────────────────────
const inputStyle: React.CSSProperties = {
  width: "100%", padding: "9px 12px", border: "1px solid rgba(0,0,0,0.1)", borderRadius: "8px",
  fontSize: "13px", color: "#0a0a0a", background: "#ffffff", outline: "none", boxSizing: "border-box",
};
function focus(e: React.FocusEvent<HTMLElement>) { e.currentTarget.style.borderColor = "var(--accent)"; }
function blur(e: React.FocusEvent<HTMLElement>) { e.currentTarget.style.borderColor = "rgba(0,0,0,0.1)"; }

function Field({ label, value, onChange, type = "text", placeholder, span = 1 }: {
  label: string; value: string; onChange: (v: string) => void; type?: string; placeholder?: string; span?: 1 | 2 | 3;
}) {
  return (
    <div style={{ gridColumn: `span ${span}` }}>
      <label style={{ fontSize: "11px", fontWeight: 600, color: "#444440", display: "block", marginBottom: "5px" }}>{label}</label>
      <input type={type} value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} onFocus={focus} onBlur={blur} style={inputStyle} />
    </div>
  );
}

function Area({ label, value, onChange, span = 3 }: { label: string; value: string; onChange: (v: string) => void; span?: 1 | 2 | 3 }) {
  return (
    <div style={{ gridColumn: `span ${span}` }}>
      <label style={{ fontSize: "11px", fontWeight: 600, color: "#444440", display: "block", marginBottom: "5px" }}>{label}</label>
      <textarea value={value} onChange={(e) => onChange(e.target.value)} onFocus={focus} onBlur={blur} rows={3} style={{ ...inputStyle, resize: "vertical", fontFamily: "inherit" }} />
    </div>
  );
}

function Section({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "20px 22px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
        <div style={{ width: "24px", height: "24px", borderRadius: "7px", background: "var(--accent)", color: "#0a0a0a", fontSize: "12px", fontWeight: 800, display: "flex", alignItems: "center", justifyContent: "center" }}>{n}</div>
        <h3 style={{ fontSize: "13px", fontWeight: 700, color: "#0a0a0a" }}>{title}</h3>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "14px" }}>{children}</div>
    </div>
  );
}

function Chip({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} style={{
      padding: "8px 14px", borderRadius: "8px", fontSize: "12px", fontWeight: active ? 700 : 500, cursor: "pointer",
      border: `1px solid ${active ? "var(--accent)" : "rgba(0,0,0,0.12)"}`,
      background: active ? "#fffbeb" : "#ffffff", color: active ? "#0a0a0a" : "#888882",
    }}>{label}</button>
  );
}

const STATUS_META: Record<string, { label: string; bg: string; fg: string; icon: React.ReactNode }> = {
  submitted: { label: "Submitted", bg: "#d1fae5", fg: "#065f46", icon: <CheckCircle size={11} /> },
  draft: { label: "Draft", bg: "#fef9c3", fg: "#854d0e", icon: <Clock size={11} /> },
};

function lossTypeText(c: InsuranceClaim): string {
  const parts = [c.lossDamage && "Damage", c.lossTheft && "Theft", c.lossThirdParty && "Third Party"].filter(Boolean);
  return parts.length ? parts.join(", ") : "—";
}

export function InsurancePage({ activeVehicle, user }: InsurancePageProps) {
  const [claims, setClaims] = useState<InsuranceClaim[]>(getInsuranceClaims());
  const [mode, setMode] = useState<"list" | "form">("list");
  const [form, setForm] = useState<InsuranceClaim | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  useEffect(() => {
    const load = () => setClaims(getInsuranceClaims());
    load();
    return subscribeInsurance(load);
  }, []);

  const stats = useMemo(() => ({
    total: claims.length,
    submitted: claims.filter((c) => c.status === "submitted").length,
    draft: claims.filter((c) => c.status === "draft").length,
  }), [claims]);

  function startNew() {
    setError(null);
    setForm(
      emptyInsuranceClaim({
        vehicleId: activeVehicle?.id,
        policyNo: activeVehicle?.insurancePolicyNumber ?? "",
        vehicleNo: activeVehicle?.licenseplate ?? "",
        chassisNo: activeVehicle?.vin ?? "",
        name: user?.name ?? activeVehicle?.ownerName ?? "",
        mobile: user?.phone ?? activeVehicle?.ownerPhone ?? "",
        email: getSettings().email || activeVehicle?.ownerEmail || "",
      }),
    );
    setMode("form");
  }

  function edit(c: InsuranceClaim) {
    setError(null);
    setForm({ ...c });
    setMode("form");
  }

  function set<K extends keyof InsuranceClaim>(key: K, value: InsuranceClaim[K]) {
    setForm((f) => (f ? { ...f, [key]: value } : f));
  }

  function persist(status: InsuranceClaim["status"]) {
    if (!form) return;
    if (status === "submitted") {
      if (!form.name.trim() || !form.vehicleNo.trim()) {
        setError("Please fill at least the insured name and vehicle number before submitting.");
        return;
      }
      if (!form.agreed) {
        setError("Please accept the declaration to submit the claim.");
        return;
      }
    }
    saveInsuranceClaim({ ...form, status });
    setMode("list");
    setSavedMsg(status === "submitted" ? "Claim form submitted." : "Draft saved.");
    setTimeout(() => setSavedMsg(null), 2500);
  }

  // ── List / dashboard view ──────────────────────────────────────────────────
  if (mode === "list") {
    return (
      <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "18px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px" }}>
          <div>
            <h1 style={{ fontSize: "20px", fontWeight: 800, color: "#0a0a0a", display: "flex", alignItems: "center", gap: "8px" }}>
              <Shield size={20} color="var(--accent)" /> Motor Insurance Claims
            </h1>
            <p style={{ fontSize: "12px", color: "#888882", marginTop: "2px" }}>
              File and track motor insurance claim forms{activeVehicle ? ` for ${activeVehicle.year} ${activeVehicle.make} ${activeVehicle.model}` : ""}.
            </p>
          </div>
          <button onClick={startNew} style={{ display: "inline-flex", alignItems: "center", gap: "6px", padding: "10px 18px", background: "var(--accent)", color: "#0a0a0a", border: "none", borderRadius: "9px", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}>
            <Plus size={15} /> New Claim Form
          </button>
        </div>

        {savedMsg && (
          <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "10px 14px", background: "#d1fae5", border: "1px solid #6ee7b7", borderRadius: "8px", fontSize: "12px", color: "#065f46", fontWeight: 600 }}>
            <CheckCircle size={14} /> {savedMsg}
          </div>
        )}

        {/* Stats */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "14px" }}>
          {[
            { label: "Total Claim Forms", value: stats.total },
            { label: "Submitted", value: stats.submitted },
            { label: "Drafts", value: stats.draft },
          ].map((s) => (
            <div key={s.label} style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "18px 20px" }}>
              <div style={{ fontSize: "26px", fontWeight: 800, color: "#0a0a0a" }}>{s.value}</div>
              <div style={{ fontSize: "11px", color: "#888882", marginTop: "2px" }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* List */}
        {claims.length === 0 ? (
          <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px dashed rgba(0,0,0,0.12)", padding: "56px 24px", textAlign: "center" }}>
            <div style={{ width: "56px", height: "56px", background: "var(--accent)", borderRadius: "16px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
              <Shield size={26} color="#0a0a0a" />
            </div>
            <h3 style={{ fontSize: "15px", fontWeight: 700, color: "#0a0a0a", marginBottom: "6px" }}>No claim forms yet</h3>
            <p style={{ fontSize: "12px", color: "#888882", marginBottom: "20px" }}>Start a motor insurance claim form — it's prefilled from your vehicle and profile.</p>
            <button onClick={startNew} style={{ display: "inline-flex", alignItems: "center", gap: "6px", background: "#0a0a0a", color: "var(--accent)", border: "none", borderRadius: "8px", padding: "10px 20px", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}>
              <Plus size={15} /> New Claim Form
            </button>
          </div>
        ) : (
          <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", overflow: "hidden" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "#fafafa" }}>
                  {["Claim ID", "Vehicle No", "Accident Date", "Type of Loss", "Est. Cost", "Status", ""].map((h) => (
                    <th key={h} style={{ textAlign: "left", padding: "11px 16px", fontSize: "10px", fontWeight: 700, color: "#888882", letterSpacing: "0.05em" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {claims.map((c) => {
                  const m = STATUS_META[c.status] ?? STATUS_META.draft;
                  return (
                    <tr key={c.id} style={{ borderTop: "1px solid rgba(0,0,0,0.05)" }}>
                      <td style={{ padding: "12px 16px" }}><code style={{ fontSize: "11px", fontWeight: 700, color: "#0a0a0a", background: "#f5f5f0", padding: "2px 7px", borderRadius: "4px" }}>{c.id}</code></td>
                      <td style={{ padding: "12px 16px", fontSize: "12px", color: "#0a0a0a" }}>{c.vehicleNo || "—"}</td>
                      <td style={{ padding: "12px 16px", fontSize: "12px", color: "#666660" }}>{c.accidentDate ? new Date(c.accidentDate).toLocaleString("en-IN", { dateStyle: "medium" }) : "—"}</td>
                      <td style={{ padding: "12px 16px", fontSize: "12px", color: "#666660" }}>{lossTypeText(c)}</td>
                      <td style={{ padding: "12px 16px", fontSize: "12px", color: "#0a0a0a", fontWeight: 600 }}>{c.estimatedRepairCost ? `₹ ${c.estimatedRepairCost}` : "—"}</td>
                      <td style={{ padding: "12px 16px" }}>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", fontSize: "10px", fontWeight: 700, color: m.fg, background: m.bg, padding: "3px 8px", borderRadius: "20px" }}>{m.icon} {m.label}</span>
                      </td>
                      <td style={{ padding: "12px 16px", textAlign: "right", whiteSpace: "nowrap" }}>
                        <button onClick={() => edit(c)} title="Open / edit" style={{ background: "transparent", border: "none", cursor: "pointer", color: "#888882", padding: "4px" }}><Pencil size={15} /></button>
                        <button onClick={() => { if (confirm(`Delete claim form ${c.id}?`)) deleteInsuranceClaim(c.id); }} title="Delete" style={{ background: "transparent", border: "none", cursor: "pointer", color: "#d4183d", padding: "4px" }}><Trash2 size={15} /></button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  }

  // ── Form view ───────────────────────────────────────────────────────────────
  if (!form) return null;
  const f = form;
  return (
    <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "16px", maxWidth: "980px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <button onClick={() => setMode("list")} style={{ display: "inline-flex", alignItems: "center", gap: "6px", padding: "8px 14px", background: "#f5f5f0", color: "#444440", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 600, cursor: "pointer" }}>
            <ArrowLeft size={14} /> Back
          </button>
          <div>
            <h1 style={{ fontSize: "16px", fontWeight: 800, color: "#0a0a0a" }}>Motor Insurance Claim Form</h1>
            <p style={{ fontSize: "11px", color: "#888882" }}>{f.id} · Issue of this form does not imply acceptance of liability.</p>
          </div>
        </div>
      </div>

      {error && (
        <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "10px 14px", background: "#fee2e2", border: "1px solid #fca5a5", borderRadius: "8px", fontSize: "12px", color: "#b91c1c", fontWeight: 600 }}>
          <AlertCircle size={14} /> {error}
        </div>
      )}

      {/* Policy / vehicle header */}
      <div style={{ background: "#0a0a0a", borderRadius: "12px", padding: "20px 22px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "14px" }}>
          {[
            { label: "Policy No", key: "policyNo" as const },
            { label: "Vehicle No", key: "vehicleNo" as const },
            { label: "Engine No", key: "engineNo" as const },
            { label: "Chassis No", key: "chassisNo" as const },
          ].map((fld) => (
            <div key={fld.key} style={{ gridColumn: fld.key === "policyNo" ? "span 1" : "span 1" }}>
              <label style={{ fontSize: "10px", fontWeight: 700, color: "var(--accent)", display: "block", marginBottom: "5px", letterSpacing: "0.05em", textTransform: "uppercase" }}>{fld.label}</label>
              <input value={f[fld.key]} onChange={(e) => set(fld.key, e.target.value)} style={{ ...inputStyle, background: "#1a1a1a", color: "#fff", border: "1px solid #2a2a2a" }} />
            </div>
          ))}
        </div>
      </div>

      <Section n={1} title="Insured Details">
        <Field label="Name" value={f.name} onChange={(v) => set("name", v)} span={2} />
        <Field label="Mobile No." value={f.mobile} onChange={(v) => set("mobile", v)} type="tel" />
        <Area label="Address" value={f.address} onChange={(v) => set("address", v)} span={2} />
        <Field label="E-Mail Id" value={f.email} onChange={(v) => set("email", v)} type="email" />
        <Area label="Other existing insurance policy(ies) for this accident" value={f.otherInsurance} onChange={(v) => set("otherInsurance", v)} span={3} />
      </Section>

      <Section n={2} title="Loss Details">
        <Field label="Date & Time of Accident / Occurrence" value={f.accidentDate} onChange={(v) => set("accidentDate", v)} type="datetime-local" />
        <Field label="Place of Loss" value={f.placeOfLoss} onChange={(v) => set("placeOfLoss", v)} />
        <Field label="Estimated Cost of Repairs (₹)" value={f.estimatedRepairCost} onChange={(v) => set("estimatedRepairCost", v)} />
        <div style={{ gridColumn: "span 3" }}>
          <label style={{ fontSize: "11px", fontWeight: 600, color: "#444440", display: "block", marginBottom: "6px" }}>Type of Loss</label>
          <div style={{ display: "flex", gap: "8px" }}>
            <Chip active={f.lossDamage} label="Damage" onClick={() => set("lossDamage", !f.lossDamage)} />
            <Chip active={f.lossTheft} label="Theft" onClick={() => set("lossTheft", !f.lossTheft)} />
            <Chip active={f.lossThirdParty} label="Third Party" onClick={() => set("lossThirdParty", !f.lossThirdParty)} />
          </div>
        </div>
        <Area label="Short Description of Accident / Incident" value={f.accidentDescription} onChange={(v) => set("accidentDescription", v)} span={3} />
      </Section>

      <Section n={3} title="Driver Details">
        <Field label="Name" value={f.driverName} onChange={(v) => set("driverName", v)} span={2} />
        <Field label="Age" value={f.driverAge} onChange={(v) => set("driverAge", v)} />
        <div style={{ gridColumn: "span 3" }}>
          <label style={{ fontSize: "11px", fontWeight: 600, color: "#444440", display: "block", marginBottom: "6px" }}>Is Driver</label>
          <div style={{ display: "flex", gap: "8px" }}>
            <Chip active={f.driverType === "owner"} label="Owner" onClick={() => set("driverType", "owner")} />
            <Chip active={f.driverType === "paid"} label="Paid Driver" onClick={() => set("driverType", "paid")} />
            <Chip active={f.driverType === "relative"} label="Relative / Friend" onClick={() => set("driverType", "relative")} />
          </div>
        </div>
        <Field label="Driving License No" value={f.licenseNo} onChange={(v) => set("licenseNo", v)} />
        <Field label="Valid up to" value={f.licenseValidUpto} onChange={(v) => set("licenseValidUpto", v)} type="date" />
        <Field label="Authorised to drive" value={f.authorisedToDrive} onChange={(v) => set("authorisedToDrive", v)} />
        <Field label="Issuing Authority" value={f.issuingAuthority} onChange={(v) => set("issuingAuthority", v)} />
      </Section>

      <Section n={4} title="Additional Details (Commercial Vehicles)">
        <Field label="Permit No." value={f.permitNo} onChange={(v) => set("permitNo", v)} />
        <Field label="Valid Up to" value={f.permitValidUpto} onChange={(v) => set("permitValidUpto", v)} type="date" />
        <Field label="Issuing Authority" value={f.permitIssuingAuthority} onChange={(v) => set("permitIssuingAuthority", v)} />
        <Field label="Fitness Certificate Valid Up to" value={f.fitnessValidUpto} onChange={(v) => set("fitnessValidUpto", v)} type="date" />
        <Field label="No. of fare paying passengers" value={f.passengersCarried} onChange={(v) => set("passengersCarried", v)} />
        <Field label="GR / LR No." value={f.grLrNo} onChange={(v) => set("grLrNo", v)} />
        <Area label="Weight and Nature of Goods Carried" value={f.goodsWeightNature} onChange={(v) => set("goodsWeightNature", v)} span={3} />
      </Section>

      <Section n={5} title="Injury / Death Details & Police Report">
        <div style={{ gridColumn: "span 1" }}>
          <label style={{ fontSize: "11px", fontWeight: 600, color: "#444440", display: "block", marginBottom: "6px" }}>Police Report Lodged?</label>
          <div style={{ display: "flex", gap: "8px" }}>
            <Chip active={f.policeReportLodged === "yes"} label="Yes" onClick={() => set("policeReportLodged", "yes")} />
            <Chip active={f.policeReportLodged === "no"} label="No" onClick={() => set("policeReportLodged", "no")} />
          </div>
        </div>
        <Field label="FIR / GD No." value={f.firNo} onChange={(v) => set("firNo", v)} />
        <Field label="Police Station Name" value={f.policeStation} onChange={(v) => set("policeStation", v)} />
        <div style={{ gridColumn: "span 3" }}>
          <label style={{ fontSize: "11px", fontWeight: 600, color: "#444440", display: "block", marginBottom: "6px" }}>Death / Injury to any occupant / Third Party or Third Party property damage?</label>
          <div style={{ display: "flex", gap: "8px" }}>
            <Chip active={f.injuryOrDeath === "yes"} label="Yes" onClick={() => set("injuryOrDeath", "yes")} />
            <Chip active={f.injuryOrDeath === "no"} label="No" onClick={() => set("injuryOrDeath", "no")} />
          </div>
        </div>
        <Area label="Details (death / injury to third party / occupants / driver, or property damage)" value={f.injuryDetails} onChange={(v) => set("injuryDetails", v)} span={3} />
      </Section>

      {/* Declaration */}
      <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "20px 22px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "14px" }}>
          <div style={{ width: "24px", height: "24px", borderRadius: "7px", background: "var(--accent)", color: "#0a0a0a", fontSize: "12px", fontWeight: 800, display: "flex", alignItems: "center", justifyContent: "center" }}>6</div>
          <h3 style={{ fontSize: "13px", fontWeight: 700, color: "#0a0a0a" }}>Declaration</h3>
        </div>
        <p style={{ fontSize: "11px", color: "#666660", lineHeight: 1.7, marginBottom: "14px" }}>
          I/We, to the best of my/our knowledge and belief, warrant the truth of the foregoing statements in every respect. I/We agree that any false or
          fraudulent statement, suppression or concealment shall render the policy void and forfeit all rights to recover thereunder. I understand the
          company reserves the right of verification of facts and documents relating to the policy and the claim.
        </p>
        <label style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "14px", cursor: "pointer" }}>
          <input type="checkbox" checked={f.agreed} onChange={(e) => set("agreed", e.target.checked)} style={{ width: "15px", height: "15px", accentColor: "var(--accent)" }} />
          <span style={{ fontSize: "12px", color: "#0a0a0a", fontWeight: 600 }}>I accept the declaration above.</span>
        </label>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "14px" }}>
          <Field label="Date" value={f.declarationDate} onChange={(v) => set("declarationDate", v)} type="date" />
          <Field label="Place" value={f.declarationPlace} onChange={(v) => set("declarationPlace", v)} />
          <Field label="Signature of the Insured (type full name)" value={f.signature} onChange={(v) => set("signature", v)} />
        </div>
        <p style={{ fontSize: "10px", color: "#888882", marginTop: "14px", fontStyle: "italic" }}>
          N.B. Please attach a photocopy of your blank / cancelled cheque for NEFT purpose.
        </p>
      </div>

      {/* Actions */}
      <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px", paddingBottom: "8px" }}>
        <button onClick={() => persist("draft")} style={{ padding: "11px 22px", background: "#f5f5f0", color: "#0a0a0a", border: "1px solid rgba(0,0,0,0.1)", borderRadius: "9px", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}>
          Save Draft
        </button>
        <button onClick={() => persist("submitted")} style={{ display: "inline-flex", alignItems: "center", gap: "6px", padding: "11px 24px", background: "var(--accent)", color: "#0a0a0a", border: "none", borderRadius: "9px", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}>
          <FileText size={15} /> Submit Claim
        </button>
      </div>
    </div>
  );
}

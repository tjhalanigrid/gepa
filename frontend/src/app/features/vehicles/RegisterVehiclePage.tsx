import { useState } from "react";
import { CheckCircle, Car, User, Shield } from "lucide-react";
import { saveVehicle, newVehicleId, newRegistrationNumber } from "../../lib/vehiclesStore";
import type { VehicleRegistration } from "../../types/vehicle";

type Step = "vehicle" | "owner" | "insurance" | "review";

const steps: { key: Step; label: string; icon: React.ReactNode }[] = [
  { key: "vehicle", label: "Vehicle Details", icon: <Car size={14} /> },
  { key: "owner", label: "Owner Info", icon: <User size={14} /> },
  { key: "insurance", label: "Insurance", icon: <Shield size={14} /> },
  { key: "review", label: "Review & Submit", icon: <CheckCircle size={14} /> },
];

interface FormData {
  make: string; model: string; year: string; color: string;
  vin: string; licenseplate: string; mileage: string; notes: string;
  ownerName: string; ownerPhone: string; ownerEmail: string; ownerAddress: string;
  insuranceProvider: string; policyNumber: string; insuranceExpiry: string;
}

const initialForm: FormData = {
  make: "", model: "", year: "", color: "",
  vin: "", licenseplate: "", mileage: "", notes: "",
  ownerName: "", ownerPhone: "", ownerEmail: "", ownerAddress: "",
  insuranceProvider: "", policyNumber: "", insuranceExpiry: "",
};

function FieldGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "5px" }}>
      <label style={{ fontSize: "10px", fontWeight: 700, color: "#666660", letterSpacing: "0.06em" }}>{label}</label>
      {children}
    </div>
  );
}

function TextInput({ placeholder, value, onChange, type = "text" }: { placeholder: string; value: string; onChange: (v: string) => void; type?: string }) {
  return (
    <input
      type={type}
      placeholder={placeholder}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{ padding: "9px 12px", border: "1px solid rgba(0,0,0,0.1)", borderRadius: "8px", fontSize: "13px", color: "#0a0a0a", background: "#ffffff", outline: "none", width: "100%", boxSizing: "border-box" }}
      onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
      onBlur={(e) => (e.currentTarget.style.borderColor = "rgba(0,0,0,0.1)")}
    />
  );
}

interface RegisterVehiclePageProps {
  onNavigate: (page: string) => void;
}

export function RegisterVehiclePage({ onNavigate }: RegisterVehiclePageProps) {
  const [currentStep, setCurrentStep] = useState<Step>("vehicle");
  const [form, setForm] = useState<FormData>(initialForm);
  const [submitted, setSubmitted] = useState(false);

  const stepOrder = steps.map((s) => s.key);
  const currentIndex = stepOrder.indexOf(currentStep);

  const [regNumber, setRegNumber] = useState("");

  function setField(key: keyof FormData, value: string) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function handleRegister() {
    const registrationNumber = newRegistrationNumber();
    const vehicle: VehicleRegistration = {
      id: newVehicleId(),
      registrationNumber,
      make: form.make.trim() || "Unknown",
      model: form.model.trim() || "Unknown",
      year: parseInt(form.year, 10) || new Date().getFullYear(),
      color: form.color.trim(),
      vin: form.vin.trim(),
      ownerName: form.ownerName.trim(),
      ownerPhone: form.ownerPhone.trim(),
      ownerEmail: form.ownerEmail.trim(),
      licenseplate: form.licenseplate.trim(),
      status: "active",
      registrationDate: new Date().toISOString().slice(0, 10),
      expiryDate: form.insuranceExpiry || "",
      insuranceProvider: form.insuranceProvider.trim(),
      insurancePolicyNumber: form.policyNumber.trim(),
      mileage: parseInt(form.mileage, 10) || 0,
      damageImages: [],
      notes: form.notes.trim(),
    };
    saveVehicle(vehicle);
    setRegNumber(registrationNumber);
    setSubmitted(true);
  }

  if (submitted) {
    return (
      <div style={{ padding: "24px", display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
        <div style={{ background: "#ffffff", borderRadius: "16px", padding: "48px", textAlign: "center", maxWidth: "480px", border: "1px solid rgba(0,0,0,0.06)" }}>
          <div style={{ width: "60px", height: "60px", background: "var(--accent)", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 18px" }}>
            <CheckCircle size={26} color="#0a0a0a" />
          </div>
          <h2 style={{ fontSize: "18px", fontWeight: 700, color: "#0a0a0a", marginBottom: "6px" }}>Vehicle Registered!</h2>
          <p style={{ fontSize: "12px", color: "#888882", marginBottom: "6px" }}>
            Registration ID: <strong style={{ color: "#0a0a0a" }}>{regNumber}</strong>
          </p>
          <p style={{ fontSize: "12px", color: "#888882", marginBottom: "24px" }}>
            {form.year} {form.make} {form.model} successfully registered.
          </p>
          <div style={{ display: "flex", gap: "10px", justifyContent: "center" }}>
            <button
              onClick={() => onNavigate("inspections")}
              style={{ padding: "10px 18px", background: "var(--accent)", color: "#0a0a0a", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 700, cursor: "pointer" }}
            >
              Start Inspection
            </button>
            <button
              onClick={() => { setSubmitted(false); setForm(initialForm); setCurrentStep("vehicle"); }}
              style={{ padding: "10px 18px", background: "#f5f5f0", color: "#0a0a0a", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 600, cursor: "pointer" }}
            >
              Register Another
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "18px" }}>
      {/* Stepper */}
      <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "18px 24px", display: "flex", alignItems: "center", overflowX: "auto" }}>
        {steps.map((step, i) => {
          const done = i < currentIndex;
          const active = step.key === currentStep;
          return (
            <div key={step.key} style={{ display: "flex", alignItems: "center", flex: i < steps.length - 1 ? "1" : "none" }}>
              <button
                onClick={() => setCurrentStep(step.key)}
                style={{ display: "flex", alignItems: "center", gap: "8px", background: "transparent", border: "none", cursor: "pointer", padding: "0", whiteSpace: "nowrap" }}
              >
                <div
                  style={{
                    width: "30px", height: "30px", borderRadius: "50%",
                    background: done ? "#0a0a0a" : active ? "var(--accent)" : "#f0f0eb",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    color: done ? "var(--accent)" : active ? "#0a0a0a" : "#888882",
                    flexShrink: 0, transition: "all 0.2s",
                  }}
                >
                  {done ? <CheckCircle size={14} /> : step.icon}
                </div>
                <div style={{ textAlign: "left" }}>
                  <div style={{ fontSize: "9px", color: "#888882" }}>Step {i + 1}</div>
                  <div style={{ fontSize: "11px", fontWeight: active ? 600 : 400, color: active ? "#0a0a0a" : done ? "#0a0a0a" : "#888882" }}>
                    {step.label}
                  </div>
                </div>
              </button>
              {i < steps.length - 1 && (
                <div style={{ flex: 1, height: "2px", background: done ? "var(--accent)" : "#f0f0eb", margin: "0 12px", minWidth: "20px" }} />
              )}
            </div>
          );
        })}
      </div>

      {/* Form card */}
      <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "28px" }}>
        {currentStep === "vehicle" && (
          <div>
            <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#0a0a0a", marginBottom: "4px" }}>Vehicle Details</h3>
            <p style={{ fontSize: "11px", color: "#888882", marginBottom: "20px" }}>Enter the vehicle's basic information</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
              <FieldGroup label="MAKE *"><TextInput placeholder="e.g. Toyota" value={form.make} onChange={(v) => setField("make", v)} /></FieldGroup>
              <FieldGroup label="MODEL *"><TextInput placeholder="e.g. Camry" value={form.model} onChange={(v) => setField("model", v)} /></FieldGroup>
              <FieldGroup label="YEAR *"><TextInput placeholder="e.g. 2023" value={form.year} onChange={(v) => setField("year", v)} type="number" /></FieldGroup>
              <FieldGroup label="COLOR *"><TextInput placeholder="e.g. Silver" value={form.color} onChange={(v) => setField("color", v)} /></FieldGroup>
              <FieldGroup label="VIN NUMBER *"><TextInput placeholder="17-character VIN" value={form.vin} onChange={(v) => setField("vin", v)} /></FieldGroup>
              <FieldGroup label="LICENSE PLATE *"><TextInput placeholder="e.g. ABC 1234" value={form.licenseplate} onChange={(v) => setField("licenseplate", v)} /></FieldGroup>
              <FieldGroup label="MILEAGE"><TextInput placeholder="Miles" value={form.mileage} onChange={(v) => setField("mileage", v)} type="number" /></FieldGroup>
              <FieldGroup label="NOTES"><TextInput placeholder="Additional info" value={form.notes} onChange={(v) => setField("notes", v)} /></FieldGroup>
            </div>
          </div>
        )}

        {currentStep === "owner" && (
          <div>
            <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#0a0a0a", marginBottom: "4px" }}>Owner Information</h3>
            <p style={{ fontSize: "11px", color: "#888882", marginBottom: "20px" }}>Details of the registered vehicle owner</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
              <FieldGroup label="FULL NAME *"><TextInput placeholder="e.g. James Whitfield" value={form.ownerName} onChange={(v) => setField("ownerName", v)} /></FieldGroup>
              <FieldGroup label="PHONE *"><TextInput placeholder="+1 555 000 0000" value={form.ownerPhone} onChange={(v) => setField("ownerPhone", v)} type="tel" /></FieldGroup>
              <FieldGroup label="EMAIL *"><TextInput placeholder="owner@email.com" value={form.ownerEmail} onChange={(v) => setField("ownerEmail", v)} type="email" /></FieldGroup>
              <FieldGroup label="ADDRESS"><TextInput placeholder="123 Main St, City, State" value={form.ownerAddress} onChange={(v) => setField("ownerAddress", v)} /></FieldGroup>
            </div>
          </div>
        )}

        {currentStep === "insurance" && (
          <div>
            <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#0a0a0a", marginBottom: "4px" }}>Insurance Details</h3>
            <p style={{ fontSize: "11px", color: "#888882", marginBottom: "20px" }}>Provide valid insurance information</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
              <FieldGroup label="PROVIDER *"><TextInput placeholder="e.g. SafeGuard Insurance" value={form.insuranceProvider} onChange={(v) => setField("insuranceProvider", v)} /></FieldGroup>
              <FieldGroup label="POLICY NUMBER *"><TextInput placeholder="e.g. SGI-2024-88231" value={form.policyNumber} onChange={(v) => setField("policyNumber", v)} /></FieldGroup>
              <FieldGroup label="EXPIRY DATE *"><TextInput placeholder="YYYY-MM-DD" value={form.insuranceExpiry} onChange={(v) => setField("insuranceExpiry", v)} type="date" /></FieldGroup>
            </div>
          </div>
        )}

        {currentStep === "review" && (
          <div>
            <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#0a0a0a", marginBottom: "4px" }}>Review & Submit</h3>
            <p style={{ fontSize: "11px", color: "#888882", marginBottom: "20px" }}>Verify all information before final submission</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
              {[
                { section: "Vehicle", fields: [["Make", form.make], ["Model", form.model], ["Year", form.year], ["Color", form.color], ["VIN", form.vin], ["Plate", form.licenseplate]] },
                { section: "Owner", fields: [["Name", form.ownerName], ["Phone", form.ownerPhone], ["Email", form.ownerEmail], ["Address", form.ownerAddress]] },
              ].map(({ section, fields }) => (
                <div key={section} style={{ background: "#fafafa", borderRadius: "10px", padding: "16px" }}>
                  <div style={{ fontSize: "10px", fontWeight: 700, color: "#888882", letterSpacing: "0.06em", marginBottom: "10px" }}>{section.toUpperCase()}</div>
                  {fields.map(([k, v]) => (
                    <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                      <span style={{ fontSize: "11px", color: "#888882" }}>{k}</span>
                      <span style={{ fontSize: "11px", color: "#0a0a0a", fontWeight: 500 }}>{v || "—"}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
            <div style={{ marginTop: "14px", padding: "12px 14px", background: "#fffbeb", border: "1px solid var(--accent)", borderRadius: "8px", fontSize: "11px", color: "#854d0e" }}>
              <strong>Review carefully.</strong> Contact an administrator to modify a submitted registration.
            </div>
          </div>
        )}
      </div>

      {/* Nav buttons */}
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <button
          onClick={currentIndex === 0 ? () => onNavigate("dashboard") : () => setCurrentStep(stepOrder[currentIndex - 1])}
          style={{ padding: "10px 20px", background: "#f5f5f0", color: "#444440", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 500, cursor: "pointer" }}
        >
          {currentIndex === 0 ? "Cancel" : "Back"}
        </button>
        <button
          onClick={currentStep === "review" ? handleRegister : () => setCurrentStep(stepOrder[currentIndex + 1])}
          style={{
            padding: "10px 22px", background: currentStep === "review" ? "#10b981" : "var(--accent)",
            color: currentStep === "review" ? "#ffffff" : "#0a0a0a",
            border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 700, cursor: "pointer",
            display: "flex", alignItems: "center", gap: "6px",
          }}
        >
          {currentStep === "review" ? <><CheckCircle size={13} /> Submit Registration</> : "Continue →"}
        </button>
      </div>
    </div>
  );
}

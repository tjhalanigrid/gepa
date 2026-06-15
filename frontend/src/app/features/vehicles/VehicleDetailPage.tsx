import { ArrowLeft, Car, User, Shield, Phone, Mail, MapPin, Calendar, Gauge, FileText } from "lucide-react";
import { getVehicle, getVehicles } from "../../lib/vehiclesStore";
import type { VehicleStatus } from "../../types/vehicle";

const statusColors: Record<VehicleStatus, { bg: string; text: string; dot: string }> = {
  active: { bg: "#d1fae5", text: "#065f46", dot: "#10b981" },
  inactive: { bg: "#f3f4f6", text: "#6b7280", dot: "#9ca3af" },
  maintenance: { bg: "#fff7ed", text: "#c2410c", dot: "#f97316" },
  pending: { bg: "#fef9c3", text: "#854d0e", dot: "var(--accent)" },
};

function InfoRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: "10px", padding: "10px 0", borderBottom: "1px solid rgba(0,0,0,0.04)" }}>
      <div style={{ color: "#888882", marginTop: "2px", flexShrink: 0 }}>{icon}</div>
      <div>
        <div style={{ fontSize: "10px", color: "#888882", fontWeight: 600, letterSpacing: "0.04em" }}>{label}</div>
        <div style={{ fontSize: "13px", color: "#0a0a0a", marginTop: "1px" }}>{value || "—"}</div>
      </div>
    </div>
  );
}

interface VehicleDetailPageProps {
  vehicleId?: string;
  onNavigate: (page: string) => void;
}

export function VehicleDetailPage({ vehicleId, onNavigate }: VehicleDetailPageProps) {
  const vehicle = getVehicle(vehicleId) ?? getVehicles()[0];

  if (!vehicle) {
    return (
      <div style={{ padding: "24px", display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ width: "60px", height: "60px", background: "#f5f5f0", borderRadius: "16px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
            <Car size={26} color="#888882" />
          </div>
          <h2 style={{ fontSize: "16px", fontWeight: 700, color: "#0a0a0a", marginBottom: "6px" }}>Vehicle not found</h2>
          <p style={{ fontSize: "12px", color: "#888882", marginBottom: "20px" }}>Register a vehicle to see its details.</p>
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

  const sc = statusColors[vehicle.status];

  return (
    <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Back button + header */}
      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        <button
          onClick={() => onNavigate("vehicles")}
          style={{
            padding: "8px 12px",
            background: "#ffffff",
            border: "1px solid rgba(0,0,0,0.08)",
            borderRadius: "8px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "6px",
            fontSize: "12px",
            color: "#444440",
          }}
        >
          <ArrowLeft size={13} />
          Back
        </button>
        <div>
          <h2 style={{ fontSize: "16px", fontWeight: 700, color: "#0a0a0a" }}>
            {vehicle.year} {vehicle.make} {vehicle.model}
          </h2>
          <p style={{ fontSize: "11px", color: "#888882" }}>{vehicle.registrationNumber}</p>
        </div>
        <span
          style={{
            marginLeft: "auto",
            background: sc.bg,
            color: sc.text,
            fontSize: "11px",
            fontWeight: 600,
            padding: "4px 12px",
            borderRadius: "10px",
            display: "inline-flex",
            alignItems: "center",
            gap: "5px",
            textTransform: "capitalize",
          }}
        >
          <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: sc.dot }} />
          {vehicle.status}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 300px", gap: "16px" }}>
        {/* Vehicle info */}
        <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <Car size={15} color="var(--accent)" />
            <h3 style={{ fontSize: "13px", fontWeight: 600, color: "#0a0a0a" }}>Vehicle Information</h3>
          </div>
          <InfoRow icon={<Car size={13} />} label="MAKE & MODEL" value={`${vehicle.make} ${vehicle.model}`} />
          <InfoRow icon={<Calendar size={13} />} label="YEAR" value={String(vehicle.year)} />
          <InfoRow icon={<FileText size={13} />} label="COLOR" value={vehicle.color} />
          <InfoRow icon={<FileText size={13} />} label="VIN NUMBER" value={vehicle.vin} />
          <InfoRow icon={<FileText size={13} />} label="LICENSE PLATE" value={vehicle.licenseplate} />
          <InfoRow icon={<Gauge size={13} />} label="MILEAGE" value={`${vehicle.mileage.toLocaleString()} miles`} />
          {vehicle.notes && (
            <InfoRow icon={<FileText size={13} />} label="NOTES" value={vehicle.notes} />
          )}
        </div>

        {/* Owner info */}
        <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <User size={15} color="var(--accent)" />
            <h3 style={{ fontSize: "13px", fontWeight: 600, color: "#0a0a0a" }}>Owner & Insurance</h3>
          </div>
          <InfoRow icon={<User size={13} />} label="OWNER NAME" value={vehicle.ownerName} />
          <InfoRow icon={<Phone size={13} />} label="PHONE" value={vehicle.ownerPhone} />
          <InfoRow icon={<Mail size={13} />} label="EMAIL" value={vehicle.ownerEmail} />
          <div style={{ marginTop: "16px", paddingTop: "16px", borderTop: "1px solid rgba(0,0,0,0.06)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
              <Shield size={15} color="var(--accent)" />
              <span style={{ fontSize: "13px", fontWeight: 600, color: "#0a0a0a" }}>Insurance</span>
            </div>
            <InfoRow icon={<Shield size={13} />} label="PROVIDER" value={vehicle.insuranceProvider} />
            <InfoRow icon={<FileText size={13} />} label="POLICY #" value={vehicle.insurancePolicyNumber} />
          </div>
        </div>

        {/* Registration timeline */}
        <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "20px" }}>
          <h3 style={{ fontSize: "13px", fontWeight: 600, color: "#0a0a0a", marginBottom: "16px" }}>Registration Timeline</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "0" }}>
            {[
              { label: "Registered On", date: vehicle.registrationDate, color: "#10b981", active: true },
              { label: "Last Inspected", date: "2024-05-20", color: "var(--accent)", active: true },
              { label: "Expires On", date: vehicle.expiryDate, color: vehicle.status === "inactive" ? "#ef4444" : "#888882", active: false },
            ].map((item, i) => (
              <div key={item.label} style={{ display: "flex", gap: "12px", paddingBottom: i < 2 ? "16px" : "0", position: "relative" }}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                  <div style={{ width: "10px", height: "10px", borderRadius: "50%", background: item.color, flexShrink: 0, marginTop: "2px" }} />
                  {i < 2 && <div style={{ width: "2px", flex: 1, background: "#f0f0eb", marginTop: "4px" }} />}
                </div>
                <div>
                  <div style={{ fontSize: "10px", color: "#888882", fontWeight: 600, letterSpacing: "0.04em" }}>{item.label}</div>
                  <div style={{ fontSize: "13px", color: "#0a0a0a", marginTop: "1px" }}>
                    {new Date(item.date).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" })}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Actions */}
          <div style={{ marginTop: "24px", display: "flex", flexDirection: "column", gap: "8px" }}>
            <button
              style={{
                padding: "9px",
                background: "var(--accent)",
                color: "#0a0a0a",
                border: "none",
                borderRadius: "8px",
                fontSize: "12px",
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              Edit Registration
            </button>
            <button
              style={{
                padding: "9px",
                background: "#f5f5f0",
                color: "#444440",
                border: "none",
                borderRadius: "8px",
                fontSize: "12px",
                cursor: "pointer",
              }}
            >
              Print Certificate
            </button>
            <button
              style={{
                padding: "9px",
                background: "#fff0f0",
                color: "#d4183d",
                border: "none",
                borderRadius: "8px",
                fontSize: "12px",
                cursor: "pointer",
              }}
            >
              Deactivate
            </button>
          </div>
        </div>
      </div>

      {/* Damage photos section */}
      <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
          <MapPin size={15} color="var(--accent)" />
          <h3 style={{ fontSize: "13px", fontWeight: 600, color: "#0a0a0a" }}>Damage Documentation</h3>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
          {(["front", "left", "right", "back"] as const).map((side) => (
            <div
              key={side}
              style={{
                background: "#f5f5f0",
                borderRadius: "10px",
                aspectRatio: "4/3",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "6px",
                border: "2px dashed rgba(0,0,0,0.08)",
              }}
            >
              <Car size={24} color="#cccccc" />
              <span style={{ fontSize: "10px", color: "#888882", fontWeight: 600, textTransform: "capitalize" }}>
                {side} View
              </span>
              <span style={{ fontSize: "9px", color: "#aaaaaa" }}>No image uploaded</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

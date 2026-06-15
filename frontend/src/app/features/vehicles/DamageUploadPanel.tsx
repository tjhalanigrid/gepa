import { useRef, useState } from "react";
import { Upload, X, Camera, AlertCircle } from "lucide-react";

type CarSide = "front" | "left" | "right" | "back";

interface SideDamage {
  file: File | null;
  preview: string | null;
  notes: string;
}

type DamageMap = Record<CarSide, SideDamage>;

const sideLabels: Record<CarSide, { label: string; icon: string; desc: string }> = {
  front: { label: "Front View", icon: "⬆", desc: "Bumper, hood, headlights" },
  left: { label: "Left Side", icon: "⬅", desc: "Driver side panel, doors" },
  right: { label: "Right Side", icon: "➡", desc: "Passenger side panel, doors" },
  back: { label: "Rear View", icon: "⬇", desc: "Trunk, taillights, bumper" },
};

const sides: CarSide[] = ["front", "left", "right", "back"];

export function DamageUploadPanel() {
  const [damages, setDamages] = useState<DamageMap>({
    front: { file: null, preview: null, notes: "" },
    left: { file: null, preview: null, notes: "" },
    right: { file: null, preview: null, notes: "" },
    back: { file: null, preview: null, notes: "" },
  });
  const [activeNote, setActiveNote] = useState<CarSide | null>(null);

  const fileInputRefs = useRef<Record<CarSide, HTMLInputElement | null>>({
    front: null,
    left: null,
    right: null,
    back: null,
  });

  function handleFileChange(side: CarSide, e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const preview = URL.createObjectURL(file);
    setDamages((prev) => ({ ...prev, [side]: { ...prev[side], file, preview } }));
  }

  function handleRemove(side: CarSide) {
    if (damages[side].preview) URL.revokeObjectURL(damages[side].preview!);
    setDamages((prev) => ({ ...prev, [side]: { file: null, preview: null, notes: prev[side].notes } }));
    if (fileInputRefs.current[side]) fileInputRefs.current[side]!.value = "";
  }

  function handleNotesChange(side: CarSide, value: string) {
    setDamages((prev) => ({ ...prev, [side]: { ...prev[side], notes: value } }));
  }

  const uploadedCount = sides.filter((s) => damages[s].file !== null).length;

  return (
    <div>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "16px",
        }}
      >
        <div>
          <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#0a0a0a" }}>Vehicle Damage Documentation</h3>
          <p style={{ fontSize: "11px", color: "#888882", marginTop: "2px" }}>
            Upload photos for all 4 sides. Supported: JPG, PNG, WEBP (max 10MB)
          </p>
        </div>
        <div
          style={{
            background: uploadedCount === 4 ? "#d1fae5" : "#f5f5f0",
            color: uploadedCount === 4 ? "#065f46" : "#888882",
            fontSize: "11px",
            fontWeight: 600,
            padding: "4px 10px",
            borderRadius: "20px",
          }}
        >
          {uploadedCount}/4 uploaded
        </div>
      </div>

      {/* Car diagram + upload zones */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gridTemplateRows: "auto auto",
          gap: "12px",
        }}
      >
        {sides.map((side) => {
          const info = sideLabels[side];
          const dmg = damages[side];
          const hasImage = dmg.preview !== null;

          return (
            <div
              key={side}
              style={{
                background: "#fafafa",
                border: `2px dashed ${hasImage ? "var(--accent)" : "rgba(0,0,0,0.12)"}`,
                borderRadius: "12px",
                padding: "16px",
                position: "relative",
                transition: "border-color 0.2s",
              }}
            >
              {/* Side label */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  marginBottom: "10px",
                }}
              >
                <div
                  style={{
                    width: "28px",
                    height: "28px",
                    background: hasImage ? "var(--accent)" : "#e8e8e3",
                    borderRadius: "6px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "13px",
                  }}
                >
                  {info.icon}
                </div>
                <div>
                  <div style={{ fontSize: "12px", fontWeight: 600, color: "#0a0a0a" }}>{info.label}</div>
                  <div style={{ fontSize: "10px", color: "#888882" }}>{info.desc}</div>
                </div>
              </div>

              {/* Image area */}
              {hasImage ? (
                <div style={{ position: "relative", marginBottom: "10px" }}>
                  <img
                    src={dmg.preview!}
                    alt={`${side} damage`}
                    style={{
                      width: "100%",
                      height: "120px",
                      objectFit: "cover",
                      borderRadius: "8px",
                      display: "block",
                    }}
                  />
                  <button
                    onClick={() => handleRemove(side)}
                    style={{
                      position: "absolute",
                      top: "6px",
                      right: "6px",
                      width: "22px",
                      height: "22px",
                      background: "#0a0a0a",
                      border: "none",
                      borderRadius: "50%",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      cursor: "pointer",
                      color: "#ffffff",
                    }}
                  >
                    <X size={11} />
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => fileInputRefs.current[side]?.click()}
                  style={{
                    width: "100%",
                    height: "100px",
                    background: "transparent",
                    border: "none",
                    cursor: "pointer",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "6px",
                    color: "#888882",
                    marginBottom: "10px",
                    borderRadius: "8px",
                    transition: "background 0.15s",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "#f0f0eb")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <div
                    style={{
                      width: "40px",
                      height: "40px",
                      background: "#e8e8e3",
                      borderRadius: "10px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <Camera size={18} color="#888882" />
                  </div>
                  <span style={{ fontSize: "11px", fontWeight: 500 }}>Click to upload</span>
                  <span style={{ fontSize: "10px", color: "#aaaaaa" }}>or drag & drop</span>
                </button>
              )}

              {/* Notes toggle */}
              <button
                onClick={() => setActiveNote(activeNote === side ? null : side)}
                style={{
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  fontSize: "11px",
                  color: "#888882",
                  display: "flex",
                  alignItems: "center",
                  gap: "4px",
                  padding: "0",
                }}
              >
                <AlertCircle size={11} />
                {dmg.notes ? "Edit damage notes" : "Add damage notes"}
              </button>

              {activeNote === side && (
                <textarea
                  autoFocus
                  placeholder={`Describe ${info.label.toLowerCase()} damage...`}
                  value={dmg.notes}
                  onChange={(e) => handleNotesChange(side, e.target.value)}
                  style={{
                    width: "100%",
                    marginTop: "8px",
                    padding: "8px",
                    fontSize: "11px",
                    border: "1px solid rgba(0,0,0,0.1)",
                    borderRadius: "6px",
                    resize: "vertical",
                    outline: "none",
                    background: "#ffffff",
                    color: "#0a0a0a",
                    minHeight: "60px",
                    boxSizing: "border-box",
                  }}
                />
              )}

              {/* Hidden file input */}
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                style={{ display: "none" }}
                ref={(el) => { fileInputRefs.current[side] = el; }}
                onChange={(e) => handleFileChange(side, e)}
              />

              {/* Upload button (when image exists) */}
              {!hasImage && (
                <button
                  onClick={() => fileInputRefs.current[side]?.click()}
                  style={{
                    marginTop: "8px",
                    width: "100%",
                    padding: "7px",
                    background: "#0a0a0a",
                    color: "var(--accent)",
                    border: "none",
                    borderRadius: "6px",
                    fontSize: "11px",
                    fontWeight: 600,
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "5px",
                  }}
                >
                  <Upload size={11} />
                  Upload {info.label}
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

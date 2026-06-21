import { useEffect, useRef, useState } from "react";
import { CheckCircle, Camera, Upload, X, Zap, FileText, ChevronRight, AlertTriangle } from "lucide-react";
import { getVehicles, subscribeVehicles } from "../../lib/vehiclesStore";
import type { VehicleRegistration } from "../../types/vehicle";
import {
  assessDamage,
  unwrapReport,
  getSessionId,
  prettifyLabel,
  capitalize,
  formatCostRange,
  jobMaskedImageUrl,
  jobMergedImageUrl,
  type FinalDamageReport,
  type DetectionWithBBox,
  type DamagePartEntry,
} from "../../lib/api";
import { AnnotatedImage, CLASS_COLORS } from "./AnnotatedImage";
import { ReviewPanel } from "./ReviewPanel";
import { saveClaim, makeThumbnail, makeRemoteThumbnail, newClaimId, statusFromApproval } from "../../lib/claimsStore";

type Step = "select" | "upload" | "analyze" | "report";

// A single damage finding mapped from the backend report, ready for display.
interface DisplayResult {
  area: string; // prettified part, e.g. "Front Bumper"
  damage: string; // e.g. "Moderate dent"
  severity: string; // "Minor" | "Moderate" | "Severe"
  confidence: number; // percent (0–100)
  costMin: number; // INR
  costMax: number; // INR
}

interface InspectionPageProps {
  onNavigate: (page: string) => void;
  activeVehicle: VehicleRegistration | null;
}

export function InspectionPage({ onNavigate, activeVehicle }: InspectionPageProps) {
  const [step, setStep] = useState<Step>(activeVehicle ? "upload" : "select");
  const [selectedVehicle, setSelectedVehicle] = useState<VehicleRegistration | null>(activeVehicle);
  const [vehicles, setVehicles] = useState<VehicleRegistration[]>(getVehicles());

  useEffect(() => {
    const load = () => setVehicles(getVehicles());
    load();
    return subscribeVehicles(load);
  }, []);

  // Single uploaded image.
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const [analyzing, setAnalyzing] = useState(false);
  const [analysisDone, setAnalysisDone] = useState(false);
  const [progress, setProgress] = useState(0);

  // Real assessment state, populated from the backend AI pipeline.
  const [results, setResults] = useState<DisplayResult[]>([]);
  const [detections, setDetections] = useState<DetectionWithBBox[]>([]);
  // Server-rendered overlays (served per-job by the backend).
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [maskFailed, setMaskFailed] = useState(false);
  const [mergedUrl, setMergedUrl] = useState<string | null>(null);
  const [mergedFailed, setMergedFailed] = useState(false);
  // Which overlay the result viewer is showing.
  const [imageView, setImageView] = useState<"boxes" | "mask" | "merged">("boxes");
  const [totalMin, setTotalMin] = useState(0);
  const [totalMax, setTotalMax] = useState(0);
  const [approval, setApproval] = useState<string>("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  // TEMP human-review (GEPA ground-truth collection): set when a job escalates.
  const [reviewSessionId, setReviewSessionId] = useState<string | null>(null);
  const [reviewFindings, setReviewFindings] = useState<DamagePartEntry[]>([]);

  const fileRef = useRef<HTMLInputElement | null>(null);

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    // A new image invalidates any previous result.
    setAnalysisDone(false);
    setResults([]);
    setDetections([]);
    setMaskUrl(null);
    setMaskFailed(false);
    setMergedUrl(null);
    setMergedFailed(false);
    setImageView("boxes");
    setError(null);
  }

  function handleRemove() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setAnalysisDone(false);
    setResults([]);
    setDetections([]);
    setMaskUrl(null);
    setMaskFailed(false);
    setMergedUrl(null);
    setMergedFailed(false);
    setImageView("boxes");
    if (fileRef.current) fileRef.current.value = "";
  }

  // Replace each VLM box with its tighter SAM2 mask-derived box when they overlap.
  // SAM2 boxes (source "sam2" in merged_detections) outline the actual damage, so
  // this yields more accurate boxes while keeping the VLM's damage/part/severity.
  function refineBoxes(report: FinalDamageReport): DetectionWithBBox[] {
    const dets = report.detections_with_bbox ?? [];
    const sam2 = (report.merged_detections ?? []).filter(
      (m: any) => m?.source === "sam2" && Array.isArray(m.bbox),
    ) as any[];
    if (!sam2.length) return dets;

    const iou = (a: number[], b: number[]) => {
      const x1 = Math.max(a[0], b[0]);
      const y1 = Math.max(a[1], b[1]);
      const x2 = Math.min(a[2], b[2]);
      const y2 = Math.min(a[3], b[3]);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      if (inter <= 0) return 0;
      const areaA = (a[2] - a[0]) * (a[3] - a[1]);
      const areaB = (b[2] - b[0]) * (b[3] - b[1]);
      return inter / (areaA + areaB - inter);
    };

    return dets.map((d) => {
      let best: number[] | null = null;
      let bestIou = 0.1; // require minimal overlap before trusting the refinement
      for (const s of sam2) {
        const o = iou(d.bbox, s.bbox);
        if (o > bestIou) {
          bestIou = o;
          best = s.bbox;
        }
      }
      return best ? { ...d, bbox: best } : d;
    });
  }

  function mapReport(report: FinalDamageReport): DisplayResult[] {
    const confOf = (damage: string, part: string): number => {
      const hit = report.detections_with_bbox.find(
        (d) => d.damage === damage && d.part === part,
      );
      return hit ? Math.round(hit.confidence * 100) : 0;
    };
    return report.damage_part_map.map((d) => ({
      area: prettifyLabel(d.part),
      damage: `${capitalize(d.severity)} ${d.damage.replace(/_/g, " ")}`,
      severity: capitalize(d.severity),
      confidence: confOf(d.damage, d.part),
      costMin: d.cost_min,
      costMax: d.cost_max,
    }));
  }

  async function startAnalysis() {
    if (!file) {
      setError("Upload a photo before running analysis.");
      return;
    }

    setError(null);
    setAnalyzing(true);
    setAnalysisDone(false);
    setProgress(8);
    setResults([]);
    setDetections([]);
    setMaskUrl(null);
    setMaskFailed(false);
    setMergedUrl(null);
    setMergedFailed(false);
    setImageView("boxes");
    setWarnings([]);
    setReviewSessionId(null);
    setReviewFindings([]);

    // Smooth indeterminate progress while the single VLM job runs (~40–70s).
    const ticker = setInterval(() => {
      setProgress((p) => (p < 90 ? p + Math.random() * 6 : p));
    }, 1200);

    // Captured locally too (state updates aren't readable synchronously below).
    let capturedMaskUrl = "";
    let capturedMergedUrl = "";

    try {
      const raw = await assessDamage(file, {
        vehicleId: selectedVehicle?.id,
        claimId: selectedVehicle?.id ? `CLM-${selectedVehicle.id}` : undefined,
        // Capture the job id so we can fetch the backend-rendered overlays.
        onJobStart: (jobId) => {
          capturedMaskUrl = jobMaskedImageUrl(jobId);
          capturedMergedUrl = jobMergedImageUrl(jobId);
          setMaskUrl(capturedMaskUrl);
          setMergedUrl(capturedMergedUrl);
        },
      });
      const report = unwrapReport(raw);

      // If the job escalated to human review, expose the session for the ReviewPanel.
      const sessionId = getSessionId(raw);
      setReviewSessionId(sessionId);
      setReviewFindings(report.damage_part_map);

      const findings = mapReport(report);
      const boxes = refineBoxes(report);
      setResults(findings);
      setDetections(boxes);
      setWarnings(report.warnings ?? []);
      setTotalMin(report.total_min);
      setTotalMax(report.total_max);
      setApproval(report.approval_decision);
      setProgress(100);
      setAnalysisDone(true);

      // Persist this analysis as a real claim (best-effort — never block the UI).
      try {
        const thumb = await makeThumbnail(file);
        // Snapshot the server overlays onto the claim so they persist into reports.
        const [maskThumb, mergedThumb] = boxes.length
          ? await Promise.all([
              capturedMaskUrl ? makeRemoteThumbnail(capturedMaskUrl) : Promise.resolve(null),
              capturedMergedUrl ? makeRemoteThumbnail(capturedMergedUrl) : Promise.resolve(null),
            ])
          : [null, null];
        saveClaim({
          id: newClaimId(),
          createdAt: new Date().toISOString(),
          vehicle: selectedVehicle
            ? `${selectedVehicle.year} ${selectedVehicle.make} ${selectedVehicle.model}`
            : "Unregistered vehicle",
          vehicleId: selectedVehicle?.id,
          thumbnail: thumb.dataUrl,
          maskThumbnail: maskThumb ?? undefined,
          mergedThumbnail: mergedThumb ?? undefined,
          imgW: thumb.w,
          imgH: thumb.h,
          detections: boxes.map((d) => ({
            index: d.index,
            bbox: d.bbox,
            damage: d.damage,
            part: d.part,
            severity: d.severity,
            confidence: d.confidence,
          })),
          findings,
          totalMin: report.total_min,
          totalMax: report.total_max,
          approval: report.approval_decision,
          status: statusFromApproval(report.approval_decision, boxes.length),
          inferenceS: report.total_inference_s,
        });
      } catch (saveErr) {
        console.warn("Could not save claim:", saveErr);
      }
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Could not reach the AI backend. Is it running on the configured URL?",
      );
    } finally {
      clearInterval(ticker);
      setAnalyzing(false);
    }
  }

  // Shared image style + captions for the overlay views.
  const overlayImgStyle: React.CSSProperties = {
    width: "100%", maxHeight: "460px", objectFit: "contain", background: "#0a0a0a",
    borderRadius: "12px", display: "block", border: "1px solid rgba(0,0,0,0.06)",
  };
  const VIEW_CAPTION: Record<string, string> = {
    boxes: "AI-detected damage regions with bounding boxes.",
    mask: "Pixel-level SAM2 segmentation of each damaged region.",
    merged: "Union of the VLM damage boxes and SAM2 regions (source-coloured).",
  };

  // ── Tabbed overlay viewer: Detections / SAM2 Mask / Merged ──────────────────
  function ImageTabs() {
    if (!preview) return null;
    const tabs = [
      { key: "boxes" as const, label: "Detections", show: true },
      { key: "mask" as const, label: "SAM2 Mask", show: !!maskUrl && !maskFailed },
      { key: "merged" as const, label: "Merged (VLM ∪ SAM2)", show: !!mergedUrl && !mergedFailed },
    ].filter((t) => t.show);
    const active = tabs.some((t) => t.key === imageView) ? imageView : "boxes";
    return (
      <div>
        {tabs.length > 1 && (
          <div style={{ display: "inline-flex", background: "#f0f0eb", borderRadius: "10px", padding: "3px", marginBottom: "12px", gap: "2px" }}>
            {tabs.map((t) => (
              <button
                key={t.key}
                onClick={() => setImageView(t.key)}
                style={{
                  padding: "6px 14px", borderRadius: "8px", border: "none", cursor: "pointer",
                  fontSize: "11px", fontWeight: 700,
                  background: active === t.key ? "#ffffff" : "transparent",
                  color: active === t.key ? "#0a0a0a" : "#888882",
                  boxShadow: active === t.key ? "0 1px 3px rgba(0,0,0,0.12)" : "none",
                }}
              >
                {t.label}
              </button>
            ))}
          </div>
        )}
        {active === "boxes" && <AnnotatedImage src={preview} detections={detections} />}
        {active === "mask" && maskUrl && (
          <img src={maskUrl} alt="SAM2 damage mask" onError={() => setMaskFailed(true)} style={overlayImgStyle} />
        )}
        {active === "merged" && mergedUrl && (
          <img src={mergedUrl} alt="Merged VLM ∪ SAM2 overlay" onError={() => setMergedFailed(true)} style={overlayImgStyle} />
        )}
        <p style={{ fontSize: "10px", color: "#888882", marginTop: "8px" }}>{VIEW_CAPTION[active]}</p>
      </div>
    );
  }

  // ── Stacked overlays for the report view (all images, with headings) ────────
  function StackedImages() {
    if (!preview) return null;
    const labelled = (title: string, node: React.ReactNode, caption: string) => (
      <div style={{ marginTop: "20px" }}>
        <div style={{ fontSize: "11px", fontWeight: 700, color: "#0a0a0a", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "8px" }}>{title}</div>
        {node}
        <p style={{ fontSize: "10px", color: "#888882", marginTop: "6px" }}>{caption}</p>
      </div>
    );
    return (
      <div>
        <AnnotatedImage src={preview} detections={detections} />
        {maskUrl && !maskFailed && detections.length > 0 &&
          labelled("SAM2 Damage Mask", <img src={maskUrl} alt="SAM2 damage mask" onError={() => setMaskFailed(true)} style={overlayImgStyle} />, VIEW_CAPTION.mask)}
        {mergedUrl && !mergedFailed && detections.length > 0 &&
          labelled("Merged (VLM ∪ SAM2)", <img src={mergedUrl} alt="Merged overlay" onError={() => setMergedFailed(true)} style={overlayImgStyle} />, VIEW_CAPTION.merged)}
      </div>
    );
  }

  const stepDefs: { key: Step; label: string; icon: React.ReactNode }[] = [
    { key: "select", label: "Select Vehicle", icon: <Camera size={14} /> },
    { key: "upload", label: "Upload Photo", icon: <Upload size={14} /> },
    { key: "analyze", label: "AI Analysis", icon: <Zap size={14} /> },
    { key: "report", label: "Report", icon: <FileText size={14} /> },
  ];
  const stepOrder: Step[] = ["select", "upload", "analyze", "report"];
  const currentIndex = stepOrder.indexOf(step);

  // ── Shared: tabbed overlay viewer + a legend, with the detection list ───────
  function DetectionView() {
    if (!preview) return null;
    return (
      <div style={{ display: "grid", gridTemplateColumns: detections.length ? "1.6fr 1fr" : "1fr", gap: "20px", alignItems: "start" }}>
        <div>
          <ImageTabs />
          {detections.length > 0 && imageView === "boxes" && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", marginTop: "12px" }}>
              {Array.from(new Set(detections.map((d) => d.damage))).map((cls) => (
                <div key={cls} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <span style={{ width: "12px", height: "12px", borderRadius: "3px", background: CLASS_COLORS[cls] ?? "#ef4444", display: "inline-block" }} />
                  <span style={{ fontSize: "11px", color: "#444440", textTransform: "capitalize" }}>{cls.replace(/_/g, " ")}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {detections.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            <div style={{ fontSize: "11px", fontWeight: 700, color: "#0a0a0a", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "2px" }}>
              Detected damage ({detections.length})
            </div>
            {detections.map((d, i) => {
              const color = CLASS_COLORS[d.damage] ?? "#ef4444";
              return (
                <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: "10px", padding: "10px 12px", background: "#fafafa", borderRadius: "10px", border: "1px solid rgba(0,0,0,0.05)" }}>
                  <div style={{ width: "20px", height: "20px", borderRadius: "5px", background: color, color: "#fff", fontSize: "11px", fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    {d.index}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: "12px", fontWeight: 600, color: "#0a0a0a", textTransform: "capitalize" }}>
                      {d.damage.replace(/_/g, " ")} · {prettifyLabel(d.part)}
                    </div>
                    <div style={{ fontSize: "10px", color: "#888882", textTransform: "capitalize" }}>
                      {d.severity} · confidence {Math.round(d.confidence * 100)}%
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  return (
    <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "18px" }}>
      {/* Stepper */}
      <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "18px 24px", display: "flex", alignItems: "center" }}>
        {stepDefs.map((s, i) => {
          const done = i < currentIndex;
          const active = s.key === step;
          return (
            <div key={s.key} style={{ display: "flex", alignItems: "center", flex: i < stepDefs.length - 1 ? "1" : "none" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", whiteSpace: "nowrap" }}>
                <div style={{
                  width: "30px", height: "30px", borderRadius: "50%", flexShrink: 0,
                  background: done ? "#0a0a0a" : active ? "var(--accent)" : "#f0f0eb",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  color: done ? "var(--accent)" : active ? "#0a0a0a" : "#888882",
                }}>
                  {done ? <CheckCircle size={14} /> : s.icon}
                </div>
                <div>
                  <div style={{ fontSize: "9px", color: "#888882" }}>Step {i + 1}</div>
                  <div style={{ fontSize: "11px", fontWeight: active ? 600 : 400, color: active ? "#0a0a0a" : done ? "#0a0a0a" : "#888882" }}>{s.label}</div>
                </div>
              </div>
              {i < stepDefs.length - 1 && <div style={{ flex: 1, height: "2px", background: done ? "var(--accent)" : "#f0f0eb", margin: "0 12px", minWidth: "20px" }} />}
            </div>
          );
        })}
      </div>

      {/* Step content */}
      <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "28px" }}>

        {/* ── Step 1: Select Vehicle ── */}
        {step === "select" && (
          <div>
            <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#0a0a0a", marginBottom: "4px" }}>Select Vehicle for Inspection</h3>
            <p style={{ fontSize: "11px", color: "#888882", marginBottom: "20px" }}>Choose the vehicle you want to inspect</p>
            {vehicles.length === 0 && (
              <div style={{ padding: "32px", textAlign: "center", background: "#fafafa", borderRadius: "12px", border: "1px dashed rgba(0,0,0,0.12)" }}>
                <p style={{ fontSize: "13px", color: "#888882", marginBottom: "16px" }}>No vehicles registered yet.</p>
                <button
                  onClick={() => onNavigate("register")}
                  style={{ padding: "10px 20px", background: "var(--accent)", color: "#0a0a0a", border: "none", borderRadius: "8px", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}
                >
                  + Register a Vehicle
                </button>
              </div>
            )}
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {vehicles.map((v) => {
                const selected = selectedVehicle?.id === v.id;
                return (
                  <button
                    key={v.id}
                    onClick={() => setSelectedVehicle(v)}
                    style={{
                      display: "flex", alignItems: "center", gap: "14px", padding: "14px 16px",
                      background: selected ? "#fffbeb" : "#fafafa",
                      border: `1px solid ${selected ? "var(--accent)" : "rgba(0,0,0,0.06)"}`,
                      borderRadius: "10px", cursor: "pointer", textAlign: "left", width: "100%",
                    }}
                  >
                    <div style={{ width: "40px", height: "40px", background: selected ? "var(--accent)" : "#f0f0eb", borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "11px", fontWeight: 800, color: selected ? "#0a0a0a" : "#888882", flexShrink: 0 }}>
                      {v.make.slice(0, 2).toUpperCase()}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: "13px", fontWeight: 600, color: "#0a0a0a" }}>{v.year} {v.make} {v.model}</div>
                      <div style={{ fontSize: "11px", color: "#888882" }}>{v.licenseplate} · {v.registrationNumber}</div>
                    </div>
                    {selected && <CheckCircle size={18} color="var(--accent)" />}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Step 2: Upload single photo (big) ── */}
        {step === "upload" && (
          <div>
            <div style={{ marginBottom: "16px" }}>
              <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#0a0a0a" }}>
                Damage Photo — {selectedVehicle?.make} {selectedVehicle?.model}
              </h3>
              <p style={{ fontSize: "11px", color: "#888882", marginTop: "2px" }}>
                Upload one clear photo of the damaged area. JPG, PNG, WEBP up to 10MB.
              </p>
            </div>

            {preview ? (
              <div style={{ position: "relative" }}>
                <img
                  src={preview}
                  alt="Uploaded damage"
                  style={{ width: "100%", maxHeight: "460px", objectFit: "contain", background: "#0a0a0a", borderRadius: "12px", display: "block" }}
                />
                <button
                  onClick={handleRemove}
                  style={{ position: "absolute", top: "12px", right: "12px", width: "32px", height: "32px", background: "#0a0a0a", border: "none", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", color: "#ffffff" }}
                  title="Remove photo"
                >
                  <X size={16} />
                </button>
                <button
                  onClick={() => fileRef.current?.click()}
                  style={{ marginTop: "12px", padding: "9px 16px", background: "#f5f5f0", color: "#444440", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 600, cursor: "pointer", display: "inline-flex", alignItems: "center", gap: "6px" }}
                >
                  <Upload size={13} /> Replace photo
                </button>
              </div>
            ) : (
              <button
                onClick={() => fileRef.current?.click()}
                style={{
                  width: "100%", minHeight: "320px", background: "#fafafa", border: "2px dashed rgba(0,0,0,0.15)",
                  borderRadius: "16px", cursor: "pointer", display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center", gap: "12px", color: "#888882",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f0f0eb")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "#fafafa")}
              >
                <div style={{ width: "64px", height: "64px", background: "#e8e8e3", borderRadius: "16px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Camera size={28} color="#888882" />
                </div>
                <span style={{ fontSize: "14px", fontWeight: 600, color: "#0a0a0a" }}>Click to upload a photo</span>
                <span style={{ fontSize: "12px", color: "#aaaaaa" }}>or drag &amp; drop your image here</span>
              </button>
            )}

            <input type="file" accept="image/jpeg,image/png,image/webp" style={{ display: "none" }} ref={fileRef} onChange={handleFile} />
          </div>
        )}

        {/* ── Step 3: AI Analysis ── */}
        {step === "analyze" && (
          <div>
            <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#0a0a0a", marginBottom: "4px" }}>AI Damage Analysis</h3>
            <p style={{ fontSize: "11px", color: "#888882", marginBottom: "20px" }}>
              The model scans the photo and marks each damaged region with a bounding box.
            </p>

            {!analyzing && !analysisDone && (
              <div style={{ textAlign: "center", padding: "20px 0" }}>
                {preview && (
                  <img src={preview} alt="To analyze" style={{ width: "100%", maxHeight: "380px", objectFit: "contain", background: "#0a0a0a", borderRadius: "12px", display: "block", marginBottom: "24px" }} />
                )}
                {error && (
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", justifyContent: "center", maxWidth: "440px", margin: "0 auto 18px", padding: "10px 14px", background: "#fee2e2", border: "1px solid #fca5a5", borderRadius: "8px" }}>
                    <AlertTriangle size={14} color="#b91c1c" />
                    <span style={{ fontSize: "11px", color: "#b91c1c", textAlign: "left" }}>{error}</span>
                  </div>
                )}
                <button
                  onClick={startAnalysis}
                  disabled={!file}
                  style={{ padding: "13px 32px", background: file ? "var(--accent)" : "#e8e8e3", color: file ? "#0a0a0a" : "#888882", border: "none", borderRadius: "10px", fontSize: "14px", fontWeight: 700, cursor: file ? "pointer" : "not-allowed", display: "inline-flex", alignItems: "center", gap: "8px" }}
                >
                  <Zap size={16} /> Start AI Analysis
                </button>
              </div>
            )}

            {analyzing && (
              <div style={{ textAlign: "center", padding: "20px 0" }}>
                {preview && (
                  <div style={{ position: "relative", marginBottom: "24px" }}>
                    <img src={preview} alt="Analyzing" style={{ width: "100%", maxHeight: "380px", objectFit: "contain", background: "#0a0a0a", borderRadius: "12px", display: "block", opacity: 0.6 }} />
                    <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "10px" }}>
                      <div style={{ width: "60px", height: "60px", background: "var(--accent)", borderRadius: "18px", display: "flex", alignItems: "center", justifyContent: "center", animation: "pulse 1s infinite" }}>
                        <Zap size={26} color="#0a0a0a" />
                      </div>
                      <span style={{ fontSize: "13px", fontWeight: 700, color: "#ffffff", textShadow: "0 1px 4px rgba(0,0,0,0.6)" }}>Analysing damage…</span>
                    </div>
                  </div>
                )}
                <div style={{ maxWidth: "360px", margin: "0 auto" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                    <span style={{ fontSize: "11px", color: "#888882" }}>Progress</span>
                    <span style={{ fontSize: "11px", fontWeight: 600, color: "#0a0a0a" }}>{Math.round(Math.min(progress, 100))}%</span>
                  </div>
                  <div style={{ background: "#f0f0eb", borderRadius: "4px", height: "8px", overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${Math.min(progress, 100)}%`, background: "var(--accent)", borderRadius: "4px", transition: "width 0.4s ease" }} />
                  </div>
                  <p style={{ fontSize: "10px", color: "#aaaaaa", marginTop: "10px" }}>
                    First run may take ~60–90s while the model warms up.
                  </p>
                </div>
              </div>
            )}

            {analysisDone && (
              <div>
                {(() => {
                  const escalated = approval === "ESCALATE_TO_HUMAN";
                  const banner = escalated
                    ? { bg: "#fffbeb", border: "var(--accent)", fg: "#854d0e" }
                    : { bg: "#d1fae5", border: "#10b981", fg: "#065f46" };
                  return (
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", padding: "12px 16px", background: banner.bg, borderRadius: "10px", marginBottom: "20px", border: `1px solid ${banner.border}` }}>
                      {escalated ? <AlertTriangle size={18} color="#d97706" /> : <CheckCircle size={18} color="#10b981" />}
                      <div>
                        <div style={{ fontSize: "13px", fontWeight: 600, color: banner.fg }}>
                          {escalated ? "Analysis complete — flagged for human review" : "Analysis complete — auto-approved"}
                        </div>
                        <div style={{ fontSize: "11px", color: banner.fg }}>
                          {detections.length} damage {detections.length === 1 ? "region" : "regions"} detected
                        </div>
                      </div>
                    </div>
                  );
                })()}

                {detections.length === 0 && results.length === 0 ? (
                  <div>
                    {preview && <AnnotatedImage src={preview} detections={[]} />}
                    <div style={{ textAlign: "center", padding: "20px", fontSize: "12px", color: "#888882" }}>
                      No visible damage was detected in this photo.
                    </div>
                  </div>
                ) : (
                  <DetectionView />
                )}

                {warnings.length > 0 && (
                  <div style={{ marginTop: "16px", padding: "10px 14px", background: "#fff7ed", border: "1px solid #fed7aa", borderRadius: "8px" }}>
                    <div style={{ fontSize: "11px", fontWeight: 600, color: "#9a3412", marginBottom: "4px" }}>Pipeline notes</div>
                    {warnings.map((w, i) => (
                      <div key={i} style={{ fontSize: "10px", color: "#9a3412" }}>• {w}</div>
                    ))}
                  </div>
                )}

                <div style={{ marginTop: "16px", padding: "12px 16px", background: "#fffbeb", border: "1px solid var(--accent)", borderRadius: "8px" }}>
                  <div style={{ fontSize: "11px", fontWeight: 600, color: "#0a0a0a", marginBottom: "2px" }}>Estimated Total Repair Cost</div>
                  <div style={{ fontSize: "22px", fontWeight: 800, color: "#0a0a0a" }}>{formatCostRange(totalMin, totalMax)}</div>
                  <div style={{ fontSize: "10px", color: "#888882" }}>Computed from COST_DB as of {new Date().toLocaleDateString("en-US", { month: "long", year: "numeric" })}</div>
                </div>

                {/* TEMP: human review to collect GEPA ground truth (shown when escalated). */}
                {reviewSessionId && (
                  <ReviewPanel
                    sessionId={reviewSessionId}
                    initialFindings={reviewFindings}
                  />
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Step 4: Report ── */}
        {step === "report" && (
          <div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px" }}>
              <div>
                <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#0a0a0a", marginBottom: "2px" }}>Inspection Report Generated</h3>
                <p style={{ fontSize: "11px", color: "#888882" }}>{selectedVehicle?.year} {selectedVehicle?.make} {selectedVehicle?.model}</p>
              </div>
            </div>

            <div style={{ background: "#fafafa", borderRadius: "10px", padding: "24px", border: "1px solid rgba(0,0,0,0.06)" }}>
              <div style={{ textAlign: "center", marginBottom: "20px" }}>
                <div style={{ fontSize: "10px", fontWeight: 700, color: "var(--accent)", letterSpacing: "0.1em", marginBottom: "6px" }}>VEHICLE DAMAGE ASSESSMENT REPORT</div>
                <h2 style={{ fontSize: "18px", fontWeight: 800, color: "#0a0a0a", marginBottom: "4px" }}>
                  {selectedVehicle?.year} {selectedVehicle?.make} {selectedVehicle?.model}
                </h2>
                <p style={{ fontSize: "11px", color: "#888882" }}>
                  Generated {new Date().toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" })}
                  {(() => {
                    const scored = results.filter((r) => r.confidence > 0);
                    if (scored.length === 0) return null;
                    const avg = Math.round(scored.reduce((s, r) => s + r.confidence, 0) / scored.length);
                    return ` | AI Confidence: ${avg}%`;
                  })()}
                </p>
              </div>

              {/* Annotated image + overlays at the top of the report */}
              {preview && (
                <div style={{ marginBottom: "20px" }}>
                  <StackedImages />
                </div>
              )}

              <div style={{ borderTop: "1px solid rgba(0,0,0,0.08)", paddingTop: "16px" }}>
                <h4 style={{ fontSize: "13px", fontWeight: 700, color: "#0a0a0a", marginBottom: "10px" }}>Executive Summary</h4>
                <p style={{ fontSize: "12px", color: "#444440", lineHeight: 1.75 }}>
                  This report presents an AI-assisted damage assessment for a {selectedVehicle?.year} {selectedVehicle?.make} {selectedVehicle?.model}.
                  {" "}{results.length} damaged component{results.length === 1 ? " was" : "s were"} identified, with an estimated total repair cost of <strong>{formatCostRange(totalMin, totalMax)}</strong>.
                  {" "}The claim was {approval === "ESCALATE_TO_HUMAN" ? "flagged for human review" : "auto-approved"} by the assessment pipeline.
                </p>
              </div>

              <div style={{ borderTop: "1px solid rgba(0,0,0,0.08)", paddingTop: "14px", marginTop: "14px" }}>
                <h4 style={{ fontSize: "13px", fontWeight: 700, color: "#0a0a0a", marginBottom: "10px" }}>Damage Findings</h4>
                {results.length === 0 ? (
                  <p style={{ fontSize: "11px", color: "#888882" }}>No damage findings recorded.</p>
                ) : (
                  results.map((r, i) => (
                    <div key={`${r.area}-${i}`} style={{ display: "flex", gap: "10px", marginBottom: "8px" }}>
                      <span style={{ fontSize: "11px", fontWeight: 600, color: "#0a0a0a", minWidth: "120px" }}>{r.area}</span>
                      <span style={{ fontSize: "11px", color: "#666660", flex: 1 }}>{r.damage} ({formatCostRange(r.costMin, r.costMax)})</span>
                      <span style={{ fontSize: "10px", fontWeight: 700, color: r.severity === "Severe" ? "#b91c1c" : r.severity === "Moderate" ? "#c2410c" : "#854d0e" }}>{r.severity}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Nav */}
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <button
          onClick={() => {
            if (currentIndex > 0) setStep(stepOrder[currentIndex - 1]);
            else onNavigate("dashboard");
          }}
          style={{ padding: "10px 20px", background: "#f5f5f0", color: "#444440", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 500, cursor: "pointer" }}
        >
          {currentIndex === 0 ? "Cancel" : "Back"}
        </button>

        {step === "report" ? (
          <button
            onClick={() => onNavigate("reports")}
            style={{ padding: "10px 22px", background: "#0a0a0a", color: "var(--accent)", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", gap: "6px" }}
          >
            <FileText size={13} /> View in Reports
          </button>
        ) : step === "analyze" && !analysisDone ? (
          // No button here — the centered "Start AI Analysis" handles this step.
          <span />
        ) : (
          <button
            onClick={() => {
              if (step === "select" && selectedVehicle) setStep("upload");
              else if (step === "upload" && file) setStep("analyze");
              else if (step === "analyze" && analysisDone) setStep("report");
            }}
            disabled={(step === "select" && !selectedVehicle) || (step === "upload" && !file)}
            style={{
              padding: "10px 22px",
              background: (step === "select" && !selectedVehicle) || (step === "upload" && !file) ? "#e8e8e3" : "var(--accent)",
              color: (step === "select" && !selectedVehicle) || (step === "upload" && !file) ? "#888882" : "#0a0a0a",
              border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 700,
              cursor: (step === "select" && !selectedVehicle) || (step === "upload" && !file) ? "not-allowed" : "pointer",
              display: "flex", alignItems: "center", gap: "6px",
            }}
          >
            Continue <ChevronRight size={13} />
          </button>
        )}
      </div>
    </div>
  );
}

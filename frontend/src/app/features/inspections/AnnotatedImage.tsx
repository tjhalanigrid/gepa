import { useState } from "react";
import type { DetectionWithBBox } from "../../lib/api";

// Per-damage-class colours (kept readable; loosely matches the backend palette).
const CLASS_COLORS: Record<string, string> = {
  dent: "#dd8a37",
  scratch: "#759e1d",
  crack: "#1775ba",
  glass_shatter: "#7e53d4",
  lamp_broken: "#305ad8",
  tire_flat: "#808788",
};
const DEFAULT_COLOR = "#ef4444";

interface AnnotatedImageProps {
  src: string;
  detections: DetectionWithBBox[];
  /** Hide the label chips, leaving only the boxes. */
  hideLabels?: boolean;
  /**
   * Original image dimensions the bbox coords refer to. Pass this when `src` is a
   * downscaled thumbnail (e.g. a saved claim) so boxes still map correctly. If
   * omitted, the natural size is read from the loaded <img>.
   */
  naturalSize?: { w: number; h: number };
}

/**
 * Renders an image with bounding boxes overlaid exactly where the AI detected
 * damage. Box coordinates are in the ORIGINAL image's pixel space, so we convert
 * them to percentages of the natural dimensions — the overlay then scales with
 * the displayed image at any size (no manual measurement needed).
 */
export function AnnotatedImage({ src, detections, hideLabels, naturalSize }: AnnotatedImageProps) {
  const [loaded, setLoaded] = useState<{ w: number; h: number } | null>(null);
  const nat = naturalSize ?? loaded;

  return (
    <div style={{ position: "relative", width: "100%", lineHeight: 0 }}>
      <img
        src={src}
        alt="Damage assessment"
        onLoad={(e) =>
          setLoaded({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
        }
        style={{
          width: "100%",
          height: "auto",
          display: "block",
          borderRadius: "12px",
          border: "1px solid rgba(0,0,0,0.08)",
        }}
      />

      {nat &&
        detections.map((d, i) => {
          const [x1, y1, x2, y2] = d.bbox;
          if (x2 <= x1 || y2 <= y1) return null;

          const left = (x1 / nat.w) * 100;
          const top = (y1 / nat.h) * 100;
          const width = ((x2 - x1) / nat.w) * 100;
          const height = ((y2 - y1) / nat.h) * 100;
          const color = CLASS_COLORS[d.damage] ?? DEFAULT_COLOR;
          const labelAbove = top > 7; // flip label inside the box if near the top edge

          return (
            <div
              key={i}
              style={{
                position: "absolute",
                left: `${left}%`,
                top: `${top}%`,
                width: `${width}%`,
                height: `${height}%`,
                border: `2px solid ${color}`,
                borderRadius: "4px",
                boxShadow: "0 0 0 1px rgba(0,0,0,0.35)",
                boxSizing: "border-box",
              }}
            >
              {!hideLabels && (
                <span
                  style={{
                    position: "absolute",
                    left: "-2px",
                    [labelAbove ? "bottom" : "top"]: "100%",
                    ...(labelAbove ? { marginBottom: "2px" } : { marginTop: "2px" }),
                    background: color,
                    color: "#ffffff",
                    fontSize: "10px",
                    fontWeight: 700,
                    lineHeight: 1.4,
                    padding: "2px 6px",
                    borderRadius: "4px",
                    whiteSpace: "nowrap",
                    fontFamily: "'Inter', system-ui, sans-serif",
                  }}
                >
                  {d.index}. {d.damage.replace(/_/g, " ")}
                  {d.confidence > 0 ? ` · ${Math.round(d.confidence * 100)}%` : ""}
                </span>
              )}
            </div>
          );
        })}
    </div>
  );
}

export { CLASS_COLORS };

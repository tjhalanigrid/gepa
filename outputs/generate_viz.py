"""
Generate pipeline visualization: YOLO damage bboxes + VLM reasoning overlaid on image.
Run from repo root: python3 outputs/generate_viz.py
"""

import json
import sys
import time
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config & tools ───────────────────────────────────────────────────────────
with open("configs/global_config.yaml") as f:
    config = yaml.safe_load(f)

from models.vlm_reasoning.tool_registry import get_tool_executor
executor = get_tool_executor(config)

image_path = str(list(Path("data/examples").glob("*.jpg"))[0].resolve())
img_name   = Path(image_path).name
print(f"Image: {img_name}")

# ── Run CV tools ─────────────────────────────────────────────────────────────
print("Running damage detection...")
t0 = time.time()
dmg = executor("run_damage_detection", {"image_path": image_path})
print(f"  {dmg['total_detections']} detections  ({round(time.time()-t0,2)}s)")

print("Running part segmentation...")
t0 = time.time()
seg = executor("run_part_segmentation", {"image_path": image_path})
print(f"  {seg['total_parts']} parts  ({round(time.time()-t0,2)}s)")

# Load last saved report for VLM reasoning output
report_path = Path("outputs/cv_tool_test_report.json")
report = json.loads(report_path.read_text()) if report_path.exists() else None

# ── Colour palette per damage class ─────────────────────────────────────────
CLASS_COLORS = {
    "glass shatter":  "#FF3B30",
    "glass_shatter":  "#FF3B30",
    "dent":           "#FF9500",
    "scratch":        "#FFCC00",
    "crack":          "#AF52DE",
    "lamp_broken":    "#5856D6",
    "tire_flat":      "#34C759",
}
DEFAULT_COLOR = "#007AFF"

img = np.array(Image.open(image_path).convert("RGB"))
h, w = img.shape[:2]

# ── Build figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12), facecolor="#1C1C1E")

# Left panel: annotated image
ax_img = fig.add_axes([0.01, 0.08, 0.60, 0.88])
ax_img.imshow(img)
ax_img.set_xlim(0, w)
ax_img.set_ylim(h, 0)
ax_img.axis("off")
ax_img.set_title(
    f"Pipeline Visualization — {img_name}",
    color="white", fontsize=13, fontweight="bold", pad=8
)

# Draw YOLO damage bboxes
legend_patches = []
seen_classes = set()
for det in dmg["detections"]:
    x1, y1, x2, y2 = det["bbox"]
    cls   = det["class"]
    conf  = det["confidence"]
    color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
    bw, bh = x2 - x1, y2 - y1

    rect = mpatches.FancyArrowPatch
    box = plt.Rectangle((x1, y1), bw, bh,
                         linewidth=2.5, edgecolor=color,
                         facecolor=color, alpha=0.15)
    ax_img.add_patch(box)
    border = plt.Rectangle((x1, y1), bw, bh,
                            linewidth=2.5, edgecolor=color, facecolor="none")
    ax_img.add_patch(border)

    label = f"{cls}  {conf:.2f}"
    ax_img.text(x1 + 4, y1 - 7, label,
                color="white", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color, alpha=0.85, linewidth=0))

    if cls not in seen_classes:
        legend_patches.append(mpatches.Patch(color=color, label=f"YOLO: {cls}"))
        seen_classes.add(cls)

# Draw part segmentation regions as text badges (no real masks — stub data)
part_colors = ["#30D158", "#0A84FF", "#FFD60A", "#FF375F"]
for i, part in enumerate(seg["parts"]):
    pc = part_colors[i % len(part_colors)]
    lbl = f"PART: {part['part']}  conf={part['segment_confidence']}"
    ax_img.text(10, 30 + i * 28, lbl,
                color="white", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=pc, alpha=0.85, linewidth=0))
    legend_patches.append(mpatches.Patch(color=pc,
        label=f"GDINO: {part['part']} ({part['segment_confidence']})"))

ax_img.legend(handles=legend_patches, loc="lower left",
              framealpha=0.75, facecolor="#2C2C2E", edgecolor="none",
              labelcolor="white", fontsize=9)

# ── Right panel: pipeline output ─────────────────────────────────────────────
ax_txt = fig.add_axes([0.63, 0.08, 0.36, 0.88])
ax_txt.set_facecolor("#2C2C2E")
ax_txt.axis("off")

lines = [
    ("YOLO DETECTIONS", "header"),
]
for det in dmg["detections"]:
    color = CLASS_COLORS.get(det["class"], DEFAULT_COLOR)
    lines.append((
        f"  • {det['class']}  conf={det['confidence']:.3f}\n"
        f"    bbox [{det['bbox'][0]:.0f},{det['bbox'][1]:.0f},"
        f"{det['bbox'][2]:.0f},{det['bbox'][3]:.0f}]",
        color
    ))

lines.append(("", "gap"))
lines.append(("PART SEGMENTS  (stub)", "header"))
for p in seg["parts"]:
    lines.append((
        f"  • {p['part']}\n"
        f"    conf={p['segment_confidence']}  dmg_area={p['damage_area_percent']}%",
        "#30D158"
    ))

if report:
    lines.append(("", "gap"))
    lines.append(("VLM REASONING OUTPUT", "header"))
    for entry in report.get("damage_part_map", []):
        c = CLASS_COLORS.get(entry["damage"], DEFAULT_COLOR)
        lines.append((
            f"  • {entry['damage']} → {entry['part']}\n"
            f"    severity: {entry['severity']}\n"
            f"    INR {entry['cost_min']:,} – {entry['cost_max']:,}",
            c
        ))

    lines.append(("", "gap"))
    total_str = (
        f"TOTAL: INR {report['total_min']:,} – {report['total_max']:,}\n"
        f"DECISION: {report['approval_decision']}"
    )
    lines.append((total_str, "total"))

    if report.get("warnings"):
        lines.append(("", "gap"))
        lines.append(("WARNINGS", "header"))
        for w in report["warnings"]:
            lines.append((f"  ⚠ {w}", "#FFD60A"))

y = 0.97
for text, style in lines:
    if style == "gap":
        y -= 0.025
        continue
    if style == "header":
        ax_txt.text(0.04, y, text, transform=ax_txt.transAxes,
                    color="#AEAEB2", fontsize=9.5, fontweight="bold",
                    fontfamily="monospace", va="top")
        y -= 0.04
    elif style == "total":
        ax_txt.text(0.04, y, text, transform=ax_txt.transAxes,
                    color="#FFD60A", fontsize=10, fontweight="bold",
                    fontfamily="monospace", va="top")
        y -= 0.09
    else:
        ax_txt.text(0.04, y, text, transform=ax_txt.transAxes,
                    color=style, fontsize=9, fontfamily="monospace", va="top")
        y -= 0.085

    if y < 0.02:
        break

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = Path("outputs/pipeline_visualization.jpg")
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved → {out_path.resolve()}")
print("Open with: open outputs/pipeline_visualization.jpg")

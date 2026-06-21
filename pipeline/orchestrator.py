"""
pipeline/orchestrator.py

Central entry point for the Vehicle Damage Assessment pipeline.

Architecture (Qwen brain + backend post-processing, Ollama backend):
  - PiAgent drives a free-form loop using Qwen via Ollama. The VLM is the SOLE
    brain — it classifies damage itself and picks its own optional vision tools.
  - The BACKEND (this module) does the deterministic work after the VLM returns:
        • cost = lookup_cost() over COST_DB (plain Python — NOT the LLM)
        • SAM2 (ultralytics) region boxes → merged union with the VLM damage boxes
          (UI-only: masks + source-tagged merged bboxes). No YOLO, no DINOv2.
        • annotated image, image-based approval, iteration log.
  - No trained-model fallback; if the VLM reports nothing usable, escalate.

This module owns:
  - Ollama health-check gate (_load_models)
  - Delegating Stage 1 to PiAgent (models/vlm_reasoning/pi_agent.py)
  - Backend cost + merged union + annotated/merged images + iteration log
  - Approval gate + FinalDamageReport construction + trajectory saving
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from PIL import Image

from pipeline.schema import (
    FinalDamageReport,
    DamagePartEntry,
    ToolCallRecord,
    DetectionWithBBox,
)

logger = logging.getLogger(__name__)


# VALID_DAMAGE_CLASSES = {"dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"}

VALID_DAMAGE_CLASSES = {
    "grille",
    "front_left_quarter_panel",
    "front_right_quarter_panel",
    "side_mirror",
    "radiator_support",
    "a_pillar",
    "b_pillar",
    "c_pillar",
    "rocker_panel",
    "quarter_panel",
    "tailgate",
    "license_plate",
    "fog_lamp",
    "wheel",
}


# VALID_PARTS = {
#     "front_bumper", "rear_bumper", "hood", "windshield", "rear_windshield",
#     "front_left_door", "front_right_door", "rear_left_door", "rear_right_door",
#     "left_fender", "right_fender", "trunk_lid", "roof_panel",
#     "headlight", "taillight", "tire",
# }


VALID_PARTS = {
    "front_bumper",
    "rear_bumper",
    "hood",
    "grille",
    "windshield",
    "rear_windshield",

    "left_fender",
    "right_fender",

    "front_left_door",
    "front_right_door",
    "rear_left_door",
    "rear_right_door",

    "roof_panel",
    "trunk_lid",
    "tailgate",

    "quarter_panel",

    "headlight",
    "taillight",
    "fog_lamp",

    "side_mirror",

    "wheel",
    "tire",

    "a_pillar",
    "b_pillar",
    "c_pillar",

    "rocker_panel",

    "radiator_support",
}
VALID_SEVERITY = {"minor", "moderate", "severe"}


# ── System prompt ────────────────────────────────────────────────────────────
# scripts/sft_train.py imports CODEACT_SYSTEM_PROMPT from this module. Re-export the
# single source of truth (the live brain prompt) so it can never drift.

from models.vlm_reasoning.pi_agent import CODEACT_SYSTEM_PROMPT  # noqa: E402,F401


# ── Ollama health gate ───────────────────────────────────────────────────────

def _load_models(config: dict) -> None:
    """
    Ollama health check. The VLM is served by the Ollama process (localhost:11434)
    and is NOT loaded in-process. Name kept for backward compatibility with
    backend/app.py startup, which calls _load_models(config) on /health warm-up.

    Raises RuntimeError if Ollama is unreachable or the configured model is absent.
    """
    from models.vlm_reasoning.ollama_client import check_health

    vlm_cfg  = config.get("vlm", {})
    base_url = vlm_cfg.get("ollama_base_url", "http://localhost:11434")
    model    = vlm_cfg.get("model_id", "qwen3.5:9b")

    logger.info(f"Checking Ollama: model='{model}' at {base_url}")
    if not check_health(base_url, model):
        raise RuntimeError(
            f"Ollama is not running or model '{model}' is not available at {base_url}. "
            f"Start Ollama with: ollama serve | Then pull model with: ollama pull {model}"
        )
    logger.info(f"Ollama ready: {model} @ {base_url}")


# ── Image preprocessing ────────────────────────────────────────────────────────

def _preprocess_image(image_path: str) -> str:
    """
    Validate and normalise image before sending to the VLM. Converts grayscale and
    RGBA to RGB. Returns the (possibly converted) image path.

    Raises ValueError if the image cannot be opened or is too small.
    """
    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"Image file not found: {image_path}")

    try:
        img = Image.open(path)
    except Exception as e:
        raise ValueError(f"Cannot open image at {image_path}: {e}") from e

    w, h = img.size
    if w < 100 or h < 100:
        raise ValueError(
            f"Image too small: {w}x{h}px. Minimum 100x100px required."
        )

    if img.mode in ("L", "RGBA", "P", "LA"):
        logger.info(f"Converting image from mode '{img.mode}' to RGB")
        converted = img.convert("RGB")
        out_path = path.parent / f"_rgb_{path.name}"
        converted.save(out_path)
        return str(out_path)

    return image_path


# ── Trajectory persistence ──────────────────────────────────────────────────────

def _save_trajectory(
    image_path: str,
    img_w: int,
    img_h: int,
    steps: list,
    costed: list,
    total_min: int,
    total_max: int,
    elapsed: float,
    model_id: str,
    raw_dir: str = "data/trajectories/collected",
) -> None:
    """Save raw trajectory to the configured collection folder (trajectory_filter promotes it)."""
    from pipeline.schema import Trajectory

    traj = Trajectory(
        trajectory_id=uuid.uuid4().hex,
        image_path=image_path,
        image_width=img_w,
        image_height=img_h,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_id=model_id,
        steps=steps,
        final_damage_map=costed,
        total_min=total_min,
        total_max=total_max,
        total_elapsed_s=elapsed,
        filter_status="unfiltered",
    )

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{traj.trajectory_id}.json"
    out.write_text(traj.model_dump_json(indent=2))
    logger.info(f"Trajectory saved: {out}")


# ── Backend post-processing helpers (deterministic — no LLM) ────────────────────

# Source → BGR colour for the merged (VLM ∪ SAM2) bbox view.
_SOURCE_COLORS_BGR = {
    "vlm":  (60, 180, 75),    # green  — VLM damage finding
    "sam2": (200, 130, 0),    # blue   — SAM2 region only
    "both": (0, 165, 255),    # orange — VLM finding confirmed by a SAM2 region
}
_CLASS_COLORS_BGR = {
    "dent": (221, 138, 55), "scratch": (117, 158, 29), "crack": (23, 117, 186),
    "glass_shatter": (126, 83, 212), "lamp_broken": (48, 90, 216),
    "tire_flat": (128, 135, 136),
}


def _iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _pct_to_px(bbox_pct: list, w: int, h: int) -> list:
    """
    Convert a VLM bbox_pct (0–100) to pixel coords, hardened against the VLM's
    unreliable geometry: clamp to range, fix corner ordering, enforce a minimum box
    size so a degenerate/flat box never renders as a line, and clamp extreme aspect
    ratios so a junk coordinate never renders as a full-width "ground strip".
    """
    try:
        p = [max(0.0, min(100.0, float(v))) for v in bbox_pct[:4]]
        if len(p) != 4:
            raise ValueError
    except Exception:
        p = [5, 5, 95, 95]
    x1, x2 = sorted((p[0] / 100 * w, p[2] / 100 * w))
    y1, y2 = sorted((p[1] / 100 * h, p[3] / 100 * h))
    min_w, min_h = 0.06 * w, 0.06 * h          # ≥6% of the frame each side
    if x2 - x1 < min_w:
        cx = (x1 + x2) / 2; x1, x2 = cx - min_w / 2, cx + min_w / 2
    if y2 - y1 < min_h:
        cy = (y1 + y2) / 2; y1, y2 = cy - min_h / 2, cy + min_h / 2

    # Aspect-ratio sanity clamp. A box >MAX_AR× wider-than-tall (or taller-than-wide)
    # is almost always a bad VLM coordinate (the full-width flat strip pinned to the
    # ground, or a sliver). Shrink the long side toward the box centre so it renders
    # as a sane box — the finding and its cost are untouched, only the geometry is fixed.
    MAX_AR = 3.5
    bw, bh = x2 - x1, y2 - y1
    if bh > 0 and bw / bh > MAX_AR:
        new_bw = MAX_AR * bh
        cx = (x1 + x2) / 2; x1, x2 = cx - new_bw / 2, cx + new_bw / 2
    elif bw > 0 and bh / bw > MAX_AR:
        new_bh = MAX_AR * bw
        cy = (y1 + y2) / 2; y1, y2 = cy - new_bh / 2, cy + new_bh / 2

    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    return [int(x1), int(y1), int(x2), int(y2)]


def _sam2_damage(image_path: str, detections: list, config: dict):
    """
    ONE SAM2 pass PROMPTED BY THE VLM DAMAGE BOXES (not auto "segment-everything",
    which boxes the whole scene and looks hallucinated). Each VLM damage box → one
    tight SAM2 mask on that exact damage. Returns:
      (tight_boxes, masked_overlay_path)
    The overlay is drawn here so the UI can show SAM2 masks immediately (no button,
    no second SAM2 run). Best-effort: returns ([], None) on failure.
    """
    boxes_in, det_for_box = [], []
    for d in detections:
        bb = [int(v) for v in (d.bbox if hasattr(d, "bbox") else d["bbox"])]
        if all(v == 0 for v in bb):
            continue
        boxes_in.append(bb)
        det_for_box.append(d)
    if not boxes_in:
        return [], None

    sam_cfg = config.get("part_segmentation", {}).get("sam2", {})
    weights = sam_cfg.get("weights_path", "models/damage_detection/models/sam2.1_b.pt")
    try:
        import cv2
        import numpy as np
        from shared.sam_mask import (
            _load_sam, _sam_masks_for_boxes, DAMAGE_MASK_COLORS, DEFAULT_MASK_COLOR,
        )

        if not _load_sam(weights):
            return [], None
        img = cv2.imread(image_path)
        if img is None:
            return [], None
        h, w = img.shape[:2]
        masks = _sam_masks_for_boxes(image_path, boxes_in, h, w)

        overlay = img.copy().astype(np.float32)
        tight = []
        drew = False
        for i, d in enumerate(det_for_box):
            m = masks.get(i)
            if m is None:
                continue
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            tight.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
            rgb = DAMAGE_MASK_COLORS.get(d.damage, DEFAULT_MASK_COLOR)
            bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
            overlay[m] = overlay[m] * 0.5 + bgr * 0.5
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cont, -1, tuple(int(c) for c in bgr), 2)
            drew = True

        masked_path = None
        if drew:
            out_dir = Path("data/uploads/masked")
            out_dir.mkdir(parents=True, exist_ok=True)
            masked_path = str(out_dir / f"{Path(image_path).stem}_masked_{uuid.uuid4().hex[:6]}.jpg")
            cv2.imwrite(masked_path, overlay.astype(np.uint8))
        logger.info(f"SAM2 refined {len(tight)} damage mask(s) (prompted by VLM boxes)")
        return tight, masked_path
    except Exception as e:
        logger.warning(f"SAM2 damage masks failed: {e} — merged union will be VLM-only")
        return [], None


def _sam2_masked_overlay(image_path: str, merged: list, config: dict):
    """
    Draw the SAM2 mask overlay PROMPTED BY THE MERGED (VLM ∪ SAM2) boxes, so the
    masks line up with the boxes shown in the "Merged" view. The merged set holds
    both each VLM box and its derived SAM2 tight box, so we dedupe by IoU —
    preferring the tighter SAM2 box — to avoid double-blending the same region.
    Returns the overlay path, or None on failure (caller falls back).
    """
    # Dedupe: SAM2 (tight) boxes first, then any VLM box SAM2 did not confirm.
    ordered = [m for m in merged if m.get("source") == "sam2"] + \
              [m for m in merged if m.get("source") != "sam2"]
    chosen = []
    for m in ordered:
        bb = [int(v) for v in m["bbox"]]
        if all(v == 0 for v in bb):
            continue
        if any(_iou([float(v) for v in bb], [float(v) for v in c["bbox"]]) >= 0.5 for c in chosen):
            continue
        chosen.append({"bbox": bb, "damage": m.get("damage", "")})
    if not chosen:
        return None

    sam_cfg = config.get("part_segmentation", {}).get("sam2", {})
    weights = sam_cfg.get("weights_path", "models/damage_detection/models/sam2.1_b.pt")
    try:
        import cv2
        import numpy as np
        from shared.sam_mask import (
            _load_sam, _sam_masks_for_boxes, DAMAGE_MASK_COLORS, DEFAULT_MASK_COLOR,
        )

        if not _load_sam(weights):
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        masks = _sam_masks_for_boxes(image_path, [c["bbox"] for c in chosen], h, w)

        overlay = img.copy().astype(np.float32)
        drew = False
        for i, c in enumerate(chosen):
            m = masks.get(i)
            if m is None:
                continue
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            rgb = DAMAGE_MASK_COLORS.get(c["damage"], DEFAULT_MASK_COLOR)
            bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
            overlay[m] = overlay[m] * 0.5 + bgr * 0.5
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cont, -1, tuple(int(c2) for c2 in bgr), 2)
            drew = True

        if not drew:
            return None
        out_dir = Path("data/uploads/masked")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = str(out_dir / f"{Path(image_path).stem}_masked_merged_{uuid.uuid4().hex[:6]}.jpg")
        cv2.imwrite(out, overlay.astype(np.uint8))
        logger.info(f"SAM2 mask overlay drawn from {len(chosen)} merged (VLM∪SAM2) box(es)")
        return out
    except Exception as e:
        logger.warning(f"SAM2 merged-box overlay failed: {e} — keeping VLM-prompted overlay")
        return None


def _merge_union(detections_with_bbox: list, sam2_boxes: list) -> list:
    """
    Merge the VLM damage boxes with the SAM2 tight mask boxes (which were derived
    FROM those damage boxes — so every SAM2 box lands on real damage, never the
    background). Source tags:
      • VLM box confirmed by an overlapping SAM2 mask → "both"
      • VLM box SAM2 did not segment                  → "vlm"
      • SAM2 tight mask box (precise damage outline)  → "sam2"
    Returns list of dicts: {index, bbox, damage, part, severity, confidence, source}.
    """
    merged = []
    idx = 1
    for d in detections_with_bbox:
        dbox = [float(v) for v in d.bbox]
        src = "both" if any(_iou(dbox, sb) >= 0.3 for sb in sam2_boxes) else "vlm"
        merged.append({"index": idx, "bbox": dbox, "damage": d.damage, "part": d.part,
                       "severity": d.severity, "confidence": d.confidence, "source": src})
        idx += 1
    # SAM2's tight boxes — labelled with the VLM class of the damage they refine.
    for sb in sam2_boxes:
        best, best_iou = None, 0.0
        for d in detections_with_bbox:
            o = _iou([float(v) for v in d.bbox], sb)
            if o > best_iou:
                best, best_iou = d, o
        label = best.damage if best is not None else "region"
        merged.append({"index": idx, "bbox": sb, "damage": label, "part": "unknown",
                       "severity": "minor", "confidence": 0.0, "source": "sam2"})
        idx += 1
    return merged


def _label_collides(rect: tuple, placed: list) -> bool:
    """True if label rect (x1,y1,x2,y2) overlaps any already-placed label rect."""
    ax1, ay1, ax2, ay2 = rect
    for bx1, by1, bx2, by2 in placed:
        if ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2:
            return True
    return False


def _draw_boxes(image_path: str, dets: list, out_subdir: str, color_by: str = "class") -> str:
    """Draw numbered boxes (dicts with bbox/damage/index/source) → saved JPEG path."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    h, w = img.shape[:2]
    placed_labels: list = []   # collision-avoidance for label backgrounds
    for d in dets:
        bbox = d["bbox"] if isinstance(d, dict) else d.bbox
        if not bbox or all(float(v) == 0.0 for v in bbox):
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        dmg = d.get("damage", "dent") if isinstance(d, dict) else d.damage
        src = d.get("source", "vlm") if isinstance(d, dict) else getattr(d, "source", "vlm")
        idx = d.get("index", 0) if isinstance(d, dict) else d.index
        color = (_SOURCE_COLORS_BGR.get(src, (128, 128, 128)) if color_by == "source"
                 else _CLASS_COLORS_BGR.get(dmg, (128, 128, 128)))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{idx}.{dmg}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lw, lh = tw + 4, th + 4

        # Candidate label positions, in order of preference: above the box, then inside
        # the top, then below the box. Pick the first that doesn't collide with an
        # already-drawn label and stays on-canvas — fixes the overlapping-label garble.
        candidates = [
            (x1, max(y1 - lh, 0)),                       # above
            (x1, min(y1 + 1, h - lh)),                   # inside top
            (x1, min(y2 + 1, h - lh)),                   # below
        ]
        lx, ly = candidates[0]
        for cx, cy in candidates:
            cx = min(cx, w - lw)
            rect = (cx, cy, cx + lw, cy + lh)
            if not _label_collides(rect, placed_labels):
                lx, ly = cx, cy
                break
        else:
            # All collide — nudge down in steps until clear or we run out of canvas.
            cx = min(x1, w - lw)
            cy = max(y1 - lh, 0)
            while _label_collides((cx, cy, cx + lw, cy + lh), placed_labels) and cy < h - lh:
                cy += lh
            lx, ly = cx, cy

        placed_labels.append((lx, ly, lx + lw, ly + lh))
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), color, -1)
        cv2.putText(img, label, (lx + 2, ly + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    out_dir = Path(out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / f"{Path(image_path).stem}_{uuid.uuid4().hex[:6]}.jpg")
    cv2.imwrite(out, img)
    return out


def _build_iterations(trajectory_steps: list) -> list:
    """Compact, UI-friendly iteration log: one entry per tool call + its 'why'."""
    iters = []
    for s in trajectory_steps:
        args = s.action.arguments or {}
        reason = args.get("reason") or args.get("part_query") or ""
        iters.append({
            "turn":      s.turn_index,
            "tool":      s.action.name,
            "reason":    reason,
            "summary":   s.observation_summary or "",
            "elapsed_s": s.elapsed_s,
            "ok":        s.observation_type != "error",
        })
    return iters


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    image_path: str,
    config: dict,
    claim_metadata: Optional[dict] = None,
) -> dict:
    """
    Main pipeline entry point. Called by the FastAPI backend.

    Stage 1: VLM brain loop (PiAgent). Qwen is the sole brain — free-form tools,
             returns its final damage_items (each with a bbox_pct).
    Stage 2: Build detections_with_bbox from damage_items (bbox_pct → pixels).
    Stage 3: BACKEND cost — price each item via COST_DB in plain Python (no LLM).
    Stage 4: SAM2 (backend) → region boxes; merged union (VLM ∪ SAM2) for the UI.
    Stage 5: Annotated image (boxes) + image-based approval + iteration log.

    Returns:
        FinalDamageReport as dict (via .model_dump())
    """
    from models.vlm_reasoning.cost_db import lookup_cost

    t_start = time.time()
    warnings_list: List[str] = []
    trajectory_steps: list = []

    image_path = _preprocess_image(image_path)

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception:
        img_w, img_h = 1920, 1080
        warnings_list.append("Could not read image dimensions — using fallback 1920x1080")

    # ── Stage 1: VLM brain loop (Qwen, free-form tools) ────────────────────────
    _load_models(config)   # Ollama health check (raises if not reachable)

    logger.info("Stage 1: PiAgent brain loop (Qwen via Ollama, free-form tools)")
    from models.vlm_reasoning.pi_agent import PiAgent
    from pipeline.feedback_reader import get_few_shot_examples
    few_shot = get_few_shot_examples(
        corrections_log_path=config.get("feedback", {}).get(
            "corrections_log", "data/feedback/corrections_log.jsonl"
        ),
        n=5,
    )
    if few_shot:
        logger.info("Temporal prompt: injecting few-shot corrections into system prompt")
    agent    = PiAgent(config, few_shot_examples=few_shot)
    loop_out = agent.run(image_path=image_path, trajectory_steps=trajectory_steps)
    warnings_list.extend(loop_out["warnings"])

    vlm_damage_items = loop_out.get("damage_items", [])
    vlm_detections   = loop_out.get("vlm_detections", [])

    # If the VLM ran run_damage_detection (found damage) but then Terminated with an
    # empty/sparse map, don't throw those findings away — salvage them so the report
    # reflects ALL detected damage instead of returning ₹0. The detection items carry
    # the same fields (class→damage_type, part, severity, bbox_pct).
    if not vlm_damage_items and vlm_detections:
        vlm_damage_items = [
            {
                "damage_type": d.get("class", ""),
                "part":        d.get("part", ""),
                "severity":    d.get("severity", "minor"),
                "confidence":  float(d.get("confidence", 0.0)),
                "bbox_pct":    d.get("bbox_pct", [5, 5, 95, 95]),
            }
            for d in vlm_detections
        ]
        warnings_list.append(
            f"SALVAGED_DETECTIONS: Terminate was empty but run_damage_detection found "
            f"{len(vlm_detections)} damage region(s) — using those for the report."
        )

    vlm_produced     = bool(vlm_damage_items)

    # ── Stage 2: detections_with_bbox from the VLM damage_items ─────────────────
    detections_with_bbox: List[DetectionWithBBox] = []
    for i, item in enumerate(vlm_damage_items):
        bbox_px = _pct_to_px(item.get("bbox_pct", [5, 5, 95, 95]), img_w, img_h)
        cmin, cmax = lookup_cost(item.get("damage_type", ""), item.get("part", ""))
        detections_with_bbox.append(DetectionWithBBox(
            index      = i + 1,
            bbox       = [float(v) for v in bbox_px],
            damage     = item.get("damage_type", "unknown"),
            part       = item.get("part", "unknown"),
            severity   = item.get("severity", "minor"),
            confidence = float(item.get("confidence", 0.0)),
            source     = "vlm",
            cost_min   = cmin,
            cost_max   = cmax,
        ))

    # ── Stage 3: BACKEND cost (deterministic Python over COST_DB; no LLM) ───────
    costed: List[DamagePartEntry] = []
    for item in vlm_damage_items:
        cmin, cmax = lookup_cost(item.get("damage_type", ""), item.get("part", ""))
        costed.append(DamagePartEntry(
            damage   = item.get("damage_type", ""),
            part     = item.get("part", ""),
            severity = item.get("severity", "minor"),
            cost_min = cmin,
            cost_max = cmax,
        ))
    total_min = sum(e.cost_min for e in costed)
    total_max = sum(e.cost_max for e in costed)
    if not costed:
        warnings_list.append(
            "EMPTY_DAMAGE_MAP: the VLM reported no visible damage. Escalating."
        )

    # ── Stage 4: SAM2 (backend, prompted by VLM boxes) + merged union ──────────
    # SAM2 is prompted by the VLM damage boxes, so it segments ONLY real damage
    # regions — never the background (no "segment-everything" hallucination).
    sam2_boxes, vlm_masked_path = _sam2_damage(image_path, detections_with_bbox, config)
    merged = _merge_union(detections_with_bbox, sam2_boxes)
    # SAM2 mask overlay is prompted by the MERGED (VLM ∪ SAM2) boxes so the masks
    # match the merged view; fall back to the VLM-prompted overlay if that fails.
    masked_image_path = _sam2_masked_overlay(image_path, merged, config) or vlm_masked_path
    n_both = sum(1 for m in merged if m["source"] == "both")
    logger.info(
        f"Merged union: {len(detections_with_bbox)} VLM box(es) + "
        f"{len(sam2_boxes)} SAM2 tight box(es) → {len(merged)} item(s) "
        f"({n_both} VLM confirmed by SAM2)"
    )

    # ── Stage 5: annotated image, approval, iteration log ──────────────────────
    # Always (re)draw the damage boxes so the UI's "Detected Damage" shows them.
    if detections_with_bbox:
        annotated_path = _draw_boxes(
            image_path, detections_with_bbox, "data/uploads/annotated", color_by="class")
    else:
        annotated_path = loop_out.get("annotated_image_path") or image_path
    merged_image_path = (_draw_boxes(image_path, merged, "data/uploads/merged",
                                     color_by="source") if merged else annotated_path)

    # Approval = image-based corroboration (the AI's two independent passes agree),
    # never cost. Escalate on no damage, or a final class its detection pass missed.
    det_classes = {str(d.get("class", "")).lower() for d in vlm_detections}
    # If VLM never called run_damage_detection, det_classes is empty — treat every
    # Terminate item as uncorroborated rather than silently AUTO_APPROVING.
    if not det_classes and vlm_produced:
        uncorroborated = [it.get("damage_type") for it in vlm_damage_items]
    else:
        uncorroborated = [
            it.get("damage_type") for it in vlm_damage_items
            if str(it.get("damage_type", "")).lower() not in det_classes
        ]
    if not vlm_produced:
        approval = "ESCALATE_TO_HUMAN"
    elif uncorroborated:
        approval = "ESCALATE_TO_HUMAN"
        warnings_list.append(
            f"ESCALATED: claim(s) {uncorroborated} were not corroborated by the VLM's "
            "own detection pass (low visual confidence)."
        )
    else:
        approval = "AUTO_APPROVED"

    # TEMP: while collecting ground-truth corrections for GEPA, force EVERY report
    # through human review — compulsory, irrespective of confidence/corroboration or
    # whether any damage was found (config: approval.force_human_review).
    if config.get("approval", {}).get("force_human_review", False):
        if approval != "ESCALATE_TO_HUMAN":
            warnings_list.append(
                "FORCED_HUMAN_REVIEW: auto-approve disabled while collecting "
                "ground-truth corrections (approval.force_human_review=true)."
            )
        approval = "ESCALATE_TO_HUMAN"

    iterations = _build_iterations(trajectory_steps)
    tool_log = [
        ToolCallRecord(
            tool         = step.action.name,
            args_summary = str(step.action.arguments)[:80],
            elapsed_s    = step.elapsed_s,
            result_keys  = list(step.observation_data.keys()) if step.observation_data else [],
            success      = step.observation_type != "error",
        )
        for step in trajectory_steps
    ]

    elapsed = round(time.time() - t_start, 2)

    _save_trajectory(
        image_path = image_path,
        img_w      = img_w,
        img_h      = img_h,
        steps      = trajectory_steps,
        costed     = costed,
        total_min  = total_min,
        total_max  = total_max,
        elapsed    = elapsed,
        model_id   = config["vlm"]["model_id"],
        raw_dir    = config.get("trajectory", {}).get("raw_dir", "data/trajectories/collected"),
    )

    return FinalDamageReport(
        image_path           = image_path,
        damage_part_map      = costed,
        detections_with_bbox = detections_with_bbox,
        merged_detections    = merged,
        total_min            = total_min,
        total_max            = total_max,
        currency             = "INR",
        approval_decision    = approval,
        tool_call_log        = tool_log,
        iterations           = iterations,
        total_inference_s    = elapsed,
        warnings             = list(dict.fromkeys(warnings_list)),
        raw_vlm_response     = loop_out.get("raw_vlm_response"),
        annotated_image_path = annotated_path,
        merged_image_path    = merged_image_path,
        masked_image_path    = masked_image_path,
    ).model_dump()

"""
SAM2 mask generation for damage regions.
Takes image path + list of DetectionWithBBox → overlaid mask image path.
Degrades gracefully (bbox-fill fallback) if SAM2 unavailable or weights missing.
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

_sam_model  = None
_sam_loaded = False
_sam_failed = False

# Mask colours: (R, G, B) — converted to BGR for cv2
DAMAGE_MASK_COLORS = {
    "dent":          (55,  138, 221),
    "scratch":       (29,  158, 117),
    "crack":         (186, 117,  23),
    "glass_shatter": (212,  83, 126),
    "lamp_broken":   (216,  90,  48),
    "tire_flat":     (136, 135, 128),
}
DEFAULT_MASK_COLOR = (128, 128, 128)


def _load_sam(weights_path: str) -> bool:
    """
    Load SAM2 via ultralytics (the same backend models/part_segmentation uses).
    The previous facebook `sam2` package is not installed on this stack — using
    ultralytics keeps a single SAM2 dependency and the stock sam2.1_b.pt weight.
    """
    global _sam_model, _sam_loaded, _sam_failed
    if _sam_loaded:
        return True
    if _sam_failed:
        return False

    search_paths = [
        weights_path,
        "models/damage_detection/models/sam2.1_b.pt",   # stock ultralytics SAM2 base
        "weights/sam2.1_hiera_base_plus.pt",
    ]

    found_path = next((p for p in search_paths if p and Path(p).exists()), None)
    if not found_path:
        logger.warning(
            f"SAM2 weights not found. Searched: {search_paths}. "
            "Mask overlay will use the bbox/GrabCut fallback."
        )
        _sam_failed = True
        return False

    try:
        from ultralytics import SAM
        _sam_model  = SAM(found_path)
        _sam_loaded = True
        logger.info(f"SAM2 (ultralytics) loaded from {found_path}")
        return True
    except Exception as e:
        logger.warning(f"SAM2 load failed: {e} — mask overlay will use bbox fallback")
        _sam_failed = True
        return False


def _sam_masks_for_boxes(image_path: str, boxes: list, h: int, w: int) -> dict:
    """
    Run ultralytics SAM2 once for all boxes; return {box_index: bool_mask(h,w)}.
    Uses polygon output (.xy is in original-image coords) filled into a mask, so
    the result aligns with the original frame regardless of letterboxing.
    """
    import cv2

    out: dict = {}
    if not boxes:
        return out
    try:
        results = _sam_model(image_path, bboxes=boxes, verbose=False)
    except Exception as e:
        logger.warning(f"SAM2 batch predict failed: {e}")
        return out
    if not results or results[0].masks is None or results[0].masks.xy is None:
        return out

    polys = results[0].masks.xy
    for i in range(min(len(polys), len(boxes))):
        poly = polys[i]
        if poly is None or len(poly) < 3:
            continue
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        if m.sum() > 0:
            out[i] = m.astype(bool)
    return out


def generate_masked_image(
    image_path: str,
    detections: list,
    weights_path: str = "models/damage_detection/models/sam2.1_b.pt",
    output_dir: str = "data/uploads/masked",
    alpha: float = 0.45,
) -> str:
    """
    Overlays SAM2 masks (or bbox fills as fallback) on detected damage regions.
    Returns path to output JPEG.

    Graceful degradation order:
      1. SAM2 mask (if weights exist and model loads)
      2. Semi-transparent bbox fill (if SAM2 unavailable)
    Every per-detection failure is caught individually — one failure doesn't stop others.
    """
    import cv2

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy().astype(np.float32)

    sam_ok = _load_sam(weights_path)

    # Run SAM2 (ultralytics) once for all valid boxes; map back to each detection.
    sam_masks: dict = {}
    if sam_ok:
        boxes, det_of_box = [], []
        for di, det in enumerate(detections):
            bb = _get_bbox(det)
            if all(v == 0 for v in bb):
                continue
            bx1, by1 = max(0, int(bb[0])), max(0, int(bb[1]))
            bx2, by2 = min(w, int(bb[2])), min(h, int(bb[3]))
            if bx2 <= bx1 or by2 <= by1:
                continue
            boxes.append([bx1, by1, bx2, by2])
            det_of_box.append(di)
        box_masks = _sam_masks_for_boxes(image_path, boxes, h, w)
        for bi, di in enumerate(det_of_box):
            if bi in box_masks:
                sam_masks[di] = box_masks[bi]

    for di, det in enumerate(detections):
        bbox  = _get_bbox(det)
        cls   = _get_field(det, "damage", "dent")
        idx   = _get_field(det, "index", 0)
        rgb   = DAMAGE_MASK_COLORS.get(cls, DEFAULT_MASK_COLOR)
        bgr   = (rgb[2], rgb[1], rgb[0])  # RGB → BGR for cv2

        if all(v == 0 for v in bbox):
            continue

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        mask_drawn = False
        mask = sam_masks.get(di)
        if mask is not None:
            try:
                colored = np.zeros_like(img_bgr, dtype=np.float32)
                colored[mask] = bgr
                overlay[mask] = overlay[mask] * (1 - alpha) + colored[mask] * alpha
                mask_u8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay.astype(np.uint8), contours, -1, bgr, 2)
                mask_drawn = True
            except Exception as e:
                logger.warning(f"SAM2 mask overlay failed for detection {idx}: {e}")

        if not mask_drawn:
            # Fallback: GrabCut segments actual damage shape within bbox.
            # Much tighter than full bbox fill — covers only the damaged pixels.
            try:
                gc_mask = np.zeros(img_bgr.shape[:2], np.uint8)
                bgd_m = np.zeros((1, 65), np.float64)
                fgd_m = np.zeros((1, 65), np.float64)
                rect = (x1, y1, x2 - x1, y2 - y1)
                cv2.grabCut(img_bgr, gc_mask, rect, bgd_m, fgd_m, 5, cv2.GC_INIT_WITH_RECT)
                fg = np.where((gc_mask == cv2.GC_PR_FGD) | (gc_mask == cv2.GC_FGD), 1, 0).astype(np.uint8)
                if fg.sum() > 200:
                    colored = np.zeros_like(img_bgr, dtype=np.float32)
                    colored[fg == 1] = bgr
                    overlay[fg == 1] = overlay[fg == 1] * (1 - alpha) + colored[fg == 1] * alpha
                    # Contour border
                    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    tmp = overlay.astype(np.uint8)
                    cv2.drawContours(tmp, contours, -1, bgr, 2)
                    overlay[:] = tmp.astype(np.float32)
                    mask_drawn = True
            except Exception as e:
                logger.warning(f"GrabCut failed for detection {idx}: {e}")

        if not mask_drawn:
            # Last resort: bbox outline only — no fill, won't cover undamaged areas
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), bgr, 2)

    result = overlay.astype(np.uint8)

    # Number badges on top
    for det in detections:
        bbox = _get_bbox(det)
        if all(v == 0 for v in bbox):
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cls = _get_field(det, "damage", "dent")
        idx = _get_field(det, "index", 0)
        rgb = DAMAGE_MASK_COLORS.get(cls, DEFAULT_MASK_COLOR)
        bgr = (rgb[2], rgb[1], rgb[0])

        badge_r = 14
        bx = max(x1 + badge_r + 2, badge_r + 2)
        by = (y1 - badge_r - 2) if y1 > badge_r + 10 else (y1 + badge_r + 2)
        by = max(badge_r + 2, min(h - badge_r - 2, by))
        cv2.circle(result, (bx, by), badge_r, bgr, -1)
        cv2.circle(result, (bx, by), badge_r, (255, 255, 255), 1)
        lbl = str(idx)
        fs = 0.5 if idx < 10 else 0.4
        ts = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
        cv2.putText(result, lbl,
                    (bx - ts[0] // 2, by + ts[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / (Path(image_path).stem + "_masked.jpg"))
    cv2.imwrite(out_path, result)
    return out_path


def _get_bbox(det) -> list:
    if hasattr(det, "bbox"):
        return list(det.bbox)
    return det.get("bbox", [0, 0, 0, 0])


def _get_field(det, field, default):
    if hasattr(det, field):
        return getattr(det, field)
    return det.get(field, default)

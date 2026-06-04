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
    global _sam_model, _sam_loaded, _sam_failed
    if _sam_loaded:
        return True
    if _sam_failed:
        return False

    search_paths = [
        weights_path,
        "weights/sam2.1_hiera_base_plus.pt",
        "weights/sam2.1_hiera_base_plus.pth",
        "weights/sam2_hiera_base_plus.pt",
        "/tmp/partseg_deps.ZohiE5/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    ]

    found_path = None
    for p in search_paths:
        if Path(p).exists():
            found_path = p
            logger.info(f"SAM2 weights found at: {found_path}")
            break

    if not found_path:
        logger.warning(
            f"SAM2 weights not found. Searched: {search_paths}. "
            "Run: python3 scripts/download_sam2_weights.py"
        )
        _sam_failed = True
        return False

    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        config_candidates = [
            "sam2_hiera_base_plus.yaml",
            "sam2/configs/sam2/sam2_hiera_base_plus.yaml",
        ]
        config_path = "sam2_hiera_base_plus.yaml"
        for c in config_candidates:
            if Path(c).exists():
                config_path = c
                break

        sam2_model = build_sam2(config_path, found_path, device=device)
        _sam_model  = SAM2ImagePredictor(sam2_model)
        _sam_loaded = True
        logger.info(f"SAM2 loaded on {device}")
        return True

    except Exception as e:
        logger.warning(f"SAM2 load failed: {e} — mask overlay will use bbox fallback")
        _sam_failed = True
        return False


def generate_masked_image(
    image_path: str,
    detections: list,
    weights_path: str = "weights/sam2.1_hiera_base_plus.pt",
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

    if sam_ok:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            _sam_model.set_image(img_rgb)
        except Exception as e:
            logger.warning(f"SAM2 set_image failed: {e}")
            sam_ok = False

    for det in detections:
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
        if sam_ok:
            try:
                masks, scores, _ = _sam_model.predict(
                    box=np.array([[x1, y1, x2, y2]]),
                    multimask_output=False,
                )
                if masks is not None and len(masks) > 0:
                    mask = masks[0].astype(bool)
                    colored = np.zeros_like(img_bgr, dtype=np.float32)
                    colored[mask] = bgr
                    overlay[mask] = overlay[mask] * (1 - alpha) + colored[mask] * alpha
                    mask_u8 = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay.astype(np.uint8), contours, -1, bgr, 2)
                    mask_drawn = True
            except Exception as e:
                logger.warning(f"SAM2 predict failed for detection {idx}: {e}")

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

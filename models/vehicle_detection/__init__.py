"""
models/vehicle_detection/__init__.py

UNTRAINED vehicle detection. Stock YOLOv8 (COCO weights, e.g. yolov8n.pt) locates
the vehicle in the frame and returns its region-of-interest (ROI). This model is
deliberately NOT fine-tuned on damage data: it provides spatial grounding only
(where the car is), never damage classification. The VLM brain owns all damage
reasoning; this ROI just anchors it and crops away background.

Public contract (the one rule every model follows — see CLAUDE.md):

    run(image_path: str, config: dict) -> dict

Returns:
    {
      "image_path": str,
      "vehicles": [{"bbox": [x1,y1,x2,y2], "cls": "car", "confidence": 0.97}, ...],
      "primary_roi": [x1,y1,x2,y2] | None,   # largest-area vehicle box
      "annotated_image_path": str | None,
      "total_vehicles": int,
      "warnings": [str],
    }

No damage classes are ever emitted. If YOLO finds no vehicle, `vehicles` is empty,
`primary_roi` is None, and a warning is appended — the caller falls back to the
full frame.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# COCO class names we treat as "a vehicle". Stock YOLOv8 COCO ids:
#   2=car, 3=motorcycle, 5=bus, 7=truck
_VEHICLE_COCO_CLASSES = frozenset({"car", "motorcycle", "bus", "truck"})

_model = None
_loaded_weights_path: Optional[str] = None


def _load_model(weights_path: str):
    """Lazy-load and cache the stock YOLO model (mirrors damage_detection idiom)."""
    global _model, _loaded_weights_path
    if _model is not None and weights_path == _loaded_weights_path:
        return _model
    from ultralytics import YOLO
    logger.info(f"Loading stock vehicle-detection YOLO: {weights_path}")
    _model = YOLO(weights_path)          # auto-downloads yolov8n.pt on first use
    _loaded_weights_path = weights_path
    return _model


def run(image_path: str, config: Dict) -> dict:
    """Run stock YOLOv8 vehicle detection on a single image."""
    weights_path = config.get("weights_path", "yolov8n.pt")
    conf_threshold = float(config.get("confidence_threshold", 0.25))
    device = config.get("device", "cpu")
    allowed = set(config.get("classes", _VEHICLE_COCO_CLASSES))

    warnings: List[str] = []

    if not Path(image_path).exists():
        raise ValueError(f"Image not found: {image_path}")

    try:
        model = _load_model(weights_path)
    except Exception as e:
        logger.error(f"Failed to load vehicle-detection YOLO from {weights_path}: {e}")
        raise RuntimeError(f"Vehicle detection model load failed: {e}") from e

    results = model.predict(image_path, conf=conf_threshold, verbose=False, device=device)[0]

    vehicles: List[dict] = []
    if results.boxes is not None:
        for box, cls_id, conf in zip(
            results.boxes.xyxy.tolist(),
            results.boxes.cls.tolist(),
            results.boxes.conf.tolist(),
        ):
            cls_name = model.names[int(cls_id)]
            if cls_name not in allowed:
                continue
            vehicles.append({
                "bbox": [round(float(x), 2) for x in box],
                "cls": cls_name,
                "confidence": round(float(conf), 3),
            })

    # Primary ROI = largest-area vehicle box (the subject of the photo).
    primary_roi: Optional[List[float]] = None
    if vehicles:
        def _area(v):
            x1, y1, x2, y2 = v["bbox"]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)
        primary_roi = max(vehicles, key=_area)["bbox"]
    else:
        warnings.append("No vehicle detected — caller should use the full frame as ROI")

    annotated_path = _draw_rois(image_path, vehicles, primary_roi) if vehicles else None

    logger.info(
        f"Vehicle detection: {len(vehicles)} vehicle(s) in {Path(image_path).name} "
        f"(primary_roi={[int(v) for v in primary_roi] if primary_roi else None})"
    )
    return {
        "image_path": image_path,
        "vehicles": vehicles,
        "primary_roi": primary_roi,
        "annotated_image_path": annotated_path,
        "total_vehicles": len(vehicles),
        "warnings": warnings,
    }


def _draw_rois(image_path: str, vehicles: List[dict], primary_roi) -> Optional[str]:
    """Draw vehicle ROI boxes for the annotation UI. Best-effort — returns None on failure."""
    try:
        import cv2
    except Exception as e:
        logger.warning(f"cv2 unavailable for ROI draw: {e}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"cv2 cannot read {image_path} for ROI draw")
        return None

    for v in vehicles:
        x1, y1, x2, y2 = [int(c) for c in v["bbox"]]
        is_primary = primary_roi is not None and v["bbox"] == primary_roi
        color = (0, 200, 0) if is_primary else (0, 160, 200)  # BGR
        thickness = 3 if is_primary else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f"{v['cls']} {v['confidence']:.2f}" + (" [ROI]" if is_primary else "")
        cv2.putText(img, label, (x1 + 3, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    out_dir = Path("data/uploads/vehicle_roi")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{Path(image_path).stem}_roi_{uuid.uuid4().hex[:6]}.jpg")
    cv2.imwrite(out_path, img)
    return out_path

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

_model = None
_loaded_weights_path = None


def run(image_path: str, config: Dict) -> dict:
    """
    Run YOLOv8 damage detection on a single image.

    Args:
        image_path: Absolute path to vehicle image
        config: damage_detection config slice from global_config.yaml

    Returns:
        dict with keys: image_path, detections (list), total_detections (int)
    """
    global _model, _loaded_weights_path

    weights_path = config.get("weights_path", "models/damage_detection/models/best.pt")
    conf_threshold = config.get("confidence_threshold", 0.25)
    device = config.get("device", "cpu")

    if _model is None or weights_path != _loaded_weights_path:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading damage detection model: {weights_path}")
            _model = YOLO(weights_path)
            _loaded_weights_path = weights_path
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {weights_path}: {e}")
            raise RuntimeError(f"Damage detection model load failed: {e}") from e

    if not Path(image_path).exists():
        raise ValueError(f"Image not found: {image_path}")

    results = _model.predict(image_path, conf=conf_threshold, verbose=False, device=device)[0]

    detections = []
    if results.boxes is not None:
        for box, cls_id, conf in zip(
            results.boxes.xyxy.tolist(),
            results.boxes.cls.tolist(),
            results.boxes.conf.tolist()
        ):
            detections.append({
                "bbox": [round(x, 2) for x in box],
                "class": _model.names[int(cls_id)],
                "confidence": round(float(conf), 3)
            })

    logger.info(f"Damage detection: {len(detections)} detections in {Path(image_path).name}")
    return {
        "image_path": image_path,
        "detections": detections,
        "total_detections": len(detections)
    }

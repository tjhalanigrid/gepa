import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

_segmenter = None


def run(image_path: str, config: Dict) -> dict:
    """
    Run part segmentation on a single image.

    Args:
        image_path: Absolute path to vehicle image
        config: part_segmentation config slice from global_config.yaml

    Returns:
        dict with keys: image_path, parts (list), total_parts (int), warnings (list)
    """
    global _segmenter

    sam_cfg = config.get("sam2", {})
    weights_path = sam_cfg.get("weights_path", "weights/sam2.1_hiera_base_plus.pt")
    warnings = []

    if not Path(image_path).exists():
        raise ValueError(f"Image not found: {image_path}")

    if _segmenter is None:
        try:
            from models.part_segmentation.infer import PartSegmenter
            logger.info(f"Loading part segmenter, weights: {weights_path}")
            _segmenter = PartSegmenter(weights_path=weights_path)
        except Exception as e:
            logger.error(f"Failed to load PartSegmenter: {e}")
            raise RuntimeError(f"Part segmentation model load failed: {e}") from e

    try:
        parts = _segmenter.predict(image_path)
    except Exception as e:
        logger.warning(f"Part segmentation inference failed: {e}")
        warnings.append(f"Part segmentation failed: {str(e)}")
        parts = []

    if not parts:
        warnings.append("Part segmentation returned no detections")

    logger.info(f"Part segmentation: {len(parts)} parts in {Path(image_path).name}")
    return {
        "image_path": image_path,
        "parts": parts,
        "total_parts": len(parts),
        "warnings": warnings
    }

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_segmenter = None
_segmenter_key = None


def run(image_path: str, config: Dict, rois: Optional[List[List[float]]] = None) -> dict:
    """
    Run SAM2 + DINOv2 class-agnostic region segmentation on a single image.

    Args:
        image_path: Absolute path to vehicle image
        config: part_segmentation config slice from global_config.yaml
        rois: optional list of vehicle bboxes (from vehicle_detection) — when given,
              only regions whose centre lies inside a ROI are returned.

    Returns:
        dict with keys: image_path, regions (list), total_regions (int), warnings (list)

    `regions` carry NO part/damage labels — only spatial + anomaly evidence. The
    VLM brain assigns part names and damage classes.
    """
    global _segmenter, _segmenter_key

    sam_cfg = config.get("sam2", {})
    dino_cfg = config.get("dinov2", {})
    post_cfg = config.get("postprocess", {})

    weights_path = sam_cfg.get("weights_path", "models/damage_detection/models/sam2.1_b.pt")
    warnings: List[str] = []

    if not Path(image_path).exists():
        raise ValueError(f"Image not found: {image_path}")

    key = (weights_path, dino_cfg.get("model_id", "facebook/dinov2-base"))
    if _segmenter is None or key != _segmenter_key:
        try:
            from models.part_segmentation.infer import PartSegmenter
            logger.info(f"Loading PartSegmenter (SAM2={weights_path}, DINOv2={key[1]})")
            _segmenter = PartSegmenter(
                weights_path=weights_path,
                dinov2_model_id=dino_cfg.get("model_id", "facebook/dinov2-base"),
                device=sam_cfg.get("device"),
                dedup_cosine=dino_cfg.get("dedup_cosine", 0.92),
                min_area_px=post_cfg.get("min_mask_area_px", 500),
                max_regions=post_cfg.get("max_regions", 24),
            )
            _segmenter_key = key
        except Exception as e:
            logger.error(f"Failed to load PartSegmenter: {e}")
            raise RuntimeError(f"Part segmentation model load failed: {e}") from e

    try:
        regions = _segmenter.predict(image_path, rois=rois)
    except Exception as e:
        logger.warning(f"Part segmentation inference failed: {e}")
        warnings.append(f"Part segmentation failed: {str(e)}")
        regions = []

    if not regions:
        warnings.append("Part segmentation returned no regions")

    logger.info(f"Part segmentation: {len(regions)} region(s) in {Path(image_path).name}")
    return {
        "image_path": image_path,
        "regions": regions,
        "total_regions": len(regions),
        "warnings": warnings,
    }

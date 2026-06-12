"""
models/damage_detection/ — RETIRED trained YOLOv8 damage detector.

The pipeline moved to untrained CV models + a VLM brain (see CLAUDE.md). The
trained `best.pt` weight has been archived to `models/_archive/best.pt` and is no
longer loaded anywhere in the live path. Damage classification is now done by the
VLM brain (models/vlm_reasoning/pi_agent.py); spatial grounding comes from stock
YOLO (models/vehicle_detection) and SAM2+DINOv2 (models/part_segmentation).

This folder is retained only to host the stock SAM2 weight (`models/sam2.1_b.pt`)
that the segmentation tool loads.

Calling run() here is a programming error — it points at a model that was
intentionally removed. Use models.vehicle_detection.run instead.
"""

from typing import Dict


def run(image_path: str, config: Dict) -> dict:
    raise RuntimeError(
        "models.damage_detection.run() is retired. The trained best.pt damage "
        "detector was archived to models/_archive/best.pt. Use "
        "models.vehicle_detection.run() for vehicle ROI and let the VLM brain "
        "(pi_agent) classify damage."
    )

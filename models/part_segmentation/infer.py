"""
models/part_segmentation/infer.py

UNTRAINED region segmentation: SAM2 + DINOv2.

  • SAM2 (ultralytics, stock sam2.1_b.pt) produces class-agnostic region masks.
    No fine-tuning on damage/part data — it just proposes "things" on the car.
  • DINOv2 (facebook/dinov2-base, self-supervised, no task head) embeds each
    region. The embeddings are used to (a) de-duplicate near-identical masks and
    (b) attach an `anomaly_score` = how far a region sits from the vehicle's mean
    region embedding (an unsupervised "this looks unusual" cross-check).

Crucially, this module emits NO part labels and NO damage labels. It returns raw
spatial regions only. The VLM brain decides which part each region is and whether
it is damaged. This is the anti-hallucination contract: the untrained models can
never assert damage they cannot know.

Degrades gracefully: if SAM2 or DINOv2 is unavailable, returns whatever it can
(possibly an empty list) and the caller records a warning.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PartSegmenter:
    """SAM2 region proposals + DINOv2 embeddings. Class-agnostic, untrained."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        dinov2_model_id: str = "facebook/dinov2-base",
        device: Optional[str] = None,
        dedup_cosine: float = 0.92,
        min_area_px: int = 500,
        max_regions: int = 24,
        config_path: Optional[str] = None,  # accepted for back-compat; unused
    ) -> None:
        self.weights_path = weights_path or "models/damage_detection/models/sam2.1_b.pt"
        self.dinov2_model_id = dinov2_model_id
        self.dedup_cosine = float(dedup_cosine)
        self.min_area_px = int(min_area_px)
        self.max_regions = int(max_regions)

        self._device = device or self._pick_device()
        self._sam = None
        self._sam_failed = False
        self._dino = None
        self._dino_proc = None
        self._dino_failed = False

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    # ── lazy loaders ──────────────────────────────────────────────────────────

    def _load_sam(self):
        if self._sam is not None or self._sam_failed:
            return self._sam
        if not Path(self.weights_path).exists():
            logger.warning(f"SAM2 weights missing at {self.weights_path} — segmentation disabled")
            self._sam_failed = True
            return None
        try:
            from ultralytics import SAM
            logger.info(f"Loading SAM2 (ultralytics) from {self.weights_path} on {self._device}")
            self._sam = SAM(self.weights_path)
        except Exception as e:
            logger.warning(f"SAM2 load failed: {e} — segmentation disabled")
            self._sam_failed = True
            self._sam = None
        return self._sam

    def _load_dino(self):
        if self._dino is not None or self._dino_failed:
            return self._dino
        try:
            import torch
            from transformers import AutoModel, AutoImageProcessor
            logger.info(f"Loading DINOv2 '{self.dinov2_model_id}' on {self._device}")
            self._dino_proc = AutoImageProcessor.from_pretrained(self.dinov2_model_id)
            self._dino = AutoModel.from_pretrained(self.dinov2_model_id).to(self._device).eval()
        except Exception as e:
            logger.warning(f"DINOv2 load failed: {e} — anomaly scoring disabled")
            self._dino_failed = True
            self._dino = None
        return self._dino

    # ── public API ──────────────────────────────────────────────────────────

    def predict(self, image_path: str, rois: Optional[List[List[float]]] = None) -> List[Dict]:
        """
        Return class-agnostic regions:
          [{"bbox":[x1,y1,x2,y2], "area_px":int, "mask_poly":[[x,y],...],
            "anomaly_score":float, "embedding_id":int}, ...]

        `rois` (optional): vehicle bboxes from vehicle_detection. When given, only
        regions whose centre falls inside a ROI are kept (background is dropped).
        """
        sam = self._load_sam()
        if sam is None:
            return []

        import cv2

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            logger.warning(f"cv2 cannot read {image_path}")
            return []
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # SAM2 automatic "segment everything" pass.
        try:
            results = sam.predict(image_path, verbose=False, device=self._device)
        except Exception as e:
            logger.warning(f"SAM2 predict failed: {e}")
            return []

        regions = self._extract_regions(results, w, h, rois)
        if not regions:
            return []

        # DINOv2 embeddings → anomaly score + dedup.
        embeddings = self._embed_regions(img_rgb, regions)
        if embeddings is not None:
            regions = self._score_and_dedup(regions, embeddings)

        # Cap to the most anomalous / largest regions to keep the VLM context small.
        regions.sort(key=lambda r: (r.get("anomaly_score", 0.0), r["area_px"]), reverse=True)
        regions = regions[: self.max_regions]
        for i, r in enumerate(regions):
            r["embedding_id"] = i
        logger.info(f"PartSegmenter: {len(regions)} region(s) in {Path(image_path).name}")
        return regions

    # ── helpers ───────────────────────────────────────────────────────────────

    def _extract_regions(self, results, w: int, h: int, rois) -> List[Dict]:
        """Turn SAM2 masks into bbox + simplified polygon dicts, filtered by area/ROI."""
        regions: List[Dict] = []
        if not results:
            return regions
        res = results[0]
        masks = getattr(res, "masks", None)
        if masks is None or masks.xy is None:
            return regions

        for poly in masks.xy:
            if poly is None or len(poly) < 3:
                continue
            xs = poly[:, 0]
            ys = poly[:, 1]
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area_px:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if rois and not _point_in_any(cx, cy, rois):
                continue
            regions.append({
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "area_px": int(area),
                "mask_poly": _simplify_poly(poly),
                "anomaly_score": 0.0,
                "embedding_id": -1,
            })
        return regions

    def _embed_regions(self, img_rgb: np.ndarray, regions: List[Dict]) -> Optional[np.ndarray]:
        dino = self._load_dino()
        if dino is None:
            return None
        try:
            import torch
            from PIL import Image as PILImage

            crops = []
            for r in regions:
                x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
                crop = img_rgb[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
                if crop.size == 0:
                    crop = img_rgb
                crops.append(PILImage.fromarray(crop))

            inputs = self._dino_proc(images=crops, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                out = dino(**inputs)
            # CLS token (last_hidden_state[:,0]) is the standard DINOv2 global descriptor.
            emb = out.last_hidden_state[:, 0, :].float().cpu().numpy()
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            return emb / norm
        except Exception as e:
            logger.warning(f"DINOv2 embedding failed: {e} — skipping anomaly/dedup")
            return None

    def _score_and_dedup(self, regions: List[Dict], emb: np.ndarray) -> List[Dict]:
        """anomaly_score = distance from the mean region embedding; greedy cosine dedup."""
        mean = emb.mean(axis=0)                       # (D,) — keep 1-D so dot is a scalar
        mean = mean / max(float(np.linalg.norm(mean)), 1e-8)
        for i, r in enumerate(regions):
            r["anomaly_score"] = round(1.0 - float(emb[i] @ mean), 4)

        # Greedy dedup: drop a region if it is both spatially overlapping AND
        # semantically near an already-kept region.
        order = sorted(range(len(regions)), key=lambda i: regions[i]["area_px"], reverse=True)
        kept_idx: List[int] = []
        for i in order:
            dup = False
            for j in kept_idx:
                cos = float(emb[i] @ emb[j].T)
                if cos >= self.dedup_cosine and _iou(regions[i]["bbox"], regions[j]["bbox"]) >= 0.5:
                    dup = True
                    break
            if not dup:
                kept_idx.append(i)
        return [regions[i] for i in kept_idx]


# ── module-level geometry helpers ──────────────────────────────────────────────

def _point_in_any(cx: float, cy: float, boxes: List[List[float]]) -> bool:
    for b in boxes:
        if len(b) == 4 and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
            return True
    return False


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _simplify_poly(poly: np.ndarray, max_pts: int = 24) -> List[List[int]]:
    """Downsample a SAM polygon to <= max_pts integer points to keep JSON compact."""
    n = len(poly)
    if n <= max_pts:
        step = 1
    else:
        step = int(np.ceil(n / max_pts))
    return [[int(p[0]), int(p[1])] for p in poly[::step]]

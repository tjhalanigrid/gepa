"""
Image-serving endpoints for the correction / review UI.

Both session-scoped (HITL) and job-scoped variants are provided:
  plain     — original upload (canvas background)
  annotated — numbered, class-coloured damage boxes
  masked    — SAM2 mask overlay (falls back to annotated / plain)
  merged    — VLM ∪ SAM2 source-coloured boxes (job only)
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from pipeline.schema import DetectionWithBBox

from .. import state
from ..core import config
from ..services.imaging import generate_annotated_image

logger = logging.getLogger(__name__)
router = APIRouter(tags=["images"])


# ── Session-scoped (HITL) ─────────────────────────────────────────────────────

@router.get("/session/{session_id}/plain_image")
async def get_plain_image(session_id: str):
    """Serve the original uploaded image, with size hints in headers."""
    from PIL import Image as PILImage

    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    image_path = state.sessions[session_id].report.image_path
    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    headers = {}
    try:
        with PILImage.open(image_path) as img:
            w, h = img.size
            headers = {"X-Image-Width": str(w), "X-Image-Height": str(h)}
    except Exception:
        pass
    return FileResponse(image_path, media_type="image/jpeg", headers=headers)


@router.get("/session/{session_id}/masked_image")
async def get_masked_image(session_id: str):
    """SAM2 mask overlay for a session. 503 if weights missing; falls back to annotated."""
    from shared.sam_mask import generate_masked_image, _sam_failed

    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    report = state.sessions[session_id].report

    if _sam_failed:
        raise HTTPException(
            status_code=503,
            detail="SAM2 weights not available. Run: python3 scripts/download_sam2_weights.py",
        )

    detections = report.detections_with_bbox
    if not detections:
        ann = getattr(report, "annotated_image_path", None)
        if ann and Path(ann).exists():
            return FileResponse(ann, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="No detections available for masking")

    try:
        out_path = generate_masked_image(
            image_path=report.image_path,
            detections=detections,
            weights_path=config.sam2_weights_path(config.load_config()),
        )
        return FileResponse(out_path, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Masked image generation failed: {e}")
        ann = getattr(report, "annotated_image_path", None)
        if ann and Path(ann).exists():
            return FileResponse(ann, media_type="image/jpeg")
        if Path(report.image_path).exists():
            return FileResponse(report.image_path, media_type="image/jpeg")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/annotated_image")
async def get_annotated_image(session_id: str):
    """Annotated image with numbered boxes for the Step 2 correction UI."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    report = state.sessions[session_id].report

    detections = report.detections_with_bbox
    plain_path = report.image_path
    if not detections:
        if Path(plain_path).exists():
            return FileResponse(plain_path, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="No detections with bbox and no image")

    try:
        annotated_path = generate_annotated_image(plain_path, detections)
        return FileResponse(annotated_path, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Annotated image generation failed: {e}")
        if Path(plain_path).exists():
            return FileResponse(plain_path, media_type="image/jpeg")
        raise HTTPException(status_code=500, detail=str(e))


# ── Job-scoped ────────────────────────────────────────────────────────────────

@router.get("/job/{job_id}/plain_image")
async def get_job_plain_image(job_id: str):
    """Original uploaded image for any job (canvas background)."""
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    result = state.jobs[job_id].get("result", {})
    image_path = result.get("image_path") or result.get("report", {}).get("image_path", "")
    if image_path and Path(image_path).exists():
        return FileResponse(image_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/job/{job_id}/masked_image")
async def get_job_masked_image(job_id: str):
    """SAM2 masks for a completed job. Falls back to the pre-rendered/annotated image."""
    from shared.sam_mask import generate_masked_image, _sam_failed

    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = state.jobs[job_id]
    if job.get("status") != "complete":
        raise HTTPException(status_code=400, detail="Job not complete")

    result = job.get("result", {})
    report = result.get("report", result)
    image_path = report.get("image_path", "")
    annotated = report.get("annotated_image_path", "")
    raw_dets = report.get("detections_with_bbox", [])

    # Fast path: orchestrator already rendered the SAM2 overlay this run.
    pre = report.get("masked_image_path")
    if pre and Path(pre).exists():
        return FileResponse(pre, media_type="image/jpeg")

    if _sam_failed or not raw_dets:
        if annotated and Path(annotated).exists():
            return FileResponse(annotated, media_type="image/jpeg")
        if image_path and Path(image_path).exists():
            return FileResponse(image_path, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="No image available")

    try:
        det_objects = [
            DetectionWithBBox(**d) if isinstance(d, dict) else d for d in raw_dets
        ]
        out = generate_masked_image(
            image_path=image_path,
            detections=det_objects,
            weights_path=config.sam2_weights_path(config.load_config()),
        )
        return FileResponse(out, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Job masked image failed: {e}")
        if annotated and Path(annotated).exists():
            return FileResponse(annotated, media_type="image/jpeg")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job/{job_id}/annotated_image")
async def get_job_annotated_image(job_id: str):
    """Annotated image for any completed job (approved or escalated)."""
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = state.jobs[job_id]
    if job.get("status") != "complete":
        raise HTTPException(status_code=400, detail="Job not complete yet")

    result = job.get("result", {})
    report = result.get("report", result)

    annotated = report.get("annotated_image_path")
    if annotated and Path(annotated).exists():
        return FileResponse(annotated, media_type="image/jpeg")

    image_path = report.get("image_path", "")
    if image_path:
        stem = Path(image_path).stem
        ann_dir = Path("data/uploads/yolo_annotated")
        if ann_dir.exists():
            candidates = list(ann_dir.glob(f"{stem}_yolo_*.jpg"))
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                return FileResponse(str(latest), media_type="image/jpeg")

    if image_path and Path(image_path).exists():
        return FileResponse(image_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No annotated image available")


@router.get("/job/{job_id}/merged_image")
async def get_job_merged_image(job_id: str):
    """Merged-union (VLM ∪ SAM2) source-coloured bbox image."""
    report = state.get_completed_report(job_id)
    merged = report.get("merged_image_path")
    if merged and Path(merged).exists():
        return FileResponse(merged, media_type="image/jpeg")
    annotated = report.get("annotated_image_path")
    if annotated and Path(annotated).exists():
        return FileResponse(annotated, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No merged image available")
